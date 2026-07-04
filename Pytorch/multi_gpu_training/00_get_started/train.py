"""
Treinamento distribuído (multi-GPU) com PyTorch DistributedDataParallel (DDP).

Projetado para 4x NVIDIA T4 (16GB VRAM cada), mas funciona para qualquer
número de GPUs na mesma máquina.

Como rodar:
    torchrun --standalone --nproc_per_node=4 train_ddp.py --epochs 20 --batch-size 64

Cada processo controla 1 GPU. O `torchrun` cuida de spawnar os processos
e definir as variáveis de ambiente (RANK, LOCAL_RANK, WORLD_SIZE).
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision import transforms


# --------------------------------------------------------------------------- #
# Setup / Teardown do processo distribuído
# --------------------------------------------------------------------------- #

def setup_ddp():
    """Inicializa o grupo de processos distribuídos usando NCCL (backend
    otimizado para GPU-GPU communication)."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def log(msg: str):
    """Só imprime no processo principal (rank 0), pra não poluir o terminal
    com 4 cópias do mesmo log."""
    if is_main_process():
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Modelo (troque por qualquer arquitetura sua; aqui uso ResNet18 como exemplo)
# --------------------------------------------------------------------------- #

def build_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return model


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #

def build_dataloaders(batch_size: int, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # DistributedSampler garante que cada GPU veja um pedaço diferente e
    # sem sobreposição dos dados a cada época.
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,       # batch_size é POR GPU, não total
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, train_sampler


# --------------------------------------------------------------------------- #
# Loop de treino
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch):
    model.train()
    running_loss = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device)
    running_total = torch.zeros(1, device=device)

    t0 = time.time()
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision: reduz uso de VRAM e acelera bastante em T4
        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach() * images.size(0)
        running_correct += (outputs.argmax(1) == targets).sum()
        running_total += images.size(0)

        if step % 50 == 0:
            log(f"  [epoch {epoch}] step {step}/{len(loader)} "
                f"loss={loss.item():.4f}")

    # Agrega as métricas de todas as GPUs para reportar um número único e correto
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_total, op=dist.ReduceOp.SUM)

    avg_loss = (running_loss / running_total).item()
    acc = (running_correct / running_total).item()
    dt = time.time() - t0

    log(f"[epoch {epoch}] treino: loss={avg_loss:.4f} acc={acc:.4f} "
        f"tempo={dt:.1f}s")


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device)
    running_total = torch.zeros(1, device=device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.detach() * images.size(0)
        running_correct += (outputs.argmax(1) == targets).sum()
        running_total += images.size(0)

    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_total, op=dist.ReduceOp.SUM)

    avg_loss = (running_loss / running_total).item()
    acc = (running_correct / running_total).item()
    log(f"[epoch {epoch}] validação: loss={avg_loss:.4f} acc={acc:.4f}")
    return acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64,
                         help="batch size POR GPU (total = batch-size * num_gpus)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    log(f"Treinando com {world_size} GPUs | batch efetivo total = "
        f"{args.batch_size * world_size}")

    # --- Modelo ---
    model = build_model(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])

    # --- Dados ---
    train_loader, val_loader, train_sampler = build_dataloaders(
        args.batch_size, args.num_workers
    )

    # --- Otimizador ---
    # Escala o LR linearmente com o número de GPUs (regra prática comum)
    scaled_lr = args.lr * world_size
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr,
                           momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    if is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Necessário para o shuffle ser diferente (e sincronizado) a cada época
        train_sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                         device, epoch)
        acc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        # Só o rank 0 salva checkpoint, senão as 4 GPUs escrevem o mesmo arquivo
        if is_main_process() and acc > best_acc:
            best_acc = acc
            torch.save(
                model.module.state_dict(),  # .module para pegar o modelo "puro"
                os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
            log(f"  -> novo melhor modelo salvo (acc={acc:.4f})")

    cleanup_ddp()


if __name__ == "__main__":
    main()