#include <stdio.h>
#include <cuda_runtime.h>

#define N 4
/*
BLOCO: é um grupo de threads que podem:
    - Trabalhar juntas
    - Compartilhar memória rápida (shared memory)
    - Sincronizar execução

ANALOGIA:
    - GRID: fábrica toda
    - BLOCO: cada sala/equipe da fábrica
    - THREAD: cada trabalhador daquela sala/equipe (daquele BLOCO)
*/

/*
EXPLICAÇÃO PROFUNDA threadsPerBlock:
matriz M:
[ 0  1  2  3 ]
[ 4  5  6  7 ]
[ 8  9 10 11 ]
[12 13 14 15 ]

Ela é armazenada linearmente:
[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
``` cuda
// isso cria:
// 4 threads no eixo X
// 4 threads no eixo Y
dim3 threadsPerBlock(N, N);
```
Visualmente:
(0,0) (1,0) (2,0) (3,0)
(0,1) (1,1) (2,1) (3,1)
(0,2) (1,2) (2,2) (3,2)
(0,3) (1,3) (2,3) (3,3)

Total = 16 threads.
Cada par `(x, y)` é uma thread diferente.
Thread (2,1) significa: coluna 2 linha 1

Transformando 2D → 1D
``` cuda
int idx = row * N + col;
// Thread (2,1):
idx = 1 * 4 + 2
idx = 6
```
Então essa thread trabalha no elemento 6 da memória.
Começando do 0 o indice 6 é o (2, 1) da nossa matriz de posições
(0,0) (1,0) (2,0) (3,0)
(0,1) (1,1) (2,1) (3,1)
(0,2) (1,2) (2,2) (3,2)
(0,3) (1,3) (2,3) (3,3)

```cuda
C[6] = A[6] + B[6]
```
Thread com coordenada (x,y), você cuida desse pedaço da memória.
*/

// __global__: indica que esta função roda na GPU (device) mas é chamada pela CPU (host)
// float *A, *B, *C: são ponteiros para as matrizes na memória da GPU
__global__ void matrix_add(float *A, float *B, float *C) {
    // Cada bloco tem suas próprias threads organizadas em 3D (x, y, z) -> isso é já para lidar com dados 3D
    // Para dados acima de 3 dimensões existem estrategias para contornar
    // `threadIdx`: dá a posição da thread dentro do seu bloco
    int col = threadIdx.x;  // posição horizontal da thread no bloco
    int row = threadIdx.y;  // posição vertical da thread no bloco
    
    // indice glogal linear
    int idx = row * N + col;

    // cada thread soma um elemento
    C[idx] = A[idx] + B[idx];
}

int main() {
    // bytes necessarios para salvar uma matriz 4x4 na memória
    // 4 x 4 = 16 x sizeof(float) = 16 x 4 = 64 bytes
    // esse é o espaço necessario para reservar na GPU
    int size = N * N * sizeof(float);

    // criando as matrizes na CPU (estão na RAM)
    float A_cpu[N][N] = {
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1}
    };
    float B_cpu[N][N] = {
        {2, 2, 2, 2},
        {2, 2, 2, 2},
        {2, 2, 2, 2},
        {2, 2, 2, 2}
    };
    float C_cpu[N][N];

    // cria os ponteiros que vão apontar para a memória da GPU
    float *A_gpu, *B_gpu, *C_gpu;
    // pede para reservar os 64 bytes que vem do size para as suas matrizes
    cudaMalloc((void**)&A_gpu, size);
    cudaMalloc((void**)&B_gpu, size);
    cudaMalloc((void**)&C_gpu, size);

    // faz a passagem CPU → GPU
    cudaMemcpy(A_gpu, A_cpu, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_cpu, size, cudaMemcpyHostToDevice);
    // cria as threads (seus trabalhadores) cada trabalhador vai cuidar de um número da matriz
    // aqui estamos criando N x N threads = 16 uma para cada elemento 
    dim3 threadsPerBlock(N, N);
    // 1 bloco, 16 threads
    matrix_add<<<1, threadsPerBlock>>>(A_gpu, B_gpu, C_gpu);
    // espera a gpu finalizar
    cudaDeviceSynchronize();
    // faz a passagem GPU → CPU
    cudaMemcpy(C_cpu, C_gpu, size, cudaMemcpyDeviceToHost);

    // VISUALIZAÇÕES
    printf("MATRIX A:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%.0f ", A_cpu[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("MATRIX B:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%.0f ", B_cpu[i][j]);
        }
        printf("\n");
    }

    printf("---------------------------------");
    printf("\n");

    printf("Resultado da soma:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            printf("%.0f ", C_cpu[i][j]);
        }
        printf("\n");
    }
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}