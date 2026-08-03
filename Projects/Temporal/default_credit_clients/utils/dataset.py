import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch import Tensor

from torch.utils.data import Dataset

class DefaultCredit(Dataset):
    def __init__(self, file_path: Path) -> None:
        self.dataframe = pd.read_parquet(file_path)
        self.data = self._preprocess_data()


    def __len__(self) -> int:
        return len(self.data[1])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        X, y = self.data
        return X[idx], y[idx]

    def _preprocess_data(self) -> tuple[Tensor, Tensor]:
        MONTHS = [
            ("1", "PAY_0"),
            ("2", "PAY_2"),
            ("3", "PAY_3"),
            ("4", "PAY_4"),
            ("5", "PAY_5"),
            ("6", "PAY_6"),
        ]
        sequences = []
        for month, pay_col in MONTHS:
            month_features = self.dataframe[
                [
                    f"BILL_AMT{month}",  # valor da fatura
                    f"PAY_AMT{month}",  # valor pago
                    pay_col,  # status de pagamento: são os meses de atraso 0 está em dia, 1, 2, 3 meses em atraso
                    "LIMIT_BAL"  # limite de crédito
                ]
            ].values
            sequences.append(month_features)
        X = np.stack(
            sequences,
            axis=1
        )
        y = self.dataframe["label"].values
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y