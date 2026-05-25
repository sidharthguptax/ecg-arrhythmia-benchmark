import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 180) → (N, 1, 180) for Conv1d
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ECGDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader