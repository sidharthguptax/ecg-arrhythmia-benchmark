import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import yaml

from src.utils.preprocessing import load_splits
from src.dataset import get_dataloaders


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_name, cfg):
    if model_name == 'cnn':
        from src.models.cnn import ECG_CNN
        return ECG_CNN(
            num_classes=cfg['classes']['num_classes'],
            dropout=cfg['models']['cnn']['dropout']
        )
    elif model_name == 'lstm':
        from src.models.lstm import ECG_BiLSTM
        return ECG_BiLSTM(
            num_classes=cfg['classes']['num_classes'],
            hidden_size=cfg['models']['lstm']['hidden_size'],
            num_layers=cfg['models']['lstm']['num_layers'],
            dropout=cfg['models']['lstm']['dropout']
        )
    elif model_name == 'transformer':
        from src.models.transformer import ECG_Transformer
        return ECG_Transformer(
            num_classes=cfg["classes"]["num_classes"],
            dropout=cfg["models"]["transformer"]["dropout"],
            d_model=cfg["models"]["transformer"]["d_model"],
            nhead=cfg["models"]["transformer"]["nhead"],
            num_layers=cfg["models"]["transformer"]["num_layers"],
            max_len=cfg["data"]["segment_length"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * len(y)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


def train(model_name):
    cfg = load_config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(cfg['data']['processed_dir'])
    train_loader, val_loader, _ = get_dataloaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        batch_size=cfg['training']['batch_size']
    )

    # Model
    model = get_model(model_name, cfg).to(device)
    print(f"Model: {model_name} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])

    # Tracking
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    os.makedirs('results/metrics', exist_ok=True)

    print(f"\nTraining {model_name}...\n")
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1     = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'results/{model_name}_best.pt')
            print(f"           ✓ Best model saved (val F1: {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= cfg['training']['early_stopping_patience']:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # Save history
    with open(f'results/metrics/{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'lstm', 'transformer'])
    args = parser.parse_args()
    train(args.model)