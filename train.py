"""
train.py
Training loop shared by all three HeartNet models.
Uses CrossEntropyLoss, Adam, ReduceLROnPlateau, early stopping,
and saves the best checkpoint per model.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR    = './data'
MODELS_DIR  = './models'
RESULTS_DIR = './results'

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(data_dir=DATA_DIR):
    """Load train and validation numpy arrays from disk."""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val   = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val   = np.load(os.path.join(data_dir, 'y_val.npy'))
    return X_train, y_train, X_val, y_val


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Wrap numpy arrays into PyTorch DataLoaders with shape (N, 1, 180)."""
    def _make_ds(X, y):
        Xt = torch.FloatTensor(X).unsqueeze(1)  # add channel dim
        yt = torch.LongTensor(y)
        return TensorDataset(Xt, yt)

    train_loader = DataLoader(_make_ds(X_train, y_train),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(_make_ds(X_val,   y_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _run_epoch(model, loader, criterion, optimizer, device, training):
    """One forward pass over the loader; backward pass only when training."""
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y_b.size(0)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += y_b.size(0)

    return total_loss / total, correct / total


def _plot_curves(history, model_name, save_dir):
    """Save loss and accuracy curves as a PNG."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=f'{model_name} - Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'],   label='Val')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title=f'{model_name} - Accuracy')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    tag  = model_name.lower().replace(' ', '_')
    path = os.path.join(save_dir, f'{tag}_training_curves.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Training curves saved to {path}")


def train_model(model, model_name, train_loader, val_loader,
                num_epochs=50, lr=1e-3, patience=10, device=DEVICE):
    """
    Train a model with early stopping and save the best checkpoint.
    Returns (history_dict, checkpoint_path).
    """
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {model_name}")
    print(f"  Device: {device}  Epochs: {num_epochs}  LR: {lr}  Patience: {patience}")
    print(f"{'=' * 60}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    history         = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss   = float('inf')
    no_improve      = 0
    best_epoch      = 0
    tag             = model_name.lower().replace(' ', '_')
    checkpoint_path = os.path.join(MODELS_DIR, f'{tag}_best.pth')

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, optimizer, device, training=False)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"  Epoch {epoch:3d}/{num_epochs}"
              f" | Train loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f" | Val loss {vl_loss:.4f} acc {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_loss': vl_loss,
                        'val_acc':  vl_acc},
                       checkpoint_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  best val loss {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Checkpoint saved to {checkpoint_path}")

    _plot_curves(history, model_name, RESULTS_DIR)
    return history, checkpoint_path


if __name__ == '__main__':
    from models.cnn         import PlainCNN
    from models.heartnet    import HeartNet
    from models.transformer import ECGTransformer

    print(f"Device: {DEVICE}")
    X_train, y_train, X_val, y_val = load_data()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val)

    train_model(PlainCNN(5),       'Plain_CNN',   train_loader, val_loader)
    train_model(HeartNet(5),       'HeartNet',    train_loader, val_loader)
    train_model(ECGTransformer(5), 'Transformer', train_loader, val_loader)
