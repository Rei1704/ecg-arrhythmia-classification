"""
main.py
Entry point for the full HeartNet pipeline.

Stages:
  1. Preprocess  - segment MIT-BIH beats, SMOTE balance, save numpy arrays
  2. Load data   - read saved splits
  3. Train       - PlainCNN, HeartNet, ECGTransformer (skips if checkpoint exists)
  4. Evaluate    - metrics + confusion matrix + ROC curves for each model
  5. Compare     - print formatted comparison table
"""

import os
import sys
import time
import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess             import preprocess
from train                  import load_data, make_dataloaders, train_model, DEVICE
from evaluate               import load_test_data, evaluate_model
from models.cnn             import PlainCNN
from models.heartnet        import HeartNet
from models.transformer     import ECGTransformer

DATA_DIR    = './data'
MODELS_DIR  = './models'
RESULTS_DIR = './results'

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_REGISTRY = [
    (PlainCNN,       'Plain_CNN',   'plain_cnn_best.pth'),
    (HeartNet,       'HeartNet',    'heartnet_best.pth'),
    (ECGTransformer, 'Transformer', 'transformer_best.pth'),
]


def _banner(title):
    w = 65
    print("\n" + "=" * w)
    print(f"{title:^{w}}")
    print("=" * w)


def main():
    t_start = time.time()

    _banner("HeartNet  |  Multi-Class Cardiac Arrhythmia Classification")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : {DEVICE}")
    print(f"  Seed    : {SEED}")

    # stage 1: preprocessing (skip if already done)
    _banner("STAGE 1 - DATA PREPROCESSING")

    expected_files = ['X_train.npy', 'y_train.npy',
                      'X_val.npy',   'y_val.npy',
                      'X_test.npy',  'y_test.npy']
    already_done = all(
        os.path.exists(os.path.join(DATA_DIR, f)) for f in expected_files
    )

    if already_done:
        print("Preprocessed arrays already exist - skipping.")
        print("  (delete ./data/*.npy to force re-preprocessing)")
    else:
        preprocess()

    # stage 2: load data
    _banner("STAGE 2 - LOADING DATA")

    X_train, y_train, X_val, y_val = load_data(DATA_DIR)
    X_test,  y_test                = load_test_data(DATA_DIR)

    print(f"  Train  : {X_train.shape}  labels {y_train.shape}")
    print(f"  Val    : {X_val.shape}    labels {y_val.shape}")
    print(f"  Test   : {X_test.shape}   labels {y_test.shape}")

    # Transformer needs smaller batches to avoid OOM on CPU
    train_loader64, val_loader64 = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=64)
    train_loader32, val_loader32 = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

    # stage 3: train (skip models with existing checkpoints)
    _banner("STAGE 3 - TRAINING MODELS")

    for ModelClass, name, ckpt_file in MODEL_REGISTRY:
        ckpt_path = os.path.join(MODELS_DIR, ckpt_file)
        if os.path.exists(ckpt_path):
            print(f"\n  {name}: checkpoint found - skipping training.")
            print(f"    ({ckpt_path})")
            continue

        model    = ModelClass(num_classes=5)
        n_params = sum(p.numel() for p in model.parameters())

        t_loader = train_loader32 if name == 'Transformer' else train_loader64
        v_loader = val_loader32   if name == 'Transformer' else val_loader64
        bs       = 32 if name == 'Transformer' else 64

        print(f"\n  {name}  |  {n_params:,} parameters  |  batch_size={bs}")

        train_model(
            model, name, t_loader, v_loader,
            num_epochs=50, lr=1e-3, patience=10, device=DEVICE,
        )

    # stage 4: evaluate
    _banner("STAGE 4 - EVALUATING MODELS")

    results = []
    for ModelClass, name, ckpt_file in MODEL_REGISTRY:
        ckpt_path = os.path.join(MODELS_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"\n  WARNING: checkpoint missing for {name} - skipping.")
            continue
        model   = ModelClass(num_classes=5)
        metrics = evaluate_model(model, name, ckpt_path, X_test, y_test)
        results.append(metrics)

    # stage 5: comparison table
    _banner("STAGE 5 - MODEL COMPARISON")

    COL = [18, 10, 10, 11, 8, 9]
    HDR = ['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC']
    SEP = "+" + "+".join("-" * (c + 2) for c in COL) + "+"

    def _row(items):
        return "|" + "|".join(f" {str(v):^{w}} " for v, w in zip(items, COL)) + "|"

    print(SEP)
    print(_row(HDR))
    print(SEP)
    for r in results:
        print(_row([
            r['model'],
            f"{r['accuracy']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['auc_roc']:.4f}",
        ]))
    print(SEP)

    elapsed = time.time() - t_start
    print(f"\n  Total time : {elapsed / 60:.1f} min")
    print(f"  Results    : {os.path.abspath(RESULTS_DIR)}/")
    print("\n  HeartNet pipeline complete.\n")


if __name__ == '__main__':
    main()
