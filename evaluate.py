"""
evaluate.py
Loads saved model checkpoints, runs inference on the held-out test set,
and computes accuracy, macro F1, precision, recall, and AUC-ROC.
Saves confusion matrix and ROC curve plots to ./results/.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR    = './data'
MODELS_DIR  = './models'
RESULTS_DIR = './results'

CLASS_NAMES = ['N (Normal)', 'S (SVE)', 'V (VE)', 'F (Fusion)', 'Q (Unknown)']
SHORT_NAMES = ['N', 'S', 'V', 'F', 'Q']
PALETTE     = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data(data_dir=DATA_DIR):
    """Load the held-out test split."""
    X = np.load(os.path.join(data_dir, 'X_test.npy'))
    y = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X, y


def get_predictions(model, X_test, batch_size=256):
    """Run batched inference and return (predicted labels, class probabilities)."""
    model.eval()
    model = model.to(DEVICE)

    Xt  = torch.FloatTensor(X_test).unsqueeze(1)
    dl  = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb.to(DEVICE))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1)
    return preds, probs


def _plot_confusion_matrix(y_true, y_pred, model_name):
    """Save a row-normalized confusion matrix as a PNG."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES, ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title(f'{model_name} - Confusion Matrix (row-normalized)')
    plt.tight_layout()

    tag  = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{tag}_confusion_matrix.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Confusion matrix saved to {path}")


def _plot_roc_curves(y_true, y_probs, model_name):
    """Save per-class one-vs-rest ROC curves as a PNG."""
    n_classes = y_probs.shape[1]
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (cname, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{cname} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} - ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tag  = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{tag}_roc_curves.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  ROC curves saved to {path}")


def evaluate_model(model, model_name, checkpoint_path, X_test, y_test):
    """
    Load a checkpoint, run evaluation, save plots.
    Returns a dict with keys: model, accuracy, f1, precision, recall, auc_roc.
    """
    print(f"\n{'=' * 60}")
    print(f"  EVALUATING: {model_name}")
    print(f"{'=' * 60}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded epoch {ckpt['epoch']} checkpoint "
          f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.4f})")

    y_pred, y_probs = get_predictions(model, X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

    try:
        y_bin   = label_binarize(y_test, classes=list(range(5)))
        auc_roc = roc_auc_score(y_bin, y_probs, multi_class='ovr', average='macro')
    except Exception as exc:
        print(f"  AUC-ROC error: {exc}")
        auc_roc = float('nan')

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}  (macro)")
    print(f"  Precision : {prec:.4f} (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  AUC-ROC   : {auc_roc:.4f} (macro OvR)")
    print(f"\n  Per-class Report:\n")
    print(classification_report(y_test, y_pred, target_names=SHORT_NAMES, zero_division=0))

    _plot_confusion_matrix(y_test, y_pred, model_name)
    _plot_roc_curves(y_test, y_probs, model_name)

    return dict(model=model_name, accuracy=acc, f1=f1,
                precision=prec, recall=rec, auc_roc=auc_roc)


if __name__ == '__main__':
    from models.cnn         import PlainCNN
    from models.heartnet    import HeartNet
    from models.transformer import ECGTransformer

    X_test, y_test = load_test_data()
    print(f"Test set: {X_test.shape}")

    configs = [
        (PlainCNN,       'Plain_CNN',   os.path.join(MODELS_DIR, 'plain_cnn_best.pth')),
        (HeartNet,       'HeartNet',    os.path.join(MODELS_DIR, 'heartnet_best.pth')),
        (ECGTransformer, 'Transformer', os.path.join(MODELS_DIR, 'transformer_best.pth')),
    ]

    results = []
    for ModelClass, name, ckpt in configs:
        m = evaluate_model(ModelClass(5), name, ckpt, X_test, y_test)
        results.append(m)

    print("\n" + "=" * 72)
    print(f"{'MODEL COMPARISON':^72}")
    print("=" * 72)
    print(f"{'Model':<18} {'Accuracy':>10} {'F1-Score':>10} "
          f"{'Precision':>10} {'Recall':>10} {'AUC-ROC':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['model']:<18} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['auc_roc']:>10.4f}")
    print("=" * 72)
