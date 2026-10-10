"""
preprocess.py
Loads MIT-BIH ECG records, extracts 180-sample beat windows centered
on R-peaks, maps annotations to 5 AAMI classes, splits 70/15/15,
normalizes with StandardScaler, applies SMOTE, and saves numpy arrays.
"""

import os
import numpy as np
import wfdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

DATA_DIR    = './data'
MITBIH_DIR  = os.path.join(DATA_DIR, 'mitbih')

WINDOW_SIZE = 180       # 90 samples before R-peak + 90 after
HALF_WINDOW = WINDOW_SIZE // 2

# AAMI 5-class label mapping (ANSI/AAMI EC57 standard)
BEAT_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'B': 0, 'n': 0,          # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1, 'e': 1, 'j': 1,   # SVE
    'V': 2, 'E': 2,                                     # Ventricular
    'F': 3,                                              # Fusion
    '/': 4, 'f': 4, 'Q': 4,                             # Unknown/paced
}

CLASS_NAMES = ['N (Normal)', 'S (SVE)', 'V (VE)', 'F (Fusion)', 'Q (Unknown)']

MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221', '222',
    '223', '228', '230', '231', '232', '233', '234',
]


def load_mitbih_beats(data_dir, records):
    """Extract 180-sample windows centered on annotated R-peaks from all records."""
    all_beats, all_labels = [], []
    print(f"Loading {len(records)} MIT-BIH records from '{data_dir}'...")

    for rec_id in records:
        rec_path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(rec_path, channels=[0])
            ann    = wfdb.rdann(rec_path, 'atr')
        except Exception as exc:
            print(f"  [SKIP] record {rec_id}: {exc}")
            continue

        signal  = record.p_signal[:, 0].astype(np.float32)
        r_peaks = ann.sample
        symbols = ann.symbol
        n_sig   = len(signal)
        count   = 0

        for peak, sym in zip(r_peaks, symbols):
            if sym not in BEAT_MAP:
                continue
            start = peak - HALF_WINDOW
            end   = peak + HALF_WINDOW
            if start < 0 or end > n_sig:
                continue
            all_beats.append(signal[start:end])
            all_labels.append(BEAT_MAP[sym])
            count += 1

        print(f"  Record {rec_id}: {count} beats")

    X = np.array(all_beats,  dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def preprocess():
    """Run the full preprocessing pipeline and save splits to ./data/."""
    print("=" * 60)
    print("  PREPROCESSING - MIT-BIH ARRHYTHMIA DATABASE")
    print("=" * 60)

    X, y = load_mitbih_beats(MITBIH_DIR, MITBIH_RECORDS)
    print(f"\nTotal beats loaded : {X.shape[0]:,}")
    print(f"Signal window size : {X.shape[1]}")
    print(f"Class distribution : {dict(sorted(Counter(y).items()))}")

    # stratified 70/15/15 split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=SEED, stratify=y_tv
    )
    print(f"\nSplit sizes  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # fit scaler on training data only to avoid data leakage
    print("\nNormalizing with StandardScaler (fit on train set only)...")
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n   = scaler.transform(X_val)
    X_test_n  = scaler.transform(X_test)

    # SMOTE applied to training set only
    print(f"\nApplying SMOTE to balance training classes...")
    print(f"  Before: {dict(sorted(Counter(y_train).items()))}")
    smote = SMOTE(random_state=SEED)
    X_train_r, y_train_r = smote.fit_resample(X_train_n, y_train)
    print(f"  After : {dict(sorted(Counter(y_train_r).items()))}")
    print(f"  Train size after SMOTE: {len(X_train_r):,}")

    os.makedirs(DATA_DIR, exist_ok=True)
    saves = {
        'X_train': X_train_r.astype(np.float32),
        'y_train': y_train_r.astype(np.int64),
        'X_val':   X_val_n.astype(np.float32),
        'y_val':   y_val.astype(np.int64),
        'X_test':  X_test_n.astype(np.float32),
        'y_test':  y_test.astype(np.int64),
    }
    for name, arr in saves.items():
        path = os.path.join(DATA_DIR, f'{name}.npy')
        np.save(path, arr)
        print(f"  Saved {path}  shape={arr.shape}")

    print("\nPreprocessing complete.")
    return (X_train_r, y_train_r, X_val_n, y_val, X_test_n, y_test)


if __name__ == '__main__':
    preprocess()
