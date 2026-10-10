"""
download_data.py - Downloads the MIT-BIH Arrhythmia Database from PhysioNet.

Usage:
    python download_data.py

Downloads ~100 MB of .hea, .dat, and .atr files to ./data/mitbih/.
Skip this step if the files are already present.
Source: https://physionet.org/content/mitdb/1.0.0/
"""

import os
import wfdb

DATA_DIR = './data/mitbih'

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading MIT-BIH Arrhythmia Database from PhysioNet...")
print("This may take a few minutes (~100 MB)...")

wfdb.dl_database('mitdb', DATA_DIR)

files = os.listdir(DATA_DIR)
print(f"\nDownload complete — {len(files)} files saved to {DATA_DIR}/")
print("You can now run: python main.py")
