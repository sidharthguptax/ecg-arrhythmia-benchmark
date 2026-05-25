import os
import wfdb
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml


# MIT-BIH record IDs
MITBIH_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

# AAMI standard mapping: raw MIT-BIH symbols → 5 classes
AAMI_MAP = {
    'N': 'N', '.': 'N', 'e': 'N', 'j': 'N',        # Normal
    'L': 'L',                                          # LBBB
    'R': 'R',                                          # RBBB
    'V': 'V', 'E': 'V',                               # PVC
    'A': 'A', 'a': 'A', 'S': 'A', 'J': 'A',         # APC
}

LABEL_TO_IDX = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4}


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_data(raw_dir):
    """Download all MIT-BIH records into raw_dir using wfdb."""
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Downloading {len(MITBIH_RECORDS)} records to '{raw_dir}'...")
    for rec_id in MITBIH_RECORDS:
        rec_name = str(rec_id)
        out_path = os.path.join(raw_dir, rec_name)
        if os.path.exists(out_path + '.dat'):
            continue  # already downloaded
        try:
            wfdb.dl_database('mitdb', dl_dir=raw_dir, records=[rec_name])
        except Exception as e:
            print(f"  Warning: could not download record {rec_id}: {e}")
    print("Download complete.")


def segment_record(rec_id, raw_dir, half_len=90):
    """
    Extract heartbeat segments from a single record.
    Each segment: signal[r_peak - half_len : r_peak + half_len]
    Returns segments (N, 2*half_len) and labels (N,) as AAMI class indices.
    """
    rec_path = os.path.join(raw_dir, str(rec_id))
    try:
        record = wfdb.rdrecord(rec_path)
        annotation = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        print(f"  Skipping record {rec_id}: {e}")
        return np.array([]), np.array([])

    signal = record.p_signal[:, 0]  # use lead II (channel 0)
    r_peaks = annotation.sample
    symbols = annotation.symbol
    seg_len = 2 * half_len

    segments, labels = [], []
    for peak, sym in zip(r_peaks, symbols):
        if sym not in AAMI_MAP:
            continue
        start, end = peak - half_len, peak + half_len
        if start < 0 or end > len(signal):
            continue
        seg = signal[start:end]
        segments.append(seg)
        labels.append(LABEL_TO_IDX[AAMI_MAP[sym]])

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


def normalize(segments):
    """Z-score normalisation per segment."""
    mean = segments.mean(axis=1, keepdims=True)
    std = segments.std(axis=1, keepdims=True) + 1e-8
    return (segments - mean) / std


def build_dataset(raw_dir, half_len=90):
    """Load and segment all records, return normalized X and y."""
    all_segs, all_labels = [], []
    for rec_id in MITBIH_RECORDS:
        segs, labels = segment_record(rec_id, raw_dir, half_len)
        if len(segs) == 0:
            continue
        all_segs.append(segs)
        all_labels.append(labels)

    X = np.concatenate(all_segs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    X = normalize(X)

    print(f"\nDataset built: {X.shape[0]} segments, {X.shape[1]} samples each")
    print("Class distribution:", {k: v for k, v in sorted(Counter(y).items())})
    return X, y


def split_and_resample(X, y, val_size=0.15, test_size=0.15, random_seed=42):
    """Train/val/test split then SMOTE on train only."""
    # First split off test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )
    # Then split val from remaining
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=random_seed
    )

    print(f"\nBefore SMOTE — Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # SMOTE on train split only
    smote = SMOTE(random_state=random_seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE  — Train: {len(y_train)}")
    print("Train class distribution:", {k: v for k, v in sorted(Counter(y_train).items())})

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(processed_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'X_val.npy'),   X_val)
    np.save(os.path.join(processed_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_val.npy'),   y_val)
    np.save(os.path.join(processed_dir, 'y_test.npy'),  y_test)
    print(f"\nSplits saved to '{processed_dir}'")


def load_splits(processed_dir):
    return (
        np.load(os.path.join(processed_dir, 'X_train.npy')),
        np.load(os.path.join(processed_dir, 'X_val.npy')),
        np.load(os.path.join(processed_dir, 'X_test.npy')),
        np.load(os.path.join(processed_dir, 'y_train.npy')),
        np.load(os.path.join(processed_dir, 'y_val.npy')),
        np.load(os.path.join(processed_dir, 'y_test.npy')),
    )


if __name__ == '__main__':
    cfg = load_config()
    download_data(cfg['data']['raw_dir'])
    X, y = build_dataset(cfg['data']['raw_dir'], half_len=cfg['data']['segment_length'] // 2)
    splits = split_and_resample(X, y, cfg['data']['val_size'], cfg['data']['test_size'], cfg['training']['random_seed'] if 'random_seed' in cfg.get('training', {}) else cfg['data']['random_seed'])
    save_splits(cfg['data']['processed_dir'], *splits)