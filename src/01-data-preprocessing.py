import os
import json

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR
from utils import setup_logger

logger = setup_logger()

RAW_DATA_DIR = DATA_DIR

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MAX_SEQ_LEN = 64

RANDOM_SEED = 42

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_MAP: Dict[str, int] = {
    "Bullish Normal": 0,
    "Bullish Wedge": 1,
    "Bullish Pennant": 2,
    "Bearish Normal": 3,
    "Bearish Wedge": 4,
    "Bearish Pennant": 5,
}

FILE_MAPPING: Dict[str, str] = {
    "d992bfe2-aapl_5min.csv": "AAPL_5min_001.csv",
    "3f6bd4f4-spy_15min.csv": "SPY_15min_001.csv",
    "400e7f75-spy_15min_2.csv": "SPY_15min_002.csv",
}

def _resolve_csv_filename(file_upload: str) -> str:
    raw_files = set(os.listdir(RAW_DATA_DIR))

    if file_upload in raw_files:
        return file_upload

    if file_upload in FILE_MAPPING:
        return FILE_MAPPING[file_upload]

    logger.warning(
        f"Cannot resolve CSV for file_upload='{file_upload}'. "
        f"Consider adding it to FILE_MAPPING in 01-data-preprocessing.py."
    )
    return None


def _load_labels(labels_path: str) -> List[Dict]:
    logger.info(f"Loading labels from {labels_path}")
    with open(labels_path, "r") as f:
        data = json.load(f)

    all_labels = []

    for task in data:
        file_upload = task.get("file_upload")
        if not file_upload:
            logger.warning("Task without 'file_upload' field encountered, skipping.")
            continue

        csv_filename = _resolve_csv_filename(file_upload)
        if csv_filename is None:
            continue

        annotations = task.get("annotations", [])
        if not annotations:
            logger.info(f"No annotations for file_upload='{file_upload}', skipping.")
            continue

        for ann in annotations:
            results = ann.get("result", [])
            for result in results:
                if result.get("type") != "timeserieslabels":
                    continue

                value = result.get("value", {})
                label_list = value.get("timeserieslabels", [])
                if not label_list:
                    continue

                label_str = label_list[0]
                if label_str not in CLASS_MAP:
                    logger.warning(f"Unknown label '{label_str}', skipping.")
                    continue

                start_str = value.get("start")
                end_str = value.get("end")
                if start_str is None or end_str is None:
                    logger.warning(
                        f"Missing start/end in annotation for file '{csv_filename}', skipping."
                    )
                    continue

                all_labels.append(
                    {
                        "csv_filename": csv_filename,
                        "label_str": label_str,
                        "start_ts": start_str,
                        "end_ts": end_str,
                    }
                )

    logger.info(f"Collected {len(all_labels)} labeled segments from labels.json")
    return all_labels


def _load_price_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading price data from {csv_path}")
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {csv_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        logger.warning(f"Some timestamps could not be parsed in {csv_path}.")

    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df


def _extract_segment(
    df: pd.DataFrame, start_ts_str: str, end_ts_str: str
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_ts_str, errors="coerce")
    end_ts = pd.to_datetime(end_ts_str, errors="coerce")

    if pd.isna(start_ts) or pd.isna(end_ts):
        logger.warning(
            f"Could not parse start/end timestamps ('{start_ts_str}', '{end_ts_str}'), skipping."
        )
        return pd.DataFrame()

    start_ts = start_ts.tz_localize(None)
    end_ts = end_ts.tz_localize(None)

    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    seg_df = df.loc[mask].copy()
    return seg_df


def _normalize_segment(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-8
    return (values - mean) / std


def _pad_or_truncate(values: np.ndarray, max_len: int) -> np.ndarray:
    length, num_features = values.shape

    if length >= max_len:
        return values[-max_len:, :]

    out = np.zeros((max_len, num_features), dtype=np.float32)
    out[:length, :] = values
    return out


def _split_dataset(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(X) == len(y)
    n = len(X)

    indices = np.arange(n)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(indices)

    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(
        f"Dataset split: "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)} (total={n})"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess():
    logger.info("=== Starting data preprocessing ===")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    logger.info(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")

    labels_path = os.path.join(RAW_DATA_DIR, "labels.json")
    if not os.path.exists(labels_path):
        logger.error(f"labels.json not found at {labels_path}")
        return

    all_labels = _load_labels(labels_path)

    if not all_labels:
        logger.error("No valid labels found in labels.json. Nothing to preprocess.")
        return

    segments = []
    label_ids = []

    df_cache: Dict[str, pd.DataFrame] = {}

    for label in all_labels:
        csv_filename = label["csv_filename"]
        csv_path = os.path.join(RAW_DATA_DIR, csv_filename)

        if not os.path.exists(csv_path):
            logger.warning(f"CSV file '{csv_filename}' not found in {RAW_DATA_DIR}, skipping label.")
            continue

        if csv_filename not in df_cache:
            df_cache[csv_filename] = _load_price_data(csv_path)

        df = df_cache[csv_filename]

        seg_df = _extract_segment(df, label["start_ts"], label["end_ts"])
        if seg_df.empty:
            logger.warning(
                f"No rows found for '{csv_filename}' between {label['start_ts']} and {label['end_ts']}, skipping."
            )
            continue

        for col in ["open", "high", "low", "close"]:
            if col not in seg_df.columns:
                logger.error(f"Column '{col}' not found in {csv_filename}.")
                return

        seg_vals = seg_df[["open", "high", "low", "close"]].values.astype(np.float32)

        seg_vals = _normalize_segment(seg_vals)
        seg_vals = _pad_or_truncate(seg_vals, MAX_SEQ_LEN)

        segments.append(seg_vals)
        label_ids.append(CLASS_MAP[label["label_str"]])

    if not segments:
        logger.error("No segments were extracted. Check your labels and CSV alignment.")
        return

    X = np.stack(segments, axis=0)
    y = np.array(label_ids, dtype=np.int64)

    logger.info(
        f"Prepared dataset with {X.shape[0]} samples, "
        f"sequence length {X.shape[1]}, features {X.shape[2]}"
    )

    unique, counts = np.unique(y, return_counts=True)
    dist_str = ", ".join(
        f"class {cls} ({label_name}): {cnt}"
        for cls, cnt, label_name in [
            (u, c, next(k for k, v in CLASS_MAP.items() if v == u))
            for u, c in zip(unique, counts)
        ]
    )
    logger.info(f"Class distribution: {dist_str}")

    X_train, y_train, X_val, y_val, X_test, y_test = _split_dataset(X, y)

    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test)

    meta = {
        "max_seq_len": MAX_SEQ_LEN,
        "num_features": X.shape[2],
        "class_map": CLASS_MAP,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    meta_path = os.path.join(PROCESSED_DATA_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved processed datasets and meta information to {PROCESSED_DATA_DIR}")
    logger.info("=== Data preprocessing completed successfully ===")

if __name__ == "__main__":
    preprocess()