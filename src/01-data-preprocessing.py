import os
import json
from typing import List, Dict, Tuple, Optional

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

ADD_NO_FLAG = True
NO_FLAG_RATIO = 2.0
NO_FLAG_BUFFER_BARS = 32
NO_FLAG_MAX_PER_FILE = 500

CLASS_MAP: Dict[str, int] = {
    "Bullish Normal": 0,
    "Bullish Wedge": 1,
    "Bullish Pennant": 2,
    "Bearish Normal": 3,
    "Bearish Wedge": 4,
    "Bearish Pennant": 5,
    "No Flag": 6,
}

FILE_MAPPING: Dict[str, str] = {
    "d992bfe2-aapl_5min.csv": "AAPL_5min_001.csv",
    "3f6bd4f4-spy_15min.csv": "SPY_15min_001.csv",
    "400e7f75-spy_15min_2.csv": "SPY_15min_002.csv",
}

def _list_csv_files() -> List[str]:
    if not os.path.isdir(RAW_DATA_DIR):
        return []
    return [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(".csv")]

def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    if s.endswith(".csv"):
        s = s[:-4]
    for ch in [" ", "_", "-", ".", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    return s

def _strip_hash_prefix(name: str) -> str:
    base = name
    if base.lower().endswith(".csv"):
        base = base[:-4]
    if "-" in base:
        base = base.split("-", 1)[1]
    return base + ".csv"

def _resolve_csv_filename(file_upload: str) -> Optional[str]:
    csv_files = _list_csv_files()
    csv_set = set(csv_files)
    if file_upload in csv_set:
        return file_upload

    if file_upload in FILE_MAPPING and FILE_MAPPING[file_upload] in csv_set:
        return FILE_MAPPING[file_upload]

    candidate = _strip_hash_prefix(file_upload)
    if candidate in csv_set:
        return candidate
    if candidate in FILE_MAPPING and FILE_MAPPING[candidate] in csv_set:
        return FILE_MAPPING[candidate]

    fu_norm = _normalize_name(file_upload)
    if fu_norm:
        for f in csv_files:
            if _normalize_name(f) == fu_norm:
                return f

        for f in csv_files:
            if fu_norm in _normalize_name(f) or _normalize_name(f) in fu_norm:
                return f

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
                if label_str not in CLASS_MAP or label_str == "No Flag":
                    logger.warning(f"Unknown label '{label_str}', skipping.")
                    continue

                start_str = value.get("start")
                end_str = value.get("end")
                if start_str is None or end_str is None:
                    logger.warning(f"Missing start/end in annotation for file '{csv_filename}', skipping.")
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

    ts = df["timestamp"]

    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, errors="coerce", unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, errors="coerce", utc=True)

    if df["timestamp"].isna().any():
        logger.warning(f"Some timestamps could not be parsed in {csv_path}.")

    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def _nearest_idx_for_ts(ts_series: pd.Series, ts: pd.Timestamp) -> int:
    arr = ts_series.to_numpy(dtype="datetime64[ns]")
    ts64 = np.datetime64(ts.to_datetime64())

    pos = int(np.searchsorted(arr, ts64))
    if pos <= 0:
        return 0
    if pos >= len(arr):
        return len(arr) - 1

    before = arr[pos - 1]
    after = arr[pos]

    if abs(after - ts64) < abs(ts64 - before):
        return pos
    return pos - 1

def _extract_segment_with_indices(
    df: pd.DataFrame, start_ts_str: str, end_ts_str: str
) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
    start_ts = pd.to_datetime(start_ts_str, errors="coerce", utc=True)
    end_ts = pd.to_datetime(end_ts_str, errors="coerce", utc=True)

    if pd.isna(start_ts) or pd.isna(end_ts):
        logger.warning(f"Could not parse start/end timestamps ('{start_ts_str}', '{end_ts_str}'), skipping.")
        return pd.DataFrame(), None, None

    start_ts = start_ts.tz_convert(None)
    end_ts = end_ts.tz_convert(None)

    if len(df) == 0:
        return pd.DataFrame(), None, None

    start_idx = _nearest_idx_for_ts(df["timestamp"], start_ts)
    end_idx = _nearest_idx_for_ts(df["timestamp"], end_ts)

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    seg_df = df.iloc[start_idx : end_idx + 1].copy()
    return seg_df, start_idx, end_idx

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

def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _window_overlaps_any(start: int, end: int, intervals: List[Tuple[int, int]]) -> bool:
    for s, e in intervals:
        if start <= e and end >= s:
            return True
    return False

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

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(
        f"Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} (total={n})"
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
    forbidden_by_file: Dict[str, List[Tuple[int, int]]] = {}
    pos_count_by_file: Dict[str, int] = {}

    required_cols = ["open", "high", "low", "close"]

    for label in all_labels:
        csv_filename = label["csv_filename"]
        csv_path = os.path.join(RAW_DATA_DIR, csv_filename)

        if not os.path.exists(csv_path):
            logger.warning(f"CSV file '{csv_filename}' not found in {RAW_DATA_DIR}, skipping label.")
            continue

        if csv_filename not in df_cache:
            df_cache[csv_filename] = _load_price_data(csv_path)

        df = df_cache[csv_filename]

        if any(col not in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.error(f"Missing columns {missing} in {csv_filename}.")
            return

        seg_df, start_idx, end_idx = _extract_segment_with_indices(df, label["start_ts"], label["end_ts"])
        if seg_df.empty or start_idx is None or end_idx is None:
            logger.warning(
                f"No rows found for '{csv_filename}' between {label['start_ts']} and {label['end_ts']}, skipping."
            )
            continue

        seg_vals = seg_df[required_cols].values.astype(np.float32)
        seg_vals = _normalize_segment(seg_vals)
        seg_vals = _pad_or_truncate(seg_vals, MAX_SEQ_LEN)

        segments.append(seg_vals)
        label_ids.append(CLASS_MAP[label["label_str"]])

        s = max(0, start_idx - NO_FLAG_BUFFER_BARS)
        e = min(len(df) - 1, end_idx + NO_FLAG_BUFFER_BARS)
        forbidden_by_file.setdefault(csv_filename, []).append((s, e))
        pos_count_by_file[csv_filename] = pos_count_by_file.get(csv_filename, 0) + 1

    if not segments:
        logger.error("No segments were extracted. Check your labels and CSV alignment.")
        return

    if ADD_NO_FLAG:
        rng = np.random.default_rng(RANDOM_SEED)
        for csv_filename, intervals in forbidden_by_file.items():
            df = df_cache.get(csv_filename)
            if df is None or len(df) < MAX_SEQ_LEN:
                continue

            intervals = _merge_intervals(intervals)
            max_start = len(df) - MAX_SEQ_LEN
            if max_start <= 0:
                continue

            valid_starts = []
            for s in range(0, max_start + 1):
                e = s + MAX_SEQ_LEN - 1
                if not _window_overlaps_any(s, e, intervals):
                    valid_starts.append(s)

            if not valid_starts:
                continue

            n_pos = pos_count_by_file.get(csv_filename, 0)
            n_neg_target = int(round(n_pos * NO_FLAG_RATIO))
            n_neg = min(n_neg_target, NO_FLAG_MAX_PER_FILE, len(valid_starts))
            if n_neg <= 0:
                continue

            chosen = rng.choice(np.array(valid_starts, dtype=np.int64), size=n_neg, replace=False)
            for s in chosen.tolist():
                window_df = df.iloc[s : s + MAX_SEQ_LEN]
                vals = window_df[required_cols].values.astype(np.float32)
                vals = _normalize_segment(vals)
                segments.append(vals)
                label_ids.append(CLASS_MAP["No Flag"])

    X = np.stack(segments, axis=0)
    y = np.array(label_ids, dtype=np.int64)

    logger.info(
        f"Prepared dataset with {X.shape[0]} samples, "
        f"sequence length {X.shape[1]}, features {X.shape[2]}"
    )

    unique, counts = np.unique(y, return_counts=True)
    inv_class_map = {v: k for k, v in CLASS_MAP.items()}
    dist_str = ", ".join(f"class {u} ({inv_class_map.get(u, 'unknown')}): {c}" for u, c in zip(unique, counts))
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
        "num_features": int(X.shape[2]),
        "class_map": CLASS_MAP,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "add_no_flag": bool(ADD_NO_FLAG),
        "no_flag_ratio": float(NO_FLAG_RATIO),
        "no_flag_buffer_bars": int(NO_FLAG_BUFFER_BARS),
    }
    meta_path = os.path.join(PROCESSED_DATA_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved processed datasets and meta information to {PROCESSED_DATA_DIR}")
    logger.info("=== Data preprocessing completed successfully ===")

if __name__ == "__main__":
    preprocess()
