import os
import json
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as config
from utils import setup_logger

logger = setup_logger()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LSTMFlagClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

def load_meta_and_class_map(data_dir: str):
    processed_dir = os.path.join(data_dir, "processed")
    meta_path = os.path.join(processed_dir, "meta.json")

    if not os.path.exists(meta_path):
        logger.warning(f"meta.json not found at {meta_path}. "
                       "Class names will be shown as numeric ids only.")
        return None, None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    class_map = meta.get("class_map", {})
    idx_to_label = {int(v): k for k, v in class_map.items()}
    return meta, idx_to_label

def load_example_data(data_dir: str) -> np.ndarray:
    processed_dir = os.path.join(data_dir, "processed")
    X_test_path = os.path.join(processed_dir, "X_test.npy")

    if not os.path.exists(X_test_path):
        raise FileNotFoundError(
            f"X_test.npy not found at {X_test_path}. "
            "Run preprocessing and training first."
        )

    X_test = np.load(X_test_path)
    logger.info(f"Loaded X_test for inference with shape: {X_test.shape}")
    return X_test

def choose_sample_indices(n_samples: int, max_index: int) -> List[int]:
    n = min(n_samples, max_index)
    return list(range(n))

def predict():
    logger.info("=== Starting inference ===")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    meta, idx_to_label = load_meta_and_class_map(config.DATA_DIR)

    X_test = load_example_data(config.DATA_DIR)

    if X_test.shape[0] == 0:
        logger.error("X_test is empty. Nothing to predict on.")
        return

    input_size = X_test.shape[2]
    if meta is not None and "class_map" in meta:
        num_classes = len(meta["class_map"])
    else:
        logger.warning("Could not determine num_classes from meta.json, "
                       "defaulting to 6.")
        num_classes = 6

    model = LSTMFlagClassifier(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        logger.error(
            f"Model file not found at {config.MODEL_SAVE_PATH}. "
            "Run 02-training.py before running inference."
        )
        return

    logger.info(f"Loading model weights from {config.MODEL_SAVE_PATH}")
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sample_indices = choose_sample_indices(n_samples=5, max_index=X_test.shape[0])
    logger.info(f"Running inference on sample indices: {sample_indices}")

    X_samples = X_test[sample_indices]
    X_t = torch.tensor(X_samples, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_t)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    for i, idx in enumerate(sample_indices):
        pred_class = int(preds_np[i])
        prob_vec = probs_np[i]

        if idx_to_label is not None and pred_class in idx_to_label:
            pred_label_name = idx_to_label[pred_class]
        else:
            pred_label_name = f"class_{pred_class}"

        top_prob = float(prob_vec[pred_class])

        logger.info(
            f"Sample {idx}: predicted class={pred_class} "
            f"({pred_label_name}), confidence={top_prob:.4f}, "
            f"full_probs={prob_vec.round(4).tolist()}"
        )
    logger.info("=== Inference completed ===")

if __name__ == "__main__":
    predict()