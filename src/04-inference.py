import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as config
from utils import setup_logger, load_meta

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

def class_names_from_map(class_map: dict) -> List[str]:
    inv = {int(v): str(k) for k, v in class_map.items()}
    n = len(inv)
    return [inv.get(i, f"class_{i}") for i in range(n)]

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

    meta = load_meta(config.DATA_DIR)
    if meta is None or "class_map" not in meta or not isinstance(meta["class_map"], dict) or len(meta["class_map"]) == 0:
        logger.error("meta.json missing or invalid. Run preprocessing first.")
        return

    class_names = class_names_from_map(meta["class_map"])
    num_classes = len(class_names)

    X_test = load_example_data(config.DATA_DIR)
    if X_test.shape[0] == 0:
        logger.error("X_test is empty. Nothing to predict on.")
        return

    input_size = int(X_test.shape[2])

    model = LSTMFlagClassifier(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=int(num_classes),
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

    sample_indices = choose_sample_indices(n_samples=5, max_index=int(X_test.shape[0]))
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
        pred_label_name = class_names[pred_class] if 0 <= pred_class < len(class_names) else f"class_{pred_class}"
        top_prob = float(prob_vec[pred_class])

        logger.info(
            f"Sample {idx}: predicted class={pred_class} "
            f"({pred_label_name}), confidence={top_prob:.4f}, "
            f"full_probs={np.round(prob_vec, 4).tolist()}"
        )

    logger.info("=== Inference completed ===")

if __name__ == "__main__":
    predict()