import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

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
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

def load_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    processed_dir = os.path.join(data_dir, "processed")

    logger.info(f"Loading test data from {processed_dir}")

    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test

def evaluate_on_numpy(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    
    criterion = nn.CrossEntropyLoss()
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(X_t)
        loss = criterion(logits, y_t)
        preds = torch.argmax(logits, dim=1)

    correct = (preds == y_t).sum().item()
    total = y_t.size(0)
    acc = correct / total if total > 0 else 0.0

    return loss.item(), acc, y_t.cpu().numpy(), preds.cpu().numpy()

def evaluate():
    logger.info("=== Starting evaluation on test set ===")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_test, y_test = load_test_data(config.DATA_DIR)

    input_size = X_test.shape[2]
    num_classes = int(np.max(y_test)) + 1

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
            "Run 02-training.py before 03-evaluation.py."
        )
        return

    logger.info(f"Loading model weights from {config.MODEL_SAVE_PATH}")
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)

    test_loss, test_acc, y_true, y_pred = evaluate_on_numpy(
        model, X_test, y_test, device
    )

    logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"Classification report:\n{report}")

    logger.info("=== Evaluation on test set completed ===")


if __name__ == "__main__":
    evaluate()