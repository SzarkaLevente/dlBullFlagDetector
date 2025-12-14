import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

def load_train_val_data(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_dir = os.path.join(data_dir, "processed")

    logger.info(f"Loading processed data from {processed_dir}")

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")

    return X_train, y_train, X_val, y_val


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[DataLoader, DataLoader]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def train():
    logger.info("=== Starting training process ===")
    logger.info(
        f"Config: EPOCHS={config.EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}, "
        f"LR={config.LEARNING_RATE}, PATIENCE={config.EARLY_STOPPING_PATIENCE}"
    )

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_train, y_train, X_val, y_val = load_train_val_data(config.DATA_DIR)

    input_size = X_train.shape[2]
    num_classes = int(np.max(y_train)) + 1

    model = LSTMFlagClassifier(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    num_params = count_parameters(model)
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Number of trainable parameters: {num_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    model_dir = os.path.dirname(config.MODEL_SAVE_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        logger.info(
            f"Epoch {epoch}/{config.EPOCHS} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(
                f"New best model at epoch {epoch}, val_loss={val_loss:.4f}. "
                f"Model saved to {config.MODEL_SAVE_PATH}"
            )
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}"
                )
                break

    logger.info("=== Training process completed (model saved based on validation loss) ===")

if __name__ == "__main__":
    train()