import os

EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 800
BATCH_SIZE = 32
LEARNING_RATE = 0.001

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

MODEL_SAVE_PATH = os.path.join(os.path.dirname(BASE_DIR), "model.pth")