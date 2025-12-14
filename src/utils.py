import logging
import sys
import os
import json

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def load_meta(data_dir: str) -> dict:
    processed_dir = os.path.join(data_dir, "processed")
    meta_path = os.path.join(processed_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)