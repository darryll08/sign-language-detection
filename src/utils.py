import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)