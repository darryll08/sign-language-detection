from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import IMAGE_SIZE, MODEL_PATH, LABEL_MAP_PATH
from src.utils import load_json


def load_trained_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_label_map(label_map_path=LABEL_MAP_PATH):
    label_map_path = Path(label_map_path)
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    return load_json(label_map_path)


def preprocess_pil_image(image: Image.Image, image_size=IMAGE_SIZE) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(image_size)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def preprocess_image_path(image_path: str, image_size=IMAGE_SIZE) -> np.ndarray:
    image = Image.open(image_path)
    return preprocess_pil_image(image, image_size=image_size)


def predict_image(
    model,
    image_path: str,
    label_map: Dict[str, str]
) -> Dict[str, float]:
    image_array = preprocess_image_path(image_path)
    preds = model.predict(image_array, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    pred_label = label_map[str(pred_idx)]
    confidence = float(preds[pred_idx])

    return {
        "label": pred_label,
        "confidence": confidence
    }


def predict_pil_image(
    model,
    image: Image.Image,
    label_map: Dict[str, str]
) -> Dict[str, float]:
    image_array = preprocess_pil_image(image)
    preds = model.predict(image_array, verbose=0)[0]

    pred_idx = int(np.argmax(preds))
    pred_label = label_map[str(pred_idx)]
    confidence = float(preds[pred_idx])

    return {
        "label": pred_label,
        "confidence": confidence
    }


def predict_top_k(
    model,
    image_path: str,
    label_map: Dict[str, str],
    k: int = 3
) -> List[Tuple[str, float]]:
    image_array = preprocess_image_path(image_path)
    preds = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append((label_map[str(int(idx))], float(preds[idx])))

    return results