from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from PIL import Image

from src.config import MODELS_DIR
from src.landmark_features import HandLandmarkFeatureExtractor

LANDMARK_MODEL_PATH = MODELS_DIR / "landmark_mlp.joblib"
LANDMARK_LABEL_ENCODER_PATH = MODELS_DIR / "landmark_label_encoder.joblib"


def load_landmark_model(model_path=LANDMARK_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Landmark model not found: {model_path}")
    return joblib.load(model_path)


def load_landmark_label_encoder(label_encoder_path=LANDMARK_LABEL_ENCODER_PATH):
    label_encoder_path = Path(label_encoder_path)
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Landmark label encoder not found: {label_encoder_path}")
    return joblib.load(label_encoder_path)


def predict_landmark_pil(
    model,
    label_encoder,
    extractor: HandLandmarkFeatureExtractor,
    image: Image.Image,
    top_k: int = 3
) -> Dict:
    features = extractor.extract_features_from_pil(image)

    if features is None:
        return {
            "label": "nothing",
            "confidence": 1.0,
            "topk": [{"label": "nothing", "confidence": 1.0}],
            "used_landmarks": False,
        }

    probs = model.predict_proba([features])[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    topk: List[Dict] = []
    for idx in top_indices:
        decoded_label = label_encoder.inverse_transform([int(idx)])[0]
        topk.append({
            "label": str(decoded_label),
            "confidence": float(probs[idx]),
        })

    best = topk[0]

    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "topk": topk,
        "used_landmarks": True,
    }