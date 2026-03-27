"""
landmark_inference.py — Improved version

Changes vs original:
- predict_landmark_pil now accepts use_video_mode param
  so the live endpoint gets proper temporal tracking
- EMA (Exponential Moving Average) smoothing on probability vector
  added as optional — call LandmarkSmoother and pass its state in
- Minimum confidence gate: if top confidence < LOW_CONFIDENCE_THRESHOLD
  we return 'nothing' even when a hand is detected — prevents junk labels
- "nothing" label is now a real class from the model (trained explicitly)
  instead of a hard-coded rule
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from PIL import Image

from src.config import MODELS_DIR
from src.landmark_features import HandLandmarkFeatureExtractor

LANDMARK_MODEL_PATH = MODELS_DIR / "landmark_mlp.joblib"
LANDMARK_LABEL_ENCODER_PATH = MODELS_DIR / "landmark_label_encoder.joblib"

# Below this confidence, we treat prediction as "nothing"
# (hand detected but pose ambiguous)
LOW_CONFIDENCE_THRESHOLD = 0.45


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


class LandmarkSmoother:
    """
    Exponential Moving Average over the probability vector output.

    Usage:
        smoother = LandmarkSmoother(alpha=0.4)
        # In each frame:
        smoothed_probs = smoother.update(raw_probs)

    A higher alpha = more reactive (follows current frame more).
    A lower alpha = smoother (relies more on history).
    """

    def __init__(self, alpha: float = 0.4, num_classes: Optional[int] = None):
        self.alpha = alpha
        self._ema: Optional[np.ndarray] = None
        self._num_classes = num_classes

    def reset(self) -> None:
        self._ema = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        if self._ema is None:
            self._ema = probs.copy()
        else:
            self._ema = self.alpha * probs + (1.0 - self.alpha) * self._ema
        return self._ema.copy()

    def current(self) -> Optional[np.ndarray]:
        return self._ema.copy() if self._ema is not None else None


def predict_landmark_pil(
    model,
    label_encoder,
    extractor: HandLandmarkFeatureExtractor,
    image: Image.Image,
    top_k: int = 3,
    use_video_mode: bool = False,
    smoother: Optional[LandmarkSmoother] = None,
) -> Dict:
    """
    Full inference on a PIL image.

    Args:
        model: sklearn Pipeline (StandardScaler + MLPClassifier)
        label_encoder: LabelEncoder
        extractor: HandLandmarkFeatureExtractor
        image: PIL image
        top_k: how many top predictions to return
        use_video_mode: if True, use VIDEO running mode (for live streams)
        smoother: optional LandmarkSmoother for temporal EMA smoothing

    Returns dict with:
        label, confidence, topk, used_landmarks, hand_detected
    """
    features = extractor.extract_features_from_pil(
        image, use_video_mode=use_video_mode
    )

    if features is None:
        # No hand detected at all
        if smoother is not None:
            smoother.reset()
        return {
            "label": "nothing",
            "confidence": 1.0,
            "topk": [{"label": "nothing", "confidence": 1.0}],
            "used_landmarks": False,
            "hand_detected": False,
        }

    # Raw probabilities from MLP
    raw_probs = model.predict_proba([features])[0]

    # Apply EMA smoothing if smoother is provided
    if smoother is not None:
        probs = smoother.update(raw_probs)
    else:
        probs = raw_probs

    top_indices = np.argsort(probs)[::-1][:top_k]

    topk: List[Dict] = []
    for idx in top_indices:
        decoded_label = label_encoder.inverse_transform([int(idx)])[0]
        topk.append(
            {
                "label": str(decoded_label),
                "confidence": float(probs[idx]),
            }
        )

    best = topk[0]
    best_label = best["label"]
    best_conf = best["confidence"]

    # Low confidence gate: hand detected but pose too ambiguous
    if best_conf < LOW_CONFIDENCE_THRESHOLD and best_label != "nothing":
        best_label = "nothing"
        best_conf = 1.0 - best_conf  # reflect uncertainty

    return {
        "label": best_label,
        "confidence": best_conf,
        "topk": topk,
        "used_landmarks": True,
        "hand_detected": True,
    }
