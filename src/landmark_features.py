from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

from src.config import MODELS_DIR

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "mediapipe" / "hand_landmarker.task"


def ensure_hand_landmarker_model(
    model_path: Path = HAND_LANDMARKER_MODEL_PATH,
    model_url: str = HAND_LANDMARKER_MODEL_URL,
) -> Path:
    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, str(model_path))
    return model_path


class HandLandmarkFeatureExtractor:
    """
    Extract 21 hand landmarks, lalu normalisasi agar lebih robust
    terhadap posisi, skala, dan handedness.
    """

    def __init__(
        self,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = ensure_hand_landmarker_model()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def close(self) -> None:
        close_fn = getattr(self.landmarker, "close", None)
        if callable(close_fn):
            close_fn()

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert("RGB")

    def _to_mp_image(self, image: Image.Image) -> mp.Image:
        np_image = np.asarray(image)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

    def _extract_raw_landmarks(
        self,
        image: Image.Image,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        image = self._prepare_image(image)
        mp_image = self._to_mp_image(image)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        hand_landmarks = result.hand_landmarks[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32
        )

        handedness = None
        try:
            handedness = result.handedness[0][0].category_name
        except Exception:
            handedness = None

        return coords, handedness

    def normalize_landmarks(
        self,
        coords: np.ndarray,
        handedness: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalisasi:
        1. mirror jika tangan kiri supaya lebih invariant
        2. center ke wrist
        3. scale berdasarkan jarak maksimum
        """
        coords = coords.copy()

        if handedness == "Left":
            coords[:, 0] = 1.0 - coords[:, 0]

        wrist = coords[0].copy()
        coords = coords - wrist

        scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
        scale = max(scale, 1e-6)
        coords = coords / scale

        return coords.astype(np.float32)

    def extract_features_from_pil(self, image: Image.Image) -> Optional[np.ndarray]:
        coords, handedness = self._extract_raw_landmarks(image)
        if coords is None:
            return None

        coords = self.normalize_landmarks(coords, handedness=handedness)
        features = coords.flatten()
        return features.astype(np.float32)

    def extract_features_from_path(self, image_path: str) -> Optional[np.ndarray]:
        with Image.open(image_path) as image:
            return self.extract_features_from_pil(image)