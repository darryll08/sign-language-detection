from __future__ import annotations

import base64
import io
import threading
import urllib.request
from dataclasses import dataclass
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


@dataclass
class HandROIResult:
    ok: bool
    message: str
    roi_image: Optional[Image.Image] = None
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
    handedness: Optional[str] = None


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


def pil_to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class HandROIDetector:
    def __init__(
        self,
        margin_ratio: float = 0.30,
        min_roi_size: int = 80,
        min_hand_detection_confidence: float = 0.65,
        min_hand_presence_confidence: float = 0.65,
        min_tracking_confidence: float = 0.60,
    ):
        self.model_path = ensure_hand_landmarker_model()
        self.margin_ratio = margin_ratio
        self.min_roi_size = min_roi_size

        self._lock = threading.Lock()
        self._video_timestamp_ms = 0

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        self._image_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

        self._video_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def close(self) -> None:
        for landmarker in [self._image_landmarker, self._video_landmarker]:
            close_fn = getattr(landmarker, "close", None)
            if callable(close_fn):
                close_fn()

    def _prepare_pil(self, image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert("RGB")

    def _to_mp_image(self, image: Image.Image) -> mp.Image:
        np_image = np.asarray(image)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

    def _extract_handedness(self, result) -> Optional[str]:
        try:
            return result.handedness[0][0].category_name
        except Exception:
            return None

    def _compute_square_bbox(
        self,
        hand_landmarks,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        x_min = max(0.0, min(xs))
        x_max = min(1.0, max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(1.0, max(ys))

        box_w = max(1, int((x_max - x_min) * width))
        box_h = max(1, int((y_max - y_min) * height))

        size = int(max(box_w, box_h) * (1.0 + self.margin_ratio))
        size = max(size, self.min_roi_size)
        size = min(size, width, height)

        cx = int(((x_min + x_max) / 2.0) * width)
        cy = int(((y_min + y_max) / 2.0) * height)

        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > width:
            shift = x2 - width
            x1 -= shift
            x2 = width
        if y2 > height:
            shift = y2 - height
            y1 -= shift
            y2 = height

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < self.min_roi_size or (y2 - y1) < self.min_roi_size:
            return None

        return int(x1), int(y1), int(x2), int(y2)

    def extract_hand_roi(
        self,
        image: Image.Image,
        use_video_mode: bool = False,
    ) -> HandROIResult:
        prepared = self._prepare_pil(image)
        width, height = prepared.size
        mp_image = self._to_mp_image(prepared)

        with self._lock:
            if use_video_mode:
                self._video_timestamp_ms += 33
                result = self._video_landmarker.detect_for_video(
                    mp_image,
                    self._video_timestamp_ms,
                )
            else:
                result = self._image_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return HandROIResult(
                ok=False,
                message="No hand detected. Letakkan satu tangan lebih jelas di dalam kotak.",
            )

        bbox = self._compute_square_bbox(result.hand_landmarks[0], width, height)
        if bbox is None:
            return HandROIResult(
                ok=False,
                message="Hand detected but ROI is too small. Dekatkan tangan ke kamera.",
            )

        roi_image = prepared.crop(bbox)
        handedness = self._extract_handedness(result)

        return HandROIResult(
            ok=True,
            message="ok",
            roi_image=roi_image,
            roi_bbox=bbox,
            handedness=handedness,
        )