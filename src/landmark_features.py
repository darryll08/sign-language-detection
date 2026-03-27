"""
landmark_features.py — Improved version

Changes vs original:
- Feature vector expanded from 63 (raw coords) to ~120 features:
    * 63 normalized (x,y,z) coords  (same as before)
    * 20 relative fingertip distances to wrist (scale-invariant)
    * 15 inter-fingertip pairwise distances (thumb-index, index-middle, ...)
    * 20 finger-joint angle features (flexion per bone segment)
- Normalization uses middle-finger MCP (landmark 9) as secondary anchor
  instead of only wrist, making scale more stable
- Z-axis normalized separately (depth) for better depth robustness
- VIDEO mode landmarker created alongside IMAGE mode
  so live inference gets proper temporal tracking from MediaPipe
"""

from __future__ import annotations

import threading
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

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_BONES = [
    # (base, mid, tip) for each finger — used for angle calculation
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
]


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


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine angle (radians) between two vectors. Handles zero-length gracefully."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def _build_extra_features(coords: np.ndarray) -> np.ndarray:
    """
    Build extra geometric features beyond raw (x,y,z) coordinates.

    Returns a 1D float32 array of:
    - 5 wrist-to-tip distances (normalized by hand scale)
    - 15 inter-fingertip pairwise distances
    - 5×3 = 15 per-bone angle features (one per joint, per finger)
    - 5 fingertip y-positions relative to their MCP (finger curled vs extended)
    Total: 5 + 15 + 15 + 5 = 40 extra features
    """
    features = []

    # Hand scale: distance from wrist to middle-finger MCP
    hand_scale = np.linalg.norm(coords[MIDDLE_MCP, :2] - coords[WRIST, :2])
    hand_scale = max(hand_scale, 1e-6)

    # 5 wrist-to-fingertip distances (xy plane, scale-normalised)
    for tip_idx in FINGERTIPS:
        d = np.linalg.norm(coords[tip_idx, :2] - coords[WRIST, :2]) / hand_scale
        features.append(d)

    # 15 inter-fingertip pairwise distances (combinations C(5,2)=10... we do sequential pairs + cross)
    tips = [coords[i, :2] for i in FINGERTIPS]
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            d = np.linalg.norm(tips[i] - tips[j]) / hand_scale
            features.append(d)  # 10 features

    # Add 5 more: each fingertip to index-tip (good for sign shapes)
    index_tip = coords[INDEX_TIP, :2]
    for tip_idx in [THUMB_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST]:
        d = np.linalg.norm(coords[tip_idx, :2] - index_tip) / hand_scale
        features.append(d)  # 5 more → total 15

    # 15 joint-bend angles (3 per finger × 5 fingers)
    for bone_indices in FINGER_BONES:
        for k in range(len(bone_indices) - 2):
            a = coords[bone_indices[k], :3]
            b = coords[bone_indices[k + 1], :3]
            c = coords[bone_indices[k + 2], :3]
            v1 = a - b
            v2 = c - b
            angle = _angle_between(v1, v2)
            features.append(angle / np.pi)  # normalize to [0,1]

    # 5 fingertip elevation (y relative to MCP, positive = extended upward)
    mcp_indices = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    for tip_idx, mcp_idx in zip(FINGERTIPS, mcp_indices):
        elev = (coords[mcp_idx, 1] - coords[tip_idx, 1]) / hand_scale
        features.append(elev)

    return np.array(features, dtype=np.float32)


class HandLandmarkFeatureExtractor:
    """
    Extract 21 hand landmarks, normalize, then build a rich feature vector
    combining raw coordinates + geometric/angular features.

    Supports two running modes:
    - IMAGE mode  (default) — for single captures, independent frames
    - VIDEO mode            — for live streams, enables temporal tracking
                              which improves detection consistency across frames
    """

    def __init__(
        self,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_video_mode: bool = False,
    ):
        model_path = ensure_hand_landmarker_model()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        # Always create IMAGE landmarker
        self.image_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

        # VIDEO landmarker — only if requested (live mode)
        self.video_landmarker = None
        self._video_lock = threading.Lock()
        self._video_timestamp_ms = 0

        if enable_video_mode:
            self.video_landmarker = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=min_hand_detection_confidence,
                    min_hand_presence_confidence=min_hand_presence_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            )

    def close(self) -> None:
        for landmarker in [self.image_landmarker, self.video_landmarker]:
            if landmarker is None:
                continue
            close_fn = getattr(landmarker, "close", None)
            if callable(close_fn):
                close_fn()

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert("RGB")

    def _to_mp_image(self, image: Image.Image) -> mp.Image:
        np_image = np.asarray(image)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

    def _detect(self, image: Image.Image, use_video_mode: bool = False):
        """Run MediaPipe detection. Returns raw result object."""
        mp_image = self._to_mp_image(image)

        if use_video_mode and self.video_landmarker is not None:
            with self._video_lock:
                self._video_timestamp_ms += 30  # ~33fps
                result = self.video_landmarker.detect_for_video(
                    mp_image,
                    self._video_timestamp_ms,
                )
        else:
            result = self.image_landmarker.detect(mp_image)

        return result

    def _extract_raw_landmarks(
        self,
        image: Image.Image,
        use_video_mode: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        image = self._prepare_image(image)
        result = self._detect(image, use_video_mode=use_video_mode)

        if not result.hand_landmarks:
            return None, None

        hand_landmarks = result.hand_landmarks[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )

        handedness = None
        try:
            handedness = result.handedness[0][0].category_name
        except Exception:
            pass

        return coords, handedness

    def normalize_landmarks(
        self,
        coords: np.ndarray,
        handedness: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalize raw 21-landmark coords:
        1. Mirror left hand so model sees everything as right-hand
        2. Center to wrist
        3. Scale by max 2D distance from wrist (x,y plane)
        4. Normalize Z separately by max depth range
        """
        coords = coords.copy()

        # Step 1: Mirror left hand
        if handedness == "Left":
            coords[:, 0] = 1.0 - coords[:, 0]

        # Step 2: Center to wrist
        wrist = coords[WRIST].copy()
        coords -= wrist

        # Step 3: XY scale
        xy_scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
        xy_scale = max(xy_scale, 1e-6)
        coords[:, :2] /= xy_scale

        # Step 4: Z scale independently (depth robustness)
        z_range = coords[:, 2].max() - coords[:, 2].min()
        z_range = max(z_range, 1e-6)
        coords[:, 2] = coords[:, 2] / z_range

        return coords.astype(np.float32)

    def extract_features_from_pil(
        self,
        image: Image.Image,
        use_video_mode: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Full feature extraction pipeline.
        Returns float32 array of shape (N,) or None if no hand detected.

        Feature vector layout (total ~103 features):
        [0:63]   — normalized (x,y,z) for 21 landmarks
        [63:103] — geometric extras (distances, angles, elevations)
        """
        coords, handedness = self._extract_raw_landmarks(
            image, use_video_mode=use_video_mode
        )
        if coords is None:
            return None

        norm_coords = self.normalize_landmarks(coords, handedness=handedness)
        base_features = norm_coords.flatten()          # 63 features
        extra_features = _build_extra_features(norm_coords)  # 40 features

        features = np.concatenate([base_features, extra_features])
        return features.astype(np.float32)

    def extract_features_from_path(self, image_path: str) -> Optional[np.ndarray]:
        with Image.open(image_path) as image:
            return self.extract_features_from_pil(image, use_video_mode=False)
