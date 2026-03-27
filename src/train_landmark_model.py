"""
train_landmark_model.py — Improved version

Changes vs original:
- Larger MLP: (512, 256, 128) instead of (256, 128)
- More training iterations: max_iter=300 instead of 80
- Better regularization: alpha=5e-4 (was 1e-4)
- Landmark augmentation: Gaussian jitter + random scale + random 2D rotation
  applied on-the-fly during training to make model robust to:
  - slight hand tremor
  - varying distances from camera
  - slight wrist rotation
- "nothing" class is now trained explicitly using hard-negative examples
  (random noise features that represent non-gesture states)
- Uses the new richer feature extractor (103 features vs 63)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import MODELS_DIR
from src.data_loader import get_dataframes
from src.landmark_features import HandLandmarkFeatureExtractor
from src.utils import save_json


LANDMARK_MODEL_PATH = MODELS_DIR / "landmark_mlp.joblib"
LANDMARK_LABEL_MAP_PATH = MODELS_DIR / "landmark_label_map.json"
LANDMARK_LABEL_ENCODER_PATH = MODELS_DIR / "landmark_label_encoder.joblib"

# ── Augmentation settings ───────────────────────────────────────────────────
AUGMENT_COPIES = 3          # how many augmented copies per real sample
JITTER_STD = 0.015          # Gaussian noise std on landmark coords
SCALE_RANGE = (0.88, 1.12)  # random scale factor
ROTATION_MAX_DEG = 12       # max 2D rotation in degrees
# ────────────────────────────────────────────────────────────────────────────

# How many synthetic "nothing" samples to inject during training
NOTHING_SAMPLES = 800


def _rotate_2d(coords: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate XY of landmark coords around origin by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    coords = coords.copy()
    coords[:, :2] = coords[:, :2] @ R.T
    return coords


def augment_landmark_coords(
    coords_21x3: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply random augmentation to a (21, 3) landmark array.
    Assumes coords are already wrist-centered and normalized.
    """
    coords = coords_21x3.copy()

    # 1. Gaussian jitter on all axes
    coords += rng.normal(0, JITTER_STD, size=coords.shape).astype(np.float32)

    # 2. Random uniform scale
    scale = rng.uniform(*SCALE_RANGE)
    coords *= scale

    # 3. Random 2D rotation
    angle = rng.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG) * (np.pi / 180.0)
    coords = _rotate_2d(coords, float(angle))

    return coords.astype(np.float32)


def _extract_raw_coords_and_handedness(extractor, filepath: str):
    """
    Get raw landmark coords + handedness before normalization,
    so we can augment in landmark space.
    Returns (coords_21x3, handedness_str) or (None, None).
    """
    from PIL import Image
    try:
        with Image.open(filepath) as img:
            coords, handedness = extractor._extract_raw_landmarks(img)
        return coords, handedness
    except Exception:
        return None, None


def prepare_landmark_dataset(
    df: pd.DataFrame,
    extractor: HandLandmarkFeatureExtractor,
    drop_nothing: bool = False,   # now False by default — we handle nothing explicitly
    augment: bool = False,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build feature matrix from dataframe filepath-label.

    When augment=True, each real sample spawns AUGMENT_COPIES extra
    augmented variants (only for training split, not validation).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    feature_list: List[np.ndarray] = []
    label_list: List[str] = []
    skipped_by_label: Dict[str, int] = {}

    total = len(df)
    start_time = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        label = row["label"]
        filepath = row["filepath"]

        if drop_nothing and label == "nothing":
            continue

        # For "nothing" label, we still try to extract features
        # but most images won't have a hand → we generate synthetic negatives later
        if label == "nothing":
            skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
            continue

        if augment:
            # Get raw coords for augmentation
            coords, handedness = _extract_raw_coords_and_handedness(extractor, filepath)
            if coords is None:
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue

            # Normalize once (original)
            from src.landmark_features import _build_extra_features
            norm = extractor.normalize_landmarks(coords, handedness=handedness)
            base = norm.flatten()
            extra = _build_extra_features(norm)
            feature_list.append(np.concatenate([base, extra]).astype(np.float32))
            label_list.append(label)

            # Augmented copies
            for _ in range(AUGMENT_COPIES):
                aug_coords = augment_landmark_coords(norm, rng)
                aug_base = aug_coords.flatten()
                aug_extra = _build_extra_features(aug_coords)
                feature_list.append(np.concatenate([aug_base, aug_extra]).astype(np.float32))
                label_list.append(label)

        else:
            features = extractor.extract_features_from_path(filepath)
            if features is None:
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue
            feature_list.append(features)
            label_list.append(label)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i + 1}/{total}] features extracted | {elapsed:.1f}s elapsed")

    if not feature_list:
        raise ValueError("No landmark features extracted. Check MediaPipe detection.")

    X = np.vstack(feature_list).astype(np.float32)
    y = np.array(label_list)

    return X, y, skipped_by_label


def inject_nothing_samples(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic 'nothing' samples as hard negatives.

    Strategy: mix of
    1. Random uniform noise in [-1, 1] range (totally invalid hands)
    2. Slightly perturbed real samples labeled as nothing
       (simulate when MediaPipe detects a false positive)
    """
    nothing_features = []

    # Half: pure random noise
    n_random = n_samples // 2
    random_noise = rng.uniform(-1.0, 1.0, size=(n_random, feature_dim)).astype(np.float32)
    nothing_features.append(random_noise)

    # Half: take random real samples and add heavy noise
    n_perturbed = n_samples - n_random
    if len(X) > 0:
        indices = rng.integers(0, len(X), size=n_perturbed)
        perturbed = X[indices].copy()
        perturbed += rng.normal(0, 0.4, size=perturbed.shape).astype(np.float32)
        nothing_features.append(perturbed)

    nothing_X = np.vstack(nothing_features).astype(np.float32)
    nothing_y = np.array(["nothing"] * n_samples)

    X_combined = np.vstack([X, nothing_X])
    y_combined = np.concatenate([y, nothing_y])

    return X_combined, y_combined


def build_landmark_pipeline(random_state: int = 42) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(512, 256, 128),  # bigger than before
                    activation="relu",
                    solver="adam",
                    alpha=5e-4,              # stronger regularization
                    batch_size=256,
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    max_iter=300,            # more iterations
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,     # more patience
                    random_state=random_state,
                    verbose=True,
                ),
            ),
        ]
    )
    return clf


def train_landmark_model():
    print("[INFO] Loading train/val splits...")
    train_df, val_df = get_dataframes()

    print("[INFO] Creating landmark extractor...")
    extractor = HandLandmarkFeatureExtractor(enable_video_mode=False)

    rng = np.random.default_rng(42)

    try:
        print("[INFO] Extracting TRAIN landmark features (with augmentation)...")
        X_train, y_train_str, train_skipped = prepare_landmark_dataset(
            train_df, extractor, drop_nothing=False, augment=True, rng=rng
        )

        print(f"[INFO] X_train shape (after augmentation): {X_train.shape}")
        feature_dim = X_train.shape[1]

        print(f"[INFO] Injecting {NOTHING_SAMPLES} synthetic 'nothing' samples...")
        X_train, y_train_str = inject_nothing_samples(
            X_train, y_train_str, NOTHING_SAMPLES, rng, feature_dim
        )
        print(f"[INFO] X_train shape (with nothing): {X_train.shape}")

        print("[INFO] Extracting VAL landmark features (no augmentation)...")
        X_val, y_val_str, val_skipped = prepare_landmark_dataset(
            val_df, extractor, drop_nothing=False, augment=False
        )
        # Inject a smaller set of nothing samples into val too for fair evaluation
        X_val, y_val_str = inject_nothing_samples(
            X_val, y_val_str, NOTHING_SAMPLES // 5, rng, feature_dim
        )
        print(f"[INFO] X_val shape: {X_val.shape}")

        print("[INFO] Skipped TRAIN by label:", train_skipped)
        print("[INFO] Skipped VAL by label  :", val_skipped)

        print("[INFO] Encoding labels...")
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_str)
        y_val = label_encoder.transform(y_val_str)

        print("[INFO] Classes:", list(label_encoder.classes_))

        print("[INFO] Training landmark classifier...")
        clf = build_landmark_pipeline()
        clf.fit(X_train, y_train)

        print("[INFO] Evaluating on validation set...")
        y_pred = clf.predict(X_val)
        class_names = list(label_encoder.classes_)

        print("\n===== CLASSIFICATION REPORT =====")
        print(
            classification_report(
                y_val,
                y_pred,
                target_names=class_names,
                digits=4,
            )
        )

        print("\n===== CONFUSION MATRIX SHAPE =====")
        print(confusion_matrix(y_val, y_pred).shape)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, LANDMARK_MODEL_PATH)
        joblib.dump(label_encoder, LANDMARK_LABEL_ENCODER_PATH)

        label_map = {str(i): label for i, label in enumerate(class_names)}
        save_json(label_map, LANDMARK_LABEL_MAP_PATH)

        print(f"\n[INFO] Landmark model saved to: {LANDMARK_MODEL_PATH}")
        print(f"[INFO] Landmark label encoder saved to: {LANDMARK_LABEL_ENCODER_PATH}")
        print(f"[INFO] Landmark label map saved to: {LANDMARK_LABEL_MAP_PATH}")
        print(f"[INFO] Feature dimension: {feature_dim}")

    finally:
        extractor.close()


if __name__ == "__main__":
    train_landmark_model()
