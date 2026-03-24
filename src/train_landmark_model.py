from __future__ import annotations

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


def prepare_landmark_dataset(
    df: pd.DataFrame,
    extractor: HandLandmarkFeatureExtractor,
    drop_nothing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build feature matrix dari dataframe filepath-label.
    nothing diperlakukan rule-based (no hand) jadi defaultnya tidak ikut training.
    """
    feature_list: List[np.ndarray] = []
    label_list: List[str] = []
    skipped_by_label: Dict[str, int] = {}

    for _, row in df.iterrows():
        label = row["label"]
        filepath = row["filepath"]

        if drop_nothing and label == "nothing":
            continue

        features = extractor.extract_features_from_path(filepath)

        if features is None:
            skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
            continue

        feature_list.append(features)
        label_list.append(label)

    if not feature_list:
        raise ValueError("No landmark features extracted. Check MediaPipe detection.")

    X = np.vstack(feature_list).astype(np.float32)
    y = np.array(label_list)

    return X, y, skipped_by_label


def build_landmark_pipeline(random_state: int = 42) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=80,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
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
    extractor = HandLandmarkFeatureExtractor()

    try:
        print("[INFO] Extracting TRAIN landmark features...")
        X_train, y_train_str, train_skipped = prepare_landmark_dataset(train_df, extractor)

        print("[INFO] Extracting VAL landmark features...")
        X_val, y_val_str, val_skipped = prepare_landmark_dataset(val_df, extractor)

        print(f"[INFO] X_train shape: {X_train.shape}")
        print(f"[INFO] X_val shape  : {X_val.shape}")

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

        print("[INFO] Evaluating...")
        y_pred = clf.predict(X_val)

        class_names = list(label_encoder.classes_)

        print("\n===== CLASSIFICATION REPORT =====")
        print(
            classification_report(
                y_val,
                y_pred,
                target_names=class_names,
                digits=4
            )
        )

        print("\n===== CONFUSION MATRIX SHAPE =====")
        print(confusion_matrix(y_val, y_pred).shape)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, LANDMARK_MODEL_PATH)
        joblib.dump(label_encoder, LANDMARK_LABEL_ENCODER_PATH)

        label_map = {str(i): label for i, label in enumerate(class_names)}
        save_json(label_map, LANDMARK_LABEL_MAP_PATH)

        print(f"[INFO] Landmark model saved to: {LANDMARK_MODEL_PATH}")
        print(f"[INFO] Landmark label encoder saved to: {LANDMARK_LABEL_ENCODER_PATH}")
        print(f"[INFO] Landmark label map saved to: {LANDMARK_LABEL_MAP_PATH}")

    finally:
        extractor.close()


if __name__ == "__main__":
    train_landmark_model()