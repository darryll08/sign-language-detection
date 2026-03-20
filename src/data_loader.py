from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TRAIN_DIR, CLASS_NAMES, RANDOM_SEED


def build_train_dataframe(train_dir: Path = TRAIN_DIR) -> pd.DataFrame:
    """
    Scan folder train dan buat dataframe berisi:
    - filepath
    - label
    """
    records = []

    for class_name in CLASS_NAMES:
        class_dir = Path(train_dir) / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        for img_path in class_dir.glob("*.jpg"):
            records.append(
                {
                    "filepath": str(img_path.resolve()),
                    "label": class_name,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No training images found. Check your TRAIN_DIR path.")

    return df


def split_train_val(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe menjadi train dan validation secara stratified.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def get_dataframes(
    train_dir: Path = TRAIN_DIR,
    val_size: float = 0.2,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper utama:
    - build dataframe full
    - split train/val
    """
    full_df = build_train_dataframe(train_dir=train_dir)
    train_df, val_df = split_train_val(
        full_df,
        val_size=val_size,
        random_state=random_state
    )
    return train_df, val_df


def summarize_dataframe(df: pd.DataFrame, name: str = "dataset") -> pd.DataFrame:
    """
    Ringkasan jumlah data per kelas.
    """
    summary = (
        df["label"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    summary.columns = ["label", "count"]

    print(f"\n===== SUMMARY: {name.upper()} =====")
    print(f"Total samples: {len(df)}")
    print(summary.to_string(index=False))

    return summary