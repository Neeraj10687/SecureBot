"""Dataset loading and basic inspection for the AI4I 2020 dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
DROP_COLUMNS = ["UDI", "Product ID"]


def load_raw() -> pd.DataFrame:
    """Load the AI4I 2020 CSV. Handles the UTF-8 BOM in the source file."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download it from "
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset "
            "and place the CSV in the data/ folder."
        )
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def load_clean() -> pd.DataFrame:
    """Return the cleaned dataframe (identifiers dropped)."""
    df = load_raw()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into X (raw feature columns) and y (target)."""
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset stats used in the Overview tab."""
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "failure_counts": df[TARGET].value_counts().to_dict(),
        "failure_pct": (df[TARGET].value_counts(normalize=True) * 100).round(2).to_dict(),
        "type_counts": df["Type"].value_counts().to_dict(),
        "describe": df.describe(),
    }
