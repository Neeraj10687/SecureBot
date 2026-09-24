"""Preprocessing: encoding, scaling, train/test split, and class balancing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES, split_features_target


@dataclass
class ProcessedData:
    """Container for one full set of processed train/test arrays."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    encoder: OneHotEncoder
    label: str


def _split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _encode_and_scale(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, OneHotEncoder]:
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_raw[NUMERIC_FEATURES])
    X_test_num = scaler.transform(X_test_raw[NUMERIC_FEATURES])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train_raw[CATEGORICAL_FEATURES])
    X_test_cat = encoder.transform(X_test_raw[CATEGORICAL_FEATURES])

    X_train_final = np.hstack([X_train_cat, X_train_num])
    X_test_final = np.hstack([X_test_cat, X_test_num])
    return X_train_final, X_test_final, scaler, encoder


def build_original_pipeline(df: pd.DataFrame) -> ProcessedData:
    """No class balancing — used for the 'before balancing' baseline."""
    X_train, X_test, y_train, y_test = _split(df)
    X_tr, X_te, scaler, encoder = _encode_and_scale(X_train, X_test)
    return ProcessedData(
        X_train=X_tr, X_test=X_te,
        y_train=y_train, y_test=y_test,
        scaler=scaler, encoder=encoder,
        label="original",
    )


def build_balanced_pipeline(
    df: pd.DataFrame,
    oversample_ratio: float = 0.05,
    random_state: int = 42,
) -> ProcessedData:
    """Apply oversample→undersample on the TRAINING rows only."""
    X_train, X_test, y_train, y_test = _split(df, random_state=random_state)

    oversampler = RandomOverSampler(
        sampling_strategy=oversample_ratio, random_state=random_state
    )
    X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

    undersampler = RandomUnderSampler(
        sampling_strategy="auto", random_state=random_state
    )
    X_train_bal, y_train_bal = undersampler.fit_resample(X_train_over, y_train_over)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_bal[NUMERIC_FEATURES])
    X_test_num = scaler.transform(X_test[NUMERIC_FEATURES])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train_bal[CATEGORICAL_FEATURES])
    X_test_cat = encoder.transform(X_test[CATEGORICAL_FEATURES])

    X_train_final = np.hstack([X_train_cat, X_train_num])
    X_test_final = np.hstack([X_test_cat, X_test_num])

    return ProcessedData(
        X_train=X_train_final, X_test=X_test_final,
        y_train=y_train_bal, y_test=y_test,
        scaler=scaler, encoder=encoder,
        label="balanced",
    )
