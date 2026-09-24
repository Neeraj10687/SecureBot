"""Live prediction utilities — encode user input and score it with the chosen model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .models import TrainedModel


def build_feature_row(
    air_temp: float,
    process_temp: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    machine_type: str,
) -> pd.DataFrame:
    """Build a 1-row DataFrame with the same schema the model was trained on."""
    return pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )


def encode_user_row(
    row: pd.DataFrame, scaler, encoder
) -> np.ndarray:
    """Apply the SAME fitted scaler + encoder used during training."""
    num = scaler.transform(row[NUMERIC_FEATURES])
    cat = encoder.transform(row[CATEGORICAL_FEATURES])
    return np.hstack([cat, num])


def predict(model: TrainedModel, row_vector: np.ndarray) -> dict:
    """Run a single prediction and return both the label and the failure score."""
    pred = int(model.estimator.predict(row_vector)[0])
    if hasattr(model.estimator, "predict_proba"):
        proba = float(model.estimator.predict_proba(row_vector)[0, 1])
    else:
        # For SVM without predict_proba, use decision_function and squash.
        raw = float(model.estimator.decision_function(row_vector)[0])
        proba = 1.0 / (1.0 + np.exp(-raw))  # sigmoid
    return {
        "prediction": pred,
        "failure_probability": proba,
        "label": "FAILURE predicted" if pred == 1 else "No failure predicted",
    }
