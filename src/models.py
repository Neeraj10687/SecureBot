"""Model training & evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_REGISTRY = {
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": lambda: SVC(kernel="rbf", random_state=42),
}


@dataclass
class TrainedModel:
    name: str
    estimator: object
    y_pred: np.ndarray
    y_score: np.ndarray
    metrics: Dict[str, float]


def _scores(estimator, X: np.ndarray) -> np.ndarray:
    """Return the failure-class score for ROC-AUC."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def train_all(X_train, y_train, X_test, y_test) -> Dict[str, TrainedModel]:
    """Train KNN / DT / RF / SVM and return them with their metrics."""
    out: Dict[str, TrainedModel] = {}
    for name, factory in MODEL_REGISTRY.items():
        est = factory()
        est.fit(X_train, y_train)

        y_pred = est.predict(X_test)
        y_score = _scores(est, X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_score),
        }
        out[name] = TrainedModel(
            name=name, estimator=est,
            y_pred=y_pred, y_score=y_score, metrics=metrics,
        )
    return out


def confusion_matrix_for(model: TrainedModel, y_test) -> np.ndarray:
    return confusion_matrix(y_test, model.y_pred)


def metrics_table(models: Dict[str, TrainedModel]) -> pd.DataFrame:
    """Return a tidy comparison DataFrame."""
    rows = []
    for name, m in models.items():
        rows.append({"Model": name, **m.metrics})
    df = pd.DataFrame(rows)
    return df


def best_model_by(models: Dict[str, TrainedModel], metric: str = "F1-Score") -> Tuple[str, TrainedModel]:
    """Pick the best model by a given metric (default: F1)."""
    best_name = max(models, key=lambda n: models[n].metrics[metric])
    return best_name, models[best_name]
