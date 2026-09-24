"""All plotting helpers for the Streamlit app."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from .models import TrainedModel

# --- font setup ------------------------------------------------------------
for path in (
    "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
):
    try:
        fm.fontManager.addfont(path)
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="Blues_r")


# --- EDA plots -------------------------------------------------------------

def fig_failure_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.countplot(data=df, x="Machine failure", ax=ax)
    ax.set_title("Machine Failure Distribution")
    ax.set_xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
    ax.set_ylabel("Number of Records")
    return fig


def fig_type_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.countplot(data=df, x="Type", ax=ax)
    ax.set_title("Machine Type Distribution")
    ax.set_xlabel("Machine Type")
    ax.set_ylabel("Number of Records")
    return fig


def fig_numeric_histograms(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.flatten()
    for ax, feat in zip(axes, features):
        df[feat].hist(bins=30, ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("count", fontsize=8)
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Distribution of Numerical Features", fontsize=12)
    return fig


def fig_boxplots_vs_target(df: pd.DataFrame, features: List[str]) -> List[plt.Figure]:
    figs = []
    for feat in features:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        sns.boxplot(data=df, x="Machine failure", y=feat, ax=ax)
        ax.set_title(f"{feat} vs Machine Failure")
        ax.set_xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
        ax.set_ylabel(feat)
        figs.append(fig)
    return figs


def fig_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> plt.Figure:
    cols = features + ["Machine failure"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, square=True)
    ax.set_title("Correlation Matrix of Numerical Features and Machine Failure")
    return fig


# --- Model-evaluation plots ------------------------------------------------

def fig_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"], ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def fig_confusion_grid(models: Dict[str, TrainedModel], y_test, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (name, m) in zip(axes.flatten(), models.items()):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, m.y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Failure", "Failure"],
                    yticklabels=["No Failure", "Failure"])
        ax.set_title(name)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
    fig.suptitle(title, fontsize=16)
    return fig


def fig_roc_curves(models: Dict[str, TrainedModel], y_test, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m.y_score)
        auc = m.metrics["ROC-AUC"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig


def fig_metric_bars(df_metrics: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of Accuracy / Precision / Recall / F1 / ROC-AUC."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models = df_metrics["Model"].tolist()
    x = np.arange(len(models))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for i, metric in enumerate(metrics):
        vals = df_metrics[metric].values
        ax.bar(x + (i - 2) * width, vals, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_xlabel("Models")
    ax.set_title("Performance Comparison of Machine Learning Models")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig


def fig_recall_before_after(
    before: Dict[str, float], after: Dict[str, float]
) -> plt.Figure:
    models = list(before.keys())
    rec_before = [v * 100 for v in before.values()]
    rec_after = [v * 100 for v in after.values()]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars1 = ax.bar(x - width / 2, rec_before, width, label="Before Balancing")
    bars2 = ax.bar(x + width / 2, rec_after, width, label="After Balancing")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.2f}%", ha="center", fontsize=9,
            )

    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Recall Before vs After Class Balancing")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig
