"""SecureBot — Predictive Maintenance Streamlit App.

Run with:
    ./venv/bin/streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    dataset_summary,
    load_clean,
)
from src.models import best_model_by, metrics_table, train_all
from src.preprocessing import build_balanced_pipeline, build_original_pipeline
from src.predict import build_feature_row, encode_user_row, predict
from src import visualize as viz


st.set_page_config(
    page_title="SecureBot — Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading dataset…")
def get_dataset() -> pd.DataFrame:
    return load_clean()


@st.cache_resource(show_spinner="Training baseline models (before balancing)…")
def get_original_pipeline_and_models(df: pd.DataFrame):
    proc = build_original_pipeline(df)
    models = train_all(proc.X_train, proc.y_train, proc.X_test, proc.y_test)
    return proc, models


@st.cache_resource(show_spinner="Training balanced models (oversample + undersample)…")
def get_balanced_pipeline_and_models(df: pd.DataFrame):
    proc = build_balanced_pipeline(df)
    models = train_all(proc.X_train, proc.y_train, proc.X_test, proc.y_test)
    return proc, models


df = get_dataset()
orig_proc, orig_models = get_original_pipeline_and_models(df)
bal_proc, bal_models = get_balanced_pipeline_and_models(df)

st.sidebar.title("🛠️ SecureBot")
st.sidebar.caption("Predictive Maintenance · AI4I 2020")
st.sidebar.divider()

tab = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 EDA", "🤖 Models (Before)", "⚖️ Models (After)", "🎯 Live Prediction"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    **Dataset**
    - Rows: `{len(df):,}`
    - Cols: `{df.shape[1]}`
    - Failures: `{df["Machine failure"].sum():,}`
    - Failure rate: `{df["Machine failure"].mean() * 100:.2f}%`
    """
)


if tab == "📊 Overview":
    st.title("📊 Dataset Overview")
    st.write(
        "The AI4I 2020 predictive maintenance dataset contains 10,000 synthetic "
        "machine records. Each row represents one machine's operating conditions "
        "(air temperature, process temperature, rotational speed, torque, tool "
        "wear, machine type) and whether the machine failed during the observed "
        "window."
    )

    s = dataset_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{s['rows']:,}")
    c2.metric("Columns", s["cols"])
    c3.metric("Missing Values", s["missing_values"])
    c4.metric("Duplicates", s["duplicates"])

    st.subheader("Class distribution")
    fail_counts = s["failure_counts"]
    fail_pct = s["failure_pct"]
    cd1, cd2 = st.columns(2)
    cd1.metric("No Failure (0)", f"{fail_counts.get(0, 0):,}", f"{fail_pct.get(0, 0):.2f}%")
    cd2.metric("Failure (1)", f"{fail_counts.get(1, 0):,}", f"{fail_pct.get(1, 0):.2f}%")

    st.subheader("Raw sample (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistical summary")
    st.dataframe(s["describe"], use_container_width=True)


elif tab == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Machine failure distribution")
    st.pyplot(viz.fig_failure_distribution(df))

    st.subheader("Machine type distribution")
    st.pyplot(viz.fig_type_distribution(df))

    st.subheader("Numerical feature distributions")
    st.pyplot(viz.fig_numeric_histograms(df, NUMERIC_FEATURES))

    st.subheader("Feature behaviour vs Machine Failure")
    st.write(
        "Each boxplot compares the distribution of a numerical feature between "
        "machines that failed (1) and those that did not (0). A large gap "
        "between the two boxes is a good early signal that the feature carries "
        "predictive signal."
    )
    for fig in viz.fig_boxplots_vs_target(df, NUMERIC_FEATURES):
        st.pyplot(fig)

    st.subheader("Correlation matrix")
    st.pyplot(viz.fig_correlation_heatmap(df, NUMERIC_FEATURES))


elif tab == "🤖 Models (Before)":
    st.title("🤖 Baseline Models — Before Class Balancing")
    st.write(
        "Four classifiers were trained on the original (highly imbalanced) "
        "training set. Accuracy looks great on paper, but **Recall on the "
        "minority failure class is poor** — that is the whole point of the "
        "next tab."
    )

    df_metrics = metrics_table(orig_models).round(4)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.subheader("Confusion matrices")
    st.pyplot(viz.fig_confusion_grid(orig_models, orig_proc.y_test,
                                     "Confusion Matrices — Before Balancing"))

    st.subheader("ROC curves")
    st.pyplot(viz.fig_roc_curves(orig_models, orig_proc.y_test,
                                 "ROC Curves — Before Class Balancing"))

    st.subheader("Metric comparison")
    st.pyplot(viz.fig_metric_bars(df_metrics))

    best_name, best_m = best_model_by(orig_models, "F1-Score")
    st.info(f"**Best baseline model by F1-Score:** {best_name} "
            f"(F1 = {best_m.metrics['F1-Score']:.4f}, "
            f"Recall = {best_m.metrics['Recall']:.4f})")


elif tab == "⚖️ Models (After)":
    st.title("⚖️ Models — After Class Balancing")
    st.write(
        "Training data was resampled with **RandomOverSampler** (failure class "
        "grown to 5% of the majority) followed by **RandomUnderSampler** "
        "(majority trimmed to match). The test set is untouched — only the "
        "training data changed. Recall on the failure class jumps dramatically."
    )

    bc1, bc2, bc3 = st.columns(3)
    bc1.markdown("**Original training**")
    bc1.write(orig_proc.y_train.value_counts())
    bc2.markdown("**Balanced training**")
    bc2.write(bal_proc.y_train.value_counts())
    bc3.markdown("**Test (untouched)**")
    bc3.write(bal_proc.y_test.value_counts())

    df_metrics = metrics_table(bal_models).round(4)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.subheader("Confusion matrices")
    st.pyplot(viz.fig_confusion_grid(bal_models, bal_proc.y_test,
                                     "Confusion Matrices — After Balancing"))

    st.subheader("ROC curves")
    st.pyplot(viz.fig_roc_curves(bal_models, bal_proc.y_test,
                                 "ROC Curves — After Class Balancing"))

    st.subheader("Recall: before vs after balancing")
    before_recalls = {n: m.metrics["Recall"] for n, m in orig_models.items()}
    after_recalls = {n: m.metrics["Recall"] for n, m in bal_models.items()}
    st.pyplot(viz.fig_recall_before_after(before_recalls, after_recalls))

    st.subheader("Metric comparison")
    st.pyplot(viz.fig_metric_bars(df_metrics))

    best_name, best_m = best_model_by(bal_models, "F1-Score")
    st.success(f"**Best balanced model by F1-Score:** {best_name} "
               f"(F1 = {best_m.metrics['F1-Score']:.4f}, "
               f"Recall = {best_m.metrics['Recall']:.4f})")


elif tab == "🎯 Live Prediction":
    st.title("🎯 Live Failure Prediction")
    st.write(
        "Enter the machine's operating conditions and pick which trained model "
        "should classify it. Balanced models are recommended — they actually "
        "catch failures, whereas the baseline models almost always predict "
        "'no failure'."
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.slider("Air temperature [K]", 290.0, 310.0, 298.0, 0.1)
            process_temp = st.slider("Process temperature [K]", 300.0, 320.0, 308.5, 0.1)
            torque = st.slider("Torque [Nm]", 0.0, 80.0, 40.0, 0.5)
        with col2:
            rotational_speed = st.slider("Rotational speed [rpm]", 1100.0, 2900.0, 1500.0, 10.0)
            tool_wear = st.slider("Tool wear [min]", 0, 250, 30, 5)
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"],
                                         help="L = Low, M = Medium, H = High quality variant")

        col_a, col_b = st.columns(2)
        with col_a:
            pipeline_choice = st.radio(
                "Pipeline",
                ["Balanced (recommended)", "Original baseline"],
                index=0,
            )
        with col_b:
            model_name = st.selectbox(
                "Model",
                list(bal_models.keys()),
                index=2,
            )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        if pipeline_choice.startswith("Balanced"):
            proc, models = bal_proc, bal_models
        else:
            proc, models = orig_proc, orig_models

        model = models[model_name]
        row = build_feature_row(
            air_temp, process_temp, rotational_speed, torque, tool_wear, machine_type
        )
        row_vec = encode_user_row(row, proc.scaler, proc.encoder)
        result = predict(model, row_vec)

        st.divider()
        rc1, rc2 = st.columns(2)
        with rc1:
            label = result["label"]
            if result["prediction"] == 1:
                st.error(f"🔴 {label}")
            else:
                st.success(f"🟢 {label}")
        with rc2:
            prob = result["failure_probability"]
            st.metric("Failure probability", f"{prob * 100:.2f}%")
            st.progress(min(prob, 1.0))

        with st.expander("Input features (after encoding)"):
            st.code(repr(row_vec[0]))
