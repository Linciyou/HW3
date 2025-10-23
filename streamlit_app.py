"""Streamlit dashboard for the spam analytics project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from spam_analytics.config import Config, load_config
from spam_analytics.data import DataRepository
from spam_analytics.modeling import ModelTrainer, extract_top_features
from spam_analytics.preprocessing import TextPreprocessor

DEFAULT_CONFIG_PATH = Path("src") / "spam_analytics" / "config.yaml"


@st.cache_data(show_spinner=False)
def load_app_config(config_path: Optional[Path]) -> Config:
    """Load configuration and memoize for the Streamlit session."""
    if config_path and config_path.exists():
        return load_config(config_path)
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(DEFAULT_CONFIG_PATH)
    return load_config(None)


@st.cache_data(show_spinner=True)
def load_preprocessed_data(config: Config) -> pd.DataFrame:
    """Load raw dataset and compute preprocessing steps."""
    repo = DataRepository(config)
    df = repo.load_raw()
    preprocessor = TextPreprocessor(config)
    result = preprocessor.run(df)
    return result.data


@st.cache_resource(show_spinner=False)
def load_trained_pipeline(config: Config):
    """Load the persisted model pipeline if it exists."""
    trainer = ModelTrainer(config)
    try:
        return trainer.load_existing()
    except FileNotFoundError:
        return None


def load_metrics(config: Config) -> Dict:
    metrics_path = config.paths.metrics_path
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Messages", len(df))
    label_counts = df["label"].value_counts()
    col2.metric("Ham", int(label_counts.get("ham", 0)))
    col3.metric("Spam", int(label_counts.get("spam", 0)))

    chart = px.histogram(
        df,
        x="label",
        title="Label Distribution",
        color="label",
        text_auto=True,
        category_orders={"label": ["ham", "spam"]},
    )
    st.plotly_chart(chart, use_container_width=True)

    st.dataframe(df[["message_id", "label", "text"]].head(5), use_container_width=True)


def render_preprocessing_preview(df: pd.DataFrame) -> None:
    st.subheader("Preprocessing Steps")
    limit = st.slider("Rows to display", min_value=3, max_value=50, value=5)
    sample_cols = ["message_id", "label", "text", "normalized_text", "clean_text"]
    st.dataframe(df[sample_cols].head(limit), use_container_width=True)

    feature_cols = ["char_count", "word_count", "avg_word_len", "digit_ratio", "punct_ratio", "upper_ratio"]
    st.write("Engineered feature summary")
    st.dataframe(df[feature_cols].describe(), use_container_width=True)

    length_fig = px.histogram(
        df,
        x="char_count",
        nbins=60,
        color="label",
        marginal="box",
        title="Message Length Distribution",
    )
    st.plotly_chart(length_fig, use_container_width=True)


def render_metrics(metrics: Dict) -> None:
    st.subheader("Evaluation Metrics")
    if not metrics:
        st.info("No metrics found. Train the model via `spam-analytics model train`.")
        return

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    cols[1].metric("Precision", f"{metrics.get('precision', 0):.3f}")
    cols[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
    cols[3].metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")

    cm = metrics.get("confusion_matrix")
    if cm:
        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Ham", "Spam"],
                y=["Ham", "Spam"],
                colorscale="Blues",
                showscale=True,
            )
        )
        cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(cm_fig, use_container_width=True)

    roc_curve = metrics.get("roc_curve")
    if roc_curve:
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=roc_curve["fpr"],
                y=roc_curve["tpr"],
                mode="lines",
                name="ROC Curve",
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(dash="dash"),
            )
        )
        roc_fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(roc_fig, use_container_width=True)


def render_top_features(pipeline, config: Config) -> None:
    st.subheader("Top Indicative Tokens")
    if pipeline is None:
        st.info("Train the model to view feature importance.")
        return

    top_features = extract_top_features(pipeline, config.training.report_top_features)
    spam_df = top_features["spam"]
    ham_df = top_features["ham"]

    spam_chart = px.bar(
        spam_df,
        x="weight",
        y="feature",
        orientation="h",
        title="Spam Indicators",
        color="weight",
        color_continuous_scale="Reds",
    )
    ham_chart = px.bar(
        ham_df,
        x="weight",
        y="feature",
        orientation="h",
        title="Ham Indicators",
        color="weight",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(spam_chart, use_container_width=True)
    st.plotly_chart(ham_chart, use_container_width=True)


def render_inference(pipeline, config: Config) -> None:
    st.subheader("Interactive Inference")
    if pipeline is None:
        st.info("Train a model to enable inference.")
        return

    user_input = st.text_area(
        "Enter a message to classify",
        height=100,
        placeholder="Congratulations! You've won a prize...",
    )

    if not user_input.strip():
        return

    preprocessor = TextPreprocessor(config)
    inference_df = pd.DataFrame({"message_id": [0], "label": ["unknown"], "text": [user_input]})
    processed = preprocessor.run(inference_df).data

    feature_columns = ["clean_text", *config.numeric_features]
    features = processed[feature_columns]

    prediction = pipeline.predict(features)[0]
    proba_idx = list(pipeline.classes_).index("spam")
    probability = pipeline.predict_proba(features)[0][proba_idx]

    st.write(f"**Prediction:** {prediction.upper()}")
    st.progress(float(probability) if prediction == "spam" else 1 - float(probability))
    st.caption(f"Spam probability: {probability:.3f}")


def main() -> None:
    st.set_page_config(page_title="Spam Analytics Dashboard", layout="wide")
    st.title("📬 Spam Analytics Dashboard")

    config_path_str = st.sidebar.text_input(
        "Config path (optional)", value=str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else ""
    )
    config_path = Path(config_path_str) if config_path_str else None

    config = load_app_config(config_path)
    df = load_preprocessed_data(config)
    pipeline = load_trained_pipeline(config)
    metrics = load_metrics(config)

    render_dataset_overview(df)
    render_preprocessing_preview(df)
    render_metrics(metrics)
    render_top_features(pipeline, config)
    render_inference(pipeline, config)


if __name__ == "__main__":
    main()

