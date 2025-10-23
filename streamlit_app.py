"""Streamlit dashboard for the spam analytics project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_analytics.config import Config, load_config
from spam_analytics.data import DataRepository
from spam_analytics.modeling import ModelTrainer, extract_top_features
from spam_analytics.preprocessing import TextPreprocessor

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    px = None  # type: ignore
    go = None  # type: ignore
    PLOTLY_AVAILABLE = False

DEFAULT_CONFIG_PATH = Path("src") / "spam_analytics" / "config.yaml"


@st.cache_data(show_spinner=False)
def load_app_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration and memoize for the Streamlit session."""
    target_path: Optional[Path] = None
    if config_path and config_path.exists():
        target_path = config_path
    elif DEFAULT_CONFIG_PATH.exists():
        target_path = DEFAULT_CONFIG_PATH

    if target_path is not None:
        try:
            return load_config(target_path)
        except ImportError:
            pass
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
def load_trained_pipeline(config: Config, model_signature: float):
    """Load the persisted model pipeline if it exists."""
    if model_signature <= 0.0:
        return None
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


def get_model_signature(config: Config) -> float:
    """Return a value that changes whenever the persisted model updates."""
    model_path = config.paths.model_path
    if not model_path.exists():
        return 0.0
    try:
        return model_path.stat().st_mtime
    except OSError:
        return 0.0


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Messages", len(df))
    label_counts = df["label"].value_counts()
    col2.metric("Ham", int(label_counts.get("ham", 0)))
    col3.metric("Spam", int(label_counts.get("spam", 0)))

    if PLOTLY_AVAILABLE:
        chart = px.histogram(
            df,
            x="label",
            title="Label Distribution",
            color="label",
            text_auto=True,
            category_orders={"label": ["ham", "spam"]},
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning("Plotly is not installed. Run `pip install plotly` (or `pip install -e .`) to enable charts.")

    st.dataframe(df[["message_id", "label", "text"]].head(5), use_container_width=True)


def render_preprocessing_preview(df: pd.DataFrame) -> None:
    st.subheader("Preprocessing Steps")
    limit = st.slider("Rows to display", min_value=3, max_value=50, value=5)
    sample_cols = ["message_id", "label", "text", "normalized_text", "clean_text"]
    st.dataframe(df[sample_cols].head(limit), use_container_width=True)

    feature_cols = ["char_count", "word_count", "avg_word_len", "digit_ratio", "punct_ratio", "upper_ratio"]
    st.write("Engineered feature summary")
    st.dataframe(df[feature_cols].describe(), use_container_width=True)

    if PLOTLY_AVAILABLE:
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
    if cm and PLOTLY_AVAILABLE:
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
    elif cm:
        st.warning("Install plotly to view the confusion matrix heatmap.")

    roc_curve = metrics.get("roc_curve")
    if roc_curve and PLOTLY_AVAILABLE:
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
    elif roc_curve:
        st.warning("Install plotly to view the ROC curve.")


def render_top_features(pipeline, config: Config) -> None:
    st.subheader("Top Indicative Tokens")
    if pipeline is None:
        st.info("Train the model to view feature importance.")
        return
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly to visualize top tokens.")
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

    config = load_app_config()
    df = load_preprocessed_data(config)
    model_signature = get_model_signature(config)
    pipeline = load_trained_pipeline(config, model_signature)
    metrics = load_metrics(config)

    if not PLOTLY_AVAILABLE:
        st.sidebar.warning("Plotly is missing. Install via `pip install plotly` or `pip install -e .` for full charts.")

    render_dataset_overview(df)
    render_preprocessing_preview(df)
    render_metrics(metrics)
    render_top_features(pipeline, config)
    render_inference(pipeline, config)


if __name__ == "__main__":
    main()
