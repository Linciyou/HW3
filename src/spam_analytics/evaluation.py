"""Model evaluation and reporting utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .config import Config


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    positive_label: str = "spam",
) -> Dict[str, Any]:
    """Compute a rich set of metrics for the provided pipeline."""
    y_pred = pipeline.predict(X_test)

    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=positive_label
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1_score"] = float(f1)

    metrics["classification_report"] = classification_report(
        y_test, y_pred, digits=3, output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    metrics["confusion_matrix"] = cm.tolist()

    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(positive_label)]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_scores))
        except ValueError:
            metrics["roc_auc"] = None
        fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=positive_label)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
    else:
        metrics["roc_auc"] = None
        metrics["roc_curve"] = None

    metrics["generated_at"] = datetime.utcnow().isoformat() + "Z"
    return metrics


def save_metrics(metrics: Dict[str, Any], config: Config) -> Path:
    """Persist metrics to the reports directory as JSON."""
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.paths.metrics_path
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics_path

