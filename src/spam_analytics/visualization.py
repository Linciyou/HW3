"""Visualization utilities for reports and dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import Config

plt.switch_backend("Agg")


def plot_label_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot class distribution of the dataset."""
    sns.set_theme(style="whitegrid")
    ax = sns.countplot(data=df, x="label", palette="muted")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    for container in ax.containers:
        ax.bar_label(container)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_message_length_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    """Plot distribution of message lengths (characters)."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="char_count",
        hue="label",
        ax=ax,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.4,
    )
    ax.set_title("Message Length Distribution")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Density")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_top_tokens(top_features: Dict[str, pd.DataFrame], output_path: Path) -> Path:
    """Plot top indicative tokens for each class."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, label in enumerate(["spam", "ham"]):
        data = top_features.get(label)
        if data is None or data.empty:
            axes[idx].set_visible(False)
            continue
        sns.barplot(
            data=data,
            y="feature",
            x="weight",
            ax=axes[idx],
            palette="viridis",
        )
        axes[idx].set_title(f"Top tokens for {label}")
        axes[idx].set_xlabel("Weight")
        axes[idx].set_ylabel("Token")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(conf_matrix, output_path: Path) -> Path:
    """Render the confusion matrix heatmap."""
    labels = ["ham", "spam"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def refresh_static_reports(
    df: pd.DataFrame,
    metrics: Dict,
    top_features: Dict[str, pd.DataFrame],
    config: Config,
) -> Dict[str, Path]:
    """Generate the full visual report suite."""
    figures_dir = config.paths.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "label_distribution": figures_dir / "label_distribution.png",
        "length_distribution": figures_dir / "message_length_distribution.png",
        "top_tokens": figures_dir / "top_tokens.png",
        "confusion_matrix": figures_dir / "confusion_matrix.png",
    }

    plot_label_distribution(df, outputs["label_distribution"])
    plot_message_length_distribution(df, outputs["length_distribution"])

    if top_features:
        plot_top_tokens(top_features, outputs["top_tokens"])

    if metrics and metrics.get("confusion_matrix"):
        plot_confusion_matrix(metrics["confusion_matrix"], outputs["confusion_matrix"])

    return outputs

