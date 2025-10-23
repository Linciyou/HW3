"""Command line entry point for the spam analytics toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import Config, load_config
from .data import DataRepository
from .evaluation import evaluate_pipeline, save_metrics
from .modeling import ModelTrainer, extract_top_features
from .preprocessing import TextPreprocessor
from .visualization import refresh_static_reports

app = typer.Typer(help="Spam analytics CLI for preprocessing, training, and reporting.")
data_app = typer.Typer(help="Dataset inspection utilities.")
model_app = typer.Typer(help="Model training and evaluation commands.")
viz_app = typer.Typer(help="Visualization and report generation.")

app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(viz_app, name="visualize")

console = Console()
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def _resolve_config(config_path: Optional[Path]) -> Config:
    target_path: Optional[Path]
    if config_path is not None:
        target_path = config_path
    elif DEFAULT_CONFIG_PATH.exists():
        target_path = DEFAULT_CONFIG_PATH
    else:
        target_path = None

    try:
        return load_config(target_path)
    except ImportError as exc:
        if config_path is not None:
            console.print(f"[yellow]{exc} Falling back to built-in defaults.[/]")
        return load_config(None)


@data_app.command("summary")
def data_summary(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional path to a YAML config file.",
    ),
    limit: int = typer.Option(5, help="Rows to preview from the dataset."),
) -> None:
    """Display dataset statistics and preview rows."""
    config = _resolve_config(config_path)
    repo = DataRepository(config)
    df = repo.load_raw()

    console.print(f"[bold green]Dataset loaded:[/] {config.paths.dataset} ({len(df)} rows)")
    label_counts = df["label"].value_counts()

    stats_table = Table(title="Dataset Stats")
    stats_table.add_column("Metric")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Rows", str(len(df)))
    stats_table.add_row("Unique labels", str(df["label"].nunique()))
    stats_table.add_row("Ham", str(label_counts.get("ham", 0)))
    stats_table.add_row("Spam", str(label_counts.get("spam", 0)))
    console.print(stats_table)

    if limit > 0:
        preview_table = Table(title=f"Preview (first {limit} rows)", show_lines=False)
        preview_table.add_column("ID", justify="right")
        preview_table.add_column("Label")
        preview_table.add_column("Text", overflow="fold")
        for _, row in df.head(limit).iterrows():
            preview_table.add_row(str(row["message_id"]), row["label"], row["text"])
        console.print(preview_table)


@data_app.command("preprocess")
def data_preprocess(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config path."),
    limit: int = typer.Option(5, help="Rows to preview after preprocessing."),
) -> None:
    """Run preprocessing pipeline and inspect intermediate outputs."""
    config = _resolve_config(config_path)
    repo = DataRepository(config)
    df = repo.load_raw()
    preprocessor = TextPreprocessor(config)
    result = preprocessor.run(df).data

    console.print(f"[bold green]Preprocessing complete[/] — {len(result)} rows.")
    columns = ["message_id", "label", "text", "normalized_text", "clean_text"]

    preview_table = Table(title=f"Preprocessed sample (first {limit} rows)")
    for column in columns:
        preview_table.add_column(column, overflow="fold")

    for _, row in result.head(limit).iterrows():
        preview_table.add_row(*(str(row[col]) for col in columns))
    console.print(preview_table)


@model_app.command("train")
def model_train(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config path."),
    cache_steps: bool = typer.Option(True, help="Persist preprocessing steps to artifacts/preprocessing_steps.parquet"),
) -> None:
    """Train the spam classifier and persist artifacts."""
    config = _resolve_config(config_path)
    config.ensure_directories()

    repo = DataRepository(config)
    raw_df = repo.load_raw()

    preprocessor = TextPreprocessor(config)
    preprocessing_result = preprocessor.run(raw_df)
    processed_df = preprocessing_result.data

    feature_columns = ["clean_text", *config.numeric_features]
    features = processed_df[feature_columns]
    target = processed_df["label"]

    bundle = repo.split(features, target)

    trainer = ModelTrainer(config)
    pipeline = trainer.fit(bundle.train_X, bundle.train_y)
    trainer.save(pipeline)

    metrics = evaluate_pipeline(pipeline, bundle.test_X, bundle.test_y)
    save_metrics(metrics, config)

    top_features = extract_top_features(pipeline, config.training.report_top_features)
    refresh_static_reports(processed_df, metrics, top_features, config)

    cache_path = None
    if cache_steps:
        cache_path = _persist_preprocessing(processed_df, config)

    console.print("[bold green]Training complete![/]")
    console.print(f"Model saved to: {config.paths.model_path}")
    console.print(f"Metrics saved to: {config.paths.metrics_path}")
    if cache_path:
        console.print(f"Cached preprocessing steps at: {cache_path}")


@model_app.command("evaluate")
def model_evaluate(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config path."),
) -> None:
    """Evaluate the existing model against the configured test split."""
    config = _resolve_config(config_path)
    trainer = ModelTrainer(config)
    pipeline = trainer.load_existing()

    repo = DataRepository(config)
    raw_df = repo.load_raw()
    preprocessor = TextPreprocessor(config)
    processed_df = preprocessor.run(raw_df).data

    feature_columns = ["clean_text", *config.numeric_features]
    features = processed_df[feature_columns]
    target = processed_df["label"]
    bundle = repo.split(features, target)

    metrics = evaluate_pipeline(pipeline, bundle.test_X, bundle.test_y)
    save_metrics(metrics, config)
    console.print("[bold green]Evaluation complete![/]")
    console.print(json.dumps({k: metrics[k] for k in ("accuracy", "precision", "recall", "f1_score")}, indent=2))


@viz_app.command("report")
def visualize_report(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config path."),
) -> None:
    """Generate static visual reports using latest artifacts."""
    config = _resolve_config(config_path)
    repo = DataRepository(config)
    raw_df = repo.load_raw()
    preprocessor = TextPreprocessor(config)
    processed_df = preprocessor.run(raw_df).data

    trainer = ModelTrainer(config)
    try:
        pipeline = trainer.load_existing()
        top_features = extract_top_features(pipeline, config.training.report_top_features)
    except FileNotFoundError:
        pipeline = None
        top_features = {}
        console.print("[yellow]No existing model found. Skipping top token plot.[/]")

    metrics = _load_metrics(config)
    refresh_static_reports(processed_df, metrics, top_features, config)
    console.print(f"[bold green]Figures refreshed in {config.paths.figures_dir}[/]")


def _persist_preprocessing(df: pd.DataFrame, config: Config) -> Path | None:
    """Persist preprocessing steps with parquet fallback to CSV."""
    output_path = config.paths.preprocessing_cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
        return output_path
    except (ImportError, ValueError):
        fallback = output_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        console.print(
            f"[yellow]PyArrow/FastParquet not available; wrote preprocessing steps to {fallback} instead.[/]"
        )
        return fallback


def _load_metrics(config: Config) -> dict:
    """Load metrics JSON if present, else return empty dict."""
    metrics_path = config.paths.metrics_path
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
