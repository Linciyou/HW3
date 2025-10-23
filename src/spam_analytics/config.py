"""Configuration management for the spam analytics toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


@dataclass
class PathConfig:
    """Filesystem locations used by the toolkit."""

    dataset: Path = Path("dataset.csv")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    figures_dirname: str = "figures"
    preprocessing_cache_name: str = "preprocessing_steps.parquet"
    model_name: str = "model.joblib"
    metrics_name: str = "metrics.json"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / self.figures_dirname

    @property
    def preprocessing_cache(self) -> Path:
        return self.artifacts_dir / self.preprocessing_cache_name

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_name

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / self.metrics_name


@dataclass
class PreprocessingConfig:
    """Text normalization options."""

    lowercase: bool = True
    strip_urls: bool = True
    strip_html: bool = True
    strip_emails: bool = True
    strip_numbers: bool = True
    strip_punctuation: bool = True
    strip_accents: bool = True
    collapse_whitespace: bool = True
    min_token_length: int = 2
    use_stemmer: bool = True
    stop_words: str = "english"
    extra_stop_words: Tuple[str, ...] = field(default_factory=tuple)
    keep_internal_digits: bool = False


@dataclass
class VectorizerConfig:
    """TfidfVectorizer parameters."""

    ngram_range: Tuple[int, int] = (1, 2)
    analyzer: str = "word"
    max_df: float = 0.95
    min_df: int = 2
    max_features: int | None = 20000
    sublinear_tf: bool = True
    norm: str = "l2"


@dataclass
class ModelConfig:
    """Classifier hyperparameters."""

    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Training split and reproducibility options."""

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    report_top_features: int = 25


@dataclass
class Config:
    """Top-level configuration container."""

    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    numeric_features: Tuple[str, ...] = (
        "char_count",
        "word_count",
        "avg_word_len",
        "digit_ratio",
        "punct_ratio",
        "upper_ratio",
        "unique_char_ratio",
    )

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config to a dictionary."""
        return _dataclass_to_dict(self)


def _dataclass_to_dict(instance: Any) -> Dict[str, Any]:
    if not is_dataclass(instance):
        raise TypeError("Expected dataclass instance.")
    output: Dict[str, Any] = {}
    for field_info in fields(instance):
        value = getattr(instance, field_info.name)
        if is_dataclass(value):
            output[field_info.name] = _dataclass_to_dict(value)
        elif isinstance(value, (tuple, list)):
            output[field_info.name] = list(value)
        else:
            output[field_info.name] = value
    return output


def load_config(path: Path | None = None) -> Config:
    """
    Load configuration from YAML if provided, otherwise return defaults.

    Parameters
    ----------
    path:
        Optional path to a YAML configuration file.
    """
    config = Config()
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    _apply_updates(config, raw)
    return config


def _apply_updates(instance: Any, updates: Dict[str, Any]) -> None:
    """Recursively apply overrides onto a dataclass instance."""
    for name, value in updates.items():
        if not hasattr(instance, name):
            continue
        current = getattr(instance, name)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_updates(current, value)
        elif is_dataclass(current) and value is None:
            continue
        else:
            setattr(instance, name, _coerce_value(current, value))


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, Path):
        return Path(value)
    if isinstance(current, tuple) and isinstance(value, Iterable):
        return tuple(value)
    return value

