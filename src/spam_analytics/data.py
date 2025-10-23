"""Dataset loading and splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


LABEL_NORMALIZATION = {
    "ham": "ham",
    "spam": "spam",
    "0": "ham",
    "1": "spam",
}


@dataclass
class DatasetBundle:
    """Structured representation of training data splits."""

    train_X: pd.DataFrame
    test_X: pd.DataFrame
    train_y: pd.Series
    test_y: pd.Series


class DataRepository:
    """Thin wrapper around dataset access patterns."""

    def __init__(self, config: Config):
        self.config = config

    def load_raw(self) -> pd.DataFrame:
        """Load the CSV dataset and normalize schema."""
        dataset_path: Path = self.config.paths.dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        df = pd.read_csv(
            dataset_path,
            encoding="utf-8",
            header=None,
            names=["label", "text"],
        )

        df["label"] = (
            df["label"].astype(str).str.strip().str.lower().map(LABEL_NORMALIZATION).fillna("ham")
        )
        df["text"] = df["text"].fillna("").astype(str)
        df["message_id"] = range(1, len(df) + 1)
        return df[["message_id", "label", "text"]]

    def split(
        self, features: pd.DataFrame, target: pd.Series
    ) -> DatasetBundle:
        """Perform a train/test split with reproducible settings."""
        cfg = self.config.training
        stratify = target if cfg.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify,
        )
        return DatasetBundle(X_train, X_test, y_train, y_test)

