"""Model training and persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


class ModelTrainer:
    """Assemble and fit the scikit-learn pipeline."""

    def __init__(self, config: Config):
        self.config = config

    def build_pipeline(self) -> Pipeline:
        """Construct the feature + classifier pipeline."""
        vector_cfg = self.config.vectorizer
        text_transformer = TfidfVectorizer(
            ngram_range=vector_cfg.ngram_range,
            analyzer=vector_cfg.analyzer,
            max_df=vector_cfg.max_df,
            min_df=vector_cfg.min_df,
            max_features=vector_cfg.max_features,
            sublinear_tf=vector_cfg.sublinear_tf,
            norm=vector_cfg.norm,
        )

        numeric_features = list(self.config.numeric_features)

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", text_transformer, "clean_text"),
                ("numeric", StandardScaler(with_mean=False), numeric_features),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        model_cfg = self.config.model
        classifier = LogisticRegression(
            penalty=model_cfg.penalty,
            C=model_cfg.C,
            solver=model_cfg.solver,
            max_iter=model_cfg.max_iter,
            class_weight=model_cfg.class_weight,
            random_state=model_cfg.random_state,
        )

        return Pipeline(
            steps=[
                ("features", preprocessor),
                ("classifier", classifier),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Fit a new pipeline instance on the provided data."""
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline

    def save(self, pipeline: Pipeline) -> Path:
        """Persist the trained pipeline to disk."""
        model_path = self.config.paths.model_path
        self.config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        return model_path

    def load_existing(self) -> Pipeline:
        """Load a previously persisted pipeline."""
        model_path = self.config.paths.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                "Model artifact not found. Train a model first using `spam-analytics model train`."
            )
        return joblib.load(model_path)


def extract_top_features(
    pipeline: Pipeline, top_n: int, positive_label: str = "spam"
) -> Dict[str, pd.DataFrame]:
    """
    Extract top weighted features for each class.

    Returns a dictionary with positive and negative class rankings.
    """
    column_transformer: ColumnTransformer = pipeline.named_steps["features"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]

    vectorizer: TfidfVectorizer = column_transformer.named_transformers_["text"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    top_positive_idx = coefs.argsort()[-top_n:][::-1]
    top_negative_idx = coefs.argsort()[:top_n]

    top_positive = pd.DataFrame(
        {"feature": feature_names[top_positive_idx], "weight": coefs[top_positive_idx]}
    )
    top_negative = pd.DataFrame(
        {"feature": feature_names[top_negative_idx], "weight": coefs[top_negative_idx]}
    )

    return {
        "spam": top_positive.reset_index(drop=True),
        "ham": top_negative.reset_index(drop=True),
    }

