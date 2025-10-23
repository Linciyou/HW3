"""Text preprocessing utilities for the spam analytics toolkit."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import Config

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b\S+@\S+\.\S+\b")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_WORD_PATTERN = re.compile(r"[^\w\s]")
DIGIT_PATTERN = re.compile(r"\d+")
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass
class PreprocessingResult:
    """Container for preprocessing outputs."""

    data: pd.DataFrame
    text_column: str = "normalized_text"
    final_text_column: str = "clean_text"


class TextPreprocessor:
    """Perform normalization, tokenization, and feature extraction."""

    def __init__(self, config: Config):
        self.config = config
        if config.preprocessing.use_stemmer:
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = None
        stop_words = set(ENGLISH_STOP_WORDS)
        if config.preprocessing.extra_stop_words:
            stop_words.update(word.lower() for word in config.preprocessing.extra_stop_words)
        self.stop_words = stop_words

    def run(self, df: pd.DataFrame) -> PreprocessingResult:
        """Return dataframe with intermediate preprocessing steps."""
        processed = df.copy()
        processed["normalized_text"] = processed["text"].apply(self._normalize_text)
        processed["tokens"] = processed["normalized_text"].apply(self._tokenize)
        processed["filtered_tokens"] = processed["tokens"].apply(self._remove_stopwords)
        processed["stemmed_tokens"] = processed["filtered_tokens"].apply(self._stem_tokens)
        processed["clean_text"] = processed["stemmed_tokens"].apply(lambda tokens: " ".join(tokens))

        # Feature engineering
        processed["char_count"] = processed["text"].str.len()
        processed["word_count"] = processed["tokens"].apply(len)
        processed["avg_word_len"] = processed.apply(
            lambda row: (sum(len(token) for token in row["tokens"]) / row["word_count"])
            if row["word_count"] else 0.0,
            axis=1,
        )
        processed["digit_ratio"] = processed["text"].apply(self._digit_ratio)
        processed["punct_ratio"] = processed["text"].apply(self._punct_ratio)
        processed["upper_ratio"] = processed["text"].apply(self._uppercase_ratio)
        processed["unique_char_ratio"] = processed["normalized_text"].apply(self._unique_char_ratio)

        return PreprocessingResult(data=processed)

    #
    # Internal helpers
    #
    def _normalize_text(self, text: str) -> str:
        cfg = self.config.preprocessing
        working = text or ""
        if cfg.strip_html:
            working = HTML_PATTERN.sub(" ", working)
        if cfg.strip_urls:
            working = URL_PATTERN.sub(" ", working)
        if cfg.strip_emails:
            working = EMAIL_PATTERN.sub(" ", working)
        if cfg.lowercase:
            working = working.lower()
        if cfg.strip_accents:
            working = unicodedata.normalize("NFKD", working).encode("ascii", "ignore").decode("utf-8", errors="ignore")
        if cfg.strip_numbers and not cfg.keep_internal_digits:
            working = DIGIT_PATTERN.sub(" ", working)
        if cfg.strip_punctuation:
            working = NON_WORD_PATTERN.sub(" ", working)
        if cfg.collapse_whitespace:
            working = MULTISPACE_PATTERN.sub(" ", working)
        return working.strip()

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text)
        min_len = self.config.preprocessing.min_token_length
        if min_len > 0:
            tokens = [token for token in tokens if len(token) >= min_len]
        return tokens

    def _remove_stopwords(self, tokens: Sequence[str]) -> List[str]:
        if not self.stop_words:
            return list(tokens)
        return [token for token in tokens if token not in self.stop_words]

    def _stem_tokens(self, tokens: Sequence[str]) -> List[str]:
        if not self.stemmer:
            return list(tokens)
        return [self.stemmer.stem(token) for token in tokens]

    @staticmethod
    def _digit_ratio(text: str) -> float:
        if not text:
            return 0.0
        digit_count = sum(char.isdigit() for char in text)
        return digit_count / len(text)

    @staticmethod
    def _punct_ratio(text: str) -> float:
        if not text:
            return 0.0
        punct_count = sum(1 for char in text if re.match(r"[^\w\s]", char))
        return punct_count / len(text)

    @staticmethod
    def _uppercase_ratio(text: str) -> float:
        if not text:
            return 0.0
        upper_count = sum(1 for char in text if char.isupper())
        return upper_count / len(text)

    @staticmethod
    def _unique_char_ratio(text: str) -> float:
        filtered = text.replace(" ", "")
        if not filtered:
            return 0.0
        return len(set(filtered)) / len(filtered)

