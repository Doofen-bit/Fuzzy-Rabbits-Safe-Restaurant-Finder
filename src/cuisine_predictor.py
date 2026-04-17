"""
cuisine_predictor.py
--------------------
Predicts the cuisine type of a restaurant from its name alone, using scikit-learn.

Pipeline
--------
1. Normalise the restaurant name (lowercase, strip punctuation).
2. Vectorise with TF-IDF using *character* n-grams (2–4 chars) – these capture
   linguistic patterns (e.g. "burger", "pizza", "sushi") even for unseen names.
3. Classify with Logistic Regression (L2, balanced class weights) to produce
   calibrated probabilities for all cuisine classes.
4. Return the top-3 predicted cuisines with their softmax probabilities.

Two split strategies are supported
------------------------------------
"random"   – sklearn stratified train/test split on the full dataset.
"by_area"  – one entire NYC borough is held out as the test set; the rest train.
             This tests whether the model generalises geographically.

Only cuisine types with at least `min_cuisine_count` restaurants are retained so
that rare one-off labels don't pollute the classifier.
"""

from __future__ import annotations

import re
import string
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CUISINE_COUNT = 10   # drop cuisines with fewer than this many restaurants
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

SplitMethod = Literal["random", "by_area"]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    name = name.lower()
    name = name.translate(str.maketrans("", "", string.punctuation))
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_cuisine_data(
    df: pd.DataFrame,
    split_method: SplitMethod = "random",
    test_fraction: float = 0.20,
    test_area: str = "Manhattan",
    random_state: int = 42,
    min_cuisine_count: int = MIN_CUISINE_COUNT,
) -> tuple[
    list[str], list[str], list[str], list[str],   # X_train, X_test, y_train, y_test
    pd.DataFrame, pd.DataFrame,                    # train_df, test_df
    list[str],                                     # kept cuisine labels
]:
    """Split restaurant-name/cuisine pairs into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        restaurants.csv with at least ``dba`` (name) and ``cuisine`` columns,
        plus a ``boro`` column for the area split.
    split_method : {"random", "by_area"}
        How to split the data.
    test_fraction : float
        Fraction used as test when split_method == "random".
    test_area : str
        Borough name held out entirely as test when split_method == "by_area".
    random_state : int
        Seed for reproducible random splits.
    min_cuisine_count : int
        Cuisine types with fewer restaurants are dropped before splitting.

    Returns
    -------
    X_train, X_test : list[str]
        Normalised restaurant names.
    y_train, y_test : list[str]
        Cuisine labels.
    train_df, test_df : pd.DataFrame
        Original rows for reference (includes ``dba``, ``cuisine``, ``boro``).
    kept_cuisines : list[str]
        Sorted list of cuisine labels used in training.
    """
    # Keep rows with both a name and a cuisine
    needed = df[["dba", "cuisine", "boro"]].dropna(subset=["dba", "cuisine"]).copy()
    needed["dba"] = needed["dba"].astype(str).str.strip()
    needed["cuisine"] = needed["cuisine"].astype(str).str.strip()

    # Filter rare cuisines
    counts = needed["cuisine"].value_counts()
    kept = counts[counts >= min_cuisine_count].index.tolist()
    needed = needed[needed["cuisine"].isin(kept)].copy()

    # Normalise names
    needed["name_norm"] = needed["dba"].apply(_normalise)

    # Split
    if split_method == "by_area":
        # Normalise borough name for matching
        boros_in_data = needed["boro"].str.strip().str.title().unique()
        test_area_title = test_area.strip().title()
        train_df = needed[needed["boro"].str.strip().str.title() != test_area_title].copy()
        test_df  = needed[needed["boro"].str.strip().str.title() == test_area_title].copy()

        # Keep only cuisine labels present in training (test may have unseen labels)
        kept_in_train = train_df["cuisine"].value_counts()
        kept_in_train = kept_in_train[kept_in_train >= min_cuisine_count].index.tolist()
        train_df = train_df[train_df["cuisine"].isin(kept_in_train)].copy()
        test_df  = test_df[test_df["cuisine"].isin(kept_in_train)].copy()
        kept = sorted(kept_in_train)
    else:
        # Stratified random split
        # Need each cuisine to appear at least twice for stratify to work
        strat_counts = needed["cuisine"].value_counts()
        valid_strat = strat_counts[strat_counts >= 2].index
        needed = needed[needed["cuisine"].isin(valid_strat)].copy()

        train_df, test_df = train_test_split(
            needed,
            test_size=test_fraction,
            stratify=needed["cuisine"],
            random_state=random_state,
        )
        kept = sorted(needed["cuisine"].unique().tolist())

    X_train = train_df["name_norm"].tolist()
    X_test  = test_df["name_norm"].tolist()
    y_train = train_df["cuisine"].tolist()
    y_test  = test_df["cuisine"].tolist()

    return X_train, X_test, y_train, y_test, train_df, test_df, kept


# ---------------------------------------------------------------------------
# Cuisine Predictor
# ---------------------------------------------------------------------------

class CuisinePredictor:
    """TF-IDF + Logistic Regression cuisine-type predictor.

    Attributes
    ----------
    classes_ : list[str]
        All cuisine labels known to the model (sorted).
    vectorizer_ : TfidfVectorizer
        Fitted character-n-gram vectorizer.
    clf_ : LogisticRegression
        Fitted multi-class logistic regression.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 4),
        max_features: int = 50_000,
        C: float = 1.0,
        max_iter: int = 1_000,
        random_state: int = 42,
    ) -> None:
        self.ngram_range   = ngram_range
        self.max_features  = max_features
        self.C             = C
        self.max_iter      = max_iter
        self.random_state  = random_state

        self.vectorizer_: TfidfVectorizer | None = None
        self.clf_: LogisticRegression | None = None
        self.classes_: list[str] = []
        self.label_encoder_: LabelEncoder | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train: list[str], y_train: list[str]) -> "CuisinePredictor":
        """Fit the vectorizer and classifier on training names/labels."""
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        X_vec = self.vectorizer_.fit_transform(X_train)

        self.clf_ = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.clf_.fit(X_vec, y_train)
        self.classes_ = list(self.clf_.classes_)
        return self

    # ------------------------------------------------------------------
    def predict_top3(
        self, name: str
    ) -> list[tuple[str, float]]:
        """Return the top-3 predicted cuisines and their probabilities.

        Parameters
        ----------
        name : str
            Raw restaurant name (normalisation applied internally).

        Returns
        -------
        list of (cuisine, probability) tuples, sorted descending by probability.
        """
        if self.vectorizer_ is None or self.clf_ is None:
            raise RuntimeError("Call fit() before predict_top3().")

        norm = _normalise(name)
        X_vec = self.vectorizer_.transform([norm])
        proba = self.clf_.predict_proba(X_vec)[0]  # shape: (n_classes,)

        top3_idx = np.argsort(proba)[::-1][:3]
        return [(self.classes_[i], float(proba[i])) for i in top3_idx]

    # ------------------------------------------------------------------
    def predict_batch(self, names: list[str]) -> list[str]:
        """Return the single most-likely cuisine for each name (for eval)."""
        if self.vectorizer_ is None or self.clf_ is None:
            raise RuntimeError("Call fit() before predict_batch().")
        normed = [_normalise(n) for n in names]
        X_vec = self.vectorizer_.transform(normed)
        return list(self.clf_.predict(X_vec))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def cuisine_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Overall accuracy."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def top3_accuracy(
    predictor: CuisinePredictor, names: list[str], y_true: list[str]
) -> float:
    """Fraction of test examples where true label appears in top-3 predictions."""
    hits = 0
    for name, true_label in zip(names, y_true):
        top3 = predictor.predict_top3(name)
        if any(c == true_label for c, _ in top3):
            hits += 1
    return hits / len(y_true) if y_true else 0.0


def per_cuisine_f1(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> pd.DataFrame:
    """Per-class precision, recall, F1, support."""
    rows = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        support = sum(t == label for t in y_true)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)
        rows.append({
            "Cuisine":   label,
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "F1":        round(f1, 4),
            "Support":   support,
        })
    return pd.DataFrame(rows).sort_values("Support", ascending=False)
