"""
knn_classifier.py
-----------------
From-scratch K-Nearest Neighbors classifier for restaurant grade prediction.

Features used per restaurant:
  - mean_score               : mean inspection score across all inspections
  - critical_violations      : total critical violation count
  - total_violations         : total violation count
  - days_since_last_inspection: recency of most recent inspection

Algorithm
---------
1. Z-score normalise the training feature matrix.
2. For each query restaurant, compute cosine similarity against every training
   restaurant (fully vectorised via NumPy; chunked for memory safety).
3. Collect the K training restaurants with highest cosine similarity.
4. Return the majority grade (A / B / C) among those K neighbours.

Evaluation
----------
Precision, recall, and F1-score are implemented from scratch for each class
and as macro averages.

Temporal train/test split
--------------------------
To ensure the model never sees future inspections during training we split by
`latest_inspection_date`:
  • training   – restaurants whose most-recent inspection is BEFORE the cutoff
  • test        – restaurants whose most-recent inspection is ON/AFTER the cutoff

This guarantees no temporal leakage: a training restaurant's feature vector
is derived purely from its own history, independently of any test restaurant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FEATURES: list[str] = [
    "mean_score",
    "critical_violations",
    "total_violations",
    "days_since_last_inspection",
]
TARGET: str = "latest_grade"
VALID_GRADES: list[str] = ["A", "B", "C"]


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_knn_data(
    restaurants: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Build a temporally-correct train/test split from ``restaurants.csv``.

    Parameters
    ----------
    restaurants : pd.DataFrame
        Output of ``preprocessor.build_restaurant_table()`` (or the saved CSV).
    test_fraction : float
        Fraction of restaurants (sorted by ``latest_inspection_date``) to put
        in the test set.  E.g. 0.20 keeps the 20 % most-recently-inspected
        restaurants as the held-out test set.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    train_df, test_df               : filtered DataFrames (for display)
    cutoff_date                     : the date used as the split boundary
    """
    # Keep only rows that have all required features AND a letter grade
    df = restaurants[restaurants[TARGET].isin(VALID_GRADES)].copy()
    df = df.dropna(subset=FEATURES + ["latest_inspection_date"])

    # Sort by inspection date, newest last
    df = df.sort_values("latest_inspection_date").reset_index(drop=True)

    n = len(df)
    cutoff_idx = int(n * (1.0 - test_fraction))
    cutoff_date: pd.Timestamp = df["latest_inspection_date"].iloc[cutoff_idx]

    train_df = df.iloc[:cutoff_idx].copy()
    test_df = df.iloc[cutoff_idx:].copy()

    X_train = train_df[FEATURES].values.astype(float)
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values.astype(float)
    y_test = test_df[TARGET].values

    return X_train, y_train, X_test, y_test, train_df, test_df, cutoff_date


# ──────────────────────────────────────────────────────────────────────────────
# KNN classifier
# ──────────────────────────────────────────────────────────────────────────────

class KNNClassifier:
    """K-Nearest Neighbours classifier using cosine similarity.

    No external ML libraries are used; only NumPy for numerical operations.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to consider.
    """

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zscore(self, X: np.ndarray) -> np.ndarray:
        """Apply stored z-score normalisation to X."""
        return (X - self._mean) / (self._std + 1e-8)

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> str:
        """Return the most frequent label in *labels*."""
        unique, counts = np.unique(labels, return_counts=True)
        return str(unique[np.argmax(counts)])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store the training data after z-score normalisation.

        Parameters
        ----------
        X : ndarray, shape (n_train, n_features)
        y : ndarray, shape (n_train,)   – string class labels
        """
        X = np.asarray(X, dtype=float)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._X_train = self._zscore(X)
        self._y_train = np.asarray(y)
        return self

    def predict_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Predict grades for a batch of test samples (vectorised).

        Parameters
        ----------
        X_batch : ndarray, shape (batch_size, n_features)

        Returns
        -------
        predictions : ndarray, shape (batch_size,) of str
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict_batch().")

        X_norm = self._zscore(np.asarray(X_batch, dtype=float))  # (B, F)

        # Cosine similarity: (B, N_train)
        dots = X_norm @ self._X_train.T                          # (B, N)
        norms_batch = np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-10  # (B, 1)
        norms_train = np.linalg.norm(self._X_train, axis=1) + 1e-10          # (N,)
        sims = dots / (norms_batch * norms_train)                # (B, N)

        # Top-k indices for each test sample (partial sort for efficiency)
        k = min(self.k, self._X_train.shape[0])
        top_k_indices = np.argpartition(sims, -k, axis=1)[:, -k:]

        predictions = np.array([
            self._majority_vote(self._y_train[idx])
            for idx in top_k_indices
        ])
        return predictions

    def predict(
        self,
        X_test: np.ndarray,
        chunk_size: int = 256,
        progress_callback=None,
    ) -> np.ndarray:
        """Predict grades for the full test set, processing in chunks.

        Parameters
        ----------
        X_test : ndarray, shape (n_test, n_features)
        chunk_size : int
            Number of test samples processed per batch (trades speed for RAM).
        progress_callback : callable(float) | None
            Called with a float in [0, 1] after each chunk; useful for
            Streamlit progress bars.

        Returns
        -------
        predictions : ndarray, shape (n_test,) of str
        """
        X_test = np.asarray(X_test, dtype=float)
        n = len(X_test)
        results: list[np.ndarray] = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch_preds = self.predict_batch(X_test[start:end])
            results.append(batch_preds)
            if progress_callback is not None:
                progress_callback(end / n)

        return np.concatenate(results)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def feature_means(self) -> np.ndarray | None:
        return self._mean

    @property
    def feature_stds(self) -> np.ndarray | None:
        return self._std


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (from scratch — no sklearn)
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str] | None = None,
) -> dict:
    """Compute per-class and macro precision, recall, F1 from scratch.

    Parameters
    ----------
    y_true, y_pred : array-like of str
    classes : list of str, optional
        If None, uses the union of y_true and y_pred.

    Returns
    -------
    metrics : dict
        ``metrics[label]`` → ``{"precision": float, "recall": float, "f1": float}``
        ``metrics["macro"]`` → macro-averaged values
        ``metrics["accuracy"]`` → overall accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))

    per_class: dict[str, dict[str, float]] = {}
    for label in classes:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(np.sum(y_true == label))

        per_class[label] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "tp":        tp,
            "fp":        fp,
            "fn":        fn,
            "support":   support,
        }

    # Macro averages
    macro_p  = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_r  = float(np.mean([v["recall"]    for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"]        for v in per_class.values()]))
    per_class["macro"] = {
        "precision": round(macro_p,  4),
        "recall":    round(macro_r,  4),
        "f1":        round(macro_f1, 4),
        "support":   len(y_true),
    }

    accuracy = float(np.sum(y_true == y_pred)) / len(y_true)
    per_class["accuracy"] = {"value": round(accuracy, 4)}

    return per_class


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    """Return a (len(classes) x len(classes)) confusion matrix as a NumPy array.

    Row = actual class, Column = predicted class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            cm[i, j] = int(np.sum((y_true == actual) & (y_pred == predicted)))
    return cm
