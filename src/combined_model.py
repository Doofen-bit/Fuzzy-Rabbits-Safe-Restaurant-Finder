"""
combined_model.py  —  Part 6
-----------------------------
Blends KNN (Part 2) and Decision Tree (Part 3) grade predictions into a
single continuous safety score used as the RL reward signal.

Why blend?
----------
• KNN skews heavily toward Grade A because class A makes up ~86 % of
  graded restaurants and the majority-vote rule defaults to the dominant
  class.  This causes the reward grid to treat nearly every restaurant
  as a Grade-A establishment — making the RL largely useless.

• Decision Tree uses balanced class weights so it recovers B and C grades
  well, but can over-correct toward non-A predictions.

Blending the two probability distributions gives a more calibrated estimate:

    P_combined(g) = α · P_KNN(g) + (1−α) · P_DT(g)
    safety_score  = 3·P(A) + 2·P(B) + 1·P(C)   ∈ [1, 3]

This replaces the binary grade look-up (A→3, B→2, C→1) used in Parts 1–5,
producing a smooth, model-driven RL reward signal.

Efficiency
----------
KNN inference requires cosine-similarity against all ~24 k training
restaurants.  predict_safety_scores() processes query rows in chunks of
size `chunk_size` (default 256) to bound peak RAM usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.knn_classifier import KNNClassifier, FEATURES as KNN_FEATURES, VALID_GRADES
from src.decision_tree import (
    DecisionTreeClassifier,
    engineer_features,
    DT_ALL_FEATURES,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES        = ["A", "B", "C"]
SAFETY_WEIGHTS = np.array([3.0, 2.0, 1.0])      # A=3, B=2, C=1
GRADE_FALLBACK = {"A": 3.0, "B": 2.0, "C": 1.0}

# Columns the models need
_KNN_REQUIRED = KNN_FEATURES   # mean_score, critical_violations, total_violations,
                                #  days_since_last_inspection
_DT_REQUIRED  = [              # superset of KNN_REQUIRED
    "mean_score", "critical_violations", "total_violations",
    "days_since_last_inspection", "inspection_count",
    "min_score", "max_score",
]


# ── Helper ────────────────────────────────────────────────────────────────────

def _align_proba(proba_raw: np.ndarray, source_classes: list[str]) -> np.ndarray:
    """Reorder / pad a (n, k) probability array to the canonical CLASSES order."""
    aligned = np.zeros((len(proba_raw), len(CLASSES)), dtype=float)
    for gi, grade in enumerate(CLASSES):
        if grade in source_classes:
            di = source_classes.index(grade)
            aligned[:, gi] = proba_raw[:, di]
    return aligned


# ── Main class ────────────────────────────────────────────────────────────────

class CombinedGradePredictor:
    """Weighted blend of KNN and Decision Tree grade probability distributions.

    Parameters
    ----------
    knn_weight : float
        Weight given to KNN probabilities (0–1).  DT weight = 1 − knn_weight.
        Default 0.35 because DT is better calibrated for minority grades.
    k : int
        KNN neighbourhood size.
    dt_max_depth : int
        Decision Tree max depth.
    """

    def __init__(
        self,
        knn_weight:   float = 0.35,
        k:            int   = 7,
        dt_max_depth: int   = 12,
    ) -> None:
        self.knn_weight   = max(0.0, min(1.0, knn_weight))
        self.dt_weight    = 1.0 - self.knn_weight
        self.k            = k
        self.dt_max_depth = dt_max_depth

        self._knn:           KNNClassifier | None          = None
        self._dt:            DecisionTreeClassifier | None = None
        self._dt_feat_names: list[str]                     = []
        self._n_train:       int                           = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        restaurants: pd.DataFrame,
        dt_progress_callback=None,
    ) -> "CombinedGradePredictor":
        """Train both models on all graded restaurants.

        Parameters
        ----------
        restaurants : pd.DataFrame
            Full restaurants.csv DataFrame (one row per restaurant).
        dt_progress_callback : callable(n_nodes) | None
            Passed to DecisionTreeClassifier.fit(); used for Streamlit progress.
        """
        df = restaurants[restaurants["latest_grade"].isin(VALID_GRADES)].copy()
        df = df.dropna(subset=_KNN_REQUIRED + ["latest_inspection_date"])
        self._n_train = len(df)

        y = df["latest_grade"].values

        # ── 1. KNN ───────────────────────────────────────────────────────
        X_knn = df[_KNN_REQUIRED].values.astype(float)
        self._knn = KNNClassifier(k=self.k)
        self._knn.fit(X_knn, y)

        # ── 2. Decision Tree ─────────────────────────────────────────────
        X_dt, feat_names = engineer_features(df)
        self._dt_feat_names = feat_names
        self._dt = DecisionTreeClassifier(
            max_depth=self.dt_max_depth,
            class_weight="balanced",
        )
        self._dt.fit(X_dt, y, progress_callback=dt_progress_callback)

        return self

    # ------------------------------------------------------------------
    # KNN probabilities (batched)
    # ------------------------------------------------------------------

    def _knn_proba_chunk(self, X_chunk: np.ndarray) -> np.ndarray:
        """Return (batch, 3) KNN probability matrix [P(A), P(B), P(C)]."""
        knn    = self._knn
        X_norm = knn._zscore(np.asarray(X_chunk, dtype=float))

        # Cosine similarity: (batch, n_train)
        dots    = X_norm @ knn._X_train.T
        norms_q = np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-10
        norms_t = np.linalg.norm(knn._X_train, axis=1)          + 1e-10
        sims    = dots / (norms_q * norms_t)

        k     = min(self.k, knn._X_train.shape[0])
        proba = np.zeros((len(X_chunk), 3), dtype=float)

        for i in range(len(X_chunk)):
            top_idx   = np.argpartition(sims[i], -k)[-k:]
            neighbors = knn._y_train[top_idx]
            for gi, grade in enumerate(CLASSES):
                proba[i, gi] = float(np.sum(neighbors == grade)) / k

        return proba

    def _knn_proba(
        self,
        X_raw:      np.ndarray,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Chunked KNN probability prediction for arbitrary-size arrays."""
        n      = len(X_raw)
        proba  = np.zeros((n, 3), dtype=float)
        for start in range(0, n, chunk_size):
            end            = min(start + chunk_size, n)
            proba[start:end] = self._knn_proba_chunk(X_raw[start:end])
        return proba

    # ------------------------------------------------------------------
    # DT probabilities
    # ------------------------------------------------------------------

    def _dt_proba(self, df_sub: pd.DataFrame) -> np.ndarray:
        """Return (n, 3) DT probability matrix [P(A), P(B), P(C)]."""
        X_dt, _ = engineer_features(df_sub, self._dt_feat_names)
        raw      = self._dt.predict_proba(X_dt)
        return _align_proba(raw, self._dt.classes_)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_safety_scores(
        self,
        df:         pd.DataFrame,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Compute a continuous safety score ∈ [1, 3] for every row in *df*.

        Algorithm
        ---------
        1. Identify rows that have all required features.
        2. For those rows, blend KNN and DT probability distributions.
        3. Map blended proba to safety score:  score = 3·P(A) + 2·P(B) + 1·P(C).
        4. Rows missing features fall back to the actual grade look-up (or 2.0).

        Parameters
        ----------
        df : pd.DataFrame
            restaurants.csv schema; may contain NaN features.
        chunk_size : int
            KNN prediction chunk size (trades speed for RAM).

        Returns
        -------
        scores : np.ndarray, shape (len(df),)
        """
        if self._knn is None or self._dt is None:
            raise RuntimeError("Call fit() before predict_safety_scores().")

        n      = len(df)
        scores = np.full(n, 2.0, dtype=float)   # neutral default

        # Grade fallback for rows that will miss the combined path
        if "latest_grade" in df.columns:
            for i, g in enumerate(df["latest_grade"].values):
                if g in GRADE_FALLBACK:
                    scores[i] = GRADE_FALLBACK[g]

        # Rows with all required features get the blended score
        avail   = [c for c in set(_KNN_REQUIRED) | set(_DT_REQUIRED) if c in df.columns]
        has_all = df[avail].notna().all(axis=1)

        if not has_all.any():
            return scores

        df_sub  = df[has_all].copy()
        X_knn   = df_sub[_KNN_REQUIRED].values.astype(float)

        knn_p   = self._knn_proba(X_knn, chunk_size=chunk_size)  # (m, 3)
        dt_p    = self._dt_proba(df_sub)                          # (m, 3)

        blended     = self.knn_weight * knn_p + self.dt_weight * dt_p
        sub_scores  = blended @ SAFETY_WEIGHTS

        scores[np.where(has_all.values)[0]] = sub_scores
        return scores

    def predict_grade_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted grade (A / B / C) based on the blended safety score."""
        scores = self.predict_safety_scores(df)
        return np.where(scores >= 2.5, "A", np.where(scores >= 1.5, "B", "C"))

    def predict_proba_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns P_A, P_B, P_C, predicted_grade,
        knn_grade, dt_grade, safety_score — useful for Streamlit visualisation.
        """
        has_all = df[[c for c in set(_KNN_REQUIRED) | set(_DT_REQUIRED)
                       if c in df.columns]].notna().all(axis=1)
        df_sub  = df[has_all].copy()
        X_knn   = df_sub[_KNN_REQUIRED].values.astype(float)

        knn_p   = self._knn_proba(X_knn)
        dt_p    = self._dt_proba(df_sub)
        blended = self.knn_weight * knn_p + self.dt_weight * dt_p

        out = pd.DataFrame({
            "P_A":           blended[:, 0].round(4),
            "P_B":           blended[:, 1].round(4),
            "P_C":           blended[:, 2].round(4),
            "knn_pred":      np.array(CLASSES)[knn_p.argmax(axis=1)],
            "dt_pred":       np.array(CLASSES)[dt_p.argmax(axis=1)],
            "combined_pred": np.array(CLASSES)[blended.argmax(axis=1)],
            "safety_score":  (blended @ SAFETY_WEIGHTS).round(3),
        }, index=df_sub.index)

        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def dt_feature_importances(self) -> dict[str, float]:
        if self._dt is None:
            return {}
        return dict(zip(self._dt_feat_names, self._dt.feature_importances_.tolist()))
