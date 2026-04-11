"""
decision_tree.py
----------------
A fully from-scratch Decision Tree classifier for restaurant grade prediction.

Key improvements over the KNN baseline
---------------------------------------
1. **Weighted Gini splits** — each sample can carry an inverse-frequency class
   weight so that minority grades (B, C) contribute equally to the split
   criterion despite being far less common.
2. **Learned decision boundaries** — instead of memorising training points the
   tree discovers explicit thresholds ("if mean_score ≤ 11.2 AND critical_rate
   ≤ 0.43 → Grade A"), which generalise much better.
3. **Richer feature set** — five engineered features are added on top of the
   original four, giving the tree more signal per split.
4. **Feature importance** — the weighted Gini decrease accumulated at every
   split tells us which features actually drive the predictions.

No external ML libraries (sklearn, scipy …) are used anywhere in this file.
Only NumPy is used for numerical array operations.

Algorithm
---------
  • Criterion  : weighted Gini impurity
  • Thresholds : up to `n_thresholds` quantile-sampled values per feature
                 (avoids evaluating every unique value, keeps fitting O(n log n))
  • Splitting  : fully vectorised per feature (T×N matrix multiply for speed)
  • Stopping   : max_depth | min_samples_split | min_samples_leaf | pure leaf
  • Prediction : tree traversal, majority class or softmax-style probability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Extended feature set used by the Decision Tree
# ---------------------------------------------------------------------------
DT_BASE_FEATURES: list[str] = [
    "mean_score",
    "critical_violations",
    "total_violations",
    "days_since_last_inspection",
]

DT_EXTRA_FEATURES: list[str] = [
    "min_score",
    "max_score",
    "score_range",               # max − min  (score consistency)
    "inspection_count",
    "critical_rate",             # critical / total violations
    "violations_per_inspection", # total / inspection_count
]

DT_ALL_FEATURES: list[str] = DT_BASE_FEATURES + DT_EXTRA_FEATURES

VALID_GRADES: list[str] = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, feature_list: list[str] | None = None):
    """Add engineered columns to *df* and return X (np.ndarray) and feature names.

    Parameters
    ----------
    df : pd.DataFrame  (restaurants.csv schema)
    feature_list : list[str] | None
        Subset of DT_ALL_FEATURES to include.  None → all.

    Returns
    -------
    X          : np.ndarray  (n, n_features)
    feat_names : list[str]
    """
    import pandas as pd  # local import keeps module importable without pandas

    df = df.copy()
    df["score_range"] = (df["max_score"] - df["min_score"]).clip(lower=0)
    df["critical_rate"] = (
        df["critical_violations"] /
        df["total_violations"].clip(lower=1)
    )
    df["violations_per_inspection"] = (
        df["total_violations"] /
        df["inspection_count"].clip(lower=1)
    )

    if feature_list is None:
        feature_list = DT_ALL_FEATURES

    # Only keep features actually present in df
    valid = [f for f in feature_list if f in df.columns]
    X = df[valid].values.astype(float)
    return X, valid


# ---------------------------------------------------------------------------
# Internal tree node
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    # Leaf data
    is_leaf: bool = False
    label: str = ""
    proba: dict = field(default_factory=dict)
    # Internal node split
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    # Display / importance bookkeeping
    n_samples: int = 0
    weighted_n: float = 0.0
    gini: float = 0.0
    impurity_decrease: float = 0.0
    depth: int = 0


# ---------------------------------------------------------------------------
# Decision Tree classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """Weighted Decision Tree Classifier (from scratch, NumPy only).

    Parameters
    ----------
    max_depth : int
        Maximum tree depth (root is depth 0).
    min_samples_split : int
        A node must have at least this many raw samples to be split.
    min_samples_leaf : int
        Each child must have at least this many raw samples after a split.
    class_weight : 'balanced' | 'uniform'
        'balanced'  → weight[i] = n / (n_classes * count_of_class_i)
        'uniform'   → weight[i] = 1
    n_thresholds : int
        Number of quantile-sampled candidate thresholds per feature.
        Higher values are more accurate but slower.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        n_thresholds: int = 50,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.n_thresholds = n_thresholds
        self._root: _Node | None = None
        self._classes: np.ndarray | None = None
        self._feature_importances: np.ndarray | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights from the class_weight strategy."""
        if self.class_weight == "uniform":
            return np.ones(len(y), dtype=float)
        n = len(y)
        n_classes = len(self._classes)
        w = np.ones(n, dtype=float)
        for c in self._classes:
            mask = y == c
            cnt = mask.sum()
            if cnt > 0:
                w[mask] = n / (n_classes * cnt)
        return w

    @staticmethod
    def _gini(class_weights: np.ndarray, total_w: float) -> float:
        """Gini impurity given per-class weight sums and total weight."""
        if total_w <= 0:
            return 0.0
        p = class_weights / total_w
        return float(1.0 - np.dot(p, p))

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[int, float, float]:
        """Find the feature + threshold that maximises weighted Gini decrease.

        Fully vectorised over candidate thresholds using (T, N) matrix ops.

        Returns
        -------
        best_feature, best_threshold, best_decrease
        """
        n, n_feats = X.shape
        total_w = weights.sum()

        # Per-class weight indicator vectors  shape (C, N)
        C = len(self._classes)
        class_w = np.zeros((C, n), dtype=float)
        for ci, c in enumerate(self._classes):
            class_w[ci] = weights * (y == c)

        parent_class_w = class_w.sum(axis=1)  # (C,)
        parent_gini = self._gini(parent_class_w, total_w)

        best_feature = -1
        best_threshold = 0.0
        best_decrease = -np.inf

        q_grid = np.linspace(0.01, 0.99, self.n_thresholds)

        for f in range(n_feats):
            col = X[:, f]
            lo, hi = col.min(), col.max()
            if lo == hi:
                continue

            thresholds = np.unique(np.quantile(col, q_grid))
            T = len(thresholds)

            # left_masks[t, i] = True if col[i] <= thresholds[t]  shape (T, N)
            left_masks = col[None, :] <= thresholds[:, None]  # broadcast

            # left_total[t]  = sum of weights for samples going left at threshold t
            left_total = left_masks.astype(float) @ weights    # (T,)
            right_total = total_w - left_total                  # (T,)

            # Ignore splits where one child is empty
            valid = (left_total > 0) & (right_total > 0)
            if not valid.any():
                continue

            # Per-class weight sums for left children  shape (T, C)
            # left_cw[t, c] = sum_{i in left} class_w[c, i]
            left_cw = left_masks.astype(float) @ class_w.T  # (T, N) @ (N, C) = (T, C)
            right_cw = parent_class_w[None, :] - left_cw     # (T, C)

            # Gini for each split (vectorised)
            left_p  = left_cw  / (left_total[:, None]  + 1e-12)  # (T, C)
            right_p = right_cw / (right_total[:, None] + 1e-12)  # (T, C)

            left_gini  = 1.0 - (left_p  ** 2).sum(axis=1)  # (T,)
            right_gini = 1.0 - (right_p ** 2).sum(axis=1)  # (T,)

            weighted_child_gini = (left_total * left_gini + right_total * right_gini) / total_w
            decreases = parent_gini - weighted_child_gini  # (T,)
            decreases[~valid] = -np.inf

            best_t_idx = int(decreases.argmax())
            if decreases[best_t_idx] > best_decrease:
                best_decrease = float(decreases[best_t_idx])
                best_feature = f
                best_threshold = float(thresholds[best_t_idx])

        return best_feature, best_threshold, best_decrease

    def _leaf(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        n_samples: int,
        gini: float,
    ) -> _Node:
        """Create a leaf node predicting the majority weighted class."""
        C = len(self._classes)
        class_w = np.array([weights[y == c].sum() for c in self._classes])
        total_w = class_w.sum() + 1e-12
        proba = {str(c): float(w / total_w) for c, w in zip(self._classes, class_w)}
        label = str(self._classes[int(class_w.argmax())])
        node = _Node(
            is_leaf=True,
            label=label,
            proba=proba,
            n_samples=n_samples,
            weighted_n=float(weights.sum()),
            gini=gini,
            depth=depth,
        )
        return node

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        progress_state: dict | None,
    ) -> _Node:
        """Recursively build the tree, returning the root node."""
        n = len(y)

        # Compute current Gini for node stats
        C = len(self._classes)
        class_w = np.array([weights[y == c].sum() for c in self._classes])
        total_w = class_w.sum()
        gini = self._gini(class_w, total_w)

        # --- Stopping criteria ---
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or gini == 0.0  # pure node
        ):
            return self._leaf(y, weights, depth, n, gini)

        best_f, best_t, best_dec = self._best_split(X, y, weights)

        if best_f == -1 or best_dec <= 0.0:
            return self._leaf(y, weights, depth, n, gini)

        left_mask = X[:, best_f] <= best_t
        right_mask = ~left_mask

        # Enforce min_samples_leaf on raw (unweighted) counts
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return self._leaf(y, weights, depth, n, gini)

        # Accumulate feature importance
        self._feature_importances[best_f] += total_w * best_dec

        if progress_state is not None:
            progress_state["nodes"] += 1
            if progress_state.get("callback") and progress_state["nodes"] % 20 == 0:
                progress_state["callback"](progress_state["nodes"])

        node = _Node(
            is_leaf=False,
            feature=best_f,
            threshold=best_t,
            n_samples=n,
            weighted_n=float(total_w),
            gini=gini,
            impurity_decrease=best_dec,
            depth=depth,
        )
        node.left  = self._build(X[left_mask],  y[left_mask],  weights[left_mask],  depth + 1, progress_state)
        node.right = self._build(X[right_mask], y[right_mask], weights[right_mask], depth + 1, progress_state)
        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_callback: Callable[[int], None] | None = None,
    ) -> "DecisionTreeClassifier":
        """Fit the decision tree.

        Parameters
        ----------
        X : ndarray  (n_train, n_features)
        y : ndarray  (n_train,)  — string class labels
        progress_callback : callable(n_nodes_built) | None
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._classes = np.array(sorted(set(y)))
        self._n_features = X.shape[1]
        self._feature_importances = np.zeros(self._n_features, dtype=float)

        weights = self._sample_weights(y)
        ps = {"nodes": 0, "callback": progress_callback}
        self._root = self._build(X, y, weights, depth=0, progress_state=ps)

        # Normalise importances to sum to 1
        total_imp = self._feature_importances.sum()
        if total_imp > 0:
            self._feature_importances /= total_imp
        return self

    def _traverse(self, x: np.ndarray, node: _Node) -> _Node:
        """Walk the tree for a single sample, return the leaf node."""
        if node.is_leaf:
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.  Returns ndarray of str."""
        if self._root is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse(x, self._root).label for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        Returns
        -------
        ndarray, shape (n, n_classes)  — columns ordered as self.classes_
        """
        if self._root is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = np.asarray(X, dtype=float)
        classes = [str(c) for c in self._classes]
        rows = []
        for x in X:
            leaf = self._traverse(x, self._root)
            rows.append([leaf.proba.get(c, 0.0) for c in classes])
        return np.array(rows)

    # ------------------------------------------------------------------
    # Introspection / display helpers
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> list[str]:
        return [str(c) for c in self._classes] if self._classes is not None else []

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._feature_importances if self._feature_importances is not None else np.array([])

    def get_depth(self) -> int:
        """Return the maximum depth of the fitted tree."""
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self._root)

    def get_n_leaves(self) -> int:
        """Count the total number of leaf nodes."""
        def _leaves(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _leaves(node.left) + _leaves(node.right)
        return _leaves(self._root)

    def get_n_nodes(self) -> int:
        """Count all nodes (internal + leaves)."""
        def _count(node):
            if node is None:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        return _count(self._root)

    def extract_top_rules(
        self, feature_names: list[str], max_rules: int = 15
    ) -> list[dict]:
        """Walk the tree depth-first; return the `max_rules` leaf paths
        with the most training samples.

        Each returned dict has:
          conditions : list[str]
          label      : str
          n_samples  : int
          proba      : dict[str, float]
        """
        rules: list[dict] = []

        def _walk(node: _Node, conditions: list[str]) -> None:
            if node is None:
                return
            if node.is_leaf:
                rules.append({
                    "conditions": conditions.copy(),
                    "label":      node.label,
                    "n_samples":  node.n_samples,
                    "proba":      node.proba,
                })
                return
            fname = feature_names[node.feature] if node.feature < len(feature_names) else f"f{node.feature}"
            _walk(node.left,  conditions + [f"{fname} <= {node.threshold:.3g}"])
            _walk(node.right, conditions + [f"{fname} > {node.threshold:.3g}"])

        _walk(self._root, [])
        rules.sort(key=lambda r: r["n_samples"], reverse=True)
        return rules[:max_rules]

    def describe_tree(self, feature_names: list[str], max_depth: int = 4) -> str:
        """Return an indented text representation of the top levels."""
        lines: list[str] = []

        def _fmt(node: _Node, indent: int, prefix: str) -> None:
            if node is None or indent // 2 > max_depth:
                return
            pad = "  " * indent
            if node.is_leaf:
                proba_str = "  ".join(f"{k}:{v:.2f}" for k, v in node.proba.items())
                lines.append(f"{pad}{prefix}[LEAF] → {node.label}  ({proba_str})  n={node.n_samples}")
                return
            fname = feature_names[node.feature] if node.feature < len(feature_names) else f"f{node.feature}"
            lines.append(f"{pad}{prefix}if {fname} <= {node.threshold:.3g}  "
                         f"[n={node.n_samples}, gini={node.gini:.3f}]")
            _fmt(node.left,  indent + 1, "├─ YES: ")
            _fmt(node.right, indent + 1, "└─ NO:  ")

        _fmt(self._root, 0, "")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data preparation (temporal split, same logic as KNN)
# ---------------------------------------------------------------------------

def prepare_dt_data(
    restaurants,
    feature_list: list[str] | None = None,
    test_fraction: float = 0.20,
):
    """Temporally correct train/test split with engineered features.

    Parameters
    ----------
    restaurants : pd.DataFrame  (restaurants.csv schema)
    feature_list : list[str] | None — subset of DT_ALL_FEATURES; None → all
    test_fraction : float

    Returns
    -------
    X_train, y_train, X_test, y_test,
    train_df, test_df,
    cutoff_date,
    feat_names
    """
    import pandas as pd

    df = restaurants[restaurants["latest_grade"].isin(VALID_GRADES)].copy()

    # Need these columns to exist
    required = ["mean_score", "critical_violations", "total_violations",
                "days_since_last_inspection", "latest_inspection_date",
                "inspection_count", "min_score", "max_score"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Sort temporally
    df = df.sort_values("latest_inspection_date").reset_index(drop=True)

    n = len(df)
    cutoff_idx = int(n * (1.0 - test_fraction))
    cutoff_date = df["latest_inspection_date"].iloc[cutoff_idx]

    train_df = df.iloc[:cutoff_idx].copy()
    test_df  = df.iloc[cutoff_idx:].copy()

    X_train, feat_names = engineer_features(train_df, feature_list)
    X_test,  _          = engineer_features(test_df,  feature_list)

    y_train = train_df["latest_grade"].values
    y_test  = test_df["latest_grade"].values

    return X_train, y_train, X_test, y_test, train_df, test_df, cutoff_date, feat_names
