"""
decision_tree.py
----------------
Part 3: From-scratch weighted Decision Tree classifier for restaurant grade
prediction in the NYC Safe Restaurant Finder project.

This file intentionally does not use scikit-learn or any external ML library.
The Decision Tree algorithm itself is implemented with NumPy only.

What this file provides
-----------------------
1. Weighted Gini impurity for class imbalance.
2. A from-scratch threshold-based Decision Tree classifier.
3. Engineered features:
   - min_score
   - max_score
   - score_range
   - inspection_count
   - critical_rate
   - violations_per_inspection
4. Feature importances from total weighted Gini decrease.
5. Top decision rules for interpretability.
6. A readable tree sketch for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Feature constants used by Streamlit and Part 6 combined model
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
    "score_range",
    "inspection_count",
    "critical_rate",
    "violations_per_inspection",
]

DT_ALL_FEATURES: list[str] = DT_BASE_FEATURES + DT_EXTRA_FEATURES

VALID_GRADES: list[str] = ["A", "B", "C"]
TARGET: str = "latest_grade"
DATE_COL: str = "latest_inspection_date"

# Raw columns needed to build each possible model feature.
_FEATURE_DEPENDENCIES: dict[str, list[str]] = {
    "mean_score": ["mean_score"],
    "critical_violations": ["critical_violations"],
    "total_violations": ["total_violations"],
    "days_since_last_inspection": ["days_since_last_inspection"],
    "min_score": ["min_score"],
    "max_score": ["max_score"],
    "score_range": ["min_score", "max_score"],
    "inspection_count": ["inspection_count"],
    "critical_rate": ["critical_violations", "total_violations"],
    "violations_per_inspection": ["total_violations", "inspection_count"],
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, feature_list: list[str] | None = None):
    """Add engineered feature columns and return X plus feature names.

    Parameters
    ----------
    df : pandas.DataFrame
        Restaurant-level table produced by src.preprocessor.
    feature_list : list[str] | None
        Requested features. If None, all Decision Tree features are used.

    Returns
    -------
    X : np.ndarray
        Numeric feature matrix with shape (n_rows, n_features).
    feat_names : list[str]
        Feature names in the same column order as X.
    """
    df = df.copy()

    if feature_list is None:
        feature_list = DT_ALL_FEATURES

    valid_features: list[str] = []

    for feature in feature_list:
        dependencies = _FEATURE_DEPENDENCIES.get(feature)
        if dependencies is None:
            continue
        if all(col in df.columns for col in dependencies):
            valid_features.append(feature)

    if not valid_features:
        raise ValueError(
            "No valid Decision Tree features are available in the input DataFrame."
        )

    if "score_range" in valid_features:
        df["score_range"] = (df["max_score"] - df["min_score"]).clip(lower=0)

    if "critical_rate" in valid_features:
        df["critical_rate"] = (
            df["critical_violations"] / df["total_violations"].clip(lower=1)
        )

    if "violations_per_inspection" in valid_features:
        df["violations_per_inspection"] = (
            df["total_violations"] / df["inspection_count"].clip(lower=1)
        )

    X = (
        df[valid_features]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .values
    )

    if np.isnan(X).any():
        raise ValueError(
            "Decision Tree feature matrix contains NaN values. "
            "Use prepare_dt_data() or drop rows with missing required features."
        )

    return X, valid_features


# ---------------------------------------------------------------------------
# Internal tree node
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    # Leaf data
    is_leaf: bool = False
    label: str = ""
    proba: dict[str, float] = field(default_factory=dict)

    # Internal split data
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    # Display and bookkeeping
    n_samples: int = 0
    weighted_n: float = 0.0
    gini: float = 0.0
    impurity_decrease: float = 0.0
    depth: int = 0


# ---------------------------------------------------------------------------
# Decision Tree classifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """Weighted Decision Tree classifier implemented from scratch with NumPy.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth. Root depth is 0.
    min_samples_split : int
        Minimum raw sample count needed to split a node.
    min_samples_leaf : int
        Minimum raw sample count required in each child node.
    class_weight : {"balanced", "uniform"}
        "balanced" gives each class equal total weight using:
        n / (n_classes * class_count).
        "uniform" gives each row weight 1.
    n_thresholds : int
        Number of quantile-sampled candidate thresholds per feature.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        n_thresholds: int = 50,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if class_weight not in {"balanced", "uniform"}:
            raise ValueError("class_weight must be either 'balanced' or 'uniform'.")
        if n_thresholds < 1:
            raise ValueError("n_thresholds must be at least 1.")

        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.class_weight = class_weight
        self.n_thresholds = int(n_thresholds)

        self._root: _Node | None = None
        self._classes: np.ndarray | None = None
        self._feature_importances: np.ndarray | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Weighting and impurity helpers
    # ------------------------------------------------------------------

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Return one weight per training sample."""
        if self._classes is None:
            raise RuntimeError("Classes are not initialized.")

        if self.class_weight == "uniform":
            return np.ones(len(y), dtype=float)

        n_samples = len(y)
        n_classes = len(self._classes)
        weights = np.ones(n_samples, dtype=float)

        for cls in self._classes:
            mask = y == cls
            class_count = int(mask.sum())
            if class_count > 0:
                weights[mask] = n_samples / (n_classes * class_count)

        return weights

    @staticmethod
    def _gini(class_weight_sums: np.ndarray, total_weight: float) -> float:
        """Compute weighted Gini impurity."""
        if total_weight <= 0:
            return 0.0

        probabilities = class_weight_sums / total_weight
        return float(1.0 - np.sum(probabilities ** 2))

    # ------------------------------------------------------------------
    # Split search
    # ------------------------------------------------------------------

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[int, float, float]:
        """Find the best feature and threshold for one node."""
        if self._classes is None:
            raise RuntimeError("Call fit() before searching for splits.")

        n_samples, n_features = X.shape
        total_weight = float(weights.sum())
        n_classes = len(self._classes)

        class_weight_matrix = np.zeros((n_classes, n_samples), dtype=float)
        for class_index, cls in enumerate(self._classes):
            class_weight_matrix[class_index] = weights * (y == cls)

        parent_class_weights = class_weight_matrix.sum(axis=1)
        parent_gini = self._gini(parent_class_weights, total_weight)

        best_feature = -1
        best_threshold = 0.0
        best_decrease = -np.inf

        quantile_grid = np.linspace(0.01, 0.99, self.n_thresholds)

        for feature_index in range(n_features):
            column = X[:, feature_index]

            if np.nanmin(column) == np.nanmax(column):
                continue

            thresholds = np.unique(np.quantile(column, quantile_grid))

            if len(thresholds) == 0:
                continue

            left_masks_bool = column[None, :] <= thresholds[:, None]
            left_counts = left_masks_bool.sum(axis=1)
            right_counts = n_samples - left_counts

            valid_counts = (
                (left_counts >= self.min_samples_leaf)
                & (right_counts >= self.min_samples_leaf)
            )

            if not valid_counts.any():
                continue

            left_masks = left_masks_bool.astype(float)

            left_weights = left_masks @ weights
            right_weights = total_weight - left_weights

            valid_weights = (left_weights > 0) & (right_weights > 0)
            valid = valid_counts & valid_weights

            if not valid.any():
                continue

            left_class_weights = left_masks @ class_weight_matrix.T
            right_class_weights = parent_class_weights[None, :] - left_class_weights

            left_probabilities = left_class_weights / (left_weights[:, None] + 1e-12)
            right_probabilities = right_class_weights / (right_weights[:, None] + 1e-12)

            left_gini = 1.0 - np.sum(left_probabilities ** 2, axis=1)
            right_gini = 1.0 - np.sum(right_probabilities ** 2, axis=1)

            child_gini = (
                left_weights * left_gini + right_weights * right_gini
            ) / total_weight

            decreases = parent_gini - child_gini
            decreases[~valid] = -np.inf

            best_index_for_feature = int(np.argmax(decreases))
            feature_best_decrease = float(decreases[best_index_for_feature])

            if feature_best_decrease > best_decrease:
                best_decrease = feature_best_decrease
                best_feature = feature_index
                best_threshold = float(thresholds[best_index_for_feature])

        return best_feature, best_threshold, best_decrease

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _make_leaf(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        gini: float,
    ) -> _Node:
        """Create a leaf node using weighted class probabilities."""
        if self._classes is None:
            raise RuntimeError("Classes are not initialized.")

        class_weight_sums = np.array(
            [weights[y == cls].sum() for cls in self._classes],
            dtype=float,
        )

        total_weight = float(class_weight_sums.sum()) + 1e-12

        proba = {
            str(cls): float(class_weight / total_weight)
            for cls, class_weight in zip(self._classes, class_weight_sums)
        }

        label = str(self._classes[int(np.argmax(class_weight_sums))])

        return _Node(
            is_leaf=True,
            label=label,
            proba=proba,
            n_samples=int(len(y)),
            weighted_n=float(weights.sum()),
            gini=float(gini),
            depth=int(depth),
        )

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
        progress_state: dict | None,
    ) -> _Node:
        """Recursively build the tree."""
        if self._classes is None or self._feature_importances is None:
            raise RuntimeError("Tree state is not initialized.")

        n_samples = len(y)
        class_weight_sums = np.array(
            [weights[y == cls].sum() for cls in self._classes],
            dtype=float,
        )
        total_weight = float(class_weight_sums.sum())
        node_gini = self._gini(class_weight_sums, total_weight)

        should_stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or node_gini <= 1e-12
        )

        if should_stop:
            return self._make_leaf(y, weights, depth, node_gini)

        best_feature, best_threshold, best_decrease = self._best_split(
            X,
            y,
            weights,
        )

        if best_feature == -1 or best_decrease <= 1e-12:
            return self._make_leaf(y, weights, depth, node_gini)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf:
            return self._make_leaf(y, weights, depth, node_gini)

        if right_mask.sum() < self.min_samples_leaf:
            return self._make_leaf(y, weights, depth, node_gini)

        self._feature_importances[best_feature] += total_weight * best_decrease

        if progress_state is not None:
            progress_state["nodes"] += 1
            callback = progress_state.get("callback")
            if callback is not None and progress_state["nodes"] % 20 == 0:
                callback(progress_state["nodes"])

        node = _Node(
            is_leaf=False,
            feature=int(best_feature),
            threshold=float(best_threshold),
            n_samples=int(n_samples),
            weighted_n=float(total_weight),
            gini=float(node_gini),
            impurity_decrease=float(best_decrease),
            depth=int(depth),
        )

        node.left = self._build(
            X[left_mask],
            y[left_mask],
            weights[left_mask],
            depth + 1,
            progress_state,
        )

        node.right = self._build(
            X[right_mask],
            y[right_mask],
            weights[right_mask],
            depth + 1,
            progress_state,
        )

        return node

    # ------------------------------------------------------------------
    # Public fit and prediction API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_callback: Callable[[int], None] | None = None,
    ) -> "DecisionTreeClassifier":
        """Fit the Decision Tree classifier."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        if len(y) == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values.")

        self._classes = np.array(sorted(set(y)))
        self._n_features = X.shape[1]
        self._feature_importances = np.zeros(self._n_features, dtype=float)

        weights = self._sample_weights(y)
        progress_state = {
            "nodes": 0,
            "callback": progress_callback,
        }

        self._root = self._build(
            X,
            y,
            weights,
            depth=0,
            progress_state=progress_state,
        )

        total_importance = float(self._feature_importances.sum())
        if total_importance > 0:
            self._feature_importances = self._feature_importances / total_importance

        return self

    def _traverse(self, x: np.ndarray, node: _Node) -> _Node:
        """Traverse the tree for one row and return a leaf node."""
        if node.is_leaf:
            return node

        if x[node.feature] <= node.threshold:
            if node.left is None:
                raise RuntimeError("Malformed tree: missing left child.")
            return self._traverse(x, node.left)

        if node.right is None:
            raise RuntimeError("Malformed tree: missing right child.")

        return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict grade labels for each row in X."""
        if self._root is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return np.array(
            [self._traverse(row, self._root).label for row in X],
            dtype=object,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each row in X.

        Columns follow self.classes_.
        """
        if self._root is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        classes = self.classes_
        probabilities = np.zeros((len(X), len(classes)), dtype=float)

        for row_index, row in enumerate(X):
            leaf = self._traverse(row, self._root)
            probabilities[row_index] = [
                leaf.proba.get(cls, 0.0)
                for cls in classes
            ]

        return probabilities

    # ------------------------------------------------------------------
    # Public inspection helpers used by Streamlit and Part 6
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> list[str]:
        """Return class labels in probability column order."""
        if self._classes is None:
            return []
        return [str(cls) for cls in self._classes]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return normalized weighted Gini decrease per feature."""
        if self._feature_importances is None:
            return np.array([], dtype=float)
        return self._feature_importances.copy()

    def get_depth(self) -> int:
        """Return the maximum number of split levels in the tree."""
        def _depth(node: _Node | None) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self._root)

    def get_n_leaves(self) -> int:
        """Return the number of leaf nodes."""
        def _leaves(node: _Node | None) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _leaves(node.left) + _leaves(node.right)

        return _leaves(self._root)

    def get_n_nodes(self) -> int:
        """Return the total number of nodes."""
        def _nodes(node: _Node | None) -> int:
            if node is None:
                return 0
            return 1 + _nodes(node.left) + _nodes(node.right)

        return _nodes(self._root)

    def extract_top_rules(
        self,
        feature_names: list[str],
        max_rules: int = 15,
    ) -> list[dict]:
        """Return paths to the largest leaf nodes as readable rules."""
        if self._root is None:
            raise RuntimeError("Call fit() before extract_top_rules().")

        rules: list[dict] = []

        def _walk(node: _Node | None, conditions: list[str]) -> None:
            if node is None:
                return

            if node.is_leaf:
                rules.append(
                    {
                        "conditions": conditions.copy(),
                        "label": node.label,
                        "n_samples": node.n_samples,
                        "proba": node.proba.copy(),
                    }
                )
                return

            if node.feature < len(feature_names):
                feature_name = feature_names[node.feature]
            else:
                feature_name = f"feature_{node.feature}"

            threshold_text = f"{node.threshold:.3g}"

            _walk(
                node.left,
                conditions + [f"{feature_name} <= {threshold_text}"],
            )

            _walk(
                node.right,
                conditions + [f"{feature_name} > {threshold_text}"],
            )

        _walk(self._root, [])

        rules.sort(key=lambda rule: rule["n_samples"], reverse=True)
        return rules[:max_rules]

    def describe_tree(self, feature_names: list[str], max_depth: int = 4) -> str:
        """Return an indented text sketch of the tree."""
        if self._root is None:
            raise RuntimeError("Call fit() before describe_tree().")

        lines: list[str] = []

        def _walk(node: _Node | None, depth: int, prefix: str) -> None:
            indent = "  " * depth

            if node is None:
                lines.append(f"{indent}{prefix}<missing node>")
                return

            if depth > max_depth:
                lines.append(f"{indent}{prefix}...")
                return

            if node.is_leaf:
                proba_text = "  ".join(
                    f"P({cls})={prob:.2f}"
                    for cls, prob in sorted(node.proba.items())
                )

                lines.append(
                    f"{indent}{prefix}[LEAF] -> Grade {node.label} "
                    f"n={node.n_samples}, "
                    f"gini={node.gini:.3f}, "
                    f"{proba_text}"
                )
                return

            if node.feature < len(feature_names):
                feature_name = feature_names[node.feature]
            else:
                feature_name = f"feature_{node.feature}"

            lines.append(
                f"{indent}{prefix}if {feature_name} <= {node.threshold:.3g} "
                f"[n={node.n_samples}, "
                f"weighted_n={node.weighted_n:.1f}, "
                f"gini={node.gini:.3f}, "
                f"gain={node.impurity_decrease:.4f}]"
            )

            _walk(node.left, depth + 1, "YES: ")
            _walk(node.right, depth + 1, "NO:  ")

        _walk(self._root, depth=0, prefix="")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data preparation for Part 3
# ---------------------------------------------------------------------------

def _required_columns_for_features(feature_list: list[str]) -> list[str]:
    """Return raw DataFrame columns required for a feature list."""
    required: set[str] = {TARGET, DATE_COL}

    for feature in feature_list:
        dependencies = _FEATURE_DEPENDENCIES.get(feature, [])
        for column in dependencies:
            required.add(column)

    return sorted(required)


def prepare_dt_data(
    restaurants,
    feature_list: list[str] | None = None,
    test_fraction: float = 0.20,
):
    """Prepare train and test data for the Decision Tree.

    This uses the same temporal split idea as Part 2:
    restaurants are sorted by latest_inspection_date, and the most recent
    fraction becomes the test set.

    Parameters
    ----------
    restaurants : pandas.DataFrame
        Restaurant-level dataset from data/restaurants.csv.
    feature_list : list[str] | None
        Features selected in the Streamlit UI. If None, use all features.
    test_fraction : float
        Fraction of rows reserved for testing.

    Returns
    -------
    X_train, y_train, X_test, y_test,
    train_df, test_df, cutoff_date, feature_names
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if feature_list is None:
        feature_list = DT_ALL_FEATURES

    requested_features = [
        feature for feature in feature_list
        if feature in DT_ALL_FEATURES
    ]

    if not requested_features:
        raise ValueError(
            "feature_list must include at least one valid Decision Tree feature."
        )

    df = restaurants[restaurants[TARGET].isin(VALID_GRADES)].copy()

    required_columns = _required_columns_for_features(requested_features)
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"restaurants is missing required columns: {missing_columns}"
        )

    df = df.dropna(subset=required_columns)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "Not enough graded restaurants with complete features to split."
        )

    cutoff_index = int(len(df) * (1.0 - test_fraction))
    cutoff_index = min(max(cutoff_index, 1), len(df) - 1)

    cutoff_date = df[DATE_COL].iloc[cutoff_index]

    train_df = df.iloc[:cutoff_index].copy()
    test_df = df.iloc[cutoff_index:].copy()

    X_train, feature_names = engineer_features(train_df, requested_features)
    X_test, _ = engineer_features(test_df, requested_features)

    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        train_df,
        test_df,
        cutoff_date,
        feature_names,
    )
