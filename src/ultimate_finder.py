"""
ultimate_finder.py  —  Part 6
------------------------------
Ultimate Restaurant Finder: combines every previous part into one system.

What is new in Part 6
---------------------
1. **Combined grade scoring (CombinedGradePredictor)**
   Instead of mapping actual letters A→3 / B→2 / C→1, each restaurant
   receives a blended KNN+DT *predicted* safety score.  This lets the RL
   reward grid reflect model confidence rather than a binary label.

2. **Restaurant-level NLP embedding**
   The user may input:
     - A food / dish description ("spicy ramen", "wood-fired pizza")
     - A cuisine style                ("Korean BBQ")
     - A restaurant name as a reference ("something like Nobu")
   The text is embedded via TF-IDF (char n-grams) against a *per-restaurant*
   document corpus (name + cuisine + borough).  Cosine similarity gives a
   match score for every restaurant — this is the "matrix embedding" step.
   Food-keyword expansion (from Part 5) ensures dish names map to cuisines.

3. **Dual navigation modes**
   - *Area mode*  — Value Iteration over the reward grid → walk to the
     highest-scoring cluster (same RL algorithm as Part 5).
   - *Direct mode* — Skip RL; rank all restaurants by
       combined_safety_score × nlp_match_score × proximity_weight
     and route the user directly to the highest-ranked restaurant.
     The caller can advance through the ranked list (next / previous).

RL reward grid (both modes)
----------------------------
    weighted_score[i] = safety_score[i] × nlp_score[i]
where
    safety_score  — from CombinedGradePredictor   ∈ [1, 3]
    nlp_score     — per-restaurant TF-IDF cosine   ∈ [0.05, 1.0]

Grid cells accumulate weighted_scores of all restaurants within them.
Value Iteration then propagates this reward through the grid (area mode).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import RL machinery from Part 5
from src.rl_route_finder import (
    _expand_query,
    latlng_to_cell,
    cell_to_latlng,
    path_distance_km,
    apply_proximity_bias,
    value_iteration,
    trace_path,
    streetify_path,
    get_osrm_route,
    GRID_ROWS, GRID_COLS,
    LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX,
    GRID_LAT_RES, GRID_LNG_RES,
    METRES_PER_ROW, METRES_PER_COL,
    WALK_SPEED_KMH,
    WALK_PRESETS,
)


# ── RestaurantEmbedder ────────────────────────────────────────────────────────

class RestaurantEmbedder:
    """TF-IDF embedding over the full restaurant corpus.

    Each restaurant's document = ``"{name} {cuisine} {borough}"``.

    Supports two query modes:
    • Food / dish description  → food-keyword expansion → TF-IDF cosine
    • Restaurant-name reference → the matched restaurant's own vector is
      used as the query, finding restaurants with similar name/cuisine/boro.
    """

    _STOPWORDS = frozenset([
        "the", "and", "a", "an", "of", "in", "at", "on", "for",
        "i", "want", "like", "find", "me", "give", "something",
        # Common food/ingredient words — should NOT trigger name detection
        "noodles", "noodle", "chicken", "beef", "pork", "fish", "rice",
        "pizza", "burger", "burgers", "taco", "tacos", "sushi", "ramen",
        "curry", "salad", "soup", "bread", "cake", "food", "restaurant",
        "cafe", "bar", "grill", "kitchen", "house", "place", "spot",
        "spicy", "crispy", "fresh", "best", "good", "great", "nice",
        "thai", "chinese", "japanese", "korean", "indian", "italian",
        "mexican", "american", "greek", "french", "latin", "bbq",
    ])

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._rest_vecs  = None          # CSR (n_restaurants, n_features)
        self._names:     list[str] = []  # lowercase DBA names
        self._cuisines:  list[str] = []  # raw cuisine values
        self._n:         int       = 0

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "RestaurantEmbedder":
        """Build TF-IDF matrix from restaurant names, cuisines and boroughs."""
        self._names    = df["dba"].fillna("").str.strip().str.lower().tolist()
        self._cuisines = df["cuisine"].fillna("").tolist()
        self._n        = len(df)

        docs = (
            df["dba"].fillna("").str.strip()
            + " " + df["cuisine"].fillna("").str.strip()
            + " " + df["boro"].fillna("").str.strip()
        ).str.lower().tolist()

        self._vectorizer = TfidfVectorizer(
            analyzer   = "char_wb",
            ngram_range= (2, 4),
            max_features= 100_000,
            sublinear_tf= True,
        )
        self._rest_vecs = self._vectorizer.fit_transform(docs)
        return self

    # ------------------------------------------------------------------
    def _detect_restaurant_name(self, query: str) -> Optional[int]:
        """Return the index (into fit df) of the best-matching restaurant name,
        or None if no plausible match is found."""
        q = query.lower().strip()

        # Remove common stopword prefixes
        tokens = [t for t in q.split() if t not in self._STOPWORDS and len(t) >= 3]
        if not tokens:
            return None

        best_idx   = None
        best_score = 0

        # Check tokens against names (prefer longer name matches)
        for token in tokens:
            for i, name in enumerate(self._names):
                if not name or len(name) < 3:
                    continue
                if token in name or name in token:
                    score = min(len(token), len(name))
                    if score > best_score:
                        best_score = score
                        best_idx   = i

        # Require a match of at least 4 characters
        return best_idx if best_score >= 4 else None

    # ------------------------------------------------------------------
    def score_per_restaurant(self, query: str) -> np.ndarray:
        """Return per-restaurant NLP match scores array, shape (n_restaurants,).

        Scores are normalised to [0.05, 1.0].
        If query is empty, returns all-ones (uniform weight).
        """
        if self._vectorizer is None:
            raise RuntimeError("Call fit() first.")

        if not query or not query.strip():
            return np.ones(self._n, dtype=float)

        # Check for restaurant-name reference
        ref_idx = self._detect_restaurant_name(query)
        if ref_idx is not None:
            q_vec = self._rest_vecs[ref_idx]  # sparse row vector
        else:
            expanded = _expand_query(query)
            q_vec    = self._vectorizer.transform([expanded])

        sims    = cosine_similarity(q_vec, self._rest_vecs)[0]
        max_sim = sims.max()
        if max_sim < 1e-8:
            return np.full(self._n, 0.05, dtype=float)

        return 0.05 + 0.95 * (sims / max_sim)

    # ------------------------------------------------------------------
    def cuisine_scores_from_query(
        self, query: str, cuisines: list[str]
    ) -> dict[str, float]:
        """Per-cuisine similarity scores (used for RL hard-filter, same role as Part 5).

        When the query names a specific restaurant, its cuisine gets score 1.0
        and related cuisines get partial scores.
        """
        if not query or not query.strip():
            return {c: 1.0 for c in cuisines}

        ref_idx = self._detect_restaurant_name(query)
        if ref_idx is not None:
            ref_cuisine = self._cuisines[ref_idx]
            scores: dict[str, float] = {}
            for c in cuisines:
                if c == ref_cuisine:
                    scores[c] = 1.0
                elif (ref_cuisine.lower() in c.lower()
                      or c.lower() in ref_cuisine.lower()):
                    scores[c] = 0.6
                else:
                    scores[c] = 0.08
            return scores

        # Food-keyword expansion + TF-IDF vs cuisine labels
        expanded     = _expand_query(query)
        cuisine_docs = [c.lower() for c in cuisines]
        cuisine_vecs = self._vectorizer.transform(cuisine_docs)
        q_vec        = self._vectorizer.transform([expanded])
        sims         = cosine_similarity(q_vec, cuisine_vecs)[0]
        max_sim      = sims.max()
        if max_sim < 1e-8:
            return {c: 0.05 for c in cuisines}
        normalised = 0.05 + 0.95 * (sims / max_sim)
        return {c: float(s) for c, s in zip(cuisines, normalised)}

    def top_matches(self, query: str, n: int = 5) -> list[tuple[str, float]]:
        """Top-n (cuisine, score) pairs for display, sorted descending."""
        if not query or not query.strip():
            return []
        scores = self.cuisine_scores_from_query(
            query,
            sorted(set(c for c in self._cuisines if c)),
        )
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def detected_reference(self, query: str) -> Optional[str]:
        """Return the detected restaurant name (DBA) if the query refers to one."""
        idx = self._detect_restaurant_name(query)
        return self._names[idx].title() if idx is not None else None


# ── Reward grid (Part-6 version) ──────────────────────────────────────────────

def build_reward_grid_v2(
    df:            pd.DataFrame,
    safety_scores: np.ndarray,         # pre-computed combined safety (1-3)
    nlp_scores:    np.ndarray,         # per-restaurant NLP match (0.05-1)
    danger_filter: bool,
    smooth_sigma:  float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build per-cell RL reward grid using model-predicted scores.

    Parameters
    ----------
    safety_scores : ndarray, shape (len(df),)
        Combined KNN+DT safety predictions (replaces actual grade lookup).
    nlp_scores : ndarray, shape (len(df),)
        Per-restaurant TF-IDF cosine similarity to user query.
    danger_filter : bool
        If True, restaurants whose *actual* grade is C get safety zeroed.

    Returns
    -------
    R        : normalised reward grid (GRID_ROWS, GRID_COLS)
    safe_df  : restaurants that contributed to the reward
    terr_df  : Grade-C restaurants (for warning display)
    """
    needed = ["latitude", "longitude", "latest_grade", "cuisine", "dba",
              "boro", "building", "street", "zipcode", "latest_score"]
    avail  = [c for c in needed if c in df.columns]
    valid  = df[avail].dropna(subset=["latitude", "longitude"]).copy()
    valid  = valid[
        valid["latitude"].between(LAT_MIN, LAT_MAX)
        & valid["longitude"].between(LNG_MIN, LNG_MAX)
    ].copy()

    valid["grid_row"] = (
        ((valid["latitude"]  - LAT_MIN) / GRID_LAT_RES).astype(int).clip(0, GRID_ROWS - 1)
    )
    valid["grid_col"] = (
        ((valid["longitude"] - LNG_MIN) / GRID_LNG_RES).astype(int).clip(0, GRID_COLS - 1)
    )

    # Attach pre-computed scores (align by position in df)
    valid_pos              = valid.index       # positional index into *df*
    valid["safety"]        = safety_scores[df.index.get_indexer(valid_pos)]
    valid["nlp_score"]     = nlp_scores[df.index.get_indexer(valid_pos)]

    # Grade-C actual flag (for warning layer — even if model predicts higher)
    terr_df = valid[valid["latest_grade"] == "C"].copy() if "latest_grade" in valid else pd.DataFrame()

    # Danger filter: zero out restaurants whose actual grade is C
    if danger_filter and "latest_grade" in valid.columns:
        valid.loc[valid["latest_grade"] == "C", "safety"] = 0.0

    valid["weighted_score"] = valid["safety"] * valid["nlp_score"]
    safe_df = valid[valid["weighted_score"] > 0].copy()

    # Accumulate into grid
    R = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    if not safe_df.empty:
        np.add.at(R,
                  (safe_df["grid_row"].values, safe_df["grid_col"].values),
                  safe_df["weighted_score"].values)

    if smooth_sigma > 0:
        R = gaussian_filter(R, sigma=smooth_sigma)

    r_max = R.max()
    if r_max > 1e-8:
        R /= r_max

    return R, safe_df, terr_df


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AreaRouteResult:
    """Outcome of the Area-mode RL route planner."""
    street_route: Optional[list[tuple[float, float]]]
    grid_route:   list[tuple[float, float]]
    cell_path:    list[tuple[int, int]]

    destination_lat: float
    destination_lng: float

    destination_restaurants: pd.DataFrame
    terrible_restaurants:    pd.DataFrame

    top_matches:     list[tuple[str, float]]
    query_applied:   bool
    detected_ref:    Optional[str]   # restaurant name detected in query

    path_distance_km:          float
    walk_minutes:              float
    n_destination_restaurants: int
    mean_destination_safety:   float

    value_map:  np.ndarray
    reward_map: np.ndarray
    gamma:      float
    sigma_km:   float
    walk_preset: str


@dataclass
class DirectRouteResult:
    """Outcome of the Direct-mode restaurant ranker."""
    ranked_restaurants: pd.DataFrame   # all candidates, sorted by combined score
    current_idx:        int            # which row is currently selected

    street_route: Optional[list[tuple[float, float]]]
    grid_route:   list[tuple[float, float]]

    path_distance_km: float
    walk_minutes:     float

    top_matches:  list[tuple[str, float]]
    detected_ref: Optional[str]

    @property
    def current_restaurant(self) -> Optional[pd.Series]:
        if self.ranked_restaurants.empty:
            return None
        idx = min(self.current_idx, len(self.ranked_restaurants) - 1)
        return self.ranked_restaurants.iloc[idx]

    @property
    def n_candidates(self) -> int:
        return len(self.ranked_restaurants)


# ── Area mode (RL) ────────────────────────────────────────────────────────────

def find_area_route(
    df:             pd.DataFrame,
    safety_scores:  np.ndarray,        # pre-computed for all rows of df
    user_lat:       float,
    user_lng:       float,
    query:          str          = "",
    walk_preset_key: str         = "15 min (~1.1 km)",
    cuisine_importance: float    = 0.7,
    danger_filter:   bool        = True,
    smooth_sigma:    float       = 1.0,
    use_osrm:        bool        = True,
    destination_radius_cells: int = 2,
) -> AreaRouteResult:
    """Plan a walking route to the highest-scoring restaurant cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Full restaurants DataFrame (must be the same df used to build safety_scores).
    safety_scores : ndarray, shape (len(df),)
        Pre-computed combined model safety scores.
    query : str
        Free-text food / dish / restaurant description.
    """
    preset   = WALK_PRESETS.get(walk_preset_key, WALK_PRESETS["15 min (~1.1 km)"])
    gamma    = preset["gamma"]
    max_steps= preset["max_steps"]
    sigma_km = preset["sigma_km"]

    # ── NLP embedding ───────────────────────────────────────────────────
    embedder = RestaurantEmbedder().fit(df)
    has_query = bool(query and query.strip())

    cuisines = sorted(df["cuisine"].dropna().unique().tolist())
    top_matches  = embedder.top_matches(query, n=5) if has_query else []
    detected_ref = embedder.detected_reference(query) if has_query else None

    if has_query:
        nlp_scores = embedder.score_per_restaurant(query)
        cuisine_scores = embedder.cuisine_scores_from_query(query, cuisines)
    else:
        nlp_scores     = np.ones(len(df), dtype=float)
        cuisine_scores = {c: 1.0 for c in cuisines}

    # Apply cuisine importance to NLP scores (hard-filter low-match restaurants)
    if has_query and cuisine_importance > 0:
        max_cs  = max(cuisine_scores.values()) if cuisine_scores else 1.0
        thresh  = cuisine_importance * max_cs
        for i, cuisine in enumerate(df["cuisine"].fillna("").values):
            if cuisine_scores.get(cuisine, 0.05) < thresh:
                nlp_scores[i] = 0.0

    # ── Reward grid ─────────────────────────────────────────────────────
    R_raw, safe_df, terr_df = build_reward_grid_v2(
        df, safety_scores, nlp_scores,
        danger_filter=danger_filter,
        smooth_sigma=smooth_sigma,
    )

    # ── Proximity Gaussian bias ─────────────────────────────────────────
    start_row, start_col = latlng_to_cell(user_lat, user_lng)
    R_biased = apply_proximity_bias(R_raw, start_row, start_col, sigma_km)
    r_max = R_biased.max()
    if r_max > 1e-8:
        R_biased /= r_max

    # ── Value Iteration ─────────────────────────────────────────────────
    V = value_iteration(R_biased, gamma=gamma)

    # ── Greedy trace ────────────────────────────────────────────────────
    cell_path = trace_path(V, start_row, start_col, max_steps=max_steps)

    latlng_raw  = [cell_to_latlng(r, c) for r, c in cell_path]
    grid_route  = streetify_path(latlng_raw)
    dest_row, dest_col = cell_path[-1]
    dest_lat, dest_lng = cell_to_latlng(dest_row, dest_col)

    # ── OSRM real routing ───────────────────────────────────────────────
    street_route: Optional[list[tuple[float, float]]] = None
    if use_osrm:
        street_route = get_osrm_route(user_lat, user_lng, dest_lat, dest_lng, mode="walking")

    # ── Destination restaurants ─────────────────────────────────────────
    if "grid_row" in safe_df.columns and "grid_col" in safe_df.columns:
        mask = (
            (safe_df["grid_row"] - dest_row).abs() <= destination_radius_cells
        ) & (
            (safe_df["grid_col"] - dest_col).abs() <= destination_radius_cells
        )
        dest_restaurants = safe_df[mask].copy()
    else:
        dest_restaurants = safe_df.iloc[:0].copy()

    # ── Terrible restaurants near path ──────────────────────────────────
    terrible_near = pd.DataFrame()
    if danger_filter and not terr_df.empty and "grid_row" in terr_df.columns:
        TERR_R     = 4
        path_rows  = np.array([r for r, _ in cell_path])
        path_cols  = np.array([c for _, c in cell_path])
        tr         = terr_df["grid_row"].values
        tc         = terr_df["grid_col"].values
        near       = np.zeros(len(terr_df), dtype=bool)
        for pr, pc in zip(path_rows, path_cols):
            near |= (np.abs(tr - pr) <= TERR_R) & (np.abs(tc - pc) <= TERR_R)
        terrible_near = terr_df[near].copy()

    dist_km  = path_distance_km(cell_path)
    walk_min = (dist_km / WALK_SPEED_KMH) * 60.0
    mean_sf  = float(dest_restaurants["safety"].mean()) if not dest_restaurants.empty else 0.0

    return AreaRouteResult(
        street_route             = street_route,
        grid_route               = grid_route,
        cell_path                = cell_path,
        destination_lat          = dest_lat,
        destination_lng          = dest_lng,
        destination_restaurants  = dest_restaurants,
        terrible_restaurants     = terrible_near,
        top_matches              = top_matches,
        query_applied            = has_query,
        detected_ref             = detected_ref,
        path_distance_km         = dist_km,
        walk_minutes             = walk_min,
        n_destination_restaurants= len(dest_restaurants),
        mean_destination_safety  = mean_sf,
        value_map                = V,
        reward_map               = R_biased,
        gamma                    = gamma,
        sigma_km                 = sigma_km,
        walk_preset              = walk_preset_key,
    )


# ── Direct mode ───────────────────────────────────────────────────────────────

def rank_restaurants_direct(
    df:            pd.DataFrame,
    safety_scores: np.ndarray,
    user_lat:      float,
    user_lng:      float,
    query:         str   = "",
    walk_preset_key: str = "15 min (~1.1 km)",
    danger_filter: bool  = True,
    top_n:         int   = 30,
) -> tuple[pd.DataFrame, list[tuple[str, float]], Optional[str]]:
    """Rank restaurants for Direct mode.

    Ranking score = safety_score × nlp_score × proximity_weight
    where proximity_weight = Gaussian centred at user's location.

    Parameters
    ----------
    walk_preset_key : str
        Determines the proximity Gaussian sigma.

    Returns
    -------
    ranked_df  : top_n restaurants sorted by combined score (descending)
    top_matches: top cuisine matches for display
    detected_ref: restaurant name detected in query (or None)
    """
    preset   = WALK_PRESETS.get(walk_preset_key, WALK_PRESETS["15 min (~1.1 km)"])
    walk_km  = preset["walk_km"]
    sigma_km = preset["sigma_km"]

    # ── NLP scores ──────────────────────────────────────────────────────
    embedder = RestaurantEmbedder().fit(df)
    has_query    = bool(query and query.strip())
    top_matches  = embedder.top_matches(query, n=5) if has_query else []
    detected_ref = embedder.detected_reference(query) if has_query else None
    nlp_scores   = embedder.score_per_restaurant(query) if has_query else np.ones(len(df))

    # ── Build scoring frame ──────────────────────────────────────────────
    avail  = [c for c in ["latitude", "longitude", "latest_grade", "dba",
                           "cuisine", "boro", "building", "street",
                           "zipcode", "latest_score"] if c in df.columns]
    valid  = df[avail].dropna(subset=["latitude", "longitude"]).copy()
    valid  = valid[
        valid["latitude"].between(LAT_MIN, LAT_MAX)
        & valid["longitude"].between(LNG_MIN, LNG_MAX)
    ].copy()

    if valid.empty:
        return pd.DataFrame(), top_matches, detected_ref

    # Attach scores (align by df index)
    pos_in_df         = df.index.get_indexer(valid.index)
    valid["safety"]   = safety_scores[pos_in_df]
    valid["nlp_score"]= nlp_scores[pos_in_df]

    # Danger filter
    if danger_filter and "latest_grade" in valid.columns:
        valid.loc[valid["latest_grade"] == "C", "safety"] = 0.0

    # Proximity Gaussian (km) — prefer closer restaurants
    lat_arr   = valid["latitude"].values
    lng_arr   = valid["longitude"].values
    dist_m    = np.sqrt(
        ((lat_arr - user_lat) * 111_000) ** 2
        + ((lng_arr - user_lng) * 85_000) ** 2
    )
    dist_km   = dist_m / 1000.0
    # Hard limit at 1.5× walking budget
    too_far   = dist_km > walk_km * 1.5
    sigma_c   = max(sigma_km, 0.1)
    prox      = np.exp(-(dist_km ** 2) / (2.0 * sigma_c ** 2))
    prox[too_far] = 0.0

    valid["proximity"]     = prox
    valid["combined_score"]= valid["safety"] * valid["nlp_score"] * prox

    # Filter to non-zero and sort
    ranked = valid[valid["combined_score"] > 0].copy()
    ranked = ranked.sort_values("combined_score", ascending=False).head(top_n)
    ranked = ranked.reset_index(drop=True)

    return ranked, top_matches, detected_ref


def get_direct_route_to_restaurant(
    user_lat:  float,
    user_lng:  float,
    dest_lat:  float,
    dest_lng:  float,
    use_osrm:  bool = True,
) -> tuple[Optional[list[tuple[float, float]]], list[tuple[float, float]], float, float]:
    """Route from user location to a specific restaurant.

    Returns
    -------
    street_route : real OSRM route or None
    grid_route   : rectilinear fallback
    dist_km      : estimated distance
    walk_min     : estimated walking time
    """
    street_route: Optional[list[tuple[float, float]]] = None
    if use_osrm:
        street_route = get_osrm_route(user_lat, user_lng, dest_lat, dest_lng, mode="walking")

    # Rectilinear grid fallback
    latlng_pair = [(user_lat, user_lng), (dest_lat, dest_lng)]
    grid_route  = streetify_path(latlng_pair)

    dist_m   = math.sqrt(
        ((dest_lat - user_lat) * 111_000) ** 2
        + ((dest_lng - user_lng) * 85_000) ** 2
    )
    dist_km  = dist_m / 1000.0
    walk_min = (dist_km / WALK_SPEED_KMH) * 60.0

    return street_route, grid_route, dist_km, walk_min
