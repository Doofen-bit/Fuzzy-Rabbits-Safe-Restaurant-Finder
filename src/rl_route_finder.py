"""
rl_route_finder.py
------------------
Reinforcement Learning walk-route planner: finds the optimal short walking path
from a user's NYC location to the nearest cluster of safe, cuisine-matching
restaurants.

RL Formulation
==============
• Environment  : NYC map discretised into ~400 m × 420 m grid cells
• State s      : (row, col) — the agent's current grid cell
• Actions      : move in one of 8 compass directions (N/NE/E/SE/S/SW/W/NW)
• Reward R(s)  : cuisine-filtered, proximity-biased aggregate safety score of
                 all eligible restaurants within cell s
• Discount γ   : derived from walking-time budget — keeps search within a
                 realistic walking radius
• Algorithm    : Value Iteration (model-based RL; fully vectorised NumPy)

Bellman Optimality Equation
----------------------------
    V⁰(s)    = R_biased(s)
    Vⁿ⁺¹(s) = R_biased(s) + γ · max_{a} Vⁿ(T(s, a))

Key Improvements over Previous Version
----------------------------------------
1. **Cuisine hard-filter** — when a meal description is given, only restaurants
   whose cuisine type's TF-IDF similarity to the description exceeds a
   user-controlled threshold contribute to the reward grid.  Non-matching
   restaurants are zeroed out entirely so the agent routes to the nearest
   *matching* cluster, not just the densest cluster.

2. **Proximity Gaussian bias** — the reward grid is multiplied by a Gaussian
   centred at the user's starting cell before VI runs.  Nearby good clusters
   therefore beat far-away larger clusters, keeping routes within a realistic
   walking radius.

3. **Reasonable walking range** — walking time (5–20 min) is the primary
   hyperparameter; gamma and max_steps are derived from it automatically so
   the agent never plans a 15 km cross-borough trek.

4. **Real-street routing** — after VI finds the destination, the actual walking
   path is fetched from the OSRM public pedestrian routing API
   (router.project-osrm.org).  Falls back to a rectilinear grid approximation
   if the network is unavailable.

Text matching (sklearn — permitted by spec)
--------------------------------------------
sklearn TF-IDF with character n-grams vectorises cuisine labels and the user's
meal description.  Cosine similarity produces per-cuisine match scores that
serve as both a hard filter threshold and a soft reward multiplier.

No external ML libraries are used for the RL algorithm itself — only NumPy and
scipy.ndimage are used inside the RL pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Food-keyword → cuisine-label expansion
# Maps common dish/ingredient words to their cuisine category so TF-IDF
# can match them against the dataset's generic cuisine labels.
# ---------------------------------------------------------------------------
_FOOD_EXPANSIONS: list[tuple[str, str]] = [
    # Japanese
    (r"\bramen\b",          "japanese noodles asian"),
    (r"\bsushi\b",          "japanese"),
    (r"\bsashimi\b",        "japanese seafood"),
    (r"\budon\b",           "japanese noodles"),
    (r"\btonkatsu\b",       "japanese"),
    (r"\bbento\b",          "japanese"),
    (r"\bteriyaki\b",       "japanese"),
    (r"\bgyoza\b",          "japanese chinese"),
    # Chinese
    (r"\bdim\s*sum\b",      "chinese"),
    (r"\bdumplings?\b",     "chinese asian"),
    (r"\bfried\s*rice\b",   "chinese asian"),
    (r"\bwonton\b",         "chinese"),
    (r"\bchow\s*mein\b",    "chinese"),
    (r"\bpeking\b",         "chinese"),
    # Korean
    (r"\bbbq\b",            "barbecue korean"),
    (r"\bkimchi\b",         "korean"),
    (r"\bbibimbap\b",       "korean"),
    # Thai
    (r"\bpad\s*thai\b",     "thai southeast asian"),
    (r"\bthai\s*curry\b",   "thai"),
    # Vietnamese
    (r"\bpho\b",            "southeast asian vietnamese"),
    (r"\bbanh\s*mi\b",      "southeast asian vietnamese"),
    # Indian
    (r"\bcurry\b",          "indian thai"),
    (r"\bnaan\b",           "indian"),
    (r"\bbiryani\b",        "indian"),
    (r"\btandoori\b",       "indian"),
    (r"\bsamosa\b",         "indian"),
    # Mexican
    (r"\btacos?\b",         "mexican tex-mex"),
    (r"\bburritos?\b",      "mexican"),
    (r"\bguacamole\b",      "mexican"),
    (r"\bquesadillas?\b",   "mexican"),
    (r"\bnachos?\b",        "tex-mex mexican"),
    # Italian
    (r"\bpizzas?\b",        "pizza italian"),
    (r"\bpastas?\b",        "italian"),
    (r"\brisotto\b",        "italian"),
    (r"\bgnocchi\b",        "italian"),
    (r"\blasagna\b",        "italian"),
    # American
    (r"\bburgers?\b",       "hamburgers american"),
    (r"\bsteaks?\b",        "steakhouse american"),
    (r"\bwings?\b",         "chicken american"),
    (r"\bsandwiches?\b",    "sandwiches american"),
    # Middle Eastern
    (r"\bshawarma\b",       "middle eastern mediterranean"),
    (r"\bfalafel\b",        "middle eastern"),
    (r"\bhummus\b",         "middle eastern mediterranean"),
    (r"\bgyros?\b",         "greek mediterranean"),
    (r"\bkebabs?\b",        "middle eastern turkish mediterranean"),
    # Other
    (r"\bnoodles?\b",       "asian chinese japanese"),
    (r"\bseafood\b",        "seafood"),
    (r"\bsalads?\b",        "salads"),
    (r"\bsushi\s*roll\b",   "japanese"),
    (r"\bcrepes?\b",        "french"),
    (r"\bcroissants?\b",    "french bakery"),
    (r"\bbagels?\b",        "bagels jewish"),
    (r"\bjerk\b",           "caribbean"),
    (r"\bpoke\b",           "hawaiian japanese"),
    (r"\btapas?\b",         "tapas spanish"),
    (r"\bpaella\b",         "spanish"),
]


def _expand_query(description: str) -> str:
    """Expand food keywords in the query to cuisine-compatible vocabulary.

    E.g. "spicy ramen" → "spicy ramen japanese noodles asian"
    This bridges the gap between dish names and the generic cuisine labels
    in the dataset, improving TF-IDF cosine-similarity matching.
    """
    desc = description.lower().strip()
    extras: list[str] = []
    for pattern, expansion in _FOOD_EXPANSIONS:
        if re.search(pattern, desc):
            extras.append(expansion)
    if extras:
        desc = desc + " " + " ".join(extras)
    return desc

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 40.477, 40.920
LNG_MIN, LNG_MAX = -74.260, -73.699

GRID_LAT_RES = 0.004   # ≈ 444 m per row
GRID_LNG_RES = 0.005   # ≈ 420 m per col

GRID_ROWS = int(math.ceil((LAT_MAX - LAT_MIN) / GRID_LAT_RES))   # 111
GRID_COLS = int(math.ceil((LNG_MAX - LNG_MIN) / GRID_LNG_RES))   # 113

METRES_PER_ROW = 444.0
METRES_PER_COL = 420.0
AVG_CELL_KM    = (METRES_PER_ROW + METRES_PER_COL) / 2 / 1000    # ≈ 0.432 km

WALK_SPEED_KMH = 4.5   # pedestrian walking speed

# Safety score by grade
SAFETY_SCORES: dict[str, float] = {"A": 3.0, "B": 2.0, "C": 1.0}

# ---------------------------------------------------------------------------
# Walking-time → route hyperparameters
# (walking minutes → max walk km, gamma, max_steps, proximity sigma)
# Capped at 20 min for realistic NYC walking.
# ---------------------------------------------------------------------------
WALK_PRESETS: dict[str, dict] = {
    "5 min  (~375 m)":  {"walk_km": 0.375, "gamma": 0.50, "max_steps": 2,  "sigma_km": 0.25},
    "8 min  (~600 m)":  {"walk_km": 0.600, "gamma": 0.65, "max_steps": 3,  "sigma_km": 0.35},
    "10 min (~750 m)":  {"walk_km": 0.750, "gamma": 0.70, "max_steps": 4,  "sigma_km": 0.45},
    "12 min (~900 m)":  {"walk_km": 0.900, "gamma": 0.75, "max_steps": 5,  "sigma_km": 0.55},
    "15 min (~1.1 km)": {"walk_km": 1.125, "gamma": 0.80, "max_steps": 6,  "sigma_km": 0.70},
    "20 min (~1.5 km)": {"walk_km": 1.500, "gamma": 0.85, "max_steps": 8,  "sigma_km": 1.00},
}

# ---------------------------------------------------------------------------
# NYC neighbourhood look-up table
# ---------------------------------------------------------------------------
NYC_NEIGHBORHOODS: dict[str, tuple[float, float]] = {
    "Midtown Manhattan (Times Sq.)":   (40.7580, -73.9855),
    "Hell's Kitchen / 9th Ave":        (40.7638, -73.9918),
    "Upper East Side":                  (40.7736, -73.9566),
    "Upper West Side":                  (40.7870, -73.9754),
    "Harlem":                           (40.8116, -73.9465),
    "East Village":                     (40.7264, -73.9815),
    "Lower East Side":                  (40.7157, -73.9863),
    "Greenwich Village / NYU":          (40.7335, -74.0027),
    "SoHo / Nolita":                   (40.7233, -74.0030),
    "Tribeca / Financial District":     (40.7163, -74.0086),
    "Chelsea / Flatiron":               (40.7465, -74.0014),
    "Flushing (Queens)":                (40.7675, -73.8330),
    "Astoria (Queens)":                 (40.7721, -73.9302),
    "Jackson Heights (Queens)":         (40.7557, -73.8831),
    "Williamsburg (Brooklyn)":          (40.7081, -73.9571),
    "Park Slope (Brooklyn)":            (40.6710, -73.9777),
    "Brooklyn Heights / DUMBO":         (40.6981, -73.9949),
    "Bensonhurst (Brooklyn)":           (40.6036, -73.9951),
    "Grand Concourse (Bronx)":          (40.8448, -73.9285),
    "Fordham / Arthur Ave (Bronx)":     (40.8599, -73.8924),
    "St. George (Staten Island)":       (40.6437, -74.0739),
    "Click on the map to set location": (40.7580, -73.9855),
}


# ---------------------------------------------------------------------------
# Coordinate ↔ grid cell helpers
# ---------------------------------------------------------------------------

def latlng_to_cell(lat: float, lng: float) -> tuple[int, int]:
    row = int((lat - LAT_MIN) / GRID_LAT_RES)
    col = int((lng - LNG_MIN) / GRID_LNG_RES)
    return max(0, min(GRID_ROWS - 1, row)), max(0, min(GRID_COLS - 1, col))


def cell_to_latlng(row: int, col: int) -> tuple[float, float]:
    return (LAT_MIN + (row + 0.5) * GRID_LAT_RES,
            LNG_MIN + (col + 0.5) * GRID_LNG_RES)


def path_distance_km(cell_path: list[tuple[int, int]]) -> float:
    dist = 0.0
    for i in range(len(cell_path) - 1):
        dr = abs(cell_path[i+1][0] - cell_path[i][0])
        dc = abs(cell_path[i+1][1] - cell_path[i][1])
        dist += math.sqrt((dr * METRES_PER_ROW)**2 + (dc * METRES_PER_COL)**2)
    return dist / 1000.0


# ---------------------------------------------------------------------------
# Meal matcher — sklearn TF-IDF cosine similarity (allowed by spec)
# ---------------------------------------------------------------------------

class MealMatcher:
    """Convert a free-text meal description → per-cuisine similarity scores.

    Scores are normalised so the best-matching cuisine always gets 1.0.
    Non-matching cuisines receive a minimum of 0.05.
    """

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._cuisine_vecs = None
        self._cuisines: list[str] = []

    def fit(self, cuisines: list[str]) -> "MealMatcher":
        self._cuisines = list(cuisines)
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
        )
        self._cuisine_vecs = self._vectorizer.fit_transform(
            [c.lower() for c in self._cuisines]
        )
        return self

    def score(self, description: str) -> dict[str, float]:
        """Return {cuisine: score ∈ [0.05, 1.0]} for each cuisine.

        The description is first expanded via _expand_query so that dish
        keywords like 'ramen' map onto cuisine labels like 'Japanese'.
        """
        if not description or not description.strip():
            return {c: 1.0 for c in self._cuisines}
        expanded = _expand_query(description)
        q_vec = self._vectorizer.transform([expanded])
        sims = cosine_similarity(q_vec, self._cuisine_vecs)[0]
        max_sim = sims.max()
        if max_sim < 1e-8:
            return {c: 0.05 for c in self._cuisines}
        sims_norm = 0.05 + 0.95 * (sims / max_sim)
        return {c: float(s) for c, s in zip(self._cuisines, sims_norm)}

    def top_matches(self, description: str, n: int = 5) -> list[tuple[str, float]]:
        """Return top-n (cuisine, score) pairs sorted descending."""
        if not description or not description.strip():
            return []
        scores = self.score(description)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Reward grid construction
# ---------------------------------------------------------------------------

def build_reward_grid(
    df: pd.DataFrame,
    cuisine_scores: dict[str, float],
    cuisine_importance: float,         # 0.0 → ignore cuisine; 1.0 → strict filter
    danger_filter: bool,
    smooth_sigma: float = 1.0,
    has_description: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build per-cell reward grid, optionally filtering by cuisine.

    Parameters
    ----------
    cuisine_importance : float in [0, 1]
        0 — no cuisine filter, all restaurants counted equally.
        >0 — restaurants whose cuisine_match < (importance × max_match) are
             zeroed out.  At 1.0 only the single best-matching cuisine counts.
    has_description : bool
        True when user provided a meal description.  When False, cuisine
        importance is ignored regardless of its value (no description → no filter).

    Returns
    -------
    R        : normalised reward grid (GRID_ROWS, GRID_COLS)
    safe_df  : restaurants that contributed to the reward
    terr_df  : Grade-C restaurants that were filtered by danger_filter
    """
    needed = ["latitude", "longitude", "latest_grade", "cuisine", "dba",
              "boro", "building", "street", "zipcode", "latest_score"]
    available = [c for c in needed if c in df.columns]
    valid = df[available].dropna(subset=["latitude", "longitude"]).copy()

    valid = valid[
        valid["latitude"].between(LAT_MIN, LAT_MAX) &
        valid["longitude"].between(LNG_MIN, LNG_MAX)
    ].copy()

    valid["grid_row"] = (
        ((valid["latitude"]  - LAT_MIN) / GRID_LAT_RES).astype(int).clip(0, GRID_ROWS - 1)
    )
    valid["grid_col"] = (
        ((valid["longitude"] - LNG_MIN) / GRID_LNG_RES).astype(int).clip(0, GRID_COLS - 1)
    )

    # Safety scores
    valid["safety"] = valid["latest_grade"].map(SAFETY_SCORES).fillna(0.0)

    # Separate Grade-C restaurants for later display
    terr_df = valid[valid["latest_grade"] == "C"].copy()

    # Apply danger filter (zero out Grade-C contribution)
    if danger_filter:
        valid.loc[valid["latest_grade"] == "C", "safety"] = 0.0

    # Cuisine match score
    valid["cuisine_match"] = valid["cuisine"].map(cuisine_scores).fillna(0.05)

    # Cuisine hard-filter — only applied when user gave a description
    if has_description and cuisine_importance > 0:
        max_match = max(cuisine_scores.values()) if cuisine_scores else 1.0
        threshold = cuisine_importance * max_match
        # Zero out non-matching restaurants
        valid.loc[valid["cuisine_match"] < threshold, "safety"] = 0.0

    # Weighted score: safety × cuisine_match (match score still acts as multiplier)
    valid["weighted_score"] = valid["safety"] * valid["cuisine_match"]

    safe_df = valid[valid["weighted_score"] > 0].copy()

    # Accumulate into grid (O(n) with np.add.at)
    R = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    if not safe_df.empty:
        np.add.at(R,
                  (safe_df["grid_row"].values, safe_df["grid_col"].values),
                  safe_df["weighted_score"].values)

    # Gaussian blur to spread density to adjacent cells
    if smooth_sigma > 0:
        R = gaussian_filter(R, sigma=smooth_sigma)

    # Normalise to [0, 1]
    r_max = R.max()
    if r_max > 1e-8:
        R /= r_max

    return R, safe_df, terr_df


# ---------------------------------------------------------------------------
# Proximity Gaussian bias
# ---------------------------------------------------------------------------

def apply_proximity_bias(
    R: np.ndarray,
    start_row: int,
    start_col: int,
    sigma_km: float,
) -> np.ndarray:
    """Multiply R by a Gaussian centred at the start cell.

    Cells farther than ~2σ from the start are suppressed, ensuring the RL
    agent routes to nearby clusters rather than the city's global maximum.

    sigma_km should be ~½ of the user's max walking radius.
    """
    rows = np.arange(GRID_ROWS)
    cols = np.arange(GRID_COLS)
    col_grid, row_grid = np.meshgrid(cols, rows)   # both (GRID_ROWS, GRID_COLS)

    dist_m = np.sqrt(
        ((row_grid - start_row) * METRES_PER_ROW) ** 2
        + ((col_grid - start_col) * METRES_PER_COL) ** 2
    )
    dist_km = dist_m / 1000.0

    # Gaussian weight: w = exp(- dist² / (2σ²))
    sigma_clamped = max(sigma_km, 0.1)
    weight = np.exp(-(dist_km ** 2) / (2.0 * sigma_clamped ** 2))

    return R * weight


# ---------------------------------------------------------------------------
# Value Iteration — pure NumPy RL
# ---------------------------------------------------------------------------

def value_iteration(
    R: np.ndarray,
    gamma: float,
    max_iter: int = 400,
    tol: float = 1e-5,
) -> np.ndarray:
    """Bellman optimality update (vectorised, zero-padding at boundaries).

        Vⁿ⁺¹(s) = R(s) + γ · max_{a ∈ {N,NE,E,SE,S,SW,W,NW}} Vⁿ(T(s,a))
    """
    V = R.copy()

    for _ in range(max_iter):
        P = np.pad(V, 1, mode="constant", constant_values=0.0)

        max_nb = np.maximum.reduce([
            P[0:-2, 1:-1],   # N
            P[2:,   1:-1],   # S
            P[1:-1, 0:-2],   # W
            P[1:-1, 2:  ],   # E
            P[0:-2, 0:-2],   # NW
            P[0:-2, 2:  ],   # NE
            P[2:,   0:-2],   # SW
            P[2:,   2:  ],   # SE
        ])

        V_new = R + gamma * max_nb
        if float(np.max(np.abs(V_new - V))) < tol:
            V = V_new
            break
        V = V_new

    return V


# ---------------------------------------------------------------------------
# Greedy policy trace
# ---------------------------------------------------------------------------

_DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1)]


def trace_path(
    V: np.ndarray,
    start_row: int,
    start_col: int,
    max_steps: int = 8,
) -> list[tuple[int, int]]:
    """Follow the greedy policy (always step toward highest V-neighbour).

    Stops when no neighbour improves V (local/global max) or after max_steps.
    Visited-cell set prevents cycles.
    """
    H, W = V.shape
    path = [(start_row, start_col)]
    visited: set[tuple[int, int]] = {(start_row, start_col)}
    r, c = start_row, start_col

    for _ in range(max_steps):
        best_v = V[r, c]
        best_next: Optional[tuple[int, int]] = None

        for dr, dc in _DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if V[nr, nc] > best_v:
                    best_v = V[nr, nc]
                    best_next = (nr, nc)

        if best_next is None:
            break
        r, c = best_next
        path.append((r, c))
        visited.add((r, c))

    return path


# ---------------------------------------------------------------------------
# Real-street routing via OSRM public API
# ---------------------------------------------------------------------------

def get_osrm_route(
    start_lat: float,
    start_lng: float,
    dest_lat: float,
    dest_lng: float,
    mode: str = "foot",
    timeout: float = 6.0,
) -> Optional[list[tuple[float, float]]]:
    """Fetch a real pedestrian route from the OSRM public API.

    Returns list of (lat, lng) waypoints following actual streets, or
    None if the request fails (network error, timeout, bad response).

    The OSRM demo server is free to use but not guaranteed.  The caller
    should fall back to streetify_path() when this returns None.
    """
    base = (
        "https://routing.openstreetmap.de/routed-foot"
        if mode == "foot"
        else "http://router.project-osrm.org"
    )
    url = (
        f"{base}/route/v1/{mode}/"
        f"{start_lng:.6f},{start_lat:.6f};"
        f"{dest_lng:.6f},{dest_lat:.6f}"
    )
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        coords = data["routes"][0]["geometry"]["coordinates"]
        # OSRM returns [lng, lat]; convert to (lat, lng)
        return [(lat, lng) for lng, lat in coords]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rectilinear street-grid fallback
# ---------------------------------------------------------------------------

def streetify_path(
    latlng_path: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Add rectilinear corner waypoints (horizontal then vertical) to mimic
    NYC's street-grid geometry when OSRM is unavailable."""
    if len(latlng_path) < 2:
        return latlng_path

    result: list[tuple[float, float]] = [latlng_path[0]]
    for i in range(len(latlng_path) - 1):
        lat1, lng1 = latlng_path[i]
        lat2, lng2 = latlng_path[i + 1]
        if abs(lat2 - lat1) > 1e-7 and abs(lng2 - lng1) > 1e-7:
            # Alternate horizontal-first / vertical-first at each hop
            if i % 2 == 0:
                result.append((lat1, lng2))
            else:
                result.append((lat2, lng1))
        result.append((lat2, lng2))
    return result


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    # Walking route — (lat, lng) waypoints.  street_route is the OSRM result
    # if available; grid_route is the rectilinear fallback.
    street_route: Optional[list[tuple[float, float]]]  # None if OSRM unavailable
    grid_route:   list[tuple[float, float]]             # always present

    cell_path: list[tuple[int, int]]

    destination_lat: float
    destination_lng: float

    destination_restaurants: pd.DataFrame   # restaurants near destination
    terrible_restaurants:    pd.DataFrame   # Grade-C near route (if filtered)

    # Meal-matching
    top_matches:     list[tuple[str, float]]   # [(cuisine, score), …] top-5
    cuisine_applied: bool                      # was a cuisine filter applied?

    # Route stats
    path_distance_km:         float
    walk_minutes:             float
    n_destination_restaurants: int
    mean_destination_safety:  float   # avg safety score at destination

    # RL internals for visualisation
    value_map:  np.ndarray
    reward_map: np.ndarray   # proximity-biased reward used for VI

    gamma:       float
    sigma_km:    float
    walk_preset: str


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_safe_route(
    df: pd.DataFrame,
    user_lat: float,
    user_lng: float,
    meal_description: str = "",
    walk_preset_key: str = "15 min (~1.1 km)",
    cuisine_importance: float = 0.7,
    danger_filter: bool = True,
    destination_radius_cells: int = 2,
    smooth_sigma: float = 1.0,
    use_osrm: bool = True,
) -> RouteResult:
    """Plan a short walking route to the nearest safe, cuisine-matching cluster.

    Parameters
    ----------
    walk_preset_key : str
        One of WALK_PRESETS keys.  Determines gamma, max_steps, sigma_km.
    cuisine_importance : float in [0, 1]
        How strictly to filter restaurants to the user's preferred cuisine.
        0 → no filter; 1 → only best-matching cuisine counts.
    danger_filter : bool
        If True, Grade-C restaurants contribute 0 to the reward.
    use_osrm : bool
        If True, attempt to fetch a real street route from the OSRM API.
    """
    preset = WALK_PRESETS.get(walk_preset_key, WALK_PRESETS["15 min (~1.1 km)"])
    gamma     = preset["gamma"]
    max_steps = preset["max_steps"]
    sigma_km  = preset["sigma_km"]
    walk_km   = preset["walk_km"]

    # ── 1. Cuisine matching ─────────────────────────────────────────────────
    cuisines = sorted(df["cuisine"].dropna().unique().tolist())
    matcher  = MealMatcher().fit(cuisines)
    has_desc = bool(meal_description and meal_description.strip())
    cuisine_scores = matcher.score(meal_description) if has_desc else {c: 1.0 for c in cuisines}
    top_matches    = matcher.top_matches(meal_description, n=5) if has_desc else []

    # ── 2. Build reward grid ────────────────────────────────────────────────
    R_raw, safe_df, terr_df = build_reward_grid(
        df,
        cuisine_scores=cuisine_scores,
        cuisine_importance=cuisine_importance if has_desc else 0.0,
        danger_filter=danger_filter,
        smooth_sigma=smooth_sigma,
        has_description=has_desc,
    )

    # ── 3. Apply proximity Gaussian bias (NEW) ───────────────────────────────
    start_row, start_col = latlng_to_cell(user_lat, user_lng)
    R_biased = apply_proximity_bias(R_raw, start_row, start_col, sigma_km)

    # Re-normalise after proximity weighting
    r_max = R_biased.max()
    if r_max > 1e-8:
        R_biased /= r_max

    # ── 4. Value Iteration ──────────────────────────────────────────────────
    V = value_iteration(R_biased, gamma=gamma)

    # ── 5. Trace greedy path (capped at max_steps) ──────────────────────────
    cell_path = trace_path(V, start_row, start_col, max_steps=max_steps)

    # ── 6. Convert to (lat, lng) waypoints ──────────────────────────────────
    latlng_raw = [cell_to_latlng(r, c) for r, c in cell_path]
    grid_route = streetify_path(latlng_raw)

    dest_row, dest_col = cell_path[-1]
    dest_lat, dest_lng = cell_to_latlng(dest_row, dest_col)

    # ── 7. Real street routing via OSRM ─────────────────────────────────────
    street_route: Optional[list[tuple[float, float]]] = None
    if use_osrm:
        street_route = get_osrm_route(user_lat, user_lng, dest_lat, dest_lng)

    # ── 8. Destination restaurants ──────────────────────────────────────────
    if "grid_row" in safe_df.columns and "grid_col" in safe_df.columns:
        mask = (
            (safe_df["grid_row"] - dest_row).abs() <= destination_radius_cells
        ) & (
            (safe_df["grid_col"] - dest_col).abs() <= destination_radius_cells
        )
        dest_restaurants = safe_df[mask].copy()
    else:
        dest_restaurants = safe_df.iloc[:0].copy()

    # ── 9. Grade-C restaurants near the route ───────────────────────────────
    terrible_near = pd.DataFrame()
    if danger_filter and not terr_df.empty and "grid_row" in terr_df.columns:
        TERR_R = 4
        path_rows = np.array([r for r, _ in cell_path])
        path_cols = np.array([c for _, c in cell_path])
        tr = terr_df["grid_row"].values
        tc = terr_df["grid_col"].values
        near = np.zeros(len(terr_df), dtype=bool)
        for pr, pc in zip(path_rows, path_cols):
            near |= (np.abs(tr - pr) <= TERR_R) & (np.abs(tc - pc) <= TERR_R)
        terrible_near = terr_df[near].copy()

    # ── 10. Stats ─────────────────────────────────────────────────────────
    dist_km  = path_distance_km(cell_path)
    walk_min = (dist_km / WALK_SPEED_KMH) * 60.0
    mean_sf  = float(dest_restaurants["safety"].mean()) if not dest_restaurants.empty else 0.0

    return RouteResult(
        street_route=street_route,
        grid_route=grid_route,
        cell_path=cell_path,
        destination_lat=dest_lat,
        destination_lng=dest_lng,
        destination_restaurants=dest_restaurants,
        terrible_restaurants=terrible_near,
        top_matches=top_matches,
        cuisine_applied=(has_desc and cuisine_importance > 0),
        path_distance_km=dist_km,
        walk_minutes=walk_min,
        n_destination_restaurants=len(dest_restaurants),
        mean_destination_safety=mean_sf,
        value_map=V,
        reward_map=R_biased,
        gamma=gamma,
        sigma_km=sigma_km,
        walk_preset=walk_preset_key,
    )
