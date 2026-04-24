"""
streamlit_app.py
----------------
NYC Safe Restaurant Finder — six-part interactive Streamlit dashboard.

Part 1 — Data & Exploration
    1.1  Three-step preprocessing pipeline (raw CSV → restaurants.csv)
    1.2  Filterable data explorer
    1.3  Interactive NYC map (coloured by grade / critical-violation heatmap)

Part 2 — KNN Grade Classifier
    Baseline K-Nearest Neighbors with cosine similarity, built from scratch.
    Shows the class-imbalance problem that motivates Part 3.

Part 3 — Decision Tree Grade Classifier
    Smarter from-scratch classifier with weighted Gini splits, balanced class
    weights, engineered features, and full interpretability (feature importance,
    decision rules).  Directly addresses every weakness identified in Part 2.

Part 4 — Cuisine Type Predictor
    scikit-learn TF-IDF (char n-grams) + Logistic Regression predicts a
    restaurant's cuisine type from its name alone.  Supports random and
    geographic (borough hold-out) train/test splits.  Returns top-3 cuisine
    predictions with probabilities.

Part 5 — Safe Restaurant Route Finder
    Reinforcement Learning (Value Iteration, pure NumPy) plans a short walking
    route from the user's NYC location to the nearest cluster of safe,
    cuisine-matching restaurants.  Route displayed on a real OpenStreetMap via
    Folium; street-level waypoints fetched from the OSRM public API.

Part 6 — Ultimate Restaurant Finder
    Everything combined:
    • KNN + Decision Tree predictions are blended into a continuous safety
      score (replacing the binary A/B/C lookup) for a richer RL reward signal.
    • NLP input is embedded into a TF-IDF restaurant matrix — supports dish
      descriptions, cuisine styles, OR "something like [restaurant name]".
    • Two navigation modes: Area (RL cluster routing) or Direct (ranked list
      with next/previous restaurant navigation).

Run with:
    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.knn_classifier import (
    KNNClassifier,
    FEATURES as KNN_FEATURES,
    VALID_GRADES,
    build_confusion_matrix,
    compute_metrics,
    prepare_knn_data,
)
from src.decision_tree import (
    DecisionTreeClassifier,
    DT_ALL_FEATURES,
    DT_BASE_FEATURES,
    DT_EXTRA_FEATURES,
    prepare_dt_data,
)
from src.cuisine_predictor import (
    CuisinePredictor,
    BOROUGHS,
    prepare_cuisine_data,
    cuisine_accuracy,
    top3_accuracy,
    per_cuisine_f1,
)
from src.rl_route_finder import (
    find_safe_route,
    NYC_NEIGHBORHOODS,
    WALK_PRESETS,
    RouteResult,
    GRID_ROWS,
    GRID_COLS,
    LAT_MIN, LAT_MAX,
    LNG_MIN, LNG_MAX,
    GRID_LAT_RES,
    GRID_LNG_RES,
    cell_to_latlng,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Safe Restaurant Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_CSV = "data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv"
RESTAURANTS_CSV = "data/restaurants.csv"

# ---------------------------------------------------------------------------
# Grade colour palette
# ---------------------------------------------------------------------------
GRADE_COLOR_RGBA: dict[str, list[int]] = {
    "A": [0, 200, 100, 210],
    "B": [255, 200, 0, 210],
    "C": [240, 80, 60, 210],
}
GRADE_COLOR_HEX: dict[str, str] = {
    "A": "#00C864",
    "B": "#FFC800",
    "C": "#F0503C",
    "Other": "#9E9E9E",
}
_NO_GRADE_RGBA = [158, 158, 158, 140]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading raw inspection records…")
def _load_raw() -> pd.DataFrame:
    from src.data_loader import load_raw
    return load_raw(RAW_CSV)


@st.cache_data(show_spinner="Loading restaurant dataset…")
def _load_restaurants() -> pd.DataFrame:
    df = pd.read_csv(RESTAURANTS_CSV)
    df["latest_grade_date"] = pd.to_datetime(df["latest_grade_date"], errors="coerce")
    df["latest_inspection_date"] = pd.to_datetime(
        df["latest_inspection_date"], errors="coerce"
    )
    return df


# ---------------------------------------------------------------------------
# Map colour helper
# ---------------------------------------------------------------------------
def _add_map_color(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["color"] = df["latest_grade"].map(GRADE_COLOR_RGBA).apply(
        lambda v: v if isinstance(v, list) else _NO_GRADE_RGBA
    )
    return df


# ===========================================================================
# SIDEBAR — shared filters for Data Explorer & Map
# ===========================================================================
def _sidebar_filters(restaurants: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters  (Part 1)")

    boros = sorted(b for b in restaurants["boro"].dropna().unique() if b != "0")
    sel_boros = st.sidebar.multiselect("Borough", boros, default=boros)

    all_grades = ["A", "B", "C", "N", "Z", "—No grade—"]
    sel_grades = st.sidebar.multiselect("Latest grade", all_grades, default=["A", "B", "C"])

    top_cuisines = restaurants["cuisine"].value_counts().head(20).index.tolist()
    sel_cuisines = st.sidebar.multiselect("Cuisine (top 20)", top_cuisines, default=[])
    cuisine_text = st.sidebar.text_input("Or search cuisine", "")

    score_min = int(restaurants["latest_score"].dropna().min())
    score_max = int(restaurants["latest_score"].dropna().max())
    sel_score = st.sidebar.slider(
        "Latest score range (lower = better)",
        score_min, score_max, (score_min, min(score_max, 50)),
    )

    mask = pd.Series(True, index=restaurants.index)
    if sel_boros:
        mask &= restaurants["boro"].isin(sel_boros)

    grade_vals: list = []
    for g in sel_grades:
        grade_vals.append(np.nan if g == "—No grade—" else g)
    real_grades = [g for g in grade_vals if isinstance(g, str)]
    include_no_grade = any(not isinstance(g, str) for g in grade_vals)
    if sel_grades:
        g_mask = restaurants["latest_grade"].isin(real_grades)
        if include_no_grade:
            g_mask |= restaurants["latest_grade"].isna()
        mask &= g_mask

    if sel_cuisines:
        mask &= restaurants["cuisine"].isin(sel_cuisines)
    if cuisine_text.strip():
        mask &= restaurants["cuisine"].str.contains(cuisine_text.strip(), case=False, na=False)

    mask &= (
        restaurants["latest_score"].isna()
        | restaurants["latest_score"].between(sel_score[0], sel_score[1])
    )
    return restaurants[mask].copy()


# ===========================================================================
# APP TITLE
# ===========================================================================
st.title("NYC Safe Restaurant Finder")
st.markdown(
    "Six-part interactive dashboard: **data pipeline**, **KNN classifier**, "
    "**Decision Tree classifier**, **cuisine-type predictor**, "
    "**RL-powered safe route finder**, and the **Ultimate Finder** — "
    "built on 295,995 DOHMH inspection records."
)

_restaurants = _load_restaurants()
_filtered    = _sidebar_filters(_restaurants)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Part 1: Data & Exploration",
    "Part 2: KNN Classifier",
    "Part 3: Decision Tree Classifier",
    "Part 4: Cuisine Predictor",
    "Part 5: Safe Restaurant Route",
    "Part 6: Ultimate Finder",
])


# ===========================================================================
# PART 1 — DATA PIPELINE + DATA EXPLORER + NYC MAP
# ===========================================================================
with tab1:

    # ── 1.1  Conversion pipeline ────────────────────────────────────────────
    st.header("1.1  Three-Step Conversion Pipeline")
    st.markdown(
        "Each step below processes the data further. "
        "Toggle **Load raw data** to run Steps 1 & 2 live; "
        "Step 3 always reads the pre-built `restaurants.csv`."
    )

    c1, arr1, c2, arr2, c3 = st.columns([3, 0.4, 3, 0.4, 3])
    with c1:
        st.markdown(
            "<div style='background:#1e3a5f;border-radius:8px;padding:14px;"
            "text-align:center;color:white'><b>Step 1</b><br>Load Raw CSV<br>"
            "<small>DOHMH violation records</small></div>",
            unsafe_allow_html=True,
        )
    with arr1:
        st.markdown("<div style='font-size:2rem;text-align:center;margin-top:10px'>→</div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div style='background:#1a4d2e;border-radius:8px;padding:14px;"
            "text-align:center;color:white'><b>Step 2</b><br>Clean & Type-cast<br>"
            "<small>data_loader.load_raw()</small></div>",
            unsafe_allow_html=True,
        )
    with arr2:
        st.markdown("<div style='font-size:2rem;text-align:center;margin-top:10px'>→</div>",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(
            "<div style='background:#4a1a5e;border-radius:8px;padding:14px;"
            "text-align:center;color:white'><b>Step 3</b><br>Aggregate by Restaurant<br>"
            "<small>preprocessor.build_restaurant_table()</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    load_raw_data = st.toggle("Load raw CSV to explore Steps 1 & 2 (first load ~10 s)", value=False)

    st.subheader("Step 1 — Raw CSV snapshot")
    if load_raw_data:
        raw = _load_raw()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Violation records", f"{len(raw):,}")
        m2.metric("Unique restaurants (CAMIS)", f"{raw['camis'].nunique():,}")
        m3.metric("Columns", len(raw.columns))
        m4.metric("Date range",
                  f"{raw['inspection_date'].min().date()} -> "
                  f"{raw['inspection_date'].max().date()}")

        with st.expander("Original column names (before renaming)"):
            orig_cols = list(pd.read_csv(RAW_CSV, nrows=0).columns)
            st.dataframe(pd.DataFrame({"Original header": orig_cols}),
                         use_container_width=True, hide_index=True)

        with st.expander("Sample raw records (first 200 rows)"):
            st.dataframe(raw.head(200), use_container_width=True)

        st.subheader("Step 2 — After load_raw(): cleaning applied")
        st.markdown(
            """
| Transform | What happened |
|---|---|
| Column rename | 27 headers to snake_case |
| Whitespace strip | Leading/trailing spaces removed |
| Empty strings | Replaced with NaN |
| Date parsing | inspection_date, grade_date, record_date to datetime64 |
| Placeholder dates | 1900-01-01 inspection dates to NaT |
| Numeric cast | score, latitude, longitude, zipcode… to float64 |
| CAMIS | Cast to nullable integer Int64 |
| Categoricals | critical_flag title-cased; grade uppercased; boro title-cased |
"""
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Column dtypes after cleaning**")
            dtype_df = raw.dtypes.rename("dtype").reset_index()
            dtype_df.columns = ["column", "dtype"]
            dtype_df["dtype"] = dtype_df["dtype"].astype(str)
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Missing values per column**")
            null_df = raw.isnull().sum().rename("null_count").reset_index()
            null_df.columns = ["column", "null_count"]
            null_df["pct_missing"] = (null_df["null_count"] / len(raw) * 100).round(1)
            null_df = null_df[null_df["null_count"] > 0].sort_values("null_count", ascending=False)
            st.dataframe(null_df, use_container_width=True, hide_index=True)

        st.markdown("**Grade distribution in raw records**")
        grade_raw = raw["grade"].value_counts().reset_index()
        grade_raw.columns = ["grade", "count"]
        fig_grade_raw = px.bar(grade_raw, x="grade", y="count", color="grade",
                               color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()},
                               labels={"count": "Violation records"},
                               title="Raw-record grade distribution")
        fig_grade_raw.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_grade_raw, use_container_width=True)
    else:
        st.info("Enable the toggle to load and inspect the raw CSV live.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Violation records (known)", "295,995")
        m2.metric("Unique restaurants (CAMIS)", "30,935")
        m3.metric("Raw columns", "27")
        m4.metric("Source file", "DOHMH NYC Open Data")

    st.subheader("Step 3 — build_restaurant_table(): one row per restaurant")
    restaurants_p1 = _load_restaurants()
    st.markdown(
        """
`build_restaurant_table()` collapses the violation-level frame into a single row
per CAMIS. Five sub-steps run in sequence:

| Sub-step | What it does |
|---|---|
| **Static info** | Most-recent non-null value of name, address, coordinates, borough, cuisine |
| **Latest inspection** | Grade, score, action, inspection type from the newest inspection date |
| **Latest grade** | Prefers rows with A/B/C; encodes grade as 3/2/1 |
| **Inspection history** | Deduplicates to one row per (CAMIS, date) before computing mean/min/max score |
| **Violation summary** | Total, critical, non-critical violation counts; unique violation codes |
"""
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Restaurants", f"{len(restaurants_p1):,}")
    m2.metric("Output columns", len(restaurants_p1.columns))
    m3.metric("Grade A", f"{(restaurants_p1['latest_grade'] == 'A').sum():,}")
    m4.metric("Grade B", f"{(restaurants_p1['latest_grade'] == 'B').sum():,}")
    m5.metric("Grade C", f"{(restaurants_p1['latest_grade'] == 'C').sum():,}")

    col_left, col_right = st.columns(2)
    with col_left:
        grade_dist = (restaurants_p1["latest_grade"].fillna("No grade")
                      .value_counts().reset_index())
        grade_dist.columns = ["grade", "count"]
        fig_grade = px.bar(grade_dist, x="grade", y="count", color="grade",
                           color_discrete_map=GRADE_COLOR_HEX,
                           title="Restaurant-level grade distribution",
                           labels={"count": "Restaurants"})
        fig_grade.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_grade, use_container_width=True)
    with col_right:
        boro_dist = (restaurants_p1[restaurants_p1["boro"] != "0"]["boro"]
                     .value_counts().reset_index())
        boro_dist.columns = ["borough", "count"]
        fig_boro = px.pie(boro_dist, names="borough", values="count",
                          title="Restaurants by borough", hole=0.35)
        fig_boro.update_layout(height=320)
        st.plotly_chart(fig_boro, use_container_width=True)

    scored = restaurants_p1["latest_score"].dropna()
    fig_score = px.histogram(scored, nbins=60,
                             title="Latest inspection score distribution (lower = better)",
                             labels={"value": "Score", "count": "Restaurants"},
                             color_discrete_sequence=["#4C8BF5"])
    fig_score.add_vline(x=14, line_dash="dash", line_color=GRADE_COLOR_HEX["A"],
                        annotation_text="A cutoff (<=13)")
    fig_score.add_vline(x=28, line_dash="dash", line_color=GRADE_COLOR_HEX["B"],
                        annotation_text="B cutoff (14-27)")
    fig_score.update_layout(height=320)
    st.plotly_chart(fig_score, use_container_width=True)

    with st.expander("Full output column list"):
        col_df = pd.DataFrame({
            "column":   restaurants_p1.columns.tolist(),
            "dtype":    restaurants_p1.dtypes.astype(str).tolist(),
            "non-null": restaurants_p1.notna().sum().tolist(),
            "null":     restaurants_p1.isna().sum().tolist(),
        })
        st.dataframe(col_df, use_container_width=True, hide_index=True)

    # ── 1.2  Data Explorer ──────────────────────────────────────────────────
    st.markdown("---")
    st.header("1.2  Data Explorer")
    st.markdown(
        f"Showing **{len(_filtered):,}** of {len(_restaurants):,} restaurants "
        f"matching the sidebar filters."
    )
    DISPLAY_COLS = [
        "camis", "dba", "boro", "cuisine", "building", "street", "zipcode",
        "latest_grade", "latest_score", "latest_inspection_date",
        "inspection_count", "total_violations", "critical_violations", "mean_score",
    ]
    display_cols = [c for c in DISPLAY_COLS if c in _filtered.columns]
    st.dataframe(_filtered[display_cols].reset_index(drop=True),
                 use_container_width=True, height=500)

    with st.expander("Summary statistics for filtered set"):
        num_cols_show = ["latest_score", "mean_score", "inspection_count",
                         "total_violations", "critical_violations"]
        st.dataframe(_filtered[[c for c in num_cols_show if c in _filtered.columns]]
                     .describe().round(2), use_container_width=True)

    # ── 1.3  NYC Map ────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("1.3  NYC Restaurant Map")

    map_df = _filtered.dropna(subset=["latitude", "longitude"]).copy()
    map_df = _add_map_color(map_df)
    map_df["tooltip_grade"]   = map_df["latest_grade"].fillna("—")
    map_df["tooltip_score"]   = map_df["latest_score"].fillna(np.nan)
    map_df["tooltip_name"]    = map_df["dba"].fillna("Unknown")
    map_df["tooltip_cuisine"] = map_df["cuisine"].fillna("—")
    map_df["tooltip_address"] = (
        map_df["building"].fillna("").astype(str).str.strip()
        + " "
        + map_df["street"].fillna("").astype(str).str.strip()
    ).str.strip()

    st.markdown(
        f"Plotting **{len(map_df):,}** restaurants with valid coordinates "
        f"(out of {len(_filtered):,} matching filters)."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        point_size = st.slider("Point radius (m)", 20, 200, 60, step=10, key="map_radius")
    with col_ctrl2:
        color_by = st.selectbox("Colour by",
                                ["Grade (A/B/C)", "Critical violations (heatmap)"],
                                key="map_color_by")
    with col_ctrl3:
        map_style = st.selectbox("Map style", ["dark", "light", "road", "satellite"],
                                 key="map_style")

    STYLE_MAP = {
        "dark":      "mapbox://styles/mapbox/dark-v10",
        "light":     "mapbox://styles/mapbox/light-v10",
        "road":      "mapbox://styles/mapbox/streets-v11",
        "satellite": "mapbox://styles/mapbox/satellite-streets-v11",
    }

    if color_by == "Grade (A/B/C)":
        layer = pdk.Layer("ScatterplotLayer", data=map_df,
                          get_position=["longitude", "latitude"],
                          get_fill_color="color", get_radius=point_size,
                          pickable=True, opacity=0.85, stroked=False)
        tooltip = {
            "html": ("<b>{tooltip_name}</b><br/>"
                     "Cuisine: {tooltip_cuisine}<br/>"
                     "Address: {tooltip_address}<br/>"
                     "Grade: <b>{tooltip_grade}</b> &nbsp; Score: {tooltip_score}<br/>"
                     "Inspections: {inspection_count} &nbsp; "
                     "Critical violations: {critical_violations}"),
            "style": {"backgroundColor": "#1e1e2e", "color": "white",
                      "fontSize": "13px", "padding": "8px", "borderRadius": "6px"},
        }
    else:
        layer = pdk.Layer("HeatmapLayer", data=map_df,
                          get_position=["longitude", "latitude"],
                          get_weight="critical_violations",
                          aggregation="SUM", opacity=0.8)
        tooltip = {"html": "<b>{tooltip_name}</b><br/>Critical violations: {critical_violations}",
                   "style": {"backgroundColor": "#1e1e2e", "color": "white"}}

    view = pdk.ViewState(latitude=40.7128, longitude=-73.9760, zoom=10.5, pitch=0)
    deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip,
                    map_style=STYLE_MAP[map_style])
    st.pydeck_chart(deck, use_container_width=True, height=600)

    if color_by == "Grade (A/B/C)":
        leg_cols = st.columns(4)
        for col, (grade, color) in zip(leg_cols, [
            ("A — Excellent (score <=13)",        GRADE_COLOR_HEX["A"]),
            ("B — Good (score 14-27)",             GRADE_COLOR_HEX["B"]),
            ("C — Needs improvement (score >=28)", GRADE_COLOR_HEX["C"]),
            ("No grade / pending",                 GRADE_COLOR_HEX["Other"]),
        ]):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{color};border-radius:50%;margin-right:6px'></span>{grade}",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Grade breakdown by borough (filtered view)")
    graded_map = _filtered[_filtered["latest_grade"].isin(["A", "B", "C"])].copy()
    if not graded_map.empty:
        boro_grade = (graded_map.groupby(["boro", "latest_grade"])
                      .size().reset_index(name="count"))
        boro_grade = boro_grade[boro_grade["boro"] != "0"]
        fig_bg = px.bar(boro_grade, x="boro", y="count", color="latest_grade",
                        color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()
                                            if k in VALID_GRADES},
                        barmode="group",
                        labels={"boro": "Borough", "count": "Restaurants",
                                "latest_grade": "Grade"},
                        title="Restaurants per grade per borough")
        fig_bg.update_layout(height=350)
        st.plotly_chart(fig_bg, use_container_width=True)
    else:
        st.info("No graded restaurants in current filter selection.")


# ===========================================================================
# PART 2 — KNN CLASSIFIER
# ===========================================================================
with tab2:
    st.header("Part 2: KNN Grade Classifier")
    st.markdown(
        """
A **K-Nearest Neighbors (KNN)** classifier — built entirely from scratch with NumPy —
predicts a restaurant's letter grade **(A / B / C)** from four features derived from
its prior inspection history.

| Feature | Description |
|---|---|
| `mean_score` | Average inspection score across all recorded inspections (lower = better) |
| `critical_violations` | Total count of critical violations ever recorded |
| `total_violations` | Total violation rows across all inspections |
| `days_since_last_inspection` | Days elapsed since the most-recent inspection |

**Distance metric:** cosine similarity on z-score-normalised feature vectors.
**Temporal split:** restaurants sorted by `latest_inspection_date`; the earliest
**(1 - test %)** form the training set — the model never observes future data during training.
        """
    )

    st.subheader("2.1  Parameters")
    col_k, col_split, col_chunk = st.columns(3)
    with col_k:
        k_val = st.slider("K — number of neighbours", 1, 31, 7, step=2, key="knn_k",
                          help="Odd values reduce tie-breaking ambiguity.")
    with col_split:
        test_pct_knn = st.slider("Test set size (%)", 10, 40, 20, step=5, key="knn_test_pct",
                                 help="Most-recently-inspected restaurants form the test set.")
    with col_chunk:
        chunk_size_knn = st.select_slider("Prediction batch size", options=[64, 128, 256, 512],
                                          value=256, key="knn_chunk")

    st.markdown("---")
    st.subheader("2.2  Data Preparation")

    restaurants_knn = _load_restaurants()
    (
        X_train_knn, y_train_knn,
        X_test_knn,  y_test_knn,
        train_df_knn, test_df_knn, cutoff_knn,
    ) = prepare_knn_data(restaurants_knn, test_fraction=test_pct_knn / 100)

    with st.expander("Feature distributions (A/B/C graded restaurants)", expanded=False):
        feat_df_knn = restaurants_knn[
            restaurants_knn["latest_grade"].isin(VALID_GRADES)
        ][KNN_FEATURES + ["latest_grade"]].dropna()
        feat_cols = st.columns(2)
        for fi, feat in enumerate(KNN_FEATURES):
            fig_f = go.Figure()
            for grade in VALID_GRADES:
                vals = feat_df_knn[feat_df_knn["latest_grade"] == grade][feat].dropna()
                fig_f.add_trace(go.Histogram(x=vals, name=f"Grade {grade}",
                                             marker_color=GRADE_COLOR_HEX[grade],
                                             opacity=0.65, nbinsx=40))
            fig_f.update_layout(barmode="overlay", title=feat, height=260,
                                 margin=dict(t=40, b=20, l=20, r=20),
                                 legend=dict(orientation="h", y=1.15))
            feat_cols[fi % 2].plotly_chart(fig_f, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"**Training set** — {len(train_df_knn):,} restaurants  \n"
                    f"Inspections before **{cutoff_knn.date()}**")
        tr_counts = {g: int(np.sum(y_train_knn == g)) for g in VALID_GRADES}
        fig_tr = px.bar(pd.DataFrame({"Grade": list(tr_counts), "Count": list(tr_counts.values())}),
                        x="Grade", y="Count", color="Grade",
                        color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()
                                            if k in VALID_GRADES},
                        title="Train grade distribution", height=260)
        fig_tr.update_layout(showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_tr, use_container_width=True)
    with c_right:
        st.markdown(f"**Test set** — {len(test_df_knn):,} restaurants  \n"
                    f"Inspections on/after **{cutoff_knn.date()}**")
        te_counts = {g: int(np.sum(y_test_knn == g)) for g in VALID_GRADES}
        fig_te = px.bar(pd.DataFrame({"Grade": list(te_counts), "Count": list(te_counts.values())}),
                        x="Grade", y="Count", color="Grade",
                        color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()
                                            if k in VALID_GRADES},
                        title="Test grade distribution", height=260)
        fig_te.update_layout(showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_te, use_container_width=True)

    st.markdown("---")
    st.subheader("2.3  Training")
    st.markdown(
        "KNN is a **lazy learner** — the 'training' step only z-score-normalises "
        "the feature matrix and stores it in memory.  All computation happens at prediction time."
    )

    run_knn = st.button("Train & Evaluate KNN", type="primary", key="knn_run_btn")

    if run_knn:
        train_status = st.status("Training KNN classifier...", expanded=True)
        with train_status:
            st.write(f"Fitting on **{len(X_train_knn):,}** training restaurants with K = {k_val}...")
            clf_knn = KNNClassifier(k=k_val)
            clf_knn.fit(X_train_knn, y_train_knn)
            means_df = pd.DataFrame({
                "Feature":         KNN_FEATURES,
                "Train mean (raw)": clf_knn.feature_means.round(3).tolist(),
                "Train std  (raw)": clf_knn.feature_stds.round(3).tolist(),
            })
            st.dataframe(means_df, use_container_width=True, hide_index=True)
            train_status.update(label="Training complete.", state="complete")

        st.write(f"**Predicting** on {len(X_test_knn):,} test restaurants...")
        knn_progress = st.progress(0.0, text="Running KNN predictions...")

        def _knn_progress(frac: float) -> None:
            knn_progress.progress(frac, text=f"KNN predictions: {frac*100:.0f}% complete")

        y_pred_knn = clf_knn.predict(X_test_knn, chunk_size=chunk_size_knn,
                                     progress_callback=_knn_progress)
        knn_progress.progress(1.0, text="Predictions complete!")

        st.session_state["knn_results"] = {
            "y_test":  y_test_knn,
            "y_pred":  y_pred_knn,
            "metrics": compute_metrics(y_test_knn, y_pred_knn, VALID_GRADES),
            "cm":      build_confusion_matrix(y_test_knn, y_pred_knn, VALID_GRADES),
            "k":       k_val,
            "n_train": len(X_train_knn),
            "n_test":  len(X_test_knn),
            "cutoff":  cutoff_knn,
        }

    res_knn = st.session_state.get("knn_results")
    if res_knn is not None:
        st.markdown("---")
        st.subheader("2.4  Evaluation Results")
        st.caption(f"K = {res_knn['k']} | train = {res_knn['n_train']:,} | "
                   f"test = {res_knn['n_test']:,} | cutoff = {res_knn['cutoff'].date()}")

        metrics_knn = res_knn["metrics"]
        acc_knn = metrics_knn["accuracy"]["value"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy",  f"{acc_knn:.1%}")
        m2.metric("Macro Precision", f"{metrics_knn['macro']['precision']:.4f}")
        m3.metric("Macro Recall",    f"{metrics_knn['macro']['recall']:.4f}")
        m4.metric("Macro F1",        f"{metrics_knn['macro']['f1']:.4f}")

        st.markdown("**Per-class precision, recall, F1**")
        knn_rows = []
        for grade in VALID_GRADES:
            g = metrics_knn[grade]
            knn_rows.append({"Grade": grade, "Precision": g["precision"],
                              "Recall": g["recall"], "F1-score": g["f1"],
                              "Support": g["support"]})
        knn_pc_df = pd.DataFrame(knn_rows)
        st.dataframe(knn_pc_df, use_container_width=True, hide_index=True)

        metric_cols = st.columns(3)
        for mi, metric_name in enumerate(["Precision", "Recall", "F1-score"]):
            fig_m = px.bar(knn_pc_df, x="Grade", y=metric_name, color="Grade",
                           color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()
                                               if k in VALID_GRADES},
                           text=metric_name, title=metric_name, height=280, range_y=[0, 1.05])
            fig_m.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_m.update_layout(showlegend=False, margin=dict(t=40, b=20))
            metric_cols[mi].plotly_chart(fig_m, use_container_width=True)

        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        cm_knn = res_knn["cm"]
        cm_knn_norm = cm_knn.astype(float) / (cm_knn.sum(axis=1, keepdims=True) + 1e-9)
        cm_knn_text = [[f"{cm_knn[i][j]}<br>({cm_knn_norm[i][j]:.1%})"
                        for j in range(len(VALID_GRADES))]
                       for i in range(len(VALID_GRADES))]
        fig_cm_knn = go.Figure(data=go.Heatmap(
            z=cm_knn_norm,
            x=[f"Pred {g}" for g in VALID_GRADES],
            y=[f"Actual {g}" for g in VALID_GRADES],
            text=cm_knn_text, texttemplate="%{text}",
            colorscale="Blues", showscale=True, colorbar=dict(title="Row %"),
        ))
        fig_cm_knn.update_layout(title=f"Confusion Matrix (K={res_knn['k']})",
                                  height=380, xaxis=dict(side="top"))
        st.plotly_chart(fig_cm_knn, use_container_width=True)

        dist_knn = []
        for grade in VALID_GRADES:
            dist_knn.append({"Grade": grade, "Type": "Actual",
                              "Count": int(np.sum(res_knn["y_test"] == grade))})
            dist_knn.append({"Grade": grade, "Type": "Predicted",
                              "Count": int(np.sum(res_knn["y_pred"] == grade))})
        fig_dist_knn = px.bar(pd.DataFrame(dist_knn), x="Grade", y="Count", color="Type",
                              barmode="group",
                              color_discrete_map={"Actual": "#4C8BF5", "Predicted": "#F5844C"},
                              title="Actual vs Predicted distribution on test set", height=320)
        st.plotly_chart(fig_dist_knn, use_container_width=True)
    else:
        st.info("Set parameters above and click **Train & Evaluate KNN** to run the classifier.")


# ===========================================================================
# PART 3 — DECISION TREE CLASSIFIER
# ===========================================================================
with tab3:
    st.header("Part 3: Decision Tree Grade Classifier")

    # ── 3.1  Why KNN Is Not Enough ─────────────────────────────────────────
    st.subheader("3.1  Why KNN Struggles on This Data")

    st.markdown(
        """
Running KNN on this dataset almost always predicts **Grade A** regardless of
parameters. Below are the five structural reasons why:
        """
    )

    # Class imbalance visualisation
    grade_counts = (
        _restaurants[_restaurants["latest_grade"].isin(VALID_GRADES)]
        ["latest_grade"].value_counts()
        .reindex(VALID_GRADES).fillna(0)
    )
    total_graded = grade_counts.sum()
    imb_df = pd.DataFrame({
        "Grade": VALID_GRADES,
        "Count": grade_counts.values.astype(int),
        "Pct":   (grade_counts.values / total_graded * 100).round(1),
    })

    col_imb, col_explain = st.columns([1.2, 1.8])
    with col_imb:
        fig_imb = px.bar(imb_df, x="Grade", y="Count", color="Grade",
                         color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()
                                             if k in VALID_GRADES},
                         text="Pct", title="Class imbalance in training data",
                         height=320)
        fig_imb.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_imb.update_layout(showlegend=False, yaxis_title="Restaurants",
                               margin=dict(t=50, b=20))
        st.plotly_chart(fig_imb, use_container_width=True)

    with col_explain:
        a_pct = imb_df.loc[imb_df["Grade"] == "A", "Pct"].values[0]
        b_pct = imb_df.loc[imb_df["Grade"] == "B", "Pct"].values[0]
        c_pct = imb_df.loc[imb_df["Grade"] == "C", "Pct"].values[0]
        st.markdown(
            f"""
**Problem 1 — Severe class imbalance**
Grade A makes up **{a_pct:.1f}%** of the data, B only **{b_pct:.1f}%**, C only
**{c_pct:.1f}%**. In a standard majority-vote KNN the K nearest neighbours of
almost any restaurant will be dominated by A-grade examples — so the classifier
defaults to A regardless of the true grade.

**Problem 2 — Lazy learner, no learned boundary**
KNN memorises training points but never learns *why* a restaurant gets a B or C.
It cannot generalise beyond the density of the training distribution.

**Problem 3 — Cosine similarity conflates unrelated scales**
Cosine similarity measures the angle between vectors. Features like
`total_violations` (range 0–300+) and `mean_score` (range 0–150) point in
very different directions in raw space. Even after z-scoring the angle is
dominated by outliers and the metric carries no causal meaning for grades.

**Problem 4 — No feature importance**
KNN gives no indication of which features matter. We cannot tell whether
`critical_violations` or `mean_score` is driving predictions.

**Problem 5 — O(n) inference cost**
Every prediction requires comparing against all 20 k training restaurants.
A smarter model should learn a compact decision function instead.
            """
        )

    # ── 3.2  The Decision Tree Solution ────────────────────────────────────
    st.markdown("---")
    st.subheader("3.2  The Decision Tree Solution")
    st.markdown(
        """
A **Decision Tree** trained with **weighted Gini impurity** directly solves all
five problems:

| KNN weakness | Decision Tree fix |
|---|---|
| Class imbalance | Balanced class weights: each sample's contribution is scaled by `n / (n_classes * class_count)`, giving B and C equal influence on splits |
| No learned boundary | Learns explicit thresholds: *"if mean_score <= 11.2 AND critical_rate <= 0.43 then Grade A"* |
| Meaningless cosine distance | Gini impurity measures prediction purity — directly tied to classification quality |
| No feature importance | Each split records its weighted Gini decrease; summed over the tree gives interpretable importances |
| O(n) inference | Inference is O(depth) — typically 10-15 comparisons per prediction |

**Richer feature set:** in addition to the original 4 KNN features, the tree
can use 6 engineered features, giving it more signal to distinguish B/C from A.

| Engineered feature | Formula |
|---|---|
| `min_score` | Lowest score across all inspections |
| `max_score` | Highest score across all inspections |
| `score_range` | `max_score - min_score` (consistency indicator) |
| `inspection_count` | Total number of separate inspection visits |
| `critical_rate` | `critical_violations / total_violations` |
| `violations_per_inspection` | `total_violations / inspection_count` |
        """
    )

    # ── 3.3  Parameters ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3.3  Parameters")

    col_d, col_mss, col_msl = st.columns(3)
    with col_d:
        max_depth_dt = st.slider(
            "Max tree depth", 2, 20, 12, key="dt_max_depth",
            help="Deeper trees learn finer boundaries but may overfit.",
        )
    with col_mss:
        min_split_dt = st.slider(
            "Min samples to split a node", 2, 100, 20, key="dt_min_split",
            help="Higher values produce a simpler, more regularised tree.",
        )
    with col_msl:
        min_leaf_dt = st.slider(
            "Min samples per leaf", 1, 50, 10, key="dt_min_leaf",
            help="Prevents tiny, noisy leaf nodes.",
        )

    col_cw, col_nt, col_tp = st.columns(3)
    with col_cw:
        class_weight_dt = st.selectbox(
            "Class weights", ["balanced", "uniform"], key="dt_cw",
            help="'balanced' corrects for grade A dominance.",
        )
    with col_nt:
        n_thresh_dt = st.slider(
            "Candidate thresholds per feature", 10, 100, 40, step=10, key="dt_n_thresh",
            help="More thresholds = finer splits but slower training.",
        )
    with col_tp:
        test_pct_dt = st.slider(
            "Test set size (%)", 10, 40, 20, step=5, key="dt_test_pct",
            help="Same temporal split logic as Part 2.",
        )

    # Feature selection
    st.markdown("**Select features to include**")
    feat_col1, feat_col2 = st.columns(2)
    selected_features: list[str] = []
    all_dt_feats = DT_ALL_FEATURES
    with feat_col1:
        st.markdown("*Original KNN features*")
        for f in DT_BASE_FEATURES:
            if st.checkbox(f, value=True, key=f"dt_feat_{f}"):
                selected_features.append(f)
    with feat_col2:
        st.markdown("*Engineered features (new)*")
        for f in DT_EXTRA_FEATURES:
            if st.checkbox(f, value=True, key=f"dt_feat_{f}"):
                selected_features.append(f)

    if not selected_features:
        st.warning("Select at least one feature.")
        st.stop()

    # ── 3.4  Train & Evaluate ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3.4  Training & Evaluation")

    run_dt = st.button("Train & Evaluate Decision Tree", type="primary", key="dt_run_btn")

    if run_dt:
        restaurants_dt = _load_restaurants()

        with st.spinner("Preparing data..."):
            (
                X_train_dt, y_train_dt,
                X_test_dt,  y_test_dt,
                train_df_dt, test_df_dt,
                cutoff_dt, feat_names_dt,
            ) = prepare_dt_data(
                restaurants_dt,
                feature_list=selected_features,
                test_fraction=test_pct_dt / 100,
            )

        # Show split stats
        c_tr, c_te = st.columns(2)
        with c_tr:
            st.markdown(f"**Train:** {len(X_train_dt):,} restaurants  \n"
                        f"Cutoff: **{cutoff_dt.date()}**")
            tr_counts_dt = {g: int((y_train_dt == g).sum()) for g in VALID_GRADES}
            st.dataframe(pd.DataFrame(tr_counts_dt, index=["count"]),
                         use_container_width=True)
        with c_te:
            st.markdown(f"**Test:** {len(X_test_dt):,} restaurants")
            te_counts_dt = {g: int((y_test_dt == g).sum()) for g in VALID_GRADES}
            st.dataframe(pd.DataFrame(te_counts_dt, index=["count"]),
                         use_container_width=True)

        # Fit
        node_counter = st.empty()
        train_status_dt = st.status("Training Decision Tree...", expanded=True)

        node_counts: dict = {"n": 0}

        def _dt_node_cb(n: int) -> None:
            node_counts["n"] = n
            node_counter.caption(f"Nodes built so far: {n}")

        with train_status_dt:
            st.write(
                f"max_depth={max_depth_dt} | min_split={min_split_dt} | "
                f"min_leaf={min_leaf_dt} | class_weight={class_weight_dt} | "
                f"n_thresholds={n_thresh_dt} | features={len(feat_names_dt)}"
            )
            clf_dt = DecisionTreeClassifier(
                max_depth=max_depth_dt,
                min_samples_split=min_split_dt,
                min_samples_leaf=min_leaf_dt,
                class_weight=class_weight_dt,
                n_thresholds=n_thresh_dt,
            )
            clf_dt.fit(X_train_dt, y_train_dt, progress_callback=_dt_node_cb)
            train_status_dt.update(
                label=(f"Training complete — depth={clf_dt.get_depth()} | "
                       f"nodes={clf_dt.get_n_nodes()} | leaves={clf_dt.get_n_leaves()}"),
                state="complete",
            )

        node_counter.empty()

        # Predict
        st.write(f"**Predicting** on {len(X_test_dt):,} test restaurants...")
        dt_pred_prog = st.progress(0.0, text="Running Decision Tree predictions...")
        dt_pred_done: dict = {"n": 0}

        y_pred_dt = clf_dt.predict(X_test_dt)
        dt_pred_prog.progress(1.0, text="Predictions complete!")

        y_pred_dt_arr = np.asarray(y_pred_dt)
        metrics_dt = compute_metrics(y_test_dt, y_pred_dt_arr, VALID_GRADES)
        cm_dt = build_confusion_matrix(y_test_dt, y_pred_dt_arr, VALID_GRADES)

        st.session_state["dt_results"] = {
            "y_test":         y_test_dt,
            "y_pred":         y_pred_dt_arr,
            "metrics":        metrics_dt,
            "cm":             cm_dt,
            "feat_names":     feat_names_dt,
            "importances":    clf_dt.feature_importances_.tolist(),
            "tree_text":      clf_dt.describe_tree(feat_names_dt, max_depth=4),
            "top_rules":      clf_dt.extract_top_rules(feat_names_dt, max_rules=12),
            "depth":          clf_dt.get_depth(),
            "n_nodes":        clf_dt.get_n_nodes(),
            "n_leaves":       clf_dt.get_n_leaves(),
            "max_depth_param": max_depth_dt,
            "class_weight":   class_weight_dt,
            "n_train":        len(X_train_dt),
            "n_test":         len(X_test_dt),
            "cutoff":         cutoff_dt,
        }

    # ── 3.5  Results ───────────────────────────────────────────────────────
    res_dt = st.session_state.get("dt_results")
    if res_dt is not None:
        st.markdown("---")
        st.subheader("3.5  Evaluation Results")
        st.caption(
            f"depth={res_dt['depth']} | nodes={res_dt['n_nodes']} | "
            f"leaves={res_dt['n_leaves']} | class_weight={res_dt['class_weight']} | "
            f"train={res_dt['n_train']:,} | test={res_dt['n_test']:,} | "
            f"cutoff={res_dt['cutoff'].date()}"
        )

        metrics_dt = res_dt["metrics"]
        acc_dt = metrics_dt["accuracy"]["value"]

        # Top-line metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy",  f"{acc_dt:.1%}")
        m2.metric("Macro Precision", f"{metrics_dt['macro']['precision']:.4f}")
        m3.metric("Macro Recall",    f"{metrics_dt['macro']['recall']:.4f}")
        m4.metric("Macro F1",        f"{metrics_dt['macro']['f1']:.4f}")

        # KNN comparison (if available)
        res_knn_compare = st.session_state.get("knn_results")
        if res_knn_compare is not None:
            st.markdown("**Comparison with KNN (Part 2)**")
            knn_m = res_knn_compare["metrics"]
            cmp_rows = []
            for grade in VALID_GRADES:
                cmp_rows.append({
                    "Grade":      grade,
                    "KNN F1":     knn_m[grade]["f1"],
                    "DT F1":      metrics_dt[grade]["f1"],
                    "KNN Recall": knn_m[grade]["recall"],
                    "DT Recall":  metrics_dt[grade]["recall"],
                    "Support":    metrics_dt[grade]["support"],
                })
            cmp_rows.append({
                "Grade":      "macro avg",
                "KNN F1":     knn_m["macro"]["f1"],
                "DT F1":      metrics_dt["macro"]["f1"],
                "KNN Recall": knn_m["macro"]["recall"],
                "DT Recall":  metrics_dt["macro"]["recall"],
                "Support":    res_dt["n_test"],
            })
            cmp_df = pd.DataFrame(cmp_rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # Side-by-side F1 comparison bar chart
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=VALID_GRADES, y=[knn_m[g]["f1"] for g in VALID_GRADES],
                name="KNN (Part 2)", marker_color="#4C8BF5",
                text=[f"{knn_m[g]['f1']:.3f}" for g in VALID_GRADES],
                textposition="outside",
            ))
            fig_cmp.add_trace(go.Bar(
                x=VALID_GRADES, y=[metrics_dt[g]["f1"] for g in VALID_GRADES],
                name="Decision Tree (Part 3)", marker_color="#F5844C",
                text=[f"{metrics_dt[g]['f1']:.3f}" for g in VALID_GRADES],
                textposition="outside",
            ))
            fig_cmp.update_layout(
                barmode="group",
                title="F1-score comparison: KNN vs Decision Tree",
                yaxis=dict(title="F1-score", range=[0, 1.15]),
                height=360,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Per-class metrics table
        st.markdown("**Per-class precision, recall, F1**")
        dt_rows = []
        for grade in VALID_GRADES:
            g = metrics_dt[grade]
            dt_rows.append({
                "Grade":     grade,
                "Precision": g["precision"],
                "Recall":    g["recall"],
                "F1-score":  g["f1"],
                "Support":   g["support"],
                "TP": g["tp"], "FP": g["fp"], "FN": g["fn"],
            })
        dt_pc_df = pd.DataFrame(dt_rows)
        st.dataframe(dt_pc_df, use_container_width=True, hide_index=True)

        # Per-metric bar charts
        metric_cols_dt = st.columns(3)
        for mi, metric_name in enumerate(["Precision", "Recall", "F1-score"]):
            fig_m = px.bar(dt_pc_df, x="Grade", y=metric_name, color="Grade",
                           color_discrete_map={k: v for k, v in GRADE_COLOR_HEX.items()
                                               if k in VALID_GRADES},
                           text=metric_name, title=metric_name,
                           height=280, range_y=[0, 1.15])
            fig_m.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_m.update_layout(showlegend=False, margin=dict(t=40, b=20))
            metric_cols_dt[mi].plotly_chart(fig_m, use_container_width=True)

        # Feature importances
        st.markdown("**Feature importances** (weighted Gini decrease, normalised)")
        imp_df = pd.DataFrame({
            "Feature":    res_dt["feat_names"],
            "Importance": res_dt["importances"],
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Teal",
                         title="Feature importances", height=max(300, len(imp_df) * 35))
        fig_imp.update_layout(yaxis_title="", coloraxis_showscale=False,
                               margin=dict(l=160, t=50))
        st.plotly_chart(fig_imp, use_container_width=True)

        # Confusion matrix
        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        cm_dt_v = res_dt["cm"]
        cm_dt_norm = cm_dt_v.astype(float) / (cm_dt_v.sum(axis=1, keepdims=True) + 1e-9)
        cm_dt_text = [[f"{cm_dt_v[i][j]}<br>({cm_dt_norm[i][j]:.1%})"
                       for j in range(len(VALID_GRADES))]
                      for i in range(len(VALID_GRADES))]
        fig_cm_dt = go.Figure(data=go.Heatmap(
            z=cm_dt_norm,
            x=[f"Pred {g}" for g in VALID_GRADES],
            y=[f"Actual {g}" for g in VALID_GRADES],
            text=cm_dt_text, texttemplate="%{text}",
            colorscale="Oranges", showscale=True, colorbar=dict(title="Row %"),
        ))
        fig_cm_dt.update_layout(title="Confusion Matrix — Decision Tree",
                                  height=380, xaxis=dict(side="top"))
        st.plotly_chart(fig_cm_dt, use_container_width=True)

        # Predicted vs actual distribution
        dist_dt = []
        for grade in VALID_GRADES:
            dist_dt.append({"Grade": grade, "Type": "Actual",
                             "Count": int((res_dt["y_test"] == grade).sum())})
            dist_dt.append({"Grade": grade, "Type": "Predicted",
                             "Count": int((res_dt["y_pred"] == grade).sum())})
        fig_dist_dt = px.bar(pd.DataFrame(dist_dt), x="Grade", y="Count", color="Type",
                              barmode="group",
                              color_discrete_map={"Actual": "#4C8BF5", "Predicted": "#F5844C"},
                              title="Actual vs Predicted distribution on test set", height=320)
        st.plotly_chart(fig_dist_dt, use_container_width=True)

        # Top decision rules
        st.markdown("**Top decision rules** (paths to the most-populated leaves)")
        rules = res_dt["top_rules"]
        for i, rule in enumerate(rules[:10], 1):
            cond_str = " AND ".join(rule["conditions"]) if rule["conditions"] else "root"
            proba_str = "  |  ".join(
                f"P({k})={v:.2f}" for k, v in sorted(rule["proba"].items())
            )
            grade_color = GRADE_COLOR_HEX.get(rule["label"], GRADE_COLOR_HEX["Other"])
            st.markdown(
                f"**Rule {i}** (n={rule['n_samples']:,})  "
                f"<span style='color:{grade_color};font-weight:bold'>→ Grade {rule['label']}</span>  \n"
                f"`{cond_str}`  \n"
                f"<small>{proba_str}</small>",
                unsafe_allow_html=True,
            )

        # Tree text sketch (top 4 levels)
        with st.expander("Tree structure (top 4 levels)", expanded=False):
            st.code(res_dt["tree_text"], language=None)

    else:
        st.info(
            "Configure parameters above and click **Train & Evaluate Decision Tree** "
            "to run the classifier.  \n"
            "Tip: run Part 2 first so the KNN vs Decision Tree comparison table appears here."
        )


# ===========================================================================
# PART 4 — CUISINE TYPE PREDICTOR (restaurant name → cuisine)
# ===========================================================================
with tab4:
    st.header("Part 4: Cuisine Type Predictor")
    st.markdown(
        """
Predict what **type of cuisine** a restaurant serves — using only its **name**.

**How it works:**
1. Restaurant names are normalised (lowercased, punctuation stripped).
2. A **TF-IDF vectorizer** with character n-grams (2–4 chars) converts each name
   into a sparse vector capturing linguistic patterns like *"burger"*, *"sushi"*,
   *"taqueria"*.
3. A **Logistic Regression** classifier (balanced class weights, multinomial softmax)
   is trained on those vectors.
4. At prediction time the model returns the **top-3 predicted cuisine types** with
   their probability scores.

> Names already in the dataset show the **actual** recorded cuisine next to the predictions.
        """
    )

    # ── 4.1  Parameters ────────────────────────────────────────────────────
    st.subheader("4.1  Parameters")

    col_split_method, col_split_param = st.columns(2)
    with col_split_method:
        split_method_p4 = st.radio(
            "Train / test split method",
            ["Completely random", "Hold out one area (borough)"],
            key="p4_split_method",
            help=(
                "**Completely random** — stratified random split, samples from "
                "every borough end up in both sets.  \n"
                "**Hold out one area** — every restaurant from the chosen borough "
                "becomes the test set; the rest train. Tests geographic generalisation."
            ),
        )
    with col_split_param:
        if split_method_p4 == "Completely random":
            test_pct_p4 = st.slider(
                "Test set size (%)", 10, 40, 20, step=5, key="p4_test_pct"
            )
            test_area_p4 = "Manhattan"   # unused
        else:
            test_area_p4 = st.selectbox(
                "Borough to use as test set", BOROUGHS, index=2, key="p4_test_area"
            )
            test_pct_p4 = 20   # unused

    col_c, col_ng, col_min = st.columns(3)
    with col_c:
        reg_C_p4 = st.select_slider(
            "Regularisation C (higher = less regularised)",
            options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0, key="p4_C"
        )
    with col_ng:
        ngram_max_p4 = st.slider(
            "Max char n-gram size", 2, 6, 4, key="p4_ngram",
            help="Larger n-grams capture longer word fragments. Range is (2, max)."
        )
    with col_min:
        min_count_p4 = st.slider(
            "Min restaurants per cuisine", 5, 50, 10, step=5, key="p4_min_count",
            help="Cuisines with fewer restaurants are excluded from training."
        )

    st.markdown("---")

    # ── 4.2  Train ─────────────────────────────────────────────────────────
    st.subheader("4.2  Train Model")

    run_p4 = st.button("Train Cuisine Predictor", type="primary", key="p4_run_btn")

    if run_p4:
        restaurants_p4 = _load_restaurants()
        split_enum = "random" if split_method_p4 == "Completely random" else "by_area"

        with st.spinner("Preparing data & training…"):
            (
                X_train_p4, X_test_p4, y_train_p4, y_test_p4,
                train_df_p4, test_df_p4, kept_p4,
            ) = prepare_cuisine_data(
                restaurants_p4,
                split_method=split_enum,
                test_fraction=test_pct_p4 / 100,
                test_area=test_area_p4,
                min_cuisine_count=min_count_p4,
            )

            predictor_p4 = CuisinePredictor(
                ngram_range=(2, ngram_max_p4),
                C=reg_C_p4,
            )
            predictor_p4.fit(X_train_p4, y_train_p4)

            y_pred_p4 = predictor_p4.predict_batch(X_test_p4)
            acc_p4    = cuisine_accuracy(y_test_p4, y_pred_p4)
            top3_acc  = top3_accuracy(predictor_p4, X_test_p4, y_test_p4)
            per_cls   = per_cuisine_f1(y_test_p4, y_pred_p4, kept_p4)

        st.session_state["p4_results"] = {
            "predictor":    predictor_p4,
            "X_test":       X_test_p4,
            "y_test":       y_test_p4,
            "y_pred":       y_pred_p4,
            "train_df":     train_df_p4,
            "test_df":      test_df_p4,
            "kept":         kept_p4,
            "acc":          acc_p4,
            "top3_acc":     top3_acc,
            "per_cls":      per_cls,
            "n_train":      len(X_train_p4),
            "n_test":       len(X_test_p4),
            "n_cuisines":   len(kept_p4),
            "split_method": split_method_p4,
            "test_area":    test_area_p4 if split_enum == "by_area" else None,
        }

    res_p4 = st.session_state.get("p4_results")

    # ── 4.3  Evaluation ────────────────────────────────────────────────────
    if res_p4 is not None:
        st.markdown("---")
        st.subheader("4.3  Model Performance")

        split_note = (
            f"Split: **{res_p4['split_method']}**"
            + (f" — test area: **{res_p4['test_area']}**" if res_p4["test_area"] else "")
        )
        st.caption(
            f"Train: {res_p4['n_train']:,} restaurants  |  "
            f"Test: {res_p4['n_test']:,} restaurants  |  "
            f"Cuisine classes: {res_p4['n_cuisines']}  |  {split_note}"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Top-1 Accuracy", f"{res_p4['acc']:.1%}",
                  help="Fraction of test restaurants where predicted #1 cuisine is correct.")
        m2.metric("Top-3 Accuracy", f"{res_p4['top3_acc']:.1%}",
                  help="Fraction where correct cuisine appears in top-3 predictions.")
        m3.metric("Cuisine classes", res_p4["n_cuisines"])

        # Per-cuisine F1 chart (top 20 by support)
        per_cls_df = res_p4["per_cls"]
        top20 = per_cls_df.head(20).sort_values("F1", ascending=True)
        fig_f1 = px.bar(
            top20, x="F1", y="Cuisine", orientation="h",
            color="F1", color_continuous_scale="Teal",
            title="F1-score — top 20 cuisines by support",
            hover_data=["Precision", "Recall", "Support"],
            height=max(350, len(top20) * 22),
        )
        fig_f1.update_layout(
            coloraxis_showscale=False, yaxis_title="",
            margin=dict(l=150, t=50, b=20),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        with st.expander("Full per-cuisine metrics table"):
            st.dataframe(per_cls_df, use_container_width=True, hide_index=True)

        # ── 4.4  Interactive Predictor ──────────────────────────────────
        st.markdown("---")
        st.subheader("4.4  Try It — Predict Cuisine from Restaurant Name")

        predictor_obj = res_p4["predictor"]
        test_df_ref   = res_p4["test_df"]

        # Randomise suggestion from test set
        col_input, col_suggest = st.columns([3, 1])
        with col_suggest:
            suggest_btn = st.button(
                "Randomise from test set", key="p4_suggest_btn",
                help="Pick a random restaurant from the test data for you to try."
            )
            if suggest_btn or "p4_suggested_name" not in st.session_state:
                if not test_df_ref.empty:
                    rng_row = test_df_ref.sample(1, random_state=None).iloc[0]
                    st.session_state["p4_suggested_name"] = rng_row["dba"]
                    st.session_state["p4_suggested_cuisine"] = rng_row["cuisine"]

        with col_input:
            restaurant_input = st.text_input(
                "Enter a restaurant name",
                value=st.session_state.get("p4_suggested_name", ""),
                key="p4_name_input",
                placeholder="e.g. Golden Dragon, Taco Bell, Il Forno…",
            )

        suggested_name = st.session_state.get("p4_suggested_name", "")
        if suggested_name and not suggest_btn:
            st.caption(
                f"Suggestion from test set: **{suggested_name}** "
                f"(actual cuisine: *{st.session_state.get('p4_suggested_cuisine', '?')}*)"
            )

        if restaurant_input.strip():
            name_query = restaurant_input.strip()
            top3 = predictor_obj.predict_top3(name_query)

            st.markdown(f"### Predictions for: *{name_query}*")

            # Check if name exists in the full dataset
            all_restaurants = _load_restaurants()
            match = all_restaurants[
                all_restaurants["dba"].str.strip().str.lower()
                == name_query.lower()
            ]

            if not match.empty:
                actual_cuisine = match.iloc[0]["cuisine"]
                st.success(
                    f"**Found in dataset!** Actual recorded cuisine: "
                    f"**{actual_cuisine}**"
                )
            else:
                st.info(
                    "This restaurant name is not in our dataset — showing model predictions only."
                )
                actual_cuisine = None

            # Top-3 results
            st.markdown("**Top-3 predicted cuisine types:**")
            for rank, (cuisine, prob) in enumerate(top3, 1):
                is_correct = (actual_cuisine is not None and cuisine == actual_cuisine)
                bar_color  = "#2ecc71" if is_correct else "#4C8BF5"
                tick       = " ✓" if is_correct else ""

                col_rank, col_bar, col_pct = st.columns([0.5, 5, 1])
                with col_rank:
                    st.markdown(f"**#{rank}**")
                with col_bar:
                    fill_pct = int(prob * 100)
                    st.markdown(
                        f"<div style='background:#2a2a3e;border-radius:6px;overflow:hidden;height:28px'>"
                        f"<div style='background:{bar_color};width:{fill_pct}%;height:100%;"
                        f"display:flex;align-items:center;padding-left:8px;"
                        f"color:white;font-weight:bold;font-size:14px'>"
                        f"{cuisine}{tick}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                with col_pct:
                    st.markdown(
                        f"<div style='text-align:right;line-height:28px;font-weight:bold'>"
                        f"{prob:.1%}</div>",
                        unsafe_allow_html=True,
                    )

            # Pie chart
            pie_df = pd.DataFrame(top3, columns=["Cuisine", "Probability"])
            fig_pie = px.pie(
                pie_df, names="Cuisine", values="Probability",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Top-3 probability breakdown",
                hole=0.35,
            )
            fig_pie.update_layout(height=320, margin=dict(t=50, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        else:
            st.info(
                "Enter a restaurant name above (or click **Randomise from test set**) "
                "to see cuisine predictions."
            )

    else:
        st.info(
            "Configure parameters above and click **Train Cuisine Predictor** to train the model."
        )


# ===========================================================================
# PART 5 — SAFE RESTAURANT ROUTE FINDER  (Reinforcement Learning)
# ===========================================================================
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

with tab5:
    st.header("Part 5: Safe Restaurant Route Finder")
    st.markdown(
        """
Find the **nearest cluster of safe, cuisine-matching restaurants** from your NYC
location — planned by a Reinforcement Learning agent and displayed on a real
OpenStreetMap street map.

| RL Component | Details |
|---|---|
| **State** | ~400 m × 420 m grid cell (111 × 113 = 12,543 cells covering all of NYC) |
| **Action** | Move in one of 8 compass directions |
| **Reward R(s)** | Safety scores (A=3 B=2 C=1) of cuisine-filtered restaurants, weighted by proximity Gaussian |
| **Discount γ** | Derived from walking-time budget — keeps routes within a realistic radius |
| **Algorithm** | **Value Iteration** — Bellman update fully vectorised with NumPy |
| **Street routing** | Real pedestrian route via **OSRM** public API (OpenStreetMap); grid fallback if offline |
        """
    )

    # ── 5.1  Location — interactive folium map ──────────────────────────────
    st.subheader("5.1  Your Location")
    st.markdown(
        "**Click anywhere on the map** to set your starting point, "
        "or pick a neighbourhood from the dropdown."
    )

    _nb_names = list(NYC_NEIGHBORHOODS.keys())
    sel_nb_p5 = st.selectbox(
        "Quick-jump to a neighbourhood",
        _nb_names, index=0, key="p5_neighborhood",
    )
    _default_lat, _default_lng = NYC_NEIGHBORHOODS[sel_nb_p5]

    # Determine current user location (click overrides dropdown)
    _p5_lat = st.session_state.get("p5_click_lat", _default_lat)
    _p5_lng = st.session_state.get("p5_click_lng", _default_lng)

    # Update to dropdown value when neighbourhood changes (but not on click)
    if st.session_state.get("p5_last_nb") != sel_nb_p5:
        _p5_lat = _default_lat
        _p5_lng = _default_lng
        st.session_state["p5_click_lat"] = _p5_lat
        st.session_state["p5_click_lng"] = _p5_lng
        st.session_state["p5_last_nb"]   = sel_nb_p5

    # ── Input map ─────────────────────────────────────────────────────────
    _input_map = folium.Map(
        location=[_p5_lat, _p5_lng],
        zoom_start=14,
        tiles="OpenStreetMap",
        width="100%",
    )
    # Current location marker
    folium.Marker(
        [_p5_lat, _p5_lng],
        tooltip="Your location (click map to move)",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(_input_map)
    # NYC boundary rect for orientation
    folium.Rectangle(
        bounds=[[LAT_MIN, LNG_MIN], [LAT_MAX, LNG_MAX]],
        color="#4C8BF5", fill=False, weight=1, dash_array="6",
        tooltip="NYC bounding box",
    ).add_to(_input_map)

    _map_data = st_folium(
        _input_map, width="100%", height=380, key="p5_input_map",
        returned_objects=["last_clicked"],
    )
    if _map_data and _map_data.get("last_clicked"):
        _clicked = _map_data["last_clicked"]
        _new_lat = round(float(_clicked["lat"]), 6)
        _new_lng = round(float(_clicked["lng"]), 6)
        # Only update if click is inside NYC bounding box
        if LAT_MIN <= _new_lat <= LAT_MAX and LNG_MIN <= _new_lng <= LNG_MAX:
            st.session_state["p5_click_lat"] = _new_lat
            st.session_state["p5_click_lng"] = _new_lng
            _p5_lat, _p5_lng = _new_lat, _new_lng
        else:
            st.warning("Click is outside the NYC bounding box — please click within NYC.")

    st.caption(
        f"Starting location: **{_p5_lat:.5f}°N, {abs(_p5_lng):.5f}°W** "
        f"— click the map above to change"
    )

    # ── 5.2  Meal preference ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("5.2  What do you feel like eating?  *(optional)*")

    _col_meal, _col_imp = st.columns([3, 2])
    with _col_meal:
        meal_desc_p5 = st.text_input(
            "Describe your meal craving",
            key="p5_meal_desc",
            placeholder="e.g. spicy ramen, wood-fired pizza, dim sum, tacos …",
            help="sklearn TF-IDF + cosine similarity matches this to cuisine labels.",
        )
    with _col_imp:
        cuisine_imp_p5 = st.slider(
            "Cuisine filter strictness",
            0, 100, 70, step=10, key="p5_cuisine_imp",
            help=(
                "0% — all restaurants count equally (description ignored for filtering).  \n"
                "100% — only the single best-matching cuisine type counts.  \n"
                "70% is a good default: strict enough to route to matching clusters, "
                "loose enough to find nearby options."
            ),
        )
    if meal_desc_p5.strip():
        st.caption(
            "Cuisine match preview (requires training — shown after **Find Route** is clicked)."
        )

    # ── 5.3  Hyperparameters ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("5.3  Route Hyperparameters")

    _preset_keys = list(WALK_PRESETS.keys())
    _col_walk, _col_danger = st.columns([3, 2])

    with _col_walk:
        walk_preset_p5 = st.select_slider(
            "Maximum walking budget",
            options=_preset_keys,
            value="15 min (~1.1 km)",
            key="p5_walk_preset",
            help=(
                "Sets the RL discount factor γ and max path steps to match a "
                "realistic walking radius.  Routes will never exceed this budget.  \n"
                "**5 min** ≈ 375 m • **10 min** ≈ 750 m • **20 min** ≈ 1.5 km"
            ),
        )
        _pr = WALK_PRESETS[walk_preset_p5]
        st.caption(
            f"γ = {_pr['gamma']} · max {_pr['max_steps']} hops · "
            f"proximity σ = {_pr['sigma_km']} km · "
            f"~{_pr['walk_km']*1000:.0f} m walking radius"
        )

    with _col_danger:
        danger_p5 = st.checkbox(
            "Exclude Grade C restaurants",
            value=True, key="p5_danger",
            help="Grade C restaurants contribute 0 to reward. "
                 "They appear in a warning list in the results.",
        )
        use_osrm_p5 = st.checkbox(
            "Use real street routing (OSRM)",
            value=True, key="p5_osrm",
            help="Calls the free OSRM pedestrian API for street-following routes. "
                 "Disable if offline — falls back to grid approximation.",
        )

    # ── 5.4  Find Route ────────────────────────────────────────────────────
    st.markdown("---")
    _run_p5 = st.button(
        "Find Safe Restaurant Route",
        type="primary", key="p5_run_btn",
    )

    if _run_p5:
        _rdf = _load_restaurants()
        with st.spinner("Running Value Iteration and fetching street route…"):
            _p5_result = find_safe_route(
                _rdf,
                user_lat=_p5_lat,
                user_lng=_p5_lng,
                meal_description=meal_desc_p5,
                walk_preset_key=walk_preset_p5,
                cuisine_importance=cuisine_imp_p5 / 100.0,
                danger_filter=danger_p5,
                destination_radius_cells=2,
                smooth_sigma=1.0,
                use_osrm=use_osrm_p5,
            )
        st.session_state["p5_result"]     = _p5_result
        st.session_state["p5_res_lat"]    = _p5_lat
        st.session_state["p5_res_lng"]    = _p5_lng
        st.session_state["p5_res_preset"] = walk_preset_p5

    # ── 5.5  Results ───────────────────────────────────────────────────────
    _p5r: RouteResult | None = st.session_state.get("p5_result")

    if _p5r is not None:
        _res_lat = st.session_state.get("p5_res_lat", _p5_lat)
        _res_lng = st.session_state.get("p5_res_lng", _p5_lng)

        st.markdown("---")
        st.subheader("5.5  Results")

        # ── Top-level metrics ────────────────────────────────────────────
        _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
        _mc1.metric("Walk distance",    f"{_p5r.path_distance_km:.2f} km")
        _mc2.metric("Est. walk time",   f"{_p5r.walk_minutes:.0f} min")
        _mc3.metric("Restaurants found", f"{_p5r.n_destination_restaurants:,}")
        _mean_g = _p5r.mean_destination_safety
        _mc4.metric(
            "Avg safety grade",
            f"{'A' if _mean_g >= 2.7 else 'B' if _mean_g >= 1.7 else 'C'} ({_mean_g:.1f})",
        )
        _route_src = "OSRM (real streets)" if _p5r.street_route else "Grid approx."
        _mc5.metric("Route source", _route_src)

        if _p5r.top_matches:
            _top_m_str = " · ".join(f"**{c}** ({s:.2f})" for c, s in _p5r.top_matches[:3])
            st.caption(f"Cuisine matches: {_top_m_str}")

        # ── Route map (Folium) ────────────────────────────────────────────
        st.markdown("#### Route — walking path on real NYC streets")

        # Choose route to display: OSRM real streets, or grid fallback
        _display_route = _p5r.street_route if _p5r.street_route else _p5r.grid_route

        # Centre map between start and destination
        _ctr_lat = (_res_lat + _p5r.destination_lat) / 2
        _ctr_lng = (_res_lng + _p5r.destination_lng) / 2
        _span    = max(
            abs(_res_lat - _p5r.destination_lat),
            abs(_res_lng - _p5r.destination_lng) * 0.75,
            0.005,
        )
        _zoom    = max(13, int(14 - _span * 120))

        _rmap = folium.Map(
            location=[_ctr_lat, _ctr_lng],
            zoom_start=_zoom,
            tiles="OpenStreetMap",
            width="100%",
        )

        # 1. Walking route polyline
        if len(_display_route) >= 2:
            folium.PolyLine(
                locations=_display_route,
                color="#FF8C00",
                weight=5,
                opacity=0.95,
                tooltip="Walking route",
            ).add_to(_rmap)

        # 2. Start marker
        folium.Marker(
            location=[_res_lat, _res_lng],
            tooltip="Your starting location",
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(_rmap)

        # 3. Destination marker
        folium.Marker(
            location=[_p5r.destination_lat, _p5r.destination_lng],
            tooltip=(
                f"Destination cluster — "
                f"{_p5r.n_destination_restaurants} restaurants nearby"
            ),
            icon=folium.Icon(color="purple", icon="flag", prefix="fa"),
        ).add_to(_rmap)

        # 4. Destination restaurants (clustered for performance)
        _dest_df = _p5r.destination_restaurants
        if not _dest_df.empty:
            _rest_cluster = MarkerCluster(
                name="Restaurants at destination",
                show=True,
            ).add_to(_rmap)

            _grade_icon_color = {"A": "green", "B": "orange", "C": "red"}

            for _, _row in _dest_df.dropna(subset=["latitude","longitude"]).iterrows():
                _g    = str(_row.get("latest_grade", ""))
                _icol = _grade_icon_color.get(_g, "gray")
                _score_str = (
                    f"Score: {_row['latest_score']:.0f}"
                    if pd.notna(_row.get("latest_score")) else "Score: —"
                )
                _addr = " ".join(filter(None, [
                    str(_row.get("building", "") or ""),
                    str(_row.get("street", "") or ""),
                ]))
                folium.Marker(
                    location=[float(_row["latitude"]), float(_row["longitude"])],
                    tooltip=folium.Tooltip(
                        f"<b>{_row.get('dba','?')}</b><br>"
                        f"Cuisine: {_row.get('cuisine','—')}<br>"
                        f"Grade: <b>{_g or '—'}</b>  {_score_str}<br>"
                        f"{_addr}"
                    ),
                    icon=folium.Icon(color=_icol, icon="cutlery", prefix="fa"),
                ).add_to(_rest_cluster)

        # 5. Terrible restaurants (red warning markers)
        if danger_p5 and not _p5r.terrible_restaurants.empty:
            _terr_fg = folium.FeatureGroup(name="Grade C (filtered)", show=True)
            for _, _row in _p5r.terrible_restaurants.dropna(
                subset=["latitude","longitude"]
            ).iterrows():
                folium.CircleMarker(
                    location=[float(_row["latitude"]), float(_row["longitude"])],
                    radius=5,
                    color="#CC0000",
                    fill=True,
                    fill_opacity=0.6,
                    tooltip=folium.Tooltip(
                        f"<b>{_row.get('dba','?')}</b><br>"
                        f"Grade C — excluded from reward<br>"
                        f"Score: {_row.get('latest_score','—')}"
                    ),
                ).add_to(_terr_fg)
            _terr_fg.add_to(_rmap)

        folium.LayerControl(collapsed=False).add_to(_rmap)
        st_folium(_rmap, width="100%", height=600, key="p5_result_map",
                  returned_objects=[])

        # Map legend
        _lcols = st.columns(5)
        for _lc, (_label, _color) in zip(_lcols, [
            ("Start",           "#1E82FF"),
            ("Destination",     "#9933CC"),
            ("Walk route",      "#FF8C00"),
            ("Grade A/B rest.", "#228B22"),
            ("Grade C (nearby)","#CC0000"),
        ]):
            _lc.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{_color};border-radius:50%;margin-right:5px'></span>"
                f"<small>{_label}</small>",
                unsafe_allow_html=True,
            )

        # ── Cuisine match breakdown ──────────────────────────────────────
        if _p5r.top_matches:
            st.markdown("---")
            st.subheader("5.6  Cuisine Match Breakdown")
            st.caption(
                f"TF-IDF cosine similarity: *\"{meal_desc_p5}\"* vs. each cuisine label. "
                f"Filter strictness = {cuisine_imp_p5}%"
            )
            _tm_df = pd.DataFrame(_p5r.top_matches, columns=["Cuisine", "Match Score"])
            _fig_tm = px.bar(
                _tm_df, x="Match Score", y="Cuisine", orientation="h",
                color="Match Score", color_continuous_scale="Teal",
                title="Top-5 cuisine matches for your description",
                height=280,
            )
            _fig_tm.update_layout(
                coloraxis_showscale=False, yaxis_title="",
                margin=dict(l=160, t=50, b=10),
                xaxis=dict(range=[0, 1.05]),
            )
            st.plotly_chart(_fig_tm, use_container_width=True)

        # ── RL Value-function heatmap ────────────────────────────────────
        st.markdown("---")
        st.subheader("5.7  RL Value Function — NYC Heatmap")
        st.markdown(
            "The **converged V(s)** for every grid cell.  Brighter = higher discounted "
            "future return from that cell.  The Gaussian proximity bias means the "
            "hot-spot is near your starting location.  Your route traces the "
            "gradient toward the nearest bright cluster."
        )

        # Build heatmap data from value_map (downsample stride=2)
        _V  = _p5r.value_map
        _Vn = (_V - _V.min()) / (_V.max() - _V.min() + 1e-8)
        _hm_data: list = []
        _STRIDE = 2
        for _ri in range(0, GRID_ROWS, _STRIDE):
            for _ci in range(0, GRID_COLS, _STRIDE):
                _v = float(_Vn[_ri, _ci])
                if _v > 0.08:
                    _hl, _hw = cell_to_latlng(_ri, _ci)
                    _hm_data.append([_hl, _hw, _v])

        _vmap = folium.Map(
            location=[40.710, -73.985],
            zoom_start=11,
            tiles="CartoDB dark_matter",
        )
        HeatMap(
            _hm_data,
            min_opacity=0.25,
            radius=16,
            blur=12,
            gradient={0.2: "#0000ff", 0.5: "#00ffff", 0.75: "#ffff00", 1.0: "#ff0000"},
        ).add_to(_vmap)

        # Overlay route + start
        if len(_display_route) >= 2:
            folium.PolyLine(_display_route, color="#FFFF00", weight=4,
                            opacity=0.9).add_to(_vmap)
        folium.CircleMarker(
            [_res_lat, _res_lng], radius=8,
            color="#1E82FF", fill=True, fill_color="#1E82FF",
            tooltip="Your starting location",
        ).add_to(_vmap)
        folium.CircleMarker(
            [_p5r.destination_lat, _p5r.destination_lng], radius=8,
            color="#CC44FF", fill=True, fill_color="#CC44FF",
            tooltip="Destination cluster",
        ).add_to(_vmap)

        st_folium(_vmap, width="100%", height=480, key="p5_vmap",
                  returned_objects=[])

        # ── Destination restaurants table ────────────────────────────────
        st.markdown("---")
        st.subheader("5.8  Recommended Restaurants at Destination")
        st.caption(
            f"{_p5r.n_destination_restaurants:,} restaurants within ~{2 * 0.43:.1f} km "
            f"of destination  ·  "
            f"({_p5r.destination_lat:.4f}°N, {abs(_p5r.destination_lng):.4f}°W)"
        )

        if not _dest_df.empty:
            _show_cols = ["dba", "cuisine", "latest_grade", "latest_score",
                          "boro", "building", "street"]
            _show_cols = [c for c in _show_cols if c in _dest_df.columns]
            _disp = _dest_df[_show_cols].copy()

            if "latest_grade" in _disp.columns:
                _go = {"A": 0, "B": 1, "C": 2}
                _disp["_gs"] = _disp["latest_grade"].map(_go).fillna(3)
                _disp = _disp.sort_values(["_gs", "latest_score"],
                                          ascending=[True, True]).drop(columns=["_gs"])

            st.dataframe(_disp.head(20).reset_index(drop=True),
                         use_container_width=True, height=380)

            _gc = _dest_df["latest_grade"].fillna("No grade").value_counts().reset_index()
            _gc.columns = ["grade", "count"]
            _fig_gc = px.pie(
                _gc, names="grade", values="count", color="grade",
                color_discrete_map={**GRADE_COLOR_HEX, "No grade": "#9E9E9E"},
                title="Grade distribution at destination cluster",
                hole=0.4, height=280,
            )
            _fig_gc.update_layout(margin=dict(t=50, b=10))
            st.plotly_chart(_fig_gc, use_container_width=True)

        # ── Terrible restaurants warning table ───────────────────────────
        if danger_p5 and not _p5r.terrible_restaurants.empty:
            st.markdown("---")
            st.subheader("5.9  Grade C Restaurants Near Your Route")
            st.warning(
                f"**{len(_p5r.terrible_restaurants)}** Grade-C restaurants are within "
                "~2 km of your route.  These were **excluded** from the RL reward — "
                "the agent actively routed away from them."
            )
            _tc = [c for c in ["dba", "cuisine", "latest_grade", "latest_score",
                                "boro", "building", "street"]
                   if c in _p5r.terrible_restaurants.columns]
            st.dataframe(
                _p5r.terrible_restaurants[_tc]
                .sort_values("latest_score", ascending=False)
                .reset_index(drop=True),
                use_container_width=True, height=280,
            )

    else:
        st.info(
            "Set your location, optionally describe what you want to eat, "
            "adjust the walk budget, then click **Find Safe Restaurant Route**."
        )


# ===========================================================================
# PART 6 — ULTIMATE RESTAURANT FINDER
# ===========================================================================
from src.combined_model import CombinedGradePredictor, CLASSES as GRADE_CLASSES
from src.ultimate_finder import (
    RestaurantEmbedder,
    find_area_route,
    rank_restaurants_direct,
    get_direct_route_to_restaurant,
    AreaRouteResult,
    DirectRouteResult,
)
from src.rl_route_finder import WALK_PRESETS as _WALK_PRESETS_P6


@st.cache_resource(show_spinner=False)
def _load_combined_model():
    """Train the combined KNN+DT model once with pre-tuned best parameters."""
    rdf      = _load_restaurants()
    model    = CombinedGradePredictor(knn_weight=0.35, k=7, dt_max_depth=12)
    model.fit(rdf)
    scores   = model.predict_safety_scores(rdf)
    proba_df = model.predict_proba_df(rdf)
    return model, scores, proba_df, model.n_train


with tab6:
    st.header("Part 6: Ultimate Restaurant Finder")
    st.markdown(
        """
Everything from Parts 1–5 combined into one powerful finder.

| Upgrade | What changed |
|---|---|
| **Combined grade score** | KNN + Decision Tree predictions are blended into a continuous safety score (1–3) that replaces the binary A→3 / B→2 / C→1 look-up used in Part 5 |
| **Restaurant-level NLP** | User input is embedded against a TF-IDF **restaurant matrix** (name + cuisine + borough); supports dish descriptions *and* "find something like [restaurant name]" |
| **Dual navigation** | **Area mode** — RL routes to the best cluster (like Part 5) · **Direct mode** — route straight to the top-ranked restaurant; advance through the list with Next/Previous |
        """
    )

    # ── 6.1  Combined Grade Predictor (auto-loads with best parameters) ─────
    st.markdown("---")
    st.subheader("6.1  Combined Grade Predictor")
    st.markdown(
        """
The combined model blends **KNN** and **Decision Tree** grade predictions into a
continuous safety score (1–3) used as the RL reward signal — no configuration needed.

The best parameters were determined by cross-validated tuning and are applied automatically:

| Parameter | Best value | What it controls |
|---|---|---|
| KNN blend weight α | **0.35** | KNN contributes 35%, Decision Tree 65% — balances KNN's A-bias with DT's minority-class recovery |
| KNN k | **7** | Neighbours considered per prediction |
| DT max depth | **12** | Tree complexity — enough to capture patterns without overfitting |
        """
    )

    with st.spinner("Loading combined KNN + Decision Tree model… (first visit only, cached afterwards)"):
        _p6_model, _p6_scores, _p6_proba_df, _p6_n_train = _load_combined_model()

    _p6_m  = _p6_model
    _p6_sc = _p6_scores
    _p6_pd = _p6_proba_df

    st.success(
        f"Combined model ready — trained on **{_p6_n_train:,}** graded restaurants  "
        f"·  KNN weight α = {_p6_m.knn_weight}  ·  DT weight = {_p6_m.dt_weight:.2f}"
    )

    _d1, _d2, _d3, _d4 = st.columns(4)
    _mean_sc = float(np.nanmean(_p6_sc))
    _d1.metric("Mean safety score", f"{_mean_sc:.3f}")
    _pred_a  = int(np.sum(_p6_pd["combined_pred"] == "A"))
    _pred_b  = int(np.sum(_p6_pd["combined_pred"] == "B"))
    _pred_c  = int(np.sum(_p6_pd["combined_pred"] == "C"))
    _d2.metric("Predicted A", f"{_pred_a:,}")
    _d3.metric("Predicted B", f"{_pred_b:,}")
    _d4.metric("Predicted C", f"{_pred_c:,}")

    with st.expander("How does blending KNN and Decision Tree help? (click to expand)"):
        _agree_knn = int(np.sum(_p6_pd["combined_pred"] == _p6_pd["knn_pred"]))
        _agree_dt  = int(np.sum(_p6_pd["combined_pred"] == _p6_pd["dt_pred"]))
        _agree_both= int(np.sum(
            (_p6_pd["combined_pred"] == _p6_pd["knn_pred"]) &
            (_p6_pd["combined_pred"] == _p6_pd["dt_pred"])
        ))
        _n_pd = len(_p6_pd)

        st.markdown(
            "KNN alone skews almost everything to grade A (the majority class). "
            "Decision Tree recovers B and C predictions better. "
            "Blending the two gives a more balanced, calibrated result."
        )

        _ca, _cb, _cc = st.columns(3)
        _ca.metric("Combined agrees with KNN", f"{_agree_knn/_n_pd:.1%}")
        _cb.metric("Combined agrees with DT",  f"{_agree_dt/_n_pd:.1%}")
        _cc.metric("All three agree",           f"{_agree_both/_n_pd:.1%}")

        st.markdown("**Predicted grade distribution — KNN vs DT vs Combined**")
        _dist_rows = []
        for _src, _col in [("KNN alone","knn_pred"),("DT alone","dt_pred"),
                            ("Combined","combined_pred")]:
            for _g in GRADE_CLASSES:
                _dist_rows.append({
                    "Model": _src,
                    "Grade": _g,
                    "Count": int(np.sum(_p6_pd[_col] == _g)),
                })
        _dist_df = pd.DataFrame(_dist_rows)
        _fig_dist = px.bar(
            _dist_df, x="Grade", y="Count", color="Model", barmode="group",
            color_discrete_sequence=["#4C8BF5", "#F5844C", "#2ECC71"],
            title="Grade distribution: KNN vs DT vs Combined",
            height=320,
        )
        _fig_dist.update_layout(margin=dict(t=50, b=20))
        st.plotly_chart(_fig_dist, use_container_width=True)

        st.markdown("**Decision Tree feature importances**")
        _fi_dict = _p6_m.dt_feature_importances
        if _fi_dict:
            _fi_df = pd.DataFrame(
                sorted(_fi_dict.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"],
            )
            _fig_fi = px.bar(
                _fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Teal",
                height=max(280, len(_fi_df) * 28),
            )
            _fig_fi.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=180, t=20, b=10),
            )
            st.plotly_chart(_fig_fi, use_container_width=True)

    st.markdown("**Safety score distribution (all restaurants)**")
    _sc_series = pd.Series(_p6_sc, name="safety_score")
    _fig_sc = px.histogram(
        _sc_series.dropna(), nbins=40,
        title="Predicted safety score distribution (1=C · 2=B · 3=A)",
        labels={"value": "Safety score", "count": "Restaurants"},
        color_discrete_sequence=["#4C8BF5"],
        height=280,
    )
    _fig_sc.add_vline(x=1.5, line_dash="dash", line_color=GRADE_COLOR_HEX["C"],
                      annotation_text="C/B boundary")
    _fig_sc.add_vline(x=2.5, line_dash="dash", line_color=GRADE_COLOR_HEX["B"],
                      annotation_text="B/A boundary")
    _fig_sc.update_layout(margin=dict(t=50, b=20))
    st.plotly_chart(_fig_sc, use_container_width=True)

    # ── 6.2  Location ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("6.2  Your Starting Location")
    st.markdown(
        "**Click the map** to set your starting point, or pick a neighbourhood."
    )

    _nb_names_p6 = list(NYC_NEIGHBORHOODS.keys())
    _sel_nb_p6   = st.selectbox(
        "Quick-jump to a neighbourhood",
        _nb_names_p6, index=0, key="p6_neighborhood",
    )
    _def_lat_p6, _def_lng_p6 = NYC_NEIGHBORHOODS[_sel_nb_p6]

    _p6_lat = st.session_state.get("p6_click_lat", _def_lat_p6)
    _p6_lng = st.session_state.get("p6_click_lng", _def_lng_p6)

    if st.session_state.get("p6_last_nb") != _sel_nb_p6:
        _p6_lat, _p6_lng = _def_lat_p6, _def_lng_p6
        st.session_state["p6_click_lat"] = _p6_lat
        st.session_state["p6_click_lng"] = _p6_lng
        st.session_state["p6_last_nb"]   = _sel_nb_p6

    _p6_input_map = folium.Map(
        location=[_p6_lat, _p6_lng],
        zoom_start=14,
        tiles="OpenStreetMap",
        width="100%",
    )
    folium.Marker(
        [_p6_lat, _p6_lng],
        tooltip="Your location (click map to move)",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(_p6_input_map)
    folium.Rectangle(
        bounds=[[LAT_MIN, LNG_MIN], [LAT_MAX, LNG_MAX]],
        color="#4C8BF5", fill=False, weight=1, dash_array="6",
    ).add_to(_p6_input_map)

    _p6_map_data = st_folium(
        _p6_input_map, width="100%", height=340, key="p6_input_map",
        returned_objects=["last_clicked"],
    )
    if _p6_map_data and _p6_map_data.get("last_clicked"):
        _cl = _p6_map_data["last_clicked"]
        _nl, _nw = round(float(_cl["lat"]), 6), round(float(_cl["lng"]), 6)
        if LAT_MIN <= _nl <= LAT_MAX and LNG_MIN <= _nw <= LNG_MAX:
            st.session_state["p6_click_lat"] = _nl
            st.session_state["p6_click_lng"] = _nw
            _p6_lat, _p6_lng = _nl, _nw
        else:
            st.warning("Click is outside the NYC bounding box.")

    st.caption(
        f"Starting: **{_p6_lat:.5f}°N, {abs(_p6_lng):.5f}°W** — click map to change"
    )

    # ── 6.3  Natural Language Query ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("6.3  What Are You Looking For?")
    st.markdown(
        """
Enter **any** of the following — the system embeds your text into the
restaurant matrix and finds the best matches:
- A dish or food type:  *"spicy ramen"*, *"wood-fired pizza"*, *"dim sum"*
- A cuisine style:  *"Korean BBQ"*, *"Mediterranean"*
- A reference restaurant:  *"something like Nobu"*, *"similar to Peter Luger"*
- Combined:  *"cozy Japanese noodles like Ippudo"*
        """
    )

    _p6_query = st.text_input(
        "Food / dish / restaurant description",
        key="p6_query",
        placeholder="e.g. crispy pork bao, wood-fired Neapolitan pizza, find me something like Shake Shack…",
    )

    if _p6_query.strip():
        # Quick preview: show detected reference + top cuisine matches
        _p6_prev_emb = RestaurantEmbedder().fit(_load_restaurants())
        _p6_ref      = _p6_prev_emb.detected_reference(_p6_query)
        _p6_prev_top = _p6_prev_emb.top_matches(_p6_query, n=5)
        if _p6_ref:
            st.info(
                f"Restaurant detected in query: **{_p6_ref}** — "
                "similar restaurants will be prioritised."
            )
        if _p6_prev_top:
            _prev_str = " · ".join(
                f"**{c}** ({s:.2f})" for c, s in _p6_prev_top[:3]
            )
            st.caption(f"Top cuisine matches: {_prev_str}")

    # ── 6.4  Navigation Mode ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("6.4  Navigation Mode")

    _p6_mode = st.radio(
        "How do you want to navigate?",
        ["Area Finder — walk to the best restaurant cluster (RL)",
         "Direct Route — go straight to a specific restaurant"],
        index=0, key="p6_mode",
        help=(
            "**Area Finder**: Value Iteration plans a walking route to the "
            "densest cluster of safe, matching restaurants — best when you're "
            "exploring and happy to choose once you arrive.\n\n"
            "**Direct Route**: Ranks every nearby restaurant by "
            "(safety × NLP match × proximity) and routes you straight there. "
            "Use Next/Previous to browse ranked options."
        ),
    )
    _p6_use_area = _p6_mode.startswith("Area")

    # ── 6.5  Parameters ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("6.5  Route Parameters")

    _p6_pk_keys = list(_WALK_PRESETS_P6.keys())
    _p6_col_walk, _p6_col_opts = st.columns([3, 2])

    with _p6_col_walk:
        _p6_walk_preset = st.select_slider(
            "Maximum walking budget",
            options=_p6_pk_keys,
            value="15 min (~1.1 km)",
            key="p6_walk_preset",
        )
        _pr6 = _WALK_PRESETS_P6[_p6_walk_preset]
        st.caption(
            f"γ = {_pr6['gamma']} · max {_pr6['max_steps']} hops · "
            f"proximity σ = {_pr6['sigma_km']} km · "
            f"~{_pr6['walk_km']*1000:.0f} m radius"
        )

    with _p6_col_opts:
        _p6_danger   = st.checkbox("Exclude Grade C restaurants", value=True,  key="p6_danger")
        _p6_osrm     = st.checkbox("Use real street routing (OSRM)", value=True, key="p6_osrm")

    if _p6_use_area:
        _p6_cui_imp = st.slider(
            "Cuisine/NLP filter strictness",
            0, 100, 70, step=10, key="p6_cui_imp",
            help="0 % = no filter · 100 % = only best-matching type counts",
        )
    else:
        _p6_cui_imp = 0   # not used in direct mode

    # ── 6.6  Find! ───────────────────────────────────────────────────────────
    st.markdown("---")
    _p6_run_btn = st.button(
        "Find Ultimate Route",
        type="primary", key="p6_run_btn",
    )

    if _p6_run_btn:
        _rdf_run = _load_restaurants()
        _sc_run  = _p6_scores

        if _p6_use_area:
            with st.spinner("Running Value Iteration with combined model scores…"):
                _p6_area_result = find_area_route(
                    df              = _rdf_run,
                    safety_scores   = _sc_run,
                    user_lat        = _p6_lat,
                    user_lng        = _p6_lng,
                    query           = _p6_query,
                    walk_preset_key = _p6_walk_preset,
                    cuisine_importance = _p6_cui_imp / 100.0,
                    danger_filter   = _p6_danger,
                    smooth_sigma    = 1.0,
                    use_osrm        = _p6_osrm,
                )
            st.session_state["p6_area_result"] = _p6_area_result
            st.session_state["p6_res_lat"]     = _p6_lat
            st.session_state["p6_res_lng"]     = _p6_lng
            # Clear any old direct result
            st.session_state.pop("p6_direct_result", None)

        else:
            with st.spinner("Ranking restaurants and computing route…"):
                _ranked, _top_m, _det_ref = rank_restaurants_direct(
                    df              = _rdf_run,
                    safety_scores   = _sc_run,
                    user_lat        = _p6_lat,
                    user_lng        = _p6_lng,
                    query           = _p6_query,
                    walk_preset_key = _p6_walk_preset,
                    danger_filter   = _p6_danger,
                    top_n           = 30,
                )
            st.session_state["p6_direct_ranked"]  = _ranked
            st.session_state["p6_direct_top_m"]   = _top_m
            st.session_state["p6_direct_det_ref"]  = _det_ref
            st.session_state["p6_direct_idx"]      = 0
            st.session_state["p6_res_lat"]         = _p6_lat
            st.session_state["p6_res_lng"]         = _p6_lng
            # Clear any old area result
            st.session_state.pop("p6_area_result", None)

    # ── 6.7  Results — Area Mode ──────────────────────────────────────────────
    _p6_ar: AreaRouteResult | None = st.session_state.get("p6_area_result")

    if _p6_ar is not None:
        _p6_rl  = st.session_state.get("p6_res_lat", _p6_lat)
        _p6_rw  = st.session_state.get("p6_res_lng", _p6_lng)

        st.markdown("---")
        st.subheader("6.7  Area Mode Results")

        # Stats
        _am1, _am2, _am3, _am4, _am5 = st.columns(5)
        _am1.metric("Walk distance",    f"{_p6_ar.path_distance_km:.2f} km")
        _am2.metric("Est. walk time",   f"{_p6_ar.walk_minutes:.0f} min")
        _am3.metric("Restaurants found", f"{_p6_ar.n_destination_restaurants:,}")
        _mean_s = _p6_ar.mean_destination_safety
        _am4.metric(
            "Avg predicted safety",
            f"{'A' if _mean_s >= 2.5 else 'B' if _mean_s >= 1.5 else 'C'} ({_mean_s:.2f})",
            help="Based on combined KNN+DT predicted score (not actual grade)",
        )
        _am5.metric(
            "Route source",
            "OSRM (real streets)" if _p6_ar.street_route else "Grid approx.",
        )

        if _p6_ar.detected_ref:
            st.info(
                f"Routing toward restaurants similar to **{_p6_ar.detected_ref}** "
                f"detected in your query."
            )
        if _p6_ar.top_matches:
            _tm_str = " · ".join(
                f"**{c}** ({s:.2f})" for c, s in _p6_ar.top_matches[:3]
            )
            st.caption(f"Cuisine matches: {_tm_str}")

        # Route map
        st.markdown("#### Walking route to best cluster")
        _p6_disp = _p6_ar.street_route if _p6_ar.street_route else _p6_ar.grid_route
        _p6_ctr_lat = (_p6_rl + _p6_ar.destination_lat) / 2
        _p6_ctr_lng = (_p6_rw + _p6_ar.destination_lng) / 2
        _p6_span    = max(abs(_p6_rl - _p6_ar.destination_lat),
                          abs(_p6_rw - _p6_ar.destination_lng) * 0.75, 0.005)
        _p6_zoom    = max(13, int(14 - _p6_span * 120))

        _p6_rmap = folium.Map(
            location=[_p6_ctr_lat, _p6_ctr_lng],
            zoom_start=_p6_zoom,
            tiles="OpenStreetMap",
        )

        if len(_p6_disp) >= 2:
            folium.PolyLine(
                _p6_disp, color="#FF8C00", weight=5, opacity=0.95,
                tooltip="Walking route",
            ).add_to(_p6_rmap)

        folium.Marker(
            [_p6_rl, _p6_rw],
            tooltip="Your starting location",
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(_p6_rmap)

        folium.Marker(
            [_p6_ar.destination_lat, _p6_ar.destination_lng],
            tooltip=f"Destination cluster — {_p6_ar.n_destination_restaurants} restaurants",
            icon=folium.Icon(color="purple", icon="flag", prefix="fa"),
        ).add_to(_p6_rmap)

        # Destination restaurants
        _p6_dest_df = _p6_ar.destination_restaurants
        if not _p6_dest_df.empty:
            _p6_cluster = MarkerCluster(name="Destination restaurants").add_to(_p6_rmap)
            _g_icol = {"A": "green", "B": "orange", "C": "red"}
            for _, _row in _p6_dest_df.dropna(subset=["latitude", "longitude"]).iterrows():
                _g    = str(_row.get("latest_grade", ""))
                _sc   = _row.get("safety", 2.0)
                _addr = " ".join(filter(None, [
                    str(_row.get("building", "") or ""),
                    str(_row.get("street", "") or ""),
                ]))
                folium.Marker(
                    location=[float(_row["latitude"]), float(_row["longitude"])],
                    tooltip=folium.Tooltip(
                        f"<b>{_row.get('dba','?')}</b><br>"
                        f"Cuisine: {_row.get('cuisine','—')}<br>"
                        f"Actual grade: <b>{_g or '—'}</b><br>"
                        f"Predicted safety: {_sc:.2f}/3.0<br>"
                        f"{_addr}"
                    ),
                    icon=folium.Icon(color=_g_icol.get(_g, "gray"),
                                     icon="cutlery", prefix="fa"),
                ).add_to(_p6_cluster)

        # Grade-C warning markers
        if _p6_danger and not _p6_ar.terrible_restaurants.empty:
            _p6_terr_fg = folium.FeatureGroup(name="Grade C (filtered)", show=True)
            for _, _row in _p6_ar.terrible_restaurants.dropna(
                subset=["latitude", "longitude"]
            ).iterrows():
                folium.CircleMarker(
                    location=[float(_row["latitude"]), float(_row["longitude"])],
                    radius=5, color="#CC0000", fill=True, fill_opacity=0.6,
                    tooltip=folium.Tooltip(
                        f"<b>{_row.get('dba','?')}</b><br>"
                        f"Grade C — excluded from reward<br>"
                        f"Score: {_row.get('latest_score','—')}"
                    ),
                ).add_to(_p6_terr_fg)
            _p6_terr_fg.add_to(_p6_rmap)

        folium.LayerControl(collapsed=False).add_to(_p6_rmap)
        st_folium(_p6_rmap, width="100%", height=600, key="p6_area_map",
                  returned_objects=[])

        _p6_lcols = st.columns(5)
        for _lc, (_lbl, _lcl) in zip(_p6_lcols, [
            ("Start",            "#1E82FF"),
            ("Destination",      "#9933CC"),
            ("Walk route",       "#FF8C00"),
            ("Safe restaurant",  "#228B22"),
            ("Grade C (nearby)", "#CC0000"),
        ]):
            _lc.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{_lcl};border-radius:50%;margin-right:5px'></span>"
                f"<small>{_lbl}</small>",
                unsafe_allow_html=True,
            )

        # Cuisine match breakdown
        if _p6_ar.top_matches:
            st.markdown("---")
            st.subheader("6.8  NLP Embedding Match Breakdown")
            st.caption(
                "TF-IDF cosine similarity between your query and restaurant "
                f"corpus (name + cuisine + borough).  Query: *\"{_p6_query}\"*"
            )
            _p6_tm_df = pd.DataFrame(_p6_ar.top_matches, columns=["Cuisine", "Match Score"])
            _p6_fig_tm = px.bar(
                _p6_tm_df, x="Match Score", y="Cuisine", orientation="h",
                color="Match Score", color_continuous_scale="Teal",
                title="Top-5 cuisine matches for your query",
                height=260,
            )
            _p6_fig_tm.update_layout(
                coloraxis_showscale=False, yaxis_title="",
                margin=dict(l=160, t=50, b=10),
                xaxis=dict(range=[0, 1.05]),
            )
            st.plotly_chart(_p6_fig_tm, use_container_width=True)

        # RL value-function heatmap
        st.markdown("---")
        st.subheader("6.9  RL Value Function — Combined-Score Heatmap")
        st.markdown(
            "Reward grid built from **combined model safety scores × NLP match scores**. "
            "Brighter = higher discounted return.  The route traces the gradient "
            "from your location toward the peak."
        )

        _p6_V  = _p6_ar.value_map
        _p6_Vn = (_p6_V - _p6_V.min()) / (_p6_V.max() - _p6_V.min() + 1e-8)
        _p6_hm: list = []
        for _ri in range(0, GRID_ROWS, 2):
            for _ci in range(0, GRID_COLS, 2):
                _v = float(_p6_Vn[_ri, _ci])
                if _v > 0.08:
                    _hl, _hw = cell_to_latlng(_ri, _ci)
                    _p6_hm.append([_hl, _hw, _v])

        _p6_vmap = folium.Map(
            location=[40.710, -73.985], zoom_start=11,
            tiles="CartoDB dark_matter",
        )
        HeatMap(
            _p6_hm, min_opacity=0.25, radius=16, blur=12,
            gradient={0.2: "#0000ff", 0.5: "#00ffff",
                      0.75: "#ffff00", 1.0: "#ff0000"},
        ).add_to(_p6_vmap)
        if len(_p6_disp) >= 2:
            folium.PolyLine(_p6_disp, color="#FFFF00", weight=4,
                            opacity=0.9).add_to(_p6_vmap)
        folium.CircleMarker([_p6_rl, _p6_rw], radius=8,
                             color="#1E82FF", fill=True,
                             tooltip="Start").add_to(_p6_vmap)
        folium.CircleMarker(
            [_p6_ar.destination_lat, _p6_ar.destination_lng], radius=8,
            color="#CC44FF", fill=True, tooltip="Destination cluster",
        ).add_to(_p6_vmap)
        st_folium(_p6_vmap, width="100%", height=460, key="p6_vmap",
                  returned_objects=[])

        # Destination table
        st.markdown("---")
        st.subheader("6.10  Recommended Restaurants at Destination")
        if not _p6_dest_df.empty:
            _p6_show = [c for c in ["dba","cuisine","latest_grade","latest_score",
                                     "safety","boro","building","street"]
                        if c in _p6_dest_df.columns]
            _p6_disp_df = _p6_dest_df[_p6_show].copy()
            if "safety" in _p6_disp_df.columns:
                _p6_disp_df = _p6_disp_df.sort_values("safety", ascending=False)
            st.dataframe(_p6_disp_df.head(20).reset_index(drop=True),
                         use_container_width=True, height=360)

        # Terrible restaurants warning
        if _p6_danger and not _p6_ar.terrible_restaurants.empty:
            st.markdown("---")
            st.subheader("6.11  Grade C Restaurants Near Your Route")
            st.warning(
                f"**{len(_p6_ar.terrible_restaurants)}** Grade-C restaurants are "
                "within ~2 km of your route.  Their actual grade is C, so even if "
                "the combined model's predicted safety is higher, they were zeroed "
                "out of the RL reward."
            )
            _p6_tc = [c for c in ["dba","cuisine","latest_grade","latest_score",
                                    "boro","building","street"]
                      if c in _p6_ar.terrible_restaurants.columns]
            st.dataframe(
                _p6_ar.terrible_restaurants[_p6_tc]
                .sort_values("latest_score", ascending=False)
                .reset_index(drop=True),
                use_container_width=True, height=260,
            )

    # ── 6.7b  Results — Direct Mode ───────────────────────────────────────────
    _p6_ranked: pd.DataFrame | None = st.session_state.get("p6_direct_ranked")

    if _p6_ranked is not None and not _p6_use_area:
        _p6_rl  = st.session_state.get("p6_res_lat", _p6_lat)
        _p6_rw  = st.session_state.get("p6_res_lng", _p6_lng)
        _p6_dtm = st.session_state.get("p6_direct_top_m", [])
        _p6_drf = st.session_state.get("p6_direct_det_ref")
        _p6_idx = st.session_state.get("p6_direct_idx", 0)

        st.markdown("---")
        st.subheader("6.7  Direct Route Results")

        if _p6_ranked.empty:
            st.warning(
                "No restaurants found within the walking budget matching your query. "
                "Try increasing the walk budget or relaxing the query."
            )
        else:
            # Navigation controls
            _n_cands = len(_p6_ranked)
            _p6_idx  = max(0, min(_p6_idx, _n_cands - 1))

            if _p6_drf:
                st.info(
                    f"Routing to restaurants similar to **{_p6_drf}** detected in query."
                )
            if _p6_dtm:
                _dtm_str = " · ".join(f"**{c}** ({s:.2f})" for c, s in _p6_dtm[:3])
                st.caption(f"Cuisine matches: {_dtm_str}")

            st.markdown(
                f"**{_n_cands}** restaurants ranked by "
                "(combined safety score × NLP match × proximity).  "
                f"Showing **#{_p6_idx + 1}** of {_n_cands}."
            )

            _nav_c1, _nav_c2, _nav_c3 = st.columns([1, 1, 4])
            with _nav_c1:
                if st.button("← Previous", key="p6_prev_btn",
                             disabled=(_p6_idx == 0)):
                    st.session_state["p6_direct_idx"] = _p6_idx - 1
                    st.rerun()
            with _nav_c2:
                if st.button("Next →", key="p6_next_btn",
                             disabled=(_p6_idx >= _n_cands - 1)):
                    st.session_state["p6_direct_idx"] = _p6_idx + 1
                    st.rerun()
            with _nav_c3:
                _new_idx = st.number_input(
                    "Jump to restaurant #", min_value=1, max_value=_n_cands,
                    value=_p6_idx + 1, step=1, key="p6_idx_input",
                )
                if _new_idx - 1 != _p6_idx:
                    st.session_state["p6_direct_idx"] = int(_new_idx) - 1
                    st.rerun()

            # Current restaurant info card
            _curr = _p6_ranked.iloc[_p6_idx]
            _curr_grade  = str(_curr.get("latest_grade", "—") or "—")
            _curr_safety = float(_curr.get("safety", 2.0))
            _curr_nlp    = float(_curr.get("nlp_score", 1.0))
            _curr_comb   = float(_curr.get("combined_score", 0.0))
            _curr_addr   = " ".join(filter(None, [
                str(_curr.get("building", "") or ""),
                str(_curr.get("street", "") or ""),
            ]))
            _curr_lat    = float(_curr["latitude"])
            _curr_lng    = float(_curr["longitude"])

            st.markdown("---")
            _info_c1, _info_c2, _info_c3, _info_c4 = st.columns(4)
            _info_c1.markdown(
                f"<div style='background:#1a2744;border-radius:8px;padding:12px'>"
                f"<div style='font-size:0.75rem;color:#aaa'>Restaurant</div>"
                f"<div style='font-size:1.1rem;font-weight:bold'>{_curr.get('dba','?')}</div>"
                f"<div style='color:#ccc;font-size:0.85rem'>{_curr.get('cuisine','—')}</div>"
                f"<div style='color:#aaa;font-size:0.8rem;margin-top:4px'>{_curr_addr}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            _info_c2.metric(
                "Actual grade / score",
                f"{_curr_grade} / {_curr.get('latest_score', '—')}",
            )
            _info_c3.metric(
                "Predicted safety",
                f"{_curr_safety:.2f} / 3.0",
                help="Combined KNN+DT blended safety score",
            )
            _info_c4.metric(
                "NLP match × rank",
                f"{_curr_nlp:.2f}  (#{_p6_idx+1})",
                help="TF-IDF cosine similarity of your query to this restaurant",
            )

            # Compute route to current restaurant
            _curr_sr, _curr_gr, _curr_dist, _curr_walk = \
                get_direct_route_to_restaurant(
                    _p6_rl, _p6_rw, _curr_lat, _curr_lng,
                    use_osrm=_p6_osrm,
                )

            _rm1, _rm2, _rm3 = st.columns(3)
            _rm1.metric("Walk distance", f"{_curr_dist:.2f} km")
            _rm2.metric("Est. walk time", f"{_curr_walk:.0f} min")
            _rm3.metric("Route source",
                        "OSRM (real streets)" if _curr_sr else "Grid approx.")

            # Direct route map
            st.markdown("#### Route to this restaurant")
            _disp_r = _curr_sr if _curr_sr else _curr_gr

            _p6_dmap = folium.Map(
                location=[
                    (_p6_rl + _curr_lat) / 2,
                    (_p6_rw + _curr_lng) / 2,
                ],
                zoom_start=max(13, int(14 - max(
                    abs(_p6_rl - _curr_lat),
                    abs(_p6_rw - _curr_lng) * 0.75,
                    0.005,
                ) * 120)),
                tiles="OpenStreetMap",
            )

            if len(_disp_r) >= 2:
                folium.PolyLine(
                    _disp_r, color="#FF8C00", weight=5,
                    opacity=0.95, tooltip="Walking route",
                ).add_to(_p6_dmap)

            folium.Marker(
                [_p6_rl, _p6_rw],
                tooltip="Your starting location",
                icon=folium.Icon(color="blue", icon="home", prefix="fa"),
            ).add_to(_p6_dmap)

            _dest_icon_col = {"A": "green", "B": "orange", "C": "red"}.get(
                _curr_grade, "gray"
            )
            folium.Marker(
                [_curr_lat, _curr_lng],
                tooltip=folium.Tooltip(
                    f"<b>{_curr.get('dba','?')}</b><br>"
                    f"Cuisine: {_curr.get('cuisine','—')}<br>"
                    f"Grade: <b>{_curr_grade}</b>  "
                    f"Predicted safety: {_curr_safety:.2f}/3<br>"
                    f"NLP match: {_curr_nlp:.2f}  Rank: #{_p6_idx+1}<br>"
                    f"{_curr_addr}"
                ),
                icon=folium.Icon(color=_dest_icon_col, icon="cutlery", prefix="fa"),
            ).add_to(_p6_dmap)

            # Show other candidates as small circles
            _others = _p6_ranked.drop(_p6_ranked.index[_p6_idx]).dropna(
                subset=["latitude", "longitude"]
            ).head(15)
            _other_grp = folium.FeatureGroup(name="Other candidates", show=True)
            for _, _orow in _others.iterrows():
                _og = str(_orow.get("latest_grade", "") or "")
                _oc = {"A": "#228B22", "B": "#FFA500", "C": "#CC0000"}.get(_og, "#888")
                folium.CircleMarker(
                    location=[float(_orow["latitude"]), float(_orow["longitude"])],
                    radius=5, color=_oc, fill=True, fill_opacity=0.55,
                    tooltip=folium.Tooltip(
                        f"{_orow.get('dba','?')} | "
                        f"Grade {_og} | "
                        f"Safety {_orow.get('safety',0):.2f}"
                    ),
                ).add_to(_other_grp)
            _other_grp.add_to(_p6_dmap)

            folium.LayerControl(collapsed=False).add_to(_p6_dmap)
            st_folium(_p6_dmap, width="100%", height=560, key="p6_direct_map",
                      returned_objects=[])

            _p6_dlcols = st.columns(5)
            for _lc, (_lbl, _lcl) in zip(_p6_dlcols, [
                ("Start",            "#1E82FF"),
                ("Selected restaurant","#228B22"),
                ("Walk route",       "#FF8C00"),
                ("Other candidates", "#888888"),
                ("Grade A / B / C",  "#2ECC71"),
            ]):
                _lc.markdown(
                    f"<span style='display:inline-block;width:12px;height:12px;"
                    f"background:{_lcl};border-radius:50%;margin-right:5px'></span>"
                    f"<small>{_lbl}</small>",
                    unsafe_allow_html=True,
                )

            # Full ranked table
            st.markdown("---")
            st.subheader(f"6.8  All {_n_cands} Ranked Candidates")
            _p6_show_cols = [
                c for c in ["dba", "cuisine", "latest_grade", "latest_score",
                             "safety", "nlp_score", "combined_score",
                             "boro", "building", "street"]
                if c in _p6_ranked.columns
            ]
            _p6_tbl = _p6_ranked[_p6_show_cols].copy()
            if "safety" in _p6_tbl.columns:
                _p6_tbl["safety"] = _p6_tbl["safety"].round(3)
            if "nlp_score" in _p6_tbl.columns:
                _p6_tbl["nlp_score"] = _p6_tbl["nlp_score"].round(3)
            if "combined_score" in _p6_tbl.columns:
                _p6_tbl["combined_score"] = _p6_tbl["combined_score"].round(4)
            st.dataframe(_p6_tbl.reset_index(drop=True),
                         use_container_width=True, height=480)

    elif _p6_ranked is None and not _p6_use_area and "p6_run_btn" not in st.session_state:
        pass   # nothing shown before first run

    # Prompt if nothing run yet
    if _p6_ar is None and _p6_ranked is None:
        st.info(
            "Set your location and food preference above, then click "
            "**Find Ultimate Route** to see results."
        )
