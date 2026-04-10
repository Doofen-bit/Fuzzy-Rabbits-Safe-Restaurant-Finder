"""
streamlit_app.py
----------------
Interactive dashboard that shows:
  1. The three-step pipeline that converts raw DOHMH violation records into a
     restaurant-level dataset (restaurants.csv).
  2. A filterable data explorer for the resulting table.
  3. An interactive NYC map where every dot is a restaurant, coloured by its
     latest health-inspection grade.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# Make sure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
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
# Grade colour palette (used in both charts and the map)
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
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading raw inspection records (one-time, ~10 s)…")
def _load_raw() -> pd.DataFrame:
    from src.data_loader import load_raw  # noqa: PLC0415

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
# Helper: add a colour column for the map layer
# ---------------------------------------------------------------------------
def _add_map_color(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a new 'color' column (list[int] RGBA) keyed on grade."""
    df = df.copy()
    df["color"] = df["latest_grade"].map(GRADE_COLOR_RGBA).apply(
        lambda v: v if isinstance(v, list) else _NO_GRADE_RGBA
    )
    return df


# ===========================================================================
# SIDEBAR FILTERS  (shared between Data Explorer and Map)
# ===========================================================================
def _sidebar_filters(restaurants: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Borough
    boros = sorted(b for b in restaurants["boro"].dropna().unique() if b != "0")
    sel_boros = st.sidebar.multiselect("Borough", boros, default=boros)

    # Grade
    all_grades = ["A", "B", "C", "N", "Z", "—No grade—"]
    sel_grades = st.sidebar.multiselect("Latest grade", all_grades, default=["A", "B", "C"])

    # Cuisine (top-20 + free text)
    top_cuisines = (
        restaurants["cuisine"]
        .value_counts()
        .head(20)
        .index.tolist()
    )
    sel_cuisines = st.sidebar.multiselect(
        "Cuisine (top 20)", top_cuisines, default=[]
    )
    cuisine_text = st.sidebar.text_input("Or search cuisine", "")

    # Score range
    score_min = int(restaurants["latest_score"].dropna().min())
    score_max = int(restaurants["latest_score"].dropna().max())
    sel_score = st.sidebar.slider(
        "Latest score range (lower = better)",
        score_min,
        score_max,
        (score_min, min(score_max, 50)),
    )

    # --- Apply filters ---
    mask = pd.Series(True, index=restaurants.index)

    if sel_boros:
        mask &= restaurants["boro"].isin(sel_boros)

    grade_vals: list[str | float] = []
    for g in sel_grades:
        if g == "—No grade—":
            grade_vals.append(np.nan)
        else:
            grade_vals.append(g)
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
        mask &= restaurants["cuisine"].str.contains(
            cuisine_text.strip(), case=False, na=False
        )

    mask &= (
        restaurants["latest_score"].isna()
        | restaurants["latest_score"].between(sel_score[0], sel_score[1])
    )

    return restaurants[mask].copy()


# ===========================================================================
# MAIN APP
# ===========================================================================
st.title("NYC Safe Restaurant Finder")
st.markdown(
    "This dashboard shows how **295,995 raw violation records** from the NYC "
    "Department of Health are transformed into a clean **restaurant-level dataset** "
    "— and lets you explore those restaurants on an interactive map."
)

tab_pipeline, tab_explore, tab_map = st.tabs(
    ["Conversion Pipeline", "Data Explorer", "NYC Map"]
)

# Load once and apply sidebar filters once — reused by both Data Explorer and Map tabs.
_restaurants = _load_restaurants()
_filtered = _sidebar_filters(_restaurants)


# ===========================================================================
# TAB 1 — CONVERSION PIPELINE
# ===========================================================================
with tab_pipeline:
    st.header("Three-step conversion pipeline")
    st.markdown(
        "Each step below processes the data further. "
        "Click **Load raw data** to run Steps 1 and 2 live; "
        "Step 3 always reads the pre-built `restaurants.csv`."
    )

    # ── Step visual header ──────────────────────────────────────────────────
    c1, arr1, c2, arr2, c3 = st.columns([3, 0.4, 3, 0.4, 3])
    with c1:
        st.markdown(
            "<div style='background:#1e3a5f;border-radius:8px;padding:14px;"
            "text-align:center;color:white'>"
            "<b>Step 1</b><br>Load Raw CSV<br>"
            "<small>DOHMH violation records</small></div>",
            unsafe_allow_html=True,
        )
    with arr1:
        st.markdown(
            "<div style='font-size:2rem;text-align:center;margin-top:10px'>→</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='background:#1a4d2e;border-radius:8px;padding:14px;"
            "text-align:center;color:white'>"
            "<b>Step 2</b><br>Clean & Type-cast<br>"
            "<small>data_loader.load_raw()</small></div>",
            unsafe_allow_html=True,
        )
    with arr2:
        st.markdown(
            "<div style='font-size:2rem;text-align:center;margin-top:10px'>→</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div style='background:#4a1a5e;border-radius:8px;padding:14px;"
            "text-align:center;color:white'>"
            "<b>Step 3</b><br>Aggregate by Restaurant<br>"
            "<small>preprocessor.build_restaurant_table()</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Raw data toggle ─────────────────────────────────────────────────────
    load_raw_data = st.toggle(
        "Load raw CSV to explore Steps 1 & 2 (first load ~10 s)",
        value=False,
    )

    # ── STEP 1 & 2 ──────────────────────────────────────────────────────────
    st.subheader("Step 1 — Raw CSV snapshot")

    if load_raw_data:
        raw = _load_raw()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Violation records", f"{len(raw):,}")
        m2.metric("Unique restaurants (CAMIS)", f"{raw['camis'].nunique():,}")
        m3.metric("Columns", len(raw.columns))
        m4.metric("Date range",
                  f"{raw['inspection_date'].min().date()} → "
                  f"{raw['inspection_date'].max().date()}")

        with st.expander("Original column names (before renaming)"):
            orig_cols = list(pd.read_csv(RAW_CSV, nrows=0).columns)
            st.dataframe(
                pd.DataFrame({"Original header": orig_cols}),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Sample raw records (first 200 rows)"):
            st.dataframe(raw.head(200), use_container_width=True)

        st.subheader("Step 2 — After `load_raw()`: cleaning applied")
        st.markdown(
            """
| Transform | What happened |
|---|---|
| Column rename | 27 headers → snake_case (e.g. `CUISINE DESCRIPTION` → `cuisine`) |
| Whitespace strip | Leading/trailing spaces removed from all string columns |
| Empty strings | Replaced with `NaN` so pandas treats them as missing |
| Date parsing | `inspection_date`, `grade_date`, `record_date` → `datetime64` |
| Placeholder dates | `1900-01-01` inspection dates → `NaT` (restaurant never inspected) |
| Numeric cast | `score`, `latitude`, `longitude`, `zipcode`, `community_board`, … → `float64` |
| CAMIS | Cast to nullable integer `Int64` |
| Categoricals | `critical_flag` title-cased; `grade` uppercased; `boro` title-cased |
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
            null_df = (
                raw.isnull().sum()
                .rename("null_count")
                .reset_index()
            )
            null_df.columns = ["column", "null_count"]
            null_df["pct_missing"] = (null_df["null_count"] / len(raw) * 100).round(1)
            null_df = null_df[null_df["null_count"] > 0].sort_values(
                "null_count", ascending=False
            )
            st.dataframe(null_df, use_container_width=True, hide_index=True)

        # Grade distribution in raw data
        st.markdown("**Grade distribution in raw records**")
        grade_raw = raw["grade"].value_counts().reset_index()
        grade_raw.columns = ["grade", "count"]
        fig_grade_raw = px.bar(
            grade_raw,
            x="grade",
            y="count",
            color="grade",
            color_discrete_map={
                "A": GRADE_COLOR_HEX["A"],
                "B": GRADE_COLOR_HEX["B"],
                "C": GRADE_COLOR_HEX["C"],
            },
            labels={"count": "Violation records"},
            title="Raw-record grade distribution",
        )
        fig_grade_raw.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_grade_raw, use_container_width=True)

    else:
        st.info(
            "Enable the toggle above to load and inspect the raw CSV live. "
            "The pre-computed stats below are based on the already-built `restaurants.csv`."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Violation records (known)", "295,995")
        m2.metric("Unique restaurants (CAMIS)", "30,935")
        m3.metric("Raw columns", "27")
        m4.metric("Source file", "DOHMH NYC Open Data")

    # ── STEP 3 ──────────────────────────────────────────────────────────────
    st.subheader("Step 3 — `build_restaurant_table()`: one row per restaurant")

    restaurants = _load_restaurants()

    st.markdown(
        """
`build_restaurant_table()` collapses the violation-level frame into a single row
per CAMIS identifier. Five sub-steps run in sequence:

| Sub-step | What it does |
|---|---|
| **Static info** | For each restaurant, picks the most-recent non-null value of name, address, coordinates, borough, cuisine … |
| **Latest inspection** | Finds the row with the newest `inspection_date` per CAMIS and extracts grade, score, action, and inspection type |
| **Latest grade** | Prefers a row that carries a letter grade (A/B/C) over rows without; also encodes grade as 3/2/1 |
| **Inspection history** | Deduplicates to one row per *(CAMIS, date)* before computing mean / min / max score and total inspection count |
| **Violation summary** | Counts total, critical, and non-critical violations; collects unique violation codes |
"""
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Restaurants", f"{len(restaurants):,}")
    m2.metric("Output columns", len(restaurants.columns))
    m3.metric("Grade A", f"{(restaurants['latest_grade'] == 'A').sum():,}")
    m4.metric("Grade B", f"{(restaurants['latest_grade'] == 'B').sum():,}")
    m5.metric("Grade C", f"{(restaurants['latest_grade'] == 'C').sum():,}")

    # Grade distribution
    col_left, col_right = st.columns(2)

    with col_left:
        grade_dist = (
            restaurants["latest_grade"]
            .fillna("No grade")
            .value_counts()
            .reset_index()
        )
        grade_dist.columns = ["grade", "count"]
        color_map = {
            "A": GRADE_COLOR_HEX["A"],
            "B": GRADE_COLOR_HEX["B"],
            "C": GRADE_COLOR_HEX["C"],
        }
        fig_grade = px.bar(
            grade_dist,
            x="grade",
            y="count",
            color="grade",
            color_discrete_map=color_map,
            title="Restaurant-level grade distribution",
            labels={"count": "Restaurants"},
        )
        fig_grade.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_grade, use_container_width=True)

    with col_right:
        boro_dist = (
            restaurants[restaurants["boro"] != "0"]["boro"]
            .value_counts()
            .reset_index()
        )
        boro_dist.columns = ["borough", "count"]
        fig_boro = px.pie(
            boro_dist,
            names="borough",
            values="count",
            title="Restaurants by borough",
            hole=0.35,
        )
        fig_boro.update_layout(height=320)
        st.plotly_chart(fig_boro, use_container_width=True)

    # Score distribution
    scored = restaurants["latest_score"].dropna()
    fig_score = px.histogram(
        scored,
        nbins=60,
        title="Latest inspection score distribution (lower = better)",
        labels={"value": "Score", "count": "Restaurants"},
        color_discrete_sequence=["#4C8BF5"],
    )
    fig_score.add_vline(x=14, line_dash="dash", line_color=GRADE_COLOR_HEX["A"],
                        annotation_text="A cutoff (≤13)")
    fig_score.add_vline(x=28, line_dash="dash", line_color=GRADE_COLOR_HEX["B"],
                        annotation_text="B cutoff (14–27)")
    fig_score.update_layout(height=320)
    st.plotly_chart(fig_score, use_container_width=True)

    with st.expander("Full output column list"):
        col_df = pd.DataFrame(
            {
                "column": restaurants.columns.tolist(),
                "dtype": restaurants.dtypes.astype(str).tolist(),
                "non-null": restaurants.notna().sum().tolist(),
                "null": restaurants.isna().sum().tolist(),
            }
        )
        st.dataframe(col_df, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2 — DATA EXPLORER
# ===========================================================================
with tab_explore:
    st.header("Data Explorer")
    restaurants = _restaurants
    filtered = _filtered

    st.markdown(
        f"Showing **{len(filtered):,}** of {len(restaurants):,} restaurants "
        f"matching the sidebar filters."
    )

    DISPLAY_COLS = [
        "camis", "dba", "boro", "cuisine", "building", "street", "zipcode",
        "latest_grade", "latest_score", "latest_inspection_date",
        "inspection_count", "total_violations", "critical_violations",
        "mean_score",
    ]
    display_cols = [c for c in DISPLAY_COLS if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=520,
    )

    # Summary stats for filtered set
    with st.expander("Summary statistics for filtered set"):
        num_cols = [
            "latest_score", "mean_score", "inspection_count",
            "total_violations", "critical_violations",
        ]
        existing_num = [c for c in num_cols if c in filtered.columns]
        st.dataframe(
            filtered[existing_num].describe().round(2),
            use_container_width=True,
        )


# ===========================================================================
# TAB 3 — NYC MAP
# ===========================================================================
with tab_map:
    st.header("NYC Restaurant Map")

    restaurants = _restaurants
    filtered_map = _filtered

    # Keep only rows with valid coordinates
    map_df = filtered_map.dropna(subset=["latitude", "longitude"]).copy()
    map_df = _add_map_color(map_df)

    # Tooltip fields
    map_df["tooltip_grade"] = map_df["latest_grade"].fillna("—")
    map_df["tooltip_score"] = map_df["latest_score"].fillna(np.nan)
    map_df["tooltip_name"] = map_df["dba"].fillna("Unknown")
    map_df["tooltip_cuisine"] = map_df["cuisine"].fillna("—")
    map_df["tooltip_address"] = (
        map_df["building"].fillna("").astype(str).str.strip()
        + " "
        + map_df["street"].fillna("").astype(str).str.strip()
    ).str.strip()

    st.markdown(
        f"Plotting **{len(map_df):,}** restaurants with valid coordinates "
        f"(out of {len(filtered_map):,} matching filters)."
    )

    # Map controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        point_size = st.slider("Point radius (m)", 20, 200, 60, step=10)
    with col_ctrl2:
        color_by = st.selectbox(
            "Colour by",
            ["Grade (A/B/C)", "Critical violations (heatmap)"],
        )
    with col_ctrl3:
        map_style = st.selectbox(
            "Map style",
            ["dark", "light", "road", "satellite"],
        )

    STYLE_MAP = {
        "dark": "mapbox://styles/mapbox/dark-v10",
        "light": "mapbox://styles/mapbox/light-v10",
        "road": "mapbox://styles/mapbox/streets-v11",
        "satellite": "mapbox://styles/mapbox/satellite-streets-v11",
    }

    # Build the pydeck layer
    if color_by == "Grade (A/B/C)":
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius=point_size,
            pickable=True,
            opacity=0.85,
            stroked=False,
        )
        tooltip = {
            "html": (
                "<b>{tooltip_name}</b><br/>"
                "Cuisine: {tooltip_cuisine}<br/>"
                "Address: {tooltip_address}<br/>"
                "Grade: <b>{tooltip_grade}</b> &nbsp; Score: {tooltip_score}<br/>"
                "Inspections: {inspection_count} &nbsp; "
                "Critical violations: {critical_violations}"
            ),
            "style": {
                "backgroundColor": "#1e1e2e",
                "color": "white",
                "fontSize": "13px",
                "padding": "8px",
                "borderRadius": "6px",
            },
        }
    else:
        # Heatmap by critical violations
        layer = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_weight="critical_violations",
            aggregation="SUM",
            opacity=0.8,
        )
        tooltip = {
            "html": "<b>{tooltip_name}</b><br/>Critical violations: {critical_violations}",
            "style": {"backgroundColor": "#1e1e2e", "color": "white"},
        }

    view = pdk.ViewState(
        latitude=40.7128,
        longitude=-73.9760,
        zoom=10.5,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style=STYLE_MAP[map_style],
    )

    st.pydeck_chart(deck, use_container_width=True, height=620)

    # Grade legend
    if color_by == "Grade (A/B/C)":
        leg_cols = st.columns(4)
        for col, (grade, color) in zip(
            leg_cols,
            [
                ("A — Excellent (score ≤ 13)", GRADE_COLOR_HEX["A"]),
                ("B — Good (score 14–27)", GRADE_COLOR_HEX["B"]),
                ("C — Needs improvement (score ≥ 28)", GRADE_COLOR_HEX["C"]),
                ("No grade / pending", GRADE_COLOR_HEX["Other"]),
            ],
        ):
            col.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{color};border-radius:50%;margin-right:6px'></span>"
                f"{grade}",
                unsafe_allow_html=True,
            )

    # Borough breakdown chart (for filtered set)
    st.markdown("---")
    st.subheader("Grade breakdown by borough (filtered view)")
    graded = filtered_map[filtered_map["latest_grade"].isin(["A", "B", "C"])].copy()
    if not graded.empty:
        boro_grade = (
            graded.groupby(["boro", "latest_grade"])
            .size()
            .reset_index(name="count")
        )
        boro_grade = boro_grade[boro_grade["boro"] != "0"]
        fig_bg = px.bar(
            boro_grade,
            x="boro",
            y="count",
            color="latest_grade",
            color_discrete_map={
                "A": GRADE_COLOR_HEX["A"],
                "B": GRADE_COLOR_HEX["B"],
                "C": GRADE_COLOR_HEX["C"],
            },
            barmode="group",
            labels={"boro": "Borough", "count": "Restaurants", "latest_grade": "Grade"},
            title="Restaurants per grade per borough",
        )
        fig_bg.update_layout(height=350)
        st.plotly_chart(fig_bg, use_container_width=True)
    else:
        st.info("No graded restaurants in current filter selection.")
