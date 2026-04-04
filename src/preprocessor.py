"""
preprocessor.py
---------------
Transforms the violation-level DataFrame produced by data_loader.load_raw()
into a restaurant-level DataFrame where every row is a unique restaurant
(identified by CAMIS).

Each output row aggregates:
  - Static restaurant info  (name, address, cuisine, coordinates …)
  - Latest inspection result (grade, score, date)
  - Full inspection history  (counts, mean score, days since last visit …)
  - Violation summary        (total, critical, non-critical counts)

The resulting table is ready for downstream use by the KNN classifier and
the SentenceTransformer-based text retrieval module.

Typical usage
-------------
    from src.data_loader import load_raw
    from src.preprocessor import build_restaurant_table

    raw = load_raw("data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv")
    restaurants = build_restaurant_table(raw)
    restaurants.to_csv("data/restaurants.csv", index=False)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_raw

# ---------------------------------------------------------------------------
# Grade constants
# ---------------------------------------------------------------------------
# Only these grades represent a final scored result
LETTER_GRADES = {"A", "B", "C"}

# Numeric encoding used as the ML target label
GRADE_TO_INT: dict[str, int] = {"A": 3, "B": 2, "C": 1}
INT_TO_GRADE: dict[int, str] = {v: k for k, v in GRADE_TO_INT.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _most_common(series: pd.Series) -> object:
    """Return the most frequent non-null value in *series*, or NaN."""
    counts = series.dropna().value_counts()
    return counts.index[0] if not counts.empty else np.nan


def _latest_value(df: pd.DataFrame, value_col: str, date_col: str = "inspection_date") -> pd.Series:
    """For each CAMIS group return the *value_col* from the row with the
    most recent *date_col*, ignoring NaT rows."""
    valid = df[df[date_col].notna()].copy()
    idx = valid.groupby("camis")[date_col].idxmax()
    return valid.loc[idx].set_index("camis")[value_col]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_restaurant_table(
    raw: pd.DataFrame,
    *,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate the violation-level *raw* DataFrame into one row per restaurant.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``data_loader.load_raw()``.
    reference_date : pd.Timestamp, optional
        Date used to compute *days_since_last_inspection*.
        Defaults to today.

    Returns
    -------
    pd.DataFrame
        One row per unique CAMIS with the columns described below.

    Output columns
    --------------
    Identity
        camis, dba, boro, building, street, zipcode, phone, cuisine,
        latitude, longitude, nta, community_board, council_district, location

    Latest inspection
        latest_grade         – most recent letter grade (A / B / C or NaN)
        latest_grade_encoded – numeric encoding (A=3, B=2, C=1, else NaN)
        latest_grade_date    – date the current grade was issued
        latest_score         – score from the most recent inspection
        latest_inspection_date
        latest_action        – action taken at the most recent inspection
        latest_inspection_type

    Inspection history
        inspection_count          – number of unique inspection dates
        mean_score                – mean score across all graded inspections
        min_score / max_score
        days_since_last_inspection – days between latest inspection and reference_date

    Violation summary
        total_violations          – total violation rows (excluding no-violation rows)
        critical_violations       – rows where critical_flag == "Critical"
        non_critical_violations   – rows where critical_flag == "Not Critical"
        unique_violation_codes    – comma-separated list of distinct violation codes seen
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    df = raw.copy()

    # ------------------------------------------------------------------
    # 1. Static restaurant info
    #    Take the most-recent non-null value for each identity column.
    # ------------------------------------------------------------------
    IDENTITY_COLS = [
        "dba", "boro", "building", "street", "zipcode", "phone", "cuisine",
        "latitude", "longitude", "nta", "community_board", "council_district",
        "location",
    ]

    def _latest_static(group: pd.DataFrame) -> pd.Series:
        """From the group, pick the most-recent non-null value per column."""
        # Sort descending so first valid value is the most recent
        grp = group.sort_values("inspection_date", ascending=False, na_position="last")
        result: dict = {}
        for col in IDENTITY_COLS:
            if col in grp.columns:
                val = grp[col].dropna()
                result[col] = val.iloc[0] if not val.empty else np.nan
        return pd.Series(result)

    static = df.groupby("camis", sort=False).apply(_latest_static)

    # ------------------------------------------------------------------
    # 2. Latest inspection result
    #    One inspection can have multiple violation rows; we identify the
    #    most recent inspection by date and pull the per-inspection columns
    #    from any row of that inspection (score / grade / action are the
    #    same across violation rows of the same inspection).
    # ------------------------------------------------------------------
    inspected = df[df["inspection_date"].notna()].copy()

    if not inspected.empty:
        latest_idx = inspected.groupby("camis")["inspection_date"].idxmax()
        latest_rows = inspected.loc[latest_idx].set_index("camis")

        latest_cols = {
            "latest_inspection_date": latest_rows["inspection_date"],
            "latest_score":           latest_rows["score"],
            "latest_action":          latest_rows["action"],
            "latest_inspection_type": latest_rows["inspection_type"],
        }

        # Latest *grade* – prefer a row that actually has a letter grade
        def _latest_grade(group: pd.DataFrame) -> pd.Series:
            grp = group.sort_values("inspection_date", ascending=False, na_position="last")
            # Try to find a row with a letter grade first
            graded = grp[grp["grade"].isin(LETTER_GRADES)]
            if not graded.empty:
                row = graded.iloc[0]
            else:
                row = grp.iloc[0]
            return pd.Series({
                "latest_grade":      row.get("grade", np.nan),
                "latest_grade_date": row.get("grade_date", np.nan),
            })

        grade_info = inspected.groupby("camis").apply(_latest_grade)
    else:
        # No inspections at all – fill with NaN
        idx = df["camis"].unique()
        latest_cols = {col: pd.Series(np.nan, index=idx) for col in [
            "latest_inspection_date", "latest_score",
            "latest_action", "latest_inspection_type",
        ]}
        grade_info = pd.DataFrame(
            {"latest_grade": np.nan, "latest_grade_date": np.nan},
            index=idx,
        )

    # ------------------------------------------------------------------
    # 3. Inspection history aggregates
    #    Deduplicate to one row per (camis, inspection_date) before
    #    computing score statistics so multi-violation inspections are not
    #    double-counted.
    # ------------------------------------------------------------------
    insp_dedup = (
        inspected
        .dropna(subset=["score"])
        .drop_duplicates(subset=["camis", "inspection_date"])
    )

    insp_stats = insp_dedup.groupby("camis")["score"].agg(
        mean_score="mean",
        min_score="min",
        max_score="max",
    )

    inspection_count = (
        inspected
        .drop_duplicates(subset=["camis", "inspection_date"])
        .groupby("camis")
        .size()
        .rename("inspection_count")
    )

    # ------------------------------------------------------------------
    # 4. Violation summary
    #    Only count rows that actually have a violation code.
    # ------------------------------------------------------------------
    has_violation = df[df["violation_code"].notna()].copy()

    total_violations = (
        has_violation.groupby("camis").size().rename("total_violations")
    )

    critical_violations = (
        has_violation[has_violation["critical_flag"] == "Critical"]
        .groupby("camis").size()
        .rename("critical_violations")
    )

    non_critical_violations = (
        has_violation[has_violation["critical_flag"] == "Not Critical"]
        .groupby("camis").size()
        .rename("non_critical_violations")
    )

    unique_violation_codes = (
        has_violation.groupby("camis")["violation_code"]
        .apply(lambda s: ",".join(sorted(s.dropna().unique())))
        .rename("unique_violation_codes")
    )

    # ------------------------------------------------------------------
    # 5. Assemble final table
    # ------------------------------------------------------------------
    result = static.copy()

    # Latest inspection columns
    for col, series in latest_cols.items():
        result[col] = series

    result = result.join(grade_info, how="left")

    # Numeric grade encoding
    result["latest_grade_encoded"] = result["latest_grade"].map(GRADE_TO_INT)

    # History / stats
    result = result.join(insp_stats, how="left")
    result = result.join(inspection_count, how="left")
    result["inspection_count"] = result["inspection_count"].fillna(0).astype(int)

    # Days since last inspection
    result["days_since_last_inspection"] = (
        reference_date - result["latest_inspection_date"]
    ).dt.days

    # Violation counts (fill 0 for restaurants with no recorded violations)
    result = result.join(total_violations, how="left")
    result = result.join(critical_violations, how="left")
    result = result.join(non_critical_violations, how="left")
    result = result.join(unique_violation_codes, how="left")

    for col in ("total_violations", "critical_violations", "non_critical_violations"):
        result[col] = result[col].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 6. Column ordering
    # ------------------------------------------------------------------
    ordered = [
        # identity
        "dba", "boro", "building", "street", "zipcode", "phone", "cuisine",
        "latitude", "longitude", "nta", "community_board", "council_district",
        "location",
        # latest inspection
        "latest_grade", "latest_grade_encoded", "latest_grade_date",
        "latest_score", "latest_inspection_date", "latest_action",
        "latest_inspection_type",
        # history
        "inspection_count", "mean_score", "min_score", "max_score",
        "days_since_last_inspection",
        # violations
        "total_violations", "critical_violations", "non_critical_violations",
        "unique_violation_codes",
    ]
    existing = [c for c in ordered if c in result.columns]
    result = result[existing]

    return result.reset_index()   # brings 'camis' back as a regular column


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    csv_path: str = "data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv",
    output_path: str = "data/restaurants.csv",
) -> None:
    """Load raw data, aggregate to restaurant level, and save to CSV."""
    print(f"Loading raw data from: {csv_path}")
    raw = load_raw(csv_path)
    print(f"  {len(raw):,} violation records, {raw['camis'].nunique():,} unique restaurants")

    print("Building restaurant table ...")
    restaurants = build_restaurant_table(raw)
    print(f"  {len(restaurants):,} rows in output (one per restaurant)")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    restaurants.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    kwargs: dict = {}
    if len(args) >= 1:
        kwargs["csv_path"] = args[0]
    if len(args) >= 2:
        kwargs["output_path"] = args[1]
    main(**kwargs)
