"""
data_loader.py
--------------
Loads and performs initial cleaning of the raw DOHMH NYC Restaurant
Inspection Results CSV.  The output is a tidy DataFrame where every
row still represents one violation record, but all columns have the
right Python types, snake_case names, and sentinel / placeholder values
have been replaced with proper NaN / NaT.

Typical usage
-------------
    from src.data_loader import load_raw

    df = load_raw("data/DOHMH_New_York_City_Restaurant_Inspection_Results_20260403.csv")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column rename map  (original CSV header → snake_case name used downstream)
# ---------------------------------------------------------------------------
COLUMN_RENAME: dict[str, str] = {
    "CAMIS": "camis",
    "DBA": "dba",
    "BORO": "boro",
    "BUILDING": "building",
    "STREET": "street",
    "ZIPCODE": "zipcode",
    "PHONE": "phone",
    "CUISINE DESCRIPTION": "cuisine",
    "INSPECTION DATE": "inspection_date",
    "ACTION": "action",
    "VIOLATION CODE": "violation_code",
    "VIOLATION DESCRIPTION": "violation_description",
    "CRITICAL FLAG": "critical_flag",
    "SCORE": "score",
    "GRADE": "grade",
    "GRADE DATE": "grade_date",
    "RECORD DATE": "record_date",
    "INSPECTION TYPE": "inspection_type",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Community Board": "community_board",
    "Council District": "council_district",
    "Census Tract": "census_tract",
    "BIN": "bin",
    "BBL": "bbl",
    "NTA": "nta",
    "Location": "location",
}

# Inspection dates set to this value indicate the restaurant has never
# been inspected yet – treat as NaT after parsing.
_PLACEHOLDER_DATE = pd.Timestamp("1900-01-01")


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw DOHMH CSV and return a cleaned, typed violation-level DataFrame.

    Each row represents one violation record (multiple rows can share the
    same restaurant and inspection date).  Placeholder inspection dates
    (1/1/1900) are converted to NaT.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw DOHMH CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with snake_case column names.
    """
    csv_path = Path(csv_path)

    # 1. Read everything as strings first to avoid mixed-type warnings
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # 2. Rename to snake_case; drop any extra auto-generated region columns
    df = df.rename(columns=COLUMN_RENAME)
    keep_cols = [c for c in COLUMN_RENAME.values() if c in df.columns]
    df = df[keep_cols].copy()

    # 3. Strip whitespace, then replace empty strings with NaN
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    df.replace("", np.nan, inplace=True)

    # 4. Parse date columns
    for col in ("inspection_date", "grade_date", "record_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%m/%d/%Y")

    # Replace placeholder 1/1/1900 inspection dates with NaT
    df.loc[df["inspection_date"] == _PLACEHOLDER_DATE, "inspection_date"] = pd.NaT

    # 5. Cast numeric columns
    for col in ("score", "latitude", "longitude",
                "community_board", "council_district", "census_tract",
                "bin", "bbl", "zipcode"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # CAMIS is a stable integer identifier
    df["camis"] = pd.to_numeric(df["camis"], errors="coerce").astype("Int64")

    # 6. Standardise categoricals
    df["critical_flag"] = df["critical_flag"].str.title()   # e.g. "Critical"
    df["grade"] = df["grade"].str.upper()
    df["boro"] = df["boro"].str.title()

    return df.reset_index(drop=True)
