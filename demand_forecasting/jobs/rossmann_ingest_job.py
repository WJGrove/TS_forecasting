from __future__ import annotations

from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# This file lives in demand_forecasting/jobs/, so:
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "demand_forecasting" / "data" / "kaggle_rossmann"

RAW_TRAIN = DATA_DIR / "train.csv"
RAW_STORE = DATA_DIR / "store.csv"
OUT_PANEL = DATA_DIR / "rossmann_panel.parquet"


# ------------------------------------------------------------------
# Core prep logic
# ------------------------------------------------------------------


def load_and_merge_rossmann() -> pd.DataFrame:
    """Load Rossmann train + store metadata and build a clean panel DataFrame."""

    # 1) Load raw CSVs
    train = pd.read_csv(RAW_TRAIN)
    store = pd.read_csv(RAW_STORE)

    # 2) Merge store metadata onto the daily sales
    df = train.merge(store, on="Store", how="left")

    # 3) Basic cleaning
    # - keep only days when the store is open
    # - and with strictly positive sales
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # 4) Parse dates
    df["Date"] = pd.to_datetime(df["Date"])

    # 5) Rename key columns to match your forecasting pipeline conventions
    df = df.rename(
        columns={
            "Store": "time_series_id",  # panel id
            "Date": "ds",  # time index
            "Sales": "y",  # target (integer sales)
        }
    )

    # 6) Sort for sanity
    df = df.sort_values(["time_series_id", "ds"])

    # 7) Add a placeholder short-series flag; your preprocessor/diagnostics
    #    can overwrite this based on actual history length.
    if "is_short_series" not in df.columns:
        df["is_short_series"] = False

    return df


def _normalize_dtypes_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column dtypes so pyarrow can write parquet cleanly.

    - Ensure ds is datetime.
    - Force known categorical/text columns to string to avoid mixed-type object columns.
    """
    # 1) Ensure ds is datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # 2) Columns that should be treated as text
    string_cols = [
        "time_series_id",
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval",
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")  # or .astype(str)

    return df


def main() -> None:
    df = load_and_merge_rossmann()

    df = _normalize_dtypes_for_parquet(df)

    # Quick sanity prints so you can see what's going on
    print(f"Rows: {len(df):,}")
    print("Columns:", list(df.columns))
    print()
    print("Unique series (time_series_id):", df["time_series_id"].nunique())
    print()
    print(df.head().to_string())

    # Save as Parquet for downstream use (Spark or pandas)
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PANEL, index=False)
    print(f"\nWrote cleaned Rossmann panel to: {OUT_PANEL}")


if __name__ == "__main__":
    main()
