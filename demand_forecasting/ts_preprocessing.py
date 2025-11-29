# ts_preprocessing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# These come from what used to be `%run "/.../general_forecasting_functions"`
from general_forecasting_functions import (
    upsample_groupedweeklyts_spark,
    remove_data_prior_to_inactive_periods,
    remove_series_w_insuff_data,
    fill_dim_nulls_bfill_ffill,
    spark_remove_outliers,
    spark_pandas_interpolate_convert,
)


@dataclass
class TSPreprocessingConfig:
    """
    Configuration for time-series preprocessing.

    The defaults are chosen to match your legacy sales preprocessing,
    but you can override them per project.
    """

    # Where to read the raw data
    source_table: str = "forecast_dev.data_science.sales_forecast_source_view"

    # Column names in the *raw* table
    raw_value_col: str = "ordered_qty_fc"
    raw_date_col: str = "req_del_fw_start_date"
    group_col: str = "time_series_id"

    # Standardized column names used during processing
    value_col: str = "y"
    date_col: str = "ds"

    # Numeric columns
    numerical_cols: List[str] = None
    cols_for_outlier_removal: List[str] = None

    # Interpolation config
    interpolation_method: str = "linear"  # 'linear', 'polynomial', 'spline'
    interpolation_order: int = 3  # used for 'polynomial' / 'spline'

    # Thresholds
    inactive_threshold: int = 4
    insufficient_data_threshold: int = 1
    short_series_threshold: int = 52
    outlier_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.numerical_cols is None:
            # Same idea as your legacy script: y, y_clean, price
            self.numerical_cols = [self.value_col, "y_clean", "gross_price_fc"]

        if self.cols_for_outlier_removal is None:
            self.cols_for_outlier_removal = [self.value_col]

    @property
    def cols_for_interpolation(self) -> List[str]:
        # mimic: [col for col in numerical_cols if col not in [value_col]]
        return [c for c in self.numerical_cols if c != self.value_col]
