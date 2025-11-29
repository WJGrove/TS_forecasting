from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# The following functions still have to be checked and many will likely need to be rewritten.
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

    # Lists of columns
    numerical_cols: List[str] = None
    cols_for_outlier_removal: List[str] = None

    # Interpolation config
    interpolation_method: str = "linear"  # 'linear', 'polynomial', 'spline'
    interpolation_order: int = 3  # used for 'polynomial' / 'spline'

    # Thresholds (these are )
    inactive_threshold: int = (
        4  # the number of consecutive 0s before you consider a series inactive.
    )
    insufficient_data_threshold: int = (
        1  # the number of observations...doublecheck the function to word this comment properly
    )
    short_series_threshold: int = (
        52  # the number of observations...doublecheck the function to word this comment properly
    )
    outlier_threshold: float = 3.0  # standard deviations

    def __post_init__(self) -> None:
        if self.numerical_cols is None:
            # Same idea as your legacy script: y, y_clean, price
            self.numerical_cols = [self.value_col, "y_clean", "gross_price_fc"]

        if self.cols_for_outlier_removal is None:
            self.cols_for_outlier_removal = [self.value_col]

    @property
    def cols_for_interpolation(self) -> List[str]:
        # we only need to interpolate columns with missing values
        return [c for c in self.numerical_cols if c != self.value_col]
