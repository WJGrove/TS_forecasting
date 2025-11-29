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

    Change the names as needed for each project.
    """

    # Where to read the raw data
    source_table: str = "forecast_dev.data_science.sales_forecast_source_view"

    # Column names in the *raw* table
    raw_value_col: str = "ordered_qty_fc"
    raw_date_col: str = "req_del_fw_start_date"
    group_col: str = (
        "time_series_id"  # This ID is constructed baased on the desired granularity of the forecast.
    )

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


class TSPreprocessor:
    """
    Preprocessing pipeline for panel time series such as weekly sales by customer/product.
    """

    def __init__(self, spark: SparkSession, config: TSPreprocessingConfig) -> None:
        self.spark = spark
        self.config = config

    # ---------- Top-level orchestrator ----------

    def run(self) -> DataFrame:
        """
        Run the full preprocessing pipeline and return the final, interpolated DataFrame.
        """
        df_raw = self.load_source_data()
        df_clean = self.basic_cleaning(df_raw)
        df_weekly = self.aggregate_to_period(df_clean)
        df_dates_filled = self.fill_missing_dates(df_weekly)
        df_active = self.remove_inactive_and_insufficient(df_dates_filled)
        df_dims_filled = self.fill_dimensions(df_active)
        df_out_rem = self.remove_outliers(df_dims_filled)
        df_interpolated = self.interpolate(df_out_rem)
        return df_interpolated

    # ---------- Individual steps (roughly matching your notebook cells) ----------

    def load_source_data(self) -> DataFrame:
        """
        SELECT * from the configured source table.
        """
        return self.spark.sql(f"SELECT * FROM {self.config.source_table}")

    def basic_cleaning(self, df: DataFrame) -> DataFrame:
        """
        Drop duplicates and normalize column names to (value_col, date_col).
        """
        c = self.config
        df = df.dropDuplicates()
        df = df.withColumnRenamed(c.raw_value_col, c.value_col)
        df = df.withColumnRenamed(c.raw_date_col, c.date_col)
        return df

    def aggregate_to_period(self, df: DataFrame) -> DataFrame:
        """
        Aggregate to the target granularity: one row per (group_col, date_col).
        Currently this matches your weekly aggregation logic.
        """
        c = self.config

        df_weekly = df.groupBy(c.date_col, c.group_col).agg(
            F.sum(c.value_col).alias(c.value_col),
            (
                F.sum(F.col("gross_price_fc") * F.col(c.value_col))
                / F.sum(F.col(c.value_col))
            ).alias("gross_price_fc"),
            F.first("parent_company_fc").alias("parent_company_fc"),
            F.first("soldto_id").alias("soldto_id"),
            F.first("item_id_fc").alias("item_id_fc"),
            F.first("bottle_type").alias("bottle_type"),
        )

        return df_weekly.orderBy(c.date_col, c.group_col)

    def fill_missing_dates(self, df_weekly: DataFrame) -> DataFrame:
        """
        Use upsample_groupedweeklyts_spark to fill gaps in each series.
        """
        c = self.config
        return upsample_groupedweeklyts_spark(df_weekly, c.date_col, c.group_col)

    def remove_inactive_and_insufficient(self, df_dates_filled: DataFrame) -> DataFrame:
        """
        Remove sections of series before long inactive periods and series with too few points.
        """
        c = self.config

        df_inact_rem = remove_data_prior_to_inactive_periods(
            df_dates_filled,
            c.value_col,
            c.date_col,
            c.group_col,
            c.inactive_threshold,
        )

        df_insuff = remove_series_w_insuff_data(
            df_inact_rem,
            c.group_col,
            c.insufficient_data_threshold,
            c.value_col,
        )
        return df_insuff

    def fill_dimensions(self, df: DataFrame) -> DataFrame:
        """
        Forward/backward fill dimension columns for dummy rows.
        """
        c = self.config

        cols_excluded = [c.date_col, c.group_col] + c.numerical_cols
        cols_to_fill = [col for col in df.columns if col not in cols_excluded]

        return fill_dim_nulls_bfill_ffill(df, c.group_col, c.date_col, cols_to_fill)

    def remove_outliers(self, df: DataFrame) -> DataFrame:
        """
        Replace outliers in selected columns with NULLs.
        """
        c = self.config
        return spark_remove_outliers(
            df,
            c.cols_for_outlier_removal,
            outlier_threshold=c.outlier_threshold,
            group_col=c.group_col,
        )

    def interpolate(self, df: DataFrame) -> DataFrame:
        """
        Interpolate numeric columns using pandas-based interpolation.
        """
        c = self.config
        return spark_pandas_interpolate_convert(
            df,
            c.group_col,
            c.cols_for_interpolation,
            c.date_col,
            c.interpolation_method,
            c.interpolation_order,
        )
