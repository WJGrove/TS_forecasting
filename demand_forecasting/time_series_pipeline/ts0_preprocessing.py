from __future__ import annotations

from datetime import datetime
import pytz

from dataclasses import dataclass, field
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# The functions below are defined in a legacy reference folder and will be refactored after the preprocessing, diagnostics, and plotting modules are stable.
from demand_forecasting.legacy_folder.general_fc_functions_legacy import (
    upsample_groupedweeklyts_spark,
    flag_data_prior_to_inactive_periods,
    remove_insuff_series,
    fill_dim_nulls_groupwise,
    groupwise_clean_flag_outliers,
    interpolate_groupwise_numeric,
    boxcox_multi_ts_sps,
)


@dataclass
class TSPreprocessingConfig:
    """
    Configuration for time-series preprocessing.

    Change the names as needed for each project.
    """

    # source and destination tables
    source_table: str = "forecast_dev.data_science.ts_source_view"
    output_catalog: str = "forecast_dev.data_science"
    output_table_name: str = "ts_preprocessed"

    # Column names in the input table
    raw_value_col: str = "ordered_qty"
    raw_date_col: str = "req_del_fw_start_date"
    group_col: str = "time_series_id"

    customer_parent_company_col: str | None = "parent_company_fc"
    customer_col: str | None = "soldto_id"
    customer_state_col: str | None = "soldto_state"
    customer_market_col: str | None = "market_area"
    customer_age_segment_col: str | None = "age_segment"

    product_id_col: str | None = "item_id_fc"
    product_unit_price_col: str | None = "gross_price_per_unit"
    product_wavg_unit_price_col: str = "product_wavg_unit_price"
    product_dim_col1: str | None = "package_type"
    product_dim_col2: str | None = "recipe"
    product_dim_col3: str | None = None

    # Time-series ID / grouping configuration
    # group_col is the name of the column used to identify a time series in the pipeline.
    # if group_key_cols is non-empty when the config is instantiated
    # , we construct group_col (the ts IDs) from these columns.
    # If group_key_cols is empty when the config is instantiated
    # , we assume group_col already exists in the source and use it.
    group_key_cols: List[str] = field(default_factory=list)
    group_key_separator: str = "|"

    # Time axis configuration
    time_granularity: str = "week"  # this is the period the analysis is done on

    # Standardized column names for the date and value columns for processing
    value_col: str = "y"
    date_col: str = "ds"

    # Lists of columns
    # "field(default_factory=list)" creates a fresh list per instance; "[]" would be
    # treated like the same list everywhere it appears.
    numerical_cols: List[str] = field(default_factory=list)
    cols_for_outlier_removal: List[str] = field(default_factory=list)

    # Interpolation config
    interpolation_method: str = "linear"  # 'linear', 'polynomial', 'spline'
    interpolation_order: int = 3  # used for 'polynomial' / 'spline'

    # Seasonality and thresholds:
    seasonal_period: int = 52
    # the number of periods below which a series is considered "short" (e.g., weeks):
    short_series_threshold: int = 52
    # the number of consecutive periods with 0s before you consider a series inactive:
    inactive_threshold: int = 4
    # the number of periods...[doublecheck the function to word this comment properly]:
    insufficient_data_threshold: int = 1

    outlier_threshold: float = 3.0  # standard deviations

    # Diagnostics configuration (short/new series volume warning thresholds)
    short_series_vol_warn1: float = 3.0  # percent of total volume
    short_series_vol_warn2: float = 5.0  # percent of total volume
    short_series_vol_warn3: float = 10.0  # percent of total volume

    timezone_name: str = "America/Chicago"

    def __post_init__(self) -> None:
        # 1) Validate thresholds and time granularity
        if self.short_series_threshold <= 0:
            raise ValueError("short_series_threshold must be positive")
        if self.inactive_threshold <= 0:
            raise ValueError("inactive_threshold must be positive")
        if self.insufficient_data_threshold <= 0:
            raise ValueError("insufficient_data_threshold must be positive")
        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")
        if self.time_granularity.lower() not in {"week"}:
            raise ValueError("time_granularity must be 'week'")

        # 2) Fill defaults for lists
        if not self.numerical_cols:
            self.numerical_cols = [self.value_col, "y_clean"]
            if self.product_wavg_unit_price_col:
                self.numerical_cols.append(self.product_wavg_unit_price_col)

        if not self.cols_for_outlier_removal:
            self.cols_for_outlier_removal = [self.value_col]

    @property
    def cols_for_interpolation(self) -> List[str]:
        # we only need to interpolate columns with missing values
        return [c for c in self.numerical_cols if c != self.value_col]

    @property
    def dim_cols(self) -> List[str]:
        """
        All configured dimension columns that should be carried/aggregated.

        Any that are None (or later set to None for a particular project)
        are automatically skipped.
        """
        return [
            c
            for c in [
                self.customer_parent_company_col,
                self.customer_col,
                self.customer_state_col,
                self.customer_market_col,
                self.customer_age_segment_col,
                self.product_id_col,
                self.product_dim_col1,
                self.product_dim_col2,
                self.product_dim_col3,
            ]
            if c is not None
        ]

    @property
    def output_table_fqn(self) -> str:
        return f"{self.output_catalog}.{self.output_table_name}"


class TSPreprocessor:
    """
    Preprocessing pipeline for panel time series such as weekly sales by customer/product.
    """

    def __init__(self, spark: SparkSession, config: TSPreprocessingConfig) -> None:
        self.spark = spark
        self.config = config

    # ---------- Top-level orchestrator ----------

    def run(self, *, with_boxcox: bool = True) -> DataFrame:
        """
        Run the full preprocessing pipeline and return the final transformed DataFrame.

        Steps:
        - load & clean
        - ensure group_col exists (time series IDs exist)
        = add a period column
        - aggregate to period
        - fill gaps & dimensions
        - remove outliers & interpolate
        - flag short series
        - optionally apply Box-Cox and per-series median
        """

        df_raw = self.load_source_data()
        df_clean = self.basic_cleaning(df_raw)
        df_with_id = self.ensure_group_col(df_clean)
        df_with_ds = self.add_period_column(df_with_id)
        df_weekly = self.aggregate_to_period(df_with_ds)
        df_dates_filled = self.fill_missing_dates(df_weekly)
        df_active = self.remove_inactive_and_insufficient(df_dates_filled)
        df_dims_filled = self.fill_dimensions(df_active)
        df_out_rem = self.remove_outliers(df_dims_filled)
        df_interpolated = self.interpolate(df_out_rem)
        df_shorts_flagged = self.flag_short_series(df_interpolated)

        if with_boxcox:
            transformed = self.apply_boxcox_and_median(df_shorts_flagged)
        else:
            transformed = df_shorts_flagged

        return transformed

    # ---------- Individual steps ----------

    def load_source_data(self) -> DataFrame:
        """
        SELECT * from the configured source table.
        """
        return self.spark.sql(f"SELECT * FROM {self.config.source_table}")

    def basic_cleaning(self, df: DataFrame) -> DataFrame:
        """
        Drop duplicates and normalize target value column name to (value_col).

        The raw_date_col is left as-is; add_period_column() will derive the
        standardized period key (date_col, e.g. 'ds') later.
        """
        c = self.config

        # 1) Check required columns exist in the source
        required = [c.raw_value_col, c.raw_date_col]
        if c.group_key_cols:
            required.extend(c.group_key_cols)
        else:
            required.append(c.group_col)

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in source data: {missing}")

        df = df.dropDuplicates()

        # 2) Rename value to standardized name if needed
        if c.raw_value_col != c.value_col:
            df = df.withColumnRenamed(c.raw_value_col, c.value_col)

        # raw_date_col is intentionally *not* renamed here
        return df

    def ensure_group_col(self, df: DataFrame) -> DataFrame:
        """
        Ensure the time-series ID column (group_col) exists.

        - If group_key_cols is non-empty, build group_col from those columns.
        - If group_key_cols is empty, assume group_col already exists in the data.
        """
        c = self.config

        if c.group_key_cols:
            # Make sure all key columns are present
            missing = [col for col in c.group_key_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing group key columns in source data: {missing}")

            # Build a string series ID by concatenating key columns
            df = df.withColumn(
                c.group_col,
                F.concat_ws(
                    c.group_key_separator,
                    *[F.col(col).cast("string") for col in c.group_key_cols],
                ),
            )
        else:
            # No keys specified; expect group_col to already exist
            if c.group_col not in df.columns:
                raise ValueError(
                    f"group_key_cols is empty and '{c.group_col}' is not present "
                    "in the data frame; cannot define series IDs."
                )

        return df

    def add_period_column(self, df: DataFrame) -> DataFrame:
        """
        Derive the period key (date_col, e.g. 'ds') from the raw date column,
        according to the configured time_granularity.

        Currently only 'week' is supported (weekly start-of-week).
        """
        c = self.config

        if c.raw_date_col not in df.columns:
            raise ValueError(
                f"raw_date_col '{c.raw_date_col}' not present in DataFrame."
            )

        gran = c.time_granularity.lower()

        if gran == "week":
            # Ensure we have a timestamp, then truncate to start-of-week and cast back to date.
            base_ts = F.to_timestamp(F.col(c.raw_date_col))
            df = df.withColumn(
                c.date_col,
                F.to_date(F.date_trunc("week", base_ts)),
            )
        # # Later, when all the functions are refactored, we can add support for daily:
        # elif gran == "day":
        #     df = df.withColumn(
        #         c.date_col,
        #         F.to_date(c.raw_date_col),
        #     )
        else:
            raise ValueError(
                f"time_granularity '{c.time_granularity}' is not supported yet; "
                "TSPreprocessor currently implements weekly processing only."
            )

        return df

    def aggregate_to_period(self, df: DataFrame) -> DataFrame:
        """
        Aggregate to the target granularity: one row per (group_col, date_col).

        - Sums the value column.
        - Optionally computes a weighted average price using product_unit_price_col.
        - Carries dimension columns using first non-null value.
        """
        c = self.config

        agg_exprs = [
            F.sum(c.value_col).alias(c.value_col),
        ]

        # calculate weighted average unit price for each period
        if (
            c.product_unit_price_col is not None
            and c.product_unit_price_col in df.columns
        ):
            agg_exprs.append(
                (
                    F.sum(F.col(c.product_unit_price_col) * F.col(c.value_col))
                    / F.sum(F.col(c.value_col))
                ).alias(c.product_wavg_unit_price_col)
            )

        # Dimension columns: use FIRST() per group to carry them through
        for dim_col in c.dim_cols:
            if dim_col in df.columns:
                agg_exprs.append(F.first(dim_col).alias(dim_col))

        df_agg = df.groupBy(c.date_col, c.group_col).agg(*agg_exprs)
        return df_agg.orderBy(c.date_col, c.group_col)

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

        df_inact_rem = flag_data_prior_to_inactive_periods(
            df_dates_filled,
            c.value_col,
            c.date_col,
            c.group_col,
            c.inactive_threshold,
            drop_pre_inactive=False,
        )

        df_insuff = remove_insuff_series(
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

        return fill_dim_nulls_groupwise(df, c.group_col, c.date_col, cols_to_fill)

    def remove_outliers(self, df: DataFrame) -> DataFrame:
        """
        Replace outliers in selected columns with NULLs.
        """
        c = self.config
        return groupwise_clean_flag_outliers(
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
        return interpolate_groupwise_numeric(
            df,
            c.group_col,
            c.cols_for_interpolation,
            c.date_col,
            c.interpolation_method,
            c.interpolation_order,
        )

    def flag_short_series(self, df: DataFrame) -> DataFrame:
        """
        Add series_length and is_short_series flags based on short_series_threshold.
        """
        c = self.config

        series_lengths = df.groupBy(c.group_col).agg(
            F.count(c.date_col).alias("series_length")
        )

        df_with_len = df.join(
            series_lengths,
            on=c.group_col,
            how="left",
        )

        df_with_flag = df_with_len.withColumn(
            "is_short_series",
            F.col("series_length") < F.lit(c.short_series_threshold),
        )

        return df_with_flag

    def apply_boxcox_and_median(self, df: DataFrame) -> DataFrame:
        """
        Apply Box-Cox transform (using helper) and add per-series median of the INTERPOLATED values.
        """
        c = self.config

        transformed = boxcox_multi_ts_sps(
            df,
            group_col=c.group_col,
            value_col="y_clean_int",
            date_col=c.date_col,
        )

        window_spec = Window.partitionBy(c.group_col)
        transformed = transformed.withColumn(
            "series_median",
            F.expr("percentile_approx(y_clean_int, 0.5)").over(window_spec),
        )

        return transformed

    def write_output_table(self, df: DataFrame) -> None:
        """
        Add a table_update_datetime column and write out to the configured table.
        """
        c = self.config
        tz = pytz.timezone(c.timezone_name)
        table_update_date = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        df_with_update = df.withColumn(
            "table_update_datetime", F.lit(table_update_date)
        )

        df_with_update.createOrReplaceGlobalTempView("transformed_df_with_update_date")

        self.spark.sql(
            f"""
            CREATE OR REPLACE TABLE {c.output_table_fqn} AS
            SELECT * FROM global_temp.transformed_df_with_update_date
            """
        )
