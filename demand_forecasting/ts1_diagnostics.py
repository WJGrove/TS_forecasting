from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .ts0_preprocessing import TSPreprocessingConfig


# ---------- Dataclasses for structured outputs ----------

@dataclass
class ShortSeriesStats:
    """
    Summary statistics for short time series.
    All volumes are in the same units as the value column (e.g., cases, units).
    Percentages are 0–100.
    """
    total_series: int
    short_series: int
    short_series_ratio: float  # short_series / total_series

    short_series_volume: float
    total_volume_window: float
    short_series_volume_pct: float  # 0–100

    min_date_short_series: Optional[str]

    warn_level: str
    warn_message: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Main diagnostics class ----------

class TSDiagnostics:
    """
    Layer 2 diagnostics for preprocessed time series data.

    This class is intentionally *read-only*: it computes metrics on the
    preprocessed DataFrame produced by TSPreprocessor, but does not modify it.
    """

    def __init__(
        self,
        spark: SparkSession,
        config: TSPreprocessingConfig,
        preprocessed_df: DataFrame,
    ) -> None:
        self.spark = spark
        self.config = config
        self.df = preprocessed_df

    # ---- Short series diagnostics ----

    def compute_short_series_stats(
        self,
        *,
        value_col: str = "y_clean", # whether to use the cleaned value column or the original column is dependent on the use case
    ) -> ShortSeriesStats:
        """
        Compute summary statistics for the "short" time series classified by the threshold in the config.
        - Uses 'is_short_series' and 'series_length' from TSPreprocessor.flag_short_series().
        - Computes:
          * total number of series
          * number of short series
          * share of series that are short
          * share of volume represented by short series over a comparable time window
        - Classifies the warning level based on config.short_series_vol_warn1/2/3.
        """
        c = self.config
        df = self.df

        required_cols = [c.group_col, c.date_col, value_col, "is_short_series"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for short-series diagnostics: {missing}"
            )

        # Filter to short series
        short_df = df.filter(F.col("is_short_series") == True)

        # Handle the edge case where there are no short series
        if short_df.rdd.isEmpty():
            total_series = df.select(c.group_col).distinct().count()
            return ShortSeriesStats(
                total_series=total_series,
                short_series=0,
                short_series_ratio=0.0,
                short_series_volume=0.0,
                total_volume_window=0.0,
                short_series_volume_pct=0.0,
                min_date_short_series=None,
                warn_level="none",
                warn_message=(
                    "No short series present with the current short_series_threshold; "
                    "short-series handling will not affect the accuracy of the analysis."
                ),
            )

        # Total # of series and # short
        total_series = df.select(c.group_col).distinct().count()
        short_series = short_df.select(c.group_col).distinct().count()
        short_series_ratio = short_series / total_series if total_series else 0.0

        # Sum of y_clean for short series and earliest date where any short series has data
        sums_row = short_df.agg(
            F.sum(F.col(value_col)).alias("sum_short"),
            F.min(F.col(c.date_col)).alias("min_date"),
        ).collect()[0]

        total_y_short = float(sums_row["sum_short"] or 0.0)
        min_date_short = sums_row["min_date"]

        # Filter all series to the same date window as in your legacy code
        window_df = df.filter(F.col(c.date_col) >= F.lit(min_date_short))
        total_row = window_df.agg(
            F.sum(F.col(value_col)).alias("sum_total")
        ).collect()[0]

        total_y_window = float(total_row["sum_total"] or 0.0)

        if total_y_window > 0.0:
            volume_pct = (total_y_short / total_y_window) * 100.0
        else:
            volume_pct = 0.0

        warn_level, warn_message = self._classify_short_series_volume(volume_pct)

        return ShortSeriesStats(
            total_series=total_series,
            short_series=short_series,
            short_series_ratio=short_series_ratio,
            short_series_volume=total_y_short,
            total_volume_window=total_y_window,
            short_series_volume_pct=volume_pct,
            min_date_short_series=str(min_date_short),
            warn_level=warn_level,
            warn_message=warn_message,
        )

    def _classify_short_series_volume(self, pct: float) -> Tuple[str, str]:
        """
        Use the three configured thresholds to classify how scary the
        short-series volume share is.

        The behavior is entirely parameterized by your config values, not hardcoded.
        """
        c = self.config
        w1 = c.short_series_vol_warn1
        w2 = c.short_series_vol_warn2
        w3 = c.short_series_vol_warn3

        if pct < w1:
            level = "low"
            msg = (
                f"Short series represent {pct:.2f}% of volume (< {w1}%). "
                "Impact of short-series handling on overall forecast is likely low."
            )
        elif pct < w2:
            level = "moderate"
            msg = (
                f"Short series represent {pct:.2f}% of volume (between {w1}% and {w2}%). "
                "Review the short-series forecasting approach to confirm it is appropriate."
            )
        elif pct < w3:
            level = "high"
            msg = (
                f"Short series represent {pct:.2f}% of volume (between {w2}% and {w3}%). "
                "Short-series handling may materially affect total forecast; validate assumptions and review methodology."
            )
        else:
            level = "very_high"
            msg = (
                f"Short series represent {pct:.2f}% of volume (>= {w3}%). "
                "Short-series forecasting method is critical to overall forecast accuracy; consider richer models for these series."
            )

        return level, msg

    # ---- Length & volume distributions ----

    def series_length_distribution(self) -> DataFrame:
        """
        Return one row per series with its length (in periods) and short-series flag.
        This is a pure Spark DataFrame for layer-3 plotting / EDA.
        """
        c = self.config

        if "series_length" in self.df.columns and "is_short_series" in self.df.columns:
            # Already computed by TSPreprocessor.flag_short_series()
            return (
                self.df
                .select(c.group_col, "series_length", "is_short_series")
                .dropDuplicates([c.group_col])
            )

        # Fallback: compute lengths and short flag here
        lengths = self.df.groupBy(c.group_col).agg(
            F.count(c.date_col).alias("series_length")
        )
        lengths = lengths.withColumn(
            "is_short_series",
            F.col("series_length") < F.lit(c.short_series_threshold),
        )
        return lengths

    def series_volume_distribution(
        self,
        *,
        value_col: str = "y_clean",
    ) -> DataFrame:
        """
        Return one row per series with its total volume.
        Useful for 'volume by series' plots or Pareto-style charts.
        """
        c = self.config

        if value_col not in self.df.columns:
            raise ValueError(
                f"value_col '{value_col}' not present in diagnostics DataFrame."
            )

        return (
            self.df.groupBy(c.group_col)
            .agg(F.sum(F.col(value_col)).alias("series_volume"))
            .orderBy(F.desc("series_volume"))
        )

    def series_level_summary(
        self,
        *,
        value_col: str = "y_clean",
    ) -> DataFrame:
        """
        Per-series summary stats, including median.

        Returns a Spark DataFrame with:
        - group_col
        - series_length
        - series_volume
        - mean_value
        - stddev_value
        - median_value (percentile_approx)
        - is_short_series
        """
        c = self.config

        if value_col not in self.df.columns:
            raise ValueError(
                f"value_col '{value_col}' not present in diagnostics DataFrame."
            )

        base = self.df.groupBy(c.group_col).agg(
            F.count(c.date_col).alias("series_length"),
            F.sum(F.col(value_col)).alias("series_volume"),
            F.avg(F.col(value_col)).alias("mean_value"),
            F.stddev(F.col(value_col)).alias("stddev_value"),
            F.expr(f"percentile_approx({value_col}, 0.5)").alias("median_value"),
        )

        # Attach short-series flag (reusing precomputed column if available)
        if "series_length" in self.df.columns and "is_short_series" in self.df.columns:
            short_flags = (
                self.df
                .select(c.group_col, "is_short_series")
                .dropDuplicates([c.group_col])
            )
            base = base.join(short_flags, on=c.group_col, how="left")
        else:
            base = base.withColumn(
                "is_short_series",
                F.col("series_length") < F.lit(c.short_series_threshold),
            )

        return base

    # ---- Missingness / null counts ----

    def column_null_counts(self) -> DataFrame:
        """
        Compute null counts and null percentages for each column in the DataFrame.

        Returns a tidy Spark DataFrame:
        - column_name
        - null_count
        - null_pct  (0–1)
        """
        df = self.df

        # Aggregate a single row with null counts for each column
        agg_exprs = [
            F.count(F.when(F.col(col).isNull(), 1)).alias(col)
            for col in df.columns
        ]
        counts_row = df.agg(*agg_exprs)

        # Unpivot into (column_name, null_count)
        n = len(df.columns)
        stack_expr = "stack({}, {}) as (column_name, null_count)".format(
            n,
            ", ".join([f"'{col}', `{col}`" for col in df.columns]),
        )
        result = counts_row.selectExpr(stack_expr)

        total_rows = df.count()
        result = result.withColumn(
            "null_pct",
            F.when(
                F.lit(total_rows) > 0,
                F.col("null_count") / F.lit(total_rows),
            ).otherwise(F.lit(0.0)),
        )

        return result.orderBy(F.desc("null_pct"))
