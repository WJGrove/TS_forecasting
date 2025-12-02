from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import numpy as np, numpy.typing as npt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, DoubleType, BooleanType
from typing import Literal



def upsample_weeklyts_groupwise(df, date_col="ds", group_col="time_series_id"):

    # Get the first and last dates for each group using aggregation
    date_ranges = df.groupBy(group_col).agg(
        F.min(date_col).alias("first_date"), F.max(date_col).alias("last_date")
    )

    # Generate the full sequence of dates for each group
    expanded_dates = date_ranges.withColumn(
        "ds", F.explode(F.expr("sequence(first_date, last_date, interval 1 week)"))
    ).select(group_col, "ds")

    # join with the original DataFrame
    df = expanded_dates.join(df, on=[group_col, date_col], how="left_outer")
    df = df.orderBy(group_col, date_col)

    return df


def flag_data_prior_to_inactive_periods(
    df: DataFrame,
    value_col: str = "y",
    date_col: str = "ds",
    group_col: str = "time_series_id",
    inactive_threshold: int = 4,
    *,
    drop_pre_inactive: bool = False,
) -> DataFrame:
    """
    Mark inactivity patterns within each series and optionally drop historical data
    that precedes the last long inactive period.

    Definitions (per series):

    - is_inactive:
        1 where `value_col` is NULL or 0, else 0.

    - rolling_period_sum_inactive:
        rolling sum of is_inactive over the last `inactive_threshold` rows
        (ordered by `date_col`).

    - max_rolling_period_sum_inactive:
        maximum rolling sum over the entire series (i.e., longest run of inactivity).

    - is_obsolete:
        True if series_max_date is more than `inactive_threshold` weeks older than the
        global max date in the dataset.

    - iswas_inactive:
        True if the series ever has a run of `inactive_threshold` or more consecutive
        inactive periods.

    - never_inactive:
        True if the series never has such a run.

    - precedes_inactive_period:
        For iswas_inactive series, True for all rows with date_col <= max_inactive_date,
        where max_inactive_date is the most recent date at which the rolling sum hits
        its maximum. False for rows after that date. For never_inactive series, False.

    If drop_pre_inactive is True:
        - All rows from obsolete series are dropped.
        - For iswas_inactive series, rows with precedes_inactive_period=True are dropped
          (i.e., you retain only the post-inactive segment).
    """
    if inactive_threshold <= 0:
        raise ValueError("inactive_threshold must be a positive integer")

    # 1) Determine the global max date in the dataset
    max_date_row = df.agg(F.max(F.col(date_col)).alias("max_date")).collect()[0]
    global_max_date = max_date_row["max_date"]

    # If there is no date at all, just return the input
    if global_max_date is None:
        return df

    # Use the global max as the "data horizon" (no dependency on wall-clock time)
    inactivity_cutoff_date = global_max_date - timedelta(weeks=inactive_threshold)

    # 2) Per-series min/max dates
    maxdate_window = Window.partitionBy(group_col)
    df = df.withColumn(
        "series_max_date", F.max(F.col(date_col)).over(maxdate_window)
    ).withColumn("series_min_date", F.min(F.col(date_col)).over(maxdate_window))

    # 3) Inactivity indicator (zeros or NULLs)
    df = df.withColumn(
        "is_inactive",
        F.when(F.col(value_col).isNull() | (F.col(value_col) == 0), F.lit(1)).otherwise(
            F.lit(0)
        ),
    )

    # 4) Rolling sum of inactive weeks over the last `inactive_threshold` periods
    inactivity_window = (
        Window.partitionBy(group_col)
        .orderBy(F.col(date_col))
        .rowsBetween(-(inactive_threshold - 1), 0)
    )

    df = df.withColumn(
        "rolling_period_sum_inactive",
        F.sum(F.col("is_inactive")).over(inactivity_window),
    )

    # 5) Max rolling sum per series
    series_window = Window.partitionBy(group_col)
    df = df.withColumn(
        "max_rolling_period_sum_inactive",
        F.max(F.col("rolling_period_sum_inactive")).over(series_window),
    )

    # 6) Series-level flags
    df = df.withColumn(
        "is_obsolete",
        F.col("series_max_date") < F.lit(inactivity_cutoff_date),
    )

    df = df.withColumn(
        "iswas_inactive",
        F.col("max_rolling_period_sum_inactive") >= F.lit(inactive_threshold),
    )

    df = df.withColumn(
        "never_inactive",
        F.col("max_rolling_period_sum_inactive") < F.lit(inactive_threshold),
    )

    # 7) For iswas_inactive series, find the most recent date where the rolling sum
    #    reaches its maximum -> this defines the last long inactive window.
    df = df.withColumn(
        "max_inactive_date",
        F.max(
            F.when(
                F.col("rolling_period_sum_inactive")
                == F.col("max_rolling_period_sum_inactive"),
                F.col(date_col),
            )
        ).over(series_window),
    )

    # 8) Flag rows that precede that window (including the inactive window itself)
    df = df.withColumn(
        "precedes_inactive_period",
        F.when(
            F.col("iswas_inactive") & (F.col(date_col) <= F.col("max_inactive_date")),
            F.lit(True),
        ).otherwise(F.lit(False)),
    )

    # 9) Optionally drop obsolete series and pre-inactive rows to mimic legacy behavior
    if drop_pre_inactive:
        df = df.filter(
            (~F.col("is_obsolete"))
            & (
                (~F.col("iswas_inactive"))
                | (F.col("precedes_inactive_period") == F.lit(False))
            )
        )

    # 10) Drop heavy intermediate columns that we don't usually need downstream.
    # Keep the main flags and series_min_date/series_max_date.
    df = df.drop(
        "rolling_period_sum_inactive",
        "max_rolling_period_sum_inactive",
        "max_inactive_date",
    )
    df_inact_rem = df
    return df_inact_rem


def remove_insuff_series(df, group_col, insufficient_data_threshold, value_col):
    """
    Removes series from a DataFrame based on a threshold of minimum data points.

    Parameters:
    df (DataFrame): The input DataFrame.
    group_col (str): The name of the column used for grouping.
    insufficient_data_threshold (int): The minimum number of rows required for each series, nulls included.
    value_col (str): The column that represents the series' values to be analyzed.

    Returns:
    DataFrame: The DataFrame with series having insufficient data (as defined by the insufficient_data_threshold) removed.
    """
    if insufficient_data_threshold <= 0:
        raise ValueError("The insufficient_data_threshold must be a positive integer.")

    if value_col is None:
        raise ValueError("'value_col' must be specified.")

    # Aggregate to compute the lengths
    series_length_df = df.groupBy(group_col).agg(
        F.count(F.col(value_col)).alias("series_length")
    )

    # Filter out series with series_length less than the specified threshold
    valid_series = series_length_df.filter(
        F.col("series_length") >= insufficient_data_threshold
    ).select(group_col)

    # Join back with the original DataFrame to filter out the insufficient series
    df_filtered = df.join(valid_series, on=group_col, how="inner")

    return df_filtered


def fill_dim_nulls_groupwise(df, group_col, date_col, cols_to_fill):
    """
    Efficiently fill null values in specified columns using both backward fill and forward fill methods.

    :param df: Spark DataFrame
    :param group_col: Column name of the time series group identifier
    :param date_col: Column name for the date
    :param cols_to_fill: List of column names to fill nulls
    :return: DataFrame with nulls filled
    """
    # Define window specifications
    windowSpec = (
        Window.partitionBy(group_col)
        .orderBy(F.col(date_col))
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    windowSpecDesc = (
        Window.partitionBy(group_col)
        .orderBy(F.col(date_col).desc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    # Combine backward and forward fill using coalesce
    for col in cols_to_fill:
        bfill_col = F.last(F.col(col), ignorenulls=True).over(windowSpec)
        ffill_col = F.first(F.col(col), ignorenulls=True).over(windowSpecDesc)
        df = df.withColumn(col, F.coalesce(bfill_col, ffill_col, F.col(col)))

    return df


def groupwise_clean_flag_outliers(
    df,
    cols_for_outlier_removal,
    group_col="time_series_id",
    outlier_threshold=3.0,
    absolute_threshold=1e5,
):
    """
    Removes outliers from specified columns in a PySpark DataFrame based on Z-score and absolute thresholds.

    Outliers are replaced with None, and a flag column is added to indicate whether each
    value is considered an outlier. The function operates within groups defined by the
    specified `group_col`, allowing for group-wise outlier detection and removal.

    Parameters:
    - df: input SQL DataFrame.
    - cols_for_outlier_removal (list): A list of column names (strings) for which outliers will be detected.
    - group_col (str, optional): The column name used to define groups for calculating Z-scores. Defaults to "time_series_id".
    - outlier_threshold (int, optional): The threshold for Z-scores above which a value is
      considered an outlier. It defaults to 3.
    - absolute_threshold (float, optional): The absolute value threshold is to account for scenarios that could cause the z-score calculation to be     less useful for identifying extreme values. It defaults to 100,000.

    Returns:
    - pyspark.sql.dataframe.DataFrame: A DataFrame with outliers flagged and removed as specified.
    """

    # Check if group_col exists in the DataFrame
    if group_col not in df.columns:
        raise ValueError(f"There's no column named '{group_col}' in the DataFrame.")

    for col in cols_for_outlier_removal:
        # Define a window for each group
        windowSpec = Window.partitionBy(group_col)

        # Calculate mean and standard deviation within each group
        df = df.withColumn(f"{col}_mean", F.avg(col).over(windowSpec))
        df = df.withColumn(f"{col}_stddev", F.stddev(col).over(windowSpec))

        # Calculate Z-score with handling for zero or near-zero standard deviation
        df = df.withColumn(
            f"{col}_zscore",
            F.when(
                F.col(f"{col}_stddev") > 1e-5,  # Ensure stddev is not zero or near-zero
                (F.col(col) - F.col(f"{col}_mean")) / F.col(f"{col}_stddev"),
            ).otherwise(
                F.lit(0)
            ),  # Set Z-score to 0 where stddev is zero or near-zero
        )

        # Flag any values with |Z-score| > outlier_threshold or > absolute_threshold
        df = df.withColumn(
            f"{col}_is_outlier",
            F.when(
                (
                    (F.abs(F.col(f"{col}_zscore")) > outlier_threshold)
                    & F.col(f"{col}_zscore").isNotNull()
                )
                | (
                    F.abs(F.col(col)) > absolute_threshold
                ),  # Secondary absolute threshold
                F.lit(True),
            ).otherwise(F.lit(False)),
        )

        # Replace outliers with None based on the conditions
        df = df.withColumn(
            f"{col}_clean",
            F.when(F.col(f"{col}_is_outlier") == True, None).otherwise(F.col(col)),
        )

        # Drop intermediate columns used for Z-score calculation
        df = df.drop(f"{col}_mean", f"{col}_stddev", f"{col}_zscore")

    return df


# Define a type for interpolation methods
InterpMethod = Literal["linear", "spline", "polynomial"] # spelling these out to satisfy Pylance in the "else" line below.

def interpolate_groupwise_numeric(
    df: DataFrame,
    group_col: str,
    numerical_cols: list[str],
    date_col: str,
    interpolation_method: InterpMethod = "linear",
    order: int = 3,
) -> DataFrame:
    """
    Interpolates specified numeric columns in a Spark DataFrame for each group,
    identified by group_col, after sorting each group by date_col.

    - Keeps all original columns.
    - Adds one new column per numeric col with "_int" suffix containing the
      interpolated values (float64).

    Implementation uses groupBy(...).applyInPandas(...) so interpolation is done
    per-group in parallel, without converting the entire DataFrame to pandas.
    """

    # Basic validation
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in DataFrame for interpolation.")

    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame.")

    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame.")

    # Build the return schema: original fields + one DoubleType field per "<col>_int"
    original_schema: StructType = df.schema
    int_fields = [
        StructField(f"{col}_int", DoubleType(), nullable=True) for col in numerical_cols
    ]
    result_schema = StructType(list(original_schema.fields) + int_fields)

    # Define the per-group interpolation function in pandas
    def _interpolate_group(pdf: pd.DataFrame) -> pd.DataFrame:

        # Sort by date within each group; keep all columns
        pdf = pdf.sort_values(by=date_col).reset_index(drop=True)
        # Ensure numeric columns are float64 for interpolation
        pdf[numerical_cols] = pdf[numerical_cols].astype("float64")

        for col in numerical_cols:
            series = pdf[col]
            if interpolation_method in ("spline", "polynomial"):
                interp = (
                    series.interpolate(
                        method=interpolation_method,
                        order=order,
                    )
                    .ffill()
                    .bfill()
                )
            else:
                interp = series.interpolate(method=interpolation_method).ffill().bfill()

            pdf[f"{col}_int"] = interp

        # Ensure column order matches result_schema
        return pdf[[field.name for field in result_schema.fields]]

    # Apply the interpolation per group
    interpolated_df = df.groupBy(group_col).applyInPandas(
        _interpolate_group,
        schema=result_schema,
    )

    return interpolated_df


# -------------------------------------------------------------------------------
# # FUTURE FUNCTIONS TO CONSIDER ADDING
# -------------------------------------------------------------------------------
# # I probably need a function for the Augmented Dickey-Fuller test

# # INVESTIGATE LEVENE'S TEST AND THE ARCH TEST AS POSSIBLES FOR
# # SKEDASTICITY TESTING
# -------------------------------------------------------------------------------


# make spark an explicit dependency argument or use SparkSession.getActiveSession()
def boxcox_transform_groupwise(
    df: DataFrame,
    group_col: str = "time_series_id",
    date_col: str = "ds",
    value_col: str = "y_clean_int",
    lower_lambda_threshold: float = 0.9,
    upper_lambda_threshold: float = 1.1,
) -> DataFrame:
    """
    Apply Box-Cox per series (group_col) to value_col, in a scalable way.

    - Operates per-group using groupBy(...).applyInPandas(...)
    - Adds:
        * `<value_col>_transformed` : transformed or original values
        * `series_lambda`           : fitted lambda where transform applied, else null
        * `is_constant`             : True if series is constant (and left untransformed)
    """

    # Basic validation
    for col in (group_col, date_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Build output schema: all original fields + 3 new fields
    original_schema: StructType = df.schema
    new_fields = [
        StructField(f"{value_col}_transformed", DoubleType(), nullable=True),
        StructField("series_lambda", DoubleType(), nullable=True),
        StructField("is_constant", BooleanType(), nullable=True),
    ]
    result_schema = StructType(list(original_schema.fields) + new_fields)

    def _boxcox_group(pdf: pd.DataFrame) -> pd.DataFrame:
        # Sort by date within group
        pdf = pdf.sort_values(by=date_col).reset_index(drop=True)

        # Make y a well-typed ndarray[float64] so numpy stubs are happier
        y: npt.NDArray[np.float64] = pdf[value_col].to_numpy(dtype="float64")
        n = y.shape[0]

        transformed = np.full(n, np.nan, dtype="float64")
        lambdas = np.full(n, np.nan, dtype="float64")
        is_const = np.zeros(n, dtype=bool)

        # Valid = non-NaN
        valid_mask = np.logical_not(np.isnan(y))
        y_valid = y[valid_mask]

        if y_valid.size == 0:
            # All nulls: leave as NaN, no lambda
            pass

        elif np.any(y_valid <= 0):
            # Non-positive values → cannot Box-Cox; keep original
            transformed[valid_mask] = y_valid

        elif np.unique(y_valid).size == 1:
            # Constant series → keep original, flag as constant
            transformed[valid_mask] = y_valid
            is_const[valid_mask] = True

        else:
            # Proper Box-Cox
            # SciPy's stub is overly general; treat result as a generic tuple
            bc_res = boxcox(y_valid)  # type: ignore[misc]
            y_trans = np.asarray(bc_res[0], dtype="float64")
            lam = float(bc_res[1])

            if lower_lambda_threshold < lam < upper_lambda_threshold:
                # Near-identity lambda → don't transform
                transformed[valid_mask] = y_valid
            else:
                transformed[valid_mask] = y_trans
                lambdas[valid_mask] = lam

        pdf[f"{value_col}_transformed"] = transformed
        pdf["series_lambda"] = lambdas
        pdf["is_constant"] = is_const

        # Enforce column order to match result_schema
        return pdf[[field.name for field in result_schema.fields]]

    result_df = df.groupBy(group_col).applyInPandas(
        _boxcox_group,
        schema=result_schema,
    )
    return result_df


# When you forecast a transformed series, you need to inverse transform the predictions and the prediction interval endpoints.
# This means you'll need to join the lambda info to the forecast data on the group_col.

import numpy as np
import pandas as pd

from scipy.special import inv_boxcox  # make sure scipy is installed
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType


def groupwise_inv_boxcox_transform(
    df: DataFrame,
    group_col: str = "time_series_id",
    lambda_col: str = "series_lambda",
    cols_to_inverse: list[str] | None = None,
    constant_flag_col: str | None = "is_constant",
) -> DataFrame:
    """
    Apply the inverse Box-Cox transform per series, using the lambda stored in `lambda_col`.

    Intended usage:
      - `df` contains:
          * one or more forecast columns on Box-Cox scale (e.g. y_hat_bc, y_hat_bc_lower, y_hat_bc_upper)
          * a per-row lambda column (e.g. 'series_lambda') produced by the forward
            Box-Cox step (one lambda per series).
      - For each column in `cols_to_inverse`:
          * If both the value and lambda are non-null, apply inv_boxcox(value, lambda).
          * Otherwise, leave the value unchanged.

    This function modifies the specified columns **in place** and returns a DataFrame
    with the same schema as the input.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame with forecast columns and a lambda column.
    group_col : str
        Name of the column that identifies a time series (e.g. 'time_series_id').
    lambda_col : str
        Name of the column containing the Box-Cox lambda for each row (e.g. 'series_lambda').
    cols_to_inverse : list[str]
        List of column names (on Box-Cox scale) to inverse transform back to original units.

    Returns
    -------
    DataFrame
        DataFrame with the same schema as `df`, with the specified columns inverse-transformed
        where a valid lambda is present.
    """
    # ---------- Defensive checks ----------
    if cols_to_inverse is None or len(cols_to_inverse) == 0:
        raise ValueError(
            "cols_to_inverse must be a non-empty list of column names to inverse-transform."
        )

    missing_required = [c for c in (group_col, lambda_col) if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Required column(s) {missing_required} not found in DataFrame."
        )

    missing_inverse_cols = [c for c in cols_to_inverse if c not in df.columns]
    if missing_inverse_cols:
        raise ValueError(
            f"Columns specified in cols_to_inverse not found in DataFrame: {missing_inverse_cols}"
        )

    # We'll preserve the exact schema and column order
    original_schema: StructType = df.schema

    def _inv_group(pdf: pd.DataFrame) -> pd.DataFrame:
        # Lambda as float64 array
        lam: npt.NDArray[np.float64] = pdf[lambda_col].to_numpy(dtype="float64")

        # Optional constant flag (to skip fully constant series)
        if constant_flag_col and constant_flag_col in pdf.columns:
            const_flags: npt.NDArray[np.bool_] = (
                pdf[constant_flag_col].astype(bool).to_numpy()
            )
        else:
            const_flags = np.zeros(len(pdf), dtype=bool)

        # Precompute "not constant"
        not_const: npt.NDArray[np.bool_] = np.logical_not(const_flags)

        for col in cols_to_inverse:
            # Values on Box-Cox scale
            vals: npt.NDArray[np.float64] = pdf[col].to_numpy(dtype="float64")

            valid_vals = np.logical_not(np.isnan(vals))
            valid_lam = np.logical_not(np.isnan(lam))

            # Only inverse where we have a value, a lambda, and the row is not constant
            mask: npt.NDArray[np.bool_] = valid_vals & valid_lam & not_const

            if mask.any():
                vals[mask] = inv_boxcox(vals[mask], lam[mask])

            pdf[col] = vals

        return pdf[[field.name for field in original_schema.fields]]

    # Use applyInPandas so Spark handles grouping + distribution
    result_df = df.groupBy(group_col).applyInPandas(
        _inv_group,
        schema=original_schema,
    )

    return result_df
