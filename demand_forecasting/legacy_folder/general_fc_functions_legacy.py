from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType

# The following function was reviewed on 12/01/2025. Good to go.

def upsample_groupedweeklyts_spark(df, date_col="ds", group_col="time_series_id"):

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

def remove_data_prior_to_inactive_periods(
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
    ).withColumn(
        "series_min_date", F.min(F.col(date_col)).over(maxdate_window)
    )

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
            F.col("iswas_inactive")
            & (F.col(date_col) <= F.col("max_inactive_date")),
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


# The following function was reviewed on 12/01/2025. Good to go.

def remove_series_w_insuff_data(df, group_col, insufficient_data_threshold, value_col):
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


# The following function was reviewed on 12/01/2025. Good to go.

def fill_dim_nulls_bfill_ffill(df, group_col, date_col, cols_to_fill):
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

# The following function was reviewed on 12/01/2025. Good to go.
# rename to include the word "clean" somewhere.
def spark_remove_outliers(
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



def spark_pandas_interpolate_convert(
    df: DataFrame,
    group_col: str,
    numerical_cols: list[str],
    date_col: str,
    interpolation_method: str = "linear",
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
        raise ValueError(
            f"Columns {missing} not found in DataFrame for interpolation."
        )

    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame.")

    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame.")

    # Build the return schema: original fields + one DoubleType field per "<col>_int"
    original_schema: StructType = df.schema
    int_fields = [
        StructField(f"{col}_int", DoubleType(), nullable=True)
        for col in numerical_cols
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


# The following function must be refactored to avoid converting the whole Spark DataFrame to Pandas. It is not scalable.
# information about which series were transformed and their lambda values needs to be stored for inverse transformation as well as reporting later.
# make spark an explicit dependency argument or use SparkSession.getActiveSession()
def boxcox_multi_ts_sps(
    df,
    group_col="time_series_id",
    date_col="ds",
    value_col="y_clean_int",
    lower_lambda_threshold=0.9,
    upper_lambda_threshold=1.1,
):
    """
    Transforms the specified value column in a Spark DataFrame using the Box-Cox transformation if the estimated
    lambda value is outside a specified threshold range. This function is designed for grouped time series data,
    where each group is identified by a unique value in the group_col. The transformation aims to stabilize variance
    and make the data more closely resemble a normal distribution.
    """
    transformed_df = df.toPandas()

    # Initialize new columns for transformed values, lambda values, and constant flag
    transformed_column_name = f"{value_col}_transformed"
    transformed_df[transformed_column_name] = np.nan
    transformed_df["series_lambda"] = np.nan
    transformed_df["is_constant"] = False

    # Sort DataFrame by group and date for grouped operations
    transformed_df.sort_values(by=[group_col, date_col], inplace=True)

    # Loop through each group to apply Box-Cox where needed
    for group_key, group in transformed_df.groupby(group_col):
        y = group[value_col].values

        # Skip groups with non-positive values
        if any(y <= 0):
            print(f"Skipping {group_col} {group_key} due to non-positive values.")
            transformed_df.loc[group.index, transformed_column_name] = (
                y  # No transformation
            )
            continue

        # Skip groups with constant values and set 'is_constant' flag
        if len(np.unique(y)) == 1:
            # print(f"Skipping {group_col} {group_key} due to constant values. Value: {y[0]}, Length: {len(y)}")
            transformed_df.loc[group.index, transformed_column_name] = y
            transformed_df.loc[group.index, "is_constant"] = True
            continue

        # Estimate the Box-Cox lambda for the group
        transformed_y, fitted_lambda = boxcox(y)

        # Determine if transformation should be applied
        if lower_lambda_threshold < fitted_lambda < upper_lambda_threshold:
            # Do not transform if lambda is within the threshold
            transformed_df.loc[group.index, transformed_column_name] = y
        else:
            # Apply transformation and record lambda
            transformed_df.loc[group.index, transformed_column_name] = transformed_y
            transformed_df.loc[group.index, "series_lambda"] = fitted_lambda

    # Convert back to Spark DataFrame
    transformed_spark_df = spark.createDataFrame(transformed_df)

    return transformed_spark_df

# The following function must be refactored
# It relies on a global variable and schema
# needs newer syntax for clarity (old style grouped map pandas udf currently used)
# it's also detached from the boxcox_multi_ts_sps function above, which writes series lambda info to a dataframe
#
#A more coherent design would be:

#   1. boxcox_multi_ts_sps writes:

#       -value_col → transformed (or original if not transformed).

#       -lambda_col (e.g. "boxcox_lambda") with the fitted lambda or None.

#   2. inverse_boxcox uses a UDF or pandas UDF that reads:

#       -y_hat and lambda_col from the same row.

#       -Applies inv_boxcox when lambda_col is not null.
# That way we don't need global dictionaries and the logic is clearer.

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def inverse_boxcox_transform(pdf):
    time_series_id = pdf["time_series_id"].iloc[
        0
    ]  # Assuming time_series_id is uniform within each group
    lam_info = lambda_values_dict.get(
        time_series_id, {"fitted_lambda": None, "is_transformed": False}
    )

    if lam_info["is_transformed"]:
        lam = lam_info["fitted_lambda"]
        pdf["y_hat"] = pdf["y_hat"].apply(
            lambda x: inv_boxcox(x, lam) if pd.notnull(x) else x
        )
        pdf["y_hat_upper"] = pdf["y_hat_upper"].apply(
            lambda x: inv_boxcox(x, lam) if pd.notnull(x) else x
        )
        pdf["y_hat_lower"] = pdf["y_hat_lower"].apply(
            lambda x: inv_boxcox(x, lam) if pd.notnull(x) else x
        )

    return pdf


# def plot_aggregated_data(spark_df, group_col1, value_col, group_col2=None):
#     """
#     Aggregates data in a Spark DataFrame by one or two grouping columns and plots
#     a color-coded bar graph of the specified value column.

#     Parameters:
#     - spark_df: Spark DataFrame
#     - group_col1: string, the first grouping column
#     - value_col: string, the column whose values are to be aggregated and plotted
#     - group_col2: string (optional), the second grouping column

#     Output:
#     - A color-coded bar graph titled "{value_col} by {group_col1} and {group_col2}"
#       with a grid, and labeled axes.
#     """
#     # Check if the DataFrame is empty
#     if spark_df.rdd.isEmpty():
#         print("DataFrame is empty, cannot plot.")
#         return  # Exit the function

#     # Aggregate data
#     if group_col2:
#         aggregated_df = spark_df.groupBy(group_col1, group_col2).agg(
#             F.sum(value_col).alias("sum")
#         )
#         plot_title = f"{value_col} by {group_col1} and {group_col2}"
#     else:
#         aggregated_df = spark_df.groupBy(group_col1).agg(F.sum(value_col).alias("sum"))
#         plot_title = f"{value_col} by {group_col1}"

#     # Convert to Pandas DataFrame for plotting
#     pd_df = aggregated_df.toPandas()

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     if group_col2:
#         # If group_col2 is provided, we use it to color-code the bars
#         pd_df.pivot(index=group_col1, columns=group_col2, values="sum").plot(
#             kind="bar", ax=plt.gca()
#         )
#     else:
#         pd_df.plot(kind="bar", x=group_col1, y="sum", ax=plt.gca(), color="skyblue")

#     plt.title(plot_title)
#     plt.xlabel(group_col1)
#     plt.ylabel(value_col)
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.legend(title=group_col2 if group_col2 else "")
#     plt.tight_layout()
#     plt.show()
