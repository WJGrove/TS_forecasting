from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType


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
    df, value_col="y", date_col="ds", group_col="time_series_id", inactive_threshold=4
):
    """
    1. Calculates inactivity_threshold_date and series_max_date, creates the is_inactive and rolling_period_sum_inactive fields,
    and determines the activity status of each series.
    2. Remove "obsolete" series.
    3. Keep only the period of "iswas_inactive" series occurring after the most recent occurrence of
       rolling_period_sum_inactive = max_rolling_period_sum_inactive.
    4. Retain all "never_inactive" series.

    :param df: Input DataFrame containing the time series data.
    :param value_col: Column name of the value/order (default 'y').
    :param date_col: Column name of the date (default 'ds').
    :param group_col: Column name of the time series group identifier (default 'time_series_id').
    :param inactive_threshold: Necessary period of inactivity in weeks (default 4).
    :return: DataFrame with the computed fields and activity_status columns.
    """
    # Initial count of distinct time series
    initial_series_count = df.select(group_col).distinct().count()
    print(f"Initial series count: {initial_series_count}")

    # Step 1: Calculate the inactivity_threshold_date
    current_date = datetime.now().date()
    global_max_possible_order_date = (
        df.filter(F.col(date_col) <= F.lit(current_date))
        .agg(F.max(F.col(date_col)).alias("global_max_possible_order_date"))
        .collect()[0]["global_max_possible_order_date"]
    )
    inactivity_threshold_date = global_max_possible_order_date - timedelta(
        weeks=inactive_threshold
    )

    # Define window specifications
    windowSpecForInactivity = (
        Window.partitionBy(F.col(group_col))
        .orderBy(F.col(date_col))
        .rowsBetween(-(inactive_threshold - 1), 0)
    )
    windowSpecForMaxDate = Window.partitionBy(F.col(group_col))

    # Step 2: Calculate series_max_date and series_min_date
    df = df.withColumn("series_max_date", F.max(date_col).over(windowSpecForMaxDate))
    df = df.withColumn("series_min_date", F.min(date_col).over(windowSpecForMaxDate))

    # Step 3: Identify inactive weeks (zeros and NULLs)
    df = df.withColumn(
        "is_inactive",
        F.when(F.col(value_col).isNull() | (F.col(value_col) == 0), 1).otherwise(0),
    )

    # Step 4: Calculate rolling sum of inactive weeks
    df = df.withColumn(
        "rolling_period_sum_inactive",
        F.sum(F.col("is_inactive")).over(windowSpecForInactivity),
    )

    # Step 5: Determine activity_status for each series
    # Calculate the max value of rolling_period_sum_inactive for each series
    max_rolling_sum_window = Window.partitionBy(F.col(group_col))
    df = df.withColumn(
        "max_rolling_period_sum_inactive",
        F.max(F.col("rolling_period_sum_inactive")).over(max_rolling_sum_window),
    )

    # Create activity status flag fields
    df = df.withColumn(
        "is_obsolete", F.col("series_max_date") < F.lit(inactivity_threshold_date)
    )
    df = df.withColumn(
        "iswas_inactive", F.col("max_rolling_period_sum_inactive") >= inactive_threshold
    )
    df = df.withColumn(
        "never_inactive", F.col("max_rolling_period_sum_inactive") < inactive_threshold
    )

    # df.show(30)

    # # Print counts of time series satisfying each criterion
    # obsolete_count = df.filter(F.col("is_obsolete")).select(group_col).distinct().count()
    # print(f"Number of 'obsolete' time series: {obsolete_count}")

    # iswas_inactive_count = df.filter(F.col("iswas_inactive")).select(group_col).distinct().count()
    # print(f"Number of 'iswas_inactive' time series: {iswas_inactive_count}")

    # never_inactive_count = df.filter(F.col("never_inactive")).select(group_col).distinct().count()
    # print(f"Number of 'never_inactive' time series: {never_inactive_count}")

    # Filter out "obsolete" series
    # These series can overlap with the "iswas_inactive" series, so we need to filter them out first.
    df = df.filter(~F.col("is_obsolete"))
    # Count of distinct time series after filtering out "obsolete"
    # series_count_after_obsolete = df.select(group_col).distinct().count()
    # print(f"Series count after filtering out 'obsolete' series: {series_count_after_obsolete}")

    # Separate "iswas_inactive" and "never_inactive" series
    df_iswas_inactive = df.filter(F.col("iswas_inactive"))
    # print(f"iswas_inactive :")
    # df_iswas_inactive.show(30)
    # print()
    df_never_inactive = df.filter(F.col("never_inactive"))
    # Count of distinct time series in each subset
    iswas_inactive_series_count = df_iswas_inactive.select(group_col).distinct().count()
    never_inactive_series_count = df_never_inactive.select(group_col).distinct().count()
    # print(f"Series count of 'iswas_inactive': {iswas_inactive_series_count}")

    # Handle "iswas_inactive" series
    # Identify the most recent occurrence of max_rolling_period_sum_inactive for "iswas_inactive" series

    # Find the most recent date where rolling_period_sum_inactive equals max_rolling_period_sum_inactive
    df_iswas_inactive = df_iswas_inactive.withColumn(
        "max_inactive_date",
        F.max(
            F.when(
                F.col("rolling_period_sum_inactive")
                == F.col("max_rolling_period_sum_inactive"),
                F.col(date_col),
            )
        ).over(Window.partitionBy(F.col(group_col))),
    )
    # print(f"iswas_inactive after adding max_inactive_date :")
    # df_iswas_inactive.show(30)
    # print()

    # Filter out data before the max_inactive_date

    # row_count_before_filter = df_iswas_inactive.count()
    # print(f"Row count before filtering 'iswas_inactive' based on 'max_inactive_date': {row_count_before_filter}")
    df_iswas_inactive_filtered = df_iswas_inactive.filter(
        F.col(date_col) > F.col("max_inactive_date")
    )
    # print(f"iswas_inactive after filtering on max_inactive_date :")
    # df_iswas_inactive_filtered.show(30)
    # print()
    # Row count after filtering based on max_inactive_date
    row_count_after_filter = df_iswas_inactive_filtered.count()
    print(f"Row count of filtered 'iswas_inactive' series: {row_count_after_filter}")

    # Drop the helper column that is no longer needed
    df_iswas_inactive_filtered = df_iswas_inactive_filtered.drop(
        "is_inactive",
        "max_inactive_date",
        "rolling_period_sum_inactive",
        "is_obsolete",
        "iswas_inactive",
        "never_inactive",
        "max_rolling_period_sum_inactive",
    )
    df_never_inactive = df_never_inactive.drop(
        "is_obsolete",
        "iswas_inactive",
        "never_inactive",
        "is_inactive",
        "rolling_period_sum_inactive",
        "max_rolling_period_sum_inactive",
    )

    # Combine the "never_inactive" series and the filtered "iswas_inactive" series
    # Row counts before union operation
    never_inactive_row_count = df_never_inactive.count()
    # print(f"Series count of 'never_inactive': {never_inactive_series_count}")
    print(f"Row count of 'never_inactive' series: {never_inactive_row_count}")

    # Perform the union to combine both DataFrames
    df_filtered = df_never_inactive.unionByName(df_iswas_inactive_filtered)
    # Row count after union
    final_row_count = df_filtered.count()
    print(
        f"Final row count after combining 'never_inactive' and filtered 'iswas_inactive': {final_row_count}"
    )

    return df_filtered


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
    df, group_col, numerical_cols, date_col, interpolation_method="linear", order=3
):
    """
    Interpolates specified columns in a Spark DataFrame for each group, identified by group_col,
    after sorting each group by the date_col. Retains all columns from the input DataFrame.

    :param df: Spark DataFrame to be interpolated.
    :param group_col: Column name that identifies the group.
    :param numerical_cols: List of column names to interpolate.
    :param date_col: Column name by which to sort within each group.
    :param interpolation_method: Method of interpolation (default is 'linear').
    :param order: Order of interpolation for spline or polynomial methods (default is 2).
    :return: Spark DataFrame with interpolated values and all original columns.
    """
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.toPandas()

    # Ensure numerical columns are of float64 type for interpolation
    pandas_df[numerical_cols] = pandas_df[numerical_cols].astype("float64")

    # Define a function to interpolate within each group
    def interpolate_group(group):
        group = group.sort_values(by=date_col).reset_index(drop=True)
        for col in numerical_cols:
            # Interpolate and create a new column with "_int" suffix for the interpolated values
            if interpolation_method in ["spline", "polynomial"]:
                group[f"{col}_int"] = (
                    group[col]
                    .interpolate(method=interpolation_method, order=order)
                    .ffill()
                    .bfill()
                )
            else:
                group[f"{col}_int"] = (
                    group[col].interpolate(method=interpolation_method).ffill().bfill()
                )
        return group

    # Apply interpolation for each group
    interpolated_df = (
        pandas_df.groupby(group_col).apply(interpolate_group).reset_index(drop=True)
    )

    # Convert back to Spark DataFrame
    spark = SparkSession.builder.getOrCreate()
    df_interp = spark.createDataFrame(interpolated_df)

    return df_interp


def upsample_monthlyts_to_weeklyts_spark(df, date_col="ds"):
    """
    Note: This function is written using Sunday as an example, but it doesn't
      matter what day of the week the series is on.

    This function upsamples a REGULAR weekly time series DataFrame to include all Sundays between the first and last dates.

    Parameters:
    df (DataFrame): The input Spark DataFrame with a date column.
    date_col (str): The name of the date column in `df` to be used for upsampling.

    Returns:
    DataFrame: A new DataFrame with weekly frequency on Sundays.
    """
    # Ensure that df has only one row per date
    df = df.dropDuplicates([date_col])
    # create a list of original cols to keep
    original_cols_to_select = [col for col in df.columns if col not in [date_col]]

    # Create a DataFrame with a range of Sundays between min_date and max_date
    all_weeks_df = df.select(
        F.explode(
            F.sequence(
                F.to_date(F.lit(F.min(date_col))),
                F.to_date(F.lit(F.max(date_col))),
                F.expr("interval 7 days"),  # Increment by 7 days to get only Sundays
            )
        ).alias("all_weeks")
    )

    # Perform a left join with the original DataFrame to include all Sundays
    expanded_df = all_weeks_df.join(
        df, all_weeks_df.all_weeks == df[date_col], "left"
    ).select("all_weeks", *original_cols_to_select)
    # Return the expanded DataFrame with weekly frequency
    return expanded_df.orderBy("all_weeks")


# -------------------------------------------------------------------------------
# # Augmented Dickey-Fuller test

# # INVESTIGATE LEVENE'S TEST AND THE ARCH TEST AS POSSIBLES FOR
# # SKEDASTICITY TESTING
# # def test_heteroskedasticity
# -------------------------------------------------------------------------------


def boxcox_multi_ts_sps(
    df,
    group_col="time_series_id",
    date_col="ds",
    value_col="y",
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


# Define the inverse Box-Cox transformation function
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


def plot_aggregated_data(spark_df, group_col1, value_col, group_col2=None):
    """
    Aggregates data in a Spark DataFrame by one or two grouping columns and plots
    a color-coded bar graph of the specified value column.

    Parameters:
    - spark_df: Spark DataFrame
    - group_col1: string, the first grouping column
    - value_col: string, the column whose values are to be aggregated and plotted
    - group_col2: string (optional), the second grouping column

    Output:
    - A color-coded bar graph titled "{value_col} by {group_col1} and {group_col2}"
      with a grid, and labeled axes.
    """
    # Check if the DataFrame is empty
    if spark_df.rdd.isEmpty():
        print("DataFrame is empty, cannot plot.")
        return  # Exit the function

    # Aggregate data
    if group_col2:
        aggregated_df = spark_df.groupBy(group_col1, group_col2).agg(
            F.sum(value_col).alias("sum")
        )
        plot_title = f"{value_col} by {group_col1} and {group_col2}"
    else:
        aggregated_df = spark_df.groupBy(group_col1).agg(F.sum(value_col).alias("sum"))
        plot_title = f"{value_col} by {group_col1}"

    # Convert to Pandas DataFrame for plotting
    pd_df = aggregated_df.toPandas()

    # Plotting
    plt.figure(figsize=(10, 6))
    if group_col2:
        # If group_col2 is provided, we use it to color-code the bars
        pd_df.pivot(index=group_col1, columns=group_col2, values="sum").plot(
            kind="bar", ax=plt.gca()
        )
    else:
        pd_df.plot(kind="bar", x=group_col1, y="sum", ax=plt.gca(), color="skyblue")

    plt.title(plot_title)
    plt.xlabel(group_col1)
    plt.ylabel(value_col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title=group_col2 if group_col2 else "")
    plt.tight_layout()
    plt.show()
