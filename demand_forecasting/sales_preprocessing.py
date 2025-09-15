# Databricks notebook source
# MAGIC %md
# MAGIC #Installs, Functions, and Inputs

# COMMAND ----------

# MAGIC %md
# MAGIC ##Installs/Imports

# COMMAND ----------

# MAGIC %md
# MAGIC >The cell below can be replaced with a job task to run the imports notebook.

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Forecast Files/demand_forecasting/general_forecasting_imports"

# COMMAND ----------

t0 = time.time()
print("start time: ", t0)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Functions

# COMMAND ----------

# MAGIC %md
# MAGIC >The cell below can be replaced with a job task to run the functions notebook.

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Forecast Files/demand_forecasting/general_forecasting_functions"

# COMMAND ----------

# FOR DEBUGGING

def find_duplicate_dates(df, time_series_col, date_col, index_col=None):
    """
    Identifies duplicate dates in a DataFrame grouped by a time series identifier
    and optionally an index column. Prints the count of duplicates per series and
    the rows where the duplicates occur.

    Parameters:
        df (DataFrame): The input Spark DataFrame.
        time_series_col (str): The column name representing the time series identifier.
        date_col (str): The column name representing the date field.
        index_col (str, optional): The column name representing the index field.
                                   If None, the index field is ignored.
    """
    # Group by time series and date, optionally including the index field
    group_cols = [time_series_col, date_col]
    if index_col:
        group_cols.append(index_col)
    
    # Count occurrences of each (time_series_id, ds, [index_col]) grouping
    duplicate_counts = df.groupBy(*group_cols).count().filter(F.col("count") > 1)
    
    # Count total number of series in the input DataFrame
    total_series = df.select(time_series_col).distinct().count()
    print(f"Total number of series in the DataFrame: {total_series}")
    
    # Count the number of series that have duplicates
    series_with_duplicates = duplicate_counts.select(time_series_col).distinct().count()
    print(f"Number of series with duplicate dates: {series_with_duplicates}")


    # Count duplicates per series
    duplicates_per_series = duplicate_counts.groupBy(time_series_col).agg(
        F.count("*").alias("duplicate_count")
    )
    
    # Show the counts of duplicates per series
    print("Count of duplicate dates per series:")
    duplicates_per_series.show(truncate=False)
    
    # Find and display the rows with duplicate dates
    if index_col:
        print("Rows with duplicate dates (including index):")
        duplicates = df.join(
            duplicate_counts.select(time_series_col, date_col, index_col),
            on=group_cols,
            how="inner"
        )
    else:
        print("Rows with duplicate dates (excluding index):")
        duplicates = df.join(
            duplicate_counts.select(time_series_col, date_col),
            on=group_cols[:2],  # Only time_series_col and date_col
            how="inner"
        )
    
    duplicates = duplicates.orderBy(time_series_col, date_col)
    duplicates.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variables/Inputs

# COMMAND ----------

local_timezone = pytz.timezone("America/Chicago")

forecasting_files_save_location = "forecast_dev.data_science"

value_col = 'y' # origin: 'ordered_qty_fc'
date_col = 'ds' # origin: 'req_del_fw_start_date'
group_col = 'time_series_id' # origin: soldto_id + item_id_fc I.e., distribution center and item.

cols_for_outlier_removal = ["y"] # "y_clean" column will be created
numerical_cols = ["y","y_clean", "gross_price_fc"]
cols_for_interpolation = [col for col in numerical_cols if col not in [value_col]] # we'll interpolate y_clean instead 'y'

interpolation_method = 'linear' # 'polynomial' 'spline'
interpolation_order = 3  # only applies when interpolation_method is 'polynomial' or 'spline'.
seasonal_period = 52 

# COMMAND ----------

inactive_threshold = 4 # Any series ending with this number of consecutive zeros is discarded.
# If it reactivates, only the new portion of the series persists. Everything before that point is filtered out.
insufficient_data_threshold = 1 # Any series with less than this number of observations (weeks) is discarded.(Default state is 1.)
short_series_threshold = 52 # (weeks) Short series are those having less than this number of observations. They can be difficult to predict (very little data for something with a 52-week seasonality), but they generally only account for a small fraction of the volume.
outlier_threshold = 3 # standard deviations

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve and Interpolate Data

# COMMAND ----------

# Grab sales data
df = sql("""
    SELECT *
    FROM forecast_dev.data_science.sales_forecast_source_view
""")

df.show(5)
print(f"'df' length = {df.count()}")

# COMMAND ----------

# Remove duplicate records.
df = df.dropDuplicates()
# Rename columns as needed
df = df.withColumnRenamed("ordered_qty_fc", "y")
# Measuring demand, we use the intended delivery week, not the order week:
df = df.withColumnRenamed("req_del_fw_start_date", "ds")

print(f"'df' length after dropping duplicates= {df.count()}")

# COMMAND ----------

# # NULL check
# null_counts = df.agg(
#     F.count(F.when(F.col('parent_company_fc').isNull(), 1)).alias('nulls_in_parent_company_fc'),
#     F.count(F.when(F.col('soldto_id').isNull(), 1)).alias('nulls_in_soldto_id'),
#     F.count(F.when(F.col('item_id_fc').isNull(), 1)).alias('nulls_in_item_id_fc'),
#     F.count(F.when(F.col('gross_price_fc').isNull(), 1)).alias('nulls_in_gross_price_fc'),
#     F.count(F.when(F.col('y').isNull(), 1)).alias('nulls_in_y'),
#     F.count(F.when(F.col('y') == 0, 1)).alias('y_is_zero')
# )
# null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Aggregate (Define Granularity)

# COMMAND ----------

# Aggregate data by 'time_series_id' and 'ds'
df_weekly = df.groupBy("ds", "time_series_id").agg(
    F.sum("y").alias("y"),
    # Weighted average of 'gross_price_fc' using 'y' as weights
    (F.sum(F.col("gross_price_fc") * F.col("y")) / F.sum(F.col("y"))).alias("gross_price_fc"), 
    F.first("parent_company_fc").alias("parent_company_fc"),
    F.first("soldto_id").alias("soldto_id"),
    F.first("item_id_fc").alias("item_id_fc"),
    F.first("bottle_type").alias("bottle_type")
)

df_weekly = df_weekly.orderBy("ds", "time_series_id")
# df_weekly.show(10)
print(f"'df_weekly' length = {df_weekly.count()}")

# COMMAND ----------

# FOR DEBUGGING

# This function expects multiple time series (a group column).
# find_duplicate_dates(df_weekly, "time_series_id", "ds")

df_weekly.filter(F.col("time_series_id") == "ALDI655160000").show(150, truncate=False)


# COMMAND ----------

# # NULL check
# null_counts = df_weekly.agg(
#     F.count(F.when(F.col('parent_company_fc').isNull(), 1)).alias('nulls_in_parent_company_fc'),
#     F.count(F.when(F.col('soldto_id').isNull(), 1)).alias('nulls_in_soldto_id'),
#     F.count(F.when(F.col('item_id_fc').isNull(), 1)).alias('nulls_in_item_id_fc'),
#     F.count(F.when(F.col('gross_price_fc').isNull(), 1)).alias('nulls_in_gross_price_fc'),
#     F.count(F.when(F.col('y').isNull(), 1)).alias('nulls_in_y'),
#     F.count(F.when(F.col('y') == 0, 1)).alias('y_is_zero')
# )
# null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Complete Timelines and interpolate

# COMMAND ----------

# MAGIC %md
# MAGIC >### This section adds dummy records to fill gaps in the timelines.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Add Missing Dates

# COMMAND ----------

# fill in missing dates
df_dates_filled  = upsample_groupedweeklyts_spark(df_weekly, date_col, group_col)
# df_dates_filled.show(10)
print(f"'df_dates_filled' length = {df_dates_filled.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Remove Inactive Series and Those with Insufficient Data

# COMMAND ----------

# MAGIC %md
# MAGIC > This section removes any part of any series preceding an inactive period of weeks equal to the 'inactive_threshold' defined at the top of the notebook. E.g., if a dc stops ordering an item for four weeks and then begins ordering that item again, the data from before the pause is filtered out. 

# COMMAND ----------

# # This cell creates a custom test set with the same col headers as df_dates_filled.

# from decimal import Decimal
# # Define the schema of the test set
# schema = StructType([
#     StructField("ds", DateType(), True),
#     StructField("time_series_id", StringType(), True),
#     StructField("parent_company_fc", StringType(), True),
#     StructField("soldto_id", StringType(), True),
#     StructField("item_id_fc", StringType(), True),
#     StructField("gross_price_fc", DecimalType(19, 2), True),
#     StructField("bottle_type", StringType(), True),
#     StructField("y", DecimalType(38, 7), True)
# ])

# # Add data for the three test series with 10 weekly observations each
# data = [
#     # Series 1 (zeros for observations 5-10)
#     (datetime.strptime("2024-01-01", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("10.0000000")),
#     (datetime.strptime("2024-01-08", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("15.0000000")),
#     (datetime.strptime("2024-01-15", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("20.0000000")),
#     (datetime.strptime("2024-01-22", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("25.0000000")),
#     (datetime.strptime("2024-01-29", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-05", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-12", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-19", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-26", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("0.0000000")),
#     (datetime.strptime("2024-03-04", "%Y-%m-%d").date(), "1", "Company A", "1001", "Item1", Decimal("25.00"), "Type1", Decimal("0.0000000")),
    
#     # Series 2 (zeros for observations 3-9)
#     (datetime.strptime("2024-01-01", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("5.0000000")),
#     (datetime.strptime("2024-01-08", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("10.0000000")),
#     (datetime.strptime("2024-01-15", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-01-22", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-01-29", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-05", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-12", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-19", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-02-26", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("0.0000000")),
#     (datetime.strptime("2024-03-04", "%Y-%m-%d").date(), "2", "Company B", "1002", "Item2", Decimal("30.00"), "Type2", Decimal("15.0000000")),

#     # Series 3 (all positive integers with an outlier)
#     (datetime.strptime("2024-01-01", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-01-08", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-01-15", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-01-22", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-01-29", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("5000000000.0000000")),
#     (datetime.strptime("2024-02-05", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-02-12", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-02-19", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-02-26", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000")),
#     (datetime.strptime("2024-03-04", "%Y-%m-%d").date(), "3", "Company C", "1003", "Item3", Decimal("35.00"), "Type3", Decimal("1.0000000"))
# ]

# # Create DataFrame
# test_df_dates_filled = spark.createDataFrame(data, schema)

# # Show the DataFrame
# test_df_dates_filled.show(30)


# COMMAND ----------

# Remove series data prior to any period of inactivity longer than the specified threshold.
df_inact_rem = remove_data_prior_to_inactive_periods(df_dates_filled, value_col
                                                    , date_col, group_col, inactive_threshold)
print()
# df_inact_rem.show(10) 
print(f"'df_inact_rem' length = {df_inact_rem.count()}")

# COMMAND ----------

# Remove series with fewer than the specified threshold for the number of observations.
df_insuff_data_rem = remove_series_w_insuff_data(df_inact_rem, group_col, insufficient_data_threshold, value_col)
# df_insuff_data_rem.show(10) 
print(f"'df_insuff_data_rem' length = {df_insuff_data_rem.count()}")

# COMMAND ----------

# NULL check
null_counts = df_insuff_data_rem.agg(
    F.count(F.when(F.col('parent_company_fc').isNull(), 1)).alias('nulls_in_parent_company_fc'),
    F.count(F.when(F.col('soldto_id').isNull(), 1)).alias('nulls_in_soldto_id'),
    F.count(F.when(F.col('item_id_fc').isNull(), 1)).alias('nulls_in_item_id_fc'),
    F.count(F.when(F.col('gross_price_fc').isNull(), 1)).alias('nulls_in_gross_price_fc'),
    F.count(F.when(F.col('y').isNull(), 1)).alias('nulls_in_y'),
    F.count(F.when(F.col('y') == 0, 1)).alias('y_is_zero')
)
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Fill in Dimensions

# COMMAND ----------

# Fill in dimensions for dummy records.
cols_excluded_from_filling = [date_col, group_col] + numerical_cols
cols_to_fill = [col for col in df_dates_filled.columns if col not in cols_excluded_from_filling]


df_dims_filled = fill_dim_nulls_bfill_ffill(df_insuff_data_rem, group_col, date_col, cols_to_fill)
# df_dims_filled.show(10)
print(f"'df_dims_filled' length = {df_dims_filled.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the only columns that should contain NULLs are those about to be interpolated.

# COMMAND ----------

# NULL check
null_counts = df_dims_filled.agg(
    F.count(F.when(F.col('parent_company_fc').isNull(), 1)).alias('nulls_in_parent_company_fc'),
    F.count(F.when(F.col('soldto_id').isNull(), 1)).alias('nulls_in_soldto_id'),
    F.count(F.when(F.col('item_id_fc').isNull(), 1)).alias('nulls_in_item_id_fc'),
    F.count(F.when(F.col('gross_price_fc').isNull(), 1)).alias('nulls_in_gross_price_fc'),
    F.count(F.when(F.col('y').isNull(), 1)).alias('nulls_in_y'),
    F.count(F.when(F.col('y') == 0, 1)).alias('y_is_zero')
)
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Flag and Remove Outliers

# COMMAND ----------

# Remove the outliers (replacing with NULLs).
df_out_rem = spark_remove_outliers(df_dims_filled, cols_for_outlier_removal, outlier_threshold=outlier_threshold, group_col="time_series_id")
# df_out_rem.show(10)
print(f"'df_out_rem' length = {df_out_rem.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Interpolate

# COMMAND ----------

interpolated_df = spark_pandas_interpolate_convert(df_out_rem, group_col, cols_for_interpolation, date_col, interpolation_method, interpolation_order)
# interpolated_df.show(10)
print(f"'interpolated_df' length = {interpolated_df.count()}")

# COMMAND ----------

# NULL check
null_counts = interpolated_df.agg(
    # F.count(F.when(F.col('parent_company_fc').isNull(), 1)).alias('nulls_in_parent_company_fc'),
    # F.count(F.when(F.col('soldto_id').isNull(), 1)).alias('nulls_in_soldto_id'),
    # F.count(F.when(F.col('item_id_fc').isNull(), 1)).alias('nulls_in_item_id_fc'),
    F.count(F.when(F.col('gross_price_fc').isNull(), 1)).alias('nulls_in_gross_price_fc'),
    F.count(F.when(F.col('gross_price_fc_int').isNull(), 1)).alias('nulls_in_gross_price_fc_int'),
    F.count(F.when(F.col('y').isNull(), 1)).alias('nulls_in_y'),
    # F.count(F.when(F.col('y') == 0, 1)).alias('y_is_zero'),
    F.count(F.when(F.col('y_clean').isNull(), 1)).alias('nulls_in_y_clean'),
    F.count(F.when(F.col('y_clean_int').isNull(), 1)).alias('nulls_in_y_clean_int')
)
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Identify Short (New) Series

# COMMAND ----------

# MAGIC %md
# MAGIC > Shorter series are harder to predict and are sometimes deprioritized for the sake of modeling more voluminous series.
# MAGIC Before we deprioritize them, we need to know how many of them there are and how much volume they represent. The 'short_series_threshold' is defined in the "Variables" section at the top of the notebook.

# COMMAND ----------

# count the lengths of the series.
series_lengths_all = interpolated_df.groupBy('time_series_id').agg(F.count('ds').alias('series_length'))
# add a length column to the df
interpolated_df = interpolated_df.join(
    series_lengths_all.select('time_series_id', 'series_length'),
    on='time_series_id',
    how='left'
)
# add a boolean for 'is_short_series'
interpolated_df = interpolated_df.withColumn(
    'is_short_series',
    F.when(interpolated_df['series_length'] < short_series_threshold, True).otherwise(False)
)

# Count the total number of time series in interpolated_df
total_series_count = series_lengths_all.count()
print(f"Total number of time series: {total_series_count}")

# Create a DataFrame to hold only the short series' data.
short_series_df = interpolated_df.filter(interpolated_df['is_short_series'] == True)

# Count the short series.
short_series_count = short_series_df.select("time_series_id").distinct().count()
print(f"Number of series with <{short_series_threshold} weeks: {short_series_count}")
print()

# Collect short series' lengths for histogram plotting
short_series_lengths = series_lengths_all.filter(series_lengths_all['series_length'] < short_series_threshold)
short_series_lengths_pd = short_series_lengths.toPandas()
# Plot histogram of series lengths
plt.figure(figsize=(12, 6))
plt.hist(short_series_lengths_pd['series_length'], bins=20, alpha=0.75)
plt.xlabel('Length of Series')
plt.ylabel('Frequency')
plt.title(f'Histogram of Lengths for Series with <{short_series_threshold} Weeks')
plt.grid(True)
plt.show()

# # Print short series' details.

# # Identify dimensional columns
# dimensional_cols = [
#     f.name for f in short_series_df.schema.fields if not isinstance(f.dataType, (DoubleType, IntegerType, DateType, DecimalType))
#     ]
# # Create a list of aggregation expressions that just grab the first non-NULL value ion the dim cols for each series.
# agg_exprs = [F.first(F.col(col), ignorenulls=True).alias(col) for col in dimensional_cols]
# # Group by 'time_series_id' and aggregate using the list of expressions
# short_series_dimensional_details_df = short_series_df.groupBy('time_series_id').agg(*agg_exprs)
# print("Dim. Details for Short Series: ")
# short_series_dimensional_details_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC >###How much volume do the short series represent?

# COMMAND ----------

# Calculate the percentage of current volume flow represented by the short series.

# Sum 'y' for short series.
total_y_short_series = short_series_df.agg(F.sum('y_clean').alias('sum_y_short')).collect()[0]['sum_y_short']

# Sum 'y' for all series.
#   We want to compare against current volume flow, not compare to the entire cumulative history of longer
#   series, so we must filter the longer series down to the same dates:
min_date_short_series = short_series_df.agg(F.min('ds').alias('min_date')).collect()[0]['min_date']
filtered_interpolated_df = interpolated_df.filter(interpolated_df['ds'] >= min_date_short_series)
sum_y_filtered_interpolated_df = filtered_interpolated_df.agg(F.sum('y_clean').alias('sum_y_filtered')).collect()[0]['sum_y_filtered']

# Calculate the percentage
percent_of_total_vol = (total_y_short_series / sum_y_filtered_interpolated_df) * 100

if percent_of_total_vol < 5:
    print(Back.GREEN + 
          f"The percent of the last {short_series_threshold} weeks' volume represented by short series is "
          f"{percent_of_total_vol:.2f}%" + Style.RESET_ALL)
elif 5 <= percent_of_total_vol < 9:
    print(Back.MAGENTA + 
          f"The percent of the last {short_series_threshold} weeks' volume represented by short series is {percent_of_total_vol:.2f}%. "
          f"Is the forecast method used for short series appropriate for this volume?" + Style.RESET_ALL)
else:  # percent_of_total_vol >= 9
    print(Back.RED + 
          f"The percent of the last {short_series_threshold} weeks' volume represented by short series is "
          f"{percent_of_total_vol:.2f}%. Confirm that the forecast method used for short series is appropriate for this volume." + Style.RESET_ALL)


# COMMAND ----------

# MAGIC %md
# MAGIC #Visualize

# COMMAND ----------

# MAGIC %md
# MAGIC >## Length and Volume Distributions

# COMMAND ----------

# Trim cols that aren't necessary for visualizations to use Pandas more efficiently
vis_trim_df = interpolated_df.drop('gross_price_fc', 'series_max_date')
# number of series
ts_count = vis_trim_df.select("time_series_id").distinct().count()
print(Style.BRIGHT + f"Total no. of series: {ts_count}" + Style.RESET_ALL)

# COMMAND ----------

# Calculate series lengths for a histogram
series_lengths = vis_trim_df.groupBy('time_series_id').agg(F.count('ds').alias('length')).cache()
# Collect lengths for histogram plotting
lengths_pd = series_lengths.toPandas()
# Plot histogram of series lengths
plt.figure(figsize=(12, 6))
plt.hist(lengths_pd['length'], bins=20, alpha=0.75)
plt.xlabel('Length of Time Series')
plt.ylabel('Frequency')
plt.title('Histogram of All Series Lengths')
plt.grid(True)
plt.show()

# Calculate series volumes for a histogram
series_volumes = vis_trim_df.groupBy('time_series_id').agg(F.sum('y_clean').alias('total_volume')).cache()
# Collect lengths for histogram plotting
volumes_pd = series_volumes.toPandas()
# Plot histogram of series volumes
plt.figure(figsize=(12, 6))
plt.hist(volumes_pd['total_volume'], bins=20, alpha=0.75)
plt.xlabel('Total Volume of Time Series')
plt.ylabel('Frequency')
plt.title('Histogram of All Series Volumes')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC >The next one may seem pointless because- all things equal- of course the longer a series is active the more volume it'll account for. But this still brings value in that it allows us to identify any new series that may be short and hard to forecast, yet worth additional effort due to relatively high early volume.
# MAGIC
# MAGIC >So, we expect to see a non-linearly increasing trend (more skus AND higher volume) with density matching the length hystogram, but we're looking for outstanding volume anywhere under the 52-wk mark to inform our definition of 'short_series_threshold'. 

# COMMAND ----------

# Join the series_lengths and series_volumes DataFrames on time_series_id to plot vol vs len.
combined_df = series_lengths.join(series_volumes, 'time_series_id')
# Collect the combined data for plotting
combined_pd = combined_df.toPandas()
# Plotting
plt.figure(figsize=(12, 6))
plt.scatter(combined_pd['length'], combined_pd['total_volume'], alpha=0.75)
plt.xlabel('Length of Time Series')
plt.ylabel('Total Volume of Time Series')
plt.title('Series Volume vs. Length')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC >## Mix

# COMMAND ----------

# Plot a bar chart of volume by bottle type for the last 52 weeks of the data occuring prior to this week.
# Since our data is at the weekly level - and our fiscal weeks begin on Sundays, we attached all our records to Sundays ('ds').
# 'ds' is also based on required delivery date, so there are future dates in the data. These don't necessarily represent the true volume of those dates because more orders could be placed for those dates between now and then.
# This all means we want to plot the volume for the last 52 weeks of the data occuring prior to last Sunday.

# Get today's date without the time part. Remeber, we're plotting with Pandas.
current_date = pd.Timestamp.now().normalize().date()
# Calculate the number of days to go back to reach last Sunday
days_to_last_sunday = (current_date.weekday() + 1) % 7
# Subtract the offset from current_date to get last Sunday
last_sunday = current_date - pd.Timedelta(days=days_to_last_sunday)

# Filter to exclude records from dates on or after last Sunday and keep only the last 52 weeks
last_52_weeks_df = vis_trim_df.filter((F.col('ds') < F.lit(last_sunday)) &
                                       (F.col('ds') >= F.lit(last_sunday - pd.Timedelta(weeks=52))))

# # Verify last_sunday and number of weeks in the filtered DataFrame (for debugging)
# print("Last Sunday:", last_sunday)
# last_52_weeks_count_pd = last_52_weeks_df.groupBy('ds').count().toPandas()
# print("Number of weeks in last 52 weeks data:", len(last_52_weeks_count_pd['ds'].unique()))

# Aggregate by bottle type within the last 52 weeks and convert to Pandas for plotting
bottle_type_df = last_52_weeks_df.groupBy('bottle_type').agg(F.sum('y_clean_int').alias('total_volume')).orderBy('total_volume', ascending=False)
bottle_type_pd = bottle_type_df.toPandas()

# Plotting total volume by bottle type for the last 52 weeks
plt.figure(figsize=(10, 5))
bottle_type_pd.plot.bar(x='bottle_type', y='total_volume', rot=45, legend=False)
plt.xlabel('Bottle Type')
plt.ylabel('Total Volume')
plt.title('T52W Volume by Bottle Type')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC >## Top Companies

# COMMAND ----------

# Aggregate to the company and bottle class level for a look.
# Make a Pandas DataFrame for plotting.
vis_trim_df_pd = vis_trim_df.groupBy(['ds', 'parent_company_fc', 'bottle_type'])\
                            .agg(F.sum('y_clean_int').alias('y_clean_int'))\
                            .orderBy('ds')\
                            .toPandas()

# Calculate the date 52 weeks prior to last Sunday
last_52_weeks_start_date = last_sunday - pd.Timedelta(weeks=52) # last_sunday was defined previously

# Iterate over each distinct bottle type
for bottle_type in vis_trim_df_pd['bottle_type'].unique():
    
    # Filter DataFrame for the current bottle type and exclude future dates
    bottle_df = vis_trim_df_pd[(vis_trim_df_pd['bottle_type'] == bottle_type) & 
                               (vis_trim_df_pd['ds'] < last_sunday)]
    
    # Step 1: Top ten T52W bar chart
    bottle_df_52_weeks = bottle_df[bottle_df['ds'] >= last_52_weeks_start_date]
    # # DEBUGGING - Calculate and print the number of unique weeks in the 52-week DataFrame for the bar chart
    # num_weeks = bottle_df_52_weeks['ds'].nunique()
    # print(f"Number of unique weeks in the last 52-week data for bar chart (up to last Sunday): {num_weeks}")

    # Identify top ten companies by total volume for this bottle type within the last 52 weeks
    top_companies = bottle_df_52_weeks.groupby('parent_company_fc')['y_clean_int'].sum().nlargest(10).index
    top_bottle_df = bottle_df_52_weeks[bottle_df_52_weeks['parent_company_fc'].isin(top_companies)]
    
    # Plot bar chart of total volume for the top ten companies (within the last 52 weeks)
    plt.figure(figsize=(10, 5))
    top_bottle_df.groupby('parent_company_fc')['y_clean_int'] \
        .sum() \
        .sort_values(ascending=False) \
        .plot(kind='bar', legend=False)
    plt.xlabel('Company')
    plt.ylabel('Total Ordered Quantity')
    plt.title(f"Top 10 Companies by Volume - {bottle_type.title()} (Last 52 Weeks)")
    plt.xticks(rotation=45)
    plt.show()
    

    # Step 2: Line Charts (include only the top five companies and past dates)
    top_five_companies = bottle_df.groupby('parent_company_fc')['y_clean_int'].sum().nlargest(5).index
    top_five_bottle_df = bottle_df[bottle_df['parent_company_fc'].isin(top_five_companies)]

    # Plot time series for the top five companies for the current bottle type
    plt.figure(figsize=(12, 6))
    for grp_id in top_five_bottle_df['parent_company_fc'].unique():
        ts_data = top_five_bottle_df[top_five_bottle_df['parent_company_fc'] == grp_id]
        plt.plot(ts_data['ds'], ts_data['y_clean_int'], label=f'{grp_id}')
    plt.xlabel('Date')
    plt.ylabel('Ordered Quantity')
    plt.title(f"Top 5 Companies - {bottle_type.title()} at the Company Level")
    plt.legend()
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC >## Seasonality

# COMMAND ----------

# Experimenting with 3D line plots for rotating (to look for lags/leads)

line_plt_agg_df = vis_trim_df.groupBy("bottle_type", F.year("ds").alias("year"), F.dayofyear("ds").alias("day_of_year")) \
                             .agg(F.sum("y_clean_int").alias("sum_y_clean_int"))
# Collect the data into a Pandas df
pandas_agg_df = line_plt_agg_df.toPandas()
# define a list bottle_typees
bottle_classes = pandas_agg_df['bottle_type'].unique()

for bottle_class in bottle_classes:
    # Filter the DataFrame for the current bottle_type
    class_df = pandas_agg_df[pandas_agg_df['bottle_type'] == bottle_class]
    
    # Pivot the DataFrame for the current bottle_type
    pivot_df = class_df.pivot(index='day_of_year', columns='year', values='sum_y_clean_int')
    
    # Ensure pivot_df covers the full range of days for each year
    x = pivot_df.columns.values
    y = np.arange(1, 367)  # Days of the year
    pivot_df = pivot_df.reindex(columns=x, index=y, fill_value=np.nan).fillna(method='ffill', axis=0)

    # Plotting
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    years = pivot_df.columns.values
    year_indices = np.arange(len(years))

    for i, year in enumerate(years):
        ys = pivot_df.index
        zs = pivot_df[year]
        ax.plot(np.full(ys.shape, i), ys, zs, label=str(year))

    ax.set_xticks(year_indices)
    ax.set_xticklabels(years)
    ax.set_xlabel('Year')
    ax.set_ylabel('Day of the Year')
    ax.set_zlabel(f'Sum of y_clean_int for {bottle_class}')
    ax.view_init(elev=0, azim=0)
    ax.legend(title='Year', bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(f'3D Line Plot of Interpolated Sales by Year for {bottle_class}')
    plt.show()


# COMMAND ----------

line_plt_agg_df = vis_trim_df.groupBy("bottle_type", F.year("ds").alias("year"), F.dayofyear("ds").alias("day_of_year")) \
                             .agg(F.sum("y_clean_int").alias("sum_y_clean_int"))
# Collect the data into a Pandas df
pandas_agg_df = line_plt_agg_df.toPandas()
# define a list of bottle_typees
bottle_classes = pandas_agg_df['bottle_type'].unique()

for bottle_class in bottle_classes:
    # Filter the DataFrame
    class_df = pandas_agg_df[pandas_agg_df['bottle_type'] == bottle_class]
    
    # Pivot the DataFrame
    pivot_df = class_df.pivot(index='day_of_year', columns='year', values='sum_y_clean_int')
    
    # Ensure pivot_df covers the full range of days for each year
    # No need to define 'x' and 'y' as before since we're not using a 3D plot
    pivot_df = pivot_df.fillna(method='ffill', axis=0)  # Forward fill to handle missing values

    # Plotting
    plt.figure(figsize=(10, 7))
    for year in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[year], label=str(year))

    plt.xlabel('Day of the Year')
    plt.ylabel(f'Sum of y_clean_int for {bottle_class}')
    plt.title(f'Line Plot of Interpolated Sales by Year for {bottle_class}')
    plt.legend(title='Year')
    plt.grid(True)
    plt.xticks(np.arange(0, 367, step=30))  # Adjust x-ticks to show day of year clearly
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Transform Interpolated Data

# COMMAND ----------

# MAGIC %md
# MAGIC ##Box-Cox

# COMMAND ----------

# apply Box-Cox transformation
transformed_df = boxcox_multi_ts_sps(interpolated_df, group_col, value_col='y_clean_int', date_col='ds')

# Add a column to hold the series' medians. This will serve as the forecast value for constant/near-constant series.
windowSpec = Window.partitionBy("time_series_id")
# Calculate the median of y_clean_int for each series and add it as a new column
transformed_df = transformed_df.withColumn(
    "series_median",
    F.expr("percentile_approx(y_clean_int, 0.5)").over(windowSpec)
)

transformed_df.show(5)
print()
print(f"'transformed_df' length = {transformed_df.count()}")

# COMMAND ----------

# NULL check
null_counts = transformed_df.agg(
    # F.count(F.when(F.col('parent_company_fc').isNull(), 1)).alias('nulls_in_parent_company_fc'),
    # F.count(F.when(F.col('soldto_id').isNull(), 1)).alias('nulls_in_soldto_id'),
    # F.count(F.when(F.col('item_id_fc').isNull(), 1)).alias('nulls_in_item_id_fc'),
    F.count(F.when(F.col('gross_price_fc').isNull(), 1)).alias('nulls_in_gross_price_fc'),
    F.count(F.when(F.col('gross_price_fc_int').isNull(), 1)).alias('nulls_in_gross_price_fc_int'),
    F.count(F.when(F.col('y').isNull(), 1)).alias('nulls_in_y'),
    # F.count(F.when(F.col('y') == 0, 1)).alias('y_is_zero'),
    F.count(F.when(F.col('y_clean').isNull(), 1)).alias('nulls_in_y_clean'),
    F.count(F.when(F.col('y_clean_int').isNull(), 1)).alias('nulls_in_y_clean_int'),
    F.count(F.when(F.col('y_clean_int_transformed').isNull(), 1)).alias('nulls_in_y_clean_int_transformed'),
    F.count(F.when(F.col('series_median').isNull(), 1)).alias('nulls_in_series_median')
)
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Create a Temp View and Table

# COMMAND ----------

# Add a table_update_date column and replace the global temporary view with the new df
table_update_date = datetime.now(local_timezone).strftime('%Y-%m-%d %H:%M:%S')
print(f"update datetime: {table_update_date}")

transformed_df_with_update_date = transformed_df.withColumn("table_update_datetime", F.lit(table_update_date))

transformed_df_with_update_date.createOrReplaceGlobalTempView("transformed_df_with_update_date")

# COMMAND ----------

# spark.sql("SHOW TABLES IN global_temp").show(truncate=False)

# COMMAND ----------

# Use SQL to create a table in the catalog
table_name = "sales_preprocessed"
spark.sql(f"""
    CREATE OR REPLACE TABLE {forecasting_files_save_location}.{table_name} AS
    SELECT * FROM global_temp.transformed_df_with_update_date
""")

# COMMAND ----------

t1 = time.time()
print("end time: ", t1)
print("Runtime (sec): ", (t1 - t0))
print("Runtime (min): ", (t1 - t0) / 60)

# COMMAND ----------

# dbutils.notebook.exit ('stop')