# Databricks notebook source
# MAGIC %md
# MAGIC >Confirm Mount to Azure Storage Account

# COMMAND ----------

# dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md
# MAGIC #Installs, Functions, and Variables

# COMMAND ----------

# MAGIC %md
# MAGIC ##Installs/Imports

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Imports"

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/Data_Prep_Functions"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Notebook Variables

# COMMAND ----------

file_path = "/mnt/MilosAI_Storage_Container/"

# COMMAND ----------

# MAGIC %md
# MAGIC #CPI Data

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variables

# COMMAND ----------

cpi_file_path = file_path
cpi_file_names =  ['CPI_Food_usCityAvg_2020-max.csv', 'CPI_Food_usSizeClassA_2020-max.csv'
                   , 'CPI_FoodAFH_MidwestSizeClassB&C_2020-max.csv', 'CPI_FoodAFH_usSizeClassB&C_2020-max.csv'
                   , 'CPI_FoodAndBev_All_SizeClassA_2020-max.csv', 'CPI_FoodAndBev_South_SizeClassB&C_2020-max.csv']
cpi_cols_to_drop = ['Series ID', 'Period', 'Year']


# COMMAND ----------

# MAGIC %md
# MAGIC Note that the CPI data is monthly and has no specific day/week attached to it. Since we forecast at the weekly level (Sundays), I'm attaching the CPI numbers to the second Sunday of the corresponding month. This is to get near the middle of the month prior to upsampling/interpolation. 

# COMMAND ----------

# I wrote a function specific to CPI files that are saved as CSVs. It will pull the files, rename some col.s
# , drop some col.s, and join the df's together on the 'year + month' field.
cpi_df = read_and_combine_cpi_csvs(cpi_file_path, cpi_file_names, cpi_cols_to_drop)
# cpi_df.show()
# print(f"'cpi_df' length = {cpi_df.count()}")

# Rename column with special character in header
cpi_df = cpi_df.withColumnRenamed('year + month', 'year_and_month')
# Shorten column names to stop the output from wrapping.
for col in cpi_df.columns:
    if col != 'year_and_month':
        new_col_name = col[:-9] # Remove the last nine characters from the column names
        cpi_df = cpi_df.withColumnRenamed(col, new_col_name)
# Parse the 'year_and_month' column to a proper date format.
# Note: I'm just arbitrarily adding a day element for a complete date.
# Specifically, I'm using the first day of the month to assign the record to
#   the first or second Sunday of the corresponding month in the next step.
# The original record is for the month and has no specific day attached to it. 
cpi_df = cpi_df.withColumn(
    "date",
    F.to_date(F.expr("concat(substring(`year_and_month`, 1, 4), '-', substring(`year_and_month`, 6, 3), '-01')"), "yyyy-MMM-dd")
)
cpi_df.show()
print(f"'cpi_df' length = {cpi_df.count()}")

# COMMAND ----------

# Add a column for the second Sunday of each month. This will be the initial anchor for a month's cpi value.

# First, find the first Sunday of the month.
# The day of the week index used by 'dayofweek()' starts with Sunday as 1.
# "%" is the modulo operator, used here to handle the case when the first day of the month is a Sunday. 
cpi_df = cpi_df.withColumn(
    "first_sunday",
   F.expr("date_add(date, (7 - dayofweek(date) + 1) % 7)")
)
cpi_df = cpi_df.withColumn(
    "second_sunday",
    F.expr("date_add(first_sunday, 7)")
)
cpi_df = cpi_df.drop('year_and_month', "first_sunday", "date").orderBy("second_sunday")
cpi_df.show()
print(f"'cpi_df' length = {cpi_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Upsample and Interpolate

# COMMAND ----------

# Increase granularity from monthly to weekly.
upsampled_cpi_df = upsample_monthlyts_to_weeklyts_spark(cpi_df,date_col='second_sunday')
upsampled_cpi_df.show()
print(f"'upsampled_cpi_df' length = {upsampled_cpi_df.count()}")

# Add a group_col (uniform placeholder) to utilize the 'spark_pandas_interpolate_convert' function
# ...maybe rename the function until it's generalized for cases w/ no group_col?
# There is no grouping in this case, to be clear.
upsampled_cpi_df = upsampled_cpi_df.withColumn('groupID_placeholder', F.lit("groupID_placeholder"))
# Define the list of cols to interpolate, interpolate df, and rename date col to "date"
cpi_cols_to_interpolate = [col for col in upsampled_cpi_df.columns if col not in ['all_weeks', 'groupID_placeholder']]
interpolated_cpi_df = spark_pandas_interpolate_convert(upsampled_cpi_df, 'groupID_placeholder', cpi_cols_to_interpolate, 'all_weeks')
interpolated_cpi_df = interpolated_cpi_df.drop('groupID_placeholder')
interpolated_cpi_df = interpolated_cpi_df.withColumnRenamed('all_weeks', 'date')
interpolated_cpi_df.show()
print(f"'interpolated_cpi_df' length = {interpolated_cpi_df.count()}")

# I'm rounding to the three decimal places to match the original.
# Define the list of cols to round
columns_to_round = [col for col in interpolated_cpi_df.columns if col not in ['date', 'groupID_placeholder']]
for col_name in columns_to_round:
    interpolated_cpi_df = interpolated_cpi_df.withColumn(col_name, F.round(interpolated_cpi_df[col_name], 3))
interpolated_cpi_df.show()
print(f"'interpolated_cpi_df' length = {interpolated_cpi_df.count()}")

# COMMAND ----------

# Trim the cpi df to include only dates found in 'interpolated_df'.
# Select only the 'ds' column from 'interpolated_df' and rename it to 'date' for matching
dates_to_keep = interpolated_df.select('ds').withColumnRenamed('ds', 'date')
# Perform a semi-join to trim interpolated_cpi_df
trimmed_interpolated_cpi_df = interpolated_cpi_df.join(dates_to_keep, 'date', 'semi')
trimmed_interpolated_cpi_df.show()
print(f"'trimmed_interpolated_cpi_df' length = {trimmed_interpolated_cpi_df.count()}")

# COMMAND ----------

# Incomplete and put on pause.
# def spark_pandas_calc_correl_coeffs_multicol(df1, df2, value_col, metric_columns_to_check_in_df2, date1='ds', date2='date'):
#     # Convert Spark DataFrames to Pandas DataFrames
#     pandas_df1 = df1.toPandas()
#     pandas_df2 = df2.toPandas()

#     # Ensure that both DataFrames have the same index for proper correlation calculation
#     pandas_df1.set_index(date1, inplace=True)
#     pandas_df2.set_index(date2, inplace=True)

#     # Align DataFrames to ensure they have the same date indices
#     pandas_df1, pandas_df2 = pandas_df1.align(pandas_df2, join='inner', axis=0)

#     # Dictionary to store the correlation results
#     correlations = {}

#     # Create a figure for the plots
#     fig, axes = plt.subplots(nrows=len(metric_columns_to_check_in_df2), ncols=1, figsize=(10, 5 * len(metric_columns_to_check_in_df2)))
#     if len(metric_columns_to_check_in_df2) == 1:
#         axes = [axes]  # Ensure axes is iterable for a single plot

#     # Iterate over each column to calculate correlation and create a plot
#     for idx, col in enumerate(metric_columns_to_check_in_df2):
#         if col in pandas_df2.columns:
#             correlation = pandas_df1[value_col].corr(pandas_df2[col])
#             correlations[col] = correlation

#             # Plotting
#             axes[idx].scatter(pandas_df2[col], pandas_df1[value_col])
#             axes[idx].set_xlabel(col)
#             axes[idx].set_ylabel(value_col)
#             axes[idx].set_title(f"Correlation between {col} and {value_col}: {correlation:.2f}")

#         else:
#             correlations[col] = 'Column not found in df2'

#     plt.tight_layout()
#     plt.show()

#     return correlations


# Aggregate sales data as necessary for coerrelation checks.
aggregated_sales = interpolated_df.groupby('ds', 'BottleClass').agg(F.sum('y').alias('y'))
print(f"aggregated_sales:")
aggregated_sales.show()
gallons_sales = aggregated_sales.filter(aggregated_sales['BottleClass'] == 'Gallon')
print(f"gallons_sales:")
gallons_sales.show()
singles_sales = aggregated_sales.filter(aggregated_sales['BottleClass'] == 'Singles')  
print(f"singles_sales:")
singles_sales.show()
# Normalize the sales time series for correlation calculation checks.
windowSpec = Window.partitionBy()
# Add mean, standard deviation, and the normalized value column as new columns
# gallons:
gallons_sales_with_stats = gallons_sales.withColumn(
    'gal_mean', F.mean(F.col('y')).over(windowSpec)
).withColumn(
    'gal_stddev', F.stddev(F.col('y')).over(windowSpec)
)
norm_gallons_sales = gallons_sales_with_stats.withColumn(
    'y_normalized',
    (F.col('y') - F.col('gal_mean')) / F.col('gal_stddev')
)
print(f"norm_gallons_sales:")
norm_gallons_sales.show()
# singles:
singles_sales_with_stats = singles_sales.withColumn(
    'sing_mean', F.mean(F.col('y')).over(windowSpec)
).withColumn(
    'sing_stddev', F.stddev(F.col('y')).over(windowSpec)
)
norm_singles_sales = singles_sales_with_stats.withColumn(
    'y_normalized',
    (F.col('y') - F.col('sing_mean')) / F.col('sing_stddev')
)
print(f"norm_singles_sales:")
norm_singles_sales.show()
print()
# Normalize the CPI time series for correlation calculation checks.
list_of_cpi_cols_to_check = [col for col in trimmed_interpolated_cpi_df.columns if col != 'date']
windowSpec = Window.partitionBy()
for col_name in list_of_cpi_cols_to_check:
    # Add mean and standard deviation as new columns for each column in list_of_cpi_cols_to_check
    trimmed_interpolated_cpi_df = trimmed_interpolated_cpi_df.withColumn(
        f'{col_name}_mean', F.mean(F.col(col_name)).over(windowSpec)
    ).withColumn(
        f'{col_name}_stddev', F.stddev(F.col(col_name)).over(windowSpec)
    )
    # Apply Z-score normalization
    trimmed_interpolated_cpi_df = trimmed_interpolated_cpi_df.withColumn(
        f'{col_name}_normalized',
        (F.col(col_name) - F.col(f'{col_name}_mean')) / F.col(f'{col_name}_stddev')
    ).na.fill({f'{col_name}_normalized': 0})  # Replace NaN values with zero
print(f"Normalized CPI DataFrame:")
trimmed_interpolated_cpi_df.show()
print()
# List now-normalized cols to check
list_of_norm_cpi_cols_to_check = [col for col in trimmed_interpolated_cpi_df.columns if '_normalized' in col]

# Calculate the correlation coefficients
gallons_correlation_results = spark_pandas_calc_correl_coeffs_multicol(norm_gallons_sales, trimmed_interpolated_cpi_df
                                                                                , 'y_normalized', list_of_norm_cpi_cols_to_check)
print(f"gallons_correlation_results:")
print(gallons_correlation_results)
print()
singles_correlation_results = spark_pandas_calc_correl_coeffs_multicol(norm_singles_sales, trimmed_interpolated_cpi_df
                                                                                , 'y_normalized', list_of_norm_cpi_cols_to_check)
print(f"singles_correlation_results:")
print(singles_correlation_results)

# COMMAND ----------

dbutils.notebook.exit ('stop')

# COMMAND ----------

# Check the column names, data types, size, etc. of the data.
# Is it ready for Prophet? SARIMAX?
# Are there NULLS? 