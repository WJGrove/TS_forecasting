# Databricks notebook source
# Confirm Mount to Azure Storage Account

# dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md
# MAGIC #Installs, Functions, and Inputs

# COMMAND ----------

# MAGIC %md
# MAGIC ##Installs/Imports

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Imports"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Functions

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Functions"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variables

# COMMAND ----------

FiscalYear = '2024'
FiscalPeriod = 'P3'


date_col = 'ds' # origin: 'ReqDelFWstart'
seasonal_period = 52
group_col = 'TimeSeriesID' # origin: concat(col("CustomerNumber"), col("ItemNumber"))


# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve Data

# COMMAND ----------

# Grab sales data from the 'Data_Pre-Prep_Sales' notebook.
transformed_nonconstant_df = spark.sql("SELECT * FROM global_temp.transformed_nonconstant_df_temp_view")
transformed_nonconstant_df.show()
print(f"'transformed_nonconstant_df' length = {transformed_nonconstant_df.count()}")


# COMMAND ----------

# Stop here for SARIMAX models.
dbutils.notebook.exit("stop")

# COMMAND ----------

# PUT LINES HERE TO RUN THE INTERNAL PRE-PROCESSING NOTEBOOKS

# COMMAND ----------

# MAGIC %md
# MAGIC #Normalization

# COMMAND ----------

# MAGIC %md
# MAGIC >Using standard z-score normalization will help maintain information represented by extreme values.

# COMMAND ----------

# Check
def groupedts_z_score_norm_pd(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Normalizes the specified value column within each group of a DataFrame using Z-score normalization.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - group_col: The column name to group by.
    - date_col: The column name that contains date information. Data will be sorted by this column within each group.
    - value_col: The column name of the values to normalize.

    Returns:
    - A Pandas DataFrame with an additional column 'normalized' containing the Z-score normalized values.

    Notes:
    - Groups with a standard deviation of zero will have their 'normalized' values filled with NaN.
    - The function will sort the DataFrame by the group and date columns.
    """
    # Sort the DataFrame by group and date columns
    df_sorted = df.sort_values(by=[group_col, date_col])
    
    # Initialize a column for normalized data
    df_sorted['normalized'] = np.nan
    
    # Group by the specified group column and normalize within each group
    for group_name, group in df_sorted.groupby(group_col):
        mean = group[value_col].mean()
        std = group[value_col].std()
        
        # Check if standard deviation is zero
        if std == 0:
            warnings.warn(f"Standard deviation is zero for group {group_name}. Filling with NULLs.")
            df_sorted.loc[group.index, 'normalized'] = np.nan  # Fill with NULLs (np.nan)
        else:
            # Apply Z-score normalization
            df_sorted.loc[group.index, 'normalized'] = (group[value_col] - mean) / std
        
    return df_sorted

# COMMAND ----------

def groupedts_z_score_norm_spk(df, group_col, date_col, value_col):
    """
    Normalizes the specified value column within each group of a Spark DataFrame using Z-score normalization.
    Marks groups with a standard deviation of zero distinctly.
    
    Parameters:
    - df: Spark DataFrame containing the data.
    - group_col: The column name to group by.
    - date_col: The column name that contains date information. Data will be sorted by this column within each group.
    - value_col: The column name of the values to normalize.
    
    Returns:
    - A Spark DataFrame with additional columns:
      1. 'normalized_{value_col}' containing the Z-score normalized values.
      2. 'zero_std_dev' boolean column indicating if the group has a standard deviation of zero.
    """
    
    # Define a window specification over each group, ordered by the date column
    windowSpec = Window.partitionBy(group_col).orderBy(date_col)
    
    # Calculate mean and standard deviation for each group
    df = df.withColumn(f"{value_col}_mean", F.avg(value_col).over(windowSpec))
    df = df.withColumn(f"{value_col}_std", F.stddev(value_col).over(windowSpec))
    
    # Calculate Z-score normalization, handling division by zero by setting result to NULL where std dev is zero
    df = df.withColumn(f'normalized_target', 
                       F.when(F.col(f"{value_col}_std") != 0, 
                              (F.col(value_col) - F.col(f"{value_col}_mean")) / F.col(f"{value_col}_std"))
                       .otherwise(F.lit(None)))
    
    # Identify groups with a standard deviation of zero
    df = df.withColumn('zero_std_dev', F.when(F.col(f"{value_col}_std") == 0, True).otherwise(False))
    
    # Drop intermediate calculation columns
    df = df.drop(f"{value_col}_mean", f"{value_col}_std")
    
    return df


# COMMAND ----------

# Make sure to isolate series with NULLs in the normalized column.....

# redefine value_col
value_col = 'y_differenced' 
# normalize using z-score normalization
normalized_df = groupedts_z_score_norm_spk(differenced_df, group_col, date_col, value_col)
normalized_df.show()
print(f"'normalized_df' length = {normalized_df.count()}")

null_counts = normalized_df.agg(
    F.count(F.when(F.col('normalized_target').isNull(), 1)).alias('nulls_in_normalized_target'))
null_counts.show()

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC %md
# MAGIC #Price Features (move to ML notebook)

# COMMAND ----------

# OLD EXAMPLES:

# COMMAND ----------

# Rev


# COMMAND ----------

# price change

# Define a window specification.
windowSpec = Window.partitionBy(group_col).orderBy(date_col)
# Use the lag function to get the previous price, and then compute the difference
expanded_df = expanded_df.withColumn("PrevGrossPrice", F.lag("GrossPrice", 1).over(windowSpec))
expanded_df = expanded_df.withColumn("PriceChangeAmount", F.col("GrossPrice") - F.col("PrevGrossPrice"))
# Replace null values in 'PriceChangeAmount' with 0 (applies to the first date of each series)
expanded_df = expanded_df.na.fill({"PriceChangeAmount": 0})

# COMMAND ----------

# price change percentage. (Handle the possibility of division by zero if 'PrevGrossPrice' is 0.)

expanded_df = expanded_df.withColumn("PriceChangePercentage", 
                   F.when(F.col("PrevGrossPrice") != 0, 
                        (F.col("PriceChangeAmount") / F.col("PrevGrossPrice")) * 100)
                   .otherwise(0))
expanded_df = expanded_df.drop("PrevGrossPrice")
# # Replace null values in 'PriceChangeAmount' with 0 (applies to the first date of each series)
# expanded_df = expanded_df.na.fill({"PriceChangeAmount": 0})
# Change data types where necessary
# expanded_df = expanded_df.withColumn("Rev?", F.col("Rev?").cast("decimal(10, 2)")) # money should be 'decimal'
expanded_df = expanded_df.withColumn("PriceChangePercentage", F.col("PriceChangePercentage").cast("double"))
expanded_df.show()
print(f"'expanded_df' length = {expanded_df.count()}")

# COMMAND ----------

# Add more features (including lags that seem important, see notes)

# COMMAND ----------

# MAGIC %md
# MAGIC #Store

# COMMAND ----------

# Check the column names, data types, size, etc. of the data.
# Are there NULLS? (Check the series individually for NULLS as the series are different lengths.)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Global Temporary View

# COMMAND ----------

# MAGIC %md
# MAGIC ###Non-Constant Series

# COMMAND ----------

# Create a temporary view
expanded_df.createOrReplaceGlobalTempView("expanded_sales_df_temp_view")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Constant Series

# COMMAND ----------

# Create a temporary view
constant_series_df.createOrReplaceGlobalTempView("constant_series_df_temp_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Write to Storage

# COMMAND ----------

# change this
# expanded_df.write.format("parquet").save(f"{sales_file_path}Expanded_Sales")