# Databricks notebook source
# MAGIC %md
# MAGIC #Installs, Functions, and Variables

# COMMAND ----------

import matplotlib.pyplot as plt
plt.rcParams['figure_size'] = [15, 10]

from math import sqrt

from datetime import datetime
import pandas as pd
import numpy as np
import pdb

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn.metrics.cluster import contingency_matrix

from dtaidistance import dtw

%run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Imports"

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/Data_Prep_Functions"

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve Data

# COMMAND ----------

# Grab sales data from the 'Data_Pre-Prep_Sales' notebook.
expanded_sales_df = spark.sql("SELECT * FROM global_temp.expanded_sales_df_temp_view")
expanded_sales_df.show()
print(f"'expanded_sales_df' length = {expanded_sales_df.count()}")

# COMMAND ----------

# Using dynamic time warping to compute the distance is computationally intensive
# , so we'll first attempt to demonstrate the efficacy of this process by using DTW on
# a smaller set. First, we'll set a baseline by clustering with the Euclidean distance (and
# modeling and testing the best of two or three models), then clustering again with a custom
# distance (and modeling and testing the best of two or three models), then clustering again
# with DTW (and modeling and testing the best of two or three models). By then, I should have an
# idea of the effort vs impact of DTW. 

# Since DTW is the only method that can handle comparing series having different lengths, we'll
# need to exclude the short series, limiting the sample to series with at least some minimum number
# of observations (probably 52 weeks). At the same time, we want to include the maximum number of
# series possible under this constraint, so we probably shouldn't require more than the most recent
# year (t-52 where t = "global max date"). The bottom line: drop all series withless than 52 weeks
# of data and trim off the dates more than 52 weeks prior to the global max date. 

# Ideas:
# (NOTICE: These are dependent on sample size, and Nielsen market still must be added to data)

# 1. First, we'll aggregate to the company, market, and item level. This may allow us to still
#       measure an entire set of series. (There's also the option of just taking the top N
#       companies and excluding the rest.)
# 2. If that's still too many, we'll stay with the market and item level, but only for Walmart.
# 3. If that's still too many, we'll try the company, market, and bottleclass level.

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

# MAGIC %md
# MAGIC ##Sampling

# COMMAND ----------

# MAGIC %md
# MAGIC #Distance Measures

# COMMAND ----------

# MAGIC %md
# MAGIC ## Euclidean Distance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Distance

# COMMAND ----------

# I've thought about using 1 minus the correlation coefficient, all scaled by the difference in volumes as a percentage of their total volume: (1-r)(|V_a - V_b|/(V_a + V_b))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Dynamic Time Warping Distance