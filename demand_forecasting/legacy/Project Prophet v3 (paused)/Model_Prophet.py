# Databricks notebook source
# MAGIC %md
# MAGIC #Installs, Functions, and Variables

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Imports"

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve Data

# COMMAND ----------

# Grab sales data from the 'Data_Pre-Prep_Sales' notebook.
expanded_sales_df = spark.sql("SELECT * FROM global_temp.expanded_sales_df_temp_view")
expanded_sales_df.show()
print(f"'expanded_sales_df' length = {expanded_sales_df.count()}")

# COMMAND ----------

# Add desired shifted columns.

# COMMAND ----------

# MAGIC %md
# MAGIC #Normality, Skedasticity, and Transformations 

# COMMAND ----------

# Test and handle normality and skedasticity for any necessary Box-Cox transformations.
# Add code for transformations here.
# Since some series will be transformed and some won't be, make sure any necessary scaling is done here.

# COMMAND ----------

# MAGIC %md
# MAGIC #Add Holidays

# COMMAND ----------

# Create 'Holiday df' for the Prophet model.

# I'm not sure if the effects of the holidays can be adjusted individually or only for
# the holiday df as a whole using Prophet. If the latter is the case (find out!), then adding a
# supplemental indicator for July 4th, Thanksgiving, and Christmas as an additional regressor may be helpful.