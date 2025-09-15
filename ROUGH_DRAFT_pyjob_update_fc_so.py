# Databricks notebook source
# Step 1: Retrieve the last processed timestamp from the metadata table
last_processed_timestamp = spark.sql("""
    SELECT last_processed_timestamp 
    FROM forecast_dev.data_science.metadata
""").collect()[0]["last_processed_timestamp"]


# COMMAND ----------

# Step 2: Query the view for new/updated rows based on dw_update_date_utc_soh
incremental_df = spark.sql(f"""
    SELECT *
    FROM forecast_dev.data_science.view_forecast_sales_orders
    WHERE dw_update_date_utc_soh > '{last_processed_timestamp}'
""")

# COMMAND ----------

# Step 3: Perform the MERGE operation to update/insert the changes
incremental_df.createOrReplaceTempView("incremental_updates")

spark.sql("""
    MERGE INTO forecast_dev.data_science.forecast_sales_orders AS target
    USING (
      SELECT *
      FROM forecast_dev.data_science.view_forecast_sales_orders
      WHERE dw_update_date_utc_soh > 'last_processed_timestamp' -- Replace with the stored value of last_update_ts
    ) AS source
    ON target.order_id = source.order_id
      AND target.item_id_soq = source.item_id_soq
      AND target.line_id = source.line_id -- Match at the line level

    WHEN MATCHED THEN
      -- Update the existing rows in the target table if a match is found
      UPDATE SET
        target.soldto_id = source.soldto_id,
        target.ordered_qty_fc = source.ordered_qty_fc,
        target.gross_price_fc = source.gross_price_fc,
        target.required_delivery_date = source.required_delivery_date,
        target.product_status = source.product_status,
        target.stock_facility_id = source.stock_facility_id,
        target.dw_update_date_utc_soh = source.dw_update_date_utc_soh,
        target.req_del_fw_start_date = source.req_del_fw_start_date,
        target.ordered_qty_soq = source.ordered_qty_soq,
        target.gross_price_sop = source.gross_price_sop,
        target.units_per_stock = source.units_per_stock,
        target.item_desc_1 = source.item_desc_1,
        target.cust_company_name = source.cust_company_name,
        target.item_id_fc = source.item_id_fc,
        target.time_series_id = source.time_series_id

    WHEN NOT MATCHED THEN
      -- Insert new rows from the source if no match is found in the target
      INSERT (
        order_id,
        soldto_id,
        required_delivery_date,
        stock_facility_id, 
        dw_update_date_utc_soh,
        req_del_fw_start_date,
        item_id_soq,
        line_id, 
        ordered_qty_soq,
        gross_price_sop,
        product_status,
        units_per_stock, 
        item_desc_1,
        cust_company_name,
        item_id_fc,
        time_series_id, 
        ordered_qty_fc,
        gross_price_fc
      )
      VALUES (
        source.order_id,
        source.soldto_id,
        source.required_delivery_date,
        source.stock_facility_id, 
        source.dw_update_date_utc_soh,
        source.req_del_fw_start_date,
        source.item_id_soq,
        source.line_id, 
        source.ordered_qty_soq,
        source.gross_price_sop,
        source.product_status,
        source.units_per_stock, 
        source.item_desc_1,
        source.cust_company_name,
        source.item_id_fc,
        source.time_series_id, 
        source.ordered_qty_fc,
        source.gross_price_fc
      );
""")

# COMMAND ----------

# Step 4: Get the latest timestamp from the updated data
new_last_processed_timestamp = spark.sql("""
    SELECT MAX(dw_update_date_utc_soh) AS new_last_update
    FROM forecast_dev.data_science.view_forecast_sales_orders
""").collect()[0]["new_last_update"]

# COMMAND ----------

# Step 5: Update the metadata table with the new timestamp
spark.sql(f"""
    UPDATE forecast_dev.data_science.metadata 
    SET last_processed_timestamp = '{new_last_processed_timestamp}'
""")