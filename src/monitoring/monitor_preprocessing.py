import os
from pyspark.sql import DataFrame
from delta.tables import DeltaTable
from pyspark.sql.functions import col
from src.common.utility_functions import unpack_requests
from compute_metrics import compute_num_tokens, flesch_kincaid_grade, automated_readability_index, compute_toxicity, compute_perplexity
from src.config.configuration import catalog_name, gold_schema_name, app_inference_table_name, app_inference_processed_table_name, input_request_json_path, input_json_path_type, output_request_json_path, output_json_path_type, keep_last_question_only


def prepare_data_to_monitor(app_inference_table_name, column_to_measure = ["input", "output"]):
    # Unpack using provided helper function
    app_inference_df = spark.readStream.option("ignoreDeletes", "true").table(app_inference_table_name).where('status_code == 200').limit(10)
    app_inference_processed_df = unpack_requests(
        app_inference_df,
        input_request_json_path,
        input_json_path_type,
        output_request_json_path,
        output_json_path_type,
        keep_last_question_only)
    
    # Drop un-necessary columns for monitoring jobs
    app_inference_processed_df = app_inference_processed_df.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id")
    
    for column_name in column_to_measure:
        app_inference_processed_df = (
        app_inference_processed_df.withColumn(f"toxicity({column_name})", compute_toxicity(col(column_name)))
                            .withColumn(f"perplexity({column_name})", compute_perplexity(col(column_name)))
                            .withColumn(f"token_count({column_name})", compute_num_tokens(col(column_name)))
                            .withColumn(f"flesch_kincaid_grade({column_name})", flesch_kincaid_grade(col(column_name)))
                            .withColumn(f"automated_readability_index({column_name})", automated_readability_index(col(column_name)))
                )
    return app_inference_processed_df

def create_processed_table_if_not_exists(app_inference_processed_table_name, app_inference_processed_df, checkpoint_location):
    """
    Helper method to create processed table using schema
    """
    (
      DeltaTable.createIfNotExists(spark)
        .tableName(app_inference_processed_table_name)
        .addColumns(app_inference_processed_df.schema)
        .property("delta.enableChangeDataFeed", "true")
        .property("delta.columnMapping.mode", "name")
        .execute()
    )

    # Append new unpacked payloads & metrics
    (app_inference_processed_df.writeStream
                        .trigger(availableNow=True)
                        .format("delta")
                        .outputMode("append")
                        .option("checkpointLocation", checkpoint_location)
                        .toTable(app_inference_processed_table_name).awaitTermination())


if  __name__ == "__main__":

    app_infernece_table_name = f"{catalog_name}.{gold_schema_name}.{app_inference_table_name}"
    app_inference_processed_table_name = f"{catalog_name}.{gold_schema_name}.{app_inference_processed_table_name}"
    checkpoint_location =  os.path.join(os.getcwd(), "checkpoint")

    app_inference_processed_df = prepare_data_to_monitor(app_infernece_table_name)

    create_processed_table_if_not_exists(app_inference_processed_table_name, app_inference_processed_df, checkpoint_location)


    