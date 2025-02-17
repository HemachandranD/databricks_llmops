import os
import sys
sys.path.append(os.getcwd().rsplit("/src")[0])

from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from src.common.utility_functions import read_data_handler, write_data_to_delta
from src.config.configuration import datasets_path, catalog_name, bronze_schema_name, pdf_raw_table_name

# Get a active session
spark = SparkSession.getActiveSession()

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)


def process_pdf_files(logger, datasets_path, catalog_name, schema_name, raw_table_name):
    """Function to process PDF files and write them to Delta tables."""
    table_name = f"{catalog_name}.{schema_name}.{raw_table_name}"

    # Read files
    logger.info(f"Reading the PDF data from {datasets_path}")
    raw_df = spark.read.format('binaryfile').option("recursiveFileLookup", "true").load(datasets_path)

    # Save list of files to Delta table
    logger.info(f"Writing the raw data from pdf to the delta table {table_name}")
    write_data_to_delta(df=raw_df, mode='overwrite', external_path=None, table_name=table_name)
    logger.info(f"Writing the raw data from pdf to the delta table {table_name} is completed.")

    return raw_df


# if __name__ == "__main__":
#     process_pdf_files(datasets_path, catalog_name, bronze_schema_name, pdf_raw_table_name)