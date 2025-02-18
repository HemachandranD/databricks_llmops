import logging
import os
import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Deploy Model").getOrCreate()


sys.path.append(os.getcwd().rsplit("/src")[0])
from src.config.configuration import (
    app_inference_processed_table_name,
    app_inference_table_name,
    catalog_name,
    gold_schema_name,
)
from src.monitoring.monitor_preprocessing import (
    create_processed_table_if_not_exists,
    prepare_data_to_monitor,
)
from src.monitoring.realtime_monitor import create_lhm_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)

# Suppress Py4J logs
logging.getLogger("py4j").setLevel(logging.WARNING)

# Create logger instance
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    app_infernece_table_name = (
        f"{catalog_name}.{gold_schema_name}.{app_inference_table_name}"
    )
    app_inference_processed_table_name = (
        f"{catalog_name}.{gold_schema_name}.{app_inference_processed_table_name}"
    )
    checkpoint_location = os.path.join(os.getcwd(), "checkpoint")

    logger.info(
        f"Preprocesing the data from {app_infernece_table_name} for monitoring."
    )
    app_inference_processed_df = prepare_data_to_monitor(app_infernece_table_name)

    logger.info(
        f"Writing the prperocessed data to the table {app_inference_processed_table_name}"
    )
    create_processed_table_if_not_exists(
        app_inference_processed_table_name,
        app_inference_processed_df,
        checkpoint_location,
    )

    logger.info(
        f"Setting up the Lakehouse Monitoring from the {app_inference_processed_table_name}"
    )
    create_lhm_monitor(
        catalog_name, gold_schema_name, app_inference_processed_table_name
    )
