import logging
import os
import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Ask HemzAI").getOrCreate()


sys.path.append(os.getcwd().rsplit("/src")[0])
from src.config.configuration import (
    catalog_name,
    gold_schema_name,
    chain_model_name
)
from src.inferencing.batch_inference import get_ready_for_batch_inference
from src.inferencing.realtime_inference import query_realtime_serving_endpoint

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
    # Prompt the user for Inference Type
    user_pref_inference_type = input("Please enter the type of Inference You would like to use[batch/realtime]: ")
    # Print the Inference Type
    print("You entered:", user_pref_inference_type)

     # Prompt the user for query
    user_question = input("Please Ask your Question: ")
    # Print the user query
    print("Your Question:", user_question)

    question = {"input": user_question}

    if user_pref_inference_type.strip().lower()== "batch":
        logger.info(
        f"Switching to Batch Ineference mode. Processing the following question: {question}")
        loaded_model = get_ready_for_batch_inference(chain_model_name)
        print(loaded_model.invoke(question))
    elif user_pref_inference_type.strip().lower()== "realtime":
        logger.info(
        f"Switching to Realtime Ineference mode. Processing the following question: {question}")
        assistant_answer=query_realtime_serving_endpoint(question)
        print(assistant_answer)
