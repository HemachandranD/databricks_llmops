import os
import sys

import mlflow

sys.path.append(os.getcwd().rsplit("/src")[0])
from src.common.utility_functions import set_alias
from src.config.configuration import catalog_name, chain_model_name, gold_schema_name


def get_ready_for_batch_inference(model_name):
    model_name = f"{catalog_name}.{gold_schema_name}.{chain_model_name}"

    current_model_version = set_alias(model_name=model_name)
    model_uri = f"models:/{model_name}/{current_model_version}"
    loaded_model = mlflow.langchain.load_model(model_uri)

    return loaded_model

