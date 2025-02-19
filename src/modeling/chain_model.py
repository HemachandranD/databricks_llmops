import os
import sys
from datetime import datetime

import mlflow
import langchain
sys.path.append(os.getcwd().rsplit("/src")[0])
from langchain.prompts import PromptTemplate
from mlflow.models import infer_signature
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

from src.modeling.document_retriever import get_retriever


def save_chain_model(
    logger,
    chain,
    catalog_name,
    schema_name,
    chain_model_name,
    llm_endpoint,
    resource_index_name,
    question,
    answer,
):
    # set model registry to UC
    mlflow.set_registry_uri("databricks-uc")
    model_name = f"{catalog_name}.{schema_name}.{chain_model_name}"

    # Set experiment path
    t = datetime.now()
    date_now = t.strftime("%Y-%m-%d")
    time_now = t.strftime("%H%M%S")
    experiment_path = f"/Workspace/Applications/Experiments/{date_now}/{time_now}"

    # End any existing runs (in the case this notebook is being run for a second time)
    mlflow.end_run()

    # Create the directory if it does not exist
    if not os.path.exists(experiment_path):
        try:
            os.makedirs(experiment_path)
            print("Created experiments folder since it doesn't exist", end="\n")
        except Exception as e:
            print(f"{str(e)}", end="\n")

    logger.info(f"Setting up the experiment {chain_model_name}")
    mlflow.set_experiment(f"{experiment_path}/{chain_model_name}")

    logger.info(f"Starting the run {chain_model_name}")
    with mlflow.start_run(run_name=chain_model_name) as run:
        signature = infer_signature(question, answer)
        model_info = mlflow.langchain.log_model(
            chain,
            loader_fn=get_retriever,
            artifact_path="chain",
            registered_model_name=model_name,
            pip_requirements=[
                "mlflow==" + mlflow.__version__,
                "langchain==" + langchain.__version__,
                "langchain-community==0.3.14",
                "databricks-vectorsearch==0.40",
                "flashrank==0.2.8",
            ],
            code_paths=[f"{os.getcwd().rsplit('/src')[0]}/src"], # Packaging the codes
            resources=[
                DatabricksVectorSearchIndex(
                    index_name=f"{catalog_name}.{schema_name}.{resource_index_name}"
                ),
                DatabricksServingEndpoint(endpoint_name=llm_endpoint),
            ],
            input_example=question,
            signature=signature,
        )

    logger.info(f"Registered the Model at {model_name}")

    logger.info(f"Ending the Model {chain_model_name} run")
    # End any existing runs (in the case this notebook is being run for a second time)
    mlflow.end_run()
    return model_info


def create_prompt():
    TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
    Use the following pieces of context to answer the question at the end:

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """
    prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "input"])

    return prompt


# unwrap the longchain document from the context to be a dict so we can register the signature in mlflow
def unwrap_document(answer):
    return answer | {
        "context": [
            {"metadata": r.metadata, "page_content": r.page_content}
            for r in answer["context"]
        ]
    }
