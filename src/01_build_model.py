import logging
import os
import sys

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Build Model").getOrCreate()


sys.path.append(os.getcwd().rsplit("/src")[0])
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatDatabricks
from langchain_core.runnables import RunnableLambda

from src.config.configuration import (
    catalog_name,
    chain_model_name,
    gold_schema_name,
    llm_endpoint,
    pdf_self_managed_vector_index_name,
    vector_search_endpoint_sub_name,
)
from src.modeling.chain_model import create_prompt, save_chain_model, unwrap_document
from src.modeling.document_retriever import get_retriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
    stream=sys.stdout,  # Redirect logs to stdout
)

# Suppress Py4J logs
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.INFO)

# Create logger instance
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    vs_endpoint_prefix = "vs_endpoint_"
    vs_endpoint_name = vs_endpoint_prefix + str(vector_search_endpoint_sub_name)

    logger.info(f"Creating the Chain Model")
    # Databricks Foundation LLM model
    chat_model = ChatDatabricks(endpoint=llm_endpoint, max_tokens=300)
    question_answer_chain = create_stuff_documents_chain(chat_model, create_prompt())
    chain = create_retrieval_chain(
        get_retriever(
            vs_endpoint_name=vs_endpoint_name,
            vs_index_fullname=f"{catalog_name}.{gold_schema_name}.{pdf_self_managed_vector_index_name}",
        ),
        question_answer_chain,
    ) | RunnableLambda(unwrap_document)

    question = {"input": "How does Generative AI impact humans?"}
    answer = chain.invoke(question)

    logger.info(f"Registering the Chain Model {chain_model_name}")
    model_info = save_chain_model(
        logger=logger,
        chain=chain,
        catalog_name=catalog_name,
        schema_name=gold_schema_name,
        chain_model_name=chain_model_name,
        llm_endpoint=llm_endpoint,
        resource_index_name=pdf_self_managed_vector_index_name,
        question=question,
        answer=answer,
    )
