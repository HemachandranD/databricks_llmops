import os
import sys
import logging
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Build Model').getOrCreate()

from pyspark.sql.functions import explode
sys.path.append(os.getcwd().rsplit("/src")[0])
from databricks.vector_search.client import VectorSearchClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from src.modeling.chain_model import save_chain_model, create_prompt, unwrap_document
from src.modeling.document_retriever import get_retriever
from src.common.utility_functions import read_data_handler, write_data_to_delta, write_data_with_cdc, write_embedding_data_handler, index_exists, wait_for_vs_endpoint_to_be_ready, wait_for_index_to_be_ready
from src.config.configuration import datasets_path, catalog_name, bronze_schema_name, silver_schema_name, gold_schema_name, pdf_raw_table_name, pdf_chunks_table_name, pdf_embeddings_table_name, vector_search_endpoint_sub_name, pdf_self_managed_vector_index_name, pdf_managed_vector_index_name, llm_endpoint, chain_model_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Date format
)

# Suppress Py4J logs
logging.getLogger("py4j").setLevel(logging.WARNING)


if __name__ == "__main__":
    # Create logger instance
    logger = logging.getLogger(__name__)
    vs_endpoint_prefix = "vs_endpoint_"
    vs_endpoint_name = vs_endpoint_prefix+str(vector_search_endpoint_sub_name)

    logger.info(f"Creating the Chain Model")
    # Databricks Foundation LLM model
    chat_model = ChatDatabricks(endpoint=llm_endpoint, max_tokens = 300)
    question_answer_chain = create_stuff_documents_chain(chat_model, create_prompt())
    chain = create_retrieval_chain(get_retriever(vs_endpoint_name=vs_endpoint_name, vs_index_fullname=f"{catalog_name}.{gold_schema_name}.{pdf_self_managed_vector_index_name}"), question_answer_chain)|RunnableLambda(unwrap_document)

    question = {"input": "How does Generative AI impact humans?"}
    answer = chain.invoke(question)

    logger.info(f"Registering the Chain Model {chain_model_name}")
    model_info = save_chain_model(logger=logger, chain=chain, catalog_name=catalog_name, schema_name=gold_schema_name, chain_model_name=chain_model_name, llm_endpoint=llm_endpoint, resource_index_name=pdf_self_managed_vector_index_name, question=question, answer=answer)