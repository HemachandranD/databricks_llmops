import os
import sys
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Build Model').getOrCreate()

from pyspark.sql.functions import explode
sys.path.append(os.getcwd().rsplit("/src")[0])
from databricks.vector_search.client import VectorSearchClient
from src.ingestion.data_preparation import process_pdf_files
from src.preprocessing.data_chunking import read_as_chunk
from src.preprocessing.data_embedding import get_embedding
from src.common.utility_functions import read_data_handler, write_data_to_delta, write_data_with_cdc, write_embedding_data_handler, index_exists, wait_for_vs_endpoint_to_be_ready, wait_for_index_to_be_ready
from src.config.configuration import datasets_path, catalog_name, bronze_schema_name, silver_schema_name, gold_schema_name, pdf_raw_table_name, pdf_chunks_table_name, pdf_embeddings_table_name, vector_search_endpoint_sub_name, pdf_self_managed_vector_index_name, pdf_managed_vector_index_name


if __name__ == "__main__":
    # Read the PDF files and store them as raw data in a Delta table
    raw_df = process_pdf_files(datasets_path=datasets_path, catalog_name=catalog_name, schema_name=bronze_schema_name, raw_table_name=pdf_raw_table_name)

    df_chunks = (raw_df
                .withColumn("content", explode(read_as_chunk("path")))
                .selectExpr('path as pdf_name', 'content')
                )
    
    write_data_to_delta(df=df_chunks, mode='overwrite', external_path=None, table_name=f"{catalog_name}.{silver_schema_name}.{pdf_chunks_table_name}")

    df_embed = (df_chunks
                .withColumn("embedding", get_embedding("content"))
                .selectExpr("pdf_name", "content", "embedding")
                )
    
    # Embedding Table Full Name
    embeddings_fqn = f"{catalog_name}.{gold_schema_name}.{pdf_embeddings_table_name}"
    
    # Embedding table write handler
    write_embedding_data_handler(df_embed, embeddings_fqn)

    # Create the vector search client
    vector_client = VectorSearchClient(disable_notice=True)

    vs_endpoint_prefix = "vs_endpoint_"
    vs_endpoint_name = vs_endpoint_prefix+str(vector_search_endpoint_sub_name)

    if (vs_endpoint_name not in [e["name"] for e in vector_client.list_endpoints()["endpoints"]]) or (vector_client.list_endpoints()==''):
        vector_client.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")
        # check the status of the endpoint
        wait_for_vs_endpoint_to_be_ready(vector_client, vs_endpoint_name)
        print(f"Endpoint named {vs_endpoint_name} is ready.")
    
    # the table we'd like to index
    vs_source_table_fullname = f"{catalog_name}.{gold_schema_name}.{pdf_embeddings_table_name}"

    # the table t to store index
    vs_index_fullname = f"{catalog_name}.{gold_schema_name}.{pdf_self_managed_vector_index_name}"

    # create the self managed vector index
    self_managed_vector_index(vector_client, vs_endpoint_name, vs_index_fullname, vs_source_table_fullname)

    # create manged vector index
    # managed_vector_index(vector_client, vs_endpoint_name, f"{catalog_name}.{gold_schema_name}.{pdf_managed_vector_index_name}", vs_source_table_fullname)