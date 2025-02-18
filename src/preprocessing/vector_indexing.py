import os
import sys

sys.path.append(os.getcwd().rsplit("/src")[0])
from src.common.utility_functions import index_exists, wait_for_index_to_be_ready


def self_managed_vector_index(
    vector_client, vs_endpoint_name, vs_index_fullname, source_table_fullname
):
    # create or sync the index
    if not index_exists(vector_client, vs_endpoint_name, vs_index_fullname):
        print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
        vector_client.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=vs_index_fullname,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED",  # Sync needs to be manually triggered
            primary_key="id",
            embedding_dimension=1024,  # Match your model embedding size (gte)
            embedding_vector_column="embedding",
        )
        # let's wait for the index to be ready and all our embeddings to be created and indexed
        wait_for_index_to_be_ready(vector_client, vs_endpoint_name, vs_index_fullname)
    else:
        # trigger a sync to update our vs content with the new data saved in the table
        vector_client.get_index(vs_endpoint_name, vs_index_fullname).sync()


def managed_vector_index(
    vector_client, vs_endpoint_name, vs_index_fullname, source_table_fullname
):
    # create or sync the index
    if not index_exists(vector_client, vs_endpoint_name, vs_index_fullname):
        print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
        vector_client.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=vs_index_fullname,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED",  # Sync needs to be manually triggered
            primary_key="id",
            embedding_source_column="content",
            embedding_model_endpoint_name="databricks-bge-large-en",
        )
        # let's wait for the index to be ready and all our embeddings to be created and indexed
        wait_for_index_to_be_ready(vector_client, vs_endpoint_name, vs_index_fullname)
    else:
        # trigger a sync to update our vs content with the new data saved in the table
        vector_client.get_index(vs_endpoint_name, vs_index_fullname).sync()

