import __init__
import yaml
import os


varaiable_yaml_path = f"{os.path.dirname(os.path.abspath(__file__))}/variables.yaml"

with open(varaiable_yaml_path, "r") as file:
    config = yaml.safe_load(file)


# Subscribe variables
datasets_path = config.get("data_parameters")["datasets_path"]
catalog_name = config.get("data_parameters")["catalog_name"]
bronze_schema_name = config.get("data_parameters")["bronze_schema_name"]
silver_schema_name = config.get("data_parameters")["silver_schema_name"]
gold_schema_name = config.get("data_parameters")["gold_schema_name"]
pdf_raw_table_name = config.get("data_parameters")["pdf_raw_table_name"]
pdf_chunks_table_name = config.get("data_parameters")["pdf_chunk_table_name"]
pdf_embeddings_table_name = config.get("data_parameters")["pdf_embeddings_table_name"]
vector_search_endpoint_sub_name = config.get("data_parameters")["vector_search_endpoint_sub_name"]
pdf_self_managed_vector_index_name = config.get("data_parameters")["pdf_self_managed_vector_index_name"]
pdf_managed_vector_index_name = config.get("data_parameters")["pdf_managed_vector_index_name"]
chain_model_name = config.get("data_parameters")["chain_model_name"]
