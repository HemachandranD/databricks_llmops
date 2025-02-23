import os

import yaml

varaiable_yaml_path = f"{os.path.dirname(os.path.abspath(__file__))}/variables.yaml"

with open(varaiable_yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Extract project parameters
project_parameters = config.get("project_parameters", {})


# Subscribe variables using dictionary unpacking
(
    datasets_path,
    catalog_name,
    bronze_schema_name,
    silver_schema_name,
    gold_schema_name,
    pdf_raw_table_name,
    pdf_chunk_size,
    pdf_chunks_table_name,
    pdf_embeddings_table_name,
    vector_search_endpoint_sub_name,
    pdf_self_managed_vector_index_name,
    pdf_managed_vector_index_name,
    llm_endpoint,
    chain_model_name,
    scope_name,
    serving_endpoint_name,
    app_inference_table_name_prefix,
    app_inference_processed_table_name,
    input_request_json_path,
    input_json_path_type,
    output_request_json_path,
    output_json_path_type,
    keep_last_question_only,
) = (
    project_parameters.get(key)
    for key in [
        "datasets_path",
        "catalog_name",
        "bronze_schema_name",
        "silver_schema_name",
        "gold_schema_name",
        "pdf_raw_table_name",
        "pdf_chunk_size",
        "pdf_chunk_table_name",
        "pdf_embeddings_table_name",
        "vector_search_endpoint_sub_name",
        "pdf_self_managed_vector_index_name",
        "pdf_managed_vector_index_name",
        "llm_endpoint",
        "chain_model_name",
        "scope_name",
        "serving_endpoint_name",
        "app_inference_table_name_prefix",
        "app_inference_processed_table_name",
        "input_request_json_path",
        "input_json_path_type",
        "output_request_json_path",
        "output_json_path_type",
        "keep_last_question_only",
    ]
)
