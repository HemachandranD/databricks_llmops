

def load_endpoint_config(catalog_name, gold_schema_name, app_inference_table_name, model_name, latest_model_version, scope_name):

    # Configure the endpoint
    endpoint_config_dict = {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": latest_model_version,
                "scale_to_zero_enabled": True,
                "workload_size": "Small",
                "environment_vars": {
                    # "DATABRICKS_TOKEN": "{{{{secrets/{0}/depl_demo_token}}}}".format(scope_name),
                    # "DATABRICKS_HOST": "{{{{secrets/{0}/depl_demo_host}}}}".format(scope_name),
                    "DATABRICKS_TOKEN": "",
                    "DATABRICKS_HOST": 'https://dbc-434c5869-d401.cloud.databricks.com/', 
                },
            },
        ],
        "auto_capture_config":{
            "catalog_name": catalog_name,
            "schema_name": gold_schema_name,
            "table_name_prefix": app_inference_table_name
        }
    }

    return endpoint_config_dict