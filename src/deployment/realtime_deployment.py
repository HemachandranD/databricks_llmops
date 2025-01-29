import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
from src.config.configuration import catalog_name, gold_schema_name, chain_model_name, scope_name, serving_endpoint_name, app_inference_table_name
from src.common.utility_functions import set_alias
from src.config.endpoint_config import load_endpoint_config


def get_ready_for_realtime_inference(model_name):
    
    latest_model_version = set_alias(model_name=model_name)
    endpoint_config_dict = load_endpoint_config(catalog_name, gold_schema_name, app_inference_table_name, model_name, latest_model_version, scope_name)
    endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

    return latest_model_version, endpoint_config


def real_time_deploy(latest_model_version, endpoint_config):
    # Initiate the workspace client
    w = WorkspaceClient()

    # Get endpoint if it exists 
    existing_endpoint = next(
        (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
    )

    db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").value()
    serving_endpoint_url = f"{db_host}/ml/endpoints/{serving_endpoint_name}"

    # If endpoint doesn't exist, create it
    if existing_endpoint == None:
        print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
        w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)

    # If endpoint does exist, update it to serve the new version
    else:
        print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
        w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)



if __name__ == "__main__":
    model_name = f"{catalog_name}.{gold_schema_name}.{chain_model_name}"

    latest_model_version, endpoint_config = get_ready_for_realtime_inference(model_name)
    real_time_deploy(latest_model_version, endpoint_config)



    