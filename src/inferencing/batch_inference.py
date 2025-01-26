import mlflow
from src.config.configuration import catalog_name, gold_schema_name, chain_model_name


def get_latest_model_version(func):
    """
    Helper method to programmatically get latest model's version from the registry
    """
    def wrapper(model_name):
        """
        Wrapper to fetch the latest model version before executing the main function.
        """
        client = mlflow.tracking.MlflowClient()
        model_version_infos = client.search_model_versions("name = '%s'" % model_name)
        latest_version = max(
            [model_version_info.version for model_version_info in model_version_infos]
        )
        # Call the original function with the latest version
        return func(model_name, latest_version)

    return wrapper


@get_latest_model_version
def set_alias(model_name, current_model_version):
    # Set @alias to the latest model version
    mlflow.tracking.MlflowClient().set_registered_model_alias(
        name=model_name, alias="champion",
        version=current_model_version
        )
    
    return current_model_version
    

if __name__ == "__main__":
    model_name = f"{catalog_name}.{gold_schema_name}.{chain_model_name}"

    current_model_version = set_alias(model_name=model_name)
    model_uri = f"models:/hemzai.gold.hemz-pilot-chain/{current_model_version}"
    
    # To verify that the model has been logged correctly, load the agent and call `invoke`:
    model = mlflow.langchain.load_model(model_uri)
    question = {"input": "How does Generative AI impact humans?"}
    print(model.invoke(question))