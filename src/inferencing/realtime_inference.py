from databricks.sdk import WorkspaceClient

from src.config.configuration import serving_endpoint_name


def main(question="What is PPO?"):
    ws_client = WorkspaceClient()

    answer = ws_client.serving_endpoints.query(
        serving_endpoint_name, inputs=[{"input": question}]
    )
    return answer


if __name__ == "__main__":
    main()
