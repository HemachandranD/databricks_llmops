import os
import sys

sys.path.append(os.getcwd().rsplit("/src")[0])
from databricks.sdk import WorkspaceClient

from src.config.configuration import serving_endpoint_name


def query_realtime_serving_endpoint(question):
    ws_client = WorkspaceClient()

    answer = ws_client.serving_endpoints.query(
        serving_endpoint_name, inputs=[{"input": question}]
    )
    return answer
