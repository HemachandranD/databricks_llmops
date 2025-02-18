import os
import sys

import mlflow.deployments
import pandas as pd
from pyspark.sql.functions import pandas_udf

sys.path.append(os.getcwd().rsplit("/src")[0])
from src.common.utility_functions import read_data_handler, write_embedding_data_handler
from src.config.configuration import (
    catalog_name,
    gold_schema_name,
    pdf_chunks_table_name,
    pdf_embeddings_table_name,
    silver_schema_name,
)


@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    deploy_client = mlflow.deployments.get_deploy_client("databricks")

    def get_embeddings(batch):
        # NOTE: this will fail if an exception is thrown during embedding creation (add try/except if needed)
        response = deploy_client.predict(
            endpoint="databricks-gte-large-en", inputs={"input": batch}
        )
        return [e["embedding"] for e in response.data]

    # splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [
        contents.iloc[i : i + max_batch_size]
        for i in range(0, len(contents), max_batch_size)
    ]

    # process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

