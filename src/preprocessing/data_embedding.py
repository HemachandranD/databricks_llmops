import logging
import mlflow
import pandas as pd
from typing import List, Iterator
from pyspark.sql.functions import explode, pandas_udf
from src.common.utility_functions import read_data_handler, write_data_with_cdc


@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        # NOTE: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": batch})
        return [e["embedding"] for e in response.data]

    # splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)


if __name__ == "__main__"
    df_chunks = read_data_handler(format="del_table", schema=None, external_path=None, table_name="hemzai.silver.pdf_text_chunks")

    df_embed = (df_chunks
                .withColumn("embedding", get_embedding("content"))
                .selectExpr("pdf_name", "content", "embedding")
                )
    
    write_data_with_cdc(df_embed, mode='append', external_path=None, table_name='hemzai.silver.pdf_text_embeddings')