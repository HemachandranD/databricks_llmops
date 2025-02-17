import io
import time
import mlflow
import logging
from typing import List
from databricks.sdk.runtime import *
from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from pyspark.sql import functions as F
from pyspark.sql.functions import explode, col
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Date format
)

# Suppress Py4J logs
logging.getLogger("py4j").setLevel(logging.WARNING)

# Create logger instance
logger = logging.getLogger(__name__)

# Load Data
def load_data(file_type: str, file_loc: str) -> List[Document]:
    try:
        """Load data into a list of Documents
        Args:
            file_type: the type of file to load
        Returns:    list of Documents
        """
        if file_type == "PDF":
            loader = PyPDFLoader(file_loc)
            data = loader.load()

        elif file_type == "Text":
            loader = TextLoader(r"{}".format(file_loc))
            data = loader.load()

        elif file_type == "DOCX":
            loader = Docx2txtLoader(file_loc)
            data = loader.load()

        elif file_type == "Markdown":
            loader = UnstructuredMarkdownLoader(file_loc)
            data = loader.load()
        logger.info(f"Loaded the {file_type} Document from {file_loc}")

        return data

    except Exception as e:
        logger.error(f"An error occurred in load_data: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")


def read_data_handler(format, schema=None, external_path=None, table_name=None):
    """
    Attempt to read data from external path.
    If path is not found, create an empty dataframe with the given schema and write it to the external path.

    :param schema: Schema to use for creating an empty dataframe
    :param format: Format to read data in
    :param external_path: Path to read data from
    :return: Dataframe read from path, or an empty dataframe if path is not found
    """
    df = None
    try:
        if format == "parquet":
            df = spark.read.schema(schema).format(format).load(external_path)
        elif format == "delta":
            df = spark.read.format(format).load(external_path)
        elif format == "del_table":
            df = spark.read.table(table_name)
    except AnalysisException as e:
        if ("[PATH_NOT_FOUND]" in str(e)) and (schema is not None):
            raise "Provide a proper external path "
        elif ("[PATH_NOT_FOUND]" in str(e)) and (schema is None):
            raise "Provide a schema to create an empty dataframe"
        else:
            raise e

    except Exception as e:
        raise e

    return df


def write_data_to_delta(df, mode, external_path, table_name):
    try: 
        # Extract database/schema namefor Unity Catalog.
        schema = table_name.split(".")[1]
        spark.sql("CREATE DATABASE IF NOT EXISTS " + schema)
        if external_path is not None:
            df.write.format("delta").mode(mode).option("overwriteSchema", "false").option(
                "path", external_path
            ).saveAsTable(table_name)
        elif external_path is None:
            df.write.format("delta").mode(mode).option(
                "overwriteSchema", "false"
            ).saveAsTable(table_name)

    except Exception as e:
        logger.error(f"An error occurred in write_data_to_delta: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")


def write_data_with_cdc(df, mode, external_path, table_name):
    try: 
        # Extract database/schema namefor Unity Catalog.
        schema = table_name.split(".")[1]
        spark.sql("CREATE DATABASE IF NOT EXISTS " + schema)
        if external_path is not None:
            df.write.format("delta").mode(mode).option("delta.enableChangeDataFeed", "true")\
                .option("overwriteSchema", "false")\
                    .option("path", external_path)\
                        .saveAsTable(table_name)
        elif external_path is None:
            # Write the empty DataFrame to create the Delta table with CDC enabled
            df.write.format("delta").mode(mode).option("delta.enableChangeDataFeed", "true")\
                .option("overwriteSchema", "false" )\
                    .saveAsTable(table_name)

    except Exception as e:
        logger.error(f"An error occurred in write_data_with_cdc: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")


def write_embedding_data_handler(df_embed, embeddings_fqn):
    """
    write the embeddings data to delta tables
    """
    try: 
        if not (spark.catalog.tableExists(embeddings_fqn)):
            catalog_name = embeddings_fqn.split(".")[0]
            schema_name = embeddings_fqn.split(".")[1]
            embed_table_name = embeddings_fqn.split(".")[2]

            # spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
            spark.sql(f"USE CATALOG {catalog_name}")

            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            spark.sql(f"USE SCHEMA {schema_name}")
            
            spark.sql(spark.sql(f"""CREATE TABLE IF NOT EXISTS {embed_table_name}(
                id BIGINT GENERATED BY DEFAULT AS IDENTITY,
                pdf_name STRING,
                content STRING,
                embedding ARRAY <FLOAT>
                -- NOTE: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
            ) TBLPROPERTIES (delta.enableChangeDataFeed = true)"""))
        
        write_data_with_cdc(df_embed, mode='append', external_path=None, table_name=embeddings_fqn)
    
    except Exception as e:
        logger.error(f"An error occurred in write_embedding_data_handler: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")


def index_exists(vsc, endpoint_name, index_full_name):
    """
    Checks if Vector Search Index exists or not.
    """
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready', False)
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
    """
    Checks the status of the Vector Search endpoint.
    """
    for i in range(180):
        endpoint = vsc.get_endpoint(vs_endpoint_name)
        status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
        if "ONLINE" in status:
            return endpoint
        elif "PROVISIONING" in status or i <6:
            if i % 20 == 0: 
                print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
                time.sleep(30)
            else:
                raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
    raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")


def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    """
    Checks the status of the Vector Search Index.
    """
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get('status', idx.get('index_status', {}))
        status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
        url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
            time.sleep(10)
        else:
            raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
    raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")


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


def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str,
                    keep_last_question_only: False) -> DataFrame:
    # Rename the date column and convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumn("timestamp", (col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "model_id",
        F.concat(
            col("request_metadata").getItem("model_name"),
            F.lit("_"),
            col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(col("status_code") == "200")

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type))
        .withColumn("response", F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type)))
    
    if keep_last_question_only:
        requests_unpacked = requests_unpacked.withColumn("request", F.array(F.element_at(F.col("request"), -1)))

    # Explode batched requests into individual rows.
    requests_exploded = (requests_unpacked
        .withColumn("__db_request_response", F.explode(F.arrays_zip(col("request").alias("input"), col("response").alias("output"))))
        .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
        )

    return requests_exploded