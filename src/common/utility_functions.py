import io
import logging
from typing import List
from databricks.sdk.runtime import *
from pyspark.sql.utils import AnalysisException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from pypdf import PdfReader

# Creating an object
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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
        logger.error(f"An error occurred in load_data: {str(e)}")
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
        logger.error(f"An error occurred in load_data: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")
