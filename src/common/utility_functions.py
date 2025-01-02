from databricks.sdk.runtime import *
from pyspark.sql.utils import AnalysisException

def read_data_unfound_handled(schema, format, external_path):
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
        elif format == "binaryfile":
            df = spark.read.format(format).option("recursiveFileLookup", "true").load(external_path)
        elif format == "delta":
            df = spark.read.format(format).load(external_path)
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


def read_table(table_name, logger):
    """
    Reads a table from the Spark session and logs the process.

    Args:
        logger: The logger object for logging information and errors.
        table_name: The name of the table to be read.

    Returns:
        A DataFrame containing the data read from the specified table.

    Raises:
        SystemExit: If an error occurs while reading the table.
    """
    try:
        logger.info(f"Reading the data from {table_name}")
        table_df = spark.read.table(table_name)
        logger.info(f"Completed reading the data from {table_name}")
        # logger.info(f"Count of records in table {table_name}: {table_df.count()}")
        return table_df
    except Exception as read_data_error:
        print(f"An error occurred while reading data: {str(read_data_error)}")
        raise SystemExit(f"Exiting due to the error: {str(read_data_error)}")


def write_data_to_delta(df, mode, external_path, table_name):
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