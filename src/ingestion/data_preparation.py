import os
import sys
sys.path.append(os.getcwd().rsplit("/src")[0])

from pyspark.sql.functions import *
from src.config.configuration import datasets_path, catalog_name, bronze_schema_name, pdf_raw_table_name
from src.common.utility_functions import read_data_handler, write_data_to_delta


# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)


if __name__ == "__main__":
    table_name = f"{catalog_name}.{bronze_schema_name}.{pdf_raw_table_name}"

    # read files
    df = spark.read.format('binaryfile').option("recursiveFileLookup", "true").load(datasets_path)
    
    # save list of the files to table
    write_data_to_delta(df=df, mode='overwrite', external_path=None, table_name=table_name)