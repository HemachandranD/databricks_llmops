import os
import sys
from typing import Iterator

sys.path.append(os.getcwd().rsplit("/src")[0])
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import (
    SentenceTransformersTokenTextSplitter,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf

from src.config.configuration import pdf_chunk_size

# Get a active session
spark = SparkSession.getActiveSession()


# Define a function to split the text content into chunks
@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Sentence splitter from llama_index to split on sentences
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=pdf_chunk_size, chunk_overlap=50
    )

    def extract_and_split(path):
        loader = PyPDFLoader(
            path.replace("dbfs:", "/dbfs/")
        )  # Use custom function to parse the bytes pdf
        data = loader.load()
        if data is None:
            return []
        docs = splitter.split_documents(data)
        return [doc.page_content for doc in docs]

    for x in batch_iter:
        yield x.apply(extract_and_split)

