import logging
import pandas as pd
from typing import List, Iterator
from pyspark.sql.functions import explode, pandas_udf
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from src.common.utility_functions import read_data_handler, write_data_to_delta


# Define a function to split the text content into chunks
@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Sentence splitter from llama_index to split on sentences
    splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=500, chunk_overlap=50)
    
    def extract_and_split(b):
        loader = PyPDFLoader(b.replace('dbfs:','/dbfs/'))  # Use custom function to parse the bytes pdf
        data = loader.load()
        if data is None:
            return []
        docs = splitter.split_documents(data)
        return [doc.page_content for doc in docs]

    for x in batch_iter:
        yield x.apply(extract_and_split)


if __name__ == "__main__"
    df = read_data_handler(format="del_table", schema=None, external_path=None, table_name="hemzai.bronze.pdf_raw_text")

    df_chunks = (df
                .withColumn("content", explode(read_as_chunk("path")))
                .selectExpr('path as pdf_name', 'content')
                )
    
    write_data_to_delta(df_chunks, mode='overwrite', external_path=None, table_name='hemzai.silver.pdf_text_chunks')
