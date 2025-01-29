import tiktoken, textstat, evaluate
import pandas as pd
from pyspark.sql.functions import pandas_udf


@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))

@pandas_udf("double")
def flesch_kincaid_grade(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.flesch_kincaid_grade(text) for text in texts])
 
@pandas_udf("double")
def automated_readability_index(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.automated_readability_index(text) for text in texts])

@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = evaluate.load("toxicity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = evaluate.load("perplexity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(perplexity.compute(data=texts.fillna(""), model_id="gpt2")["perplexities"]).where(texts.notna(), None)

