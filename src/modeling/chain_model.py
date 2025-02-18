from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from databricks_langchain import ChatDatabricks
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,)
from mlflow.models import infer_signature
from datetime import datetime
import mlflow
import langchain
import os

from src.modeling.document_retriever import get_retriever
from src.config.configuration import catalog_name, gold_schema_name, vector_search_endpoint_sub_name, pdf_self_managed_vector_index_name, pdf_managed_vector_index_name, chain_model_name,llm_endpoint

def save_chain_model(logger, chain, catalog_name, schema_name, chain_model_name, llm_endpoint, resource_index_name, question, answer):
    # set model registry to UC
    mlflow.set_registry_uri("databricks-uc")
    model_name = f"{catalog_name}.{schema_name}.{chain_model_name}"

    # Set experiment path
    t = datetime.now()
    date_now = t.strftime("%Y-%m-%d")
    time_now = t.strftime("%H%M%S")
    experiment_path = f"/Workspace/Applications/Experiments/{date_now}/{time_now}"

    # End any existing runs (in the case this notebook is being run for a second time)
    mlflow.end_run()
    
    # Create the directory if it does not exist
    if not os.path.exists(experiment_path):
        try:
            os.makedirs(experiment_path)
            print("Created experiments folder since it doesn't exist", end="\n")
        except Exception as e:
            print(f"{str(e)}", end="\n")

    logger.info(f"Setting up the experiment {chain_model_name}")
    mlflow.set_experiment(f"{experiment_path}/{chain_model_name}")

    logger.info(f"Starting the run {chain_model_name}")
    with mlflow.start_run(run_name=chain_model_name) as run:
        signature = infer_signature(question, answer)
        model_info = mlflow.langchain.log_model(
            chain,
            loader_fn=get_retriever, 
            artifact_path="chain",
            registered_model_name=model_name,
            pip_requirements=[
                "mlflow==" + mlflow.__version__,
                "langchain==" + langchain.__version__,
                "langchain-community==0.3.14",
                "databricks-vectorsearch==0.49",
                "flashrank==0.2.8",
                "databricks-langchain==0.3.0",
                "sentence-transformers==3.4.1"
            ],
            resources=[
            DatabricksVectorSearchIndex(index_name=f"{catalog_name}.{schema_name}.{resource_index_name}"),
            DatabricksServingEndpoint(endpoint_name=llm_endpoint),
            ],
            input_example=question,
            signature=signature
        )

    logger.info(f"Registered the Model at {model_name}")
    return model_info


def create_prompt():
    TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
    Use the following pieces of context to answer the question at the end:

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """
    prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "input"])

    return prompt
    

# unwrap the longchain document from the context to be a dict so we can register the signature in mlflow
def unwrap_document(answer):
  return answer | {"context": [{"metadata": r.metadata, "page_content": r.page_content} for r in answer["context"]]}


# if __name__ == "__main__":
#     vs_endpoint_prefix = "vs_endpoint_"
#     vs_endpoint_name = vs_endpoint_prefix+str(vector_search_endpoint_sub_name)

#     # test Databricks Foundation LLM model
#     chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens = 300)
#     question_answer_chain = create_stuff_documents_chain(chat_model, create_prompt())
#     chain = create_retrieval_chain(get_retriever(vs_endpoint_name=vs_endpoint_name, vs_index_fullname=f"{catalog_name}.{gold_schema_name}.{pdf_self_managed_vector_index_name}"), question_answer_chain)|RunnableLambda(unwrap_document)

#     question = {"input": "How does Generative AI impact humans?"}
#     answer = chain.invoke(question)
#     # print(answer)
#     model_info = save_chain_model(chain=chain, catalog_name=catalog_name, schema_name=gold_schema_name, chain_model_name=chain_model_name, llm_endpoint=llm_endpoint, resource_index_name=pdf_self_managed_vector_index_name, question=question, answer=answer)