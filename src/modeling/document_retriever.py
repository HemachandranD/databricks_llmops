from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.docstore.document import Document
from flashrank import Ranker, RerankRequest

from src.config.configuration import catalog_name, gold_schema_name, vector_search_endpoint_sub_name, pdf_self_managed_vector_index_name, pdf_managed_vector_index_name


def get_retriever(vs_endpoint_name, vs_index_fullname, cache_dir="/tmp"):

    def retrieve(query, k: int=10):
        if isinstance(query, dict):
            query = next(iter(query.values()))

        # get the vector search index
        vsc = VectorSearchClient(disable_notice=True)
        vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)
        
        # get the query vector
        embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
        query_vector = embeddings.embed_query(query)
        
        # get similar k documents
        return query, vs_index.similarity_search(
            query_vector=query_vector,
            columns=["pdf_name", "content"],
            num_results=k)

    def rerank(query, retrieved, cache_dir, k: int=2):
        # format result to align with reranker lib format 
        passages = []
        for doc in retrieved.get("result", {}).get("data_array", []):
            new_doc = {"file": doc[0], "text": doc[1]}
            passages.append(new_doc)       
        # Load the flashrank ranker
        ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

        # rerank the retrieved documents
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerankrequest)[:k]

        # format the results of rerank to be ready for prompt
        return [Document(page_content=r.get("text"), metadata={"source": r.get("file")}) for r in results]

    # the retriever is a runnable sequence of retrieving and reranking.
    return RunnableLambda(retrieve) | RunnableLambda(lambda x: rerank(x[0], x[1], cache_dir))


# if __name__ == "__main__":
#     vs_endpoint_prefix = "vs_endpoint_"
#     vs_endpoint_name = vs_endpoint_prefix+str(vector_search_endpoint_sub_name)

#     # test our retriever
#     question = {"input": "How does Generative AI impact humans?"}
#     retriever = get_retriever(vs_endpoint_name=vs_endpoint_name, vs_index_fullname=f"{catalog_name}.{gold_schema_name}.{pdf_self_managed_vector_index_name}")
#     similar_documents = retriever.invoke(question)
#     print(f"Relevant documents: {similar_documents}")