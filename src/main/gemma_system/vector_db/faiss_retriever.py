import logging

from main.gemma_system.vector_db.constants import VECTOR_DB_PATH
from main.gemma_system.vector_db.faiss_db import vector_db
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def retriever(query: str, top_k: int = 5):
    """
    Retrieves relevant documents from the Faiss index based on the query.

    Args:
        query (str): The input query for which to retrieve relevant documents.
        top_k (int): The number of top relevant documents to retrieve.
    Returns:
        List[Document]: A list of retrieved documents relevant to the query.
    """
    # Step 1: Convert the query into an embedding vector
    logger.info(f"Loading embedding model and Faiss index for query: {query}")
    _, embeddings = vector_db()

    # Step 2: Search the Faiss index for the top_k most similar documents
    logger.info(f"Searching Faiss index for top {top_k} documents for query: {query}")
    vector_store = FAISS.load_local(
        str(VECTOR_DB_PATH), embeddings, allow_dangerous_deserialization=True
    )

    logger.info(f"Performing similarity search for query: {query}")
    retrieved_documents = vector_store.similarity_search_with_relevance_scores(query=query, k=top_k)

    # Step 3: Retrieve the corresponding documents based on the indices
    logger.info(f"Retrieved {len(retrieved_documents)} documents for query: {query}")
    retrieved_documents = [(page.page_content, page.metadata, score) for page, score in retrieved_documents]

    return retrieved_documents


if __name__ == "__main__":
    # Example usage
    query = "What is Nalanda University?"
    top_k = 3
    results = retriever(query, top_k)
    for idx, doc in enumerate(results):
        print(f"Document {idx + 1}: {doc[0]}")
        print(f"Similarity score: {doc[2]:.2f}")
        print(f"Source: {doc[1].get('source', 'Unknown')}")
        print("-" * 80)
        print()