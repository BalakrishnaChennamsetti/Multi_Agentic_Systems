from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from main.gemma_system.exception_handling.exceptions import VectorDBError


def vector_db(embedding_model: str = "embeddinggemma"):
    try:
        embeddings = OllamaEmbeddings(model=embedding_model)
    except Exception as e:
        raise VectorDBError(f"Error initializing vector database: {str(e)}")
    return FAISS, embeddings
