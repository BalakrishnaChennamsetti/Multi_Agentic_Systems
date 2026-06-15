import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from main.gemma_system.exception_handling.exceptions import (
    DocumentLoadError,
    VectorDBError,
)
from main.gemma_system.vector_db.constants import EMBEDDING_BATCH_SIZE, VECTOR_DB_PATH
from main.gemma_system.vector_db.faiss_db import vector_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_docs(data_path: str):
    docs = []
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        for file in Path(data_path).rglob("*"):
            if file.suffix in [".txt", ".md"]:
                logger.info(f"Loading document: {file}")
                try:
                    content = file.read_text(encoding="utf-8")
                except UnicodeDecodeError as e:
                    logger.error(f"Failed to decode file: {file}")
                    raise
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(file),
                            "filename": file.name,
                            "type": file.suffix,
                        },
                    )
                )
            elif file.suffix in [".pdf"]:
                logger.info(f"Loading PDF document: {file}")
                try:
                    loader = PyPDFLoader(str(file))
                    pdf_docs = loader.load()
                    docs.extend(pdf_docs)
                    print(len(docs))
                except Exception as e:
                    logger.error(f"Failed to load PDF file: {file}")
                    raise
            else:
                logger.warning(f"Unsupported file type: {file}, skipping.")
                raise DocumentLoadError(f"Unsupported file type: {file}")
        logger.info(f"Loaded {len(docs)} documents from {data_path}")
    except Exception as e:
        raise DocumentLoadError(f"Error loading documents: {str(e)}")
    return docs


def chunk_docs(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
    )
    try:
        chunked_docs = text_splitter.create_documents(
            [doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs]
        )
        logger.info(f"Chunked documents into {len(chunked_docs)} chunks.")
    except Exception as e:
        raise DocumentLoadError(f"Error chunking documents: {str(e)}")
    return chunked_docs


def save_vector_store(
    vector_store, save_path: str = VECTOR_DB_PATH
):
    vector_store.save_local(save_path)
    logger.info(f"Vector store saved to {save_path}")


def ingest_docs(data_path: str, embedding_model: str = "embeddinggemma"):
    docs = load_docs(data_path)
    chunked_docs = chunk_docs(docs)
    vector_store_class, embeddings = vector_db(embedding_model)
    first_batch = chunked_docs[:EMBEDDING_BATCH_SIZE]
    vector_store = vector_store_class.from_documents(
        first_batch,
        embeddings
    )
    try:
        counter = EMBEDDING_BATCH_SIZE
        for i in range(EMBEDDING_BATCH_SIZE, len(chunked_docs), EMBEDDING_BATCH_SIZE):
            chunks_50 = chunked_docs[i : i + EMBEDDING_BATCH_SIZE]
            # logger.info(f"Testing vector store creation with {len(chunks_50)} chunks...")
            # vector_store = vector_store_class.from_documents(chunks_50, embeddings)
            counter += len(chunks_50)
            logger.info("Adding vector store...")
            vector_store.add_documents(chunks_50)
            logger.info(
            f"Indexed {min(i + EMBEDDING_BATCH_SIZE, len(chunked_docs))}/{len(chunked_docs)} chunks"
        )
            # if counter >= 200:
            #     # save_vector_store(vector_store)
            #     break
        save_vector_store(vector_store)
        if counter >= len(chunked_docs):
            logger.info(f"Indexed {counter}/{len(chunked_docs)} chunks")
        

    except VectorDBError as e:
        raise VectorDBError(f"Error creating or saving vector store: {str(e)}")
    return "Successfully ingested documents into the vector store."


if __name__ == "__main__":
    vdb_path = Path(VECTOR_DB_PATH)
    if vdb_path.exists():
        delete_confirmation = input(
            f"Vector store already exists at {vdb_path}. Do you want to delete it and re-ingest? (y/n): "
        )
        if delete_confirmation.lower() != "y":
            logger.info("Skipping ingestion.")
            exit(0)
        else:
            import shutil

            shutil.rmtree(vdb_path)
            logger.info(f"Deleted existing vector store at {vdb_path}.")
            
    data_path = "src/main/gemma_system/raw_knowledge_source"
    embedding_model = "embeddinggemma"
    result = ingest_docs(data_path, embedding_model)
    logger.info("Ingestion completed.")
    assert result == "Successfully ingested documents into the vector store."
