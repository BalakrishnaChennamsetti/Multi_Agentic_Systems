import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()  # Load environment variables from .env file


def llm_client(model_name: str = "gemma4:12b") -> ChatOllama:
    """
    This is a simple LLM client creation through Ollam local server
    Args:
    model_name: str model name, by deafult to gemma4:12b

    returns:
    ChatOllama client object
    """
    llm = ChatOllama(
        model=model_name, temperature=0, base_url=os.getenv("GEMMA4_ENDPOINT")
    )
    return llm
