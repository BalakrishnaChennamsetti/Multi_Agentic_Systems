from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from ..vector_db.faiss_retriever import retriever


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """
    return str(eval(expression))


@tool
def retriever_as_tools(query: str, top_k: int = 5):
    """
    Retrieve relevant documents based on the query.
    """
    return retriever(query, top_k)


TOOLS = [calculator, retriever_as_tools]


def get_common_tools():
    return ToolNode(TOOLS)
