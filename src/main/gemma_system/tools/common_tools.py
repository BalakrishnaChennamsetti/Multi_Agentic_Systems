from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """
    return str(eval(expression))


TOOLS = [calculator]


def get_common_tools():
    return ToolNode(TOOLS)