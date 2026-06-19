from ..agents_state import State
from ..tools.common_tools import TOOLS
from ..utils import llm_client


def run_frontier(state: State):
    """This is the frontier agent which will be responsible for calling the tools and getting the response from the LLM."""
    client = llm_client()

    llm_with_tools = client.bind_tools(TOOLS)

    response = llm_with_tools.invoke(state["messages"])

    print("\n=== TOOL CALLS ===")
    print(response.tool_calls)

    return {"messages": [response]}
