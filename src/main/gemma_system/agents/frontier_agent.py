from ..agents_state import State
from ..tools.common_tools import TOOLS
from ..utils import llm_client


def run_frontier(state: State):
    client = llm_client()

    llm_with_tools = client.bind_tools(TOOLS)

    response = llm_with_tools.invoke(state["messages"])

    print("\n=== TOOL CALLS ===")
    print(response.tool_calls)

    return {"messages": [response]}
