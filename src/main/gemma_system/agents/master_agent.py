from pprint import pprint

from main.gemma_system.tools.common_tools import TOOLS

from agents_state import State

from main.gemma_system.utils import llm_client


def run_master(state: State):
    print("Running master agent with messages:")
    client = llm_client()
    llm_with_tools = client.bind_tools(TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    # llm_response = client.invoke(state["messages"])
    print(f"Master agent response: {response.content}")
    new_state = {"messages": [response]}
    pprint("The state of the master agent:")
    print("\nINPUT STATE")
    pprint(state)

    print("\nOUTPUT STATE")
    pprint(new_state)

    return new_state
