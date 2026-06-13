from pprint import pprint

from agents_state import State

from main.gemma_system.utils import llm_client


def run_master(state: State):
    print("Running master agent with messages:")
    client = llm_client()
    llm_response = client.invoke(state["messages"])
    print(f"Master agent response: {llm_response.content}")
    new_state = {"messages": [llm_response]}
    pprint("The state of the master agent:")
    print("\nINPUT STATE")
    pprint(state)

    print("\nOUTPUT STATE")
    pprint(new_state)

    return new_state
