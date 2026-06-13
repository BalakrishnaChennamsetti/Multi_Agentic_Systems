from pprint import pprint
from unittest import result

from agents.frontier_agent import run_frontier
from agents.master_agent import run_master
from agents_state import State
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from tools.common_tools import get_common_tools

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", run_frontier)
graph_builder.add_node("tools", get_common_tools())
graph_builder.add_node("master", run_master)

graph_builder.add_edge(START, "master")
# graph_builder.add_edge("master", "chatbot")
graph_builder.add_conditional_edges(
    "master",
    tools_condition,
)

graph_builder.add_edge(
    "tools",
    "master",
)

graph = graph_builder.compile()


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test"}}
    for event in graph.stream(
        {
            "messages": [
                (
                    "user",
                    "Find value of x in equation X^2+2x+5=0, along with the steps used to solve it.",
                )
            ]
        },
        stream_mode="updates",
    ):
        print("\nNODE UPDATE")
        pprint(event)
    print(graph.get_graph().draw_mermaid())
    print("\n=== FINAL RESPONSE ===")
    # print(result["messages"][-1].content)
