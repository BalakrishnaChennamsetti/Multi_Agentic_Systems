from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

from agents.frontier_agent import run_frontier
from agents_state import State
from tools.common_tools import get_common_tools


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", run_frontier)
graph_builder.add_node("tools", get_common_tools())

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge(
    "tools",
    "chatbot",
)

graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [
                ("user", "What is 235 * 8 + 48?")
            ]
        }
    )

    print("\n=== FINAL RESPONSE ===")
    print(result["messages"][-1].content)