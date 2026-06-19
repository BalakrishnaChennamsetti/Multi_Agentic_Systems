from pprint import pprint

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from .agents.frontier_agent import run_frontier
from .agents.master_agent import run_master
from .agents_state import State
from .tools.common_tools import get_common_tools

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", run_frontier)
graph_builder.add_node("tools", get_common_tools())
graph_builder.add_node("master", run_master)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "master")
graph_builder.add_conditional_edges(
    "master",
    tools_condition,
)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge(
    "tools",
    "master",
)

graph_builder.add_edge(
    "tools",
    "chatbot",
)

graph_builder.add_edge("master", END)

graph = graph_builder.compile()


def run_graph(user_input: str):
    config = {"configurable": {"thread_id": "test"}}
    # for event in graph.stream(
    #     {
    #         "messages": [
    #             (
    #                 "user",
    #                 # "What is the biggest Prime Number? Any faster ways to find the whether a number is prime or not? Based on ancient Indian mathematics.",
    #                 user_input,
    #             )
    #         ]
    #     },
    #     stream_mode="updates",
    # ):
    #     print("\nNODE UPDATE")
    #     pprint(event)
    result = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    # "What is the biggest Prime Number? Any faster ways to find the whether a number is prime or not? Based on ancient Indian mathematics.",
                    user_input,
                )
            ]
        },
        config=config,
    )
    print(graph.get_graph().draw_mermaid())
    print("\n=== FINAL RESPONSE ===")
    # print(result["messages"][-1].content)
    return result["messages"][-1].content


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test"}}
    for event in graph.stream(
        {
            "messages": [
                (
                    "user",
                    # "What is the biggest Prime Number? Any faster ways to find the whether a number is prime or not? Based on ancient Indian mathematics.",
                    "Find the next prime number after 2^{136,279,841}-1",
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
