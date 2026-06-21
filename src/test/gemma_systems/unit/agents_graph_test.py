from unittest.mock import MagicMock, patch

from src.main.gemma_system.agents_graph import graph, run_graph


def test_graph_compiles():
    """Verify graph object exists and compiles."""
    assert graph is not None


def test_graph_contains_expected_nodes():
    """Verify expected nodes are present."""
    graph_structure = graph.get_graph()

    node_names = set(graph_structure.nodes.keys())

    assert "chatbot" in node_names
    assert "master" in node_names
    assert "tools" in node_names


@patch("src.main.gemma_system.agents_graph.graph")
def test_run_graph_returns_final_message(mock_graph):
    """Verify run_graph returns final message content."""

    mock_response = {
        "messages": [
            MagicMock(content="ignored"),
            MagicMock(content="Test response"),
        ]
    }

    mock_graph.invoke.return_value = mock_response
    mock_graph.get_graph.return_value.draw_mermaid.return_value = "mock graph"

    result = run_graph("hello")

    assert result == "Test response"


@patch("src.main.gemma_system.agents_graph.graph")
def test_run_graph_invokes_graph_with_correct_payload(mock_graph):
    """Verify graph.invoke receives expected payload."""

    mock_response = {
        "messages": [
            MagicMock(content="Success"),
        ]
    }

    mock_graph.invoke.return_value = mock_response
    mock_graph.get_graph.return_value.draw_mermaid.return_value = ""

    user_input = "What is AI?"

    run_graph(user_input)

    mock_graph.invoke.assert_called_once()

    args, kwargs = mock_graph.invoke.call_args

    payload = args[0]

    assert payload["messages"][0][0] == "user"
    assert payload["messages"][0][1] == user_input

    assert kwargs["config"] == {"configurable": {"thread_id": "test"}}


@patch("src.main.gemma_system.agents_graph.graph")
def test_run_graph_handles_empty_response(mock_graph):
    """Ensure predictable behavior for malformed output."""

    mock_graph.invoke.return_value = {"messages": [MagicMock(content="")]}

    mock_graph.get_graph.return_value.draw_mermaid.return_value = ""

    result = run_graph("hello")

    assert result == ""
