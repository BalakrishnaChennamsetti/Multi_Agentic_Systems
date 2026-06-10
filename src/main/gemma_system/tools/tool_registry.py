from langgraph.prebuilt import ToolNode

from main.gemma_system.tools.common_tools import calculator

tool_node = ToolNode([calculator])
