from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    report_structure_planner_node,
    human_feedback_node,
    human_feedback_router
)

memory_saver = MemorySaver()

builder = StateGraph(AgentState)

builder.add_node("report_structure_planner", report_structure_planner_node)
builder.add_node("human_feedback", human_feedback_node)

builder.set_entry_point("report_structure_planner")
builder.add_edge("report_structure_planner", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    human_feedback_router,
)
graph = builder.compile(checkpointer=memory_saver)
