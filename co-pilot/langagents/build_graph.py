from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langagents.state import CoPilotState
from langagents.state import create_plan, human_loop


def build():
    graph = StateGraph(CoPilotState)
    graph.add_node("plan", create_plan)
    graph.add_node("feedback", human_loop)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "feedback")
    graph.add_edge("feedback", END)
    active_graph = graph.compile(checkpointer=MemorySaver())
    return active_graph