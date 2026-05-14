import os
import psycopg
from langgraph.graph import END, StateGraph
from langagents.state import CoPilotState
from langagents.state import create_plan, human_loop
from langgraph.checkpoint.postgres import PostgresSaver


CHECKPOINT_DATABASE_URL = os.getenv("CHECKPOINT_DATABASE_URL", os.getenv("DATABASE_URL"))
if not CHECKPOINT_DATABASE_URL:
    raise RuntimeError("Set CHECKPOINT_DATABASE_URL or DATABASE_URL for Postgres checkpoints")

# Keep one connection/checkpointer alive for process lifetime in this POC.
conn = psycopg.connect(CHECKPOINT_DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()


def build():
    graph = StateGraph(CoPilotState)
    graph.add_node("plan", create_plan)
    graph.add_node("feedback", human_loop)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "feedback")
    graph.add_edge("feedback", END)
    active_graph = graph.compile(checkpointer=checkpointer)
    return active_graph