"""Human-in-the-loop trade approval flow built with LangGraph.

Flow:
1) Analyze trade risk with an LLM.
2) If high risk, interrupt and request human approval.
3) Execute or reject the trade based on decision.
"""

import re
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    HumanMessage
)

# 1. State Definition
class State(TypedDict):
    # add_messages appends new messages and preserves chat history correctly.
    messages: Annotated[list[BaseMessage], add_messages]
    risk_score: float
    requires_approval: bool
    is_approved: Literal["approved", "rejected", "NA"]

# 2. Structured Output
class AnalyzerOutput(BaseModel):
    risk: Literal["low", "medium", "high"] = Field(description="Risk category for trade")
    reason: str = Field(description="What is your reasoning for assigning this category?")

# LLMs
model = ChatOpenAI(base_url='http://localhost:8080/v1', api_key="na")
structured_model = model.with_structured_output(AnalyzerOutput)


def _risk_to_score(risk: Literal["low", "medium", "high"]) -> float:
    return {"low": 0.1, "medium": 0.5, "high": 0.9}[risk]


def _extract_risk_from_text(text: str) -> Literal["low", "medium", "high"]:
    lowered = text.lower()
    for risk in ("high", "medium", "low"):
        if re.search(rf"\b{risk}\b", lowered):
            return risk
    return "high"


def _safe_analyze(prompt: list[BaseMessage]) -> AnalyzerOutput:
    try:
        analysis = structured_model.invoke(prompt)
        if analysis:
            return analysis
    except Exception:
        pass

    fallback = model.invoke(prompt)
    fallback_text = fallback.content if isinstance(fallback.content, str) else str(fallback.content)
    risk = _extract_risk_from_text(fallback_text)
    reason = fallback_text.strip() or "Fallback parser used due to missing structured output."
    return AnalyzerOutput(risk=risk, reason=reason)


def analyzer_node(state: State):
    # Evaluate trade risk from the latest user message.
    prompt = [
        SystemMessage(content="You are a trade risk evaluator. If the user tries to sell more than 100,000 units, flag it as high risk."),
        HumanMessage(content = state["messages"][-1].content)
    ]
    analysis = _safe_analyze(prompt)
    risk_score = _risk_to_score(analysis.risk)
    requires_approval = risk_score >= 0.5
    return {
        "messages": [AIMessage(content=f"Risk={analysis.risk}. Reason: {analysis.reason}")],
        "risk_score": risk_score,
        "requires_approval": requires_approval,
        "is_approved": "NA",
    }


def execution_node(state: State):
    if state["is_approved"] in ("approved", "NA"):
        return {"messages": [AIMessage(content="Trade executed successfully.")]}
    return {"messages": [AIMessage(content="Trade rejected.")]}


def human_gate(state: State):
    # First call: interrupt() pauses the graph and returns control to the caller.
    # Second call (on resume): interrupt() returns the value passed via Command(resume=...).
    if state["requires_approval"] and state.get("is_approved") == "NA":
        decision = interrupt("High risk trade. Awaiting approval.")
        return {"is_approved": decision}
    return {}

# 3. Build Graph with Persistence
workflow = StateGraph(State)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("executor", execution_node)
workflow.add_node("human_gate", human_gate)
workflow.set_entry_point("analyzer")

# Conditional Edge: Only proceed if risk is low
workflow.add_conditional_edges(
    "analyzer",
    lambda x: "executor" if not x["requires_approval"] else "human_gate"
)
workflow.add_edge("human_gate", "executor")

# 4. The "Secret Sauce": Checkpointer and Breakpoints
app = workflow.compile(checkpointer=MemorySaver())

def run_demo() -> None:
    config = {"configurable": {"thread_id": "123"}}
    initial_input = {"messages": [HumanMessage(content="Sell 200,000 units of Asset X")]}
    # initial_input = {"messages": [HumanMessage(content="Sell 100,000 units of Asset X")]}

    print("--- Starting AI Analysis ---")
    for event in app.stream(initial_input, config, stream_mode="values"):
        print(event)

    state = app.get_state(config).values
    if state["requires_approval"]:
        decision = input("Approve? (approved/rejected): ").strip().lower()
        if decision not in {"approved", "rejected"}:
            decision = "rejected"
        # Pass the decision string directly — interrupt() returns this value on the second run.
        app.invoke(Command(resume=decision), config)

    final_msg = app.get_state(config).values["messages"][-1]
    print(final_msg.content)


if __name__ == "__main__":
    run_demo()

