from langgraph.types import interrupt
from typing import List, TypedDict, Annotated
from langagents.llm import LLM
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    HumanMessage
)
import operator
import re


class CoPilotState(TypedDict, total=False):
    """State carried through the CoPilot LangGraph workflow."""
    user_query: str
    plan: str
    human_feedback: str
    break_plan: List[str]
    messages: Annotated[list, operator.add]


def human_loop(state: CoPilotState) -> CoPilotState:
    plan = state["plan"]
    # Interrupt is not called second time when the graph resumes.
    resume_value = interrupt(
        {
            "message": "Review the generated plan and resume with feedback or approval(approve/reject).",
            "plan": plan   
        }
    )
    if resume_value == "reject":
        print('Plan rejected by user. Ending workflow.')
        return {
                "human_feedback": "reject",
                "break_plan": []
                }
    # Update works because you are updating a local variable, not the langgraph state directly.
    updates: CoPilotState = {"plan": plan, "human_feedback": "approve"} 
    updates["break_plan"] = re.findall(r"<step>(.*?)</step>", plan, re.DOTALL)
    return updates

def create_plan(state: CoPilotState) -> CoPilotState:
    # Initialize or extend messages
    messages = list(state.get("messages", []))
    messages.append(SystemMessage(content="""You are a helpful assistant. 
You will be given a user's request and your job is to create a clear, step by step plan to address the user's request.
The sub-plan can ideally be broken down into parallelizable steps, but that is not a requirement.
Each sub-plan must be decorate within <step>...</step> tags in the response."""))
    messages.append(HumanMessage(content=state["user_query"]))
    response = LLM.invoke(messages)
    plan = response.content if hasattr(response, "content") else str(response)
    messages.append(AIMessage(content=plan))
    return {"plan": plan, "messages": messages}
    

