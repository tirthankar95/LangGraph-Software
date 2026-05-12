import asyncio
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_results: Annotated[list, operator.add]
    
# =============================================================================
# PART 4: Alternative – inline @tool wrapping an MCP call (simpler pattern)
# =============================================================================
#
# If you don't want a full MCP server subprocess, you can write MCP-style tools
# directly as LangChain @tool functions and still plug them into the same graph.
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    # Replace with a real API call (e.g. yfinance, Alpha Vantage)
    import random
    price = round(random.uniform(50, 500), 2)
    return f"{ticker}: ${price}"

@tool
def summarise_news(topic: str) -> str:
    """Fetch and summarise latest news for a topic."""
    # Replace with NewsAPI / requests call
    return f"Top news for '{topic}': Markets react to latest Fed decision..."


def build_inline_tool_agent():
    """Build a LangGraph agent using plain @tool functions (no MCP server)."""
    inline_tools = [get_stock_price, summarise_news]

    llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="na")
    llm_with_tools = llm.bind_tools(inline_tools)

    def call_llm(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else "end"

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(inline_tools))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")

    return graph.compile()

if __name__ == '__main__':
    inline_agent = build_inline_tool_agent()
    response = inline_agent.invoke({
        "messages": [HumanMessage(content="Get the stock price of google.")]
    })
    response['messages'][-1].pretty_print()