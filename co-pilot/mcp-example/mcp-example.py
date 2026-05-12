import asyncio
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- MCP server imports ------------------------------------------------------
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# --- langchain-mcp-adapters --------------------------------------------------
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import stdio_client


# =============================================================================
# PART 2: LangGraph Agent State
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_results: Annotated[list, operator.add]


# =============================================================================
# PART 3: Load MCP tools and build the agent graph
# =============================================================================
async def build_agent_with_mcp_tools():
    """
    Connect to the MCP server, load its tools as LangChain tools,
    and wire them into a LangGraph agent.
    """

    # ── 3a. Launch the MCP server as a subprocess and open a session ─────────
    server_params = {
        "command": "python",
        "args": ["weather_server.py"],
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Initialise the MCP session (handshake)
            await session.initialize()

            # ── 3b. Convert MCP tools → LangChain tools ──────────────────────
            # load_mcp_tools introspects the server's tool schemas and wraps
            # each one in a @tool-decorated function that calls the MCP server.
            mcp_tools = await load_mcp_tools(session)
            # mcp_tools is now a plain list of LangChain BaseTool objects:
            # [get_weather(...), get_forecast(...)]

            print("Loaded MCP tools:", [t.name for t in mcp_tools])

            # ── 3c. Bind tools to the LLM ─────────────────────────────────────
            llm = ChatOpenAI(base_url="http://localhost:8080/v1", api_key="na")
            llm_with_tools = llm.bind_tools(mcp_tools)

            # ── 3d. Define graph nodes ─────────────────────────────────────────

            def call_llm(state: AgentState) -> dict:
                """LLM node: decide whether to call a tool or respond."""
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}

            def should_continue(state: AgentState) -> str:
                """Router: if the LLM wants to call a tool, go to tools node."""
                last = state["messages"][-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    return "tools"
                return "end"

            # ToolNode handles tool_calls on the last AIMessage automatically
            tool_node = ToolNode(mcp_tools)

            # ── 3e. Build the StateGraph ───────────────────────────────────────
            graph = StateGraph(AgentState)

            graph.add_node("llm", call_llm)
            graph.add_node("tools", tool_node)

            graph.set_entry_point("llm")

            graph.add_conditional_edges(
                "llm",
                should_continue,
                {"tools": "tools", "end": END},
            )

            # After tools run, always go back to LLM to interpret results
            graph.add_edge("tools", "llm")

            agent = graph.compile()

            # ── 3f. Run the agent ──────────────────────────────────────────────
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=(
                    "What's the weather in Kolkata? "
                    "Also give me a 3-day forecast."
                ))],
                "tool_results": [],
            })

            print("\n=== Final Answer ===")
            print(result["messages"][-1].content)
            return result



# =============================================================================
# PART 5: Entrypoint
# =============================================================================

if __name__ == "__main__":
    # Run the MCP-backed agent
    asyncio.run(build_agent_with_mcp_tools())

    # Or the inline tool agent (sync)
    # agent = build_inline_tool_agent()
    # result = agent.invoke({
    #     "messages": [HumanMessage(content="What is AAPL's stock price?")],
    #     "tool_results": [],
    # })
    # print(result["messages"][-1].content)