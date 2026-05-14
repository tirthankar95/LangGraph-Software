# Co-Pilot POC: LangGraph, MCP, and Tooling

This repository is a proof-of-concept project that demonstrates advanced agent patterns using:

- LangGraph stateful workflows
- Human-in-the-loop approvals with resume commands
- MCP (Model Context Protocol) server and client integration
- FastAPI backend + SQLAlchemy user auth layer
- Tool-style execution helpers (pytest and shell command runner)

## What This Project Shows

1. Stateful agent flow with checkpoints and interrupts.
2. Human approval loop (`/reply` -> interrupt -> `/approve_reply` resume).
3. MCP tool server pattern (`weather-server.py`) and client/agent usage (`mcp-example.py`).
4. Backend API and persistence-ready user model using PostgreSQL + SQLAlchemy.

## Repository Structure

- `main.py`: FastAPI app with user creation, agent request, and approval endpoints.
- `backend/`: Database config and SQLAlchemy models.
- `langagents/`: LangGraph graph construction, state, LLM wiring, and tools.
- `mcp-example/`: Minimal MCP server/client examples for tool execution.
- `experiment/`: Notebook/scripts for experiments.

## API Flow

### 1) Create user

`POST /users`

Creates a user record required for authenticated requests.

### 2) Start planning request

`POST /reply`

- Validates `user_id`
- Invokes LangGraph with a unique `thread_id`
- Returns an interrupt payload with:
	- human review message
	- generated plan
	- thread id to resume later

### 3) Approve or reject

`POST /approve_reply`

- Uses `Command(resume=...)` with the same `thread_id`
- Continues graph execution from checkpoint
- Returns parsed breakdown/feedback response

## MCP Example

`mcp-example/weather-server.py` exposes example MCP tools:

- `get_weather`
- `get_forecast`

`mcp-example/mcp-example.py` launches the MCP server over stdio, loads tools through adapters, binds them to an LLM, and runs a LangGraph agent loop.

## Prerequisites

- Python 3.12+
- PostgreSQL
- Optional local OpenAI-compatible endpoint (configured in `langagents/llm.py`)

## Setup

1. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e .
```

3. Set database URL:

```bash
export DATABASE_URL="postgresql://copilot_app:strong_password@localhost/copilot"
```

4. Start API server:

```bash
python -m fastapi dev main.py
```

## Swagger UI

After startup, open:

- `http://127.0.0.1:8000/docs`

## Example Requests

Create user:

```json
{
	"user_id": "t.mittra",
	"email": "tmittra@example.com",
	"name": "T Mittra"
}
```

Start reply:

```json
{
	"user_id": "t.mittra",
	"user_request": "Plan a CI/CD migration"
}
```

Approve reply:

```json
{
	"user_id": "t.mittra",
	"thread_id": "t.mittra_dd143356-05bc-4805-b081-8070329dd722",
	"user_request": "approve"
}
```

## Notes

- Checkpointer state is process-local when using `MemorySaver`.
- Restarting the server clears in-memory thread checkpoints.
- For production durability, move to a persistent checkpointer backend.
