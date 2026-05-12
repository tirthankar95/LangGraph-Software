from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio, json, random

app = Server("weather-server")

# ── Tool 1: get_weather ──────────────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        ),
        types.Tool(
            name="get_forecast",
            description="Get 3-day weather forecast for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Number of days (1-3)"},
                },
                "required": ["city"],
            },
        ),
    ]

# ── Tool implementations ─────────────────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        # Simulated weather data (replace with real API call)
        weather = {
            "city": city,
            "temperature": random.randint(15, 35),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Windy"]),
            "humidity": random.randint(40, 90),
        }
        return [types.TextContent(type="text", text=json.dumps(weather))]

    elif name == "get_forecast":
        city = arguments["city"]
        days = arguments.get("days", 3)
        forecast = [
            {
                "day": i + 1,
                "temperature": random.randint(15, 35),
                "condition": random.choice(["Sunny", "Cloudy", "Rainy"]),
            }
            for i in range(days)
        ]
        return [types.TextContent(type="text", text=json.dumps({"city": city, "forecast": forecast}))]

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())