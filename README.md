# agent-llm-mock

Local LLM mock server with OpenAI and Anthropic compatible endpoints, plus a web dashboard for manual response control. Useful for testing AI agent tool-calling behavior, debugging prompt chains, and simulating LLM responses without external API calls.

## Features

- **Dual API support** — OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`
- **Three operating modes** — scripted responses, upstream forwarding, or manual operator control
- **Web dashboard** — real-time request inspection, manual response editing, and tool call mockup
- **Tool call support** — both OpenAI `tool_calls` and Anthropic `tool_use` formats, with auto-extraction from forwarded responses
- **Streaming** — SSE streaming for both endpoints
- **Forward proxy** — transparently proxy requests to real LLM APIs with rule-based matching
- **Zero dependencies beyond FastAPI** — single file server, easy to hack on

## Installation

```bash
pip install agent-llm-mock
```

Or from source:

```bash
git clone https://github.com/your-org/agent-llm-mock.git
cd agent-llm-mock
pip install -e .
```

Requires Python ≥ 3.10.

## Quick Start

```bash
# Start the server
agent-llm-mock --port 9999
```

Open `http://localhost:9999` in your browser. Point your AI agent at:

```
base_url: http://localhost:9999/v1
```

Send a request — it appears in the dashboard. Type a response and click **Submit Response**.

## Usage

### CLI

```bash
agent-llm-mock --port 9999                          # basic usage
agent-llm-mock --port 9999 --scripts rules.json     # with scripted responses
agent-llm-mock --port 9999 --forward-config fw.json  # with upstream forwarding
```

### Python API

```python
from agent_llm_mock import serve

serve(port=9999)                            # blocking
serve(port=9999, scripts_path="rules.json")
serve(port=9999, forward_config_path="fw.json")
```

## Operating Modes

Each incoming request goes through three checks in order:

### 1. Scripted Responses (`--scripts`)

Predefined response rules in a JSON file. Useful for deterministic testing.

```json
[
  {
    "match": {
      "text_contains": ["classify intent", "what is the weather"]
    },
    "response": {
      "content": "{\"intent\": \"tool_call\", \"confidence\": 0.9}",
      "tool_calls": [
        {
          "id": "call_abc",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"Beijing\"}"
          }
        }
      ]
    }
  }
]
```

**Match fields:**
- `text_contains` — matches if any list item appears in any message content
- `api_format` — matches `"openai"` or `"anthropic"`

### 2. Forward Proxy (`--forward-config`)

Proxy requests to real LLM APIs based on rules. The dashboard shows the full upstream request/response, extracted tool calls, and request tool definitions.

```json
[
  {
    "match": {"api_format": "openai"},
    "target_url": "https://api.deepseek.com/v1",
    "timeout": 30
  },
  {
    "match": {"api_format": "anthropic"},
    "target_url": "https://api.anthropic.com/v1",
    "timeout": 60
  }
]
```

The endpoint path (`/chat/completions` or `/messages`) is appended automatically to `target_url`.

### 3. Manual Control (default)

Requests that don't match any script or forward rule are queued. Open the dashboard, expand the request card, type a response and/or fill in tool call parameters, then click **Submit Response**.

## Dashboard

The dashboard (`http://localhost:9999`) updates in real-time via WebSocket.

### Request List
- Shows all requests with status badges: `pending`, `forwarded`, `completed`, `skipped`
- Click a request to expand details

### Pending Requests (Manual Mode)
- **Response Text** — free-form textarea for the assistant's text response
- **Tool Call Mocks** — each tool from the request is shown with:
  - Tool name and description (visible, selectable)
  - Parameter input fields (typed: string, number, boolean, enum, array, object)
  - Enable/disable checkbox per tool
  - Editable call_id
- **Live JSON preview** of assembled `tool_calls`
- Buttons: **Submit Response**, **Skip (empty response)**

### Forwarded Requests
- **Forwarding Rule** — which rule matched
- **Upstream Request** — URL, headers
- **Upstream Response** — status, headers, body
- **Extracted Tool Calls** — tool calls parsed from upstream response, with descriptions cross-referenced from request tool definitions
- **Request Tools** — read-only display of tool definitions (name, description, parameter list)

## API Endpoints

### LLM Endpoints

| Endpoint | Format |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages` | Anthropic Messages |

Both support `stream: true` for SSE streaming.

### Dashboard API

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/requests` | List all requests |
| `GET /api/requests/{id}` | Get a single request |
| `POST /api/requests/{id}/respond` | Submit manual response |
| `POST /api/requests/{id}/skip` | Skip (empty response) |
| `WS /ws` | WebSocket for real-time updates |

### POST `/api/requests/{id}/respond`

```json
{
  "content": "The assistant's text response",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Beijing\"}"
      }
    }
  ]
}
```

## Project Structure

```
agent_llm_mock/
  __init__.py      — Public API: serve(), main(), MockLLMServer
  __main__.py      — python -m agent_llm_mock
  server.py        — Server implementation (single file)
  scripts/
    response.json              — Example scripted response
    forward_rules.example.json — Example forward config
pyproject.toml    — Package metadata
```

## Architecture

```
                    ┌──────────────────┐
                    │   AI Agent / App  │
                    └────────┬─────────┘
                             │ POST /v1/chat/completions
                             │ POST /v1/messages
                             ▼
                 ┌───────────────────────┐
                 │   agent-llm-mock       │
                 │                        │
                 │  1. Script match? ────► Return scripted response
                 │  2. Forward match? ───► Proxy to upstream LLM
                 │  3. Otherwise ────────► Queue for operator
                 └───────────┬───────────┘
                             │ WebSocket
                             ▼
                 ┌───────────────────────┐
                 │   Web Dashboard        │
                 │   http://localhost:N   │
                 │                        │
                 │  • View requests       │
                 │  • Edit responses      │
                 │  • Mock tool calls     │
                 │  • Inspect forwarded   │
                 └───────────────────────┘
```

## Tool Call Format

Tools in requests and responses use OpenAI's function-calling format:

```json
// Request tool definition
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"}
      },
      "required": ["city"]
    }
  }
}

// Response tool call
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\": \"Beijing\"}"
  }
}
```

Anthropic-format tools are auto-converted for dashboard display. Both formats are supported transparently.

## License

MIT
