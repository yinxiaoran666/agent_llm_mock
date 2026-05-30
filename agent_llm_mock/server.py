"""Mock LLM server — OpenAI-compatible and Anthropic endpoints with web dashboard for manual response control."""

import asyncio
import json
import logging
import sys
import time
import uuid
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("agent-llm-mock")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
))
logger.addHandler(_console_handler)

# Keep noisy lib logs at WARNING, uvicorn.access at INFO for audit trail
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PendingRequest:
    id: str
    model: str
    messages: List[Dict]
    tools: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    response_format: Optional[Dict] = None
    extra_body: Optional[Dict] = None
    api_format: str = "openai"       # "openai" | "anthropic"
    system_prompt: Optional[str] = None  # Anthropic top-level system
    created_at: float = field(default_factory=time.time)
    status: str = "pending"          # pending | completed | skipped | forwarded
    response_content: Optional[str] = None
    response_tool_calls: Optional[List[Dict]] = None
    event: threading.Event = field(default_factory=threading.Event)
    # Forwarding fields
    raw_request_body: Optional[Dict] = None
    raw_request_headers: Optional[Dict] = None
    forwarded_request: Optional[Dict] = None   # {url, headers, body}
    forwarded_response: Optional[Dict] = None  # {status_code, headers, body}
    forwarded_rule: Optional[Dict] = None      # matched rule
    forwarded_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text_or_messages) -> int:
    """Rough token estimate: ~4 chars per token."""
    if isinstance(text_or_messages, str):
        return max(1, len(text_or_messages) // 4)
    if isinstance(text_or_messages, list):
        total = 0
        for m in text_or_messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
        return max(1, total // 4)
    return 0


def _build_chat_completion(req: PendingRequest) -> dict:
    """Wrap the operator's text response into a valid OpenAI ChatCompletion."""
    content = req.response_content or ""
    tool_calls = req.response_tool_calls
    finish_reason = "tool_calls" if tool_calls else "stop"

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    prompt_tokens = _estimate_tokens(req.messages)
    completion_tokens = _estimate_tokens(content)

    return {
        "id": f"chatcmpl-mock-{req.id[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Anthropic Messages API helpers
# ---------------------------------------------------------------------------

def _convert_openai_tool_to_anthropic(tool: dict) -> dict:
    """OpenAI tool → Anthropic tool format."""
    fn = tool.get("function", tool)
    return {
        "name": fn.get("name", "unknown"),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _convert_tool_calls_to_anthropic_content(tool_calls: list) -> list:
    """OpenAI tool_calls → Anthropic content blocks (tool_use)."""
    blocks = []
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            inp = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", "toolu_" + uuid.uuid4().hex[:12]),
            "name": fn.get("name", "unknown"),
            "input": inp,
        })
    return blocks


def _build_anthropic_response(req: PendingRequest) -> dict:
    """Build an Anthropic-format Message response from operator input."""
    content = []
    if req.response_content:
        content.append({"type": "text", "text": req.response_content})
    if req.response_tool_calls:
        content.extend(_convert_tool_calls_to_anthropic_content(req.response_tool_calls))

    stop_reason = "end_turn"
    if req.response_tool_calls:
        stop_reason = "tool_use"

    prompt_tokens = _estimate_tokens(req.messages)
    if req.system_prompt:
        prompt_tokens += _estimate_tokens(req.system_prompt)
    completion_tokens = _estimate_tokens(req.response_content or "")

    return {
        "id": f"msg_mock-{req.id[:12]}",
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def _anthropic_stream_response(req: PendingRequest):
    """SSE streaming response in Anthropic format."""
    msg = _build_anthropic_response(req)

    async def event_stream():
        # message_start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': msg}, ensure_ascii=False)}\n\n"
        # content blocks
        for i, block in enumerate(msg["content"]):
            if block["type"] == "text":
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'text', 'text': ''}}, ensure_ascii=False)}\n\n"
                text = block.get("text", "")
                chunk_size = 20
                for pos in range(0, len(text), chunk_size):
                    snippet = text[pos:pos + chunk_size]
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'text_delta', 'text': snippet}}, ensure_ascii=False)}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i}, ensure_ascii=False)}\n\n"
            elif block["type"] == "tool_use":
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}}, ensure_ascii=False)}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block.get('input', {}), ensure_ascii=False)}}, ensure_ascii=False)}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i}, ensure_ascii=False)}\n\n"
        # message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': msg['stop_reason'], 'stop_sequence': None}, 'usage': {'output_tokens': msg['usage']['output_tokens']}}, ensure_ascii=False)}\n\n"
        # message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Scripted responses
# ---------------------------------------------------------------------------

def _load_scripts(path: Optional[str]) -> List[Dict]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _extract_text_content(content) -> str:
    """Extract plain text from OpenAI/Anthropic content field (str | list | None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("type", "")
                if t == "text":
                    parts.append(part.get("text", ""))
                elif t in ("image_url", "image"):
                    parts.append("[image]")
                elif t == "input_audio":
                    parts.append("[audio]")
                elif t == "file":
                    parts.append("[file]")
                elif t == "tool_use":
                    inp = json.dumps(part.get("input", {}), ensure_ascii=False)
                    parts.append(f"[tool_use: {part.get('name', 'unknown')}({inp})]")
                elif t == "tool_result":
                    inner = part.get("content", "")
                    parts.append(f"[tool_result: {_extract_text_content(inner)}]")
                else:
                    parts.append(str(part))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return str(content)


def _match_script(req: PendingRequest, scripts: List[Dict]) -> Optional[Dict]:
    """Return the first matching script entry, or None."""
    combined = " ".join(
        _extract_text_content(m.get("content"))
        for m in req.messages
        if m.get("role") in ("user", "system")
    ).lower()
    for script in scripts:
        match_cfg = script.get("match", {})
        texts = match_cfg.get("text_contains", [])
        if texts and all(t.lower() in combined for t in texts):
            return script.get("response", {})
    return None


# ---------------------------------------------------------------------------
# Forwarding helpers
# ---------------------------------------------------------------------------

def _load_forward_config(path: Optional[str]) -> List[Dict]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load forward config from %s: %s", path, exc)
        return []


def _match_forward_rule(req: PendingRequest, forward_config: List[Dict]) -> Optional[Dict]:
    """Return the first forward rule matching this request, or None.

    Rules evaluated in order; first match wins.
    All match fields are AND-ed. Omitted fields match everything.
    """
    if not forward_config:
        return None

    combined = " ".join(
        _extract_text_content(m.get("content"))
        for m in req.messages
        if m.get("role") in ("user", "system")
    ).lower()

    for rule in forward_config:
        match_cfg = rule.get("match", {})
        if not match_cfg:
            return rule

        api_format = match_cfg.get("api_format")
        model_contains = match_cfg.get("model_contains")
        texts = match_cfg.get("text_contains", [])
        stream_match = match_cfg.get("stream")

        if api_format and api_format != req.api_format:
            continue
        if model_contains and model_contains.lower() not in req.model.lower():
            continue
        if texts and not all(t.lower() in combined for t in texts):
            continue
        if stream_match is not None and stream_match != req.stream:
            continue

        return rule

    return None


def _build_forward_url(rule: Dict, req: PendingRequest) -> str:
    """Build the full upstream URL from the rule's target_url and the request path.

    target_url should be the API base path, e.g.:
      - OpenAI/DeepSeek:  https://api.deepseek.com/v1
      - Anthropic:         https://api.anthropic.com/v1

    The endpoint path (/chat/completions or /messages) is appended automatically.
    The resulting URL is stored in forwarded_request.url for dashboard display.
    """
    target_base = rule["target_url"].rstrip("/")
    if req.api_format == "openai":
        url = f"{target_base}/chat/completions"
    else:
        url = f"{target_base}/messages"
    logger.debug("FORWARD %s -> URL %s", req.id, url)
    return url


async def _forward_request(
    req: PendingRequest,
    rule: Dict,
    client: httpx.AsyncClient,
    request_headers: Dict[str, str],
) -> Optional[Dict]:
    """Forward a non-streaming request upstream. Returns response dict or None on failure."""
    url = _build_forward_url(rule, req)
    timeout = rule.get("timeout", 30)
    body = deepcopy(req.raw_request_body) if req.raw_request_body else {}

    # Build forwarding headers (pass through client headers selectively)
    forward_headers = {"Content-Type": "application/json"}
    for h in ("authorization", "x-api-key", "anthropic-version", "x-custom"):
        if h in request_headers:
            forward_headers[h] = request_headers[h]

    req.forwarded_request = {"url": url, "headers": dict(forward_headers), "body": body}

    try:
        resp = await client.post(url, json=body, headers=forward_headers, timeout=timeout)
        try:
            resp_body = resp.json()
        except json.JSONDecodeError:
            resp_body = resp.text

        req.forwarded_response = {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp_body,
        }
        return req.forwarded_response
    except httpx.TimeoutException:
        logger.warning("FORWARD %s -> TIMEOUT after %ds", req.id, timeout)
        req.forwarded_error = f"Timeout after {timeout}s"
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning("FORWARD %s -> HTTP %s: %s", req.id, exc.response.status_code,
                       exc.response.text[:500] if exc.response.text else "(no body)")
        req.forwarded_error = f"HTTP {exc.response.status_code}"
        req.forwarded_response = {
            "status_code": exc.response.status_code,
            "headers": dict(exc.response.headers),
            "body": exc.response.text[:2000] if exc.response.text else "",
        }
        return None
    except httpx.RequestError as exc:
        logger.warning("FORWARD %s -> REQUEST_ERROR: %s", req.id, exc)
        req.forwarded_error = f"Request error: {exc}"
        return None


async def _proxy_stream_response(
    req: PendingRequest,
    rule: Dict,
    client: httpx.AsyncClient,
    request_headers: Dict[str, str],
) -> StreamingResponse:
    """Forward a streaming request upstream and proxy the SSE back to the client."""
    url = _build_forward_url(rule, req)
    timeout = rule.get("timeout", 30)
    body = deepcopy(req.raw_request_body) if req.raw_request_body else {}

    forward_headers = {"Content-Type": "application/json"}
    for h in ("authorization", "x-api-key", "anthropic-version", "x-custom"):
        if h in request_headers:
            forward_headers[h] = request_headers[h]

    req.forwarded_request = {"url": url, "headers": dict(forward_headers), "body": body}

    async def stream_proxy():
        buffer = bytearray()
        upstream_status = None
        upstream_resp_headers = {}
        error = None
        try:
            async with client.stream("POST", url, json=body, headers=forward_headers,
                                     timeout=timeout) as upstream_resp:
                upstream_status = upstream_resp.status_code
                upstream_resp_headers = dict(upstream_resp.headers)
                async for chunk in upstream_resp.aiter_bytes():
                    buffer.extend(chunk)
                    yield chunk
        except Exception as exc:
            logger.warning("FORWARD_STREAM %s -> FAILED: %s", req.id, exc)
            error = str(exc)
        finally:
            full_text = buffer.decode("utf-8", errors="replace")
            req.forwarded_response = {
                "status_code": upstream_status or 0,
                "headers": upstream_resp_headers,
                "body": full_text,
                "stream": True,
            }
            if error:
                req.forwarded_error = error
            req.status = "forwarded"
            req.event.set()
            try:
                await req._broadcast_callback({
                    "type": "request_updated",
                    "request": req._serialize_callback(req),
                    "pending": req._stats_callback()["pending"],
                    "completed": req._stats_callback()["completed"],
                })
            except Exception:
                pass

    return StreamingResponse(stream_proxy(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Web dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Mock Dashboard</title>
<style>
  :root { --bg: #1a1a2e; --card: #16213e; --text: #e0e0e0; --accent: #0f3460;
          --green: #4caf50; --orange: #ff9800; --red: #f44336; --blue: #2196f3;
          --purple: #7c3aed; --border: #2a2a4a; --input-bg: #0d0d1a; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
  .header h1 { font-size: 1.5rem; }
  .stats { display: flex; gap: 16px; }
  .stat { background: var(--card); padding: 8px 16px; border-radius: 8px; }
  .stat .num { font-size: 1.4rem; font-weight: bold; }
  .stat.pending .num { color: var(--orange); }
  .stat.completed .num { color: var(--green); }
  .req-list { display: flex; flex-direction: column; gap: 12px; }
  .req-card { background: var(--card); border-radius: 10px; overflow: hidden; transition: border 0.15s; }
  .req-card.expanded { border: 2px solid var(--blue); }
  .req-card-header { padding: 14px; cursor: pointer; user-select: none; }
  .req-card-header:hover { background: #1a2550; }
  .req-card-header .meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .req-card-header .model { font-size: 0.85rem; color: #888; }
  .req-card-header .time { font-size: 0.8rem; color: #666; }
  .req-card-header .preview { font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90vw; }
  .req-card-header .badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }
  .req-card-header .expand-icon { color: #666; margin-left: 8px; font-size: 0.7rem; transition: transform 0.2s; display: inline-block; }
  .req-card.expanded .expand-icon { transform: rotate(180deg); }
  .badge.pending { background: var(--orange); color: #000; }
  .badge.completed { background: var(--green); color: #000; }
  .badge.skipped { background: #666; color: #fff; }
  .badge.forwarded { background: var(--blue); color: #fff; }
  .badge.forwarding { background: var(--purple); color: #fff; }
  .req-card-detail { display: none; padding: 0 16px 16px; border-top: 1px solid var(--border); }
  .req-card.expanded .req-card-detail { display: block; }
  .section-title { font-size: 1rem; font-weight: bold; margin: 14px 0 6px; color: #aaa; }
  .raw-json { background: var(--input-bg); border-radius: 6px; padding: 12px; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.8rem; white-space: pre-wrap; word-break: break-all; max-height: 260px; overflow-y: auto; border: 1px solid var(--border); }
  .detail-response-text { width: 100%; padding: 12px; border-radius: 6px; border: 1px solid #444; background: var(--input-bg); color: var(--text); font-family: inherit; font-size: 0.9rem; resize: vertical; min-height: 80px; display: block; }
  .btn-row { display: flex; gap: 8px; margin-top: 16px; }
  .btn { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn-submit { background: var(--green); color: #fff; }
  .btn-skip { background: var(--red); color: #fff; }
  .empty-state { text-align: center; padding: 40px; color: #666; }
  .empty-state .icon { font-size: 2rem; }

  /* Forward URL display */
  .fwd-url-box { background: #0a0a18; border: 1px solid #444; border-radius: 6px; padding: 10px 14px;
    font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.85rem; color: #8be9fd;
    word-break: break-all; margin-bottom: 4px; }
  .fwd-method { background: var(--green); color: #000; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 700; margin-right: 6px; }

  /* Tool mock module */
  .tool-module { background: var(--input-bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
  .tool-module-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #0f0f23; cursor: pointer; user-select: none; }
  .tool-module-header:hover { background: #141430; }
  .tool-module-header .tool-name { font-weight: 600; color: var(--purple); font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.9rem; }
  .tool-module-header .collapse-icon { color: #888; transition: transform 0.2s; font-size: 0.7rem; }
  .tool-module.collapsed .collapse-icon { transform: rotate(-90deg); }
  .tool-module-body { padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; }
  .tool-module.collapsed .tool-module-body { display: none; }
  .tool-enable-row { display: flex; align-items: center; gap: 8px; margin-bottom: 2px; }
  .tool-enable-row input[type=checkbox] { width: 16px; height: 16px; accent-color: var(--green); cursor: pointer; }
  .tool-enable-row label { font-size: 0.85rem; color: #ccc; cursor: pointer; }
  .field-row { display: flex; align-items: center; gap: 10px; }
  .field-row label { min-width: 100px; font-size: 0.82rem; color: #888; text-align: right; }
  .field-row label.required::after { content: ' *'; color: var(--red); }
  .field-row input, .field-row select, .field-row textarea { flex: 1; padding: 6px 10px; border-radius: 4px; border: 1px solid #444; background: #0a0a18; color: var(--text); font-family: inherit; font-size: 0.85rem; }
  .field-row input:focus, .field-row select:focus, .field-row textarea:focus { outline: none; border-color: var(--blue); }
  .field-row textarea { font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.78rem; min-height: 50px; resize: vertical; }
  .field-row select { cursor: pointer; }
  .call-id-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .call-id-row input { flex: 1; padding: 5px 8px; border-radius: 4px; border: 1px solid #444; background: #0a0a18; color: #888; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.78rem; }
  .call-id-row button { padding: 4px 10px; font-size: 0.75rem; background: #2a2a4a; color: #aaa; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
  .call-id-row button:hover { background: #3a3a5a; }

  /* JSON preview */
  .preview-box { background: #0a0a18; border: 1px solid #333; border-radius: 6px; padding: 12px; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.78rem; white-space: pre-wrap; max-height: 200px; overflow-y: auto; color: #8be9fd; }
  .tools-empty { font-size: 0.85rem; color: #666; padding: 8px 0; }

  @media (max-width: 768px) {
    body { padding: 10px; }
    .req-card-header .preview { max-width: 70vw; }
    .field-row { flex-direction: column; align-items: flex-start; gap: 4px; }
    .field-row label { text-align: left; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>LLM Mock Dashboard</h1>
  <div class="stats">
    <div class="stat pending"><div class="num" id="pendingCount">0</div>Pending</div>
    <div class="stat completed"><div class="num" id="completedCount">0</div>Done</div>
  </div>
</div>

<div class="req-list" id="reqList">
  <div class="empty-state">
    <div class="icon">&#128179;</div>
    <p>Waiting for requests...</p>
    <p style="font-size:0.8rem;margin-top:4px;">OpenAI: /v1/chat/completions  |  Anthropic: /v1/messages</p>
  </div>
</div>

<script>
let expandedId = null;
let requestCache = {};
let ws = null;

function genCallId() { return 'call_' + crypto.randomUUID().replace(/-/g,'').substring(0,12); }

function esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function extractPreview(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    var parts = [];
    content.forEach(function(p) {
      if (!p || typeof p !== 'object') { parts.push(String(p)); return; }
      if (p.type === 'text') parts.push(p.text || '');
      else if (p.type === 'tool_use') parts.push('[tool_use: ' + (p.name || '') + ']');
      else if (p.type === 'tool_result') parts.push('[tool_result]');
      else if (p.type === 'image' || p.type === 'image_url') parts.push('[image]');
      else parts.push('[' + (p.type || 'block') + ']');
    });
    return parts.join(' ');
  }
  return String(content || '');
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

// ---- WebSocket ----
function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    if (msg.type === 'init') handleInit(msg);
    else if (msg.type === 'new_request') handleNewRequest(msg);
    else if (msg.type === 'request_updated') handleRequestUpdated(msg);
  };
  ws.onclose = function() { setTimeout(connectWs, 2000); };
  ws.onerror = function() { ws.close(); };
}

function handleInit(msg) {
  requestCache = {};
  (msg.requests || []).forEach(function(r){ requestCache[r.id] = r; });
  document.getElementById('pendingCount').textContent = msg.pending || 0;
  document.getElementById('completedCount').textContent = msg.completed || 0;
  buildAllCards(msg.requests || []);
}

function buildAllCards(requests) {
  var el = document.getElementById('reqList');
  if (!requests || requests.length === 0) {
    el.innerHTML = '<div class="empty-state"><div class="icon">&#128179;</div><p>Waiting for requests...</p><p style=\"font-size:0.8rem;margin-top:4px;\">Point your agent\'s base_url to http://HOST:PORT/v1</p></div>';
    return;
  }
  var html = '';
  requests.forEach(function(r) {
    html += buildCardHTML(r);
  });
  el.innerHTML = html;
}

function buildCardHTML(r) {
  var isExpanded = r.id === expandedId;
  return '<div class="req-card'+(isExpanded?' expanded':'')+'" id="card-'+r.id+'">'
    + buildCardHeaderHTML(r)
    + '<div class="req-card-detail" id="detail-'+r.id+'">'
    + (isExpanded ? buildDetailHTML(r) : '')
    + '</div></div>';
}

function buildCardHeaderHTML(r) {
  var lastUser = (r.messages || []).filter(function(m){ return m.role === 'user'; }).pop();
  var preview = lastUser ? extractPreview(lastUser.content) : '(system prompt)';
  var toolCount = r.tools ? r.tools.length : 0;
  var badges = toolCount > 0 ? '<span style=\"color:#888;font-size:0.75rem;\">+'+toolCount+' tools</span>' : '';
  return '<div class=\"req-card-header\" onclick=\"toggleRequest(\''+r.id+'\')\">'
    +'<div class=\"meta\"><span class=\"model\">'+esc(r.model)+' '+badges+'</span>'
    +'<span class=\"time\">'+new Date(r.created_at*1000).toLocaleTimeString()
    +' <span class=\"expand-icon\">&#9660;</span></span></div>'
    +'<div class=\"preview\">'+esc(preview)+'</div>'
    +'<span class=\"badge '+r.status+'\">'+r.status+'</span>'
    +'</div>';
}

function handleNewRequest(msg) {
  requestCache[msg.request.id] = msg.request;
  document.getElementById('pendingCount').textContent = msg.pending || 0;
  document.getElementById('completedCount').textContent = msg.completed || 0;
  var el = document.getElementById('reqList');
  var emptyState = el.querySelector('.empty-state');
  if (emptyState) emptyState.remove();
  var card = document.createElement('div');
  card.className = 'req-card';
  card.id = 'card-' + msg.request.id;
  card.innerHTML = buildCardHeaderHTML(msg.request)
    + '<div class=\"req-card-detail\" id=\"detail-'+msg.request.id+'\"></div>';
  el.insertBefore(card, el.firstChild);
}

function handleRequestUpdated(msg) {
  requestCache[msg.request.id] = msg.request;
  document.getElementById('pendingCount').textContent = msg.pending || 0;
  document.getElementById('completedCount').textContent = msg.completed || 0;
  var card = document.getElementById('card-'+msg.request.id);
  if (!card) return;
  var badge = card.querySelector('.badge');
  if (badge) {
    badge.textContent = msg.request.status;
    badge.className = 'badge ' + msg.request.status;
  }
  // If expanded and now non-pending, hide action buttons
  if (msg.request.id === expandedId && msg.request.status !== 'pending') {
    var detail = document.getElementById('detail-'+msg.request.id);
    if (detail) {
      // For forwarded, rebuild detail to show upstream request/response
      if (msg.request.status === 'forwarded') {
        detail.innerHTML = buildDetailHTML(msg.request);
      } else {
        var btnRow = detail.querySelector('.btn-row');
        if (btnRow) btnRow.style.display = 'none';
      }
    }
  }
}

// ---- Detail HTML builder ----
function buildDetailHTML(r) {
  var tools = r.tools || [];
  var existingTC = r.response_tool_calls || [];
  var existingByName = {};
  existingTC.forEach(function(tc) {
    if (tc.function && tc.function.name) existingByName[tc.function.name] = tc;
  });

  var toolsHTML = '';
  if (tools.length === 0) {
    toolsHTML = '<div class=\"tools-empty\">(no tools in this request)</div>';
  } else {
    toolsHTML = '<div class=\"section-title\">Tool Call Mocks ('+tools.length+' tool'+(tools.length>1?'s':'')+')</div>';
    tools.forEach(function(tool, idx) {
      var fn = tool.function || {};
      var name = fn.name || 'unknown_tool_'+idx;
      var params = fn.parameters || {};
      var props = params.properties || {};
      var required = params.required || [];
      var existing = existingByName[name] || {};
      var wasEnabled = !!existing.id;

      var callId = existing.id || genCallId();
      var existingArgs = {};
      if (existing.function && existing.function.arguments) {
        try { existingArgs = JSON.parse(existing.function.arguments); } catch(e) {}
      }

      var fieldsHTML = Object.keys(props).map(function(propName) {
        var prop = props[propName] || {};
        var isReq = required.indexOf(propName) >= 0;
        var val = existingArgs[propName] !== undefined ? existingArgs[propName] : (prop.default !== undefined ? prop.default : '');
        var desc = prop.description ? ' title=\"'+esc(prop.description)+'\"' : '';

        if (prop.enum && Array.isArray(prop.enum)) {
          var opts = (isReq ? '' : '<option value=\"\">-- not set --</option>');
          prop.enum.forEach(function(v) {
            opts += '<option value=\"'+esc(String(v))+'\"'+(String(val)===String(v)?' selected':'')+'>'+esc(String(v))+'</option>';
          });
          return '<div class=\"field-row\"><label class=\"'+(isReq?'required':'')+'\"'+desc+'>'+esc(propName)+'</label>'
            +'<select data-field=\"'+esc(propName)+'\" data-type=\"enum\">'+opts+'</select></div>';
        }
        if (prop.type === 'boolean') {
          var bopts = '<option value=\"\">-- not set --</option>';
          ['true','false'].forEach(function(v) { bopts += '<option value=\"'+v+'\"'+(String(val)===v?' selected':'')+'>'+v+'</option>'; });
          return '<div class=\"field-row\"><label class=\"'+(isReq?'required':'')+'\"'+desc+'>'+esc(propName)+'</label>'
            +'<select data-field=\"'+esc(propName)+'\" data-type=\"boolean\">'+bopts+'</select></div>';
        }
        if (prop.type === 'number' || prop.type === 'integer') {
          return '<div class=\"field-row\"><label class=\"'+(isReq?'required':'')+'\"'+desc+'>'+esc(propName)+'</label>'
            +'<input type=\"number\" data-field=\"'+esc(propName)+'\" data-type=\"'+prop.type+'\" value=\"'+esc(String(val))+'\"'
            +(prop.minimum!==undefined?' min=\"'+prop.minimum+'\"':'')+(prop.maximum!==undefined?' max=\"'+prop.maximum+'\"':'')+' /></div>';
        }
        if (prop.type === 'array' || prop.type === 'object') {
          var jsonVal = typeof val === 'object' ? JSON.stringify(val) : String(val);
          return '<div class=\"field-row\"><label class=\"'+(isReq?'required':'')+'\"'+desc+'>'+esc(propName)+'</label>'
            +'<textarea data-field=\"'+esc(propName)+'\" data-type=\"'+prop.type+'\" rows=\"2\">'+esc(jsonVal)+'</textarea></div>';
        }
        return '<div class=\"field-row\"><label class=\"'+(isReq?'required':'')+'\"'+desc+'>'+esc(propName)+'</label>'
          +'<input type=\"text\" data-field=\"'+esc(propName)+'\" data-type=\"'+(prop.type||'string')+'\" value=\"'+esc(String(val))+'\" /></div>';
      }).join('');

      if (!fieldsHTML) {
        fieldsHTML = '<div style=\"color:#666;font-size:0.8rem;\">(no parameters defined)</div>';
      }

      toolsHTML += '<div class=\"tool-module'+(wasEnabled?'':' collapsed')+'\" id=\"toolMod-'+r.id+'-'+idx+'\">'
        +'<div class=\"tool-module-header\" onclick=\"event.stopPropagation();document.getElementById(\'toolMod-'+r.id+'-'+idx+'\').classList.toggle(\'collapsed\')\">'
        +'<span class=\"tool-name\">'+esc(name)+'</span>'
        +'<span style=\"font-size:0.78rem;color:#666;\">params: '+Object.keys(props).length+'</span>'
        +'<span class=\"collapse-icon\">&#9660;</span></div>'
        +'<div class=\"tool-module-body\">'
        +'<div class=\"tool-enable-row\">'
        +'<input type=\"checkbox\" id=\"enable-'+r.id+'-'+idx+'\" '+(wasEnabled?'checked':'')+' onchange=\"onToolToggle(\''+r.id+'\')\">'
        +'<label for=\"enable-'+r.id+'-'+idx+'\">Include this tool call in response</label></div>'
        +'<div class=\"call-id-row\"><span style=\"font-size:0.78rem;color:#888;\">call_id:</span>'
        +'<input type=\"text\" id=\"callId-'+r.id+'-'+idx+'\" value=\"'+esc(callId)+'\" />'
        +'<button onclick=\"event.stopPropagation();document.getElementById(\'callId-'+r.id+'-'+idx+'\').value=genCallId()\">Regen</button></div>'
        +fieldsHTML
        +'</div></div>';
    });

    toolsHTML += '<div id=\"preview-'+r.id+'\" style=\"display:none;\">'
      +'<div class=\"section-title\">Assembled tool_calls (JSON preview)</div>'
      +'<div class=\"preview-box\" id=\"previewContent-'+r.id+'\"></div></div>';
  }

  // ---- Forwarded request/response panel (read-only) ----
  function buildForwardedHTML(r) {
    var html = '';
    html += '<div class=\"section-title\">Forwarding Rule</div>';
    html += '<div class=\"raw-json\">'+esc(JSON.stringify(r.forwarded_rule||{}, null, 2))+'</div>';

    if (r.forwarded_request) {
      var fr = r.forwarded_request;
      html += '<div class=\"section-title\">Upstream Request</div>';
      html += '<div class=\"fwd-url-box\"><span class=\"fwd-method\">POST</span> '+esc(fr.url||'?')+'</div>';
      html += '<div class=\"section-title\" style=\"margin-top:12px;\">Request Headers</div>';
      html += '<div class=\"raw-json\">'+esc(JSON.stringify(fr.headers||{}, null, 2))+'</div>';
    }

    if (r.forwarded_response) {
      var frp = r.forwarded_response;
      html += '<div class=\"section-title\">Upstream Response</div>';
      html += '<div style=\"font-size:0.85rem;margin-bottom:8px;\">Status: <b style=\"color:'+(frp.status_code>=200&&frp.status_code<300?'var(--green)':'var(--red)')+'\">'+frp.status_code+'</b>';
      if (frp.stream) html += ' <span style=\"color:var(--purple);\">[stream]</span>';
      html += '</div>';
      html += '<div class=\"section-title\">Response Headers</div>';
      html += '<div class=\"raw-json\">'+esc(JSON.stringify(frp.headers||{}, null, 2))+'</div>';
      html += '<div class=\"section-title\">Response Body</div>';
      html += '<div class=\"raw-json\">'+esc(typeof frp.body==='string'?frp.body:JSON.stringify(frp.body||{}, null, 2))+'</div>';
    }

    if (r.forwarded_error) {
      html += '<div class=\"section-title\" style=\"color:var(--red);\">Forward Error</div>';
      html += '<div style=\"color:var(--red);font-size:0.85rem;\">'+esc(r.forwarded_error)+'</div>';
    }
    return html;
  }

  var buttonsHTML = '';
  if (r.status === 'pending') {
    buttonsHTML = '<div class=\"btn-row\">'
      +'<button class=\"btn btn-submit\" onclick=\"submitResponse(\''+r.id+'\')\">Submit Response</button>'
      +'<button class=\"btn btn-skip\" onclick=\"skipRequest(\''+r.id+'\')\">Skip (empty response)</button>'
      +'</div>';
  } else if (r.status !== 'forwarded') {
    buttonsHTML = '<div style=\"margin-top:12px;color:#666;font-size:0.85rem;\">Request already '+esc(r.status)+' &mdash; no further action needed.</div>';
  }

  var detailHTML = '<div class=\"section-title\">Request Detail</div>'
    +'<div class=\"raw-json\">'+esc(JSON.stringify(r.raw_request_body||{
      model: r.model, messages: r.messages, temperature: r.temperature,
      max_tokens: r.max_tokens, stream: r.stream, tools: r.tools,
      response_format: r.response_format, extra_body: r.extra_body,
    }, null, 2))+'</div>';

  if (r.status === 'forwarded') {
    detailHTML += buildForwardedHTML(r);
    detailHTML += buttonsHTML;
  } else {
    detailHTML += '<div class=\"section-title\">Response Text</div>'
      +'<textarea class=\"detail-response-text\" id=\"respText-'+r.id+'\" placeholder=\"Type assistant text response here...\">'+esc(r.response_content||'')+'</textarea>'
      +toolsHTML
      +buttonsHTML;
  }

  return detailHTML;
}

// ---- Expand / collapse ----
async function toggleRequest(id) {
  if (expandedId === id) {
    // Collapse — keep detail HTML so re-expand preserves form state
    document.getElementById('card-'+id).classList.remove('expanded');
    expandedId = null;
    return;
  }
  // Collapse previous (keep its detail HTML)
  if (expandedId) {
    var prevCard = document.getElementById('card-'+expandedId);
    if (prevCard) prevCard.classList.remove('expanded');
  }
  // Expand this one
  expandedId = id;
  document.getElementById('card-'+id).classList.add('expanded');
  var detailEl = document.getElementById('detail-'+id);
  // Only rebuild if empty; otherwise user was already editing
  if (!detailEl.innerHTML.trim()) {
    var r = requestCache[id];
    if (!r) {
      r = await fetchJSON('/api/requests/' + id);
      requestCache[id] = r;
    }
    detailEl.innerHTML = buildDetailHTML(r);
    refreshPreview(id);
  } else {
    // Detail already built — just refresh preview
    refreshPreview(id);
  }
}

// ---- Tool enable toggle ----
function onToolToggle(reqId) {
  refreshPreview(reqId);
}

// Debounce preview refresh on input
document.addEventListener('input', function(e) {
  var detailEl = e.target.closest('.req-card-detail');
  if (detailEl) {
    var reqId = detailEl.id.replace('detail-','');
    clearTimeout(window._previewTimer);
    window._previewTimer = setTimeout(function(){ refreshPreview(reqId); }, 300);
  }
});

// ---- Collect & preview tool_calls ----
function collectToolCalls(reqId) {
  var r = requestCache[reqId];
  if (!r || !r.tools || r.tools.length === 0) return null;
  var result = [];
  r.tools.forEach(function(tool, idx) {
    var cb = document.getElementById('enable-'+reqId+'-'+idx);
    if (!cb || !cb.checked) return;
    var fn = tool.function || {};
    var name = fn.name || 'unknown_tool_'+idx;
    var callIdEl = document.getElementById('callId-'+reqId+'-'+idx);
    var callId = callIdEl ? callIdEl.value : genCallId();
    var args = {};
    var detailEl = document.getElementById('detail-'+reqId);
    if (detailEl) {
      var inputs = detailEl.querySelectorAll('[data-field]');
      inputs.forEach(function(inp) {
        var mod = inp.closest('.tool-module');
        if (!mod) return;
        if (mod.id !== 'toolMod-'+reqId+'-'+idx) return;
        var field = inp.dataset.field;
        var val = inp.value;
        if (val === '' || val === null || val === undefined) return;
        var dtype = inp.dataset.type;
        if (dtype === 'number' || dtype === 'integer') {
          val = dtype === 'integer' ? parseInt(val,10) : parseFloat(val);
          if (isNaN(val)) return;
        } else if (dtype === 'boolean') {
          val = val === 'true';
        } else if (dtype === 'array' || dtype === 'object') {
          try { val = JSON.parse(val); } catch(e) {}
        }
        args[field] = val;
      });
    }
    result.push({
      id: callId,
      type: 'function',
      function: { name: name, arguments: JSON.stringify(args) }
    });
  });
  return result.length > 0 ? result : null;
}

function refreshPreview(reqId) {
  var previewDiv = document.getElementById('preview-'+reqId);
  var previewContent = document.getElementById('previewContent-'+reqId);
  if (!previewDiv || !previewContent) return;
  var tc = collectToolCalls(reqId);
  if (tc && tc.length > 0) {
    previewDiv.style.display = 'block';
    previewContent.textContent = JSON.stringify(tc, null, 2);
  } else {
    previewDiv.style.display = 'none';
  }
}

// ---- Submit / Skip ----
async function submitResponse(reqId) {
  var content = document.getElementById('respText-'+reqId).value;
  var tool_calls = collectToolCalls(reqId);
  await fetch('/api/requests/' + reqId + '/respond', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content: content, tool_calls: tool_calls}),
  });
}

async function skipRequest(reqId) {
  await fetch('/api/requests/' + reqId + '/skip', {method: 'POST'});
}

connectWs();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# MockLLMServer
# ---------------------------------------------------------------------------

class MockLLMServer:
    """Local LLM mock server with OpenAI and Anthropic endpoints + web dashboard."""

    def __init__(self, port: int = 9999, host: str = "0.0.0.0",
                 scripts_path: str = None, forward_config_path: str = None):
        self.port = port
        self.host = host
        self._pending: Dict[str, PendingRequest] = {}
        self._lock = threading.Lock()
        self._scripts = _load_scripts(scripts_path)
        self._forward_config = _load_forward_config(forward_config_path)
        self._http_client = httpx.AsyncClient(timeout=30.0) if self._forward_config else None
        self._server_thread: Optional[threading.Thread] = None
        self._ws_clients: set = set()
        self._app = self._create_app()

    # ---- public API ----

    def start(self):
        """Start the server (blocking)."""
        print(f"\n  Mock LLM Server running at http://{self.host}:{self.port}")
        print(f"  Web dashboard:      http://localhost:{self.port}")
        print(f"  OpenAI endpoint:    http://localhost:{self.port}/v1/chat/completions")
        print(f"  Anthropic endpoint: http://localhost:{self.port}/v1/messages")
        print(f"  Set agent.yaml:     base_url: http://localhost:{self.port}/v1")
        if self._scripts:
            print(f"  Scripts loaded: {len(self._scripts)} patterns")
        if self._forward_config:
            print(f"  Forward rules:  {len(self._forward_config)} configured")
        print()
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")

    def start_in_thread(self) -> threading.Thread:
        """Start the server in a background thread. Returns the thread."""
        def _run():
            uvicorn.run(self._app, host=self.host, port=self.port, log_level="info")
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._server_thread = t
        time.sleep(0.5)  # brief wait for startup
        return t

    def stop(self):
        """Signal shutdown (uvicorn runs in daemon thread, will exit with process)."""
        pass

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._pending.values() if r.status == "pending")

    def get_pending_requests(self) -> List[PendingRequest]:
        with self._lock:
            return [r for r in self._pending.values() if r.status == "pending"]

    def _serialize(self, req: PendingRequest) -> dict:
        return {
            "id": req.id, "model": req.model, "messages": req.messages,
            "tools": req.tools, "temperature": req.temperature,
            "max_tokens": req.max_tokens, "stream": req.stream,
            "status": req.status, "created_at": req.created_at,
            "response_content": req.response_content,
            "response_tool_calls": req.response_tool_calls,
            "response_format": req.response_format,
            "extra_body": req.extra_body,
            "api_format": req.api_format,
            "system_prompt": req.system_prompt,
            "raw_request_body": req.raw_request_body,
            "raw_request_headers": req.raw_request_headers,
            "forwarded_request": req.forwarded_request,
            "forwarded_response": req.forwarded_response,
            "forwarded_rule": req.forwarded_rule,
            "forwarded_error": req.forwarded_error,
        }

    async def _broadcast(self, msg: dict):
        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    def _stats(self):
        with self._lock:
            items = list(self._pending.values())
        return {
            "pending": sum(1 for r in items if r.status == "pending"),
            "completed": sum(1 for r in items if r.status != "pending"),
        }

    # ---- internals ----

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="LLM Mock Server")

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            request_headers = {k.lower(): v for k, v in request.headers.items()}

            req_id = uuid.uuid4().hex[:16]
            model = body.get("model", "unknown")
            messages = body.get("messages", [])
            tools = body.get("tools")
            stream = body.get("stream", False)
            extra = body.get("extra_body")

            # Summarize last user message for logging
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = (m.get("content") or "")[:120]
                    break

            logger.info(
                "REQ %s | model=%s stream=%s tools=%d msgs=%d extra=%s | user: %s",
                req_id, model, stream,
                len(tools) if tools else 0,
                len(messages),
                json.dumps(extra, ensure_ascii=False) if extra else "{}",
                last_user or "(none)",
            )
            logger.debug("REQ %s BODY %s", req_id, json.dumps(body, ensure_ascii=False)[:2000])

            req = PendingRequest(
                id=req_id,
                model=model,
                messages=messages,
                tools=tools,
                temperature=body.get("temperature"),
                max_tokens=body.get("max_tokens"),
                stream=stream,
                response_format=body.get("response_format"),
                extra_body=extra,
                raw_request_body=body,
                raw_request_headers=request_headers,
            )

            # Check scripted responses
            script_match = _match_script(req, self._scripts)
            if script_match:
                req.response_content = script_match.get("content", "")
                req.response_tool_calls = script_match.get("tool_calls")
                req.status = "completed"
                req.event.set()
                with self._lock:
                    self._pending[req_id] = req
                await self._broadcast({
                    "type": "request_updated",
                    "request": self._serialize(req),
                    "pending": self._stats()["pending"],
                    "completed": self._stats()["completed"],
                })
                logger.info(
                    "REQ %s -> SCRIPT_MATCH response=%d chars, tool_calls=%d",
                    req_id,
                    len(req.response_content or ""),
                    len(req.response_tool_calls) if req.response_tool_calls else 0,
                )
                if req.stream:
                    return _stream_response(req)
                return JSONResponse(_build_chat_completion(req))

            # Forwarding check
            forward_rule = _match_forward_rule(req, self._forward_config)
            if forward_rule:
                req.forwarded_rule = forward_rule
                logger.info(
                    "REQ %s -> FORWARD_MATCH target=%s timeout=%d",
                    req_id, forward_rule.get("target_url"),
                    forward_rule.get("timeout", 30),
                )

                if req.stream:
                    with self._lock:
                        self._pending[req_id] = req
                    await self._broadcast({
                        "type": "new_request",
                        "request": self._serialize(req),
                        "pending": self._stats()["pending"],
                        "completed": self._stats()["completed"],
                    })
                    req._broadcast_callback = self._broadcast
                    req._serialize_callback = self._serialize
                    req._stats_callback = self._stats
                    return await _proxy_stream_response(req, forward_rule, self._http_client, request_headers)

                upstream_resp = await _forward_request(req, forward_rule, self._http_client, request_headers)
                if upstream_resp:
                    req.status = "forwarded"
                    req.event.set()
                    with self._lock:
                        self._pending[req_id] = req
                    await self._broadcast({
                        "type": "request_updated",
                        "request": self._serialize(req),
                        "pending": self._stats()["pending"],
                        "completed": self._stats()["completed"],
                    })
                    logger.info(
                        "REQ %s -> FORWARDED status=%d",
                        req_id, upstream_resp["status_code"],
                    )
                    return JSONResponse(upstream_resp["body"])

                logger.warning("REQ %s -> FORWARD_FAILED, falling through to queue", req_id)

            # Queue and wait
            with self._lock:
                self._pending[req_id] = req

            await self._broadcast({
                "type": "new_request",
                "request": self._serialize(req),
                "pending": self._stats()["pending"],
                "completed": self._stats()["completed"],
            })

            logger.info(
                "REQ %s -> QUEUED (pending=%d, total=%d). Open http://localhost:%d to respond.",
                req_id, self.pending_count(), len(self._pending), self.port,
            )

            # Wait for operator response (event.set() from /api/requests/{id}/respond)
            # Use run_in_executor — threading.Event.wait() blocks the asyncio event loop otherwise
            loop = asyncio.get_event_loop()
            signalled = await loop.run_in_executor(None, req.event.wait, 3600)
            if not signalled:
                req.status = "skipped"
                req.response_content = "[timeout — no response from operator]"
                logger.warning("REQ %s -> TIMEOUT (3600s)", req_id)
            else:
                logger.info(
                    "REQ %s -> %s response=%d chars, tool_calls=%d",
                    req_id,
                    req.status.upper(),
                    len(req.response_content or ""),
                    len(req.response_tool_calls) if req.response_tool_calls else 0,
                )

            if req.stream:
                return _stream_response(req)
            return JSONResponse(_build_chat_completion(req))

        @app.post("/v1/messages")
        async def anthropic_messages(request: Request):
            body = await request.json()
            request_headers = {k.lower(): v for k, v in request.headers.items()}

            req_id = uuid.uuid4().hex[:16]
            model = body.get("model", "claude-sonnet-4-6")
            messages = body.get("messages", [])
            system = body.get("system", "")
            anthropic_tools = body.get("tools")
            stream = body.get("stream", False)

            # Convert Anthropic tools → OpenAI format for dashboard compatibility
            openai_tools = None
            if anthropic_tools:
                openai_tools = []
                for t in anthropic_tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name", "unknown"),
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                        }
                    })

            # Summarize last user message for logging
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = _extract_text_content(m.get("content", ""))
                    break

            logger.info(
                "REQ %s | model=%s stream=%s tools=%d msgs=%d format=anthropic | user: %s",
                req_id, model, stream,
                len(anthropic_tools) if anthropic_tools else 0,
                len(messages),
                last_user or "(none)",
            )
            logger.debug("REQ %s BODY %s", req_id, json.dumps(body, ensure_ascii=False)[:2000])

            req = PendingRequest(
                id=req_id,
                model=model,
                messages=messages,
                tools=openai_tools,
                temperature=body.get("temperature"),
                max_tokens=body.get("max_tokens"),
                stream=stream,
                api_format="anthropic",
                system_prompt=system if isinstance(system, str) else (json.dumps(system, ensure_ascii=False) if system else None),
                raw_request_body=body,
                raw_request_headers=request_headers,
            )

            # Check scripted responses
            script_match = _match_script(req, self._scripts)
            if script_match:
                req.response_content = script_match.get("content", "")
                req.response_tool_calls = script_match.get("tool_calls")
                req.status = "completed"
                req.event.set()
                with self._lock:
                    self._pending[req_id] = req
                await self._broadcast({
                    "type": "request_updated",
                    "request": self._serialize(req),
                    "pending": self._stats()["pending"],
                    "completed": self._stats()["completed"],
                })
                logger.info(
                    "REQ %s -> SCRIPT_MATCH response=%d chars, tool_calls=%d",
                    req_id,
                    len(req.response_content or ""),
                    len(req.response_tool_calls) if req.response_tool_calls else 0,
                )
                if req.stream:
                    return _anthropic_stream_response(req)
                return JSONResponse(_build_anthropic_response(req))

            # Forwarding check
            forward_rule = _match_forward_rule(req, self._forward_config)
            if forward_rule:
                req.forwarded_rule = forward_rule
                logger.info(
                    "REQ %s -> FORWARD_MATCH target=%s timeout=%d",
                    req_id, forward_rule.get("target_url"),
                    forward_rule.get("timeout", 30),
                )

                if req.stream:
                    with self._lock:
                        self._pending[req_id] = req
                    await self._broadcast({
                        "type": "new_request",
                        "request": self._serialize(req),
                        "pending": self._stats()["pending"],
                        "completed": self._stats()["completed"],
                    })
                    req._broadcast_callback = self._broadcast
                    req._serialize_callback = self._serialize
                    req._stats_callback = self._stats
                    return await _proxy_stream_response(req, forward_rule, self._http_client, request_headers)

                upstream_resp = await _forward_request(req, forward_rule, self._http_client, request_headers)
                if upstream_resp:
                    req.status = "forwarded"
                    req.event.set()
                    with self._lock:
                        self._pending[req_id] = req
                    await self._broadcast({
                        "type": "request_updated",
                        "request": self._serialize(req),
                        "pending": self._stats()["pending"],
                        "completed": self._stats()["completed"],
                    })
                    logger.info(
                        "REQ %s -> FORWARDED status=%d",
                        req_id, upstream_resp["status_code"],
                    )
                    return JSONResponse(upstream_resp["body"])

                logger.warning("REQ %s -> FORWARD_FAILED, falling through to queue", req_id)

            # Queue and wait
            with self._lock:
                self._pending[req_id] = req

            await self._broadcast({
                "type": "new_request",
                "request": self._serialize(req),
                "pending": self._stats()["pending"],
                "completed": self._stats()["completed"],
            })

            logger.info(
                "REQ %s -> QUEUED (pending=%d, total=%d). Open http://localhost:%d to respond.",
                req_id, self.pending_count(), len(self._pending), self.port,
            )

            loop = asyncio.get_event_loop()
            signalled = await loop.run_in_executor(None, req.event.wait, 3600)
            if not signalled:
                req.status = "skipped"
                req.response_content = "[timeout — no response from operator]"
                logger.warning("REQ %s -> TIMEOUT (3600s)", req_id)
            else:
                logger.info(
                    "REQ %s -> %s response=%d chars, tool_calls=%d",
                    req_id,
                    req.status.upper(),
                    len(req.response_content or ""),
                    len(req.response_tool_calls) if req.response_tool_calls else 0,
                )

            if req.stream:
                return _anthropic_stream_response(req)
            return JSONResponse(_build_anthropic_response(req))

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return DASHBOARD_HTML

        @app.get("/api/requests")
        async def list_requests():
            with self._lock:
                items = list(self._pending.values())
            items.sort(key=lambda r: r.created_at, reverse=True)
            return {
                "requests": [self._serialize(r) for r in items],
                "pending": sum(1 for r in items if r.status == "pending"),
                "completed": sum(1 for r in items if r.status != "pending"),
            }

        @app.get("/api/requests/{req_id}")
        async def get_request(req_id: str):
            with self._lock:
                req = self._pending.get(req_id)
            if not req:
                raise HTTPException(404, "Request not found")
            return self._serialize(req)

        @app.post("/api/requests/{req_id}/respond")
        async def respond(req_id: str, body: dict):
            with self._lock:
                req = self._pending.get(req_id)
            if not req:
                logger.warning("RESPOND %s -> 404 not found", req_id)
                raise HTTPException(404, "Request not found")
            if req.status != "pending":
                logger.warning("RESPOND %s -> 409 already handled (status=%s)", req_id, req.status)
                raise HTTPException(409, "Request already handled")
            req.response_content = body.get("content", "")
            req.response_tool_calls = body.get("tool_calls")
            req.status = "completed"
            req.event.set()
            logger.info(
                "RESPOND %s -> COMPLETED content=%d chars, tool_calls=%d",
                req_id,
                len(req.response_content or ""),
                len(req.response_tool_calls) if req.response_tool_calls else 0,
            )
            await self._broadcast({
                "type": "request_updated",
                "request": self._serialize(req),
                "pending": self._stats()["pending"],
                "completed": self._stats()["completed"],
            })
            return {"status": "ok"}

        @app.post("/api/requests/{req_id}/skip")
        async def skip(req_id: str):
            with self._lock:
                req = self._pending.get(req_id)
            if not req:
                logger.warning("SKIP %s -> 404 not found", req_id)
                raise HTTPException(404, "Request not found")
            if req.status != "pending":
                logger.warning("SKIP %s -> 409 already handled (status=%s)", req_id, req.status)
                raise HTTPException(409, "Request already handled")
            req.response_content = ""
            req.status = "skipped"
            req.event.set()
            logger.info("SKIP %s -> SKIPPED", req_id)
            await self._broadcast({
                "type": "request_updated",
                "request": self._serialize(req),
                "pending": self._stats()["pending"],
                "completed": self._stats()["completed"],
            })
            return {"status": "ok"}

        @app.get("/api/stats")
        async def stats():
            with self._lock:
                items = list(self._pending.values())
            return {
                "pending": sum(1 for r in items if r.status == "pending"),
                "completed": sum(1 for r in items if r.status == "completed"),
                "skipped": sum(1 for r in items if r.status == "skipped"),
                "total": len(items),
            }

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.add(websocket)
            # Send full current state on connect
            with self._lock:
                items = list(self._pending.values())
            items.sort(key=lambda r: r.created_at, reverse=True)
            st = self._stats()
            await websocket.send_json({
                "type": "init",
                "requests": [self._serialize(r) for r in items],
                "pending": st["pending"],
                "completed": st["completed"],
            })
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self._ws_clients.discard(websocket)

        return app


def _stream_response(req: PendingRequest):
    """Return a minimal SSE streaming response for callers that pass stream=True."""
    completion = _build_chat_completion(req)
    chunk = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [{
            "index": 0,
            "delta": completion["choices"][0]["message"],
            "finish_reason": completion["choices"][0]["finish_reason"],
        }],
    }

    async def event_stream():
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
