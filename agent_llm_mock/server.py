"""Mock LLM server — OpenAI-compatible endpoint with web dashboard for manual response control."""

import json
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn


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
    created_at: float = field(default_factory=time.time)
    status: str = "pending"          # pending | completed | skipped
    response_content: Optional[str] = None
    response_tool_calls: Optional[List[Dict]] = None
    event: threading.Event = field(default_factory=threading.Event)


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


def _match_script(req: PendingRequest, scripts: List[Dict]) -> Optional[Dict]:
    """Return the first matching script entry, or None."""
    combined = " ".join(
        m.get("content", "") or ""
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
          --green: #4caf50; --orange: #ff9800; --red: #f44336; --blue: #2196f3; }
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
  .req-card { background: var(--card); border-radius: 10px; padding: 14px; cursor: pointer; transition: border 0.15s; }
  .req-card:hover { border: 1px solid var(--accent); }
  .req-card.selected { border: 2px solid var(--blue); }
  .req-card .meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .req-card .model { font-size: 0.85rem; color: #888; }
  .req-card .time { font-size: 0.8rem; color: #666; }
  .req-card .preview { font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90vw; }
  .req-card .badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }
  .badge.pending { background: var(--orange); color: #000; }
  .badge.completed { background: var(--green); color: #000; }
  .badge.skipped { background: #666; color: #fff; }
  .detail-panel { background: var(--card); border-radius: 10px; padding: 16px; margin-top: 16px; display: none; }
  .detail-panel.active { display: block; }
  .section-title { font-size: 1rem; font-weight: bold; margin: 12px 0 6px; color: #aaa; }
  .raw-json { background: #0d0d1a; border-radius: 6px; padding: 12px; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.8rem; white-space: pre-wrap; word-break: break-all; max-height: 400px; overflow-y: auto; }
  .tools-json { background: #0d0d1a; border-radius: 6px; padding: 12px; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.78rem; white-space: pre-wrap; max-height: 250px; overflow-y: auto; }
  .response-area { margin-top: 16px; display: flex; flex-direction: column; gap: 10px; }
  .response-area textarea { width: 100%; padding: 12px; border-radius: 6px; border: 1px solid #444; background: #0d0d1a; color: #e0e0e0; font-family: inherit; font-size: 0.9rem; resize: vertical; min-height: 80px; }
  .response-area textarea.tools-input { min-height: 100px; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.8rem; }
  .btn-row { display: flex; gap: 8px; }
  .btn { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn-submit { background: var(--green); color: #fff; }
  .btn-skip { background: var(--red); color: #fff; }
  .btn-tools-toggle { background: transparent; border: 1px solid #555; color: #aaa; font-size: 0.8rem; padding: 4px 12px; }
  .tools-section { display: none; margin-top: 8px; }
  .tools-section.visible { display: flex; flex-direction: column; gap: 6px; }
  .tools-hint { font-size: 0.75rem; color: #888; }
  .empty-state { text-align: center; padding: 40px; color: #666; }
  .empty-state .icon { font-size: 2rem; }
  @media (max-width: 768px) {
    body { padding: 10px; }
    .req-card .preview { max-width: 70vw; }
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
    <p style="font-size:0.8rem;margin-top:4px;">Point your agent's base_url to http://HOST:PORT/v1</p>
  </div>
</div>

<div class="detail-panel" id="detailPanel">
  <div class="section-title">Request Detail</div>
  <div class="raw-json" id="detailJson"></div>
  <div class="section-title">Tools</div>
  <div class="tools-json" id="toolsJson">(none)</div>
  <div class="response-area">
    <label for="responseText" style="font-weight:600;">Response (plain text)</label>
    <textarea id="responseText" placeholder="Type assistant response here..."></textarea>
    <button class="btn-tools-toggle" onclick="toggleTools()">+ Tool Calls (advanced)</button>
    <div class="tools-section" id="toolsSection">
      <label for="toolsInput" style="font-weight:600;">Tool Calls (JSON array)</label>
      <textarea class="tools-input" id="toolsInput" placeholder='[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/x\"}"}}]'></textarea>
      <span class="tools-hint">Leave empty if no tool calls. Paste valid JSON array matching OpenAI tool_calls format.</span>
    </div>
    <div class="btn-row">
      <button class="btn btn-submit" onclick="submitResponse()">Submit Response</button>
      <button class="btn btn-skip" onclick="skipRequest()">Skip (empty response)</button>
    </div>
  </div>
</div>

<script>
let selectedId = null;
let currentPort = location.port || '9999';

function toggleTools() {
  document.getElementById('toolsSection').classList.toggle('visible');
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

async function poll() {
  try {
    const data = await fetchJSON('/api/requests');
    renderList(data.requests);
    document.getElementById('pendingCount').textContent = data.pending;
    document.getElementById('completedCount').textContent = data.completed;
  } catch(e) { console.error(e); }
}

function renderList(requests) {
  const el = document.getElementById('reqList');
  if (!requests || requests.length === 0) {
    el.innerHTML = '<div class="empty-state"><div class="icon">&#128179;</div><p>Waiting for requests...</p></div>';
    return;
  }
  el.innerHTML = requests.map(r => {
    const lastUser = (r.messages || []).filter(m => m.role === 'user').pop();
    const preview = lastUser ? (lastUser.content || '').substring(0, 100) : '(system prompt)';
    const toolCount = r.tools ? r.tools.length : 0;
    const badges = toolCount > 0 ? `<span style="color:#888;font-size:0.75rem;">+${toolCount} tools</span>` : '';
    const sel = r.id === selectedId ? ' selected' : '';
    return `<div class="req-card${sel}" onclick="selectRequest('${r.id}')">
      <div class="meta">
        <span class="model">${esc(r.model)} ${badges}</span>
        <span class="time">${new Date(r.created_at * 1000).toLocaleTimeString()}</span>
      </div>
      <div class="preview">${esc(preview)}</div>
      <span class="badge ${r.status}">${r.status}</span>
    </div>`;
  }).join('');
}

function esc(s) { return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function selectRequest(id) {
  selectedId = id;
  try {
    const r = await fetchJSON('/api/requests/' + id);
    document.getElementById('detailPanel').classList.add('active');
    document.getElementById('detailJson').textContent = JSON.stringify({
      model: r.model, messages: r.messages, temperature: r.temperature,
      max_tokens: r.max_tokens, stream: r.stream, tools: r.tools,
      response_format: r.response_format, extra_body: r.extra_body,
    }, null, 2);
    document.getElementById('toolsJson').textContent = r.tools ? JSON.stringify(r.tools, null, 2) : '(none)';
    document.getElementById('responseText').value = r.response_content || '';
    document.getElementById('toolsInput').value = r.response_tool_calls ? JSON.stringify(r.response_tool_calls, null, 2) : '';
  } catch(e) { console.error(e); }
  poll();
}

async function submitResponse() {
  if (!selectedId) return alert('Select a request first');
  const content = document.getElementById('responseText').value;
  let tool_calls = null;
  const tcText = document.getElementById('toolsInput').value.trim();
  if (tcText) {
    try { tool_calls = JSON.parse(tcText); } catch(e) { return alert('Invalid tool_calls JSON: ' + e.message); }
  }
  await fetch('/api/requests/' + selectedId + '/respond', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content: content, tool_calls: tool_calls}),
  });
  document.getElementById('detailPanel').classList.remove('active');
  selectedId = null;
  poll();
}

async function skipRequest() {
  if (!selectedId) return;
  await fetch('/api/requests/' + selectedId + '/skip', {method: 'POST'});
  document.getElementById('detailPanel').classList.remove('active');
  selectedId = null;
  poll();
}

setInterval(poll, 2000);
poll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# MockLLMServer
# ---------------------------------------------------------------------------

class MockLLMServer:
    """Local OpenAI-compatible LLM mock server with web dashboard."""

    def __init__(self, port: int = 9999, host: str = "0.0.0.0", scripts_path: str = None):
        self.port = port
        self.host = host
        self._pending: Dict[str, PendingRequest] = {}
        self._lock = threading.Lock()
        self._scripts = _load_scripts(scripts_path)
        self._server_thread: Optional[threading.Thread] = None
        self._app = self._create_app()

    # ---- public API ----

    def start(self):
        """Start the server (blocking)."""
        print(f"\n  Mock LLM Server running at http://{self.host}:{self.port}")
        print(f"  Web dashboard:  http://localhost:{self.port}")
        print(f"  API endpoint:   http://localhost:{self.port}/v1/chat/completions")
        print(f"  Set agent.yaml: base_url: http://localhost:{self.port}/v1")
        if self._scripts:
            print(f"  Scripts loaded: {len(self._scripts)} patterns")
        print()
        uvicorn.run(self._app, host=self.host, port=self.port, log_level="warning")

    def start_in_thread(self) -> threading.Thread:
        """Start the server in a background thread. Returns the thread."""
        def _run():
            uvicorn.run(self._app, host=self.host, port=self.port, log_level="warning")
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

    # ---- internals ----

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="LLM Mock Server")

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()

            req_id = uuid.uuid4().hex[:16]
            req = PendingRequest(
                id=req_id,
                model=body.get("model", "unknown"),
                messages=body.get("messages", []),
                tools=body.get("tools"),
                temperature=body.get("temperature"),
                max_tokens=body.get("max_tokens"),
                stream=body.get("stream", False),
                response_format=body.get("response_format"),
                extra_body=body.get("extra_body"),
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
                if req.stream:
                    return _stream_response(req)
                return JSONResponse(_build_chat_completion(req))

            # Queue and wait
            with self._lock:
                self._pending[req_id] = req

            # Wait for operator response (event.set() from /api/requests/{id}/respond)
            signalled = req.event.wait(timeout=3600)
            if not signalled:
                req.status = "skipped"
                req.response_content = "[timeout — no response from operator]"

            if req.stream:
                return _stream_response(req)
            return JSONResponse(_build_chat_completion(req))

        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return DASHBOARD_HTML

        @app.get("/api/requests")
        async def list_requests():
            with self._lock:
                items = list(self._pending.values())
            items.sort(key=lambda r: r.created_at, reverse=True)
            return {
                "requests": [
                    {
                        "id": r.id, "model": r.model, "messages": r.messages,
                        "tools": r.tools, "temperature": r.temperature,
                        "max_tokens": r.max_tokens, "stream": r.stream,
                        "status": r.status, "created_at": r.created_at,
                        "response_content": r.response_content,
                        "response_tool_calls": r.response_tool_calls,
                    }
                    for r in items
                ],
                "pending": sum(1 for r in items if r.status == "pending"),
                "completed": sum(1 for r in items if r.status != "pending"),
            }

        @app.get("/api/requests/{req_id}")
        async def get_request(req_id: str):
            with self._lock:
                req = self._pending.get(req_id)
            if not req:
                raise HTTPException(404, "Request not found")
            return {
                "id": req.id, "model": req.model, "messages": req.messages,
                "tools": req.tools, "temperature": req.temperature,
                "max_tokens": req.max_tokens, "stream": req.stream,
                "status": req.status, "created_at": req.created_at,
                "response_content": req.response_content,
                "response_tool_calls": req.response_tool_calls,
                "response_format": req.response_format,
                "extra_body": req.extra_body,
            }

        @app.post("/api/requests/{req_id}/respond")
        async def respond(req_id: str, body: dict):
            with self._lock:
                req = self._pending.get(req_id)
            if not req:
                raise HTTPException(404, "Request not found")
            if req.status != "pending":
                raise HTTPException(409, "Request already handled")
            req.response_content = body.get("content", "")
            req.response_tool_calls = body.get("tool_calls")
            req.status = "completed"
            req.event.set()
            return {"status": "ok"}

        @app.post("/api/requests/{req_id}/skip")
        async def skip(req_id: str):
            with self._lock:
                req = self._pending.get(req_id)
            if not req:
                raise HTTPException(404, "Request not found")
            if req.status != "pending":
                raise HTTPException(409, "Request already handled")
            req.response_content = ""
            req.status = "skipped"
            req.event.set()
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
