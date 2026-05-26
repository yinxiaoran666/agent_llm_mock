"""agent-llm-mock — Local OpenAI-compatible LLM mock server.

Usage:
    CLI:        agent-llm-mock --port 9999
    Module:     python -m agent_llm_mock --port 9999
    Python API: from agent_llm_mock import serve; serve(port=9999)
"""

import argparse
from .server import MockLLMServer


def serve(port: int = 9999, scripts_path: str = None) -> MockLLMServer:
    """Start the mock LLM server (blocking)."""
    server = MockLLMServer(port=port, scripts_path=scripts_path)
    server.start()
    return server


def main():
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible LLM mock server")
    parser.add_argument("--port", type=int, default=9999, help="Server port (default: 9999)")
    parser.add_argument("--scripts", type=str, default=None, help="Path to pre-scripted responses JSON")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    args = parser.parse_args()
    server = MockLLMServer(port=args.port, scripts_path=args.scripts, host=args.host)
    server.start()


__all__ = ["MockLLMServer", "serve", "main"]
