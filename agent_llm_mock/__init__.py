"""agent-llm-mock — Local LLM mock server (OpenAI + Anthropic endpoints).

Usage:
    CLI:        agent-llm-mock --port 9999
    Module:     python -m agent_llm_mock --port 9999
    Python API: from agent_llm_mock import serve; serve(port=9999)
"""

import argparse
from .server import MockLLMServer


def serve(port: int = 9999, scripts_path: str = None, host: str = "0.0.0.0",
          forward_config_path: str = None) -> MockLLMServer:
    """Start the mock LLM server (blocking)."""
    server = MockLLMServer(port=port, host=host, scripts_path=scripts_path,
                           forward_config_path=forward_config_path)
    server.start()
    return server


def main():
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible LLM mock server")
    parser.add_argument("--port", type=int, default=9999, help="Server port (default: 9999)")
    parser.add_argument("--scripts", type=str, default=None, help="Path to pre-scripted responses JSON")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--forward-config", type=str, default=None,
                        help="Path to JSON file with forwarding rules")
    args = parser.parse_args()
    server = MockLLMServer(port=args.port, scripts_path=args.scripts, host=args.host,
                           forward_config_path=args.forward_config)
    server.start()


__all__ = ["MockLLMServer", "serve", "main"]
