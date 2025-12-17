"""Helper functions for Hybrid LLM."""
from typing import Any

def convert_tool_to_ollama(tool: Any) -> dict:
    """Convert a Home Assistant Tool definition to Ollama JSON Schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters.schema if hasattr(tool.parameters, 'schema') else tool.parameters,
        }
    }
