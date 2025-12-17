"""Performance Tracer for Hybrid LLM Integration."""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

class PerformanceTracer:
    """Buffers trace events and dumps them to JSON (Chrome Tracing format)."""

    def __init__(self, trace_dir: str):
        """Initialize the tracer."""
        self._trace_dir = trace_dir
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        
        # Ensure trace directory exists
        if not os.path.exists(trace_dir):
            try:
                os.makedirs(trace_dir, exist_ok=True)
            except Exception as e:
                _LOGGER.error("Failed to create trace directory %s: %s", trace_dir, e)

    def start_trace(self, run_id: str):
        """Initialize a trace buffer for a specific run ID."""
        self._buffers[run_id] = []

    def trace_event(
        self,
        run_id: str,
        name: str,
        phase: str, # "B" (Begin), "E" (End), "i" (Instant)
        category: str = "hybrid_llm",
        args: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        """Record a trace event."""
        if run_id not in self._buffers:
            return

        if timestamp is None:
            timestamp = time.time()

        event = {
            "name": name,
            "cat": category,
            "ph": phase,
            "ts": timestamp * 1000000, # Microseconds
            "pid": 1, # Dummy PID
            "tid": 1, # Dummy TID (could map to thread ID if needed)
            "args": args or {}
        }
        self._buffers[run_id].append(event)

    def dump(self, run_id: str):
        """Dump the buffered events to a file."""
        if run_id not in self._buffers:
            return

        events = self._buffers.pop(run_id)
        if not events:
            return

        filename = f"trace_{run_id}.json"
        filepath = os.path.join(self._trace_dir, filename)
        
        try:
            with open(filepath, "w") as f:
                json.dump(events, f, indent=2)
            _LOGGER.info("Performance trace saved to %s", filepath)
        except Exception as e:
            _LOGGER.error("Failed to write trace file %s: %s", filepath, e)
