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
        self._prune_buffers()
        self._buffers[run_id] = []

    def _prune_buffers(self):
        """Remove buffers older than 5 minutes to prevent memory leaks."""
        now = time.time()
        # Find stale IDs
        stale_ids = []
        for rid, events in self._buffers.items():
            if not events:
                continue
            # Check timestamp of first event
            first_ts = events[0].get("ts", 0) / 1000000
            if now - first_ts > 300: # 5 minutes
                stale_ids.append(rid)
        
        for rid in stale_ids:
            _LOGGER.debug(f"Pruning stale trace buffer: {rid}")
            del self._buffers[rid]

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

    async def dump(self, hass: Any, run_id: str, clear_buffer: bool = True):
        """Dump the buffered events to a file (async)."""
        if run_id not in self._buffers:
            return

        if clear_buffer:
            events = self._buffers.pop(run_id)
        else:
            events = list(self._buffers[run_id])
            
        if not events:
            return

        filename = f"trace_{run_id}.json"
        filepath = os.path.join(self._trace_dir, filename)
        
        await hass.async_add_executor_job(self._write_trace, filepath, events)

    def _write_trace(self, filepath: str, events: List[Dict[str, Any]]):
        """Write events to file (blocking)."""
        try:
            with open(filepath, "w") as f:
                json.dump(events, f, indent=2)
            _LOGGER.info("Performance trace saved to %s", filepath)
        except Exception as e:
            _LOGGER.error("Failed to write trace file %s: %s", filepath, e)
