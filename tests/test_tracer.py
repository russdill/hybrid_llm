"""Tests for PerformanceTracer."""
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from custom_components.hybrid_llm.tracer import PerformanceTracer

def test_tracer_buffered_dump(tmp_path):
    """Test standard trace buffering and dumping."""
    tracer = PerformanceTracer(str(tmp_path))
    run_id = "run_123"
    
    tracer.start_trace(run_id)
    tracer.trace_event(run_id, "Test Event", "B", "category1", timestamp=100.0)
    tracer.trace_event(run_id, "Test Event", "E", "category1", timestamp=101.0)
    
    tracer.dump(run_id)
    
    trace_file = tmp_path / f"trace_{run_id}.json"
    assert trace_file.exists()
    
    with open(trace_file, "r") as f:
        events = json.load(f)
    
    assert len(events) == 2
    assert events[0]["name"] == "Test Event"
    assert events[0]["ph"] == "B"
    assert events[0]["ts"] == 100000000.0 # Microseconds
    
    assert events[1]["ph"] == "E"

def test_tracer_ignore_unknown_run(tmp_path):
    """Test ignoring events for unknown runs."""
    tracer = PerformanceTracer(str(tmp_path))
    tracer.trace_event("unknown_id", "Event", "i")
    tracer.dump("unknown_id")
    
    assert not any(tmp_path.iterdir())
