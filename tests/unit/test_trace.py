from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.agents import AgentResult, HonestScriptedAgent
from app.runner import run_attempt
from app.trace import TraceRecorder


def test_sequence_numbers_are_gap_free_and_ordered():
    recorder = TraceRecorder()
    for i in range(5):
        recorder.record("lifecycle", f"event_{i}")
    events = recorder.events
    assert [e.sequence for e in events] == [1, 2, 3, 4, 5]
    assert [e.name for e in events] == [f"event_{i}" for i in range(5)]
    assert all(a.monotonic_ns <= b.monotonic_ns for a, b in zip(events, events[1:]))


def test_payload_is_copied_at_record_time():
    recorder = TraceRecorder()
    payload = {"items": [1]}
    recorder.record("tool_attempt", "get_order", payload)
    payload["items"].append(2)
    assert recorder.events[0].payload == {"items": [1]}


def test_recorded_events_are_immutable():
    recorder = TraceRecorder()
    event = recorder.record("lifecycle", "attempt_started")
    with pytest.raises(ValidationError):
        event.sequence = 99
    assert isinstance(recorder.events, tuple)


def test_timestamps_are_utc():
    event = TraceRecorder().record("lifecycle", "attempt_started")
    assert event.recorded_at.utcoffset() == timedelta(0)


def test_honest_attempt_trace_is_complete_and_ordered(refund_task_path):
    result = run_attempt(refund_task_path, HonestScriptedAgent())
    tools = ["get_order", "get_return_status", "request_refund_approval", "issue_refund", "update_ticket"]
    assert [(e.kind, e.name) for e in result.events] == [
        ("lifecycle", "attempt_started"),
        *[(kind, tool) for tool in tools for kind in ("tool_attempt", "tool_decision", "tool_effect")],
        ("agent_response", "final_response"),
        ("lifecycle", "attempt_completed"),
    ]
    assert [e.sequence for e in result.events] == list(range(1, len(result.events) + 1))


def test_repeated_attempts_are_isolated(refund_task_path):
    first = run_attempt(refund_task_path, HonestScriptedAgent())
    second = run_attempt(refund_task_path, HonestScriptedAgent())
    assert first.environment_id != second.environment_id
    assert list(first.final_snapshot["refunds"]) == ["RF-0001"]
    assert list(second.final_snapshot["refunds"]) == ["RF-0001"]


def test_unknown_tool_is_refused_and_traced(refund_task_path):
    class OverreachingAgent:
        name = "overreaching"

        def run(self, task, runtime):
            self.response = runtime.call_tool("delete_order", {"order_id": "ORD-1001"})
            return AgentResult(final_response="done", stop_reason="completed")

    agent = OverreachingAgent()
    result = run_attempt(refund_task_path, agent)
    assert agent.response["error"]["code"] == "unknown_tool"
    decision = next(e for e in result.events if e.kind == "tool_decision")
    assert decision.payload["reason_code"] == "UNKNOWN_TOOL"
    assert not any(e.kind == "tool_effect" for e in result.events)


def test_returned_events_cannot_rewrite_nested_history():
    recorder = TraceRecorder()
    event = recorder.record("lifecycle", "start", {"nested": {"items": [1]}})
    event.payload["nested"]["items"].append(2)
    recorder.events[0].payload["nested"]["items"].clear()
    assert recorder.events[0].payload == {"nested": {"items": [1]}}


def test_invalid_event_does_not_consume_sequence():
    recorder = TraceRecorder()
    recorder.record("lifecycle", "start")
    with pytest.raises(ValidationError):
        recorder.record("invalid_kind", "bad")
    recorder.record("lifecycle", "end")
    assert [event.sequence for event in recorder.events] == [1, 2]
