import asyncio
import gc
import json
from typing import Any, Dict, List

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Insights import (
    LiveInsightSettings,
    LiveMeetingInsights,
)


class _DummyWebSocket:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.messages.append(payload)


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs) -> Dict[str, Any]:
        self.calls.append(kwargs)
        content = {
            "summary": ["Discussed roadmap and blockers."],
            "action_items": [{"description": "Follow up with Alex on blockers", "owner": "Alex"}],
            "decisions": ["Proceed with Q2 launch timeline"],
            "topics": ["Roadmap"],
        }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(content),
                    }
                }
            ]
        }


@pytest.mark.asyncio
async def test_live_insights_emits_summary():
    websocket = _DummyWebSocket()
    fake_llm = _FakeLLM()
    settings = LiveInsightSettings(
        enabled=True,
        provider="openai",
        model="test-model",
        summary_interval_seconds=0.0,
        context_window_segments=2,
        live_updates=True,
    )
    engine = LiveMeetingInsights(websocket, settings, chat_call=fake_llm)

    segment = {
        "text": "We walked through the roadmap and assigned follow ups to Alex.",
        "is_final": True,
        "segment_id": 1,
        "segment_start": 0.0,
        "segment_end": 15.0,
        "chunk_start": 0.0,
        "chunk_end": 15.0,
    }

    await engine.on_transcript(segment)
    await asyncio.sleep(0.05)
    await engine.close()

    assert websocket.messages, "Expected live insight message to be emitted"
    insight_messages = [m for m in websocket.messages if m.get("type") == "insight"]
    assert insight_messages, "No insight messages captured"
    latest = insight_messages[-1]
    assert latest["stage"] == "live"
    assert latest["summary"]
    assert latest["action_items"]


@pytest.mark.asyncio
async def test_live_insights_final_summary():
    websocket = _DummyWebSocket()
    fake_llm = _FakeLLM()
    settings = LiveInsightSettings(
        enabled=True,
        provider="openai",
        model="test-model",
        live_updates=False,
        final_summary=True,
    )
    engine = LiveMeetingInsights(websocket, settings, chat_call=fake_llm)

    segment = {
        "text": "Initial discussion on launch timing.",
        "is_final": True,
        "segment_id": 1,
        "segment_start": 0.0,
        "segment_end": 30.0,
        "chunk_start": 0.0,
        "chunk_end": 30.0,
    }
    await engine.on_transcript(segment)
    await engine.on_commit("Initial discussion on launch timing.")
    await engine.close()

    insight_messages = [m for m in websocket.messages if m.get("type") == "insight"]
    assert insight_messages, "Expected final insight message"
    final_msg = insight_messages[-1]
    assert final_msg["stage"] == "final"
    assert final_msg["summary"]


@pytest.mark.asyncio
async def test_live_insights_error_sanitizes_llm_exception():
    websocket = _DummyWebSocket()

    def _failing_llm(**_kwargs):
        raise RuntimeError("llm provider failed at /private/llm/config.json")

    settings = LiveInsightSettings(
        enabled=True,
        provider="openai",
        model="test-model",
        live_updates=True,
    )
    engine = LiveMeetingInsights(websocket, settings, chat_call=_failing_llm)

    await engine._generate_and_send(
        [
            {
                "text": "The team discussed operational blockers.",
                "is_final": True,
                "segment_id": 1,
            }
        ],
        stage="live",
    )

    error_messages = [message for message in websocket.messages if message.get("type") == "insight_error"]
    assert error_messages
    assert error_messages[-1]["message"] == "Live insights request failed"
    assert "llm provider failed" not in str(error_messages)
    assert "/private/llm/config.json" not in str(error_messages)


@pytest.mark.asyncio
async def test_live_insights_task_exceptions_retrieved():
    websocket = _DummyWebSocket()
    settings = LiveInsightSettings(
        enabled=True,
        provider="openai",
        model="test-model",
        live_updates=True,
        summary_interval_seconds=0.0,
    )
    engine = LiveMeetingInsights(websocket, settings, chat_call=_FakeLLM())

    loop = asyncio.get_running_loop()
    errors: List[Dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()

    def _handler(_loop, context):  # type: ignore[no-untyped-def]
        errors.append(context)

    loop.set_exception_handler(_handler)

    async def _boom():
        raise RuntimeError("boom")

    engine._schedule_task(_boom)
    await asyncio.sleep(0.05)
    gc.collect()
    await asyncio.sleep(0.05)
    await engine.close()

    if previous_handler:
        loop.set_exception_handler(previous_handler)
    else:
        loop.set_exception_handler(None)

    assert not any(
        "Task exception was never retrieved" in str(ctx.get("message", ""))
        for ctx in errors
    )
