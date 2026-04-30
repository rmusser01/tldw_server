import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint


def test_queue_estimate_sanitizes_base64_payload():


    payload = "data:image/png;base64," + ("a" * 400)
    raw = (
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\""
        + payload
        + "\"}}]}]}"
    )

    raw_est = max(1, len(raw) // 4)
    sanitized = chat_endpoint._sanitize_json_for_rate_limit(raw)
    sanitized_est = max(1, len(sanitized) // 4)
    helper_est = chat_endpoint._estimate_tokens_for_queue(raw)

    assert helper_est == sanitized_est
    assert sanitized_est < raw_est


def test_build_test_mode_chat_response_returns_mermaid_for_mind_map_prompt():
    content = chat_endpoint._build_test_mode_chat_response(
        [
            {
                "role": "system",
                "content": "You are a mind map generator. Return ONLY Mermaid mindmap syntax.",
            },
            {
                "role": "user",
                "content": "Sources:\nProgram Alpha briefing\nGovernance and evidence review.",
            },
        ]
    )

    assert content.startswith("mindmap")
    assert "Governance" in content
    assert "Evidence Review" in content


def test_build_test_mode_chat_response_returns_markdown_table_for_data_table_prompt():
    content = chat_endpoint._build_test_mode_chat_response(
        [
            {
                "role": "system",
                "content": (
                    "You are a data table generator. Return ONLY a markdown table "
                    "with pipe delimiters, a header row, and a separator row."
                ),
            },
            {
                "role": "user",
                "content": "Sources:\nProgram Alpha briefing\nProgram Beta briefing",
            },
        ]
    )

    assert content.startswith("| Topic |")
    assert "| --- | --- | --- |" in content
    assert "Program Alpha" in content


def test_build_test_mode_chat_response_supports_message_objects():
    content = chat_endpoint._build_test_mode_chat_response(
        [
            SimpleNamespace(
                role="system",
                content="You are a mind map generator. Return ONLY Mermaid mindmap syntax.",
            ),
            SimpleNamespace(
                role="user",
                content="Sources:\nProgram Alpha briefing\nGovernance and evidence review.",
            ),
        ]
    )

    assert content.startswith("mindmap")


def test_build_test_mode_chat_response_uses_explicit_system_message():
    content = chat_endpoint._build_test_mode_chat_response(
        [
            {
                "role": "user",
                "content": "Sources:\nProgram Alpha briefing\nGovernance and evidence review.",
            },
        ],
        system_message="You are a mind map generator. Return ONLY Mermaid mindmap syntax.",
    )

    assert content.startswith("mindmap")


def test_chat_error_audit_metadata_sanitizes_exception_message():
    exc = RuntimeError("provider failed at /private/chat/request.json")

    metadata = chat_endpoint._build_chat_error_audit_metadata(
        exc,
        provider="openai",
        model="gpt-4o-mini",
    )

    assert metadata == {
        "error_type": "RuntimeError",
        "error_message": "Chat completion failed",
        "provider": "openai",
        "model": "gpt-4o-mini",
    }


@pytest.mark.asyncio
async def test_schedule_audit_background_task_observes_exception(monkeypatch):
    captured: list[tuple[str, tuple]] = []

    class _DummyLogger:
        def debug(self, message, *args):
            captured.append((message, args))

    monkeypatch.setattr(chat_endpoint, "logger", _DummyLogger())

    async def _boom():
        raise RuntimeError("boom")

    task = chat_endpoint._schedule_audit_background_task(_boom(), task_name="chat.endpoint.audit")
    assert task is not None
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert any("chat.endpoint.audit" in str(args) for _, args in captured)


@pytest.mark.asyncio
async def test_schedule_audit_background_task_cancelled_is_silent(monkeypatch):
    captured: list[tuple[str, tuple]] = []

    class _DummyLogger:
        def debug(self, message, *args):
            captured.append((message, args))

    monkeypatch.setattr(chat_endpoint, "logger", _DummyLogger())

    gate = asyncio.Event()

    async def _slow():
        await gate.wait()

    task = chat_endpoint._schedule_audit_background_task(_slow(), task_name="chat.endpoint.cancel")
    assert task is not None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert not any("chat.endpoint.cancel" in str(args) and "failed" in msg for msg, args in captured)


@pytest.mark.asyncio
async def test_process_content_for_db_sync_large_image_path_handles_processor_rejection(monkeypatch):
    class _RejectingProcessor:
        async def process_image_url(self, _url: str, _max_size_bytes: int):
            return False, None, "text/plain", "Unsupported image MIME type: text/plain"

    monkeypatch.setattr(chat_endpoint, "get_image_processor", lambda: _RejectingProcessor())

    large_payload = "A" * 100001
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:text/plain;base64,{large_payload}"},
        }
    ]

    text_parts, images = await chat_endpoint._process_content_for_db_sync(content, "conv_test")

    assert images == []
    assert any(
        part.startswith("<Image failed: Unsupported image MIME type: text/plain>")
        for part in text_parts
    )
