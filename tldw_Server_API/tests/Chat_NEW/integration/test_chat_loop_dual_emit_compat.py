"""Integration tests for dual-emission stream compatibility mode."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _mock_provider_stream():
    yield "Hello"
    yield " world"
    yield "!"


@pytest.mark.integration
def test_dual_emit_preserves_legacy_and_loop_events(
    test_client,
    auth_headers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STREAMS_UNIFIED", "1")

    with patch(
        "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
        return_value=_mock_provider_stream(),
    ):
        with test_client.stream(
            "POST",
            "/api/v1/chat/completions",
            headers={**auth_headers, "X-TLDW-Loop-Compat": "1"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines() if line]

    payload = "\n".join(lines)
    assert "event: stream_start" in payload
    assert "event: stream_end" in payload
    assert "event: run_started" in payload
    assert "event: run_complete" in payload
