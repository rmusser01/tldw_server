from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.RAG.rag_service import payload_exemplars


pytestmark = pytest.mark.unit


class _FailingSink:
    def open(self, *args, **kwargs):
        raise OSError("payload exemplar sink exploded /private/rag_payload.jsonl")


def test_maybe_record_exemplar_write_failure_log_is_sanitized(monkeypatch):
    fake_logger = MagicMock()

    monkeypatch.setattr(payload_exemplars, "logger", fake_logger)
    monkeypatch.setattr(payload_exemplars, "_sampling_random", lambda: 0.0)
    monkeypatch.setattr(payload_exemplars, "_safe_sink", lambda **_kwargs: _FailingSink())

    payload_exemplars.maybe_record_exemplar(
        query="test query",
        documents=[{"id": "doc-1", "content": "test content", "score": 0.8}],
        answer="test answer",
        reason="test_reason",
        user_id="user-1",
    )

    fake_logger.debug.assert_called_once_with("Failed to write exemplar")
    rendered = repr(fake_logger.debug.call_args)
    assert "payload exemplar sink exploded" not in rendered
    assert "/private/rag_payload.jsonl" not in rendered
