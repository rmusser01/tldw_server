from __future__ import annotations

from loguru import logger

import pytest


pytestmark = pytest.mark.unit

_LEAK = "backend exploded /tmp/secret-token"


def _assert_safe_log(rendered: str) -> None:
    assert "backend exploded" not in rendered
    assert "/tmp/secret-token" not in rendered
    assert "exc_info" not in rendered


def test_sync_executor_shutdown_log_omits_raw_exception_metadata(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_orchestrator

    class _ExplodingExecutor:
        def shutdown(self, *args, **kwargs):
            raise RuntimeError(_LEAK)

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    monkeypatch.setattr(chat_orchestrator, "_SYNC_EXECUTOR", _ExplodingExecutor())
    try:
        chat_orchestrator._shutdown_sync_executor()
    finally:
        logger.remove(sink_id)
        monkeypatch.setattr(chat_orchestrator, "_SYNC_EXECUTOR", None)

    rendered = "\n".join(records)
    _assert_safe_log(rendered)


def test_token_count_fallback_log_omits_raw_exception(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_orchestrator

    class _ExplodingHistory:
        def __iter__(self):
            raise RuntimeError(_LEAK)

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    try:
        assert chat_orchestrator.approximate_token_count(_ExplodingHistory()) == 0
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    _assert_safe_log(rendered)
