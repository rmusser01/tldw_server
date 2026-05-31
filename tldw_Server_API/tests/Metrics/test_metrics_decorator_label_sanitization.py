import pytest

from tldw_Server_API.app.core.Metrics import decorators


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


def _failing_label_extractor(*args, **kwargs):
    raise RuntimeError("metrics label exploded /private/metrics.db")


def test_count_calls_sync_label_extractor_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(decorators, "logger", logger_stub)

    @decorators.count_calls(
        metric_name="test_count_calls_sync_label_extractor_failure_log_is_sanitized_total",
        label_extractor=_failing_label_extractor,
    )
    def wrapped() -> str:
        return "ok"

    assert wrapped() == "ok"
    assert logger_stub.debugs == ["label_extractor failed (sync)"]
    assert "metrics label exploded" not in str(logger_stub.debugs)
    assert "/private/metrics.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_count_calls_async_label_extractor_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(decorators, "logger", logger_stub)

    @decorators.count_calls(
        metric_name="test_count_calls_async_label_extractor_failure_log_is_sanitized_total",
        label_extractor=_failing_label_extractor,
    )
    async def wrapped() -> str:
        return "ok"

    assert await wrapped() == "ok"
    assert logger_stub.debugs == ["label_extractor failed (async)"]
    assert "metrics label exploded" not in str(logger_stub.debugs)
    assert "/private/metrics.db" not in str(logger_stub.debugs)
