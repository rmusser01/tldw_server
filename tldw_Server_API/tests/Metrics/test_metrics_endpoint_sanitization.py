import pytest
from fastapi import HTTPException


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)


def _failing_metrics_registry():
    raise RuntimeError("metrics registry exploded at /private/metrics.sock")


@pytest.mark.asyncio
async def test_get_json_metrics_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import metrics

    logger_stub = _LoggerStub()
    monkeypatch.setattr(metrics, "get_metrics_registry", _failing_metrics_registry)
    monkeypatch.setattr(metrics, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await metrics.get_json_metrics()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve metrics"
    assert logger_stub.errors == ["Error getting metrics"]
    assert "metrics registry exploded" not in str(logger_stub.errors)
    assert "/private/metrics.sock" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_metrics_health_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import metrics

    def _failing_get_chat_metrics():
        raise RuntimeError("chat metrics exploded at /private/chat-metrics.sock")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(metrics, "get_chat_metrics", _failing_get_chat_metrics)
    monkeypatch.setattr(metrics, "logger", logger_stub)

    response = await metrics.health_check_with_metrics()

    assert response["status"] == "unhealthy"
    assert response["message"] == "Metrics Health check failed: ERROR - SEE LOGS"
    assert logger_stub.errors == ["Metrics Health check failed"]
    assert "chat metrics exploded" not in str(logger_stub.errors)
    assert "/private/chat-metrics.sock" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_get_chat_metrics_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import metrics

    def _failing_get_chat_metrics():
        raise RuntimeError("chat metrics endpoint exploded at /private/chat-metrics.sock")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(metrics, "get_chat_metrics", _failing_get_chat_metrics)
    monkeypatch.setattr(metrics, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await metrics.get_chat_metrics_endpoint()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve chat metrics"
    assert logger_stub.errors == ["Error getting chat metrics"]
    assert "chat metrics endpoint exploded" not in str(logger_stub.errors)
    assert "/private/chat-metrics.sock" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_reset_metrics_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import metrics

    logger_stub = _LoggerStub()
    monkeypatch.setattr(metrics, "get_metrics_registry", _failing_metrics_registry)
    monkeypatch.setattr(metrics, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await metrics.reset_metrics()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to reset metrics"
    assert logger_stub.errors == ["Error resetting metrics"]
    assert "metrics registry exploded" not in str(logger_stub.errors)
    assert "/private/metrics.sock" not in str(logger_stub.errors)
