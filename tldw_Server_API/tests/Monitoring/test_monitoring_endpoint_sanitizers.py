import pytest

from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_ep
from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import Watchlist


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.exceptions: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        del kwargs
        self.errors.append(message.format(*args) if args else message)

    def exception(self, message: str, *args, **kwargs) -> None:
        del kwargs
        self.exceptions.append(message.format(*args) if args else message)


@pytest.mark.asyncio
async def test_upsert_watchlist_sanitizes_unexpected_error_log(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingMonitoringService:
        def upsert_watchlist(self, _payload):
            raise RuntimeError("monitoring backend exploded /private/monitoring.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(monitoring_ep, "logger", logger_stub)
    monkeypatch.setattr(
        monitoring_ep,
        "get_topic_monitoring_service",
        lambda: _FailingMonitoringService(),
    )

    with pytest.raises(monitoring_ep.HTTPException) as exc_info:
        await monitoring_ep.upsert_watchlist(Watchlist(name="secret-watchlist", rules=[]))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to upsert watchlist"
    assert logger_stub.errors == ["Failed to upsert watchlist"]
    assert logger_stub.exceptions == []
    assert "secret-watchlist" not in str(logger_stub.errors)
    assert "monitoring backend exploded" not in str(logger_stub.errors)
    assert "/private/monitoring.db" not in str(logger_stub.errors)
