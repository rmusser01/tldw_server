from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


pytestmark = pytest.mark.unit

_TEST_USER_ID = 424242
_SENSITIVE_MARKERS = (
    str(_TEST_USER_ID),
    "watchlists backend exploded",
    "/private/watchlists.db",
)


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


class _FailingTelemetryDb:
    def record_onboarding_event(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("watchlists backend exploded /private/watchlists.db")

    def summarize_onboarding_events(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("watchlists backend exploded /private/watchlists.db")

    def record_ia_experiment_event(self, **_kwargs: Any) -> bool:
        raise RuntimeError("watchlists backend exploded /private/watchlists.db")

    def summarize_ia_experiment_events(self, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("watchlists backend exploded /private/watchlists.db")


def _test_user() -> User:
    return User(id=_TEST_USER_ID, username="watchlists-logs", email=None, is_active=True)


def _assert_sanitized_log(
    records: list[tuple[str, tuple[object, ...], dict[str, object]]],
    expected_message: str,
) -> None:
    assert records
    rendered = repr(records)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered
    assert records == [(expected_message, (), {})]


@pytest.mark.asyncio
async def test_onboarding_telemetry_ingest_failure_log_is_sanitized(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    logger_stub = _LoggerStub()
    monkeypatch.setattr(watchlists, "logger", logger_stub)
    monkeypatch.setattr(watchlists, "record_onboarding_ingest_result", lambda _status: None)

    response = await watchlists.record_watchlists_onboarding_telemetry(
        payload=watchlists.WatchlistOnboardingTelemetryIngestRequest(
            session_id="watchlists-session-0001",
            event_type="quick_setup_opened",
            event_at="2026-02-23T18:00:00Z",
        ),
        current_user=_test_user(),
        db=_FailingTelemetryDb(),
    )

    assert response.accepted is False
    assert response.code == "onboarding_telemetry_ingest_failed"
    _assert_sanitized_log(
        logger_stub.debugs,
        "watchlists onboarding telemetry ingest failed",
    )


@pytest.mark.asyncio
async def test_onboarding_telemetry_summary_failure_log_is_sanitized(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    logger_stub = _LoggerStub()
    summary_requests: list[tuple[str, str]] = []
    monkeypatch.setattr(watchlists, "logger", logger_stub)
    monkeypatch.setattr(
        watchlists,
        "record_summary_request",
        lambda name, status, _duration: summary_requests.append((name, status)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await watchlists.get_watchlists_onboarding_telemetry_summary(
            since="2026-02-23T18:00:00Z",
            until="2026-02-23T19:00:00Z",
            current_user=_test_user(),
            db=_FailingTelemetryDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "watchlists_onboarding_telemetry_summary_failed"
    assert summary_requests == [("onboarding_summary", "error")]
    _assert_sanitized_log(
        logger_stub.errors,
        "watchlists onboarding telemetry summary failed",
    )


@pytest.mark.asyncio
async def test_ia_telemetry_ingest_failure_log_is_sanitized(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    logger_stub = _LoggerStub()
    monkeypatch.setattr(watchlists, "logger", logger_stub)

    response = await watchlists.record_watchlists_ia_experiment_telemetry(
        payload=watchlists.WatchlistIaExperimentTelemetryIngestRequest(
            variant="experimental",
            session_id="watchlists-session-0002",
            current_tab="sources",
            transitions=0,
            visited_tabs=["sources"],
            first_seen_at="2026-02-23T18:00:00Z",
            last_seen_at="2026-02-23T18:00:00Z",
        ),
        current_user=_test_user(),
        db=_FailingTelemetryDb(),
    )

    assert response.accepted is False
    _assert_sanitized_log(
        logger_stub.debugs,
        "watchlists IA telemetry ingest failed",
    )


@pytest.mark.asyncio
async def test_rc_telemetry_summary_failure_log_is_sanitized(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    logger_stub = _LoggerStub()
    summary_requests: list[tuple[str, str]] = []
    monkeypatch.setattr(watchlists, "logger", logger_stub)
    monkeypatch.setattr(
        watchlists,
        "record_summary_request",
        lambda name, status, _duration: summary_requests.append((name, status)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await watchlists.get_watchlists_rc_telemetry_summary(
            since="2026-02-23T18:00:00Z",
            until="2026-02-23T19:00:00Z",
            current_user=_test_user(),
            db=_FailingTelemetryDb(),
            collections_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "watchlists_rc_telemetry_summary_failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert summary_requests == [("rc_summary", "error")]
    _assert_sanitized_log(
        logger_stub.errors,
        "watchlists RC telemetry summary failed",
    )


@pytest.mark.asyncio
async def test_ia_telemetry_summary_failure_log_is_sanitized(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    logger_stub = _LoggerStub()
    monkeypatch.setattr(watchlists, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await watchlists.get_watchlists_ia_experiment_telemetry_summary(
            since="2026-02-23T18:00:00Z",
            until="2026-02-23T19:00:00Z",
            current_user=_test_user(),
            db=_FailingTelemetryDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "watchlists_ia_telemetry_summary_failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    _assert_sanitized_log(
        logger_stub.errors,
        "watchlists IA telemetry summary failed",
    )
