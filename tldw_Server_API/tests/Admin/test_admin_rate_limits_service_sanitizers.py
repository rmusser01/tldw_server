from __future__ import annotations

import pytest

from tldw_Server_API.app.services import admin_rate_limits_service as service

pytestmark = pytest.mark.unit

_LEAK = "rate-limit backend leaked token at /tmp/rate-limit-secret-token"


class _UnusedDb:
    pass


def _capture_warning_logs() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    return messages, sink_id


def _assert_safe_log(rendered: str) -> None:
    assert "RuntimeError" in rendered
    assert "rate-limit backend leaked token" not in rendered
    assert "/tmp/rate-limit-secret-token" not in rendered
    assert "exc_info" not in rendered


@pytest.mark.asyncio
async def test_simulate_rate_limit_user_fallback_warning_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_fetch_user_rate_limits(*_args, **_kwargs):
        raise RuntimeError(_LEAK)

    async def empty_fetch_role_rate_limits(*_args, **_kwargs):
        return []

    monkeypatch.setattr(service, "fetch_user_rate_limits", fail_fetch_user_rate_limits)
    monkeypatch.setattr(service, "fetch_role_rate_limits", empty_fetch_role_rate_limits)

    messages, sink_id = _capture_warning_logs()
    try:
        result = await service.simulate_rate_limit(db=_UnusedDb(), user_id=42, endpoint="/api/v1/rag/search")
    finally:
        service.logger.remove(sink_id)

    assert result["user_limits"] == []
    assert result["role_limits"] == []
    _assert_safe_log("\n".join(messages))


@pytest.mark.asyncio
async def test_simulate_rate_limit_role_fallback_warning_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def empty_fetch_user_rate_limits(*_args, **_kwargs):
        return []

    async def fail_fetch_role_rate_limits(*_args, **_kwargs):
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(service, "fetch_user_rate_limits", empty_fetch_user_rate_limits)
    monkeypatch.setattr(service, "fetch_role_rate_limits", fail_fetch_role_rate_limits)

    messages, sink_id = _capture_warning_logs()
    try:
        result = await service.simulate_rate_limit(db=_UnusedDb(), user_id=43, endpoint="/api/v1/rag/search")
    finally:
        service.logger.remove(sink_id)

    assert result["user_limits"] == []
    assert result["role_limits"] == []
    _assert_safe_log("\n".join(messages))
