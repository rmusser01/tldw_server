"""Tests for the storage quota enforcement FastAPI dependency."""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import storage_quota_guard as quota_guard_mod
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import (
    guard_storage_quota,
)


class _RecordingLogger:
    def __init__(self):
        self.warning_calls = []

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


def _assert_warning_logs_are_sanitized(logger: _RecordingLogger):
    assert logger.warning_calls
    for args, kwargs in logger.warning_calls:
        assert not kwargs.get("exc_info")
        rendered = " ".join(str(arg) for arg in args)
        assert "/tmp/source" not in rendered
        assert "token=secret" not in rendered


def _fake_user(
    *,
    user_id: int | str = 1,
    org_ids: Sequence[int] | None = None,
    active_org_id: int | None = None,
) -> SimpleNamespace:
    """Return a minimal User-like object."""
    return SimpleNamespace(
        id=user_id,
        org_ids=list(org_ids or []),
        active_org_id=active_org_id,
    )


def _fake_request(*, org_id: int | None = None, content_length: str | None = None):
    """Return a minimal Request-like object."""
    state = SimpleNamespace()
    if org_id is not None:
        state.org_id = org_id
    headers = {}
    if content_length is not None:
        headers["content-length"] = content_length
    return SimpleNamespace(state=state, headers=headers)


def _fake_response():
    return SimpleNamespace(headers={})


# ---------------------------------------------------------------------------
# Disabled / skipped paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard._is_enabled", return_value=False)
async def test_guard_skips_when_disabled(_mock_enabled):
    """When STORAGE_QUOTA_ENFORCEMENT=0 the guard is a no-op."""
    result = await guard_storage_quota(
        request=_fake_request(),
        response=_fake_response(),
        current_user=_fake_user(),
    )
    assert result is None


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=True,
)
async def test_guard_skips_single_user_mode(_mock_mode):
    """Single-user / desktop deployments skip enforcement."""
    result = await guard_storage_quota(
        request=_fake_request(),
        response=_fake_response(),
        current_user=_fake_user(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Quota allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={"allowed": True, "reason": "Quota check passed", "used_mb": 50, "quota_mb": 1024, "remaining_mb": 974},
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_allows_when_under_quota(_pool, _check, _mode):
    """Upload proceeds when quota is not exceeded."""
    result = await guard_storage_quota(
        request=_fake_request(org_id=10),
        response=_fake_response(),
        current_user=_fake_user(user_id=1, org_ids=[10]),
    )
    assert result is None
    _check.assert_awaited_once()


# ---------------------------------------------------------------------------
# Quota exceeded → HTTP 413
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={
        "allowed": False,
        "reason": "Storage quota exceeded (at hard limit)",
        "used_mb": 1024,
        "quota_mb": 1024,
        "remaining_mb": 0,
    },
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_blocks_when_quota_exceeded(_pool, _check, _mode):
    """Upload is rejected with 413 when storage is at hard limit."""
    with pytest.raises(HTTPException) as exc_info:
        await guard_storage_quota(
            request=_fake_request(org_id=10),
            response=_fake_response(),
            current_user=_fake_user(user_id=1, org_ids=[10]),
        )
    assert exc_info.value.status_code == 413
    assert "storage_quota_exceeded" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Soft limit → warning header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={
        "allowed": True,
        "reason": "Warning: Approaching storage limit (soft limit reached)",
        "used_mb": 900,
        "quota_mb": 1024,
        "remaining_mb": 124,
    },
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_sets_warning_header_at_soft_limit(_pool, _check, _mode):
    """Soft limit adds X-Storage-Warning header but allows request."""
    resp = _fake_response()
    result = await guard_storage_quota(
        request=_fake_request(org_id=10),
        response=resp,
        current_user=_fake_user(user_id=1, org_ids=[10]),
    )
    assert result is None
    assert "X-Storage-Warning" in resp.headers
    assert "soft limit" in resp.headers["X-Storage-Warning"].lower()


# ---------------------------------------------------------------------------
# Fail-open on errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    side_effect=ConnectionError("db unavailable"),
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_fails_open_on_error(_pool, _check, _mode):
    """When the quota check itself fails, the request is allowed (fail-open)."""
    result = await guard_storage_quota(
        request=_fake_request(org_id=10),
        response=_fake_response(),
        current_user=_fake_user(user_id=1, org_ids=[10]),
    )
    assert result is None


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    side_effect=ConnectionError("db unavailable /tmp/source token=secret"),
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_fail_open_log_is_sanitized(_pool, _check, _mode, monkeypatch):
    """Quota backend failures are swallowed without logging raw backend details."""
    recording_logger = _RecordingLogger()
    monkeypatch.setattr(quota_guard_mod, "logger", recording_logger)

    result = await guard_storage_quota(
        request=_fake_request(org_id=10),
        response=_fake_response(),
        current_user=_fake_user(user_id=1, org_ids=[10]),
    )

    assert result is None
    _assert_warning_logs_are_sanitized(recording_logger)


# ---------------------------------------------------------------------------
# Org resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={"allowed": True, "reason": "No quota limit set", "used_mb": 0, "quota_mb": None, "remaining_mb": None},
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_resolves_org_from_active_org_id(_pool, mock_check, _mode):
    """When request.state has no org_id, falls back to user.active_org_id."""
    await guard_storage_quota(
        request=_fake_request(),  # no org_id on state
        response=_fake_response(),
        current_user=_fake_user(user_id=1, active_org_id=42),
    )
    call_kwargs = mock_check.call_args.kwargs
    assert call_kwargs["org_id"] == 42


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={"allowed": True, "reason": "No quota limit set", "used_mb": 0, "quota_mb": None, "remaining_mb": None},
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_resolves_org_from_org_ids_list(_pool, mock_check, _mode):
    """When request.state and active_org_id are absent, falls back to org_ids[0]."""
    await guard_storage_quota(
        request=_fake_request(),
        response=_fake_response(),
        current_user=_fake_user(user_id=1, org_ids=[99, 100]),
    )
    call_kwargs = mock_check.call_args.kwargs
    assert call_kwargs["org_id"] == 99


# ---------------------------------------------------------------------------
# Content-Length estimation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={"allowed": True, "reason": "Quota check passed", "used_mb": 0, "quota_mb": 1024, "remaining_mb": 1024},
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
async def test_guard_passes_content_length_as_file_size(_pool, mock_check, _mode):
    """Content-Length header is used as an upload size estimate."""
    await guard_storage_quota(
        request=_fake_request(org_id=10, content_length="5242880"),
        response=_fake_response(),
        current_user=_fake_user(user_id=1, org_ids=[10]),
    )
    call_kwargs = mock_check.call_args.kwargs
    assert call_kwargs["file_size_bytes"] == 5242880


@pytest.mark.asyncio
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.is_single_user_profile_mode",
    return_value=False,
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.check_storage_quota",
    new_callable=AsyncMock,
    return_value={"allowed": True, "reason": "Quota check passed", "used_mb": 0, "quota_mb": 1024, "remaining_mb": 1024},
)
@patch(
    "tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.get_db_pool",
    new_callable=AsyncMock,
    return_value=MagicMock(),
)
@patch("tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard.logger.warning")
async def test_guard_logs_when_user_id_falls_back_to_zero(_warning, _pool, mock_check, _mode):
    """Invalid user ids fall back to 0 and emit a warning for operator visibility."""
    await guard_storage_quota(
        request=_fake_request(org_id=10),
        response=_fake_response(),
        current_user=_fake_user(user_id="not-an-int", org_ids=[10]),
    )

    call_kwargs = mock_check.call_args.kwargs
    assert call_kwargs["user_id"] == 0
    _warning.assert_any_call("Storage quota guard falling back to anonymous user_id=0 for user {}", "not-an-int")
