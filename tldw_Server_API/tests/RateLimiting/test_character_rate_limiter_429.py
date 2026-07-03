"""Assert rate limits actually fire: 429 + Retry-After (audit F7)."""
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as crl


@pytest.fixture
def limiter(monkeypatch):
    monkeypatch.setattr(crl, "_rg_character_enabled", lambda: True)
    monkeypatch.setattr(crl, "_rg_character_enforce_requests", lambda: True)
    lim = crl.CharacterRateLimiter()
    lim.enabled = True
    return lim


@pytest.mark.unit
async def test_denied_decision_raises_429_with_retry_after(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character",
        AsyncMock(return_value={"allowed": False, "retry_after": 7, "policy_id": "p1"}),
    )
    with pytest.raises(HTTPException) as exc:
        await limiter.check_rate_limit(user_id=1, operation="character_op")
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "7"


@pytest.mark.unit
async def test_denied_decision_without_retry_after_defaults_to_60(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character",
        AsyncMock(return_value={"allowed": False, "policy_id": "p1"}),
    )
    with pytest.raises(HTTPException) as exc:
        await limiter.check_rate_limit(user_id=1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "60"


@pytest.mark.unit
async def test_allowed_decision_passes(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character",
        AsyncMock(return_value={"allowed": True}),
    )
    allowed, _ = await limiter.check_rate_limit(user_id=1)
    assert allowed is True


@pytest.mark.unit
async def test_unavailable_governor_fails_open(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character", AsyncMock(return_value=None)
    )
    allowed, _ = await limiter.check_rate_limit(user_id=1)
    assert allowed is True  # documented fail-open behavior (crl.py:159-163)


@pytest.mark.unit
async def test_disabled_limiter_short_circuits():
    lim = crl.CharacterRateLimiter()
    lim.enabled = False
    allowed, _ = await lim.check_rate_limit(user_id=1)
    assert allowed is True
