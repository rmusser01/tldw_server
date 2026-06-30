import asyncio

import pytest

from fastapi import HTTPException

from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as crl
from tldw_Server_API.app.core import config as core_config
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter


class _FakeGovernor:
    def __init__(self, payload):
        self.payload = payload
        self.peek_calls = []

    async def reserve(self, *_args, **_kwargs):
        class _Decision:
            def __init__(self, allowed, retry_after):
                self.allowed = allowed
                self.retry_after = retry_after

        return _Decision(self.payload["allowed"], self.payload.get("retry_after")), "handle"

    async def commit(self, *_args, **_kwargs):
        return None

    async def peek_with_policy(self, entity: str, categories: list[str], policy_id: str):
        self.peek_calls.append((entity, categories, policy_id))
        return {"requests": {"remaining": 7, "reset": 0}}


@pytest.mark.unit
def test_rate_limiter_allows_when_rg_disabled(monkeypatch):
    monkeypatch.setattr(crl, "_rg_character_enabled", lambda: False)
    limiter = CharacterRateLimiter(enabled=True)

    allowed, remaining = asyncio.run(limiter.check_rate_limit(user_id=123, operation="test"))

    assert allowed is True
    assert remaining == 0


@pytest.mark.unit
def test_rate_limiter_denies_when_rg_denies(monkeypatch):
    monkeypatch.setattr(crl, "_rg_character_enabled", lambda: True)
    monkeypatch.setenv("RG_CHARACTER_CHAT_ENFORCE_REQUESTS", "1")
    async def _deny(**_kwargs):
        return {"allowed": False, "retry_after": 5, "policy_id": "character_chat.default"}

    monkeypatch.setattr(crl, "_maybe_enforce_with_rg_character", _deny)

    limiter = CharacterRateLimiter(enabled=True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(limiter.check_rate_limit(user_id=42, operation="test"))

    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "5"


@pytest.mark.unit
def test_rate_limiter_allows_when_rg_allows(monkeypatch):
    monkeypatch.setattr(crl, "_rg_character_enabled", lambda: True)
    monkeypatch.setenv("RG_CHARACTER_CHAT_ENFORCE_REQUESTS", "1")
    async def _allow(**_kwargs):
        return {"allowed": True, "retry_after": None, "policy_id": "character_chat.default"}

    monkeypatch.setattr(crl, "_maybe_enforce_with_rg_character", _allow)

    limiter = CharacterRateLimiter(enabled=True)

    allowed, remaining = asyncio.run(limiter.check_rate_limit(user_id=7, operation="test"))

    assert allowed is True
    assert remaining == 0


@pytest.mark.unit
def test_character_guardrails_raise():
    limiter = CharacterRateLimiter(max_characters=2, max_import_size_mb=1, max_chats_per_user=1, max_messages_per_chat=1)

    with pytest.raises(HTTPException):
        asyncio.run(limiter.check_character_limit(user_id=1, current_count=2))

    with pytest.raises(HTTPException):
        limiter.check_import_size(2 * 1024 * 1024)

    with pytest.raises(HTTPException):
        asyncio.run(limiter.check_chat_limit(user_id=1, current_chat_count=1))

    assert asyncio.run(limiter.check_message_limit(chat_id="c1", projected_message_count=1)) is True

    with pytest.raises(HTTPException):
        asyncio.run(limiter.check_message_limit(chat_id="c1", projected_message_count=2))


@pytest.mark.unit
def test_soft_message_limit_guardrail():
    limiter = CharacterRateLimiter(max_messages_per_chat_soft=1)

    with pytest.raises(HTTPException):
        asyncio.run(limiter.check_soft_message_limit(chat_id="c1", current_message_count=1))


@pytest.mark.unit
def test_get_usage_stats_peeks_rg(monkeypatch):
    fake = _FakeGovernor({"allowed": True})
    monkeypatch.setattr(crl, "_rg_character_enabled", lambda: True)
    async def _get_fake():
        return fake

    monkeypatch.setattr(crl, "_get_character_rg_governor", _get_fake)

    limiter = CharacterRateLimiter(enabled=True)

    stats = asyncio.run(limiter.get_usage_stats(99))

    assert stats["requests"]["remaining"] == 7
    assert fake.peek_calls


@pytest.mark.unit
def test_get_character_rate_limiter_ignores_global_rate_limit_enabled(monkeypatch):
    crl._rate_limiter = None
    monkeypatch.delenv("CHARACTER_RATE_LIMIT_ENABLED", raising=False)
    monkeypatch.setattr(crl, "is_test_mode", lambda: False)
    monkeypatch.setattr(crl, "env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(
        core_config,
        "settings",
        {
            "SINGLE_USER_MODE": True,
            "RATE_LIMIT_ENABLED": True,
        },
    )
    try:
        limiter = crl.get_character_rate_limiter()
        assert limiter.enabled is False
    finally:
        crl._rate_limiter = None
