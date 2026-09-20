from collections.abc import Callable, Generator
from typing import Any

import pytest


@pytest.fixture
def healthy_absent_provider_override_snapshot() -> Generator[None, None, None]:
    """Expose an explicitly healthy empty override snapshot for legacy routes."""

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides

    with llm_provider_overrides._OVERRIDE_LOCK:
        original_overrides = dict(llm_provider_overrides._OVERRIDE_CACHE)
        original_healthy = llm_provider_overrides._OVERRIDE_CACHE_HEALTHY
        original_ttl_disabled = (
            llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
        )

    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            original_overrides,
            healthy=original_healthy,
            ttl_enabled=not original_ttl_disabled,
        )


@pytest.fixture
def character_provider_adapter_boundary() -> tuple[
    list[dict[str, Any]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    """Bind authentic runtime capabilities at a recording provider boundary."""

    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    )
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
        bind_provider_call_credentials,
    )

    calls: list[dict[str, Any]] = []

    def bind_and_record(request: dict[str, Any]) -> dict[str, Any]:
        provider = str(request.get("api_endpoint") or "")
        bound, credentials = bind_provider_call_credentials(
            provider,
            request,
            consume=True,
        )
        assert credentials is not None
        assert bound["credentials_resolved"] is True
        assert isinstance(bound["api_key"], str) and bound["api_key"]
        assert PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in bound
        calls.append(bound)
        return bound

    return calls, bind_and_record


@pytest.fixture(autouse=True)
def _override_character_chat_rate_limits_for_character_chat(monkeypatch):
    """Relax Character-Chat rate limits for this test package to avoid flakiness.

    Tests focused on behavior, not rate enforcement, should not fail due to
    incidental shared limiter state. Specific rate-limit tests can override
    these env vars in their own scope when needed.
    """
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_OPS", "1000000")
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("MAX_CHARACTERS_PER_USER", "1000000")
    monkeypatch.setenv("MAX_CHATS_PER_USER", "1000000")
    monkeypatch.setenv("MAX_MESSAGES_PER_CHAT", "1000000")
    monkeypatch.setenv("MAX_CHAT_COMPLETIONS_PER_MINUTE", "1000000")
    monkeypatch.setenv("MAX_MESSAGE_SENDS_PER_MINUTE", "1000000")

    try:
        from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as _crl

        _crl._rate_limiter = None  # type: ignore[attr-defined]
    except Exception:
        _ = None

    yield

    try:
        from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as _crl

        _crl._rate_limiter = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
