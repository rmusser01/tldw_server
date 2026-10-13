"""Test helpers for the LLM quota checks (LLM usage accounting spec §4a, §4b)."""

import asyncio
from typing import Any

import pytest


def refuse_llm_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every LLM quota check refuse: the user's monthly LLM token limit is 0 and nothing else is limited."""

    async def _user_quota(_uid: object, key: str) -> int | None:
        """Only the LLM token key is set, at 0."""
        return 0 if key == "limits.llm_tokens_per_month" else None

    async def _no_tokens(_uid: object) -> float:
        """No tokens used this month."""
        return 0.0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _user_quota)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _no_tokens)


def refuse_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """``refuse_llm_work``, and switch on the check at the call (spec §4b) for a test without the application lifespan.

    The call-site check reads the quotas switch itself, so it is turned on for that
    check only, and cached limits are dropped. Called from a coroutine, it also makes
    the running loop the accounting loop, which the lifespan would otherwise set.
    """
    from tldw_Server_API.app.core.Usage import llm_accounting, quota_resolver

    refuse_llm_work(monkeypatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.llm_quota_gate.usage_quotas_enabled", lambda: True)
    quota_resolver.invalidate_all()
    try:
        monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    except RuntimeError:
        return


def assert_llm_quota_402(response: Any) -> None:
    """The response is the public LLM quota refusal: 402, limit_exceeded, Retry-After."""
    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    assert (detail["error"], detail["category"], detail["limit"]) == ("limit_exceeded", "llm_tokens_month", 0)
    assert int(response.headers["Retry-After"]) >= 1
