"""Test helpers for the LLM quota boundary checks (LLM usage accounting spec §4a)."""

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


def assert_llm_quota_402(response: Any) -> None:
    """The response is the public LLM quota refusal: 402, limit_exceeded, Retry-After."""
    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    assert (detail["error"], detail["category"], detail["limit"]) == ("limit_exceeded", "llm_tokens_month", 0)
    assert int(response.headers["Retry-After"]) >= 1
