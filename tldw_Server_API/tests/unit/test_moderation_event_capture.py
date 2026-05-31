from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy, PatternRule
from tldw_Server_API.app.core.Moderation.review_service import (
    capture_moderation_review_item,
    get_moderation_review_service,
)


class _Metrics:
    def track_moderation_input(self, *_args, **_kwargs) -> None:
        return None


class _Audit:
    async def log_event(self, *_args, **_kwargs) -> None:
        return None

    async def flush(self, *, raise_on_failure: bool = False) -> None:
        return None


class _Moderation:
    def __init__(self, policy: ModerationPolicy, action: str = "block") -> None:
        self.policy = policy
        self.action = action

    def get_effective_policy(self, _user_id: str) -> ModerationPolicy:
        return self.policy

    def evaluate_action_with_match(self, text: str, _policy: ModerationPolicy, _phase: str):
        return self.action, "[REDACTED]" if self.action == "redact" else None, "secret", "pii", (6, 12)

    def build_sanitized_snippet(self, text: str, policy: ModerationPolicy, match_span, pattern):
        return "hello [REDACTED]"

    def redact_text(self, text: str, _policy: ModerationPolicy) -> str:
        return text.replace("secret", "[REDACTED]")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_input_moderation_capture_is_gated_off_by_default(monkeypatch):
    captured: list[dict] = []
    monkeypatch.delenv("MODERATION_REVIEW_CAPTURE_ENABLED", raising=False)
    monkeypatch.setattr(chat_service, "capture_moderation_review_item", lambda **payload: captured.append(payload))

    request_data = SimpleNamespace(messages=[SimpleNamespace(role="user", content="hello secret")])
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))
    policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        input_action="warn",
        block_patterns=[PatternRule(regex=__import__("re").compile("secret"), action="warn", categories={"pii"})],
    )

    await chat_service.moderate_input_messages(
        request_data=request_data,
        request=request,
        moderation_service=_Moderation(policy, action="warn"),
        topic_monitoring_service=None,
        metrics=_Metrics(),
        audit_service=_Audit(),
        audit_context=object(),
        client_id="client-1",
    )

    assert captured == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_input_moderation_capture_records_sanitized_idempotent_payload_when_enabled(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setenv("MODERATION_REVIEW_CAPTURE_ENABLED", "1")
    monkeypatch.setattr(chat_service, "capture_moderation_review_item", lambda **payload: captured.append(payload))

    request_data = SimpleNamespace(messages=[SimpleNamespace(role="user", content="hello secret raw-value")])
    request = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))
    policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        input_action="block",
        block_patterns=[PatternRule(regex=__import__("re").compile("secret"), action="block", categories={"pii"})],
    )

    with pytest.raises(HTTPException):
        await chat_service.moderate_input_messages(
            request_data=request_data,
            request=request,
            moderation_service=_Moderation(policy, action="block"),
            topic_monitoring_service=None,
            metrics=_Metrics(),
            audit_service=_Audit(),
            audit_context=object(),
            client_id="client-1",
        )

    assert len(captured) == 1
    payload = captured[0]
    assert payload["phase"] == "input"
    assert payload["action"] == "block"
    assert payload["category"] == "pii"
    assert payload["excerpt"] == "hello [REDACTED]"
    assert "raw-value" not in str(payload)


@pytest.mark.unit
def test_capture_moderation_review_item_is_idempotent_for_repeated_outcomes(monkeypatch, tmp_path):
    monkeypatch.setenv("MODERATION_REVIEW_CAPTURE_ENABLED", "1")
    monkeypatch.setenv("MODERATION_REVIEW_DB_PATH", str(tmp_path / "review.db"))
    get_moderation_review_service.cache_clear()
    try:
        first = capture_moderation_review_item(
            phase="input",
            action="block",
            excerpt="hello [REDACTED]",
            category="pii",
            matched_pattern="secret",
            effective_policy={"enabled": True, "block_patterns": ["secret"]},
            source_type="chat",
            source_id="conversation-1",
            user_id="user-1",
        )
        second = capture_moderation_review_item(
            phase="input",
            action="block",
            excerpt="hello [REDACTED]",
            category="pii",
            matched_pattern="secret",
            effective_policy={"enabled": True, "block_patterns": ["secret"]},
            source_type="chat",
            source_id="conversation-1",
            user_id="user-1",
        )

        service = get_moderation_review_service()
        items = service.list_items(status="needs_review", limit=10)
    finally:
        get_moderation_review_service.cache_clear()

    assert first is not None
    assert second is not None
    assert first["id"] == second["id"]
    assert items["total"] == 1
    assert "secret" not in str(first["effective_policy"])
