from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceChatRequest,
    SharedWorkspaceChatResponse,
)
from tldw_Server_API.app.core.Chat.chat_target_resolution import ResolvedChatTarget
from tldw_Server_API.app.core.Sharing.shared_workspace_chat_service import (
    SharedSourceSnapshot,
    SharedSourceSnapshotItem,
    SharedWorkspaceGeneratedAnswer,
    SharedWorkspacePromptBudget,
    SharedWorkspaceRetrievalUnavailable,
    SharedWorkspaceSourceChanged,
    VerifiedSharedEvidence,
)

pytestmark = pytest.mark.asyncio

REQUEST_ID = UUID("de305d54-75b4-431b-adb2-eb6b9e546014")
NOW = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _resolved_target(monkeypatch):
    monkeypatch.setattr(
        sharing,
        "resolve_chat_target",
        lambda **kwargs: ResolvedChatTarget("openai", "gpt-model"),
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        share_id=42,
        recipient_user_id=9,
        owner_user_id=7,
        workspace_id="workspace-1",
        share_scope_type="team",
        share_scope_id=73,
        workspace={"name": "Shared research"},
    )


def _body() -> SharedWorkspaceChatRequest:
    return SharedWorkspaceChatRequest(
        request_id=REQUEST_ID,
        query="  What supports the conclusion?  ",
        source_scope={"mode": "include", "source_ids": ["source-1"]},
        provider="openai",
        model="gpt-model",
    )


def _snapshot() -> SharedSourceSnapshot:
    return SharedSourceSnapshot(
        mode="include",
        items=(
            SharedSourceSnapshotItem(
                source_id="source-1",
                media_id=11,
                media_uuid="media-uuid",
                content_hash="content-hash",
                readiness_class="ready",
            ),
        ),
        snapshot_hash="snapshot-hash",
    )


def _turn() -> SimpleNamespace:
    user_message = SimpleNamespace(
        message_id="user-message",
        role="user",
        content="What supports the conclusion?",
        created_at=NOW,
    )
    assistant_message = SimpleNamespace(
        message_id="assistant-message",
        role="assistant",
        content="Grounded answer",
        created_at=NOW,
    )
    return SimpleNamespace(
        request_id=REQUEST_ID,
        conversation_id="conversation-1",
        user_message=user_message,
        assistant_message=assistant_message,
        citations=(
            {
                "citation_id": "citation-1",
                "source_id": "source-1",
                "source_title": "Source",
                "locator": {"chunk": 2, "start_char": 0, "end_char": 8},
                "quote": "Evidence",
                "score": 0.9,
            },
        ),
        provider="openai",
        model="gpt-model",
        source_mode="include",
        effective_source_count=1,
    )


class _Access:
    def __init__(self, events: list[str], *, fail_call: int | None = None) -> None:
        self.events = events
        self.calls = 0
        self.fail_call = fail_call

    async def resolve(self, *, share_id: int, recipient_user_id: int):
        self.calls += 1
        self.events.append(f"authorize:{self.calls}")
        if self.calls == self.fail_call:
            from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
                SharedWorkspaceNotFound,
            )

            raise SharedWorkspaceNotFound()
        return _context()


class _Store:
    def __init__(
        self,
        events: list[str],
        disposition: str = "claimed",
        *,
        frozen: bool = False,
    ) -> None:
        self.events = events
        self.disposition = disposition
        self.frozen = frozen
        self.retry_codes: list[str] = []
        self.conflict_codes: list[str] = []
        self.completed = False

    def get_or_create_thread(self, **kwargs):
        self.events.append("thread")
        return SimpleNamespace(conversation_id="conversation-1")

    def claim_request(self, **kwargs):
        self.events.append("claim")
        return SimpleNamespace(
            disposition=self.disposition,
            completed_turn=_turn() if self.disposition == "replay" else None,
            retry_after_ms=1500 if self.disposition == "in_progress" else None,
            source_mode="include" if self.frozen else None,
            source_ids=("source-1",) if self.frozen else (),
            source_snapshot_hash="old-snapshot-hash" if self.frozen else None,
            provider="openai" if self.frozen else None,
            model="gpt-model" if self.frozen else None,
            **kwargs,
            lease_epoch=1,
            lease_token="lease-token",
        )

    def freeze_sources(self, **kwargs):
        self.events.append("freeze")
        return True

    def mark_retryable(self, *, claim, error_code: str):
        self.events.append(f"retryable:{error_code}")
        self.retry_codes.append(error_code)
        return True

    def mark_conflicted(self, *, claim, error_code: str):
        self.events.append(f"conflicted:{error_code}")
        self.conflict_codes.append(error_code)
        return True

    def complete_turn(self, **kwargs):
        self.events.append("complete")
        self.completed = True
        return _turn()


class _ChatService:
    def __init__(self, events: list[str], *, failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure

    def resolve_source_snapshot(self, **kwargs):
        self.events.append("snapshot")
        return _snapshot()

    async def retrieve_verified_evidence(self, **kwargs):
        self.events.append("retrieve")
        if self.failure is not None:
            raise self.failure
        return (
            VerifiedSharedEvidence(
                label="E1",
                source_id="source-1",
                source_title="Source",
                content="Evidence",
                score=0.9,
                chunk_index=2,
                start_char=0,
                end_char=8,
            ),
        )

    def revalidate_source_snapshot(self, **kwargs):
        self.events.append("revalidate")
        if isinstance(self.failure, SharedWorkspaceSourceChanged):
            raise self.failure
        return kwargs["snapshot"]

    async def generate_grounded_answer(self, **kwargs):
        self.events.append("generate")
        return SharedWorkspaceGeneratedAnswer(
            answer="Grounded answer",
            citations=_turn().citations,
            budgeted_evidence=(),
            prompt_budget=SharedWorkspacePromptBudget(8192, 1200, 820, 1000, "utf8_bytes"),
        )


class _Limiter:
    def __init__(self, events: list[str], allowed: bool = True) -> None:
        self.events = events
        self.allowed = allowed
        self.calls: list[tuple[str, str, int]] = []

    async def check_rate_limit(self, user_id, conversation_id, estimated_tokens):
        self.events.append("rate")
        self.calls.append((user_id, conversation_id, estimated_tokens))
        return self.allowed, None if self.allowed else "private limiter detail"


class _Audit:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, dict]] = []

    async def log(self, event_type: str, **kwargs):
        self.events.append("audit")
        self.calls.append((event_type, kwargs))


def _resources(store: _Store, chat: _ChatService):
    @asynccontextmanager
    async def _load(context):
        yield SimpleNamespace(store=store, chat_service=chat)

    return _load


async def _run(
    *,
    disposition: str = "claimed",
    allowed: bool = True,
    failure: Exception | None = None,
    fail_access_call: int | None = None,
    frozen: bool = False,
):
    events: list[str] = []
    store = _Store(events, disposition, frozen=frozen)
    chat = _ChatService(events, failure=failure)
    limiter = _Limiter(events, allowed)
    audit = _Audit(events)
    access = _Access(events, fail_call=fail_access_call)
    try:
        response = await sharing._orchestrate_shared_workspace_chat(
            share_id=42,
            body=_body(),
            request=SimpleNamespace(),
            recipient_user_id=9,
            access_service=access,
            resource_loader=_resources(store, chat),
            rate_limiter=limiter,
            audit=audit,
        )
        return response, events, store, limiter, audit
    except Exception as exc:
        exc.test_state = (events, store, limiter, audit)
        raise


async def test_claimant_orders_authorization_rate_freeze_generation_and_completion() -> None:
    response, events, store, limiter, audit = await _run()

    assert isinstance(response, SharedWorkspaceChatResponse)
    assert events == [
        "authorize:1",
        "thread",
        "claim",
        "rate",
        "snapshot",
        "freeze",
        "retrieve",
        "authorize:2",
        "revalidate",
        "generate",
        "authorize:3",
        "revalidate",
        "complete",
        "audit",
    ]
    assert limiter.calls == [("9", "shared:9:42", len(_body().query.strip().encode("utf-8")))]
    assert store.completed is True
    assert response.replay.replayed is False
    assert response.source_scope.effective_source_count == 1
    assert audit.calls


@pytest.mark.parametrize(
    ("disposition", "code"),
    [("in_progress", "request_in_progress"), ("request_id_conflict", "request_id_conflict")],
)
async def test_nonclaim_dispositions_do_not_reserve_or_generate(
    disposition: str,
    code: str,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(disposition=disposition)

    events, store, limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == code
    assert "rate" not in events
    assert "retrieve" not in events
    assert "generate" not in events
    assert store.completed is False
    assert limiter.calls == []


async def test_completed_replay_returns_stored_turn_without_generation_reservation() -> None:
    response, events, store, limiter, audit = await _run(disposition="replay")

    assert response.replay.replayed is True
    assert events == ["authorize:1", "thread", "claim", "audit"]
    assert store.completed is False
    assert limiter.calls == []
    assert audit.calls[0][1]["metadata"]["replay"] is True


async def test_recipient_share_rate_rejection_releases_receipt() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(allowed=False)

    events, store, _limiter, audit = exc_info.value.test_state
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["code"] == "shared_chat_rate_limited"
    assert store.retry_codes == ["shared_chat_rate_limited"]
    assert "snapshot" not in events
    assert audit.calls[0][1]["metadata"]["outcome"] == "shared_chat_rate_limited"


async def test_transient_retrieval_failure_marks_retryable_without_persistence() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(failure=SharedWorkspaceRetrievalUnavailable())

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "retrieval_unavailable"
    assert store.retry_codes == ["retrieval_unavailable"]
    assert "complete" not in events


async def test_frozen_source_change_marks_conflicted_without_persistence() -> None:
    chat_failure = SharedWorkspaceSourceChanged()
    with pytest.raises(HTTPException) as exc_info:
        await _run(failure=chat_failure)

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "shared_source_changed"
    assert store.conflict_codes == ["shared_source_changed"]
    assert store.retry_codes == []
    assert "complete" not in events


async def test_reclaimed_frozen_receipt_rejects_persisted_snapshot_hash_drift() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(frozen=True)

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "shared_source_changed"
    assert store.conflict_codes == ["shared_source_changed"]
    assert "retrieve" not in events
    assert "complete" not in events


async def test_revocation_before_generation_releases_receipt_and_saves_nothing() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(fail_access_call=2)

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["code"] == "shared_workspace_not_found"
    assert store.retry_codes == ["shared_workspace_not_found"]
    assert "generate" not in events
    assert "complete" not in events


async def test_audit_metadata_is_bounded_and_contains_no_content() -> None:
    response, _events, _store, _limiter, audit = await _run()

    event_type, kwargs = audit.calls[0]
    serialized = str(kwargs).lower()
    assert event_type == "share.chat.completed"
    assert set(kwargs["metadata"]) == {
        "effective_source_count",
        "provider",
        "model",
        "outcome",
        "replay",
        "timings_ms",
    }
    assert response.turn.assistant_message.content not in serialized
    assert _body().query.strip().lower() not in serialized
    assert "evidence" not in serialized


async def test_receipt_lease_derives_from_provider_timeout_and_stays_bounded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sharing,
        "load_and_log_configs",
        lambda: {
            "openai_api": {"api_timeout": "90"},
            "ollama_api": {"api_timeout": "9009"},
        },
    )

    ordinary = _body()
    local = ordinary.model_copy(update={"provider": "ollama"})

    assert sharing._shared_chat_lease_seconds(ordinary) == 300
    assert sharing._shared_chat_lease_seconds(local) == 1_800
