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
        frozen_snapshot_hash: str = "old-snapshot-hash",
        transition_outcome: bool | Exception = True,
        reload_disposition: str | None = None,
    ) -> None:
        self.events = events
        self.disposition = disposition
        self.frozen = frozen
        self.frozen_snapshot_hash = frozen_snapshot_hash
        self.transition_outcome = transition_outcome
        self.reload_disposition = reload_disposition
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
            source_snapshot_hash=self.frozen_snapshot_hash if self.frozen else None,
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
        if isinstance(self.transition_outcome, Exception):
            raise self.transition_outcome
        if self.transition_outcome:
            self.retry_codes.append(error_code)
        return self.transition_outcome

    def mark_conflicted(self, *, claim, error_code: str):
        self.events.append(f"conflicted:{error_code}")
        if isinstance(self.transition_outcome, Exception):
            raise self.transition_outcome
        if self.transition_outcome:
            self.conflict_codes.append(error_code)
        return self.transition_outcome

    def reload_claim_state(self, *, claim, now):
        self.events.append("reload_claim")
        if self.reload_disposition == "replay":
            return SimpleNamespace(
                disposition="replay",
                completed_turn=_turn(),
                lease_epoch=claim.lease_epoch + 1,
                retry_after_ms=None,
            )
        if self.reload_disposition == "in_progress":
            return SimpleNamespace(
                disposition="in_progress",
                completed_turn=None,
                lease_epoch=claim.lease_epoch + 1,
                retry_after_ms=1750,
            )
        return None

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


def _resources(
    store: _Store,
    chat: _ChatService,
    *,
    owner_failure: bool = False,
    owner_exit_failure: bool = False,
):
    @asynccontextmanager
    async def _load(context):
        store.events.append("owner:enter")
        try:
            if owner_failure:
                raise RuntimeError("private owner resource failure")
            yield SimpleNamespace(chat_service=chat)
        finally:
            store.events.append("owner:exit")
            if owner_exit_failure:
                raise RuntimeError("private owner resource exit failure")

    return _load


def _store_loader(store: _Store):
    async def _load(context):
        store.events.append("recipient:store")
        return store

    return _load


async def _run(
    *,
    disposition: str = "claimed",
    allowed: bool = True,
    failure: Exception | None = None,
    fail_access_call: int | None = None,
    frozen: bool = False,
    frozen_snapshot_hash: str = "old-snapshot-hash",
    transition_outcome: bool | Exception = True,
    reload_disposition: str | None = None,
    owner_failure: bool = False,
    owner_exit_failure: bool = False,
):
    events: list[str] = []
    store = _Store(
        events,
        disposition,
        frozen=frozen,
        frozen_snapshot_hash=frozen_snapshot_hash,
        transition_outcome=transition_outcome,
        reload_disposition=reload_disposition,
    )
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
            store_loader=_store_loader(store),
            resource_loader=_resources(
                store,
                chat,
                owner_failure=owner_failure,
                owner_exit_failure=owner_exit_failure,
            ),
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
        "recipient:store",
        "thread",
        "claim",
        "rate",
        "owner:enter",
        "snapshot",
        "freeze",
        "retrieve",
        "authorize:2",
        "revalidate",
        "generate",
        "authorize:3",
        "revalidate",
        "complete",
        "owner:exit",
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
    assert "owner:enter" not in events
    assert store.completed is False
    assert limiter.calls == []


async def test_completed_replay_returns_stored_turn_without_generation_reservation() -> None:
    response, events, store, limiter, audit = await _run(disposition="replay")

    assert response.replay.replayed is True
    assert events == ["authorize:1", "recipient:store", "thread", "claim", "audit"]
    assert store.completed is False
    assert limiter.calls == []
    assert audit.calls[0][1]["metadata"]["replay"] is True


async def test_completed_replay_does_not_open_owner_resources() -> None:
    response, events, _store, _limiter, _audit = await _run(disposition="replay")

    assert response.replay.replayed is True
    assert "owner:enter" not in events
    assert "owner:exit" not in events


async def test_recipient_share_rate_rejection_releases_receipt() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(allowed=False)

    events, store, _limiter, audit = exc_info.value.test_state
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["code"] == "shared_chat_rate_limited"
    assert store.retry_codes == ["shared_chat_rate_limited"]
    assert "snapshot" not in events
    assert audit.calls[0][1]["metadata"]["outcome"] == "shared_chat_rate_limited"


async def test_owner_resource_acquisition_failure_is_typed_and_retryable() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(owner_failure=True)

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "shared_workspace_unavailable"
    assert store.retry_codes == ["shared_workspace_unavailable"]
    assert "complete" not in events
    assert store.completed is False


async def test_owner_resource_exit_failure_recovers_completed_turn() -> None:
    response, events, store, _limiter, _audit = await _run(
        owner_exit_failure=True,
        transition_outcome=False,
        reload_disposition="replay",
    )

    assert response.replay.replayed is True
    assert "complete" in events
    assert "owner:exit" in events
    assert "reload_claim" in events
    assert store.completed is True


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


async def test_reclaimed_frozen_receipt_rechecks_current_provider_override(
    monkeypatch,
) -> None:
    captured: list[dict[str, str | None]] = []

    def _drifted_target(**kwargs):
        captured.append(kwargs)
        return ResolvedChatTarget("openai", "newly-allowed-model")

    monkeypatch.setattr(sharing, "resolve_chat_target", _drifted_target)

    with pytest.raises(HTTPException) as exc_info:
        await _run(frozen=True, frozen_snapshot_hash="snapshot-hash")

    events, store, _limiter, _audit = exc_info.value.test_state
    assert captured == [
        {"requested_provider": "openai", "requested_model": "gpt-model"}
    ]
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "no_provider_configured"
    assert store.retry_codes == ["no_provider_configured"]
    assert "retrieve" not in events


async def test_reclaimed_frozen_receipt_rejects_unavailable_adapter(
    monkeypatch,
) -> None:
    def _unavailable_target(**kwargs):
        raise sharing.ChatConfigurationError(message="private adapter state")

    monkeypatch.setattr(sharing, "resolve_chat_target", _unavailable_target)

    with pytest.raises(HTTPException) as exc_info:
        await _run(frozen=True, frozen_snapshot_hash="snapshot-hash")

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "no_provider_configured"
    assert store.retry_codes == ["no_provider_configured"]
    assert "retrieve" not in events


@pytest.mark.parametrize("transition_outcome", [False, RuntimeError("transition failed")])
@pytest.mark.parametrize("failure_path", ["rate", "transient", "source_conflict"])
async def test_required_failure_transition_must_succeed_before_original_error(
    failure_path: str,
    transition_outcome: bool | Exception,
) -> None:
    kwargs = {"transition_outcome": transition_outcome}
    if failure_path == "rate":
        kwargs["allowed"] = False
    elif failure_path == "transient":
        kwargs["failure"] = SharedWorkspaceRetrievalUnavailable()
    else:
        kwargs["failure"] = SharedWorkspaceSourceChanged()

    with pytest.raises(HTTPException) as exc_info:
        await _run(**kwargs)

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "shared_workspace_unavailable"
    assert "reload_claim" in events
    assert "complete" not in events
    assert store.completed is False


async def test_failed_transition_returns_completed_race_winner() -> None:
    response, events, store, _limiter, _audit = await _run(
        allowed=False,
        transition_outcome=False,
        reload_disposition="replay",
    )

    assert response.replay.replayed is True
    assert "reload_claim" in events
    assert store.completed is False


async def test_failed_transition_classifies_newer_active_race_winner() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _run(
            allowed=False,
            transition_outcome=False,
            reload_disposition="in_progress",
        )

    events, store, _limiter, _audit = exc_info.value.test_state
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "request_in_progress"
    assert exc_info.value.detail["retry_after_ms"] == 1750
    assert "reload_claim" in events
    assert store.completed is False


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
    from tldw_Server_API.app.core.Chat import chat_target_resolution

    monkeypatch.setattr(chat_target_resolution, "get_default_provider", lambda: "openai")
    monkeypatch.setattr(
        sharing,
        "load_and_log_configs",
        lambda: {
            "openai_api": {"api_timeout": "90"},
            "ollama_api": {"api_timeout": "9009"},
        },
    )

    ordinary = _body()
    defaulted = ordinary.model_copy(update={"provider": None, "model": None})
    local = ordinary.model_copy(update={"provider": "ollama"})

    assert sharing._shared_chat_lease_seconds(ordinary) == 300
    assert sharing._shared_chat_lease_seconds(defaulted) == 300
    assert sharing._shared_chat_lease_seconds(local) == 1_800


async def test_receipt_lease_uses_qualified_model_provider_and_registry_alias(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sharing,
        "load_and_log_configs",
        lambda: {
            "anthropic_api": {"api_timeout": "900"},
            "llama_api": {"api_timeout": "420"},
        },
    )

    qualified = _body().model_copy(
        update={"provider": None, "model": "anthropic/claude-special"}
    )
    aliased = _body().model_copy(
        update={"provider": "llamacpp", "model": "local-model"}
    )

    assert sharing._shared_chat_lease_seconds(qualified) == 960
    assert sharing._shared_chat_lease_seconds(aliased) == 480
