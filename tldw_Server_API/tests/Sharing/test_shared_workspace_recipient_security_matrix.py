"""Integrated owner/member/nonmember recipient shared-workspace security matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceChatRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chat.chat_target_resolution import ResolvedChatTarget
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessService,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_chat_service import (
    SharedSourceSnapshot,
    SharedSourceSnapshotItem,
    SharedWorkspaceGeneratedAnswer,
    SharedWorkspacePromptBudget,
    SharedWorkspaceSourceChanged,
    VerifiedSharedEvidence,
)

pytestmark = pytest.mark.integration

OWNER_ID = 1
MEMBER_ID = 2
NONMEMBER_ID = 3
TEAM_ID = 10
WORKSPACE_ID = "shared-security-workspace"
OWNER_NOTE_SENTINEL = "OWNER-NOTE-SENTINEL-DO-NOT-LEAK"
OWNER_CHAT_SENTINEL = "OWNER-CHAT-SENTINEL-DO-NOT-LEAK"
OWNER_MEDIA_SENTINEL = "OWNER-UNRELATED-MEDIA-SENTINEL-DO-NOT-LEAK"
RECIPIENT_LOCAL_SENTINEL = "RECIPIENT-LOCAL-WORKSPACE-SENTINEL-DO-NOT-LEAK"
NOW = datetime(2026, 8, 22, 17, 0, tzinfo=timezone.utc)


class _UsersRepo:
    async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        if user_id == OWNER_ID:
            return {"id": OWNER_ID, "username": "Research owner"}
        return None


class _AllowLimiter:
    async def check_rate_limit(
        self,
        user_id: str,
        conversation_id: str,
        estimated_tokens: int,
    ) -> tuple[bool, None]:
        assert user_id in {str(OWNER_ID), str(MEMBER_ID)}
        assert conversation_id.startswith(f"shared:{user_id}:")
        assert estimated_tokens > 0
        return True, None


Hook = Callable[[], Awaitable[None] | None]


@dataclass
class _DeterministicChatService:
    owner_db: CharactersRAGDB
    media: dict[int, dict[str, str]]
    generation_calls: int = 0
    after_retrieval: Hook | None = None
    after_generation: Hook | None = None
    _evidence: tuple[VerifiedSharedEvidence, ...] = field(default_factory=tuple)

    async def _run_hook(self, hook: Hook | None) -> None:
        if hook is None:
            return
        result = hook()
        if result is not None:
            await result

    def resolve_source_snapshot(
        self,
        *,
        mode: str,
        source_ids: list[str] | tuple[str, ...],
        frozen_source_ids: tuple[str, ...] | None = None,
    ) -> SharedSourceSnapshot:
        sources = {
            str(source["id"]): source
            for source in self.owner_db.list_workspace_sources(WORKSPACE_ID)
            if source.get("selected", True)
        }
        requested = (
            tuple(frozen_source_ids)
            if frozen_source_ids is not None
            else tuple(sorted(sources))
            if mode == "all"
            else tuple(source_ids)
        )
        if not requested or any(source_id not in sources for source_id in requested):
            raise SharedWorkspaceSourceChanged()

        items = []
        for source_id in requested:
            source = sources[source_id]
            media_id = int(source["media_id"])
            media = self.media.get(media_id)
            if media is None:
                raise SharedWorkspaceSourceChanged()
            items.append(
                SharedSourceSnapshotItem(
                    source_id=source_id,
                    media_id=media_id,
                    media_uuid=media["uuid"],
                    content_hash=media["content_hash"],
                    readiness_class="ready",
                )
            )
        serialized = json.dumps(
            [
                (
                    item.source_id,
                    item.media_id,
                    item.media_uuid,
                    item.content_hash,
                    item.readiness_class,
                )
                for item in items
            ],
            separators=(",", ":"),
        )
        return SharedSourceSnapshot(
            mode=mode,
            items=tuple(items),
            snapshot_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    async def retrieve_verified_evidence(
        self,
        *,
        query: str,
        snapshot: SharedSourceSnapshot,
    ) -> tuple[VerifiedSharedEvidence, ...]:
        assert query
        self._evidence = tuple(
            VerifiedSharedEvidence(
                label=f"E{index}",
                source_id=item.source_id,
                source_title=f"Shared source {item.source_id[-1]}",
                content=f"Evidence from {item.source_id}",
                score=0.9,
                chunk_index=index,
                start_char=0,
                end_char=24,
            )
            for index, item in enumerate(snapshot.items, start=1)
        )
        await self._run_hook(self.after_retrieval)
        return self._evidence

    def revalidate_source_snapshot(
        self,
        *,
        snapshot: SharedSourceSnapshot,
    ) -> SharedSourceSnapshot:
        current = self.resolve_source_snapshot(
            mode=snapshot.mode,
            source_ids=snapshot.source_ids,
            frozen_source_ids=snapshot.source_ids if snapshot.mode == "all" else None,
        )
        if current.snapshot_hash != snapshot.snapshot_hash:
            raise SharedWorkspaceSourceChanged()
        return current

    async def generate_grounded_answer(
        self,
        *,
        query: str,
        evidence: tuple[VerifiedSharedEvidence, ...],
        **_kwargs: Any,
    ) -> SharedWorkspaceGeneratedAnswer:
        assert query
        self.generation_calls += 1
        citations = tuple(
            {
                "citation_id": f"citation-{item.label.lower()}",
                "source_id": item.source_id,
                "source_title": item.source_title,
                "locator": {
                    "chunk": item.chunk_index,
                    "start_char": item.start_char,
                    "end_char": item.end_char,
                },
                "quote": item.content,
                "score": item.score,
            }
            for item in evidence
        )
        await self._run_hook(self.after_generation)
        return SharedWorkspaceGeneratedAnswer(
            answer="Grounded recipient answer",
            citations=citations,
            budgeted_evidence=evidence,
            prompt_budget=SharedWorkspacePromptBudget(
                context_window=8192,
                max_output_tokens=1200,
                safety_tokens=400,
                prompt_tokens=800,
                counter="utf8_bytes",
            ),
        )


@dataclass
class RecipientSecurityHarness:
    repo: Any
    sharing_db: Any
    owner_db: CharactersRAGDB
    recipient_dbs: dict[int, CharactersRAGDB]
    share_id: int
    media: dict[int, dict[str, str]]
    owner_loader_calls: list[int]
    chat_service: _DeterministicChatService

    @property
    def access_service(self) -> SharedWorkspaceAccessService:
        async def load_owner(owner_user_id: int) -> CharactersRAGDB:
            self.owner_loader_calls.append(owner_user_id)
            return self.owner_db

        return SharedWorkspaceAccessService(self.repo, _UsersRepo(), load_owner)

    def user(self, user_id: int) -> User:
        return User(
            id=user_id,
            username=f"user-{user_id}",
            email=f"user-{user_id}@example.test",
            password_hash="hash",
        )

    def principal(self, user_id: int, *, can_read: bool = True) -> AuthPrincipal:
        return AuthPrincipal(
            kind="user",
            user_id=user_id,
            username=f"user-{user_id}",
            permissions=["sharing.read"] if can_read else [],
        )

    def client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        user_id: int,
        *,
        can_read: bool = True,
    ) -> TestClient:
        async def get_principal() -> AuthPrincipal:
            return self.principal(user_id, can_read=can_read)

        async def get_user() -> User:
            return self.user(user_id)

        async def allow_rate_limit(*_args: Any, **_kwargs: Any) -> None:
            return None

        async def load_sources(context: Any) -> list[dict[str, Any]]:
            assert context.owner_user_id == OWNER_ID
            return [dict(item) for item in self.owner_db.list_workspace_sources(WORKSPACE_ID)]

        async def project_sources(
            context: Any,
            sources: list[dict[str, Any]],
        ) -> dict[str, Any]:
            assert context.owner_user_id == OWNER_ID
            return {
                "sources": [
                    {
                        "id": source["id"],
                        "state": "queryable",
                        "status_reason": "source_queryable",
                        "readiness": {
                            "text_extracted": True,
                            "fts_ready": True,
                            "citation_ready": True,
                            "tool_accessible": True,
                        },
                    }
                    for source in sources
                ],
                "summary": {
                    "total": len(sources),
                    "queryable": len(sources),
                    "processing": 0,
                    "failed": 0,
                },
                "partial_errors": [],
            }

        async def generation_default(_context: Any) -> dict[str, Any]:
            return {
                "provider": "openai",
                "model": "shared-test-model",
                "ready": True,
                "reason_code": None,
            }

        async def history(
            context: Any,
            *,
            before: str | None,
            limit: int,
        ) -> tuple[Any, str | None]:
            store = self.recipient_dbs[context.recipient_user_id].shared_workspace_chat_store
            page = store.list_messages(share_id=self.share_id, before=before, limit=limit)
            thread = store.get_thread(share_id=self.share_id)
            return page, thread.conversation_id if thread is not None else None

        async def preview(
            context: Any,
            source: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, Any]:
            media_id = int(source["media_id"])
            media = self.media[media_id]
            return {
                "source_id": source["id"],
                "title": source["title"],
                "source_type": source["source_type"],
                "origin_url": "https://evidence.example.test",
                "origin_host": "evidence.example.test",
                "state": "queryable",
                "reason_code": "source_queryable",
                "content_available": True,
                "preview_mode": "content_excerpt",
                "unavailable_reason": None,
                "text_preview": media["content"],
                "text_total_chars": len(media["content"]),
                "text_truncated": False,
                "snippets": [],
                "generated_at": NOW.isoformat(),
            }

        monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", allow_rate_limit)
        monkeypatch.setattr(sharing, "_load_recipient_workspace_sources", load_sources)
        monkeypatch.setattr(sharing, "_project_recipient_source_status", project_sources)
        monkeypatch.setattr(sharing, "_resolve_recipient_generation_default", generation_default)
        monkeypatch.setattr(sharing, "_load_recipient_chat_history", history)
        monkeypatch.setattr(sharing, "_build_recipient_source_preview", preview)

        app = FastAPI()
        app.include_router(sharing.router, prefix="/api/v1")
        app.dependency_overrides[auth_deps.get_auth_principal] = get_principal
        app.dependency_overrides[sharing.get_request_user] = get_user
        app.dependency_overrides[sharing.get_shared_workspace_access_service] = (
            lambda: self.access_service
        )
        return TestClient(app, raise_server_exceptions=False)

    def body(
        self,
        request_id: str,
        *,
        mode: str = "all",
        source_ids: tuple[str, ...] = (),
        query: str = "What does the shared evidence support?",
    ) -> SharedWorkspaceChatRequest:
        return SharedWorkspaceChatRequest(
            request_id=UUID(request_id),
            query=query,
            source_scope={"mode": mode, "source_ids": list(source_ids)},
            provider="openai",
            model="shared-test-model",
        )

    async def chat(self, body: SharedWorkspaceChatRequest):
        async def store_loader(context: Any):
            return self.recipient_dbs[context.recipient_user_id].shared_workspace_chat_store

        @asynccontextmanager
        async def resource_loader(context: Any):
            assert context.owner_user_id == OWNER_ID
            yield SimpleNamespace(chat_service=self.chat_service)

        return await sharing._orchestrate_shared_workspace_chat(
            share_id=self.share_id,
            body=body,
            request=SimpleNamespace(),
            recipient_user_id=MEMBER_ID,
            access_service=self.access_service,
            store_loader=store_loader,
            resource_loader=resource_loader,
            rate_limiter=_AllowLimiter(),
            audit=None,
        )

    def saved_messages(self) -> tuple[Any, ...]:
        page = self.recipient_dbs[MEMBER_ID].shared_workspace_chat_store.list_messages(
            share_id=self.share_id,
            before=None,
            limit=100,
        )
        return page.messages


@pytest.fixture
async def recipient_security_harness(
    repo: Any,
    sharing_db: Any,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> RecipientSecurityHarness:
    sharing_db.execute(
        "INSERT OR IGNORE INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
        (NONMEMBER_ID, "nonmember", "nonmember@example.test", "hash"),
    )
    sharing_db.execute(
        "INSERT OR IGNORE INTO team_members (team_id, user_id, status) VALUES (?, ?, 'active')",
        (TEAM_ID, MEMBER_ID),
    )
    sharing_db.commit()
    share = await repo.create_share(
        workspace_id=WORKSPACE_ID,
        owner_user_id=OWNER_ID,
        share_scope_type="team",
        share_scope_id=TEAM_ID,
        access_level="view_chat",
        allow_clone=False,
        created_by=OWNER_ID,
    )

    owner_db = CharactersRAGDB(tmp_path / "owner.db", client_id=str(OWNER_ID))
    member_db = CharactersRAGDB(tmp_path / "member.db", client_id=str(MEMBER_ID))
    owner_recipient_db = CharactersRAGDB(
        tmp_path / "owner-recipient.db", client_id=str(OWNER_ID)
    )
    owner_db.upsert_workspace(WORKSPACE_ID, "Shared security review")
    for index in (1, 2):
        owner_db.add_workspace_source(
            WORKSPACE_ID,
            {
                "id": f"source-{index}",
                "media_id": 100 + index,
                "title": f"Shared source {index}",
                "source_type": "pdf",
                "position": index - 1,
                "selected": True,
            },
        )
    owner_db.add_note(OWNER_NOTE_SENTINEL, OWNER_NOTE_SENTINEL)
    owner_db.add_conversation(
        {
            "id": "owner-chat-sentinel",
            "root_id": "owner-chat-sentinel",
            "title": OWNER_CHAT_SENTINEL,
            "client_id": str(OWNER_ID),
        }
    )
    member_db.upsert_workspace("recipient-local", RECIPIENT_LOCAL_SENTINEL)

    media = {
        101: {
            "uuid": "media-uuid-101",
            "content_hash": "hash-101",
            "content": "First shared source evidence.",
        },
        102: {
            "uuid": "media-uuid-102",
            "content_hash": "hash-102",
            "content": "Second shared source evidence.",
        },
        999: {
            "uuid": "owner-unrelated-media",
            "content_hash": "owner-unrelated-hash",
            "content": OWNER_MEDIA_SENTINEL,
        },
    }
    harness = RecipientSecurityHarness(
        repo=repo,
        sharing_db=sharing_db,
        owner_db=owner_db,
        recipient_dbs={OWNER_ID: owner_recipient_db, MEMBER_ID: member_db},
        share_id=int(share["id"]),
        media=media,
        owner_loader_calls=[],
        chat_service=_DeterministicChatService(owner_db=owner_db, media=media),
    )
    monkeypatch.setattr(
        sharing,
        "resolve_chat_target",
        lambda **_kwargs: ResolvedChatTarget("openai", "shared-test-model"),
    )
    monkeypatch.setattr(
        sharing,
        "load_and_log_configs",
        lambda: {"openai_api": {"api_timeout": "30"}},
    )
    try:
        yield harness
    finally:
        owner_db.close_all_connections()
        member_db.close_all_connections()
        owner_recipient_db.close_all_connections()


def _assert_no_sentinels(payload: Any) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    assert OWNER_NOTE_SENTINEL not in serialized
    assert OWNER_CHAT_SENTINEL not in serialized
    assert OWNER_MEDIA_SENTINEL not in serialized
    assert RECIPIENT_LOCAL_SENTINEL not in serialized


def test_owner_and_member_reads_project_only_shared_sources_and_recipient_history(
    recipient_security_harness: RecipientSecurityHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = recipient_security_harness
    for user_id in (OWNER_ID, MEMBER_ID):
        client = harness.client(monkeypatch, user_id)
        bootstrap = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
        )
        sources = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/sources"
        )
        preview = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/sources/source-1/preview"
        )

        assert bootstrap.status_code == 200
        assert sources.status_code == 200
        assert preview.status_code == 200
        assert [item["source_id"] for item in sources.json()["items"]] == [
            "source-1",
            "source-2",
        ]
        assert preview.json()["text_preview"] == "First shared source evidence."
        assert bootstrap.json()["conversation"]["messages"] == []
        _assert_no_sentinels(bootstrap.json())
        _assert_no_sentinels(sources.json())
        _assert_no_sentinels(preview.json())

    missing = harness.client(monkeypatch, MEMBER_ID).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/sources/owner-unrelated-media/preview"
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "shared_workspace_not_found"


def test_nonmember_and_missing_share_are_neutral_equivalent_before_owner_data(
    recipient_security_harness: RecipientSecurityHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = recipient_security_harness
    nonmember = harness.client(monkeypatch, NONMEMBER_ID).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
    )
    missing = harness.client(monkeypatch, MEMBER_ID).get(
        "/api/v1/sharing/shared-with-me/999999/workspace"
    )
    assert harness.owner_loader_calls == []

    assert nonmember.status_code == missing.status_code == 404
    assert nonmember.json() == missing.json() == {
        "detail": {
            "code": "shared_workspace_not_found",
            "message": "Shared workspace not found.",
            "retryable": False,
        }
    }

@pytest.mark.asyncio
async def test_revoked_share_matches_neutral_denial(
    recipient_security_harness: RecipientSecurityHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = recipient_security_harness
    await harness.repo.revoke_share(harness.share_id)
    revoked = harness.client(monkeypatch, MEMBER_ID).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
    )
    assert revoked.status_code == 404
    assert revoked.json()["detail"] == {
        "code": "shared_workspace_not_found",
        "message": "Shared workspace not found.",
        "retryable": False,
    }
    assert harness.owner_loader_calls == []


def test_sharing_read_permission_is_required_before_share_resolution(
    recipient_security_harness: RecipientSecurityHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = recipient_security_harness
    response = harness.client(monkeypatch, MEMBER_ID, can_read=False).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "sharing_permission_required"
    assert harness.owner_loader_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "source_ids", "expected_sources"),
    [
        ("all", (), {"source-1", "source-2"}),
        ("include", ("source-2",), {"source-2"}),
    ],
)
async def test_all_and_subset_chat_freeze_only_workspace_sources_and_replay(
    recipient_security_harness: RecipientSecurityHarness,
    mode: str,
    source_ids: tuple[str, ...],
    expected_sources: set[str],
) -> None:
    harness = recipient_security_harness
    body = harness.body(
        "de305d54-75b4-431b-adb2-eb6b9e546014",
        mode=mode,
        source_ids=source_ids,
    )

    first = await harness.chat(body)
    replay = await harness.chat(body)

    assert {citation.source_id for citation in first.citations} == expected_sources
    assert first.replay.replayed is False
    assert replay.replay.replayed is True
    assert harness.chat_service.generation_calls == 1
    assert len(harness.saved_messages()) == 2
    _assert_no_sentinels(first.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_matching_concurrent_claim_and_mismatched_fingerprint_fail_without_messages(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    body = harness.body("de305d54-75b4-431b-adb2-eb6b9e546015")
    store = harness.recipient_dbs[MEMBER_ID].shared_workspace_chat_store
    context = await harness.access_service.resolve(
        share_id=harness.share_id,
        recipient_user_id=MEMBER_ID,
    )
    thread = store.get_or_create_thread(
        share_id=harness.share_id,
        owner_user_id=str(context.owner_user_id),
        workspace_id=context.workspace_id,
        workspace_name=context.workspace["name"],
    )
    claimed = store.claim_request(
        share_id=harness.share_id,
        request_id=body.request_id,
        request_fingerprint=sharing._shared_chat_fingerprint(body),
        conversation_id=thread.conversation_id,
        lease_seconds=300,
        now=NOW,
    )
    assert claimed.disposition == "claimed"

    with pytest.raises(HTTPException) as matching_info:
        await harness.chat(body)
    assert matching_info.value.status_code == 409
    assert matching_info.value.detail["code"] == "request_in_progress"

    mismatched = body.model_copy(update={"query": "A different request body"})
    with pytest.raises(HTTPException) as mismatch_info:
        await harness.chat(mismatched)
    assert mismatch_info.value.status_code == 409
    assert mismatch_info.value.detail["code"] == "request_id_conflict"
    assert harness.saved_messages() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "mutation"),
    [
        ("after_retrieval", "revoke"),
        ("after_retrieval", "membership"),
        ("after_retrieval", "source"),
        ("after_retrieval", "media"),
        ("after_generation", "revoke"),
        ("after_generation", "membership"),
        ("after_generation", "source"),
        ("after_generation", "media"),
    ],
)
async def test_authority_and_frozen_scope_changes_fail_before_persistence(
    recipient_security_harness: RecipientSecurityHarness,
    boundary: str,
    mutation: str,
) -> None:
    harness = recipient_security_harness

    async def mutate() -> None:
        if mutation == "revoke":
            await harness.repo.revoke_share(harness.share_id)
        elif mutation == "membership":
            harness.sharing_db.execute(
                "UPDATE team_members SET status = 'suspended' WHERE team_id = ? AND user_id = ?",
                (TEAM_ID, MEMBER_ID),
            )
            harness.sharing_db.commit()
        elif mutation == "source":
            harness.owner_db.delete_workspace_source(WORKSPACE_ID, "source-2")
        else:
            harness.media[102]["content_hash"] = "changed-content-hash"

    setattr(harness.chat_service, boundary, mutate)
    body = harness.body("de305d54-75b4-431b-adb2-eb6b9e546016")

    with pytest.raises(HTTPException) as exc_info:
        await harness.chat(body)

    assert exc_info.value.status_code in {404, 409}
    assert exc_info.value.detail["code"] in {
        "shared_workspace_not_found",
        "shared_source_changed",
    }
    assert harness.saved_messages() == ()
