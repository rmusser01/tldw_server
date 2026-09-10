"""Integrated recipient security matrix over production shared-workspace routes."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import (
    ChaCha_Notes_DB_Deps,
    DB_Deps,
    auth_deps,
    jobs_deps,
)
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.repos import users_repo
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    dumps_envelope,
    encrypt_byok_payload,
)
from tldw_Server_API.app.core.Chat import chat_target_resolution
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.legacy_state import (
    mark_media_as_processed,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Sharing import shared_workspace_chat_service as shared_chat

pytestmark = pytest.mark.integration

OWNER_ID = 1
MEMBER_ID = 2
NONMEMBER_ID = 3
TEAM_ID = 10
ORG_ID = 20
WORKSPACE_ID = "shared-security-workspace"
OWNER_NOTE_SENTINEL = "OWNER-NOTE-SENTINEL-DO-NOT-LEAK"
OWNER_CHAT_SENTINEL = "OWNER-CHAT-SENTINEL-DO-NOT-LEAK"
OWNER_MEDIA_SENTINEL = "OWNER-UNRELATED-MEDIA-SENTINEL-DO-NOT-LEAK"
RECIPIENT_LOCAL_SENTINEL = "RECIPIENT-LOCAL-WORKSPACE-SENTINEL-DO-NOT-LEAK"

Hook = Callable[[], Awaitable[None] | None]


class _UsersRepo:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        if user_id == OWNER_ID:
            return {"id": OWNER_ID, "username": "Research owner"}
        return None


def _credential_row(api_key: str, project_id: str) -> dict[str, Any]:
    encrypted = encrypt_byok_payload(
        build_secret_payload(api_key, {"project_id": project_id})
    )
    return {
        "encrypted_blob": dumps_envelope(encrypted),
        "revoked_at": None,
        "last_used_at": None,
    }


class _UserCredentialRepo(AuthnzUserProviderSecretsRepo):
    def __init__(self, rows: dict[int, dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int, str]] = []
        self.touches: list[tuple[int, str]] = []

    async def fetch_secret_for_active_user(
        self,
        user_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        assert include_revoked is True
        self.calls.append(("active", user_id, provider))
        return self.rows.get(user_id)

    async def fetch_secret_for_user(
        self,
        user_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        assert include_revoked is True
        self.calls.append(("unrestricted", user_id, provider))
        return self.rows.get(user_id)

    async def touch_last_used(
        self,
        user_id: int,
        provider: str,
        _last_used_at: datetime,
    ) -> None:
        self.touches.append((user_id, provider))


class _SharedCredentialRepo(AuthnzOrgProviderSecretsRepo):
    def __init__(self, rows: dict[tuple[str, int], dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int, str]] = []
        self.touches: list[tuple[str, int, str]] = []

    async def fetch_secret(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
        *,
        include_revoked: bool = False,
    ) -> dict[str, Any] | None:
        assert include_revoked is True
        self.calls.append((scope_type, scope_id, provider))
        return self.rows.get((scope_type, scope_id))

    async def touch_last_used(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
        _last_used_at: datetime,
    ) -> None:
        self.touches.append((scope_type, scope_id, provider))


@dataclass
class _ExternalTransports:
    media_db: MediaDatabase
    retrieval_calls: list[dict[str, Any]] = field(default_factory=list)
    generation_calls: int = 0
    generation_requests: list[dict[str, Any]] = field(default_factory=list)
    after_retrieval: Hook | None = None
    after_generation: Hook | None = None
    generation_entered: asyncio.Event | None = None
    release_generation: asyncio.Event | None = None

    async def _run_hook(self, hook: Hook | None) -> None:
        if hook is None:
            return
        result = hook()
        if result is not None:
            await result

    async def retrieve(self, *, query: str, **kwargs: Any) -> dict[str, Any]:
        self.retrieval_calls.append({"query": query, **kwargs})
        documents: list[dict[str, Any]] = []
        for index, media_id in enumerate(kwargs["include_media_ids"], start=1):
            media = self.media_db.get_media_by_id(media_id)
            assert media is not None
            content = str(media["content"])
            documents.append(
                {
                    "id": f"chunk-{media_id}",
                    "content": content,
                    "source": "media_db",
                    "score": 0.9,
                    "metadata": {
                        "source": "media_db",
                        "media_id": media_id,
                        "chunk_id": f"chunk-{media_id}",
                        "chunk_index": index,
                        "start_char": 0,
                        "end_char": len(content),
                    },
                }
            )
        await self._run_hook(self.after_retrieval)
        return {
            "documents": documents,
            "query": query,
            "expanded_queries": [],
            "errors": [],
            "generated_answer": None,
            "cache_hit": False,
            "metadata": {
                "original_query": query,
                "retrieval_cache_hit": False,
                "generation_executed": False,
                "explicit_source_selection": {
                    "enabled": True,
                    "requested_sources": ["media_db"],
                    "resolved_sources": ["media_db"],
                    "include_media_ids_count": len(kwargs["include_media_ids"]),
                    "include_note_ids_count": 0,
                    "scope_intersection_empty": False,
                    "cache_disabled": False,
                },
                "sources_requested": ["media_db"],
                "sources_searched": ["media_db"],
                "documents_retrieved": len(documents),
                "retrieval_plan": {
                    "query": query,
                    "sources": ["media_db"],
                    "search_mode": "fts",
                    "top_k": 20,
                    "index_namespace": f"user_{OWNER_ID}_media_embeddings",
                },
            },
        }

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.generation_calls += 1
        self.generation_requests.append(dict(kwargs))
        if self.generation_entered is not None:
            self.generation_entered.set()
        if self.release_generation is not None:
            await self.release_generation.wait()
        await self._run_hook(self.after_generation)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "Grounded recipient answer",
                                "citations": ["E1", "E2"],
                            }
                        )
                    }
                }
            ]
        }


@dataclass
class RecipientSecurityHarness:
    repo: Any
    sharing_db: Any
    owner_db: CharactersRAGDB
    recipient_dbs: dict[int, CharactersRAGDB]
    media_db: MediaDatabase
    media_ids: dict[str, int]
    share_id: int
    transports: _ExternalTransports
    owner_db_calls: list[int]
    recipient_db_calls: list[int]
    media_db_calls: list[int]
    user_credential_repo: _UserCredentialRepo
    shared_credential_repo: _SharedCredentialRepo
    clock_now: datetime

    def app(self, user_id: int, *, can_read: bool = True) -> FastAPI:
        async def get_principal():
            from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

            return AuthPrincipal(
                kind="user",
                user_id=user_id,
                username=f"user-{user_id}",
                permissions=["sharing.read"] if can_read else [],
            )

        async def get_user():
            from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

            return User(
                id=user_id,
                username=f"user-{user_id}",
                email=f"user-{user_id}@example.test",
                password_hash="hash",
            )

        app = FastAPI()
        app.include_router(sharing.router, prefix="/api/v1")
        app.dependency_overrides[auth_deps.get_auth_principal] = get_principal
        app.dependency_overrides[sharing.get_request_user] = get_user
        return app

    def client(self, user_id: int, *, can_read: bool = True) -> TestClient:
        return TestClient(
            self.app(user_id, can_read=can_read),
            raise_server_exceptions=False,
        )

    def chat_body(
        self,
        request_id: str,
        *,
        mode: str = "all",
        source_ids: tuple[str, ...] = (),
        query: str = "What does the shared evidence support?",
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "query": query,
            "source_scope": {"mode": mode, "source_ids": list(source_ids)},
            "provider": "openai",
            "model": "shared-test-model",
        }

    async def post_chat(self, body: dict[str, Any]) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app(MEMBER_ID))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                f"/api/v1/sharing/shared-with-me/{self.share_id}/chat",
                json=body,
            )

    def saved_messages(self, recipient_id: int = MEMBER_ID) -> tuple[Any, ...]:
        page = self.recipient_dbs[recipient_id].shared_workspace_chat_store.list_messages(
            share_id=self.share_id,
            before=None,
            limit=100,
        )
        return page.messages

    def receipt_rows(self) -> list[dict[str, Any]]:
        with self.recipient_dbs[MEMBER_ID].transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT request_id, status, lease_epoch, lease_expires_at,
                           user_message_id, assistant_message_id, completed_at
                      FROM shared_workspace_chat_requests
                     WHERE recipient_user_id = ? AND share_id = ?
                    """,
                    (str(MEMBER_ID), self.share_id),
                ).fetchall()
            ]

    def mutate_media_hash(self, source_id: str) -> None:
        media_id = self.media_ids[source_id]
        with self.media_db.transaction() as connection:
            connection.execute(
                """
                UPDATE Media
                   SET content_hash = ?, version = version + 1,
                       last_modified = ?, client_id = ?
                 WHERE id = ?
                """,
                (
                    f"changed-{media_id}",
                    self.media_db._get_current_utc_timestamp_str(),
                    self.media_db.client_id,
                    media_id,
                ),
            )


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
        tmp_path / "owner-recipient.db",
        client_id=str(OWNER_ID),
    )
    media_db = MediaDatabase(
        db_path=str(tmp_path / "owner-media.db"),
        client_id=str(OWNER_ID),
    )
    media_db.initialize_db()
    media_ids: dict[str, int] = {}
    for source_id, title, content in (
        ("source-1", "Shared source 1", "First shared source evidence."),
        ("source-2", "Shared source 2", "Second shared source evidence."),
        ("owner-sentinel", "Owner sentinel", OWNER_MEDIA_SENTINEL),
    ):
        media_id, _media_uuid, _status = media_db.add_media_with_keywords(
            url=f"https://evidence.example.test/{source_id}",
            title=title,
            media_type="pdf",
            content=content,
            keywords=None,
        )
        assert media_id is not None
        mark_media_as_processed(media_db, media_id)
        media_ids[source_id] = media_id

    owner_db.upsert_workspace(WORKSPACE_ID, "Shared security review")
    for index in (1, 2):
        owner_db.add_workspace_source(
            WORKSPACE_ID,
            {
                "id": f"source-{index}",
                "media_id": media_ids[f"source-{index}"],
                "title": f"Shared source {index}",
                "source_type": "pdf",
                "url": f"https://evidence.example.test/source-{index}",
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

    owner_db_calls: list[int] = []
    recipient_db_calls: list[int] = []
    media_db_calls: list[int] = []
    recipient_dbs = {OWNER_ID: owner_recipient_db, MEMBER_ID: member_db}

    async def get_owner_db(owner_user_id: int) -> CharactersRAGDB:
        owner_db_calls.append(owner_user_id)
        assert owner_user_id == OWNER_ID
        return owner_db

    async def get_recipient_db(
        recipient_user_id: int,
        client_id: str | None = None,
    ) -> CharactersRAGDB:
        del client_id
        recipient_db_calls.append(recipient_user_id)
        return recipient_dbs[recipient_user_id]

    @contextmanager
    def get_owner_media(owner_user_id: int):
        media_db_calls.append(owner_user_id)
        assert owner_user_id == OWNER_ID
        yield media_db

    transports = _ExternalTransports(media_db=media_db)

    def build_chat_service(**kwargs: Any) -> shared_chat.SharedWorkspaceChatService:
        return shared_chat.SharedWorkspaceChatService(
            **kwargs,
            rag_pipeline=transports.retrieve,
        )

    async def allow_rate_limit(*_args: Any, **_kwargs: Any) -> None:
        return None

    clock_now = datetime.now(timezone.utc)

    class _ControlledDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            if tz is None:
                return clock_now.replace(tzinfo=None)
            return clock_now.astimezone(tz)

    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "shared-test-model")
    monkeypatch.setenv(
        "BYOK_ENCRYPTION_KEY",
        base64.b64encode(b"r" * 32).decode("ascii"),
    )
    reset_settings()
    user_credential_repo = _UserCredentialRepo(
        {
            OWNER_ID: _credential_row("owner-private-key", "owner-private-project"),
        }
    )
    shared_credential_repo = _SharedCredentialRepo(
        {
            ("team", TEAM_ID): _credential_row(
                "recipient-team-key",
                "recipient-team-project",
            ),
            ("org", ORG_ID): _credential_row(
                "recipient-org-key",
                "recipient-org-project",
            ),
        }
    )

    async def get_user_credential_repo() -> _UserCredentialRepo:
        return user_credential_repo

    async def get_shared_credential_repo() -> _SharedCredentialRepo:
        return shared_credential_repo

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_user_credential_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", get_shared_credential_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"model": "shared-test-model"}},
    )
    monkeypatch.setattr(
        chat_target_resolution,
        "get_override_default_model",
        lambda _provider: None,
    )
    monkeypatch.setattr(
        chat_target_resolution,
        "get_default_model_for_provider",
        lambda _provider: "shared-test-model",
    )
    monkeypatch.setattr(
        chat_target_resolution,
        "get_llm_provider_override",
        lambda _provider: None,
    )
    monkeypatch.setattr(
        chat_target_resolution,
        "validate_provider_override",
        lambda _provider, _model: None,
    )
    monkeypatch.setattr(sharing, "_get_repo", lambda: repo)
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: None)
    monkeypatch.setattr(users_repo, "AuthnzUsersRepo", _UsersRepo)
    monkeypatch.setattr(sharing, "SharedWorkspaceChatService", build_chat_service)
    monkeypatch.setattr(shared_chat, "perform_chat_api_call_async", transports.generate)
    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", allow_rate_limit)
    monkeypatch.setattr(ChaCha_Notes_DB_Deps, "get_chacha_db_for_owner", get_owner_db)
    monkeypatch.setattr(
        ChaCha_Notes_DB_Deps,
        "get_chacha_db_for_user_id",
        get_recipient_db,
    )
    monkeypatch.setattr(DB_Deps, "managed_media_db_for_owner", get_owner_media)
    monkeypatch.setattr(
        DB_Deps,
        "get_media_db_path_for_rag",
        lambda db: str(db.db_path_str),
    )
    monkeypatch.setattr(jobs_deps, "try_get_job_manager", lambda: None)
    monkeypatch.setattr(sharing, "datetime", _ControlledDateTime)

    harness = RecipientSecurityHarness(
        repo=repo,
        sharing_db=sharing_db,
        owner_db=owner_db,
        recipient_dbs=recipient_dbs,
        media_db=media_db,
        media_ids=media_ids,
        share_id=int(share["id"]),
        transports=transports,
        owner_db_calls=owner_db_calls,
        recipient_db_calls=recipient_db_calls,
        media_db_calls=media_db_calls,
        user_credential_repo=user_credential_repo,
        shared_credential_repo=shared_credential_repo,
        clock_now=clock_now,
    )
    try:
        yield harness
    finally:
        owner_db.close_all_connections()
        member_db.close_all_connections()
        owner_recipient_db.close_all_connections()
        media_db.close_connection()
        reset_settings()


def _assert_no_sentinels(payload: Any) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    assert OWNER_NOTE_SENTINEL not in serialized
    assert OWNER_CHAT_SENTINEL not in serialized
    assert OWNER_MEDIA_SENTINEL not in serialized
    assert RECIPIENT_LOCAL_SENTINEL not in serialized


def test_matrix_keeps_production_credential_resolver_active(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    assert recipient_security_harness.share_id > 0
    assert sharing.resolve_byok_credentials is byok_runtime.resolve_byok_credentials
    assert shared_chat.resolve_byok_credentials is byok_runtime.resolve_byok_credentials


def test_owner_and_member_reads_use_production_db_selection_and_projections(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    for user_id in (OWNER_ID, MEMBER_ID):
        client = harness.client(user_id)
        bootstrap = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
        )
        sources = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/sources"
        )
        preview = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/sources/source-1/preview"
        )
        history = client.get(
            f"/api/v1/sharing/shared-with-me/{harness.share_id}/chat/messages"
        )

        assert bootstrap.status_code == sources.status_code == preview.status_code == 200
        assert history.status_code == 200
        assert [item["source_id"] for item in sources.json()["items"]] == [
            "source-1",
            "source-2",
        ]
        assert preview.json()["text_preview"] == "First shared source evidence."
        assert bootstrap.json()["generation_default"] == {
            "provider": "openai",
            "model": "shared-test-model",
            "ready": True,
            "reason_code": None,
        }
        assert bootstrap.json()["conversation"]["messages"] == []
        assert history.json()["messages"] == []
        _assert_no_sentinels(bootstrap.json())
        _assert_no_sentinels(sources.json())
        _assert_no_sentinels(preview.json())

    assert harness.owner_db_calls
    assert set(harness.owner_db_calls) == {OWNER_ID}
    assert harness.media_db_calls
    assert set(harness.media_db_calls) == {OWNER_ID}
    assert set(harness.recipient_db_calls) == {OWNER_ID, MEMBER_ID}
    assert ("active", OWNER_ID, "openai") in harness.user_credential_repo.calls
    assert ("active", MEMBER_ID, "openai") in harness.user_credential_repo.calls
    assert ("unrestricted", MEMBER_ID, "openai") in harness.user_credential_repo.calls
    assert ("team", TEAM_ID, "openai") in harness.shared_credential_repo.calls

    missing = harness.client(MEMBER_ID).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/sources/owner-sentinel/preview"
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "shared_workspace_not_found"


@pytest.mark.asyncio
async def test_production_credential_resolver_preserves_recipient_precedence_and_scope(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    harness.user_credential_repo.calls.clear()
    harness.shared_credential_repo.calls.clear()
    harness.user_credential_repo.rows[MEMBER_ID] = _credential_row(
        "recipient-user-key",
        "recipient-user-project",
    )

    user_credentials = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=MEMBER_ID,
        team_ids=[TEAM_ID],
        org_ids=[ORG_ID],
        trusted_base_url_override=False,
    )
    del harness.user_credential_repo.rows[MEMBER_ID]
    team_credentials = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=MEMBER_ID,
        team_ids=[TEAM_ID],
        org_ids=[ORG_ID],
        trusted_base_url_override=False,
    )
    del harness.shared_credential_repo.rows[("team", TEAM_ID)]
    org_credentials = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=MEMBER_ID,
        team_ids=[TEAM_ID],
        org_ids=[ORG_ID],
        trusted_base_url_override=False,
    )

    assert [
        (credentials.source, credentials.api_key)
        for credentials in (user_credentials, team_credentials, org_credentials)
    ] == [
        ("user", "recipient-user-key"),
        ("team", "recipient-team-key"),
        ("org", "recipient-org-key"),
    ]
    assert {
        credentials.app_config["openai_api"]["project_id"]
        for credentials in (user_credentials, team_credentials, org_credentials)
    } == {
        "recipient-user-project",
        "recipient-team-project",
        "recipient-org-project",
    }
    assert {
        user_id for _lookup, user_id, _provider in harness.user_credential_repo.calls
    } == {MEMBER_ID}
    assert harness.shared_credential_repo.calls == [
        ("team", TEAM_ID, "openai"),
        ("team", TEAM_ID, "openai"),
        ("org", ORG_ID, "openai"),
    ]


def test_nonmember_and_missing_share_are_neutral_before_owner_data(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    harness.owner_db_calls.clear()
    harness.media_db_calls.clear()
    nonmember = harness.client(NONMEMBER_ID).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
    )
    missing = harness.client(MEMBER_ID).get(
        "/api/v1/sharing/shared-with-me/999999/workspace"
    )

    assert nonmember.status_code == missing.status_code == 404
    assert nonmember.json() == missing.json() == {
        "detail": {
            "code": "shared_workspace_not_found",
            "message": "Shared workspace not found.",
            "retryable": False,
        }
    }
    assert harness.owner_db_calls == []
    assert harness.media_db_calls == []


@pytest.mark.asyncio
async def test_revoked_share_matches_neutral_denial(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    await harness.repo.revoke_share(harness.share_id)
    harness.owner_db_calls.clear()
    response = harness.client(MEMBER_ID).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == {
        "code": "shared_workspace_not_found",
        "message": "Shared workspace not found.",
        "retryable": False,
    }
    assert harness.owner_db_calls == []


def test_sharing_read_is_required_before_share_resolution(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    harness.owner_db_calls.clear()
    response = harness.client(MEMBER_ID, can_read=False).get(
        f"/api/v1/sharing/shared-with-me/{harness.share_id}/workspace"
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "sharing_permission_required"
    assert harness.owner_db_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "source_ids", "expected_sources"),
    [
        ("all", (), {"source-1", "source-2"}),
        ("include", ("source-2",), {"source-2"}),
    ],
)
async def test_chat_uses_production_snapshot_credentials_persistence_and_replay(
    recipient_security_harness: RecipientSecurityHarness,
    mode: str,
    source_ids: tuple[str, ...],
    expected_sources: set[str],
) -> None:
    harness = recipient_security_harness
    harness.user_credential_repo.calls.clear()
    harness.shared_credential_repo.calls.clear()
    body = harness.chat_body(
        "de305d54-75b4-431b-adb2-eb6b9e546014",
        mode=mode,
        source_ids=source_ids,
    )

    first = await harness.post_chat(body)
    replay = await harness.post_chat(body)
    conflict = await harness.post_chat(
        {**body, "query": "A different request body"}
    )

    assert first.status_code == replay.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "request_id_conflict"
    assert {
        citation["source_id"] for citation in first.json()["citations"]
    } == expected_sources
    assert first.json()["replay"]["replayed"] is False
    assert replay.json()["replay"]["replayed"] is True
    assert harness.transports.generation_calls == 1
    assert len(harness.saved_messages()) == 2
    assert len(harness.receipt_rows()) == 1
    assert harness.receipt_rows()[0]["status"] == "completed"
    retrieval = harness.transports.retrieval_calls[0]
    assert set(retrieval["include_media_ids"]) == {
        harness.media_ids[source_id] for source_id in expected_sources
    }
    assert harness.media_ids["owner-sentinel"] not in retrieval["include_media_ids"]
    assert harness.user_credential_repo.calls == [
        ("active", MEMBER_ID, "openai"),
        ("unrestricted", MEMBER_ID, "openai"),
    ]
    assert harness.shared_credential_repo.calls == [("team", TEAM_ID, "openai")]
    generation_request = harness.transports.generation_requests[-1]
    assert generation_request["api_key"] == "recipient-team-key"
    assert generation_request["app_config"]["openai_api"]["project_id"] == (
        "recipient-team-project"
    )
    assert generation_request["user_identifier"] == str(MEMBER_ID)
    _assert_no_sentinels(first.json())


@pytest.mark.asyncio
async def test_matching_requests_overlap_with_one_winner_and_durable_replay(
    recipient_security_harness: RecipientSecurityHarness,
) -> None:
    harness = recipient_security_harness
    body = harness.chat_body("de305d54-75b4-431b-adb2-eb6b9e546015")
    entered = asyncio.Event()
    release = asyncio.Event()
    harness.transports.generation_entered = entered
    harness.transports.release_generation = release

    winner_task = asyncio.create_task(harness.post_chat(body))
    await asyncio.wait_for(entered.wait(), timeout=5)
    active_receipt = harness.receipt_rows()[0]
    lease_expires_at = datetime.fromisoformat(active_receipt["lease_expires_at"])
    assert active_receipt["status"] == "in_progress"
    assert active_receipt["lease_epoch"] == 1
    assert harness.clock_now + timedelta(minutes=5) <= lease_expires_at
    assert lease_expires_at <= harness.clock_now + timedelta(minutes=30)
    overlapping_task = asyncio.create_task(harness.post_chat(body))
    overlapping = await asyncio.wait_for(overlapping_task, timeout=5)
    release.set()
    winner = await asyncio.wait_for(winner_task, timeout=5)
    replay = await harness.post_chat(body)

    assert winner.status_code == 200
    assert winner.json()["replay"]["replayed"] is False
    assert overlapping.status_code == 409
    assert overlapping.json()["detail"]["code"] == "request_in_progress"
    assert replay.status_code == 200
    assert replay.json()["replay"]["replayed"] is True
    assert replay.json()["turn"] == winner.json()["turn"]
    assert harness.transports.generation_calls == 1
    assert len(harness.saved_messages()) == 2
    receipts = harness.receipt_rows()
    assert len(receipts) == 1
    assert receipts[0]["status"] == "completed"
    assert receipts[0]["user_message_id"] is not None
    assert receipts[0]["assistant_message_id"] is not None
    assert receipts[0]["completed_at"] is not None


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
            harness.mutate_media_hash("source-2")

    setattr(harness.transports, boundary, mutate)
    response = await harness.post_chat(
        harness.chat_body("de305d54-75b4-431b-adb2-eb6b9e546016")
    )

    assert response.status_code in {404, 409}
    assert response.json()["detail"]["code"] in {
        "shared_workspace_not_found",
        "shared_source_changed",
    }
    assert harness.saved_messages() == ()
    assert harness.receipt_rows()[0]["status"] in {"retryable", "conflicted"}
    assert harness.transports.generation_calls == (
        0 if boundary == "after_retrieval" else 1
    )
