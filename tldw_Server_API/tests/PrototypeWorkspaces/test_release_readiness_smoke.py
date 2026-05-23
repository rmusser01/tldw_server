"""Risk Gate 8 smoke coverage for prototype workspace release readiness."""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from tldw_Server_API.app.core.AuthNZ.migrations import (
    migration_001_create_users_table,
    migration_077_create_sharing_tables,
    migration_086_create_prototype_workspace_tables,
    migration_087_expand_share_tokens_resource_type_for_prototypes,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = pytest.mark.integration

_T = TypeVar("_T")


def _dict_row(cursor: sqlite3.Cursor, row: Any) -> dict[str, Any] | None:
    """Convert a SQLite cursor row to a dict when the statement returned columns."""
    if row is None or cursor.description is None:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row, strict=True))


class _AsyncCursor:
    """Async cursor wrapper that returns dict rows for repository transaction paths."""

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> dict[str, Any] | None:
        row = self._cursor.fetchone()
        return _dict_row(self._cursor, row)


class _FakePool:
    """Minimal DatabasePool stand-in backed by one in-memory SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return _dict_row(cur, row)

    async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        cur = self._conn.execute(sql, params)
        if cur.description is None:
            return []
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in rows]

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        """Yield a transaction adapter that defers commit until the context exits."""

        class _TxConn:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self._conn = conn

            async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _AsyncCursor:
                return _AsyncCursor(self._conn.execute(sql, params))

            async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
                cur = self._conn.execute(sql, params)
                row = cur.fetchone()
                return _dict_row(cur, row)

            async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
                cur = self._conn.execute(sql, params)
                if cur.description is None:
                    return []
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in rows]

        self._conn.execute("BEGIN")
        try:
            yield _TxConn(self._conn)
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()


class _NoOpAuditService:
    """Audit service stub for public share exchange smoke tests."""

    async def log(self, *args: Any, **kwargs: Any) -> None:
        return None


class _MutablePublishValidator:
    """Publish validator stub that can switch between failure and success."""

    def __init__(self, *, ok: bool) -> None:
        self.ok = ok

    async def validate_publish_candidate(self, **kwargs: Any) -> dict[str, Any]:
        if not self.ok:
            return {"ok": False, "reason": "gate8 synthetic publish validation failure"}
        return {
            "ok": True,
            "runtime_target_url": f"runtime://gate8/{kwargs['candidate']['snapshot_id']}",
        }


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run async repo/service calls from synchronous TestClient smoke tests."""
    return asyncio.run(coro)


def _assert_prototype_error(response: Response, *, category: str, frontend_state: str) -> None:
    """Assert the stable prototype error contract without parsing messages."""
    detail = response.json()["detail"]
    assert detail["category"] == category
    assert detail["frontend_state"] == frontend_state
    assert detail["retryable"] is False
    assert isinstance(detail["message"], str)
    assert detail["message"]


def _build_release_smoke_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jobs_db_path: Path,
    publish_validation_ok: bool,
) -> SimpleNamespace:
    """Build a combined sharing/prototype FastAPI app with in-memory dependencies."""
    from tldw_Server_API.app.api.v1.endpoints import prototype_workspaces as prototype_endpoints
    from tldw_Server_API.app.api.v1.endpoints import sharing as sharing_endpoints
    from tldw_Server_API.app.api.v1.endpoints.prototype_workspaces import router as prototype_router
    from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Prototype_Workspaces.access import PrototypeAccessService
    from tldw_Server_API.app.core.Prototype_Workspaces.jobs import PrototypeWorkspaceJobs
    from tldw_Server_API.app.core.Prototype_Workspaces.preview_broker import PrototypePreviewBroker
    from tldw_Server_API.app.core.Prototype_Workspaces.service import PrototypeWorkspaceService
    from tldw_Server_API.app.core.Sharing.share_token_service import ShareTokenService

    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migration_001_create_users_table(conn)
    migration_077_create_sharing_tables(conn)
    migration_087_expand_share_tokens_resource_type_for_prototypes(conn)
    migration_086_create_prototype_workspace_tables(conn)
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (1, 'owner', 'owner@test.com', 'hash')"
    )
    conn.commit()

    pool = _FakePool(conn)
    prototype_repo = PrototypeWorkspacesRepo(db_pool=pool)
    sharing_repo = SharedWorkspaceRepo(db_pool=pool)
    token_service = ShareTokenService(sharing_repo)
    access_service = PrototypeAccessService(prototype_repo, signing_secret=secrets.token_urlsafe(32))
    preview_broker = PrototypePreviewBroker(
        repo=prototype_repo,
        base_preview_path="/api/v1/prototype-previews",
        signing_secret=secrets.token_urlsafe(32),
    )
    validator = _MutablePublishValidator(ok=publish_validation_ok)
    service = PrototypeWorkspaceService(
        repo=prototype_repo,
        preview_broker=preview_broker,
        publish_validator=validator,
    )
    jobs_service = PrototypeWorkspaceJobs(
        repo=prototype_repo,
        jobs_manager=JobManager(db_path=str(jobs_db_path)),
    )

    PrototypePreviewBroker._records.clear()
    PrototypePreviewBroker._active_scope_handles.clear()
    monkeypatch.setattr(prototype_endpoints, "_get_repo", lambda: prototype_repo)
    monkeypatch.setattr(prototype_endpoints, "_get_service", lambda: service)
    monkeypatch.setattr(prototype_endpoints, "_get_jobs_service", lambda: jobs_service)
    monkeypatch.setattr(prototype_endpoints, "_get_access_service", lambda: access_service)
    monkeypatch.setattr(prototype_endpoints, "_get_preview_broker", lambda: preview_broker)

    monkeypatch.setattr(sharing_endpoints, "_get_repo", lambda: sharing_repo)
    monkeypatch.setattr(sharing_endpoints, "_get_token_service", lambda: token_service)
    monkeypatch.setattr(sharing_endpoints, "_get_prototype_repo", lambda: prototype_repo)
    monkeypatch.setattr(sharing_endpoints, "_get_prototype_access_service", lambda: access_service)
    monkeypatch.setattr(sharing_endpoints, "_get_audit_service", lambda: _NoOpAuditService())
    monkeypatch.setattr(sharing_endpoints, "_check_public_rate_limit", lambda _request: None)

    app = FastAPI()
    app.include_router(sharing_router, prefix="/api/v1")
    app.include_router(prototype_router, prefix="/api/v1")

    async def _owner_user() -> User:
        return User(
            id=1,
            username="owner",
            email="owner@test.com",
            roles=["admin"],
            permissions=["prototype.promote"],
        )

    app.dependency_overrides[get_request_user] = _owner_user

    def _set_publish_validation(ok: bool) -> None:
        validator.ok = ok

    return SimpleNamespace(
        client=TestClient(app),
        conn=conn,
        repo=prototype_repo,
        service=service,
        token_service=token_service,
        set_publish_validation=_set_publish_validation,
    )


def test_owner_collaborator_promotion_smoke_covers_failure_and_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the minimum Gate 8 owner-collaborator-promotion path."""
    harness = _build_release_smoke_app(
        monkeypatch,
        jobs_db_path=tmp_path / "prototype_jobs.db",
        publish_validation_ok=False,
    )
    client = harness.client

    created = client.post(
        "/api/v1/prototype-workspaces",
        json={
            "title": "Gate 8 smoke prototype",
            "creation_source": "prompt",
            "prompt": "Build a smoke-test dashboard",
        },
    )
    assert created.status_code == 201
    workspace_id = created.json()["id"]
    original_canonical = created.json()["canonical_snapshot_id"]

    share = _run(
        harness.token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=workspace_id,
            owner_user_id=1,
        )
    )
    exchange = client.post(
        f"/api/v1/sharing/public/{share['raw_token']}/prototype-session",
        json={"display_name": "Gate 8 collaborator"},
    )
    assert exchange.status_code == 200
    session_token = exchange.json()["session_token"]
    shared_actor_id = exchange.json()["shared_actor_id"]

    branch = client.post(
        "/api/v1/prototype-sessions",
        json={"session_token": session_token, "request_nonce": "req_gate8_branch"},
    )
    assert branch.status_code == 202
    branch_body = branch.json()
    assert branch_body["prototype_workspace_id"] == workspace_id
    session_id = branch_body["prototype_session_id"]

    candidate = _run(
        harness.service.save_session_snapshot(
            prototype_session_id=session_id,
            snapshot_id="psnap_gate8_candidate_failed",
            storage_ref="prototype://gate8/candidate-failed",
            preview_health={"status": "ready", "source": "gate8-smoke-stub"},
        )
    )
    promotion = client.post(
        "/api/v1/prototype-promotions",
        json={
            "prototype_workspace_id": workspace_id,
            "prototype_session_id": session_id,
            "candidate_snapshot_id": candidate["snapshot_id"],
            "session_token": session_token,
            "request_reason": "Ready for Gate 8 review",
        },
    )
    assert promotion.status_code == 201
    promotion_request_id = promotion.json()["id"]

    failed_review = client.post(
        f"/api/v1/prototype-promotions/{promotion_request_id}/review",
        json={"decision": "approve", "review_notes": "Exercise failed validator"},
    )
    assert failed_review.status_code == 200
    assert failed_review.json()["status"] == "failed"
    after_failure = _run(harness.repo.get_workspace(workspace_id))
    assert after_failure["canonical_snapshot_id"] == original_canonical
    assert after_failure["publish_validation_status"] == "failed"

    harness.set_publish_validation(True)
    promoted_candidate = _run(
        harness.service.save_session_snapshot(
            prototype_session_id=session_id,
            snapshot_id="psnap_gate8_candidate_promoted",
            storage_ref="prototype://gate8/candidate-promoted",
            preview_health={"status": "ready", "source": "gate8-smoke-stub"},
        )
    )
    second_promotion = client.post(
        "/api/v1/prototype-promotions",
        json={
            "prototype_workspace_id": workspace_id,
            "prototype_session_id": session_id,
            "candidate_snapshot_id": promoted_candidate["snapshot_id"],
            "session_token": session_token,
            "request_reason": "Ready for successful Gate 8 review",
        },
    )
    assert second_promotion.status_code == 201

    successful_review = client.post(
        f"/api/v1/prototype-promotions/{second_promotion.json()['id']}/review",
        json={"decision": "approve", "review_notes": "Exercise passing validator"},
    )
    assert successful_review.status_code == 200
    assert successful_review.json()["status"] == "promoted"
    assert successful_review.json()["canonical_snapshot_id"] == promoted_candidate["snapshot_id"]
    assert successful_review.json()["preview_handle"]

    after_success = _run(harness.repo.get_workspace(workspace_id))
    assert after_success["canonical_snapshot_id"] == promoted_candidate["snapshot_id"]
    assert after_success["last_known_good_snapshot_id"] == promoted_candidate["snapshot_id"]
    assert after_success["publish_validation_status"] == "validated"
    actor = _run(harness.repo.get_shared_actor(shared_actor_id))
    assert actor is not None


def test_revoked_and_expired_prototype_links_fail_without_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gate 8 negative smoke: unavailable links stay non-enumerating."""
    harness = _build_release_smoke_app(
        monkeypatch,
        jobs_db_path=tmp_path / "prototype_jobs.db",
        publish_validation_ok=True,
    )
    workspace = _run(
        harness.repo.create_workspace(
            owner_user_id=1,
            title="Gate 8 negative smoke",
            creation_source="prompt",
        )
    )
    revoked = _run(
        harness.token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=workspace["id"],
            owner_user_id=1,
        )
    )
    _run(harness.token_service.revoke_token(revoked["id"]))
    expired = _run(
        harness.token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=workspace["id"],
            owner_user_id=1,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
    )

    for raw_token in (revoked["raw_token"], expired["raw_token"]):
        response = harness.client.post(
            f"/api/v1/sharing/public/{raw_token}/prototype-session",
            json={"display_name": "Gate 8 collaborator"},
        )

        assert response.status_code == 404
        _assert_prototype_error(
            response,
            category="invalid_or_unavailable_link",
            frontend_state="link_unavailable",
        )
