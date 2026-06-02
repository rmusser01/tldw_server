"""Integration tests for the prototype workspace API surface."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = pytest.mark.integration


@pytest.fixture
def owner_user() -> User:
    return User(
        id=1,
        username="owner",
        email="owner@test.com",
        roles=["admin"],
        permissions=["prototype.promote"],
    )


def _run(coro):
    return asyncio.run(coro)


def _assert_prototype_error(
    response,
    *,
    category: str,
    frontend_state: str,
    retryable: bool,
) -> None:
    detail = response.json()["detail"]
    assert detail["category"] == category
    assert detail["frontend_state"] == frontend_state
    assert detail["retryable"] is retryable
    assert isinstance(detail["message"], str)
    assert detail["message"]


def _assert_openapi_error_response(openapi: dict[str, Any], path: str, method: str, status_code: str) -> None:
    schema = openapi["paths"][path][method]["responses"][status_code]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/PrototypeErrorResponse")


def _assert_openapi_error_or_validation_response(
    openapi: dict[str, Any],
    path: str,
    method: str,
    status_code: str,
) -> None:
    schema = openapi["paths"][path][method]["responses"][status_code]["content"]["application/json"]["schema"]
    refs = {entry["$ref"].rsplit("/", maxsplit=1)[-1] for entry in schema["anyOf"]}
    assert {"PrototypeErrorResponse", "HTTPValidationError"} <= refs


def _seed_workspace(
    services: SimpleNamespace,
    *,
    title: str = "Sales dashboard",
    designated_promoter_ids: list[int] | None = None,
) -> tuple[dict, dict]:
    workspace = _run(
        services.repo.create_workspace(
            owner_user_id=1,
            title=title,
            creation_source="prompt",
            share_policy={"allow_browser_session_resume": True},
            runtime_policy={
                "owner_profile": "owner_collab",
                "external_collaborator_profile": "locked_collab",
            },
            designated_promoter_ids=designated_promoter_ids,
        )
    )
    seed_snapshot = _run(
        services.repo.create_snapshot(
            prototype_workspace_id=workspace["id"],
            snapshot_id=f"psnap_seed_{workspace['id']}",
            created_by_user_id=1,
            storage_ref="prototype://seed",
            prompt_summary=f"Seed prompt for {title}",
        )
    )
    workspace = _run(
        services.repo.update_workspace_state(
            workspace["id"],
            canonical_snapshot_id=seed_snapshot["snapshot_id"],
            last_known_good_snapshot_id=seed_snapshot["snapshot_id"],
            canonical_preview_status="uninitialized",
            publish_validation_status="pending",
        )
    )
    return workspace, seed_snapshot


def _seed_external_access(
    services: SimpleNamespace,
    *,
    prototype_workspace_id: str,
    share_link_id: int = 41,
):
    return _run(
        services.access_service.exchange_external_collaborator(
            prototype_workspace_id=prototype_workspace_id,
            share_link_id=share_link_id,
            display_name="Acme PM",
            resume_cookie_value=None,
        )
    )


def _seed_pending_promotion_request(
    services: SimpleNamespace,
    *,
    title: str,
    share_link_id: int,
    snapshot_id: str,
    designated_promoter_ids: list[int] | None = None,
) -> tuple[dict, dict, dict, dict]:
    workspace, _seed_snapshot = _seed_workspace(
        services,
        title=title,
        designated_promoter_ids=designated_promoter_ids,
    )
    access_context = _seed_external_access(
        services,
        prototype_workspace_id=workspace["id"],
        share_link_id=share_link_id,
    )
    session_result = _run(
        services.service.create_or_reuse_branch_session(
            prototype_workspace_id=workspace["id"],
            actor_type="external_collaborator",
            actor_shared_actor_id=access_context.shared_actor_id,
            request_nonce=f"req_{snapshot_id}",
            share_link_id=share_link_id,
        )
    )
    session = session_result["session"]
    candidate_snapshot = _run(
        services.repo.create_snapshot(
            prototype_workspace_id=workspace["id"],
            snapshot_id=snapshot_id,
            created_by_shared_actor_id=access_context.shared_actor_id,
            parent_snapshot_id=session["base_snapshot_id"],
            created_from_session_id=session["id"],
            storage_ref=f"prototype://{snapshot_id}",
            prompt_summary="Ready for review",
        )
    )
    promotion_request = _run(
        services.repo.create_promotion_request(
            prototype_workspace_id=workspace["id"],
            prototype_session_id=session["id"],
            candidate_snapshot_id=candidate_snapshot["snapshot_id"],
            requested_by_shared_actor_id=access_context.shared_actor_id,
        )
    )
    return workspace, session, candidate_snapshot, promotion_request


@pytest.fixture
def test_services(monkeypatch, repo, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import prototype_workspaces as prototype_endpoints
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Prototype_Workspaces.access import PrototypeAccessService
    from tldw_Server_API.app.core.Prototype_Workspaces.jobs import PrototypeWorkspaceJobs
    from tldw_Server_API.app.core.Prototype_Workspaces.preview_broker import (
        PrototypePreviewBroker,
    )
    from tldw_Server_API.app.core.Prototype_Workspaces.service import PrototypeWorkspaceService

    PrototypePreviewBroker._records.clear()
    PrototypePreviewBroker._active_scope_handles.clear()

    preview_broker = PrototypePreviewBroker(
        repo=repo,
        base_preview_path="/api/v1/prototype-previews",
        signing_secret="preview-test-secret",
    )
    access_service = PrototypeAccessService(
        repo,
        signing_secret="session-test-secret",
    )
    service = PrototypeWorkspaceService(
        repo=repo,
        preview_broker=preview_broker,
    )
    jobs_service = PrototypeWorkspaceJobs(
        repo=repo,
        jobs_manager=JobManager(db_path=tmp_path / "prototype_jobs.db"),
    )

    monkeypatch.setattr(prototype_endpoints, "_get_repo", lambda: repo)
    monkeypatch.setattr(prototype_endpoints, "_get_service", lambda: service)
    monkeypatch.setattr(prototype_endpoints, "_get_jobs_service", lambda: jobs_service)
    monkeypatch.setattr(prototype_endpoints, "_get_access_service", lambda: access_service)
    monkeypatch.setattr(prototype_endpoints, "_get_preview_broker", lambda: preview_broker)

    return SimpleNamespace(
        repo=repo,
        service=service,
        access_service=access_service,
        preview_broker=preview_broker,
        jobs_service=jobs_service,
    )


@pytest.fixture
def test_app(owner_user: User, test_services: SimpleNamespace) -> FastAPI:
    from tldw_Server_API.app.api.v1.endpoints.prototype_workspaces import router
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    async def _fake_user() -> User:
        return owner_user

    app.dependency_overrides[get_request_user] = _fake_user
    app.state.prototype_test_services = test_services
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    return TestClient(test_app)


class TestPrototypeWorkspaceEndpoints:
    def test_prototype_workspace_openapi_declares_error_contract(self, test_app: FastAPI) -> None:
        openapi = test_app.openapi()

        _assert_openapi_error_response(openapi, "/api/v1/prototype-workspaces/{prototype_workspace_id}", "get", "403")
        _assert_openapi_error_response(
            openapi,
            "/api/v1/prototype-workspaces/{prototype_workspace_id}/sessions",
            "post",
            "409",
        )
        _assert_openapi_error_response(openapi, "/api/v1/prototype-sessions", "post", "403")
        _assert_openapi_error_or_validation_response(openapi, "/api/v1/prototype-sessions", "post", "422")
        _assert_openapi_error_response(openapi, "/api/v1/prototype-promotions", "post", "404")
        _assert_openapi_error_response(
            openapi,
            "/api/v1/prototype-promotions/{promotion_request_id}/review",
            "post",
            "403",
        )
        _assert_openapi_error_response(openapi, "/api/v1/prototype-previews/{preview_handle}/renew", "post", "409")

    def test_request_validation_422_uses_fastapi_validation_shape(
        self,
        client: TestClient,
    ) -> None:
        resp = client.post("/api/v1/prototype-sessions", json={})

        assert resp.status_code == 422
        assert isinstance(resp.json()["detail"], list)

    def test_owner_can_fetch_workspace_detail_with_snapshot_and_session_inventory(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, seed_snapshot = _seed_workspace(test_services, title="Detail prototype")
        owner_session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="owner",
                actor_user_id=1,
                request_nonce="req_owner_detail_view",
            )
        )
        owner_session = owner_session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_candidate_detail_1",
                created_by_user_id=1,
                parent_snapshot_id=seed_snapshot["snapshot_id"],
                created_from_session_id=owner_session["id"],
                storage_ref="prototype://candidate-detail",
                prompt_summary="Candidate snapshot for detail view",
            )
        )

        resp = client.get(f"/api/v1/prototype-workspaces/{workspace['id']}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == workspace["id"]
        assert body["viewer_role"] == "owner"
        assert body["canonical_snapshot_id"] == seed_snapshot["snapshot_id"]
        assert len(body["sessions"]) == 1
        assert body["sessions"][0]["id"] == owner_session["id"]
        assert len(body["snapshots"]) == 2
        assert body["snapshots"][0]["snapshot_id"] == candidate_snapshot["snapshot_id"]
        assert body["snapshots"][0]["is_canonical"] is False
        assert body["snapshots"][1]["snapshot_id"] == seed_snapshot["snapshot_id"]
        assert body["snapshots"][1]["is_canonical"] is True

    def test_owner_workspace_detail_includes_promotion_request_inventory(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        """Workspace detail includes pending promotion requests needed by the owner UI."""
        workspace, seed_snapshot = _seed_workspace(test_services, title="Promotion inventory prototype")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=63,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_promotion_inventory",
                share_link_id=63,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_promotion_inventory_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=seed_snapshot["snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://promotion-inventory-candidate",
                prompt_summary="Inventory candidate",
            )
        )
        promotion_request = _run(
            test_services.repo.create_promotion_request(
                prototype_workspace_id=workspace["id"],
                prototype_session_id=session["id"],
                candidate_snapshot_id=candidate_snapshot["snapshot_id"],
                requested_by_shared_actor_id=access_context.shared_actor_id,
            )
        )

        resp = client.get(f"/api/v1/prototype-workspaces/{workspace['id']}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["promotion_requests"] == [
            {
                "id": promotion_request["id"],
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "requested_by_user_id": None,
                "requested_by_shared_actor_id": access_context.shared_actor_id,
                "status": "pending",
                "reviewed_by_user_id": None,
                "review_notes": None,
                "created_at": promotion_request["created_at"],
                "updated_at": promotion_request["updated_at"],
            }
        ]

    def test_owner_can_create_workspace_and_request_branch_session(self, client: TestClient) -> None:
        created = client.post(
            "/api/v1/prototype-workspaces",
            json={
                "title": "Sales dashboard",
                "creation_source": "prompt",
                "prompt": "Build a B2B dashboard",
            },
        )

        assert created.status_code == 201
        created_body = created.json()
        assert created_body["title"] == "Sales dashboard"
        assert created_body["id"]

        session = client.post(
            f"/api/v1/prototype-workspaces/{created_body['id']}/sessions",
            json={},
        )
        assert session.status_code == 202
        session_body = session.json()
        assert session_body["job_type"] == "branch_session_bootstrap"
        assert session_body["prototype_workspace_id"] == created_body["id"]
        assert session_body["prototype_session_id"].startswith("pss_")
        assert session_body["actor_type"] == "owner"

    def test_collaborator_can_create_session_with_prototype_session_token(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Collaborator prototype")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
        )

        resp = client.post(
            "/api/v1/prototype-sessions",
            json={
                "session_token": access_context.session_token,
                "request_nonce": "req_collab_token_1",
            },
        )

        assert resp.status_code == 202
        body = resp.json()
        assert body["actor_type"] == "external_collaborator"
        assert body["shared_actor_id"] == access_context.shared_actor_id
        assert body["prototype_workspace_id"] == workspace["id"]
        assert body["job_type"] == "branch_session_bootstrap"
        assert body["prototype_session_id"].startswith("pss_")

    def test_revoked_collaborator_session_token_returns_403(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Revoked collaborator prototype")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
        )
        _run(
            test_services.repo.revoke_shared_actor(
                access_context.shared_actor_id,
                revoked_at=datetime.now(timezone.utc).isoformat(),
            )
        )

        resp = client.post(
            "/api/v1/prototype-sessions",
            json={
                "session_token": access_context.session_token,
                "request_nonce": "req_revoked_collab_token_1",
            },
        )

        assert resp.status_code == 403
        _assert_prototype_error(
            resp,
            category="inactive_session",
            frontend_state="session_inactive",
            retryable=False,
        )

    def test_owner_workspace_detail_forbidden_returns_stable_error(
        self,
        test_app: FastAPI,
        test_services: SimpleNamespace,
    ) -> None:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        async def _fake_non_owner() -> User:
            return User(
                id=2,
                username="other",
                email="other@test.com",
                roles=[],
                permissions=[],
            )

        test_app.dependency_overrides[get_request_user] = _fake_non_owner
        client = TestClient(test_app)
        workspace, _ = _seed_workspace(test_services, title="Forbidden detail")

        resp = client.get(f"/api/v1/prototype-workspaces/{workspace['id']}")

        assert resp.status_code == 403
        _assert_prototype_error(
            resp,
            category="unauthorized",
            frontend_state="unauthorized",
            retryable=False,
        )

    def test_collaborator_can_submit_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Promotion request prototype")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=52,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_promotion_submission",
                share_link_id=52,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_demo_1",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://candidate",
                prompt_summary="Ready for owner review",
            )
        )

        resp = client.post(
            "/api/v1/prototype-promotions",
            json={
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "session_token": access_context.session_token,
                "request_reason": "Ready for owner review",
            },
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["prototype_workspace_id"] == workspace["id"]
        assert body["prototype_session_id"] == session["id"]
        assert body["candidate_snapshot_id"] == candidate_snapshot["snapshot_id"]
        assert body["status"] == "pending"
        assert body["requested_by_shared_actor_id"] == access_context.shared_actor_id

    def test_collaborator_cannot_submit_promotion_request_for_another_branch(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Promotion auth prototype")
        first_actor = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=81,
        )
        second_actor = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=82,
        )
        first_session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=first_actor.shared_actor_id,
                request_nonce="req_first_branch",
                share_link_id=81,
            )
        )
        first_session = first_session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_other_actor_candidate",
                created_by_shared_actor_id=first_actor.shared_actor_id,
                parent_snapshot_id=first_session["base_snapshot_id"],
                created_from_session_id=first_session["id"],
                storage_ref="prototype://other-actor-candidate",
                prompt_summary="Another actor's candidate",
            )
        )

        resp = client.post(
            "/api/v1/prototype-promotions",
            json={
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": first_session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "session_token": second_actor.session_token,
            },
        )

        assert resp.status_code == 403

    def test_revoked_shared_actor_cannot_submit_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Revoked promotion actor")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=83,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_revoked_promotion_actor",
                share_link_id=83,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_revoked_actor_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://revoked-actor-candidate",
                prompt_summary="Revoked actor candidate",
            )
        )
        _run(
            test_services.repo.revoke_shared_actor(
                access_context.shared_actor_id,
                revoked_at=datetime.now(timezone.utc).isoformat(),
            )
        )

        resp = client.post(
            "/api/v1/prototype-promotions",
            json={
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "session_token": access_context.session_token,
            },
        )

        assert resp.status_code == 403
        _assert_prototype_error(
            resp,
            category="inactive_session",
            frontend_state="session_inactive",
            retryable=False,
        )

    def test_expired_shared_actor_cannot_submit_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Expired promotion actor")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=84,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_expired_promotion_actor",
                share_link_id=84,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_expired_actor_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://expired-actor-candidate",
                prompt_summary="Expired actor candidate",
            )
        )
        _run(
            test_services.repo.update_shared_actor_expiry(
                access_context.shared_actor_id,
                expires_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            )
        )

        resp = client.post(
            "/api/v1/prototype-promotions",
            json={
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "session_token": access_context.session_token,
            },
        )

        assert resp.status_code == 403
        _assert_prototype_error(
            resp,
            category="inactive_session",
            frontend_state="session_inactive",
            retryable=False,
        )

    def test_malformed_shared_actor_expiry_cannot_submit_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Malformed promotion actor expiry")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=85,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_malformed_expiry_promotion_actor",
                share_link_id=85,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_malformed_expiry_actor_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://malformed-expiry-actor-candidate",
                prompt_summary="Malformed expiry actor candidate",
            )
        )
        _run(
            test_services.repo.update_shared_actor_expiry(
                access_context.shared_actor_id,
                expires_at="not-a-timestamp",
            )
        )

        resp = client.post(
            "/api/v1/prototype-promotions",
            json={
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "session_token": access_context.session_token,
            },
        )

        assert resp.status_code == 403
        _assert_prototype_error(
            resp,
            category="inactive_session",
            frontend_state="session_inactive",
            retryable=False,
        )

    def test_session_share_link_mismatch_cannot_submit_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Session share-link mismatch")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=86,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_session_share_link_mismatch",
                share_link_id=86,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_session_share_link_mismatch_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://session-share-link-mismatch-candidate",
                prompt_summary="Session share-link mismatch candidate",
            )
        )
        original_get_session = test_services.repo.get_session

        async def get_mismatched_session(prototype_session_id: str) -> dict[str, Any] | None:
            row = await original_get_session(prototype_session_id)
            if row and row.get("id") == session["id"]:
                row = dict(row)
                row["share_link_id"] = 999
            return row

        monkeypatch.setattr(test_services.repo, "get_session", get_mismatched_session)

        resp = client.post(
            "/api/v1/prototype-promotions",
            json={
                "prototype_workspace_id": workspace["id"],
                "prototype_session_id": session["id"],
                "candidate_snapshot_id": candidate_snapshot["snapshot_id"],
                "session_token": access_context.session_token,
            },
        )

        assert resp.status_code == 403
        _assert_prototype_error(
            resp,
            category="inactive_session",
            frontend_state="session_inactive",
            retryable=False,
        )

    def test_owner_can_review_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _ = _seed_workspace(test_services, title="Promotion review prototype")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=61,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_review",
                share_link_id=61,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_review_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://review-candidate",
                prompt_summary="Looks good",
            )
        )
        promotion_request = _run(
            test_services.repo.create_promotion_request(
                prototype_workspace_id=workspace["id"],
                prototype_session_id=session["id"],
                candidate_snapshot_id=candidate_snapshot["snapshot_id"],
                requested_by_shared_actor_id=access_context.shared_actor_id,
            )
        )

        review = client.post(
            f"/api/v1/prototype-promotions/{promotion_request['id']}/review",
            json={
                "decision": "approve",
                "review_notes": "Looks good",
            },
        )

        assert review.status_code == 200
        body = review.json()
        assert body["status"] == "promoted"
        assert body["preview_handle"]
        assert body["canonical_snapshot_id"] == candidate_snapshot["snapshot_id"]

    def test_designated_promoter_can_review_promotion_request(
        self,
        test_app: FastAPI,
        test_services: SimpleNamespace,
    ) -> None:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        async def _fake_promoter() -> User:
            return User(
                id=2,
                username="promoter",
                email="promoter@test.com",
                roles=[],
                permissions=["prototype.promote"],
            )

        test_app.dependency_overrides[get_request_user] = _fake_promoter
        client = TestClient(test_app)

        workspace, _ = _seed_workspace(
            test_services,
            title="Designated promotion review prototype",
            designated_promoter_ids=[2],
        )
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=62,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_designated_review",
                share_link_id=62,
            )
        )
        session = session_result["session"]
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_designated_review_candidate",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=session["base_snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://designated-review-candidate",
                prompt_summary="Looks good to designated promoter",
            )
        )
        promotion_request = _run(
            test_services.repo.create_promotion_request(
                prototype_workspace_id=workspace["id"],
                prototype_session_id=session["id"],
                candidate_snapshot_id=candidate_snapshot["snapshot_id"],
                requested_by_shared_actor_id=access_context.shared_actor_id,
            )
        )

        review = client.post(
            f"/api/v1/prototype-promotions/{promotion_request['id']}/review",
            json={
                "decision": "approve",
                "review_notes": "Looks good from designated promoter",
            },
        )

        assert review.status_code == 200
        body = review.json()
        assert body["status"] == "promoted"
        assert body["canonical_snapshot_id"] == candidate_snapshot["snapshot_id"]

    def test_review_endpoint_delegates_decision_to_service(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict] = []

        async def fake_review_promotion_request(**kwargs):
            calls.append(kwargs)
            return {
                "status": "rejected",
                "prototype_workspace_id": "pw_delegated",
                "candidate_snapshot_id": "psnap_delegated",
                "canonical_snapshot_id": "psnap_canonical",
                "details": {"review_notes": "Delegated"},
            }

        async def fail_if_endpoint_reads_request(_promotion_request_id: str):
            raise AssertionError("endpoint should delegate promotion review lookup to the service")

        monkeypatch.setattr(
            test_services.service,
            "review_promotion_request",
            fake_review_promotion_request,
            raising=False,
        )
        monkeypatch.setattr(
            test_services.repo,
            "get_promotion_request",
            fail_if_endpoint_reads_request,
        )

        review = client.post(
            "/api/v1/prototype-promotions/ppr_delegated/review",
            json={
                "decision": "reject",
                "review_notes": "Delegated",
            },
        )

        assert review.status_code == 200
        assert review.json() == {
            "status": "rejected",
            "failure_code": None,
            "prototype_workspace_id": "pw_delegated",
            "candidate_snapshot_id": "psnap_delegated",
            "canonical_snapshot_id": "psnap_canonical",
            "preview_handle": None,
            "details": {"review_notes": "Delegated"},
        }
        assert calls == [
            {
                "promotion_request_id": "ppr_delegated",
                "reviewer_user_id": 1,
                "decision": "reject",
                "review_notes": "Delegated",
                "review_baseline_snapshot_id": None,
            }
        ]

    def test_owner_can_reject_promotion_request(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, _session, candidate_snapshot, promotion_request = _seed_pending_promotion_request(
            test_services,
            title="Promotion reject prototype",
            share_link_id=63,
            snapshot_id="psnap_reject_candidate",
        )

        review = client.post(
            f"/api/v1/prototype-promotions/{promotion_request['id']}/review",
            json={
                "decision": "reject",
                "review_notes": "Needs another pass",
            },
        )

        assert review.status_code == 200
        body = review.json()
        assert body["status"] == "rejected"
        assert body["prototype_workspace_id"] == workspace["id"]
        assert body["candidate_snapshot_id"] == candidate_snapshot["snapshot_id"]
        assert body["canonical_snapshot_id"] == workspace["canonical_snapshot_id"]
        assert body["details"] == {"review_notes": "Needs another pass"}
        updated_request = _run(test_services.repo.get_promotion_request(promotion_request["id"]))
        assert updated_request["status"] == "rejected"
        assert updated_request["reviewed_by_user_id"] == 1

    def test_non_promoter_cannot_review_promotion_request(
        self,
        test_app: FastAPI,
        test_services: SimpleNamespace,
    ) -> None:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

        async def _fake_non_promoter() -> User:
            return User(
                id=3,
                username="viewer",
                email="viewer@test.com",
                roles=[],
                permissions=[],
            )

        test_app.dependency_overrides[get_request_user] = _fake_non_promoter
        client = TestClient(test_app)
        _workspace, _session, _candidate_snapshot, promotion_request = _seed_pending_promotion_request(
            test_services,
            title="Promotion forbidden prototype",
            share_link_id=64,
            snapshot_id="psnap_forbidden_review_candidate",
        )

        review = client.post(
            f"/api/v1/prototype-promotions/{promotion_request['id']}/review",
            json={
                "decision": "reject",
                "review_notes": "I should not be able to review this",
            },
        )

        assert review.status_code == 403
        assert review.json()["detail"] == "Reviewer does not have promotion permissions"
        updated_request = _run(test_services.repo.get_promotion_request(promotion_request["id"]))
        assert updated_request["status"] == "pending"

    def test_review_missing_promotion_request_returns_404(
        self,
        client: TestClient,
    ) -> None:
        review = client.post(
            "/api/v1/prototype-promotions/ppr_missing/review",
            json={
                "decision": "reject",
                "review_notes": "Cannot find it",
            },
        )

        assert review.status_code == 404
        assert review.json()["detail"] == "Prototype promotion request not found"

    def test_preview_grant_renewal_returns_updated_expiry(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, seed_snapshot = _seed_workspace(test_services, title="Preview renewal prototype")
        preview_grant = _run(
            test_services.preview_broker.issue_preview_grant(
                prototype_workspace_id=workspace["id"],
                snapshot_id=seed_snapshot["snapshot_id"],
                runtime_target_url="runtime://canonical/preview-renewal",
            )
        )

        renewed = client.post(
            f"/api/v1/prototype-previews/{preview_grant['preview_handle']}/renew",
            json={},
        )

        assert renewed.status_code == 200
        body = renewed.json()
        assert body["preview_handle"] == preview_grant["preview_handle"]
        assert body["expires_at"]
        assert body["preview_url"].startswith(f"/api/v1/prototype-previews/{preview_grant['preview_handle']}?")
        assert body["token"]

    def test_preview_grant_renewal_recovers_after_broker_memory_clear(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, seed_snapshot = _seed_workspace(test_services, title="Preview renewal recovered")
        preview_grant = _run(
            test_services.preview_broker.issue_preview_grant(
                prototype_workspace_id=workspace["id"],
                snapshot_id=seed_snapshot["snapshot_id"],
                runtime_target_url="runtime://canonical/preview-renewal-recovered",
            )
        )
        broker_cls = type(test_services.preview_broker)
        broker_cls._records.clear()
        broker_cls._active_scope_handles.clear()

        renewed = client.post(
            f"/api/v1/prototype-previews/{preview_grant['preview_handle']}/renew",
            json={},
        )

        assert renewed.status_code == 200
        assert renewed.json()["preview_handle"] == preview_grant["preview_handle"]

    def test_preview_grant_renewal_rejects_unknown_body_fields(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, seed_snapshot = _seed_workspace(test_services, title="Preview renewal strict body")
        preview_grant = _run(
            test_services.preview_broker.issue_preview_grant(
                prototype_workspace_id=workspace["id"],
                snapshot_id=seed_snapshot["snapshot_id"],
                runtime_target_url="runtime://canonical/preview-renewal-strict",
            )
        )

        renewed = client.post(
            f"/api/v1/prototype-previews/{preview_grant['preview_handle']}/renew",
            json={"extend_seconds": 600},
        )

        assert renewed.status_code == 422

    def test_preview_grant_renewal_maps_typed_missing_handle_to_404(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace, _seed_snapshot = _seed_workspace(test_services, title="Preview renewal typed error")
        broker_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.preview_broker")
        missing_handle_error = getattr(broker_module, "PrototypePreviewHandleNotFound", RuntimeError)

        def fake_record(_preview_handle: str) -> dict[str, str]:
            return {"prototype_workspace_id": workspace["id"]}

        async def fake_renew(_preview_handle: str) -> dict[str, str]:
            raise missing_handle_error("preview handle disappeared")

        monkeypatch.setattr(test_services.preview_broker, "get_preview_record", fake_record)
        monkeypatch.setattr(test_services.preview_broker, "renew_preview_grant", fake_renew)

        renewed = client.post(
            "/api/v1/prototype-previews/ph_raced_away/renew",
            json={},
        )

        assert renewed.status_code == 404
        _assert_prototype_error(
            renewed,
            category="preview_unavailable",
            frontend_state="preview_unavailable",
            retryable=False,
        )
        assert renewed.json()["detail"]["message"] == "Prototype preview is unavailable"

    def test_preview_grant_renewal_maps_runtime_error_to_stable_message(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace, _seed_snapshot = _seed_workspace(test_services, title="Preview renewal conflict")

        def fake_record(_preview_handle: str) -> dict[str, str]:
            return {"prototype_workspace_id": workspace["id"]}

        async def fake_renew(_preview_handle: str) -> dict[str, str]:
            raise RuntimeError("runtime target leaked: http://internal-preview-host")

        monkeypatch.setattr(test_services.preview_broker, "get_preview_record", fake_record)
        monkeypatch.setattr(test_services.preview_broker, "renew_preview_grant", fake_renew)

        renewed = client.post(
            "/api/v1/prototype-previews/ph_conflict/renew",
            json={},
        )

        assert renewed.status_code == 409
        _assert_prototype_error(
            renewed,
            category="preview_unavailable",
            frontend_state="preview_unavailable",
            retryable=True,
        )
        assert renewed.json()["detail"]["message"] == "Prototype preview renewal conflict; please retry"

    def test_revoked_preview_grant_renewal_returns_404(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, seed_snapshot = _seed_workspace(test_services, title="Preview revoke prototype")
        preview_grant = _run(
            test_services.preview_broker.issue_preview_grant(
                prototype_workspace_id=workspace["id"],
                snapshot_id=seed_snapshot["snapshot_id"],
                runtime_target_url="runtime://canonical/preview-revoked",
            )
        )
        _run(test_services.preview_broker.revoke_preview_handle(preview_grant["preview_handle"]))

        renewed = client.post(
            f"/api/v1/prototype-previews/{preview_grant['preview_handle']}/renew",
            json={},
        )

        assert renewed.status_code == 404
        _assert_prototype_error(
            renewed,
            category="preview_unavailable",
            frontend_state="preview_unavailable",
            retryable=False,
        )

    def test_stale_promotion_response_shape(
        self,
        client: TestClient,
        test_services: SimpleNamespace,
    ) -> None:
        workspace, seed_snapshot = _seed_workspace(test_services, title="Stale review prototype")
        access_context = _seed_external_access(
            test_services,
            prototype_workspace_id=workspace["id"],
            share_link_id=73,
        )
        session_result = _run(
            test_services.service.create_or_reuse_branch_session(
                prototype_workspace_id=workspace["id"],
                actor_type="external_collaborator",
                actor_shared_actor_id=access_context.shared_actor_id,
                request_nonce="req_stale_review",
                share_link_id=73,
            )
        )
        session = session_result["session"]
        fresh_canonical = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_canonical_fresh",
                created_by_user_id=1,
                parent_snapshot_id=seed_snapshot["snapshot_id"],
                storage_ref="prototype://fresh-canonical",
                prompt_summary="Fresh canonical state",
            )
        )
        _run(
            test_services.repo.update_workspace_state(
                workspace["id"],
                canonical_snapshot_id=fresh_canonical["snapshot_id"],
                last_known_good_snapshot_id=fresh_canonical["snapshot_id"],
            )
        )
        candidate_snapshot = _run(
            test_services.repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id="psnap_candidate_stale",
                created_by_shared_actor_id=access_context.shared_actor_id,
                parent_snapshot_id=seed_snapshot["snapshot_id"],
                created_from_session_id=session["id"],
                storage_ref="prototype://candidate-stale",
                prompt_summary="Stale candidate",
            )
        )
        promotion_request = _run(
            test_services.repo.create_promotion_request(
                prototype_workspace_id=workspace["id"],
                prototype_session_id=session["id"],
                candidate_snapshot_id=candidate_snapshot["snapshot_id"],
                requested_by_shared_actor_id=access_context.shared_actor_id,
            )
        )

        resp = client.post(
            f"/api/v1/prototype-promotions/{promotion_request['id']}/review",
            json={
                "decision": "approve",
                "review_baseline_snapshot_id": fresh_canonical["snapshot_id"],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "stale"
        assert body["failure_code"] == "stale_candidate"
        assert body["prototype_workspace_id"] == workspace["id"]
        assert body["candidate_snapshot_id"] == candidate_snapshot["snapshot_id"]
        assert body["canonical_snapshot_id"] == fresh_canonical["snapshot_id"]
