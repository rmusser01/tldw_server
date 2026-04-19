"""Integration tests for the prototype workspace API surface."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


def _seed_workspace(services: SimpleNamespace, *, title: str = "Sales dashboard") -> tuple[dict, dict]:
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
        assert body["preview_url"].startswith(
            f"/api/v1/prototype-previews/{preview_grant['preview_handle']}?"
        )
        assert body["token"]

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
