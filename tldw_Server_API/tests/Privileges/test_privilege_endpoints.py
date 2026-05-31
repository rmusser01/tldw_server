import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.PrivilegeMaps.introspection import DependencyMetadata, RouteMetadata
from tldw_Server_API.app.core.PrivilegeMaps.service import PrivilegeMapService
from tldw_Server_API.app.core.PrivilegeMaps.snapshots import PrivilegeSnapshotStore, get_privilege_snapshot_store
from tldw_Server_API.app.core.PrivilegeMaps.service import get_privilege_map_service


class InMemoryTrendStore:
    def __init__(self) -> None:
        self.snapshots = []

    async def record_snapshot(self, *, scope, group_by, catalog_version, generated_at, buckets, team_id=None):  # type: ignore[no-untyped-def]
        self.snapshots.append((scope, group_by, generated_at, buckets, team_id))

    async def compute_trends(self, *, scope, group_by, bucket_counts, window_start, window_end, team_id=None, org_id=None):  # type: ignore[no-untyped-def]
        trends = []
        for key in bucket_counts.keys():
            trends.append(
                {
                    "key": key,
                    "window": {
                        "start": window_start.isoformat(),
                        "end": window_end.isoformat(),
                    },
                    "delta_users": 0,
                    "delta_endpoints": 0,
                    "delta_scopes": 0,
                }
            )
        return trends


class FakePrivilegeMapService(PrivilegeMapService):
    def __init__(self) -> None:
        fake_registry = {
            "media.ingest": [
                RouteMetadata(
                    path="/api/v1/media/process",
                    methods=("POST",),
                    name="media_process",
                    tags=("media",),
                    endpoint="tests.fake.media.process",
                    dependencies=(
                        DependencyMetadata(
                            id="auth.require_token_scope",
                            type="dependency",
                            module="tldw.fake.auth",
                        ),
                    ),
                    dependency_sources=("tldw.fake.require_token_scope",),
                    rate_limit_resources=("media.ingest",),
                    summary="Ingest new media assets",
                    description="Allows uploading media for processing.",
                )
            ],
            "rag.search": [
                RouteMetadata(
                    path="/api/v1/rag/search",
                    methods=("POST",),
                    name="rag_search",
                    tags=("rag",),
                    endpoint="tests.fake.rag.search",
                    dependencies=(
                        DependencyMetadata(
                            id="auth.require_token_scope",
                            type="dependency",
                            module="tldw.fake.auth",
                        ),
                    ),
                    dependency_sources=("tldw.fake.require_token_scope",),
                    rate_limit_resources=("rag.search",),
                    summary="Perform RAG search",
                    description="Execute retrieval-augmented generation queries.",
                )
            ],
            "media.catalog.view": [
                RouteMetadata(
                    path="/api/v1/media/catalog",
                    methods=("GET",),
                    name="media_catalog",
                    tags=("media", "catalog"),
                    endpoint="tests.fake.media.catalog",
                    dependencies=(
                        DependencyMetadata(
                            id="auth.require_token_scope",
                            type="dependency",
                            module="tldw.fake.auth",
                        ),
                    ),
                    dependency_sources=("tldw.fake.require_token_scope",),
                    rate_limit_resources=("media.catalog.view",),
                    summary="View media catalog",
                    description="List media records.",
                )
            ],
        }
        super().__init__(route_registry=fake_registry, trend_store=InMemoryTrendStore())
        self.sample_users = [
            {
                "id": "user-1",
                "username": "Alex Rivera",
                "primary_role": "admin",
                "roles": ["admin"],
                "permissions": [],
                "feature_flags": {flag.id for flag in self.catalog.feature_flags},
                "allowed_scopes": {scope.id for scope in self.catalog.scopes},
            },
            {
                "id": "user-2",
                "username": "Priya Patel",
                "primary_role": "analyst",
                "roles": ["analyst"],
                "permissions": {"rag.search"},
                "feature_flags": {"media_ingest_beta"},
                "allowed_scopes": {"rag.search"},
            },
            {
                "id": "user-3",
                "username": "Morgan Lee",
                "primary_role": "viewer",
                "roles": ["viewer"],
                "permissions": {"media.catalog.view"},
                "feature_flags": set(),
                "allowed_scopes": {"media.catalog.view"},
            },
        ]
        self.sample_memberships = [
            {"team_id": "team-1", "user_id": "user-1", "team_name": "Core Admins", "org_id": "acme"},
            {"team_id": "team-1", "user_id": "user-2", "team_name": "Core Admins", "org_id": "acme"},
            {"team_id": "team-2", "user_id": "user-3", "team_name": "Viewers", "org_id": "acme"},
        ]
        self.sample_org_memberships = [
            {"org_id": "acme", "user_id": "user-1", "membership_role": "member"},
            {"org_id": "acme", "user_id": "user-2", "membership_role": "member"},
            {"org_id": "acme", "user_id": "user-3", "membership_role": "member"},
        ]

    async def _fetch_users(self):
        return list(self.sample_users)

    async def _fetch_team_memberships(self):
        return list(self.sample_memberships)

    async def _fetch_org_memberships(self):
        return list(self.sample_org_memberships)


@pytest.fixture()
def privilege_test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_service = FakePrivilegeMapService()
    db_path = tmp_path / "privilege_snapshots.db"
    monkeypatch.setenv("DATABASE_URL", db_path.as_uri())
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    asyncio.run(reset_db_pool())
    snapshot_store = PrivilegeSnapshotStore()

    async def seed_snapshots():
        await snapshot_store.clear()
        await snapshot_store.add_snapshot(
            {
                "snapshot_id": "snap-2025-01-15-001",
                "generated_at": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc).isoformat(),
                "generated_by": "user-42",
                "target_scope": "org",
                "org_id": "acme",
                "team_id": None,
                "catalog_version": fake_service.catalog.version,
                "summary": {
                    "users": 2,
                    "scopes": 3,
                    "endpoints": 3,
                    "scope_ids": ["media.ingest", "chat.admin", "rag.search"],
                    "sensitivity_breakdown": {"high": 1, "restricted": 1, "moderate": 1},
                },
            },
            detail_items=[
                {
                    "user_id": "user-1",
                    "endpoint": "/api/v1/media/process",
                    "method": "POST",
                    "privilege_scope_id": "media.ingest",
                    "feature_flag_id": "media_ingest_beta",
                    "sensitivity_tier": "high",
                    "ownership_predicates": ["same_org"],
                    "status": "allowed",
                    "blocked_reason": None,
                    "dependencies": [
                        {
                            "id": "auth.require_token_scope",
                            "type": "dependency",
                            "module": "tldw.fake.auth",
                        }
                    ],
                    "dependency_sources": ["tldw.fake.require_token_scope"],
                    "rate_limit_class": "elevated",
                    "rate_limit_resources": ["media.ingest"],
                    "source_module": "tests.fake.media",
                    "summary": "Ingest new media assets",
                    "tags": ["media", "ingestion"],
                }
            ],
        )
        await snapshot_store.add_snapshot(
            {
                "snapshot_id": "snap-2025-01-12-001",
                "generated_at": datetime(2025, 1, 12, 12, 30, tzinfo=timezone.utc).isoformat(),
                "generated_by": "user-99",
                "target_scope": "team",
                "org_id": "beta",
                "team_id": "team-2",
                "catalog_version": fake_service.catalog.version,
                "summary": {
                    "users": 1,
                    "scopes": 1,
                    "endpoints": 1,
                    "scope_ids": ["media.catalog.view"],
                    "sensitivity_breakdown": {"low": 1},
                },
            },
            detail_items=[
                {
                    "user_id": "user-3",
                    "endpoint": "/api/v1/media/catalog",
                    "method": "GET",
                    "privilege_scope_id": "media.catalog.view",
                    "feature_flag_id": None,
                    "sensitivity_tier": "low",
                    "ownership_predicates": [],
                    "status": "allowed",
                    "blocked_reason": None,
                    "dependencies": [
                        {
                            "id": "auth.require_token_scope",
                            "type": "dependency",
                            "module": "tldw.fake.auth",
                        }
                    ],
                    "dependency_sources": ["tldw.fake.require_token_scope"],
                    "rate_limit_class": "standard",
                    "rate_limit_resources": ["media.catalog.view"],
                    "source_module": "tests.fake.media",
                    "summary": "List media records.",
                    "tags": ["media", "catalog"],
                }
            ],
        )

    asyncio.run(seed_snapshots())

    def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="Admin User",
            roles=["admin"],
            permissions=[],
            is_admin=True,
            subject="user:1",
        )

    fastapi_app.dependency_overrides[get_auth_principal] = override_principal
    fastapi_app.dependency_overrides[get_privilege_map_service] = lambda: fake_service
    fastapi_app.dependency_overrides[get_privilege_snapshot_store] = lambda: snapshot_store

    with TestClient(fastapi_app) as client:
        yield client

    asyncio.run(snapshot_store.clear())
    asyncio.run(reset_db_pool())
    fastapi_app.dependency_overrides.clear()


def test_get_org_summary_group_by_role(privilege_test_client: TestClient):
    response = privilege_test_client.get("/api/v1/privileges/org")
    assert response.status_code == 200
    payload = response.json()
    assert payload["group_by"] == "role"
    assert "trends" in payload
    assert isinstance(payload["trends"], list)
    keys = {bucket["key"] for bucket in payload["buckets"]}
    assert "admin" in keys
    assert "analyst" in keys


def test_org_summary_trends_requested(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/org",
        params={"include_trends": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["trends"]
    sample = payload["trends"][0]
    assert {"key", "delta_users", "delta_endpoints", "delta_scopes"}.issubset(sample.keys())


def test_get_org_detail_pagination(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/org",
        params={"view": "detail", "page": 1, "page_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 1
    assert payload["page_size"] == 5
    assert payload["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 5,
        "total": payload["total_items"],
        "total_pages": (payload["total_items"] + 4) // 5,
        "has_more": payload["total_items"] > 5,
    }
    assert payload["items"], "Expected detail items to be present"
    statuses = {item["status"] for item in payload["items"]}
    assert statuses.issubset({"allowed", "blocked"})
    first = payload["items"][0]
    assert isinstance(first["dependencies"], list)
    assert first["dependencies"], "Expected at least one dependency entry"
    assert {"id", "type"}.issubset(first["dependencies"][0].keys())
    assert isinstance(first["rate_limit_resources"], list)
    assert first["source_module"]


def test_org_detail_dependency_filter(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/org",
        params={
            "view": "detail",
            "page": 1,
            "page_size": 20,
            "dependency": "ratelimit.media.catalog.view",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["items"], "Expected filtered results"
    for item in payload["items"]:
        dependency_ids = {dep["id"] for dep in item["dependencies"]}
        assert "ratelimit.media.catalog.view" in dependency_ids


def test_team_detail_filters(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/teams/team-1",
        params={"view": "detail", "page": 1, "page_size": 5, "resource": "media"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_items"] >= 1
    assert payload["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 5,
        "total": payload["total_items"],
        "total_pages": (payload["total_items"] + 4) // 5,
        "has_more": payload["total_items"] > 5,
    }
    for item in payload["items"]:
        assert "media" in item["endpoint"]


def test_user_detail_includes_canonical_page_pagination(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/users/user-1",
        params={"page": 1, "page_size": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 5,
        "total": payload["total_items"],
        "total_pages": (payload["total_items"] + 4) // 5,
        "has_more": payload["total_items"] > 5,
    }


def test_team_detail_dependency_filter(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/teams/team-1",
        params={
            "view": "detail",
            "page": 1,
            "page_size": 20,
            "dependency": "ratelimit.media.ingest",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["items"]
    for item in payload["items"]:
        dependency_ids = {dep["id"] for dep in item["dependencies"]}
        assert "ratelimit.media.ingest" in dependency_ids


def test_snapshot_list_filters(privilege_test_client: TestClient):
    response = privilege_test_client.get(
        "/api/v1/privileges/snapshots",
        params={"org_id": "acme", "include_counts": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_items"] == 1
    assert payload["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 50,
        "total": 1,
        "total_pages": 1,
        "has_more": False,
    }
    assert payload["items"][0]["org_id"] == "acme"
    assert payload["items"][0]["target_scope"] == "org"

    response = privilege_test_client.get(
        "/api/v1/privileges/snapshots",
        params={"scope": "media.catalog.view"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_items"] == 1
    assert payload["items"][0]["snapshot_id"] == "snap-2025-01-12-001"


def test_get_self_map(privilege_test_client: TestClient):
    response = privilege_test_client.get("/api/v1/privileges/self")
    assert response.status_code == 200
    payload = response.json()
    assert "items" in payload
    assert "recommended_actions" in payload


def test_get_snapshot_detail(privilege_test_client: TestClient):
    response = privilege_test_client.get("/api/v1/privileges/snapshots/snap-2025-01-15-001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot_id"] == "snap-2025-01-15-001"
    assert payload["detail"]["total_items"] >= 1
    assert payload["detail"]["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 500,
        "total": payload["detail"]["total_items"],
        "total_pages": 1,
        "has_more": False,
    }
    endpoints = {item["endpoint"] for item in payload["detail"]["items"]}
    assert "/api/v1/media/process" in endpoints
    assert payload["target_scope"] == "org"


def test_export_snapshot_json(privilege_test_client: TestClient):
    response = privilege_test_client.get("/api/v1/privileges/snapshots/snap-2025-01-15-001/export.json")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment; filename=" in response.headers.get("content-disposition", "")
    payload = response.json()
    assert payload["snapshot_id"] == "snap-2025-01-15-001"
    assert isinstance(payload.get("detail_items"), list)
    assert payload["total_items"] == len(payload.get("detail_items"))


def test_export_snapshot_csv(privilege_test_client: TestClient):
    response = privilege_test_client.get("/api/v1/privileges/snapshots/snap-2025-01-15-001/export.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=" in response.headers.get("content-disposition", "")
    body_lines = response.text.strip().splitlines()
    assert len(body_lines) >= 2
    header = body_lines[0].split(",")
    assert "privilege_scope_id" in header


def test_create_snapshot_org(privilege_test_client: TestClient):
    response = privilege_test_client.post(
        "/api/v1/privileges/snapshots",
        json={
            "target_scope": "org",
            "org_id": "acme",
            "notes": "automation-test",
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["target_scope"] == "org"
    assert payload["summary"]["users"] >= 1


def test_org_requires_admin_claim(privilege_test_client: TestClient):
    previous = fastapi_app.dependency_overrides[get_auth_principal]
    fastapi_app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="user",
        user_id=2,
        username="Viewer User",
        roles=["viewer"],
        permissions=[],
        is_admin=False,
        subject="user:2",
    )
    try:
        response = privilege_test_client.get("/api/v1/privileges/org")
        assert response.status_code == 403
        assert "Administrator privileges required" in response.json()["detail"]
    finally:
        fastapi_app.dependency_overrides[get_auth_principal] = previous


def test_self_requires_user_principal(privilege_test_client: TestClient):
    previous = fastapi_app.dependency_overrides[get_auth_principal]
    fastapi_app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="service",
        user_id=None,
        username=None,
        roles=[],
        permissions=[],
        is_admin=False,
        subject="service:watchlist-worker",
    )
    try:
        response = privilege_test_client.get("/api/v1/privileges/self")
        assert response.status_code == 403
        assert "User principal required" in response.json()["detail"]
    finally:
        fastapi_app.dependency_overrides[get_auth_principal] = previous
