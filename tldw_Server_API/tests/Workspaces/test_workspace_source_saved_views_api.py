"""Contract tests for workspace source saved-view API routes."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.schemas import workspace_schemas
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    SOURCE_VIEW_LIMIT_REACHED,
    SOURCE_VIEW_MAX_COUNT,
    CharactersRAGDB,
    WorkspaceSourceSavedViewConflictError,
)

pytestmark = pytest.mark.integration

RESPONSE_KEYS = {
    "id",
    "workspace_id",
    "name",
    "schema_version",
    "state",
    "valid",
    "invalid_reason",
    "version",
    "created_at",
    "updated_at",
}

DEFAULT_STATE = {
    "type_filters": [],
    "status_filters": [],
    "review_state_filters": [],
    "lifecycle_state_filters": [],
    "date_field": "added_at",
    "date_from": None,
    "date_to": None,
    "require_url": False,
    "require_file_size": False,
    "require_duration": False,
    "require_page_count": False,
    "file_size_min": None,
    "file_size_max": None,
    "duration_min": None,
    "duration_max": None,
    "page_count_min": None,
    "page_count_max": None,
    "sort": "manual",
}


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(db_path=str(tmp_path / "saved-views.db"), client_id="1")
    database.upsert_workspace("ws-1", "Workspace One")
    database.upsert_workspace("ws-2", "Workspace Two")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture
def app() -> FastAPI:
    fastapi_app = FastAPI()
    fastapi_app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return fastapi_app


@pytest.fixture
def client(app: FastAPI, db: CharactersRAGDB) -> Iterator[TestClient]:
    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _create_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": "My view", "schema_version": 1, "state": {}}
    payload.update(overrides)
    return payload


def _post(client: TestClient, payload: dict[str, Any], workspace_id: str = "ws-1"):
    return client.post(f"/api/v1/workspaces/{workspace_id}/source-views", json=payload)


def _raw_state(db: CharactersRAGDB, workspace_id: str, view_id: str) -> str:
    return db.get_workspace_source_saved_view("1", workspace_id, view_id)["state_json"]


def test_create_list_patch_delete_and_canonicalize(client: TestClient, db: CharactersRAGDB) -> None:
    state = {
        "type_filters": ["text", "pdf", "text", "video"],
        "status_filters": ["error", "processing", "error"],
        "review_state_filters": ["reviewed", "unset", "reviewed"],
        "lifecycle_state_filters": ["unknown", "queued", "failed", "queued"],
        "date_field": "source_created_at",
        "date_from": "2024-02-29",
        "date_to": "2024-03-01",
        "file_size_min": 0,
        "file_size_max": 12.5,
        "sort": "page_count_desc",
    }

    created_response = _post(client, _create_payload(name="  My view  ", state=state))

    assert created_response.status_code == 201, created_response.text
    created = created_response.json()
    assert set(created) == RESPONSE_KEYS
    assert created["name"] == "My view"
    assert created["workspace_id"] == "ws-1"
    assert created["schema_version"] == 1
    assert created["valid"] is True
    assert created["invalid_reason"] is None
    assert created["version"] == 1
    assert created["state"] == {
        **DEFAULT_STATE,
        "type_filters": ["pdf", "video", "text"],
        "status_filters": ["processing", "error"],
        "review_state_filters": ["unset", "reviewed"],
        "lifecycle_state_filters": ["queued", "failed", "unknown"],
        "date_field": "source_created_at",
        "date_from": "2024-02-29",
        "date_to": "2024-03-01",
        "file_size_min": 0.0,
        "file_size_max": 12.5,
        "sort": "page_count_desc",
    }
    expected_json = json.dumps(
        created["state"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert _raw_state(db, "ws-1", created["id"]) == expected_json

    listed = client.get("/api/v1/workspaces/ws-1/source-views")
    assert listed.status_code == 200
    assert listed.json() == {"items": [created]}

    patched_response = client.patch(
        f"/api/v1/workspaces/ws-1/source-views/{created['id']}",
        json={
            "version": 1,
            "name": "Renamed",
            "schema_version": 1,
            "state": {"sort": "name_asc", "require_url": True},
        },
    )
    assert patched_response.status_code == 200, patched_response.text
    patched = patched_response.json()
    assert set(patched) == RESPONSE_KEYS
    assert patched["name"] == "Renamed"
    assert patched["version"] == 2
    assert patched["state"] == {**DEFAULT_STATE, "sort": "name_asc", "require_url": True}

    deleted = client.delete(f"/api/v1/workspaces/ws-1/source-views/{created['id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.delete(f"/api/v1/workspaces/ws-1/source-views/{created['id']}").status_code == 404


def test_v1_accepts_exact_enums_and_canonicalizes_in_declaration_order(client: TestClient) -> None:
    types = ["pdf", "video", "audio", "website", "document", "text"]
    statuses = ["processing", "ready", "error"]
    reviews = ["unset", "needs_review", "reviewed"]
    lifecycles = [
        "queued",
        "ingesting",
        "extracting",
        "chunking",
        "indexing",
        "queryable",
        "partially_queryable",
        "failed",
        "retrying",
        "missing_media",
        "blocked_by_permissions",
        "unknown",
    ]
    sorts = [
        "manual",
        "name_asc",
        "name_desc",
        "added_desc",
        "added_asc",
        "source_created_desc",
        "source_created_asc",
        "file_size_desc",
        "file_size_asc",
        "duration_desc",
        "duration_asc",
        "page_count_desc",
        "page_count_asc",
    ]
    state = {
        "type_filters": [*reversed(types), *types],
        "status_filters": list(reversed(statuses)),
        "review_state_filters": list(reversed(reviews)),
        "lifecycle_state_filters": list(reversed(lifecycles)),
        "date_field": "added_at",
        "sort": sorts[-1],
    }

    response = _post(client, _create_payload(state=state))

    assert response.status_code == 201, response.text
    canonical = response.json()["state"]
    assert canonical["type_filters"] == types
    assert canonical["status_filters"] == statuses
    assert canonical["review_state_filters"] == reviews
    assert canonical["lifecycle_state_filters"] == lifecycles
    assert canonical["sort"] == sorts[-1]
    properties = workspace_schemas.WorkspaceSourceSavedViewStateV1.model_json_schema()["properties"]
    assert properties["type_filters"]["items"]["enum"] == types
    assert properties["status_filters"]["items"]["enum"] == statuses
    assert properties["review_state_filters"]["items"]["enum"] == reviews
    assert properties["lifecycle_state_filters"]["items"]["enum"] == lifecycles
    assert properties["date_field"]["enum"] == ["added_at", "source_created_at"]
    assert properties["sort"]["enum"] == sorts


@pytest.mark.parametrize(
    ("payload", "method"),
    [
        ({"name": "x", "schema_version": True, "state": {}}, "post"),
        ({"name": "x", "schema_version": 0, "state": {}}, "post"),
        ({"name": "x", "schema_version": 2, "state": {}}, "post"),
        ({"name": "x", "state": {}}, "post"),
        ({"name": "x", "schema_version": 1}, "post"),
        ({"name": "x", "schema_version": 1, "state": {}, "extra": 1}, "post"),
        ({"name": "x", "schema_version": 1, "state": {"extra": 1}}, "post"),
        ({"name": " ", "schema_version": 1, "state": {}}, "post"),
        ({"name": "x" * 121, "schema_version": 1, "state": {}}, "post"),
        ({"name": "x", "schema_version": 1, "state": {"date_from": "2023-02-29"}}, "post"),
        (
            {
                "name": "x",
                "schema_version": 1,
                "state": {"date_from": "2024-02-02", "date_to": "2024-02-01"},
            },
            "post",
        ),
        ({"name": "x", "schema_version": 1, "state": {"file_size_min": True}}, "post"),
        ({"name": "x", "schema_version": 1, "state": {"duration_min": -0.1}}, "post"),
        ({"name": "x", "schema_version": 1, "state": {"duration_min": 10**400}}, "post"),
        (
            {
                "name": "x",
                "schema_version": 1,
                "state": {"page_count_min": 3, "page_count_max": 2},
            },
            "post",
        ),
        ({"name": "x", "schema_version": 1, "state": {"sort": "relevance"}}, "post"),
        ({"version": True, "name": "x"}, "patch"),
        ({"version": 0, "name": "x"}, "patch"),
        ({"version": -1, "name": "x"}, "patch"),
        ({"version": 2_147_483_648, "name": "x"}, "patch"),
        ({"name": "x"}, "patch"),
        ({"version": 1}, "patch"),
        ({"version": 1, "name": None}, "patch"),
        ({"version": 1, "state": None, "schema_version": 1}, "patch"),
        ({"version": 1, "schema_version": None, "state": {}}, "patch"),
        ({"version": 1, "state": {}}, "patch"),
        ({"version": 1, "schema_version": 1}, "patch"),
        ({"version": 1, "schema_version": True, "state": {}}, "patch"),
        ({"version": 1, "schema_version": 2, "state": {}}, "patch"),
        ({"version": 1, "name": "x", "extra": 1}, "patch"),
    ],
)
def test_strict_write_validation_returns_422(
    client: TestClient,
    db: CharactersRAGDB,
    payload: dict[str, Any],
    method: str,
) -> None:
    if method == "post":
        response = _post(client, payload)
    else:
        existing = db.create_workspace_source_saved_view(
            "1", "ws-1", name="Existing", schema_version=1, state_json="{}"
        )
        response = client.patch(
            f"/api/v1/workspaces/ws-1/source-views/{existing['id']}", json=payload
        )

    assert response.status_code == 422, response.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_state_model_rejects_non_finite_and_boolean_numbers(value: Any) -> None:
    with pytest.raises(ValidationError):
        workspace_schemas.WorkspaceSourceSavedViewStateV1(file_size_min=value)


def test_response_model_enforces_validity_invariants() -> None:
    common = {
        "id": "v1",
        "workspace_id": "ws-1",
        "name": "View",
        "schema_version": 1,
        "version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    with pytest.raises(ValidationError):
        workspace_schemas.WorkspaceSourceSavedViewResponse.model_validate(
            {**common, "state": None, "valid": True, "invalid_reason": None}
        )
    with pytest.raises(ValidationError):
        workspace_schemas.WorkspaceSourceSavedViewResponse.model_validate(
            {**common, "state": {}, "valid": False, "invalid_reason": "invalid_state"}
        )
    with pytest.raises(ValidationError):
        workspace_schemas.WorkspaceSourceSavedViewResponse.model_validate(
            {**common, "state": None, "valid": False, "invalid_reason": None}
        )


def test_duplicate_limit_and_version_conflicts_have_exact_detail(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _post(client, _create_payload(name="Duplicate")).json()
    duplicate = _post(client, _create_payload(name=" duplicate "))
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == {
        "code": "source_view_name_exists",
        "view_id": created["id"],
        "version": 1,
    }

    conflict = client.patch(
        f"/api/v1/workspaces/ws-1/source-views/{created['id']}",
        json={"version": 2, "name": "New name"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == {
        "code": "source_view_version_conflict",
        "view_id": created["id"],
        "current_version": 1,
    }

    def _limit(*args: Any, **kwargs: Any) -> None:
        raise WorkspaceSourceSavedViewConflictError(
            SOURCE_VIEW_LIMIT_REACHED, {"limit": SOURCE_VIEW_MAX_COUNT}
        )

    monkeypatch.setattr(db, "create_workspace_source_saved_view", _limit)
    limit = _post(client, _create_payload(name="Limit"), workspace_id="ws-2")
    assert limit.status_code == 409
    assert limit.json()["detail"] == {"code": "source_view_limit_reached", "limit": 100}


def test_missing_workspace_view_and_client_id_mismatch_are_404(
    client: TestClient,
    app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    missing_workspace = client.get("/api/v1/workspaces/missing/source-views")
    assert missing_workspace.status_code == 404
    assert missing_workspace.json() == {"detail": "Source view not found"}
    assert client.patch(
        "/api/v1/workspaces/ws-1/source-views/missing", json={"version": 1, "name": "x"}
    ).status_code == 404
    assert client.delete("/api/v1/workspaces/ws-1/source-views/missing").status_code == 404

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=2)
    assert client.get("/api/v1/workspaces/ws-1/source-views").status_code == 404
    assert _post(client, _create_payload(), workspace_id="ws-1").status_code == 404


def test_cross_owner_patch_and_delete_are_404_and_leave_view_unchanged(
    client: TestClient,
    app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    created = _post(client, _create_payload(name="Owner one")).json()
    original = db.get_workspace_source_saved_view("1", "ws-1", created["id"])
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=2)

    patched = client.patch(
        f"/api/v1/workspaces/ws-1/source-views/{created['id']}",
        json={"version": 1, "name": "Owner two"},
    )
    deleted = client.delete(f"/api/v1/workspaces/ws-1/source-views/{created['id']}")

    assert patched.status_code == 404
    assert patched.json() == {"detail": "Source view not found"}
    assert deleted.status_code == 404
    assert deleted.json() == {"detail": "Source view not found"}
    unchanged = db.get_workspace_source_saved_view("1", "ws-1", created["id"])
    assert unchanged == original


def test_views_are_isolated_by_workspace(client: TestClient) -> None:
    first = _post(client, _create_payload(name="First"), workspace_id="ws-1")
    second = _post(client, _create_payload(name="Second"), workspace_id="ws-2")
    assert first.status_code == second.status_code == 201

    assert [v["name"] for v in client.get("/api/v1/workspaces/ws-1/source-views").json()["items"]] == [
        "First"
    ]
    assert [v["name"] for v in client.get("/api/v1/workspaces/ws-2/source-views").json()["items"]] == [
        "Second"
    ]
    assert client.patch(
        f"/api/v1/workspaces/ws-2/source-views/{first.json()['id']}",
        json={"version": 1, "name": "Leak"},
    ).status_code == 404


def test_list_order_is_updated_desc_then_name_key_then_id(
    client: TestClient,
    db: CharactersRAGDB,
) -> None:
    ids = {}
    for name in ("Zulu", "beta", "Alpha"):
        row = db.create_workspace_source_saved_view(
            "1", "ws-1", name=name, schema_version=1, state_json="{}"
        )
        ids[name] = row["id"]
    with db.transaction() as conn:
        conn.execute(
            "UPDATE workspace_source_saved_views SET updated_at = ? WHERE id = ?",
            ("2026-01-02T00:00:00+00:00", ids["Zulu"]),
        )
        conn.execute(
            "UPDATE workspace_source_saved_views SET updated_at = ? WHERE id IN (?, ?)",
            ("2026-01-01T00:00:00+00:00", ids["beta"], ids["Alpha"]),
        )

    response = client.get("/api/v1/workspaces/ws-1/source-views")

    assert response.status_code == 200
    assert [row["id"] for row in response.json()["items"]] == [
        ids["Zulu"],
        ids["Alpha"],
        ids["beta"],
    ]


def test_invalid_rows_are_recoverable_and_unsupported_version_precedes_json_parse(
    client: TestClient,
    db: CharactersRAGDB,
) -> None:
    malformed = db.create_workspace_source_saved_view(
        "1", "ws-1", name="Malformed", schema_version=1, state_json="{not-json"
    )
    invalid = db.create_workspace_source_saved_view(
        "1",
        "ws-1",
        name="Invalid",
        schema_version=1,
        state_json='{"file_size_min":true}',
    )
    unsupported = db.create_workspace_source_saved_view(
        "1", "ws-1", name="Unsupported", schema_version=2, state_json="{not-json"
    )

    response = client.get("/api/v1/workspaces/ws-1/source-views")

    assert response.status_code == 200, response.text
    by_id = {item["id"]: item for item in response.json()["items"]}
    assert by_id[malformed["id"]]["invalid_reason"] == "invalid_json"
    assert by_id[invalid["id"]]["invalid_reason"] == "invalid_state"
    assert by_id[unsupported["id"]]["invalid_reason"] == "unsupported_schema_version"
    for item in by_id.values():
        assert set(item) == RESPONSE_KEYS
        assert item["state"] is None
        assert item["valid"] is False


def test_huge_integer_json_is_invalid_without_failing_other_list_rows(
    client: TestClient,
    db: CharactersRAGDB,
) -> None:
    valid = db.create_workspace_source_saved_view(
        "1", "ws-1", name="Valid", schema_version=1, state_json="{}"
    )
    huge_integer = db.create_workspace_source_saved_view(
        "1",
        "ws-1",
        name="Huge integer",
        schema_version=1,
        state_json='{"file_size_min":' + ("9" * 5_000) + "}",
    )

    response = client.get("/api/v1/workspaces/ws-1/source-views")

    assert response.status_code == 200, response.text
    by_id = {item["id"]: item for item in response.json()["items"]}
    assert by_id[valid["id"]]["valid"] is True
    assert by_id[huge_integer["id"]]["valid"] is False
    assert by_id[huge_integer["id"]]["invalid_reason"] == "invalid_json"


def test_saved_view_handlers_are_sync_for_threadpool_offload() -> None:
    handlers = (
        workspaces_endpoint.list_source_saved_views,
        workspaces_endpoint.create_source_saved_view,
        workspaces_endpoint.update_source_saved_view,
        workspaces_endpoint.delete_source_saved_view,
    )

    assert all(not inspect.iscoroutinefunction(handler) for handler in handlers)


def test_saved_view_openapi_matches_patch_response_and_error_contracts(app: FastAPI) -> None:
    schema = app.openapi()
    components = schema["components"]["schemas"]

    patch_schema = components["WorkspaceSourceSavedViewPatchRequest"]
    patch_refs = {branch["$ref"].rsplit("/", 1)[-1] for branch in patch_schema["anyOf"]}
    assert patch_refs == {
        "WorkspaceSourceSavedViewRenamePatch",
        "WorkspaceSourceSavedViewStatePatch",
        "WorkspaceSourceSavedViewCombinedPatch",
    }
    for component_name in patch_refs:
        branch = components[component_name]
        required = set(branch["required"])
        assert "version" in required
        assert required & {"name", "state"}
        if "state" in required:
            assert "schema_version" in required
        for operation in {"name", "state", "schema_version"} & set(branch["properties"]):
            assert {"type": "null"} not in branch["properties"][operation].get("anyOf", [])

    response_schema = components["WorkspaceSourceSavedViewResponse"]
    response_refs = {branch["$ref"].rsplit("/", 1)[-1] for branch in response_schema["anyOf"]}
    assert response_refs == {
        "WorkspaceSourceSavedViewValidResponse",
        "WorkspaceSourceSavedViewInvalidResponse",
    }
    valid_response = components["WorkspaceSourceSavedViewValidResponse"]
    invalid_response = components["WorkspaceSourceSavedViewInvalidResponse"]
    assert valid_response["properties"]["valid"]["const"] is True
    assert valid_response["properties"]["state"]["$ref"].endswith(
        "/WorkspaceSourceSavedViewStateV1"
    )
    assert valid_response["properties"]["invalid_reason"]["type"] == "null"
    assert invalid_response["properties"]["valid"]["const"] is False
    assert invalid_response["properties"]["state"]["type"] == "null"
    assert invalid_response["properties"]["invalid_reason"]["enum"] == [
        "invalid_json",
        "invalid_state",
        "unsupported_schema_version",
    ]

    conflict = components["WorkspaceSourceSavedViewConflictResponse"]["properties"]["detail"]
    assert set(conflict["discriminator"]["mapping"]) == {
        "source_view_name_exists",
        "source_view_limit_reached",
        "source_view_version_conflict",
    }
    conflict_refs = {branch["$ref"].rsplit("/", 1)[-1] for branch in conflict["oneOf"]}
    assert conflict_refs == {
        "WorkspaceSourceSavedViewNameExistsDetail",
        "WorkspaceSourceSavedViewLimitReachedDetail",
        "WorkspaceSourceSavedViewVersionConflictDetail",
    }

    paths = schema["paths"]
    collection_responses = paths["/api/v1/workspaces/{workspace_id}/source-views"]
    item_responses = paths["/api/v1/workspaces/{workspace_id}/source-views/{view_id}"]
    assert set(collection_responses["get"]["responses"]) >= {"200", "404"}
    assert set(collection_responses["post"]["responses"]) >= {"201", "404", "409"}
    assert set(item_responses["patch"]["responses"]) >= {"200", "404", "409"}
    assert set(item_responses["delete"]["responses"]) >= {"204", "404"}
    not_found_ref = "#/components/schemas/WorkspaceSourceSavedViewNotFoundResponse"
    conflict_ref = "#/components/schemas/WorkspaceSourceSavedViewConflictResponse"
    for operation in (
        collection_responses["get"],
        collection_responses["post"],
        item_responses["patch"],
        item_responses["delete"],
    ):
        assert operation["responses"]["404"]["content"]["application/json"]["schema"] == {
            "$ref": not_found_ref
        }
    for operation in (collection_responses["post"], item_responses["patch"]):
        assert operation["responses"]["409"]["content"]["application/json"]["schema"] == {
            "$ref": conflict_ref
        }


def test_invalid_view_reset_is_an_atomic_ordinary_patch(
    client: TestClient,
    db: CharactersRAGDB,
) -> None:
    invalid = db.create_workspace_source_saved_view(
        "1", "ws-1", name="Reset me", schema_version=2, state_json="{}"
    )

    response = client.patch(
        f"/api/v1/workspaces/ws-1/source-views/{invalid['id']}",
        json={"version": 1, "schema_version": 1, "state": {}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["state"] == DEFAULT_STATE
    assert response.json()["valid"] is True
    assert response.json()["invalid_reason"] is None
    assert response.json()["version"] == 2


class _OwnerRecordingDB:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        return {"id": workspace_id, "client_id": "7"}

    def list_workspace_source_saved_views(self, *args: Any) -> list[dict[str, Any]]:
        self.calls.append(("list", *args))
        return []

    def create_workspace_source_saved_view(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create", *args, kwargs))
        return {
            "id": "view-1",
            "workspace_id": "ws",
            "name": kwargs["name"],
            "schema_version": kwargs["schema_version"],
            "state_json": kwargs["state_json"],
            "version": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }

    def update_workspace_source_saved_view(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("update", *args, kwargs))
        row = self.create_workspace_source_saved_view(
            args[0], args[1], name=kwargs.get("name", "View"), schema_version=1, state_json="{}"
        )
        row["id"] = args[2]
        row["version"] = kwargs["expected_version"] + 1
        return row

    def delete_workspace_source_saved_view(self, *args: Any) -> None:
        self.calls.append(("delete", *args))


def test_every_db_call_receives_authenticated_owner(app: FastAPI) -> None:
    async def _allow_rate_limit() -> None:
        return None

    recording_db = _OwnerRecordingDB()
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: recording_db
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as api:
        assert api.get("/api/v1/workspaces/ws/source-views").status_code == 200
        assert _post(api, _create_payload(), workspace_id="ws").status_code == 201
        assert api.patch(
            "/api/v1/workspaces/ws/source-views/view-1",
            json={"version": 1, "name": "Renamed"},
        ).status_code == 200
        assert api.delete("/api/v1/workspaces/ws/source-views/view-1").status_code == 204

    assert recording_db.calls[0] == ("list", "7", "ws")
    assert recording_db.calls[1][0:3] == ("create", "7", "ws")
    assert recording_db.calls[2][0:4] == ("update", "7", "ws", "view-1")
    assert recording_db.calls[-1] == ("delete", "7", "ws", "view-1")
