from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tldw_profile_core import (
    ActorType,
    ProfileProposal,
    ProfileProvenance,
    ProposalOperation,
    ProvenanceSource,
)

from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import (
    get_personal_context_service,
)
from tldw_Server_API.app.api.v1.endpoints.personal_context import router
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)

pytestmark = pytest.mark.unit


def _ids():
    counters: defaultdict[str, int] = defaultdict(int)

    def issue(label: str) -> str:
        counters[label] += 1
        return f"{label}-{counters[label]}"

    return issue


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    service = PersonalContextService(
        PersonalContextRepository(PersonalizationDB(str(tmp_path / "personalization.db"))),
        clock=lambda: datetime(2026, 8, 30, 19, 0, tzinfo=UTC),
        id_factory=_ids(),
        workspace_access=lambda workspace_id: workspace_id == "workspace-owned",
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personal-context")
    app.dependency_overrides[get_personal_context_service] = lambda: service
    with TestClient(app) as client:
        yield client, service


def _record_body(scope_id: str, value: str = "concise") -> dict:
    return {
        "scope_id": scope_id,
        "payload": {
            "schema_version": 1,
            "kind": "preference",
            "subject": "response.detail",
            "polarity": "like",
            "value": value,
        },
        "semantic_key": {
            "namespace": "preference",
            "subject": "response.detail",
        },
        "controls": {
            "sync_mode": "syncable",
            "agent_visibility": "agent_visible",
        },
    }


def test_status_manifest_and_strict_profile_creation(api) -> None:
    client, _service = api
    assert client.get("/api/v1/personal-context/status").json()["state"] == "absent"

    invalid = client.post(
        "/api/v1/personal-context/manifest",
        json={"runtime_enabled": False, "unknown": True},
    )
    assert invalid.status_code == 422

    created = client.post(
        "/api/v1/personal-context/manifest",
        json={"runtime_enabled": False},
    )
    assert created.status_code == 201
    assert created.json()["revision"] == 0
    assert client.get("/api/v1/personal-context/status").json()["state"] == "disabled"
    assert client.get("/api/v1/personal-context/manifest").status_code == 200


def test_profile_creation_can_atomically_enable_server_runtime(api) -> None:
    client, _service = api

    created = client.post(
        "/api/v1/personal-context/manifest",
        json={"runtime_enabled": True},
    )

    assert created.status_code == 201
    assert client.get("/api/v1/personal-context/status").json()["state"] == "available"
    runtime = client.get("/api/v1/personal-context/runtime").json()
    assert runtime["enabled"] is True
    assert runtime["version_id"] is not None


def test_record_routes_are_bounded_and_conflicts_are_typed(api) -> None:
    client, _service = api
    client.post("/api/v1/personal-context/manifest", json={"runtime_enabled": False})
    scope_id = client.get("/api/v1/personal-context/scopes").json()["items"][0]["scope_id"]

    created = client.post(
        "/api/v1/personal-context/records",
        json=_record_body(scope_id),
    )
    assert created.status_code == 201
    record = created.json()

    no_op = client.patch(
        f"/api/v1/personal-context/records/{record['record_id']}",
        json={"expected_version_id": record["version_id"]},
    )
    assert no_op.status_code == 422

    listed = client.get(
        "/api/v1/personal-context/records",
        params={"q": "concise"},
    )
    assert listed.status_code == 200
    assert listed.json()["limit"] == 5
    assert len(listed.json()["items"]) == 1
    assert client.get("/api/v1/personal-context/records", params={"limit": 21}).status_code == 422

    stale = client.patch(
        f"/api/v1/personal-context/records/{record['record_id']}",
        json={
            "expected_version_id": "stale",
            "payload": _record_body(scope_id, "structured")["payload"],
        },
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "profile_version_conflict"

    updated = client.patch(
        f"/api/v1/personal-context/records/{record['record_id']}",
        json={
            "expected_version_id": record["version_id"],
            "payload": _record_body(scope_id, "structured")["payload"],
        },
    )
    assert updated.status_code == 200
    assert updated.json()["payload"]["value"] == "structured"


def test_runtime_export_and_purge_routes_require_explicit_confirmation(api) -> None:
    client, _service = api
    client.post("/api/v1/personal-context/manifest", json={"runtime_enabled": False})

    runtime = client.patch(
        "/api/v1/personal-context/runtime",
        json={"enabled": True, "expected_version_id": None},
    )
    assert runtime.status_code == 200
    assert runtime.json()["enabled"] is True

    refused_export = client.post(
        "/api/v1/personal-context/export",
        json={"mode": "plaintext", "confirmation": "yes"},
    )
    assert refused_export.status_code == 422
    exported = client.post(
        "/api/v1/personal-context/export",
        json={"mode": "plaintext", "confirmation": "EXPORT PLAINTEXT"},
    )
    assert exported.status_code == 200
    assert exported.json()["mode"] == "plaintext"

    local_copy = client.post(
        "/api/v1/personal-context/purge",
        json={
            "mode": "local_copy",
            "confirmation": "DELETE EVERYWHERE",
            "expected_purge_generation": 0,
        },
    )
    assert local_copy.status_code == 409
    assert local_copy.json()["detail"]["code"] == "server_local_copy_unsupported"

    purged = client.post(
        "/api/v1/personal-context/purge",
        json={
            "mode": "everywhere",
            "confirmation": "DELETE EVERYWHERE",
            "expected_purge_generation": 0,
        },
    )
    assert purged.status_code == 200
    assert purged.json()["purge_generation"] == 1

    fenced = client.post(
        "/api/v1/personal-context/scopes/workspace",
        json={"workspace_id": "workspace-owned", "label": "Resurrection attempt"},
    )
    assert fenced.status_code == 409
    assert fenced.json()["detail"]["code"] == "profile_purge_pending"


def test_payload_over_16_kib_is_rejected(api) -> None:
    client, _service = api
    client.post("/api/v1/personal-context/manifest", json={"runtime_enabled": False})
    scope_id = client.get("/api/v1/personal-context/scopes").json()["items"][0]["scope_id"]
    body = _record_body(scope_id, "x" * (16 * 1024))

    response = client.post("/api/v1/personal-context/records", json=body)

    assert response.status_code == 422


def test_proposal_conflicts_and_pagination_are_bounded(api) -> None:
    client, service = api
    client.post("/api/v1/personal-context/manifest", json={"runtime_enabled": False})
    manifest = service.get_manifest()
    scope = service.list_scopes()[0]
    created_at = datetime(2026, 8, 30, 19, 0, tzinfo=UTC)
    proposed = service.build_manual_record(**_record_body(scope.scope_id, "proposal one"))
    first = ProfileProposal(
        proposal_id="proposal-api-a",
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=proposed,
        provenance=ProfileProvenance(
            source=ProvenanceSource.AGENT,
            actor=ActorType.AGENT,
            reason_code="conversation_learning",
        ),
        confidence=0.8,
        created_at=created_at,
        expires_at=created_at + timedelta(days=90),
    )

    assert (
        client.post(
            "/api/v1/personal-context/proposals",
            json=first.model_dump(mode="json"),
        ).status_code
        == 201
    )
    duplicate = client.post(
        "/api/v1/personal-context/proposals",
        json=first.model_dump(mode="json"),
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "profile_version_conflict"

    second_record = service.build_manual_record(**_record_body(scope.scope_id, "proposal two"))
    second = first.model_copy(
        update={
            "proposal_id": "proposal-api-b",
            "proposed_record": second_record,
        }
    )
    assert (
        client.post(
            "/api/v1/personal-context/proposals",
            json=second.model_dump(mode="json"),
        ).status_code
        == 201
    )

    page = client.get(
        "/api/v1/personal-context/proposals",
        params={"limit": 1, "offset": 1},
    )
    assert page.status_code == 200
    assert page.json()["limit"] == 1
    assert page.json()["offset"] == 1
    assert page.json()["items"][0]["proposal_id"] == second.proposal_id
    assert (
        client.get(
            "/api/v1/personal-context/proposals",
            params={"offset": 1_001},
        ).status_code
        == 422
    )


def test_personal_context_router_is_registered_in_canonical_groups() -> None:
    from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    content = {spec.name for spec in iter_content_router_specs()}
    minimal = {spec.name for spec in iter_minimal_optional_router_specs()}
    assert "personal-context" in content
    assert "personal-context" in minimal
