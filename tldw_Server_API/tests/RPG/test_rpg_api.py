import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import AuthPrincipal, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import rpg as rpg_endpoint
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.PrivilegeCatalog.test_endpoint_scope_catalog_sync import load_catalog_scope_ids


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _headers(**extra):
    return {"X-API-KEY": get_settings().SINGLE_USER_API_KEY, **extra}


def _create_campaign_and_session(client: TestClient, prefix: str) -> tuple[int, int]:
    campaign = client.post(
        "/api/v1/rpg/campaigns",
        headers=_headers(**{"Idempotency-Key": f"{prefix}-campaign"}),
        json={"title": f"{prefix} Campaign", "default_adapter_key": "fate"},
    )
    assert campaign.status_code == 201  # nosec B101
    campaign_id = campaign.json()["id"]

    session = client.post(
        f"/api/v1/rpg/campaigns/{campaign_id}/sessions",
        headers=_headers(**{"Idempotency-Key": f"{prefix}-session"}),
        json={"title": f"{prefix} Opening", "adapter_key": "fate"},
    )
    assert session.status_code == 201  # nosec B101
    return campaign_id, session.json()["id"]


def _route_permissions(method: str, path: str) -> set[str]:
    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == path and method.upper() in route.methods
    )
    permissions: set[str] = set()
    for dependency in route.dependant.dependencies:
        call = dependency.call
        closure = getattr(call, "__closure__", None) or ()
        for cell in closure:
            value = cell.cell_contents
            if isinstance(value, list):
                permissions.update(str(item) for item in value)
            elif isinstance(value, str) and "." in value:
                permissions.add(value)
    return permissions


def _route_dependency_calls(method: str, path: str) -> set[object]:
    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == path and method.upper() in route.methods
    )
    calls: set[object] = set()

    def collect(dependant) -> None:
        for dependency in getattr(dependant, "dependencies", []):
            calls.add(dependency.call)
            collect(dependency)

    collect(route.dependant)
    return calls


def test_rpg_endpoint_scopes_are_cataloged():
    catalog_path = _REPO_ROOT / "tldw_Server_API" / "Config_Files" / "privilege_catalog.yaml"
    catalog_ids = load_catalog_scope_ids(catalog_path)
    expected_ids = {
        rpg_endpoint.RPG_RULES_READ,
        rpg_endpoint.RPG_CAMPAIGNS_READ,
        rpg_endpoint.RPG_CAMPAIGNS_MANAGE,
        rpg_endpoint.RPG_SESSIONS_READ,
        rpg_endpoint.RPG_SESSIONS_MANAGE,
        rpg_endpoint.RPG_PROPOSALS_REVIEW,
        rpg_endpoint.MEDIA_READ,
    }

    assert expected_ids <= catalog_ids  # nosec B101


def test_campaign_rules_pack_refs_get_requires_campaign_read_and_media_read():
    permissions = _route_permissions("GET", "/api/v1/rpg/campaigns/{campaign_id}/rules-packs")

    assert {rpg_endpoint.RPG_CAMPAIGNS_READ, rpg_endpoint.MEDIA_READ} <= permissions  # nosec B101


def test_campaign_rules_pack_refs_put_requires_campaign_manage_and_media_read():
    permissions = _route_permissions("PUT", "/api/v1/rpg/campaigns/{campaign_id}/rules-packs")

    assert {rpg_endpoint.RPG_CAMPAIGNS_MANAGE, rpg_endpoint.MEDIA_READ} <= permissions  # nosec B101


def test_session_rules_pack_refs_get_requires_session_read_and_media_read():
    permissions = _route_permissions("GET", "/api/v1/rpg/sessions/{session_id}/rules-packs")

    assert {rpg_endpoint.RPG_SESSIONS_READ, rpg_endpoint.MEDIA_READ} <= permissions  # nosec B101


def test_session_rules_pack_refs_put_requires_session_manage_and_media_read():
    permissions = _route_permissions("PUT", "/api/v1/rpg/sessions/{session_id}/rules-packs")

    assert {rpg_endpoint.RPG_SESSIONS_MANAGE, rpg_endpoint.MEDIA_READ} <= permissions  # nosec B101


def test_lookup_requires_rules_read_and_media_read():
    permissions = _route_permissions("POST", "/api/v1/rpg/sessions/{session_id}/rules/lookup")

    assert {rpg_endpoint.RPG_RULES_READ, rpg_endpoint.MEDIA_READ} <= permissions  # nosec B101


def test_non_rules_pack_routes_do_not_open_media_databases():
    calls = _route_dependency_calls("GET", "/api/v1/rpg/rules/adapters")

    assert get_media_db_for_user not in calls  # nosec B101
    assert get_collections_db_for_user not in calls  # nosec B101


def test_endpoint_scope_catalog_includes_rules_pack_routes():
    catalog_path = _REPO_ROOT / "tldw_Server_API" / "Config_Files" / "privilege_catalog.yaml"
    catalog_ids = load_catalog_scope_ids(catalog_path)
    for method, path in (
        ("GET", "/api/v1/rpg/campaigns/{campaign_id}/rules-packs"),
        ("PUT", "/api/v1/rpg/campaigns/{campaign_id}/rules-packs"),
        ("GET", "/api/v1/rpg/sessions/{session_id}/rules-packs"),
        ("PUT", "/api/v1/rpg/sessions/{session_id}/rules-packs"),
    ):
        assert _route_permissions(method, path) <= catalog_ids  # nosec B101


def test_rpg_adapters_endpoint_lists_default_adapters():
    client = TestClient(app)

    response = client.get("/api/v1/rpg/rules/adapters", headers=_headers())

    assert response.status_code == 200  # nosec B101
    keys = [item["adapter_key"] for item in response.json()["adapters"]]
    assert keys == ["dnd5e_srd", "fate", "pf2e"]  # nosec B101


def test_media_item_validator_rejects_owner_mismatch():
    class FakeMediaDb:
        client_id = "1"

        def get_media_by_id(self, media_id, *, include_deleted, include_trash):
            assert include_deleted is False  # nosec B101
            assert include_trash is False  # nosec B101
            return {"id": media_id, "title": "Other User Rules", "owner_user_id": 2}

    validator = rpg_endpoint.RPGRulesSourceValidator(
        media_db=FakeMediaDb(),
        collections_db=SimpleNamespace(),
    )

    result = asyncio.run(validator.validate_media_item(owner_user_id=1, media_id=42))

    assert result.readable is False  # nosec B101
    assert result.ready_media_ids == []  # nosec B101


def test_collection_validator_filters_unreadable_ready_media_ids():
    class FakeMediaDb:
        client_id = "1"

        def __init__(self):
            self.calls = []

        def get_media_by_id(self, media_id, *, include_deleted, include_trash):
            self.calls.append((media_id, include_deleted, include_trash))
            rows = {
                10: {"id": 10, "title": "Readable Rules", "owner_user_id": 1},
                12: {"id": 12, "title": "Other User Rules", "owner_user_id": 2},
            }
            return rows.get(media_id)

    class FakeCollectionsDb:
        def get_media_collection(self, collection_id):
            assert collection_id == 3  # nosec B101
            return SimpleNamespace(
                name="Mixed Collection",
                items=[
                    SimpleNamespace(media_id=10, status="completed"),
                    SimpleNamespace(media_id=11, status="completed"),
                    SimpleNamespace(media_id=12, status="skipped_existing"),
                    SimpleNamespace(media_id=13, status="failed"),
                    SimpleNamespace(media_id=None, status="completed"),
                ],
            )

    media_db = FakeMediaDb()
    validator = rpg_endpoint.RPGRulesSourceValidator(
        media_db=media_db,
        collections_db=FakeCollectionsDb(),
    )

    result = asyncio.run(validator.validate_media_collection(owner_user_id=1, collection_id=3))

    assert result.readable is True  # nosec B101
    assert result.display_name == "Mixed Collection"  # nosec B101
    assert result.ready_media_ids == [10]  # nosec B101
    assert media_db.calls == [(10, False, False), (11, False, False), (12, False, False)]  # nosec B101


def test_create_campaign_session_and_record_user_event():
    client = TestClient(app)

    _, session_id = _create_campaign_and_session(client, "api-main")

    event_response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-main-event"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "note.added",
                    "event_payload": {"note_id": "n1", "text": "At the docks"},
                }
            ],
        },
    )

    assert event_response.status_code == 200  # nosec B101
    payload = event_response.json()
    assert payload["committed_events"][0]["sequence_number"] == 1  # nosec B101
    assert payload["proposal"] is None  # nosec B101


def test_create_campaign_requires_idempotency_key():
    client = TestClient(app)

    response = client.post(
        "/api/v1/rpg/campaigns",
        headers=_headers(),
        json={"title": "Missing Header", "default_adapter_key": "fate"},
    )

    assert response.status_code == 422  # nosec B101


def test_record_events_rejects_stale_expected_sequence():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-stale")

    first = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-stale-event-1"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "note.added",
                    "event_payload": {"note_id": "n1", "text": "First note"},
                }
            ],
        },
    )
    assert first.status_code == 200  # nosec B101

    stale = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-stale-event-2"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "note.added",
                    "event_payload": {"note_id": "n2", "text": "Second note"},
                }
            ],
        },
    )

    assert stale.status_code == 409  # nosec B101
    assert stale.json()["detail"] == "stale_event_sequence"  # nosec B101


def test_rules_lookup_and_context_endpoints():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-rules")
    scene = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-rules-scene"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "scene.updated",
                    "event_payload": {"scene_id": "scene-1", "summary": "Lanterns in the rain"},
                }
            ],
        },
    )
    assert scene.status_code == 200  # nosec B101

    lookup = client.post(
        f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
        headers=_headers(),
        json={"query": "stress"},
    )

    assert lookup.status_code == 200  # nosec B101
    lookup_payload = lookup.json()
    assert lookup_payload["query"] == "stress"  # nosec B101
    assert lookup_payload["diagnostics"]["bundled_policy"] == "citations_only"  # nosec B101
    assert lookup_payload["diagnostics"]["result_mode"] == "citation_index"  # nosec B101
    assert all(item["text"] == "" for item in lookup_payload["results"])  # nosec B101

    context = client.post(
        f"/api/v1/rpg/sessions/{session_id}/context",
        headers=_headers(),
        json={"query": "stress", "max_chars": 1000},
    )

    assert context.status_code == 200  # nosec B101
    context_payload = context.json()
    assert "Lanterns in the rain" in context_payload["text"]  # nosec B101
    assert context_payload["diagnostics"]["rules_result_count"] >= 1  # nosec B101


def test_replace_campaign_rules_pack_refs_returns_version_and_refs():
    client = TestClient(app)
    campaign_id, _ = _create_campaign_and_session(client, "api-campaign-rules-packs")

    response = client.put(
        f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs",
        headers=_headers(),
        json={
            "expected_version": 1,
            "idempotency_key": "api-campaign-rules-packs-replace",
            "refs": [
                {
                    "source_type": "media_item",
                    "source_id": 7,
                    "display_name": "Disabled Rules",
                    "enabled": False,
                }
            ],
        },
    )

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["version"] == 2  # nosec B101
    assert payload["replayed"] is False  # nosec B101
    assert payload["refs"][0]["ref_id"] == "media_item:7"  # nosec B101
    assert payload["refs"][0]["display_name"] == "Disabled Rules"  # nosec B101

    listed = client.get(f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs", headers=_headers())
    assert listed.status_code == 200  # nosec B101
    assert listed.json()["version"] == 2  # nosec B101
    assert listed.json()["refs"] == payload["refs"]  # nosec B101


def test_replace_session_rules_pack_refs_returns_version_and_refs():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-session-rules-packs")

    response = client.put(
        f"/api/v1/rpg/sessions/{session_id}/rules-packs",
        headers=_headers(),
        json={
            "expected_version": 1,
            "idempotency_key": "api-session-rules-packs-replace",
            "refs": [
                {
                    "source_type": "media_collection",
                    "source_id": 3,
                    "display_name": "Disabled Collection",
                    "enabled": False,
                }
            ],
        },
    )

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["version"] == 2  # nosec B101
    assert payload["replayed"] is False  # nosec B101
    assert payload["refs"][0]["ref_id"] == "media_collection:3"  # nosec B101

    listed = client.get(f"/api/v1/rpg/sessions/{session_id}/rules-packs", headers=_headers())
    assert listed.status_code == 200  # nosec B101
    assert listed.json()["version"] == 2  # nosec B101
    assert listed.json()["refs"] == payload["refs"]  # nosec B101


def test_replace_rules_pack_refs_rejects_stale_version_with_409():
    client = TestClient(app)
    campaign_id, _ = _create_campaign_and_session(client, "api-rules-packs-stale")

    response = client.put(
        f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs",
        headers=_headers(),
        json={
            "expected_version": 999,
            "idempotency_key": "api-rules-packs-stale-replace",
            "refs": [{"source_type": "media_item", "source_id": 7, "enabled": False}],
        },
    )

    assert response.status_code == 409  # nosec B101
    assert response.json()["detail"] == "stale_rules_pack_ref_version"  # nosec B101


def test_replace_rules_pack_refs_replays_idempotency_key():
    client = TestClient(app)
    campaign_id, _ = _create_campaign_and_session(client, "api-rules-packs-replay")
    request = {
        "expected_version": 1,
        "idempotency_key": "api-rules-packs-replay-replace",
        "refs": [{"source_type": "media_item", "source_id": 7, "enabled": False}],
    }

    first = client.put(
        f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs",
        headers=_headers(),
        json=request,
    )
    second = client.put(
        f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs",
        headers=_headers(),
        json=request,
    )

    assert first.status_code == 200  # nosec B101
    assert second.status_code == 200  # nosec B101
    assert second.json()["replayed"] is True  # nosec B101
    assert second.json()["version"] == first.json()["version"]  # nosec B101
    assert second.json()["refs"] == first.json()["refs"]  # nosec B101


def test_replace_rules_pack_refs_rejects_non_boolean_enabled():
    client = TestClient(app)
    campaign_id, _ = _create_campaign_and_session(client, "api-rules-packs-enabled")

    response = client.put(
        f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs",
        headers=_headers(),
        json={
            "expected_version": 1,
            "idempotency_key": "api-rules-packs-enabled-replace",
            "refs": [{"source_type": "media_item", "source_id": 7, "enabled": "false"}],
        },
    )

    assert response.status_code == 422  # nosec B101


def test_rules_lookup_accepts_lookup_mode():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-lookup-mode")

    response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
        headers=_headers(),
        json={"query": "stress", "mode": "lookup"},
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["query"] == "stress"  # nosec B101


def test_rules_lookup_accepts_answer_mode():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-answer-mode")

    response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
        headers=_headers(),
        json={"query": "stress", "mode": "answer"},
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["query"] == "stress"  # nosec B101


def test_rules_lookup_rejects_unknown_mode():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-bad-mode")

    response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
        headers=_headers(),
        json={"query": "stress", "mode": "summarize"},
    )

    assert response.status_code == 422  # nosec B101


def test_rules_lookup_denies_principal_missing_media_read():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-lookup-no-media")

    async def no_media_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            roles=[],
            permissions=[rpg_endpoint.RPG_RULES_READ],
            is_admin=False,
        )

    app.dependency_overrides[get_auth_principal] = no_media_principal
    try:
        response = client.post(
            f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
            headers=_headers(),
            json={"query": "stress"},
        )
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)

    assert response.status_code == 403  # nosec B101


def test_rules_pack_refs_denies_principal_missing_media_read():
    client = TestClient(app)
    campaign_id, _ = _create_campaign_and_session(client, "api-rules-pack-no-media")

    async def no_media_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            roles=[],
            permissions=[rpg_endpoint.RPG_CAMPAIGNS_READ],
            is_admin=False,
        )

    app.dependency_overrides[get_auth_principal] = no_media_principal
    try:
        response = client.get(
            f"/api/v1/rpg/campaigns/{campaign_id}/rules-packs",
            headers=_headers(),
        )
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)

    assert response.status_code == 403  # nosec B101


def test_context_endpoint_rejects_tiny_budget():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-context-budget")

    response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/context",
        headers=_headers(),
        json={"max_chars": 999},
    )

    assert response.status_code == 422  # nosec B101
