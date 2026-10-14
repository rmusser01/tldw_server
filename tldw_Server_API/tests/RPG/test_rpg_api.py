import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import AuthPrincipal, get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import rpg as rpg_endpoint
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RPG.context import SessionContext
from tldw_Server_API.app.core.RPG.rules.answering import RulesAnswerOptions
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupResult
from tldw_Server_API.app.core.RPG.service import RPGService
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.PrivilegeCatalog.test_endpoint_scope_catalog_sync import load_catalog_scope_ids

pytestmark = pytest.mark.integration

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
        for route in iter_served_routes(app.routes)
        if isinstance(route.route, APIRoute) and route.path == path and method.upper() in route.methods
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
        for route in iter_served_routes(app.routes)
        if isinstance(route.route, APIRoute) and route.path == path and method.upper() in route.methods
    )
    calls: set[object] = set()

    def collect(dependant) -> None:
        for dependency in getattr(dependant, "dependencies", []):
            calls.add(dependency.call)
            collect(dependency)

    collect(route.dependant)
    return calls


def test_rules_service_closes_rag_retriever(monkeypatch):
    class FakeRetriever:
        def __init__(self, *args, **kwargs) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_retriever = FakeRetriever()
    monkeypatch.setattr(rpg_endpoint, "MultiDatabaseRetriever", lambda *args, **kwargs: fake_retriever)
    monkeypatch.setattr(rpg_endpoint, "get_media_db_path_for_rag", lambda media_db: "/tmp/media.db")
    monkeypatch.setattr(rpg_endpoint.RPGRepository, "initialized", staticmethod(lambda db: object()))

    dependency = rpg_endpoint._rules_service(
        db=object(),
        media_db=object(),
        collections_db=object(),
        current_user=SimpleNamespace(id_int=42),
    )
    service = next(dependency)

    dependency.close()

    assert service.owner_user_id == 42  # nosec B101
    assert fake_retriever.closed is True  # nosec B101


def test_rpg_endpoint_scopes_are_cataloged():
    catalog_path = _REPO_ROOT / "tldw_Server_API" / "Config_Files" / "privilege_catalog.yaml"
    catalog_ids = load_catalog_scope_ids(catalog_path)
    expected_ids = {
        rpg_endpoint.CHAT_COMPLETIONS,
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


def test_context_requires_session_read_rules_read_and_media_read():
    permissions = _route_permissions("POST", "/api/v1/rpg/sessions/{session_id}/context")

    assert {rpg_endpoint.RPG_SESSIONS_READ, rpg_endpoint.RPG_RULES_READ, rpg_endpoint.MEDIA_READ} <= permissions  # nosec B101


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


def test_context_endpoint_reports_rules_lookup_diagnostics():
    client = TestClient(app)

    class FakeService:
        async def build_context(self, *, session_id, query, max_chars):
            assert session_id == 99  # nosec B101
            assert query == "stress"  # nosec B101
            assert max_chars == 1000  # nosec B101
            return SessionContext(
                text="Rules evidence:\n- Retrieved stress rule",
                diagnostics={
                    "truncated": False,
                    "max_chars": 1000,
                    "original_chars": 40,
                    "returned_chars": 40,
                    "rules_result_count": 1,
                    "omitted_sections": [],
                    "rules_lookup": {"retrieval_result_count": 1, "skipped_refs": []},
                },
            )

    app.dependency_overrides[rpg_endpoint._rules_service] = lambda: FakeService()
    try:
        response = client.post(
            "/api/v1/rpg/sessions/99/context",
            headers=_headers(),
            json={"query": "stress", "max_chars": 1000},
        )
    finally:
        app.dependency_overrides.pop(rpg_endpoint._rules_service, None)

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert "Retrieved stress rule" in payload["text"]  # nosec B101
    assert payload["diagnostics"]["rules_lookup"]["retrieval_result_count"] == 1  # nosec B101


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


def test_rules_lookup_answer_mode_passes_generation_options():
    client = TestClient(app)

    class FakeService:
        async def lookup_rules(self, *, session_id, query, mode, answer_options):
            assert session_id == 99  # nosec B101
            assert query == "stress"  # nosec B101
            assert mode == "answer"  # nosec B101
            assert isinstance(answer_options, RulesAnswerOptions)  # nosec B101
            assert answer_options.provider == "openai"  # nosec B101
            assert answer_options.model == "gpt-test"  # nosec B101
            assert answer_options.temperature == 0.4  # nosec B101
            assert answer_options.max_tokens == 321  # nosec B101
            return RuleLookupResult(
                query=query,
                mode=mode,
                results=[],
                answer=None,
                answer_status="no_evidence",
                answer_citation_ids=[],
                diagnostics={
                    "bundled_policy": "no_match",
                    "result_mode": "citation_index",
                    "linked_rules_pack_count": 0,
                    "enabled_rules_pack_count": 0,
                    "ready_media_item_count": 0,
                    "retrieval_result_count": 0,
                    "bundled_citation_count": 0,
                    "skipped_refs": [],
                    "broad_fallback_used": False,
                },
            )

    app.dependency_overrides[rpg_endpoint._rules_service] = lambda: FakeService()
    try:
        response = client.post(
            "/api/v1/rpg/sessions/99/rules/lookup",
            headers=_headers(),
            json={
                "query": "stress",
                "mode": "answer",
                "provider": "openai",
                "model": "gpt-test",
                "temperature": 0.4,
                "max_tokens": 321,
            },
        )
    finally:
        app.dependency_overrides.pop(rpg_endpoint._rules_service, None)

    assert response.status_code == 200  # nosec B101
    assert response.json()["answer_status"] == "no_evidence"  # nosec B101


def test_rules_lookup_answer_mode_requires_chat_completion_permission():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-answer-mode-no-chat")

    async def no_chat_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            roles=[],
            permissions=[rpg_endpoint.RPG_RULES_READ, rpg_endpoint.MEDIA_READ],
            is_admin=False,
        )

    app.dependency_overrides[get_auth_principal] = no_chat_principal
    try:
        lookup_response = client.post(
            f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
            headers=_headers(),
            json={"query": "stress", "mode": "lookup"},
        )
        answer_response = client.post(
            f"/api/v1/rpg/sessions/{session_id}/rules/lookup",
            headers=_headers(),
            json={"query": "stress", "mode": "answer"},
        )
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)

    assert lookup_response.status_code == 200  # nosec B101
    assert answer_response.status_code == 403  # nosec B101
    assert "chat.completions" in answer_response.json()["detail"]  # nosec B101


def test_answer_mode_generation_controls_pass_bearer_credentials(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_permission_guard(principal):
        captured["principal"] = principal

    async def fake_scope_guard(request, credentials=None):
        captured["credentials"] = credentials

    async def fake_llm_budget(request):
        captured["budget_checked"] = True

    async def fake_rbac_limit(request, resource, db_pool):
        captured["rate_limit_resource"] = resource
        captured["db_pool"] = db_pool

    async def fake_get_db_pool():
        return object()

    monkeypatch.setattr(rpg_endpoint, "_answer_mode_permission_guard", fake_permission_guard)
    monkeypatch.setattr(rpg_endpoint, "_answer_mode_scope_guard", fake_scope_guard)
    monkeypatch.setattr(rpg_endpoint, "enforce_llm_budget", fake_llm_budget)
    monkeypatch.setattr(rpg_endpoint, "enforce_rbac_rate_limit", fake_rbac_limit)
    monkeypatch.setattr(rpg_endpoint, "get_db_pool", fake_get_db_pool)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/rpg/sessions/99/rules/lookup",
            "headers": [(b"authorization", b"Bearer scoped-token")],
        }
    )
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        roles=[],
        permissions=[rpg_endpoint.CHAT_COMPLETIONS],
        is_admin=False,
    )

    asyncio.run(rpg_endpoint._enforce_answer_mode_generation_controls(request, principal))

    credentials = captured["credentials"]
    assert credentials.scheme == "Bearer"  # nosec B101
    assert credentials.credentials == "scoped-token"  # nosec B101
    assert captured["principal"] is principal  # nosec B101
    assert captured["budget_checked"] is True  # nosec B101
    assert captured["rate_limit_resource"] == rpg_endpoint.CHAT_CREATE_RATE_LIMIT  # nosec B101


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


def test_list_campaigns_route_returns_created_campaign():
    client = TestClient(app)

    campaign = client.post(
        "/api/v1/rpg/campaigns",
        headers=_headers(**{"Idempotency-Key": "route-list-c"}),
        json={"title": "Route Campaign", "default_adapter_key": "fate"},
    )
    listed = client.get("/api/v1/rpg/campaigns", headers=_headers())

    assert listed.status_code == 200  # nosec B101
    body = listed.json()
    assert body["total_count"] >= 1  # nosec B101
    assert any(item["id"] == campaign.json()["id"] for item in body["items"])  # nosec B101


def test_read_routes_scope_matrix():
    expected = {
        ("GET", "/api/v1/rpg/campaigns"): {rpg_endpoint.RPG_CAMPAIGNS_READ},
        ("GET", "/api/v1/rpg/campaigns/{campaign_id}"): {rpg_endpoint.RPG_CAMPAIGNS_READ},
        ("GET", "/api/v1/rpg/campaigns/{campaign_id}/sessions"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}/snapshot"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}/events"): {rpg_endpoint.RPG_SESSIONS_READ},
        ("GET", "/api/v1/rpg/sessions/{session_id}/proposals"): {rpg_endpoint.RPG_PROPOSALS_REVIEW},
    }
    for (method, path), scopes in expected.items():
        assert scopes <= _route_permissions(method, path), f"{method} {path} missing {scopes - _route_permissions(method, path)}"  # nosec B101


def test_session_snapshot_events_and_foreign_404():
    client = TestClient(app)
    _campaign_id, session_id = _create_campaign_and_session(client, "route-read")

    snap = client.get(f"/api/v1/rpg/sessions/{session_id}/snapshot", headers=_headers())

    assert snap.status_code == 200 and snap.json()["snapshot_version"] == 0  # nosec B101

    events = client.get(f"/api/v1/rpg/sessions/{session_id}/events", headers=_headers())

    assert events.status_code == 200 and events.json()["items"] == []  # nosec B101
    assert client.get("/api/v1/rpg/campaigns/999999", headers=_headers()).status_code == 404  # nosec B101
    assert client.get("/api/v1/rpg/sessions/999999/events", headers=_headers()).status_code == 404  # nosec B101
    assert client.get("/api/v1/rpg/sessions/999999/proposals", headers=_headers()).status_code == 404  # nosec B101


def test_session_events_list_with_rows_and_proposals(tmp_path):
    # The Phase-1 POST /events route hardcodes source_type="user" (no mcp/model
    # ingest route yet), so a real service is shared with the routes via
    # dependency override purely to seed one mcp-sourced event; every assertion
    # below still exercises the HTTP surface. The DB must be file-backed and
    # unique: TestClient serves on a worker thread, and a fresh connection to
    # ":memory:" from another thread is a new empty database.
    service = RPGService(
        repo=RPGRepository.initialized(CharactersRAGDB(str(tmp_path / "rpg_events_rows.db"), "rpg-api-events-test")),
        owner_user_id=42,
    )
    app.dependency_overrides[rpg_endpoint._service] = lambda: service
    try:
        client = TestClient(app)
        _campaign_id, session_id = _create_campaign_and_session(client, "route-events-rows")

        for index in range(3):
            recorded = client.post(
                f"/api/v1/rpg/sessions/{session_id}/events",
                headers=_headers(**{"Idempotency-Key": f"route-events-rows-{index}"}),
                json={
                    "expected_last_event_sequence": index,
                    "events": [
                        {
                            "event_type": "note.added",
                            "event_payload": {"note_id": f"n{index}", "text": f"Note {index}"},
                        }
                    ],
                },
            )
            assert recorded.status_code == 200  # nosec B101
            assert recorded.json()["committed_events"][0]["sequence_number"] == index + 1  # nosec B101

        listed = client.get(f"/api/v1/rpg/sessions/{session_id}/events", headers=_headers())

        assert listed.status_code == 200  # nosec B101
        body = listed.json()
        assert [item["sequence_number"] for item in body["items"]] == [1, 2, 3]  # nosec B101
        assert body["next_after_sequence"] == 3  # nosec B101
        assert all(item["event_type"] == "note.added" for item in body["items"])  # nosec B101
        assert [item["event_payload"]["note_id"] for item in body["items"]] == ["n0", "n1", "n2"]  # nosec B101

        paged = client.get(
            f"/api/v1/rpg/sessions/{session_id}/events",
            params={"limit": 1, "after_sequence": 1},
            headers=_headers(),
        )

        assert paged.status_code == 200  # nosec B101
        paged_body = paged.json()
        assert [item["sequence_number"] for item in paged_body["items"]] == [2]  # nosec B101
        assert paged_body["next_after_sequence"] == 2  # nosec B101

        # Seed one mcp-sourced event directly (default authority settings keep
        # mcp events pending as proposals), then verify the list route shape.
        mcp_result = service.record_events(
            session_id=session_id,
            events=[{"event_type": "note.added", "event_payload": {"note_id": "m1", "text": "Mcp note"}}],
            source_type="mcp",
            expected_last_event_sequence=3,
            idempotency_key="route-events-rows-mcp",
        )
        assert mcp_result.proposal is not None  # nosec B101

        proposals = client.get(f"/api/v1/rpg/sessions/{session_id}/proposals", headers=_headers())

        assert proposals.status_code == 200  # nosec B101
        proposals_body = proposals.json()
        assert proposals_body["total_count"] >= 1  # nosec B101
        assert proposals_body["items"][0]["session_id"] == session_id  # nosec B101
        assert proposals_body["items"][0]["status"] == "pending"  # nosec B101
    finally:
        app.dependency_overrides.pop(rpg_endpoint._service, None)


def test_patch_campaign_archive_and_conflict():
    client = TestClient(app)
    campaign_id, _session_id = _create_campaign_and_session(client, "route-patch")
    patched = client.patch(
        f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-1"}),
        json={"status": "archived", "expected_version": 1},
    )
    assert patched.status_code == 200 and patched.json()["status"] == "archived"  # nosec B101
    assert patched.json()["version"] == 2  # nosec B101
    replay = client.patch(
        f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-1"}),
        json={"status": "archived", "expected_version": 1},
    )
    assert replay.status_code == 200 and replay.json()["version"] == 2  # nosec B101  (replay)
    conflict = client.patch(
        f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-2"}),
        json={"title": "Stale", "expected_version": 1},
    )
    assert conflict.status_code == 409  # nosec B101
    bad_status = client.patch(
        f"/api/v1/rpg/campaigns/{campaign_id}", headers=_headers(**{"Idempotency-Key": "route-pc-3"}),
        json={"status": "deleted", "expected_version": 2},
    )
    # The Task 5 schema rejects unknown statuses at the Pydantic Literal layer,
    # so FastAPI answers with its standard request-validation code (422), not 400.
    assert bad_status.status_code == 422  # nosec B101


def test_patch_session_title_and_conflict():
    client = TestClient(app)
    _campaign_id, session_id = _create_campaign_and_session(client, "route-patch-s")

    patched = client.patch(
        f"/api/v1/rpg/sessions/{session_id}",
        headers=_headers(**{"Idempotency-Key": "route-ps-1"}),
        json={"title": "Renamed Opening", "expected_version": 1},
    )
    assert patched.status_code == 200  # nosec B101
    assert patched.json()["title"] == "Renamed Opening"  # nosec B101
    assert patched.json()["version"] == 2  # nosec B101

    conflict = client.patch(
        f"/api/v1/rpg/sessions/{session_id}",
        headers=_headers(**{"Idempotency-Key": "route-ps-2"}),
        json={"status": "archived", "expected_version": 1},
    )
    assert conflict.status_code == 409  # nosec B101


def test_authority_settings_put_roundtrip():
    client = TestClient(app)
    _campaign_id, session_id = _create_campaign_and_session(client, "route-auth")
    payload = {
        "import_auto_commit": False, "mcp_auto_commit": True,
        "model_auto_commit": False, "model_auto_commit_event_types": [],
        "expected_version": 1,
    }
    updated = client.put(
        f"/api/v1/rpg/sessions/{session_id}/authority-settings",
        headers=_headers(**{"Idempotency-Key": "route-au-1"}), json=payload,
    )
    assert updated.status_code == 200  # nosec B101
    assert updated.json()["authority_settings"]["mcp_auto_commit"] is True  # nosec B101
    detail = client.get(f"/api/v1/rpg/sessions/{session_id}", headers=_headers())
    assert detail.json()["authority_settings"]["mcp_auto_commit"] is True  # nosec B101
    bad = dict(payload, model_auto_commit_event_types=["nope.nope"], expected_version=2)
    assert client.put(
        f"/api/v1/rpg/sessions/{session_id}/authority-settings",
        headers=_headers(**{"Idempotency-Key": "route-au-2"}), json=bad,
    ).status_code == 400  # nosec B101


def test_snapshot_rebuild_route():
    client = TestClient(app)
    _campaign_id, session_id = _create_campaign_and_session(client, "route-rb")
    rebuilt = client.post(
        f"/api/v1/rpg/sessions/{session_id}/snapshot/rebuild", headers=_headers(**{"Idempotency-Key": "route-rb-1"}),
    )
    assert rebuilt.status_code == 200  # nosec B101
    assert rebuilt.json()["event_count"] == 0 and rebuilt.json()["replayed"] is False  # nosec B101
    replay = client.post(
        f"/api/v1/rpg/sessions/{session_id}/snapshot/rebuild", headers=_headers(**{"Idempotency-Key": "route-rb-1"}),
    )
    assert replay.status_code == 200 and replay.json()["replayed"] is True  # nosec B101


def test_mutation_routes_scope_matrix():
    expected = {
        ("PATCH", "/api/v1/rpg/campaigns/{campaign_id}"): {rpg_endpoint.RPG_CAMPAIGNS_MANAGE},
        ("PATCH", "/api/v1/rpg/sessions/{session_id}"): {rpg_endpoint.RPG_SESSIONS_MANAGE},
        ("PUT", "/api/v1/rpg/sessions/{session_id}/authority-settings"): {rpg_endpoint.RPG_SESSIONS_MANAGE},
        ("POST", "/api/v1/rpg/sessions/{session_id}/snapshot/rebuild"): {rpg_endpoint.RPG_SESSIONS_MANAGE},
    }
    for (method, path), scopes in expected.items():
        assert scopes <= _route_permissions(method, path), f"{method} {path} missing {scopes - _route_permissions(method, path)}"  # nosec B101
