
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import notes_graph as notes_graph_module
from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    NoteGraphRequest,
    NoteLinkCreate,
    NoteLinkRestore,
    NoteLinkUpdate,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, reset_jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLink
from tldw_Server_API.app.core.Sync.v2.models import DEFAULT_M1_ENCRYPTION_POLICY, SyncDataset
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import NotesLinkPreflightError
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginBatchResult,
    SyncServerOriginBatchAppendError,
    SyncServerOriginBatchMaterializationError,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("model", [NoteLinkCreate, NoteLinkUpdate, NoteLinkRestore])
@pytest.mark.parametrize("field", ["dataset_id", "idempotency_key"])
def test_link_authority_fields_trim_and_reject_blank(model, field: str) -> None:
    payload: dict[str, object] = {field: "  value  "}
    if model is NoteLinkCreate:
        payload["to_note_id"] = "note-2"
    elif model is NoteLinkUpdate:
        payload["weight"] = 2.0

    request = model.model_validate(payload)
    assert getattr(request, field) == "value"

    payload[field] = "   "
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_graph_dataset_authority_field_trims_and_rejects_blank() -> None:
    assert NoteGraphRequest(dataset_id="  dataset-1  ").dataset_id == "dataset-1"
    with pytest.raises(ValidationError):
        NoteGraphRequest(dataset_id="   ")


@pytest.fixture()
def test_app(monkeypatch) -> FastAPI:
     # Configure JWT for tests (multi-user style virtual key)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "graph_rbac_tests_secret_1234567890")
    reset_settings()
    reset_jwt_service()

    app = FastAPI()
    app.include_router(notes_graph_module.router, prefix="/api/v1/notes")

    class _StubChaChaDB:
        def __init__(self, user_id: int) -> None:
            self.client_id = str(user_id)

        def create_manual_note_edge(
            self,
            *,
            user_id: str,
            from_note_id: str,
            to_note_id: str,
            directed: bool,
            weight: float,
            metadata: object,
            created_by: str,
        ) -> dict:
            return {
                "id": "edge:test",
                "user_id": user_id,
                "from_note_id": from_note_id,
                "to_note_id": to_note_id,
                "directed": directed,
                "weight": weight,
                "metadata": metadata,
                "created_by": created_by,
            }

        # Graph query stubs needed by NoteGraphService
        def count_user_notes(self, include_deleted=True):
            return 0

        def get_all_note_ids_for_graph(self, include_deleted=True, limit=500):
            return []

        def get_notes_batch(self, note_ids, include_deleted=True):
            return []

        def get_manual_edges_for_notes(self, user_id, note_ids):
            return []

        def get_note_tag_edges(self, note_ids):
            return []

        def count_notes_per_tag(self):
            return {}

        def get_note_source_info(self, note_ids):
            return []

    async def override_user():
        # Provide a benign user object; auth is enforced by token scope and claims
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            roles=["user"],
            permissions=[
                "notes.graph.read",
                "notes.graph.write",
            ],
        )

    async def override_chacha_db():
        return _StubChaChaDB(user_id=1)

    async def _override_auth_principal(request: Request):  # type: ignore[override]
        # Provide a principal with the same user id and graph permissions.
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=["notes.graph.read", "notes.graph.write"],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[auth_deps.get_auth_principal] = _override_auth_principal
    return app


@pytest.fixture()
def client_with_user_override(test_app: FastAPI):
    with TestClient(test_app) as client:
        yield client


def _make_token(scope: str) -> str:
    svc = JWTService(get_settings())
    return svc.create_virtual_access_token(user_id=1, username="tester", role="user", scope=scope, ttl_minutes=5)


def test_graph_read_forbidden_with_wrong_scope(client_with_user_override: TestClient):
    bad_token = _make_token(scope="media")  # Endpoint requires scope="notes"
    headers = {"Authorization": f"Bearer {bad_token}"}
    resp = client_with_user_override.get(
        "/api/v1/notes/graph",
        headers=headers,
        params={"request": "graph"},
    )
    assert resp.status_code == 403


def test_graph_read_allows_with_correct_scope(client_with_user_override: TestClient):
    good_token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {good_token}"}
    resp = client_with_user_override.get(
        "/api/v1/notes/graph",
        headers=headers,
        params={"request": "graph"},
    )
    # Handler is a stub; it should pass auth and return a 200 with empty graph structure
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "nodes" in data and "edges" in data


def test_graph_authorizes_dataset_before_cache_lookup(
    client_with_user_override: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_authority(**_kwargs):
        raise notes_graph_module.NotesLinkDatasetConflictError()

    monkeypatch.setattr(
        notes_graph_module,
        "resolve_notes_link_dataset_authority",
        reject_authority,
    )
    monkeypatch.setattr(
        notes_graph_module._GRAPH_CACHE,
        "get",
        lambda _key: pytest.fail("cache consulted before dataset authorization"),
    )
    token = _make_token(scope="notes")
    response = client_with_user_override.get(
        "/api/v1/notes/graph",
        headers={"Authorization": f"Bearer {token}"},
        params={"dataset_id": "wrong-dataset"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == "notes_link_dataset_conflict"


def test_graph_projection_not_ready_maps_to_safe_503(
    client_with_user_override: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        notes_graph_module.NoteGraphService,
        "generate_graph",
        lambda _self, _request: (_ for _ in ()).throw(
            notes_graph_module.GraphProjectionNotReadyError("rebuilding")
        ),
    )
    token = _make_token(scope="notes")
    response = client_with_user_override.get(
        "/api/v1/notes/graph",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == "notes_graph_projection_not_ready"


@pytest.mark.asyncio
async def test_link_detail_uses_read_authority_without_requiring_write_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge_id = "11111111-1111-4111-8111-111111111111"
    link = NotesLink(
        edge_id=edge_id,
        owner_user_id="1",
        source_note_id="22222222-2222-4222-8222-222222222222",
        target_note_id="33333333-3333-4333-8333-333333333333",
        type="manual",
        directed=False,
        weight=1.0,
        label=None,
        properties={},
        created_at="2026-08-10T12:00:00+00:00",
        last_modified="2026-08-10T12:00:00+00:00",
        created_by="device-1",
        version=1,
        deleted=False,
        deleted_at=None,
    )

    class _LinkStore:
        @staticmethod
        def get(requested_edge_id: str) -> NotesLink | None:
            return link if requested_edge_id == edge_id else None

    class _Db:
        notes_link_store = _LinkStore()

    monkeypatch.setattr(
        notes_graph_module,
        "_graph_dataset_key",
        lambda **_kwargs: "dataset-1",
    )
    monkeypatch.setattr(
        notes_graph_module,
        "resolve_notes_link_coordinator",
        lambda **_kwargs: pytest.fail("read-only link detail required write readiness"),
    )

    result = await notes_graph_module.get_manual_link(
        edge_id=edge_id,
        dataset_id="dataset-1",
        current_user=User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
        ),
        db=_Db(),  # type: ignore[arg-type]
        _=None,
        __=None,
        ___=None,
    )

    assert result["edge_id"] == edge_id


@pytest.mark.asyncio
async def test_legacy_create_without_edge_id_skips_canonical_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "11111111-1111-4111-8111-111111111111"
    target_id = "22222222-2222-4222-8222-222222222222"

    class _LinkStore:
        @staticmethod
        def get(_edge_id: str) -> None:
            pytest.fail("empty legacy edge id must not be read")

    class _Db:
        notes_link_store = _LinkStore()

        @staticmethod
        def create_manual_note_edge(**_kwargs: object) -> dict[str, object]:
            return {"from_note_id": source_id, "to_note_id": target_id}

    monkeypatch.setattr(
        notes_graph_module,
        "resolve_notes_link_coordinator",
        lambda **_kwargs: None,
    )
    result = await notes_graph_module.create_manual_link(
        note_id=source_id,
        link=NoteLinkCreate(to_note_id=target_id),
        current_user=User(id=1, username="tester", email="t@e.com", is_active=True),
        db=_Db(),  # type: ignore[arg-type]
        _=None,
        __=None,
        ___=None,
    )

    assert result["edge"] == {"from_note_id": source_id, "to_note_id": target_id}


def test_graph_write_forbidden_with_wrong_scope(client_with_user_override: TestClient):
    bad_token = _make_token(scope="media")
    headers = {"Authorization": f"Bearer {bad_token}"}
    resp = client_with_user_override.post(
        "/api/v1/notes/n-1/links",
        headers=headers,
        json={"to_note_id": "n-2"},
        params={"request": "graph"},
    )
    assert resp.status_code == 403


def test_graph_write_allows_with_correct_scope(client_with_user_override: TestClient):
    good_token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {good_token}"}
    resp = client_with_user_override.post(
        "/api/v1/notes/n-1/links",
        headers=headers,
        json={"to_note_id": "n-2"},
        params={"request": "graph"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    # Endpoint may be a stub or implemented; accept either
    assert payload.get("status") in {"stub", "created"}


def test_active_link_mutations_require_expected_version(
    client_with_user_override: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ActiveCoordinator:
        pass

    monkeypatch.setattr(
        notes_graph_module,
        "resolve_notes_link_coordinator",
        lambda **_kwargs: _ActiveCoordinator(),
    )
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}
    edge_id = "11111111-1111-4111-8111-111111111111"

    patch = client_with_user_override.patch(
        f"/api/v1/notes/links/{edge_id}",
        headers=headers,
        json={"weight": 2.0},
        params={"request": "graph"},
    )
    delete = client_with_user_override.delete(
        f"/api/v1/notes/links/{edge_id}",
        headers=headers,
        params={"request": "graph"},
    )
    restore = client_with_user_override.post(
        f"/api/v1/notes/links/{edge_id}/restore",
        headers=headers,
        json={},
        params={"request": "graph"},
    )

    assert [patch.status_code, delete.status_code, restore.status_code] == [428, 428, 428]


def test_inactive_sync_rejects_supplied_dataset(client_with_user_override: TestClient):
    token = _make_token(scope="notes")
    headers = {"Authorization": f"Bearer {token}"}
    response = client_with_user_override.post(
        "/api/v1/notes/n-1/links",
        headers=headers,
        json={"to_note_id": "n-2", "dataset_id": "dataset-explicit"},
        params={"request": "graph"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "notes_link_sync_inactive_dataset"


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_code"),
    [
        (NotesLinkPreflightError(), 409, "notes_link_preflight_failed"),
        (
            SyncServerOriginBatchAppendError("group-1"),
            503,
            "sync_server_origin_batch_append_failed",
        ),
        (
            SyncServerOriginBatchMaterializationError(
                ServerOriginBatchResult(
                    dataset=SyncDataset(
                        dataset_id="dataset-1",
                        owner_user_id="1",
                        scope_type="personal",
                        encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
                        domains=["notes.note", "notes.link"],
                        workspace_id=None,
                        metadata={"notes_link_v1": {"state": "ready"}},
                        created_at="2026-08-10T12:00:00+00:00",
                        updated_at="2026-08-10T12:00:00+00:00",
                    ),
                    envelopes=(),
                    fully_applied=False,
                ),
                retryable=True,
            ),
            503,
            "sync_server_origin_batch_materialization_failed",
        ),
    ],
)
def test_active_link_capture_failures_have_stable_http_mapping(
    client_with_user_override: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    class _FailingCoordinator:
        def create(self, **_kwargs):
            raise failure

    monkeypatch.setattr(
        notes_graph_module,
        "resolve_notes_link_coordinator",
        lambda **_kwargs: _FailingCoordinator(),
    )
    token = _make_token(scope="notes")
    response = client_with_user_override.post(
        "/api/v1/notes/n-1/links",
        headers={"Authorization": f"Bearer {token}"},
        json={"to_note_id": "n-2", "idempotency_key": "route-failure"},
        params={"request": "graph"},
    )
    assert response.status_code == expected_status
    assert response.json()["detail"]["error_code"] == expected_code
