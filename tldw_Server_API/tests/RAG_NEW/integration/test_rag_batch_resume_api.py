import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")


@pytest.fixture()
def client_with_overrides(monkeypatch, auth_headers):
    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _noop():
        return None

    async def _fake_principal(request: Request) -> AuthPrincipal:  # noqa: ARG001
        return AuthPrincipal(
            kind="service",
            user_id=None,
            api_key_id=None,
            subject="service:rag-batch-resume-test",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.logs", "media.read"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    # Disable RBAC enforcement to avoid DB access
    import tldw_Server_API.app.api.v1.API_Deps.auth_deps as auth_deps
    async def _no_rbac(*args, **kwargs):  # noqa: ARG001
        return None
    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _no_rbac)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _fake_principal
    fastapi_app.dependency_overrides[check_rate_limit] = _noop
    # Avoid DB initialization by overriding DB deps to return None
    try:
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as _get_media_db
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user as _get_chacha_db
        async def _none_media_db():
            return None
        async def _none_chacha_db():
            return None
        fastapi_app.dependency_overrides[_get_media_db] = _none_media_db
        fastapi_app.dependency_overrides[_get_chacha_db] = _none_chacha_db
    except Exception:
        _ = None

    try:
        # Prefer disabling lifespan to avoid DB/services startup in CI; skip if not supported
        import inspect as _inspect
        if 'lifespan' in _inspect.signature(TestClient.__init__).parameters:
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False, lifespan='off') as client:
                yield client
        else:
            with TestClient(fastapi_app, headers=auth_headers, raise_server_exceptions=False) as client:
                yield client
    finally:
        fastapi_app.dependency_overrides.clear()


def _set_checkpoint_dir(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.RAG.rag_service.checkpoint as cp_mod
    monkeypatch.setattr(
        cp_mod.CheckpointManager,
        "DEFAULT_CHECKPOINT_DIR",
        str(tmp_path / "checkpoints"),
    )
    return cp_mod


def test_rag_batch_resume_respects_query_indices(client_with_overrides, monkeypatch, tmp_path):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    manager = cp_mod.CheckpointManager()

    queries = ["alpha", "alpha", "beta"]
    checkpoint = manager.create(
        "rag_batch",
        total_items=len(queries),
        config={"queries": queries, "max_concurrent": 2},
    )
    checkpoint = manager.save_progress(
        checkpoint,
        {"query_index": 2, "query": "beta", "status": "ok"},
    )

    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    async def fake_pipeline(query: str, **kwargs):  # noqa: ARG001
        return up.UnifiedSearchResult(documents=[], query=query, errors=[])

    monkeypatch.setattr(up, "unified_rag_pipeline", fake_pipeline)

    resp = client_with_overrides.post(f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data.get("total_queries") == 2
    assert data.get("successful") == 2
    assert data.get("failed") == 0

    loaded = manager.load_by_id(checkpoint.checkpoint_id)
    assert loaded.total_items == 3
    assert loaded.completed_items == 3

    indices = [entry.get("query_index") for entry in loaded.results]
    assert sorted(indices) == [0, 1, 2]
    assert len(indices) == len(set(indices))
    status_map = {entry.get("query_index"): entry.get("status") for entry in loaded.results}
    assert status_map.get(0) == "ok"
    assert status_map.get(1) == "ok"
    assert status_map.get(2) == "ok"


def test_rag_batch_resume_records_errors(client_with_overrides, monkeypatch, tmp_path):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    manager = cp_mod.CheckpointManager()

    queries = ["good", "bad"]
    checkpoint = manager.create(
        "rag_batch",
        total_items=len(queries),
        config={"queries": queries, "max_concurrent": 2},
    )

    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    async def fake_pipeline(query: str, **kwargs):  # noqa: ARG001
        if query == "bad":
            raise RuntimeError("boom")
        return up.UnifiedSearchResult(documents=[], query=query, errors=[])

    monkeypatch.setattr(up, "unified_rag_pipeline", fake_pipeline)

    resp = client_with_overrides.post(f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data.get("total_queries") == 2
    assert data.get("successful") == 1
    assert data.get("failed") == 1

    loaded = manager.load_by_id(checkpoint.checkpoint_id)
    assert loaded.total_items == 2
    assert loaded.completed_items == 2

    status_map = {entry.get("query_index"): entry for entry in loaded.results}
    assert status_map.get(0, {}).get("status") == "ok"
    bad_entry = status_map.get(1)
    assert bad_entry is not None
    assert bad_entry.get("status") == "error"
    assert any("boom" in str(err) for err in bad_entry.get("errors", []))


def test_rag_batch_resume_reuses_shared_batch_resolution(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    manager = cp_mod.CheckpointManager()

    checkpoint = manager.create(
        "rag_batch",
        total_items=1,
        config={
            "queries": ["resume me"],
            "max_concurrent": 2,
            "corpus": "resume-corpus",
        },
    )

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    captured = {"kwargs": None}

    async def fake_batch_pipeline(queries, **kwargs):  # noqa: ANN001
        captured["kwargs"] = {"queries": queries, **kwargs}
        return [up.UnifiedSearchResult(documents=[], query=queries[0], errors=[])]

    monkeypatch.setattr(rag_ep, "unified_batch_pipeline", fake_batch_pipeline)

    resp = client_with_overrides.post(f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}")
    assert resp.status_code == 200, resp.text

    kwargs = captured["kwargs"]
    assert kwargs is not None
    assert kwargs["queries"] == ["resume me"]
    assert kwargs["index_namespace"] == "resume-corpus"
    assert kwargs["sources"] == ["media_db", "notes", "characters"]
    assert kwargs["resolved_request"].index_namespace == "resume-corpus"
    assert kwargs["retrieval_plan"].index_namespace == "resume-corpus"
    assert kwargs["retrieval_plan"].search_mode == kwargs["search_mode"]
    assert kwargs["retrieval_plan"].top_k == kwargs["top_k"]
    assert kwargs["retrieval_plan"].min_score == kwargs["min_score"]
    assert list(kwargs["retrieval_plan"].sources) == kwargs["sources"]
    assert kwargs["resolved_request"].user_id == kwargs["user_id"]
    assert kwargs["resolved_request"].feedback_user_id == kwargs["feedback_user_id"]
