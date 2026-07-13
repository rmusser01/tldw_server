from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.scope_context import (
    get_scope,
    scoped_context,
)
from tldw_Server_API.app.main import app as fastapi_app

pytestmark = pytest.mark.integration


class _RecordingRuntime:
    instances = []

    def __init__(self, **scope):
        self.scope = scope
        self.close_calls = 0
        self.close_scope = None
        type(self).instances.append(self)

    async def close(self):
        self.close_calls += 1
        self.close_scope = get_scope()


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
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user as _get_chacha_db
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user as _get_media_db
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


def _override_principal(principal):
    async def _principal(request: Request):  # noqa: ARG001
        return principal

    fastapi_app.dependency_overrides[get_auth_principal] = _principal


def _override_user(user_id: int):
    async def _user():
        return User(id=user_id, username=f"user-{user_id}", email=None)

    fastapi_app.dependency_overrides[get_request_user] = _user


def _install_recording_runtime(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    _RecordingRuntime.instances.clear()
    monkeypatch.setattr(rag_ep, "ProviderCredentialRuntime", _RecordingRuntime)


def _patch_current_memberships(monkeypatch, *, team_ids=(), org_ids=(), calls=None):
    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs_teams

    async def _teams(user_id: int):
        if calls is not None:
            calls.append(("team", user_id))
        return [{"team_id": team_id} for team_id in team_ids]

    async def _orgs(user_id: int):
        if calls is not None:
            calls.append(("org", user_id))
        return [{"org_id": org_id, "status": "active"} for org_id in org_ids]

    monkeypatch.setattr(orgs_teams, "list_active_team_memberships_for_user", _teams)
    monkeypatch.setattr(orgs_teams, "list_org_memberships_for_user", _orgs)


def _user_principal(user_id: int, **claims):
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        subject=f"user:{user_id}",
        token_type="access",
        roles=claims.get("roles", []),
        permissions=claims.get("permissions", ["media.read"]),
        is_admin=claims.get("is_admin", False),
        org_ids=claims.get("org_ids", []),
        team_ids=claims.get("team_ids", []),
    )


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
    assert bad_entry.get("errors") == ["batch_query_failed"]


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


def test_rag_batch_resume_rejects_owner_mismatch_before_complete_response_or_runtime(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=0,
        config={"queries": []},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": [],
                "org_ids": [],
            }
        },
    )
    _override_principal(_user_principal(1))
    _install_recording_runtime(monkeypatch)

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "checkpoint_owner_forbidden"
    assert _RecordingRuntime.instances == []


@pytest.mark.parametrize(
    "claims",
    [
        {"is_admin": True},
        {"roles": [" AdMiN "]},
        {"permissions": ["*"]},
        {"permissions": ["media.read", "SYSTEM.CONFIGURE"]},
    ],
)
def test_rag_batch_resume_accepts_explicit_admin_claims(
    client_with_overrides,
    monkeypatch,
    tmp_path,
    claims,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=0,
        config={"queries": []},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": [],
                "org_ids": [],
            }
        },
    )
    _override_principal(_user_principal(99, **claims))

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 200, response.text


def test_rag_batch_resume_rejects_revoked_checkpoint_membership(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=0,
        config={"queries": []},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": [7],
                "org_ids": [11],
            }
        },
    )
    _override_principal(_user_principal(42))
    _patch_current_memberships(monkeypatch)

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "credential_scope_revoked"


def test_rag_batch_resume_fails_closed_when_membership_store_is_unavailable(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=0,
        config={"queries": []},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": [7],
                "org_ids": [],
            }
        },
    )
    _override_principal(_user_principal(42))

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs_teams

    async def _unavailable(user_id: int):  # noqa: ARG001
        raise RuntimeError("private membership database detail")

    monkeypatch.setattr(
        orgs_teams,
        "list_active_team_memberships_for_user",
        _unavailable,
    )

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "credential_scope_unavailable"
    assert "private membership database detail" not in response.text


@pytest.mark.parametrize(
    ("scope_kind", "invalid_id"),
    [
        ("team", 7.9),
        ("team", True),
        ("team", "7"),
        ("org", 11.9),
        ("org", True),
        ("org", "11"),
    ],
)
def test_rag_batch_resume_rejects_malformed_current_membership_rows(
    client_with_overrides,
    monkeypatch,
    tmp_path,
    scope_kind,
    invalid_id,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    stored_team_ids = [7] if scope_kind == "team" else []
    stored_org_ids = [11] if scope_kind == "org" else []
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=0,
        config={"queries": []},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": stored_team_ids,
                "org_ids": stored_org_ids,
            }
        },
    )
    _override_principal(_user_principal(42))
    _install_recording_runtime(monkeypatch)

    import tldw_Server_API.app.core.AuthNZ.orgs_teams as orgs_teams

    async def _teams(user_id: int):  # noqa: ARG001
        return [{"team_id": invalid_id}]

    async def _orgs(user_id: int):  # noqa: ARG001
        return [{"org_id": invalid_id, "status": "active"}]

    monkeypatch.setattr(orgs_teams, "list_active_team_memberships_for_user", _teams)
    monkeypatch.setattr(orgs_teams, "list_org_memberships_for_user", _orgs)

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "credential_scope_unavailable"
    assert _RecordingRuntime.instances == []


def test_rag_batch_resume_rejects_present_but_malformed_scope(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=0,
        config={"queries": []},
        metadata={
            "credential_scope": {
                "owner_user_id": "42",
                "team_ids": [],
                "org_ids": [],
            }
        },
    )
    _override_principal(_user_principal(42))
    _install_recording_runtime(monkeypatch)

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "credential_scope_invalid"
    assert _RecordingRuntime.instances == []


@pytest.mark.parametrize(
    ("permissions", "expected_trusted"),
    [(["media.read"], False), (["media.read", "system.configure"], True)],
)
def test_rag_batch_resume_rebuilds_owner_runtime_with_current_base_url_authority(
    client_with_overrides,
    monkeypatch,
    tmp_path,
    permissions,
    expected_trusted,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=1,
        config={"queries": ["resume me"], "max_concurrent": 1},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": [7],
                "org_ids": [11],
            }
        },
    )
    _override_principal(_user_principal(42, permissions=permissions))
    _override_user(42)
    _patch_current_memberships(monkeypatch, team_ids=[7], org_ids=[11])
    _install_recording_runtime(monkeypatch)

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    owner_media = SimpleNamespace(db_path="owner-media.db")
    owner_chacha = SimpleNamespace(db_path="owner-chacha.db")
    owner_prompts = SimpleNamespace(db_path_str="owner-prompts.db")
    owner_resource_calls = []
    captured = {}

    @contextmanager
    def _owner_media(owner_user_id: int):
        owner_resource_calls.append(("media_enter", owner_user_id))
        captured["media_scope"] = get_scope()
        try:
            yield owner_media
        finally:
            captured["media_exit_scope"] = get_scope()
            owner_resource_calls.append(("media_exit", owner_user_id))

    async def _owner_chacha(owner_user_id: int):
        owner_resource_calls.append(("chacha", owner_user_id))
        return owner_chacha

    async def _owner_prompts(request: Request, owner_user: User):  # noqa: ARG001
        owner_resource_calls.append(("prompts", owner_user.id))
        return owner_prompts

    async def fake_batch_pipeline(**kwargs):
        captured["pipeline"] = kwargs
        captured["pipeline_scope"] = get_scope()
        return [up.UnifiedSearchResult(documents=[], query="resume me", errors=[])]

    monkeypatch.setattr(rag_ep, "managed_media_db_for_owner", _owner_media)
    monkeypatch.setattr(rag_ep, "get_chacha_db_for_owner", _owner_chacha)
    monkeypatch.setattr(rag_ep, "get_prompts_db_for_user", _owner_prompts)
    monkeypatch.setattr(rag_ep, "unified_batch_pipeline", fake_batch_pipeline)

    with scoped_context(
        user_id=42,
        org_ids=[199],
        team_ids=[299],
        active_org_id=199,
        active_team_id=299,
        is_admin=True,
    ) as request_scope:
        response = client_with_overrides.post(
            f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
        )
        restored_scope = get_scope()

    assert response.status_code == 200, response.text
    runtime = _RecordingRuntime.instances[0]
    assert runtime.scope["user_id"] == 42
    assert runtime.scope["team_ids"] == [7]
    assert runtime.scope["org_ids"] == [11]
    assert runtime.scope["trusted_base_url_override"] is expected_trusted
    assert runtime.close_calls == 1
    assert captured["pipeline"]["media_db"] is owner_media
    assert captured["pipeline"]["chacha_db"] is owner_chacha
    assert captured["pipeline"]["prompts_db"] is owner_prompts
    for scope in (
        captured["media_scope"],
        captured["pipeline_scope"],
        runtime.close_scope,
        captured["media_exit_scope"],
    ):
        assert scope.user_id == 42
        assert scope.org_ids == [11]
        assert scope.team_ids == [7]
        assert scope.active_org_id == 11
        assert scope.active_team_id == 7
        assert scope.is_admin is False
    assert restored_scope == request_scope
    assert owner_resource_calls == [
        ("media_enter", 42),
        ("chacha", 42),
        ("prompts", 42),
        ("media_exit", 42),
    ]


def test_rag_batch_admin_resume_uses_checkpoint_owner_scope(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=1,
        config={"queries": ["resume me"], "max_concurrent": 1},
        metadata={
            "credential_scope": {
                "owner_user_id": 42,
                "team_ids": [7],
                "org_ids": [11],
            }
        },
    )
    _override_principal(
        _user_principal(99, permissions=["media.read", "system.configure"])
    )
    membership_calls = []
    _patch_current_memberships(
        monkeypatch,
        team_ids=[7],
        org_ids=[11],
        calls=membership_calls,
    )
    _install_recording_runtime(monkeypatch)

    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    owner_media = SimpleNamespace(db_path="owner-media.db")
    owner_chacha = SimpleNamespace(db_path="owner-chacha.db")
    owner_prompts = SimpleNamespace(db_path_str="owner-prompts.db")
    owner_resource_calls = []
    captured = {}

    @contextmanager
    def _owner_media(owner_user_id: int):
        owner_resource_calls.append(("media_enter", owner_user_id))
        captured["media_scope"] = get_scope()
        try:
            yield owner_media
        finally:
            captured["media_exit_scope"] = get_scope()
            owner_resource_calls.append(("media_exit", owner_user_id))

    async def _owner_chacha(owner_user_id: int):
        owner_resource_calls.append(("chacha", owner_user_id))
        return owner_chacha

    async def _owner_prompts(request: Request, owner_user: User):  # noqa: ARG001
        owner_resource_calls.append(("prompts", owner_user.id))
        return owner_prompts

    original_build_bundle = rag_ep._build_batch_request_bundle

    def _recording_build_bundle(*args, **kwargs):
        bundle = original_build_bundle(*args, **kwargs)
        captured["current_user"] = kwargs["current_user"]
        captured["resolved_request"] = bundle.resolved_request
        captured["db_paths"] = kwargs["db_paths"]
        return bundle

    async def fake_batch_pipeline(**kwargs):
        captured["pipeline"] = kwargs
        captured["pipeline_scope"] = get_scope()
        return [up.UnifiedSearchResult(documents=[], query="resume me", errors=[])]

    monkeypatch.setattr(
        rag_ep,
        "managed_media_db_for_owner",
        _owner_media,
        raising=False,
    )
    monkeypatch.setattr(
        rag_ep,
        "get_chacha_db_for_owner",
        _owner_chacha,
        raising=False,
    )
    monkeypatch.setattr(rag_ep, "get_prompts_db_for_user", _owner_prompts)
    monkeypatch.setattr(rag_ep, "_build_batch_request_bundle", _recording_build_bundle)
    monkeypatch.setattr(rag_ep, "unified_batch_pipeline", fake_batch_pipeline)

    with scoped_context(
        user_id=99,
        org_ids=[199],
        team_ids=[299],
        active_org_id=199,
        active_team_id=299,
        is_admin=True,
    ) as admin_scope:
        response = client_with_overrides.post(
            f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
        )
        restored_scope = get_scope()

    assert response.status_code == 200, response.text
    runtime = _RecordingRuntime.instances[0]
    assert runtime.scope["user_id"] == 42
    assert runtime.scope["team_ids"] == [7]
    assert runtime.scope["org_ids"] == [11]
    assert runtime.scope["trusted_base_url_override"] is True
    assert membership_calls == [("team", 42), ("org", 42)]
    assert captured["pipeline"]["media_db"] is owner_media
    assert captured["pipeline"]["chacha_db"] is owner_chacha
    assert captured["pipeline"]["prompts_db"] is owner_prompts
    assert captured["current_user"].id == 42
    assert captured["resolved_request"].user_id == "42"
    assert captured["resolved_request"].feedback_user_id == "42"
    assert captured["db_paths"]["media_db_path"] == "owner-media.db"
    assert captured["db_paths"]["notes_db_path"] == "owner-chacha.db"
    assert captured["db_paths"]["character_db_path"] == "owner-chacha.db"
    assert captured["db_paths"]["prompts_db_path"] == "owner-prompts.db"
    assert "42" in captured["db_paths"]["kanban_db_path"]
    for scope in (
        captured["media_scope"],
        captured["pipeline_scope"],
        runtime.close_scope,
        captured["media_exit_scope"],
    ):
        assert scope.user_id == 42
        assert scope.org_ids == [11]
        assert scope.team_ids == [7]
        assert scope.active_org_id == 11
        assert scope.active_team_id == 7
        assert scope.is_admin is False
    assert restored_scope == admin_scope
    assert owner_resource_calls == [
        ("media_enter", 42),
        ("chacha", 42),
        ("prompts", 42),
        ("media_exit", 42),
    ]


def test_rag_batch_legacy_checkpoint_uses_server_runtime_scope(
    client_with_overrides,
    monkeypatch,
    tmp_path,
):
    cp_mod = _set_checkpoint_dir(monkeypatch, tmp_path)
    checkpoint = cp_mod.CheckpointManager().create(
        "rag_batch",
        total_items=1,
        config={"queries": ["resume me"], "max_concurrent": 1},
    )
    _override_principal(
        _user_principal(42, permissions=["media.read", "system.configure"])
    )
    _install_recording_runtime(monkeypatch)

    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    async def fake_pipeline(query: str, **kwargs):  # noqa: ARG001
        return up.UnifiedSearchResult(documents=[], query=query, errors=[])

    monkeypatch.setattr(up, "unified_rag_pipeline", fake_pipeline)

    response = client_with_overrides.post(
        f"/api/v1/rag/batch/resume/{checkpoint.checkpoint_id}"
    )

    assert response.status_code == 200, response.text
    runtime = _RecordingRuntime.instances[0]
    assert runtime.scope["user_id"] is None
    assert runtime.scope["team_ids"] == []
    assert runtime.scope["org_ids"] == []
    assert runtime.scope["trusted_base_url_override"] is False
