"""Prove Workspace origin and replay through the native research API."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import research_runs, workspaces
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Research import service as research_service
from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

ROOT = "/api/v1/workspaces/workspace-a/research-runs"
RUNS = "/api/v1/research/runs"
pytestmark = pytest.mark.integration


@pytest.fixture
def owner(tmp_path, monkeypatch):
    cha = CharactersRAGDB(str(tmp_path / "chacha.db"), client_id="1")
    for workspace_id in ("workspace-a", "workspace-b"):
        cha.workspace_store.upsert_workspace(workspace_id, workspace_id)
        cha.workspace_store.add_workspace_source(workspace_id, {"id": "source-1", "media_id": 12, "title": "Source"})
    manager = JobManager(tmp_path / "jobs.db", backend="sqlite")
    service = research_service.ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=manager,
    )
    app = FastAPI()
    app.include_router(workspaces.router, prefix="/api/v1/workspaces")
    app.include_router(research_runs.router, prefix="/api/v1")

    async def allow_rate_limit():
        pass

    async def principal(_request):
        return SimpleNamespace(user_id=app.dependency_overrides[get_request_user]().id)

    monkeypatch.setattr(auth_deps, "get_auth_principal", principal)
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: cha
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = allow_rate_limit
    app.dependency_overrides[research_runs.get_research_service] = lambda: service
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, app=app, service=service, manager=manager, cha=cha)
    cha.close_all_connections()


def _launch():
    return {
        "launch_id": "launch-workspace",
        "query": "Which evidence supports this?",
        "created_at": "2026-11-24T20:00:00Z",
        "source_snapshot": [
            {
                "source_id": "source-1",
                "catalog_item_id": "12",
                "source_type": "media",
                "association_version": 1,
                "ready_modes": ["fts"],
            }
        ],
    }


def test_workspace_origin_survives_replay_and_restart(owner, tmp_path):
    client = owner.client
    assert client.get(ROOT).json() == {"contract": "research-workspace-runs.v1"}
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: client.post(ROOT, json=_launch()), range(2)))
    assert [reply.status_code for reply in responses] == [200, 200], [reply.text for reply in responses]
    first, second = (reply.json() for reply in responses)
    run_id = first["id"]
    assert second["id"] == run_id
    origin = first["workspace_launch"]
    assert origin["workspace_id"] == "workspace-a"
    assert origin["owner_user_id"] == "1"
    assert origin["launch"]["launch_id"] == "launch-workspace"
    assert first["chat_id"] is None
    assert len(owner.service.list_sessions(owner_user_id="1")) == 1

    restarted = research_service.ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=owner.manager,
    )
    owner.app.dependency_overrides[research_runs.get_research_service] = lambda: restarted
    assert client.get(f"{RUNS}/{run_id}").json()["workspace_launch"] == origin
    native = restarted.get_session(owner_user_id="1", session_id=run_id)
    assert native.active_job_id
    job = owner.manager.get_job(int(native.active_job_id))
    assert job["payload"]["session_id"] == run_id and job["payload"]["phase"] == "drafting_plan"
    assert restarted._db_for_user("1").get_chat_handoff(run_id) is None

    artifacts = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=restarted._db_for_user("1"))
    artifacts.write_json(
        owner_user_id=1,
        session_id=run_id,
        artifact_name="bundle.json",
        phase="packaging",
        job_id=native.active_job_id,
        payload={"question": _launch()["query"], "report_markdown": "# Native report", "run_id": "untrusted"},
    )
    bundle = client.get(f"{RUNS}/{run_id}/bundle").json()
    assert bundle["run_id"] == run_id and bundle["workspace_launch"] == origin
    assert bundle["report_markdown"] == "# Native report"

    body = {**_launch(), "query": "A changed question"}
    assert client.post(ROOT, json=body).status_code == 409
    assert client.post(ROOT.replace("workspace-a", "workspace-b"), json=_launch()).status_code == 409
    owner.app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=2)
    assert client.get(f"{RUNS}/{run_id}").status_code == 404
    assert client.post(ROOT, json=_launch()).status_code == 404
    owner.app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    assert client.delete(f"{RUNS}/{run_id}").status_code == 200
    assert client.post(ROOT, json=_launch()).status_code == 409


def test_workspace_launch_rejects_missing_or_changed_origin(owner):
    client = owner.client
    assert client.get(ROOT.replace("workspace-a", "missing")).status_code == 404
    assert client.post(ROOT, json=_launch(), headers={"X-TLDW-Expected-User-ID": "2"}).status_code == 412
    stale = _launch()
    stale["source_snapshot"][0]["association_version"] = 999
    assert client.post(ROOT, json=stale).status_code == 409
    unknown_output = {
        **_launch(),
        "initiating_output": {
            "owner_kind": "server_artifact",
            "owner_id": "missing",
            "owner_version": 1,
        },
    }
    assert client.post(ROOT, json=unknown_output).status_code == 409
    assert owner.service.list_sessions(owner_user_id="1") == []


@pytest.mark.parametrize("after_dispatch", (False, True))
def test_retry_replays_committed_native_dispatch_after_interruption(owner, monkeypatch, after_dispatch):
    actual = research_service.dispatch_pending_phase

    def lost_response(manager, database, session):
        if after_dispatch:
            actual(manager, database, session)
        raise RuntimeError("Simulated interruption around native dispatch")

    monkeypatch.setattr(research_service, "dispatch_pending_phase", lost_response)
    assert owner.client.post(ROOT, json=_launch()).status_code == 500
    saved = owner.service.list_sessions(owner_user_id="1")[0]
    monkeypatch.setattr(research_service, "dispatch_pending_phase", actual)
    replay = owner.client.post(ROOT, json=_launch())
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == saved.id
    assert replay.json()["active_job_id"]
    if after_dispatch:
        assert replay.json()["active_job_id"] == saved.active_job_id
    native = owner.service.get_session(owner_user_id="1", session_id=saved.id)
    assert native.phase_dispatch_json["dispatch_id"] == saved.phase_dispatch_json["dispatch_id"]
    ordinary = owner.client.post(RUNS, json={"query": "An ordinary run"})
    assert ordinary.status_code == 200
    assert owner.client.get(f"{RUNS}/{ordinary.json()['id']}").json().get("workspace_launch") is None
