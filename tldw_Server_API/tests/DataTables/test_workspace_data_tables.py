"""Exercise Workspace admission through native Media and Jobs."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import data_tables, workspaces
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.Data_Tables import jobs_worker, publication
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.DataTables.test_data_tables_jobs_integration import _run_worker_once

ROOT = "/api/v1/workspaces/workspace-a/research-data-tables"
NATIVE = "/api/v1/data-tables"
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("healthy_no_override_tts_credential_snapshot")]


@pytest.fixture
def owner(tmp_path, monkeypatch, data_tables_app_factory):
    path = tmp_path / "media.db"
    media = MediaDatabase(db_path=str(path), client_id="1")
    media_id, _, _ = media.add_media_with_keywords(
        title="Selected source",
        media_type="document",
        content="Orchid grows in the first plot.",
        owner_user_id=1,
        author="Fixture",
        ingestion_date="2026-11-24T12:00:00",
        keywords=[],
    )
    cha = CharactersRAGDB(str(tmp_path / "chacha.db"), client_id="1")
    cha.workspace_store.upsert_workspace("workspace-a", "Workspace A")
    cha.workspace_store.add_workspace_source(
        "workspace-a", {"id": "source-1", "media_id": media_id, "title": "Selected source", "selected": True}
    )
    app, jobs_path = data_tables_app_factory(path)
    app.include_router(workspaces.router, prefix="/api/v1/workspaces")
    jobs = JobManager(jobs_path, backend="sqlite")
    app.dependency_overrides[data_tables.get_job_manager] = lambda: jobs
    app.dependency_overrides[get_chacha_db_for_user] = lambda: cha
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = lambda: None
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = lambda: None

    async def principal(_request):
        return SimpleNamespace(user_id=1)

    monkeypatch.setattr(auth_deps, "get_auth_principal", principal)
    monkeypatch.setattr(jobs_worker, "get_user_media_db_path", lambda _user_id: str(path))
    monkeypatch.setattr(publication, "get_user_media_db_path", lambda _user_id: str(path))
    monkeypatch.setattr(jobs_worker, "get_default_provider", lambda: "openai")
    monkeypatch.setattr(jobs_worker, "load_and_log_configs", lambda: {"ollama_api": {"model": "orchid-model"}})
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    calls = []

    class Adapter:
        def chat(self, request, **_kwargs):
            calls.append(request)
            content = json.dumps({"columns": [{"name": "Plant"}], "rows": [["Orchid"]]})
            return {"choices": [{"message": {"content": content}}]}

    def adapter(provider):
        assert provider == "ollama"
        return Adapter()

    monkeypatch.setattr(jobs_worker, "_get_adapter", adapter)
    jobs_worker._MEDIA_DB_CACHE.clear()
    request = {
        "operation_id": "studio-table-1",
        "provider": "ollama",
        "model": "orchid-model",
        "name": "Plot plants",
        "prompt": "Extract the plants from the evidence.",
        "sources": [{
            "source_id": "source-1",
            "catalog_item_id": media_id,
            "association_version": 1,
            "catalog_item_version": media.get_media_by_id(media_id)["version"],
        }],
    }
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, request=request, jobs=jobs, media=media, cha=cha, calls=calls)
    jobs_worker._MEDIA_DB_CACHE.clear()
    cha.close_all_connections()
    media.close_connection()


def test_exact_route_and_selected_media_are_required(owner):
    client = owner.client
    chosen = {"provider": "ollama", "model": "orchid-model"}
    capability = client.get(ROOT, params=chosen)
    assert capability.status_code == 200, capability.text
    assert capability.json()["available"] is True, capability.text
    assert capability.json()["contract"] == "research-workspace-data-tables.v1"
    assert client.get(ROOT, params={**chosen, "model": "wrong"}).json()["available"] is False
    assert client.post(ROOT, json={**owner.request, "model": "wrong"}).status_code == 409
    assert client.get(ROOT.replace("workspace-a", "missing"), params=chosen).status_code == 404
    expected_user = {"X-TLDW-Expected-User-ID": "2"}
    assert client.post(ROOT, json=owner.request, headers=expected_user).status_code == 412
    for field in ("association_version", "catalog_item_version", "catalog_item_id"):
        [source] = owner.request["sources"]
        changed = {**source, field: source[field] + 1}
        assert client.post(ROOT, json={**owner.request, "sources": [changed]}).status_code == 409
    assert owner.media.list_data_tables(owner_user_id=1) == []


def test_concurrent_admission_publishes_once_and_replays_original_job(owner):
    client = owner.client
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: client.post(ROOT, json=owner.request), range(2)))
    assert [response.status_code for response in responses] == [202, 202], [r.text for r in responses]
    first, second = [response.json() for response in responses]
    table_uuid = first["table"]["uuid"]
    job_id = first["job_id"]
    assert second["table"]["uuid"] == table_uuid and second["job_uuid"] == first["job_uuid"]
    [stored] = owner.media.list_data_tables(owner_user_id=1)
    assert stored["uuid"] == table_uuid
    assert len(owner.media.list_data_table_sources(stored["id"], owner_user_id=1)) == 1
    assert client.post(ROOT, json={**owner.request, "prompt": "Changed goal"}).status_code == 409

    asyncio.run(_run_worker_once(owner.jobs))
    [call] = owner.calls
    assert call["model"] == "orchid-model"
    assert "Orchid grows" in call["messages"][0]["content"]

    inspected = client.get(f"{NATIVE}/{table_uuid}")
    assert inspected.status_code == 200, inspected.text
    assert inspected.json()["table"]["status"] == "ready"
    assert "Orchid" in str(inspected.json()["rows"])
    exported = client.get(f"{NATIVE}/{table_uuid}/export?format=csv&async_mode=sync&download=true")
    assert exported.status_code == 200, exported.text
    assert "Orchid" in exported.text

    owner.cha.workspace_store.delete_workspace_source("workspace-a", "source-1")
    replay = client.post(ROOT, json=owner.request).json()
    assert replay["job_id"] == job_id and replay["published_version"] == inspected.json()["table"]["version"]
    assert client.delete(f"{NATIVE}/{table_uuid}").status_code < 300
    assert client.post(ROOT, json=owner.request).status_code == 409


def test_retry_recovers_reservation_after_jobs_storage_failure(owner, monkeypatch):
    create = owner.jobs.create_job

    def interrupted(**kwargs):
        raise RuntimeError("temporary admission interruption")

    monkeypatch.setattr(owner.jobs, "create_job", interrupted)
    failed = owner.client.post(ROOT, json=owner.request)
    assert failed.status_code == 503, failed.text
    [reserved] = owner.media.list_data_tables(owner_user_id=1)
    monkeypatch.setattr(owner.jobs, "create_job", create)
    resumed = owner.client.post(ROOT, json=owner.request)
    assert resumed.status_code == 202, resumed.text
    assert resumed.json()["table"]["uuid"] == reserved["uuid"]


def test_worker_refuses_a_media_revision_changed_after_admission(owner):
    reply = owner.client.post(ROOT, json=owner.request)
    assert reply.status_code == 202, reply.text
    job_id = reply.json()["job_id"]
    [source] = owner.request["sources"]
    owner.media.execute_query(
        "UPDATE Media SET content = ?, version = version + 1 WHERE id = ?",
        ("Newer, unseen evidence", source["catalog_item_id"]),
        commit=True,
    )

    async def invoke():
        job = owner.jobs.acquire_next_job(domain="data_tables", queue="default", worker_id="revision-check", lease_seconds=30)
        assert job["id"] == job_id
        with pytest.raises(jobs_worker.DataTablesJobError, match="workspace_table_source_changed"):
            await jobs_worker._handle_job(job, owner.jobs)

    asyncio.run(invoke())
    assert owner.calls == []
