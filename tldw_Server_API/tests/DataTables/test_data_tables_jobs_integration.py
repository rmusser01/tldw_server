import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Data_Tables import jobs_worker, publication
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

pytestmark = pytest.mark.integration


class _StubAdapter:
    def __init__(self, payload):
        self._payload = payload

    def chat(self, _request, **_kwargs):
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def _configure_worker_paths(tmp_path: Path, monkeypatch) -> Path:
    media_path = tmp_path / "media.db"
    chacha_path = tmp_path / "chacha.db"
    monkeypatch.setattr(jobs_worker, "get_user_media_db_path", lambda _user_id: str(media_path))
    monkeypatch.setattr(publication, "get_user_media_db_path", lambda _user_id: str(media_path))
    monkeypatch.setattr(jobs_worker, "get_user_chacha_db_path", lambda _user_id: str(chacha_path))
    jobs_worker._MEDIA_DB_CACHE.clear()
    jobs_worker._CHACHA_DB_CACHE.clear()
    return media_path


def _stub_llm(monkeypatch, payload):
    monkeypatch.setattr(jobs_worker, "_get_adapter", lambda _provider: _StubAdapter(payload))
    monkeypatch.setattr(jobs_worker, "_resolve_model", lambda *_args, **_kwargs: "test-model")
    monkeypatch.setattr(jobs_worker, "provider_requires_api_key", lambda _p: False)
    monkeypatch.setattr(jobs_worker, "resolve_provider_api_key", lambda *_a, **_k: ("", {}))
    monkeypatch.setattr(jobs_worker, "get_default_provider", lambda: "openai")
    monkeypatch.setattr(jobs_worker, "load_and_log_configs", lambda: {})


async def _run_worker_once(jm: JobManager):
    cfg = WorkerConfig(
        domain="data_tables",
        queue="default",
        worker_id="worker-test",
        lease_seconds=5,
        renew_threshold_seconds=1,
        renew_jitter_seconds=0,
        bind_completion_token=True,
    )
    sdk = WorkerSDK(jm, cfg)

    async def handler(job_row):
        result = await jobs_worker._handle_job(job_row, jm)
        sdk.stop()
        return result

    async def completed(job_row, _result):
        jobs_worker.publish_completed_generation(jm, job_row["id"], expected_uuid=job_row["uuid"])

    await asyncio.wait_for(sdk.run(handler=handler, on_completed=completed), timeout=2)


@pytest.mark.asyncio
async def test_data_tables_job_lifecycle(monkeypatch, tmp_path, data_tables_app_factory):
    media_path = _configure_worker_paths(tmp_path, monkeypatch)
    app, jobs_db_path = data_tables_app_factory(media_path)

    llm_payload = {
        "columns": [
            {"name": "Term", "type": "text"},
            {"name": "Count", "type": "number"},
        ],
        "rows": [["alpha", 1], ["beta", 2]],
    }
    _stub_llm(monkeypatch, llm_payload)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Lifecycle Table",
                "prompt": "Summarize",
                "description": "demo",
                "sources": [
                    {
                        "source_type": "rag_query",
                        "source_id": "summarize",
                        "snapshot": {"chunks": [{"chunk_text": "alpha beta"}]},
                    }
                ],
                "column_hints": [{"name": "Term", "type": "text"}],
            },
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        job_id = payload["job_id"]
        table_uuid = payload["table"]["uuid"]

    jm = JobManager(Path(jobs_db_path))
    await _run_worker_once(jm)

    job_row = jm.get_job(int(job_id))
    assert job_row["status"] == "completed"

    db = MediaDatabase(db_path=str(media_path), client_id="test_client")
    try:
        table_row = db.get_data_table_by_uuid(table_uuid)
        assert table_row["status"] == "ready"
        assert table_row["row_count"] == 2
    finally:
        db.close_connection()
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_data_tables_job_cancel(monkeypatch, tmp_path, data_tables_app_factory):
    media_path = _configure_worker_paths(tmp_path, monkeypatch)
    app, jobs_db_path = data_tables_app_factory(media_path)

    llm_payload = {
        "columns": [{"name": "Term", "type": "text"}],
        "rows": [["alpha"]],
    }
    _stub_llm(monkeypatch, llm_payload)

    def _cancel_on_check(jm, job_id):
        jm.cancel_job(int(job_id), reason="test")
        return True

    monkeypatch.setattr(jobs_worker, "_is_job_cancelled", _cancel_on_check)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Cancel Table",
                "prompt": "Summarize",
                "sources": [
                    {
                        "source_type": "rag_query",
                        "source_id": "cancel",
                        "snapshot": {"chunks": [{"chunk_text": "alpha beta"}]},
                    }
                ],
            },
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        job_id = payload["job_id"]
        table_uuid = payload["table"]["uuid"]

    jm = JobManager(Path(jobs_db_path))
    await _run_worker_once(jm)

    job_row = jm.get_job(int(job_id))
    assert job_row["status"] == "cancelled"

    db = MediaDatabase(db_path=str(media_path), client_id="test_client")
    try:
        table_row = db.get_data_table_by_uuid(table_uuid)
        assert table_row["status"] == "cancelled"
    finally:
        db.close_connection()
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_data_tables_regenerate_uses_snapshot(monkeypatch, tmp_path, data_tables_app_factory):
    media_path = _configure_worker_paths(tmp_path, monkeypatch)
    app, jobs_db_path = data_tables_app_factory(media_path)

    llm_payload = {
        "columns": [
            {"name": "Term", "type": "text"},
            {"name": "Count", "type": "number"},
        ],
        "rows": [["alpha", 1], ["beta", 2]],
    }
    _stub_llm(monkeypatch, llm_payload)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/data-tables/generate",
            json={
                "name": "Regen Table",
                "prompt": "Summarize",
                "sources": [
                    {
                        "source_type": "rag_query",
                        "source_id": "regen",
                        "snapshot": {"chunks": [{"chunk_text": "alpha beta"}]},
                    }
                ],
            },
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        table_uuid = payload["table"]["uuid"]

    jm = JobManager(Path(jobs_db_path))
    await _run_worker_once(jm)

    async def _fail_rag(*_args, **_kwargs):
        raise AssertionError("rag_query should not be resolved during regenerate")

    monkeypatch.setattr(jobs_worker, "_resolve_rag_query_source", _fail_rag)

    with TestClient(app) as client:
        regen = client.post(f"/api/v1/data-tables/{table_uuid}/regenerate", json={})
        assert regen.status_code == 202, regen.text
        regen_job_id = regen.json()["job_id"]

    await _run_worker_once(jm)

    regen_job = jm.get_job(int(regen_job_id))
    assert regen_job["status"] == "completed"

    db = MediaDatabase(db_path=str(media_path), client_id="test_client")
    try:
        table_row = db.get_data_table_by_uuid(table_uuid)
        assert table_row["status"] == "ready"
        assert table_row["row_count"] == 2
    finally:
        db.close_connection()
        app.dependency_overrides.clear()


@contextmanager
def _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory):
    """Seed saved content through Media abstractions and acquire its real regeneration."""
    media_path = _configure_worker_paths(tmp_path, monkeypatch)
    db = MediaDatabase(db_path=str(media_path), client_id="test_client")
    try:
        table = db.create_data_table(name="Saved Rowan facts", prompt="Extract facts", status="ready")
        table_id = table["id"]
        db.persist_data_table_generation(
            table_id,
            columns=[{"column_id": "old-fact", "name": "Fact", "type": "text"}],
            rows=[{"row_id": "old-row", "row_json": {"old-fact": "Selene"}}],
            sources=[
                {
                    "source_type": "rag_query",
                    "source_id": "Rowan",
                    "snapshot_json": {"chunks": [{"chunk_text": "The telescope is named Selene."}]},
                }
            ],
        )
        original = _s03_content(db, table_id)
        app, jobs_path = data_tables_app_factory(media_path)
        _stub_llm(
            monkeypatch,
            {
                "columns": [{"name": "Fact", "type": "text"}],
                "rows": [["Replacement"]],
            },
        )
        with TestClient(app) as client:
            response = client.post(f"/api/v1/data-tables/{table['uuid']}/regenerate", json={})
            assert response.status_code == 202, response.text  # nosec B101 # pytest API oracle.
        jm = JobManager(Path(jobs_path))
        job = jm.acquire_next_job(
            domain="data_tables",
            queue="default",
            lease_seconds=60,
            worker_id="s03-publication-test",
        )
        assert job and job["id"] == response.json()["job_id"]  # nosec B101 # acquired identity oracle.
        yield db, jm, job, app, original
    finally:
        db.close_connection()


def _s03_content(db, table_id):
    return {"columns": db.list_data_table_columns(table_id), "rows": db.list_data_table_rows(table_id)}


def _s03_complete(jm, job, result):
    return jm.complete_job(
        job["id"],
        result=result,
        expected_uuid=job["uuid"],
        worker_id=job["worker_id"],
        lease_id=job["lease_id"],
        completion_token=job["lease_id"],
        enforce=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_via", ["dedicated", "manager", "admin_batch"])
async def test_regeneration_cancel_before_completion_preserves_saved_content(
    monkeypatch, tmp_path, data_tables_app_factory, cancel_via
):
    """Any accepted cancellation must win before generated data becomes visible."""
    from tldw_Server_API.app.core.Jobs.admin_operations import batch_cancel

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, original = case
        await jobs_worker._handle_job(job, jm)
        if cancel_via == "dedicated":
            with TestClient(app) as client:
                cancelled = client.delete(f"/api/v1/data-tables/jobs/{job['id']}")
            accepted = cancelled.status_code == 200 and cancelled.json()["success"]
        elif cancel_via == "manager":
            accepted = jm.cancel_job(job["id"], reason="s03_regression")
        else:
            accepted = (
                batch_cancel(jm, domain="data_tables", queue=None, job_type=None, job_id=job["id"], dry_run=False) == 1
            )
        assert accepted  # nosec B101 # cancellation acknowledgement oracle.
        assert jm.get_job(job["id"])["status"] == "cancelled"  # nosec B101 # real Jobs oracle.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # preserved content oracle.


def test_generation_status_read_failure_never_grants_publication(monkeypatch, tmp_path):
    """An unavailable authoritative Jobs read cannot authorize a Media write."""
    import sqlite3

    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    jm = JobManager(tmp_path / "jobs.db")

    def unavailable(_job_id):
        raise sqlite3.OperationalError("S03 status unavailable")

    monkeypatch.setattr(jm, "get_job", unavailable)
    with pytest.raises(DataTablesJobError):
        jobs_worker._is_job_cancelled(jm, 71)


@pytest.mark.asyncio
async def test_completed_generation_without_callback_recovers_through_authenticated_job_get(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, original = case
        result = await jobs_worker._handle_job(job, jm)
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # no premature publication.
        assert _s03_complete(jm, job, result)  # nosec B101 # actual completion oracle.
        with TestClient(app) as client:
            response = client.get(f"/api/v1/data-tables/jobs/{job['id']}")
        assert response.status_code == 200, response.text  # nosec B101 # API oracle.
        public = response.json()
        assert public["result"]["publication_status"] == "ready"  # nosec B101 # actual saved state.
        assert "_data_table_publication" not in public["result"]  # nosec B101 # private staging guard.
        assert "The telescope is named Selene." not in response.text  # nosec B101 # snapshot projection guard.
        current = _s03_content(db, job["payload"]["table_id"])
        assert json.loads(current["rows"][0]["row_json"]) == {current["columns"][0]["column_id"]: "Replacement"}  # nosec B101 # persisted cell oracle.


@pytest.mark.asyncio
async def test_completed_generation_replay_preserves_a_later_real_content_edit(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, _original = case
        result = await jobs_worker._handle_job(job, jm)
        assert _s03_complete(jm, job, result)  # nosec B101 # completion oracle.
        assert jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db) == "ready"  # nosec B101 # publication oracle.
        columns = db.list_data_table_columns(job["payload"]["table_id"])
        with TestClient(app) as client:
            edit = client.put(
                f"/api/v1/data-tables/{job['payload']['table_uuid']}/content",
                json={
                    "columns": [
                        {"column_id": col["column_id"], "name": col["name"], "type": col["type"]} for col in columns
                    ],
                    "rows": [{columns[0]["column_id"]: "Later user edit"}],
                },
            )
        assert edit.status_code == 200, edit.text  # nosec B101 # legitimate edit oracle.
        edited = _s03_content(db, job["payload"]["table_id"])
        assert jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db) == "ready"  # nosec B101 # idempotent replay oracle.
        assert _s03_content(db, job["payload"]["table_id"]) == edited  # nosec B101 # later edit preservation.


@pytest.mark.asyncio
async def test_completed_generation_conflict_preserves_edits_and_is_not_artifact_success(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, _original = case
        result = await jobs_worker._handle_job(job, jm)
        with TestClient(app) as client:
            edit = client.put(
                f"/api/v1/data-tables/{job['payload']['table_uuid']}/content",
                json={
                    "columns": [{"column_id": "old-fact", "name": "Fact", "type": "text"}],
                    "rows": [{"old-fact": "Newer saved edit"}],
                },
            )
            assert edit.status_code == 200, edit.text  # nosec B101 # actual intervening edit.
            saved = _s03_content(db, job["payload"]["table_id"])
            assert _s03_complete(jm, job, result)  # nosec B101 # computation completion oracle.
            status = client.get(f"/api/v1/data-tables/jobs/{job['id']}")
        assert status.status_code == 200, status.text  # nosec B101 # truthful status response.
        assert status.json()["status"] == "completed"  # nosec B101 # actual Jobs lifecycle retained.
        assert status.json()["result"]["publication_status"] == "conflict"  # nosec B101 # no false artifact success.
        assert _s03_content(db, job["payload"]["table_id"]) == saved  # nosec B101 # conflict preserves edits.


@pytest.mark.asyncio
async def test_rejected_completion_cannot_publish_staged_generation(monkeypatch, tmp_path, data_tables_app_factory):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, _app, original = case
        result = await jobs_worker._handle_job(job, jm)
        assert jm.cancel_job(job["id"], reason="s03_cancel")  # nosec B101 # real cancel wins.
        assert not _s03_complete(jm, job, result)  # nosec B101 # exact completion rejects cancelled row.
        assert (
            jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db) == "not_completed"
        )  # nosec B101 # publisher refuses rejected receipt.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # original saved content.


@pytest.mark.asyncio
async def test_publisher_uses_stored_truncation_instead_of_the_original_callback_argument(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, original = case
        result = await jobs_worker._handle_job(job, jm)
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "32")
        monkeypatch.setenv("JOBS_JSON_TRUNCATE", "true")
        assert _s03_complete(jm, job, result)  # nosec B101 # real Jobs truncation path.
        with TestClient(app) as client:
            status = client.get(f"/api/v1/data-tables/jobs/{job['id']}")
        assert status.status_code == 200, status.text  # nosec B101 # explicit result state.
        assert status.json()["result"]["publication_status"] == "unavailable"  # nosec B101 # truncation cannot publish.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # original preserved.


@pytest.mark.parametrize("change", ["cancel", "wrong_lease", "changed_payload", "changed_payload_type"])
def test_legacy_protocol_upgrade_requires_the_unchanged_acquired_delivery(tmp_path, change):
    jm = JobManager(tmp_path / "jobs.db")
    created = jm.create_job(
        domain="data_tables",
        queue="default",
        job_type="data_table_generate",
        owner_user_id="1",
        payload={"table_id": 9},
    )
    job = jm.acquire_next_job(domain="data_tables", queue="default", worker_id="s03-marker", lease_seconds=60)
    if change == "cancel":
        jm.cancel_job(job["id"], reason="before marker")
    elif change == "changed_payload":
        jm.replace_job_payload(job["id"], payload={"table_id": 10}, expected_uuid=job["uuid"])
    elif change == "changed_payload_type":
        jm.replace_job_payload(job["id"], payload={"table_id": 9.0}, expected_uuid=job["uuid"])
    before = jm.get_job(created["id"])["payload"]
    upgraded = jm.replace_job_payload(
        job["id"],
        payload={**job["payload"], "data_table_publication_version": 1},
        expected_uuid=job["uuid"],
        expected_domain="data_tables",
        expected_worker_id=job["worker_id"],
        expected_lease_id="stale" if change == "wrong_lease" else job["lease_id"],
        expected_payload=job["payload"],
    )
    assert not upgraded  # nosec B101 # exact acquired delivery CAS oracle.
    assert jm.get_job(created["id"])["payload"] == before  # nosec B101 # no overwritten payload.


@pytest.mark.asyncio
async def test_legacy_queued_generation_without_table_uuid_upgrades_and_publishes(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, _app, _original = case
        payload = dict(job["payload"])
        payload.pop("data_table_publication_version")
        payload.pop("table_uuid")
        assert jm.replace_job_payload(job["id"], payload=payload, expected_uuid=job["uuid"])  # nosec B101 # legacy setup.
        job = jm.get_job(job["id"])
        result = await jobs_worker._handle_job(job, jm)
        assert _s03_complete(jm, job, result)  # nosec B101 # exact legacy delivery completes.
        assert jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db) == "ready"  # nosec B101 # legacy upgrade remains usable.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "foreign_owner",
        "foreign_job",
        "foreign_table",
        "version",
        "completion_token",
        "protocol_bool",
        "protocol_float",
        "receipt_bool",
        "receipt_float",
    ],
)
async def test_invalid_stored_publication_never_changes_saved_content(
    monkeypatch, tmp_path, data_tables_app_factory, damage
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, _app, original = case
        result = await jobs_worker._handle_job(job, jm)
        if damage == "missing":
            del result["_data_table_publication"]
        elif damage.startswith("protocol_"):
            changed_payload = {
                **job["payload"],
                "data_table_publication_version": True if damage.endswith("bool") else 1.0,
            }
            jm.replace_job_payload(job["id"], payload=changed_payload, expected_uuid=job["uuid"])
        elif damage.startswith("receipt_"):
            result["_data_table_publication"]["version"] = True if damage.endswith("bool") else 1.0
        elif damage != "completion_token":
            key, value = {
                "foreign_owner": ("owner", "other"),
                "foreign_job": ("job_uuid", "other"),
                "foreign_table": ("table_uuid", "other"),
                "version": ("expected_version", True),
            }[damage]
            result["_data_table_publication"][key] = value
        assert jm.complete_job(
            job["id"],
            result=result,
            expected_uuid=job["uuid"],
            worker_id=job["worker_id"],
            lease_id=job["lease_id"],
            completion_token="other-attempt" if damage == "completion_token" else job["lease_id"],
            enforce=True,
        )  # nosec B101 # actual stored result boundary.
        assert (
            jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db) == "unavailable"
        )  # nosec B101 # invalid authority rejected.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # original content intact.


@pytest.mark.asyncio
@pytest.mark.parametrize("archived", [False, True])
async def test_table_reopen_recovers_a_completed_generation_without_the_original_client_receipt(
    monkeypatch, tmp_path, data_tables_app_factory, archived
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, _original = case
        result = await jobs_worker._handle_job(job, jm)
        assert _s03_complete(jm, job, result)  # nosec B101 # computation completed, callback absent.
        if archived:
            monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
            assert jm.prune_jobs(statuses=["completed"], older_than_days=0, domain="data_tables") == 1  # nosec B101 # private archive transition.
            assert jm.get_job(job["id"]) is None  # nosec B101 # active row genuinely gone.
        with TestClient(app) as client:
            response = client.get(f"/api/v1/data-tables/{job['payload']['table_uuid']}")
        assert response.status_code == 200, response.text  # nosec B101 # ordinary reopen, no client job ID.
        body = response.json()
        assert body["table"]["status"] == "ready"  # nosec B101 # publication recovered.
        assert body["rows"][0]["data"] == {body["columns"][0]["column_id"]: "Replacement"}  # nosec B101 # actual generated cell.


@pytest.mark.asyncio
async def test_an_older_acquired_generation_cannot_change_a_newer_admission(
    monkeypatch, tmp_path, data_tables_app_factory
):
    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, original = case
        with TestClient(app) as client:
            newer = client.post(
                f"/api/v1/data-tables/{job['payload']['table_uuid']}/regenerate",
                json={"prompt": "Latest user intent"},
            )
        assert newer.status_code == 202, newer.text  # nosec B101 # real newer admission.
        before = db.get_data_table(job["payload"]["table_id"])
        with pytest.raises(DataTablesJobError):
            await jobs_worker._handle_job(job, jm)
        assert db.get_data_table(job["payload"]["table_id"]) == before  # nosec B101 # no stale status/version writes.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # saved content preserved.


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "error"])
async def test_superseded_generation_cannot_mark_newer_work_failed_or_cancelled(
    monkeypatch, tmp_path, data_tables_app_factory, failure
):
    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, original = case
        progress = jm.update_job_progress
        after_admission = []

        def newer_request(job_id, **kwargs):
            if kwargs.get("progress_percent") == 5.0:
                with TestClient(app) as client:
                    response = client.post(
                        f"/api/v1/data-tables/{job['payload']['table_uuid']}/regenerate",
                        json={"prompt": "Newer request"},
                    )
                assert response.status_code == 202, response.text  # nosec B101 # real replacement admission.
                after_admission.append(db.get_data_table(job["payload"]["table_id"]))
                if failure == "error":
                    raise DataTablesJobError("old_generation_failed", retryable=False)
                assert jm.cancel_job(job_id, reason="old request")  # nosec B101 # actual old Job cancellation.
            return progress(job_id, **kwargs)

        monkeypatch.setattr(jm, "update_job_progress", newer_request)
        if failure == "error":
            with pytest.raises(DataTablesJobError):
                await jobs_worker._handle_job(job, jm)
        else:
            assert (await jobs_worker._handle_job(job, jm))["cancelled"]  # nosec B101 # worker saw real cancel.
        assert len(after_admission) == 1  # nosec B101 # exactly one superseding request.
        assert db.get_data_table(job["payload"]["table_id"]) == after_admission[0]  # nosec B101 # no stale metadata.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # no stale content.


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["owner", "queue", "uuid", "key", "malformed"])
async def test_worker_refuses_a_foreign_or_malformed_generation_reservation_without_writes(
    monkeypatch, tmp_path, data_tables_app_factory, damage
):
    from uuid import uuid4

    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, _app, original = case
        reservation = json.loads(db.get_data_table(job["payload"]["table_id"])["generation_job_json"])
        if damage == "malformed":
            reservation = "not a reservation"
        else:
            reservation[damage] = str(uuid4()) if damage in {"uuid", "key"} else "foreign"
        before = db.update_data_table(job["payload"]["table_id"], generation_job=reservation)
        with pytest.raises(DataTablesJobError):
            await jobs_worker._handle_job(job, jm)
        assert db.get_data_table(job["payload"]["table_id"]) == before  # nosec B101 # invalid binding has no side effects.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # saved cells intact.


def test_fast_worker_completion_is_not_overwritten_by_the_admission_ack(monkeypatch, tmp_path, data_tables_app_factory):
    from concurrent.futures import ThreadPoolExecutor

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, _original = case
        create = JobManager.create_job

        def complete_before_ack(manager, **kwargs):
            admitted = create(manager, **kwargs)
            acquired = jm.acquire_next_job(
                domain="data_tables", queue="default", worker_id="fast-worker", lease_seconds=60
            )
            assert acquired["uuid"] == admitted["uuid"]  # nosec B101 # exact newly admitted work.

            def finish():
                result = asyncio.run(jobs_worker._handle_job(acquired, jm))
                assert _s03_complete(jm, acquired, result)  # nosec B101 # real exact completion.
                assert (
                    publication.publish_completed_generation(jm, acquired["id"], expected_uuid=acquired["uuid"], db=db)
                    == "ready"
                )  # nosec B101 # real pre-ACK publication.

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(finish).result(timeout=15)
            return admitted

        monkeypatch.setattr(JobManager, "create_job", complete_before_ack)
        with TestClient(app) as client:
            response = client.post(f"/api/v1/data-tables/{job['payload']['table_uuid']}/regenerate", json={})
        assert response.status_code == 202, response.text  # nosec B101 # admission itself succeeded.
        assert response.json()["table"]["status"] == "ready"  # nosec B101 # queued ACK cannot replace ready state.
        assert db.get_data_table(job["payload"]["table_id"])["status"] == "ready"  # nosec B101 # canonical state ready.


@pytest.mark.asyncio
async def test_admin_reopen_uses_the_actual_media_owner_for_publication(monkeypatch, tmp_path, data_tables_app_factory):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, _app, _original = case
        result = await jobs_worker._handle_job(job, jm)
        assert _s03_complete(jm, job, result)  # nosec B101 # completion has no callback.
        admin = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="admin-request")
        try:
            recovered = publication.recover_table_generation(jm, admin, db.get_data_table(job["payload"]["table_id"]))
            assert recovered["status"] == "ready"  # nosec B101 # request principal does not become Media owner.
            assert _s03_content(db, job["payload"]["table_id"])["rows"][0]["row_json"] != '{"old-fact": "Selene"}'  # nosec B101 # generated cells applied.
        finally:
            admin.close_connection()


@pytest.mark.asyncio
async def test_modern_job_cannot_claim_a_missing_generation_reservation(monkeypatch, tmp_path, data_tables_app_factory):
    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, _app, original = case
        before = db.update_data_table(job["payload"]["table_id"], generation_job=None)
        with pytest.raises(DataTablesJobError):
            await jobs_worker._handle_job(job, jm)
        assert db.get_data_table(job["payload"]["table_id"]) == before  # nosec B101 # modern NULL is not a legacy claim.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # no content writes.


def test_definitive_regeneration_admission_refusal_retires_its_queued_marker(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, _jm, job, app, original = case

        def refuse(_manager, **_kwargs):
            raise ValueError("Quota exceeded")

        monkeypatch.setattr(JobManager, "create_job", refuse)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(f"/api/v1/data-tables/{job['payload']['table_uuid']}/regenerate", json={})
        assert response.status_code == 409  # nosec B101 # explicit refusal, not endless queued.
        saved = db.get_data_table(job["payload"]["table_id"])
        assert saved["status"] == "failed"  # nosec B101 # own rejected admission is retired.
        assert saved["generation_job_json"] is not None  # nosec B101 # reservation retained for audit/recovery.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # no saved data loss.


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ["before_bind", "after_stage"])
async def test_late_old_lease_error_does_not_invalidate_the_new_attempt_receipt(
    monkeypatch, tmp_path, data_tables_app_factory, ordering
):
    from concurrent.futures import ThreadPoolExecutor

    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, old, _app, _original = case
        progress = jm.update_job_progress
        newer = []
        if ordering == "before_bind":
            update = MediaDatabase.update_data_table
            running_calls = []

            def old_error_before_new_bind(media, table_id, **kwargs):
                if kwargs.get("status") == "running":
                    running_calls.append(table_id)
                    if len(running_calls) == 2:
                        current = media.get_data_table(table_id)
                        jobs_worker._mark_table_for_job_error(
                            media,
                            table_id=table_id,
                            user_id=current["client_id"],
                            exc=DataTablesJobError("old_error_before_new_bind", retryable=False),
                            generation_job=current["generation_job_json"],
                            expected_version=current["version"],
                        )
                return update(media, table_id, **kwargs)

            monkeypatch.setattr(MediaDatabase, "update_data_table", old_error_before_new_bind)

        def redeliver(job_id, **kwargs):
            if kwargs.get("progress_percent") == 5.0 and not newer:
                assert jm.release_job(job_id, worker_id=old["worker_id"], lease_id=old["lease_id"], enforce=True)  # nosec B101 # real lease relinquishment.
                acquired = jm.acquire_next_job(
                    domain="data_tables", queue="default", worker_id="new-lease", lease_seconds=60
                )
                assert acquired["uuid"] == old["uuid"] and acquired["lease_id"] != old["lease_id"]  # nosec B101 # same Job, actual different attempt.
                newer.append(acquired)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(lambda: asyncio.run(jobs_worker._handle_job(acquired, jm))).result(timeout=15)
                newer.extend([result, db.get_data_table(old["payload"]["table_id"])])
                raise DataTablesJobError("late_old_attempt_error", retryable=False)
            return progress(job_id, **kwargs)

        monkeypatch.setattr(jm, "update_job_progress", redeliver)
        with pytest.raises(DataTablesJobError):
            await jobs_worker._handle_job(old, jm)
        assert len(newer) == 3, "Replacement lease could not bind and stage its result"  # nosec B101 # preserve the actual handoff failure.
        assert db.get_data_table(old["payload"]["table_id"]) == newer[2]  # nosec B101 # stale lease cannot mutate state/version.
        assert _s03_complete(jm, newer[0], newer[1])  # nosec B101 # newer exact completion remains valid.
        assert (
            publication.publish_completed_generation(jm, newer[0]["id"], expected_uuid=newer[0]["uuid"], db=db)
            == "ready"
        )  # nosec B101 # stale error did not poison its publication.


@pytest.mark.asyncio
async def test_reopening_a_publication_conflict_does_not_rewrite_failure_metadata_repeatedly(
    monkeypatch, tmp_path, data_tables_app_factory
):
    with _s03_publication_case(monkeypatch, tmp_path, data_tables_app_factory) as case:
        db, jm, job, app, original = case
        result = await jobs_worker._handle_job(job, jm)
        assert _s03_complete(jm, job, result)  # nosec B101 # completed receipt.
        db.update_data_table(job["payload"]["table_id"], name="Saved newer edit")
        with TestClient(app) as client:
            first = client.get(f"/api/v1/data-tables/{job['payload']['table_uuid']}")
            before = db.get_data_table(job["payload"]["table_id"])
            second = client.get(f"/api/v1/data-tables/{job['payload']['table_uuid']}")
        assert first.status_code == second.status_code == 200  # nosec B101 # normal table reopen.
        assert first.json()["table"]["status"] == "failed"  # nosec B101 # conflict explicitly retired.
        assert db.get_data_table(job["payload"]["table_id"]) == before  # nosec B101 # failure replay is read-only.
        assert _s03_content(db, job["payload"]["table_id"]) == original  # nosec B101 # never replaces a newer edit.


@pytest.mark.asyncio
async def test_local_provider_alias_without_key_completes_generation(monkeypatch, tmp_path, data_tables_app_factory):
    """The real credential policy must admit a registered keyless local alias."""
    media_path = _configure_worker_paths(tmp_path, monkeypatch)
    app, jobs_db_path = data_tables_app_factory(media_path)
    calls = []

    class LocalAdapter:
        def chat(self, request, **_kwargs):
            calls.append(request)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "columns": [{"name": "Fact", "type": "text"}],
                                    "rows": [["Selene"]],
                                }
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(jobs_worker, "_get_adapter", lambda _provider: LocalAdapter())
    monkeypatch.setattr(jobs_worker, "resolve_provider_api_key", lambda *_args, **_kwargs: ("", {}))
    monkeypatch.setattr(jobs_worker, "DEFAULT_LLM_PROVIDER", "llamacpp")
    monkeypatch.setattr(jobs_worker, "load_and_log_configs", lambda: {"llama_api": {"model": "local-test-model"}})
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/data-tables/generate",
                json={
                    "name": "Keyless local alias",
                    "prompt": "Extract the telescope name",
                    "sources": [
                        {
                            "source_type": "rag_query",
                            "source_id": "local",
                            "snapshot": {"chunks": [{"chunk_text": "The telescope is Selene."}]},
                        }
                    ],
                },
            )
            assert response.status_code == 202  # nosec B101 -- actual API admission.
            receipt = response.json()
        manager = JobManager(Path(jobs_db_path))
        sdk = WorkerSDK(
            manager,
            WorkerConfig(
                domain="data_tables",
                queue="default",
                worker_id="keyless-local-test",
                lease_seconds=5,
                renew_threshold_seconds=1,
                renew_jitter_seconds=0,
                bind_completion_token=True,
            ),
        )

        async def handle(job):
            try:
                return await jobs_worker._handle_job(job, manager)
            finally:
                sdk.stop()

        async def completed(job, _result):
            jobs_worker.publish_completed_generation(manager, job["id"], expected_uuid=job["uuid"])

        await asyncio.wait_for(sdk.run(handler=handle, on_completed=completed), timeout=2)
        job = manager.get_job(receipt["job_id"])
        assert job["status"] == "completed", job.get("error_message")  # nosec B101 -- real SDK completion.
        assert len(calls) == 1 and calls[0]["api_key"] == "" and calls[0]["model"] == "local-test-model"  # nosec B101 -- actual keyless configured model.
        db = MediaDatabase(db_path=str(media_path), client_id="test_client")
        try:
            table = db.get_data_table_by_uuid(receipt["table"]["uuid"])
            assert table["status"] == "ready" and table["row_count"] == 1  # nosec B101 -- actual published result.
            rows = db.list_data_table_rows(table["id"], limit=10, offset=0)
            assert list(json.loads(rows[0]["row_json"]).values()) == ["Selene"]  # nosec B101 -- real stored content.
        finally:
            db.close_connection()
    finally:
        app.dependency_overrides.clear()
