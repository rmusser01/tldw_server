"""Real Media/Jobs publication, upgrade and concurrency across supported backends."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from uuid import uuid4

import pytest
from loguru import logger

from tldw_Server_API.app.core.Data_Tables import jobs_worker, publication
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("healthy_no_override_tts_credential_snapshot")]


@pytest.fixture(params=["sqlite", "postgres"])
def publication_store(request, tmp_path):
    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgres" else None
    path = str(tmp_path / "media.db")
    metadata = []

    def observe(message):
        extra = message.record["extra"]
        if extra.get("sqlstate"):
            metadata.append((str(extra.get("exception_type")), str(extra["sqlstate"])))

    sink = logger.add(observe, level="ERROR")
    db = None
    try:
        with scoped_context(user_id=401, is_admin=False):
            db = MediaDatabase(db_path=path, client_id="401", backend=backend)
        yield db, backend, path
    except Exception as exc:
        exc.add_note(f"S03 database error classes only: {metadata}")
        raise
    finally:
        if metadata:
            print("S03_DB_CLASSES=" + json.dumps(metadata))
        logger.remove(sink)
        if db is not None:
            db.close_connection()


def _replace(db, table_id, value):
    column_id = str(uuid4())
    return db.persist_data_table_generation(
        table_id,
        columns=[{"column_id": column_id, "name": "Fact", "type": "text", "position": 0}],
        rows=[{"row_id": str(uuid4()), "row_index": 0, "row_json": {column_id: value}}],
        owner_user_id="401",
    )


def _contents(db, table_id):
    return db.list_data_table_columns(table_id), db.list_data_table_rows(table_id)


def test_existing_index_upgrade_retains_history_and_live_uniqueness(publication_store):
    db, backend, path = publication_store
    with scoped_context(user_id=401, is_admin=False):
        table = db.create_data_table(name="Rowan history", prompt="Extract facts", status="ready")
        _replace(db, table["id"], "Selene")
        # Recreate the old released index contract, then use normal schema reopen.
        for table_name, position, old_index, new_index in (
            (
                "data_table_columns",
                "position",
                "ux_data_table_columns_table_position_active",
                "ux_data_table_columns_table_position_live",
            ),
            (
                "data_table_rows",
                "row_index",
                "ux_data_table_rows_table_index_active",
                "ux_data_table_rows_table_index_live",
            ),
        ):
            db.execute_query(
                f"CREATE UNIQUE INDEX {old_index} ON {table_name}(table_id, {position}, deleted)", commit=True
            )  # nosec B608 # fixed schema identifiers above.
            db.execute_query(f"DROP INDEX {new_index}", commit=True)
        db.execute_query("ALTER TABLE data_tables DROP COLUMN generation_job_json", commit=True)
        db.close_connection()
        reopened = MediaDatabase(db_path=path, client_id="401", backend=backend)
        try:
            assert reopened.get_data_table(table["id"])["generation_job_json"] is None  # nosec B101 # genuine older schema gets nullable reservation.
            for value in ("1988", "Friday", "Friday at 18:00"):
                _replace(reopened, table["id"], value)
            columns = reopened.list_data_table_columns(table["id"], include_deleted=True)
            rows = reopened.list_data_table_rows(table["id"], include_deleted=True)
            assert len(columns) == len(rows) == 4  # nosec B101 # every generation retained.
            assert json.loads(rows[0]["row_json"]) == {columns[0]["column_id"]: "Selene"}  # nosec B101 # original history retained.
            for insert in (
                lambda: reopened.insert_data_table_columns(
                    table["id"], [{"column_id": str(uuid4()), "name": "Duplicate", "type": "text", "position": 0}]
                ),
                lambda: reopened.insert_data_table_rows(
                    table["id"],
                    [{"row_id": str(uuid4()), "row_index": 0, "row_json": {columns[-1]["column_id"]: "duplicate"}}],
                ),
            ):
                with pytest.raises((DatabaseError, sqlite3.Error)):
                    with reopened.transaction():
                        insert()
            active_columns, active_rows = _contents(reopened, table["id"])
            assert len(active_columns) == len(active_rows) == 1  # nosec B101 # active uniqueness intact.
        finally:
            reopened.close_connection()


def _complete_generation(
    db, backend, tmp_path, monkeypatch, *, modern=False, background=False, document_source=False, source_owner=401
):
    if backend is None:
        jm = JobManager(tmp_path / "jobs.db")
    else:
        c = backend.config
        dsn = f"postgresql://{quote(c.pg_user, safe='')}:{quote(c.pg_password, safe='')}@{c.pg_host}:{c.pg_port}/{quote(c.pg_database, safe='')}"
        jm = JobManager(backend="postgres", db_url=dsn)
    reservation = {"key": str(uuid4()), "owner": "401", "queue": "default"} if modern else None
    table = db.create_data_table(
        name="Rowan completion", prompt="Extract facts", status="ready", generation_job=reservation
    )
    _replace(db, table["id"], "Selene")
    if document_source:
        with scoped_context(user_id=source_owner, is_admin=False):
            source_path = tmp_path / ("media.db" if backend is not None or source_owner == 401 else f"media-{source_owner}.db")
            source_db = MediaDatabase(
                db_path=str(source_path), client_id=str(source_owner), backend=backend,
                existing_only=backend is not None or source_owner == 401,
            )
            try:
                media_id, _media_uuid, _content_hash = source_db.add_media_with_keywords(
                    title="Rowan background source",
                    media_type="document",
                    content="The telescope is named Selene.",
                    owner_user_id=source_owner,
                    author="Rowan curator",
                    ingestion_date="2026-10-27T12:00:00",
                    keywords=[],
                )
            finally:
                source_db.close_connection()
    payload = {
        "table_id": table["id"],
        "table_uuid": table["uuid"],
        "user_id": "401",
        "data_table_publication_version": 1,
        "prompt": "Extract facts",
        "sources": (
            [{"source_type": "document", "source_id": str(media_id)}]
            if document_source
            else [
                {"source_type": "rag_query", "source_id": "Rowan", "snapshot": {"chunks": [{"chunk_text": "Selene"}]}}
            ]
        ),
    }
    if modern:
        payload["generation_key"] = reservation["key"]
    jm.create_job(
        domain="data_tables",
        queue="default",
        job_type="data_table_generate",
        owner_user_id="401",
        payload=payload,
        idempotency_key=reservation["key"] if modern else None,
    )
    job = jm.acquire_next_job(domain="data_tables", queue="default", worker_id="publication-test", lease_seconds=60)

    class Adapter:
        def chat(self, _request, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"columns": [{"name": "Fact", "type": "text"}], "rows": [["Replacement"]]}
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(jobs_worker, "_get_media_db", lambda owner: db if owner == "401" else None)
    monkeypatch.setattr(jobs_worker, "_get_adapter", lambda _provider: Adapter())
    monkeypatch.setattr(jobs_worker, "_resolve_model", lambda *_args: "test-model")
    monkeypatch.setattr(jobs_worker, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(jobs_worker, "resolve_provider_api_key", lambda *_args, **_kwargs: ("", {}))
    monkeypatch.setattr(jobs_worker, "load_and_log_configs", lambda: {})
    if background:
        with scoped_context(user_id=None, is_admin=False):
            result = asyncio.run(jobs_worker._handle_job(job, jm))
    else:
        result = asyncio.run(jobs_worker._handle_job(job, jm))
    assert jm.complete_job(
        job["id"],
        result=result,
        expected_uuid=job["uuid"],
        worker_id=job["worker_id"],
        lease_id=job["lease_id"],
        completion_token=job["lease_id"],
        enforce=True,
    )  # nosec B101 # exact real completion.
    return jm, job, table


def test_concurrent_publication_is_idempotent_and_later_edits_survive(publication_store, tmp_path, monkeypatch):
    db, backend, path = publication_store
    with scoped_context(user_id=401, is_admin=False):
        jm, job, table = _complete_generation(db, backend, tmp_path, monkeypatch)
    barrier = threading.Barrier(2)

    def publish():
        with scoped_context(user_id=401, is_admin=False):
            handle = MediaDatabase(db_path=path, client_id="401", backend=backend, existing_only=True)
            try:
                barrier.wait(timeout=15)
                return jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=handle)
            finally:
                handle.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
        assert [future.result(timeout=15) for future in futures] == ["ready", "ready"]  # nosec B101 # concurrent replay.
    with scoped_context(user_id=401, is_admin=False):
        assert len(db.list_data_table_rows(table["id"], include_deleted=True)) == 2  # nosec B101 # exactly one publication.
        _replace(db, table["id"], "Later user edit")
        edited = _contents(db, table["id"])
        assert jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db) == "ready"  # nosec B101 # durable replay.
        assert _contents(db, table["id"]) == edited  # nosec B101 # no overwrite of later edits.
    with scoped_context(user_id=402, is_admin=False):
        foreign = MediaDatabase(db_path=path, client_id="402", backend=backend, existing_only=True)
        try:
            assert foreign.get_data_table(table["id"]) is None  # nosec B101 # normal owner isolation.
            assert (
                jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=foreign)
                == "unavailable"
            )  # nosec B101 # foreign borrowed store refused.
        finally:
            foreign.close_connection()


@pytest.mark.parametrize("archived", [False, True])
def test_modern_reserved_generation_recovers_the_exact_active_or_archived_attempt(
    publication_store, tmp_path, monkeypatch, archived
):
    db, backend, _path = publication_store
    with scoped_context(user_id=401, is_admin=False):
        jm, job, table = _complete_generation(db, backend, tmp_path, monkeypatch, modern=True)
        pending = db.get_data_table(table["id"])
        binding = json.loads(pending["generation_job_json"])
        assert pending["status"] == "running"  # nosec B101 # completed computation has not published.
        assert binding["key"] == jm.get_job(job["id"])["idempotency_key"]  # nosec B101 # actual modern admission key.
        assert binding["uuid"] == job["uuid"]  # nosec B101 # exact acquired identity.
        if archived:
            monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
            assert jm.prune_jobs(statuses=["completed"], older_than_days=0, domain="data_tables") == 1  # nosec B101 # real archive transition.
            assert jm.get_job(job["id"]) is None  # nosec B101 # active storage no longer supplies authority.
        recovered = publication.recover_table_generation(jm, db, pending)
        assert recovered["status"] == "ready"  # nosec B101 # exact completion-token replay works.
        columns, rows = _contents(db, table["id"])
        assert json.loads(rows[0]["row_json"]) == {columns[0]["column_id"]: "Replacement"}  # nosec B101 # canonical generated cell.


@pytest.mark.parametrize("response_kind", ["table", "job"])
def test_real_backend_timestamp_responses_keep_the_public_string_contract(publication_store, tmp_path, response_kind):
    """Native PostgreSQL values pass through the same response builders as real requests."""
    from datetime import datetime

    from tldw_Server_API.app.api.v1.endpoints import data_tables as endpoint
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    db, backend, _path = publication_store
    with scoped_context(user_id=401, is_admin=False):
        if response_kind == "table":
            row = db.create_data_table(name="Rowan native timestamps", prompt="Keep saved facts", status="ready")
            response = endpoint._table_summary_from_row(row)
            fields = ("created_at", "updated_at", "last_modified")
        else:
            if backend is None:
                jm = JobManager(tmp_path / "timestamp-jobs.db")
            else:
                cfg = backend.config
                dsn = f"postgresql://{quote(cfg.pg_user, safe='')}:{quote(cfg.pg_password, safe='')}@{cfg.pg_host}:{cfg.pg_port}/{quote(cfg.pg_database, safe='')}"
                jm = JobManager(backend="postgres", db_url=dsn)
            row = jm.create_job(
                domain="data_tables", queue="default", job_type="data_table_generate", owner_user_id="401", payload={}
            )
            response = asyncio.run(
                endpoint.get_data_table_job(
                    row["id"],
                    current_user=User(id=401, username="timestamp-owner", is_active=True, is_admin=False),
                    principal=AuthPrincipal(
                        kind="user", user_id=401, subject="timestamp-owner", roles=["user"], permissions=["media.read"]
                    ),
                    jm=jm,
                )
            )
            fields = ("created_at", "started_at", "completed_at", "cancelled_at")
        expected = {
            name: row.get(name).isoformat() if isinstance(row.get(name), datetime) else row.get(name) for name in fields
        }
        actual = {name: response.model_dump(mode="json")[name] for name in fields}
        assert (
            actual == expected
        )  # nosec B101 -- actual database timestamp values retain the public string/null contract.


@pytest.mark.parametrize("document_source", [False, True], ids=["snapshot", "document"])
def test_background_generation_opens_only_the_admitted_owner_scope(
    publication_store, tmp_path, monkeypatch, document_source
):
    """A background delivery cannot depend on the API request's content scope."""
    db, backend, _path = publication_store
    with scoped_context(user_id=401, is_admin=False):
        jm, job, table = _complete_generation(
            db, backend, tmp_path, monkeypatch, modern=True, background=True, document_source=document_source
        )
    with scoped_context(user_id=None, is_admin=False):
        state = jobs_worker.publish_completed_generation(jm, job["id"], expected_uuid=job["uuid"], db=db)
    with scoped_context(user_id=401, is_admin=False):
        columns, rows = _contents(db, table["id"])
        assert (
            state == "ready"
            and len(columns) == len(rows) == 1
            and json.loads(rows[0]["row_json"]) == {columns[0]["column_id"]: "Replacement"}
        )  # nosec B101 -- real background stage/completion/publication needs no foreground scope.


def test_document_generation_does_not_borrow_ambient_admin_access(publication_store, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope
    from tldw_Server_API.app.core.exceptions import DataTablesJobError

    db, backend, _path = publication_store
    with scoped_context(user_id=999, is_admin=True) as parent:
        with pytest.raises(DataTablesJobError, match="media_not_found"):
            _complete_generation(
                db, backend, tmp_path, monkeypatch, modern=True, document_source=True, source_owner=402
            )
        assert get_scope() is parent  # nosec B101 -- failed source reads restore the caller's exact content context.
