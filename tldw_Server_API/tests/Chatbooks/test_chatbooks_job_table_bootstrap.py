"""Chatbooks metadata bootstrap must preserve current and legacy real job rows."""

from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportJob, ExportStatus, ImportStatus
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("storage", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
@pytest.mark.parametrize("column_state", ["present", "legacy"])
def test_job_table_bootstrap_retains_rows(storage, column_state, request, tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("JOBS_DB_URL", "sqlite:///" + str(tmp_path / "core-jobs.db"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "core-jobs.db"))
    backend = (
        request.getfixturevalue("pg_restricted_backend")
        if storage == "postgres"
        else SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "chacha.db")))
    )
    with ExitStack() as cleanup, scoped_context(user_id=101, is_admin=False):
        if storage == "sqlite":
            cleanup.callback(backend.get_pool().close_all)
        db = CharactersRAGDB(db_path=tmp_path / "chacha.db", client_id="101", owner_user_id="101", backend=backend)
        cleanup.callback(db.close_all_connections)
        service = ChatbookService(user_id="101", user_id_int=101, db=db)
        import_id = service.create_import_job("rowan.chatbook")["job_id"]
        export = ExportJob(
            job_id=str(uuid4()),
            user_id="101",
            status=ExportStatus.PENDING,
            chatbook_name="Rowan preserved export",
            metadata={"source": "s02"},
        )
        service._save_export_job(export)
        imported = replace(
            service.get_import_job(import_id),
            status=ImportStatus.COMPLETED,
            started_at=datetime(2026, 10, 26, 12, tzinfo=timezone.utc),
            completed_at=datetime(2026, 10, 26, 12, 1, tzinfo=timezone.utc),
            total_items=3,
            successful_items=2,
            skipped_items=1,
            conflicts=[{"item": "Rowan"}],
            warnings=["One skipped item"],
            metadata={"source": "s02"},
        )
        exported = replace(
            export,
            status=ExportStatus.COMPLETED,
            output_path="rowan.chatbook",
            started_at=datetime(2026, 10, 26, 12, tzinfo=timezone.utc),
            completed_at=datetime(2026, 10, 26, 12, 1, tzinfo=timezone.utc),
            total_items=3,
            file_size_bytes=123,
            download_url="/rowan",
            expires_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
        )
        service._save_import_job(imported)
        service._save_export_job(exported)
        assert (
            imported.progress_percentage,
            imported.processed_items,
            exported.progress_percentage,
            exported.processed_items,
        ) == (100, 3, 100, 3)  # nosec B101 -- Completion normalization has an independent expected result.
        assert service.get_import_job(import_id).to_dict() == imported.to_dict()  # nosec B101 -- Real upsert preserves every import field and completion normalization.
        assert service.get_export_job(export.job_id).to_dict() == exported.to_dict()  # nosec B101 -- Real upsert preserves every export field and completion normalization.
        # Clearing nullable fields must replace the prior values, not coalesce them.
        imported = replace(imported, started_at=None, completed_at=None, metadata={})
        exported = replace(exported, output_path=None, download_url=None, expires_at=None, metadata={})
        service._save_import_job(imported)
        service._save_export_job(exported)
        before_import = imported.to_dict()
        before_export = exported.to_dict()
        with pytest.raises(RuntimeError, match="rollback both jobs"):
            with db.transaction():
                service._save_import_job(replace(imported, error_message="rolled back import"), commit=False)
                service._save_export_job(replace(exported, error_message="rolled back export"), commit=False)
                raise RuntimeError("rollback both jobs")
        assert service.get_import_job(import_id).to_dict() == before_import  # nosec B101 -- Caller rollback retains the prior import update and cleared fields.
        assert service.get_export_job(export.job_id).to_dict() == before_export  # nosec B101 -- Caller rollback retains the prior export update and cleared fields.
        if column_state == "legacy":
            # Publish the legacy shape before testing a fresh service initialization.
            with db.transaction():
                db.execute_query("ALTER TABLE import_jobs DROP COLUMN metadata")
                db.execute_query("ALTER TABLE export_jobs DROP COLUMN metadata")
                before_import["metadata"] = {}
                before_export["metadata"] = {}
                if storage == "postgres":
                    # A same-named table in another schema must not suppress this user's upgrade.
                    db.execute_query("CREATE SCHEMA s02_other_jobs")
                    db.execute_query("CREATE TABLE s02_other_jobs.import_jobs (job_id TEXT PRIMARY KEY, metadata TEXT)")
                    db.execute_query("CREATE TABLE s02_other_jobs.export_jobs (job_id TEXT PRIMARY KEY, metadata TEXT)")
        with db.transaction():
            reopened = ChatbookService(user_id="101", user_id_int=101, db=db)
            repeated = ChatbookService(user_id="101", user_id_int=101, db=db)
        assert reopened.get_import_job(import_id).to_dict() == before_import  # nosec B101 -- Real legacy/current row preservation.
        assert repeated.get_export_job(export.job_id).to_dict() == before_export  # nosec B101 -- Repeat initialization is idempotent.
        for table in ("import_jobs", "export_jobs"):
            columns = backend.get_table_info(table)
            assert sum(row["name"] == "metadata" for row in columns) == 1  # nosec B101 -- Both real tables acquire exactly one metadata column.
