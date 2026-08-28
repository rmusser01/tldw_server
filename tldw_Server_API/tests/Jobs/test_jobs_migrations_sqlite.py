import base64
import gzip
import importlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.Jobs.migrations import (
    JOBS_SQLITE_DDL,
    ensure_jobs_tables,
    normalize_slides_archive_projection,
)


def test_sqlite_schema_persists_owner_scoped_idempotency_receipts(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "jobs_receipts.db")
    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(job_idempotency_receipts)"
            ).fetchall()
        }
        assert columns == {
            "receipt_id",
            "domain",
            "queue",
            "job_type",
            "owner_user_id",
            "key_digest",
            "request_fingerprint",
            "operation_scope",
            "job_uuid",
            "job_id",
            "created_at",
            "expires_at",
        }
        assert not {"idempotency_key", "raw_key", "client_key"} & columns

        indexes = {
            row[1]: tuple(
                column[2]
                for column in conn.execute(
                    f"PRAGMA index_info('{row[1]}')"
                ).fetchall()
            )
            for row in conn.execute(
                "PRAGMA index_list('job_idempotency_receipts')"
            ).fetchall()
        }
        assert indexes["idx_job_idempotency_receipts_owner_key"] == (
            "domain",
            "queue",
            "job_type",
            "owner_user_id",
            "key_digest",
        )
        assert indexes["idx_job_idempotency_receipts_job_uuid"] == ("job_uuid",)
        assert indexes["idx_job_idempotency_receipts_job_id"] == ("job_id",)
        assert indexes["idx_job_idempotency_receipts_scope"] == (
            "operation_scope",
            "owner_user_id",
            "expires_at",
        )

        values = (
            "sharing",
            "workspace-clone",
            "workspace_clone",
            "recipient-1",
            "a" * 64,
            "b" * 64,
            "share:share-1",
            "job-1",
            1,
            "2026-08-25T00:00:00+00:00",
            "2026-09-24T00:00:00+00:00",
        )
        conn.execute(
            "INSERT INTO job_idempotency_receipts "
            "(domain, queue, job_type, owner_user_id, key_digest, "
            "request_fingerprint, operation_scope, job_uuid, job_id, "
            "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        conn.execute(
            "INSERT INTO job_idempotency_receipts "
            "(domain, queue, job_type, owner_user_id, key_digest, "
            "request_fingerprint, operation_scope, job_uuid, job_id, "
            "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*values[:3], "recipient-2", *values[4:7], "job-2", 2, *values[9:]),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO job_idempotency_receipts "
                "(domain, queue, job_type, owner_user_id, key_digest, "
                "request_fingerprint, operation_scope, job_uuid, job_id, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*values[:7], "job-3", 3, *values[9:]),
            )
    finally:
        conn.close()


def test_sqlite_schema_has_expected_columns_and_indexes(tmp_path):


    db_path = ensure_jobs_tables(tmp_path / "jobs_mig.db")
    conn = sqlite3.connect(db_path)
    try:
        # Columns present
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for expected in [
            "completion_token",
            "failure_timeline",
            "request_id",
            "trace_id",
            "progress_percent",
            "progress_message",
            "error_code",
            "error_class",
            "error_stack",
            "expired_lease_policy",
            "quarantine_threshold",
            "no_attempt_recovery_fingerprint",
        ]:
            assert expected in cols
        column_details = {
            row[1]: row
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        assert column_details["expired_lease_policy"][3] == 1
        assert column_details["expired_lease_policy"][4] == "'consume_retry'"
        assert column_details["quarantine_threshold"][3] == 0
        assert column_details["no_attempt_recovery_fingerprint"][3] == 0
        # Archive table exists
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs_archive'").fetchone()
        assert row and row[0] == "jobs_archive"
        # Partial unique index for idempotency exists
        idx = [r[1] for r in conn.execute("PRAGMA index_list('jobs')").fetchall()]
        assert any("idx_jobs_idempotent" in x for x in idx)
        # job_counters exists
        row2 = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_counters'").fetchone()
        assert row2 and row2[0] == "job_counters"
        # Idempotent ensure re-runs without error
        ensure_jobs_tables(Path(db_path))
    finally:
        conn.close()


def test_jobs_migrations_uses_shared_sqlite_policy_helper(tmp_path):
    jobs_migrations = importlib.import_module("tldw_Server_API.app.core.Jobs.migrations")
    calls: list[dict[str, object]] = []

    def fake_configure(conn, **kwargs):
        calls.append(kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sqlite_policy, "configure_sqlite_connection", fake_configure)
        jobs_migrations = importlib.reload(jobs_migrations)
        jobs_migrations.ensure_jobs_tables(tmp_path / "jobs_helper.db")

    importlib.reload(jobs_migrations)

    assert calls == [{
        "use_wal": True,
        "synchronous": "NORMAL",
        "busy_timeout_ms": 5000,
        "foreign_keys": False,
        "temp_store": None,
    }]


def test_ensure_jobs_tables_sanitizes_schema_failure_log(tmp_path, monkeypatch):
    jobs_migrations = importlib.import_module("tldw_Server_API.app.core.Jobs.migrations")
    secret = "sk_jobsMigrationSecret1234567890"
    db_path = tmp_path / secret / "jobs.db"
    fake_logger = MagicMock()

    def fail_connect(path):
        raise sqlite3.OperationalError(f"unable to open database file {path} token={secret}")

    monkeypatch.setattr(jobs_migrations.sqlite3, "connect", fail_connect)
    monkeypatch.setattr(jobs_migrations, "logger", fake_logger)

    with pytest.raises(sqlite3.OperationalError):
        jobs_migrations.ensure_jobs_tables(db_path)

    fake_logger.warning.assert_called_once()
    warning_args = fake_logger.warning.call_args.args
    rendered = " ".join(str(arg) for arg in warning_args)
    assert "Failed to ensure Jobs schema" in rendered
    assert str(db_path) not in rendered
    assert secret not in rendered


def test_sqlite_forward_migration_backfills_execution_controls(tmp_path):
    db_path = tmp_path / "legacy_jobs.db"
    legacy_ddl = JOBS_SQLITE_DDL.replace(
        "  expired_lease_policy TEXT NOT NULL DEFAULT 'consume_retry' "
        "CHECK (expired_lease_policy IN ('consume_retry','requeue_no_attempt')),\n",
        "",
    ).replace(
        "  quarantine_threshold INTEGER CHECK "
        "(quarantine_threshold IS NULL OR quarantine_threshold > 0),\n",
        "",
    ).replace(
        "  no_attempt_recovery_fingerprint TEXT CHECK (\n"
        "    no_attempt_recovery_fingerprint IS NULL OR (\n"
        "      LENGTH(no_attempt_recovery_fingerprint) = 64 AND\n"
        "      no_attempt_recovery_fingerprint NOT GLOB '*[^0-9a-f]*'\n"
        "    )\n"
        "  ),\n",
        "",
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(legacy_ddl)
        conn.execute(
            "INSERT INTO jobs(uuid, domain, queue, job_type, payload, status) "
            "VALUES('legacy', 'legacy', 'default', 'work', '{}', 'queued')"
        )

    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT expired_lease_policy, quarantine_threshold, "
            "no_attempt_recovery_fingerprint FROM jobs WHERE id=1"
        ).fetchone()
        assert row == ("consume_retry", None, None)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE jobs SET expired_lease_policy='invalid' WHERE id=1"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE jobs SET quarantine_threshold=0 WHERE id=1")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE jobs SET no_attempt_recovery_fingerprint='invalid' WHERE id=1"
            )

    ensure_jobs_tables(db_path)


@pytest.mark.parametrize(
    "variant",
    ("base64_whitespace", "concatenated", "trailing", "truncated", "malformed"),
)
def test_archive_projection_rejects_noncanonical_gzip_framing(variant) -> None:
    payload = {"delivery_id": "00000000-0000-4000-8000-000000000001"}
    encoded_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    member = gzip.compress(encoded_json)
    if variant == "base64_whitespace":
        encoded = base64.b64encode(member).decode("ascii")
        blob = "gzip64:" + encoded[:8] + "\n" + encoded[8:]
    elif variant == "concatenated":
        blob = "gzip64:" + base64.b64encode(
            member + gzip.compress(b" ")
        ).decode("ascii")
    elif variant == "trailing":
        blob = "gzip64:" + base64.b64encode(member + b"\0").decode("ascii")
    elif variant == "truncated":
        blob = "gzip64:" + base64.b64encode(member[:-4]).decode("ascii")
    else:
        blob = "gzip64:!!!!"

    normalized = normalize_slides_archive_projection(
        {
            "payload": None,
            "result": None,
            "payload_compressed": blob,
            "result_compressed": None,
        }
    )

    assert normalized["payload"] != payload
