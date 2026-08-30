import base64
import gzip
import importlib
import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.Jobs import migrations as jobs_migrations
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
            "prepared_disposition_fingerprint",
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
        assert column_details["prepared_disposition_fingerprint"][3] == 0
        assert column_details["no_attempt_recovery_fingerprint"][3] == 0
        # Archive table exists
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs_archive'").fetchone()
        assert row and row[0] == "jobs_archive"
        archive_columns = {
            item[1]
            for item in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()
        }
        assert {
            "expired_lease_policy",
            "quarantine_threshold",
            "prepared_disposition_fingerprint",
            "no_attempt_recovery_fingerprint",
        } <= archive_columns
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
        "  prepared_disposition_fingerprint TEXT CHECK (\n"
        "    prepared_disposition_fingerprint IS NULL OR (\n"
        "      LENGTH(prepared_disposition_fingerprint) = 64 AND\n"
        "      prepared_disposition_fingerprint NOT GLOB '*[^0-9a-f]*'\n"
        "    )\n"
        "  ),\n",
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
        conn.execute(
            "INSERT INTO jobs_archive(uuid, domain, queue, job_type, payload, status) "
            "VALUES('legacy-archive', 'legacy', 'default', 'work', '{}', 'completed')"
        )

    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT expired_lease_policy, quarantine_threshold, "
            "prepared_disposition_fingerprint, no_attempt_recovery_fingerprint "
            "FROM jobs WHERE id=1"
        ).fetchone()
        assert row == ("consume_retry", None, None, None)
        archive_row = conn.execute(
            "SELECT expired_lease_policy, quarantine_threshold, "
            "prepared_disposition_fingerprint, no_attempt_recovery_fingerprint "
            "FROM jobs_archive WHERE uuid='legacy-archive'"
        ).fetchone()
        assert archive_row == ("consume_retry", None, None, None)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE jobs SET expired_lease_policy='invalid' WHERE id=1"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE jobs SET quarantine_threshold=0 WHERE id=1")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE jobs_archive SET prepared_disposition_fingerprint='invalid' "
                "WHERE uuid='legacy-archive'"
            )
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

    with pytest.raises(jobs_migrations.SlidesArchiveNormalizationError) as exc_info:
        normalize_slides_archive_projection(
            {
                "payload": None,
                "result": None,
                "payload_compressed": blob,
                "result_compressed": None,
            }
        )

    assert exc_info.value.args == ()
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def _archive_sidecar(value, *, backend: str):
    compressed = gzip.compress(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    )
    if backend == "sqlite":
        return "gzip64:" + base64.b64encode(compressed).decode("ascii")
    return compressed


@pytest.mark.parametrize("backend", ("sqlite", "postgres"))
def test_archive_projection_rejects_malformed_sidecar_with_primary_json(
    backend,
) -> None:
    malformed = "gzip64:!!!!" if backend == "sqlite" else b"sensitive-destination"

    with pytest.raises(jobs_migrations.SlidesArchiveNormalizationError) as exc_info:
        normalize_slides_archive_projection(
            {
                "payload": '{"schema_version":1}',
                "result": None,
                "payload_compressed": malformed,
                "result_compressed": None,
            }
        )

    assert exc_info.value.args == ()
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("backend", ("sqlite", "postgres"))
def test_archive_projection_rejects_divergent_primary_and_sidecar_json(
    backend,
) -> None:
    primary = {"schema_version": 1, "delivery_id": "delivery-primary"}
    sidecar = {"schema_version": 1, "delivery_id": "delivery-sidecar"}

    with pytest.raises(jobs_migrations.SlidesArchiveNormalizationError) as exc_info:
        normalize_slides_archive_projection(
            {
                "payload": json.dumps(primary),
                "result": None,
                "payload_compressed": _archive_sidecar(sidecar, backend=backend),
                "result_compressed": None,
            }
        )

    assert exc_info.value.args == ()
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("backend", ("sqlite", "postgres"))
def test_archive_projection_accepts_matching_primary_and_sidecar_json(
    backend,
) -> None:
    payload = {"schema_version": 1, "delivery_id": "delivery-matching"}

    normalized = normalize_slides_archive_projection(
        {
            "payload": json.dumps(payload),
            "result": None,
            "payload_compressed": _archive_sidecar(payload, backend=backend),
            "result_compressed": None,
        }
    )

    assert normalized["payload"] == payload


@pytest.mark.parametrize(
    ("primary", "sidecar"),
    (
        ({"nested": {"value": True}}, {"nested": {"value": 1}}),
        ({"nested": {"value": 1}}, {"nested": {"value": 1.0}}),
    ),
)
def test_archive_projection_rejects_nested_json_type_mismatches(
    primary,
    sidecar,
) -> None:
    with pytest.raises(jobs_migrations.SlidesArchiveNormalizationError):
        normalize_slides_archive_projection(
            {
                "payload": json.dumps(primary),
                "result": None,
                "payload_compressed": _archive_sidecar(
                    sidecar,
                    backend="sqlite",
                ),
                "result_compressed": None,
            }
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_archive_projection_accepts_matching_nested_nonfinite_float(value) -> None:
    logical = {"nested": [value]}

    normalized = normalize_slides_archive_projection(
        {
            "payload": json.dumps(logical),
            "result": None,
            "payload_compressed": _archive_sidecar(
                logical,
                backend="sqlite",
            ),
            "result_compressed": None,
        }
    )

    actual = normalized["payload"]["nested"][0]
    assert type(actual) is float
    assert math.isnan(actual) if math.isnan(value) else actual == value


@pytest.mark.parametrize(
    ("primary", "sidecar"),
    (
        (float("nan"), float("inf")),
        (float("inf"), float("-inf")),
        (float("-inf"), float("inf")),
    ),
)
def test_archive_projection_rejects_nonfinite_category_or_sign_mismatch(
    primary,
    sidecar,
) -> None:
    with pytest.raises(jobs_migrations.SlidesArchiveNormalizationError):
        normalize_slides_archive_projection(
            {
                "payload": json.dumps({"nested": [primary]}),
                "result": None,
                "payload_compressed": _archive_sidecar(
                    {"nested": [sidecar]},
                    backend="sqlite",
                ),
                "result_compressed": None,
            }
        )


@pytest.mark.parametrize("field", ("payload", "result"))
def test_archive_projection_treats_present_json_null_as_authoritative(field) -> None:
    row = {
        "payload": None,
        "result": None,
        "payload_compressed": None,
        "result_compressed": None,
        f"__slides_archive_{field}_present": True,
    }
    row[f"{field}_compressed"] = _archive_sidecar(
        {"divergent": True},
        backend="sqlite",
    )

    with pytest.raises(jobs_migrations.SlidesArchiveNormalizationError):
        normalize_slides_archive_projection(row)


@pytest.mark.parametrize("field", ("payload", "result"))
def test_archive_projection_json_null_controls_and_internal_metadata(field) -> None:
    present = f"__slides_archive_{field}_present"
    row = {
        "payload": None,
        "result": None,
        "payload_compressed": None,
        "result_compressed": None,
        present: True,
    }
    row[f"{field}_compressed"] = _archive_sidecar(None, backend="sqlite")

    matching_null = normalize_slides_archive_projection(row)
    assert matching_null[field] is None
    assert present not in matching_null

    row[present] = False
    row[f"{field}_compressed"] = _archive_sidecar(
        {"sidecar_only": True},
        backend="sqlite",
    )
    sidecar_only = normalize_slides_archive_projection(row)
    assert sidecar_only[field] == {"sidecar_only": True}
    assert present not in sidecar_only

    row[f"{field}_compressed"] = None
    no_sidecar = normalize_slides_archive_projection(row)
    assert no_sidecar[field] is None
    assert present not in no_sidecar


def test_archive_normalization_failure_contract_survives_migrations_reload() -> None:
    normalization_error = jobs_migrations.SlidesArchiveNormalizationError

    reloaded = importlib.reload(jobs_migrations)

    assert reloaded.SlidesArchiveNormalizationError is normalization_error
