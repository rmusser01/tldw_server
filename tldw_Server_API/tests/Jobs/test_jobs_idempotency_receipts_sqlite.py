"""SQLite coverage for durable Jobs idempotency receipts."""

from __future__ import annotations

import base64
import gzip
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import SLIDES_ARCHIVE_EXACT_FIELDS
from tldw_Server_API.app.core.Jobs.operations import contracts

pytestmark = pytest.mark.unit


def _receipt_contracts():
    required = (
        "IdempotentOperationAdmission",
        "IdempotentOperationCommand",
        "IdempotentOperationConflict",
        "IdempotentOperationConflictReason",
        "IdempotentOperationDisposition",
    )
    missing = [name for name in required if not hasattr(contracts, name)]
    assert not missing, f"missing durable idempotency contracts: {missing}"
    return tuple(getattr(contracts, name) for name in required)


def _create_job_command(*, owner_user_id: str | None = "recipient-1"):
    return contracts.CreateJobCommand(
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        payload={"schema_version": 1},
        owner_user_id=owner_user_id,
        batch_group="share:share-1",
        priority=5,
        max_retries=0,
    )


def _operation_command(**overrides):
    (
        _,
        command_type,
        _,
        _,
        _,
    ) = _receipt_contracts()
    values = {
        "job": _create_job_command(),
        "key_digest": "a" * 64,
        "request_fingerprint": "b" * 64,
        "operation_scope": "share:share-1",
        "receipt_expires_at": datetime.now(timezone.utc) + timedelta(days=31),
    }
    values.update(overrides)
    return command_type(**values)


@pytest.mark.parametrize("owner_user_id", (None, "", "   "))
def test_idempotent_operation_command_requires_owner(owner_user_id):
    with pytest.raises(ValueError, match="owner_user_id"):
        _operation_command(job=_create_job_command(owner_user_id=owner_user_id))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("key_digest", "a" * 63),
        ("key_digest", "A" * 64),
        ("key_digest", "g" * 64),
        ("request_fingerprint", "b" * 65),
        ("request_fingerprint", "B" * 64),
    ),
)
def test_idempotent_operation_command_requires_sha256_hex_digests(
    field_name,
    invalid_value,
):
    with pytest.raises(ValueError, match=field_name):
        _operation_command(**{field_name: invalid_value})


@pytest.mark.parametrize("operation_scope", ("", "   ", "x" * 201, "share:\N{SNOWMAN}"))
def test_idempotent_operation_command_requires_bounded_ascii_scope(operation_scope):
    with pytest.raises(ValueError, match="operation_scope"):
        _operation_command(operation_scope=operation_scope)


def test_idempotent_operation_command_requires_timezone_aware_expiry():
    with pytest.raises(ValueError, match="receipt_expires_at"):
        _operation_command(receipt_expires_at=datetime.now())


def test_idempotent_operation_admission_defensively_copies_job_row():
    admission_type, _, _, _, disposition_type = _receipt_contracts()
    row = {"uuid": "job-1", "status": "queued", "payload": {"schema_version": 1}}

    admission = admission_type.created(row)
    row["status"] = "failed"
    row["payload"]["schema_version"] = 2

    assert admission.disposition is disposition_type.CREATED
    assert admission.job == {
        "uuid": "job-1",
        "status": "queued",
        "payload": {"schema_version": 1},
    }


def test_idempotent_operation_conflict_exposes_bounded_reason_and_job_uuid():
    _, _, conflict_type, reason_type, _ = _receipt_contracts()

    conflict = conflict_type(reason_type.KEY_REUSED, job_uuid="job-1")

    assert str(conflict) == "idempotency_key_reused"
    assert conflict.reason is reason_type.KEY_REUSED
    assert conflict.job_uuid == "job-1"


def test_idempotent_operation_command_requires_job_scope_alignment():
    with pytest.raises(ValueError, match="batch_group"):
        _operation_command(
            job=contracts.CreateJobCommand(
                domain="sharing",
                queue="workspace-clone",
                job_type="workspace_clone",
                payload={"schema_version": 1},
                owner_user_id="recipient-1",
                batch_group="share:other",
            )
        )


def test_idempotent_operation_command_rejects_legacy_job_idempotency_key():
    with pytest.raises(ValueError, match="idempotency_key"):
        _operation_command(
            job=contracts.CreateJobCommand(
                domain="sharing",
                queue="workspace-clone",
                job_type="workspace_clone",
                payload={"schema_version": 1},
                owner_user_id="recipient-1",
                batch_group="share:share-1",
                idempotency_key="legacy-global-key",
            )
        )


@pytest.fixture
def receipt_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    return JobManager(tmp_path / "jobs.db")


def _receipt_rows(manager: JobManager):
    conn = sqlite3.connect(manager.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM job_idempotency_receipts ORDER BY receipt_id"
            )
        ]
    finally:
        conn.close()


def _table_count(manager: JobManager, table: str) -> int:
    queries = {
        "job_events": "SELECT COUNT(*) FROM job_events",
        "job_idempotency_receipts": (
            "SELECT COUNT(*) FROM job_idempotency_receipts"
        ),
        "jobs": "SELECT COUNT(*) FROM jobs",
    }
    conn = sqlite3.connect(manager.db_path)
    try:
        row = conn.execute(queries[table]).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _archive_job(manager: JobManager, job_uuid: str, *, retain_active: bool = False):
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    conn = sqlite3.connect(manager.db_path)
    try:
        with conn:
            conn.execute(
                f"INSERT INTO jobs_archive ({projection}) "  # nosec B608
                f"SELECT {projection} FROM jobs WHERE uuid = ?",  # nosec B608
                (job_uuid,),
            )
            if not retain_active:
                conn.execute("DELETE FROM jobs WHERE uuid = ?", (job_uuid,))
    finally:
        conn.close()


def test_first_request_atomically_creates_job_and_receipt(receipt_manager):
    result = receipt_manager.admit_idempotent_operation(_operation_command())

    jobs = receipt_manager.list_jobs(
        domain="sharing",
        owner_user_id="recipient-1",
    )
    receipts = _receipt_rows(receipt_manager)

    assert result.disposition is contracts.IdempotentOperationDisposition.CREATED
    assert result.job["uuid"] == jobs[0]["uuid"] == receipts[0]["job_uuid"]
    assert receipts[0]["key_digest"] == "a" * 64
    assert jobs[0]["idempotency_key"] is None
    assert _table_count(receipt_manager, "job_events") == 1


def test_same_key_and_fingerprint_replays_same_job(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    replay = receipt_manager.admit_idempotent_operation(_operation_command())

    assert replay.disposition is contracts.IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == first.job["uuid"]
    assert len(_receipt_rows(receipt_manager)) == 1
    assert _table_count(receipt_manager, "job_events") == 1


def test_exact_replay_survives_queue_policy_change(receipt_manager, monkeypatch):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES_SHARING")

    replay = receipt_manager.admit_idempotent_operation(_operation_command())

    assert replay.disposition is contracts.IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == first.job["uuid"]


def test_exact_replay_does_not_require_a_new_retention_window(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())

    replay = receipt_manager.admit_idempotent_operation(
        _operation_command(
            receipt_expires_at=datetime.now(timezone.utc) + timedelta(days=1)
        )
    )

    assert replay.disposition is contracts.IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == first.job["uuid"]


def test_uuid_lookup_normalizes_active_and_archived_job(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())

    active = receipt_manager.get_job_or_archived_by_uuid(
        first.job["uuid"],
        domain="sharing",
        owner_user_id="recipient-1",
    )
    _archive_job(receipt_manager, first.job["uuid"])
    archived = receipt_manager.get_job_or_archived_by_uuid(
        first.job["uuid"],
        domain="sharing",
        owner_user_id="recipient-1",
    )

    assert active is not None
    assert archived is not None
    assert active["archived"] is False
    assert archived["archived"] is True
    assert {key: value for key, value in active.items() if key != "archived"} == {
        key: value for key, value in archived.items() if key != "archived"
    }
    assert active["payload"] == archived["payload"] == {"schema_version": 1}
    assert active["result"] == archived["result"]
    assert active["uuid"] == archived["uuid"] == first.job["uuid"]


def test_exact_replay_survives_job_archival(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    _archive_job(receipt_manager, first.job["uuid"])

    replay = receipt_manager.admit_idempotent_operation(_operation_command())

    assert replay.disposition is contracts.IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == first.job["uuid"]
    assert replay.job["archived"] is True


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
@pytest.mark.parametrize(
    ("sidecar_kind", "primary_json_present"),
    (("malformed", False), ("malformed", True), ("divergent", True)),
)
def test_archived_receipt_lookup_and_replay_reject_invalid_sidecar_without_mutation(
    receipt_manager,
    compressed_field,
    sidecar_kind,
    primary_json_present,
):
    command = _operation_command()
    first = receipt_manager.admit_idempotent_operation(command)
    _archive_job(receipt_manager, first.job["uuid"])
    sidecar = "gzip64:c2Vuc2l0aXZlLWRlc3RpbmF0aW9u"
    if sidecar_kind == "divergent":
        divergent = (
            {"schema_version": 2}
            if compressed_field == "payload"
            else {"status": "divergent"}
        )
        sidecar = "gzip64:" + base64.b64encode(
            gzip.compress(json.dumps(divergent).encode("utf-8"))
        ).decode("ascii")
    with sqlite3.connect(receipt_manager.db_path) as conn:
        conn.execute(
            "UPDATE jobs_archive SET result=? WHERE uuid=?",
            ('{"status":"completed"}', first.job["uuid"]),
        )
        if compressed_field == "payload":
            conn.execute(
                "UPDATE jobs_archive SET payload=CASE WHEN ? THEN payload ELSE NULL END, "
                "payload_compressed=? "
                "WHERE uuid=?",
                (primary_json_present, sidecar, first.job["uuid"]),
            )
        else:
            conn.execute(
                "UPDATE jobs_archive SET result=CASE WHEN ? THEN result ELSE NULL END, "
                "result_compressed=? "
                "WHERE uuid=?",
                (primary_json_present, sidecar, first.job["uuid"]),
            )
        before_archive = tuple(
            conn.execute(
                "SELECT payload, payload_compressed, result, result_compressed, "
                "status FROM jobs_archive WHERE uuid=?",
                (first.job["uuid"],),
            ).fetchone()
        )
    before = (
        before_archive,
        _receipt_rows(receipt_manager),
        _table_count(receipt_manager, "jobs"),
        _table_count(receipt_manager, "job_events"),
    )

    calls = (
        lambda: receipt_manager.get_job_or_archived_by_uuid(first.job["uuid"]),
        lambda: receipt_manager.replay_idempotent_operation(command),
        lambda: receipt_manager.admit_idempotent_operation(command),
    )
    for call in calls:
        with pytest.raises(contracts.IdempotentOperationUnavailableError) as exc_info:
            call()
        assert str(exc_info.value) == "job archive projection is unavailable"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert "sensitive-destination" not in str(exc_info.value)

    with sqlite3.connect(receipt_manager.db_path) as conn:
        after_archive = tuple(
            conn.execute(
                "SELECT payload, payload_compressed, result, result_compressed, "
                "status FROM jobs_archive WHERE uuid=?",
                (first.job["uuid"],),
            ).fetchone()
        )
    after = (
        after_archive,
        _receipt_rows(receipt_manager),
        _table_count(receipt_manager, "jobs"),
        _table_count(receipt_manager, "job_events"),
    )
    assert after == before


def test_receipt_backed_prune_collision_remap_has_no_exception_chain_or_mutation(
    receipt_manager,
):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    _archive_job(receipt_manager, first.job["uuid"], retain_active=True)
    with sqlite3.connect(receipt_manager.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status='completed', "
            "completed_at='2000-01-01 00:00:00' WHERE uuid=?",
            (first.job["uuid"],),
        )
        conn.execute(
            "UPDATE jobs_archive SET payload_compressed=? WHERE uuid=?",
            ("gzip64:c2Vuc2l0aXZlLWRlc3RpbmF0aW9u", first.job["uuid"]),
        )
        active_before = tuple(
            conn.execute(
                "SELECT status, completed_at, payload, result FROM jobs WHERE uuid=?",
                (first.job["uuid"],),
            ).fetchone()
        )
        archive_before = tuple(
            conn.execute(
                "SELECT payload, payload_compressed, result, result_compressed, status "
                "FROM jobs_archive WHERE uuid=?",
                (first.job["uuid"],),
            ).fetchone()
        )
    before = (
        active_before,
        archive_before,
        _receipt_rows(receipt_manager),
        _table_count(receipt_manager, "jobs"),
        _table_count(receipt_manager, "job_events"),
    )

    with pytest.raises(contracts.IdempotentOperationUnavailableError) as exc_info:
        receipt_manager.prune_jobs(statuses=["completed"], older_than_days=30)

    assert exc_info.value.args == ("job archive projection is unavailable",)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    with sqlite3.connect(receipt_manager.db_path) as conn:
        active_after = tuple(
            conn.execute(
                "SELECT status, completed_at, payload, result FROM jobs WHERE uuid=?",
                (first.job["uuid"],),
            ).fetchone()
        )
        archive_after = tuple(
            conn.execute(
                "SELECT payload, payload_compressed, result, result_compressed, status "
                "FROM jobs_archive WHERE uuid=?",
                (first.job["uuid"],),
            ).fetchone()
        )
    after = (
        active_after,
        archive_after,
        _receipt_rows(receipt_manager),
        _table_count(receipt_manager, "jobs"),
        _table_count(receipt_manager, "job_events"),
    )
    assert after == before


def test_uuid_lookup_rejects_duplicate_active_and_archived_authority(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    _archive_job(receipt_manager, first.job["uuid"], retain_active=True)

    with pytest.raises(
        contracts.IdempotentOperationUnavailableError,
        match="exactly one Job",
    ):
        receipt_manager.get_job_or_archived_by_uuid(first.job["uuid"])


@pytest.mark.parametrize(
    ("column", "value"),
    (("job_uuid", "not-a-job-uuid"), ("job_id", 999_999)),
)
def test_corrupt_receipt_correlation_fails_closed(
    receipt_manager,
    column,
    value,
):
    receipt_manager.admit_idempotent_operation(_operation_command())
    conn = sqlite3.connect(receipt_manager.db_path)
    try:
        conn.execute(
            f"UPDATE job_idempotency_receipts SET {column} = ?",  # nosec B608
            (value,),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(contracts.IdempotentOperationUnavailableError):
        receipt_manager.admit_idempotent_operation(_operation_command())


def test_same_key_is_isolated_between_owners(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    second = receipt_manager.admit_idempotent_operation(
        _operation_command(job=_create_job_command(owner_user_id="recipient-2"))
    )

    assert first.job["uuid"] != second.job["uuid"]
    assert len(receipt_manager.list_jobs(domain="sharing")) == 2
    assert len(_receipt_rows(receipt_manager)) == 2


def test_same_key_with_different_fingerprint_conflicts(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())

    with pytest.raises(contracts.IdempotentOperationConflict) as exc_info:
        receipt_manager.admit_idempotent_operation(
            _operation_command(request_fingerprint="c" * 64)
        )

    assert exc_info.value.reason is contracts.IdempotentOperationConflictReason.KEY_REUSED
    assert exc_info.value.job_uuid == first.job["uuid"]


def test_second_key_with_same_scope_and_fingerprint_converges(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    converged = receipt_manager.admit_idempotent_operation(
        _operation_command(key_digest="d" * 64)
    )

    assert converged.disposition is contracts.IdempotentOperationDisposition.CONVERGED
    assert converged.job["uuid"] == first.job["uuid"]
    assert {row["key_digest"] for row in _receipt_rows(receipt_manager)} == {
        "a" * 64,
        "d" * 64,
    }


def test_second_key_with_active_scope_and_different_fingerprint_conflicts(
    receipt_manager,
):
    first = receipt_manager.admit_idempotent_operation(_operation_command())

    with pytest.raises(contracts.IdempotentOperationConflict) as exc_info:
        receipt_manager.admit_idempotent_operation(
            _operation_command(
                key_digest="d" * 64,
                request_fingerprint="c" * 64,
            )
        )

    assert exc_info.value.reason is contracts.IdempotentOperationConflictReason.SCOPE_ACTIVE
    assert exc_info.value.job_uuid == first.job["uuid"]


def test_active_scope_without_authoritative_receipt_fails_closed(receipt_manager):
    existing = receipt_manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="workspace_clone",
        payload={"schema_version": 1},
        owner_user_id="recipient-1",
        batch_group="share:share-1",
        priority=5,
        max_retries=0,
    )

    with pytest.raises(
        contracts.IdempotentOperationUnavailableError,
        match="one fingerprint",
    ):
        receipt_manager.admit_idempotent_operation(_operation_command())

    jobs = receipt_manager.list_jobs(domain="sharing")
    assert [job["uuid"] for job in jobs] == [existing["uuid"]]
    assert _receipt_rows(receipt_manager) == []


def test_active_scope_with_malformed_receipt_fails_closed(receipt_manager):
    first = receipt_manager.admit_idempotent_operation(_operation_command())
    conn = sqlite3.connect(receipt_manager.db_path)
    try:
        conn.execute(
            "UPDATE job_idempotency_receipts SET job_id = ?",
            (int(first.job["id"]) + 1000,),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(
        contracts.IdempotentOperationUnavailableError,
        match="correlation",
    ):
        receipt_manager.admit_idempotent_operation(
            _operation_command(key_digest="d" * 64)
        )

    assert len(receipt_manager.list_jobs(domain="sharing")) == 1


def test_receipt_insert_failure_rolls_back_new_job(receipt_manager, monkeypatch):
    from tldw_Server_API.app.core.Jobs.operations.sqlite import idempotency

    def _fail_receipt_insert(*_args, **_kwargs):
        raise sqlite3.IntegrityError("forced receipt failure")

    monkeypatch.setattr(idempotency, "_insert_receipt", _fail_receipt_insert)

    with pytest.raises(sqlite3.IntegrityError, match="forced receipt failure"):
        receipt_manager.admit_idempotent_operation(_operation_command())

    assert receipt_manager.list_jobs(domain="sharing") == []
    assert _receipt_rows(receipt_manager) == []


def test_receipt_expiry_must_preserve_thirty_day_replay_window(receipt_manager):
    with pytest.raises(ValueError, match="at least 30 days"):
        receipt_manager.admit_idempotent_operation(
            _operation_command(
                receipt_expires_at=datetime.now(timezone.utc) + timedelta(days=29)
            )
        )

    assert _table_count(receipt_manager, "jobs") == 0
    assert _table_count(receipt_manager, "job_idempotency_receipts") == 0


def test_missing_receipt_job_fails_closed_without_replacement(receipt_manager):
    receipt_manager.admit_idempotent_operation(_operation_command())
    conn = sqlite3.connect(receipt_manager.db_path)
    try:
        conn.execute("DELETE FROM jobs")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(
        contracts.IdempotentOperationUnavailableError,
        match="exactly one Job",
    ):
        receipt_manager.admit_idempotent_operation(_operation_command())

    assert _table_count(receipt_manager, "jobs") == 0
    assert _table_count(receipt_manager, "job_idempotency_receipts") == 1


def test_concurrent_keys_converge_on_one_active_job(receipt_manager):
    barrier = Barrier(8)

    def _admit(index: int):
        manager = JobManager(receipt_manager.db_path)
        barrier.wait(timeout=10)
        return manager.admit_idempotent_operation(
            _operation_command(key_digest=("a" if index % 2 == 0 else "d") * 64)
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_admit, range(8)))

    assert len({result.job["uuid"] for result in results}) == 1
    assert len(receipt_manager.list_jobs(domain="sharing")) == 1
    assert len(_receipt_rows(receipt_manager)) == 2
