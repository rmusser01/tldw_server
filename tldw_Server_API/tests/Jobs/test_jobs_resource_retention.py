"""Resource payload and request-retention boundaries."""

import pytest

from tldw_Server_API.tests.Jobs import test_jobs_resource_lifecycle as lifecycle
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import make_command, make_request, snapshot, sql

manager_fixture = lifecycle.manager
resource_scenario_fixture = lifecycle.resource_scenario


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("status", ["queued", "processing"])
@pytest.mark.parametrize("key_state", ["keyless", "expired", "retained"])
def test_pruning_retains_active_resource_identity(resource_scenario, monkeypatch, archived, status, key_state):
    s = resource_scenario
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", str(archived).lower())
    job = s.admit(key=None if key_state == "keyless" else "active-key")
    if status == "processing":
        assert s.acquire()["uuid"] == job["uuid"]
    if key_state == "expired":
        sql(s.manager, "UPDATE job_resource_request_receipts SET expires_at='2000-01-01'")
    # Exercise the public admin cutoff, not a zero-day approximation.
    sql(s.manager, "UPDATE jobs SET created_at='2000-01-01',updated_at='2000-01-01' WHERE uuid=?", (job["uuid"],))
    before = snapshot(s.manager)
    assert s.manager.prune_jobs(older_than_days=1, statuses=[status], domain="claims") == 0
    after = snapshot(s.manager)
    for table in ("jobs", "jobs_archive", "job_resource_heads", "job_counters"):
        assert after[table] == before[table], table
    assert [event for event in after["job_events"] if event["job_id"] == job["id"]] == before["job_events"]
    audit = [event for event in after["job_events"] if event["job_id"] is None]
    assert all(event["event_type"] == "jobs.pruned" and event["attrs_json"]["deleted"] == 0 for event in audit)
    reused = s.manager.admit_resource_job(make_command(), request=make_request(key=None))
    assert reused.kind == "active_reused"
    assert reused.job["uuid"] == job["uuid"]
    changed = make_command(payload={**make_command().payload, "min_size": 3})
    conflict = s.manager.admit_resource_job(changed, request=make_request(changed, key=None))
    assert conflict.kind == "active_conflict"
    assert conflict.job["uuid"] == job["uuid"]
    final = snapshot(s.manager)
    for table in ("jobs", "jobs_archive", "job_resource_heads", "job_counters", "job_events"):
        assert final[table] == after[table], table


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("status", ["queued", "processing"])
def test_pruning_active_opted_out_jobs_keeps_legacy_behavior(manager, monkeypatch, archived, status):
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", str(archived).lower())
    job = manager.create_job(domain="chatbooks", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    if status == "processing":
        acquired = manager.acquire_next_job(domain="chatbooks", queue="default", worker_id="worker", lease_seconds=300)
        assert acquired["uuid"] == job["uuid"]
    sql(manager, "UPDATE jobs SET created_at='2000-01-01',updated_at='2000-01-01' WHERE uuid=?", (job["uuid"],))
    assert manager.prune_jobs(older_than_days=1, statuses=[status], domain="chatbooks") == 1
    assert sql(manager, "SELECT * FROM jobs WHERE uuid=?", (job["uuid"],), rows=True) == []
    assert len(sql(manager, "SELECT * FROM jobs_archive WHERE uuid=?", (job["uuid"],), rows=True)) == int(archived)
    assert sql(manager, "SELECT * FROM job_resource_heads", rows=True) == []


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"owner_user_id": "2"},
        {"method": "embeddings", "similarity_threshold": 0.5},
        {"min_size": 3},
        {"version": 2},
        {"algorithm_version": 2},
        {"_truncated": True},
    ],
)
def test_resource_payload_replacement_preserves_admitted_contract(resource_scenario, monkeypatch, archived, changes):
    s = resource_scenario
    job = s.admit()
    if archived:
        assert s.manager.cancel_job(job["id"])
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
        assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    before = snapshot(s.manager)
    replace_payload = s.manager.replace_archived_job_payload if archived else s.manager.replace_job_payload
    payload = {**make_command().payload, **changes}
    with pytest.raises(ValueError, match="^resource_payload_conflict$"):
        replace_payload(job["id"], payload=payload, expected_uuid=job["uuid"])
    assert snapshot(s.manager) == before
    assert replace_payload(
        job["id"], payload=dict(reversed(list(make_command().payload.items()))), expected_uuid=job["uuid"]
    )
    after = snapshot(s.manager)
    assert after["job_resource_heads"] == before["job_resource_heads"]
    assert after["job_resource_request_receipts"] == before["job_resource_request_receipts"]


@pytest.mark.parametrize("archived", [False, True])
def test_unexpired_request_receipt_keeps_exact_target(resource_scenario, monkeypatch, archived):
    s = resource_scenario
    job = s.admit(key="retained-key")
    assert s.manager.cancel_job(job["id"])
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", str(archived).lower())
    before = snapshot(s.manager)
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 0
    after = snapshot(s.manager)
    assert [event for event in after["job_events"] if event["job_id"] == job["id"]] == before["job_events"]
    assert {table: rows for table, rows in after.items() if table != "job_events"} == {
        table: rows for table, rows in before.items() if table != "job_events"
    }
    replay = s.manager.admit_resource_job(make_command(), request=make_request(key="retained-key"))
    assert replay.kind == "request_replayed"
    assert replay.job["uuid"] == job["uuid"]
    assert s.head()["generation"] == 1


def test_expired_request_pruning_never_resets_head(resource_scenario):
    s = resource_scenario
    job = s.admit(key="expired-key")
    assert s.manager.cancel_job(job["id"])
    sql(s.manager, "UPDATE job_resource_request_receipts SET expires_at='2000-01-01'")
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    assert s.head()["generation"] == 1
    assert sql(s.manager, "SELECT * FROM job_resource_request_receipts", rows=True) == []
    assert s.admit()["resource_generation"] == 2


@pytest.mark.parametrize("archived", [False, True])
def test_registered_unbound_payload_replacement_fails_closed(manager, archived):
    job = manager.create_job(domain="claims", queue="default", job_type="legacy", payload={}, owner_user_id="1")
    sql(manager, "UPDATE jobs SET job_type='claims_rebuild_clusters' WHERE id=?", (job["id"],))
    if archived:
        from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import archive

        archive(manager, job)
    before = snapshot(manager)
    replace_payload = manager.replace_archived_job_payload if archived else manager.replace_job_payload
    with pytest.raises(ValueError, match="^resource_payload_conflict$"):
        replace_payload(job["id"], expected_uuid=job["uuid"], payload=make_command().payload)
    assert snapshot(manager) == before


@pytest.mark.parametrize("archived", [False, True])
def test_semantic_encrypted_rewrite_and_truncation_guard(resource_scenario, monkeypatch, archived):
    import base64
    import os

    s = resource_scenario
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.setenv("JOBS_ENCRYPT_CLAIMS", "true")
    job = s.admit()
    if archived:
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
        monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
        monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
        assert s.manager.cancel_job(job["id"])
        assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    replace_payload = s.manager.replace_archived_job_payload if archived else s.manager.replace_job_payload
    before = snapshot(s.manager)
    assert replace_payload(job["id"], expected_uuid=job["uuid"], payload=make_command().payload)
    after = snapshot(s.manager)
    table = "jobs_archive" if archived else "jobs"
    assert before[table][0]["payload"] != after[table][0]["payload"]
    assert before["job_resource_heads"] == after["job_resource_heads"]
    monkeypatch.setenv("JOBS_JSON_TRUNCATE", "true")
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1")
    with pytest.raises(ValueError, match="^resource_payload_conflict$"):
        replace_payload(job["id"], expected_uuid=job["uuid"], payload=make_command().payload)
    assert snapshot(s.manager) == after


def test_archive_payload_copies_must_both_match_binding(resource_scenario, monkeypatch):
    import gzip
    import json

    s = resource_scenario
    job = s.admit()
    assert s.manager.cancel_job(job["id"])
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    blob = gzip.compress(json.dumps({**make_command().payload, "min_size": 3}).encode())
    sql(s.manager, "UPDATE jobs_archive SET payload_compressed=? WHERE uuid=?", (blob, job["uuid"]))
    before = snapshot(s.manager)
    with pytest.raises(ValueError, match="^resource_payload_conflict$"):
        s.manager.replace_archived_job_payload(job["id"], expected_uuid=job["uuid"], payload=make_command().payload)
    assert snapshot(s.manager) == before


def test_request_retention_uses_jobs_clock(resource_scenario, monkeypatch):
    from datetime import datetime, timedelta, timezone

    s = resource_scenario
    job = s.admit(key="clock-retained")
    assert s.manager.cancel_job(job["id"])
    expires = snapshot(s.manager)["job_resource_request_receipts"][0]["expires_at"]
    if isinstance(expires, str):
        expires = datetime.fromisoformat(expires).replace(tzinfo=timezone.utc)
    monkeypatch.setattr(s.manager._clock, "now_utc", lambda: expires + timedelta(seconds=1))
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    assert s.head()["generation"] == 1
    assert snapshot(s.manager)["job_resource_request_receipts"] == []


@pytest.mark.parametrize("archived", [False, True])
def test_encrypted_input_with_equal_contract_can_be_rewritten(resource_scenario, monkeypatch, archived):
    import base64
    import os

    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob_with_key

    s = resource_scenario
    key = base64.b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", key)
    monkeypatch.setenv("JOBS_ENCRYPT_CLAIMS", "true")
    job = s.admit()
    if archived:
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
        assert s.manager.cancel_job(job["id"])
        assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    replacement = {"_encrypted": encrypt_json_blob_with_key(make_command().payload, key)}
    replace_payload = s.manager.replace_archived_job_payload if archived else s.manager.replace_job_payload
    assert replace_payload(job["id"], expected_uuid=job["uuid"], payload=replacement)
    assert (
        s.manager.get_job_or_archived_by_uuid(job["uuid"], domain="claims", owner_user_id="1")["payload"]
        == make_command().payload
    )


def test_archive_numeric_identity_requires_exact_resource_target(resource_scenario, monkeypatch):
    s = resource_scenario
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    first = s.admit()
    assert s.manager.cancel_job(first["id"])
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    if s.manager.backend == "postgres":
        sql(s.manager, "SELECT setval(pg_get_serial_sequence('jobs','id'), ?, false)", (first["id"],), rows=True)
    second = s.admit()
    assert second["id"] == first["id"]
    assert s.manager.cancel_job(second["id"])
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    before = snapshot(s.manager)
    with pytest.raises(ValueError, match="^resource_payload_conflict$"):
        s.manager.replace_archived_job_payload(first["id"], payload=make_command().payload)
    assert snapshot(s.manager) == before
    assert s.manager.replace_archived_job_payload(
        first["id"], expected_uuid=first["uuid"], payload=make_command().payload
    )


@pytest.mark.parametrize("archived", [False, True])
def test_corrupt_stored_payload_cannot_be_accepted_or_repaired(resource_scenario, monkeypatch, archived):
    import json

    s = resource_scenario
    job = s.admit(key="immutable-contract")
    if archived:
        assert s.manager.cancel_job(job["id"])
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
        # Retained aliases intentionally keep the live replay target.
        sql(s.manager, "UPDATE job_resource_request_receipts SET expires_at='2000-01-01'")
        assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    table = "jobs_archive" if archived else "jobs"
    corrupt = {**make_command().payload, "min_size": 3}
    sql(s.manager, f"UPDATE {table} SET payload=? WHERE uuid=?", (json.dumps(corrupt), job["uuid"]))
    before = snapshot(s.manager)
    replace_payload = s.manager.replace_archived_job_payload if archived else s.manager.replace_job_payload
    for proposed in (corrupt, make_command().payload):
        with pytest.raises(ValueError, match="^resource_payload_conflict$"):
            replace_payload(job["id"], expected_uuid=job["uuid"], payload=proposed)
        assert snapshot(s.manager) == before


@pytest.mark.parametrize("archived", [False, True])
def test_embedding_threshold_is_part_of_admitted_contract(resource_scenario, monkeypatch, archived):
    s = resource_scenario
    command = make_command(payload={**make_command().payload, "method": "embeddings", "similarity_threshold": 0.5})
    job = s.manager.admit_resource_job(command, request=make_request(command, key=None)).job
    if archived:
        assert s.manager.cancel_job(job["id"])
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
        assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    before = snapshot(s.manager)
    replace_payload = s.manager.replace_archived_job_payload if archived else s.manager.replace_job_payload
    with pytest.raises(ValueError, match="^resource_payload_conflict$"):
        replace_payload(job["id"], expected_uuid=job["uuid"], payload={**command.payload, "similarity_threshold": 0.6})
    assert snapshot(s.manager) == before
    assert replace_payload(job["id"], expected_uuid=job["uuid"], payload=command.payload)


@pytest.mark.parametrize("archived", [False, True])
def test_opted_out_payload_replacement_remains_unrestricted(manager, monkeypatch, archived):
    job = manager.create_job(domain="chatbooks", queue="default", job_type="legacy", payload={}, owner_user_id="1")
    if archived:
        assert manager.cancel_job(job["id"])
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
        assert manager.prune_jobs(older_than_days=0, domain="chatbooks") == 1
    replace_payload = manager.replace_archived_job_payload if archived else manager.replace_job_payload
    before = snapshot(manager)
    assert not replace_payload(job["id"], expected_uuid="mismatched", payload={"new": "value"})
    assert snapshot(manager) == before
    assert replace_payload(job["id"], expected_uuid=job["uuid"], payload={"new": "value"})


@pytest.mark.unit
def test_partial_resource_metadata_fails_closed():
    from itertools import product

    from tldw_Server_API.app.core.Jobs.operations.resource import resource_scope_for
    from tldw_Server_API.app.core.Jobs.resource_contracts import RESOURCE_METADATA_FIELDS

    complete = dict(zip(RESOURCE_METADATA_FIELDS, ("cluster-rebuild", "authority", 1, "{}", "fingerprint")))
    for present in product((False, True), repeat=len(RESOURCE_METADATA_FIELDS)):
        if all(present):
            continue
        row = {"domain": "claims", "owner_user_id": "1", "job_type": "claims_rebuild_clusters"}
        row.update(
            {field: complete[field] if exists else None for field, exists in zip(RESOURCE_METADATA_FIELDS, present)}
        )
        with pytest.raises(ValueError, match="resource Job binding is invalid"):
            resource_scope_for(row)


def test_archive_encrypted_copies_compare_decrypted_contracts(resource_scenario, monkeypatch):
    import base64
    import gzip
    import json
    import os

    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob_with_key

    s = resource_scenario
    key = base64.b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", key)
    monkeypatch.setenv("JOBS_ENCRYPT_CLAIMS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    job = s.admit()
    assert s.manager.cancel_job(job["id"])
    assert s.manager.prune_jobs(older_than_days=0, domain="claims") == 1
    different_ciphertext = {"_encrypted": encrypt_json_blob_with_key(make_command().payload, key)}
    sql(
        s.manager,
        "UPDATE jobs_archive SET payload_compressed=? WHERE uuid=?",
        (gzip.compress(json.dumps(different_ciphertext).encode()), job["uuid"]),
    )
    assert s.manager.replace_archived_job_payload(job["id"], expected_uuid=job["uuid"], payload=make_command().payload)
    assert (
        s.manager.get_job_or_archived_by_uuid(job["uuid"], domain="claims", owner_user_id="1")["payload"]
        == make_command().payload
    )


def test_rotation_keeps_bounded_selection_and_contracts(resource_scenario, monkeypatch):
    import base64
    import json
    import os

    from tldw_Server_API.app.core.Security.crypto import decrypt_json_blob_with_key

    s = resource_scenario
    old_key, new_key = [base64.b64encode(os.urandom(32)).decode() for _ in range(2)]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", old_key)
    monkeypatch.setenv("JOBS_ENCRYPT_CLAIMS", "true")
    s.admit()
    other = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
    s.manager.admit_resource_job(other, request=make_request(other, key=None))
    before = snapshot(s.manager)
    assert (
        s.manager.rotate_encryption_keys(
            domain="claims", fields=["payload"], old_key_b64=old_key, new_key_b64=new_key, limit=1
        )
        == 1
    )
    after = snapshot(s.manager)
    rows = sorted(after["jobs"], key=lambda row: row["id"])
    for row, key, expected in zip(rows, (new_key, old_key), (make_command().payload, other.payload)):
        stored = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        assert decrypt_json_blob_with_key(stored["_encrypted"], key) == expected
    assert after["job_resource_heads"] == before["job_resource_heads"]
    assert after["job_resource_request_receipts"] == before["job_resource_request_receipts"]


def test_opted_out_duplicate_archive_ids_keep_legacy_replacement_behavior(manager, monkeypatch):
    import json

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    for number in range(2):
        job = manager.create_job(
            domain="chatbooks", queue="default", job_type="legacy", payload={"number": number}, owner_user_id="1"
        )
        assert manager.cancel_job(job["id"])
        assert manager.prune_jobs(older_than_days=0, domain="chatbooks") == 1
        if manager.backend == "postgres":
            sql(manager, "SELECT setval(pg_get_serial_sequence('jobs','id'), ?, false)", (job["id"],), rows=True)
    assert manager.replace_archived_job_payload(job["id"], payload={"replacement": True})
    archived = snapshot(manager)["jobs_archive"]
    assert len(archived) == 2
    assert all(
        (json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]) == {"replacement": True}
        for row in archived
    )


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
def test_sqlite_frozen_targets_are_indexed_linear_and_null_safe(tmp_path, table):
    import sqlite3

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.operations.resource import ResourceAdmissionStore, ResourceMutation

    jm = JobManager(tmp_path / "target-plan.db")
    conn = jm._connect()
    steps = []
    try:
        for count in (128, 256, 1024):
            conn.execute(f"DELETE FROM {table}")
            conn.executemany(
                f"INSERT INTO {table}(id,uuid,domain,queue,job_type,owner_user_id,status) "
                "VALUES (?,?,'ordinary','default','ordinary','1','queued')",
                [(number, None if number == 1 else str(number)) for number in range(1, count + 1)],
            )
            mutation = ResourceMutation(ResourceAdmissionStore(conn), where="1=1", archived=table == "jobs_archive")
            mutation._freeze_rows(
                [{"id": number, "uuid": None if number == 1 else str(number)} for number in range(1, count + 1)]
            )
            query = f"SELECT id FROM {table} WHERE " + mutation.where
            plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + query, mutation.params)]
            instructions = []
            conn.set_progress_handler(lambda instructions=instructions: instructions.append(1) or 0, 1)
            assert len(conn.execute(query, mutation.params).fetchall()) == count
            conn.set_progress_handler(None, 0)
            steps.append(len(instructions))
            conn.execute(f"UPDATE {table} SET uuid='replacement' WHERE id=2")
            assert len(conn.execute(query, mutation.params).fetchall()) == count - 1
        assert steps[1] < 3 * steps[0], (steps, plan)
        assert any(f"SEARCH {table} USING INTEGER PRIMARY KEY" in detail for detail in plan), plan
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 10)
        assert len(conn.execute(query, mutation.params).fetchall()) == 1023
    finally:
        conn.close()
