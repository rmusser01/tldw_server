"""Behavioral resource-admission cases, shared with the official PG fixture."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import CreateJobCommand, IdempotentOperationCommand
from tldw_Server_API.app.core.Jobs.resource_contracts import ResourceAdmissionRequest, ResourceScope

TYPE = "claims_rebuild_clusters"
CONFLICT_REFERENCE_FIELDS = {"id", "uuid", "domain", "owner_user_id", "job_type", "resource"}


def make_command(**changes):
    payload = {
        "version": 1,
        "owner_user_id": "1",
        "method": "exact",
        "min_size": 2,
        "similarity_threshold": None,
        "algorithm_version": 1,
    }
    return replace(CreateJobCommand("claims", "default", TYPE, payload, "1", priority=5), **changes)


def make_request(command=None, key="request-1", **changes):
    command = command or make_command()
    settings = {k: v for k, v in command.payload.items() if k != "owner_user_id"}
    if settings["method"] == "exact":
        settings.pop("similarity_threshold")
    return ResourceAdmissionRequest(
        ResourceScope(command.domain, command.owner_user_id, "cluster-rebuild"),
        settings,
        key,
        **changes,
    )


def sql(manager, statement, params=(), *, rows=False):
    conn = manager._connect()
    try:
        with conn:
            if manager.backend == "postgres":
                with manager._pg_cursor(conn) as cur:
                    cur.execute(statement.replace("?", "%s"), params)
                    return [dict(row) for row in cur.fetchall()] if rows else None
            cur = conn.execute(statement, params)
            return [dict(row) for row in cur.fetchall()] if rows else None
    finally:
        conn.close()


def snapshot(manager):
    tables = (
        "jobs",
        "jobs_archive",
        "job_resource_heads",
        "job_resource_request_receipts",
        "job_idempotency_receipts",
        "job_events",
        "job_counters",
    )
    return {table: sql(manager, f"SELECT * FROM {table}", rows=True) for table in tables}


def archive(manager, job):
    # Task 3 owns lifecycle/archive paths. Seed the supported complete projection here.
    fields = (
        "id",
        "uuid",
        "domain",
        "queue",
        "job_type",
        "owner_user_id",
        "payload",
        "status",
        "resource",
        "resource_authority_uuid",
        "resource_generation",
        "resource_settings_json",
        "resource_settings_fingerprint",
        "created_at",
        "updated_at",
    )
    row = sql(manager, "SELECT * FROM jobs WHERE uuid=?", (job["uuid"],), rows=True)[0]
    values = [row[field] for field in fields]
    if manager.backend == "postgres":
        from psycopg.types.json import Jsonb

        values[fields.index("payload")] = Jsonb(values[fields.index("payload")])
    sql(manager, f"INSERT INTO jobs_archive ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
    sql(manager, "DELETE FROM jobs WHERE uuid=?", (job["uuid"],))


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CLAIMS", "default,other")
    return JobManager(tmp_path / "resources.db")


class ResourceAdmissionCases:
    @pytest.mark.parametrize("target", ["active", "retained_active", "retained_archive", "retained_with_successor"])
    def test_conflict_returns_only_exact_authorized_reference_without_mutation(self, manager, target):
        first = manager.admit_resource_job(make_command(), request=make_request())
        sql(
            manager,
            "UPDATE jobs SET result=?,error_message=? WHERE uuid=?",
            ('{"private":"must-not-be-exposed"}', "private raw failure", first.job["uuid"]),
        )
        if target in ("retained_archive", "retained_with_successor"):
            sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (first.job["uuid"],))
            sql(manager, "UPDATE job_resource_heads SET active_job_uuid=NULL")
            if target == "retained_archive":
                archive(manager, first.job)
            else:
                successor = manager.admit_resource_job(make_command(), request=make_request(key="successor"))
                assert successor.job["uuid"] != first.job["uuid"]
        changed = make_command(queue="other", payload={**make_command().payload, "min_size": 3})
        request = make_request(changed, key="new-alias" if target == "active" else "request-1")
        before = snapshot(manager)
        conflict = manager.admit_resource_job(changed, request=request)
        assert conflict.kind == ("active_conflict" if target == "active" else "request_conflict")
        assert conflict.job == {field: first.job[field] for field in CONFLICT_REFERENCE_FIELDS}
        assert snapshot(manager) == before

    @pytest.mark.parametrize("key", [None, "request-1"])
    def test_conflict_reference_never_discloses_another_owner(self, manager, key):
        foreign = manager.admit_resource_job(make_command(), request=make_request())
        own_command = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
        own = manager.admit_resource_job(own_command, request=make_request(own_command))
        changed = replace(own_command, payload={**own_command.payload, "min_size": 3})
        before = snapshot(manager)
        conflict = manager.admit_resource_job(changed, request=make_request(changed, key=key))
        assert conflict.kind == ("active_conflict" if key is None else "request_conflict")
        assert conflict.job == {field: own.job[field] for field in CONFLICT_REFERENCE_FIELDS}
        assert foreign.job["uuid"] not in str(conflict)
        with pytest.raises(ValueError, match="scope does not match"):
            manager.admit_resource_job(
                changed, request=replace(make_request(changed, key=key), scope=make_request().scope)
            )
        assert snapshot(manager) == before

    @pytest.mark.parametrize("different", [False, True])
    def test_older_active_generation_reuses_or_conflicts_without_rewinding_head(self, manager, different):
        first = manager.admit_resource_job(make_command(), request=make_request())
        sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (first.job["uuid"],))
        sql(manager, "UPDATE job_resource_heads SET active_job_uuid=NULL")
        newer = manager.admit_resource_job(make_command(), request=make_request(key="newer"))
        assert newer.job["resource_generation"] == 2
        # Fixture state only: Task 3 owns the reactivation API and its bookkeeping.
        sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (newer.job["uuid"],))
        sql(manager, "UPDATE jobs SET status='queued' WHERE uuid=?", (first.job["uuid"],))
        sql(manager, "UPDATE job_resource_heads SET active_job_uuid=?", (first.job["uuid"],))
        sql(manager, "UPDATE job_counters SET ready_count=1,scheduled_count=0,processing_count=0")
        command = make_command(payload={**make_command().payload, "min_size": 3 if different else 2})
        before = snapshot(manager)
        outcome = manager.admit_resource_job(command, request=make_request(command, key=None))
        assert outcome.kind == ("active_conflict" if different else "active_reused")
        assert outcome.job["uuid"] == first.job["uuid"]
        if different:
            assert outcome.job == {field: first.job[field] for field in CONFLICT_REFERENCE_FIELDS}
        else:
            assert outcome.job["resource_generation"] == 1
        assert snapshot(manager) == before
        assert before["job_resource_heads"][0]["generation"] == 2

    def test_active_generation_above_head_still_fails_without_reference_or_mutation(self, manager):
        first = manager.admit_resource_job(make_command(), request=make_request())
        sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (first.job["uuid"],))
        sql(manager, "UPDATE job_resource_heads SET active_job_uuid=NULL")
        manager.admit_resource_job(make_command(), request=make_request(key="newer"))
        sql(manager, "UPDATE job_resource_heads SET generation=1")
        before = snapshot(manager)
        with pytest.raises(RuntimeError, match="resource Job authority is invalid"):
            manager.admit_resource_job(make_command(), request=make_request(key=None))
        assert snapshot(manager) == before

    @pytest.mark.parametrize("job_type", ["claimsXrebuildXclusters.v1", TYPE + ".v0", TYPE + ".version"])
    def test_unregistered_alias_lookalikes_do_not_occupy_resource(self, manager, job_type):
        manager.create_job(domain="claims", queue="default", job_type=job_type, payload={}, owner_user_id="1")
        assert manager.admit_resource_job(make_command(), request=make_request()).kind == "created"

    def test_active_reuse_does_not_submit_again(self, manager):
        command, request = make_command(), make_request()
        first = manager.admit_resource_job(command, request=request)
        second = manager.admit_resource_job(command, request=request)
        assert (first.kind, second.kind) == ("created", "request_replayed")
        assert first.job["uuid"] == second.job["uuid"]
        facts = snapshot(manager)
        assert len(facts["jobs"]) == len(facts["job_events"]) == 1
        assert facts["job_counters"][0]["ready_count"] == 1
        assert first.job["idempotency_key"] is None
        assert "request-1" not in str(facts)
        assert first.job["resource_generation"] == 1

    def test_retained_replay_survives_queue_policy_change(self, manager, monkeypatch):
        first = manager.admit_resource_job(make_command(), request=make_request())
        monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CLAIMS", "disallowed")
        monkeypatch.setenv("JOBS_ALLOWED_JOB_TYPES_CLAIMS", "disallowed")
        monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
        monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN", "1")
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1")
        before = snapshot(manager)
        replay = manager.admit_resource_job(make_command(queue="other"), request=make_request())
        assert replay.kind == "request_replayed"
        assert replay.job["uuid"] == first.job["uuid"]
        assert snapshot(manager) == before

    def test_keyless_reuse_and_method_conflict_cross_queues(self, manager, monkeypatch):
        first = manager.admit_resource_job(make_command(), request=make_request(key=None))
        monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CLAIMS", "disallowed")
        reused = manager.admit_resource_job(make_command(queue="other"), request=make_request(key=None))
        assert reused.kind == "active_reused"
        assert reused.job["uuid"] == first.job["uuid"]
        changed = make_command(payload={**make_command().payload, "method": "embeddings", "similarity_threshold": 0.8})
        conflict = manager.admit_resource_job(changed, request=make_request(changed, key=None))
        assert conflict.kind == "active_conflict"
        assert len(snapshot(manager)["jobs"]) == 1

    def test_full_settings_equality_even_with_digest_collision(self, manager, monkeypatch):
        monkeypatch.setattr("tldw_Server_API.app.core.Jobs.resource_contracts.settings_fingerprint", lambda _: "a" * 64)
        manager.admit_resource_job(make_command(), request=make_request())
        other = make_command(payload={**make_command().payload, "min_size": 3})
        assert manager.admit_resource_job(other, request=make_request(other)).kind == "request_conflict"
        assert manager.admit_resource_job(other, request=make_request(other, key=None)).kind == "active_conflict"

    @pytest.mark.parametrize("archived", [False, True])
    def test_retained_key_after_terminal_and_archive(self, manager, archived):
        first = manager.admit_resource_job(make_command(), request=make_request())
        sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (first.job["uuid"],))
        sql(manager, "UPDATE job_resource_heads SET active_job_uuid=NULL")
        if archived:
            archive(manager, first.job)
        before = snapshot(manager)
        replay = manager.admit_resource_job(make_command(), request=make_request())
        assert replay.kind == "request_replayed"
        assert replay.job["uuid"] == first.job["uuid"]
        assert replay.job["resource_generation"] == 1
        assert snapshot(manager) == before

    def test_expired_key_is_fresh_and_preserves_authority(self, manager):
        first = manager.admit_resource_job(make_command(), request=make_request())
        sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (first.job["uuid"],))
        sql(manager, "UPDATE job_resource_heads SET active_job_uuid=NULL")
        sql(manager, "UPDATE job_resource_request_receipts SET expires_at=?", (datetime(2000, 1, 1),))
        second = manager.admit_resource_job(make_command(), request=make_request())
        assert second.kind == "created"
        assert second.job["uuid"] != first.job["uuid"]
        assert second.job["resource_generation"] == 2
        assert second.job["resource_authority_uuid"] == first.job["resource_authority_uuid"]

    @pytest.mark.parametrize("mismatch", ["owner", "domain", "resource", "settings", "payload_owner"])
    def test_contract_mismatch_fails_before_lookup(self, manager, mismatch):
        manager.admit_resource_job(make_command(), request=make_request())
        request, command = make_request(), make_command()
        if mismatch in ("owner", "domain", "resource"):
            field = {"owner": "owner_user_id", "domain": "domain", "resource": "resource"}[mismatch]
            request = replace(request, scope=replace(request.scope, **{field: "other"}))
        elif mismatch == "settings":
            request = replace(request, canonical_settings={**request.canonical_settings, "min_size": 3})
        else:
            command = replace(command, payload={**command.payload, "owner_user_id": "2"})
        before = snapshot(manager)
        with pytest.raises(ValueError):
            manager.admit_resource_job(command, request=request)
        assert snapshot(manager) == before

    def test_independent_owners(self, manager):
        first = manager.admit_resource_job(make_command(), request=make_request())
        other = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
        second = manager.admit_resource_job(other, request=make_request(other))
        assert second.kind == "created"
        assert second.job["uuid"] != first.job["uuid"]

    @pytest.mark.parametrize("keys", [(None, None), ("same", "same"), ("first", "second")])
    @pytest.mark.parametrize("different", [False, "min_size", "method"])
    def test_concurrent_managers_serialize_across_queues(self, manager, different, keys):
        second = (
            JobManager(manager.db_path)
            if manager.backend == "sqlite"
            else JobManager(backend="postgres", db_url=manager.db_url)
        )
        barrier = Barrier(2)
        commands = [make_command(), make_command(queue="other")]
        if different == "min_size":
            commands[1] = replace(commands[1], payload={**commands[1].payload, "min_size": 3})
        elif different == "method":
            commands[1] = replace(
                commands[1], payload={**commands[1].payload, "method": "embeddings", "similarity_threshold": 0.8}
            )

        def submit(index):
            barrier.wait(timeout=10)
            return (manager, second)[index].admit_resource_job(
                commands[index], request=make_request(commands[index], key=keys[index])
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, (0, 1)))
        retained = keys[0] is not None and keys[0] == keys[1]
        expected = (
            ("request_conflict" if retained else "active_conflict")
            if different
            else ("request_replayed" if retained else "active_reused")
        )
        assert sorted(result.kind for result in results) == sorted(["created", expected])
        assert len(snapshot(manager)["jobs"]) == len(snapshot(manager)["job_events"]) == 1

    def test_alias_ceiling_preserves_old_replay_and_keyless_reuse(self, manager):
        first = manager.admit_resource_job(make_command(), request=make_request())
        for index in range(127):
            assert (
                manager.admit_resource_job(make_command(), request=make_request(key=f"alias-{index}")).kind
                == "active_reused"
            )
        before = snapshot(manager)
        rejected = manager.admit_resource_job(make_command(), request=make_request(key="overflow"))
        assert rejected.kind == "rejected"
        assert rejected.rejection.admission_rejection_reason.value == "quota_exceeded"
        assert manager.admit_resource_job(make_command(), request=make_request()).kind == "request_replayed"
        assert (
            manager.admit_resource_job(make_command(), request=make_request(key=None)).job["uuid"] == first.job["uuid"]
        )
        assert snapshot(manager) == before

    @pytest.mark.parametrize("fault", ["_insert_created_event", "_bump_counters"])
    def test_creation_fault_rolls_back_everything(self, manager, monkeypatch, fault):
        module = f"tldw_Server_API.app.core.Jobs.operations.{manager.backend}.admission"
        from importlib import import_module

        original = getattr(import_module(module), fault)

        def fail(*_args, **_kwargs):
            original(*_args, **_kwargs)
            raise RuntimeError("injected admission fault")

        monkeypatch.setattr(f"{module}.{fault}", fail)
        before = snapshot(manager)
        with pytest.raises(RuntimeError, match="injected admission fault"):
            manager.admit_resource_job(make_command(), request=make_request())
        assert snapshot(manager) == before

    @pytest.mark.parametrize("corruption", ["missing_head_target", "duplicate_archive_target", "unbound_active"])
    def test_unsafe_authority_fails_closed(self, manager, corruption):
        first = manager.admit_resource_job(make_command(), request=make_request())
        if corruption == "missing_head_target":
            sql(manager, "DELETE FROM jobs WHERE uuid=?", (first.job["uuid"],))
        elif corruption == "duplicate_archive_target":
            # Retain live row as well as archive to make UUID resolution ambiguous.
            archive(manager, first.job)
            row = snapshot(manager)["jobs_archive"][0]
            command = make_command(job_type="legacy")
            legacy = manager.create_job(
                domain=command.domain, queue=command.queue, job_type="legacy", payload={}, owner_user_id="1"
            )
            sql(manager, "UPDATE jobs SET uuid=? WHERE id=?", (row["uuid"], legacy["id"]))
        else:
            sql(manager, "DELETE FROM job_resource_request_receipts")
            sql(manager, "DELETE FROM job_resource_heads")
        before = snapshot(manager)
        for key in (None, "request-1"):
            with pytest.raises(RuntimeError, match="resource"):
                manager.admit_resource_job(make_command(), request=make_request(key=key))
        assert snapshot(manager) == before

    def test_resource_payload_cannot_be_destructively_truncated(self, manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1")
        monkeypatch.setenv("JOBS_JSON_TRUNCATE", "true")
        before = snapshot(manager)
        with pytest.raises(ValueError):
            manager.admit_resource_job(make_command(), request=make_request())
        assert snapshot(manager) == before

    @pytest.mark.parametrize("boundary", ["insert", "head", "receipt"])
    def test_attachment_fault_rolls_back_job_head_receipt_counter_event(self, manager, monkeypatch, boundary):
        from tldw_Server_API.app.core.Jobs.operations.resource import ResourceAdmissionStore

        method = "attach_receipt" if boundary == "head" else "attach_created"
        original = getattr(ResourceAdmissionStore, method)

        def fail_after_write(store, *args, **kwargs):
            if boundary == "insert":
                assert args[1]["resource_generation"] == 1
                assert args[1]["resource_settings_json"]
            if boundary == "receipt":
                original(store, *args, **kwargs)
            raise RuntimeError("attachment fault")

        monkeypatch.setattr(ResourceAdmissionStore, method, fail_after_write)
        before = snapshot(manager)
        with pytest.raises(RuntimeError, match="attachment fault"):
            manager.admit_resource_job(make_command(), request=make_request())
        assert snapshot(manager) == before

    @pytest.mark.parametrize("quota", ["JOBS_QUOTA_MAX_QUEUED", "JOBS_QUOTA_SUBMITS_PER_MIN"])
    def test_fresh_quota_rejection_does_not_allocate_authority_or_submit(self, manager, monkeypatch, quota):
        manager.create_job(domain="claims", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
        monkeypatch.setenv(quota, "1")
        before = snapshot(manager)
        result = manager.admit_resource_job(make_command(), request=make_request())
        assert result.kind == "rejected"
        assert result.rejection.admission_rejection_reason.value == "quota_exceeded"
        assert snapshot(manager) == before

    def test_unbound_legacy_active_row_blocks_resource_admission(self, manager):
        job = manager.create_job(domain="claims", queue="other", job_type="legacy", payload={}, owner_user_id="1")
        sql(manager, "UPDATE jobs SET job_type=? WHERE id=?", (TYPE, job["id"]))
        before = snapshot(manager)
        with pytest.raises(RuntimeError, match="resource"):
            manager.admit_resource_job(make_command(), request=make_request())
        assert snapshot(manager) == before

    def test_missing_head_cannot_reset_terminal_resource_authority(self, manager):
        first = manager.admit_resource_job(make_command(), request=make_request())
        sql(manager, "UPDATE jobs SET status='completed' WHERE uuid=?", (first.job["uuid"],))
        sql(manager, "DELETE FROM job_resource_request_receipts")
        sql(manager, "DELETE FROM job_resource_heads")
        before = snapshot(manager)
        with pytest.raises(RuntimeError, match="resource"):
            manager.admit_resource_job(make_command(), request=make_request(key=None))
        assert snapshot(manager) == before

    def test_encryption_preserves_admitted_contract_and_reuse(self, manager, monkeypatch):
        import base64
        import json

        monkeypatch.setenv("JOBS_ENCRYPT_CLAIMS", "true")
        monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", base64.b64encode(b"k" * 32).decode())
        first = manager.admit_resource_job(make_command(), request=make_request())
        stored = sql(manager, "SELECT payload FROM jobs", rows=True)[0]["payload"]
        stored = json.loads(stored) if isinstance(stored, str) else stored
        assert "_encrypted" in stored
        assert manager._maybe_decrypt_json(stored, fail_on_error=True) == make_command().payload
        assert manager.admit_resource_job(make_command(), request=make_request()).kind == "request_replayed"
        assert (
            first.job["resource_settings_json"]
            == snapshot(manager)["job_resource_request_receipts"][0]["settings_json"]
        )

    def test_secret_redaction_cannot_change_resource_contract(self, manager, monkeypatch):
        monkeypatch.setattr(
            manager, "_scan_and_redact_secrets", lambda payload: ({**payload, "min_size": 4}, True, ["min_size"])
        )
        before = snapshot(manager)
        with pytest.raises(ValueError):
            manager.admit_resource_job(make_command(), request=make_request())
        assert snapshot(manager) == before

    def test_shared_preparation_runs_once_only_for_creation(self, manager, monkeypatch):
        calls = []
        original = manager._prepare_job_admission

        def prepare(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(manager, "_prepare_job_admission", prepare)
        manager.admit_resource_job(make_command(), request=make_request())
        manager.admit_resource_job(make_command(), request=make_request())
        manager.admit_resource_job(make_command(), request=make_request(key=None))
        assert calls == [1]

    @pytest.mark.parametrize("job_type", [TYPE, TYPE + ".v1", TYPE + ".v2"])
    def test_public_backend_admission_import_rejects_bare_resource_commands(self, manager, job_type):
        from importlib import import_module

        operation = import_module(f"tldw_Server_API.app.core.Jobs.operations.{manager.backend}").create_job_admission
        conn = manager._connect()
        args = (conn, manager._pg_cursor) if manager.backend == "postgres" else (conn,)
        before = snapshot(manager)
        try:
            with pytest.raises(ValueError, match="^resource_admission_required$"):
                operation(
                    *args,
                    command=make_command(job_type=job_type),
                    uuid_value="unused",
                    now=datetime.now(timezone.utc),
                    max_queued_quota=0,
                    submits_per_minute_quota=0,
                    counters_enabled=True,
                )
        finally:
            conn.close()
        assert snapshot(manager) == before

    @pytest.mark.parametrize("method", ["create_job", "admit_job"])
    @pytest.mark.parametrize("job_type", [TYPE, TYPE + ".v1", TYPE + ".v2"])
    def test_bare_admission_rejects_resource_types(self, manager, method, job_type):
        with pytest.raises(ValueError, match="resource_admission_required"):
            getattr(manager, method)(domain="claims", queue="default", job_type=job_type, payload={}, owner_user_id="1")
        assert not snapshot(manager)["jobs"]

    @pytest.mark.parametrize("method", ["admit_idempotent_operation", "replay_idempotent_operation"])
    @pytest.mark.parametrize("job_type", [TYPE, TYPE + ".v1", TYPE + ".v2"])
    @pytest.mark.parametrize("entry", ["facade", "backend"])
    def test_receipt_facades_reject_resource_required_types(self, manager, method, job_type, entry):
        command = IdempotentOperationCommand(
            job=make_command(job_type=job_type, batch_group="scope"),
            key_digest="a" * 64,
            request_fingerprint="b" * 64,
            operation_scope="scope",
            receipt_expires_at=datetime.now(timezone.utc) + timedelta(days=31),
        )
        ordinary = replace(command, job=replace(command.job, job_type="ordinary-receipt"))
        admitted = manager.admit_idempotent_operation(ordinary)
        assert manager.replay_idempotent_operation(ordinary).job["uuid"] == admitted.job["uuid"]
        # Seed legacy unbound resource rows/receipts through their old shape.
        sql(manager, "UPDATE jobs SET job_type=? WHERE uuid=?", (job_type, admitted.job["uuid"]))
        sql(manager, "UPDATE job_idempotency_receipts SET job_type=?", (job_type,))
        before = snapshot(manager)
        with pytest.raises(ValueError, match="^resource_admission_required$"):
            if entry == "facade":
                getattr(manager, method)(command)
            else:
                from importlib import import_module

                operation = getattr(
                    import_module(f"tldw_Server_API.app.core.Jobs.operations.{manager.backend}.idempotency"), method
                )
                conn = manager._connect()
                args = (conn, manager._pg_cursor) if manager.backend == "postgres" else (conn,)
                try:
                    if method.startswith("replay"):
                        operation(*args, command)
                    else:
                        operation(
                            *args,
                            command=command,
                            uuid_value="unused",
                            now=datetime.now(timezone.utc),
                            max_queued_quota=0,
                            submits_per_minute_quota=0,
                            counters_enabled=True,
                        )
                finally:
                    conn.close()
        assert snapshot(manager) == before


class TestSQLite(ResourceAdmissionCases):
    pass
