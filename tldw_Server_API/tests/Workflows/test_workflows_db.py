import json
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import Workflows_DB as workflows_db_mod
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase


def test_workflows_db_crud(tmp_path):


    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    # Create definition
    defn = {
        "name": "demo",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "Hi {{ inputs.name }}"}},
        ],
    }
    wid = db.create_definition(
        tenant_id="t1",
        name="demo",
        version=1,
        owner_id="1",
        visibility="private",
        description="",
        tags=["x"],
        definition=defn,
    )
    assert wid > 0

    d = db.get_definition(wid)
    assert d and d.name == "demo"

    lst = db.list_definitions(owner_id="1")
    assert any(x.id == wid for x in lst)

    # Create run
    run_id = "run-1"
    db.create_run(
        run_id=run_id,
        tenant_id="t1",
        user_id="1",
        inputs={"name": "Alice"},
        workflow_id=wid,
        definition_version=1,
        definition_snapshot=defn,
    )
    run = db.get_run(run_id)
    assert run and run.status == "queued"

    # Update status and events
    db.update_run_status(run_id, status="running")
    seq1 = db.append_event("t1", run_id, "run_started", {"mode": "async"})
    assert seq1 == 1
    seq2 = db.append_event("t1", run_id, "step_completed", {"step_id": "s1"})
    assert seq2 == 2
    events = db.get_events(run_id)
    assert [e["event_type"] for e in events] == ["run_started", "step_completed"]
    events_since = db.get_events(run_id, since=1)
    assert len(events_since) == 1 and events_since[0]["event_seq"] == 2


def test_sqlite_append_event_uses_begin_immediate(tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    run_id = "run-immediate"
    db.create_run(
        run_id=run_id,
        tenant_id="t1",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "immediate", "version": 1, "steps": []},
    )

    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)  # type: ignore[attr-defined]
    try:
        seq = db.append_event("t1", run_id, "run_started", {"mode": "async"})
    finally:
        db._conn.set_trace_callback(None)  # type: ignore[attr-defined]

    assert seq == 1
    assert any("BEGIN IMMEDIATE" in statement.upper() for statement in statements)


def test_sqlite_append_event_releases_connection_on_serialization_error(monkeypatch, tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    run_id = "run-release-on-error"
    db.create_run(
        run_id=run_id,
        tenant_id="t1",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "release-on-error", "version": 1, "steps": []},
    )

    released: list[sqlite3.Connection] = []
    original_release = db._release_sqlite

    def _record_release(conn: sqlite3.Connection) -> None:
        released.append(conn)
        original_release(conn)

    monkeypatch.setattr(db, "_release_sqlite", _record_release)

    with pytest.raises(TypeError):
        db.append_event("t1", run_id, "bad_payload", {"not_json": object()})

    assert released == [db._conn]
    assert db._conn.in_transaction is False


def test_workflow_research_wait_db_tracks_links(tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-1",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "wait-link", "steps": []},
    )

    db.upsert_research_wait_link(
        wait_id="rw-1",
        tenant_id="tenant",
        workflow_run_id="wf-run-1",
        step_id="wait",
        research_run_id="research-session-10",
        checkpoint_id="checkpoint-4",
        checkpoint_type="sources_review",
        wait_status="waiting",
        wait_payload={
            "__status__": "waiting_human",
            "reason": "research_checkpoint",
            "run_id": "research-session-10",
        },
        active_poll_seconds=1.25,
    )

    link = db.get_research_wait_link(workflow_run_id="wf-run-1", step_id="wait")
    assert link is not None
    assert link["research_run_id"] == "research-session-10"
    assert link["checkpoint_id"] == "checkpoint-4"
    assert link["wait_status"] == "waiting"
    assert json.loads(link["wait_payload_json"])["reason"] == "research_checkpoint"


def test_workflow_research_wait_db_claims_links_for_resume_once(tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-2",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "wait-claim", "steps": []},
    )

    db.upsert_research_wait_link(
        wait_id="rw-2",
        tenant_id="tenant",
        workflow_run_id="wf-run-2",
        step_id="wait",
        research_run_id="research-session-11",
        checkpoint_id="checkpoint-5",
        checkpoint_type="outline_review",
        wait_status="waiting",
        wait_payload={
            "__status__": "waiting_human",
            "reason": "research_checkpoint",
            "run_id": "research-session-11",
        },
        active_poll_seconds=2.0,
    )

    claimed = db.claim_research_waits_for_resume(
        research_run_id="research-session-11",
        checkpoint_id="checkpoint-5",
    )
    assert len(claimed) == 1
    assert claimed[0]["workflow_run_id"] == "wf-run-2"
    assert claimed[0]["step_id"] == "wait"

    claimed_again = db.claim_research_waits_for_resume(
        research_run_id="research-session-11",
        checkpoint_id="checkpoint-5",
    )
    assert claimed_again == []


def test_workflow_step_attempt_round_trip(tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-attempt-1",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "step-attempt-round-trip", "steps": []},
    )
    db.create_step_run(
        step_run_id="wf-run-attempt-1:s1:1",
        tenant_id="tenant",
        run_id="wf-run-attempt-1",
        step_id="s1",
        name="Prompt step",
        step_type="prompt",
        status="running",
        inputs={"config": {"template": "Hello"}},
    )

    attempt_id = db.create_step_attempt(
        tenant_id="tenant",
        run_id="wf-run-attempt-1",
        step_run_id="wf-run-attempt-1:s1:1",
        step_id="s1",
        attempt_number=1,
        status="running",
        metadata={"step_type": "prompt", "retry_recommendation": "safe"},
    )

    db.complete_step_attempt(
        attempt_id=attempt_id,
        status="failed",
        reason_code_core="transient_network_error",
        reason_code_detail="RuntimeError",
        retryable=True,
        error_summary="upstream reset",
        metadata={
            "step_type": "prompt",
            "retry_recommendation": "safe",
            "failure_envelope": {"reason_code_core": "transient_network_error"},
        },
    )

    attempts = db.list_step_attempts(run_id="wf-run-attempt-1", step_id="s1")
    assert len(attempts) == 1
    assert attempts[0]["attempt_id"] == attempt_id
    assert attempts[0]["step_run_id"] == "wf-run-attempt-1:s1:1"
    assert attempts[0]["attempt_number"] == 1
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["reason_code_core"] == "transient_network_error"
    assert attempts[0]["reason_code_detail"] == "RuntimeError"
    assert bool(attempts[0]["retryable"]) is True
    assert attempts[0]["error_summary"] == "upstream reset"
    assert attempts[0]["started_at"] is not None
    assert attempts[0]["ended_at"] is not None
    assert attempts[0]["metadata_json"]["retry_recommendation"] == "safe"
    assert attempts[0]["metadata_json"]["failure_envelope"]["reason_code_core"] == "transient_network_error"


def test_workflow_step_run_listing_round_trip(tmp_path, monkeypatch):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-step-runs-1",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "step-runs", "steps": []},
    )
    timestamps = iter(
        [
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:02+00:00",
        ]
    )
    monkeypatch.setattr(workflows_db_mod, "_utcnow_iso", lambda: next(timestamps))
    db.create_step_run(
        step_run_id="wf-run-step-runs-1:s2:1",
        tenant_id="tenant",
        run_id="wf-run-step-runs-1",
        step_id="s2",
        name="Second step",
        step_type="log",
        status="running",
        inputs={"config": {"message": "later"}},
    )
    db.create_step_run(
        step_run_id="wf-run-step-runs-1:s1:1",
        tenant_id="tenant",
        run_id="wf-run-step-runs-1",
        step_id="s1",
        name="First step",
        step_type="prompt",
        status="succeeded",
        inputs={"config": {"template": "Hello"}},
    )

    step_runs = db.list_step_runs(run_id="wf-run-step-runs-1")
    assert [step_run["step_run_id"] for step_run in step_runs] == [
        "wf-run-step-runs-1:s2:1",
        "wf-run-step-runs-1:s1:1",
    ]
    assert step_runs[0]["step_id"] == "s2"
    assert step_runs[1]["step_id"] == "s1"


def test_workflow_step_attempt_requires_parent_step_run(tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-attempt-parent",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "step-attempt-parent", "steps": []},
    )

    with pytest.raises(ValueError, match="step_run_id is required"):
        db.create_step_attempt(
            tenant_id="tenant",
            run_id="wf-run-attempt-parent",
            step_run_id="",
            step_id="s1",
            attempt_number=1,
            status="running",
            metadata={"step_type": "prompt"},
        )


def test_workflow_step_attempt_rejects_duplicate_logical_attempt(tmp_path):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-attempt-duplicate",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "step-attempt-duplicate", "steps": []},
    )
    db.create_step_run(
        step_run_id="wf-run-attempt-duplicate:s1:1",
        tenant_id="tenant",
        run_id="wf-run-attempt-duplicate",
        step_id="s1",
        name="Prompt step",
        step_type="prompt",
        status="running",
        inputs={"config": {"template": "Hello"}},
    )

    db.create_step_attempt(
        tenant_id="tenant",
        run_id="wf-run-attempt-duplicate",
        step_run_id="wf-run-attempt-duplicate:s1:1",
        step_id="s1",
        attempt_number=1,
        status="running",
        metadata={"step_type": "prompt"},
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.create_step_attempt(
            tenant_id="tenant",
            run_id="wf-run-attempt-duplicate",
            step_run_id="wf-run-attempt-duplicate:s1:1",
            step_id="s1",
            attempt_number=1,
            status="running",
            metadata={"step_type": "prompt"},
        )


def test_backend_v7_migration_does_not_create_step_attempts_table() -> None:
    db = WorkflowsDatabase.__new__(WorkflowsDatabase)
    queries: list[str] = []

    class _Backend:
        @staticmethod
        def escape_identifier(identifier: str) -> str:
            return f'"{identifier}"'

        def execute(self, query: str, params=None, connection=None):  # noqa: ANN001
            queries.append(query)

    db.backend = _Backend()

    db._backend_migrate_to_v7(object())

    assert not any("workflow_step_attempts" in query for query in queries)
    assert any("workflow_research_waits" in query for query in queries)


def test_list_step_attempts_logs_malformed_metadata_json(tmp_path, monkeypatch):
    db_path = tmp_path / "workflows.db"
    db = WorkflowsDatabase(str(db_path))

    db.create_run(
        run_id="wf-run-bad-metadata",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "bad-metadata", "steps": []},
    )
    db.create_step_run(
        step_run_id="wf-run-bad-metadata:s1:1",
        tenant_id="tenant",
        run_id="wf-run-bad-metadata",
        step_id="s1",
        name="Prompt step",
        step_type="prompt",
        status="running",
        inputs={},
    )
    attempt_id = db.create_step_attempt(
        tenant_id="tenant",
        run_id="wf-run-bad-metadata",
        step_run_id="wf-run-bad-metadata:s1:1",
        step_id="s1",
        attempt_number=1,
        status="running",
        metadata={"step_type": "prompt"},
    )
    db._conn.execute(
        "UPDATE workflow_step_attempts SET metadata_json = ? WHERE attempt_id = ?",
        ("{not-json", attempt_id),
    )
    db._conn.commit()

    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(workflows_db_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args))

    attempts = db.list_step_attempts(run_id="wf-run-bad-metadata", step_id="s1")

    assert attempts[0]["metadata_json"] == {}
    assert warnings
    assert "metadata_json" in str(warnings[0][0])


def test_workflow_step_attempt_migration_rebuilds_legacy_contract(tmp_path):
    db_path = tmp_path / "workflows_legacy.db"
    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.executescript(
        """
        CREATE TABLE workflow_schema_version (version INTEGER NOT NULL);
        INSERT INTO workflow_schema_version(version) VALUES (7);

        CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            workflow_id INTEGER,
            status TEXT NOT NULL,
            status_reason TEXT,
            user_id TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            outputs_json TEXT,
            error TEXT,
            duration_ms INTEGER,
            created_at TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            definition_version INTEGER,
            definition_snapshot_json TEXT,
            idempotency_key TEXT,
            session_id TEXT,
            validation_mode TEXT DEFAULT 'block',
            tokens_input INTEGER,
            tokens_output INTEGER,
            cost_usd REAL,
            cancel_requested INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE workflow_step_runs (
            step_run_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            name TEXT,
            type TEXT,
            status TEXT,
            attempt INTEGER DEFAULT 0,
            started_at TEXT,
            ended_at TEXT,
            inputs_json TEXT,
            outputs_json TEXT,
            error TEXT,
            decision TEXT,
            assigned_to TEXT,
            approved_by TEXT,
            approved_at TEXT,
            review_comment TEXT,
            locked_by TEXT,
            locked_at TEXT,
            lock_expires_at TEXT,
            heartbeat_at TEXT,
            pid INTEGER,
            pgid INTEGER,
            workdir TEXT,
            stdout_path TEXT,
            stderr_path TEXT
        );

        INSERT INTO workflow_runs(
            run_id, tenant_id, workflow_id, status, status_reason, user_id, inputs_json, outputs_json, error,
            duration_ms, created_at, started_at, ended_at, definition_version, definition_snapshot_json,
            idempotency_key, session_id, validation_mode, tokens_input, tokens_output, cost_usd, cancel_requested
        ) VALUES (
            'run-legacy',
            'tenant',
            NULL,
            'running',
            NULL,
            'user',
            '{}',
            NULL,
            NULL,
            NULL,
            '2026-01-01T00:00:00+00:00',
            NULL,
            NULL,
            1,
            '{}',
            NULL,
            NULL,
            'block',
            NULL,
            NULL,
            NULL,
            0
        );

        INSERT INTO workflow_step_runs(
            step_run_id, tenant_id, run_id, step_id, name, type, status, attempt, started_at, ended_at, inputs_json,
            outputs_json, error, decision, assigned_to, approved_by, approved_at, review_comment, locked_by,
            locked_at, lock_expires_at, heartbeat_at, pid, pgid, workdir, stdout_path, stderr_path
        ) VALUES (
            'run-legacy:s1:1',
            'tenant',
            'run-legacy',
            's1',
            'Legacy Step',
            'prompt',
            'running',
            1,
            '2026-01-01T00:00:00+00:00',
            NULL,
            '{}',
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL
        );

        INSERT INTO workflow_step_runs(
            step_run_id, tenant_id, run_id, step_id, name, type, status, attempt, started_at, ended_at, inputs_json,
            outputs_json, error, decision, assigned_to, approved_by, approved_at, review_comment, locked_by,
            locked_at, lock_expires_at, heartbeat_at, pid, pgid, workdir, stdout_path, stderr_path
        ) VALUES (
            'run-legacy:s2:1',
            'tenant',
            'run-legacy',
            's2',
            'Legacy Step 2',
            'prompt',
            'running',
            1,
            '2026-01-01T00:00:00+00:00',
            NULL,
            '{}',
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL
        );

        CREATE TABLE workflow_step_attempts (
            attempt_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            step_run_id TEXT,
            step_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            metadata_json TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT
        );

        INSERT INTO workflow_step_attempts(
            attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, metadata_json, started_at
        ) VALUES (
            'legacy-orphan',
            'tenant',
            'run-legacy',
            NULL,
            's1',
            1,
            'running',
            '{}',
            '2026-01-01T00:00:00+00:00'
        );

        INSERT INTO workflow_step_attempts(
            attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, metadata_json, started_at, ended_at
        ) VALUES (
            'legacy-duplicate-older',
            'tenant',
            'run-legacy',
            'run-legacy:s1:1',
            's1',
            1,
            'running',
            '{"source":"older"}',
            '2026-01-01T00:00:00+00:00',
            NULL
        );

        INSERT INTO workflow_step_attempts(
            attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, metadata_json, started_at, ended_at
        ) VALUES (
            'legacy-duplicate-newer',
            'tenant',
            'run-legacy',
            'run-legacy:s1:1',
            's1',
            1,
            'failed',
            '{"source":"newer"}',
            '2026-01-01T00:00:10+00:00',
            '2026-01-01T00:00:20+00:00'
        );

        INSERT INTO workflow_step_attempts(
            attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, metadata_json, started_at, ended_at
        ) VALUES (
            'legacy-valid-survivor',
            'tenant',
            'run-legacy',
            'run-legacy:s2:1',
            's2',
            1,
            'succeeded',
            '{"source":"survivor"}',
            '2026-01-01T00:01:00+00:00',
            '2026-01-01T00:01:05+00:00'
        );

        INSERT INTO workflow_step_attempts(
            attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, metadata_json, started_at, ended_at
        ) VALUES (
            'legacy-bad-parent',
            'tenant',
            'run-legacy',
            'missing-step-run',
            's9',
            1,
            'failed',
            '{"source":"bad-parent"}',
            '2026-01-01T00:02:00+00:00',
            '2026-01-01T00:02:01+00:00'
        );
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    db = WorkflowsDatabase(str(db_path))

    migrated_conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]: row for row in migrated_conn.execute("PRAGMA table_info(workflow_step_attempts)").fetchall()
        }
        assert columns["step_run_id"][3] == 1

        indexes = migrated_conn.execute("PRAGMA index_list(workflow_step_attempts)").fetchall()
        unique_indexes = [row for row in indexes if row[2] == 1]
        assert unique_indexes
        index_names = {row[1] for row in indexes}
        assert "idx_step_attempts_run_attempts" in index_names
        assert "idx_step_attempts_run_step_attempts" in index_names
        assert "idx_step_attempts_step_run_attempts" in index_names

        version_row = migrated_conn.execute(
            "SELECT version FROM workflow_schema_version LIMIT 1"
        ).fetchone()
        assert version_row is not None
        assert int(version_row[0]) == WorkflowsDatabase._CURRENT_SCHEMA_VERSION

        rows = migrated_conn.execute(
            "SELECT attempt_id, step_run_id, attempt_number, status, metadata_json "
            "FROM workflow_step_attempts ORDER BY step_run_id ASC"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "legacy-duplicate-newer"
        assert rows[0][1] == "run-legacy:s1:1"
        assert rows[0][2] == 1
        assert rows[0][3] == "failed"
        assert rows[1][0] == "legacy-valid-survivor"
        assert rows[1][1] == "run-legacy:s2:1"
        assert all(row[0] != "legacy-bad-parent" for row in rows)
    finally:
        migrated_conn.close()

    db.create_run(
        run_id="wf-run-attempt-migrated",
        tenant_id="tenant",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "step-attempt-migrated", "steps": []},
    )
    db.create_step_run(
        step_run_id="wf-run-attempt-migrated:s1:1",
        tenant_id="tenant",
        run_id="wf-run-attempt-migrated",
        step_id="s1",
        name="Prompt step",
        step_type="prompt",
        status="running",
        inputs={"config": {"template": "Hello"}},
    )
    db.create_step_attempt(
        tenant_id="tenant",
        run_id="wf-run-attempt-migrated",
        step_run_id="wf-run-attempt-migrated:s1:1",
        step_id="s1",
        attempt_number=1,
        status="running",
        metadata={"step_type": "prompt"},
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.create_step_attempt(
            tenant_id="tenant",
            run_id="wf-run-attempt-migrated",
            step_run_id="wf-run-attempt-migrated:s1:1",
            step_id="s1",
            attempt_number=1,
            status="running",
            metadata={"step_type": "prompt"},
        )
