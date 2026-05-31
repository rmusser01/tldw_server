from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Scheduler.handlers import workflows as workflow_handler_mod


pytestmark = pytest.mark.unit


def test_create_run_persists_metadata_json(tmp_path):
    db = WorkflowsDatabase(db_path=str(tmp_path / "workflows.db"))
    metadata = {
        "source": "watchlist_audio_briefing",
        "watchlist_run_id": 7,
        "watchlist_job_id": 3,
        "audio_request_id": "wla_test_1",
    }

    db.create_run(
        run_id="wf_audio_1",
        tenant_id="default",
        user_id="1",
        inputs={"items": []},
        definition_snapshot={"name": "audio_briefing", "steps": []},
        idempotency_key="watchlist-audio-briefing:1:3:7:wla_test_1",
        metadata=metadata,
    )

    run = db.get_run("wf_audio_1")
    assert run is not None
    assert run.metadata_json is not None
    assert json.loads(run.metadata_json) == metadata

    listed_run = db.list_runs(tenant_id="default", user_id="1")[0]
    assert json.loads(listed_run.metadata_json or "{}")["audio_request_id"] == "wla_test_1"

    idempotent_run = db.get_run_by_idempotency(
        "default",
        "1",
        "watchlist-audio-briefing:1:3:7:wla_test_1",
    )
    assert idempotent_run is not None
    assert json.loads(idempotent_run.metadata_json or "{}")["watchlist_run_id"] == 7


def test_existing_sqlite_workflow_runs_schema_migrates_metadata_json(tmp_path):
    db_path = tmp_path / "legacy-workflows.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE workflow_schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO workflow_schema_version (version) VALUES (8)")
    conn.execute(
        """
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
        )
        """
    )
    conn.commit()
    conn.close()

    db = WorkflowsDatabase(db_path=str(db_path))
    columns = {
        row["name"]
        for row in db._conn.execute("PRAGMA table_info(workflow_runs)").fetchall()
    }
    assert "metadata_json" in columns

    db.create_run(
        run_id="wf_audio_migrated",
        tenant_id="default",
        user_id="1",
        inputs={},
        definition_snapshot={"name": "audio_briefing", "steps": []},
        metadata={"audio_request_id": "wla_after_migration"},
    )

    run = db.get_run("wf_audio_migrated")
    assert run is not None
    assert json.loads(run.metadata_json or "{}")["audio_request_id"] == "wla_after_migration"


def test_sqlite_metadata_migration_reraises_non_duplicate_errors():
    db = WorkflowsDatabase.__new__(WorkflowsDatabase)

    class FailingCursor:
        @staticmethod
        def execute(sql: str) -> None:
            raise sqlite3.OperationalError("disk I/O error")

    class FailingConnection:
        rollback_called = False

        @staticmethod
        def cursor() -> FailingCursor:
            return FailingCursor()

        @staticmethod
        def commit() -> None:
            raise AssertionError("commit should not run after failed migration")

        def rollback(self) -> None:
            self.rollback_called = True

    connection = FailingConnection()
    db._conn = connection

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        db._sqlite_migrate_to_v9()
    assert connection.rollback_called is True


@pytest.mark.asyncio
async def test_workflow_run_handler_persists_payload_metadata_without_mutating_definition(monkeypatch):
    created_runs: list[dict[str, Any]] = []
    submitted: list[tuple[str, Any]] = []
    ledger_records: list[tuple[str, str]] = []

    class FakeWorkflowsDB:
        def create_run(self, **kwargs: Any) -> None:
            created_runs.append(kwargs)

        def get_run(self, run_id: str) -> Any | None:  # pragma: no cover - sync mode not used
            return None

    class FakeWorkflowEngine:
        def __init__(self, *, db: Any) -> None:
            self.db = db

        def submit(self, run_id: str, mode: Any) -> None:
            submitted.append((run_id, mode))

        @staticmethod
        def set_run_secrets(run_id: str, secrets: dict[str, str]) -> None:  # pragma: no cover - not used
            raise AssertionError("secrets should not be set in this test")

    async def fake_record_workflow_run(*, entity_scope: str, entity_value: str, run_id: str, units: int) -> None:
        ledger_records.append((entity_value, run_id))

    fake_db = FakeWorkflowsDB()
    monkeypatch.setattr(workflow_handler_mod, "_get_wf_db", lambda: fake_db)
    monkeypatch.setattr(workflow_handler_mod, "WorkflowEngine", FakeWorkflowEngine)
    monkeypatch.setattr(workflow_handler_mod, "record_workflow_run", fake_record_workflow_run)

    definition_snapshot = {
        "name": "audio_briefing",
        "metadata": {"existing": "kept"},
        "steps": [],
    }
    metadata = {
        "source": "watchlist_audio_briefing",
        "watchlist_job_id": 3,
        "watchlist_run_id": 7,
        "audio_request_id": "wla_test_1",
    }

    result = await workflow_handler_mod.workflow_run(
        {
            "user_id": "42",
            "definition_snapshot": definition_snapshot,
            "inputs": {"items": []},
            "metadata": metadata,
            "mode": "async",
        }
    )

    assert result == {"run_id": created_runs[0]["run_id"], "status": "queued"}
    assert created_runs[0]["metadata"] == metadata
    assert created_runs[0]["definition_snapshot"] is not definition_snapshot
    assert created_runs[0]["definition_snapshot"]["metadata"] == {
        "existing": "kept",
        **metadata,
    }
    assert definition_snapshot["metadata"] == {"existing": "kept"}
    assert submitted == [(created_runs[0]["run_id"], workflow_handler_mod.RunMode.ASYNC)]
    assert ledger_records == [("42", created_runs[0]["run_id"])]


@pytest.mark.asyncio
async def test_workflow_run_handler_rejects_non_dict_metadata(monkeypatch):
    class FakeWorkflowsDB:
        def create_run(self, **kwargs: Any) -> None:
            raise AssertionError("invalid metadata should fail before create_run")

    monkeypatch.setattr(workflow_handler_mod, "_get_wf_db", lambda: FakeWorkflowsDB())

    with pytest.raises(ValueError, match="metadata must be a dict"):
        await workflow_handler_mod.workflow_run(
            {
                "user_id": "42",
                "definition_snapshot": {"name": "audio_briefing", "steps": []},
                "inputs": {"items": []},
                "metadata": "not-a-dict",
                "mode": "async",
            }
        )
