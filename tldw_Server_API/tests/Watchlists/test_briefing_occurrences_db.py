from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management import Watchlists_DB as watchlists_module
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

pytestmark = pytest.mark.unit


def _make_backend(tmp_path):
    return DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "briefing_occurrences.db"),
        )
    )


def _make_db(tmp_path, *, user_id: int = 1) -> WatchlistsDatabase:
    return WatchlistsDatabase(user_id=user_id, backend=_make_backend(tmp_path))


def _create_job_run(db: WatchlistsDatabase, *, label: str) -> tuple[int, int]:
    job = db.create_job(
        name=f"{label} job",
        description=None,
        scope_json=json.dumps({}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=0,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
        job_filters_json=None,
    )
    run = db.create_run(int(job.id), status="finished")
    return int(job.id), int(run.id)


def test_sqlite_schema_contains_briefing_occurrence_contract_and_indexes(tmp_path):
    db = _make_db(tmp_path)

    columns = {row["name"] for row in db.backend.get_table_info("watchlist_briefing_occurrences")}
    indexes = {
        row["name"]
        for row in db.backend.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
            ("watchlist_briefing_occurrences",),
        ).rows
    }

    assert columns == {
        "id",
        "user_id",
        "job_id",
        "run_id",
        "occurrence_key",
        "contract_json",
        "stages_json",
        "artifact_status",
        "delivery_status",
        "output_id",
        "audio_task_id",
        "delivery_task_id",
        "selected_count",
        "omitted_count",
        "created_at",
        "updated_at",
    }
    assert {
        "idx_briefing_occurrences_user_job",
        "idx_briefing_occurrences_run",
    }.issubset(indexes)


def test_create_or_get_occurrence_is_idempotent_and_preserves_initial_contract(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="daily")
    occurrence_key = f"user:1:job:{job_id}:run:{run_id}:v1"

    first = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=occurrence_key,
        contract_json='{"version":1}',
    )
    second = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=occurrence_key,
        contract_json='{"version":2}',
    )

    assert isinstance(first, watchlists_module.BriefingOccurrenceRow)
    assert second.id == first.id
    assert first.user_id == "1"
    assert first.job_id == job_id
    assert first.run_id == run_id
    assert first.contract_json == '{"version":1}'
    assert first.stages_json == "{}"
    assert first.artifact_status == "running"
    assert first.delivery_status == "waiting_for_artifacts"
    assert first.output_id is None
    assert first.audio_task_id is None
    assert first.delivery_task_id is None
    assert first.selected_count == 0
    assert first.omitted_count == 0
    assert second.contract_json == first.contract_json
    assert second.created_at == first.created_at
    assert db.backend.execute("SELECT COUNT(*) AS count FROM watchlist_briefing_occurrences").scalar == 1


def test_concurrent_create_or_get_occurrence_returns_one_logical_row(tmp_path):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="race")
    occurrence_key = f"user:1:job:{job_id}:run:{run_id}:v1"
    barrier = Barrier(6)

    def create() -> int:
        barrier.wait()
        row = db.create_or_get_briefing_occurrence(
            run_id=run_id,
            occurrence_key=occurrence_key,
            contract_json='{"version":1}',
        )
        return int(row.id)

    with ThreadPoolExecutor(max_workers=6) as pool:
        occurrence_ids = list(pool.map(lambda _: create(), range(6)))

    assert len(set(occurrence_ids)) == 1
    assert db.backend.execute("SELECT COUNT(*) AS count FROM watchlist_briefing_occurrences").scalar == 1


def test_occurrence_create_read_and_update_are_scoped_to_owned_run(tmp_path):
    backend = _make_backend(tmp_path)
    owner = WatchlistsDatabase(user_id=1, backend=backend)
    outsider = WatchlistsDatabase(user_id=2, backend=backend)
    owner_job_id, owner_run_id = _create_job_run(owner, label="owner")
    outsider_job_id, outsider_run_id = _create_job_run(outsider, label="outsider")
    occurrence_key = "shared-logical-key"
    owner_occurrence = owner.create_or_get_briefing_occurrence(
        run_id=owner_run_id,
        occurrence_key=occurrence_key,
        contract_json='{"owner":1}',
    )
    outsider_occurrence = outsider.create_or_get_briefing_occurrence(
        run_id=outsider_run_id,
        occurrence_key=occurrence_key,
        contract_json='{"owner":2}',
    )

    assert owner_occurrence.id != outsider_occurrence.id
    assert owner_occurrence.job_id == owner_job_id
    assert outsider_occurrence.job_id == outsider_job_id
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.get_briefing_occurrence(int(owner_occurrence.id))
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.update_briefing_occurrence(
            int(owner_occurrence.id),
            artifact_status="failed",
        )
    with pytest.raises(KeyError, match="run_not_found"):
        outsider.create_or_get_briefing_occurrence(
            run_id=owner_run_id,
            occurrence_key=occurrence_key,
            contract_json='{"owner":2}',
        )
    assert owner.get_briefing_occurrence(int(owner_occurrence.id)).artifact_status == "running"


def test_get_latest_occurrence_is_job_and_user_scoped(tmp_path):
    backend = _make_backend(tmp_path)
    owner = WatchlistsDatabase(user_id=1, backend=backend)
    outsider = WatchlistsDatabase(user_id=2, backend=backend)
    job_id, first_run_id = _create_job_run(owner, label="latest")
    second_run = owner.create_run(job_id, status="finished")
    first = owner.create_or_get_briefing_occurrence(
        run_id=first_run_id,
        occurrence_key="latest:first",
        contract_json='{"version":1}',
    )
    second = owner.create_or_get_briefing_occurrence(
        run_id=int(second_run.id),
        occurrence_key="latest:second",
        contract_json='{"version":1}',
    )

    assert owner.get_latest_briefing_occurrence(job_id).id == second.id
    assert owner.get_latest_briefing_occurrence(job_id).id != first.id
    with pytest.raises(KeyError, match="briefing_occurrence_not_found"):
        outsider.get_latest_briefing_occurrence(job_id)


def test_update_occurrence_serializes_stages_and_only_changes_named_fields(tmp_path, monkeypatch):
    db = _make_db(tmp_path)
    job_id, run_id = _create_job_run(db, label="update")
    occurrence = db.create_or_get_briefing_occurrence(
        run_id=run_id,
        occurrence_key=f"user:1:job:{job_id}:run:{run_id}:v1",
        contract_json='{"version":1}',
    )
    stages = {
        "collect": {"status": "ready"},
        "persist_text": {"status": "ready", "output_id": 901},
    }
    updated_at = "2026-07-10T12:34:56+00:00"
    monkeypatch.setattr(watchlists_module, "_utcnow_iso", lambda: updated_at)

    updated = db.update_briefing_occurrence(
        int(occurrence.id),
        stages=stages,
        artifact_status="ready",
        delivery_status="delivered",
        output_id=901,
        audio_task_id="audio-123",
        delivery_task_id="delivery-456",
        selected_count=4,
        omitted_count=2,
    )

    assert json.loads(updated.stages_json) == stages
    assert updated.artifact_status == "ready"
    assert updated.delivery_status == "delivered"
    assert updated.output_id == 901
    assert updated.audio_task_id == "audio-123"
    assert updated.delivery_task_id == "delivery-456"
    assert updated.selected_count == 4
    assert updated.omitted_count == 2
    assert updated.contract_json == occurrence.contract_json
    assert updated.created_at == occurrence.created_at
    assert updated.updated_at == updated_at
    with pytest.raises(TypeError):
        db.update_briefing_occurrence(  # type: ignore[call-arg]
            int(occurrence.id),
            contract_json='{"version":2}',
        )


class _CapturingPostgresBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self) -> None:
        self.ddl = ""
        self.executed: list[str] = []

    def create_tables(self, ddl: str) -> None:
        self.ddl = ddl

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append(query)

    def get_table_info(self, table_name: str) -> list[dict[str, Any]]:
        existing_columns = {
            "scrape_jobs": {"wf_schedule_id", "job_filters_json", "watchlist_id"},
            "sources": {"defer_until", "consec_not_modified", "consec_errors"},
            "scrape_run_items": {"source_id"},
            "scraped_items": {"content", "queued_for_briefing"},
        }
        return [{"name": name} for name in existing_columns.get(table_name, set())]


def test_postgres_schema_includes_briefing_occurrence_contract():
    backend = _CapturingPostgresBackend()

    WatchlistsDatabase(user_id=1, backend=backend)  # type: ignore[arg-type]

    occurrence_table_marker = "CREATE TABLE IF NOT EXISTS watchlist_briefing_occurrences"
    assert occurrence_table_marker in backend.ddl
    occurrence_ddl = backend.ddl.partition(occurrence_table_marker)[2].partition(";")[0]
    assert "id BIGSERIAL PRIMARY KEY" in occurrence_ddl
    assert "job_id BIGINT NOT NULL" in occurrence_ddl
    assert "run_id BIGINT NOT NULL" in occurrence_ddl
    assert "output_id BIGINT" in occurrence_ddl
    assert "UNIQUE (user_id, occurrence_key)" in occurrence_ddl
    assert "CREATE INDEX IF NOT EXISTS idx_briefing_occurrences_user_job" in backend.ddl
    assert "CREATE INDEX IF NOT EXISTS idx_briefing_occurrences_run" in backend.ddl
