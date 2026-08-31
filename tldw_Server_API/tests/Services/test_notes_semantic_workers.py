"""Worker, recovery cadence, kill-switch, and shutdown contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Notes_Graph.semantic_jobs import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import NotesSemanticVectorStore
from tldw_Server_API.app.services import (
    notes_semantic_index_worker,
    notes_semantic_maintenance,
)
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext
from tldw_Server_API.app.services.startup_study_privilege_jobs_pollers import (
    provide_study_privilege_jobs_worker_specs,
)

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


def _context(*, sidecar_mode: bool) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=sidecar_mode,
    )


def test_worker_config_is_exact_and_disables_unbounded_sdk_retries() -> None:
    config = notes_semantic_index_worker.build_worker_config(worker_id="semantic-1")

    assert config.domain == JOB_DOMAIN == "notes"
    assert config.queue == JOB_QUEUE
    assert config.retry_on_exception is False
    assert config.bind_completion_token is True


@pytest.mark.asyncio
async def test_app_managed_and_standalone_entrypoints_share_one_handler(monkeypatch) -> None:
    seen: list[object] = []

    async def fake_runner(*, stop_event, handler):
        seen.append((stop_event, handler))

    monkeypatch.setattr(notes_semantic_index_worker, "_run_worker", fake_runner)
    first = asyncio.Event()
    second = asyncio.Event()

    await notes_semantic_index_worker.run_notes_semantic_index_worker(first)
    await notes_semantic_index_worker.run_standalone_notes_semantic_index_worker(second)

    assert seen == [
        (first, notes_semantic_index_worker.handle_notes_semantic_index_job),
        (second, notes_semantic_index_worker.handle_notes_semantic_index_job),
    ]


def test_startup_flags_cannot_create_duplicate_app_and_sidecar_ownership(monkeypatch) -> None:
    monkeypatch.setenv("NOTES_SEMANTIC_INDEX_WORKER_ENABLED", "true")
    monkeypatch.setenv("NOTES_SEMANTIC_MAINTENANCE_ENABLED", "true")
    specs = {spec.name: spec for spec in provide_study_privilege_jobs_worker_specs()}

    for name in (
        "notes_semantic_index_jobs_task",
        "notes_semantic_maintenance_task",
    ):
        assert specs[name].enabled(_context(sidecar_mode=False)) is True
        assert specs[name].enabled(_context(sidecar_mode=True)) is False


class _Scope:
    def __init__(self, name: str, *, dirty: int, failed: int, cleanup: int) -> None:
        self.name = name
        self.dirty = dirty
        self.failed = failed
        self.cleanup = cleanup
        self.calls: list[tuple[str, int]] = []
        self.admitted: list[tuple[str, object]] = []

    def reclaim_expired(self, *, limit: int, now):
        self.calls.append(("reclaim", limit))
        return min(limit, 2)

    def claim_dirty(self, *, limit: int, now):
        self.calls.append(("dirty", limit))
        return tuple(
            SimpleNamespace(
                owner_user_id="owner-a",
                dataset_id=self.name,
                generation_id="generation-a",
                dirty_generation=index,
            )
            for index in range(min(limit, self.dirty))
        )

    def claim_failed(self, *, limit: int, now):
        self.calls.append(("failed", limit))
        return tuple(f"failed-{index}" for index in range(min(limit, self.failed)))

    def claim_cleanup(self, *, limit: int, now):
        self.calls.append(("cleanup", limit))
        return tuple(f"cleanup-{index}" for index in range(min(limit, self.cleanup)))

    def admit(self, *, mode: str, claim):
        self.admitted.append((mode, claim))
        return True

    async def cleanup_claim(self, claim):
        self.admitted.append(("cleanup", claim))
        return True


@pytest.mark.asyncio
async def test_maintenance_shares_one_bounded_claim_budget_and_coalesces_dirty_work() -> None:
    first = _Scope("dataset-a", dirty=80, failed=20, cleanup=10)
    second = _Scope("dataset-b", dirty=80, failed=20, cleanup=10)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(first, second),
        indexing_enabled=True,
    )

    result = await coordinator.run_pass(now=NOW, limit=100)

    assert result.claimed <= 100
    assert result.dirty_admitted <= 100
    dirty_keys = {
        (
            claim.owner_user_id,
            claim.dataset_id,
            claim.generation_id,
            claim.dirty_generation,
        )
        for scope in (first, second)
        for mode, claim in scope.admitted
        if mode == "maintain"
    }
    assert len(dirty_keys) == result.dirty_admitted
    assert first.calls[0] == ("reclaim", 100)


@pytest.mark.asyncio
async def test_failed_notes_retry_separately_and_cleanup_requires_confirmation() -> None:
    scope = _Scope("dataset-a", dirty=0, failed=2, cleanup=2)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(scope,),
        indexing_enabled=True,
    )

    result = await coordinator.run_pass(now=NOW, limit=10)

    assert result.failed_retries == 2
    assert result.cleanup_confirmed == 2
    assert [mode for mode, _claim in scope.admitted].count("retry_failed") == 2
    assert [mode for mode, _claim in scope.admitted].count("cleanup") == 2


@pytest.mark.asyncio
async def test_kill_switch_blocks_index_admission_but_keeps_cleanup_available() -> None:
    scope = _Scope("dataset-a", dirty=5, failed=3, cleanup=2)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(scope,),
        indexing_enabled=False,
    )

    result = await coordinator.run_pass(now=NOW, limit=20)

    assert result.dirty_admitted == 0
    assert result.failed_retries == 0
    assert result.cleanup_confirmed == 2
    assert all(mode == "cleanup" for mode, _claim in scope.admitted)
    assert not any(name in {"dirty", "failed"} for name, _limit in scope.calls)


@pytest.mark.asyncio
async def test_unresolved_generation_cleanup_confirms_no_storage_without_backend_io() -> None:
    class Authority:
        owner_user_id = "owner-a"

        def get_generation(self, dataset_id, generation_id):
            return SimpleNamespace(
                owner_user_id="owner-a",
                dataset_id=dataset_id,
                id=generation_id,
                dimension_state="pending",
                dimensions=None,
            )

    class Backend:
        def supports_dimensions(self, _dimensions):
            raise AssertionError("unresolved cleanup must not inspect backend dimensions")

        async def delete_generation(self, _binding):
            raise AssertionError("unresolved generation never created physical storage")

    vectors = NotesSemanticVectorStore(authority=Authority(), backend=Backend())

    result = await vectors.delete_generation("dataset-a", "generation-a")

    assert result.confirmed_absent is True


@pytest.mark.asyncio
async def test_maintenance_loop_drains_cleanly_on_shutdown() -> None:
    stop = asyncio.Event()
    calls: list[str] = []

    class Runner:
        async def run_pass(self, *, now, limit):
            del now, limit
            calls.append("pass")
            stop.set()

    await notes_semantic_maintenance.run_maintenance_loop(
        Runner(),
        stop,
        interval_seconds=60,
        now=lambda: NOW,
    )

    assert calls == ["pass"]


@pytest.mark.asyncio
async def test_worker_stop_event_stops_sdk_and_awaits_watcher(monkeypatch) -> None:
    calls: list[str] = []

    class SDK:
        def stop(self):
            calls.append("stop")

        async def run(self, **kwargs):
            assert kwargs["job_type"] == JOB_TYPE
            stop.set()
            await asyncio.sleep(0)

    stop = asyncio.Event()
    monkeypatch.setattr(notes_semantic_index_worker, "_build_sdk", lambda **_kwargs: SDK())
    await notes_semantic_index_worker._run_worker(
        stop_event=stop,
        handler=notes_semantic_index_worker.handle_notes_semantic_index_job,
    )

    assert calls == ["stop"]
