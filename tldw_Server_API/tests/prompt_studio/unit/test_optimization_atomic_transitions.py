from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
    _BackendPromptStudioDatabase,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    optimization_engine as optimization_engine_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import (
    JobProcessor,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import (
    OptimizationEngine,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)


@pytest.fixture
def optimization_db(
    tmp_path: Path,
) -> Iterator[tuple[PromptStudioDatabase, int]]:
    db = PromptStudioDatabase(tmp_path / "atomic-optimization.sqlite", "atomic-test")
    project = db.create_project("Atomic optimization", user_id="owner-1")
    try:
        yield db, int(project["id"])
    finally:
        db.close()


def _running_optimization(db: PromptStudioDatabase, project_id: int) -> int:
    optimization = db.create_optimization(
        project_id=project_id,
        name="Atomic transition",
        initial_prompt_id=None,
        optimizer_type="iterative",
        status="running",
    )
    return int(optimization["id"])


def test_optimization_update_expected_uuid_is_atomic_across_backends(
    prompt_studio_dual_backend_db: tuple[str, PromptStudioDatabase],
) -> None:
    _backend, db = prompt_studio_dual_backend_db
    project = db.create_project("Optimization UUID CAS", user_id="owner-1")
    optimization = db.create_optimization(
        project_id=int(project["id"]),
        name="Optimization UUID CAS",
        initial_prompt_id=None,
        optimizer_type="iterative",
        status="completed",
    )

    row, applied = db.update_optimization(
        int(optimization["id"]),
        {"status": "cancelled", "error_message": "stale callback"},
        expected_statuses=("completed",),
        expected_uuid="stale-optimization-uuid",
        _return_transition_applied=True,
    )

    assert applied is False
    assert row["uuid"] == optimization["uuid"]
    assert row["status"] == "completed"
    assert row.get("error_message") is None


def test_sqlite_cancel_committed_during_finalization_cannot_be_revived(
    optimization_db: tuple[PromptStudioDatabase, int],
) -> None:
    db, project_id = optimization_db
    optimization_id = _running_optimization(db, project_id)

    cancelled = db.set_optimization_status(
        optimization_id,
        "cancelled",
        error_message="user cancelled",
        mark_completed=True,
    )
    db.complete_optimization(
        optimization_id,
        iterations_completed=3,
        final_metrics={"accuracy": 1.0},
    )

    persisted = db.get_optimization(optimization_id)
    assert persisted is not None
    assert persisted["status"] == "cancelled"
    assert persisted["error_message"] == "user cancelled"
    assert persisted["iterations_completed"] == cancelled["iterations_completed"]
    assert persisted["final_metrics"] == cancelled["final_metrics"]


def test_sqlite_completion_committed_first_cannot_be_overwritten_by_cancel(
    optimization_db: tuple[PromptStudioDatabase, int],
) -> None:
    db, project_id = optimization_db
    optimization_id = _running_optimization(db, project_id)

    completed = db.complete_optimization(
        optimization_id,
        iterations_completed=3,
        final_metrics={"accuracy": 1.0},
    )
    db.set_optimization_status(
        optimization_id,
        "cancelled",
        error_message="late cancellation",
        mark_completed=True,
    )

    persisted = db.get_optimization(optimization_id)
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["error_message"] is None
    assert persisted["final_metrics"] == completed["final_metrics"]


@pytest.mark.asyncio
async def test_engine_retry_returns_persisted_completion_without_dispatch(
    optimization_db: tuple[PromptStudioDatabase, int],
) -> None:
    db, project_id = optimization_db
    prompt = db.create_prompt(
        project_id=project_id,
        name="Completed retry prompt",
        system_prompt="Answer accurately.",
        user_prompt="{question}",
    )
    test_case = db.create_test_case(
        project_id=project_id,
        name="Completed retry case",
        inputs={"question": "two plus two"},
        expected_outputs={"response": "four"},
    )
    optimization = db.create_optimization(
        project_id=project_id,
        name="Completed retry",
        initial_prompt_id=int(prompt["id"]),
        optimizer_type="mipro",
        optimization_config={
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": {
                "provider": "openai",
                "model": "gpt-test",
                "parameters": {},
            },
        },
        status="pending",
    )
    optimization_id = int(optimization["id"])
    db.update_optimization(
        optimization_id,
        {"test_case_ids": [int(test_case["id"])]},
    )
    db.complete_optimization(
        optimization_id,
        optimized_prompt_id=int(prompt["id"]),
        iterations_completed=2,
        final_metrics={"accuracy": 1.0},
    )

    class _MustNotDispatch:
        async def optimize(self, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("provider work must not restart")

    engine = OptimizationEngine(db)
    engine.mipro = _MustNotDispatch()  # type: ignore[assignment]

    result = await engine.optimize(
        optimization_id,
        runtime_model_config={
            "provider": "openai",
            "model": "gpt-test",
            "parameters": {},
        },
    )

    assert result == {
        "optimization_id": optimization_id,
        "optimized_prompt_id": int(prompt["id"]),
        "iterations": 2,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_engine_start_race_returns_concurrent_completion_without_dispatch() -> None:
    optimization_id = 29
    pending = {
        "id": optimization_id,
        "status": "pending",
        "initial_prompt_id": 5,
        "optimized_prompt_id": None,
        "iterations_completed": 0,
        "max_iterations": 1,
        "test_case_ids": [7],
        "optimization_config": {
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": {
                "provider": "openai",
                "model": "gpt-test",
                "parameters": {},
            },
        },
    }
    completed = {
        **pending,
        "status": "completed",
        "optimized_prompt_id": 11,
        "iterations_completed": 3,
    }

    class _RaceDB:
        client_id = "engine-start-race"

        def get_optimization(self, _optimization_id: int) -> dict[str, Any]:
            return dict(pending)

        def set_optimization_status(
            self,
            _optimization_id: int,
            status: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert status == "running"
            return dict(completed)

    class _MustNotDispatch:
        async def optimize(self, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("provider work must not start after completion")

    engine = OptimizationEngine(_RaceDB())  # type: ignore[arg-type]
    engine.mipro = _MustNotDispatch()  # type: ignore[assignment]

    result = await engine.optimize(
        optimization_id,
        runtime_model_config={
            "provider": "openai",
            "model": "gpt-test",
            "parameters": {},
        },
    )

    assert result == {
        "optimization_id": optimization_id,
        "optimized_prompt_id": 11,
        "iterations": 3,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_concurrent_direct_engine_completion_emits_once_for_cas_winner(
    optimization_db: tuple[PromptStudioDatabase, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, project_id = optimization_db
    prompt = db.create_prompt(
        project_id=project_id,
        name="Concurrent completion prompt",
        system_prompt="Answer accurately.",
        user_prompt="{question}",
    )
    test_case = db.create_test_case(
        project_id=project_id,
        name="Concurrent completion case",
        inputs={"question": "two plus two"},
        expected_outputs={"response": "four"},
    )
    optimization = db.create_optimization(
        project_id=project_id,
        name="Concurrent MCTS completion",
        initial_prompt_id=int(prompt["id"]),
        optimizer_type="mcts",
        optimization_config={
            "optimizer_type": "mcts",
            "target_metric": "accuracy",
            "model_config": {
                "provider": "openai",
                "model": "gpt-test",
                "parameters": {},
            },
        },
        max_iterations=1,
        status="pending",
    )
    optimization_id = int(optimization["id"])
    db.update_optimization(
        optimization_id,
        {"test_case_ids": [int(test_case["id"])]},
    )

    both_dispatched = asyncio.Event()
    release = asyncio.Event()
    dispatch_count = 0

    class _RacingMCTS:
        async def optimize(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal dispatch_count
            dispatch_count += 1
            if dispatch_count == 2:
                both_dispatched.set()
            await release.wait()
            return {
                "optimized_prompt_id": int(prompt["id"]),
                "iterations": 1,
                "initial_score": 0.25,
                "final_score": 0.75,
                "improvement": 0.5,
            }

    completion_events: list[int] = []

    class _RecordingBroadcaster:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def broadcast_event(self, **kwargs: Any) -> None:
            completion_events.append(int(kwargs["data"]["optimization_id"]))

    monkeypatch.setattr(
        optimization_engine_module.mcts_optimizer_module,
        "ws_connection_manager",
        object(),
        raising=True,
    )
    monkeypatch.setattr(
        optimization_engine_module.mcts_optimizer_module,
        "EventBroadcaster",
        _RecordingBroadcaster,
        raising=True,
    )
    first = OptimizationEngine(db)
    second = OptimizationEngine(db)
    racing_mcts = _RacingMCTS()
    first.mcts = racing_mcts  # type: ignore[assignment]
    second.mcts = racing_mcts  # type: ignore[assignment]
    runtime_model_config = {
        "provider": "openai",
        "model": "gpt-test",
        "parameters": {},
    }

    tasks = [
        asyncio.create_task(
            engine.optimize(
                optimization_id,
                runtime_model_config=runtime_model_config,
                emit_completion_event=True,
            )
        )
        for engine in (first, second)
    ]
    try:
        await asyncio.wait_for(both_dispatched.wait(), timeout=2)
        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()

    persisted = db.get_optimization(optimization_id) or {}
    assert persisted["status"] == "completed"
    assert completion_events == [optimization_id]


@pytest.mark.asyncio
async def test_current_jobs_lease_uses_prompt_cas_winner_result_and_event(
    optimization_db: tuple[PromptStudioDatabase, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, project_id = optimization_db
    initial_prompt = db.create_prompt(
        project_id=project_id,
        name="Lease race initial prompt",
        system_prompt="Answer accurately.",
        user_prompt="{question}",
    )
    prompt_winner = db.create_prompt(
        project_id=project_id,
        name="Prompt CAS winner",
        system_prompt="Winner.",
        user_prompt="{question}",
    )
    prompt_loser = db.create_prompt(
        project_id=project_id,
        name="Current Jobs lease attempt",
        system_prompt="Loser.",
        user_prompt="{question}",
    )
    test_case = db.create_test_case(
        project_id=project_id,
        name="Lease race case",
        inputs={"question": "two plus two"},
        expected_outputs={"response": "four"},
    )
    durable_config = {
        "optimizer_type": "mcts",
        "target_metric": "accuracy",
        "model_config": {
            "provider": "openai",
            "model": "gpt-test",
            "parameters": {},
        },
    }
    optimization = db.create_optimization(
        project_id=project_id,
        name="Split Prompt and Jobs winner",
        initial_prompt_id=int(initial_prompt["id"]),
        optimizer_type="mcts",
        optimization_config=durable_config,
        max_iterations=1,
        status="pending",
    )
    optimization_id = int(optimization["id"])
    db.update_optimization(
        optimization_id,
        {"test_case_ids": [int(test_case["id"])]},
    )
    payload = {
        "optimization_id": optimization_id,
        "optimization_uuid": str(optimization["uuid"]),
        "initial_prompt_id": int(initial_prompt["id"]),
        "optimizer_type": "mcts",
        "max_iterations": 1,
        "test_case_ids": [int(test_case["id"])],
        "optimization_config": durable_config,
    }
    manager = JobManager(tmp_path / "split-winner-jobs.sqlite")
    created = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=payload,
        owner_user_id="1",
    )
    first_lease = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        lease_seconds=30,
        worker_id="prompt-cas-winner",
    )
    assert first_lease is not None

    raw_results = [
        {
            "optimized_prompt_id": int(prompt_winner["id"]),
            "iterations": 2,
            "initial_score": 0.10,
            "final_score": 0.91,
            "improvement": 0.81,
            "final_metrics": {"score": 0.91, "winner": "prompt-cas"},
            "total_tokens": 111,
            "total_cost": 1.11,
            "scorer_provider_dispatched": False,
        },
        {
            "optimized_prompt_id": int(prompt_loser["id"]),
            "iterations": 7,
            "initial_score": 0.20,
            "final_score": 0.44,
            "improvement": 0.24,
            "final_metrics": {"score": 0.44, "winner": "jobs-lease"},
            "total_tokens": 777,
            "total_cost": 7.77,
            "scorer_provider_dispatched": False,
        },
    ]
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    dispatch_count = 0
    provider_successes = 0

    async def racing_optimize(_self: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal dispatch_count, provider_successes
        index = dispatch_count
        dispatch_count += 1
        entered[index].set()
        await release[index].wait()
        callback = kwargs.get("on_provider_success")
        if callback is not None:
            await callback()
            provider_successes += 1
        return dict(raw_results[index])

    monkeypatch.setattr(
        optimization_engine_module.MCTSOptimizer,
        "optimize",
        racing_optimize,
        raising=True,
    )
    runtime_config = {
        "provider": "openai",
        "model": "gpt-test",
        "parameters": {},
    }
    first_processor = JobProcessor(db)
    second_processor = JobProcessor(db)
    first_task = asyncio.create_task(
        first_processor.process_optimization_job(
            payload,
            optimization_id,
            runtime_model_config=runtime_config,
            on_provider_success=lambda: asyncio.sleep(0),
        )
    )
    try:
        await asyncio.wait_for(entered[0].wait(), timeout=2)
        connection = manager._connect()
        try:
            connection.execute(
                "UPDATE jobs SET leased_until = ? WHERE id = ?",
                ("2000-01-01 00:00:00", int(created["id"])),
            )
            connection.commit()
        finally:
            connection.close()
        current_lease = manager.acquire_next_job(
            domain="prompt_studio",
            queue="default",
            job_type="optimization",
            lease_seconds=30,
            worker_id="current-jobs-lease",
        )
        assert current_lease is not None
        second_task = asyncio.create_task(
            second_processor.process_optimization_job(
                payload,
                optimization_id,
                runtime_model_config=runtime_config,
                on_provider_success=lambda: asyncio.sleep(0),
            )
        )
        await asyncio.wait_for(entered[1].wait(), timeout=2)

        release[0].set()
        first_result = await asyncio.wait_for(first_task, timeout=2)
        release[1].set()
        second_result = await asyncio.wait_for(second_task, timeout=2)
    finally:
        for event in release:
            event.set()
        if not first_task.done():
            first_task.cancel()

    persisted = db.get_optimization(optimization_id) or {}
    expected = {
        "optimization_id": optimization_id,
        "status": "completed",
        "optimized_prompt_id": int(prompt_winner["id"]),
        "best_prompt_id": int(prompt_winner["id"]),
        "iterations": 2,
        "iterations_completed": 2,
        "initial_score": 0.10,
        "final_score": 0.91,
        "best_metric": 0.91,
        "improvement": 0.81,
        "initial_metrics": {"score": 0.10},
        "final_metrics": {"score": 0.91, "winner": "prompt-cas"},
        "total_tokens": 111,
        "total_cost": 1.11,
    }
    assert persisted["optimized_prompt_id"] == expected["optimized_prompt_id"]
    assert persisted["final_metrics"] == expected["final_metrics"]
    assert provider_successes == 2
    assert dispatch_count == 2
    assert second_result["final_score"] == expected["final_score"]
    assert {key: first_result.get(key) for key in expected} == expected
    assert {key: second_result.get(key) for key in expected} == expected
    assert second_result["_scorer_provider_dispatched"] is False

    assert manager.complete_job(
        int(created["id"]),
        result=first_result,
        worker_id="prompt-cas-winner",
        lease_id=str(first_lease["lease_id"]),
        enforce=True,
    ) is False

    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: second_processor,
        raising=True,
    )
    worker_result = await jobs_worker._handle_job(
        current_lease,
        job_manager=manager,
    )
    assert worker_result == expected
    for marker in (
        "provider_dispatches",
        "_provider_dispatches",
        "scorer_provider_dispatched",
        "_scorer_provider_dispatched",
    ):
        assert marker not in worker_result
    assert dispatch_count == 2

    assert manager.complete_job(
        int(created["id"]),
        result=worker_result,
        worker_id="current-jobs-lease",
        lease_id=str(current_lease["lease_id"]),
        enforce=True,
    ) is True
    core_job = manager.get_job(int(created["id"])) or {}
    core_result = core_job.get("result") or {}
    assert core_result == expected
    for marker in (
        "provider_dispatches",
        "_provider_dispatches",
        "scorer_provider_dispatched",
        "_scorer_provider_dispatched",
    ):
        assert marker not in core_result

    completion_events: list[dict[str, Any]] = []

    class _RecordingBroadcaster:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def broadcast_event(self, **kwargs: Any) -> None:
            completion_events.append(dict(kwargs["data"]))

    monkeypatch.setattr(
        optimization_engine_module.mcts_optimizer_module,
        "ws_connection_manager",
        object(),
        raising=True,
    )
    monkeypatch.setattr(
        optimization_engine_module.mcts_optimizer_module,
        "EventBroadcaster",
        _RecordingBroadcaster,
        raising=True,
    )
    await jobs_worker._broadcast_completed_optimization(current_lease, core_result)

    assert completion_events == [
        {
            "optimization_id": optimization_id,
            "optimization_uuid": str(persisted["uuid"]),
            "strategy": "mcts",
            "status": "completed",
            "iterations": 2,
            "final_score": 0.91,
            "tokens_spent": 111,
        }
    ]

    retry_result = await first_processor.process_optimization_job(
        payload,
        optimization_id,
        runtime_model_config=runtime_config,
        on_provider_success=lambda: asyncio.sleep(0),
    )
    assert retry_result == worker_result == expected
    for marker in (
        "provider_dispatches",
        "_provider_dispatches",
        "scorer_provider_dispatched",
        "_scorer_provider_dispatched",
    ):
        assert marker not in retry_result
    assert dispatch_count == 2


def test_sqlite_retry_pending_cannot_revive_a_cancelled_optimization(
    optimization_db: tuple[PromptStudioDatabase, int],
) -> None:
    db, project_id = optimization_db
    optimization_id = _running_optimization(db, project_id)
    cancelled = db.set_optimization_status(
        optimization_id,
        "cancelled",
        error_message="user cancelled",
        mark_completed=True,
    )

    jobs_worker._mark_retry_pending_safely(
        SimpleNamespace(db=db),
        optimization_id,
        jobs_worker.PromptStudioJobError(
            "provider unavailable",
            retryable=True,
            failure_code="provider_unavailable",
        ),
        expected_uuid=str(cancelled["uuid"]),
    )

    persisted = db.get_optimization(optimization_id)
    assert persisted is not None
    assert persisted["status"] == "cancelled"
    assert persisted["error_message"] == "user cancelled"
    assert persisted["completed_at"] == cancelled["completed_at"]


@pytest.mark.asyncio
async def test_jobs_state_lookup_failure_blocks_prompt_completion() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.client_id = "final-gate-test"
            self.row: dict[str, Any] = {
                "id": 7,
                "status": "pending",
                "initial_prompt_id": 11,
                "optimizer_type": "mcts",
                "max_iterations": 1,
                "test_case_ids": [13],
                "optimization_config": {
                    "optimizer_type": "mcts",
                    "target_metric": "accuracy",
                    "model_config": {
                        "provider": "openai",
                        "model": "gpt-test",
                        "parameters": {},
                    },
                },
            }
            self.complete_calls = 0

        def get_optimization(self, optimization_id: int) -> dict[str, Any]:
            assert optimization_id == 7
            return dict(self.row)

        def update_optimization(
            self,
            optimization_id: int,
            updates: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert optimization_id == 7
            self.row.update(updates)
            return dict(self.row)

        def set_optimization_status(
            self,
            optimization_id: int,
            status: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert optimization_id == 7
            self.row["status"] = status
            return dict(self.row)

        def complete_optimization(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.complete_calls += 1
            self.row["status"] = "completed"
            return dict(self.row)

    class FakeMCTS:
        async def optimize(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "optimized_prompt_id": 12,
                "iterations": 1,
                "initial_score": 0.1,
                "final_score": 0.2,
                "improvement": 0.1,
            }

    async def unavailable_final_state() -> bool:
        raise jobs_worker.PromptStudioJobError(
            "Prompt Studio job state is temporarily unavailable",
            retryable=True,
            failure_code="job_store_unavailable",
        )

    db = FakeDB()
    engine = OptimizationEngine(db)  # type: ignore[arg-type]
    engine.mcts = FakeMCTS()  # type: ignore[assignment]

    with pytest.raises(
        jobs_worker.PromptStudioJobError,
        match="job state is temporarily unavailable",
    ):
        await engine.optimize(
            7,
            runtime_model_config={
                "provider": "openai",
                "model": "gpt-test",
                "parameters": {},
            },
            before_finalize=unavailable_final_state,
            manage_failure_status=False,
        )

    assert db.complete_calls == 0
    assert db.row["status"] == "running"


class _ReturningCursor:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


def _postgres_sql_capture(
    returning_row: dict[str, Any] | None = None,
) -> tuple[
    _BackendPromptStudioDatabase,
    list[tuple[str, tuple[Any, ...]]],
]:
    database = object.__new__(_BackendPromptStudioDatabase)
    database._write_lock = threading.RLock()
    statements: list[tuple[str, tuple[Any, ...]]] = []

    @contextmanager
    def _transaction() -> Iterator[object]:
        yield object()

    def _cursor_exec(
        _connection: object,
        statement: str,
        params: list[Any],
    ) -> _ReturningCursor:
        statements.append((statement, tuple(params)))
        row = returning_row
        if row is None:
            row = {"id": 17, "uuid": "opt-17", "status": params[0]}
        return _ReturningCursor(row)

    database.transaction = _transaction  # type: ignore[method-assign]
    database._cursor_exec = _cursor_exec  # type: ignore[method-assign]
    database._row_to_dict = (  # type: ignore[method-assign]
        lambda _cursor, row: dict(row)
    )
    database._log_sync_event = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    return database, statements


def test_postgres_completion_transition_reports_cas_loser() -> None:
    database, _statements = _postgres_sql_capture()
    database._cursor_exec = (  # type: ignore[method-assign]
        lambda _connection, _statement, _params: _ReturningCursor(None)
    )
    database.get_optimization = (  # type: ignore[method-assign]
        lambda _optimization_id, **_kwargs: {
            "id": 17,
            "uuid": "opt-17",
            "status": "completed",
        }
    )

    row, applied = database.complete_optimization(
        17,
        iterations_completed=1,
        _return_transition_applied=True,
    )

    assert applied is False
    assert row["status"] == "completed"


def test_postgres_optimization_update_expected_uuid_is_in_atomic_where() -> None:
    database, statements = _postgres_sql_capture()

    database.update_optimization(
        17,
        {"status": "cancelled"},
        expected_statuses=("completed",),
        expected_uuid="opt-17",
        _return_transition_applied=True,
    )

    statement, params = statements[-1]
    normalized = " ".join(statement.split())
    where_match = re.search(r"\bWHERE\b(?P<clause>.+)", normalized, re.IGNORECASE)
    assert where_match is not None
    where_clause = where_match.group("clause")
    assert re.search(r"\buuid\b", where_clause, flags=re.IGNORECASE)
    assert re.search(r"\bstatus\b", where_clause, flags=re.IGNORECASE)
    assert "opt-17" in params


@pytest.mark.parametrize(
    "transition",
    ["complete", "cancel"],
)
def test_postgres_terminal_transition_update_has_atomic_status_guard(
    transition: str,
) -> None:
    database, statements = _postgres_sql_capture()

    if transition == "complete":
        database.complete_optimization(17, iterations_completed=1)
    else:
        database.set_optimization_status(
            17,
            "cancelled",
            error_message="user cancelled",
            mark_completed=True,
        )

    statement, params = statements[-1]
    normalized = " ".join(statement.split())
    where_match = re.search(r"\bWHERE\b(?P<clause>.+)", normalized, re.IGNORECASE)
    assert where_match is not None
    where_clause = where_match.group("clause")
    assert re.search(r"\bstatus\b", where_clause, flags=re.IGNORECASE)
    contract = f"{where_clause} {params}".lower()
    blocked_status = "cancelled" if transition == "complete" else "completed"
    assert blocked_status in contract or any(
        status in contract for status in ("pending", "queued", "running")
    )
