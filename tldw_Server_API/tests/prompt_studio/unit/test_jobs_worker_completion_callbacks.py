"""Prompt Studio worker completion callback wiring regressions."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    optimization_engine as optimization_engine_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_worker_wires_completion_event_after_core_jobs_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_kwargs: dict[str, Any] = {}

    class _SDK:
        def __init__(self, _manager: object, _config: object) -> None:
            pass

        async def run(self, **kwargs: Any) -> None:
            run_kwargs.update(kwargs)

        def stop(self) -> None:
            return None

    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object(), raising=True)
    monkeypatch.setattr(jobs_worker, "WorkerSDK", _SDK, raising=True)

    await jobs_worker.run_prompt_studio_jobs_worker()

    assert run_kwargs["on_completed"] is jobs_worker._broadcast_completed_optimization
    assert callable(run_kwargs["on_completion_rejected"])


@pytest.mark.asyncio
async def test_standalone_worker_loads_and_owns_provider_override_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def _refresh(*, force: bool = False) -> None:
        assert force is True
        events.append("refresh")

    def _start() -> None:
        events.append("start")

    async def _run() -> None:
        events.append("run")

    async def _shutdown() -> None:
        events.append("shutdown")

    monkeypatch.setattr(
        jobs_worker,
        "refresh_llm_provider_overrides",
        _refresh,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "start_llm_provider_override_refresh_service",
        _start,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "run_prompt_studio_jobs_worker",
        _run,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "shutdown_llm_provider_override_recovery",
        _shutdown,
        raising=True,
    )

    await jobs_worker.main()

    assert events == ["refresh", "start", "run", "shutdown"]


@pytest.mark.asyncio
async def test_standalone_worker_fails_closed_when_override_refresh_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def _refresh(*, force: bool = False) -> None:
        assert force is True
        events.append("refresh")
        raise RuntimeError("override store unavailable")

    async def _run() -> None:
        events.append("run")

    monkeypatch.setattr(
        jobs_worker,
        "refresh_llm_provider_overrides",
        _refresh,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "run_prompt_studio_jobs_worker",
        _run,
        raising=True,
    )

    with pytest.raises(RuntimeError, match="override store unavailable"):
        await jobs_worker.main()

    assert events == ["refresh"]


@pytest.mark.asyncio
async def test_rejected_completion_compensates_late_core_jobs_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[dict[str, Any]] = []

    class _DB:
        def get_optimization(
            self,
            optimization_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "id": optimization_id,
                "uuid": "optimization-17",
                "status": "completed",
            }

        def update_optimization(
            self,
            optimization_id: int,
            values: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            updates.append(
                {
                    "optimization_id": optimization_id,
                    "values": dict(values),
                    **kwargs,
                }
            )
            return {"id": optimization_id, **values}

    class _Manager:
        def get_job(self, _job_id: int) -> dict[str, Any]:
            return {
                "id": 41,
                "uuid": "job-41",
                "domain": "prompt_studio",
                "job_type": "optimization",
                "status": "cancelled",
                "cancellation_reason": "late admin cancellation",
            }

    job = {
        "id": 41,
        "uuid": "job-41",
        "domain": "prompt_studio",
        "job_type": "optimization",
        "owner_user_id": "7",
        "payload": {
            "optimization_id": 17,
            "optimization_uuid": "optimization-17",
        },
    }
    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "single_user", raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: type("_Processor", (), {"db": _DB()})(),
        raising=True,
    )

    await jobs_worker._reconcile_rejected_optimization_completion(
        job,
        {"status": "completed"},
        _Manager(),  # type: ignore[arg-type]
    )

    assert updates == [
        {
            "optimization_id": 17,
            "values": {
                "status": "cancelled",
                "error_message": "late admin cancellation",
            },
            "expected_statuses": (
                "pending",
                "queued",
                "running",
                "completed",
                "failed",
                "cancelled",
            ),
            "set_completed_at": True,
            "expected_uuid": "optimization-17",
            "_return_transition_applied": True,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("optimization_uuid", [None, "stale-optimization-uuid"])
async def test_rejected_completion_does_not_cancel_reused_optimization_id(
    monkeypatch: pytest.MonkeyPatch,
    optimization_uuid: str | None,
) -> None:
    updates: list[dict[str, Any]] = []

    class _DB:
        def get_optimization(
            self,
            optimization_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "id": optimization_id,
                "uuid": "current-optimization-uuid",
                "status": "completed",
            }

        def update_optimization(
            self,
            optimization_id: int,
            values: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            updates.append(
                {
                    "optimization_id": optimization_id,
                    "values": dict(values),
                    **kwargs,
                }
            )
            return {"id": optimization_id, **values}

    class _Manager:
        def get_job(self, _job_id: int) -> dict[str, Any]:
            return {
                "id": 41,
                "uuid": "job-41",
                "domain": "prompt_studio",
                "job_type": "optimization",
                "status": "cancelled",
                "cancellation_reason": "late admin cancellation",
            }

    payload: dict[str, Any] = {"optimization_id": 17}
    if optimization_uuid is not None:
        payload["optimization_uuid"] = optimization_uuid
    job = {
        "id": 41,
        "uuid": "job-41",
        "domain": "prompt_studio",
        "job_type": "optimization",
        "owner_user_id": "7",
        "payload": payload,
    }
    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "single_user", raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: type("_Processor", (), {"db": _DB()})(),
        raising=True,
    )

    await jobs_worker._reconcile_rejected_optimization_completion(
        job,
        {"status": "completed"},
        _Manager(),  # type: ignore[arg-type]
    )

    assert updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("optimization_uuid", [None, "stale-optimization-uuid"])
async def test_mcts_completion_callback_rejects_missing_or_stale_identity(
    monkeypatch: pytest.MonkeyPatch,
    optimization_uuid: str | None,
) -> None:
    broadcasts: list[tuple[int, dict[str, Any]]] = []

    class _DB:
        def get_optimization(
            self,
            optimization_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "id": optimization_id,
                "uuid": "current-optimization-uuid",
                "status": "completed",
            }

    class _Engine:
        def __init__(self, _db: object) -> None:
            pass

        async def _broadcast_mcts_completion(
            self,
            optimization_id: int,
            result: dict[str, Any],
            **_kwargs: Any,
        ) -> None:
            broadcasts.append((optimization_id, dict(result)))

    payload: dict[str, Any] = {
        "optimization_id": 17,
        "optimizer_type": "mcts",
    }
    if optimization_uuid is not None:
        payload["optimization_uuid"] = optimization_uuid
    job = {
        "job_type": "optimization",
        "owner_user_id": "7",
        "payload": payload,
    }
    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "single_user", raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: type("_Processor", (), {"db": _DB()})(),
        raising=True,
    )
    monkeypatch.setattr(
        optimization_engine_module,
        "OptimizationEngine",
        _Engine,
        raising=True,
    )

    await jobs_worker._broadcast_completed_optimization(
        job,
        {"status": "completed"},
    )

    assert broadcasts == []


@pytest.mark.asyncio
async def test_direct_engine_forwards_initial_uuid_to_mcts_completion_broadcast() -> None:
    optimization_id = 17

    class _DB:
        def __init__(self) -> None:
            self.row: dict[str, Any] = {
                "id": optimization_id,
                "uuid": "optimization-17",
                "status": "pending",
                "initial_prompt_id": 5,
                "max_iterations": 1,
                "test_case_ids": [7],
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

        def get_optimization(self, requested_id: int) -> dict[str, Any]:
            assert requested_id == optimization_id
            return dict(self.row)

        def set_optimization_status(
            self,
            requested_id: int,
            status: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert requested_id == optimization_id
            self.row["status"] = status
            return dict(self.row)

        def complete_optimization_with_transition(
            self,
            requested_id: int,
            **updates: Any,
        ) -> tuple[dict[str, Any], bool]:
            assert requested_id == optimization_id
            self.row.update(updates)
            self.row["status"] = "completed"
            return dict(self.row), True

    class _MCTS:
        async def optimize(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "optimized_prompt_id": 5,
                "iterations": 1,
                "initial_score": 0.25,
                "final_score": 0.75,
                "improvement": 0.5,
            }

    forwarded: list[tuple[int, str]] = []

    async def _capture_broadcast(
        requested_id: int,
        _results: dict[str, Any],
        *,
        expected_optimization_uuid: str,
    ) -> None:
        forwarded.append((requested_id, expected_optimization_uuid))

    engine = object.__new__(optimization_engine_module.OptimizationEngine)
    engine.db = _DB()
    engine.mcts = _MCTS()  # type: ignore[assignment]
    engine._broadcast_mcts_completion = _capture_broadcast  # type: ignore[method-assign]

    await engine.optimize(
        optimization_id,
        runtime_model_config={
            "provider": "openai",
            "model": "gpt-test",
            "parameters": {},
        },
        emit_completion_event=True,
    )

    assert forwarded == [(optimization_id, "optimization-17")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("latest_uuid", "expected_uuid", "expected_event_count"),
    [
        ("current-optimization-uuid", "current-optimization-uuid", 1),
        ("current-optimization-uuid", "stale-optimization-uuid", 0),
        ("", "", 0),
    ],
)
async def test_mcts_broadcaster_checks_expected_uuid_on_latest_row(
    monkeypatch: pytest.MonkeyPatch,
    latest_uuid: str,
    expected_uuid: str,
    expected_event_count: int,
) -> None:
    events: list[dict[str, Any]] = []

    class _DB:
        def get_optimization(self, _optimization_id: int) -> dict[str, Any]:
            return {
                "id": 17,
                "uuid": latest_uuid,
                "status": "completed",
                "project_id": 5,
            }

    class _Broadcaster:
        def __init__(self, _manager: object, _db: object) -> None:
            pass

        async def broadcast_event(self, **kwargs: Any) -> None:
            events.append(dict(kwargs))

    monkeypatch.setattr(
        optimization_engine_module.mcts_optimizer_module,
        "ws_connection_manager",
        object(),
        raising=True,
    )
    monkeypatch.setattr(
        optimization_engine_module.mcts_optimizer_module,
        "EventBroadcaster",
        _Broadcaster,
        raising=True,
    )
    engine = object.__new__(optimization_engine_module.OptimizationEngine)
    engine.db = _DB()

    await engine._broadcast_mcts_completion(
        17,
        {"status": "completed", "iterations": 1},
        expected_optimization_uuid=expected_uuid,
    )

    assert len(events) == expected_event_count
    if events:
        assert events[0]["data"]["optimization_uuid"] == expected_uuid
