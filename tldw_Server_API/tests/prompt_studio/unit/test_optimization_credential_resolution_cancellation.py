"""Cancellation gates around Prompt Studio credential resolution."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_cancellation_during_credential_resolution_prevents_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution_started = asyncio.Event()
    release_resolution = asyncio.Event()
    dispatches = 0
    runtime_closes = 0
    optimization_id = 17
    durable_config = {
        "optimizer_type": "mipro",
        "target_metric": "accuracy",
        "model_config": {
            "provider": "openai",
            "model": "gpt-test",
            "parameters": {},
        },
    }

    class _DB:
        def __init__(self) -> None:
            self.row: dict[str, Any] = {
                "id": optimization_id,
                "uuid": "optimization-17",
                "status": "running",
                "optimization_config": durable_config,
            }

        def get_optimization(
            self,
            requested_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert requested_id == optimization_id
            return dict(self.row)

        def update_optimization(
            self,
            requested_id: int,
            values: dict[str, Any],
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert requested_id == optimization_id
            self.row.update(values)
            return dict(self.row)

    class _Processor:
        def __init__(self) -> None:
            self.db = _DB()

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            requested_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            nonlocal dispatches
            assert requested_id == optimization_id
            dispatches += 1
            self.db.row["status"] = "completed"
            return {
                "optimization_id": optimization_id,
                "status": "completed",
            }

    class _Manager:
        def __init__(self) -> None:
            self.row: dict[str, Any] = {
                "id": 41,
                "uuid": "job-41",
                "domain": "prompt_studio",
                "job_type": "optimization",
                "status": "processing",
                "lease_id": "lease-41",
                "cancel_requested_at": None,
                "cancellation_reason": None,
            }

        def get_job(self, job_id: int) -> dict[str, Any]:
            assert job_id == 41
            return dict(self.row)

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def resolve(
            self,
            provider: str,
            *,
            model: str,
        ) -> SimpleNamespace:
            assert (provider, model) == ("openai", "gpt-test")
            resolution_started.set()
            await release_resolution.wait()
            return SimpleNamespace(
                api_key="resolved-key",
                app_config={},
                credentials_resolved=True,
            )

        async def close(self) -> None:
            nonlocal runtime_closes
            runtime_closes += 1

    processor = _Processor()
    manager = _Manager()
    job = {
        **manager.row,
        "owner_user_id": "7",
        "payload": {
            "optimization_id": optimization_id,
            "optimization_uuid": "optimization-17",
            "optimization_config": durable_config,
        },
    }

    async def _owner_is_active(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def _memberships(_owner_id: int) -> tuple[list[int], list[int]]:
        return [], []

    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "multi_user", raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: processor,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_secure_optimization_durable_state",
        lambda **kwargs: (dict(kwargs["payload"]), durable_config),
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_ensure_job_owner_active",
        _owner_is_active,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_owner_membership_scope",
        _memberships,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "ProviderCredentialRuntime",
        _Runtime,
        raising=True,
    )

    task = asyncio.create_task(
        jobs_worker._handle_job(job, job_manager=manager)  # type: ignore[arg-type]
    )
    try:
        await asyncio.wait_for(resolution_started.wait(), timeout=1)
        manager.row.update(
            {
                "status": "cancelled",
                "cancel_requested_at": "2026-07-16T00:00:00Z",
                "cancellation_reason": "admin cancellation",
            }
        )
        release_resolution.set()
        with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
            await task
    finally:
        release_resolution.set()
        if not task.done():
            task.cancel()

    assert exc_info.value.failure_code == "job_cancelled"
    assert dispatches == 0
    assert processor.db.row["status"] == "cancelled"
    assert runtime_closes == 1
