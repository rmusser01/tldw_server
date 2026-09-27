"""Prompt Studio Jobs owner and terminal-state boundary regressions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)

pytestmark = pytest.mark.unit

_MODEL_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "parameters": {"temperature": 0.1},
}


class _OptimizationDB:
    def __init__(self) -> None:
        self.row: dict[str, Any] = {
            "id": 11,
            "uuid": "optimization-11",
            "status": "pending",
            "optimization_config": {
                "optimizer_type": "mipro",
                "target_metric": "accuracy",
                "model_config": dict(_MODEL_CONFIG),
            },
            "completed_at": None,
            "error_message": None,
        }

    def get_optimization(
        self,
        optimization_id: int,
        *,
        include_deleted: bool,
    ) -> dict[str, Any]:
        assert optimization_id == 11
        assert include_deleted is True
        return dict(self.row)

    def update_optimization(
        self,
        optimization_id: int,
        updates: dict[str, Any],
        *,
        expected_statuses: tuple[str, ...] | None = None,
        expected_uuid: str | None = None,
        set_completed_at: bool = False,
        _return_transition_applied: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], bool]:
        assert optimization_id == 11
        applied = not (
            (expected_statuses is not None and self.row["status"] not in expected_statuses)
            or (expected_uuid is not None and self.row["uuid"] != expected_uuid)
        )
        if applied:
            self.row.update(updates)
            if set_completed_at:
                self.row["completed_at"] = "terminal"
        result = dict(self.row)
        return (result, applied) if _return_transition_applied else result

    def set_optimization_status(
        self,
        optimization_id: int,
        status: str,
        *,
        error_message: str | None = None,
        mark_completed: bool = False,
    ) -> None:
        assert optimization_id == 11
        self.row.update(
            status=status,
            error_message=error_message,
            completed_at="terminal" if mark_completed else None,
        )


class _OptimizationProcessor:
    def __init__(self, db: _OptimizationDB) -> None:
        self.db = db

    async def process_optimization_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("owner/runtime rejection should stop before optimization dispatch")


def _optimization_payload() -> dict[str, Any]:
    return {
        "optimization_id": 11,
        "optimization_uuid": "optimization-11",
        "initial_prompt_id": 12,
        "test_case_ids": [13],
        "optimizer_type": "mipro",
        "max_iterations": 1,
        "optimization_config": {
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": dict(_MODEL_CONFIG),
        },
    }


def _patch_auth_mode(
    monkeypatch: pytest.MonkeyPatch,
    *,
    auth_mode: str,
    owner_record: dict[str, Any] | None = None,
    single_user_id: int = 1,
) -> list[int]:
    lookups: list[int] = []

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
            lookups.append(user_id)
            return owner_record

    async def _from_pool() -> _UsersRepo:
        return _UsersRepo()

    monkeypatch.setattr(
        jobs_worker,
        "get_auth_settings",
        lambda: SimpleNamespace(
            AUTH_MODE=auth_mode,
            SINGLE_USER_FIXED_ID=single_user_id,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        jobs_worker,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_from_pool),
        raising=False,
    )
    return lookups


@pytest.mark.asyncio
async def test_same_code_jobs_quarantine_leaves_optimization_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    monkeypatch.setenv("TLDW_TEST_MODE", "true")
    _patch_auth_mode(monkeypatch, auth_mode="single_user")

    manager = JobManager(tmp_path / "prompt-studio-jobs.db")
    created = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_optimization_payload(),
        owner_user_id="1",
        max_retries=3,
    )
    first = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=5,
        worker_id="seed-worker",
    )
    assert first is not None
    assert manager.fail_job(
        int(created["id"]),
        error="temporarily unavailable",
        retryable=True,
        backoff_seconds=0,
        worker_id="seed-worker",
        lease_id=str(first["lease_id"]),
        error_code="credential_store_unavailable",
    )

    optimization_db = _OptimizationDB()
    processor = _OptimizationProcessor(optimization_db)

    class _UnavailableRuntime:
        async def resolve(self, provider: str, *, model: str | None = None) -> None:
            raise ByokResolutionError("credential_store_unavailable", provider)

        async def close(self) -> None:
            return None

    async def _empty_scope(_user_id: int) -> tuple[list[int], list[int]]:
        return [], []

    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: manager, raising=True)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: processor, raising=True)
    monkeypatch.setattr(jobs_worker, "_owner_membership_scope", _empty_scope, raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "ProviderCredentialRuntime",
        lambda **_kwargs: _UnavailableRuntime(),
        raising=True,
    )

    config = WorkerConfig(
        domain="prompt_studio",
        queue="default",
        worker_id="quarantine-worker",
        lease_seconds=5,
        renew_threshold_seconds=1,
        renew_jitter_seconds=0,
        retry_backoff_seconds=0,
    )
    sdk = WorkerSDK(manager, config)
    real_fail_job = manager.fail_job

    def _finalize_then_stop(job_id: int, **kwargs: Any) -> bool:
        finalized = real_fail_job(job_id, **kwargs)
        sdk.stop()
        return finalized

    monkeypatch.setattr(manager, "fail_job", _finalize_then_stop, raising=True)

    await asyncio.wait_for(sdk.run(handler=jobs_worker._handle_job), timeout=1)

    stored_job = manager.get_job(int(created["id"])) or {}
    assert stored_job["status"] == "quarantined"
    assert stored_job["failure_streak_code"] == "credential_store_unavailable"
    assert optimization_db.row["status"] in {"failed", "quarantined"}
    assert optimization_db.row["status"] != "pending"
    assert optimization_db.row["completed_at"] is not None


def test_prompt_retry_prediction_resets_a_different_failure_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    job = {
        "retry_count": 1,
        "max_retries": 3,
        "failure_streak_code": "E1",
        "failure_streak_count": 1,
    }

    assert jobs_worker._retry_attempt_remains(
        job,
        jobs_worker.PromptStudioJobError(
            "retry",
            retryable=True,
            failure_code="E2",
        ),
    )
    assert not jobs_worker._retry_attempt_remains(
        job,
        jobs_worker.PromptStudioJobError(
            "quarantine",
            retryable=True,
            failure_code="E1",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_record",
    [
        pytest.param({"id": 7, "is_active": False}, id="disabled"),
        pytest.param(None, id="deleted"),
    ],
)
async def test_inactive_owner_is_rejected_before_membership_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
    owner_record: dict[str, Any] | None,
) -> None:
    owner_lookups = _patch_auth_mode(
        monkeypatch,
        auth_mode="multi_user",
        owner_record=owner_record,
    )
    optimization_db = _OptimizationDB()
    membership_calls = 0
    runtime_calls = 0

    async def _unexpected_memberships(_user_id: int) -> tuple[list[int], list[int]]:
        nonlocal membership_calls
        membership_calls += 1
        raise AssertionError("inactive owner reached membership resolution")

    def _unexpected_runtime(**_kwargs: Any) -> None:
        nonlocal runtime_calls
        runtime_calls += 1
        raise AssertionError("inactive owner reached credential resolution")

    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _OptimizationProcessor(optimization_db),
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_owner_membership_scope",
        _unexpected_memberships,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "ProviderCredentialRuntime",
        _unexpected_runtime,
        raising=True,
    )

    with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
        await jobs_worker._handle_job(
            {
                "id": 11,
                "uuid": "owner-revalidation-job",
                "job_type": "optimization",
                "owner_user_id": "7",
                "payload": _optimization_payload(),
            }
        )

    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False
    assert owner_lookups == [7]
    assert membership_calls == 0
    assert runtime_calls == 0


@pytest.mark.asyncio
async def test_missing_owner_falls_back_only_in_explicit_single_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auth_mode(monkeypatch, auth_mode="single_user", single_user_id=41)
    monkeypatch.setattr(
        jobs_worker.DatabasePaths,
        "get_single_user_id",
        lambda: 41,
        raising=True,
    )
    processor_user_ids: list[str] = []

    class _GenerationProcessor:
        async def process_generation_job(
            self,
            payload: dict[str, Any],
            entity_id: int,
        ) -> dict[str, Any]:
            return {"project_id": entity_id, "job_id": payload["job_id"]}

    def _processor(user_id: str) -> _GenerationProcessor:
        processor_user_ids.append(user_id)
        return _GenerationProcessor()

    monkeypatch.setattr(jobs_worker, "_get_processor", _processor, raising=True)

    result = await jobs_worker._handle_job(
        {
            "id": 22,
            "uuid": "single-user-owner-fallback",
            "job_type": "generation",
            "payload": {"project_id": 42},
        }
    )

    assert result == {"project_id": 42, "job_id": "single-user-owner-fallback"}
    assert processor_user_ids == ["41"]


@pytest.mark.asyncio
async def test_missing_owner_is_rejected_in_multi_user_before_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auth_mode(monkeypatch, auth_mode="multi_user")
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("ownerless multi-user job reached processor")
        ),
        raising=True,
    )

    with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
        await jobs_worker._handle_job(
            {
                "id": 23,
                "uuid": "multi-user-owner-missing",
                "job_type": "generation",
                "payload": {"project_id": 43},
            }
        )

    assert exc_info.value.retryable is False
    assert "owner_user_id" in str(exc_info.value)
