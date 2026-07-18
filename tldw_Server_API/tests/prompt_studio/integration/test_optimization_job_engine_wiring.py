import asyncio
import contextlib
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as ps_opt_endpoints,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationCreate,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    OptimizationSimpleCreateRequest,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    mcts_optimizer as mcts_optimizer_mod,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as prompt_executor_mod,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    test_runner as test_runner_mod,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import JobProcessor
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import MCTSOptimizer
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import (
    OptimizationEngine,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import jobs_worker
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

pytestmark = pytest.mark.integration


@pytest.fixture
def optimization_runtime_boundary(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install one authoritative, observable credential runtime per worker call."""
    state: dict[str, Any] = {
        "adapter_requests": [],
        "membership_calls": [],
        "instances": [],
        "runner_boundary_crossed": False,
        "completed_events": [],
        "completion_job_statuses": [],
    }

    async def _memberships(user_id: int) -> tuple[list[int], list[int]]:
        state["membership_calls"].append(user_id)
        return [71], [72]

    class _Runtime:
        def __init__(self, **scope: Any) -> None:
            self.scope = scope
            self.resolve_calls: list[tuple[str, str | None]] = []
            self.mark_calls: list[Any] = []
            self.close_count = 0
            self.handle = SimpleNamespace(
                provider="openai",
                api_key="optimization-runtime-key",
                app_config={"openai_api": {"model": "gpt-3.5-turbo"}},
                credentials_resolved=True,
            )
            state["instances"].append(self)

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            self.resolve_calls.append((provider, model))
            return self.handle

        async def mark_used(self, handle: Any) -> bool:
            self.mark_calls.append(handle)
            return True

        async def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(jobs_worker, "_owner_membership_scope", _memberships, raising=True)
    monkeypatch.setattr(jobs_worker, "ProviderCredentialRuntime", _Runtime, raising=True)

    class _Adapter:
        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            state["adapter_requests"].append(dict(request))
            return {
                "choices": [{"message": {"content": "8"}}],
                "usage": {"total_tokens": 2},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    monkeypatch.setattr(
        prompt_executor_mod,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    monkeypatch.setattr(
        test_runner_mod,
        "get_adapter_or_raise",
        lambda _provider: _Adapter(),
        raising=True,
    )
    for module in (prompt_executor_mod, test_runner_mod):
        monkeypatch.setattr(
            module,
            "is_runtime_issued_provider_call_credentials",
            lambda value, *, provider=None: (
                value is not None
                and (provider is None or value.provider == provider)
            ),
            raising=False,
        )

    original_run_single_test = TestRunner.run_single_test

    async def boundary_then_synthetic_score(
        self,
        *,
        prompt_id: int,
        test_case_id: int,
        model_config: dict[str, Any],
        metrics: Any = None,
        provider_credentials: Any = None,
        on_provider_success: Any = None,
        strict_provider_errors: bool = False,
    ) -> dict[str, Any]:
        if not state["runner_boundary_crossed"]:
            state["runner_boundary_crossed"] = True
            return await original_run_single_test(
                self,
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config,
                metrics=metrics,
                provider_credentials=provider_credentials,
                on_provider_success=on_provider_success,
                strict_provider_errors=strict_provider_errors,
            )

        await _mark_fake_provider_success(on_provider_success)
        row = self.db.get_prompt(prompt_id) or {}
        system_text = row.get("system_prompt") or ""
        score = 0.9 if "Ensure outputs strictly follow" in system_text else 0.2
        return {"success": True, "scores": {"aggregate_score": score}}

    monkeypatch.setattr(
        TestRunner,
        "run_single_test",
        boundary_then_synthetic_score,
        raising=True,
    )

    class _CompletionBroadcaster:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def broadcast_event(
            self,
            event_type: Any,
            data: dict[str, Any],
            **kwargs: Any,
        ) -> None:
            if event_type is not mcts_optimizer_mod.EventType.OPTIMIZATION_COMPLETED:
                return
            state["completed_events"].append((event_type, dict(data), dict(kwargs)))
            observer = state.get("completion_status_observer")
            if callable(observer):
                state["completion_job_statuses"].append(observer())

    monkeypatch.setattr(
        mcts_optimizer_mod,
        "EventBroadcaster",
        _CompletionBroadcaster,
        raising=True,
    )
    monkeypatch.setattr(
        mcts_optimizer_mod,
        "ws_connection_manager",
        object(),
        raising=True,
    )
    return state


async def _mark_fake_provider_success(callback: Any) -> None:
    if callback is not None:
        await callback()


def _seed_prompt_and_case(db):
    project = db.create_project(name="OptJobEngine", description="", user_id="7")
    prompt = db.create_prompt(
        project_id=project["id"],
        name="BasePrompt",
        system_prompt="You are precise and concise.",
        user_prompt="Echo: {text}",
    )
    case = db.create_test_case(
        project_id=project["id"],
        name="Case1",
        inputs={"text": "hello"},
        expected_outputs={"response": "hello"},
    )
    return project, prompt, case


def _get_job_by_id(job_id: str) -> dict:
    jm = jobs_worker._jobs_manager()
    job = None
    with contextlib.suppress(Exception):
        job = jm.get_job_by_uuid(str(job_id))
    if job is None and str(job_id).isdigit():
        with contextlib.suppress(Exception):
            job = jm.get_job(int(job_id))
    assert job is not None, f"job not found: {job_id}"
    return job


def _fake_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
        }
    )


async def _create_optimization_via_endpoint(
    db,
    payload: dict,
) -> tuple[int, str, dict]:
    response = await ps_opt_endpoints.create_optimization(
        optimization_data=OptimizationCreate.model_validate(payload),
        request=_fake_request("/api/v1/prompt-studio/optimizations/create"),
        _=True,
        db=db,
        security_config={},
        user_context={"user_id": "7", "is_admin": True, "permissions": ["all"]},
        idempotency_key=None,
    )
    body = response.model_dump()
    assert body.get("success") is True, body
    data = body.get("data") or {}
    optimization = data.get("optimization") or {}
    return int(optimization.get("id")), str(data.get("job_id")), body


async def _cancel_optimization_via_endpoint(db, optimization_id: int) -> dict:
    response = await ps_opt_endpoints.cancel_optimization(
        request=_fake_request(f"/api/v1/prompt-studio/optimizations/cancel/{optimization_id}"),
        optimization_id=optimization_id,
        reason=None,
        db=db,
        user_context={"user_id": "7", "is_admin": True, "permissions": ["all"]},
    )
    body = response.model_dump()
    assert body.get("success") is True, body
    return body


@pytest.mark.asyncio
async def test_admin_cross_owner_create_then_cancel_targets_exact_core_job(
    prompt_studio_dual_backend_db,
    monkeypatch,
):
    async def allow_project_write_access(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_write_access",
        allow_project_write_access,
        raising=True,
    )

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)
    admin_context = {
        "user_id": "99",
        "is_admin": True,
        "permissions": ["all"],
    }
    create_response = await ps_opt_endpoints.create_optimization(
        optimization_data=OptimizationCreate.model_validate(
            {
                "project_id": project["id"],
                "initial_prompt_id": prompt["id"],
                "optimization_config": {
                    "optimizer_type": "iterative",
                    "max_iterations": 1,
                    "target_metric": "accuracy",
                },
                "test_case_ids": [case["id"]],
                "name": "Admin cross-owner cancellation",
            }
        ),
        request=_fake_request("/api/v1/prompt-studio/optimizations/create"),
        _=True,
        db=db,
        security_config={},
        user_context=admin_context,
        idempotency_key=None,
    )
    create_body = create_response.model_dump()
    create_data = create_body.get("data") or {}
    optimization = create_data.get("optimization") or {}
    job_id = str(create_data.get("job_id"))
    created_job = _get_job_by_id(job_id)

    assert created_job["owner_user_id"] == str(project["user_id"])
    assert created_job["payload"]["created_by"] == "99"

    cancel_response = await ps_opt_endpoints.cancel_optimization(
        request=_fake_request(
            f"/api/v1/prompt-studio/optimizations/cancel/{optimization['id']}"
        ),
        optimization_id=int(optimization["id"]),
        reason="admin requested cancellation",
        db=db,
        user_context=admin_context,
    )

    assert cancel_response.success is True
    cancelled_job = _get_job_by_id(job_id)
    assert cancelled_job["uuid"] == created_job["uuid"]
    assert cancelled_job["status"] == "cancelled"
    assert cancelled_job["cancellation_reason"] == "admin requested cancellation"
    assert (db.get_optimization(int(optimization["id"])) or {})["status"] == "cancelled"


@pytest.mark.asyncio
async def test_process_optimization_job_routes_mcts_to_engine_and_persists_runtime_inputs(
    prompt_studio_dual_backend_db,
    monkeypatch,
):
    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)

    optimization_config = {
        "optimizer_type": "mcts",
        "target_metric": "accuracy",
        "strategy_params": {
            "mcts_simulations": 3,
            "mcts_max_depth": 2,
            "prompt_candidates_per_node": 1,
            "token_budget": 1000,
        },
    }
    optimization = db.create_optimization(
        project_id=project["id"],
        name="MCTS-Job-Wiring",
        initial_prompt_id=prompt["id"],
        optimizer_type="mcts",
        optimization_config=optimization_config,
        max_iterations=3,
        status="pending",
    )
    optimization = db.update_optimization(
        int(optimization["id"]),
        {"test_case_ids": [int(case["id"])]},
    )

    async def fake_run_single_test(
        self,
        *,
        prompt_id: int,
        test_case_id: int,
        model_config: dict[str, Any],
        metrics: Any = None,
        on_provider_success: Any = None,
        strict_provider_errors: bool = False,
    ) -> dict[str, Any]:
        await _mark_fake_provider_success(on_provider_success)
        row = self.db.get_prompt(prompt_id) or {}
        system_text = row.get("system_prompt") or ""
        score = 0.9 if "Ensure outputs strictly follow" in system_text else 0.2
        return {"success": True, "scores": {"aggregate_score": score}}

    async def fake_rephrase_segment(
        self,
        system_text: str,
        segment_text: str,
        **_kwargs: Any,
    ) -> None:
        return None

    monkeypatch.setattr(TestRunner, "run_single_test", fake_run_single_test, raising=True)
    monkeypatch.setattr(MCTSOptimizer, "_rephrase_segment", fake_rephrase_segment, raising=True)

    payload = {
        "optimization_id": optimization["id"],
        "optimizer_type": "mcts",
        "initial_prompt_id": prompt["id"],
        "test_case_ids": [case["id"]],
        "optimization_config": optimization_config,
    }

    processor = JobProcessor(db)
    result = await processor.process_optimization_job(payload, int(optimization["id"]))

    row = db.get_optimization(int(optimization["id"])) or {}
    assert row.get("status") == "completed"
    assert isinstance((row.get("final_metrics") or {}).get("trace"), dict)
    assert [int(x) for x in (row.get("test_case_ids") or [])] == [int(case["id"])]

    assert int(result.get("optimization_id")) == int(optimization["id"])
    assert int(result.get("iterations_completed") or 0) >= 1
    assert result.get("best_prompt_id") is not None


@pytest.mark.asyncio
async def test_process_optimization_job_rejects_unsupported_strategy_without_simulation(
    prompt_studio_dual_backend_db,
    monkeypatch,
):
    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)

    optimization = db.create_optimization(
        project_id=project["id"],
        name="Unsupported-Strategy",
        initial_prompt_id=prompt["id"],
        optimizer_type="quantum_search",
        optimization_config={
            "optimizer_type": "quantum_search",
            "target_metric": "accuracy",
            "model_config": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "parameters": {},
            },
        },
        max_iterations=3,
        status="pending",
    )
    db.update_optimization(
        int(optimization["id"]),
        {"test_case_ids": [int(case["id"])]},
    )

    async def forbidden_optimize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("unsupported strategy reached optimization engine")

    monkeypatch.setattr(
        OptimizationEngine,
        "optimize",
        forbidden_optimize,
        raising=True,
    )

    payload = {
        "optimization_id": optimization["id"],
        "optimizer_type": "quantum_search",
        "initial_prompt_id": prompt["id"],
        "test_case_ids": [case["id"]],
        "max_iterations": 3,
        "optimization_config": optimization["optimization_config"],
    }
    processor = JobProcessor(db)

    with pytest.raises(ValueError, match="(?i)unknown|unsupported"):
        await processor.process_optimization_job(payload, int(optimization["id"]))

    row = db.get_optimization(int(optimization["id"])) or {}
    assert row.get("status") == "failed"
    assert int(row.get("iterations_completed") or 0) == 0


@pytest.mark.asyncio
async def test_process_optimization_job_rejects_conflicting_strategy_snapshots(
    prompt_studio_dual_backend_db,
    monkeypatch,
):
    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)
    optimization = db.create_optimization(
        project_id=project["id"],
        name="Conflicting-Strategy",
        initial_prompt_id=prompt["id"],
        optimizer_type="mipro",
        optimization_config={
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "parameters": {},
            },
        },
        max_iterations=1,
        status="pending",
    )
    db.update_optimization(
        int(optimization["id"]),
        {"test_case_ids": [int(case["id"])]},
    )

    async def forbidden_optimize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("conflicting strategy reached optimization engine")

    monkeypatch.setattr(
        OptimizationEngine,
        "optimize",
        forbidden_optimize,
        raising=True,
    )
    payload = {
        "optimization_id": optimization["id"],
        "optimizer_type": "bootstrap",
        "initial_prompt_id": prompt["id"],
        "test_case_ids": [case["id"]],
        "optimization_config": {
            **optimization["optimization_config"],
            "optimizer_type": "bootstrap",
        },
    }

    with pytest.raises(ValueError, match="Optimization strategy mismatch"):
        await JobProcessor(db).process_optimization_job(
            payload,
            int(optimization["id"]),
        )

    row = db.get_optimization(int(optimization["id"])) or {}
    assert row.get("status") == "failed"


@pytest.mark.asyncio
async def test_endpoint_created_mcts_job_executes_via_worker_path_and_persists_trace(
    prompt_studio_dual_backend_db,
    monkeypatch,
    optimization_runtime_boundary,
):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    async def allow_project_write_access(*args, **kwargs):
        return None

    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_write_access",
        allow_project_write_access,
        raising=True,
    )

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)

    async def fake_rephrase_segment(
        self,
        system_text: str,
        segment_text: str,
        **_kwargs: Any,
    ) -> None:
        return None

    monkeypatch.setattr(MCTSOptimizer, "_rephrase_segment", fake_rephrase_segment, raising=True)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: JobProcessor(db), raising=True)

    optimization_id, job_id, _body = await _create_optimization_via_endpoint(
        db,
        {
            "project_id": project["id"],
            "initial_prompt_id": prompt["id"],
            "optimization_config": {
                "optimizer_type": "mcts",
                "max_iterations": 6,
                "target_metric": "accuracy",
                "strategy_params": {
                    "mcts_simulations": 4,
                    "mcts_max_depth": 2,
                    "prompt_candidates_per_node": 1,
                    "token_budget": 1000,
                },
            },
            "test_case_ids": [case["id"]],
            "name": "MCTS Endpoint Worker",
        },
    )

    manager = jobs_worker._jobs_manager()
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="prompt_studio",
            queue="default",
            worker_id="mcts-post-complete",
            lease_seconds=30,
            renew_threshold_seconds=5,
            renew_jitter_seconds=0,
        ),
    )
    captured_result: dict[str, Any] = {}
    optimization_runtime_boundary["completion_status_observer"] = lambda: str(
        (_get_job_by_id(job_id) or {}).get("status")
    )

    async def handle_job(job: dict[str, Any]) -> dict[str, Any]:
        result = await jobs_worker._handle_job(job, job_manager=manager)
        captured_result.update(result)
        return result

    async def publish_completion(
        job: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        sdk.stop()
        await jobs_worker._broadcast_completed_optimization(job, result)

    await asyncio.wait_for(
        sdk.run(
            handler=handle_job,
            job_type="optimization",
            on_completed=publish_completion,
        ),
        timeout=5,
    )
    result = captured_result
    row = db.get_optimization(optimization_id) or {}

    assert row.get("status") == "completed"
    assert isinstance((row.get("final_metrics") or {}).get("trace"), dict)
    assert [int(x) for x in (row.get("test_case_ids") or [])] == [int(case["id"])]
    assert int(result.get("iterations") or 0) >= 1
    assert optimization_runtime_boundary["membership_calls"] == [7]
    runtime = optimization_runtime_boundary["instances"][0]
    assert runtime.resolve_calls == [("openai", "gpt-3.5-turbo")]
    assert runtime.mark_calls == [runtime.handle]
    assert runtime.close_count == 1
    assert optimization_runtime_boundary["adapter_requests"]
    for request in optimization_runtime_boundary["adapter_requests"]:
        assert request["api_key"] == "optimization-runtime-key"
        assert request["app_config"] == {
            "openai_api": {"model": "gpt-3.5-turbo"}
        }
        assert request["credentials_resolved"] is True
    assert len(optimization_runtime_boundary["completed_events"]) == 1
    completed_type, completed_data, _completed_kwargs = (
        optimization_runtime_boundary["completed_events"][0]
    )
    assert completed_type is mcts_optimizer_mod.EventType.OPTIMIZATION_COMPLETED
    assert completed_data["optimization_id"] == optimization_id
    assert completed_data["status"] == "completed"
    assert optimization_runtime_boundary["completion_job_statuses"] == ["completed"]


@pytest.mark.asyncio
async def test_history_endpoint_returns_progress_and_timeline_after_worker_run(
    prompt_studio_dual_backend_db,
    monkeypatch,
    optimization_runtime_boundary,
):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    async def allow_project_write_access(*args, **kwargs):
        return None

    async def allow_project_access(*args, **kwargs):
        return None

    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_write_access",
        allow_project_write_access,
        raising=True,
    )
    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_access",
        allow_project_access,
        raising=True,
    )

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)

    async def fake_rephrase_segment(
        self,
        system_text: str,
        segment_text: str,
        **_kwargs: Any,
    ) -> None:
        return None

    monkeypatch.setattr(MCTSOptimizer, "_rephrase_segment", fake_rephrase_segment, raising=True)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: JobProcessor(db), raising=True)

    optimization_id, job_id, _body = await _create_optimization_via_endpoint(
        db,
        {
            "project_id": project["id"],
            "initial_prompt_id": prompt["id"],
            "optimization_config": {
                "optimizer_type": "mcts",
                "max_iterations": 6,
                "target_metric": "accuracy",
                "strategy_params": {
                    "mcts_simulations": 4,
                    "mcts_max_depth": 2,
                    "prompt_candidates_per_node": 1,
                    "token_budget": 1000,
                },
            },
            "test_case_ids": [case["id"]],
            "name": "MCTS History Worker",
        },
    )

    _result = await jobs_worker._handle_job(_get_job_by_id(job_id))

    history_response = await ps_opt_endpoints.get_optimization_history(
        optimization_id=optimization_id,
        db=db,
        user_context={"user_id": "7", "is_admin": True, "permissions": ["all"]},
    )
    body = history_response.model_dump()
    assert body.get("success") is True
    data = body.get("data") or {}
    progress = data.get("progress") or {}
    timeline = data.get("timeline") or []

    assert str(progress.get("status")).lower() == "completed"
    assert int(progress.get("iterations_completed") or 0) >= 1
    assert isinstance(timeline, list) and len(timeline) >= 1
    assert optimization_runtime_boundary["membership_calls"] == [7]
    runtime = optimization_runtime_boundary["instances"][0]
    assert runtime.resolve_calls == [("openai", "gpt-3.5-turbo")]
    assert runtime.mark_calls == [runtime.handle]
    assert runtime.close_count == 1
    assert optimization_runtime_boundary["adapter_requests"]
    for request in optimization_runtime_boundary["adapter_requests"]:
        assert request["api_key"] == "optimization-runtime-key"
        assert request["app_config"] == {
            "openai_api": {"model": "gpt-3.5-turbo"}
        }
        assert request["credentials_resolved"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["iterative", "mipro", "bootstrap"])
async def test_simple_endpoint_compat_strategy_uses_real_provider_boundaries(
    prompt_studio_dual_backend_db,
    monkeypatch,
    optimization_runtime_boundary,
    strategy,
):
    _label, db = prompt_studio_dual_backend_db
    _project, prompt, case = _seed_prompt_and_case(db)
    adapter_requests: list[tuple[str, dict[str, Any]]] = []

    class _Adapter:
        def __init__(self, boundary: str) -> None:
            self.boundary = boundary

        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            adapter_requests.append((self.boundary, dict(request)))
            content = (
                "Always answer with hello."
                if self.boundary == "executor"
                else "not hello"
            )
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 2},
            }

    class _Registry:
        @staticmethod
        def is_local_provider_name(_provider: str) -> bool:
            return False

        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter("executor")

    for module in (test_runner_mod, prompt_executor_mod):
        monkeypatch.setattr(
            module,
            "ensure_app_config",
            lambda config: config or {},
            raising=True,
        )
        monkeypatch.setattr(
            module,
            "resolve_provider_api_key_from_config",
            lambda *_args, **_kwargs: None,
            raising=True,
        )
    monkeypatch.setattr(
        test_runner_mod,
        "get_adapter_or_raise",
        lambda _provider: _Adapter("runner"),
        raising=True,
    )
    monkeypatch.setattr(
        prompt_executor_mod,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: JobProcessor(db), raising=True)

    response = await ps_opt_endpoints.create_optimization_simple(
        payload=OptimizationSimpleCreateRequest.model_validate(
            {
                "prompt_id": prompt["id"],
                "project_id": _project["id"],
                "strategy": strategy,
                "name": f"WebUI {strategy} compatibility",
                "description": "WebUI-shaped provider-bound regression",
                "model_config": {
                    "provider": "openai",
                    "model_name": "gpt-3.5-turbo",
                },
                "test_case_ids": [case["id"]],
                "config": {"max_iterations": 1},
            }
        ),
        request=_fake_request("/api/v1/prompt-studio/optimizations"),
        db=db,
        user_context={"user_id": "7", "is_admin": True, "permissions": ["all"]},
    )
    job_id = str(response["id"])
    job = _get_job_by_id(job_id)
    assert job["payload"]["optimizer_type"] == strategy
    assert job["payload"]["optimization_config"]["optimizer_type"] == strategy
    assert job["payload"]["test_case_ids"] == [case["id"]]
    optimization_id = int(job["payload"]["optimization_id"])

    result = await jobs_worker._handle_job(job)
    row = db.get_optimization(optimization_id) or {}

    assert row.get("status") == "completed"
    assert result.get("status") == "completed"
    assert int(result.get("iterations_completed") or 0) >= 1
    expected_boundaries = (
        {"runner"}
        if strategy == "bootstrap"
        else {"runner", "executor"}
    )
    assert {boundary for boundary, _request in adapter_requests} == expected_boundaries
    for _boundary, request in adapter_requests:
        assert request["api_key"] == "optimization-runtime-key"
        assert request["app_config"] == {
            "openai_api": {"model": "gpt-3.5-turbo"}
        }
        assert request["credentials_resolved"] is True
    assert optimization_runtime_boundary["membership_calls"] == [7]
    runtime = optimization_runtime_boundary["instances"][0]
    assert runtime.resolve_calls == [("openai", "gpt-3.5-turbo")]
    assert runtime.mark_calls == [runtime.handle]
    assert runtime.close_count == 1


@pytest.mark.asyncio
async def test_worker_path_respects_cancelled_optimization_for_queued_and_running_states(
    prompt_studio_dual_backend_db,
    monkeypatch,
    optimization_runtime_boundary,
):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    async def allow_project_write_access(*args, **kwargs):
        return None

    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_write_access",
        allow_project_write_access,
        raising=True,
    )

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)
    provider_entered = asyncio.Event()
    release_provider = asyncio.Event()
    provider_calls = 0
    broadcast_events: list[Any] = []

    class _RecordingBroadcaster:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def broadcast_event(self, event_type: Any, **_kwargs: Any) -> None:
            broadcast_events.append(event_type)

        async def broadcast_optimization_iteration(self, **_kwargs: Any) -> None:
            return None

    async def fake_run_single_test(
        self,
        *,
        prompt_id: int,
        test_case_id: int,
        model_config: dict[str, Any],
        metrics: Any = None,
        provider_credentials: Any = None,
        on_provider_success: Any = None,
        strict_provider_errors: bool = False,
    ) -> dict[str, Any]:
        nonlocal provider_calls
        assert provider_credentials is optimization_runtime_boundary["instances"][0].handle
        provider_calls += 1
        if provider_calls == 1:
            provider_entered.set()
            await release_provider.wait()
        await _mark_fake_provider_success(on_provider_success)
        return {"success": True, "scores": {"aggregate_score": 0.25}}

    async def fake_rephrase_segment(
        self,
        system_text: str,
        segment_text: str,
        **_kwargs: Any,
    ) -> None:
        return None

    monkeypatch.setattr(TestRunner, "run_single_test", fake_run_single_test, raising=True)
    monkeypatch.setattr(MCTSOptimizer, "_rephrase_segment", fake_rephrase_segment, raising=True)
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: JobProcessor(db), raising=True)
    monkeypatch.setattr(mcts_optimizer_mod, "EventBroadcaster", _RecordingBroadcaster, raising=True)
    monkeypatch.setattr(mcts_optimizer_mod, "ws_connection_manager", object(), raising=True)

    # Queued cancellation: cancelling before worker handling should keep optimization cancelled.
    queued_opt_id, queued_job_id, _queued_body = await _create_optimization_via_endpoint(
        db,
        {
            "project_id": project["id"],
            "initial_prompt_id": prompt["id"],
            "optimization_config": {
                "optimizer_type": "mcts",
                "max_iterations": 12,
                "target_metric": "accuracy",
                "strategy_params": {
                    "mcts_simulations": 12,
                    "mcts_max_depth": 2,
                    "prompt_candidates_per_node": 1,
                    "token_budget": 1000,
                },
            },
            "test_case_ids": [case["id"]],
            "name": "Queued-Cancel",
        },
    )
    _cancelled = await _cancel_optimization_via_endpoint(db, queued_opt_id)

    queued_result = await jobs_worker._handle_job(_get_job_by_id(queued_job_id))
    queued_row = db.get_optimization(queued_opt_id) or {}
    assert queued_result.get("status") == "cancelled"
    assert str(queued_row.get("status")).lower() == "cancelled"
    assert optimization_runtime_boundary["membership_calls"] == []
    assert optimization_runtime_boundary["instances"] == []

    # Running cancellation: cancel while worker path is executing.
    running_opt_id, running_job_id, _running_body = await _create_optimization_via_endpoint(
        db,
        {
            "project_id": project["id"],
            "initial_prompt_id": prompt["id"],
            "optimization_config": {
                "optimizer_type": "mcts",
                "max_iterations": 30,
                "target_metric": "accuracy",
                "strategy_params": {
                    "mcts_simulations": 30,
                    "mcts_max_depth": 2,
                    "prompt_candidates_per_node": 1,
                    "token_budget": 10000,
                    "feedback_enabled": False,
                },
            },
            "test_case_ids": [case["id"]],
            "name": "Running-Cancel",
        },
    )

    running_job = _get_job_by_id(running_job_id)
    worker_task = asyncio.create_task(jobs_worker._handle_job(running_job))
    try:
        await asyncio.wait_for(provider_entered.wait(), timeout=2.0)
        cancel_body = await _cancel_optimization_via_endpoint(db, running_opt_id)
        row_while_provider_in_flight = db.get_optimization(running_opt_id) or {}
        job_while_provider_in_flight = _get_job_by_id(running_job_id)
    except BaseException:
        release_provider.set()
        with contextlib.suppress(BaseException):
            await worker_task
        raise
    release_provider.set()
    worker_result = await worker_task

    running_row = db.get_optimization(running_opt_id) or {}
    raw_cfg = running_row.get("optimization_config") or {}
    if isinstance(raw_cfg, str):
        with contextlib.suppress(Exception):
            raw_cfg = json.loads(raw_cfg)
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}
    requested_sims = int(((raw_cfg.get("strategy_params") or {}).get("mcts_simulations")) or 30)
    assert cancel_body.get("data") == {"message": "Optimization cancelled"}
    assert str(row_while_provider_in_flight.get("status")).lower() == "cancelled"
    assert str(job_while_provider_in_flight.get("status")).lower() == "cancelled"
    assert str(running_row.get("status")).lower() == "cancelled"
    assert int(running_row.get("iterations_completed") or 0) < requested_sims
    assert isinstance((running_row.get("final_metrics") or {}).get("trace"), dict)
    assert int(worker_result.get("iterations") or 0) < requested_sims
    assert optimization_runtime_boundary["membership_calls"] == [7]
    runtime = optimization_runtime_boundary["instances"][0]
    assert runtime.resolve_calls == [("openai", "gpt-3.5-turbo")]
    assert runtime.mark_calls == [runtime.handle]
    assert runtime.close_count == 1
    assert provider_calls == 1
    assert mcts_optimizer_mod.EventType.OPTIMIZATION_STARTED in broadcast_events
    assert mcts_optimizer_mod.EventType.OPTIMIZATION_COMPLETED not in broadcast_events
    assert optimization_runtime_boundary["completed_events"] == []


@pytest.mark.asyncio
async def test_generic_jobs_cancellation_stops_running_optimization_at_adapter_boundary(
    prompt_studio_dual_backend_db,
    monkeypatch,
    optimization_runtime_boundary,
):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    async def allow_project_write_access(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_write_access",
        allow_project_write_access,
        raising=True,
    )

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    adapter_requests: list[dict[str, Any]] = []

    class _BlockingAdapter:
        def chat(
            self,
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            adapter_requests.append(dict(request))
            provider_entered.set()
            if not release_provider.wait(timeout=5.0):
                raise TimeoutError("test provider release timed out")
            return {
                "choices": [{"message": {"content": "8"}}],
                "usage": {"total_tokens": 2},
            }

    monkeypatch.setattr(
        test_runner_mod,
        "get_adapter_or_raise",
        lambda _provider: _BlockingAdapter(),
        raising=True,
    )
    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: JobProcessor(db), raising=True)

    optimization_id, job_id, _body = await _create_optimization_via_endpoint(
        db,
        {
            "project_id": project["id"],
            "initial_prompt_id": prompt["id"],
            "optimization_config": {
                "optimizer_type": "mcts",
                "max_iterations": 10,
                "target_metric": "accuracy",
                "strategy_params": {
                    "mcts_simulations": 10,
                    "mcts_max_depth": 2,
                    "prompt_candidates_per_node": 1,
                    "feedback_enabled": False,
                },
            },
            "test_case_ids": [case["id"]],
            "name": "Generic Jobs Running Cancel",
        },
    )

    jobs_manager = jobs_worker._jobs_manager()
    queued_job = _get_job_by_id(job_id)
    acquired_job = jobs_manager.acquire_next_job(
        domain="prompt_studio",
        queue=str(queued_job["queue"]),
        lease_seconds=60,
        worker_id="generic-cancel-test",
    )
    assert acquired_job is not None
    assert str(acquired_job["uuid"]) == str(queued_job["uuid"])

    worker_task = asyncio.create_task(
        jobs_worker._handle_job(acquired_job, job_manager=jobs_manager)
    )
    try:
        provider_started = await asyncio.to_thread(provider_entered.wait, 2.0)
        assert provider_started is True
        assert jobs_manager.cancel_job(
            int(acquired_job["id"]),
            reason="generic admin cancellation",
        )
        cancelled_job = jobs_manager.get_job(int(acquired_job["id"])) or {}
        assert str(cancelled_job.get("status")).lower() == "cancelled"
    except BaseException:
        release_provider.set()
        with contextlib.suppress(BaseException):
            await worker_task
        raise
    release_provider.set()
    with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
        await worker_task

    optimization = db.get_optimization(optimization_id) or {}
    assert str(optimization.get("status")).lower() == "cancelled"
    assert exc_info.value.failure_code == "job_cancelled"
    assert exc_info.value.retryable is False
    terminal_job = jobs_manager.get_job(int(acquired_job["id"])) or {}
    assert str(terminal_job.get("status")).lower() == "cancelled"
    assert len(adapter_requests) == 1
    assert adapter_requests[0]["api_key"] == "optimization-runtime-key"
    runtime = optimization_runtime_boundary["instances"][0]
    assert runtime.mark_calls == [runtime.handle]
    assert runtime.close_count == 1
    assert optimization_runtime_boundary["completed_events"] == []


@pytest.mark.asyncio
async def test_cancelled_queued_job_reconciles_prompt_optimization_without_provider_work(
    prompt_studio_dual_backend_db,
    monkeypatch,
    optimization_runtime_boundary,
):
    async def allow_project_write_access(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ps_opt_endpoints,
        "require_project_write_access",
        allow_project_write_access,
        raising=True,
    )

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_prompt_and_case(db)
    caller_thread_id = threading.get_ident()
    close_thread_ids: list[int] = []
    original_close_connection = db.close_connection

    def record_close_connection() -> None:
        close_thread_ids.append(threading.get_ident())
        original_close_connection()

    monkeypatch.setattr(
        db,
        "close_connection",
        record_close_connection,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )
    optimization_id, job_id, _body = await _create_optimization_via_endpoint(
        db,
        {
            "project_id": project["id"],
            "initial_prompt_id": prompt["id"],
            "optimization_config": {
                "optimizer_type": "mcts",
                "max_iterations": 2,
                "target_metric": "accuracy",
            },
            "test_case_ids": [case["id"]],
            "name": "Generic Jobs Queued Cancel",
        },
    )

    jobs_manager = jobs_worker._jobs_manager()
    queued_job = _get_job_by_id(job_id)
    assert jobs_manager.cancel_job(int(queued_job["id"]), reason="queued admin cancel")
    before_reconcile = db.get_optimization(optimization_id) or {}
    assert str(before_reconcile.get("status")).lower() == "pending"

    reconciled = await jobs_worker._reconcile_cancelled_optimization_jobs(jobs_manager)

    after_reconcile = db.get_optimization(optimization_id) or {}
    assert reconciled == 1
    assert str(after_reconcile.get("status")).lower() == "cancelled"
    assert any(thread_id != caller_thread_id for thread_id in close_thread_ids)
    assert optimization_runtime_boundary["instances"] == []
    assert optimization_runtime_boundary["adapter_requests"] == []
