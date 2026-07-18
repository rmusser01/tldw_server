"""Provider/model propagation regressions across Prompt Studio optimizers."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as prompt_executor_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import (
    MCTSOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import (
    BootstrapOptimizer,
    MIPROOptimizer,
    OptimizationEngine,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_strategies import (
    IterativeRefinementOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
    TestRunner,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.types_common import (
    MetricType,
)

pytestmark = pytest.mark.unit

_CANONICAL_CONFIG = {
    "provider": "bedrock",
    "model": "anthropic.claude-3-haiku",
    "parameters": {
        "temperature": 0.29,
        "top_p": 0.61,
        "max_tokens": 96,
        "timeout_seconds": 13,
    },
}


class _BoundaryDb:
    """Smallest DB contract required by optimizer-owned PromptExecutor instances."""

    client_id = "optimizer-provider-boundary"


class _EngineDb:
    client_id = "optimizer-propagation"

    def __init__(self, strategy: str) -> None:
        self.row: dict[str, Any] = {
            "id": 77,
            "project_id": 7,
            "initial_prompt_id": 12,
            "optimizer_type": strategy,
            "optimization_config": {
                "optimizer_type": strategy,
                "target_metric": "accuracy",
                "model_configuration": {
                    "provider": "AWS_BEDROCK",
                    "model_name": "anthropic.claude-3-haiku",
                    "temperature": 0.29,
                    "timeout_seconds": 13,
                    "parameters": {"top_p": 0.61, "max_tokens": 96},
                },
            },
            "test_case_ids": [3],
            "max_iterations": 1,
            "status": "pending",
        }

    def get_optimization(self, optimization_id: int, **_kwargs: Any) -> dict[str, Any] | None:
        return copy.deepcopy(self.row) if optimization_id == self.row["id"] else None

    def update_optimization(self, optimization_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        assert optimization_id == self.row["id"]
        self.row.update(copy.deepcopy(updates))
        return copy.deepcopy(self.row)

    def set_optimization_status(
        self,
        optimization_id: int,
        status: str,
        **_kwargs: Any,
    ) -> None:
        assert optimization_id == self.row["id"]
        self.row["status"] = status

    def complete_optimization(self, optimization_id: int, **updates: Any) -> None:
        assert optimization_id == self.row["id"]
        self.row.update(copy.deepcopy(updates))
        self.row["status"] = "completed"


class _RecordingRunner:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.model_configs: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    async def run_single_test(self, *, model_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.model_configs.append(copy.deepcopy(model_config))
        self.calls.append({"model_config": copy.deepcopy(model_config), **kwargs})
        on_provider_success = kwargs.get("on_provider_success")
        if on_provider_success is not None:
            await on_provider_success()
        return {
            "success": True,
            "inputs": {"question": "q"},
            "expected_outputs": {"response": "a"},
            "actual_output": {"response": "wrong"},
            "scores": {
                "accuracy": self.score,
                "aggregate_score": self.score,
            },
        }


class _CapturingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def _call_llm(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(kwargs))
        on_provider_success = kwargs.get("on_provider_success")
        if on_provider_success is not None:
            await on_provider_success()
        return {"content": f"candidate for {kwargs['model']}", "tokens": 1}


class _ConcurrentBaselineRunner(_RecordingRunner):
    def __init__(self, initial_prompt_ids: set[int]) -> None:
        super().__init__()
        self.initial_prompt_ids = initial_prompt_ids
        self.baseline_arrivals = 0
        self.baselines_ready = asyncio.Event()
        self.lock = asyncio.Lock()

    async def run_single_test(
        self,
        *,
        prompt_id: int,
        model_config: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if prompt_id in self.initial_prompt_ids:
            async with self.lock:
                self.baseline_arrivals += 1
                if self.baseline_arrivals == len(self.initial_prompt_ids):
                    self.baselines_ready.set()
            await asyncio.wait_for(self.baselines_ready.wait(), timeout=1.0)
        return await super().run_single_test(
            prompt_id=prompt_id,
            model_config=model_config,
            **kwargs,
        )


def _assert_no_retained_provider_state(optimizer: Any) -> None:
    retained = {
        name: value
        for name, value in vars(optimizer).items()
        if "model_config" in name.lower()
        or "credential" in name.lower()
        or "provider_success" in name.lower()
        or ("provider" in name.lower() and "config" in name.lower())
        or ("runtime" in name.lower() and "config" in name.lower())
    }
    assert all(value is None or value == {} for value in retained.values()), retained


@pytest.fixture
def optimizer_db(tmp_path: Path) -> PromptStudioDatabase:
    return PromptStudioDatabase(str(tmp_path / "optimizer-provider-propagation.db"), client_id="provider-propagation")


def _seed_prompt_and_case(db: PromptStudioDatabase, label: str = "Base") -> tuple[int, int]:
    project = db.create_project(name=f"Provider propagation {label}", description="")
    prompt = db.create_prompt(
        project_id=int(project["id"]),
        name=label,
        system_prompt=f"SYSTEM_{label}: Return a precise answer in the requested format.",
        user_prompt="Answer {question}",
        version_number=1,
    )
    case = db.create_test_case(
        project_id=int(project["id"]),
        name=f"Case {label}",
        inputs={"question": "q"},
        expected_outputs={"response": "a"},
    )
    return int(prompt["id"]), int(case["id"])


@pytest.mark.parametrize("strategy", ["mipro", "bootstrap", "mcts"])
@pytest.mark.asyncio
async def test_engine_normalizes_and_propagates_selected_config_to_each_strategy(
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _EngineDb(strategy)
    engine = OptimizationEngine(db)  # type: ignore[arg-type]
    strategy_optimizer = getattr(engine, strategy)
    optimize = AsyncMock(
        return_value={
            "optimized_prompt_id": 12,
            "initial_score": 0.5,
            "final_score": 0.6,
            "improvement": 0.1,
            "iterations": 1,
        }
    )
    monkeypatch.setattr(strategy_optimizer, "optimize", optimize, raising=True)

    await engine.optimize(77)

    assert optimize.await_args.kwargs["model_config"] == _CANONICAL_CONFIG


@pytest.mark.asyncio
async def test_engine_lost_start_status_cas_exits_before_optimizer_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CancelledAtStartDb(_EngineDb):
        def set_optimization_status(
            self,
            optimization_id: int,
            status: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert optimization_id == self.row["id"]
            assert status == "running"
            self.row["status"] = "cancelled"
            self.row["error_message"] = "cancelled during start transition"
            return copy.deepcopy(self.row)

    db = _CancelledAtStartDb("mipro")
    engine = OptimizationEngine(db)  # type: ignore[arg-type]
    optimize = AsyncMock(
        return_value={
            "optimized_prompt_id": 12,
            "initial_score": 0.5,
            "final_score": 0.6,
            "improvement": 0.1,
            "iterations": 1,
        }
    )
    monkeypatch.setattr(engine.mipro, "optimize", optimize, raising=True)

    result = await engine.optimize(77)

    optimize.assert_not_awaited()
    assert result == {
        "optimization_id": 77,
        "optimized_prompt_id": 12,
        "iterations": 0,
        "status": "cancelled",
    }


@pytest.mark.parametrize("strategy", ["mipro", "bootstrap", "mcts"])
@pytest.mark.asyncio
async def test_strategy_provider_boundaries_are_strict_and_mark_valid_success(
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _RecordingRunner()
    db = _BoundaryDb()
    marks: list[str] = []

    async def _mark_success() -> None:
        marks.append(strategy)

    if strategy == "mipro":
        optimizer = MIPROOptimizer(db, runner)  # type: ignore[arg-type]
        await optimizer.optimize(
            initial_prompt_id=12,
            test_case_ids=[3],
            model_config=_CANONICAL_CONFIG,
            max_iterations=0,
            target_metric=MetricType.ACCURACY,
            on_provider_success=_mark_success,
        )
    elif strategy == "mcts":
        optimizer = MCTSOptimizer(db, runner)  # type: ignore[arg-type]
        monkeypatch.setattr(
            optimizer,
            "_get_prompt",
            lambda _prompt_id: {
                "id": 12,
                "project_id": 7,
                "name": "Base",
                "system_prompt": "System",
                "user_prompt": "User",
            },
            raising=True,
        )
        monkeypatch.setattr(optimizer.decomposer, "decompose_text", lambda _text: [], raising=True)

        async def _feedback(**_kwargs: Any) -> tuple[float, int]:
            return 0.9, 12

        monkeypatch.setattr(optimizer, "_evaluate_with_feedback", _feedback, raising=True)
        await optimizer.optimize(
            initial_prompt_id=12,
            test_case_ids=[3],
            model_config=_CANONICAL_CONFIG,
            max_iterations=1,
            target_metric=MetricType.ACCURACY,
            strategy_params={
                "mcts_simulations": 1,
                "mcts_max_depth": 1,
                "feedback_enabled": False,
            },
            on_provider_success=_mark_success,
        )
    else:
        optimizer = BootstrapOptimizer(db, runner)  # type: ignore[arg-type]

        async def _create_prompt(_prompt_id: int, _examples: list[dict[str, Any]]) -> int:
            return 13

        monkeypatch.setattr(optimizer, "_create_prompt_with_examples", _create_prompt, raising=True)
        await optimizer.optimize(
            prompt_id=12,
            test_case_ids=[3],
            model_config=_CANONICAL_CONFIG,
            num_examples=1,
            selection_strategy="best",
            on_provider_success=_mark_success,
        )

    assert runner.calls
    assert marks == [strategy] * len(runner.calls)
    for call in runner.calls:
        assert call["model_config"] == _CANONICAL_CONFIG
        assert call["strict_provider_errors"] is True
        assert call["on_provider_success"] is _mark_success


@pytest.mark.asyncio
async def test_concurrent_mipro_public_runs_keep_provider_snapshots_and_callbacks_isolated(
    optimizer_db: PromptStudioDatabase,
) -> None:
    first_prompt, first_case = _seed_prompt_and_case(optimizer_db, "A")
    second_prompt, second_case = _seed_prompt_and_case(optimizer_db, "B")
    runner = _ConcurrentBaselineRunner({first_prompt, second_prompt})
    executor = _CapturingExecutor()
    optimizer = MIPROOptimizer(optimizer_db, runner)  # type: ignore[arg-type]
    optimizer.executor = executor  # type: ignore[assignment]
    first_config = {
        "provider": "anthropic",
        "model": "claude-a",
        "parameters": {"top_p": 0.31, "timeout_seconds": 11},
        "api_key": "runtime-key-a",
        "app_config": {
            "anthropic_api": {
                "model": "claude-a",
                "api_key": "runtime-key-a",
                "runtime_marker": "first",
            }
        },
        "credentials_resolved": True,
    }
    second_config = {
        "provider": "bedrock",
        "model": "claude-b",
        "parameters": {"top_p": 0.79, "timeout_seconds": 19},
        "api_key": None,
        "app_config": {
            "bedrock_api": {
                "model": "claude-b",
                "aws_access_key_id": "runtime-access-b",
                "aws_secret_access_key": "runtime-secret-b",
                "runtime_marker": "second",
            }
        },
        "credentials_resolved": True,
    }
    first_marks: list[str] = []
    second_marks: list[str] = []

    async def _mark_first() -> None:
        first_marks.append("first")

    async def _mark_second() -> None:
        second_marks.append("second")

    await asyncio.gather(
        optimizer.optimize(
            initial_prompt_id=first_prompt,
            test_case_ids=[first_case],
            model_config=first_config,
            max_iterations=1,
            on_provider_success=_mark_first,
        ),
        optimizer.optimize(
            initial_prompt_id=second_prompt,
            test_case_ids=[second_case],
            model_config=second_config,
            max_iterations=1,
            on_provider_success=_mark_second,
        ),
    )

    first_llm_calls = [call for call in executor.calls if "SYSTEM_A" in call["prompt"]]
    second_llm_calls = [call for call in executor.calls if "SYSTEM_B" in call["prompt"]]
    assert first_llm_calls and second_llm_calls
    assert len(first_llm_calls) + len(second_llm_calls) == len(executor.calls)
    for call in first_llm_calls:
        assert call["provider"] == "anthropic"
        assert call["model"] == "claude-a"
        assert call["parameters"]["top_p"] == 0.31
        assert call["api_key_override"] == "runtime-key-a"
        assert call["app_config"] == first_config["app_config"]
        assert call["credentials_resolved"] is True
        assert call["timeout_seconds"] == 11
        assert call["on_provider_success"] is _mark_first
    for call in second_llm_calls:
        assert call["provider"] == "bedrock"
        assert call["model"] == "claude-b"
        assert call["parameters"]["top_p"] == 0.79
        assert call["api_key_override"] is None
        assert call["app_config"] == second_config["app_config"]
        assert call["credentials_resolved"] is True
        assert call["timeout_seconds"] == 19
        assert call["on_provider_success"] is _mark_second

    for call in runner.calls:
        expected_config, expected_callback = (
            (first_config, _mark_first)
            if call["test_case_id"] == first_case
            else (second_config, _mark_second)
        )
        assert call["model_config"] == expected_config
        assert call["strict_provider_errors"] is True
        assert call["on_provider_success"] is expected_callback
    assert first_marks and second_marks
    _assert_no_retained_provider_state(optimizer)


@pytest.mark.asyncio
async def test_cancelled_mipro_public_run_does_not_retain_model_or_runtime_config(
    optimizer_db: PromptStudioDatabase,
) -> None:
    prompt_id, case_id = _seed_prompt_and_case(optimizer_db, "cancel")
    runner = _RecordingRunner()
    optimizer = MIPROOptimizer(optimizer_db, runner)  # type: ignore[arg-type]
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingExecutor:
        async def _call_llm(self, **_kwargs: Any) -> dict[str, Any]:
            entered.set()
            await release.wait()
            return {"content": "late candidate", "tokens": 1}

    optimizer.executor = _BlockingExecutor()  # type: ignore[assignment]
    task = asyncio.create_task(
        optimizer.optimize(
            initial_prompt_id=prompt_id,
            test_case_ids=[case_id],
            model_config=_CANONICAL_CONFIG,
            max_iterations=1,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_no_retained_provider_state(optimizer)


@pytest.mark.asyncio
async def test_mcts_routes_candidate_evaluation_and_distinct_scorer_through_one_runtime(
    optimizer_db: PromptStudioDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id, case_id = _seed_prompt_and_case(optimizer_db)
    runner = TestRunner(optimizer_db)
    optimizer = MCTSOptimizer(optimizer_db, runner)
    llm_boundary_calls: list[dict[str, Any]] = []
    internal_adapter_calls: list[dict[str, Any]] = []
    adapter_calls: list[dict[str, Any]] = []
    marks: list[str] = []

    async def _mark_success() -> None:
        marks.append("mark")

    def _call_adapter(**kwargs: Any) -> dict[str, Any]:
        adapter_calls.append(copy.deepcopy(kwargs))
        return {
            "choices": [{"message": {"content": "a"}}],
            "usage": {"total_tokens": 2},
        }

    class _InternalAdapter:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        def chat(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            internal_adapter_calls.append(
                {
                    "provider": self.provider,
                    "request": copy.deepcopy(request),
                    "timeout": timeout,
                }
            )
            prompt_text = str(request.get("messages") or "")
            content = "9" if "Rate the clarity" in prompt_text else "Improved system prompt"
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 2},
            }

    class _InternalRegistry:
        def get_adapter(self, provider: str) -> _InternalAdapter:
            return _InternalAdapter(provider)

    original_call_llm = optimizer.executor._call_llm

    async def _capture_llm_boundary(**kwargs: Any) -> dict[str, Any]:
        llm_boundary_calls.append(copy.deepcopy(kwargs))
        return await original_call_llm(**kwargs)

    monkeypatch.setattr(runner, "_call_adapter", _call_adapter, raising=True)
    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _InternalRegistry(), raising=True)
    monkeypatch.setattr(optimizer.executor, "_call_llm", _capture_llm_boundary, raising=True)
    monkeypatch.setattr(optimizer.decomposer, "decompose_text", lambda _text: ["segment"], raising=True)

    await optimizer.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=[case_id],
        model_config=_CANONICAL_CONFIG,
        max_iterations=1,
        target_metric=MetricType.ACCURACY,
        strategy_params={
            "mcts_simulations": 1,
            "mcts_max_depth": 1,
            "prompt_candidates_per_node": 1,
            "scorer_model": "anthropic.claude-3-sonnet",
            "feedback_enabled": False,
        },
        on_provider_success=_mark_success,
    )

    assert len(llm_boundary_calls) >= 2
    candidate_calls = [
        call for call in llm_boundary_calls if not str(call["prompt"]).startswith("Rate the clarity")
    ]
    scorer_calls = [call for call in llm_boundary_calls if str(call["prompt"]).startswith("Rate the clarity")]
    assert candidate_calls and scorer_calls
    for call in candidate_calls:
        assert call["provider"] == _CANONICAL_CONFIG["provider"]
        assert call["model"] == _CANONICAL_CONFIG["model"]
        assert call["parameters"]["top_p"] == 0.61
        assert call["on_provider_success"] is _mark_success
        assert call["timeout_seconds"] == 13
    for call in scorer_calls:
        assert call["provider"] == _CANONICAL_CONFIG["provider"]
        assert call["model"] == "anthropic.claude-3-sonnet"
        assert call["parameters"]["top_p"] == 0.61
        assert call["on_provider_success"] is _mark_success
        assert call["timeout_seconds"] == 13

    assert internal_adapter_calls
    for call in internal_adapter_calls:
        assert call["provider"] == _CANONICAL_CONFIG["provider"]
        assert call["timeout"] == 13
        model = call["request"]["model"]
        assert model in {_CANONICAL_CONFIG["model"], "anthropic.claude-3-sonnet"}

    assert adapter_calls
    for call in adapter_calls:
        assert call["provider"] == _CANONICAL_CONFIG["provider"]
        assert call["model"] == _CANONICAL_CONFIG["model"]
        assert call["timeout_seconds"] == 13
    assert marks


@pytest.mark.asyncio
async def test_iterative_refinement_generation_and_evaluation_use_selected_provider_config(
    optimizer_db: PromptStudioDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id, case_id = _seed_prompt_and_case(optimizer_db)
    runner = _RecordingRunner(score=0.1)
    optimizer = IterativeRefinementOptimizer(optimizer_db, runner)  # type: ignore[arg-type]
    llm_calls: list[dict[str, Any]] = []
    marks: list[str] = []

    async def _mark_success() -> None:
        marks.append("mark")

    async def _call_llm(**kwargs: Any) -> dict[str, Any]:
        llm_calls.append(copy.deepcopy(kwargs))
        on_provider_success = kwargs.get("on_provider_success")
        if on_provider_success is not None:
            await on_provider_success()
        return {"content": "Add an explicit output constraint.", "tokens": 2}

    monkeypatch.setattr(optimizer.executor, "_call_llm", _call_llm, raising=True)

    await optimizer.optimize(
        prompt_id=prompt_id,
        test_case_ids=[case_id],
        model_config=_CANONICAL_CONFIG,
        max_iterations=1,
        on_provider_success=_mark_success,
    )

    assert runner.calls
    for call in runner.calls:
        assert call["model_config"] == _CANONICAL_CONFIG
        assert call["strict_provider_errors"] is True
        assert call["on_provider_success"] is _mark_success
    assert len(llm_calls) == 1
    assert llm_calls[0]["provider"] == _CANONICAL_CONFIG["provider"]
    assert llm_calls[0]["model"] == _CANONICAL_CONFIG["model"]
    assert llm_calls[0]["parameters"]["top_p"] == _CANONICAL_CONFIG["parameters"]["top_p"]
    assert llm_calls[0]["timeout_seconds"] == 13
    assert llm_calls[0]["on_provider_success"] is _mark_success
    assert marks


@pytest.mark.asyncio
async def test_public_mcts_post_baseline_refiner_uses_authoritative_non_openai_runtime(
    optimizer_db: PromptStudioDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id, case_id = _seed_prompt_and_case(optimizer_db)
    events: list[str] = []
    runtime_config = {
        "provider": "anthropic",
        "model": "claude-3-5-haiku-runtime",
        "parameters": {
            "temperature": 0.17,
            "top_p": 0.72,
            "max_tokens": 80,
            "timeout_seconds": 23,
        },
        "api_key": "runtime-anthropic-key",
        "app_config": {
            "anthropic_api": {
                "model": "claude-3-5-haiku-runtime",
                "api_key": "runtime-anthropic-key",
                "base_url": "https://runtime-anthropic.example/v1",
                "runtime_marker": "post-baseline-refiner",
            }
        },
        "credentials_resolved": True,
    }

    class _EventRunner(_RecordingRunner):
        async def run_single_test(self, **kwargs: Any) -> dict[str, Any]:
            events.append("evaluation")
            return await super().run_single_test(**kwargs)

    runner = _EventRunner(score=0.1)
    optimizer = MCTSOptimizer(optimizer_db, runner)  # type: ignore[arg-type]
    optimizer._refiner_cls = IterativeRefinementOptimizer
    refiner_boundary_calls: list[dict[str, Any]] = []
    refiner_adapter_calls: list[dict[str, Any]] = []
    marks: list[str] = []

    async def _mark_success() -> None:
        marks.append("mark")

    class _Adapter:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        def chat(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
            request_text = str(request.get("messages") or "")
            if "Analyze these errors" in request_text:
                events.append("refiner_adapter")
                refiner_adapter_calls.append(
                    {
                        "provider": self.provider,
                        "request": copy.deepcopy(request),
                        "timeout": timeout,
                    }
                )
                content = "Add an explicit response-format constraint."
            else:
                content = "Improved candidate system prompt."
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 2},
            }

    class _Registry:
        def get_adapter(self, provider: str) -> _Adapter:
            return _Adapter(provider)

    original_call_llm = prompt_executor_module.PromptExecutor._call_llm

    async def _capture_refiner_boundary(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if "Analyze these errors" in str(kwargs.get("prompt") or ""):
            refiner_boundary_calls.append(copy.deepcopy(kwargs))
        return await original_call_llm(self, *args, **kwargs)

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry(), raising=True)
    monkeypatch.setattr(
        prompt_executor_module.PromptExecutor,
        "_call_llm",
        _capture_refiner_boundary,
        raising=True,
    )
    monkeypatch.setattr(optimizer.decomposer, "decompose_text", lambda _text: ["segment"], raising=True)

    await optimizer.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=[case_id],
        model_config=runtime_config,
        max_iterations=1,
        target_metric=MetricType.ACCURACY,
        strategy_params={
            "mcts_simulations": 1,
            "mcts_max_depth": 1,
            "prompt_candidates_per_node": 1,
            "feedback_enabled": True,
            "feedback_threshold": 10.0,
            "feedback_max_retries": 1,
        },
        on_provider_success=_mark_success,
    )

    assert refiner_boundary_calls
    for call in refiner_boundary_calls:
        assert call["provider"] == "anthropic"
        assert call["model"] == "claude-3-5-haiku-runtime"
        assert call["api_key_override"] == "runtime-anthropic-key"
        assert call["app_config"] == runtime_config["app_config"]
        assert call["credentials_resolved"] is True
        assert call["on_provider_success"] is _mark_success
        assert call["timeout_seconds"] == 23

    assert refiner_adapter_calls
    for call in refiner_adapter_calls:
        assert call["provider"] == "anthropic"
        assert call["request"]["model"] == "claude-3-5-haiku-runtime"
        assert call["request"]["api_key"] == "runtime-anthropic-key"
        assert call["request"]["app_config"] == runtime_config["app_config"]
        assert call["request"]["credentials_resolved"] is True
        assert call["timeout"] == 23

    assert events[0] == "evaluation"
    assert events.index("evaluation") < events.index("refiner_adapter")
    assert marks
