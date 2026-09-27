"""Prompt Studio durable-worker resource and lifecycle regressions."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_config import build_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    optimization_engine as optimization_engine_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as prompt_executor_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    test_runner as test_runner_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import (
    JobProcessor,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import (
    MCTSOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import (
    PromptExecutor,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
    TestRunner,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.types_common import (
    MetricType,
)

pytestmark = pytest.mark.unit

_MODEL_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "parameters": {"temperature": 0.1, "max_tokens": 32},
}


async def _issue_provider_credentials(
    provider: str,
    *,
    endpoint: str | None = None,
    api_key: str = "runtime-key",
) -> ProviderCallCredentials:
    """Issue one authentic, execution-only provider credential handle."""

    credential_fields = {"base_url": endpoint} if endpoint is not None else {}
    app_config = build_app_config_overrides(provider, credential_fields)

    async def _resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=app_config,
            credential_fields=credential_fields,
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
        )

    runtime = ProviderCredentialRuntime(
        user_id=41,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=_resolver,
    )
    try:
        return await runtime.resolve(provider)
    finally:
        await runtime.close()


def _new_db(tmp_path: Path, name: str) -> PromptStudioDatabase:
    return PromptStudioDatabase(
        str(tmp_path / f"{name}.db"),
        client_id=f"lifecycle-{name}",
    )


def _create_project_resources(
    db: PromptStudioDatabase,
    name: str,
) -> tuple[int, int, int]:
    project = db.create_project(name=f"Project {name}", description="")
    project_id = int(project["id"])
    prompt = db.create_prompt(
        project_id=project_id,
        name=f"Prompt {name}",
        system_prompt="Answer precisely.",
        user_prompt="Question: {question}",
    )
    test_case = db.create_test_case(
        project_id=project_id,
        name=f"Case {name}",
        inputs={"question": name},
        expected_outputs={"response": name},
    )
    return project_id, int(prompt["id"]), int(test_case["id"])


def _create_optimization(
    db: PromptStudioDatabase,
    *,
    project_id: int,
    prompt_id: int | None,
    test_case_ids: list[int],
    name: str,
    status: str = "pending",
) -> int:
    optimization = db.create_optimization(
        project_id=project_id,
        name=f"Optimization {name}",
        initial_prompt_id=prompt_id,
        optimizer_type="mipro",
        optimization_config={
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": dict(_MODEL_CONFIG),
        },
        max_iterations=1,
        status=status,
    )
    optimization_id = int(optimization["id"])
    db.update_optimization(
        optimization_id,
        {"test_case_ids": list(test_case_ids)},
    )
    return optimization_id


def _optimization_payload(
    *,
    optimization_id: int,
    prompt_id: int | None,
    test_case_ids: list[int],
    optimization_uuid: str | None = None,
) -> dict[str, Any]:
    payload = {
        "optimization_id": optimization_id,
        "initial_prompt_id": prompt_id,
        "test_case_ids": list(test_case_ids),
        "optimizer_type": "mipro",
        "max_iterations": 1,
        "optimization_config": {
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_config": dict(_MODEL_CONFIG),
        },
    }
    if optimization_uuid is not None:
        payload["optimization_uuid"] = optimization_uuid
    return payload


def _create_evaluation(
    db: PromptStudioDatabase,
    *,
    project_id: int,
    prompt_id: int,
    test_case_ids: list[int],
    model_configs: list[dict[str, Any]],
) -> int:
    evaluation = db.create_evaluation(
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=test_case_ids,
        model_configs=model_configs,  # type: ignore[arg-type]
        status="pending",
    )
    return int(evaluation["id"])


def _install_dispatch_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    dispatched: list[int] = []

    async def _optimize(
        _self: Any,
        optimization_id: int,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        dispatched.append(optimization_id)
        return {
            "optimization_id": optimization_id,
            "status": "pending",
            "iterations_completed": 0,
        }

    monkeypatch.setattr(
        optimization_engine_module.OptimizationEngine,
        "optimize",
        _optimize,
        raising=True,
    )
    return dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "endpoint"),
    [
        ("local-llm", "http://127.0.0.1:8080/v1"),
        ("custom-openai-api-2", "https://custom.example/v1"),
    ],
)
async def test_prompt_executor_attaches_authentic_runtime_handle_at_adapter_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    endpoint: str,
) -> None:
    handle = await _issue_provider_credentials(
        provider,
        endpoint=endpoint,
        api_key="snapshot-key",
    )
    captured: list[dict[str, Any]] = []

    class _Adapter:
        @staticmethod
        def chat(
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            captured.append(request)
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

        @staticmethod
        def is_local_provider_name(candidate: str) -> bool:
            return candidate == "local-llm"

    db = _new_db(tmp_path, f"adapter-handle-{provider}")
    monkeypatch.setattr(
        prompt_executor_module,
        "get_registry",
        lambda: _Registry(),
        raising=True,
    )

    try:
        result = await PromptExecutor(db)._call_llm(
            provider=provider,
            model="snapshot-model",
            prompt="hello",
            api_key_override="loose-key",
            app_config={"attacker": {"base_url": "https://attacker.example/v1"}},
            credentials_resolved=True,
            provider_credentials=handle,
        )

        assert result["content"] == "ok"
        assert len(captured) == 1
        assert captured[0][PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_kind", ["forged", "wrong-provider"])
async def test_prompt_executor_rejects_inauthentic_or_mismatched_runtime_handle(
    tmp_path: Path,
    credential_kind: str,
) -> None:
    provider_credentials: object
    if credential_kind == "wrong-provider":
        provider_credentials = await _issue_provider_credentials("openai")
    else:
        provider_credentials = object()
    db = _new_db(tmp_path, f"rejected-handle-{credential_kind}")

    try:
        with pytest.raises(ChatConfigurationError):
            PromptExecutor(db)._build_adapter_request(
                provider="local-llm",
                model="snapshot-model",
                messages=[{"role": "user", "content": "hello"}],
                system_prompt=None,
                temperature=0.1,
                max_tokens=32,
                params={},
                app_config={},
                api_key_override="loose-key",
                credentials_resolved=True,
                provider_credentials=provider_credentials,
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_test_runner_attaches_authentic_runtime_handle_at_adapter_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = await _issue_provider_credentials(
        "local-llm",
        endpoint="http://127.0.0.1:8080/v1",
    )
    captured: list[dict[str, Any]] = []

    class _Adapter:
        @staticmethod
        def chat(
            request: dict[str, Any],
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            captured.append(request)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        test_runner_module,
        "get_adapter_or_raise",
        lambda _provider: _Adapter(),
        raising=True,
    )
    db = _new_db(tmp_path, "test-runner-adapter-handle")

    try:
        TestRunner(db)._call_adapter(
            provider="local-llm",
            model="snapshot-model",
            messages_payload=[{"role": "user", "content": "hello"}],
            system_message=None,
            temperature=0.1,
            max_tokens=32,
            app_config=handle.app_config,
            api_key_override=handle.api_key,
            credentials_resolved=True,
            provider_credentials=handle,
        )

        assert len(captured) == 1
        assert captured[0][PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mcts_forwards_primary_handle_to_expansion_without_success_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "mcts-primary-handle-no-callback")
    _project_id, prompt_id, test_case_id = _create_project_resources(
        db,
        "mcts-primary-handle-no-callback",
    )
    handle = await _issue_provider_credentials(
        "custom-openai-api-2",
        endpoint="https://custom.example/v1",
    )
    optimizer = MCTSOptimizer(db, TestRunner(db))
    captured_expansion: dict[str, Any] = {}

    async def _evaluate_prompt(*_args: Any, **_kwargs: Any) -> float:
        return 0.5

    async def _expand_node(
        _node: Any,
        **kwargs: Any,
    ) -> None:
        captured_expansion.update(kwargs)
        return None

    async def _evaluate_with_feedback(**_kwargs: Any) -> tuple[float, int]:
        return 0.5, prompt_id

    monkeypatch.setattr(
        optimizer.decomposer,
        "decompose_text",
        lambda _text: ["one segment"],
        raising=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_evaluate_prompt",
        _evaluate_prompt,
        raising=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_expand_node",
        _expand_node,
        raising=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_evaluate_with_feedback",
        _evaluate_with_feedback,
        raising=True,
    )

    try:
        await optimizer.optimize(
            initial_prompt_id=prompt_id,
            test_case_ids=[test_case_id],
            model_config={
                "provider": "custom-openai-api-2",
                "model": "snapshot-model",
                "parameters": {},
            },
            max_iterations=1,
            target_metric=MetricType.ACCURACY,
            strategy_params={
                "mcts_simulations": 1,
                "mcts_max_depth": 1,
                "prompt_candidates_per_node": 1,
                "feedback_enabled": False,
            },
            provider_credentials=handle,
            on_provider_success=None,
        )

        assert captured_expansion["provider_credentials"] is handle
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mcts_scorer_falls_back_to_primary_runtime_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "mcts-primary-scorer-handle")
    handle = await _issue_provider_credentials(
        "custom-openai-api-2",
        endpoint="https://custom.example/v1",
    )
    optimizer = MCTSOptimizer(db, TestRunner(db))
    optimizer._counters = {}
    optimizer._debug_top_by_depth = None
    captured_scorer: dict[str, Any] = {}

    async def _propose_candidates(
        system_so_far: str,
        _segment_text: str,
        _count: int,
        **_kwargs: Any,
    ) -> list[str]:
        return [f"{system_so_far}\nCandidate"]

    async def _score_prompt_async(**kwargs: Any) -> float:
        captured_scorer.update(kwargs)
        return 8.0

    monkeypatch.setattr(
        optimizer,
        "_propose_candidates",
        _propose_candidates,
        raising=True,
    )
    monkeypatch.setattr(
        optimizer.scorer,
        "score_prompt_async",
        _score_prompt_async,
        raising=True,
    )

    try:
        node = MCTSOptimizer._Node(
            parent=None,
            segment_index=0,
            system_text="System",
        )
        await optimizer._expand_node(
            node,
            segment="segment",
            base_user="question",
            k_candidates=1,
            score_bin_size=0.5,
            min_quality=0.0,
            model_config={
                "provider": "custom-openai-api-2",
                "model": "snapshot-model",
                "parameters": {},
            },
            scorer_model="snapshot-model",
            provider_credentials=handle,
            scorer_model_config=None,
            scorer_provider_credentials=None,
            on_provider_success=None,
            on_scorer_provider_success=None,
        )

        assert captured_scorer["provider_credentials"] is handle
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mcts_primary_handle_without_callback_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "mcts-primary-handle-strict")
    _project_id, prompt_id, test_case_id = _create_project_resources(
        db,
        "mcts-primary-handle-strict",
    )
    handle = await _issue_provider_credentials(
        "custom-openai-api-2",
        endpoint="https://custom.example/v1",
    )
    optimizer = MCTSOptimizer(db, TestRunner(db))

    async def _run_single_test(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("primary provider failed")

    async def _evaluate_with_feedback(**_kwargs: Any) -> tuple[float, int]:
        return 0.0, prompt_id

    monkeypatch.setattr(
        optimizer.decomposer,
        "decompose_text",
        lambda _text: [],
        raising=True,
    )
    monkeypatch.setattr(
        optimizer.test_runner,
        "run_single_test",
        _run_single_test,
        raising=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_evaluate_with_feedback",
        _evaluate_with_feedback,
        raising=True,
    )

    try:
        with pytest.raises(RuntimeError, match="primary provider failed"):
            await optimizer.optimize(
                initial_prompt_id=prompt_id,
                test_case_ids=[test_case_id],
                model_config={
                    "provider": "custom-openai-api-2",
                    "model": "snapshot-model",
                    "parameters": {},
                },
                max_iterations=1,
                target_metric=MetricType.ACCURACY,
                strategy_params={
                    "mcts_simulations": 1,
                    "feedback_enabled": False,
                },
                provider_credentials=handle,
                on_provider_success=None,
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_mcts_scorer_handle_without_callback_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "mcts-scorer-handle-strict")
    _project_id, prompt_id, test_case_id = _create_project_resources(
        db,
        "mcts-scorer-handle-strict",
    )
    handle = await _issue_provider_credentials(
        "custom-openai-api-2",
        endpoint="https://custom.example/v1",
    )
    optimizer = MCTSOptimizer(db, TestRunner(db))

    async def _evaluate_prompt(*_args: Any, **_kwargs: Any) -> float:
        return 0.5

    async def _propose_candidates(
        system_so_far: str,
        _segment_text: str,
        _count: int,
        **_kwargs: Any,
    ) -> list[str]:
        return [f"{system_so_far}\nCandidate"]

    async def _score_prompt_async(**_kwargs: Any) -> float:
        raise RuntimeError("scorer provider failed")

    monkeypatch.setattr(
        optimizer.decomposer,
        "decompose_text",
        lambda _text: ["one segment"],
        raising=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_evaluate_prompt",
        _evaluate_prompt,
        raising=True,
    )
    monkeypatch.setattr(
        optimizer,
        "_propose_candidates",
        _propose_candidates,
        raising=True,
    )
    monkeypatch.setattr(
        optimizer.scorer,
        "score_prompt_async",
        _score_prompt_async,
        raising=True,
    )

    try:
        with pytest.raises(RuntimeError, match="scorer provider failed"):
            await optimizer.optimize(
                initial_prompt_id=prompt_id,
                test_case_ids=[test_case_id],
                model_config=dict(_MODEL_CONFIG),
                max_iterations=1,
                target_metric=MetricType.ACCURACY,
                strategy_params={
                    "mcts_simulations": 1,
                    "mcts_max_depth": 1,
                    "prompt_candidates_per_node": 1,
                    "feedback_enabled": False,
                },
                scorer_model_config={
                    "provider": "custom-openai-api-2",
                    "model": "snapshot-model",
                    "parameters": {},
                },
                scorer_provider_credentials=handle,
                on_scorer_provider_success=None,
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_durable_worker_passes_runtime_handle_only_as_ephemeral_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "worker-runtime-handle")
    project_id, prompt_id, test_case_id = _create_project_resources(
        db,
        "worker-runtime-handle",
    )
    optimization_id = _create_optimization(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[test_case_id],
        name="worker-runtime-handle",
    )
    handle = await _issue_provider_credentials("openai", api_key="worker-key")
    captured: dict[str, Any] = {}

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(
            self,
            provider: str,
            *,
            model: str | None = None,
        ) -> ProviderCallCredentials:
            assert (provider, model) == ("openai", "gpt-4o-mini")
            return handle

        async def mark_used(self, selected: ProviderCallCredentials) -> bool:
            assert selected is handle
            return True

        async def close(self) -> None:
            return None

    class _Processor:
        def __init__(self, database: PromptStudioDatabase) -> None:
            self.db = database

        async def process_optimization_job(
            self,
            _payload: dict[str, Any],
            selected_optimization_id: int,
            **kwargs: Any,
        ) -> dict[str, Any]:
            captured.update(kwargs)
            await kwargs["on_provider_success"]()
            self.db.complete_optimization(
                selected_optimization_id,
                optimized_prompt_id=prompt_id,
                iterations_completed=1,
            )
            return {
                "optimization_id": selected_optimization_id,
                "status": "completed",
                "iterations_completed": 1,
            }

    async def _memberships(_user_id: int) -> tuple[list[int], list[int]]:
        return [], []

    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "single_user")
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: _Processor(db),
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

    try:
        result = await jobs_worker._handle_job(
            {
                "id": optimization_id,
                "uuid": "worker-runtime-handle-job",
                "job_type": "optimization",
                "owner_user_id": "7",
                "payload": _optimization_payload(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=[test_case_id],
                ),
            }
        )

        assert result["status"] == "completed"
        assert captured["provider_credentials"] is handle
        assert PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in captured[
            "runtime_model_config"
        ]
        assert all(
            value is not handle
            for value in captured["runtime_model_config"].values()
        )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_missing_persisted_prompt_is_not_auto_created_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "missing-prompt")
    project_id, _prompt_id, test_case_id = _create_project_resources(
        db,
        "missing-prompt",
    )
    optimization_id = _create_optimization(
        db,
        project_id=project_id,
        prompt_id=None,
        test_case_ids=[test_case_id],
        name="missing-prompt",
    )
    queued_prompt_id = 999_991
    dispatched = _install_dispatch_probe(monkeypatch)

    try:
        with pytest.raises(ValueError, match="prompt"):
            await JobProcessor(db).process_optimization_job(
                _optimization_payload(
                    optimization_id=optimization_id,
                    prompt_id=queued_prompt_id,
                    test_case_ids=[test_case_id],
                ),
                optimization_id,
            )

        assert db.get_prompt(queued_prompt_id) is None
        assert dispatched == []
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "invalid_id",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(1.5, id="non-integral-float"),
    ],
)
def test_resource_ids_reject_bool_and_non_integral_float(invalid_id: object) -> None:
    with pytest.raises(ValueError, match="resource is invalid"):
        JobProcessor._positive_id(invalid_id, label="resource")


@pytest.mark.asyncio
async def test_deleted_project_fails_closed_before_optimization_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "deleted-project")
    project_id, prompt_id, test_case_id = _create_project_resources(
        db,
        "deleted-project",
    )
    optimization_id = _create_optimization(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[test_case_id],
        name="deleted-project",
    )
    dispatched = _install_dispatch_probe(monkeypatch)

    try:
        assert db.delete_project(project_id) is True
        deleted_project = db.get_project(project_id, include_deleted=True) or {}
        assert bool(deleted_project.get("deleted")) is True

        with pytest.raises(ValueError, match="(?i)project"):
            await JobProcessor(db).process_optimization_job(
                _optimization_payload(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=[test_case_id],
                ),
                optimization_id,
            )

        assert dispatched == []
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_queued_test_cases_cannot_replace_persisted_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "queued-test-cases")
    project_id, prompt_id, persisted_case_id = _create_project_resources(
        db,
        "queued-test-cases",
    )
    queued_case = db.create_test_case(
        project_id=project_id,
        name="Queued replacement",
        inputs={"question": "queued"},
        expected_outputs={"response": "queued"},
    )
    queued_case_id = int(queued_case["id"])
    optimization_id = _create_optimization(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[persisted_case_id],
        name="queued-test-cases",
    )
    dispatched = _install_dispatch_probe(monkeypatch)

    try:
        with pytest.raises(ValueError, match="test case"):
            await JobProcessor(db).process_optimization_job(
                _optimization_payload(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=[queued_case_id],
                ),
                optimization_id,
            )

        stored = db.get_optimization(optimization_id, include_deleted=True) or {}
        assert stored["test_case_ids"] == [persisted_case_id]
        assert dispatched == []
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_resource", ["prompt", "test_case"])
async def test_persisted_optimization_resources_must_belong_to_its_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foreign_resource: str,
) -> None:
    db = _new_db(tmp_path, f"foreign-{foreign_resource}")
    first_project, first_prompt, first_case = _create_project_resources(db, "first")
    _second_project, second_prompt, second_case = _create_project_resources(db, "second")
    prompt_id = second_prompt if foreign_resource == "prompt" else first_prompt
    test_case_ids = [second_case if foreign_resource == "test_case" else first_case]
    optimization_id = _create_optimization(
        db,
        project_id=first_project,
        prompt_id=prompt_id,
        test_case_ids=test_case_ids,
        name=f"foreign-{foreign_resource}",
    )
    dispatched = _install_dispatch_probe(monkeypatch)

    try:
        with pytest.raises(ValueError, match="project"):
            await JobProcessor(db).process_optimization_job(
                _optimization_payload(
                    optimization_id=optimization_id,
                    prompt_id=prompt_id,
                    test_case_ids=test_case_ids,
                ),
                optimization_id,
            )

        assert dispatched == []
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_concurrent_swapped_resource_snapshots_fail_without_cross_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "concurrent-swapped-resources")
    first_project, first_prompt, first_case = _create_project_resources(db, "first")
    second_project, second_prompt, second_case = _create_project_resources(db, "second")
    first_id = _create_optimization(
        db,
        project_id=first_project,
        prompt_id=first_prompt,
        test_case_ids=[first_case],
        name="first",
    )
    second_id = _create_optimization(
        db,
        project_id=second_project,
        prompt_id=second_prompt,
        test_case_ids=[second_case],
        name="second",
    )
    dispatched = _install_dispatch_probe(monkeypatch)
    processor = JobProcessor(db)

    try:
        outcomes = await asyncio.gather(
            processor.process_optimization_job(
                _optimization_payload(
                    optimization_id=first_id,
                    prompt_id=second_prompt,
                    test_case_ids=[second_case],
                ),
                first_id,
            ),
            processor.process_optimization_job(
                _optimization_payload(
                    optimization_id=second_id,
                    prompt_id=first_prompt,
                    test_case_ids=[first_case],
                ),
                second_id,
            ),
            return_exceptions=True,
        )

        assert all(isinstance(outcome, ValueError) for outcome in outcomes)
        assert dispatched == []
        assert (db.get_optimization(first_id) or {})["test_case_ids"] == [first_case]
        assert (db.get_optimization(second_id) or {})["test_case_ids"] == [second_case]
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("swapped_field", "error_pattern"),
    [
        pytest.param("test_case_ids", "(?i)test case", id="test-cases"),
        pytest.param("model_configs", "(?i)model config", id="model-configs"),
    ],
)
async def test_evaluation_queued_snapshot_cannot_replace_persisted_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swapped_field: str,
    error_pattern: str,
) -> None:
    db = _new_db(tmp_path, f"evaluation-authority-{swapped_field}")
    project_id, prompt_id, persisted_case_id = _create_project_resources(
        db,
        f"evaluation-authority-{swapped_field}",
    )
    replacement_case = db.create_test_case(
        project_id=project_id,
        name=f"Replacement {swapped_field}",
        inputs={"question": "replacement"},
        expected_outputs={"response": "replacement"},
    )
    replacement_case_id = int(replacement_case["id"])
    persisted_models = [
        {"provider": "openai", "model": "persisted-model"},
    ]
    replacement_models = [
        {"provider": "openai", "model": "replacement-model"},
    ]
    evaluation_id = _create_evaluation(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[persisted_case_id],
        model_configs=persisted_models,
    )
    payload: dict[str, Any] = {
        "prompt_id": prompt_id,
        "test_case_ids": [persisted_case_id],
        "model_configs": persisted_models,
    }
    payload[swapped_field] = (
        [replacement_case_id]
        if swapped_field == "test_case_ids"
        else replacement_models
    )
    dispatched: list[tuple[int, int, str]] = []
    processor = JobProcessor(db)

    async def _execute_test_case(
        selected_prompt_id: int,
        selected_case_id: int,
        selected_model_config: dict[str, Any],
    ) -> dict[str, Any]:
        dispatched.append(
            (
                selected_prompt_id,
                selected_case_id,
                str(selected_model_config.get("model") or ""),
            )
        )
        return {
            "id": len(dispatched),
            "scores": {},
            "tokens_used": 0,
            "cost_estimate": 0.0,
        }

    monkeypatch.setattr(
        processor,
        "_execute_test_case",
        _execute_test_case,
        raising=True,
    )

    try:
        with pytest.raises(ValueError, match=error_pattern):
            await processor.process_evaluation_job(payload, evaluation_id)

        assert dispatched == []
        stored = db.get_evaluation(evaluation_id) or {}
        assert stored["test_case_ids"] == [persisted_case_id]
        assert stored["model_configs"] == persisted_models
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_concurrent_swapped_evaluation_snapshots_fail_without_cross_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "concurrent-swapped-evaluations")
    project_id, prompt_id, first_case_id = _create_project_resources(
        db,
        "concurrent-swapped-evaluations",
    )
    second_case = db.create_test_case(
        project_id=project_id,
        name="Second concurrent evaluation case",
        inputs={"question": "second"},
        expected_outputs={"response": "second"},
    )
    second_case_id = int(second_case["id"])
    first_models = [{"provider": "openai", "model": "first-model"}]
    second_models = [{"provider": "openai", "model": "second-model"}]
    first_id = _create_evaluation(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[first_case_id],
        model_configs=first_models,
    )
    second_id = _create_evaluation(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[second_case_id],
        model_configs=second_models,
    )
    dispatched: list[tuple[int, str]] = []
    dispatch_lock = asyncio.Lock()
    both_dispatched = asyncio.Event()
    processor = JobProcessor(db)

    async def _execute_test_case(
        _selected_prompt_id: int,
        selected_case_id: int,
        selected_model_config: dict[str, Any],
    ) -> dict[str, Any]:
        async with dispatch_lock:
            dispatched.append(
                (
                    selected_case_id,
                    str(selected_model_config.get("model") or ""),
                )
            )
            if len(dispatched) == 2:
                both_dispatched.set()
        await asyncio.wait_for(both_dispatched.wait(), timeout=1.0)
        return {
            "id": len(dispatched),
            "scores": {},
            "tokens_used": 0,
            "cost_estimate": 0.0,
        }

    monkeypatch.setattr(
        processor,
        "_execute_test_case",
        _execute_test_case,
        raising=True,
    )

    try:
        outcomes = await asyncio.gather(
            processor.process_evaluation_job(
                {
                    "prompt_id": prompt_id,
                    "test_case_ids": [second_case_id],
                    "model_configs": second_models,
                },
                first_id,
            ),
            processor.process_evaluation_job(
                {
                    "prompt_id": prompt_id,
                    "test_case_ids": [first_case_id],
                    "model_configs": first_models,
                },
                second_id,
            ),
            return_exceptions=True,
        )

        assert all(isinstance(outcome, ValueError) for outcome in outcomes)
        assert dispatched == []
        assert (db.get_evaluation(first_id) or {})["test_case_ids"] == [
            first_case_id
        ]
        assert (db.get_evaluation(second_id) or {})["test_case_ids"] == [
            second_case_id
        ]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_exhausted_lease_failure_converges_running_prompt_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "expired-lease")
    project_id, prompt_id, test_case_id = _create_project_resources(db, "expired-lease")
    optimization_id = _create_optimization(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[test_case_id],
        name="expired-lease",
        status="running",
    )
    jobs_path = tmp_path / "expired-lease-jobs.sqlite"
    manager = JobManager(db_path=jobs_path)
    queued = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_optimization_payload(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=[test_case_id],
            optimization_uuid=str(
                (db.get_optimization(optimization_id) or {})["uuid"]
            ),
        ),
        owner_user_id="7",
        max_retries=0,
    )
    acquired = manager.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=60,
        worker_id="expired-lease-worker",
    )
    assert acquired is not None
    assert acquired["id"] == queued["id"]
    connection = sqlite3.connect(jobs_path)
    try:
        connection.execute(
            "UPDATE jobs SET leased_until = DATETIME('now', '-1 minute'), "
            "retry_count = max_retries WHERE id = ?",
            (int(acquired["id"]),),
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        assert manager.acquire_next_job(
            domain="prompt_studio",
            queue="default",
            lease_seconds=60,
            worker_id="recovery-trigger",
        ) is None
        recovered = manager.get_job(int(acquired["id"])) or {}
        assert recovered["status"] == "failed"

        assert await jobs_worker._reconcile_cancelled_optimization_jobs(manager) == 1

        optimization = db.get_optimization(
            optimization_id,
            include_deleted=True,
        ) or {}
        assert optimization["status"] == "failed"
        assert optimization.get("completed_at") is not None
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_archived_cancellation_converges_tenant_before_repair_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _new_db(tmp_path, "archived-cancellation")
    project_id, prompt_id, test_case_id = _create_project_resources(
        db,
        "archived-cancellation",
    )
    optimization_id = _create_optimization(
        db,
        project_id=project_id,
        prompt_id=prompt_id,
        test_case_ids=[test_case_id],
        name="archived-cancellation",
        status="running",
    )
    jobs_path = tmp_path / "archived-cancellation-jobs.sqlite"
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
    manager = JobManager(db_path=jobs_path)
    job = manager.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload=_optimization_payload(
            optimization_id=optimization_id,
            prompt_id=prompt_id,
            test_case_ids=[test_case_id],
            optimization_uuid=str(
                (db.get_optimization(optimization_id) or {})["uuid"]
            ),
        ),
        owner_user_id="7",
    )
    assert manager.cancel_job(int(job["id"]), reason="archived cancellation")
    assert manager.prune_jobs(
        statuses=["cancelled"],
        older_than_days=0,
        domain="prompt_studio",
    ) == 1
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: JobProcessor(db),
        raising=True,
    )

    try:
        assert await jobs_worker._reconcile_cancelled_optimization_jobs(
            manager,
            include_archived=True,
        ) == 1
        optimization = db.get_optimization(
            optimization_id,
            include_deleted=True,
        ) or {}
        assert optimization["status"] == "cancelled"
        assert optimization.get("completed_at") is not None
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_live_reconciliation_offloads_list_and_tenant_db_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_ran = False
    list_calls = 0
    caller_thread_id = threading.get_ident()
    close_thread_ids: list[int] = []

    class _SlowManager:
        def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal list_calls
            list_calls += 1
            time.sleep(0.05)
            if kwargs.get("status") != "cancelled":
                return []
            return [
                {
                    "id": 1,
                    "uuid": "slow-cancelled-job",
                    "domain": "prompt_studio",
                    "queue": "default",
                    "job_type": "optimization",
                    "status": "cancelled",
                    "owner_user_id": "7",
                    "payload": {
                        "optimization_id": 1,
                        "optimization_config": {
                            "optimizer_type": "mipro",
                            "model_config": dict(_MODEL_CONFIG),
                        },
                    },
                }
            ]

    class _DB:
        def close_connection(self) -> None:
            close_thread_ids.append(threading.get_ident())

    class _Processor:
        db = _DB()

    def _slow_tenant_lookup(_user_id: str) -> _Processor:
        time.sleep(0.05)
        return _Processor()

    def _fail_tenant_work(**_kwargs: Any) -> None:
        raise RuntimeError("tenant probe complete")

    async def _heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0.01)
        heartbeat_ran = True

    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        _slow_tenant_lookup,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_secure_optimization_durable_state",
        _fail_tenant_work,
        raising=True,
    )
    heartbeat = asyncio.create_task(_heartbeat())

    await jobs_worker._reconcile_cancelled_optimization_jobs(
        _SlowManager(),  # type: ignore[arg-type]
    )
    assert heartbeat_ran is True
    assert list_calls >= 1
    assert close_thread_ids
    assert all(thread_id != caller_thread_id for thread_id in close_thread_ids)
    await heartbeat


@pytest.mark.asyncio
async def test_reconciliation_does_not_evict_worker_thread_tenant_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor reconciliation must not mutate the worker's thread-affine LRU."""

    caller_thread_id = threading.get_ident()
    factory_threads: list[int] = []
    close_calls: list[tuple[str, int]] = []

    class _DB:
        def __init__(self, user_id: str) -> None:
            self.user_id = user_id

        def get_optimization(
            self,
            optimization_id: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "uuid": f"optimization-{optimization_id}",
                "status": "running",
            }

        def close_connection(self) -> None:
            close_calls.append((self.user_id, threading.get_ident()))

    class _Processor:
        def __init__(self, db: _DB) -> None:
            self.db = db

    class _Manager:
        def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("status") != "cancelled":
                return []
            return [
                {
                    "id": owner_id,
                    "uuid": f"cancelled-{owner_id}",
                    "domain": "prompt_studio",
                    "queue": "default",
                    "job_type": "optimization",
                    "status": "cancelled",
                    "owner_user_id": str(owner_id),
                    "payload": {
                        "optimization_id": owner_id,
                        "optimization_uuid": f"optimization-{owner_id}",
                        "optimization_config": {"optimizer_type": "mipro"},
                    },
                }
                for owner_id in (8, 9)
            ]

    def _create_processor(user_id: str) -> _Processor:
        factory_threads.append(threading.get_ident())
        return _Processor(_DB(user_id))

    worker_db = _DB("worker-owner")
    worker_processor = object()
    with jobs_worker._CACHE_LOCK:
        saved_db_cache = jobs_worker._DB_CACHE.copy()
        saved_processor_cache = jobs_worker._PROCESSOR_CACHE.copy()
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._DB_CACHE["worker-owner"] = worker_db
        jobs_worker._PROCESSOR_CACHE["worker-owner"] = worker_processor  # type: ignore[assignment]

    monkeypatch.setattr(jobs_worker, "_MAX_CACHE_ENTRIES", 1, raising=True)
    monkeypatch.setattr(jobs_worker, "_auth_mode", lambda: "single_user", raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        _create_processor,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_secure_optimization_durable_state",
        lambda **kwargs: (
            kwargs["payload"],
            kwargs["payload"]["optimization_config"],
        ),
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_converge_terminal_prompt_state",
        lambda **_kwargs: True,
        raising=True,
    )

    try:
        assert await jobs_worker._reconcile_cancelled_optimization_jobs(
            _Manager(),  # type: ignore[arg-type]
        ) == 2
        assert {"worker-owner": worker_db} == jobs_worker._DB_CACHE
        assert {
            "worker-owner": worker_processor
        } == jobs_worker._PROCESSOR_CACHE
        assert all(thread_id != caller_thread_id for thread_id in factory_threads)
        assert close_calls
        assert all(
            user_id != "worker-owner" and thread_id != caller_thread_id
            for user_id, thread_id in close_calls
        )
    finally:
        with jobs_worker._CACHE_LOCK:
            jobs_worker._DB_CACHE.clear()
            jobs_worker._DB_CACHE.update(saved_db_cache)
            jobs_worker._PROCESSOR_CACHE.clear()
            jobs_worker._PROCESSOR_CACHE.update(saved_processor_cache)


@pytest.mark.asyncio
async def test_reconciliation_state_reuses_facade_across_poll_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated terminal scans must not repeat PostgreSQL schema initialization."""

    factory_calls = 0
    close_calls = 0

    class _Manager:
        def list_archived_jobs(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

        def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("status") != "cancelled":
                return []
            return [
                {
                    "id": 1,
                    "uuid": "repeated-cancelled-job",
                    "domain": "prompt_studio",
                    "queue": "default",
                    "job_type": "optimization",
                    "status": "cancelled",
                    "owner_user_id": "7",
                    "payload": {
                        "optimization_id": 1,
                        "optimization_uuid": "optimization-1",
                        "optimization_config": {"optimizer_type": "mipro"},
                    },
                }
            ]

    class _DB:
        def close_connection(self) -> None:
            nonlocal close_calls
            close_calls += 1

    class _Processor:
        db = _DB()

    def _create_processor(_user_id: str) -> _Processor:
        nonlocal factory_calls
        factory_calls += 1
        return _Processor()

    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        _create_processor,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_secure_optimization_durable_state",
        lambda **kwargs: (
            kwargs["payload"],
            kwargs["payload"]["optimization_config"],
        ),
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_converge_terminal_prompt_state",
        lambda **_kwargs: True,
        raising=True,
    )

    state = jobs_worker._CancellationReconciliationState()
    for _ in range(2):
        await jobs_worker._reconcile_cancelled_optimization_jobs(
            _Manager(),  # type: ignore[arg-type]
            include_archived=True,
            state=state,
        )

    assert factory_calls == 1
    assert close_calls == 2


def test_cache_drain_defers_active_database_close_until_scope_exits() -> None:
    close_calls: list[str] = []

    class _DB:
        def close_connection(self) -> None:
            close_calls.append("close")

    db = _DB()
    with jobs_worker._CACHE_LOCK:
        saved_db_cache = jobs_worker._DB_CACHE.copy()
        saved_processor_cache = jobs_worker._PROCESSOR_CACHE.copy()
        saved_active_counts = dict(jobs_worker._ACTIVE_USER_COUNTS)
        saved_pending_close = {
            key: list(value)
            for key, value in jobs_worker._PENDING_CLOSE.items()
        }
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._ACTIVE_USER_COUNTS.clear()
        jobs_worker._PENDING_CLOSE.clear()
        jobs_worker._DB_CACHE["7"] = db
        jobs_worker._PROCESSOR_CACHE["7"] = object()  # type: ignore[assignment]

    try:
        with jobs_worker._active_user_cache_scope("7"):
            jobs_worker._drain_tenant_db_cache()
            assert close_calls == []
            assert jobs_worker._DB_CACHE == {}
            assert jobs_worker._PROCESSOR_CACHE == {}

        assert close_calls == ["close"]
        assert jobs_worker._PENDING_CLOSE == {}
    finally:
        with jobs_worker._CACHE_LOCK:
            jobs_worker._DB_CACHE.clear()
            jobs_worker._DB_CACHE.update(saved_db_cache)
            jobs_worker._PROCESSOR_CACHE.clear()
            jobs_worker._PROCESSOR_CACHE.update(saved_processor_cache)
            jobs_worker._ACTIVE_USER_COUNTS.clear()
            jobs_worker._ACTIVE_USER_COUNTS.update(saved_active_counts)
            jobs_worker._PENDING_CLOSE.clear()
            jobs_worker._PENDING_CLOSE.update(saved_pending_close)


def test_cache_scope_releases_each_threads_connection_without_evicting_facade() -> None:
    """Concurrent users of one facade release only their thread-local handles."""

    close_thread_ids: list[int] = []
    entered = threading.Barrier(3)
    leave = threading.Barrier(3)

    class _DB:
        def close_connection(self) -> None:
            close_thread_ids.append(threading.get_ident())

    db = _DB()
    with jobs_worker._CACHE_LOCK:
        saved_db_cache = jobs_worker._DB_CACHE.copy()
        saved_processor_cache = jobs_worker._PROCESSOR_CACHE.copy()
        saved_active_counts = dict(jobs_worker._ACTIVE_USER_COUNTS)
        saved_pending_close = {
            key: list(value)
            for key, value in jobs_worker._PENDING_CLOSE.items()
        }
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._ACTIVE_USER_COUNTS.clear()
        jobs_worker._PENDING_CLOSE.clear()
        jobs_worker._DB_CACHE["7"] = db

    def _use_facade() -> None:
        with jobs_worker._active_user_cache_scope("7"):
            entered.wait(timeout=2)
            leave.wait(timeout=2)

    threads = [threading.Thread(target=_use_facade) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        entered.wait(timeout=2)
        assert close_thread_ids == []
        leave.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        assert set(close_thread_ids) == {thread.ident for thread in threads}
        assert jobs_worker._DB_CACHE["7"] is db
    finally:
        for thread in threads:
            thread.join(timeout=2)
        with jobs_worker._CACHE_LOCK:
            jobs_worker._DB_CACHE.clear()
            jobs_worker._DB_CACHE.update(saved_db_cache)
            jobs_worker._PROCESSOR_CACHE.clear()
            jobs_worker._PROCESSOR_CACHE.update(saved_processor_cache)
            jobs_worker._ACTIVE_USER_COUNTS.clear()
            jobs_worker._ACTIVE_USER_COUNTS.update(saved_active_counts)
            jobs_worker._PENDING_CLOSE.clear()
            jobs_worker._PENDING_CLOSE.update(saved_pending_close)


@pytest.mark.asyncio
async def test_worker_shutdown_drains_tenant_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = threading.Event()
    caller_thread_id = threading.get_ident()
    close_thread_ids: list[int] = []

    class _SDK:
        def __init__(self, _manager: object, _config: object) -> None:
            pass

        async def run(self, **_kwargs: Any) -> None:
            return None

        def stop(self) -> None:
            return None

    def _drain() -> None:
        close_thread_ids.append(threading.get_ident())
        closed.set()

    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object(), raising=True)
    monkeypatch.setattr(jobs_worker, "WorkerSDK", _SDK, raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_drain_tenant_db_cache",
        _drain,
        raising=False,
    )

    await jobs_worker.run_prompt_studio_jobs_worker()

    assert closed.is_set()
    assert close_thread_ids == [caller_thread_id]


@pytest.mark.asyncio
async def test_worker_shutdown_joins_blocked_reconciliation_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cannot outlive reconciliation work that still owns backends."""

    row_entered = threading.Event()
    release_row = threading.Event()
    sdk_returned = asyncio.Event()
    lifecycle_events: list[str] = []

    class _Manager:
        def list_archived_jobs(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

        def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("status") != "cancelled":
                return []
            return [
                {
                    "id": 1,
                    "uuid": "blocked-reconciliation",
                    "domain": "prompt_studio",
                    "queue": "default",
                    "job_type": "optimization",
                    "status": "cancelled",
                    "owner_user_id": "7",
                    "payload": {
                        "optimization_id": 1,
                        "optimization_uuid": "optimization-1",
                        "optimization_config": {"optimizer_type": "mipro"},
                    },
                }
            ]

    class _DB:
        def get_optimization(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"uuid": "optimization-1", "status": "running"}

        def close_connection(self) -> None:
            lifecycle_events.append("row-closed")

    class _Processor:
        db = _DB()

    class _SDK:
        def __init__(self, _manager: object, _config: object) -> None:
            pass

        async def run(self, **_kwargs: Any) -> None:
            while not row_entered.is_set():
                await asyncio.sleep(0)
            sdk_returned.set()

        def stop(self) -> None:
            return None

    def _block_row(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        row_entered.set()
        if not release_row.wait(timeout=2):
            raise AssertionError("test did not release reconciliation row")
        return kwargs["payload"], kwargs["payload"]["optimization_config"]

    monkeypatch.setattr(jobs_worker, "_jobs_manager", _Manager, raising=True)
    monkeypatch.setattr(jobs_worker, "WorkerSDK", _SDK, raising=True)
    monkeypatch.setattr(
        jobs_worker,
        "_create_reconciliation_processor",
        lambda _user_id: _Processor(),
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_secure_optimization_durable_state",
        _block_row,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_converge_terminal_prompt_state",
        lambda **_kwargs: True,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "_drain_tenant_db_cache",
        lambda: lifecycle_events.append("cache-drained"),
        raising=True,
    )

    worker_task = asyncio.create_task(jobs_worker.run_prompt_studio_jobs_worker())
    try:
        await asyncio.wait_for(sdk_returned.wait(), timeout=1)
        await asyncio.sleep(0.01)
        assert worker_task.done() is False
        assert "cache-drained" not in lifecycle_events
        worker_task.cancel()
        await asyncio.sleep(0.01)
        assert worker_task.done() is False
        assert "cache-drained" not in lifecycle_events
    finally:
        release_row.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(worker_task, timeout=1)
    assert lifecycle_events[-2:] == ["row-closed", "cache-drained"]
