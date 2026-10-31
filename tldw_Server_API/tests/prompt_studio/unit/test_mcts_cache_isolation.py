"""Tenant and provider-runtime isolation for MCTS' durable caches."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import (
    MCTSOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.types_common import (
    MetricType,
)

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._row)


class _SharedSyncLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []


class _ProviderBarrier:
    """Hold provider calls until every concurrent cache miss is in flight."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.arrived = 0
        self._release = asyncio.Event()

    async def wait(self) -> None:
        self.arrived += 1
        if self.arrived == self.expected:
            self._release.set()
        await asyncio.wait_for(self._release.wait(), timeout=5.0)


class _CacheDb:
    """Minimal DB facade whose sync-log storage is shared across tenants."""

    def __init__(self, backend: _SharedSyncLog, tenant_user_id: str) -> None:
        self.backend = backend
        self.client_id = f"mcts-cache-audit:{tenant_user_id}"
        self.tenant_user_id = tenant_user_id

    @contextmanager
    def transaction(self) -> Iterator[object]:
        yield object()

    def _cursor_exec(
        self,
        _conn: object,
        sql: str,
        params: tuple[Any, ...],
    ) -> _Cursor:
        assert "FROM sync_log" in sql
        entity, entity_uuid = params[-2:]
        row = next(
            (
                event
                for event in reversed(self.backend.events)
                if event["entity"] == entity
                and event["entity_uuid"] == entity_uuid
            ),
            None,
        )
        if row is None:
            return _Cursor(None)
        return _Cursor(
            {
                "payload": row["payload"],
                "timestamp": row["timestamp"],
            }
        )

    def _log_sync_event(
        self,
        *,
        entity: str,
        entity_uuid: str,
        operation: str,
        payload: dict[str, Any],
    ) -> None:
        self.backend.events.append(
            {
                "entity": entity,
                "entity_uuid": entity_uuid,
                "operation": operation,
                "payload": copy.deepcopy(payload),
                "timestamp": "2099-01-01T00:00:00",
            }
        )

    def get_test_cases_by_ids(
        self,
        test_case_ids: list[int],
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        assert include_deleted is False
        return [
            {
                "id": int(test_case_id),
                "inputs": {"question": f"question-{int(test_case_id)}"},
                "expected_outputs": {"answer": f"answer-{int(test_case_id)}"},
                "deleted": False,
            }
            for test_case_id in test_case_ids
        ]


class _RecordingExecutor:
    def __init__(
        self,
        content: str,
        provider_barrier: _ProviderBarrier | None = None,
    ) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.provider_barrier = provider_barrier

    async def _call_llm(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(kwargs))
        if self.provider_barrier is not None:
            await self.provider_barrier.wait()
        callback = kwargs.get("on_provider_success")
        if callback is not None:
            await callback()
        return {"content": self.content, "tokens": 1}


class _RecordingRunner:
    def __init__(
        self,
        score: float,
        provider_barrier: _ProviderBarrier | None = None,
    ) -> None:
        self.score = score
        self.calls: list[dict[str, Any]] = []
        self.provider_barrier = provider_barrier

    async def run_single_test(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(kwargs))
        if self.provider_barrier is not None:
            await self.provider_barrier.wait()
        callback = kwargs.get("on_provider_success")
        if callback is not None:
            await callback()
        scores = {metric.value: self.score for metric in MetricType}
        scores["aggregate_score"] = self.score
        return {
            "success": True,
            "scores": scores,
        }


def _runtime_config(
    *,
    endpoint: str,
    temperature: float,
    api_key: str,
) -> dict[str, Any]:
    return {
        "provider": "custom-openai-api",
        "model": "shared-model",
        "parameters": {
            "temperature": temperature,
            "top_p": 0.73,
            "max_tokens": 96,
            "timeout_seconds": 17,
        },
        "api_key": api_key,
        "app_config": {
            "custom_openai_api": {
                "api_key": api_key,
                "base_url": endpoint,
                "model": "shared-model",
            }
        },
        "credentials_resolved": True,
    }


def _surface_output(surface: str, label: str) -> str | float:
    if surface == "rephrase":
        return f"rephrased-{label}"
    if surface == "scorer":
        return {"A": "9.0", "B": "1.0", "C": "5.0"}[label]
    return {"A": 0.91, "B": 0.27, "C": 0.63}[label]


def _assert_cache_excludes_credentials(
    backend: _SharedSyncLog,
    *secrets: str,
) -> None:
    serialized_cache = json.dumps(backend.events, sort_keys=True, default=str)
    for secret in secrets:
        assert secret not in serialized_cache
        assert hashlib.sha256(secret.encode("utf-8")).hexdigest() not in serialized_cache


async def _run_cache_surface(
    *,
    surface: str,
    db: _CacheDb,
    config: dict[str, Any],
    label: str,
    target_metric: MetricType = MetricType.ACCURACY,
    provider_barrier: _ProviderBarrier | None = None,
) -> dict[str, Any]:
    output = _surface_output(surface, label)
    runner = _RecordingRunner(
        float(output) if surface == "eval" else 0.5,
        provider_barrier,
    )
    optimizer = MCTSOptimizer(db, runner)  # type: ignore[arg-type]
    executor = _RecordingExecutor(str(output), provider_barrier)
    optimizer.executor = executor  # type: ignore[assignment]
    optimizer.scorer.set_executor(executor)
    optimizer._tokens_spent = 0
    marks: list[str] = []

    async def _mark_provider() -> None:
        marks.append(label)

    if surface == "rephrase":
        result = await optimizer._rephrase_segment(
            "Shared system prompt",
            "Shared segment",
            model_config=config,
            on_provider_success=_mark_provider,
            strict_provider_errors=True,
        )
        calls = executor.calls
    elif surface == "scorer":
        async def _one_candidate(*_args: Any, **_kwargs: Any) -> list[str]:
            return ["Shared candidate"]

        optimizer._propose_candidates = _one_candidate  # type: ignore[method-assign]
        dispatch_state = {"dispatched": False}
        node = MCTSOptimizer._Node(
            parent=None,
            segment_index=0,
            system_text="Shared system prompt",
        )
        child = await optimizer._expand_node(
            node,
            segment="Shared segment",
            base_user="Shared user prompt",
            k_candidates=1,
            score_bin_size=0.01,
            min_quality=0.0,
            model_config=config,
            scorer_model="shared-scorer-model",
            scorer_model_config=config,
            on_scorer_provider_success=_mark_provider,
            scorer_dispatch_state=dispatch_state,
            strict_provider_errors=True,
        )
        assert child is not None
        result = child.score_bin
        calls = executor.calls
    else:
        optimizer._create_ephemeral_prompt_version = (  # type: ignore[method-assign]
            lambda **_kwargs: 101
        )
        result, _prompt_id = await optimizer._evaluate_with_feedback(
            base_prompt={
                "id": 1,
                "project_id": 7,
                "name": "Shared prompt",
                "version_number": 1,
            },
            system_text="Shared system prompt",
            user_text="Shared user prompt",
            test_case_ids=[11],
            model_config=config,
            target_metric=target_metric,
            feedback_enabled=False,
            feedback_threshold=10.0,
            feedback_max_retries=0,
            on_provider_success=_mark_provider,
            strict_provider_errors=True,
        )
        calls = runner.calls

    for call in calls:
        if surface == "eval":
            assert call["model_config"] == config
        else:
            assert call["provider"] == config["provider"]
            assert call["model"] in {config["model"], "shared-scorer-model"}
            assert call["api_key_override"] == config["api_key"]
            assert call["app_config"] == config["app_config"]
            assert call["credentials_resolved"] is True

    return {"result": result, "calls": calls, "marks": marks}


@pytest.mark.parametrize("surface", ["rephrase", "scorer", "eval"])
@pytest.mark.asyncio
async def test_mcts_durable_cache_isolated_by_tenant_and_runtime_behavior(
    surface: str,
) -> None:
    backend = _SharedSyncLog()
    config_a = _runtime_config(
        endpoint="https://tenant-a.example/v1",
        temperature=0.19,
        api_key="tenant-a-runtime-secret",
    )
    config_b = _runtime_config(
        endpoint="https://tenant-b.example/v1",
        temperature=0.81,
        api_key="tenant-b-runtime-secret",
    )

    first = await _run_cache_surface(
        surface=surface,
        db=_CacheDb(backend, "tenant-a"),
        config=config_a,
        label="A",
    )
    distinct_behavior = await _run_cache_surface(
        surface=surface,
        db=_CacheDb(backend, "tenant-b"),
        config=config_b,
        label="B",
    )
    distinct_tenant = await _run_cache_surface(
        surface=surface,
        db=_CacheDb(backend, "tenant-b"),
        config=config_a,
        label="C",
    )

    assert len(first["calls"]) == 1
    assert len(distinct_behavior["calls"]) == 1
    assert len(distinct_tenant["calls"]) == 1
    assert first["marks"] == ["A"]
    assert distinct_behavior["marks"] == ["B"]
    assert distinct_tenant["marks"] == ["C"]
    assert len(
        {
            first["result"],
            distinct_behavior["result"],
            distinct_tenant["result"],
        }
    ) == 3
    _assert_cache_excludes_credentials(
        backend,
        "tenant-a-runtime-secret",
        "tenant-b-runtime-secret",
    )


@pytest.mark.parametrize("surface", ["rephrase", "scorer", "eval"])
@pytest.mark.asyncio
async def test_mcts_durable_cache_identity_uses_secret_free_runtime_behavior(
    surface: str,
) -> None:
    backend = _SharedSyncLog()
    original_secret = "credential-before-rotation"
    rotated_secret = "credential-after-rotation"
    original_config = _runtime_config(
        endpoint="https://stable-runtime.example/v1",
        temperature=0.37,
        api_key=original_secret,
    )
    rotated_config = _runtime_config(
        endpoint="https://stable-runtime.example/v1",
        temperature=0.37,
        api_key=rotated_secret,
    )

    first = await _run_cache_surface(
        surface=surface,
        db=_CacheDb(backend, "tenant-a"),
        config=original_config,
        label="A",
    )
    stored_before_rotation = copy.deepcopy(backend.events)
    cache_hit = await _run_cache_surface(
        surface=surface,
        db=_CacheDb(backend, "tenant-a"),
        config=rotated_config,
        label="B",
    )

    assert len(first["calls"]) == 1
    assert first["marks"] == ["A"]
    assert cache_hit["calls"] == []
    assert cache_hit["marks"] == []
    assert cache_hit["result"] == first["result"]
    assert backend.events == stored_before_rotation

    _assert_cache_excludes_credentials(backend, original_secret, rotated_secret)


@pytest.mark.parametrize("surface", ["rephrase", "scorer", "eval"])
@pytest.mark.asyncio
async def test_mcts_durable_cache_isolation_holds_for_concurrent_tenants_and_behaviors(
    surface: str,
) -> None:
    backend = _SharedSyncLog()
    config_a = _runtime_config(
        endpoint="https://tenant-a.example/v1",
        temperature=0.19,
        api_key="tenant-a-runtime-secret",
    )
    config_b = _runtime_config(
        endpoint="https://tenant-b.example/v1",
        temperature=0.81,
        api_key="tenant-b-runtime-secret",
    )
    primed = await _run_cache_surface(
        surface=surface,
        db=_CacheDb(backend, "tenant-a"),
        config=config_a,
        label="A",
    )

    provider_barrier = _ProviderBarrier(expected=2)

    async def _scheduled_run(
        *,
        tenant: str,
        config: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        return await _run_cache_surface(
            surface=surface,
            db=_CacheDb(backend, tenant),
            config=config,
            label=label,
            provider_barrier=provider_barrier,
        )

    behavior_task = asyncio.create_task(
        _scheduled_run(tenant="tenant-a", config=config_b, label="B")
    )
    tenant_task = asyncio.create_task(
        _scheduled_run(tenant="tenant-b", config=config_a, label="C")
    )
    distinct_behavior, distinct_tenant = await asyncio.gather(
        behavior_task,
        tenant_task,
    )

    assert len(primed["calls"]) == 1
    assert len(distinct_behavior["calls"]) == 1
    assert len(distinct_tenant["calls"]) == 1
    assert primed["marks"] == ["A"]
    assert distinct_behavior["marks"] == ["B"]
    assert distinct_tenant["marks"] == ["C"]
    assert provider_barrier.arrived == 2
    assert len(
        {
            primed["result"],
            distinct_behavior["result"],
            distinct_tenant["result"],
        }
    ) == 3
    _assert_cache_excludes_credentials(
        backend,
        "tenant-a-runtime-secret",
        "tenant-b-runtime-secret",
    )


@pytest.mark.asyncio
async def test_mcts_eval_cache_isolated_by_target_metric() -> None:
    backend = _SharedSyncLog()
    config = _runtime_config(
        endpoint="https://stable-runtime.example/v1",
        temperature=0.37,
        api_key="metric-runtime-secret",
    )

    accuracy = await _run_cache_surface(
        surface="eval",
        db=_CacheDb(backend, "tenant-a"),
        config=config,
        label="A",
        target_metric=MetricType.ACCURACY,
    )
    f1_score = await _run_cache_surface(
        surface="eval",
        db=_CacheDb(backend, "tenant-a"),
        config=config,
        label="B",
        target_metric=MetricType.F1_SCORE,
    )

    assert len(accuracy["calls"]) == 1
    assert len(f1_score["calls"]) == 1
    assert accuracy["marks"] == ["A"]
    assert f1_score["marks"] == ["B"]
    assert accuracy["result"] == pytest.approx(0.91)
    assert f1_score["result"] == pytest.approx(0.27)
    _assert_cache_excludes_credentials(backend, "metric-runtime-secret")


async def _invoke_reused_optimizer_surface(
    *,
    surface: str,
    optimizer: MCTSOptimizer,
    executor: _RecordingExecutor,
    runner: _RecordingRunner,
    config: dict[str, Any],
    label: str,
    marks: list[str],
) -> str | float | int:
    output = _surface_output(surface, label)
    executor.content = str(output)
    if surface == "eval":
        runner.score = float(output)

    async def _mark_provider() -> None:
        marks.append(label)

    if surface == "rephrase":
        result = await optimizer._rephrase_segment(
            "Shared system prompt",
            "Shared segment",
            model_config=config,
            on_provider_success=_mark_provider,
            strict_provider_errors=True,
        )
        assert result is not None
        return result

    if surface == "scorer":

        async def _one_candidate(*_args: Any, **_kwargs: Any) -> list[str]:
            return ["Shared candidate"]

        optimizer._propose_candidates = _one_candidate  # type: ignore[method-assign]
        node = MCTSOptimizer._Node(
            parent=None,
            segment_index=0,
            system_text="Shared system prompt",
        )
        child = await optimizer._expand_node(
            node,
            segment="Shared segment",
            base_user="Shared user prompt",
            k_candidates=1,
            score_bin_size=0.01,
            min_quality=0.0,
            model_config=config,
            scorer_model="shared-scorer-model",
            scorer_model_config=config,
            on_scorer_provider_success=_mark_provider,
            scorer_dispatch_state={"dispatched": False},
            strict_provider_errors=True,
        )
        assert child is not None
        return child.score_bin

    optimizer._create_ephemeral_prompt_version = (  # type: ignore[method-assign]
        lambda **_kwargs: 101
    )
    result, _prompt_id = await optimizer._evaluate_with_feedback(
        base_prompt={
            "id": 1,
            "project_id": 7,
            "name": "Shared prompt",
            "version_number": 1,
        },
        system_text="Shared system prompt",
        user_text="Shared user prompt",
        test_case_ids=[11],
        model_config=config,
        target_metric=MetricType.ACCURACY,
        feedback_enabled=False,
        feedback_threshold=10.0,
        feedback_max_retries=0,
        on_provider_success=_mark_provider,
        strict_provider_errors=True,
    )
    return result


@pytest.mark.asyncio
async def test_mcts_scorer_memory_cache_scopes_resolved_runtime_behavior() -> None:
    backend = _SharedSyncLog()
    runner = _RecordingRunner(0.5)
    optimizer = MCTSOptimizer(  # type: ignore[arg-type]
        _CacheDb(backend, "tenant-a"),
        runner,
    )
    executor = _RecordingExecutor("0")
    optimizer.executor = executor  # type: ignore[assignment]
    optimizer.scorer.set_executor(executor)
    optimizer._tokens_spent = 0
    marks: list[str] = []

    first = await _invoke_reused_optimizer_surface(
        surface="scorer",
        optimizer=optimizer,
        executor=executor,
        runner=runner,
        config=_runtime_config(
            endpoint="https://runtime-a.example/v1",
            temperature=0.19,
            api_key="runtime-a-secret",
        ),
        label="A",
        marks=marks,
    )
    second = await _invoke_reused_optimizer_surface(
        surface="scorer",
        optimizer=optimizer,
        executor=executor,
        runner=runner,
        config=_runtime_config(
            endpoint="https://runtime-b.example/v1",
            temperature=0.81,
            api_key="runtime-b-secret",
        ),
        label="B",
        marks=marks,
    )

    assert len(executor.calls) == 2
    assert marks == ["A", "B"]
    assert first != second
    _assert_cache_excludes_credentials(
        backend,
        "runtime-a-secret",
        "runtime-b-secret",
    )


_MISSING_CREDENTIAL_STATE = object()


@pytest.mark.parametrize("surface", ["rephrase", "scorer", "eval"])
@pytest.mark.parametrize(
    "credential_state",
    [
        pytest.param(_MISSING_CREDENTIAL_STATE, id="missing"),
        pytest.param(False, id="false"),
        pytest.param(1, id="integer-one"),
        pytest.param("true", id="string-true"),
    ],
)
@pytest.mark.asyncio
async def test_mcts_provider_result_caches_require_exactly_resolved_credentials(
    surface: str,
    credential_state: object,
) -> None:
    backend = _SharedSyncLog()
    runner = _RecordingRunner(0.5)
    optimizer = MCTSOptimizer(  # type: ignore[arg-type]
        _CacheDb(backend, "tenant-a"),
        runner,
    )
    executor = _RecordingExecutor("0")
    optimizer.executor = executor  # type: ignore[assignment]
    optimizer.scorer.set_executor(executor)
    optimizer._tokens_spent = 0
    marks: list[str] = []

    authoritative_config = _runtime_config(
        endpoint="https://stable-runtime.example/v1",
        temperature=0.37,
        api_key="runtime-secret",
    )
    unresolved_config = copy.deepcopy(authoritative_config)
    if credential_state is _MISSING_CREDENTIAL_STATE:
        unresolved_config.pop("credentials_resolved")
    else:
        unresolved_config["credentials_resolved"] = credential_state

    first = await _invoke_reused_optimizer_surface(
        surface=surface,
        optimizer=optimizer,
        executor=executor,
        runner=runner,
        config=authoritative_config,
        label="A",
        marks=marks,
    )
    durable_cache_before = copy.deepcopy(backend.events)
    memory_caches_before = (
        copy.deepcopy(optimizer._rephrase_cache),
        copy.deepcopy(optimizer._eval_cache),
        copy.deepcopy(optimizer.scorer._cache),
    )

    second = await _invoke_reused_optimizer_surface(
        surface=surface,
        optimizer=optimizer,
        executor=executor,
        runner=runner,
        config=unresolved_config,
        label="B",
        marks=marks,
    )
    third = await _invoke_reused_optimizer_surface(
        surface=surface,
        optimizer=optimizer,
        executor=executor,
        runner=runner,
        config=unresolved_config,
        label="C",
        marks=marks,
    )

    provider_calls = runner.calls if surface == "eval" else executor.calls
    assert durable_cache_before
    assert len(provider_calls) == 3
    assert marks == ["A", "B", "C"]
    assert len({first, second, third}) == 3
    assert backend.events == durable_cache_before
    assert (
        optimizer._rephrase_cache,
        optimizer._eval_cache,
        optimizer.scorer._cache,
    ) == memory_caches_before
    _assert_cache_excludes_credentials(backend, "runtime-secret")
