import asyncio
import contextlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner


class _CancellationTrackingRuntime:
    """Record credential lifecycle ordering for endpoint cancellation tests."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.handle = SimpleNamespace(
            api_key="test_api_key",
            app_config={"openai_api": {"model": "test-model"}},
            credentials_resolved=True,
        )
        self.closed = False
        self.mark_count = 0

    async def mark_used(self, handle) -> None:
        assert handle is self.handle
        self.mark_count += 1
        self.events.append("mark")

    async def close(self) -> None:
        self.closed = True
        self.events.append("close")


class _AllowingLimiter:
    async def check_rate_limit(self, *_args, **_kwargs):
        return True, {"retry_after": 0}


class _NoopWebhookManager:
    async def send_webhook(self, **_kwargs) -> None:
        return None


def _install_cancellation_endpoint_dependencies(monkeypatch, runtime, service) -> None:
    """Install deterministic direct-call dependencies for evaluation endpoints."""

    async def _resolve(*_args, **_kwargs):
        return "openai", "test-model", runtime.handle, runtime

    monkeypatch.setattr(eval_unified, "_resolve_and_validate_eval_provider", _resolve)
    monkeypatch.setattr(eval_unified, "_build_eval_credential_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _AllowingLimiter())
    monkeypatch.setattr(eval_unified, "get_unified_evaluation_service_for_user", lambda _uid: service)
    monkeypatch.setattr(eval_unified, "_get_webhook_manager_for_user", lambda _uid: _NoopWebhookManager())
    monkeypatch.setattr(eval_unified, "_is_eval_test_mode", lambda: True)


async def _await_cancelled(task: asyncio.Task) -> None:
    """Consume an endpoint task that must preserve cancellation semantics."""

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_geval_cancellation_drains_provider_and_marks_before_runtime_close(monkeypatch):
    events: list[str] = []
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    runtime = _CancellationTrackingRuntime(events)

    class _Service:
        async def evaluate_geval(self, **_kwargs):
            async def _provider_call():
                provider_started.set()
                await release_provider.wait()
                events.append("provider_done")
                return {
                    "evaluation_id": "eval-1",
                    "evaluation_time": 0.1,
                    "results": {"metrics": {}, "average_score": 1.0},
                }

            return await asyncio.shield(asyncio.create_task(_provider_call()))

    _install_cancellation_endpoint_dependencies(monkeypatch, runtime, _Service())
    endpoint_task = asyncio.create_task(
        eval_unified.evaluate_geval(
            request=eval_unified.GEvalRequest(
                source_text="source text long enough",
                summary="summary text long enough",
                api_name="openai",
            ),
            http_request=SimpleNamespace(),
            response=None,
            user_id="user-1",
            current_user=User(id=1, username="tester", email=None, is_active=True),
        )
    )

    await provider_started.wait()
    endpoint_task.cancel()
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.closed is False
        release_provider.set()
        await _await_cancelled(endpoint_task)
        assert events == ["provider_done", "mark", "close"]
    finally:
        release_provider.set()
        if not endpoint_task.done():
            endpoint_task.cancel()
        with contextlib.suppress(BaseException):
            await endpoint_task


@pytest.mark.asyncio
async def test_sequential_batch_cancellation_drains_provider_and_marks_before_close(monkeypatch):
    events: list[str] = []
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    runtime = _CancellationTrackingRuntime(events)

    class _Service:
        async def evaluate_geval(self, **_kwargs):
            async def _provider_call():
                provider_started.set()
                await release_provider.wait()
                events.append("provider_done")
                return {"evaluation_id": "eval-1", "results": {}}

            return await asyncio.shield(asyncio.create_task(_provider_call()))

    _install_cancellation_endpoint_dependencies(monkeypatch, runtime, _Service())
    endpoint_task = asyncio.create_task(
        eval_unified.batch_evaluate(
            request=eval_unified.BatchEvaluationRequest(
                evaluation_type="geval",
                parallel_workers=1,
                items=[{"source_text": "source", "summary": "summary"}],
            ),
            http_request=SimpleNamespace(),
            user_id="user-1",
            current_user=User(id=1, username="tester", email=None, is_active=True),
            response=None,
        )
    )

    await provider_started.wait()
    endpoint_task.cancel()
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.closed is False
        release_provider.set()
        await _await_cancelled(endpoint_task)
        assert events == ["provider_done", "mark", "close"]
    finally:
        release_provider.set()
        if not endpoint_task.done():
            endpoint_task.cancel()
        with contextlib.suppress(BaseException):
            await endpoint_task


@pytest.mark.asyncio
async def test_parallel_batch_cancellation_drains_all_children_before_runtime_close(monkeypatch):
    events: list[str] = []
    both_started = asyncio.Event()
    release_provider = asyncio.Event()
    runtime = _CancellationTrackingRuntime(events)
    started = 0

    class _Service:
        async def evaluate_geval(self, *, source_text: str, **_kwargs):
            async def _provider_call():
                nonlocal started
                started += 1
                events.append(f"start:{source_text}")
                if started == 2:
                    both_started.set()
                await release_provider.wait()
                events.append(f"done:{source_text}")
                return {"evaluation_id": source_text, "results": {}}

            return await asyncio.shield(asyncio.create_task(_provider_call()))

    _install_cancellation_endpoint_dependencies(monkeypatch, runtime, _Service())
    endpoint_task = asyncio.create_task(
        eval_unified.batch_evaluate(
            request=eval_unified.BatchEvaluationRequest(
                evaluation_type="geval",
                parallel_workers=2,
                items=[
                    {"source_text": "item-1", "summary": "summary"},
                    {"source_text": "item-2", "summary": "summary"},
                ],
            ),
            http_request=SimpleNamespace(),
            user_id="user-1",
            current_user=User(id=1, username="tester", email=None, is_active=True),
            response=None,
        )
    )

    await both_started.wait()
    endpoint_task.cancel()
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.closed is False
        release_provider.set()
        await _await_cancelled(endpoint_task)
        assert runtime.mark_count == 2
        assert events[-1] == "close"
        assert events.index("close") > events.index("done:item-1")
        assert events.index("close") > events.index("done:item-2")
    finally:
        release_provider.set()
        if not endpoint_task.done():
            endpoint_task.cancel()
        with contextlib.suppress(BaseException):
            await endpoint_task


def _install_eval_runtime(monkeypatch) -> None:
    """Install a deterministic authoritative credential runtime for endpoint tests."""

    class _Runtime:
        def __init__(self, **_kwargs):
            self.handle = SimpleNamespace(
                api_key="test_api_key",
                app_config={"openai_api": {"model": "test-model"}},
                credentials_resolved=True,
            )

        async def resolve(self, _provider: str, *, model: str | None = None):
            _ = model
            return self.handle

        async def mark_used(self, handle):
            assert handle is self.handle

        async def close(self):
            return None

    monkeypatch.setattr(eval_unified, "ProviderCredentialRuntime", _Runtime)


def test_batch_parallel_strict_fail_fast_cancels_remaining(monkeypatch):
    _install_eval_runtime(monkeypatch)
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    class _Limiter:
        async def check_rate_limit(
            self,
            _user_id: str,
            *,
            endpoint: str,
            is_batch: bool,
            tokens_requested: int,
            estimated_cost: float,
        ):
            _ = (endpoint, is_batch, tokens_requested, estimated_cost)
            return True, {"retry_after": 0}

    started_items: list[int] = []
    cancelled_items: list[int] = []

    class _Service:
        async def evaluate_geval(
            self,
            source_text: str,
            summary: str,
            metrics,
            api_name: str,
            api_key: str,
            model: str | None,
            user_id: str,
            webhook_user_id: str | None = None,
            app_config=None,
            credentials_resolved: bool = False,
            provider_credentials=None,
        ):
            _ = (
                summary,
                metrics,
                api_name,
                api_key,
                model,
                user_id,
                webhook_user_id,
                app_config,
                credentials_resolved,
                provider_credentials,
            )
            idx = int(source_text.split("_")[-1])
            started_items.append(idx)
            if idx == 0:
                await asyncio.sleep(0.01)
                raise ValueError("forced failure")
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                cancelled_items.append(idx)
                raise
            return {
                "evaluation_id": f"eval_{idx}",
                "results": {"idx": idx},
            }

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _rate_limit_dep_override():
        return None

    async def _fake_apply_rate_limit_headers(_limiter, _user_id, response, _meta):
        response.headers["X-Stage3-RateLimit-Applied"] = "true"

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.check_evaluation_rate_limit] = _rate_limit_dep_override

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _Limiter())
    monkeypatch.setattr(eval_unified, "_apply_rate_limit_headers", _fake_apply_rate_limit_headers)
    monkeypatch.setattr(eval_unified, "get_unified_evaluation_service_for_user", lambda _uid: _Service())

    body = {
        "evaluation_type": "geval",
        "parallel_workers": 2,
        "continue_on_error": False,
        "items": [
            {"source_text": "item_0", "summary": "summary_0"},
            {"source_text": "item_1", "summary": "summary_1"},
            {"source_text": "item_2", "summary": "summary_2"},
            {"source_text": "item_3", "summary": "summary_3"},
            {"source_text": "item_4", "summary": "summary_4"},
        ],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/evaluations/batch", json=body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_items"] == 5
    assert payload["successful"] == 0
    assert payload["failed"] == 5
    assert response.headers.get("X-Stage3-RateLimit-Applied") == "true"
    assert len(started_items) <= 2
    assert any(
        "strict fail-fast" in (entry.get("error") or "")
        for entry in payload["results"]
    )
    assert len(cancelled_items) <= 1


def test_batch_parallel_sanitizes_item_failure(monkeypatch):
    _install_eval_runtime(monkeypatch)
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    class _Limiter:
        async def check_rate_limit(
            self,
            _user_id: str,
            *,
            endpoint: str,
            is_batch: bool,
            tokens_requested: int,
            estimated_cost: float,
        ):
            _ = (endpoint, is_batch, tokens_requested, estimated_cost)
            return True, {"retry_after": 0}

    class _Service:
        async def evaluate_geval(
            self,
            source_text: str,
            summary: str,
            metrics,
            api_name: str,
            api_key: str,
            model: str | None,
            user_id: str,
            webhook_user_id: str | None = None,
            app_config=None,
            credentials_resolved: bool = False,
            provider_credentials=None,
        ):
            _ = (
                source_text,
                summary,
                metrics,
                api_name,
                api_key,
                model,
                user_id,
                webhook_user_id,
                app_config,
                credentials_resolved,
                provider_credentials,
            )
            raise RuntimeError("evaluation backend exploded at /private/evals.db")

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _rate_limit_dep_override():
        return None

    async def _fake_apply_rate_limit_headers(_limiter, _user_id, _response, _meta):
        return None

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.check_evaluation_rate_limit] = _rate_limit_dep_override

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _Limiter())
    monkeypatch.setattr(eval_unified, "_apply_rate_limit_headers", _fake_apply_rate_limit_headers)
    monkeypatch.setattr(eval_unified, "get_unified_evaluation_service_for_user", lambda _uid: _Service())

    body = {
        "evaluation_type": "geval",
        "parallel_workers": 2,
        "continue_on_error": True,
        "items": [
            {"source_text": "item_0", "summary": "summary_0"},
        ],
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/evaluations/batch", json=body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["failed"] == 1
    assert payload["results"][0]["error"] == "Evaluation item failed"
    assert "evaluation backend exploded" not in str(payload)
    assert "/private/evals.db" not in str(payload)


@pytest.mark.asyncio
async def test_execute_evaluation_handles_metrics_none(tmp_path, monkeypatch):
    runner = EvaluationRunner(str(tmp_path / "evals_stage3.db"), max_concurrent_evals=2, eval_timeout=10)

    monkeypatch.setattr(runner.db, "update_run_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.db, "update_run_progress", lambda *_args, **_kwargs: None)

    captured_results: dict[str, object] = {}

    def _capture_store_results(_run_id, results, usage):
        captured_results["results"] = results
        captured_results["usage"] = usage

    monkeypatch.setattr(runner.db, "store_run_results", _capture_store_results)
    monkeypatch.setattr(
        runner.db,
        "get_evaluation",
        lambda _eval_id: {
            "eval_type": "model_graded",
            "eval_spec": {
                "metrics": None,
                "threshold": 0.7,
            },
        },
    )

    async def _fake_get_samples(_evaluation, _eval_config):
        return [{"id": "s1", "input": {"source_text": "a", "summary": "b"}}]

    monkeypatch.setattr(runner, "_get_samples", _fake_get_samples)
    monkeypatch.setattr(runner, "_get_evaluation_function", lambda *_args, **_kwargs: object())

    async def _fake_process_batch(
        batch,
        eval_fn,
        eval_spec,
        eval_config,
        max_workers,
        start_index,
        timeout_seconds,
    ):
        _ = (batch, eval_fn, eval_spec, eval_config, max_workers, start_index, timeout_seconds)
        return [
            {
                "sample_id": "sample_000000_s1",
                "scores": {"coherence": 0.9},
                "passed": True,
                "avg_score": 0.9,
                "usage": {"total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2},
            }
        ]

    monkeypatch.setattr(runner, "_process_batch", _fake_process_batch)

    result = await runner._execute_evaluation(
        run_id="run_stage3_metrics_none",
        eval_id="eval_stage3_metrics_none",
        eval_config={"config": {"batch_size": 1, "max_workers": 1, "timeout_seconds": 1.0}},
    )

    assert result["aggregate"]["mean_score"] == pytest.approx(0.9)
    assert result["by_metric"]["coherence"]["mean"] == pytest.approx(0.9)
    assert captured_results["results"] == result
