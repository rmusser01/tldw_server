import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner


def test_batch_parallel_strict_fail_fast_cancels_remaining(monkeypatch):
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
            user_id: str,
            webhook_user_id: str | None = None,
        ):
            _ = (summary, metrics, api_name, api_key, user_id, webhook_user_id)
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
