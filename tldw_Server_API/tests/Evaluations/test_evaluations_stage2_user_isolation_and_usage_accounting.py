import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner


@pytest.mark.asyncio
async def test_rag_pipeline_ephemeral_ops_use_run_user_context(tmp_path, monkeypatch):
    runner = EvaluationRunner(str(tmp_path / "evals_stage2.db"), max_concurrent_evals=2, eval_timeout=10)

    monkeypatch.setattr(runner.db, "update_run_progress", lambda *_args, **_kwargs: None)

    build_user_ids: list[str] = []
    cleanup_user_ids: list[str] = []
    deleted_collections: list[str] = []
    pipeline_user_ids: list[str] = []

    async def _fake_build_ephemeral_index(*, collection_name, samples, chunking_cfg, user_id):
        _ = (collection_name, samples, chunking_cfg)
        build_user_ids.append(user_id)
        return {"total_chunks": 1}

    monkeypatch.setattr(runner, "_build_ephemeral_index", _fake_build_ephemeral_index)

    class _FakeAdapter:
        async def initialize(self):
            return None

        async def delete_collection(self, collection_name: str):
            deleted_collections.append(collection_name)

    def _fake_create_from_settings_for_user(_settings, user_id):
        cleanup_user_ids.append(user_id)
        return _FakeAdapter()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.eval_runner.create_from_settings_for_user",
        _fake_create_from_settings_for_user,
    )

    async def _fake_unified_rag_pipeline(*, query, **kwargs):
        _ = query
        pipeline_user_ids.append(str(kwargs.get("user_id")))
        return {
            "documents": [{"id": "doc_1", "content": "context"}],
            "generated_answer": "answer",
            "timings": {"total": 0.001},
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.eval_runner.unified_rag_pipeline",
        _fake_unified_rag_pipeline,
    )

    class _FakeRagEvaluator:
        async def evaluate(self, **_kwargs):
            return {
                "metrics": {"relevance": {"score": 0.9}, "faithfulness": {"score": 0.8}},
                "overall_score": 0.85,
            }

    runner._rag_evaluator = _FakeRagEvaluator()

    eval_spec = {
        "sub_type": "rag_pipeline",
        "metrics": ["relevance", "faithfulness"],
        "rag_pipeline": {
            "index_namespace": "tenant_scope",
            "cleanup_collections": True,
            "custom_metrics": False,
        },
    }
    samples = [{"input": {"query": "q1", "corpus": ["doc text"]}, "expected": {"answer": "answer"}}]

    results, usage = await runner._execute_rag_pipeline_run(
        run_id="run_stage2_user_scope",
        samples=samples,
        eval_spec=eval_spec,
        eval_config={"config": {}},
        run_user_id="tenant-42",
    )

    assert results["config_count"] == 1
    assert usage["total_tokens"] == 0
    assert build_user_ids == ["tenant-42"]
    assert cleanup_user_ids == ["tenant-42"]
    assert pipeline_user_ids == ["tenant-42"]
    assert deleted_collections == ["tenant_scope_cfg_001"]


def test_ocr_pdf_uses_consistent_endpoint_key_for_check_and_record(monkeypatch):
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    class _Limiter:
        def __init__(self):
            self.checked_endpoints: list[str] = []
            self.recorded_endpoints: list[str] = []

        async def check_rate_limit(
            self,
            _user_id: str,
            *,
            endpoint: str,
            is_batch: bool,
            tokens_requested: int,
            estimated_cost: float,
            evaluations_requested: int = 1,
        ):
            _ = (is_batch, tokens_requested, estimated_cost, evaluations_requested)
            self.checked_endpoints.append(endpoint)
            return True, {"retry_after": 0}

        async def record_actual_usage(
            self,
            _user_id: str,
            endpoint: str,
            _tokens_used: int,
            _cost: float = 0.0,
            reserved_tokens: int = 0,
        ):
            _ = reserved_tokens
            self.recorded_endpoints.append(endpoint)

    limiter = _Limiter()

    async def _verify_api_key_override():
        return "user_1"

    async def _get_user_override():
        return User(id=1, username="tester", email=None, is_active=True)

    async def _rate_limit_dep_override():
        return None

    async def _fake_apply_rate_limit_headers(_limiter, _user_id, response, _meta):
        response.headers["X-Stage2-RateLimit-Applied"] = "true"

    class _Service:
        async def evaluate_ocr(self, **_kwargs):
            return {
                "evaluation_id": "eval_ocr_pdf_1",
                "results": {"metrics": {}},
                "evaluation_time": 0.01,
                "usage": {"total_tokens": 42, "cost": 0.123},
            }

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.check_evaluation_rate_limit] = _rate_limit_dep_override

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: limiter)
    monkeypatch.setattr(eval_unified, "_apply_rate_limit_headers", _fake_apply_rate_limit_headers)
    monkeypatch.setattr(eval_unified, "get_unified_evaluation_service_for_user", lambda _uid: _Service())

    files = [
        ("files", ("sample.pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF", "application/pdf")),
    ]
    with TestClient(app) as client:
        response = client.post("/api/v1/evaluations/ocr-pdf", files=files)

    assert response.status_code == 200, response.text
    assert response.json()["evaluation_id"] == "eval_ocr_pdf_1"
    assert response.headers.get("X-Stage2-RateLimit-Applied") == "true"
    assert limiter.checked_endpoints == ["evals:ocr_pdf"]
    assert limiter.recorded_endpoints == ["evals:ocr_pdf"]
