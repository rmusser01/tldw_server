import json

import pytest

from tldw_Server_API.app.core.Evaluations.embeddings_abtest_repository import (
    get_embeddings_abtest_store,
)


class _StubDb:
    def get_abtest(self, test_id, created_by=None):
        return {"test_id": test_id, "created_by": created_by or "tester"}

    def list_abtest_results(self, test_id, limit, offset, created_by=None):
        return (
            [
                {
                    "result_id": "res1",
                    "test_id": test_id,
                    "arm_id": "arm1",
                    "query_id": "q1",
                    "ranked_ids": json.dumps(["1", "2"]),
                    "scores": json.dumps([0.9, 0.1]),
                    "metrics_json": json.dumps({"recall_at_k": 1.0}),
                    "latency_ms": 12.5,
                    "ranked_distances": json.dumps([0.1, 0.9]),
                    "ranked_metadatas": json.dumps([{"media_id": "1"}]),
                    "ranked_documents": json.dumps(["doc"]),
                    "rerank_scores": json.dumps([0.95, 0.05]),
                    "created_at": "2024-01-01T00:00:00Z",
                }
            ],
            1,
        )

    def record_idempotency(self, *_args, **_kwargs):
        return None


class _StubService:
    def __init__(self):
        self.db = _StubDb()


class _StubUser:
    def __init__(self, user_id="1"):
        self.id = user_id


class _StubPrincipal:
    pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_abtest_export_json_parses_payload(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified as endpoints

    monkeypatch.setattr("tldw_Server_API.app.core.Evaluations.unified_evaluation_service.get_unified_evaluation_service_for_user", lambda _uid: _StubService())
    monkeypatch.setattr("tldw_Server_API.app.core.Evaluations.audit_adapter.log_evaluation_exported", lambda **_kwargs: None)
    monkeypatch.setattr(endpoints, "enforce_heavy_evaluations_admin", lambda _principal: None)

    resp = await endpoints.export_embeddings_abtest(
        test_id="abtest_123",
        principal=_StubPrincipal(),
        current_user=_StubUser(),
        format="json",
        user_ctx="tester",
        idempotency_key=None,
    )

    payload = json.loads(resp.body.decode("utf-8"))
    row = payload["results"][0]
    assert isinstance(row["ranked_ids"], list)
    assert isinstance(row["metrics_json"], dict)
    assert isinstance(row["ranked_metadatas"], list)
    assert isinstance(row["rerank_scores"], list)


@pytest.mark.unit
def test_abtest_store_accepts_sqlalchemy_url(tmp_path):
    db_path = tmp_path / "abtest_repo.db"
    store = get_embeddings_abtest_store(f"sqlite:///{db_path}")
    test_id = store.create_abtest(name="url", config={"arms": []}, created_by="tester")
    row = store.get_abtest(test_id)
    assert row["test_id"] == test_id
