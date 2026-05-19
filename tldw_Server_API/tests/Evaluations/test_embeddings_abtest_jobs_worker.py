import pytest

import tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs_worker as worker
from tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs import ABTEST_JOBS_JOB_TYPE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_abtest_job_invokes_runner(monkeypatch):
    called = {}

    async def _fake_run_abtest_full(db, config, test_id, user_id, media_db):
        called["db"] = db
        called["config"] = config
        called["test_id"] = test_id
        called["user_id"] = user_id
        called["media_db"] = media_db

    class _Svc:
        def __init__(self):
            self.db = object()

    monkeypatch.setattr(worker, "get_unified_evaluation_service_for_user", lambda uid: _Svc())
    monkeypatch.setattr(worker, "_build_media_db", lambda user_id: object())
    monkeypatch.setattr(worker, "run_abtest_full", _fake_run_abtest_full)

    payload = {
        "test_id": "abtest_123",
        "config": {
            "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
            "media_ids": [],
            "retrieval": {"k": 3, "search_mode": "vector"},
            "queries": [{"text": "hello"}],
            "metric_level": "media",
        },
    }
    job = {
        "job_type": ABTEST_JOBS_JOB_TYPE,
        "payload": payload,
        "owner_user_id": "1",
    }

    result = await worker.handle_abtest_job(job)
    assert result["test_id"] == "abtest_123"
    assert called["test_id"] == "abtest_123"
    assert called["user_id"] == "1"
    assert called["config"].chunking is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_abtest_job_preserves_string_owner_scope_for_bindings(monkeypatch):
    captured = {}

    async def _fake_run_abtest_full(db, config, test_id, user_id, media_db):
        captured["db"] = db
        captured["config"] = config
        captured["test_id"] = test_id
        captured["run_user_id"] = user_id
        captured["media_db"] = media_db

    class _Svc:
        def __init__(self):
            self.db = object()

    def _service_for_user(user_id):
        captured["service_user_id"] = user_id
        return _Svc()

    def _build_media_db(user_id):
        captured["media_db_user_id"] = user_id
        return {"user_id": user_id}

    monkeypatch.setattr(worker, "get_unified_evaluation_service_for_user", _service_for_user)
    monkeypatch.setattr(worker, "_build_media_db", _build_media_db)
    monkeypatch.setattr(worker, "run_abtest_full", _fake_run_abtest_full)

    payload = {
        "test_id": "abtest_tenant",
        "config": {
            "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
            "media_ids": [],
            "retrieval": {"k": 3, "search_mode": "vector"},
            "queries": [{"text": "hello"}],
            "metric_level": "media",
        },
    }
    job = {
        "job_type": ABTEST_JOBS_JOB_TYPE,
        "payload": payload,
        "owner_user_id": "tenant-user",
    }

    result = await worker.handle_abtest_job(job)
    assert result["test_id"] == "abtest_tenant"
    assert captured["service_user_id"] == "tenant-user"
    assert captured["media_db_user_id"] == "tenant-user"
    assert captured["run_user_id"] == "tenant-user"
    assert captured["media_db"] == {"user_id": "tenant-user"}
