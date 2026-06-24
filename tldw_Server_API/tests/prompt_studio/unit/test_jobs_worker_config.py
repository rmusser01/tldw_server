import os

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import jobs_worker


pytestmark = pytest.mark.unit


def test_build_worker_config_uses_env(monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS", "40")
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_RENEW_JITTER_SECONDS", "3")
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_RENEW_THRESHOLD_SECONDS", "11")

    cfg = jobs_worker._build_worker_config(worker_id="w1", queue="default")

    assert cfg.lease_seconds == 40
    assert cfg.renew_jitter_seconds == 3
    assert cfg.renew_threshold_seconds == 11
    assert cfg.worker_id == "w1"
    assert cfg.queue == "default"


def test_build_worker_config_heartbeat_override(monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS", "60")
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_RENEW_THRESHOLD_SECONDS", "10")
    monkeypatch.setenv("TLDW_PS_HEARTBEAT_SECONDS", "15")

    cfg = jobs_worker._build_worker_config(worker_id="w2", queue="default")

    assert cfg.renew_threshold_seconds == 45


def test_build_worker_config_heartbeat_exceeds_lease(monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS", "10")
    monkeypatch.setenv("TLDW_PS_HEARTBEAT_SECONDS", "20")

    cfg = jobs_worker._build_worker_config(worker_id="w3", queue="default")

    assert cfg.renew_threshold_seconds == 1


@pytest.mark.asyncio
async def test_handle_job_requires_owner_user_id(monkeypatch):
    monkeypatch.setattr(
        jobs_worker,
        "_get_processor",
        lambda _user_id: (_ for _ in ()).throw(AssertionError("processor should not be created")),
        raising=True,
    )

    with pytest.raises(jobs_worker.PromptStudioJobError) as exc_info:
        await jobs_worker._handle_job(
            {
                "id": 1,
                "uuid": "job-1",
                "job_type": "generation",
                "payload": {"project_id": 42, "user_id": "payload-user"},
            }
        )

    assert exc_info.value.retryable is False
    assert "owner_user_id" in str(exc_info.value)


def test_db_cache_is_bounded_and_closes_evicted_entries(monkeypatch):
    class FakeDB:
        def __init__(self, user_id: str) -> None:
            self.user_id = user_id
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    created: list[FakeDB] = []

    def _fake_create_prompt_studio_database(*, client_id, db_path, backend):
        db = FakeDB(db_path.rsplit("/", 1)[-1])
        created.append(db)
        return db

    jobs_worker._DB_CACHE.clear()
    jobs_worker._PROCESSOR_CACHE.clear()
    monkeypatch.setattr(jobs_worker, "_MAX_CACHE_ENTRIES", 1, raising=False)
    monkeypatch.setattr(jobs_worker, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        jobs_worker.DatabasePaths,
        "get_prompt_studio_db_path",
        lambda user_id: f"/tmp/{user_id}",
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "create_prompt_studio_database",
        _fake_create_prompt_studio_database,
        raising=True,
    )

    first = jobs_worker._get_db("user-1")
    second = jobs_worker._get_db("user-2")

    assert first is created[0]
    assert second is created[1]
    assert first.closed is True
    assert list(jobs_worker._DB_CACHE.keys()) == ["user-2"]
