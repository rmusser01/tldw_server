import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_jobs_worker_caches():
    with jobs_worker._CACHE_LOCK:
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._ACTIVE_USER_COUNTS.clear()
        jobs_worker._PENDING_CLOSE.clear()
    yield
    with jobs_worker._CACHE_LOCK:
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._ACTIVE_USER_COUNTS.clear()
        jobs_worker._PENDING_CLOSE.clear()


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


@pytest.mark.asyncio
async def test_handle_job_marks_owner_active_during_processing(monkeypatch):
    class FakeProcessor:
        async def process_generation_job(self, payload, entity_id):
            assert jobs_worker._ACTIVE_USER_COUNTS["user-1"] == 1
            return {"entity_id": entity_id, "job_id": payload["job_id"]}

    monkeypatch.setattr(jobs_worker, "_get_processor", lambda _user_id: FakeProcessor(), raising=True)

    result = await jobs_worker._handle_job(
        {
            "id": 1,
            "uuid": "job-1",
            "job_type": "generation",
            "owner_user_id": "user-1",
            "payload": {"project_id": 42},
        }
    )

    assert result == {"entity_id": 42, "job_id": "job-1"}
    assert "user-1" not in jobs_worker._ACTIVE_USER_COUNTS


def test_worker_database_separates_owner_tenant_from_audit_client(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDB:
        user_id: str | None = None

    def _fake_create_prompt_studio_database(**kwargs):
        captured.update(kwargs)
        return FakeDB()

    backend = object()
    monkeypatch.setattr(
        jobs_worker,
        "get_content_backend_instance",
        lambda: backend,
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker.DatabasePaths,
        "get_prompt_studio_db_path",
        lambda user_id: f"/tmp/{user_id}/prompt-studio.db",
        raising=True,
    )
    monkeypatch.setattr(
        jobs_worker,
        "create_prompt_studio_database",
        _fake_create_prompt_studio_database,
        raising=True,
    )

    db = jobs_worker._get_db("7")

    assert captured["client_id"] == "prompt_studio_jobs_worker:7"
    assert captured["tenant_user_id"] == "7"
    assert captured["backend"] is backend
    assert db.user_id == "7"


def test_db_cache_is_bounded_and_closes_evicted_entries(monkeypatch):
    class FakeDB:
        def __init__(self, user_id: str) -> None:
            self.user_id = user_id
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    created: list[FakeDB] = []

    def _fake_create_prompt_studio_database(
        *, client_id, db_path, tenant_user_id, backend
    ):
        assert tenant_user_id == db_path.rsplit("/", 1)[-1]
        db = FakeDB(db_path.rsplit("/", 1)[-1])
        created.append(db)
        return db

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


def test_db_cache_defers_closing_active_evicted_entries(monkeypatch):
    class FakeDB:
        def __init__(self, user_id: str) -> None:
            self.user_id = user_id
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    created: list[FakeDB] = []

    def _fake_create_prompt_studio_database(
        *, client_id, db_path, tenant_user_id, backend
    ):
        assert tenant_user_id == db_path.rsplit("/", 1)[-1]
        db = FakeDB(db_path.rsplit("/", 1)[-1])
        created.append(db)
        return db

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

    with jobs_worker._active_user_cache_scope("user-1"):
        first = jobs_worker._get_db("user-1")
        second = jobs_worker._get_db("user-2")

        assert first is created[0]
        assert second is created[1]
        assert first.closed is False
        assert jobs_worker._PENDING_CLOSE["user-1"] == [first]
        assert list(jobs_worker._DB_CACHE.keys()) == ["user-2"]

    assert first.closed is True


def test_close_db_logs_cleanup_failures(monkeypatch):
    class FailingDB:
        def close_connection(self) -> None:
            raise RuntimeError("close failed")

    class FakeLogger:
        def __init__(self) -> None:
            self.exception = None
            self.warnings: list[tuple[str, tuple[object, ...]]] = []

        def opt(self, **kwargs):
            self.exception = kwargs.get("exception")
            return self

        def warning(self, message, *args) -> None:
            self.warnings.append((message, args))

    fake_logger = FakeLogger()
    monkeypatch.setattr(jobs_worker, "logger", fake_logger, raising=True)

    jobs_worker._close_db(FailingDB(), user_id="user-1")

    assert isinstance(fake_logger.exception, RuntimeError)
    assert fake_logger.warnings == [
        ("Failed to close Prompt Studio DB for user {} via {}", ("user-1", "close_connection"))
    ]
