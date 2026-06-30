import asyncio
import contextlib
from typing import Any

import pytest

import tldw_Server_API.app.services.jobs_prune_scheduler as scheduler


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


def test_int_optional_invalid_env_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(
        scheduler,
        "_raw_setting",
        lambda _env_name, _config_key, default=None: "not-int /tmp/jobs-prune-secret sk-live-prune",
    )

    assert scheduler._int_optional("JOBS_PRUNE_INTERVAL_SEC", "prune_interval_sec") is None
    assert logger.debugs == ["jobs_prune: invalid JOBS_PRUNE_INTERVAL_SEC; using default"]
    assert logger.binds == [{"error_type": "ValueError"}]
    rendered = "\n".join(logger.debugs)
    assert "/tmp/jobs-prune-secret" not in rendered
    assert "sk-live-prune" not in rendered
    assert "not-int" not in rendered


@pytest.mark.asyncio
async def test_jobs_prune_runner_failure_log_is_sanitized(monkeypatch):
    monkeypatch.setenv("JOBS_PRUNE_ENFORCE", "true")
    monkeypatch.setenv("JOBS_PRUNE_INTERVAL_SEC", "60")

    logger = _LoggerStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "_build_retention_groups", lambda: {30: ["completed"]})
    monkeypatch.setattr(scheduler, "_split_csv", lambda *_args, **_kwargs: [])

    class _FailingJobManager:
        @staticmethod
        def set_rls_context(**_kwargs: Any) -> None:
            return None

        @staticmethod
        def clear_rls_context() -> None:
            return None

        def prune_jobs(self, **_kwargs: Any) -> int:
            raise RuntimeError("jobs prune runner leaked /tmp/jobs-prune-runner-secret sk-live-runner")

    monkeypatch.setattr(scheduler, "JobManager", _FailingJobManager)

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr(scheduler.asyncio, "sleep", _fake_sleep)

    task = await scheduler.start_jobs_prune_scheduler()
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert logger.warnings == ["Jobs prune run failed"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/jobs-prune-runner-secret" not in rendered
    assert "sk-live-runner" not in rendered
    assert "jobs prune runner leaked" not in rendered
