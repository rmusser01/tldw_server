import asyncio
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.services import workflows_artifact_gc_service as service


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


async def _run_one_gc_iteration(monkeypatch: pytest.MonkeyPatch, db: Any, logger: _LoggerStub) -> None:
    stop_event = asyncio.Event()
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_RETENTION_DAYS", "1")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service, "get_content_backend_instance", lambda: object())
    monkeypatch.setattr(service, "create_workflows_database", lambda backend: db)

    async def _fake_wait_for(awaitable: Any, timeout: float) -> None:
        if hasattr(awaitable, "close"):
            awaitable.close()
        stop_event.set()

    monkeypatch.setattr(service.asyncio, "wait_for", _fake_wait_for)

    await service.run_workflows_artifact_gc_worker(stop_event)


@pytest.mark.asyncio
async def test_file_delete_failure_log_is_sanitized(monkeypatch, tmp_path):
    artifact_path = tmp_path / "artifact-secret-token.txt"
    artifact_path.write_text("content", encoding="utf-8")

    class _FakeDB:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def list_artifacts_older_than(self, _cutoff_iso: str) -> list[dict[str, str]]:
            return [{"artifact_id": "artifact-file-id", "uri": f"file://{artifact_path}"}]

        def delete_artifact(self, artifact_id: str) -> None:
            self.deleted.append(artifact_id)

    def _fail_unlink(self: Path) -> None:
        raise OSError(f"cannot unlink {artifact_path} with sk-live-artifact-token")

    logger = _LoggerStub()
    db = _FakeDB()
    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    await _run_one_gc_iteration(monkeypatch, db, logger)

    assert db.deleted == ["artifact-file-id"]
    assert "Artifact GC: failed to delete artifact file" in logger.warnings
    assert "Artifact GC: failed to append workflow evidence" in logger.warnings
    assert {"error_type": "OSError"} in logger.binds
    assert {"error_type": "AttributeError"} in logger.binds
    rendered = "\n".join(logger.infos + logger.warnings)
    assert str(artifact_path) not in rendered
    assert "sk-live-artifact-token" not in rendered
    assert "cannot unlink" not in rendered


@pytest.mark.asyncio
async def test_per_artifact_failure_log_is_sanitized(monkeypatch):
    class _FakeDB:
        def list_artifacts_older_than(self, _cutoff_iso: str) -> list[dict[str, str]]:
            return [{"artifact_id": "artifact-secret-id", "uri": ""}]

        def delete_artifact(self, artifact_id: str) -> None:
            raise RuntimeError(f"delete failed for {artifact_id} at /tmp/workflows-artifact-secret")

    logger = _LoggerStub()

    await _run_one_gc_iteration(monkeypatch, _FakeDB(), logger)

    assert logger.warnings == ["Artifact GC: error deleting artifact"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.infos + logger.warnings)
    assert "artifact-secret-id" not in rendered
    assert "/tmp/workflows-artifact-secret" not in rendered
    assert "delete failed" not in rendered


@pytest.mark.asyncio
async def test_outer_loop_failure_log_is_sanitized(monkeypatch):
    class _FakeDB:
        def list_artifacts_older_than(self, _cutoff_iso: str) -> list[dict[str, str]]:
            raise RuntimeError("artifact list failed at /tmp/workflows-gc-loop-secret")

    logger = _LoggerStub()

    await _run_one_gc_iteration(monkeypatch, _FakeDB(), logger)

    assert logger.warnings == ["Artifact GC loop error"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.infos + logger.warnings)
    assert "/tmp/workflows-gc-loop-secret" not in rendered
    assert "artifact list failed" not in rendered
