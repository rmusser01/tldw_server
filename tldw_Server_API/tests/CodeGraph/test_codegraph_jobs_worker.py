from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.jobs import (
    CODEGRAPH_INDEX_JOB_TYPE,
    build_codegraph_index_job_payload,
)
from tldw_Server_API.app.core.CodeGraph.jobs_worker import handle_codegraph_index_job
from tldw_Server_API.app.core.CodeGraph.models import WorkspaceResolution
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository
from tldw_Server_API.app.core.exceptions import CodeGraphJobError


@pytest.fixture(autouse=True)
def _worker_index_base_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEGRAPH_JOBS_INDEX_BASE_DIR", str(tmp_path / "indexes"))


def _resolution(tmp_path: Path) -> WorkspaceResolution:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    return WorkspaceResolution(
        workspace_root=workspace_root,
        workspace_key="ws_test",
        index_db_path=tmp_path / "indexes" / "ws_test" / "codegraph.db",
        workspace_id="workspace-1",
        source="test",
    )


def _payload(
    tmp_path: Path,
    *,
    operation: str,
    index_db_path: Path | None = None,
) -> dict[str, object]:
    resolution = _resolution(tmp_path)
    if index_db_path is not None:
        resolution = WorkspaceResolution(
            workspace_root=resolution.workspace_root,
            workspace_key=resolution.workspace_key,
            index_db_path=index_db_path,
            workspace_id=resolution.workspace_id,
            source=resolution.source,
        )
    settings = CodeGraphSettings.from_mapping({"index_base_dir": str(tmp_path / "indexes")})
    return build_codegraph_index_job_payload(
        resolution=resolution,
        settings=settings,
        operation=operation,
        force=True,
        languages=["python"],
        max_files=10,
    )


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_indexes_workspace(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="index")
    workspace_root = Path(str(payload["workspace_root"]))
    (workspace_root / "app.py").write_text(
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )

    result = await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    repo = CodeGraphRepository(Path(str(payload["index_db_path"])))
    assert result["status"] == "complete"
    assert result["workspace_key"] == "ws_test"
    assert result["counters"]["files_indexed"] == 1
    assert repo.find_node_by_symbol("helper") is not None


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_syncs_workspace(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="sync")
    workspace_root = Path(str(payload["workspace_root"]))
    (workspace_root / "app.py").write_text("value = 1\n", encoding="utf-8")

    result = await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert result["status"] == "complete"
    assert result["operation"] == "sync"
    assert result["counters"]["files_indexed"] == 1


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_wraps_indexer_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(tmp_path, operation="index")

    class _ExplodingIndexer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def index_workspace(self, *_args, **_kwargs):
            raise RuntimeError("indexer exploded")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.CodeGraph.jobs_worker.CodeGraphIndexer",
        _ExplodingIndexer,
    )

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "codegraph_job_execution_failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_unsupported_job_type(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="index")

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": "other", "payload": payload})

    assert excinfo.value.retryable is False
    assert "unsupported job_type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_missing_operation(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="index")
    del payload["operation"]

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "unsupported operation: <missing>" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_unsupported_operation(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="index")
    payload["operation"] = "watch"

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "unsupported operation: watch" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_missing_paths(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="index")
    del payload["workspace_root"]

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "missing workspace_root" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_unsafe_index_path(tmp_path: Path) -> None:
    payload = _payload(
        tmp_path,
        operation="index",
        index_db_path=tmp_path / "outside" / "codegraph.db",
    )

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "index_db_path_outside_index_base" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_spoofed_payload_index_base(tmp_path: Path) -> None:
    payload = _payload(tmp_path, operation="index")
    settings_payload = payload["settings"]
    assert isinstance(settings_payload, dict)
    settings = dict(settings_payload)
    settings["index_base_dir"] = "/"
    payload["settings"] = settings

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "index_base_dir_mismatch" in str(excinfo.value)


@pytest.mark.asyncio
async def test_codegraph_jobs_worker_rejects_index_base_as_database_path(tmp_path: Path) -> None:
    payload = _payload(
        tmp_path,
        operation="index",
        index_db_path=tmp_path / "indexes",
    )

    with pytest.raises(CodeGraphJobError) as excinfo:
        await handle_codegraph_index_job({"job_type": CODEGRAPH_INDEX_JOB_TYPE, "payload": payload})

    assert excinfo.value.retryable is False
    assert "index_db_path_outside_index_base" in str(excinfo.value)
