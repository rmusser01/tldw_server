from __future__ import annotations

import json
from pathlib import Path

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.jobs import (
    CODEGRAPH_INDEX_JOB_TYPE,
    CODEGRAPH_JOBS_DOMAIN,
    build_codegraph_index_job_payload,
    enqueue_codegraph_index_job,
)
from tldw_Server_API.app.core.CodeGraph.models import WorkspaceResolution
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _workspace_resolution(tmp_path: Path) -> WorkspaceResolution:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    index_db_path = tmp_path / "indexes" / "ws_test" / "codegraph.db"
    return WorkspaceResolution(
        workspace_root=workspace_root,
        workspace_key="ws_test",
        index_db_path=index_db_path,
        workspace_id="workspace-1",
        source="test",
    )


def test_codegraph_job_payload_is_json_safe(tmp_path: Path) -> None:
    settings = CodeGraphSettings.from_mapping({"index_base_dir": str(tmp_path / "indexes")})
    payload = build_codegraph_index_job_payload(
        resolution=_workspace_resolution(tmp_path),
        settings=settings,
        operation="index",
        force=True,
        languages=["python"],
        max_files=25,
    )

    json.dumps(payload)

    assert payload["operation"] == "index"
    assert payload["force"] is True
    assert payload["languages"] == ["python"]
    assert payload["max_files"] == 25
    assert payload["workspace_root"] == str(tmp_path / "workspace")
    assert payload["workspace_key"] == "ws_test"
    assert payload["workspace_id"] == "workspace-1"
    assert payload["workspace_source"] == "test"
    assert payload["index_db_path"] == str(tmp_path / "indexes" / "ws_test" / "codegraph.db")
    assert payload["settings"]["index_base_dir"] == str(tmp_path / "indexes")
    assert isinstance(payload["settings"]["exclude_dirs"], list)


def test_codegraph_job_payload_serializes_absolute_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace_root = Path("workspace")
    index_db_path = Path("indexes") / "ws_test" / "codegraph.db"
    resolution = WorkspaceResolution(
        workspace_root=workspace_root,
        workspace_key="ws_test",
        index_db_path=index_db_path,
        workspace_id="workspace-1",
        source="test",
    )
    settings = CodeGraphSettings.from_mapping({"index_base_dir": "indexes"})

    payload = build_codegraph_index_job_payload(
        resolution=resolution,
        settings=settings,
        operation="index",
        force=False,
        languages=None,
        max_files=None,
    )

    assert payload["workspace_root"] == str((tmp_path / "workspace").resolve(strict=False))
    assert payload["index_db_path"] == str((tmp_path / "indexes" / "ws_test" / "codegraph.db").resolve(strict=False))
    assert payload["settings"]["index_base_dir"] == str((tmp_path / "indexes").resolve(strict=False))


def test_enqueue_codegraph_index_job_uses_codegraph_domain_queue_and_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CODEGRAPH_JOBS_QUEUE", "low")
    jobs = JobManager(tmp_path / "jobs.db")
    settings = CodeGraphSettings.from_mapping({"index_base_dir": str(tmp_path / "indexes")})

    job = enqueue_codegraph_index_job(
        jm=jobs,
        resolution=_workspace_resolution(tmp_path),
        settings=settings,
        operation="sync",
        force=False,
        languages=None,
        max_files=None,
        owner_user_id="7",
    )

    assert job["domain"] == CODEGRAPH_JOBS_DOMAIN
    assert job["queue"] == "low"
    assert job["job_type"] == CODEGRAPH_INDEX_JOB_TYPE
    assert job["owner_user_id"] == "7"
    assert job["status"] == "queued"
    assert job["payload"]["operation"] == "sync"
    assert job["payload"]["workspace_key"] == "ws_test"
