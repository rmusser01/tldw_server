from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import build_inventory_ignore_policy
from tldw_Server_API.app.core.Workspaces.file_inventory_jobs import (
    WORKSPACE_FILE_INVENTORY_JOB_TYPE,
    WORKSPACE_JOBS_DOMAIN,
    WorkspaceFileInventoryEnqueueError,
    enqueue_workspace_file_inventory_scan_job,
)


class _RecordingJobs:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.created_jobs: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("jobs unavailable")
        idempotency_key = kwargs.get("idempotency_key")
        if idempotency_key:
            for row in self.created_jobs:
                if row.get("idempotency_key") == idempotency_key:
                    return row
        job_id = len(self.created_jobs) + 100
        row = {
            "id": job_id,
            "uuid": f"job-{job_id}",
            "status": "queued",
            **kwargs,
        }
        self.created_jobs.append(row)
        return row

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        for row in self.created_jobs:
            if int(row["id"]) == int(job_id):
                return row
        return None


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    chacha = CharactersRAGDB(str(tmp_path / "workspace-file-inventory-jobs.db"), client_id="workspace-jobs-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


def _workspace_with_root(db: CharactersRAGDB, root_path: Path) -> dict[str, Any]:
    db.upsert_workspace("ws-1", "Workspace", workspace_profile="project")
    root_path.mkdir(parents=True, exist_ok=True)
    return db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "absolute_root": str(root_path),
            "root_state": "attached",
        },
    )


def test_enqueue_job_payload_contains_no_absolute_path_and_uses_scan_id_key(
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    policy = build_inventory_ignore_policy()
    jobs = _RecordingJobs()

    result = enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=jobs,
    )

    created = jobs.created_jobs[0]
    payload = created["payload"]
    assert created["domain"] == WORKSPACE_JOBS_DOMAIN
    assert created["job_type"] == WORKSPACE_FILE_INVENTORY_JOB_TYPE
    assert created["idempotency_key"] == f"workspace-file-inventory-scan:{result['scan']['scan_id']}"
    assert payload == {
        "workspace_id": "ws-1",
        "root_id": "root-1",
        "root_version": int(root["version"]),
        "scan_id": result["scan"]["scan_id"],
        "ignore_policy_fingerprint": policy.fingerprint,
        "requested_by": "user-1",
    }
    assert str(tmp_path) not in str(payload)
    assert result["scan"]["job_id"] == created["id"]
    assert result["status"]["scan_id"] == result["scan"]["scan_id"]


def test_enqueue_creates_scan_row_before_job_create(db: CharactersRAGDB, tmp_path: Path) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    policy = build_inventory_ignore_policy()

    class _AssertScanExistsJobs(_RecordingJobs):
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            scan_id = kwargs["payload"]["scan_id"]
            row = db.execute_query(
                "SELECT scan_id, job_id FROM workspace_file_inventory_scans WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
            assert row is not None
            assert row["job_id"] is None
            return super().create_job(**kwargs)

    enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=_AssertScanExistsJobs(),
    )


def test_enqueue_failure_marks_scan_failed_and_next_enqueue_does_not_reuse_it(
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    policy = build_inventory_ignore_policy()

    with pytest.raises(WorkspaceFileInventoryEnqueueError) as excinfo:
        enqueue_workspace_file_inventory_scan_job(
            db=db,
            workspace_id="ws-1",
            root_id=root["root_id"],
            root_version=int(root["version"]),
            policy_fingerprint=policy.fingerprint,
            requested_by="user-1",
            owner_user_id="1",
            job_manager=_RecordingJobs(fail=True),
        )

    assert excinfo.value.error_code == "workspace_file_inventory_enqueue_failed"
    failed_scan = db.execute_query("SELECT scan_id, state, job_id FROM workspace_file_inventory_scans").fetchone()
    assert failed_scan["state"] == "failed"
    assert failed_scan["job_id"] is None

    jobs = _RecordingJobs()
    result = enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=jobs,
    )

    assert result["scan"]["scan_id"] != failed_scan["scan_id"]
    assert result["scan"]["state"] == "queued"
    assert result["scan"]["job_id"] == jobs.created_jobs[0]["id"]


def test_enqueue_reuses_active_scan_only_when_job_id_exists(db: CharactersRAGDB, tmp_path: Path) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    policy = build_inventory_ignore_policy()
    stale_unattached = db.begin_workspace_file_inventory_scan(
        "ws-1",
        root["root_id"],
        int(root["version"]),
        policy.fingerprint,
        requested_by="user-1",
    )
    jobs = _RecordingJobs()

    first = enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=jobs,
    )
    second = enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=jobs,
    )

    assert first["scan"]["scan_id"] != stale_unattached["scan_id"]
    assert second["scan"]["scan_id"] == first["scan"]["scan_id"]
    assert second["job"]["id"] == first["job"]["id"]


def test_enqueue_uses_real_job_manager_idempotency(db: CharactersRAGDB, tmp_path: Path) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    policy = build_inventory_ignore_policy()
    jobs = JobManager(tmp_path / "jobs.db")

    first = enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=jobs,
    )
    second = enqueue_workspace_file_inventory_scan_job(
        db=db,
        workspace_id="ws-1",
        root_id=root["root_id"],
        root_version=int(root["version"]),
        policy_fingerprint=policy.fingerprint,
        requested_by="user-1",
        owner_user_id="1",
        job_manager=jobs,
    )

    assert second["scan"]["scan_id"] == first["scan"]["scan_id"]
    assert second["job"]["id"] == first["job"]["id"]
