from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import build_inventory_ignore_policy
from tldw_Server_API.app.core.Workspaces.file_inventory_jobs import WORKSPACE_FILE_INVENTORY_JOB_TYPE
from tldw_Server_API.app.core.Workspaces.root_binding_service import resolve_workspace_root_for_inventory_scan
from tldw_Server_API.app.services.workspace_file_inventory_jobs_worker import (
    WorkspaceFileInventoryJobError,
    handle_workspace_file_inventory_job,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    chacha = CharactersRAGDB(str(tmp_path / "workspace-file-inventory-worker.db"), client_id="workspace-worker-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


def _workspace_with_host_root(db: CharactersRAGDB, root_path: Path) -> dict[str, Any]:
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


def _workspace_with_sandbox_root(db: CharactersRAGDB) -> dict[str, Any]:
    db.upsert_workspace("ws-1", "Workspace", workspace_profile="project")
    return db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-1",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
            "sandbox_mount_state": "not_configured",
        },
    )


def _attached_scan(db: CharactersRAGDB, root: dict[str, Any], policy_fingerprint: str) -> dict[str, Any]:
    scan = db.begin_workspace_file_inventory_scan(
        "ws-1",
        root["root_id"],
        int(root["version"]),
        policy_fingerprint,
        requested_by="user-1",
    )
    return db.attach_workspace_file_inventory_job(scan["scan_id"], {"id": 501, "uuid": "job-501"})


def _job(root: dict[str, Any], scan: dict[str, Any], policy_fingerprint: str) -> dict[str, Any]:
    return {
        "id": 501,
        "uuid": "job-501",
        "owner_user_id": "1",
        "job_type": WORKSPACE_FILE_INVENTORY_JOB_TYPE,
        "payload": {
            "workspace_id": "ws-1",
            "root_id": root["root_id"],
            "root_version": int(root["version"]),
            "scan_id": scan["scan_id"],
            "ignore_policy_fingerprint": policy_fingerprint,
            "requested_by": "user-1",
        },
    }


async def test_worker_rejects_unsupported_job_type() -> None:
    with pytest.raises(WorkspaceFileInventoryJobError) as excinfo:
        await handle_workspace_file_inventory_job({"job_type": "other", "payload": {}}, db=None)

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "unsupported_job_type"


async def test_worker_rejects_malformed_payload() -> None:
    with pytest.raises(WorkspaceFileInventoryJobError) as excinfo:
        await handle_workspace_file_inventory_job(
            {"job_type": WORKSPACE_FILE_INVENTORY_JOB_TYPE, "payload": "not-json"},
            db=None,
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "invalid_job_payload"


async def test_worker_root_version_mismatch_completes_scan_failed(
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "project"
    root = _workspace_with_host_root(db, root_path)
    policy = build_inventory_ignore_policy()
    scan = _attached_scan(db, root, policy.fingerprint)
    db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": root["root_id"],
            "backend": "host_local",
            "absolute_root": str(root_path),
            "display_name": "Changed root",
        },
    )

    result = await handle_workspace_file_inventory_job(
        _job(root, scan, policy.fingerprint),
        db=db,
        allowed_roots=[tmp_path],
    )

    updated = db.execute_query(
        "SELECT state, diagnostics_json FROM workspace_file_inventory_scans WHERE scan_id = ?",
        (scan["scan_id"],),
    ).fetchone()
    assert result["state"] == "failed"
    assert updated["state"] == "failed"
    assert "root_version_mismatch" in updated["diagnostics_json"]


async def test_worker_host_local_scan_completes_and_writes_items(
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "project"
    (root_path / "src").mkdir(parents=True)
    (root_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (root_path / "node_modules" / "pkg").mkdir(parents=True)
    (root_path / "node_modules" / "pkg" / "index.js").write_text("ignored", encoding="utf-8")
    root = _workspace_with_host_root(db, root_path)
    policy = build_inventory_ignore_policy()
    scan = _attached_scan(db, root, policy.fingerprint)

    result = await handle_workspace_file_inventory_job(
        _job(root, scan, policy.fingerprint),
        db=db,
        allowed_roots=[tmp_path],
    )

    status = db.get_workspace_file_inventory_status("ws-1", policy_fingerprint=policy.fingerprint)
    page = db.list_workspace_file_inventory_items("ws-1", include_ignored=True, limit=20)

    assert result["state"] == "current"
    assert result["counts"]["files"] == 1
    assert result["counts"]["ignored"] == 1
    assert status["state"] == "current"
    assert [item["relative_path"] for item in page["items"]] == ["src", "src/app.py"]
    assert str(tmp_path) not in str(result)


async def test_worker_policy_fingerprint_mismatch_fails_scan_without_writing_items(
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "project"
    (root_path / "notes.txt").parent.mkdir(parents=True)
    (root_path / "notes.txt").write_text("notes", encoding="utf-8")
    root = _workspace_with_host_root(db, root_path)
    scan = _attached_scan(db, root, "stale-policy-fingerprint")

    result = await handle_workspace_file_inventory_job(
        _job(root, scan, "stale-policy-fingerprint"),
        db=db,
        allowed_roots=[tmp_path],
    )

    updated = db.execute_query(
        "SELECT state, diagnostics_json FROM workspace_file_inventory_scans WHERE scan_id = ?",
        (scan["scan_id"],),
    ).fetchone()
    page = db.list_workspace_file_inventory_items("ws-1", include_ignored=True, limit=20)

    assert result["state"] == "failed"
    assert updated["state"] == "failed"
    assert "ignore_policy_fingerprint_mismatch" in updated["diagnostics_json"]
    assert page["items"] == []


async def test_worker_unexpected_scan_error_finalizes_scan_failed(
    db: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.services.workspace_file_inventory_jobs_worker as worker

    root_path = tmp_path / "project"
    root = _workspace_with_host_root(db, root_path)
    policy = build_inventory_ignore_policy()
    scan = _attached_scan(db, root, policy.fingerprint)

    def _raise_scan_error(*args: object, **kwargs: object) -> object:
        raise RuntimeError(f"scanner leaked {tmp_path}")

    monkeypatch.setattr(worker, "scan_workspace_file_inventory", _raise_scan_error)

    result = await handle_workspace_file_inventory_job(
        _job(root, scan, policy.fingerprint),
        db=db,
        allowed_roots=[tmp_path],
    )

    updated = db.execute_query(
        "SELECT state, diagnostics_json FROM workspace_file_inventory_scans WHERE scan_id = ?",
        (scan["scan_id"],),
    ).fetchone()
    root_after = db.get_workspace_primary_root("ws-1")

    assert result["state"] == "failed"
    assert updated["state"] == "failed"
    assert root_after["file_inventory_state"] == "failed"
    assert "workspace_file_inventory_job_failed" in updated["diagnostics_json"]
    assert str(tmp_path) not in updated["diagnostics_json"]


async def test_worker_sandbox_root_without_mount_resolver_fails_closed(db: CharactersRAGDB) -> None:
    root = _workspace_with_sandbox_root(db)
    policy = build_inventory_ignore_policy()
    scan = _attached_scan(db, root, policy.fingerprint)

    result = await handle_workspace_file_inventory_job(
        _job(root, scan, policy.fingerprint),
        db=db,
    )

    updated = db.execute_query(
        "SELECT state, diagnostics_json FROM workspace_file_inventory_scans WHERE scan_id = ?",
        (scan["scan_id"],),
    ).fetchone()
    assert result["state"] == "failed"
    assert updated["state"] == "failed"
    assert "sandbox_mount_not_ready" in updated["diagnostics_json"]


def test_root_resolution_rejects_missing_and_symlink_roots_without_mutating_db(
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    db.upsert_workspace("ws-1", "Workspace", workspace_profile="project")
    missing_root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "absolute_root": str(tmp_path / "missing"),
            "root_state": "attached",
        },
    )
    before_state = missing_root["file_inventory_state"]
    missing_resolution = resolve_workspace_root_for_inventory_scan(
        root=missing_root,
        allowed_roots=[tmp_path],
    )

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    symlink_root_path = tmp_path / "link-root"
    try:
        symlink_root_path.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    symlink_root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "absolute_root": str(symlink_root_path),
            "replace_existing": True,
        },
    )
    symlink_resolution = resolve_workspace_root_for_inventory_scan(
        root=symlink_root,
        allowed_roots=[tmp_path],
    )
    after_root = db.get_workspace_primary_root("ws-1")

    assert missing_resolution.ok is False
    assert missing_resolution.failure_code == "workspace_project_root_missing"
    assert symlink_resolution.ok is False
    assert symlink_resolution.failure_code == "workspace_project_root_symlink"
    assert after_root["file_inventory_state"] == before_state
