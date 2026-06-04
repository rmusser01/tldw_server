from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Workspaces.file_inventory_models import INVENTORY_COUNT_KEYS


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")


def _workspace_with_root(
    db: CharactersRAGDB,
    *,
    workspace_id: str = "ws-1",
    root_id: str = "root-1",
) -> dict:
    db.upsert_workspace(workspace_id, "Workspace")
    return db.upsert_workspace_primary_root(
        workspace_id,
        {
            "root_id": root_id,
            "backend": "host_local",
            "absolute_root": "/Users/example/project",
            "root_state": "attached",
        },
    )


def _begin_attached_scan(
    db: CharactersRAGDB,
    *,
    workspace_id: str = "ws-1",
    root_id: str = "root-1",
    root_version: int,
    policy_fingerprint: str = "policy-a",
) -> dict:
    scan = db.begin_workspace_file_inventory_scan(
        workspace_id,
        root_id,
        root_version,
        policy_fingerprint,
        requested_by="user-1",
    )
    return db.attach_workspace_file_inventory_job(scan["scan_id"], {"id": 42, "uuid": "job-42"})


def test_file_inventory_schema_tables_and_indexes_exist(db: CharactersRAGDB) -> None:
    table_rows = db.execute_query("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    tables = {row["name"] for row in table_rows}
    assert {"workspace_file_inventory_scans", "workspace_file_inventory_items"} <= tables

    index_rows = db.execute_query(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name LIKE 'workspace_file_inventory_%'"
    ).fetchall()
    indexes = {row["name"] for row in index_rows}
    assert {
        "idx_ws_file_inventory_scans_root_created",
        "idx_ws_file_inventory_scans_active",
        "idx_ws_file_inventory_scans_job_id",
        "idx_ws_file_inventory_items_root_path",
    } <= indexes


def test_begin_scan_creates_queued_record_and_updates_root_state(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)

    scan = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )

    assert scan["workspace_id"] == "ws-1"
    assert scan["root_id"] == "root-1"
    assert scan["root_version"] == root["version"]
    assert scan["state"] == "queued"
    assert scan["job_id"] is None
    assert db.get_workspace_primary_root("ws-1")["file_inventory_state"] == "queued"


def test_begin_scan_reuses_active_scan_only_after_job_is_attached(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)

    orphaned = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )
    fresh = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )
    assert fresh["scan_id"] != orphaned["scan_id"]

    attached = db.attach_workspace_file_inventory_job(fresh["scan_id"], {"id": 42, "uuid": "job-42"})
    retry = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )
    assert retry["scan_id"] == attached["scan_id"]


def test_attach_job_is_idempotent_but_rejects_different_job(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )

    attached = db.attach_workspace_file_inventory_job(scan["scan_id"], {"id": 42, "uuid": "job-42"})
    retried = db.attach_workspace_file_inventory_job(scan["scan_id"], {"id": 42, "uuid": "job-42"})

    assert retried["job_id"] == attached["job_id"]
    with pytest.raises(ConflictError):
        db.attach_workspace_file_inventory_job(scan["scan_id"], {"id": 43, "uuid": "job-43"})


def test_enqueue_failure_marks_scan_and_root_failed_and_allows_retry(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )

    failed = db.mark_workspace_file_inventory_enqueue_failed(
        scan["scan_id"],
        [{"path_hint": "/Users/example/project/private.txt", "message": "Queue failed"}],
    )
    retry = db.begin_workspace_file_inventory_scan(
        "ws-1",
        "root-1",
        root["version"],
        "policy-a",
        requested_by="user-1",
    )

    assert failed["state"] == "failed"
    assert "path_hint" not in json.loads(failed["diagnostics_json"])[0]
    assert db.get_workspace_primary_root("ws-1")["file_inventory_state"] == "queued"
    assert retry["scan_id"] != scan["scan_id"]


def test_scan_completion_status_and_stale_projection(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan = _begin_attached_scan(db, root_version=root["version"])

    scanning = db.mark_workspace_file_inventory_scanning(scan["scan_id"])
    assert scanning["state"] == "scanning"
    assert db.get_workspace_primary_root("ws-1")["file_inventory_state"] == "scanning"

    completed = db.complete_workspace_file_inventory_scan(
        scan["scan_id"],
        "current",
        {"files": "2", "directories": 1, "diagnostics": -1},
        [],
        root_snapshot_token="snapshot-1",
    )
    status = db.get_workspace_file_inventory_status("ws-1")

    assert completed["state"] == "current"
    assert status["state"] == "current"
    assert status["stale"] is False
    assert set(status["counts"]) == set(INVENTORY_COUNT_KEYS)
    assert status["counts"]["files"] == 2
    assert status["counts"]["diagnostics"] == 0
    current_root = db.get_workspace_primary_root("ws-1")
    db.update_workspace_project_root_state(
        "ws-1",
        "root-1",
        {"git_state": "clean"},
        expected_version=current_root["version"],
    )

    stale_status = db.get_workspace_file_inventory_status("ws-1")

    assert stale_status["state"] == "stale"
    assert stale_status["durable_state"] == "current"
    assert stale_status["stale"] is True
    assert stale_status["scan_id"] == scan["scan_id"]


def test_status_for_unscanned_root_is_not_started(db: CharactersRAGDB) -> None:
    _workspace_with_root(db)

    status = db.get_workspace_file_inventory_status("ws-1")

    assert status["state"] == "not_started"
    assert status["scan_id"] is None
    assert status["counts"] == {key: 0 for key in INVENTORY_COUNT_KEYS}


def test_root_version_mismatch_failure_keeps_previous_completed_scan_stale(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan_1 = _begin_attached_scan(db, root_version=root["version"])
    db.complete_workspace_file_inventory_scan(
        scan_1["scan_id"],
        "current",
        {"files": 1},
        [],
        root_snapshot_token="snapshot-1",
    )
    current_root = db.get_workspace_primary_root("ws-1")
    db.update_workspace_project_root_state(
        "ws-1",
        "root-1",
        {"git_state": "clean"},
        expected_version=current_root["version"],
    )
    changed_root = db.get_workspace_primary_root("ws-1")
    scan_2 = _begin_attached_scan(
        db,
        root_version=changed_root["version"],
        policy_fingerprint="policy-b",
    )
    db.complete_workspace_file_inventory_scan(
        scan_2["scan_id"],
        "failed",
        {},
        [{"code": "root_version_mismatch", "message": "Root changed before scan started."}],
        root_snapshot_token=None,
    )

    status = db.get_workspace_file_inventory_status("ws-1")

    assert status["state"] == "stale"
    assert status["durable_state"] == "current"
    assert status["scan_id"] == scan_1["scan_id"]
    assert status["stale"] is True


def test_replace_items_tracks_full_and_partial_scan_coverage(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan_1 = _begin_attached_scan(db, root_version=root["version"])
    db.complete_workspace_file_inventory_scan(scan_1["scan_id"], "current", {}, [], root_snapshot_token="snapshot-1")
    db.replace_workspace_file_inventory_items(
        "ws-1",
        "root-1",
        scan_1["scan_id"],
        [
            {"relative_path": "src/a.py", "entry_kind": "file", "size_bytes": 10, "indexing_candidate": True},
            {"relative_path": "src/b.py", "entry_kind": "file", "ignored": True, "ignore_reason": "generated"},
        ],
        scan_coverage_complete=True,
    )

    scan_2 = _begin_attached_scan(db, root_version=root["version"], policy_fingerprint="policy-b")
    db.complete_workspace_file_inventory_scan(scan_2["scan_id"], "partial", {}, [], root_snapshot_token="snapshot-2")
    db.replace_workspace_file_inventory_items(
        "ws-1",
        "root-1",
        scan_2["scan_id"],
        [{"relative_path": "src/c.py", "entry_kind": "file", "size_bytes": 5}],
        scan_coverage_complete=False,
    )

    partial_rows = db.execute_query(
        "SELECT relative_path, coverage_state, deleted FROM workspace_file_inventory_items ORDER BY relative_path"
    ).fetchall()
    assert [(row["relative_path"], row["coverage_state"], row["deleted"]) for row in partial_rows] == [
        ("src/a.py", "previous", 0),
        ("src/b.py", "previous", 0),
        ("src/c.py", "current", 0),
    ]

    scan_3 = _begin_attached_scan(db, root_version=root["version"], policy_fingerprint="policy-c")
    db.complete_workspace_file_inventory_scan(scan_3["scan_id"], "current", {}, [], root_snapshot_token="snapshot-3")
    db.replace_workspace_file_inventory_items(
        "ws-1",
        "root-1",
        scan_3["scan_id"],
        [{"relative_path": "src/c.py", "entry_kind": "file", "size_bytes": 8}],
        scan_coverage_complete=True,
    )

    full_rows = db.execute_query(
        "SELECT relative_path, coverage_state, deleted FROM workspace_file_inventory_items ORDER BY relative_path"
    ).fetchall()
    assert [(row["relative_path"], row["coverage_state"], row["deleted"]) for row in full_rows] == [
        ("src/a.py", "previous", 1),
        ("src/b.py", "previous", 1),
        ("src/c.py", "current", 0),
    ]


def test_item_listing_filters_ignored_paths_and_paginates_by_relative_path(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan = _begin_attached_scan(db, root_version=root["version"])
    db.complete_workspace_file_inventory_scan(scan["scan_id"], "current", {}, [], root_snapshot_token="snapshot-1")
    db.replace_workspace_file_inventory_items(
        "ws-1",
        "root-1",
        scan["scan_id"],
        [
            {"relative_path": "src/b.py", "entry_kind": "file", "ignored": True, "ignore_reason": "generated"},
            {"relative_path": "src/a.py", "entry_kind": "file", "size_bytes": 1},
            {"relative_path": "src/c.py", "entry_kind": "file", "size_bytes": 3},
            {"relative_path": "README.md", "entry_kind": "file", "size_bytes": 2},
        ],
        scan_coverage_complete=True,
    )

    page = db.list_workspace_file_inventory_items("ws-1", prefix="src/", limit=1)
    next_page = db.list_workspace_file_inventory_items("ws-1", prefix="src/", cursor=page["next_cursor"], limit=5)
    all_items = db.list_workspace_file_inventory_items("ws-1", include_ignored=True, limit=10)

    assert [item["relative_path"] for item in page["items"]] == ["src/a.py"]
    assert [item["relative_path"] for item in next_page["items"]] == ["src/c.py"]
    assert next_page["next_cursor"] is None
    assert [item["relative_path"] for item in all_items["items"]] == [
        "README.md",
        "src/a.py",
        "src/b.py",
        "src/c.py",
    ]
    assert all(not str(item.get("relative_path", "")).startswith("/") for item in all_items["items"])


def test_item_listing_rejects_invalid_cursor_as_input_error(db: CharactersRAGDB) -> None:
    _workspace_with_root(db)

    with pytest.raises(InputError):
        db.list_workspace_file_inventory_items("ws-1", cursor="not-base64")


def test_root_replacement_and_workspace_hard_delete_clean_inventory_rows(db: CharactersRAGDB) -> None:
    root = _workspace_with_root(db)
    scan = _begin_attached_scan(db, root_version=root["version"])
    db.complete_workspace_file_inventory_scan(scan["scan_id"], "current", {}, [], root_snapshot_token="snapshot-1")
    db.replace_workspace_file_inventory_items(
        "ws-1",
        "root-1",
        scan["scan_id"],
        [{"relative_path": "src/a.py", "entry_kind": "file"}],
        scan_coverage_complete=True,
    )

    db.upsert_workspace_primary_root("ws-1", {"root_id": "root-2", "backend": "host_local"})

    assert db.execute_query("SELECT COUNT(*) AS c FROM workspace_file_inventory_scans").fetchone()["c"] == 0
    assert db.execute_query("SELECT COUNT(*) AS c FROM workspace_file_inventory_items").fetchone()["c"] == 0

    root = db.get_workspace_primary_root("ws-1")
    scan = _begin_attached_scan(db, root_id="root-2", root_version=root["version"])
    db.complete_workspace_file_inventory_scan(scan["scan_id"], "current", {}, [], root_snapshot_token="snapshot-2")
    db.replace_workspace_file_inventory_items(
        "ws-1",
        "root-2",
        scan["scan_id"],
        [{"relative_path": "src/b.py", "entry_kind": "file"}],
        scan_coverage_complete=True,
    )

    db.hard_delete_workspace("ws-1")

    assert db.execute_query("SELECT COUNT(*) AS c FROM workspace_file_inventory_scans").fetchone()["c"] == 0
    assert db.execute_query("SELECT COUNT(*) AS c FROM workspace_file_inventory_items").fetchone()["c"] == 0
