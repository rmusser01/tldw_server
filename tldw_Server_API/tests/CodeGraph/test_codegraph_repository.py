from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository, _create_optional_fts


def test_repository_initializes_schema_and_counts_empty_graph(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()

    counts = repo.counts()

    assert counts["files"] == 0
    assert counts["nodes"] == 0
    assert counts["edges"] == 0
    assert counts["unresolved_refs"] == 0


def test_repository_upserts_files_and_records_runs(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    run_id = repo.record_index_run_start(workspace_key="ws_test", mode="foreground_index")

    repo.upsert_file(
        path="app/main.py",
        language="python",
        size=12,
        content_hash="abc",
        modified_at=1.5,
        status="indexed",
        errors=[],
    )
    repo.finish_index_run(run_id, status="complete", counters={"files_indexed": 1}, error_summary=[])

    assert repo.counts()["files"] == 1
    assert repo.list_files(limit=10)[0].path == "app/main.py"
    last_run = repo.last_index_run()

    assert last_run is not None
    assert last_run.status == "complete"


def test_repository_list_files_treats_path_prefix_as_literal(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    for path in ("src/pkg_1.py", "src/pkgA.py", "src/pkg%literal.py", "src/pkg-other.py"):
        repo.upsert_file(
            path=path,
            language="python",
            size=12,
            content_hash=path,
            modified_at=1.5,
            status="indexed",
            errors=[],
        )

    underscore_matches = repo.list_files(limit=10, path_prefix="src/pkg_")
    percent_matches = repo.list_files(limit=10, path_prefix="src/pkg%")

    assert [item.path for item in underscore_matches] == ["src/pkg_1.py"]
    assert [item.path for item in percent_matches] == ["src/pkg%literal.py"]


def test_create_optional_fts_ignores_missing_fts5() -> None:
    class _FakeConnection:
        def execute(self, _sql: str) -> None:
            raise sqlite3.OperationalError("no such module: fts5")

    _create_optional_fts(_FakeConnection())


def test_repository_replacing_file_removes_owned_graph_rows_and_dangling_edges(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    repo.upsert_file(
        path="app/main.py",
        language="python",
        size=12,
        content_hash="old",
        modified_at=1.0,
        status="indexed",
        errors=[],
    )
    repo.seed_graph_rows_for_test(
        nodes=[
            {
                "id": "node_old",
                "identity_key": "old",
                "kind": "function",
                "name": "old",
                "file_path": "app/main.py",
            },
            {
                "id": "node_other",
                "identity_key": "other",
                "kind": "function",
                "name": "other",
                "file_path": "app/other.py",
            },
        ],
        edges=[
            {
                "id": "edge_owned",
                "source": "node_old",
                "target": "node_other",
                "kind": "calls",
                "file_path": "app/main.py",
            },
            {
                "id": "edge_dangling",
                "source": "node_other",
                "target": "node_old",
                "kind": "calls",
                "file_path": "app/other.py",
            },
        ],
        unresolved_refs=[
            {
                "from_node_id": "node_old",
                "reference_name": "missing",
                "reference_kind": "call",
                "file_path": "app/main.py",
            },
        ],
    )

    repo.prepare_file_replacement("app/main.py")

    assert repo.counts()["nodes"] == 1
    assert repo.counts()["edges"] == 0
    assert repo.counts()["unresolved_refs"] == 0
