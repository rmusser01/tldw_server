from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.models import CodeGraphEdge, CodeGraphNode, CodeGraphUnresolvedRef
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


def test_repository_persists_searches_and_fetches_graph_relationships(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    repo.upsert_file(
        path="pkg/sample.py",
        language="python",
        size=128,
        content_hash="hash",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=3,
    )
    module = CodeGraphNode(
        id="node_module",
        identity_key="module",
        kind="module",
        name="sample",
        qualified_name="pkg.sample",
        file_path="pkg/sample.py",
        language="python",
        start_line=1,
        end_line=12,
    )
    caller = CodeGraphNode(
        id="node_caller",
        identity_key="caller",
        kind="method",
        name="greet",
        qualified_name="Greeter.greet",
        file_path="pkg/sample.py",
        language="python",
        start_line=6,
        end_line=7,
    )
    callee = CodeGraphNode(
        id="node_callee",
        identity_key="callee",
        kind="function",
        name="helper",
        qualified_name="helper",
        file_path="pkg/sample.py",
        language="python",
        start_line=10,
        end_line=12,
    )
    edge = CodeGraphEdge(
        id="edge_call",
        source="node_caller",
        target="node_callee",
        kind="calls",
        file_path="pkg/sample.py",
        line=7,
        column=15,
    )
    unresolved = CodeGraphUnresolvedRef(
        from_node_id="node_callee",
        reference_name="external_call",
        reference_kind="call",
        file_path="pkg/sample.py",
        line=11,
        column=4,
        language="python",
    )

    repo.replace_file_graph(
        path="pkg/sample.py",
        nodes=[module, caller, callee],
        edges=[edge],
        unresolved_refs=[unresolved],
    )

    search_results = repo.search_nodes("helper", limit=10)
    fetched = repo.get_node("node_callee")
    callers = repo.list_callers("node_callee", limit=10)
    callees = repo.list_callees("node_caller", limit=10)

    assert [node.id for node in search_results] == ["node_callee"]
    assert fetched is not None
    assert fetched.qualified_name == "helper"
    assert [relationship["source"]["id"] for relationship in callers] == ["node_caller"]
    assert [relationship["target"]["id"] for relationship in callees] == ["node_callee"]
    assert callees[0]["kind"] == "calls"


def test_repository_replacement_removes_stale_symbols_from_search(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    repo.upsert_file(
        path="pkg/sample.py",
        language="python",
        size=128,
        content_hash="hash",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=1,
    )
    repo.replace_file_graph(
        path="pkg/sample.py",
        nodes=[
            CodeGraphNode(
                id="node_old",
                identity_key="old",
                kind="function",
                name="old_helper",
                qualified_name="old_helper",
                file_path="pkg/sample.py",
                language="python",
            )
        ],
        edges=[],
        unresolved_refs=[],
    )

    repo.replace_file_graph(
        path="pkg/sample.py",
        nodes=[
            CodeGraphNode(
                id="node_new",
                identity_key="new",
                kind="function",
                name="new_helper",
                qualified_name="new_helper",
                file_path="pkg/sample.py",
                language="python",
            )
        ],
        edges=[],
        unresolved_refs=[],
    )

    assert repo.search_nodes("old_helper", limit=10) == []
    assert [node.id for node in repo.search_nodes("new_helper", limit=10)] == ["node_new"]


def test_repository_atomic_file_and_graph_replacement_rolls_back_on_graph_error(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    repo.upsert_file(
        path="pkg/sample.py",
        language="python",
        size=128,
        content_hash="old_hash",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=1,
    )
    repo.replace_file_graph(
        path="pkg/sample.py",
        nodes=[
            CodeGraphNode(
                id="node_old",
                identity_key="old",
                kind="function",
                name="old_helper",
                qualified_name="old_helper",
                file_path="pkg/sample.py",
                language="python",
            )
        ],
        edges=[],
        unresolved_refs=[],
    )
    repo.seed_graph_rows_for_test(
        nodes=[
            {
                "id": "node_conflict",
                "identity_key": "conflict",
                "kind": "function",
                "name": "other_helper",
                "file_path": "pkg/other.py",
            }
        ],
        edges=[],
        unresolved_refs=[],
    )

    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_file_and_replace_graph(
            path="pkg/sample.py",
            language="python",
            size=256,
            content_hash="new_hash",
            modified_at=2.0,
            status="indexed",
            errors=[],
            node_count=1,
            nodes=[
                CodeGraphNode(
                    id="node_conflict",
                    identity_key="new",
                    kind="function",
                    name="new_helper",
                    qualified_name="new_helper",
                    file_path="pkg/sample.py",
                    language="python",
                )
            ],
            edges=[],
            unresolved_refs=[],
        )

    file_row = repo.list_files(limit=10)[0]

    assert file_row.content_hash == "old_hash"
    assert repo.get_node("node_old") is not None
    assert repo.search_nodes("new_helper", limit=10) == []


def test_repository_traverses_bounded_impact_graph(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_impact_graph(repo)

    impact = repo.traverse_impact("node_helper", depth=1, direction="both", limit=10)

    assert [node.id for node in impact.nodes] == [
        "node_entry",
        "node_helper",
        "node_leaf",
        "node_other",
    ]
    assert [relationship["id"] for relationship in impact.relationships] == [
        "edge_entry_helper",
        "edge_helper_leaf",
        "edge_other_helper",
    ]
    assert impact.truncated is False


def test_repository_impact_traversal_reports_truncation(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_impact_graph(repo)

    impact = repo.traverse_impact("node_helper", depth=2, direction="both", limit=1)

    assert [relationship["id"] for relationship in impact.relationships] == ["edge_entry_helper"]
    assert impact.truncated is True


def _seed_impact_graph(repo: CodeGraphRepository) -> None:
    repo.seed_graph_rows_for_test(
        nodes=[
            {
                "id": "node_entry",
                "identity_key": "entry",
                "kind": "function",
                "name": "entry",
                "file_path": "pkg/sample.py",
            },
            {
                "id": "node_helper",
                "identity_key": "helper",
                "kind": "function",
                "name": "helper",
                "file_path": "pkg/sample.py",
            },
            {
                "id": "node_leaf",
                "identity_key": "leaf",
                "kind": "function",
                "name": "leaf",
                "file_path": "pkg/sample.py",
            },
            {
                "id": "node_other",
                "identity_key": "other",
                "kind": "function",
                "name": "other",
                "file_path": "pkg/other.py",
            },
        ],
        edges=[
            {
                "id": "edge_entry_helper",
                "source": "node_entry",
                "target": "node_helper",
                "kind": "calls",
                "file_path": "pkg/sample.py",
                "line": 2,
            },
            {
                "id": "edge_helper_leaf",
                "source": "node_helper",
                "target": "node_leaf",
                "kind": "calls",
                "file_path": "pkg/sample.py",
                "line": 6,
            },
            {
                "id": "edge_other_helper",
                "source": "node_other",
                "target": "node_helper",
                "kind": "calls",
                "file_path": "pkg/other.py",
                "line": 3,
            },
        ],
        unresolved_refs=[],
    )
