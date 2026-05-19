"""Tests for the native CodeGraph SQLite repository."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.models import CodeGraphEdge, CodeGraphNode, CodeGraphUnresolvedRef
from tldw_Server_API.app.core.DB_Management.codegraph import repository as repository_module
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


def test_repository_marks_references_resolved_without_counting_them_unresolved(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_cross_file_resolution_graph(repo)

    references = repo.list_references_for_resolution()
    repo.mark_reference_resolved(
        references[0].id,
        edge=CodeGraphEdge(
            id="edge_cross_file_call",
            source="node_entry",
            target="node_helper",
            kind="calls",
            file_path="pkg/app.py",
            line=4,
            column=12,
            provenance="codegraph_resolver",
        ),
        resolution_kind="python_import",
    )

    all_references = repo.list_references_for_resolution(include_resolved=True)

    assert repo.counts()["unresolved_refs"] == 0
    assert [relationship["target"]["id"] for relationship in repo.list_callees("node_entry", limit=10)] == [
        "node_helper"
    ]
    assert len(all_references) == 1
    assert all_references[0].resolved_target == "node_helper"
    assert all_references[0].resolved_edge == "edge_cross_file_call"


def test_repository_clears_stale_reference_resolution_when_target_file_is_deleted(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_cross_file_resolution_graph(repo)
    reference = repo.list_references_for_resolution()[0]
    repo.mark_reference_resolved(
        reference.id,
        edge=CodeGraphEdge(
            id="edge_cross_file_call",
            source="node_entry",
            target="node_helper",
            kind="calls",
            file_path="pkg/app.py",
            line=4,
            column=12,
            provenance="codegraph_resolver",
        ),
        resolution_kind="python_import",
    )

    repo.delete_file("pkg/util.py")

    references = repo.list_references_for_resolution(include_resolved=True)

    assert repo.counts()["edges"] == 0
    assert repo.counts()["unresolved_refs"] == 1
    assert len(references) == 1
    assert references[0].resolved_target is None
    assert references[0].resolved_edge is None


def test_repository_counts_and_clears_resolution_with_null_edge(tmp_path: Path) -> None:
    db_path = tmp_path / "codegraph.db"
    repo = CodeGraphRepository(db_path)
    repo.initialize()
    _seed_cross_file_resolution_graph(repo)
    repo.upsert_edge(
        CodeGraphEdge(
            id="edge_unrelated",
            source="node_entry",
            target="node_helper",
            kind="calls",
            file_path="pkg/app.py",
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE unresolved_refs
            SET resolved_target = ?, resolved_edge = NULL, resolution_kind = ?
            WHERE reference_name = ?
            """,
            ("node_helper", "test_fixture", "helper"),
        )
        conn.commit()

    cleared = repo.clear_stale_reference_resolutions()
    references = repo.list_references_for_resolution(include_resolved=True)

    assert cleared == 1
    assert repo.counts()["unresolved_refs"] == 1
    assert references[0].resolved_target is None
    assert references[0].resolved_edge is None


def test_repository_list_references_for_resolution_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_cross_file_resolution_graph(repo)

    def _fail_stale_cleanup(_conn: sqlite3.Connection) -> int:
        raise AssertionError("list_references_for_resolution must not mutate stale state")

    monkeypatch.setattr(repository_module, "_clear_stale_reference_resolutions", _fail_stale_cleanup)

    references = repo.list_references_for_resolution()

    assert [reference.reference_name for reference in references] == ["helper"]


def test_repository_batches_resolved_references_and_edges(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_cross_file_resolution_graph(repo)
    reference = repo.list_references_for_resolution()[0]

    repo.mark_references_resolved(
        (
            (
                reference.id,
                CodeGraphEdge(
                    id="edge_cross_file_call",
                    source="node_entry",
                    target="node_helper",
                    kind="calls",
                    file_path="pkg/app.py",
                    line=4,
                    column=12,
                    provenance="codegraph_resolver",
                ),
                "python_import",
            ),
        )
    )

    assert repo.counts()["unresolved_refs"] == 0
    assert [relationship["target"]["id"] for relationship in repo.list_callees("node_entry", limit=10)] == [
        "node_helper"
    ]


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
    """Return a deterministic bounded impact neighborhood around a node."""
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
    """Report truncation when relationship traversal reaches the result cap."""
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_impact_graph(repo)

    impact = repo.traverse_impact("node_helper", depth=2, direction="both", limit=1)

    assert [relationship["id"] for relationship in impact.relationships] == ["edge_entry_helper"]
    assert impact.truncated is True


def test_repository_batch_impact_traversal_uses_one_neighborhood(tmp_path: Path) -> None:
    """Traverse one shared impact neighborhood for multiple starting nodes."""
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_impact_graph(repo)

    impact = repo.traverse_impact_many(("node_entry", "node_helper"), depth=1, direction="both", limit=10)

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


def test_repository_impact_traversal_passes_remaining_row_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound each relationship query to the remaining result budget plus one row."""
    assert "max_rows" in inspect.signature(repository_module._select_relationships_for_nodes).parameters
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_impact_graph(repo)
    requested_row_limits: list[int | None] = []
    original_select = repository_module._select_relationships_for_nodes

    def _spy_select_relationships_for_nodes(
        conn: sqlite3.Connection,
        node_ids: set[str],
        direction: str,
        *,
        anchor_file_path: str,
        max_rows: int | None = None,
    ) -> list[sqlite3.Row]:
        requested_row_limits.append(max_rows)
        return original_select(conn, node_ids, direction, anchor_file_path=anchor_file_path, max_rows=max_rows)

    monkeypatch.setattr(repository_module, "_select_relationships_for_nodes", _spy_select_relationships_for_nodes)

    impact = repo.traverse_impact("node_helper", depth=2, direction="both", limit=1)

    assert requested_row_limits == [2]
    assert [relationship["id"] for relationship in impact.relationships] == ["edge_entry_helper"]
    assert impact.truncated is True


def _seed_impact_graph(repo: CodeGraphRepository) -> None:
    """Seed a compact graph for impact traversal tests."""
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


def _seed_cross_file_resolution_graph(repo: CodeGraphRepository) -> None:
    """Seed source and target files plus one cross-file call reference."""
    repo.upsert_file(
        path="pkg/app.py",
        language="python",
        size=64,
        content_hash="app",
        modified_at=1.0,
        status="indexed",
        errors=[],
    )
    repo.upsert_file(
        path="pkg/util.py",
        language="python",
        size=64,
        content_hash="util",
        modified_at=1.0,
        status="indexed",
        errors=[],
    )
    repo.seed_graph_rows_for_test(
        nodes=[
            {
                "id": "node_entry",
                "identity_key": "entry",
                "kind": "function",
                "name": "entry",
                "qualified_name": "entry",
                "file_path": "pkg/app.py",
            },
            {
                "id": "node_helper",
                "identity_key": "helper",
                "kind": "function",
                "name": "helper",
                "qualified_name": "helper",
                "file_path": "pkg/util.py",
            },
        ],
        edges=[],
        unresolved_refs=[
            {
                "from_node_id": "node_entry",
                "reference_name": "helper",
                "reference_kind": "call",
                "file_path": "pkg/app.py",
                "line": 4,
                "column": 12,
                "language": "python",
            }
        ],
    )
