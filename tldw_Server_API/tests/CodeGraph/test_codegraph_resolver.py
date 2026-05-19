"""Tests for CodeGraph cross-file reference resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode, CodeGraphUnresolvedRef
from tldw_Server_API.app.core.CodeGraph.resolver import CodeGraphReferenceResolver
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository


def test_resolver_links_python_from_import_call_to_target_symbol(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_python_import_workspace(repo, import_local_name="helper", reference_name="helper")

    result = CodeGraphReferenceResolver(repo).resolve()

    callers = repo.list_callers("node_util_helper", limit=10)
    callees = repo.list_callees("node_app_entry", limit=10)

    assert result.resolved_calls == 1
    assert result.resolved_imports == 1
    assert repo.counts()["unresolved_refs"] == 0
    assert [relationship["source"]["id"] for relationship in callers] == ["node_app_entry"]
    assert [relationship["target"]["id"] for relationship in callees] == ["node_util_helper"]


def test_resolver_links_python_aliased_import_call(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_python_import_workspace(repo, import_local_name="h", reference_name="h", import_alias="h")

    result = CodeGraphReferenceResolver(repo).resolve()

    assert result.resolved_calls == 1
    assert [relationship["target"]["id"] for relationship in repo.list_callees("node_app_entry", limit=10)] == [
        "node_util_helper"
    ]


def test_resolver_links_js_ts_named_import_with_resolved_path(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_js_ts_import_workspace(repo)

    result = CodeGraphReferenceResolver(repo).resolve()

    callers = repo.list_callers("node_ts_helper", limit=10)

    assert result.resolved_calls == 1
    assert result.resolved_imports == 1
    assert [relationship["source"]["id"] for relationship in callers] == ["node_ts_main"]


def test_resolver_keeps_unmatched_external_references_unresolved(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_js_ts_import_workspace(repo, resolved_path=None)

    result = CodeGraphReferenceResolver(repo).resolve()

    assert result.resolved_calls == 0
    assert result.resolved_imports == 0
    assert repo.counts()["unresolved_refs"] == 1


def test_resolver_matches_case_sensitive_python_symbols_exactly(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    repo.upsert_file_and_replace_graph(
        path="pkg/util.py",
        language="python",
        size=64,
        content_hash="util",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=3,
        nodes=[
            _node("node_util_module", "module", "util", "pkg.util", "pkg/util.py", "python"),
            _node("node_upper_helper", "function", "Helper", "Helper", "pkg/util.py", "python"),
            _node("node_lower_helper", "function", "helper", "helper", "pkg/util.py", "python"),
        ],
        edges=[],
        unresolved_refs=[],
    )
    _seed_python_import_workspace(repo, import_local_name="helper", reference_name="helper", include_target=False)

    result = CodeGraphReferenceResolver(repo).resolve()

    assert result.resolved_calls == 1
    assert [relationship["target"]["id"] for relationship in repo.list_callees("node_app_entry", limit=10)] == [
        "node_lower_helper"
    ]


def test_resolver_batches_persistence_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_python_import_workspace(repo, import_local_name="helper", reference_name="helper")

    def _fail_single_write(*_args, **_kwargs) -> None:
        raise AssertionError("resolver should use repository batch write helpers")

    monkeypatch.setattr(repo, "mark_reference_resolved", _fail_single_write)
    monkeypatch.setattr(repo, "upsert_edge", _fail_single_write)

    result = CodeGraphReferenceResolver(repo).resolve()

    assert result.resolved_calls == 1
    assert result.resolved_imports == 1


def test_resolver_respects_reference_limit(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_python_import_workspace(repo, import_local_name="helper", reference_name="helper")
    repo.seed_graph_rows_for_test(
        nodes=[],
        edges=[],
        unresolved_refs=[
            {
                "from_node_id": "node_app_entry",
                "reference_name": "helper",
                "reference_kind": "call",
                "file_path": "pkg/app.py",
                "line": 5,
                "column": 12,
                "language": "python",
            }
        ],
    )

    result = CodeGraphReferenceResolver(repo).resolve(max_refs=1)

    assert result.resolved_calls == 1
    assert result.truncated is True
    assert repo.counts()["unresolved_refs"] == 1


def test_resolver_respects_expired_deadline(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    _seed_python_import_workspace(repo, import_local_name="helper", reference_name="helper")

    result = CodeGraphReferenceResolver(repo).resolve(deadline_monotonic=1.0, monotonic=lambda: 2.0)

    assert result.resolved_calls == 0
    assert result.resolved_imports == 0
    assert result.truncated is True
    assert repo.counts()["unresolved_refs"] == 1


def _seed_python_import_workspace(
    repo: CodeGraphRepository,
    *,
    import_local_name: str,
    reference_name: str,
    import_alias: str | None = None,
    include_target: bool = True,
) -> None:
    if include_target:
        repo.upsert_file_and_replace_graph(
            path="pkg/util.py",
            language="python",
            size=64,
            content_hash="util",
            modified_at=1.0,
            status="indexed",
            errors=[],
            node_count=2,
            nodes=[
                _node("node_util_module", "module", "util", "pkg.util", "pkg/util.py", "python"),
                _node("node_util_helper", "function", "helper", "helper", "pkg/util.py", "python"),
            ],
            edges=[],
            unresolved_refs=[],
        )
    repo.upsert_file_and_replace_graph(
        path="pkg/app.py",
        language="python",
        size=64,
        content_hash="app",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=3,
        nodes=[
            _node("node_app_module", "module", "app", "pkg.app", "pkg/app.py", "python"),
            _node(
                "node_app_import",
                "import",
                import_local_name,
                "pkg.util.helper",
                "pkg/app.py",
                "python",
                metadata={"imported": "pkg.util.helper", "alias": import_alias},
            ),
            _node("node_app_entry", "function", "entry", "entry", "pkg/app.py", "python"),
        ],
        edges=[],
        unresolved_refs=[
            CodeGraphUnresolvedRef(
                from_node_id="node_app_entry",
                reference_name=reference_name,
                reference_kind="call",
                file_path="pkg/app.py",
                line=4,
                column=12,
                language="python",
            )
        ],
    )


def _seed_js_ts_import_workspace(repo: CodeGraphRepository, *, resolved_path: str | None = "src/util.ts") -> None:
    repo.upsert_file_and_replace_graph(
        path="src/util.ts",
        language="typescript",
        size=64,
        content_hash="util",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=2,
        nodes=[
            _node("node_ts_module", "module", "util", "src.util", "src/util.ts", "typescript"),
            _node(
                "node_ts_helper",
                "function",
                "helper",
                "helper",
                "src/util.ts",
                "typescript",
                flags=("exported",),
            ),
        ],
        edges=[],
        unresolved_refs=[],
    )
    repo.upsert_file_and_replace_graph(
        path="src/app.ts",
        language="typescript",
        size=64,
        content_hash="app",
        modified_at=1.0,
        status="indexed",
        errors=[],
        node_count=3,
        nodes=[
            _node("node_ts_app_module", "module", "app", "src.app", "src/app.ts", "typescript"),
            _node(
                "node_ts_import",
                "import",
                "helper",
                "@/util:helper:helper",
                "src/app.ts",
                "typescript",
                metadata={
                    "source": "@/util",
                    "imported": "helper",
                    "alias": None,
                    "resolved_path": resolved_path,
                    "resolution_kind": "alias" if resolved_path else "external",
                },
            ),
            _node("node_ts_main", "function", "main", "main", "src/app.ts", "typescript"),
        ],
        edges=[],
        unresolved_refs=[
            CodeGraphUnresolvedRef(
                from_node_id="node_ts_main",
                reference_name="helper",
                reference_kind="call",
                file_path="src/app.ts",
                line=3,
                column=10,
                language="typescript",
            )
        ],
    )


def _node(
    node_id: str,
    kind: str,
    name: str,
    qualified_name: str,
    file_path: str,
    language: str,
    *,
    flags: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> CodeGraphNode:
    return CodeGraphNode(
        id=node_id,
        identity_key=node_id,
        kind=kind,
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        language=language,
        flags=flags,
        metadata=dict(metadata or {}),
    )
