from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.indexer import CodeGraphIndexer, _Candidate, _DiscoveryResult
from tldw_Server_API.app.core.CodeGraph.language_registry import CodeGraphLanguageRegistry
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult, LanguageInfo
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository


def test_indexer_indexes_supported_file_inventory_and_skips_excluded_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('hi')\n", encoding="utf-8")
    node_modules = workspace / "node_modules"
    node_modules.mkdir()
    (node_modules / "ignored.ts").write_text("export const ignored = true\n", encoding="utf-8")

    repo = CodeGraphRepository(tmp_path / "index" / "codegraph.db")
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"index_base_dir": str(tmp_path / "index")}),
        registry=CodeGraphLanguageRegistry(),
    )

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert result.counters["files_indexed"] == 1
    assert repo.list_files(limit=10)[0].path == "app.py"


def test_indexer_extracts_python_graph_rows_during_index(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text(
        """
class Greeter:
    def greet(self, name):
        return helper(name)


def helper(value):
    return value.upper()
""",
        encoding="utf-8",
    )
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    helper = repo.find_node_by_symbol("helper")

    assert result.status == "complete"
    assert repo.counts()["nodes"] >= 4
    assert repo.counts()["edges"] == 1
    assert repo.list_files(limit=10)[0].node_count >= 4
    assert helper is not None
    assert [relationship["source"]["qualified_name"] for relationship in repo.list_callers(helper.id, limit=10)] == [
        "Greeter.greet"
    ]


def test_indexer_keeps_javascript_typescript_as_inventory_only_until_extractor_slice(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (workspace / "ui.ts").write_text("export function helper() { return 1; }\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    files = {item.path: item for item in repo.list_files(limit=10)}

    assert result.status == "complete"
    assert files["app.py"].node_count > 0
    assert files["ui.ts"].node_count == 0
    assert [node.file_path for node in repo.search_nodes("helper", limit=10)] == ["app.py"]


def test_indexer_marks_python_extraction_errors_without_claiming_indexed_status(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )
    file_row = repo.list_files(limit=10)[0]

    assert result.status == "complete"
    assert result.counters["errors"] > 0
    assert any(error.startswith("broken.py:") for error in result.errors)
    assert file_row.status == "extraction_failed"
    assert file_row.node_count == 0
    assert file_row.errors


def test_indexer_converts_extractor_exceptions_to_file_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "broken.py").write_text("x = 1\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    class _RaisingExtractor:
        def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
            raise ValueError("source code string cannot contain null bytes")

    indexer._extractors["python"] = _RaisingExtractor()  # noqa: SLF001

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )
    file_row = repo.list_files(limit=10)[0]

    assert result.status == "complete"
    assert result.counters["errors"] == 1
    assert file_row.status == "extraction_failed"
    assert file_row.errors == ("source code string cannot contain null bytes",)


def test_indexer_does_not_read_full_inventory_only_file_into_memory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ui.ts").write_text("export const value = 1;\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    class _NoReadBytesPath:
        def __init__(self, path: Path) -> None:
            self._path = path

        def open(self, *args: Any, **kwargs: Any) -> Any:
            return self._path.open(*args, **kwargs)

        def read_bytes(self) -> bytes:
            raise AssertionError("inventory-only files should not be loaded fully")

    original_discover = indexer._discover_candidates  # noqa: SLF001

    def _discover_with_wrapped_paths(
        workspace_root: Path,
        *,
        languages: list[str] | tuple[str, ...] | None,
        counters: dict[str, int],
        max_files: int,
    ) -> _DiscoveryResult:
        result = original_discover(
            workspace_root,
            languages=languages,
            counters=counters,
            max_files=max_files,
        )
        result.candidates[0] = result.candidates[0].__class__(
            path=_NoReadBytesPath(result.candidates[0].path),  # type: ignore[arg-type]
            relative_path=result.candidates[0].relative_path,
            language_id=result.candidates[0].language_id,
            stage=result.candidates[0].stage,
            size=result.candidates[0].size,
            modified_at=result.candidates[0].modified_at,
        )
        return result

    indexer._discover_candidates = _discover_with_wrapped_paths  # type: ignore[method-assign]  # noqa: SLF001

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert repo.list_files(limit=10)[0].path == "ui.ts"


def test_indexer_records_unreadable_files_without_aborting_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "broken.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "healthy.py").write_text("y = 2\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    class _UnreadablePath:
        def __init__(self, path: Path) -> None:
            self._path = path

        def open(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError("permission denied")

        def read_bytes(self) -> bytes:
            raise OSError("permission denied")

    original_discover = indexer._discover_candidates  # noqa: SLF001

    def _discover_with_unreadable_path(
        workspace_root: Path,
        *,
        languages: list[str] | tuple[str, ...] | None,
        counters: dict[str, int],
        max_files: int,
    ) -> _DiscoveryResult:
        result = original_discover(
            workspace_root,
            languages=languages,
            counters=counters,
            max_files=max_files,
        )
        candidates = [
            _Candidate(
                path=_UnreadablePath(candidate.path),  # type: ignore[arg-type]
                relative_path=candidate.relative_path,
                language_id=candidate.language_id,
                stage=candidate.stage,
                size=candidate.size,
                modified_at=candidate.modified_at,
            )
            if candidate.relative_path == "broken.py"
            else candidate
            for candidate in result.candidates
        ]
        return _DiscoveryResult(candidates=candidates, status=result.status)

    indexer._discover_candidates = _discover_with_unreadable_path  # type: ignore[method-assign]  # noqa: SLF001

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )
    files = {item.path: item for item in repo.list_files(limit=10)}

    assert result.status == "complete"
    assert result.counters["errors"] == 1
    assert result.counters["files_skipped"] == 1
    assert result.counters["files_indexed"] == 1
    assert files["broken.py"].status == "extraction_failed"
    assert files["broken.py"].errors == ("permission denied",)
    assert files["healthy.py"].status == "indexed"


def test_indexer_opens_each_candidate_once_for_probe_and_content(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "ui.ts").write_text("export const y = 2;\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())
    wrapped_paths: dict[str, _OpenCountingPath] = {}

    class _OpenCountingPath:
        def __init__(self, path: Path) -> None:
            self._path = path
            self.open_calls = 0

        def open(self, *args: Any, **kwargs: Any) -> Any:
            self.open_calls += 1
            return self._path.open(*args, **kwargs)

        def read_bytes(self) -> bytes:
            raise AssertionError("indexer should reuse the open stream instead of read_bytes")

    original_discover = indexer._discover_candidates  # noqa: SLF001

    def _discover_with_counting_paths(
        workspace_root: Path,
        *,
        languages: list[str] | tuple[str, ...] | None,
        counters: dict[str, int],
        max_files: int,
    ) -> _DiscoveryResult:
        result = original_discover(
            workspace_root,
            languages=languages,
            counters=counters,
            max_files=max_files,
        )
        candidates: list[_Candidate] = []
        for candidate in result.candidates:
            wrapped = _OpenCountingPath(candidate.path)
            wrapped_paths[candidate.relative_path] = wrapped
            candidates.append(
                _Candidate(
                    path=wrapped,  # type: ignore[arg-type]
                    relative_path=candidate.relative_path,
                    language_id=candidate.language_id,
                    stage=candidate.stage,
                    size=candidate.size,
                    modified_at=candidate.modified_at,
                )
            )
        return _DiscoveryResult(candidates=candidates, status=result.status)

    indexer._discover_candidates = _discover_with_counting_paths  # type: ignore[method-assign]  # noqa: SLF001

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert wrapped_paths["app.py"].open_calls == 1
    assert wrapped_paths["ui.ts"].open_calls == 1


def test_indexer_rejects_over_limit_foreground_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(3):
        (workspace / f"file_{index}.py").write_text("x = 1\n", encoding="utf-8")

    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=2,
    )

    assert result.status == "index_too_large_for_foreground"
    assert repo.counts()["files"] == 0


def test_indexer_rejects_over_total_byte_budget_without_partial_index(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "small.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "large.py").write_text("x = '" + ("a" * 64) + "'\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"foreground_max_bytes": 16}),
        registry=CodeGraphLanguageRegistry(),
    )

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "index_too_large_for_foreground"
    assert repo.counts()["files"] == 0


def test_indexer_stops_when_foreground_time_budget_expires(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("y = 2\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"max_index_seconds": 1}),
        registry=CodeGraphLanguageRegistry(),
        monotonic=lambda: next(ticks),
    )

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "index_timed_out_for_foreground"
    assert result.counters["files_indexed"] == 1


def test_indexer_skips_planned_language_files_until_extractors_exist(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert result.counters["planned_language_skipped"] == 1
    assert repo.counts()["files"] == 0


def test_sync_removes_deleted_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("x = 1\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    indexer.index_workspace(workspace, "ws_test", repo, force=True, languages=None, max_files=10)
    source.unlink()
    result = indexer.sync_workspace(workspace, "ws_test", repo, languages=None, max_files=10)

    assert result.status == "complete"
    assert repo.counts()["files"] == 0


def test_sync_with_language_filter_preserves_out_of_scope_indexed_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "ui.ts").write_text("export const x = 1;\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    indexer.index_workspace(workspace, "ws_test", repo, force=True, languages=None, max_files=10)
    result = indexer.sync_workspace(workspace, "ws_test", repo, languages=["python"], max_files=10)

    assert result.status == "complete"
    assert sorted(item.path for item in repo.list_files(limit=10)) == ["app.py", "ui.ts"]


def test_indexer_stops_discovery_once_file_limit_is_exceeded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(20):
        (workspace / f"file_{index}.py").write_text("x = 1\n", encoding="utf-8")

    class _CountingRegistry:
        def __init__(self) -> None:
            self.calls = 0

        def language_for_path(self, _path: Path) -> LanguageInfo | None:
            self.calls += 1
            return LanguageInfo(
                language_id="python",
                display_name="Python",
                extensions=(".py",),
                stage="foundation",
            )

    registry = _CountingRegistry()
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=registry)

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=2,
    )

    assert result.status == "index_too_large_for_foreground"
    assert registry.calls == 3
