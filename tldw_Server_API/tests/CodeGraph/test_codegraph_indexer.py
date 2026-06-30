from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.CodeGraph.indexer as indexer_module
from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.dependencies import DependencyHealth
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.indexer import CodeGraphIndexer, _Candidate, _DiscoveryResult
from tldw_Server_API.app.core.CodeGraph.language_registry import CodeGraphLanguageRegistry
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult, LanguageInfo
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository


def _require_c_family_parsers() -> None:
    """Skip C/C++ indexer coverage unless both C-family parsers load."""
    if not (load_parser("c").available and load_parser("cpp").available):
        pytest.skip("tree-sitter-c/cpp parsers are not available")


def _require_jvm_parsers() -> None:
    """Skip JVM indexer coverage unless both Java and Kotlin parsers load."""
    if not (load_parser("java").available and load_parser("kotlin").available):
        pytest.skip("tree-sitter-java/kotlin parsers are not available")


def _require_typescript_parsers() -> None:
    """Skip TypeScript indexer coverage unless TS and TSX parsers load."""
    if not (load_parser("typescript").available and load_parser("tsx").available):
        pytest.skip("tree-sitter-typescript parser is not available")


def _require_csharp_parser() -> None:
    """Skip C# indexer coverage unless the C# parser loads."""
    if not load_parser("csharp").available:
        pytest.skip("tree-sitter-c-sharp parser is not available")


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


def test_indexer_resolves_python_cross_file_import_call(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    package.mkdir(parents=True)
    (package / "util.py").write_text(
        """
def helper(value):
    return value.upper()
""",
        encoding="utf-8",
    )
    (package / "app.py").write_text(
        """
from pkg.util import helper


def entry(value):
    return helper(value)
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
    helper = repo.find_nodes_by_file_and_name(file_path="pkg/util.py", name="helper")[0]

    assert result.status == "complete"
    assert [relationship["source"]["file_path"] for relationship in repo.list_callers(helper.id, limit=10)] == [
        "pkg/app.py"
    ]


def test_indexer_removes_stale_cross_file_edges_after_target_rename(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    package.mkdir(parents=True)
    util = package / "util.py"
    util.write_text("def helper(value):\n    return value\n", encoding="utf-8")
    (package / "app.py").write_text(
        "from pkg.util import helper\n\n\ndef entry(value):\n    return helper(value)\n",
        encoding="utf-8",
    )
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())
    indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )
    helper = repo.find_nodes_by_file_and_name(file_path="pkg/util.py", name="helper")[0]
    assert repo.list_callers(helper.id, limit=10)

    util.write_text("def renamed(value):\n    return value\n", encoding="utf-8")
    result = indexer.sync_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert repo.list_callers(helper.id, limit=10) == []
    assert repo.counts()["unresolved_refs"] == 1


def test_indexer_resolves_typescript_path_alias_import_call(tmp_path: Path) -> None:
    _require_typescript_parsers()
    workspace = tmp_path / "workspace"
    source_dir = workspace / "src"
    source_dir.mkdir(parents=True)
    (workspace / "tsconfig.json").write_text(
        '{"compilerOptions":{"baseUrl":".","paths":{"@/*":["src/*"]}}}',
        encoding="utf-8",
    )
    (source_dir / "util.ts").write_text("export function helper(value: string) { return value; }\n", encoding="utf-8")
    (source_dir / "app.ts").write_text(
        'import { helper } from "@/util";\nexport function main(value: string) { return helper(value); }\n',
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
    helper = repo.find_nodes_by_file_and_name(file_path="src/util.ts", name="helper")[0]

    assert result.status == "complete"
    assert [relationship["source"]["file_path"] for relationship in repo.list_callers(helper.id, limit=10)] == [
        "src/app.ts"
    ]


def test_indexer_passes_foreground_bounds_to_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class _FakeResolver:
        def __init__(self, _repository: CodeGraphRepository) -> None:
            pass

        def resolve(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                resolved_calls=0,
                resolved_imports=0,
                stale_resolutions_cleared=0,
                truncated=False,
                import_nodes_scanned=0,
                references_scanned=0,
            )

    monkeypatch.setattr(indexer_module, "CodeGraphReferenceResolver", _FakeResolver)
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"max_index_seconds": 20}),
        registry=CodeGraphLanguageRegistry(),
        monotonic=lambda: 5.0,
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
    assert captured["source_file_paths"] == {"app.py"}
    assert captured["max_import_nodes"] > 0
    assert captured["max_refs"] > 0
    assert captured["deadline_monotonic"] == 25.0
    assert captured["monotonic"]() == 5.0


def test_indexer_extracts_javascript_typescript_graph_rows_during_index(tmp_path: Path) -> None:
    _require_typescript_parsers()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (workspace / "ui.ts").write_text("export function helper() { return 1; }\n", encoding="utf-8")
    (workspace / "Card.tsx").write_text(
        "export function Card() { return <div />; }\n",
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

    files = {item.path: item for item in repo.list_files(limit=10)}

    assert result.status == "complete"
    assert files["app.py"].node_count > 0
    assert files["ui.ts"].node_count > 0
    assert files["Card.tsx"].node_count > 0
    assert sorted(node.file_path for node in repo.search_nodes("helper", limit=10)) == ["app.py", "ui.ts"]
    assert [node.file_path for node in repo.search_nodes("Card", kind="component", limit=10)] == ["Card.tsx"]


def test_indexer_extracts_java_kotlin_graph_rows_during_index(tmp_path: Path) -> None:
    _require_jvm_parsers()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Service.java").write_text(
        """
package com.example.app;

public class Service {
    public String greet(String name) {
        return helper(name);
    }

    private String helper(String value) {
        return value.trim();
    }
}
""",
        encoding="utf-8",
    )
    (workspace / "Greeter.kt").write_text(
        """
package com.example.app

class Greeter {
    fun greet(name: String): String {
        return helper(name)
    }

    private fun helper(value: String): String {
        return value.trim()
    }
}
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

    files = {item.path: item for item in repo.list_files(limit=10)}
    helper_paths = sorted(node.file_path for node in repo.search_nodes("helper", limit=10))

    assert result.status == "complete"
    assert files["Service.java"].node_count > 0
    assert files["Greeter.kt"].node_count > 0
    assert helper_paths == ["Greeter.kt", "Service.java"]


def test_indexer_extracts_csharp_graph_rows_during_index(tmp_path: Path) -> None:
    _require_csharp_parser()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Greeter.cs").write_text(
        """
using System;

namespace Example.App;

public class Greeter {
    public string Greet(string name) {
        return Helper(name);
    }

    private string Helper(string value) {
        return value.Trim();
    }
}
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

    files = {item.path: item for item in repo.list_files(limit=10)}
    helper_paths = sorted(node.file_path for node in repo.search_nodes("Helper", limit=10))

    assert result.status == "complete"
    assert files["Greeter.cs"].node_count > 0
    assert helper_paths == ["Greeter.cs"]


def test_indexer_extracts_c_cpp_graph_rows_during_index(tmp_path: Path) -> None:
    _require_c_family_parsers()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "greeter.c").write_text(
        """
#include <stdio.h>

int helper(int value) {
    return value + 1;
}

int greet(int name) {
    return helper(name);
}
""",
        encoding="utf-8",
    )
    (workspace / "Greeter.cpp").write_text(
        """
#include <string>

namespace demo {
class Greeter {
public:
    std::string greet(std::string name) {
        return helper(name);
    }

private:
    std::string helper(std::string value) {
        return value;
    }
};
}
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

    files = {item.path: item for item in repo.list_files(limit=10)}
    c_helpers = [node.qualified_name for node in repo.search_nodes("helper", language="c", limit=10)]
    cpp_helpers = [node.qualified_name for node in repo.search_nodes("helper", language="cpp", limit=10)]

    assert result.status == "complete"
    assert files["greeter.c"].node_count > 0
    assert files["Greeter.cpp"].node_count > 0
    assert c_helpers == ["helper"]
    assert cpp_helpers == ["demo.Greeter.helper"]


def test_indexer_does_not_count_non_extractable_jvm_files_against_foreground_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_parser(language_id: str) -> SimpleNamespace:
        return SimpleNamespace(available=language_id not in {"java", "kotlin"})

    monkeypatch.setattr(indexer_module, "load_parser", fake_load_parser)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "A.java").write_text("class A {}\n", encoding="utf-8")
    (workspace / "B.java").write_text("class B {}\n", encoding="utf-8")
    (workspace / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(
            available=True,
            missing=("tree_sitter_java", "tree_sitter_kotlin"),
            present=("tree_sitter", "tree_sitter_javascript", "tree_sitter_typescript"),
        )
    )
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=registry)

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=1,
    )

    assert result.status == "complete"
    assert result.counters["dependency_missing_language_skipped"] == 2
    assert [item.path for item in repo.list_files(limit=10)] == ["app.py"]


def test_indexer_does_not_count_non_extractable_csharp_files_against_foreground_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_parser(language_id: str) -> SimpleNamespace:
        return SimpleNamespace(available=language_id != "csharp")

    monkeypatch.setattr(indexer_module, "load_parser", fake_load_parser)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "A.cs").write_text("class A {}\n", encoding="utf-8")
    (workspace / "B.cs").write_text("class B {}\n", encoding="utf-8")
    (workspace / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(
            available=True,
            missing=("tree_sitter_c_sharp",),
            present=(
                "tree_sitter",
                "tree_sitter_javascript",
                "tree_sitter_typescript",
                "tree_sitter_java",
                "tree_sitter_kotlin",
            ),
        )
    )
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=registry)

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=1,
    )

    assert result.status == "complete"
    assert result.counters["dependency_missing_language_skipped"] == 2
    assert [item.path for item in repo.list_files(limit=10)] == ["app.py"]


def test_indexer_does_not_count_non_extractable_c_cpp_files_against_foreground_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_parser(language_id: str) -> SimpleNamespace:
        return SimpleNamespace(available=language_id not in {"c", "cpp"})

    monkeypatch.setattr(indexer_module, "load_parser", fake_load_parser)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "A.c").write_text("int a() { return 1; }\n", encoding="utf-8")
    (workspace / "B.cpp").write_text("int b() { return 1; }\n", encoding="utf-8")
    (workspace / "app.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(
            available=True,
            missing=("tree_sitter_c", "tree_sitter_cpp"),
            present=(
                "tree_sitter",
                "tree_sitter_javascript",
                "tree_sitter_typescript",
                "tree_sitter_java",
                "tree_sitter_kotlin",
                "tree_sitter_c_sharp",
            ),
        )
    )
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=registry)

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=1,
    )

    assert result.status == "complete"
    assert result.counters["dependency_missing_language_skipped"] == 2
    assert [item.path for item in repo.list_files(limit=10)] == ["app.py"]


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


def test_indexer_converts_non_value_extractor_exceptions_to_file_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "broken.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "healthy.py").write_text("y = 2\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    class _RaisingExtractor:
        def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
            if file_path == "broken.py":
                raise OSError("bad tsconfig")
            return ExtractionResult()

    indexer._extractors["python"] = _RaisingExtractor()  # noqa: SLF001

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
    assert files["broken.py"].status == "extraction_failed"
    assert files["broken.py"].errors == ("bad tsconfig",)
    assert files["healthy.py"].status == "indexed"


def test_indexer_registers_typescript_extractor_when_ts_parser_exists_without_tsx(monkeypatch) -> None:
    def fake_load_parser(language_id: str) -> SimpleNamespace:
        return SimpleNamespace(available=language_id == "typescript")

    monkeypatch.setattr(indexer_module, "load_parser", fake_load_parser)

    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    assert "typescript" in indexer._extractors  # noqa: SLF001


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
            symbol_extraction=result.candidates[0].symbol_extraction,
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
                symbol_extraction=candidate.symbol_extraction,
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
                    symbol_extraction=candidate.symbol_extraction,
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


def test_indexer_extracts_former_planned_cpp_extension_when_parser_exists(tmp_path: Path) -> None:
    _require_c_family_parsers()
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
    assert result.counters["files_indexed"] == 1
    assert repo.list_files(limit=10)[0].path == "main.cc"
    assert [node.qualified_name for node in repo.search_nodes("main", kind="function", language="cpp", limit=10)] == [
        "main"
    ]


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
                symbol_extraction=True,
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
