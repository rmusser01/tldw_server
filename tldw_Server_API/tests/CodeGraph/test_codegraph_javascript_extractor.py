from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors import js_ts_imports
from tldw_Server_API.app.core.CodeGraph.extractors.javascript_extractor import (
    JavaScriptTreeSitterExtractor,
)
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult

pytestmark = pytest.mark.skipif(
    not load_parser("javascript").available,
    reason="tree-sitter-javascript parser is not available",
)


def test_javascript_extractor_records_module_symbols_imports_and_calls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src" / "lib").mkdir(parents=True)
    (workspace / "src" / "lib" / "api.js").write_text("export const apiClient = {};\n", encoding="utf-8")
    (workspace / "src" / "shared.js").write_text("export function helper() {}\n", encoding="utf-8")

    result = _extract(
        workspace,
        """
import React from "react";
import { apiClient as client } from "./lib/api";
export { helper as sharedHelper } from "./shared";

export function Helper(value) {
  return value + 1;
}

const loadData = () => {
  return Helper(1);
};

class Widget {
  render() {
    client.fetch();
    return Helper(2);
  }
}

export function Card() {
  return <div>{loadData()}</div>;
}
""",
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}
    imports = [node for node in result.nodes if node.kind == "import"]

    assert ("module", "app") in nodes_by_kind_name
    assert ("function", "Helper") in nodes_by_kind_name
    assert ("function", "loadData") in nodes_by_kind_name
    assert ("class", "Widget") in nodes_by_kind_name
    assert ("method", "render") in nodes_by_kind_name
    assert ("component", "Card") in nodes_by_kind_name
    assert [(node.name, node.metadata["source"], node.metadata["imported"]) for node in imports] == [
        ("React", "react", "default"),
        ("client", "./lib/api", "apiClient"),
        ("sharedHelper", "./shared", "helper"),
    ]

    helper = nodes_by_kind_name[("function", "Helper")]
    load_data = nodes_by_kind_name[("function", "loadData")]
    render = nodes_by_kind_name[("method", "render")]
    card = nodes_by_kind_name[("component", "Card")]

    assert (load_data.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert (render.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert (card.id, load_data.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "call" and ref.reference_name == "client.fetch" and ref.from_node_id == render.id
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "react"
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


def test_javascript_extractor_marks_parse_errors() -> None:
    result = JavaScriptTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/broken.js",
        source=b"function broken(",
    )

    assert result == ExtractionResult(errors=("JavaScript parse error",))


def test_javascript_extractor_loads_project_config_once_per_file(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    calls: list[tuple[Path, str | Path]] = []

    def fake_load_js_ts_project_config(
        workspace_root: Path,
        source_path: str | Path,
    ) -> js_ts_imports.JsTsProjectConfig | None:
        calls.append((workspace_root, source_path))
        return None

    monkeypatch.setattr(js_ts_imports, "load_js_ts_project_config", fake_load_js_ts_project_config)

    result = JavaScriptTreeSitterExtractor(workspace_root=workspace).extract(
        workspace_key="ws",
        file_path="src/app.jsx",
        source=b"""
import React from "react";
import { helper } from "@web/helper";
export { Card } from "@tldw/ui/Card";
""",
    )

    assert result.errors == ()
    assert calls == [(workspace.resolve(), "src/app.jsx")]


def test_javascript_extractor_uses_deterministic_node_ids() -> None:
    first = JavaScriptTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/app.js",
        source=b"export function helper() { return 1; }\n",
    )
    second = JavaScriptTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/app.js",
        source=b"export function helper() { return 1; }\n",
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]


def _extract(workspace: Path, source: str) -> ExtractionResult:
    return JavaScriptTreeSitterExtractor(workspace_root=workspace).extract(
        workspace_key="ws",
        file_path="src/app.jsx",
        source=source.encode("utf-8"),
    )
