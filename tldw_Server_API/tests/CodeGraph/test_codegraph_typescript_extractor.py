from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.extractors.typescript_extractor import (
    TypeScriptTreeSitterExtractor,
)

pytestmark = pytest.mark.skipif(
    not load_parser("typescript").available,
    reason="tree-sitter-typescript parser is not available",
)


def test_typescript_extractor_records_ts_symbols_imports_and_calls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "types.ts").write_text("export interface User { id: string }\n", encoding="utf-8")

    result = TypeScriptTreeSitterExtractor(workspace_root=workspace).extract(
        workspace_key="ws",
        file_path="src/service.ts",
        source=b"""
import type { User } from "./types";
import { z } from "zod";

export interface Account { id: string }
type Result<T> = T | null;
export enum Mode { Read, Write }

export function makeAccount(id: string): Account {
  return { id };
}

class Service {
  run(): Result<Account> {
    return makeAccount("1");
  }
}
""",
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("interface", "Account") in nodes_by_kind_name
    assert ("type_alias", "Result") in nodes_by_kind_name
    assert ("enum", "Mode") in nodes_by_kind_name
    assert ("function", "makeAccount") in nodes_by_kind_name
    assert ("class", "Service") in nodes_by_kind_name
    assert ("method", "run") in nodes_by_kind_name
    assert [
        (node.name, node.metadata["source"], node.metadata["imported"])
        for node in result.nodes
        if node.kind == "import"
    ] == [
        ("User", "./types", "User"),
        ("z", "zod", "z"),
    ]

    make_account = nodes_by_kind_name[("function", "makeAccount")]
    run = nodes_by_kind_name[("method", "run")]

    assert (run.id, make_account.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "zod"
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


@pytest.mark.skipif(
    not load_parser("tsx").available,
    reason="tree-sitter-typescript TSX parser is not available",
)
def test_tsx_extractor_records_component_function() -> None:
    result = TypeScriptTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Card.tsx",
        source=b"export function Card(props: { title: string }) { return <div>{props.title}</div>; }\n",
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("component", "Card") in nodes_by_kind_name
    assert result.errors == ()


def test_typescript_extractor_uses_deterministic_node_ids() -> None:
    source = b"export interface Account { id: string }\nexport enum Mode { Read }\n"
    first = TypeScriptTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/types.ts",
        source=source,
    )
    second = TypeScriptTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/types.ts",
        source=source,
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
