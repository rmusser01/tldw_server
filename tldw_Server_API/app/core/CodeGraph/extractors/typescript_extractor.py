"""Tree-sitter TypeScript and TSX extraction for native CodeGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.CodeGraph.extractors.javascript_extractor import (
    JavaScriptGraphBuilder,
    first_named_child_of_type,
    node_text,
    qualified_name,
)
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult


class TypeScriptTreeSitterExtractor:
    """Extract conservative TypeScript and TSX symbols with Tree-sitter."""

    language_id = "typescript"

    def __init__(self, *, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root

    def extract(
        self,
        *,
        workspace_key: str,
        file_path: str,
        source: bytes,
        workspace_root: Path | None = None,
    ) -> ExtractionResult:
        """Parse one TypeScript/TSX file and return symbols, calls, unresolved refs, and errors."""
        parser_language = "tsx" if file_path.endswith(".tsx") else "typescript"
        parser_result = load_parser(parser_language)
        if parser_result.missing:
            return ExtractionResult(errors=(f"Missing Tree-sitter dependencies: {', '.join(parser_result.missing)}",))
        if parser_result.error or parser_result.parser is None:
            return ExtractionResult(errors=(parser_result.error or "TypeScript parser unavailable",))

        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ExtractionResult(errors=(str(exc),))

        tree = parser_result.parser.parse(source)
        if tree.root_node.has_error:
            return ExtractionResult(errors=("TypeScript parse error",))

        builder = _TypeScriptGraphBuilder(
            workspace_key=workspace_key,
            workspace_root=workspace_root or self.workspace_root,
            file_path=file_path,
            source=source,
            source_text=text,
            language_id=self.language_id,
        )
        return builder.build(tree.root_node)


class _TypeScriptGraphBuilder(JavaScriptGraphBuilder):
    """JS-family graph builder with TypeScript declaration support."""

    _TYPE_DECLARATION_KINDS = {
        "interface_declaration": "interface",
        "type_alias_declaration": "type_alias",
        "enum_declaration": "enum",
    }

    def _visit(self, node: Any, *, exported: bool = False) -> None:
        kind = self._TYPE_DECLARATION_KINDS.get(node.type)
        if kind is not None:
            self._visit_type_declaration(node, kind=kind, exported=exported)
            return
        super()._visit(node, exported=exported)

    def _visit_type_declaration(self, node: Any, *, kind: str, exported: bool) -> None:
        name_node = (
            node.child_by_field_name("name")
            or first_named_child_of_type(node, "type_identifier")
            or first_named_child_of_type(node, "identifier")
        )
        name = node_text(self.source, name_node)
        if not name:
            return
        self.nodes.append(
            self._make_node(
                kind=kind,
                name=name,
                qualified_name=qualified_name((*self._class_stack, name)),
                node=node,
                flags=("exported",) if exported else (),
            )
        )


__all__ = ["TypeScriptTreeSitterExtractor"]
