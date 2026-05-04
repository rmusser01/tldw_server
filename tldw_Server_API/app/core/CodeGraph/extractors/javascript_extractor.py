"""Tree-sitter JavaScript extraction for native CodeGraph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tldw_Server_API.app.core.CodeGraph.extractors.js_ts_imports as js_ts_imports
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import (
    CodeGraphEdge,
    CodeGraphNode,
    CodeGraphUnresolvedRef,
    ExtractionResult,
    make_edge_id,
    make_node_id,
)


@dataclass(frozen=True)
class _CallSite:
    """Deferred same-file JS call reference captured while walking a callable body."""

    source_node_id: str
    reference_name: str
    line: int
    column: int


class JavaScriptTreeSitterExtractor:
    """Extract conservative JavaScript and JSX symbols with Tree-sitter."""

    language_id = "javascript"

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
        """Parse one JavaScript/JSX file and return symbols, calls, unresolved refs, and errors."""
        parser_result = load_parser("javascript")
        if parser_result.missing:
            return ExtractionResult(errors=(f"Missing Tree-sitter dependencies: {', '.join(parser_result.missing)}",))
        if parser_result.error or parser_result.parser is None:
            return ExtractionResult(errors=(parser_result.error or "JavaScript parser unavailable",))

        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ExtractionResult(errors=(str(exc),))

        tree = parser_result.parser.parse(source)
        if tree.root_node.has_error:
            return ExtractionResult(errors=("JavaScript parse error",))

        builder = JavaScriptGraphBuilder(
            workspace_key=workspace_key,
            workspace_root=workspace_root or self.workspace_root,
            file_path=file_path,
            source=source,
            source_text=text,
            language_id=self.language_id,
        )
        return builder.build(tree.root_node)


class JavaScriptGraphBuilder:
    """Stateful Tree-sitter visitor for one JS-family source file."""

    def __init__(
        self,
        *,
        workspace_key: str,
        workspace_root: Path | None,
        file_path: str,
        source: bytes,
        source_text: str,
        language_id: str,
    ) -> None:
        self.workspace_key = workspace_key
        self.workspace_root = workspace_root.resolve() if workspace_root is not None else None
        self.file_path = file_path
        self.source = source
        self.source_text = source_text
        self.language_id = language_id
        self.project_config = (
            js_ts_imports.load_js_ts_project_config(self.workspace_root, self.file_path)
            if self.workspace_root is not None
            else None
        )
        self.nodes: list[CodeGraphNode] = []
        self.edges: list[CodeGraphEdge] = []
        self.unresolved_refs: list[CodeGraphUnresolvedRef] = []
        self._scope_stack: list[CodeGraphNode] = []
        self._class_stack: list[str] = []
        self._call_sites: list[_CallSite] = []
        self._callable_by_name: dict[str, CodeGraphNode | None] = {}

    def build(self, root_node: Any) -> ExtractionResult:
        """Visit a parsed JS-family program and resolve same-file call references."""
        module_name = Path(self.file_path).stem
        module_node = self._make_node(
            kind="module",
            name=module_name,
            qualified_name=_module_qualified_name(self.file_path),
            node=root_node,
            start_line=1,
            end_line=max(1, len(self.source_text.splitlines())),
        )
        self.nodes.append(module_node)
        self._scope_stack.append(module_node)
        for child in root_node.named_children:
            self._visit(child)
        self._scope_stack.pop()
        self._resolve_calls()
        return ExtractionResult(
            nodes=tuple(self.nodes),
            edges=tuple(self.edges),
            unresolved_refs=tuple(self.unresolved_refs),
        )

    def _visit(self, node: Any, *, exported: bool = False) -> None:
        if node.type == "import_statement":
            self._visit_import_statement(node)
            return
        if node.type == "export_statement":
            self._visit_export_statement(node)
            return
        if node.type in {"function_declaration", "generator_function_declaration"}:
            self._visit_function_declaration(node, exported=exported)
            return
        if node.type in {"lexical_declaration", "variable_declaration"}:
            for child in node.named_children:
                self._visit(child, exported=exported)
            return
        if node.type == "variable_declarator":
            self._visit_variable_declarator(node, exported=exported)
            return
        if node.type == "class_declaration":
            self._visit_class_declaration(node, exported=exported)
            return
        if node.type == "method_definition":
            self._visit_method_definition(node)
            return
        if node.type == "call_expression":
            self._visit_call_expression(node)
            return

        for child in node.named_children:
            self._visit(child, exported=exported)

    def _visit_export_statement(self, node: Any) -> None:
        declaration = node.child_by_field_name("declaration")
        if declaration is not None:
            self._visit(declaration, exported=True)
            return

        source_node = node.child_by_field_name("source")
        source_specifier = _string_literal_value(self.source, source_node)
        export_clause = first_named_child_of_type(node, "export_clause")
        if source_specifier and export_clause is not None:
            for specifier_node in _named_descendants_of_type(export_clause, "export_specifier"):
                imported = node_text(self.source, specifier_node.child_by_field_name("name"))
                local = node_text(self.source, specifier_node.child_by_field_name("alias")) or imported
                self._add_import_node(
                    node=specifier_node,
                    name=local,
                    source_specifier=source_specifier,
                    imported=imported,
                    alias=local if local != imported else None,
                    is_re_export=True,
                )

    def _visit_import_statement(self, node: Any) -> None:
        source_node = node.child_by_field_name("source")
        source_specifier = _string_literal_value(self.source, source_node)
        if not source_specifier:
            return

        import_clause = first_named_child_of_type(node, "import_clause")
        if import_clause is None:
            self._add_import_node(
                node=node,
                name=Path(source_specifier).name or source_specifier,
                source_specifier=source_specifier,
                imported="side_effect",
                alias=None,
                is_re_export=False,
            )
            return

        default_name = first_named_child_of_type(import_clause, "identifier")
        if default_name is not None:
            self._add_import_node(
                node=default_name,
                name=node_text(self.source, default_name),
                source_specifier=source_specifier,
                imported="default",
                alias=None,
                is_re_export=False,
            )

        for specifier_node in _named_descendants_of_type(import_clause, "import_specifier"):
            imported = node_text(self.source, specifier_node.child_by_field_name("name"))
            local = node_text(self.source, specifier_node.child_by_field_name("alias")) or imported
            self._add_import_node(
                node=specifier_node,
                name=local,
                source_specifier=source_specifier,
                imported=imported,
                alias=local if local != imported else None,
                is_re_export=False,
            )

        namespace_import = first_named_child_of_type(import_clause, "namespace_import")
        if namespace_import is not None:
            local_name = _last_identifier_text(namespace_import)
            if local_name:
                self._add_import_node(
                    node=namespace_import,
                    name=local_name,
                    source_specifier=source_specifier,
                    imported="namespace",
                    alias=local_name,
                    is_re_export=False,
                )

    def _visit_function_declaration(self, node: Any, *, exported: bool = False) -> None:
        name_node = node.child_by_field_name("name") or first_named_child_of_type(node, "identifier")
        name = node_text(self.source, name_node)
        if not name:
            return
        kind = "component" if _is_component_name(name) and _contains_jsx(node) else "function"
        function_node = self._make_node(
            kind=kind,
            name=name,
            qualified_name=qualified_name((*self._class_stack, name)),
            node=node,
            flags=("exported",) if exported else (),
        )
        self.nodes.append(function_node)
        self._remember_callable(function_node)

        self._scope_stack.append(function_node)
        body = node.child_by_field_name("body")
        if body is not None:
            self._visit(body)
        self._scope_stack.pop()

    def _visit_variable_declarator(self, node: Any, *, exported: bool = False) -> None:
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        if value_node is None or value_node.type not in {"arrow_function", "function_expression"}:
            for child in node.named_children:
                self._visit(child, exported=exported)
            return

        name = node_text(self.source, name_node)
        if not name:
            return
        kind = "component" if _is_component_name(name) and _contains_jsx(value_node) else "function"
        function_node = self._make_node(
            kind=kind,
            name=name,
            qualified_name=qualified_name((*self._class_stack, name)),
            node=node,
            flags=("exported", "arrow") if exported else ("arrow",),
        )
        self.nodes.append(function_node)
        self._remember_callable(function_node)

        self._scope_stack.append(function_node)
        self._visit(value_node)
        self._scope_stack.pop()

    def _visit_class_declaration(self, node: Any, *, exported: bool = False) -> None:
        name_node = node.child_by_field_name("name") or first_named_child_of_type(node, "identifier")
        name = node_text(self.source, name_node)
        if not name:
            return
        class_node = self._make_node(
            kind="class",
            name=name,
            qualified_name=qualified_name((*self._class_stack, name)),
            node=node,
            flags=("exported",) if exported else (),
        )
        self.nodes.append(class_node)
        self._remember_callable(class_node)

        self._scope_stack.append(class_node)
        self._class_stack.append(name)
        body = node.child_by_field_name("body") or first_named_child_of_type(node, "class_body")
        if body is not None:
            for child in body.named_children:
                self._visit(child)
        self._class_stack.pop()
        self._scope_stack.pop()

    def _visit_method_definition(self, node: Any) -> None:
        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return
        method_node = self._make_node(
            kind="method",
            name=name,
            qualified_name=qualified_name((*self._class_stack, name)),
            node=node,
        )
        self.nodes.append(method_node)
        self._remember_callable(method_node)

        self._scope_stack.append(method_node)
        body = node.child_by_field_name("body")
        if body is not None:
            self._visit(body)
        self._scope_stack.pop()

    def _visit_call_expression(self, node: Any) -> None:
        current_scope = self._scope_stack[-1] if self._scope_stack else None
        function_node = node.child_by_field_name("function")
        if current_scope is not None and current_scope.kind in {"function", "method", "component"}:
            if function_node is not None and function_node.type == "identifier":
                self._call_sites.append(
                    _CallSite(
                        source_node_id=current_scope.id,
                        reference_name=node_text(self.source, function_node),
                        line=_line(node),
                        column=_column(node),
                    )
                )
            elif function_node is not None and function_node.type == "member_expression":
                reference_name = _member_reference_name(self.source, function_node)
                if reference_name:
                    self.unresolved_refs.append(
                        CodeGraphUnresolvedRef(
                            from_node_id=current_scope.id,
                            reference_name=reference_name,
                            reference_kind="call",
                            file_path=self.file_path,
                            line=_line(node),
                            column=_column(node),
                            language=self.language_id,
                        )
                    )

        for child in node.named_children:
            self._visit(child)

    def _add_import_node(
        self,
        *,
        node: Any,
        name: str,
        source_specifier: str,
        imported: str,
        alias: str | None,
        is_re_export: bool,
    ) -> None:
        resolution = (
            js_ts_imports.resolve_js_ts_import_with_config(
                self.workspace_root,
                self.file_path,
                source_specifier,
                self.project_config,
            )
            if self.workspace_root is not None
            else None
        )
        metadata: dict[str, Any] = {
            "source": source_specifier,
            "imported": imported,
            "alias": alias,
            "re_export": is_re_export,
        }
        if resolution is not None:
            metadata.update(
                {
                    "resolution_kind": resolution.resolution_kind,
                    "resolved_path": resolution.resolved_path,
                    "resolution_reason": resolution.reason,
                }
            )

        import_node = self._make_node(
            kind="import",
            name=name,
            qualified_name=f"{source_specifier}:{imported}:{name}",
            node=node,
            metadata=metadata,
        )
        self.nodes.append(import_node)

        if resolution is not None and resolution.resolution_kind == "external":
            self.unresolved_refs.append(
                CodeGraphUnresolvedRef(
                    from_node_id=import_node.id,
                    reference_name=source_specifier,
                    reference_kind="import",
                    file_path=self.file_path,
                    line=_line(node),
                    column=_column(node),
                    language=self.language_id,
                )
            )
        elif resolution is not None and resolution.resolution_kind == "unresolved":
            self.unresolved_refs.append(
                CodeGraphUnresolvedRef(
                    from_node_id=import_node.id,
                    reference_name=source_specifier,
                    reference_kind="import",
                    file_path=self.file_path,
                    line=_line(node),
                    column=_column(node),
                    candidates=resolution.candidates,
                    language=self.language_id,
                )
            )

    def _make_node(
        self,
        *,
        kind: str,
        name: str,
        qualified_name: str,
        node: Any,
        start_line: int | None = None,
        end_line: int | None = None,
        flags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> CodeGraphNode:
        resolved_start = start_line or _line(node)
        identity_key = (
            f"{self.workspace_key}:{self.language_id}:{self.file_path}:{kind}:{qualified_name}:{resolved_start}"
        )
        return CodeGraphNode(
            id=make_node_id(self.workspace_key, self.language_id, self.file_path, kind, qualified_name, resolved_start),
            identity_key=identity_key,
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            file_path=self.file_path,
            language=self.language_id,
            start_line=resolved_start,
            end_line=end_line or _end_line(node),
            start_column=_column(node),
            end_column=_end_column(node),
            flags=flags,
            metadata=dict(metadata or {}),
        )

    def _remember_callable(self, node: CodeGraphNode) -> None:
        for key in {node.name, node.qualified_name}:
            existing = self._callable_by_name.get(key)
            if existing is None and key in self._callable_by_name:
                continue
            if existing is not None and existing.id != node.id:
                self._callable_by_name[key] = None
                continue
            self._callable_by_name[key] = node

    def _resolve_calls(self) -> None:
        for call in self._call_sites:
            target = self._callable_by_name.get(call.reference_name)
            if target is None:
                self.unresolved_refs.append(
                    CodeGraphUnresolvedRef(
                        from_node_id=call.source_node_id,
                        reference_name=call.reference_name,
                        reference_kind="call",
                        file_path=self.file_path,
                        line=call.line,
                        column=call.column,
                        language=self.language_id,
                    )
                )
                continue
            self.edges.append(
                CodeGraphEdge(
                    id=make_edge_id(
                        call.source_node_id,
                        "calls",
                        target.id,
                        self.file_path,
                        call.line,
                        call.column,
                    ),
                    source=call.source_node_id,
                    target=target.id,
                    kind="calls",
                    file_path=self.file_path,
                    line=call.line,
                    column=call.column,
                    provenance="tree_sitter",
                )
            )


def node_text(source: bytes, node: Any | None) -> str:
    """Return a node's UTF-8 source slice, or an empty string for missing nodes."""
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _string_literal_value(source: bytes, node: Any | None) -> str:
    if node is None:
        return ""
    for child in node.named_children:
        if child.type == "string_fragment":
            return node_text(source, child)
    value = node_text(source, node)
    return value[1:-1] if len(value) >= 2 and value[0] in {'"', "'"} else value


def first_named_child_of_type(node: Any, node_type: str) -> Any | None:
    """Return the first direct named child with the requested Tree-sitter node type."""
    for child in node.named_children:
        if child.type == node_type:
            return child
    return None


def _named_descendants_of_type(node: Any, node_type: str) -> tuple[Any, ...]:
    matches: list[Any] = []
    stack = list(reversed(node.named_children))
    while stack:
        current = stack.pop()
        if current.type == node_type:
            matches.append(current)
        stack.extend(reversed(current.named_children))
    return tuple(matches)


def _last_identifier_text(node: Any) -> str:
    identifiers = list(_named_descendants_of_type(node, "identifier"))
    if not identifiers:
        return ""
    return identifiers[-1].text.decode("utf-8")


def _member_reference_name(source: bytes, node: Any) -> str:
    object_node = node.child_by_field_name("object")
    property_node = node.child_by_field_name("property")
    if object_node is not None and object_node.type == "member_expression":
        object_name = _member_reference_name(source, object_node)
    else:
        object_name = node_text(source, object_node)
    property_name = node_text(source, property_node)
    if object_name and property_name:
        return f"{object_name}.{property_name}"
    return node_text(source, node)


def _contains_jsx(node: Any) -> bool:
    if node.type.startswith("jsx_"):
        return True
    return any(_contains_jsx(child) for child in node.named_children)


def _is_component_name(name: str) -> bool:
    return bool(name) and name[0].isupper()


def qualified_name(parts: tuple[str, ...]) -> str:
    """Join qualified-name parts while dropping empty segments."""
    return ".".join(part for part in parts if part)


def _module_qualified_name(file_path: str) -> str:
    path = Path(file_path)
    return ".".join((*path.with_suffix("").parts,))


def _line(node: Any) -> int:
    return int(node.start_point.row) + 1


def _column(node: Any) -> int:
    return int(node.start_point.column) + 1


def _end_line(node: Any) -> int:
    return int(node.end_point.row) + 1


def _end_column(node: Any) -> int:
    return int(node.end_point.column) + 1


__all__ = [
    "JavaScriptGraphBuilder",
    "JavaScriptTreeSitterExtractor",
    "first_named_child_of_type",
    "node_text",
    "qualified_name",
]
