"""Tree-sitter Java extraction for native CodeGraph."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.CodeGraph.extractors.jvm_common import (
    JvmCallSite,
    column,
    declaration_payload_text,
    first_named_child_of_type,
    line,
    make_jvm_node,
    module_qualified_name,
    node_text,
    qualified_name,
    remember_callable,
    resolve_call_sites,
)
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import (
    CodeGraphNode,
    CodeGraphUnresolvedRef,
    ExtractionResult,
)

_TYPE_KINDS = {
    "class_declaration": "class",
    "enum_declaration": "enum",
    "interface_declaration": "interface",
    "record_declaration": "record",
}


class JavaTreeSitterExtractor:
    """Extract conservative Java symbols and same-file calls with Tree-sitter."""

    language_id = "java"

    def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
        """Parse one Java file and return symbols, calls, unresolved refs, and errors."""
        parser_result = load_parser("java")
        if parser_result.missing:
            return ExtractionResult(errors=(f"Missing Tree-sitter dependencies: {', '.join(parser_result.missing)}",))
        if parser_result.error or parser_result.parser is None:
            return ExtractionResult(errors=(parser_result.error or "Java parser unavailable",))

        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ExtractionResult(errors=(str(exc),))

        tree = parser_result.parser.parse(source)
        if tree.root_node.has_error:
            return ExtractionResult(errors=("Java parse error",))

        builder = _JavaGraphBuilder(workspace_key=workspace_key, file_path=file_path, source=source, source_text=text)
        return builder.build(tree.root_node)


class _JavaGraphBuilder:
    """Stateful Tree-sitter visitor for one Java source file."""

    def __init__(self, *, workspace_key: str, file_path: str, source: bytes, source_text: str) -> None:
        self.workspace_key = workspace_key
        self.file_path = file_path
        self.source = source
        self.source_text = source_text
        self.nodes: list[CodeGraphNode] = []
        self.unresolved_refs: list[CodeGraphUnresolvedRef] = []
        self._package_name = ""
        self._scope_stack: list[CodeGraphNode] = []
        self._type_stack: list[str] = []
        self._call_sites: list[JvmCallSite] = []
        self._callable_by_name: dict[str, CodeGraphNode | None] = {}

    def build(self, root_node: Any) -> ExtractionResult:
        """Visit a parsed Java program and resolve captured same-file call references."""
        module_node = self._make_node(
            kind="module",
            name=module_qualified_name(self.file_path).rsplit(".", 1)[-1],
            qualified_name_value=module_qualified_name(self.file_path),
            node=root_node,
            start_line=1,
            end_line=max(1, len(self.source_text.splitlines())),
        )
        self.nodes.append(module_node)
        self._scope_stack.append(module_node)
        for child in root_node.named_children:
            self._visit_top_level(child)
        self._scope_stack.pop()

        call_edges, call_unresolved_refs = resolve_call_sites(
            call_sites=tuple(self._call_sites),
            callable_by_name=self._callable_by_name,
            file_path=self.file_path,
            language_id=JavaTreeSitterExtractor.language_id,
            provenance="tree_sitter_java",
        )
        return ExtractionResult(
            nodes=tuple(self.nodes),
            edges=call_edges,
            unresolved_refs=(*self.unresolved_refs, *call_unresolved_refs),
        )

    def _visit_top_level(self, node: Any) -> None:
        if node.type == "package_declaration":
            self._visit_package_declaration(node)
            return
        if node.type == "import_declaration":
            self._visit_import_declaration(node)
            return
        if node.type in _TYPE_KINDS:
            self._visit_type_declaration(node)

    def _visit_package_declaration(self, node: Any) -> None:
        package_name = declaration_payload_text(self.source, node, "package")
        if not package_name:
            return
        self._package_name = package_name
        self.nodes.append(
            self._make_node(
                kind="package",
                name=package_name,
                qualified_name_value=package_name,
                node=node,
            )
        )

    def _visit_import_declaration(self, node: Any) -> None:
        imported = declaration_payload_text(self.source, node, "import")
        if not imported:
            return
        import_node = self._make_node(
            kind="import",
            name=imported,
            qualified_name_value=imported,
            node=node,
            metadata={"imported": imported, "source": imported},
        )
        self.nodes.append(import_node)
        self.unresolved_refs.append(
            CodeGraphUnresolvedRef(
                from_node_id=import_node.id,
                reference_name=imported,
                reference_kind="import",
                file_path=self.file_path,
                line=line(node),
                column=column(node),
                language=JavaTreeSitterExtractor.language_id,
            )
        )

    def _visit_type_declaration(self, node: Any) -> None:
        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return
        type_node = self._make_node(
            kind=_TYPE_KINDS[node.type],
            name=name,
            qualified_name_value=qualified_name(self._package_name, *self._type_stack, name),
            node=node,
            visibility=_visibility(node),
        )
        self.nodes.append(type_node)

        self._scope_stack.append(type_node)
        self._type_stack.append(name)
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.named_children:
                if child.type in _TYPE_KINDS:
                    self._visit_type_declaration(child)
                elif child.type == "constructor_declaration":
                    self._visit_constructor_declaration(child)
                elif child.type == "method_declaration":
                    self._visit_method_declaration(child)
        self._type_stack.pop()
        self._scope_stack.pop()

    def _visit_constructor_declaration(self, node: Any) -> None:
        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return
        constructor_node = self._make_node(
            kind="constructor",
            name=name,
            qualified_name_value=qualified_name(self._package_name, *self._type_stack, name),
            node=node,
            visibility=_visibility(node),
        )
        self.nodes.append(constructor_node)
        remember_callable(self._callable_by_name, constructor_node)
        self._visit_callable_body(node, constructor_node, "body")

    def _visit_method_declaration(self, node: Any) -> None:
        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return
        method_node = self._make_node(
            kind="method",
            name=name,
            qualified_name_value=qualified_name(self._package_name, *self._type_stack, name),
            node=node,
            visibility=_visibility(node),
        )
        self.nodes.append(method_node)
        remember_callable(self._callable_by_name, method_node)
        self._visit_callable_body(node, method_node, "body")

    def _visit_callable_body(self, node: Any, callable_node: CodeGraphNode, body_field: str) -> None:
        body = node.child_by_field_name(body_field)
        if body is None:
            return
        self._scope_stack.append(callable_node)
        self._visit_executable(body)
        self._scope_stack.pop()

    def _visit_executable(self, node: Any) -> None:
        if node.type == "method_invocation":
            self._visit_method_invocation(node)
        for child in node.named_children:
            self._visit_executable(child)

    def _visit_method_invocation(self, node: Any) -> None:
        current_scope = self._scope_stack[-1] if self._scope_stack else None
        if current_scope is None or current_scope.kind not in {"constructor", "method"}:
            return

        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return

        object_node = node.child_by_field_name("object")
        if object_node is None:
            self._call_sites.append(
                JvmCallSite(
                    source_node_id=current_scope.id,
                    reference_name=name,
                    line=line(node),
                    column=column(node),
                )
            )
            return

        self.unresolved_refs.append(
            CodeGraphUnresolvedRef(
                from_node_id=current_scope.id,
                reference_name=f"{node_text(self.source, object_node)}.{name}",
                reference_kind="call",
                file_path=self.file_path,
                line=line(node),
                column=column(node),
                language=JavaTreeSitterExtractor.language_id,
            )
        )

    def _make_node(
        self,
        *,
        kind: str,
        name: str,
        qualified_name_value: str,
        node: Any,
        start_line: int | None = None,
        end_line: int | None = None,
        visibility: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CodeGraphNode:
        return make_jvm_node(
            workspace_key=self.workspace_key,
            language_id=JavaTreeSitterExtractor.language_id,
            file_path=self.file_path,
            kind=kind,
            name=name,
            qualified_name_value=qualified_name_value,
            node=node,
            start_line=start_line,
            end_line=end_line,
            visibility=visibility,
            metadata=metadata,
        )


def _visibility(node: Any) -> str | None:
    modifiers = first_named_child_of_type(node, "modifiers")
    if modifiers is None:
        return None
    for child in modifiers.children:
        if child.type in {"public", "protected", "private"}:
            return child.type
    return None


__all__ = ["JavaTreeSitterExtractor"]
