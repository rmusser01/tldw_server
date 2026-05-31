"""Tree-sitter Kotlin extraction for native CodeGraph."""

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


class KotlinTreeSitterExtractor:
    """Extract conservative Kotlin symbols and same-file calls with Tree-sitter."""

    language_id = "kotlin"

    def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
        """Parse one Kotlin file and return symbols, calls, unresolved refs, and errors."""
        parser_result = load_parser("kotlin")
        if parser_result.missing:
            return ExtractionResult(errors=(f"Missing Tree-sitter dependencies: {', '.join(parser_result.missing)}",))
        if parser_result.error or parser_result.parser is None:
            return ExtractionResult(errors=(parser_result.error or "Kotlin parser unavailable",))

        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ExtractionResult(errors=(str(exc),))

        tree = parser_result.parser.parse(source)
        if tree.root_node.has_error:
            return ExtractionResult(errors=("Kotlin parse error",))

        builder = _KotlinGraphBuilder(
            workspace_key=workspace_key,
            file_path=file_path,
            source=source,
            source_text=text,
        )
        return builder.build(tree.root_node)


class _KotlinGraphBuilder:
    """Stateful Tree-sitter visitor for one Kotlin source file."""

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
        """Visit a parsed Kotlin source file and resolve same-file calls."""
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
            language_id=KotlinTreeSitterExtractor.language_id,
            provenance="tree_sitter_kotlin",
        )
        return ExtractionResult(
            nodes=tuple(self.nodes),
            edges=call_edges,
            unresolved_refs=(*self.unresolved_refs, *call_unresolved_refs),
        )

    def _visit_top_level(self, node: Any) -> None:
        if node.type == "package_header":
            self._visit_package_header(node)
            return
        if node.type == "import":
            self._visit_import(node)
            return
        if node.type in {"class_declaration", "object_declaration"}:
            self._visit_type_declaration(node)
            return
        if node.type == "function_declaration":
            self._visit_function_declaration(node)

    def _visit_package_header(self, node: Any) -> None:
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

    def _visit_import(self, node: Any) -> None:
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
                language=KotlinTreeSitterExtractor.language_id,
            )
        )

    def _visit_type_declaration(self, node: Any) -> None:
        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return
        type_node = self._make_node(
            kind=_kotlin_type_kind(self.source, node),
            name=name,
            qualified_name_value=qualified_name(self._package_name, *self._type_stack, name),
            node=node,
            visibility=_visibility(self.source, node),
        )
        self.nodes.append(type_node)

        self._scope_stack.append(type_node)
        self._type_stack.append(name)
        body = first_named_child_of_type(node, "class_body")
        if body is not None:
            for child in body.named_children:
                if child.type in {"class_declaration", "object_declaration"}:
                    self._visit_type_declaration(child)
                elif child.type == "function_declaration":
                    self._visit_function_declaration(child)
        self._type_stack.pop()
        self._scope_stack.pop()

    def _visit_function_declaration(self, node: Any) -> None:
        name_node = node.child_by_field_name("name")
        name = node_text(self.source, name_node)
        if not name:
            return
        function_node = self._make_node(
            kind="function",
            name=name,
            qualified_name_value=qualified_name(self._package_name, *self._type_stack, name),
            node=node,
            visibility=_visibility(self.source, node),
        )
        self.nodes.append(function_node)
        remember_callable(self._callable_by_name, function_node)

        body = first_named_child_of_type(node, "function_body")
        if body is None:
            return
        self._scope_stack.append(function_node)
        self._visit_executable(body)
        self._scope_stack.pop()

    def _visit_executable(self, node: Any) -> None:
        if node.type == "call_expression":
            self._visit_call_expression(node)
        for child in node.named_children:
            self._visit_executable(child)

    def _visit_call_expression(self, node: Any) -> None:
        current_scope = self._scope_stack[-1] if self._scope_stack else None
        if current_scope is None or current_scope.kind != "function":
            return

        callee_node = node.named_children[0] if node.named_children else None
        if callee_node is None:
            return

        if callee_node.type == "identifier":
            self._call_sites.append(
                JvmCallSite(
                    source_node_id=current_scope.id,
                    reference_name=node_text(self.source, callee_node),
                    line=line(node),
                    column=column(node),
                )
            )
            return

        if callee_node.type == "navigation_expression":
            reference_name = _navigation_reference_name(self.source, callee_node)
            if reference_name:
                self.unresolved_refs.append(
                    CodeGraphUnresolvedRef(
                        from_node_id=current_scope.id,
                        reference_name=reference_name,
                        reference_kind="call",
                        file_path=self.file_path,
                        line=line(node),
                        column=column(node),
                        language=KotlinTreeSitterExtractor.language_id,
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
            language_id=KotlinTreeSitterExtractor.language_id,
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


def _kotlin_type_kind(source: bytes, node: Any) -> str:
    if node.type == "object_declaration":
        return "object"
    del source
    if any(child.type == "interface" for child in node.children):
        return "interface"
    return "class"


def _visibility(source: bytes, node: Any) -> str | None:
    modifiers = first_named_child_of_type(node, "modifiers")
    if modifiers is None:
        return None
    visibility = first_named_child_of_type(modifiers, "visibility_modifier")
    return node_text(source, visibility) or None


def _navigation_reference_name(source: bytes, node: Any) -> str:
    return node_text(source, node)


__all__ = ["KotlinTreeSitterExtractor"]
