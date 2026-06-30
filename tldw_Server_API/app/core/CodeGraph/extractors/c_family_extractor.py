"""Tree-sitter C and C++ extraction for native CodeGraph."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.CodeGraph.extractors.jvm_common import (
    JvmCallSite,
    column,
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
    "class_specifier": "class",
    "struct_specifier": "struct",
    "enum_specifier": "enum",
    "union_specifier": "union",
}
_DECLARATION_CONTAINERS = {
    "declaration",
    "declaration_list",
    "field_declaration",
    "field_declaration_list",
    "type_definition",
}
_NAME_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "namespace_identifier",
    "type_identifier",
}


class CTreeSitterExtractor:
    """Extract conservative C symbols and same-file calls with Tree-sitter."""

    language_id = "c"
    display_name = "C"
    provenance = "tree_sitter_c"

    def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
        """Parse one C file and return symbols, calls, unresolved refs, and errors."""
        return _extract_c_family(
            language_id=self.language_id,
            display_name=self.display_name,
            provenance=self.provenance,
            workspace_key=workspace_key,
            file_path=file_path,
            source=source,
        )


class CppTreeSitterExtractor:
    """Extract conservative C++ symbols and same-file calls with Tree-sitter."""

    language_id = "cpp"
    display_name = "C++"
    provenance = "tree_sitter_cpp"

    def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
        """Parse one C++ file and return symbols, calls, unresolved refs, and errors."""
        return _extract_c_family(
            language_id=self.language_id,
            display_name=self.display_name,
            provenance=self.provenance,
            workspace_key=workspace_key,
            file_path=file_path,
            source=source,
        )


def _extract_c_family(
    *,
    language_id: str,
    display_name: str,
    provenance: str,
    workspace_key: str,
    file_path: str,
    source: bytes,
) -> ExtractionResult:
    parser_result = load_parser(language_id)
    if parser_result.missing:
        return ExtractionResult(errors=(f"Missing Tree-sitter dependencies: {', '.join(parser_result.missing)}",))
    if parser_result.error or parser_result.parser is None:
        return ExtractionResult(errors=(parser_result.error or f"{display_name} parser unavailable",))

    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        return ExtractionResult(errors=(str(exc),))

    tree = parser_result.parser.parse(source)
    if tree.root_node.has_error:
        return ExtractionResult(errors=(f"{display_name} parse error",))

    builder = _CFamilyGraphBuilder(
        language_id=language_id,
        provenance=provenance,
        workspace_key=workspace_key,
        file_path=file_path,
        source=source,
        source_text=text,
    )
    return builder.build(tree.root_node)


class _CFamilyGraphBuilder:
    """Stateful Tree-sitter visitor for one C-family source file."""

    def __init__(
        self,
        *,
        language_id: str,
        provenance: str,
        workspace_key: str,
        file_path: str,
        source: bytes,
        source_text: str,
    ) -> None:
        self.language_id = language_id
        self.provenance = provenance
        self.workspace_key = workspace_key
        self.file_path = file_path
        self.source = source
        self.source_text = source_text
        self.nodes: list[CodeGraphNode] = []
        self.unresolved_refs: list[CodeGraphUnresolvedRef] = []
        self._namespace_name = ""
        self._scope_stack: list[CodeGraphNode] = []
        self._type_stack: list[str] = []
        self._call_sites: list[JvmCallSite] = []
        self._callable_by_name: dict[str, CodeGraphNode | None] = {}

    def build(self, root_node: Any) -> ExtractionResult:
        """Visit a parsed C-family translation unit and resolve same-file calls."""
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
        self._visit_declaration_children(root_node.named_children)
        self._scope_stack.pop()

        call_edges, call_unresolved_refs = resolve_call_sites(
            call_sites=tuple(self._call_sites),
            callable_by_name=self._callable_by_name,
            file_path=self.file_path,
            language_id=self.language_id,
            provenance=self.provenance,
        )
        return ExtractionResult(
            nodes=tuple(self.nodes),
            edges=call_edges,
            unresolved_refs=(*self.unresolved_refs, *call_unresolved_refs),
        )

    def _visit_declaration_children(self, children: list[Any] | tuple[Any, ...]) -> None:
        for child in children:
            self._visit_declaration_child(child)

    def _visit_declaration_child(self, node: Any) -> None:
        if node.type == "preproc_include":
            self._visit_include(node)
            return
        if node.type == "namespace_definition":
            self._visit_namespace_definition(node)
            return
        if node.type == "function_definition":
            self._visit_function_definition(node)
            return
        if node.type in _TYPE_KINDS:
            self._visit_type_specifier(node)
            return
        if node.type in _DECLARATION_CONTAINERS:
            self._visit_declaration_children(node.named_children)

    def _visit_include(self, node: Any) -> None:
        include_path = node_text(self.source, node.child_by_field_name("path")).strip()
        if not include_path:
            return
        import_node = self._make_node(
            kind="import",
            name=include_path,
            qualified_name_value=include_path,
            node=node,
            metadata={"imported": include_path, "source": include_path},
        )
        self.nodes.append(import_node)
        self.unresolved_refs.append(
            CodeGraphUnresolvedRef(
                from_node_id=import_node.id,
                reference_name=include_path,
                reference_kind="import",
                file_path=self.file_path,
                line=line(node),
                column=column(node),
                language=self.language_id,
            )
        )

    def _visit_namespace_definition(self, node: Any) -> None:
        local_name = node_text(self.source, node.child_by_field_name("name"))
        if not local_name:
            return
        previous_namespace = self._namespace_name
        namespace_name = qualified_name(previous_namespace, local_name)
        self._namespace_name = namespace_name
        self.nodes.append(
            self._make_node(
                kind="namespace",
                name=local_name,
                qualified_name_value=namespace_name,
                node=node,
            )
        )

        body = first_named_child_of_type(node, "declaration_list")
        if body is not None:
            self._visit_declaration_children(body.named_children)
        self._namespace_name = previous_namespace

    def _visit_type_specifier(self, node: Any) -> None:
        name = _node_name(self.source, node)
        if not name:
            return
        type_node = self._make_node(
            kind=_TYPE_KINDS[node.type],
            name=name,
            qualified_name_value=qualified_name(self._namespace_name, *self._type_stack, name),
            node=node,
        )
        self.nodes.append(type_node)

        body = first_named_child_of_type(node, "field_declaration_list", "declaration_list")
        if body is None:
            return
        self._scope_stack.append(type_node)
        self._type_stack.append(name)
        self._visit_declaration_children(body.named_children)
        self._type_stack.pop()
        self._scope_stack.pop()

    def _visit_function_definition(self, node: Any) -> None:
        name = _function_name(self.source, node)
        if not name:
            return
        function_node = self._make_node(
            kind="method" if self._type_stack else "function",
            name=name,
            qualified_name_value=qualified_name(self._namespace_name, *self._type_stack, name),
            node=node,
        )
        self.nodes.append(function_node)
        remember_callable(self._callable_by_name, function_node)

        body = first_named_child_of_type(node, "compound_statement")
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
        if current_scope is None or current_scope.kind not in {"function", "method"}:
            return

        function_node = node.child_by_field_name("function")
        if function_node is None:
            return

        if function_node.type in {"identifier", "field_identifier"}:
            reference_name = node_text(self.source, function_node)
            if reference_name:
                self._call_sites.append(
                    JvmCallSite(
                        source_node_id=current_scope.id,
                        reference_name=reference_name,
                        line=line(node),
                        column=column(node),
                    )
                )
            return

        reference_name = node_text(self.source, function_node)
        if reference_name:
            self.unresolved_refs.append(
                CodeGraphUnresolvedRef(
                    from_node_id=current_scope.id,
                    reference_name=reference_name,
                    reference_kind="call",
                    file_path=self.file_path,
                    line=line(node),
                    column=column(node),
                    language=self.language_id,
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
        metadata: dict[str, Any] | None = None,
    ) -> CodeGraphNode:
        return make_jvm_node(
            workspace_key=self.workspace_key,
            language_id=self.language_id,
            file_path=self.file_path,
            kind=kind,
            name=name,
            qualified_name_value=qualified_name_value,
            node=node,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata,
        )


def _node_name(source: bytes, node: Any) -> str:
    """Return the declared name for a C-family type or namespace node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(source, name_node)
    for child in node.named_children:
        if child.type in _NAME_NODE_TYPES:
            return node_text(source, child)
    return ""


def _function_name(source: bytes, node: Any) -> str:
    """Return the declared function name from a C-family function definition."""
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return ""
    return _declarator_name(source, declarator)


def _declarator_name(source: bytes, node: Any) -> str:
    """Walk nested declarator nodes to find the identifier that names a callable."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(source, name_node)
    nested = node.child_by_field_name("declarator")
    if nested is not None:
        return _declarator_name(source, nested)
    if node.type in _NAME_NODE_TYPES:
        return node_text(source, node)
    for child in node.named_children:
        name = _declarator_name(source, child)
        if name:
            return name
    return ""


__all__ = ["CTreeSitterExtractor", "CppTreeSitterExtractor"]
