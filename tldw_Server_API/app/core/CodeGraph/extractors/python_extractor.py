"""Python AST extraction for the first native CodeGraph symbol slice."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

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
    """Deferred same-file call reference captured while walking a function body."""

    source_node_id: str
    reference_name: str
    line: int
    column: int


class PythonAstExtractor:
    """Extract conservative Python symbols and same-file calls with stdlib ast."""

    language_id = "python"

    def extract(self, *, workspace_key: str, file_path: str, source: bytes) -> ExtractionResult:
        """Parse one Python file and return symbols, calls, unresolved refs, and errors."""
        try:
            text = source.decode("utf-8")
            tree = ast.parse(text, filename=file_path)
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            return ExtractionResult(errors=(str(exc),))

        builder = _PythonGraphBuilder(workspace_key=workspace_key, file_path=file_path, source=text)
        return builder.build(tree)


class _PythonGraphBuilder(ast.NodeVisitor):
    """Stateful AST visitor that emits CodeGraph rows for one Python module."""

    def __init__(self, *, workspace_key: str, file_path: str, source: str) -> None:
        """Initialize per-file extraction state."""
        self.workspace_key = workspace_key
        self.file_path = file_path
        self.source = source
        self.nodes: list[CodeGraphNode] = []
        self.edges: list[CodeGraphEdge] = []
        self.unresolved_refs: list[CodeGraphUnresolvedRef] = []
        self._scope_stack: list[CodeGraphNode] = []
        self._class_stack: list[str] = []
        self._call_sites: list[_CallSite] = []
        self._callable_by_name: dict[str, CodeGraphNode | None] = {}

    def build(self, tree: ast.Module) -> ExtractionResult:
        """Visit a parsed module and resolve captured same-file call references."""
        module_name = _module_name_for_path(self.file_path)
        module_node = self._make_node(
            kind="module",
            name=module_name.rsplit(".", 1)[-1],
            qualified_name=module_name,
            node=tree,
            start_line=1,
            end_line=max(1, len(self.source.splitlines())),
        )
        self.nodes.append(module_node)
        self._scope_stack.append(module_node)
        self.visit(tree)
        self._scope_stack.pop()
        self._resolve_calls()
        return ExtractionResult(
            nodes=tuple(self.nodes),
            edges=tuple(self.edges),
            unresolved_refs=tuple(self.unresolved_refs),
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Record import statements as graph nodes without following them."""
        for alias in node.names:
            self.nodes.append(
                self._make_node(
                    kind="import",
                    name=alias.asname or alias.name.split(".")[-1],
                    qualified_name=alias.name,
                    node=node,
                    metadata={"imported": alias.name, "alias": alias.asname},
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Record from-import statements as graph nodes without following them."""
        module = "." * int(node.level) + (node.module or "")
        for alias in node.names:
            imported = f"{module}.{alias.name}" if module else alias.name
            self.nodes.append(
                self._make_node(
                    kind="import",
                    name=alias.asname or alias.name,
                    qualified_name=imported,
                    node=node,
                    metadata={"imported": imported, "alias": alias.asname},
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        """Record a class and visit its body in class-qualified scope."""
        qualified_name = ".".join([*self._class_stack, node.name])
        class_node = self._make_node(
            kind="class",
            name=node.name,
            qualified_name=qualified_name,
            node=node,
            docstring=ast.get_docstring(node),
        )
        self.nodes.append(class_node)
        self._remember_callable(class_node)

        self._scope_stack.append(class_node)
        self._class_stack.append(node.name)
        for item in node.body:
            self.visit(item)
        self._class_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Record a synchronous function or method definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        """Record an asynchronous function or method definition."""
        self._visit_function(node, is_async=True)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool = False,
    ) -> None:
        """Record a function-like node and visit nested executable statements."""
        container = ".".join(self._class_stack)
        qualified_name = f"{container}.{node.name}" if container else node.name
        function_node = self._make_node(
            kind="method" if self._class_stack else "function",
            name=node.name,
            qualified_name=qualified_name,
            node=node,
            docstring=ast.get_docstring(node),
            flags=("async",) if is_async else (),
        )
        self.nodes.append(function_node)
        self._remember_callable(function_node)

        self._scope_stack.append(function_node)
        for item in node.body:
            self.visit(item)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Capture call expressions that originate inside functions or methods."""
        if self._scope_stack:
            reference_name = _call_reference_name(node.func)
            current_scope = self._scope_stack[-1]
            if reference_name and current_scope.kind in {"function", "method"}:
                self._call_sites.append(
                    _CallSite(
                        source_node_id=current_scope.id,
                        reference_name=reference_name,
                        line=int(getattr(node, "lineno", 0) or 0),
                        column=int(getattr(node, "col_offset", 0) or 0) + 1,
                    )
                )
            elif current_scope.kind in {"function", "method"}:
                unresolved_name = _attribute_reference_name(node.func)
                if unresolved_name:
                    self.unresolved_refs.append(
                        CodeGraphUnresolvedRef(
                            from_node_id=current_scope.id,
                            reference_name=unresolved_name,
                            reference_kind="call",
                            file_path=self.file_path,
                            line=int(getattr(node, "lineno", 0) or 0),
                            column=int(getattr(node, "col_offset", 0) or 0) + 1,
                            language="python",
                        )
                    )
        self.generic_visit(node)

    def _make_node(
        self,
        *,
        kind: str,
        name: str,
        qualified_name: str,
        node: ast.AST,
        start_line: int | None = None,
        end_line: int | None = None,
        docstring: str | None = None,
        flags: tuple[str, ...] = (),
        metadata: dict[str, str | None] | None = None,
    ) -> CodeGraphNode:
        """Create a stable CodeGraph node from an AST node and qualified name."""
        resolved_start = start_line or int(getattr(node, "lineno", 1) or 1)
        identity_key = f"{self.workspace_key}:python:{self.file_path}:{kind}:{qualified_name}:{resolved_start}"
        return CodeGraphNode(
            id=make_node_id(self.workspace_key, "python", self.file_path, kind, qualified_name, resolved_start),
            identity_key=identity_key,
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            file_path=self.file_path,
            language="python",
            start_line=resolved_start,
            end_line=end_line or int(getattr(node, "end_lineno", resolved_start) or resolved_start),
            start_column=int(getattr(node, "col_offset", 0) or 0) + 1,
            end_column=(
                int(getattr(node, "end_col_offset", 0)) + 1
                if getattr(node, "end_col_offset", None) is not None
                else None
            ),
            docstring=docstring,
            flags=flags,
            metadata=dict(metadata or {}),
        )

    def _remember_callable(self, node: CodeGraphNode) -> None:
        """Register a callable by simple and qualified names unless ambiguous."""
        for key in {node.name, node.qualified_name}:
            existing = self._callable_by_name.get(key)
            if existing is None and key in self._callable_by_name:
                continue
            if existing is not None and existing.id != node.id:
                self._callable_by_name[key] = None
                continue
            self._callable_by_name[key] = node

    def _resolve_calls(self) -> None:
        """Convert captured call sites into resolved edges or unresolved references."""
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
                        language="python",
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
                    provenance="python_ast",
                )
            )


def _module_name_for_path(file_path: str) -> str:
    """Return a dotted module name from a workspace-relative Python path."""
    path = Path(file_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else path.stem


def _call_reference_name(node: ast.AST) -> str | None:
    """Return the comparable callable name for bare-name calls only."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _attribute_reference_name(node: ast.AST) -> str | None:
    """Return a dotted best-effort name for attribute calls that stay unresolved."""
    if not isinstance(node, ast.Attribute):
        return None

    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts)) if parts else None
