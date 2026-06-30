"""Conservative cross-file reference resolution for native CodeGraph indexes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.CodeGraph.models import (
    CodeGraphEdge,
    CodeGraphNode,
    StoredCodeGraphReference,
    make_edge_id,
)
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository


@dataclass(frozen=True)
class ResolutionResult:
    """Summary returned by one cross-file resolution pass."""

    resolved_calls: int = 0
    resolved_imports: int = 0
    stale_resolutions_cleared: int = 0
    truncated: bool = False
    import_nodes_scanned: int = 0
    references_scanned: int = 0


@dataclass(frozen=True)
class _ImportBinding:
    """One local import binding in a source file."""

    import_node: CodeGraphNode
    local_name: str
    target_node: CodeGraphNode | None
    target_file_path: str | None


@dataclass(frozen=True)
class _BindingIndex:
    """Import bindings grouped for direct and namespace lookup."""

    by_file: dict[str, list[_ImportBinding]]
    by_local_name: dict[str, dict[str, _ImportBinding]]


class CodeGraphReferenceResolver:
    """Resolve import-driven references against already indexed workspace nodes."""

    def __init__(self, repository: CodeGraphRepository) -> None:
        self._repository = repository

    def resolve(
        self,
        *,
        source_file_paths: set[str] | frozenset[str] | None = None,
        max_import_nodes: int | None = None,
        max_refs: int | None = None,
        deadline_monotonic: float | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> ResolutionResult:
        """Resolve currently unresolved refs and write deterministic graph edges."""
        clock = monotonic or time.monotonic
        if _deadline_expired(deadline_monotonic, clock):
            return ResolutionResult(truncated=True)

        stale = self._repository.clear_stale_reference_resolutions(file_paths=source_file_paths)
        import_nodes = self._repository.list_import_nodes(
            file_paths=source_file_paths,
            limit=_limit_plus_one(max_import_nodes),
        )
        truncated = _over_limit(import_nodes, max_import_nodes)
        if max_import_nodes is not None:
            import_nodes = import_nodes[:max_import_nodes]
        bindings = self._build_import_bindings(import_nodes, deadline_monotonic=deadline_monotonic, monotonic=clock)
        if bindings is None:
            return ResolutionResult(stale_resolutions_cleared=stale, truncated=True)

        import_edges, import_truncated = self._collect_import_edges(
            bindings,
            deadline_monotonic=deadline_monotonic,
            monotonic=clock,
        )
        truncated = truncated or import_truncated
        if import_edges:
            self._repository.upsert_edges(import_edges)

        references = self._repository.list_references_for_resolution(
            file_paths=source_file_paths,
            limit=_limit_plus_one(max_refs),
        )
        truncated = truncated or _over_limit(references, max_refs)
        if max_refs is not None:
            references = references[:max_refs]

        resolved_references: list[tuple[int, CodeGraphEdge, str]] = []
        for reference in references:
            if _deadline_expired(deadline_monotonic, clock):
                truncated = True
                break
            if reference.reference_kind != "call":
                continue
            target = self._resolve_call_reference(reference, bindings)
            if target is None:
                continue
            line = int(reference.line or 0)
            column = int(reference.column or 0)
            edge = CodeGraphEdge(
                id=make_edge_id(reference.from_node_id, "calls", target.id, reference.file_path, line, column),
                source=reference.from_node_id,
                target=target.id,
                kind="calls",
                file_path=reference.file_path,
                line=reference.line,
                column=reference.column,
                metadata={"resolved_by": "import_binding"},
                provenance="codegraph_resolver",
            )
            resolved_references.append((reference.id, edge, "import_binding"))

        if resolved_references:
            self._repository.mark_references_resolved(resolved_references)

        return ResolutionResult(
            resolved_calls=len(resolved_references),
            resolved_imports=len(import_edges),
            stale_resolutions_cleared=stale,
            truncated=truncated,
            import_nodes_scanned=len(import_nodes),
            references_scanned=len(references),
        )

    def _build_import_bindings(
        self,
        import_nodes: list[CodeGraphNode],
        *,
        deadline_monotonic: float | None,
        monotonic: Callable[[], float],
    ) -> _BindingIndex | None:
        """Build local import bindings grouped by source file."""
        by_file: dict[str, list[_ImportBinding]] = {}
        by_local_name: dict[str, dict[str, _ImportBinding]] = {}
        for import_node in import_nodes:
            if _deadline_expired(deadline_monotonic, monotonic):
                return None
            binding = self._binding_for_import_node(import_node)
            if binding is None:
                continue
            by_file.setdefault(import_node.file_path, []).append(binding)
            by_local_name.setdefault(import_node.file_path, {})[binding.local_name] = binding
        return _BindingIndex(by_file=by_file, by_local_name=by_local_name)

    def _binding_for_import_node(self, import_node: CodeGraphNode) -> _ImportBinding | None:
        metadata = dict(import_node.metadata)
        if import_node.language == "python":
            return self._python_binding(import_node, metadata)
        if import_node.language in {"javascript", "typescript"}:
            return self._js_ts_binding(import_node, metadata)
        return None

    def _python_binding(self, import_node: CodeGraphNode, metadata: dict[str, Any]) -> _ImportBinding | None:
        imported = str(metadata.get("imported") or import_node.qualified_name or "").strip()
        if not imported or imported.startswith("."):
            return None
        local_name = str(metadata.get("alias") or import_node.name).strip()
        if not local_name:
            return None
        target = self._resolve_python_import(imported)
        return _ImportBinding(
            import_node=import_node,
            local_name=local_name,
            target_node=target,
            target_file_path=target.file_path if target is not None else None,
        )

    def _js_ts_binding(self, import_node: CodeGraphNode, metadata: dict[str, Any]) -> _ImportBinding | None:
        resolved_path = metadata.get("resolved_path")
        if not isinstance(resolved_path, str) or not resolved_path:
            return None
        imported = str(metadata.get("imported") or import_node.name).strip()
        local_name = str(metadata.get("alias") or import_node.name).strip()
        if not imported or not local_name:
            return None
        target = self._resolve_target_in_file(resolved_path, imported)
        return _ImportBinding(
            import_node=import_node,
            local_name=local_name,
            target_node=target,
            target_file_path=resolved_path,
        )

    def _resolve_python_import(self, imported: str) -> CodeGraphNode | None:
        module = self._repository.find_module_node(imported)
        if module is not None:
            return module

        parts = imported.split(".")
        for index in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:index])
            symbol_name = ".".join(parts[index:])
            module = self._repository.find_module_node(module_name)
            if module is None:
                continue
            target = self._resolve_target_in_file(module.file_path, symbol_name)
            if target is not None:
                return target
        return None

    def _resolve_target_in_file(self, file_path: str, imported: str) -> CodeGraphNode | None:
        if imported in {"namespace", "side_effect"}:
            return self._repository.find_module_node_for_file(file_path)
        if imported == "default":
            return self._first_exported_node(file_path) or self._repository.find_module_node_for_file(file_path)

        candidates = self._repository.find_nodes_by_file_and_name(file_path=file_path, name=imported)
        for node in candidates:
            if node.kind != "import" and "exported" in node.flags:
                return node
        for node in candidates:
            if node.kind not in {"import", "module"}:
                return node
        return candidates[0] if candidates else None

    def _first_exported_node(self, file_path: str) -> CodeGraphNode | None:
        module = self._repository.find_module_node_for_file(file_path)
        if module is None:
            return None
        candidates = self._repository.find_nodes_by_file_and_name(file_path=file_path, name=module.name)
        for node in candidates:
            if "exported" in node.flags and node.kind != "module":
                return node
        return None

    def _collect_import_edges(
        self,
        bindings: _BindingIndex,
        *,
        deadline_monotonic: float | None,
        monotonic: Callable[[], float],
    ) -> tuple[list[CodeGraphEdge], bool]:
        """Build import dependency edges for impact/context traversal."""
        edges: list[CodeGraphEdge] = []
        seen: set[str] = set()
        for file_bindings in bindings.by_file.values():
            for binding in file_bindings:
                if _deadline_expired(deadline_monotonic, monotonic):
                    return edges, True
                target = binding.target_node
                if target is None:
                    continue
                line = int(binding.import_node.start_line or 0)
                column = int(binding.import_node.start_column or 0)
                edge = CodeGraphEdge(
                    id=make_edge_id(binding.import_node.id, "imports", target.id, binding.import_node.file_path, line, column),
                    source=binding.import_node.id,
                    target=target.id,
                    kind="imports",
                    file_path=binding.import_node.file_path,
                    line=binding.import_node.start_line,
                    column=binding.import_node.start_column,
                    metadata={"local_name": binding.local_name},
                    provenance="codegraph_resolver",
                )
                if edge.id in seen:
                    continue
                edges.append(edge)
                seen.add(edge.id)
        return edges, False

    def _resolve_call_reference(
        self,
        reference: StoredCodeGraphReference,
        bindings: _BindingIndex,
    ) -> CodeGraphNode | None:
        binding = bindings.by_local_name.get(reference.file_path, {}).get(reference.reference_name)
        if binding is not None:
            return binding.target_node if binding.target_node is not None and binding.target_node.kind != "module" else None
        for binding in bindings.by_file.get(reference.file_path, ()):
            prefix = f"{binding.local_name}."
            if reference.reference_name.startswith(prefix) and binding.target_file_path:
                symbol_name = reference.reference_name[len(prefix) :].rsplit(".", 1)[-1]
                return self._resolve_target_in_file(binding.target_file_path, symbol_name)
        return None


def _deadline_expired(deadline_monotonic: float | None, monotonic: Callable[[], float]) -> bool:
    """Return whether a caller-provided foreground time budget is exhausted."""
    return deadline_monotonic is not None and monotonic() >= deadline_monotonic


def _limit_plus_one(limit: int | None) -> int | None:
    """Return a SQLite row cap that lets callers detect deterministic truncation."""
    if limit is None:
        return None
    return max(1, int(limit)) + 1


def _over_limit(items: list[object], limit: int | None) -> bool:
    """Return whether a result list includes the extra row used to detect truncation."""
    return limit is not None and len(items) > max(1, int(limit))


__all__ = ["CodeGraphReferenceResolver", "ResolutionResult"]
