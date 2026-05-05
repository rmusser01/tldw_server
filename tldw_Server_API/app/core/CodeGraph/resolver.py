"""Conservative cross-file reference resolution for native CodeGraph indexes."""

from __future__ import annotations

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


@dataclass(frozen=True)
class _ImportBinding:
    """One local import binding in a source file."""

    import_node: CodeGraphNode
    local_name: str
    target_node: CodeGraphNode | None
    target_file_path: str | None


class CodeGraphReferenceResolver:
    """Resolve import-driven references against already indexed workspace nodes."""

    def __init__(self, repository: CodeGraphRepository) -> None:
        self._repository = repository

    def resolve(self) -> ResolutionResult:
        """Resolve currently unresolved refs and write deterministic graph edges."""
        stale = self._repository.clear_stale_reference_resolutions()
        bindings = self._build_import_bindings()
        resolved_imports = self._write_import_edges(bindings)
        resolved_calls = 0

        for reference in self._repository.list_references_for_resolution():
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
            self._repository.mark_reference_resolved(reference.id, edge=edge, resolution_kind="import_binding")
            resolved_calls += 1

        return ResolutionResult(
            resolved_calls=resolved_calls,
            resolved_imports=resolved_imports,
            stale_resolutions_cleared=stale,
        )

    def _build_import_bindings(self) -> dict[str, list[_ImportBinding]]:
        """Build local import bindings grouped by source file."""
        by_file: dict[str, list[_ImportBinding]] = {}
        for import_node in self._repository.list_import_nodes():
            binding = self._binding_for_import_node(import_node)
            if binding is None:
                continue
            by_file.setdefault(import_node.file_path, []).append(binding)
        return by_file

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

    def _write_import_edges(self, bindings: dict[str, list[_ImportBinding]]) -> int:
        """Persist import dependency edges for impact/context traversal."""
        resolved = 0
        seen: set[str] = set()
        for file_bindings in bindings.values():
            for binding in file_bindings:
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
                self._repository.upsert_edge(edge)
                seen.add(edge.id)
                resolved += 1
        return resolved

    def _resolve_call_reference(
        self,
        reference: StoredCodeGraphReference,
        bindings: dict[str, list[_ImportBinding]],
    ) -> CodeGraphNode | None:
        for binding in bindings.get(reference.file_path, ()):
            if reference.reference_name == binding.local_name:
                return binding.target_node if binding.target_node is not None and binding.target_node.kind != "module" else None
            prefix = f"{binding.local_name}."
            if reference.reference_name.startswith(prefix) and binding.target_file_path:
                symbol_name = reference.reference_name[len(prefix) :].rsplit(".", 1)[-1]
                return self._resolve_target_in_file(binding.target_file_path, symbol_name)
        return None


__all__ = ["CodeGraphReferenceResolver", "ResolutionResult"]
