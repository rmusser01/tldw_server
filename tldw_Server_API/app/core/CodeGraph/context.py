"""Bounded source-context assembly for native CodeGraph tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode, codegraph_node_to_dict

_SNIPPET_CONTEXT_LINES = 3


class CodeGraphContextBuilder:
    """Build a compact, workspace-bounded source context payload."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        max_context_chars: int,
        max_file_size_bytes: int,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.max_context_chars = max(0, int(max_context_chars))
        self.max_file_size_bytes = max(0, int(max_file_size_bytes))

    def build(
        self,
        *,
        task: str,
        nodes: tuple[CodeGraphNode, ...],
        relationships: tuple[dict[str, Any], ...],
        max_files: int,
        include_code: bool,
    ) -> dict[str, Any]:
        """Return task-oriented nodes, relationships, files, and source snippets."""
        grouped = _group_nodes_by_file(nodes, max_files=max(1, int(max_files)))
        files: list[dict[str, Any]] = []
        used_chars = 0
        truncated = False
        skipped_files = 0

        for file_path, file_nodes in grouped:
            resolved = self._resolve_workspace_file(file_path)
            if resolved is None:
                skipped_files += 1
                continue

            file_context = {
                "path": file_path,
                "language": _dominant_language(file_nodes),
                "exists": resolved.exists(),
                "snippets": [],
                "errors": [],
            }
            if not resolved.exists():
                file_context["errors"].append("source file not found")
                files.append(file_context)
                continue

            if not resolved.is_file():
                file_context["errors"].append("source path is not a file")
                files.append(file_context)
                continue

            try:
                size = resolved.stat().st_size
            except OSError as exc:
                file_context["errors"].append(f"source file stat failed: {exc.strerror or exc.__class__.__name__}")
                files.append(file_context)
                continue

            if size > self.max_file_size_bytes:
                file_context["errors"].append("source file exceeds max_file_size_bytes")
                files.append(file_context)
                continue

            if include_code and not truncated:
                try:
                    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError as exc:
                    file_context["errors"].append(f"source file read failed: {exc.strerror or exc.__class__.__name__}")
                    files.append(file_context)
                    continue
                for node in file_nodes:
                    snippet = _make_snippet(node, lines)
                    remaining = self.max_context_chars - used_chars
                    if len(snippet["text"]) > remaining:
                        snippet["text"] = snippet["text"][: max(0, remaining)]
                        snippet["truncated"] = True
                        truncated = True
                    used_chars += len(snippet["text"])
                    file_context["snippets"].append(snippet)
                    if truncated:
                        break
            files.append(file_context)
            if truncated:
                break

        return {
            "task": task,
            "nodes": [codegraph_node_to_dict(node) for node in nodes],
            "relationships": list(relationships),
            "files": files,
            "truncation": {
                "max_context_chars": self.max_context_chars,
                "used_chars": used_chars,
                "truncated": truncated,
                "skipped_files": skipped_files,
            },
        }

    def _resolve_workspace_file(self, file_path: str) -> Path | None:
        candidate = Path(file_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        resolved = (self.workspace_root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            return None
        return resolved


def _group_nodes_by_file(
    nodes: tuple[CodeGraphNode, ...],
    *,
    max_files: int,
) -> list[tuple[str, list[CodeGraphNode]]]:
    grouped: dict[str, list[CodeGraphNode]] = {}
    for node in nodes:
        if node.file_path not in grouped and len(grouped) >= max_files:
            continue
        grouped.setdefault(node.file_path, []).append(node)
    return list(grouped.items())


def _dominant_language(nodes: list[CodeGraphNode]) -> str | None:
    for node in nodes:
        if node.language:
            return node.language
    return None


def _make_snippet(node: CodeGraphNode, lines: list[str]) -> dict[str, Any]:
    if not lines:
        return {
            "node_id": node.id,
            "start_line": 1,
            "end_line": 0,
            "text": "",
            "truncated": False,
        }

    node_start = max(1, int(node.start_line or 1))
    node_end = max(node_start, int(node.end_line or node_start))
    start_line = max(1, node_start - _SNIPPET_CONTEXT_LINES)
    end_line = min(len(lines), node_end + _SNIPPET_CONTEXT_LINES)
    text = "\n".join(lines[start_line - 1 : end_line])
    return {
        "node_id": node.id,
        "start_line": start_line,
        "end_line": end_line,
        "text": text,
        "truncated": False,
    }
