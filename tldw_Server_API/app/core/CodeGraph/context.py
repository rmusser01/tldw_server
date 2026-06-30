"""Bounded source-context assembly for native CodeGraph tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode, codegraph_node_to_dict

_SNIPPET_CONTEXT_LINES = 3
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


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


def rank_context_nodes(
    task: str,
    nodes: tuple[CodeGraphNode, ...],
    *,
    relationships: tuple[dict[str, Any], ...],
) -> tuple[CodeGraphNode, ...]:
    """Return context nodes ordered by task-token relevance and graph proximity."""
    tokens = _task_tokens(task)
    candidate_ids = {node.id for node in nodes}
    relationship_counts = _relationship_endpoint_counts(relationships, candidate_ids=candidate_ids)
    indexed_nodes = tuple(enumerate(nodes))

    return tuple(
        node
        for _index, node in sorted(
            indexed_nodes,
            key=lambda item: _ranking_key(item[0], item[1], tokens, relationship_counts),
        )
    )


def context_search_terms(task: str) -> tuple[str, ...]:
    """Return a deterministic sequence of search terms for a context task."""
    stripped = task.strip()
    terms: dict[str, None] = {}
    if stripped:
        terms[stripped] = None
    for token in _task_tokens(stripped):
        terms.setdefault(token, None)
    return tuple(terms)


def _dominant_language(nodes: list[CodeGraphNode]) -> str | None:
    for node in nodes:
        if node.language:
            return node.language
    return None


def _task_tokens(task: str) -> tuple[str, ...]:
    """Tokenize a task string into deterministic lowercase search terms."""
    seen: dict[str, None] = {}
    for match in _TOKEN_RE.finditer(task.lower()):
        token = match.group(0).strip("_")
        if len(token) >= 2:
            seen.setdefault(token, None)
    return tuple(seen)


def _relationship_endpoint_counts(
    relationships: tuple[dict[str, Any], ...],
    *,
    candidate_ids: set[str],
) -> dict[str, int]:
    """Count candidate ids that participate in candidate-to-candidate relationships."""
    counts: dict[str, int] = {}
    for relationship in relationships:
        source_id = _relationship_endpoint_id(relationship, "source")
        target_id = _relationship_endpoint_id(relationship, "target")
        if source_id not in candidate_ids or target_id not in candidate_ids:
            continue
        counts[source_id] = counts.get(source_id, 0) + 1
        counts[target_id] = counts.get(target_id, 0) + 1
    return counts


def _relationship_endpoint_id(relationship: dict[str, Any], endpoint_name: str) -> str | None:
    endpoint = relationship.get(endpoint_name)
    if not isinstance(endpoint, dict):
        return None
    node_id = endpoint.get("id")
    if not isinstance(node_id, str) or not node_id:
        return None
    return node_id


def _ranking_key(
    original_index: int,
    node: CodeGraphNode,
    tokens: tuple[str, ...],
    relationship_counts: dict[str, int],
) -> tuple[int, int, int]:
    """Build a stable sort key where lower values represent more relevant context."""
    relevance_score = _node_relevance_score(node, tokens)
    relationship_score = relationship_counts.get(node.id, 0)
    return (
        -relevance_score,
        -relationship_score,
        original_index,
    )


def _node_relevance_score(node: CodeGraphNode, tokens: tuple[str, ...]) -> int:
    """Score a node by exact and partial task-token matches in public identity fields."""
    if not tokens:
        return 0
    fields = (
        node.name.lower(),
        node.qualified_name.lower(),
        node.file_path.lower(),
    )
    score = 0
    for token in tokens:
        for field in fields:
            if field == token:
                score += 8
            elif field.endswith(f".{token}") or field.endswith(f"/{token}") or f"/{token}." in field:
                score += 5
            elif token in field:
                score += 2
    return score


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
