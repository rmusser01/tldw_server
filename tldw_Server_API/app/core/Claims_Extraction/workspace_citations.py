"""Resolve citation locators against enrolled, user-owned canonical media text."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Media.navigation import load_navigation_sections
from tldw_Server_API.app.core.Media.page_provenance import page_spans_for_range, resolve_navigation_span
from tldw_Server_API.app.core.Research_Workspace.section_attribution import (
    build_section_attribution_contexts,
    render_attribution_context,
)


def _attribution_context(snapshot: dict[str, Any], source: dict[str, Any], start: int, end: int) -> str:
    """Carry only a uniquely enclosing verified section's literal attribution text."""
    candidates = []
    nodes = snapshot.get("nodes") or []
    for node in nodes:
        if node.get("parent_id"):
            continue
        span = resolve_navigation_span(node, nodes, snapshot.get("page_map"), len(snapshot["content"]), content=snapshot["content"])
        if span and span[0] <= start < end <= span[1]:
            candidates.append((node, span))
    if len(candidates) != 1:
        return ""
    node, (left, right) = candidates[0]
    section = {"source_id": source["id"], "media_id": source["media_id"], "node_id": node["id"],
        "title": node.get("title", ""), "navigation_version": snapshot["navigation_version"],
        "start_char": left, "end_char": right}
    contexts = build_section_attribution_contexts(section, snapshot)
    return render_attribution_context(contexts, start, end)


def _resolve_quote(locator: dict[str, Any], snapshot: dict[str, Any]) -> tuple[int, int] | None:
    """Offsets are exact, or a unique literal substring is required; never normalize."""
    text, quote = snapshot["content"], locator["quote"]
    if locator.get("navigation_version") not in (None, snapshot["navigation_version"]):
        return None
    start, end = locator.get("start_char"), locator.get("end_char")
    if start is None and end is None:
        start = text.find(quote)
        if start < 0 or text.find(quote, start + 1) >= 0:
            return None
        end = start + len(quote)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != quote:
        return None
    return start, end


async def resolve_workspace_citations(
    workspace_id: str, locators: list[dict[str, Any]], *, workspace_db: Any, media_db: Any,
) -> dict[str, Any]:
    """Resolve every locator without exposing foreign sources or trusting supplied quotes."""
    if not workspace_db.get_workspace(workspace_id):
        raise LookupError("workspace_not_found")
    sources = {str(row["id"]): row for row in workspace_db.list_workspace_sources(workspace_id)}
    snapshots: dict[int, dict[str, Any] | None] = {}
    evidence = {}
    for locator in locators:
        citation_id = locator["id"]
        source = sources.get(locator["source_id"])
        media_id = source.get("media_id") if source else None
        if type(media_id) is not int or media_id <= 0:
            evidence[citation_id] = {"resolution_error": "citation_source_unavailable"}
            continue
        if media_id not in snapshots:
            try:
                snapshots[media_id] = await load_navigation_sections(media_id, media_db)
            except ValueError:
                snapshots[media_id] = None
        snapshot = snapshots[media_id]
        span = _resolve_quote(locator, snapshot) if snapshot else None
        if not span:
            evidence[citation_id] = {"resolution_error": "citation_source_changed_or_unresolved"}
            continue
        start, end = span
        item = {"text": snapshot["content"][start:end], "source_id": source["id"], "media_id": media_id,
            "start_char": start, "end_char": end, "navigation_version": snapshot["navigation_version"],
            "page_spans": page_spans_for_range(snapshot.get("page_map"), start, end)}
        context = _attribution_context(snapshot, source, start, end)
        if context:
            item["attribution_context"] = context
        evidence[citation_id] = item
    return evidence
