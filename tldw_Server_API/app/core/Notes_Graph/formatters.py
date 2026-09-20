"""Output formatters for graph responses."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.api.v1.schemas.notes_graph import NoteGraphResponse


def to_cytoscape(graph: NoteGraphResponse) -> dict[str, Any]:
    """Convert a :class:`NoteGraphResponse` into Cytoscape.js JSON.

    Returns::

        {
            "elements": {
                "nodes": [{"data": {"id": ..., "type": ..., ...}}, ...],
                "edges": [{"data": {"id": ..., "source": ..., "target": ..., ...}}, ...]
            },
            "truncated": ...,
            "truncated_by": [...],
            "has_more": ...,
            "cursor": ...,
            "limits": {...},
            "radius_cap_applied": ...
        }
    """
    cy_nodes = []
    for node in graph.nodes:
        data: dict[str, Any] = {
            "id": node.id,
            "type": node.type,
            "label": node.label,
        }
        if node.created_at is not None:
            data["created_at"] = node.created_at.isoformat() if hasattr(node.created_at, "isoformat") else str(node.created_at)
        if node.deleted is not None:
            data["deleted"] = node.deleted
        if node.degree is not None:
            data["degree"] = node.degree
        if node.tag_count is not None:
            data["tag_count"] = node.tag_count
        if node.primary_source_id is not None:
            data["primary_source_id"] = node.primary_source_id
        cy_nodes.append({"data": data})

    cy_edges = []
    for edge in graph.edges:
        data = {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "type": edge.type.value,
            "directed": edge.directed,
        }
        if edge.weight is not None:
            data["weight"] = edge.weight
        if edge.label is not None:
            data["label"] = edge.label
        cy_edges.append({"data": data})

    return {
        "elements": {
            "nodes": cy_nodes,
            "edges": cy_edges,
        },
        "truncated": graph.truncated,
        "truncated_by": graph.truncated_by,
        "has_more": graph.has_more,
        "cursor": graph.cursor,
        "limits": graph.limits.model_dump(),
        "radius_cap_applied": graph.radius_cap_applied,
        "suggestions_authorized": graph.suggestions_authorized,
    }
