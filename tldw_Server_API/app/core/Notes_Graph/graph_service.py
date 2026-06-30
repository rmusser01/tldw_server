"""Core graph service: BFS expansion, derived edges, pruning, pagination."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphEdge,
    GraphLimits,
    GraphNode,
    NoteGraphRequest,
    NoteGraphResponse,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
)
from tldw_Server_API.app.core.Notes_Graph.graph_cache import GraphCache
from tldw_Server_API.app.core.Notes_Graph.wikilink_parser import extract_wikilinks

# ---------------------------------------------------------------------------
# Config constants (env-overridable, matching PRD §11)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").lower()
    if raw in ("0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    return default


NOTES_GRAPH_ENABLED = lambda: _env_bool("NOTES_GRAPH_ENABLED", True)  # noqa: E731
MAX_NODES = lambda: _env_int("NOTES_GRAPH_MAX_NODES", 300)  # noqa: E731
MAX_EDGES = lambda: _env_int("NOTES_GRAPH_MAX_EDGES", 1200)  # noqa: E731
MAX_DEGREE = lambda: _env_int("NOTES_GRAPH_MAX_DEGREE", 40)  # noqa: E731
POPULAR_TAG_CUTOFF = lambda: _env_float("NOTES_GRAPH_POPULAR_TAG_CUTOFF", 0.15)  # noqa: E731
POPULAR_TAG_ABSOLUTE_MIN = lambda: _env_int("NOTES_GRAPH_POPULAR_TAG_ABSOLUTE_MIN", 25)  # noqa: E731

# Radius=2 stricter caps
_R2_MAX_NODES = 200
_R2_MAX_EDGES = 800
_R2_MAX_DEGREE = 20

# Per-type soft caps
_NOTE_CAP = 250
_TAG_CAP = 75
_SOURCE_CAP = 50


# ---------------------------------------------------------------------------
# Metrics helpers (best-effort, no crash on import failure)
# ---------------------------------------------------------------------------

def _metrics_increment(name: str, labels: dict[str, str] | None = None, value: int = 1) -> None:
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
        increment_counter(name, value, labels)
    except Exception as metrics_error:
        logger.debug("Notes graph counter metric emission failed", exc_info=metrics_error)


def _metrics_observe(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
        observe_histogram(name, value, labels)
    except Exception as metrics_error:
        logger.debug("Notes graph histogram metric emission failed", exc_info=metrics_error)


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

def _encode_cursor(layer: int, pos: int, last_id: str, neighbor_pos: int = 0) -> str:
    payload = json.dumps({
        "layer": layer,
        "pos": pos,
        "last_id": last_id,
        "neighbor_pos": neighbor_pos,
    })
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(raw.encode()).decode())
    except Exception as exc:
        raise InputError("Invalid graph cursor") from exc
    if not isinstance(payload, dict):
        raise InputError("Invalid graph cursor")
    for field in ("layer", "pos", "last_id"):
        if field not in payload:
            raise InputError("Invalid graph cursor")
    try:
        payload["layer"] = int(payload["layer"])
        payload["pos"] = int(payload["pos"])
        payload["neighbor_pos"] = int(payload.get("neighbor_pos", 0))
    except (TypeError, ValueError) as exc:
        raise InputError("Invalid graph cursor") from exc
    if payload["layer"] < 0 or payload["pos"] < 0 or payload["neighbor_pos"] < 0:
        raise InputError("Invalid graph cursor")
    if not isinstance(payload["last_id"], str):
        raise InputError("Invalid graph cursor")
    return payload


# ---------------------------------------------------------------------------
# Derived edge ID helpers (deterministic)
# ---------------------------------------------------------------------------

def _wl_edge_id(from_id: str, to_id: str) -> str:
    return f"wl:{from_id[:8]}:{to_id[:8]}"


def _bl_edge_id(from_id: str, to_id: str) -> str:
    return f"bl:{from_id[:8]}:{to_id[:8]}"


def _tm_edge_id(note_id: str, keyword_id: int) -> str:
    return f"tm:{note_id[:8]}:{keyword_id}"


def _sm_edge_id(note_id: str, source_key: str) -> str:
    h = hashlib.sha256(source_key.encode()).hexdigest()[:8]
    return f"sm:{note_id[:8]}:{h}"


def _source_node_id(source: str, external_ref: str | None) -> str:
    if external_ref:
        return f"source:{source}:{external_ref}"
    return f"source:{source}"


# ---------------------------------------------------------------------------
# NoteGraphService
# ---------------------------------------------------------------------------

class NoteGraphService:
    """Stateless-per-request graph service.

    Orchestrates seed resolution → BFS expansion → derived edges → pruning.
    """

    def __init__(
        self,
        *,
        user_id: str,
        db: CharactersRAGDB,
        cache: GraphCache | None = None,
        allow_heavy_limits: bool = False,
    ) -> None:
        self._user_id = user_id
        self._db = db
        self._cache = cache
        self._allow_heavy_limits = allow_heavy_limits

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate_graph(self, req: NoteGraphRequest) -> NoteGraphResponse:
        """Build and return a bounded note graph."""
        t0 = time.monotonic()

        # 1. Resolve effective limits before any graph expansion work.
        eff_max_nodes, eff_max_edges, eff_max_degree, radius_cap_applied = self._resolve_effective_limits(req)

        # 2. Check cache
        if self._cache is not None:
            cache_key = GraphCache.make_cache_key(
                self._user_id,
                {
                    "center": req.center_note_id,
                    "radius": req.radius,
                    "edge_types": [e.value for e in req.edge_types] if req.edge_types else None,
                    "tag": req.tag,
                    "source": req.source,
                    "time_range": req.time_range.model_dump() if req.time_range else None,
                    "time_range_field": req.time_range_field,
                    "max_nodes": eff_max_nodes,
                    "max_edges": eff_max_edges,
                    "max_degree": eff_max_degree,
                    "allow_heavy": req.allow_heavy and self._allow_heavy_limits,
                    "cursor": req.cursor,
                },
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                _metrics_increment("notes_graph_cache_hits_total")
                return cached
            _metrics_increment("notes_graph_cache_misses_total")
        else:
            cache_key = None

        # 3. Determine seed set
        seed_ids = self._determine_seed_set(req, eff_max_nodes)

        # 4. BFS expand — collects note IDs and manual edges
        note_ids, manual_edges, truncated, truncated_by, cursor_info = self._bfs_expand(
            seed_ids, req.radius, eff_max_nodes, eff_max_degree, req,
        )

        # 5. Fetch note data
        note_rows = self._db.get_notes_batch(note_ids, include_deleted=True)
        note_map: dict[str, dict] = {r["id"]: r for r in note_rows}
        # Prune IDs that don't actually exist
        note_ids = [nid for nid in note_ids if nid in note_map]

        # 5b. Validate center note exists
        if req.center_note_id and req.center_note_id not in note_map:
            raise InputError(f"Note {req.center_note_id} not found")

        # 6. Apply time-range filter
        if req.time_range:
            note_ids = self._apply_time_range(note_ids, note_map, req)
        note_ids = self._order_note_ids(note_ids, note_map)
        note_id_set = set(note_ids)

        # 7. Determine which edge types to compute
        wanted = set(req.edge_types) if req.edge_types else set(EdgeType)

        # 8. Compute derived edges
        edges: list[GraphEdge] = []
        tag_nodes: dict[str, GraphNode] = {}
        source_nodes: dict[str, GraphNode] = {}

        # Manual edges
        if EdgeType.manual in wanted:
            for e in manual_edges:
                if e["from_note_id"] in note_id_set and e["to_note_id"] in note_id_set:
                    edges.append(GraphEdge(
                        id=f"e:{e['edge_id']}",
                        source=e["from_note_id"],
                        target=e["to_note_id"],
                        type=EdgeType.manual,
                        directed=bool(e["directed"]),
                        weight=e.get("weight", 1.0),
                    ))

        # Wikilinks + backlinks
        if EdgeType.wikilink in wanted or EdgeType.backlink in wanted:
            wl_edges, bl_edges = self._compute_wikilink_edges(note_ids, note_id_set, note_map, wanted)
            edges.extend(wl_edges)
            edges.extend(bl_edges)

        # Tag membership
        if EdgeType.tag_membership in wanted:
            tm_edges, tag_nodes = self._compute_tag_edges(note_ids)
            edges.extend(tm_edges)

        # Source membership
        if EdgeType.source_membership in wanted:
            sm_edges, source_nodes = self._compute_source_edges(note_ids)
            edges.extend(sm_edges)

        # 9. Build note nodes
        note_node_map: dict[str, GraphNode] = {}
        for nid in note_ids:
            row = note_map.get(nid)
            if not row:
                continue
            note_node_map[nid] = GraphNode(
                id=nid,
                type="note",
                label=row["title"],
                created_at=row.get("created_at"),
                deleted=bool(row.get("deleted", 0)),
                degree=0,
                tag_count=None,
                primary_source_id=None,
            )

        # 10. Compute degree & tag counts
        for edge in edges:
            if edge.source in note_node_map and note_node_map[edge.source].degree is not None:
                note_node_map[edge.source].degree += 1
            if edge.target in note_node_map and note_node_map[edge.target].degree is not None:
                note_node_map[edge.target].degree += 1
            if edge.source in tag_nodes:
                tag_nodes[edge.source].degree = (tag_nodes[edge.source].degree or 0) + 1
            if edge.target in tag_nodes:
                tag_nodes[edge.target].degree = (tag_nodes[edge.target].degree or 0) + 1
            if edge.source in source_nodes:
                source_nodes[edge.source].degree = (source_nodes[edge.source].degree or 0) + 1
            if edge.target in source_nodes:
                source_nodes[edge.target].degree = (source_nodes[edge.target].degree or 0) + 1

        # Tag counts per note
        if tag_nodes:
            tag_edge_note_counts: dict[str, int] = {}
            for edge in edges:
                if edge.type == EdgeType.tag_membership and edge.source in note_node_map:
                    tag_edge_note_counts[edge.source] = tag_edge_note_counts.get(edge.source, 0) + 1
            for nid, cnt in tag_edge_note_counts.items():
                note_node_map[nid].tag_count = cnt

        # Source info for notes
        if source_nodes:
            for edge in edges:
                if edge.type == EdgeType.source_membership and edge.source in note_node_map:
                    if note_node_map[edge.source].primary_source_id is None:
                        note_node_map[edge.source].primary_source_id = edge.target

        # 11. Pruning
        edges = self._order_edges(edges)
        all_nodes: list[GraphNode] = list(note_node_map.values()) + list(tag_nodes.values()) + list(source_nodes.values())
        all_nodes, edges, truncated, truncated_by = self._apply_pruning(
            all_nodes, edges, eff_max_nodes, eff_max_edges, eff_max_degree,
            truncated, truncated_by,
        )

        # 12. Build cursor
        cursor_str = None
        has_more = False
        if cursor_info and cursor_info.get("has_more"):
            cursor_str = _encode_cursor(
                cursor_info["layer"],
                cursor_info["pos"],
                cursor_info["last_id"],
                cursor_info.get("neighbor_pos", 0),
            )
            has_more = True

        limits = GraphLimits(
            max_nodes=eff_max_nodes,
            max_edges=eff_max_edges,
            max_degree=eff_max_degree,
        )

        response = NoteGraphResponse(
            nodes=all_nodes,
            edges=edges,
            truncated=truncated,
            truncated_by=truncated_by,
            has_more=has_more,
            cursor=cursor_str,
            limits=limits,
            radius_cap_applied=radius_cap_applied,
        )

        # 13. Cache & metrics
        elapsed = time.monotonic() - t0
        _metrics_observe("notes_graph_generation_duration_seconds", elapsed)
        note_count = sum(1 for n in all_nodes if n.type == "note")
        tag_count = sum(1 for n in all_nodes if n.type == "tag")
        source_count = sum(1 for n in all_nodes if n.type == "source")
        if note_count:
            _metrics_increment("notes_graph_nodes_total", {"type": "note"}, value=note_count)
        if tag_count:
            _metrics_increment("notes_graph_nodes_total", {"type": "tag"}, value=tag_count)
        if source_count:
            _metrics_increment("notes_graph_nodes_total", {"type": "source"}, value=source_count)
        for reason in truncated_by:
            _metrics_increment("notes_graph_truncation_total", {"reason": reason})

        if self._cache is not None and cache_key:
            self._cache.put(cache_key, response)

        logger.debug(
            "Graph generated: {} notes, {} tags, {} sources, {} edges in {:.3f}s",
            note_count, tag_count, source_count, len(edges), elapsed,
        )
        return response

    # ------------------------------------------------------------------
    # Limit resolution
    # ------------------------------------------------------------------

    def _resolve_effective_limits(self, req: NoteGraphRequest) -> tuple[int, int, int, bool]:
        """Clamp caller-requested limits before DB traversal begins."""
        default_max_nodes = MAX_NODES()
        default_max_edges = MAX_EDGES()
        default_max_degree = MAX_DEGREE()

        heavy_allowed = bool(req.allow_heavy and self._allow_heavy_limits)
        hard_max_nodes = default_max_nodes * 2 if heavy_allowed else default_max_nodes
        hard_max_edges = default_max_edges * 2 if heavy_allowed else default_max_edges
        hard_max_degree = default_max_degree * 2 if heavy_allowed else default_max_degree

        requested_max_nodes = req.max_nodes if req.max_nodes is not None else default_max_nodes
        requested_max_edges = req.max_edges if req.max_edges is not None else default_max_edges
        requested_max_degree = req.max_degree if req.max_degree is not None else default_max_degree

        eff_max_nodes = min(requested_max_nodes, hard_max_nodes)
        eff_max_edges = min(requested_max_edges, hard_max_edges)
        eff_max_degree = min(requested_max_degree, hard_max_degree)

        radius_cap_applied = False
        if req.radius == 2 and not heavy_allowed:
            if eff_max_nodes > _R2_MAX_NODES:
                eff_max_nodes = _R2_MAX_NODES
                radius_cap_applied = True
            if eff_max_edges > _R2_MAX_EDGES:
                eff_max_edges = _R2_MAX_EDGES
                radius_cap_applied = True
            if eff_max_degree > _R2_MAX_DEGREE:
                eff_max_degree = _R2_MAX_DEGREE
                radius_cap_applied = True

        return eff_max_nodes, eff_max_edges, eff_max_degree, radius_cap_applied

    # ------------------------------------------------------------------
    # Seed set
    # ------------------------------------------------------------------

    def _determine_seed_set(self, req: NoteGraphRequest, max_nodes: int) -> list[str]:
        """Resolve initial seed note IDs for BFS."""
        if req.center_note_id:
            return [req.center_note_id]

        if req.tag:
            return self._db.get_note_ids_by_tag_for_graph(
                req.tag,
                include_deleted=True,
                limit=max_nodes,
            )

        if req.source:
            return self._db.get_note_ids_by_source_for_graph(
                req.source,
                include_deleted=True,
                limit=max_nodes,
            )

        # Seedless: full graph if small enough
        total = self._db.count_user_notes(include_deleted=True)
        if total == 0:
            return []
        if total <= max_nodes:
            return self._db.get_all_note_ids_for_graph(include_deleted=True, limit=max_nodes)

        if req.allow_heavy and self._allow_heavy_limits:
            return self._db.get_all_note_ids_for_graph(include_deleted=True, limit=max_nodes)

        raise InputError(  # noqa: TRY003
            f"Too many notes ({total}) for seedless graph. "
            f"Provide center_note_id, tag, or source filter, or request allow_heavy with elevated graph permission."
        )

    # ------------------------------------------------------------------
    # BFS expansion
    # ------------------------------------------------------------------

    def _bfs_expand(
        self,
        seed_ids: list[str],
        radius: int,
        max_nodes: int,
        max_degree: int,
        req: NoteGraphRequest,
    ) -> tuple[list[str], list[dict], bool, list[str], dict | None]:
        """Layer-by-layer BFS from seeds, collecting note IDs and manual edges."""
        visited: set[str] = set()
        visited_order: list[str] = []
        all_edges: list[dict] = []
        edge_ids_seen: set[str] = set()
        truncated = False
        truncated_by: list[str] = []
        cursor_info: dict | None = None

        # Parse cursor for resume
        cur = _decode_cursor(req.cursor)
        if cur and radius > 1:
            logger.warning("Cursor pagination only reliable for radius=1; ignoring cursor")
            cur = None
        start_layer = cur["layer"] if cur else 0
        start_pos = cur["pos"] if cur else 0
        start_neighbor_pos = cur["neighbor_pos"] if cur else 0

        frontier: deque[str] = deque()

        # Initial seeds
        for sid in seed_ids:
            if len(visited) >= max_nodes:
                truncated = True
                if "max_nodes" not in truncated_by:
                    truncated_by.append("max_nodes")
                break
            if sid not in visited:
                visited.add(sid)
                visited_order.append(sid)
                frontier.append(sid)

        for layer in range(radius):
            if not frontier:
                break
            next_frontier: deque[str] = deque()
            layer_nodes = list(frontier)
            frontier.clear()

            # Fetch manual edges for all nodes in this layer
            edges = self._db.get_manual_edges_for_notes(self._user_id, layer_nodes)
            # Also compute wikilink targets from content
            note_data = self._db.get_notes_batch(layer_nodes, include_deleted=True)
            content_map = {r["id"]: r.get("content", "") for r in note_data}

            for idx, nid in enumerate(layer_nodes):
                if layer < start_layer or (layer == start_layer and idx < start_pos):
                    continue

                neighbors: list[str] = []

                # Manual edge neighbors
                for e in edges:
                    eid = e["edge_id"]
                    if eid in edge_ids_seen:
                        continue
                    if e["from_note_id"] == nid:
                        neighbors.append(e["to_note_id"])
                        edge_ids_seen.add(eid)
                        all_edges.append(e)
                    elif e["to_note_id"] == nid:
                        neighbors.append(e["from_note_id"])
                        edge_ids_seen.add(eid)
                        all_edges.append(e)

                # Wikilink target neighbors
                content = content_map.get(nid, "")
                if content:
                    wl_refs = extract_wikilinks(content)
                    for ref in wl_refs:
                        neighbors.append(ref.target_note_id)

                # Sort neighbors: deterministic
                neighbors = sorted(set(neighbors))
                neighbor_start = start_neighbor_pos if layer == start_layer and idx == start_pos else 0

                # Enforce max_degree per node
                if len(neighbors) > max_degree:
                    neighbors = neighbors[:max_degree]
                    truncated = True
                    if "max_degree" not in truncated_by:
                        truncated_by.append("max_degree")

                for neighbor_idx, nb in enumerate(neighbors[neighbor_start:], start=neighbor_start):
                    if nb in visited:
                        continue
                    if len(visited) >= max_nodes:
                        truncated = True
                        if "max_nodes" not in truncated_by:
                            truncated_by.append("max_nodes")
                        cursor_info = {
                            "layer": layer,
                            "pos": idx,
                            "last_id": nid,
                            "neighbor_pos": neighbor_idx,
                            "has_more": True,
                        }
                        break
                    visited.add(nb)
                    visited_order.append(nb)
                    next_frontier.append(nb)

                if truncated and "max_nodes" in truncated_by:
                    break

            frontier = next_frontier

        return visited_order, all_edges, truncated, truncated_by, cursor_info

    # ------------------------------------------------------------------
    # Derived edges
    # ------------------------------------------------------------------

    def _compute_wikilink_edges(
        self,
        note_ids: list[str],
        note_id_set: set[str],
        note_map: dict[str, dict],
        wanted: set[EdgeType],
    ) -> tuple[list[GraphEdge], list[GraphEdge]]:
        """Compute wikilink and backlink edges within the graph."""
        wl_edges: list[GraphEdge] = []
        bl_edges: list[GraphEdge] = []
        seen_wl: set[str] = set()

        for nid in note_ids:
            row = note_map.get(nid)
            if not row:
                continue
            content = row.get("content", "")
            if not content:
                continue
            refs = extract_wikilinks(content)
            for ref in refs:
                target = ref.target_note_id
                if target not in note_id_set:
                    continue
                if target == nid:
                    continue

                if EdgeType.wikilink in wanted:
                    eid = _wl_edge_id(nid, target)
                    if eid not in seen_wl:
                        seen_wl.add(eid)
                        wl_edges.append(GraphEdge(
                            id=eid,
                            source=nid,
                            target=target,
                            type=EdgeType.wikilink,
                            directed=True,
                            weight=1.0,
                        ))

                if EdgeType.backlink in wanted:
                    bl_eid = _bl_edge_id(target, nid)
                    if bl_eid not in seen_wl:
                        seen_wl.add(bl_eid)
                        bl_edges.append(GraphEdge(
                            id=bl_eid,
                            source=target,
                            target=nid,
                            type=EdgeType.backlink,
                            directed=True,
                            weight=1.0,
                        ))

        return wl_edges, bl_edges

    def _compute_tag_edges(
        self, note_ids: list[str],
    ) -> tuple[list[GraphEdge], dict[str, GraphNode]]:
        """Compute tag_membership edges and tag nodes."""
        tag_data = self._db.get_note_tag_edges(list(note_ids))
        if not tag_data:
            return [], {}

        # Popularity cutoff
        tag_counts = self._db.count_notes_per_tag()
        total_notes = self._db.count_user_notes(include_deleted=False)
        cutoff_relative = POPULAR_TAG_CUTOFF()
        cutoff_absolute = POPULAR_TAG_ABSOLUTE_MIN()

        popular_kw_ids: set[int] = set()
        if total_notes > 0:
            for kw_id, cnt in tag_counts.items():
                ratio = cnt / total_notes
                if ratio > cutoff_relative and cnt >= cutoff_absolute:
                    popular_kw_ids.add(kw_id)

        edges: list[GraphEdge] = []
        tag_nodes: dict[str, GraphNode] = {}

        for row in tag_data:
            kw_id = row["keyword_id"]
            if kw_id in popular_kw_ids:
                continue
            note_id = row["note_id"]
            kw_label = row["keyword"]
            tag_nid = f"tag:{kw_label}"

            if tag_nid not in tag_nodes:
                tag_nodes[tag_nid] = GraphNode(
                    id=tag_nid,
                    type="tag",
                    label=kw_label,
                    degree=0,
                )

            edges.append(GraphEdge(
                id=_tm_edge_id(note_id, kw_id),
                source=note_id,
                target=tag_nid,
                type=EdgeType.tag_membership,
                directed=False,
                weight=1.0,
            ))

        return edges, tag_nodes

    def _compute_source_edges(
        self, note_ids: list[str],
    ) -> tuple[list[GraphEdge], dict[str, GraphNode]]:
        """Compute source_membership edges and source nodes."""
        source_data = self._db.get_note_source_info(list(note_ids))
        if not source_data:
            return [], {}

        edges: list[GraphEdge] = []
        source_nodes: dict[str, GraphNode] = {}

        for row in source_data:
            note_id = row["note_id"]
            src = row["source"]
            ext_ref = row.get("external_ref")
            src_nid = _source_node_id(src, ext_ref)
            src_label = f"{src}: {ext_ref}" if ext_ref else src

            if src_nid not in source_nodes:
                source_nodes[src_nid] = GraphNode(
                    id=src_nid,
                    type="source",
                    label=src_label,
                    degree=0,
                )

            source_key = f"{src}:{ext_ref}" if ext_ref else src
            edges.append(GraphEdge(
                id=_sm_edge_id(note_id, source_key),
                source=note_id,
                target=src_nid,
                type=EdgeType.source_membership,
                directed=False,
                weight=1.0,
            ))

        return edges, source_nodes

    # ------------------------------------------------------------------
    # Time-range filter
    # ------------------------------------------------------------------

    def _apply_time_range(
        self,
        note_ids: list[str],
        note_map: dict[str, dict],
        req: NoteGraphRequest,
    ) -> list[str]:
        """Filter notes by time range. Maps updated_at → last_modified."""
        tr = req.time_range
        if not tr:
            return note_ids

        # Map field name
        field = "last_modified" if req.time_range_field == "updated_at" else "created_at"

        filtered: list[str] = []
        start_naive = self._to_utc_naive(tr.start) if tr.start else None
        end_naive = self._to_utc_naive(tr.end) if tr.end else None
        for nid in note_ids:
            row = note_map.get(nid)
            if not row:
                continue
            val = row.get(field)
            if val is None:
                filtered.append(nid)
                continue
            ts_naive = self._to_utc_naive(val)
            if ts_naive is None:
                filtered.append(nid)
                continue

            if start_naive:
                if ts_naive < start_naive:
                    continue
            if end_naive:
                if ts_naive > end_naive:
                    continue
            filtered.append(nid)
        return filtered

    @staticmethod
    def _to_utc_naive(value: str | datetime | None) -> datetime | None:
        """Normalize datetimes to UTC-naive values for consistent comparisons."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def _order_note_ids(self, note_ids: list[str], note_map: dict[str, dict]) -> list[str]:
        """Order note IDs by updated timestamp descending, then ID ascending."""
        def _sort_key(note_id: str) -> tuple[float, str]:
            row = note_map.get(note_id, {})
            ts = self._to_utc_naive(row.get("last_modified"))
            sort_ts = ts.replace(tzinfo=timezone.utc).timestamp() if ts is not None else 0.0
            return (-sort_ts, note_id)

        return sorted(dict.fromkeys(note_ids), key=_sort_key)

    @staticmethod
    def _order_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
        """Order edges deterministically before order-sensitive pruning."""
        return sorted(
            edges,
            key=lambda edge: (edge.type.value, edge.id, edge.source, edge.target),
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _apply_pruning(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        max_nodes: int,
        max_edges: int,
        max_degree: int,
        truncated: bool,
        truncated_by: list[str],
    ) -> tuple[list[GraphNode], list[GraphEdge], bool, list[str]]:
        """Apply per-type caps and global limits."""
        truncated_by = list(truncated_by)  # copy

        # Per-type node caps
        note_nodes = [n for n in nodes if n.type == "note"]
        tag_nodes = [n for n in nodes if n.type == "tag"]
        source_nodes = [n for n in nodes if n.type == "source"]

        if len(note_nodes) > _NOTE_CAP:
            note_nodes = note_nodes[:_NOTE_CAP]
            truncated = True
            if "max_nodes" not in truncated_by:
                truncated_by.append("max_nodes")

        if len(tag_nodes) > _TAG_CAP:
            tag_nodes = tag_nodes[:_TAG_CAP]
            truncated = True
            if "max_nodes" not in truncated_by:
                truncated_by.append("max_nodes")

        if len(source_nodes) > _SOURCE_CAP:
            source_nodes = source_nodes[:_SOURCE_CAP]
            truncated = True
            if "max_nodes" not in truncated_by:
                truncated_by.append("max_nodes")

        pruned_nodes = note_nodes + tag_nodes + source_nodes

        # Global max_nodes
        if len(pruned_nodes) > max_nodes:
            pruned_nodes = pruned_nodes[:max_nodes]
            truncated = True
            if "max_nodes" not in truncated_by:
                truncated_by.append("max_nodes")

        # Filter edges to only reference present nodes
        node_id_set = {n.id for n in pruned_nodes}
        edges = [e for e in edges if e.source in node_id_set and e.target in node_id_set]

        # Global max_edges: prune in order: tag/source → wikilinks → manual
        if len(edges) > max_edges:
            # Sort by priority (manual last = highest priority to keep)
            _type_priority = {
                EdgeType.tag_membership: 0,
                EdgeType.source_membership: 0,
                EdgeType.backlink: 1,
                EdgeType.wikilink: 1,
                EdgeType.manual: 2,
            }
            edges.sort(key=lambda e: _type_priority.get(e.type, 0), reverse=True)
            edges = edges[:max_edges]
            truncated = True
            if "max_edges" not in truncated_by:
                truncated_by.append("max_edges")

        # Global max_degree enforcement
        degree_count: dict[str, int] = {}
        kept_edges: list[GraphEdge] = []
        for e in edges:
            ds = degree_count.get(e.source, 0)
            dt = degree_count.get(e.target, 0)
            if ds >= max_degree or dt >= max_degree:
                truncated = True
                if "max_degree" not in truncated_by:
                    truncated_by.append("max_degree")
                continue
            degree_count[e.source] = ds + 1
            degree_count[e.target] = dt + 1
            kept_edges.append(e)
        edges = kept_edges

        return pruned_nodes, edges, truncated, truncated_by
