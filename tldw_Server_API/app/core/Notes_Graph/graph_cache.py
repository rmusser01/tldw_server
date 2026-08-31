"""Thread-safe TTL cache for graph query results."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.api.v1.schemas.notes_graph import NoteGraphRequest
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import (
    DEFAULT_SEMANTIC_INDEX_SETTINGS,
)


@dataclass(frozen=True, slots=True)
class SemanticGraphQueryIdentity:
    """Canonical immutable identity for one semantic graph request."""

    focus_note_id: str
    radius: int
    edge_types: tuple[str, ...]
    tag: str | None
    source: str | None
    time_range_start: str | None
    time_range_end: str | None
    time_range_field: str
    semantic_threshold: float
    semantic_top_k: int
    max_nodes: int
    max_edges: int
    max_degree: int
    semantic_candidate_nodes: int
    semantic_candidate_edges: int
    allow_heavy: bool

    def __post_init__(self) -> None:
        if not self.focus_note_id:
            raise ValueError("focus_note_id must not be empty")
        canonical_edge_types = tuple(sorted(set(self.edge_types)))
        if self.edge_types != canonical_edge_types or "semantic" not in self.edge_types:
            raise ValueError("edge_types must be canonical and include semantic")
        if type(self.semantic_threshold) is not float or not math.isfinite(
            self.semantic_threshold
        ) or not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be a finite float from 0 to 1")
        if (
            type(self.semantic_top_k) is not int
            or not 1
            <= self.semantic_top_k
            <= DEFAULT_SEMANTIC_INDEX_SETTINGS.max_query_neighbors
        ):
            raise ValueError("semantic_top_k must be a bounded positive integer")
        for field_name, value, minimum in (
            ("max_nodes", self.max_nodes, 1),
            ("max_edges", self.max_edges, 0),
            ("max_degree", self.max_degree, 1),
            ("semantic_candidate_nodes", self.semantic_candidate_nodes, 0),
            ("semantic_candidate_edges", self.semantic_candidate_edges, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field_name} must be a bounded integer")
        semantic_cap = DEFAULT_SEMANTIC_INDEX_SETTINGS.max_query_neighbors
        if self.semantic_candidate_nodes > semantic_cap:
            raise ValueError("semantic_candidate_nodes exceeds the semantic cap")
        if self.semantic_candidate_edges > semantic_cap:
            raise ValueError("semantic_candidate_edges exceeds the semantic cap")
        if type(self.allow_heavy) is not bool:
            raise TypeError("allow_heavy must be a boolean")

    @classmethod
    def from_request(
        cls,
        request: NoteGraphRequest,
        *,
        semantic_threshold: float,
        semantic_top_k: int,
        max_nodes: int,
        max_edges: int,
        max_degree: int,
        semantic_candidate_nodes: int,
        semantic_candidate_edges: int,
        allow_heavy: bool,
    ) -> SemanticGraphQueryIdentity:
        """Build the complete semantic identity from validated effective inputs."""

        if not request.semantic_requested:
            raise ValueError("Semantic query identity requires semantic edge type")
        if not request.center_note_id:
            raise ValueError("Semantic query identity requires a focus note")
        if (
            request.semantic_threshold is not None
            and request.semantic_threshold != semantic_threshold
        ):
            raise ValueError("effective semantic threshold does not match the request")
        if (
            request.semantic_top_k is not None
            and request.semantic_top_k != semantic_top_k
        ):
            raise ValueError("effective semantic top_k does not match the request")
        time_range = (
            request.time_range.model_dump(mode="json")
            if request.time_range is not None
            else {}
        )
        return cls(
            focus_note_id=request.center_note_id,
            radius=request.radius,
            edge_types=tuple(
                edge_type.value for edge_type in request.resolved_edge_types
            ),
            tag=request.tag,
            source=request.source,
            time_range_start=time_range.get("start"),
            time_range_end=time_range.get("end"),
            time_range_field=request.time_range_field,
            semantic_threshold=semantic_threshold,
            semantic_top_k=semantic_top_k,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_degree=max_degree,
            semantic_candidate_nodes=semantic_candidate_nodes,
            semantic_candidate_edges=semantic_candidate_edges,
            allow_heavy=allow_heavy,
        )

    def as_payload(self) -> dict[str, object]:
        """Return the closed canonical payload used by cache and cursor hashes."""

        return {
            "focus_note_id": self.focus_note_id,
            "radius": self.radius,
            "edge_types": self.edge_types,
            "tag": self.tag,
            "source": self.source,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "time_range_field": self.time_range_field,
            "semantic_threshold": self.semantic_threshold,
            "semantic_top_k": self.semantic_top_k,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_degree": self.max_degree,
            "semantic_candidate_nodes": self.semantic_candidate_nodes,
            "semantic_candidate_edges": self.semantic_candidate_edges,
            "allow_heavy": self.allow_heavy,
        }


def _semantic_revision_payload(
    *,
    dataset_id: str,
    graph_revision: int,
    parser_version: int,
    generation_id: str,
    semantic_index_revision: int,
    configuration_revision: int,
    compatibility_hash: str,
    model_revision: str | None,
    normalization_version: str,
    chunker_version: str,
    query_identity: SemanticGraphQueryIdentity,
) -> dict[str, object]:
    """Return only immutable inputs to a final semantic projection."""

    if not isinstance(query_identity, SemanticGraphQueryIdentity):
        raise TypeError("query_identity must be SemanticGraphQueryIdentity")
    return {
        "dataset_hash": hashlib.sha256(dataset_id.encode()).hexdigest(),
        "graph_revision": graph_revision,
        "parser_version": parser_version,
        "generation_id": generation_id,
        "semantic_index_revision": semantic_index_revision,
        "configuration_revision": configuration_revision,
        "compatibility_hash": compatibility_hash,
        "model_revision": model_revision,
        "normalization_version": normalization_version,
        "chunker_version": chunker_version,
        "query": query_identity.as_payload(),
    }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


class GraphCache:
    """Simple in-memory TTL cache keyed on a hash of query parameters.

    Config via environment variables:
      - ``NOTES_GRAPH_CACHE_TTL`` – seconds before expiry (default 20)
      - ``NOTES_GRAPH_CACHE_MAX_KEYS`` – max cached entries (default 1000)
    """

    def __init__(
        self,
        ttl_seconds: int | None = None,
        max_keys: int | None = None,
    ) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else _env_int("NOTES_GRAPH_CACHE_TTL", 20)
        self._max_keys = max_keys if max_keys is not None else _env_int("NOTES_GRAPH_CACHE_MAX_KEYS", 1000)
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_cache_key(user_id: str, query_params: dict) -> str:
        """SHA-256[:32] of *user_id* + deterministic JSON of *query_params*."""
        raw = user_id + "|" + json.dumps(query_params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def make_revision_key(
        *,
        user_id: str,
        dataset_id: str,
        graph_revision: int,
        parser_version: int,
        query_params: dict,
    ) -> str:
        """Build a cache key that cannot survive a graph-visible mutation."""

        return GraphCache.make_cache_key(
            user_id,
            {
                "dataset_id": dataset_id,
                "graph_revision": graph_revision,
                "parser_version": parser_version,
                "query": query_params,
            },
        )

    @staticmethod
    def make_semantic_revision_key(
        *,
        user_id: str,
        dataset_id: str,
        graph_revision: int,
        parser_version: int,
        generation_id: str,
        semantic_index_revision: int,
        configuration_revision: int,
        compatibility_hash: str,
        model_revision: str | None,
        normalization_version: str,
        chunker_version: str,
        query_identity: SemanticGraphQueryIdentity,
    ) -> str:
        """Build the stable outer key for a final semantic projection."""

        return GraphCache.make_cache_key(
            user_id,
            _semantic_revision_payload(
                dataset_id=dataset_id,
                graph_revision=graph_revision,
                parser_version=parser_version,
                generation_id=generation_id,
                semantic_index_revision=semantic_index_revision,
                configuration_revision=configuration_revision,
                compatibility_hash=compatibility_hash,
                model_revision=model_revision,
                normalization_version=normalization_version,
                chunker_version=chunker_version,
                query_identity=query_identity,
            ),
        )

    @staticmethod
    def make_semantic_cursor_binding(
        *,
        dataset_id: str,
        graph_revision: int,
        parser_version: int,
        generation_id: str,
        semantic_index_revision: int,
        configuration_revision: int,
        compatibility_hash: str,
        model_revision: str | None,
        normalization_version: str,
        chunker_version: str,
        query_identity: SemanticGraphQueryIdentity,
    ) -> str:
        """Hash the immutable semantic request identity carried by cursors."""

        payload = _semantic_revision_payload(
            dataset_id=dataset_id,
            graph_revision=graph_revision,
            parser_version=parser_version,
            generation_id=generation_id,
            semantic_index_revision=semantic_index_revision,
            configuration_revision=configuration_revision,
            compatibility_hash=compatibility_hash,
            model_revision=model_revision,
            normalization_version=normalization_version,
            chunker_version=chunker_version,
            query_identity=query_identity,
        )
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` on miss / expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Insert or replace a cache entry."""
        with self._lock:
            now = time.monotonic()
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (now, value)
            # Evict oldest if over capacity
            while len(self._store) > self._max_keys:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        """Return cache statistics snapshot."""
        with self._lock:
            return {
                "size": len(self._store),
                "max_keys": self._max_keys,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
            }
