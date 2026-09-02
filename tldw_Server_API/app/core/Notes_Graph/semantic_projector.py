"""Async, authority-bound semantic projection for the synchronous Notes graph."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.notes_graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NoteGraphRequest,
    NoteGraphResponse,
    SemanticEdgeEvidence,
    SemanticExcerpt,
    SemanticExcerptPair,
    SemanticGraphStatus,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticProjectionChunk,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

from .graph_cache import GraphCache, SemanticGraphQueryIdentity
from .graph_service import (
    SemanticGraphCandidateResult,
    _decode_cursor,
    bind_semantic_cursor,
)
from .semantic_capabilities import (
    semantic_capability_binding_matches,
    semantic_provider_label,
)
from .semantic_content import reconstruct_semantic_chunk
from .semantic_observability import (
    record_semantic_denial,
    record_semantic_failure,
    record_semantic_health_metrics,
    record_semantic_query_metrics,
)
from .semantic_scoring import (
    SemanticChunkCandidate,
    SemanticNoteMatch,
    qualitative_similarity_band,
    rank_semantic_note_matches,
)
from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings
from .semantic_vectors import SemanticVector, SemanticVectorMatch, create_semantic_vector_store

_DEFAULT_THRESHOLD = 0.75
_DEFAULT_TOP_K = 10
_MAX_EVIDENCE_PAIRS = 3
_MAX_EXCERPT_CODE_POINTS = 480
_MAX_EDGE_EVIDENCE_CODE_POINTS = 2_880
_MAX_RESPONSE_EVIDENCE_BYTES = 256 * 1024


class SemanticProjectionError(RuntimeError):
    """Stable, content-free semantic projection error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class _SemanticBinding:
    config: Any
    generation: Any
    status: SemanticGraphStatus
    provider_label: str
    model_label: str
    signature: tuple[object, ...]


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _semantic_pair(edge: GraphEdge) -> tuple[str, str]:
    return tuple(sorted((edge.source, edge.target)))  # type: ignore[return-value]


def _edge_priority(edge: GraphEdge) -> tuple[int, float, str, str, str]:
    priorities = {
        EdgeType.manual: 0,
        EdgeType.wikilink: 1,
        EdgeType.backlink: 1,
        EdgeType.semantic: 2,
        EdgeType.tag_membership: 3,
        EdgeType.source_membership: 3,
    }
    score = float(edge.weight or 0.0) if edge.type is EdgeType.semantic else 0.0
    return (
        priorities.get(edge.type, 4),
        -score,
        edge.id,
        edge.source,
        edge.target,
    )


def compose_semantic_graph(
    *,
    public_graph: NoteGraphResponse,
    candidate_nodes: Sequence[GraphNode],
    candidate_edges: Sequence[GraphEdge],
    semantic_nodes: Sequence[GraphNode],
    semantic_edges: Sequence[GraphEdge],
    focus_note_id: str,
    authoritative_manual_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> NoteGraphResponse:
    """Compose candidates under public caps and the approved precedence order."""

    node_map = {node.id: node for node in (*public_graph.nodes, *candidate_nodes, *semantic_nodes)}
    if focus_note_id not in node_map:
        return public_graph

    manual_pairs = authoritative_manual_pairs | {
        _semantic_pair(edge) for edge in candidate_edges if edge.type is EdgeType.manual
    }
    semantic = [
        edge for edge in semantic_edges if edge.type is EdgeType.semantic and _semantic_pair(edge) not in manual_pairs
    ]
    all_edges = sorted(
        (
            *(edge for edge in candidate_edges if edge.type is not EdgeType.semantic),
            *semantic,
        ),
        key=_edge_priority,
    )

    max_nodes = public_graph.limits.max_nodes
    max_edges = public_graph.limits.max_edges
    max_degree = public_graph.limits.max_degree
    selected_node_ids: list[str] = [focus_note_id]
    selected_node_set = {focus_note_id}
    selected_edges: list[GraphEdge] = []
    degrees: Counter[str] = Counter()
    truncated_by = list(public_graph.truncated_by)
    truncated = public_graph.truncated

    for edge in all_edges:
        if len(selected_edges) >= max_edges:
            truncated = True
            if "max_edges" not in truncated_by:
                truncated_by.append("max_edges")
            break
        if edge.source not in node_map or edge.target not in node_map:
            continue
        if degrees[edge.source] >= max_degree or degrees[edge.target] >= max_degree:
            truncated = True
            if "max_degree" not in truncated_by:
                truncated_by.append("max_degree")
            continue
        missing = tuple(node_id for node_id in (edge.source, edge.target) if node_id not in selected_node_set)
        if len(selected_node_set) + len(missing) > max_nodes:
            truncated = True
            if "max_nodes" not in truncated_by:
                truncated_by.append("max_nodes")
            continue
        for node_id in missing:
            selected_node_set.add(node_id)
            selected_node_ids.append(node_id)
        selected_edges.append(edge)
        degrees[edge.source] += 1
        degrees[edge.target] += 1

    candidate_node_ids = {node.id for node in candidate_nodes}
    semantic_only_node_ids = {node.id for node in semantic_nodes if node.id not in candidate_node_ids}
    node_type_priority = {"note": 0, "tag": 1, "source": 2}
    remaining_nodes = sorted(
        (
            node
            for node_id, node in node_map.items()
            if node_id not in selected_node_set and node_id not in semantic_only_node_ids
        ),
        key=lambda node: (node_type_priority.get(node.type, 3), node.id),
    )
    for node in remaining_nodes:
        if len(selected_node_ids) >= max_nodes:
            break
        selected_node_ids.append(node.id)
        selected_node_set.add(node.id)

    nodes = [node_map[node_id].model_copy(update={"degree": degrees[node_id]}) for node_id in selected_node_ids]
    return public_graph.model_copy(
        update={
            "nodes": nodes,
            "edges": selected_edges,
            "truncated": truncated,
            "truncated_by": truncated_by,
        }
    )


def bound_semantic_evidence(
    edges: Sequence[GraphEdge],
    *,
    byte_cap: int = _MAX_RESPONSE_EVIDENCE_BYTES,
) -> tuple[tuple[GraphEdge, ...], bool]:
    """Apply a stable response-wide UTF-8 cap without dropping semantic edges."""

    if isinstance(byte_cap, bool) or not isinstance(byte_cap, int) or byte_cap < 0:
        raise ValueError("semantic evidence byte cap must be non-negative")
    used = 0
    truncated = False
    exhausted = False
    bounded: list[GraphEdge] = []
    for edge in edges:
        evidence = edge.evidence
        if evidence is None:
            bounded.append(edge)
            continue
        size = len(evidence.model_dump_json().encode("utf-8"))
        if exhausted or used + size > byte_cap:
            exhausted = True
            truncated = True
            payload = edge.model_dump(mode="python")
            payload["evidence"] = None
            payload["evidence_omitted"] = "response_byte_cap"
            bounded.append(GraphEdge.model_validate(payload))
            continue
        used += size
        bounded.append(edge)
    return tuple(bounded), truncated


def _rank_bounded_candidates(
    candidates: Sequence[SemanticChunkCandidate],
    *,
    threshold: float,
    top_k: int,
    current_chunk_ids: set[str] | frozenset[str],
) -> tuple[tuple[SemanticNoteMatch, ...], bool]:
    """Rank all bounded candidates, then report only actual note-level clipping."""

    ranked = rank_semantic_note_matches(
        candidates,
        threshold=threshold,
        top_k=max(1, len(candidates)),
        current_chunk_ids=current_chunk_ids,
    )
    return ranked[:top_k], len(ranked) > top_k


async def build_projection_vector_store(
    *,
    db: Any,
    owner_user_id: str,
    backend_name: str,
    settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
) -> Any:
    """Lazily construct the configured vector-only read facade."""

    chroma_manager = None
    postgres_backend = None
    if backend_name == "chromadb":
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
        from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

        chroma_manager = await asyncio.to_thread(
            ChromaDBManager,
            user_id=owner_user_id,
            user_embedding_config={
                "USER_DB_BASE_DIR": str(DatabasePaths.get_user_db_base_dir()),
            },
        )
    elif backend_name == "pgvector":
        postgres_backend = getattr(db, "_backend", None)
    return await create_semantic_vector_store(
        backend_name,
        authority=db.note_semantic_store,
        chroma_manager=chroma_manager,
        postgres_backend=postgres_backend,
        settings=settings,
    )


class SemanticGraphProjector:
    """Await vector I/O and merge only revalidated semantic relationships."""

    def __init__(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        db: Any,
        graph_service: Any,
        cache: GraphCache | None,
        vector_store_factory: Callable[[], Any | Awaitable[Any]],
        capability_resolver: Callable[[], Any],
        settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
    ) -> None:
        self._owner_user_id = str(owner_user_id)
        self._dataset_id = str(dataset_id)
        self._db = db
        self._store = db.note_semantic_store
        self._graph_service = graph_service
        self._cache = cache
        self._vector_store_factory = vector_store_factory
        self._capability_resolver = capability_resolver
        self._settings = settings

    def _effective_controls(self, request: NoteGraphRequest) -> tuple[float, int]:
        threshold = float(request.semantic_threshold) if request.semantic_threshold is not None else _DEFAULT_THRESHOLD
        top_k = min(
            request.semantic_top_k or _DEFAULT_TOP_K,
            self._settings.max_query_neighbors,
        )
        return threshold, top_k

    def _status(
        self,
        *,
        available: bool,
        state: str,
        detail_reason: str | None,
        config: Any | None,
        integrity: Any | None,
        effective_threshold: float,
        effective_top_k: int,
    ) -> SemanticGraphStatus:
        try:
            active_notes = max(0, int(self._db.count_user_notes(include_deleted=False)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            active_notes = 0
        return SemanticGraphStatus(
            available=available,
            state=state,
            detail_reason=detail_reason,
            generation_id=(None if config is None else getattr(config, "active_generation_id", None)),
            semantic_index_revision=(None if config is None else int(config.semantic_index_revision)),
            configuration_revision=(None if config is None else int(config.configuration_revision)),
            active_notes=active_notes,
            indexed_notes=(0 if integrity is None else int(integrity.indexed_note_count)),
            dirty_notes=(0 if integrity is None else int(integrity.pending_note_count)),
            excluded_notes=(0 if integrity is None else int(integrity.excluded_note_count)),
            failed_notes=(0 if integrity is None else int(integrity.failed_note_count)),
            effective_top_k=effective_top_k,
            effective_threshold=effective_threshold,
            max_top_k=self._settings.max_query_neighbors,
            max_admission_nodes=self._settings.max_query_neighbors,
            max_admission_edges=self._settings.max_query_neighbors,
            max_evidence_pairs=_MAX_EVIDENCE_PAIRS,
            max_excerpt_code_points=_MAX_EXCERPT_CODE_POINTS,
            max_edge_evidence_code_points=_MAX_EDGE_EVIDENCE_CODE_POINTS,
            max_response_evidence_bytes=_MAX_RESPONSE_EVIDENCE_BYTES,
        )

    async def _resolve_binding(
        self,
        *,
        effective_threshold: float,
        effective_top_k: int,
    ) -> _SemanticBinding | SemanticGraphStatus:
        if not self._settings.indexing_enabled:
            record_semantic_denial("kill_switch")
            return self._status(
                available=False,
                state="unavailable",
                detail_reason="notes_semantic_indexing_disabled",
                config=None,
                integrity=None,
                effective_threshold=effective_threshold,
                effective_top_k=effective_top_k,
            )
        config = await asyncio.to_thread(self._store.get_configuration, self._dataset_id)
        if config is None or _enum_value(config.desired_state) != "enabled":
            record_semantic_denial("configuration")
            return self._status(
                available=False,
                state="off",
                detail_reason="disabled",
                config=config,
                integrity=None,
                effective_threshold=effective_threshold,
                effective_top_k=effective_top_k,
            )
        generation_id = getattr(config, "active_generation_id", None)
        if not generation_id:
            record_semantic_denial("configuration")
            return self._status(
                available=False,
                state="preparing",
                detail_reason="active_generation_required",
                config=config,
                integrity=None,
                effective_threshold=effective_threshold,
                effective_top_k=effective_top_k,
            )
        generation = await asyncio.to_thread(
            self._store.get_generation,
            self._dataset_id,
            generation_id,
        )
        if not self._binding_is_current(config, generation):
            record_semantic_denial("configuration")
            return self._status(
                available=False,
                state="unavailable",
                detail_reason="configuration_stale",
                config=config,
                integrity=None,
                effective_threshold=effective_threshold,
                effective_top_k=effective_top_k,
            )

        capabilities = None
        try:
            capabilities = await asyncio.to_thread(self._capability_resolver)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Notes semantic capability projection failed")
        capability_hash = getattr(capabilities, "compatibility_hash", None)
        if not semantic_capability_binding_matches(
            config.compatibility_hash,
            capability_hash,
        ):
            record_semantic_denial("configuration")
            return self._status(
                available=False,
                state="unavailable",
                detail_reason="configuration_stale",
                config=config,
                integrity=None,
                effective_threshold=effective_threshold,
                effective_top_k=effective_top_k,
            )

        integrity = await asyncio.to_thread(
            self._store.get_generation_integrity,
            self._dataset_id,
            generation_id,
        )
        if int(integrity.failed_note_count) or int(integrity.excluded_note_count):
            state = "needs_attention"
        elif int(integrity.pending_note_count):
            state = "updating"
        else:
            state = "ready"
        persisted_capability_revision = getattr(config, "capability_revision", None)
        current_capability_revision = getattr(capabilities, "capability_revision", None)
        configuration_stale = bool(
            persisted_capability_revision and current_capability_revision != persisted_capability_revision
        )
        maintenance_reason = (
            "stale_configuration"
            if configuration_stale
            else getattr(capabilities, "unavailable_reason", None)
            if capabilities is not None and not bool(getattr(capabilities, "indexing_available", True))
            else None
        )
        if configuration_stale:
            state = "needs_attention"
            record_semantic_denial("configuration")
        elif maintenance_reason is not None:
            record_semantic_denial("capability")
        record_semantic_health_metrics(
            backend=str(getattr(config, "vector_backend", "unavailable")),
            counts={
                "indexed": int(integrity.indexed_note_count),
                "excluded": int(integrity.excluded_note_count),
                "failed": int(integrity.failed_note_count),
                "dirty": int(integrity.pending_note_count),
                "pending": int(integrity.pending_note_count),
            },
            stale_generations=1 if configuration_stale else 0,
        )
        status = self._status(
            available=True,
            state=state,
            detail_reason=maintenance_reason,
            config=config,
            integrity=integrity,
            effective_threshold=effective_threshold,
            effective_top_k=effective_top_k,
        )
        provider_label = semantic_provider_label(getattr(config, "provider", None))
        model_label = str(getattr(config, "model", None) or "unavailable")
        return _SemanticBinding(
            config=config,
            generation=generation,
            status=status,
            provider_label=provider_label,
            model_label=model_label,
            signature=self._semantic_binding_signature(config, generation),
        )

    def _binding_is_current(self, config: Any, generation: Any | None) -> bool:
        if generation is None:
            return False
        return bool(
            str(getattr(config, "owner_user_id", "")) == self._owner_user_id
            and str(getattr(config, "dataset_id", "")) == self._dataset_id
            and str(getattr(self._store, "owner_user_id", "")) == self._owner_user_id
            and _enum_value(getattr(config, "desired_state", None)) == "enabled"
            and str(getattr(config, "active_generation_id", "")) == str(getattr(generation, "id", ""))
            and str(getattr(generation, "owner_user_id", "")) == self._owner_user_id
            and str(getattr(generation, "dataset_id", "")) == self._dataset_id
            and _enum_value(getattr(generation, "state", None)) == "active"
            and int(getattr(generation, "configuration_revision", -1))
            == int(getattr(config, "configuration_revision", -2))
            and getattr(generation, "compatibility_hash", None) == getattr(config, "compatibility_hash", None)
            and getattr(generation, "model_revision", None) == getattr(config, "model_revision", None)
            and _enum_value(getattr(generation, "dimension_state", None)) == "resolved"
            and _enum_value(getattr(config, "dimension_state", None)) == "resolved"
            and getattr(generation, "dimensions", None) == getattr(config, "dimensions", None)
            and getattr(config, "metric", None) == "cosine"
            and bool(getattr(config, "compatibility_hash", None))
            and bool(getattr(config, "normalization_version", None))
            and bool(getattr(config, "chunker_version", None))
        )

    @staticmethod
    def _semantic_binding_signature(
        config: Any,
        generation: Any,
    ) -> tuple[object, ...]:
        return (
            config.active_generation_id,
            config.semantic_index_revision,
            config.configuration_revision,
            getattr(config, "capability_revision", None),
            getattr(config, "disclosure_hash", None),
            config.compatibility_hash,
            getattr(config, "provider", None),
            getattr(config, "model", None),
            config.model_revision,
            getattr(config, "endpoint_origin_revision", None),
            getattr(config, "data_boundary", None),
            getattr(config, "storage_boundary", None),
            config.normalization_version,
            config.chunker_version,
            generation.id,
            generation.state,
            generation.dimensions,
        )

    @staticmethod
    def _binding_signature(binding: _SemanticBinding) -> tuple[object, ...]:
        return binding.signature

    async def _assert_binding_current(
        self,
        binding: _SemanticBinding,
        *,
        effective_threshold: float,
        effective_top_k: int,
        error_code: str = "notes_semantic_projection_binding_changed",
    ) -> None:
        refreshed = await self._resolve_binding(
            effective_threshold=effective_threshold,
            effective_top_k=effective_top_k,
        )
        if not isinstance(refreshed, _SemanticBinding) or (
            self._binding_signature(refreshed) != self._binding_signature(binding)
        ):
            raise SemanticProjectionError(error_code)

    def _projection_revisions(self) -> tuple[int, int]:
        store = getattr(self._db, "note_graph_projection_store", None)
        if store is None:
            return 0, 1
        status = store.get_projection_status()
        return int(store.get_revision()), int(status.parser_version)

    def _query_identity(
        self,
        request: NoteGraphRequest,
        ordinary: NoteGraphResponse,
        *,
        threshold: float,
        top_k: int,
        admission_nodes: int,
        admission_edges: int,
    ) -> SemanticGraphQueryIdentity:
        return SemanticGraphQueryIdentity.from_request(
            request,
            semantic_threshold=threshold,
            semantic_top_k=top_k,
            max_nodes=ordinary.limits.max_nodes,
            max_edges=ordinary.limits.max_edges,
            max_degree=ordinary.limits.max_degree,
            semantic_candidate_nodes=admission_nodes,
            semantic_candidate_edges=admission_edges,
            allow_heavy=bool(request.allow_heavy),
        )

    def _cache_parts(
        self,
        binding: _SemanticBinding,
        identity: SemanticGraphQueryIdentity,
        *,
        projection_revisions: tuple[int, int],
    ) -> tuple[str | None, str]:
        graph_revision, parser_version = projection_revisions
        config = binding.config
        generation = binding.generation
        kwargs = {
            "dataset_id": self._dataset_id,
            "graph_revision": graph_revision,
            "parser_version": parser_version,
            "generation_id": str(generation.id),
            "semantic_index_revision": int(config.semantic_index_revision),
            "configuration_revision": int(config.configuration_revision),
            "capability_revision": str(getattr(config, "capability_revision", "")),
            "disclosure_hash": str(getattr(config, "disclosure_hash", "")),
            "compatibility_hash": str(config.compatibility_hash),
            "provider": str(getattr(config, "provider", "")),
            "model": str(getattr(config, "model", "")),
            "model_revision": config.model_revision,
            "endpoint_origin_revision": str(getattr(config, "endpoint_origin_revision", "")),
            "normalization_version": str(config.normalization_version),
            "chunker_version": str(config.chunker_version),
            "query_identity": identity,
        }
        cache_key = (
            None
            if self._cache is None
            else GraphCache.make_semantic_revision_key(
                user_id=self._owner_user_id,
                **kwargs,
            )
        )
        cursor_binding = GraphCache.make_semantic_cursor_binding(**kwargs)
        return cache_key, cursor_binding

    def _chunk_is_current(
        self,
        chunk: SemanticProjectionChunk,
        binding: _SemanticBinding,
    ) -> bool:
        config = binding.config
        return bool(
            chunk.owner_user_id == self._owner_user_id
            and chunk.dataset_id == self._dataset_id
            and chunk.generation_id == str(binding.generation.id)
            and chunk.content_version > 0
            and chunk.field in {"title", "content"}
            and 0 <= chunk.start_offset < chunk.end_offset
            and chunk.normalization_version == config.normalization_version
            and chunk.chunker_version == config.chunker_version
            and reconstruct_semantic_chunk(
                cast(Any, chunk),
                title=chunk.title,
                content=chunk.content,
                content_version=chunk.content_version,
            )
            is not None
        )

    async def _vector_store(self) -> Any:
        result = self._vector_store_factory()
        return await result if inspect.isawaitable(result) else result

    async def _visible_ids(self, generation_id: str, note_id: str) -> tuple[str, ...]:
        return tuple(
            await asyncio.to_thread(
                self._store.list_visible_vector_ids,
                self._dataset_id,
                generation_id,
                note_id,
            )
        )

    async def _load_chunks(
        self,
        generation_id: str,
        vector_ids: Sequence[str],
    ) -> tuple[SemanticProjectionChunk, ...]:
        return tuple(
            await asyncio.to_thread(
                self._store.load_projection_chunks,
                dataset_id=self._dataset_id,
                generation_id=generation_id,
                vector_ids=tuple(vector_ids),
            )
        )

    async def _query_current_matches(
        self,
        *,
        binding: _SemanticBinding,
        focus_note_id: str,
        threshold: float,
        top_k: int,
        request: NoteGraphRequest | None,
    ) -> tuple[
        tuple[SemanticNoteMatch, ...],
        dict[str, SemanticProjectionChunk],
        set[str],
        int,
        int,
        int,
    ]:
        generation_id = str(binding.generation.id)
        visible_source_ids = await self._visible_ids(generation_id, focus_note_id)
        source_ids = visible_source_ids[: self._settings.max_query_vectors_per_call]
        truncations: set[str] = set()
        if len(source_ids) < len(visible_source_ids):
            truncations.add("semantic_candidates")
        source_chunks = await self._load_chunks(generation_id, source_ids)
        source_by_id = {
            chunk.vector_id: chunk
            for chunk in source_chunks
            if chunk.note_id == focus_note_id and self._chunk_is_current(chunk, binding)
        }
        ordered_source_ids = tuple(vector_id for vector_id in source_ids if vector_id in source_by_id)
        if not ordered_source_ids:
            return (), source_by_id, truncations, 0, 0, 0

        vectors = await self._vector_store()
        fetched = tuple(await vectors.fetch(self._dataset_id, generation_id, ordered_source_ids))
        fetched_by_id = {
            vector.vector_id: vector
            for vector in fetched
            if isinstance(vector, SemanticVector) and vector.vector_id in source_by_id
        }
        ordered_vectors = tuple(
            fetched_by_id[vector_id] for vector_id in ordered_source_ids if vector_id in fetched_by_id
        )
        if not ordered_vectors:
            return (), source_by_id, truncations, 0, 0, 0

        batches = tuple(
            await vectors.query(
                self._dataset_id,
                generation_id,
                tuple(vector.embedding for vector in ordered_vectors),
                limit=top_k,
            )
        )
        if len(batches) != len(ordered_vectors):
            raise SemanticProjectionError("notes_semantic_vector_result_invalid")
        raw_pairs: list[tuple[str, SemanticVectorMatch]] = []
        for source_vector, batch in zip(ordered_vectors, batches):
            try:
                matches = tuple(batch)
            except TypeError as exc:
                raise SemanticProjectionError("notes_semantic_vector_result_invalid") from exc
            if len(matches) > top_k:
                raise SemanticProjectionError("notes_semantic_vector_result_invalid")
            for match in matches:
                if not isinstance(match, SemanticVectorMatch):
                    raise SemanticProjectionError("notes_semantic_vector_result_invalid")
                raw_pairs.append((source_vector.vector_id, match))
        if len(raw_pairs) > self._settings.max_query_candidates_per_call:
            raise SemanticProjectionError("notes_semantic_vector_result_invalid")

        target_ids = tuple(dict.fromkeys(match.vector_id for _, match in raw_pairs))
        current_chunks = await self._load_chunks(
            generation_id,
            (*ordered_source_ids, *target_ids),
        )
        current_by_id = {chunk.vector_id: chunk for chunk in current_chunks if self._chunk_is_current(chunk, binding)}

        await self._assert_binding_current(
            binding,
            effective_threshold=threshold,
            effective_top_k=top_k,
        )

        target_note_ids = tuple(
            sorted(
                {
                    current_by_id[match.vector_id].note_id
                    for _, match in raw_pairs
                    if match.vector_id in current_by_id and current_by_id[match.vector_id].note_id != focus_note_id
                }
            )
        )
        time_range = request.time_range if request is not None else None
        admitted_note_ids = await asyncio.to_thread(
            self._store.filter_projection_note_ids,
            dataset_id=self._dataset_id,
            generation_id=generation_id,
            note_ids=target_note_ids,
            tag=None if request is None else request.tag,
            source=None if request is None else request.source,
            time_range_start=None if time_range is None else time_range.start,
            time_range_end=None if time_range is None else time_range.end,
            time_range_field=("updated_at" if request is None else request.time_range_field),
        )
        candidates: list[SemanticChunkCandidate] = []
        for source_id, match in raw_pairs:
            source_chunk = current_by_id.get(source_id)
            target_chunk = current_by_id.get(match.vector_id)
            if (
                source_chunk is None
                or target_chunk is None
                or source_chunk.note_id != focus_note_id
                or target_chunk.note_id == focus_note_id
                or target_chunk.note_id not in admitted_note_ids
            ):
                continue
            candidates.append(
                SemanticChunkCandidate(
                    target_note_id=target_chunk.note_id,
                    source_chunk_id=source_id,
                    target_chunk_id=match.vector_id,
                    cosine_distance=match.distance,
                )
            )
        ranked, candidates_truncated = _rank_bounded_candidates(
            candidates,
            threshold=threshold,
            top_k=top_k,
            current_chunk_ids=set(current_by_id),
        )
        if candidates_truncated:
            truncations.add("semantic_candidates")
        return (
            ranked,
            current_by_id,
            truncations,
            len(raw_pairs),
            len(candidates),
            len(ranked),
        )

    async def _authoritative_manual_pairs(
        self,
        focus_note_id: str,
    ) -> frozenset[tuple[str, str]]:
        """Load direct owner-scoped manual pairs independently of display filters."""

        try:
            rows = await asyncio.to_thread(
                self._db.get_manual_edges_for_notes,
                self._owner_user_id,
                [focus_note_id],
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise SemanticProjectionError("notes_semantic_manual_authority_unavailable") from exc
        pairs: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = row.get("from_note_id")
            target = row.get("to_note_id")
            if isinstance(source, str) and source and isinstance(target, str) and target:
                pairs.add(tuple(sorted((source, target))))
        return frozenset(pairs)

    def _evidence_excerpt(
        self,
        chunk: SemanticProjectionChunk,
    ) -> SemanticExcerpt | None:
        text = reconstruct_semantic_chunk(
            cast(Any, chunk),
            title=chunk.title,
            content=chunk.content,
            content_version=chunk.content_version,
        )
        if text is None:
            return None
        bounded = text[:_MAX_EXCERPT_CODE_POINTS]
        if not bounded:
            return None
        return SemanticExcerpt(
            field=cast(Any, chunk.field),
            start_code_point=chunk.start_offset,
            end_code_point=chunk.start_offset + len(bounded),
            text=bounded,
        )

    def _semantic_items(
        self,
        matches: Sequence[SemanticNoteMatch],
        chunks: dict[str, SemanticProjectionChunk],
        binding: _SemanticBinding,
    ) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for match in matches:
            pairs: list[SemanticExcerptPair] = []
            target_chunk: SemanticProjectionChunk | None = None
            source_chunk: SemanticProjectionChunk | None = None
            code_points = 0
            for chunk_match in match.chunk_matches[:_MAX_EVIDENCE_PAIRS]:
                current_source = chunks.get(chunk_match.source_chunk_id)
                current_target = chunks.get(chunk_match.target_chunk_id)
                if current_source is None or current_target is None:
                    continue
                source_excerpt = self._evidence_excerpt(current_source)
                target_excerpt = self._evidence_excerpt(current_target)
                if source_excerpt is None or target_excerpt is None:
                    continue
                pair_points = len(source_excerpt.text) + len(target_excerpt.text)
                if code_points + pair_points > _MAX_EDGE_EVIDENCE_CODE_POINTS:
                    continue
                code_points += pair_points
                source_chunk = source_chunk or current_source
                target_chunk = target_chunk or current_target
                pairs.append(SemanticExcerptPair(source=source_excerpt, target=target_excerpt))
            if source_chunk is None or target_chunk is None:
                continue
            evidence = SemanticEdgeEvidence(
                similarity=match.similarity,
                qualitative_band=qualitative_similarity_band(match.similarity),
                source_note_id=source_chunk.note_id,
                target_note_id=target_chunk.note_id,
                source_content_version=source_chunk.content_version,
                target_content_version=target_chunk.content_version,
                generation_id=str(binding.generation.id),
                semantic_index_revision=int(binding.config.semantic_index_revision),
                configuration_revision=int(binding.config.configuration_revision),
                normalization_version=str(binding.config.normalization_version),
                chunker_version=str(binding.config.chunker_version),
                provider_label=binding.provider_label,
                model_label=binding.model_label,
                model_revision=binding.config.model_revision,
                excerpt_pairs=pairs,
            )
            edge_digest = hashlib.sha256(
                (f"{source_chunk.note_id}\0{target_chunk.note_id}\0{binding.generation.id}").encode()
            ).hexdigest()[:24]
            edges.append(
                GraphEdge(
                    id=f"semantic:{edge_digest}",
                    source=source_chunk.note_id,
                    target=target_chunk.note_id,
                    type=EdgeType.semantic,
                    directed=False,
                    weight=match.similarity,
                    evidence=evidence,
                )
            )
            nodes.append(
                GraphNode(
                    id=target_chunk.note_id,
                    type="note",
                    label=target_chunk.title,
                    created_at=target_chunk.created_at,
                    deleted=False,
                    degree=0,
                )
            )
        return tuple(nodes), tuple(edges)

    async def project(
        self,
        request: NoteGraphRequest,
        ordinary: NoteGraphResponse,
        *,
        user: Any,
    ) -> NoteGraphResponse:
        """Project one opt-in semantic first page or fail open to ordinary graph."""

        if not request.semantic_requested:
            return ordinary
        threshold, top_k = self._effective_controls(request)
        if str(getattr(user, "id_str", "")) != self._owner_user_id:
            record_semantic_denial("permission")
            return ordinary.model_copy(
                update={
                    "semantic_status": self._status(
                        available=False,
                        state="unavailable",
                        detail_reason="owner_mismatch",
                        config=None,
                        integrity=None,
                        effective_threshold=threshold,
                        effective_top_k=top_k,
                    )
                }
            )
        if not request.center_note_id:
            return ordinary.model_copy(
                update={
                    "semantic_status": self._status(
                        available=False,
                        state="focus_required",
                        detail_reason="focus_required",
                        config=None,
                        integrity=None,
                        effective_threshold=threshold,
                        effective_top_k=top_k,
                    )
                }
            )

        binding: _SemanticBinding | None = None
        try:
            binding_or_status = await self._resolve_binding(
                effective_threshold=threshold,
                effective_top_k=top_k,
            )
            if isinstance(binding_or_status, SemanticGraphStatus):
                return ordinary.model_copy(update={"semantic_status": binding_or_status})
            binding = binding_or_status
            admission_nodes = min(top_k, ordinary.limits.max_nodes)
            admission_edges = min(top_k, ordinary.limits.max_edges)
            identity = self._query_identity(
                request,
                ordinary,
                threshold=threshold,
                top_k=top_k,
                admission_nodes=admission_nodes,
                admission_edges=admission_edges,
            )
            projection_revisions = await asyncio.to_thread(self._projection_revisions)
            cache_key, cursor_binding = self._cache_parts(
                binding,
                identity,
                projection_revisions=projection_revisions,
            )

            if request.cursor is not None:
                await self._assert_binding_current(
                    binding,
                    effective_threshold=threshold,
                    effective_top_k=top_k,
                )
                try:
                    _decode_cursor(
                        request.cursor,
                        expected_semantic_binding=cursor_binding,
                    )
                except InputError as exc:
                    raise SemanticProjectionError("notes_semantic_cursor_mismatch") from exc
                return ordinary.model_copy(
                    update={
                        "cursor": bind_semantic_cursor(
                            ordinary.cursor,
                            semantic_binding=cursor_binding,
                        ),
                        "semantic_status": binding.status,
                    }
                )

            cache_allowed = binding.status.dirty_notes == 0
            if cache_allowed and self._cache is not None and cache_key is not None:
                cached = self._cache.get(cache_key)
                if isinstance(cached, NoteGraphResponse):
                    if await asyncio.to_thread(self._projection_revisions) != projection_revisions:
                        raise SemanticProjectionError("notes_semantic_projection_binding_changed")
                    await self._assert_binding_current(
                        binding,
                        effective_threshold=threshold,
                        effective_top_k=top_k,
                    )
                    cached_truncations = [] if cached.semantic_status is None else cached.semantic_status.truncated_by
                    fresh_status = binding.status.model_copy(update={"truncated_by": cached_truncations})
                    return cached.model_copy(update={"semantic_status": fresh_status})

            candidate_result = cast(
                SemanticGraphCandidateResult,
                await asyncio.to_thread(
                    self._graph_service.generate_semantic_candidates,
                    request,
                    additional_nodes=admission_nodes,
                    additional_edges=admission_edges,
                ),
            )
            if await asyncio.to_thread(self._projection_revisions) != projection_revisions:
                raise SemanticProjectionError("notes_semantic_projection_binding_changed")
            manual_pairs = await self._authoritative_manual_pairs(request.center_note_id)
            query_started = time.perf_counter()
            try:
                (
                    matches,
                    chunks,
                    truncations,
                    candidate_count,
                    filtered_count,
                    admitted_count,
                ) = await self._query_current_matches(
                    binding=binding,
                    focus_note_id=request.center_note_id,
                    threshold=threshold,
                    top_k=top_k,
                    request=request,
                )
            except Exception:
                record_semantic_query_metrics(
                    status="failed",
                    backend=str(binding.config.vector_backend),
                    duration_seconds=time.perf_counter() - query_started,
                    candidate_count=0,
                    filtered_count=0,
                    admitted_count=0,
                )
                raise
            if await asyncio.to_thread(self._projection_revisions) != projection_revisions:
                raise SemanticProjectionError("notes_semantic_projection_binding_changed")
            await self._assert_binding_current(
                binding,
                effective_threshold=threshold,
                effective_top_k=top_k,
            )
            semantic_nodes, semantic_edges = self._semantic_items(
                matches,
                chunks,
                binding,
            )
            composed = compose_semantic_graph(
                public_graph=candidate_result.public_graph,
                candidate_nodes=candidate_result.candidate_nodes,
                candidate_edges=candidate_result.candidate_edges,
                semantic_nodes=semantic_nodes,
                semantic_edges=semantic_edges,
                focus_note_id=request.center_note_id,
                authoritative_manual_pairs=manual_pairs,
            )
            bounded_edges, evidence_truncated = bound_semantic_evidence(composed.edges)
            if evidence_truncated:
                truncations.add("semantic_evidence_bytes")
                composed = composed.model_copy(update={"edges": list(bounded_edges)})
            admitted_semantic_ids = {edge.id for edge in composed.edges if edge.type is EdgeType.semantic}
            if len(admitted_semantic_ids) < len(semantic_edges):
                truncations.add("semantic_edges")
                admitted_targets = {edge.target for edge in composed.edges if edge.type is EdgeType.semantic}
                if len(admitted_targets) < len(semantic_nodes):
                    truncations.add("semantic_nodes")
            record_semantic_query_metrics(
                status="success",
                backend=str(binding.config.vector_backend),
                duration_seconds=time.perf_counter() - query_started,
                candidate_count=candidate_count,
                filtered_count=filtered_count,
                admitted_count=admitted_count,
                truncations=tuple(sorted(truncations)),
            )
            status = binding.status.model_copy(update={"truncated_by": sorted(truncations)})
            stable = composed.model_copy(
                update={
                    "cursor": bind_semantic_cursor(
                        composed.cursor,
                        semantic_binding=cursor_binding,
                    ),
                    "semantic_status": status,
                }
            )
            if cache_allowed and self._cache is not None and cache_key is not None:
                self._cache.put(cache_key, stable)
            return stable.model_copy(update={"semantic_status": status})
        except SemanticProjectionError as exc:
            if exc.code == "notes_semantic_cursor_mismatch":
                raise
            if exc.code == "notes_semantic_projection_binding_changed":
                reason = "configuration_stale"
            else:
                reason = "vector_unavailable"
            logger.warning("Notes semantic projection failed code={}", exc.code)
            record_semantic_failure(
                component="vector",
                category=(
                    "configuration"
                    if exc.code == "notes_semantic_projection_binding_changed"
                    else "invalid_response"
                    if exc.code == "notes_semantic_vector_result_invalid"
                    else "unavailable"
                ),
                backend=(str(binding.config.vector_backend) if binding is not None else "unavailable"),
            )
        except Exception as exc:  # noqa: BLE001 - semantic reads fail open by contract.
            logger.warning(
                "Notes semantic projection failed error_type={}",
                type(exc).__name__,
            )
            reason = "vector_unavailable"
            record_semantic_failure(
                component="vector",
                category="execution",
                backend=(str(binding.config.vector_backend) if binding is not None else "unavailable"),
            )
        if binding is not None:
            unavailable_status = binding.status.model_copy(
                update={
                    "available": False,
                    "state": "unavailable",
                    "detail_reason": reason,
                }
            )
        else:
            unavailable_status = self._status(
                available=False,
                state="unavailable",
                detail_reason=reason,
                config=None,
                integrity=None,
                effective_threshold=threshold,
                effective_top_k=top_k,
            )
        return ordinary.model_copy(update={"semantic_status": unavailable_status})

    async def validate_conversion(
        self,
        *,
        source_note_id: str,
        target_note_id: str,
        generation_id: str,
    ) -> None:
        """Validate one current semantic pair before canonical manual-link write."""

        if source_note_id == target_note_id:
            raise SemanticProjectionError("notes_semantic_conversion_pair_mismatch")
        threshold = 0.0
        top_k = self._settings.max_query_neighbors
        binding_or_status = await self._resolve_binding(
            effective_threshold=threshold,
            effective_top_k=top_k,
        )
        if not isinstance(binding_or_status, _SemanticBinding) or (
            str(getattr(binding_or_status.generation, "id", "")) != generation_id
        ):
            raise SemanticProjectionError("notes_semantic_conversion_generation_stale")
        source_ids, target_ids = await asyncio.gather(
            self._visible_ids(generation_id, source_note_id),
            self._visible_ids(generation_id, target_note_id),
        )
        current_pair = await self._load_chunks(
            generation_id,
            (*source_ids, *target_ids),
        )
        current_note_ids = {chunk.note_id for chunk in current_pair if self._chunk_is_current(chunk, binding_or_status)}
        if not {source_note_id, target_note_id} <= current_note_ids:
            raise SemanticProjectionError("notes_semantic_conversion_owner_mismatch")
        try:
            (
                matches,
                _chunks,
                truncations,
                candidate_count,
                filtered_count,
                admitted_count,
            ) = await self._query_current_matches(
                binding=binding_or_status,
                focus_note_id=source_note_id,
                threshold=threshold,
                top_k=top_k,
                request=None,
            )
            record_semantic_query_metrics(
                status="success",
                backend=str(binding_or_status.config.vector_backend),
                duration_seconds=0,
                candidate_count=candidate_count,
                filtered_count=filtered_count,
                admitted_count=admitted_count,
                truncations=tuple(sorted(truncations)),
            )
        except SemanticProjectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve closed conversion errors.
            raise SemanticProjectionError("notes_semantic_conversion_unavailable") from exc
        await self._assert_binding_current(
            binding_or_status,
            effective_threshold=threshold,
            effective_top_k=top_k,
            error_code="notes_semantic_conversion_generation_stale",
        )
        if target_note_id not in {match.target_note_id for match in matches}:
            raise SemanticProjectionError("notes_semantic_conversion_pair_mismatch")


__all__ = [
    "SemanticGraphProjector",
    "SemanticProjectionError",
    "_rank_bounded_candidates",
    "bound_semantic_evidence",
    "build_projection_vector_store",
    "compose_semantic_graph",
]
