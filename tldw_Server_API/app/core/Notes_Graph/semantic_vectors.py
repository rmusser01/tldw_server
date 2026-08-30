"""Authority-bound async facade for vector-only Notes semantic storage."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
)

from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings


class SemanticVectorError(RuntimeError):
    """Base error carrying only a stable, non-sensitive code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SemanticVectorCapabilityError(SemanticVectorError):
    """The selected physical backend cannot satisfy the vector-only contract."""


class SemanticVectorBindingError(SemanticVectorError):
    """ChaChaNotes does not authorize the requested generation binding."""


class SemanticVectorValidationError(SemanticVectorError, ValueError):
    """A vector or physical backend result violates the pinned contract."""


@dataclass(frozen=True, slots=True)
class SemanticVectorBinding:
    """One authoritative owner/dataset/generation vector namespace."""

    owner_user_id: str
    dataset_id: str
    generation_id: str
    dimensions: int


@dataclass(frozen=True, slots=True)
class SemanticVector:
    """An opaque vector identifier and its embedding."""

    vector_id: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SemanticVectorMatch:
    """One nearest-neighbor result with raw cosine distance."""

    vector_id: str
    distance: float


@dataclass(frozen=True, slots=True)
class SemanticVectorCleanup:
    """Confirmation that the requested physical vector target is absent."""

    confirmed_absent: bool


class SemanticGenerationAuthority(Protocol):
    """Minimal ChaChaNotes generation authority consumed by the facade."""

    @property
    def owner_user_id(self) -> str: ...

    def get_generation(self, dataset_id: str, generation_id: str) -> Any: ...


class SemanticVectorBackend(Protocol):
    """Physical vector-only backend implemented by ChromaDB and pgvector."""

    name: str

    async def check_capability(self) -> None: ...

    def supports_dimensions(self, dimensions: int) -> bool: ...

    async def create_generation_storage(self, binding: SemanticVectorBinding) -> None: ...

    async def upsert(
        self,
        binding: SemanticVectorBinding,
        vectors: tuple[SemanticVector, ...],
    ) -> int: ...

    async def fetch(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> tuple[SemanticVector, ...]: ...

    async def query(
        self,
        binding: SemanticVectorBinding,
        query_vectors: tuple[tuple[float, ...], ...],
        *,
        limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]: ...

    async def delete_ids(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> SemanticVectorCleanup: ...

    async def delete_generation(
        self,
        binding: SemanticVectorBinding,
    ) -> SemanticVectorCleanup: ...


def _dimension_state_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _validated_vector_id(value: object) -> str:
    if not isinstance(value, str):
        raise SemanticVectorValidationError("notes_semantic_vector_id_invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise SemanticVectorValidationError("notes_semantic_vector_id_invalid") from None
    if not value or len(encoded) > 512 or any(ord(char) < 32 for char in value):
        raise SemanticVectorValidationError("notes_semantic_vector_id_invalid")
    return value


def _validated_embedding(value: Sequence[float], *, dimensions: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise SemanticVectorValidationError("notes_semantic_vector_dimensions_mismatch")
    try:
        raw_values = tuple(value)
    except TypeError:
        raise SemanticVectorValidationError(
            "notes_semantic_vector_dimensions_mismatch"
        ) from None
    if len(raw_values) != dimensions:
        raise SemanticVectorValidationError("notes_semantic_vector_dimensions_mismatch")
    normalized: list[float] = []
    for raw in raw_values:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise SemanticVectorValidationError("notes_semantic_vector_value_invalid")
        number = float(raw)
        if not math.isfinite(number):
            raise SemanticVectorValidationError("notes_semantic_vector_non_finite")
        normalized.append(number)
    norm_squared = math.fsum(number * number for number in normalized)
    if not math.isfinite(norm_squared):
        raise SemanticVectorValidationError("notes_semantic_vector_non_finite")
    if norm_squared <= 0.0:
        raise SemanticVectorValidationError("notes_semantic_vector_zero_norm")
    return tuple(normalized)


def _validated_ids(values: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(_validated_vector_id(value) for value in values)
    if len(ids) != len(set(ids)):
        raise SemanticVectorValidationError("notes_semantic_vector_ids_duplicate")
    return ids


class NotesSemanticVectorStore:
    """Validate Notes authority and vector invariants before every backend call."""

    def __init__(
        self,
        *,
        authority: SemanticGenerationAuthority,
        backend: SemanticVectorBackend,
        max_query_neighbors: int = 100,
    ) -> None:
        self._authority = authority
        self._backend = backend
        self._max_query_neighbors = max_query_neighbors

    async def _binding(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> SemanticVectorBinding:
        generation = await asyncio.to_thread(
            self._authority.get_generation,
            dataset_id,
            generation_id,
        )
        if generation is None:
            raise SemanticVectorBindingError("notes_semantic_vector_binding_invalid")
        owner_user_id = str(self._authority.owner_user_id)
        if str(getattr(generation, "owner_user_id", "")) != owner_user_id:
            raise SemanticVectorBindingError("notes_semantic_vector_owner_mismatch")
        if (
            str(getattr(generation, "dataset_id", "")) != dataset_id
            or str(getattr(generation, "id", "")) != generation_id
        ):
            raise SemanticVectorBindingError("notes_semantic_vector_binding_invalid")
        if _dimension_state_value(getattr(generation, "dimension_state", None)) != SemanticDimensionState.RESOLVED.value:
            raise SemanticVectorBindingError("notes_semantic_vector_dimensions_unresolved")
        dimensions = getattr(generation, "dimensions", None)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise SemanticVectorBindingError("notes_semantic_vector_dimensions_unresolved")
        if not self._backend.supports_dimensions(dimensions):
            raise SemanticVectorBindingError("notes_semantic_vector_dimensions_unsupported")
        return SemanticVectorBinding(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            generation_id=generation_id,
            dimensions=dimensions,
        )

    async def create_generation_storage(self, dataset_id: str, generation_id: str) -> None:
        binding = await self._binding(dataset_id, generation_id)
        await self._backend.create_generation_storage(binding)

    async def upsert(
        self,
        dataset_id: str,
        generation_id: str,
        vectors: Sequence[SemanticVector],
    ) -> int:
        binding = await self._binding(dataset_id, generation_id)
        normalized: list[SemanticVector] = []
        seen_ids: set[str] = set()
        for vector in vectors:
            vector_id = _validated_vector_id(vector.vector_id)
            if vector_id in seen_ids:
                raise SemanticVectorValidationError("notes_semantic_vector_ids_duplicate")
            seen_ids.add(vector_id)
            normalized.append(
                SemanticVector(
                    vector_id=vector_id,
                    embedding=_validated_embedding(vector.embedding, dimensions=binding.dimensions),
                )
            )
        if not normalized:
            return 0
        return await self._backend.upsert(binding, tuple(normalized))

    async def fetch(
        self,
        dataset_id: str,
        generation_id: str,
        vector_ids: Sequence[str],
    ) -> tuple[SemanticVector, ...]:
        binding = await self._binding(dataset_id, generation_id)
        ids = _validated_ids(vector_ids)
        if not ids:
            return ()
        fetched = await self._backend.fetch(binding, ids)
        requested = set(ids)
        by_id: dict[str, SemanticVector] = {}
        for vector in fetched:
            vector_id = _validated_vector_id(vector.vector_id)
            if vector_id not in requested or vector_id in by_id:
                raise SemanticVectorValidationError("notes_semantic_vector_backend_result_invalid")
            by_id[vector_id] = SemanticVector(
                vector_id=vector_id,
                embedding=_validated_embedding(vector.embedding, dimensions=binding.dimensions),
            )
        return tuple(by_id[vector_id] for vector_id in ids if vector_id in by_id)

    async def query(
        self,
        dataset_id: str,
        generation_id: str,
        query_vectors: Sequence[Sequence[float]],
        *,
        limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        binding = await self._binding(dataset_id, generation_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self._max_query_neighbors:
            raise SemanticVectorValidationError("notes_semantic_vector_query_limit_invalid")
        normalized = tuple(
            _validated_embedding(vector, dimensions=binding.dimensions)
            for vector in query_vectors
        )
        if not normalized:
            return ()
        batches = await self._backend.query(binding, normalized, limit=limit)
        if len(batches) != len(normalized):
            raise SemanticVectorValidationError("notes_semantic_vector_backend_result_invalid")
        validated_batches: list[tuple[SemanticVectorMatch, ...]] = []
        for batch in batches:
            seen_ids: set[str] = set()
            matches: list[SemanticVectorMatch] = []
            for match in batch:
                vector_id = _validated_vector_id(match.vector_id)
                if isinstance(match.distance, bool):
                    raise SemanticVectorValidationError(
                        "notes_semantic_vector_backend_result_invalid"
                    )
                try:
                    distance = float(match.distance)
                except (TypeError, ValueError, OverflowError):
                    raise SemanticVectorValidationError(
                        "notes_semantic_vector_backend_result_invalid"
                    ) from None
                if vector_id in seen_ids or not math.isfinite(distance):
                    raise SemanticVectorValidationError("notes_semantic_vector_backend_result_invalid")
                seen_ids.add(vector_id)
                matches.append(SemanticVectorMatch(vector_id=vector_id, distance=distance))
            validated_batches.append(
                tuple(sorted(matches, key=lambda item: (item.distance, item.vector_id))[:limit])
            )
        return tuple(validated_batches)

    async def delete_ids(
        self,
        dataset_id: str,
        generation_id: str,
        vector_ids: Sequence[str],
    ) -> SemanticVectorCleanup:
        binding = await self._binding(dataset_id, generation_id)
        ids = _validated_ids(vector_ids)
        if not ids:
            return SemanticVectorCleanup(confirmed_absent=True)
        return await self._backend.delete_ids(binding, ids)

    async def delete_generation(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> SemanticVectorCleanup:
        binding = await self._binding(dataset_id, generation_id)
        return await self._backend.delete_generation(binding)


async def create_semantic_vector_store(
    backend_name: str,
    *,
    authority: SemanticGenerationAuthority,
    chroma_manager: Any | None = None,
    postgres_backend: Any | None = None,
    settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
) -> NotesSemanticVectorStore:
    """Construct one capable vector-only backend or fail with a stable code."""

    normalized_name = str(backend_name).strip().lower()
    if normalized_name == "chromadb":
        if chroma_manager is None:
            raise SemanticVectorCapabilityError("notes_semantic_chroma_unavailable")
        from .semantic_vectors_chroma import ChromaSemanticVectorBackend

        backend: SemanticVectorBackend = ChromaSemanticVectorBackend(chroma_manager)
    elif normalized_name == "pgvector":
        if postgres_backend is None:
            raise SemanticVectorCapabilityError("notes_semantic_pgvector_unavailable")
        from .semantic_vectors_pg import PostgresSemanticVectorBackend

        backend = PostgresSemanticVectorBackend(
            postgres_backend,
            allowed_dimensions=settings.pgvector_allowed_dimensions,
        )
    else:
        raise SemanticVectorCapabilityError("notes_semantic_vector_backend_unsupported")
    await backend.check_capability()
    return NotesSemanticVectorStore(
        authority=authority,
        backend=backend,
        max_query_neighbors=settings.max_query_neighbors,
    )


__all__ = [
    "NotesSemanticVectorStore",
    "SemanticVector",
    "SemanticVectorBackend",
    "SemanticVectorBinding",
    "SemanticVectorBindingError",
    "SemanticVectorCapabilityError",
    "SemanticVectorCleanup",
    "SemanticVectorError",
    "SemanticVectorMatch",
    "SemanticVectorValidationError",
    "create_semantic_vector_store",
]
