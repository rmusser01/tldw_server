"""Vector-only ChromaDB backend for Notes semantic generations."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from chromadb.errors import ChromaError

from .semantic_vectors import (
    SemanticVector,
    SemanticVectorBinding,
    SemanticVectorCapabilityError,
    SemanticVectorCleanup,
    SemanticVectorError,
    SemanticVectorMatch,
)

_MISSING_COLLECTION_EXCEPTIONS = {
    "InvalidCollectionException",
    "KeyError",
    "NotFoundError",
}
_CHROMA_OPERATION_ERRORS = (
    AttributeError,
    ChromaError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _namespace(binding: SemanticVectorBinding) -> str:
    payload = "\x00".join(
        (binding.owner_user_id, binding.dataset_id, binding.generation_id)
    ).encode("utf-8")
    return f"nsv_{hashlib.sha256(payload).hexdigest()[:48]}"


def _is_missing_collection(exc: BaseException) -> bool:
    return type(exc).__name__ in _MISSING_COLLECTION_EXCEPTIONS


def _sequence_or_empty(value: Any) -> tuple[Any, ...]:
    return () if value is None else tuple(value)


class ChromaSemanticVectorBackend:
    """Use direct Chroma collection operations without documents or metadata."""

    name = "chromadb"

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._client = getattr(manager, "client", None)

    async def check_capability(self) -> None:
        client = self._client
        required = (
            "delete_collection",
            "get_collection",
            "get_or_create_collection",
        )
        if client is None or any(not callable(getattr(client, name, None)) for name in required):
            raise SemanticVectorCapabilityError("notes_semantic_chroma_unavailable")
        if type(client).__name__ == "_InMemoryChromaClient":
            raise SemanticVectorCapabilityError("notes_semantic_chroma_vector_only_unavailable")

    def supports_dimensions(self, dimensions: int) -> bool:
        return isinstance(dimensions, int) and not isinstance(dimensions, bool) and dimensions > 0

    def _create_sync(self, binding: SemanticVectorBinding) -> None:
        try:
            collection = self._client.get_or_create_collection(
                name=_namespace(binding),
                metadata={"hnsw:space": "cosine"},
            )
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorCapabilityError("notes_semantic_chroma_unavailable") from None
        metadata = getattr(collection, "metadata", None)
        if not isinstance(metadata, dict) or metadata.get("hnsw:space") != "cosine":
            raise SemanticVectorCapabilityError("notes_semantic_chroma_cosine_unavailable")
        for method in ("delete", "get", "query", "upsert"):
            if not callable(getattr(collection, method, None)):
                raise SemanticVectorCapabilityError("notes_semantic_chroma_vector_only_unavailable")

    async def create_generation_storage(self, binding: SemanticVectorBinding) -> None:
        await asyncio.to_thread(self._create_sync, binding)

    def _collection_or_none(self, binding: SemanticVectorBinding) -> Any | None:
        try:
            return self._client.get_collection(name=_namespace(binding))
        except _CHROMA_OPERATION_ERRORS as exc:
            if _is_missing_collection(exc):
                return None
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None

    def _upsert_sync(
        self,
        binding: SemanticVectorBinding,
        vectors: tuple[SemanticVector, ...],
    ) -> int:
        collection = self._collection_or_none(binding)
        if collection is None:
            raise SemanticVectorError("notes_semantic_vector_storage_missing")
        try:
            collection.upsert(
                ids=[vector.vector_id for vector in vectors],
                embeddings=[list(vector.embedding) for vector in vectors],
            )
        except TypeError:
            raise SemanticVectorCapabilityError(
                "notes_semantic_chroma_vector_only_unavailable"
            ) from None
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        return len(vectors)

    async def upsert(
        self,
        binding: SemanticVectorBinding,
        vectors: tuple[SemanticVector, ...],
    ) -> int:
        return await asyncio.to_thread(self._upsert_sync, binding, vectors)

    def _fetch_sync(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> tuple[SemanticVector, ...]:
        collection = self._collection_or_none(binding)
        if collection is None:
            return ()
        try:
            result = collection.get(ids=list(vector_ids), include=["embeddings"])
            ids = _sequence_or_empty(result.get("ids"))
            embeddings = _sequence_or_empty(result.get("embeddings"))
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        if len(ids) != len(embeddings):
            raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
        try:
            return tuple(
                SemanticVector(
                    vector_id=str(vector_id),
                    embedding=tuple(float(value) for value in embedding),
                )
                for vector_id, embedding in zip(ids, embeddings)
            )
        except (TypeError, ValueError, OverflowError):
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            ) from None

    async def fetch(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> tuple[SemanticVector, ...]:
        return await asyncio.to_thread(self._fetch_sync, binding, vector_ids)

    def _query_sync(
        self,
        binding: SemanticVectorBinding,
        query_vectors: tuple[tuple[float, ...], ...],
        limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        collection = self._collection_or_none(binding)
        if collection is None:
            return tuple(() for _ in query_vectors)
        try:
            result = collection.query(
                query_embeddings=[list(vector) for vector in query_vectors],
                n_results=limit,
                include=["distances"],
            )
            ids_by_query = _sequence_or_empty(result.get("ids"))
            distances_by_query = _sequence_or_empty(result.get("distances"))
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        if len(ids_by_query) != len(query_vectors) or len(distances_by_query) != len(query_vectors):
            raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
        batches: list[tuple[SemanticVectorMatch, ...]] = []
        for ids, distances in zip(ids_by_query, distances_by_query):
            if len(ids) != len(distances):
                raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
            try:
                batches.append(
                    tuple(
                        SemanticVectorMatch(vector_id=str(vector_id), distance=float(distance))
                        for vector_id, distance in zip(ids, distances)
                    )
                )
            except (TypeError, ValueError, OverflowError):
                raise SemanticVectorError(
                    "notes_semantic_vector_backend_result_invalid"
                ) from None
        return tuple(batches)

    async def query(
        self,
        binding: SemanticVectorBinding,
        query_vectors: tuple[tuple[float, ...], ...],
        *,
        limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        return await asyncio.to_thread(
            self._query_sync,
            binding,
            query_vectors,
            limit,
        )

    def _delete_ids_sync(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> SemanticVectorCleanup:
        collection = self._collection_or_none(binding)
        if collection is None:
            return SemanticVectorCleanup(confirmed_absent=True)
        try:
            collection.delete(ids=list(vector_ids))
            remaining = collection.get(ids=list(vector_ids), include=[]).get("ids") or ()
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        return SemanticVectorCleanup(confirmed_absent=not bool(remaining))

    async def delete_ids(
        self,
        binding: SemanticVectorBinding,
        vector_ids: tuple[str, ...],
    ) -> SemanticVectorCleanup:
        return await asyncio.to_thread(self._delete_ids_sync, binding, vector_ids)

    def _delete_generation_sync(
        self,
        binding: SemanticVectorBinding,
    ) -> SemanticVectorCleanup:
        if self._collection_or_none(binding) is None:
            return SemanticVectorCleanup(confirmed_absent=True)
        try:
            self._client.delete_collection(name=_namespace(binding))
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        return SemanticVectorCleanup(
            confirmed_absent=self._collection_or_none(binding) is None
        )

    async def delete_generation(
        self,
        binding: SemanticVectorBinding,
    ) -> SemanticVectorCleanup:
        return await asyncio.to_thread(self._delete_generation_sync, binding)


__all__ = ["ChromaSemanticVectorBackend"]
