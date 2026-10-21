"""Vector-only ChromaDB backend for Notes semantic generations."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import chromadb.errors as chroma_errors

from .semantic_vectors import (
    SemanticVector,
    SemanticVectorBinding,
    SemanticVectorCapabilityError,
    SemanticVectorCleanup,
    SemanticVectorError,
    SemanticVectorMatch,
)

ChromaError = chroma_errors.ChromaError
_CHROMA_NOT_FOUND_ERRORS = tuple(
    error_type
    for error_name in ("NotFoundError", "InvalidCollectionException")
    if isinstance((error_type := getattr(chroma_errors, error_name, None)), type)
    and issubclass(error_type, BaseException)
)
_CHROMA_OPERATION_ERRORS = (
    AttributeError,
    ChromaError,
    ConnectionError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_LEGACY_COLLECTION_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$"
)
_IPV4_SHAPE = re.compile(r"^[0-9]{1,3}(?:\.[0-9]{1,3}){3}$")


def _namespace(binding: SemanticVectorBinding) -> str:
    payload = "\x00".join(
        (binding.owner_user_id, binding.dataset_id, binding.generation_id)
    ).encode("utf-8")
    return f"nsv_{hashlib.sha256(payload).hexdigest()[:48]}"


def _sequence_or_empty(value: Any) -> tuple[Any, ...]:
    return () if value is None else tuple(value)


def _strict_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
    try:
        return tuple(value)
    except _CHROMA_OPERATION_ERRORS:
        raise SemanticVectorError(
            "notes_semantic_vector_backend_result_invalid"
        ) from None


def _legacy_collection_name(value: object) -> str:
    name = value if type(value) is str else getattr(value, "name", None)
    if type(name) is not str:
        raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        raise SemanticVectorError(
            "notes_semantic_vector_backend_result_invalid"
        ) from None
    if (
        _LEGACY_COLLECTION_NAME.fullmatch(name) is None
        or ".." in name
        or _IPV4_SHAPE.fullmatch(name) is not None
    ):
        raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
    return name


class ChromaSemanticVectorBackend:
    """Use direct Chroma collection operations without documents or metadata."""

    name = "chromadb"

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._client = getattr(manager, "client", None)

    async def check_capability(self) -> None:
        client = self._client
        required = (
            "create_collection",
            "delete_collection",
            "get_collection",
            "list_collections",
        )
        if client is None or any(not callable(getattr(client, name, None)) for name in required):
            raise SemanticVectorCapabilityError("notes_semantic_chroma_unavailable")
        if type(client).__name__ == "_InMemoryChromaClient":
            raise SemanticVectorCapabilityError("notes_semantic_chroma_vector_only_unavailable")

    def supports_dimensions(self, dimensions: int) -> bool:
        return isinstance(dimensions, int) and not isinstance(dimensions, bool) and dimensions > 0

    def _create_sync(self, binding: SemanticVectorBinding) -> None:
        try:
            collection = self._collection_or_none(binding)
        except SemanticVectorError:
            raise SemanticVectorCapabilityError("notes_semantic_chroma_unavailable") from None
        if collection is None:
            try:
                collection = self._client.create_collection(
                    name=_namespace(binding),
                    metadata={"hnsw:space": "cosine"},
                )
            except _CHROMA_OPERATION_ERRORS:
                try:
                    collection = self._collection_or_none(binding)
                except SemanticVectorError:
                    raise SemanticVectorCapabilityError(
                        "notes_semantic_chroma_unavailable"
                    ) from None
                if collection is None:
                    raise SemanticVectorCapabilityError(
                        "notes_semantic_chroma_unavailable"
                    ) from None
        metadata = getattr(collection, "metadata", None)
        if not isinstance(metadata, Mapping) or metadata.get("hnsw:space") != "cosine":
            raise SemanticVectorCapabilityError("notes_semantic_chroma_cosine_unavailable")
        for method in ("delete", "get", "query", "upsert"):
            if not callable(getattr(collection, method, None)):
                raise SemanticVectorCapabilityError("notes_semantic_chroma_vector_only_unavailable")

    async def create_generation_storage(self, binding: SemanticVectorBinding) -> None:
        await asyncio.to_thread(self._create_sync, binding)

    def _collection_or_none(self, binding: SemanticVectorBinding) -> Any | None:
        collection_name = _namespace(binding)
        try:
            return self._client.get_collection(name=collection_name)
        except _CHROMA_NOT_FOUND_ERRORS:
            return None
        except ValueError:
            return self._legacy_collection_or_none(collection_name)
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None

    def _legacy_collection_or_none(self, collection_name: str) -> None:
        try:
            listed = self._client.list_collections()
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        if isinstance(listed, (str, bytes)) or not isinstance(listed, Sequence):
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            )
        names: list[str] = []
        try:
            for item in listed:
                names.append(_legacy_collection_name(item))
        except SemanticVectorError:
            raise
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            ) from None
        if len(names) != len(set(names)):
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            )
        if collection_name in names:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed")
        return None

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
        candidate_limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        collection = self._collection_or_none(binding)
        if collection is None:
            return tuple(() for _ in query_vectors)
        try:
            result = collection.query(
                query_embeddings=[list(vector) for vector in query_vectors],
                n_results=candidate_limit,
                include=["distances"],
            )
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        try:
            if (
                not isinstance(result, Mapping)
                or "ids" not in result
                or "distances" not in result
            ):
                raise SemanticVectorError(
                    "notes_semantic_vector_backend_result_invalid"
                )
            ids_by_query = _strict_sequence(result["ids"])
            distances_by_query = _strict_sequence(result["distances"])
        except SemanticVectorError:
            raise
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            ) from None
        if len(ids_by_query) != len(query_vectors) or len(distances_by_query) != len(query_vectors):
            raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
        batches: list[tuple[SemanticVectorMatch, ...]] = []
        for raw_ids, raw_distances in zip(ids_by_query, distances_by_query):
            ids = _strict_sequence(raw_ids)
            distances = _strict_sequence(raw_distances)
            if len(ids) != len(distances):
                raise SemanticVectorError("notes_semantic_vector_backend_result_invalid")
            matches: list[SemanticVectorMatch] = []
            for vector_id, distance in zip(ids, distances):
                if type(vector_id) is not str or type(distance) not in (int, float):
                    raise SemanticVectorError(
                        "notes_semantic_vector_backend_result_invalid"
                    )
                try:
                    finite = math.isfinite(distance)
                    normalized_distance = float(distance)
                except (OverflowError, TypeError, ValueError):
                    raise SemanticVectorError(
                        "notes_semantic_vector_backend_result_invalid"
                    ) from None
                if not finite:
                    raise SemanticVectorError(
                        "notes_semantic_vector_backend_result_invalid"
                    )
                matches.append(
                    SemanticVectorMatch(
                        vector_id=vector_id,
                        distance=normalized_distance,
                    )
                )
            batches.append(tuple(matches))
        return tuple(batches)

    async def query(
        self,
        binding: SemanticVectorBinding,
        query_vectors: tuple[tuple[float, ...], ...],
        *,
        limit: int,
        candidate_limit: int,
    ) -> tuple[tuple[SemanticVectorMatch, ...], ...]:
        return await asyncio.to_thread(
            self._query_sync,
            binding,
            query_vectors,
            limit,
            candidate_limit,
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
            result = collection.get(ids=list(vector_ids), include=[])
        except _CHROMA_OPERATION_ERRORS:
            raise SemanticVectorError("notes_semantic_chroma_operation_failed") from None
        if not isinstance(result, Mapping) or "ids" not in result:
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            )
        remaining = result["ids"]
        if isinstance(remaining, (str, bytes)) or not isinstance(remaining, Sequence):
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            )
        remaining_ids = tuple(remaining)
        if (
            any(not isinstance(vector_id, str) for vector_id in remaining_ids)
            or len(remaining_ids) != len(set(remaining_ids))
            or not set(remaining_ids).issubset(vector_ids)
        ):
            raise SemanticVectorError(
                "notes_semantic_vector_backend_result_invalid"
            )
        return SemanticVectorCleanup(confirmed_absent=not remaining_ids)

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
