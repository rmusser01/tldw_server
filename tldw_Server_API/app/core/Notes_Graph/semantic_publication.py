"""Fenced cross-store publication for the Notes semantic index."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticChunkRecord,
    SemanticGenerationIntegrity,
    SemanticIndexingError,
    SemanticManifestPublication,
    SemanticWorkItem,
    SemanticWorkKind,
)

from .semantic_content import SemanticChunkInput
from .semantic_vectors import SemanticVector


@dataclass(frozen=True, slots=True)
class SemanticExecutionFence:
    """Complete pinned identity required for content reads and publication."""

    owner_user_id: str
    dataset_id: str
    generation_id: str
    generation_fencing_token: str
    configuration_revision: int
    capability_revision: str
    disclosure_hash: str
    provider: str
    model: str
    model_revision: str | None
    endpoint_origin_revision: str
    compatibility_hash: str | None
    dimensions: int | None
    vector_backend: str


@dataclass(frozen=True, slots=True)
class SemanticAuthorityState:
    """Fresh authority and capability facts returned by the application boundary."""

    user_exists: bool
    owner_authorized: bool
    semantic_manage_allowed: bool
    desired_enabled: bool
    owner_user_id: str
    dataset_id: str
    generation_id: str
    generation_fencing_token: str
    configuration_revision: int
    capability_revision: str
    disclosure_hash: str
    provider: str
    model: str
    model_revision: str | None
    endpoint_origin_revision: str
    endpoint_policy_allowed: bool
    compatibility_hash: str | None
    dimensions: int | None
    vector_backend: str
    vector_capable: bool


@dataclass(frozen=True, slots=True)
class SemanticPublicationReceipt:
    """ChaChaNotes activation result returned directly to the Jobs adapter."""

    receipt: str
    generation_id: str
    configuration_revision: int
    semantic_index_revision: int
    indexed_notes: int
    excluded_notes: int
    failed_notes: int
    published_chunks: int
    degraded: bool


class SemanticFenceRevalidator(Protocol):
    def __call__(
        self,
        fence: SemanticExecutionFence,
    ) -> SemanticAuthorityState | Awaitable[SemanticAuthorityState]: ...


class SemanticPublicationStore(Protocol):
    def publish_indexed_manifest(self, **kwargs: Any) -> SemanticManifestPublication | None: ...

    def publish_note_tombstone(self, **kwargs: Any) -> SemanticManifestPublication | None: ...

    def get_generation_integrity(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> SemanticGenerationIntegrity: ...

    def assert_generation_activatable(self, integrity: SemanticGenerationIntegrity) -> None: ...

    def activate_generation_verified(self, **kwargs: Any) -> Any: ...

    def authorize_obsolete_vector_cleanup(self, **kwargs: Any) -> bool: ...

    def complete_obsolete_vector_cleanup(self, **kwargs: Any) -> bool: ...

    def authorize_generation_cleanup(self, **kwargs: Any) -> bool: ...

    def complete_generation_cleanup(self, **kwargs: Any) -> bool: ...


class SemanticPublicationVectors(Protocol):
    async def upsert(
        self,
        dataset_id: str,
        generation_id: str,
        vectors: Sequence[SemanticVector],
    ) -> int: ...

    async def fetch(
        self,
        dataset_id: str,
        generation_id: str,
        vector_ids: Sequence[str],
    ) -> tuple[SemanticVector, ...]: ...

    async def delete_ids(
        self,
        dataset_id: str,
        generation_id: str,
        vector_ids: Sequence[str],
    ) -> Any: ...

    async def delete_generation(self, dataset_id: str, generation_id: str) -> Any: ...


def validate_execution_fence(
    fence: SemanticExecutionFence,
    current: SemanticAuthorityState,
) -> SemanticAuthorityState:
    """Compare every execution fence fact and fail with one stable code."""

    if not current.user_exists:
        raise SemanticIndexingError("notes_semantic_user_missing")
    if not current.owner_authorized:
        raise SemanticIndexingError("notes_semantic_owner_authority_revoked")
    if not current.semantic_manage_allowed:
        raise SemanticIndexingError("notes_semantic_manage_permission_revoked")
    if not current.desired_enabled:
        raise SemanticIndexingError("notes_semantic_index_disabled")
    if (
        current.owner_user_id != fence.owner_user_id
        or current.dataset_id != fence.dataset_id
    ):
        raise SemanticIndexingError("notes_semantic_owner_authority_revoked")
    if current.capability_revision != fence.capability_revision:
        raise SemanticIndexingError("notes_semantic_capability_drift")
    if current.disclosure_hash != fence.disclosure_hash:
        raise SemanticIndexingError("notes_semantic_disclosure_drift")
    if current.configuration_revision != fence.configuration_revision:
        raise SemanticIndexingError("notes_semantic_configuration_drift")
    if (
        current.generation_id != fence.generation_id
        or current.generation_fencing_token != fence.generation_fencing_token
    ):
        raise SemanticIndexingError("notes_semantic_generation_fence_mismatch")
    if current.provider != fence.provider or current.model != fence.model:
        raise SemanticIndexingError("notes_semantic_provider_model_drift")
    if current.model_revision != fence.model_revision:
        raise SemanticIndexingError("notes_semantic_model_revision_drift")
    if not current.endpoint_policy_allowed:
        raise SemanticIndexingError("notes_semantic_endpoint_policy_denied")
    if current.endpoint_origin_revision != fence.endpoint_origin_revision:
        raise SemanticIndexingError("notes_semantic_endpoint_drift")
    if current.compatibility_hash != fence.compatibility_hash:
        raise SemanticIndexingError("notes_semantic_compatibility_drift")
    if current.dimensions != fence.dimensions:
        raise SemanticIndexingError("notes_semantic_dimension_drift")
    if current.vector_backend != fence.vector_backend or not current.vector_capable:
        raise SemanticIndexingError("notes_semantic_vector_capability_drift")
    return current


async def revalidate_execution_fence(
    revalidate: SemanticFenceRevalidator,
    fence: SemanticExecutionFence,
) -> SemanticAuthorityState:
    current = revalidate(fence)
    if inspect.isawaitable(current):
        current = await current
    if not isinstance(current, SemanticAuthorityState):
        raise SemanticIndexingError("notes_semantic_authority_result_invalid")
    return validate_execution_fence(fence, current)


def _chunk_record(chunk: SemanticChunkInput) -> SemanticChunkRecord:
    return SemanticChunkRecord(
        chunk_id=chunk.vector_id,
        generation_id=chunk.generation_id,
        note_id=chunk.note_id,
        content_version=chunk.content_version,
        ordinal=chunk.ordinal,
        field=chunk.field,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        chunk_fingerprint=chunk.chunk_fingerprint,
        normalization_version=chunk.normalization_version,
        chunker_version=chunk.chunker_version,
    )


class SemanticPublicationService:
    """Order vectors before Notes manifests and verify activation fail closed."""

    def __init__(
        self,
        *,
        store: SemanticPublicationStore,
        vectors: SemanticPublicationVectors,
        revalidate: SemanticFenceRevalidator,
        clock: Callable[[], datetime],
        receipt_factory: Callable[[], str],
        max_cleanup_vectors: int = 10_000,
        max_vectors_per_publication: int = 200,
    ) -> None:
        if (
            type(max_cleanup_vectors) is not int
            or max_cleanup_vectors <= 0
            or type(max_vectors_per_publication) is not int
            or max_vectors_per_publication <= 0
        ):
            raise ValueError("notes_semantic_publication_limit_invalid")
        self._store = store
        self._vectors = vectors
        self._revalidate = revalidate
        self._clock = clock
        self._receipt_factory = receipt_factory
        self._max_cleanup_vectors = max_cleanup_vectors
        self._max_vectors_per_publication = max_vectors_per_publication

    async def publish_note(
        self,
        fence: SemanticExecutionFence,
        claim: SemanticWorkItem,
        chunks: Sequence[SemanticChunkInput],
        vectors: Sequence[SemanticVector],
    ) -> SemanticManifestPublication:
        """Upsert deterministic vectors, then CAS one Note manifest."""

        chunk_tuple = tuple(chunks)
        vector_tuple = tuple(vectors)
        expected_ids = tuple(chunk.vector_id for chunk in chunk_tuple)
        if not chunk_tuple or len(chunk_tuple) > self._max_vectors_per_publication:
            raise SemanticIndexingError("notes_semantic_publication_limit_exceeded")
        if tuple(vector.vector_id for vector in vector_tuple) != expected_ids:
            raise SemanticIndexingError("notes_semantic_vector_ids_mismatch")
        if (
            claim.owner_user_id != fence.owner_user_id
            or claim.dataset_id != fence.dataset_id
            or claim.kind is not SemanticWorkKind.INDEX_NOTE
            or claim.generation_id != fence.generation_id
            or claim.dirty_generation is None
            or not claim.claim_token
            or not claim.note_id
            or not chunk_tuple
            or any(
                chunk.generation_id != fence.generation_id
                or chunk.note_id != claim.note_id
                or chunk.content_version != chunk_tuple[0].content_version
                or chunk.content_fingerprint != chunk_tuple[0].content_fingerprint
                for chunk in chunk_tuple
            )
        ):
            raise SemanticIndexingError("notes_semantic_note_claim_invalid")
        await revalidate_execution_fence(self._revalidate, fence)
        written = await self._vectors.upsert(
            fence.dataset_id,
            fence.generation_id,
            vector_tuple,
        )
        if written != len(vector_tuple):
            raise SemanticIndexingError("notes_semantic_vector_upsert_incomplete")
        await revalidate_execution_fence(self._revalidate, fence)
        publication = await asyncio.to_thread(
            self._store.publish_indexed_manifest,
            owner_user_id=fence.owner_user_id,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            generation_fencing_token=fence.generation_fencing_token,
            expected_configuration_revision=fence.configuration_revision,
            work_id=claim.id,
            claim_token=claim.claim_token,
            work_fencing_token=claim.fencing_token,
            claimed_dirty_generation=claim.dirty_generation,
            content_version=chunk_tuple[0].content_version,
            content_fingerprint=chunk_tuple[0].content_fingerprint,
            chunks=tuple(_chunk_record(chunk) for chunk in chunk_tuple),
            now=self._clock(),
        )
        if publication is None:
            raise SemanticIndexingError("notes_semantic_note_claim_stale")
        return publication

    async def publish_tombstone(
        self,
        fence: SemanticExecutionFence,
        claim: SemanticWorkItem,
    ) -> SemanticManifestPublication:
        """Hide one Note manifest transactionally before physical cleanup."""

        if (
            claim.owner_user_id != fence.owner_user_id
            or claim.dataset_id != fence.dataset_id
            or claim.kind is not SemanticWorkKind.DELETE_NOTE_VECTORS
            or claim.generation_id != fence.generation_id
            or not claim.note_id
            or claim.dirty_generation is None
            or not claim.claim_token
        ):
            raise SemanticIndexingError("notes_semantic_note_claim_invalid")
        await revalidate_execution_fence(self._revalidate, fence)
        publication = await asyncio.to_thread(
            self._store.publish_note_tombstone,
            owner_user_id=fence.owner_user_id,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            generation_fencing_token=fence.generation_fencing_token,
            expected_configuration_revision=fence.configuration_revision,
            work_id=claim.id,
            claim_token=claim.claim_token,
            work_fencing_token=claim.fencing_token,
            claimed_dirty_generation=claim.dirty_generation,
            note_id=claim.note_id,
            now=self._clock(),
        )
        if publication is None:
            raise SemanticIndexingError("notes_semantic_note_claim_stale")
        return publication

    async def activate(
        self,
        fence: SemanticExecutionFence,
    ) -> SemanticPublicationReceipt:
        """Verify Notes and vectors, then atomically activate and return the receipt."""

        if fence.dimensions is None or fence.compatibility_hash is None:
            raise SemanticIndexingError("notes_semantic_generation_dimensions_unresolved")
        integrity = await asyncio.to_thread(
            self._store.get_generation_integrity,
            fence.dataset_id,
            fence.generation_id,
        )
        if integrity.generation_fencing_token != fence.generation_fencing_token:
            raise SemanticIndexingError("notes_semantic_generation_fence_mismatch")
        if (
            integrity.dimensions != fence.dimensions
            or integrity.compatibility_hash != fence.compatibility_hash
        ):
            raise SemanticIndexingError("notes_semantic_generation_identity_mismatch")
        self._store.assert_generation_activatable(integrity)
        fetched = await self._vectors.fetch(
            fence.dataset_id,
            fence.generation_id,
            integrity.vector_ids,
        )
        if tuple(vector.vector_id for vector in fetched) != integrity.vector_ids:
            raise SemanticIndexingError("notes_semantic_vector_integrity_mismatch")
        if any(len(vector.embedding) != fence.dimensions for vector in fetched):
            raise SemanticIndexingError("notes_semantic_vector_dimension_mismatch")
        await revalidate_execution_fence(self._revalidate, fence)
        receipt_value = self._receipt_factory()
        activated = await asyncio.to_thread(
            self._store.activate_generation_verified,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            expected_configuration_revision=fence.configuration_revision,
            generation_fencing_token=fence.generation_fencing_token,
            expected_manifest_hash=integrity.manifest_hash,
            expected_vector_ids=integrity.vector_ids,
            expected_dimensions=fence.dimensions,
            expected_compatibility_hash=fence.compatibility_hash,
            publication_receipt=receipt_value,
            now=self._clock(),
        )
        if activated is None:
            raise SemanticIndexingError("notes_semantic_activation_fence_lost")
        return SemanticPublicationReceipt(
            receipt=receipt_value,
            generation_id=fence.generation_id,
            configuration_revision=activated.configuration_revision,
            semantic_index_revision=activated.semantic_index_revision,
            indexed_notes=integrity.indexed_note_count,
            excluded_notes=integrity.excluded_note_count,
            failed_notes=integrity.failed_note_count,
            published_chunks=integrity.published_chunk_count,
            degraded=integrity.degraded,
        )

    async def cleanup_obsolete(
        self,
        fence: SemanticExecutionFence,
        publication: SemanticManifestPublication,
    ) -> bool:
        """Delete only IDs proven obsolete after manifest visibility changed."""

        vector_ids = publication.old_vector_ids
        if len(vector_ids) > self._max_cleanup_vectors:
            raise SemanticIndexingError("notes_semantic_cleanup_limit_exceeded")
        if not vector_ids:
            return True
        authorized = await asyncio.to_thread(
            self._store.authorize_obsolete_vector_cleanup,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            note_id=publication.note_id,
            dirty_generation=publication.dirty_generation,
            vector_ids=vector_ids,
        )
        if not authorized:
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        cleanup = await self._vectors.delete_ids(
            fence.dataset_id,
            fence.generation_id,
            vector_ids,
        )
        if not bool(getattr(cleanup, "confirmed_absent", False)):
            raise SemanticIndexingError("notes_semantic_cleanup_unconfirmed")
        return await asyncio.to_thread(
            self._store.complete_obsolete_vector_cleanup,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            note_id=publication.note_id,
            dirty_generation=publication.dirty_generation,
            vector_ids=vector_ids,
            now=self._clock(),
        )

    async def cleanup_generation(self, claim: SemanticWorkItem) -> bool:
        """Execute one claimed delayed cleanup against only its fenced generation."""

        if (
            claim.kind.value != "delete_generation"
            or claim.note_id is not None
            or claim.generation_id is None
            or claim.dirty_generation is not None
            or not claim.claim_token
        ):
            raise SemanticIndexingError("notes_semantic_cleanup_claim_invalid")
        authorized = await asyncio.to_thread(
            self._store.authorize_generation_cleanup,
            dataset_id=claim.dataset_id,
            work_id=claim.id,
            generation_id=claim.generation_id,
            claim_token=claim.claim_token,
            fencing_token=claim.fencing_token,
        )
        if not authorized:
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        cleanup = await self._vectors.delete_generation(
            claim.dataset_id,
            claim.generation_id,
        )
        if not bool(getattr(cleanup, "confirmed_absent", False)):
            raise SemanticIndexingError("notes_semantic_cleanup_unconfirmed")
        completed = await asyncio.to_thread(
            self._store.complete_generation_cleanup,
            dataset_id=claim.dataset_id,
            work_id=claim.id,
            generation_id=claim.generation_id,
            claim_token=claim.claim_token,
            fencing_token=claim.fencing_token,
            now=self._clock(),
        )
        if not completed:
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        return True


__all__ = [
    "SemanticAuthorityState",
    "SemanticExecutionFence",
    "SemanticFenceRevalidator",
    "SemanticIndexingError",
    "SemanticPublicationReceipt",
    "SemanticPublicationService",
    "revalidate_execution_fence",
    "validate_execution_fence",
]
