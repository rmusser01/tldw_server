"""Fenced cross-store publication for the Notes semantic index."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from .semantic_endpoint import canonical_semantic_endpoint_origin
from .semantic_observability import record_semantic_cleanup_retry
from .semantic_vectors import SemanticVector


def _redacted_repr(value: object) -> str:
    return f"{type(value).__name__}(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
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
    endpoint_origin: str
    credential_source: str
    endpoint_origin_revision: str
    compatibility_hash: str | None
    dimensions: int | None
    vector_backend: str

    def __post_init__(self) -> None:
        if (
            canonical_semantic_endpoint_origin(self.endpoint_origin)
            != self.endpoint_origin
        ):
            raise ValueError("notes_semantic_endpoint_origin_invalid")
        if self.credential_source not in {"user", "server_default"}:
            raise ValueError("notes_semantic_credential_scope_invalid")

    __repr__ = _redacted_repr


@dataclass(frozen=True, slots=True, repr=False)
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
    endpoint_origin: str
    credential_source: str
    endpoint_origin_revision: str
    endpoint_policy_allowed: bool
    compatibility_hash: str | None
    dimensions: int | None
    vector_backend: str
    vector_capable: bool

    __repr__ = _redacted_repr


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


BeforeSemanticSideEffect = Callable[[], object]


async def _before_side_effect(callback: BeforeSemanticSideEffect | None) -> None:
    if callback is None:
        return
    result = callback()
    if inspect.isawaitable(result):
        await result


class SemanticPublicationStore(Protocol):
    def stage_obsolete_vector_cleanup(self, **kwargs: Any) -> int: ...

    def authorize_note_vector_upsert(self, **kwargs: Any) -> bool: ...

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

    def claim_obsolete_vector_cleanup_batch(self, **kwargs: Any) -> Any: ...

    def authorize_obsolete_vector_claim(self, **kwargs: Any) -> bool: ...

    def complete_obsolete_vector_claim(self, **kwargs: Any) -> bool: ...

    def release_obsolete_vector_claim(self, **kwargs: Any) -> bool: ...

    def retry_obsolete_vector_cleanup(self, **kwargs: Any) -> bool: ...

    def rearm_exhausted_obsolete_vector_cleanup(self, **kwargs: Any) -> int: ...

    def authorize_generation_cleanup(self, **kwargs: Any) -> bool: ...

    def complete_generation_cleanup(self, **kwargs: Any) -> bool: ...

    def list_generation_cleanup_vector_ids(self, **kwargs: Any) -> tuple[str, ...] | None: ...

    def complete_generation_vector_cleanup_page(self, **kwargs: Any) -> bool: ...


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
    if (
        canonical_semantic_endpoint_origin(current.endpoint_origin)
        != current.endpoint_origin
        or current.endpoint_origin != fence.endpoint_origin
    ):
        raise SemanticIndexingError("notes_semantic_endpoint_origin_drift")
    if current.credential_source != fence.credential_source:
        raise SemanticIndexingError("notes_semantic_credential_scope_drift")
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
    if inspect.iscoroutinefunction(revalidate):
        current = revalidate(fence)
    else:
        current = await asyncio.to_thread(revalidate, fence)
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


async def run_reconciled_transaction(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Drain a transactional thread; a committed result wins over cancellation."""

    operation = asyncio.create_task(asyncio.to_thread(function, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            current = asyncio.current_task()
            uncancel = getattr(current, "uncancel", None)
            if callable(uncancel):
                uncancel()
            continue
        except BaseException:
            if cancellation is not None:
                raise cancellation from None
            raise
    if cancellation is not None and result is None:
        raise cancellation
    return result


async def run_quiescent_operation(
    operation: Awaitable[Any],
    *,
    committed_result_wins: bool = False,
) -> Any:
    """Drain a started async side effect before resolving caller cancellation."""

    task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.done():
                if task.cancelled():
                    if cancellation is not None:
                        raise cancellation from None
                    raise
                if cancellation is None:
                    cancellation = exc
                if task.exception() is not None:
                    raise cancellation from None
                result = task.result()
                break
            if cancellation is None:
                cancellation = exc
            current = asyncio.current_task()
            uncancel = getattr(current, "uncancel", None)
            if callable(uncancel):
                uncancel()
        except BaseException:
            if cancellation is not None:
                raise cancellation from None
            raise
    if cancellation is not None and not committed_result_wins:
        raise cancellation
    return result


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
        store_call: Callable[..., Awaitable[Any]] | None = None,
        backend: str = "unavailable",
    ) -> None:
        if (
            type(max_cleanup_vectors) is not int
            or max_cleanup_vectors <= 0
            or type(max_vectors_per_publication) is not int
            or max_vectors_per_publication <= 0
        ):
            raise ValueError("notes_semantic_publication_limit_invalid")
        if backend not in {"chromadb", "pgvector", "unavailable"}:
            raise ValueError("notes_semantic_vector_backend_invalid")
        self._store = store
        self._vectors = vectors
        self._revalidate = revalidate
        self._clock = clock
        self._receipt_factory = receipt_factory
        self._max_cleanup_vectors = max_cleanup_vectors
        self._max_vectors_per_publication = max_vectors_per_publication
        self._store_call = store_call
        self._backend = backend

    async def _store_transaction(
        self,
        function: Callable[..., Any],
        /,
        **kwargs: Any,
    ) -> Any:
        if self._store_call is not None:
            return await self._store_call(function, **kwargs)
        return await run_reconciled_transaction(function, **kwargs)

    async def _release_obsolete_claim(self, claim: Any) -> bool:
        eligible_at = self._clock() + timedelta(seconds=1)
        return bool(
            await self._store_transaction(
                self._store.release_obsolete_vector_claim,
                dataset_id=claim.dataset_id,
                ledger_ids=claim.ledger_ids,
                claim_token=claim.claim_token,
                now=eligible_at,
            )
        )

    async def _retry_obsolete_claim(self, claim: Any, *, error_code: str) -> bool:
        now = self._clock()
        delay_seconds = min(300, 2 ** min(int(claim.attempt_count), 8))
        committed = bool(
            await self._store_transaction(
                self._store.retry_obsolete_vector_cleanup,
                dataset_id=claim.dataset_id,
                ledger_ids=claim.ledger_ids,
                claim_token=claim.claim_token,
                error_code=error_code,
                retry_at=now + timedelta(seconds=delay_seconds),
                now=now,
            )
        )
        if committed:
            record_semantic_cleanup_retry(status="failed", backend=self._backend)
        return committed

    async def _claim_obsolete_batch(
        self,
        *,
        dataset_id: str,
        generation_id: str,
    ) -> Any:
        now = self._clock()
        claim = await self._store_transaction(
            self._store.claim_obsolete_vector_cleanup_batch,
            dataset_id=dataset_id,
            generation_id=generation_id,
            limit=self._max_cleanup_vectors,
            now=now,
        )
        if claim is not None:
            return claim
        rearmed = await self._store_transaction(
            self._store.rearm_exhausted_obsolete_vector_cleanup,
            dataset_id=dataset_id,
            generation_id=generation_id,
            limit=self._max_cleanup_vectors,
            now=now,
        )
        if not rearmed:
            return None
        return await self._store_transaction(
            self._store.claim_obsolete_vector_cleanup_batch,
            dataset_id=dataset_id,
            generation_id=generation_id,
            limit=self._max_cleanup_vectors,
            now=now,
        )

    async def publish_note(
        self,
        fence: SemanticExecutionFence,
        claim: SemanticWorkItem,
        chunks: Sequence[SemanticChunkInput],
        vectors: Sequence[SemanticVector],
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
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
        await _before_side_effect(before_side_effect)
        staged = await self._store_transaction(
            self._store.stage_obsolete_vector_cleanup,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            vector_ids=expected_ids,
            source_kind="unpublished",
            note_id=claim.note_id,
            dirty_generation=claim.dirty_generation,
            now=self._clock(),
        )
        if staged != len(expected_ids):
            raise SemanticIndexingError("notes_semantic_cleanup_claim_conflict")
        await _before_side_effect(before_side_effect)
        authorized = await self._store_transaction(
            self._store.authorize_note_vector_upsert,
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
        if not authorized:
            raise SemanticIndexingError("notes_semantic_note_claim_stale")
        written = await run_quiescent_operation(
            self._vectors.upsert(
                fence.dataset_id,
                fence.generation_id,
                vector_tuple,
            )
        )
        if written != len(vector_tuple):
            raise SemanticIndexingError("notes_semantic_vector_upsert_incomplete")
        await revalidate_execution_fence(self._revalidate, fence)
        await _before_side_effect(before_side_effect)
        publication = await self._store_transaction(
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
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
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
        await _before_side_effect(before_side_effect)
        publication = await self._store_transaction(
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
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
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
        fetched_vectors: list[SemanticVector] = []
        for offset in range(0, len(integrity.vector_ids), self._max_cleanup_vectors):
            page_ids = integrity.vector_ids[offset : offset + self._max_cleanup_vectors]
            await _before_side_effect(before_side_effect)
            fetched_vectors.extend(
                await self._vectors.fetch(
                    fence.dataset_id,
                    fence.generation_id,
                    page_ids,
                )
            )
        fetched = tuple(fetched_vectors)
        if tuple(vector.vector_id for vector in fetched) != integrity.vector_ids:
            raise SemanticIndexingError("notes_semantic_vector_integrity_mismatch")
        if any(len(vector.embedding) != fence.dimensions for vector in fetched):
            raise SemanticIndexingError("notes_semantic_vector_dimension_mismatch")
        await revalidate_execution_fence(self._revalidate, fence)
        receipt_value = self._receipt_factory()
        await _before_side_effect(before_side_effect)
        activated = await self._store_transaction(
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
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> bool:
        """Delete only IDs proven obsolete after manifest visibility changed."""

        claim = await self._claim_obsolete_batch(
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
        )
        if claim is None:
            return not publication.old_vector_ids
        vector_ids = tuple(claim.vector_ids)
        authorized = await self._store_transaction(
            self._store.authorize_obsolete_vector_claim,
            dataset_id=fence.dataset_id,
            ledger_ids=claim.ledger_ids,
            claim_token=claim.claim_token,
        )
        if not authorized:
            await self._release_obsolete_claim(claim)
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        try:
            await _before_side_effect(before_side_effect)
            cleanup = await self._vectors.delete_ids(
                fence.dataset_id,
                fence.generation_id,
                vector_ids,
            )
        except asyncio.CancelledError:
            await self._release_obsolete_claim(claim)
            raise
        except Exception:  # noqa: BLE001 - sanitize injected vector-backend failures
            await self._retry_obsolete_claim(
                claim,
                error_code="vector_backend_failure",
            )
            raise SemanticIndexingError("notes_semantic_cleanup_backend_failed") from None
        if not bool(getattr(cleanup, "confirmed_absent", False)):
            await self._retry_obsolete_claim(
                claim,
                error_code="vector_cleanup_unconfirmed",
            )
            raise SemanticIndexingError("notes_semantic_cleanup_unconfirmed")
        await _before_side_effect(before_side_effect)
        completed = bool(
            await self._store_transaction(
                self._store.complete_obsolete_vector_claim,
                dataset_id=fence.dataset_id,
                ledger_ids=claim.ledger_ids,
                claim_token=claim.claim_token,
            )
        )
        if not completed:
            await self._release_obsolete_claim(claim)
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        return True

    async def cleanup_generation(
        self,
        claim: SemanticWorkItem,
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> bool:
        """Execute one claimed delayed cleanup against only its fenced generation."""

        if (
            claim.kind.value != "delete_generation"
            or claim.note_id is not None
            or claim.generation_id is None
            or claim.dirty_generation is not None
            or not claim.claim_token
        ):
            raise SemanticIndexingError("notes_semantic_cleanup_claim_invalid")
        authorized = await self._store_transaction(
            self._store.authorize_generation_cleanup,
            dataset_id=claim.dataset_id,
            work_id=claim.id,
            generation_id=claim.generation_id,
            claim_token=claim.claim_token,
            fencing_token=claim.fencing_token,
        )
        if not authorized:
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        obsolete = await self._claim_obsolete_batch(
            dataset_id=claim.dataset_id,
            generation_id=claim.generation_id,
        )
        if obsolete is not None:
            if not await self._store_transaction(
                self._store.authorize_obsolete_vector_claim,
                dataset_id=claim.dataset_id,
                ledger_ids=obsolete.ledger_ids,
                claim_token=obsolete.claim_token,
            ):
                await self._release_obsolete_claim(obsolete)
                raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
            try:
                await _before_side_effect(before_side_effect)
                cleanup = await self._vectors.delete_ids(
                    claim.dataset_id,
                    claim.generation_id,
                    obsolete.vector_ids,
                )
            except asyncio.CancelledError:
                await self._release_obsolete_claim(obsolete)
                raise
            except Exception:  # noqa: BLE001 - sanitize injected vector-backend failures
                await self._retry_obsolete_claim(
                    obsolete,
                    error_code="vector_backend_failure",
                )
                raise SemanticIndexingError(
                    "notes_semantic_cleanup_backend_failed"
                ) from None
            if not bool(getattr(cleanup, "confirmed_absent", False)):
                await self._retry_obsolete_claim(
                    obsolete,
                    error_code="vector_cleanup_unconfirmed",
                )
                raise SemanticIndexingError("notes_semantic_cleanup_unconfirmed")
            await _before_side_effect(before_side_effect)
            if not await self._store_transaction(
                self._store.complete_obsolete_vector_claim,
                dataset_id=claim.dataset_id,
                ledger_ids=obsolete.ledger_ids,
                claim_token=obsolete.claim_token,
            ):
                await self._release_obsolete_claim(obsolete)
                raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
            return False
        vector_ids = await self._store_transaction(
            self._store.list_generation_cleanup_vector_ids,
            dataset_id=claim.dataset_id,
            work_id=claim.id,
            generation_id=claim.generation_id,
            claim_token=claim.claim_token,
            fencing_token=claim.fencing_token,
            limit=self._max_cleanup_vectors,
        )
        if vector_ids is None:
            raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
        if vector_ids:
            await _before_side_effect(before_side_effect)
            cleanup = await self._vectors.delete_ids(
                claim.dataset_id,
                claim.generation_id,
                vector_ids,
            )
            if not bool(getattr(cleanup, "confirmed_absent", False)):
                raise SemanticIndexingError("notes_semantic_cleanup_unconfirmed")
            await _before_side_effect(before_side_effect)
            if not await self._store_transaction(
                self._store.complete_generation_vector_cleanup_page,
                dataset_id=claim.dataset_id,
                work_id=claim.id,
                generation_id=claim.generation_id,
                claim_token=claim.claim_token,
                fencing_token=claim.fencing_token,
                vector_ids=vector_ids,
            ):
                raise SemanticIndexingError("notes_semantic_cleanup_fence_lost")
            remaining = await self._store_transaction(
                self._store.list_generation_cleanup_vector_ids,
                dataset_id=claim.dataset_id,
                work_id=claim.id,
                generation_id=claim.generation_id,
                claim_token=claim.claim_token,
                fencing_token=claim.fencing_token,
                limit=1,
            )
            if remaining:
                return False
        await _before_side_effect(before_side_effect)
        cleanup = await self._vectors.delete_generation(
            claim.dataset_id,
            claim.generation_id,
        )
        if not bool(getattr(cleanup, "confirmed_absent", False)):
            raise SemanticIndexingError("notes_semantic_cleanup_unconfirmed")
        await _before_side_effect(before_side_effect)
        completed = await self._store_transaction(
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
    "run_quiescent_operation",
    "run_reconciled_transaction",
    "validate_execution_fence",
]
