"""Bounded async orchestration for fenced Notes semantic generations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGeneration,
    SemanticGenerationIntegrity,
    SemanticIndexConfig,
    SemanticIndexingError,
    SemanticSnapshotSeed,
    SemanticWorkItem,
    SemanticWorkKind,
)

from .semantic_content import (
    SemanticChunkInput,
    SemanticContentError,
    build_semantic_chunks,
    semantic_content_fingerprint,
)
from .semantic_embeddings import (
    PendingSemanticConfig,
    ResolvedDimension,
    ResolvedSemanticConfig,
    SemanticEmbeddingBatch,
    SemanticEmbeddingPlan,
    SemanticEmbeddingSystemError,
    plan_semantic_embedding_batches,
)
from .semantic_publication import (
    BeforeSemanticSideEffect,
    SemanticExecutionFence,
    SemanticFenceRevalidator,
    SemanticPublicationReceipt,
    SemanticPublicationService,
    _before_side_effect,
    revalidate_execution_fence,
    run_reconciled_transaction,
)
from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings
from .semantic_vectors import SemanticVector


@dataclass(frozen=True, slots=True)
class NoteVersionRef:
    """Content-free identity for one active Note snapshot member."""

    note_id: str
    content_version: int


@dataclass(frozen=True, slots=True)
class VersionedNoteSnapshot:
    """Ephemeral, versioned Note content returned by the authority boundary."""

    note_id: str
    title: str | None = field(repr=False)
    content: str | None = field(repr=False)
    content_version: int


@dataclass(frozen=True, slots=True)
class InitialGenerationRequest:
    """Pinned input for one initial generation build."""

    fence: SemanticExecutionFence
    embedding_config: PendingSemanticConfig


class SemanticNoteIndexingError(SemanticIndexingError):
    """A documented Note-local failure that may produce degraded activation."""

    _ALLOWED_CODES = frozenset(
        {
            "note_content_invalid",
            "note_content_unavailable",
        }
    )

    def __init__(self, code: str) -> None:
        if code not in self._ALLOWED_CODES:
            raise ValueError("notes_semantic_note_error_code_invalid")
        super().__init__(code)


@dataclass(slots=True)
class _RunBudget:
    settings: SemanticIndexSettings
    chunk_count: int = 0
    provider_bytes: int = 0
    provider_requests: int = 0

    def admit(self, chunks: Sequence[SemanticChunkInput]) -> SemanticEmbeddingPlan:
        plan = plan_semantic_embedding_batches(chunks, self.settings)
        next_chunks = self.chunk_count + plan.input_count
        next_bytes = self.provider_bytes + plan.total_bytes
        next_requests = self.provider_requests + plan.request_count
        if (
            next_chunks > self.settings.max_chunks_per_run
            or next_bytes > self.settings.max_provider_bytes_per_run
            or next_requests > self.settings.max_provider_requests_per_run
        ):
            raise SemanticIndexingError("notes_semantic_run_limit_exceeded")
        self.chunk_count = next_chunks
        self.provider_bytes = next_bytes
        self.provider_requests = next_requests
        return plan

    def reconcile_requests(
        self,
        reservation: SemanticEmbeddingPlan,
        actual_request_count: object,
    ) -> None:
        if (
            type(actual_request_count) is not int
            or actual_request_count < 0
            or actual_request_count > reservation.request_count
        ):
            raise SemanticIndexingError("notes_semantic_embedding_usage_invalid")
        self.provider_requests -= reservation.request_count - actual_request_count


class VersionedNoteReader(Protocol):
    async def list_note_versions(
        self,
        owner_user_id: str,
        dataset_id: str,
        *,
        limit: int,
    ) -> Sequence[NoteVersionRef]: ...

    async def read_note_version(
        self,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        content_version: int,
    ) -> VersionedNoteSnapshot | None: ...


class SemanticGenerationEmbedder(Protocol):
    async def resolve_dimensions(
        self,
        config: PendingSemanticConfig,
        *,
        user_id: str,
    ) -> ResolvedDimension: ...

    async def embed_chunks(
        self,
        chunks: Sequence[SemanticChunkInput],
        config: ResolvedSemanticConfig,
        *,
        user_id: str,
    ) -> SemanticEmbeddingBatch: ...


class SemanticGenerationVectors(Protocol):
    async def create_generation_storage(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> None: ...

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


class SemanticGenerationStore(Protocol):
    def get_configuration(self, dataset_id: str) -> SemanticIndexConfig | None: ...

    def get_generation(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> SemanticGeneration | None: ...

    def resolve_generation_dimensions(self, **kwargs: Any) -> SemanticGeneration | None: ...

    def seed_generation_snapshot(self, **kwargs: Any) -> bool: ...

    def claim_work_batch(self, **kwargs: Any) -> tuple[SemanticWorkItem, ...]: ...

    def release_work_claim(self, **kwargs: Any) -> bool: ...

    def fail_claimed_note(self, **kwargs: Any) -> bool: ...

    def get_generation_integrity(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> SemanticGenerationIntegrity: ...

    def fail_generation(self, **kwargs: Any) -> bool: ...


@dataclass(frozen=True, slots=True)
class _SnapshotPlan:
    refs: tuple[NoteVersionRef, ...]
    seeds: tuple[SemanticSnapshotSeed, ...]
    chunks_by_note: dict[str, tuple[SemanticChunkInput, ...]] = field(repr=False)


def _build_snapshot_note(
    fence: SemanticExecutionFence,
    snapshot: VersionedNoteSnapshot,
    settings: SemanticIndexSettings,
) -> tuple[str, tuple[SemanticChunkInput, ...]]:
    fingerprint = semantic_content_fingerprint(
        snapshot.title,
        snapshot.content,
        snapshot.content_version,
    )
    chunks = build_semantic_chunks(
        generation_id=fence.generation_id,
        note_id=snapshot.note_id,
        title=snapshot.title,
        content=snapshot.content,
        content_version=snapshot.content_version,
        settings=settings,
    )
    return fingerprint, chunks


class SemanticGenerationBuilder:
    """Resolve, seed, publish, converge, and activate one staging generation."""

    def __init__(
        self,
        *,
        store: SemanticGenerationStore,
        note_reader: VersionedNoteReader,
        embedder: SemanticGenerationEmbedder,
        vectors: SemanticGenerationVectors,
        revalidate: SemanticFenceRevalidator,
        compatibility_hash_for_dimension: Callable[[ResolvedDimension], str],
        settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
        clock: Callable[[], datetime],
        receipt_factory: Callable[[], str],
    ) -> None:
        self._store = store
        self._note_reader = note_reader
        self._embedder = embedder
        self._vectors = vectors
        self._revalidate = revalidate
        self._compatibility_hash_for_dimension = compatibility_hash_for_dimension
        self._settings = settings
        self._clock = clock
        self._publication = SemanticPublicationService(
            store=store,  # type: ignore[arg-type]
            vectors=vectors,
            revalidate=revalidate,
            clock=clock,
            receipt_factory=receipt_factory,
            max_cleanup_vectors=settings.max_cleanup_vectors_per_run,
            max_vectors_per_publication=settings.max_chunks_per_note,
        )

    async def build_initial_generation(
        self,
        request: InitialGenerationRequest,
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> SemanticPublicationReceipt:
        """Build and activate one bounded initial snapshot or fail it closed."""

        fence = request.fence
        try:
            fence, resolved_config = await self._resolve_generation(
                fence,
                request.embedding_config,
                before_side_effect=before_side_effect,
            )
            await revalidate_execution_fence(self._revalidate, fence)
            await _before_side_effect(before_side_effect)
            await self._vectors.create_generation_storage(
                fence.dataset_id,
                fence.generation_id,
            )
            run_budget = _RunBudget(self._settings)
            for _ in range(self._settings.max_retries + 1):
                try:
                    plan = await self._read_snapshot(
                        fence,
                        before_side_effect=before_side_effect,
                    )
                except SemanticIndexingError as exc:
                    if exc.code == "notes_semantic_snapshot_changed":
                        continue
                    raise
                await revalidate_execution_fence(self._revalidate, fence)
                await _before_side_effect(before_side_effect)
                seeded = await asyncio.to_thread(
                    self._store.seed_generation_snapshot,
                    dataset_id=fence.dataset_id,
                    generation_id=fence.generation_id,
                    expected_configuration_revision=fence.configuration_revision,
                    generation_fencing_token=fence.generation_fencing_token,
                    seeds=plan.seeds,
                    now=self._clock(),
                )
                if not seeded:
                    continue
                await self._publish_claimed_notes(
                    fence,
                    resolved_config,
                    plan,
                    run_budget=run_budget,
                    before_side_effect=before_side_effect,
                )
                integrity = await asyncio.to_thread(
                    self._store.get_generation_integrity,
                    fence.dataset_id,
                    fence.generation_id,
                )
                current_refs = await self._list_refs(
                    fence,
                    before_side_effect=before_side_effect,
                )
                if integrity.pending_note_count == 0 and current_refs == plan.refs:
                    return await self._publication.activate(
                        fence,
                        before_side_effect=before_side_effect,
                    )
            raise SemanticIndexingError("notes_semantic_convergence_exhausted")
        except asyncio.CancelledError:
            raise
        except SemanticEmbeddingSystemError:
            code = "notes_semantic_embedding_system_failure"
            await self._record_systemic_failure(fence, code)
            raise SemanticIndexingError(code) from None
        except SemanticIndexingError as exc:
            await self._record_systemic_failure(fence, exc.code)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize dependency failures at the boundary
            if getattr(exc, "failure_code", None) == "notes_semantic_run_cancelled":
                raise
            code = "notes_semantic_build_system_failure"
            await self._record_systemic_failure(fence, code)
            raise SemanticIndexingError(code) from None

    async def maintain_generation(
        self,
        request: InitialGenerationRequest,
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> SemanticGenerationIntegrity:
        """Publish bounded dirty or failed Note work into an active generation."""

        fence = request.fence
        fence, resolved_config = await self._resolve_generation(
            fence,
            request.embedding_config,
            before_side_effect=before_side_effect,
        )
        plan = await self._read_snapshot(
            fence,
            before_side_effect=before_side_effect,
        )
        await self._publish_claimed_notes(
            fence,
            resolved_config,
            plan,
            run_budget=_RunBudget(self._settings),
            before_side_effect=before_side_effect,
        )
        return await asyncio.to_thread(
            self._store.get_generation_integrity,
            fence.dataset_id,
            fence.generation_id,
        )

    async def _resolve_generation(
        self,
        fence: SemanticExecutionFence,
        pending: PendingSemanticConfig,
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> tuple[SemanticExecutionFence, ResolvedSemanticConfig]:
        if (
            pending.provider != fence.provider
            or pending.model != fence.model
            or pending.model_revision != fence.model_revision
            or pending.endpoint_origin != fence.endpoint_origin
            or pending.credential_source != fence.credential_source
        ):
            raise SemanticIndexingError("notes_semantic_execution_config_drift")
        authority = await revalidate_execution_fence(self._revalidate, fence)
        if fence.dimensions is None:
            await _before_side_effect(before_side_effect)
            resolved = await self._embedder.resolve_dimensions(
                pending,
                user_id=fence.owner_user_id,
            )
            if resolved.provider != fence.provider or resolved.model != fence.model:
                raise SemanticIndexingError("notes_semantic_provider_model_drift")
            if (
                fence.model_revision is not None
                and resolved.model_revision != fence.model_revision
            ):
                raise SemanticIndexingError("notes_semantic_model_revision_drift")
            if (
                resolved.endpoint_origin != authority.endpoint_origin
                or resolved.credential_source != authority.credential_source
            ):
                raise SemanticIndexingError("notes_semantic_execution_config_drift")
            compatibility_hash = await asyncio.to_thread(
                self._compatibility_hash_for_dimension,
                resolved,
            )
            config = await asyncio.to_thread(
                self._store.get_configuration,
                fence.dataset_id,
            )
            generation = await asyncio.to_thread(
                self._store.get_generation,
                fence.dataset_id,
                fence.generation_id,
            )
            already_resolved = (
                config is not None
                and generation is not None
                and config.dimension_state is SemanticDimensionState.RESOLVED
                and generation.dimension_state is SemanticDimensionState.RESOLVED
                and config.dimensions == resolved.dimensions
                and generation.dimensions == resolved.dimensions
                and config.compatibility_hash == compatibility_hash
                and generation.compatibility_hash == compatibility_hash
                and config.model_revision == resolved.model_revision
                and generation.model_revision == resolved.model_revision
            )
            if not already_resolved:
                await revalidate_execution_fence(self._revalidate, fence)
                await _before_side_effect(before_side_effect)
                generation = await asyncio.to_thread(
                    self._store.resolve_generation_dimensions,
                    dataset_id=fence.dataset_id,
                    generation_id=fence.generation_id,
                    expected_configuration_revision=fence.configuration_revision,
                    dimensions=resolved.dimensions,
                    compatibility_hash=compatibility_hash,
                    model_revision=resolved.model_revision,
                    now=self._clock(),
                )
                if generation is None:
                    raise SemanticIndexingError("notes_semantic_dimension_fence_lost")
                config = await asyncio.to_thread(
                    self._store.get_configuration,
                    fence.dataset_id,
                )
            if config is None or generation is None:
                raise SemanticIndexingError("notes_semantic_generation_missing")
            fence = replace(
                fence,
                configuration_revision=config.configuration_revision,
                model_revision=resolved.model_revision,
                compatibility_hash=compatibility_hash,
                dimensions=resolved.dimensions,
            )
        else:
            resolved = ResolvedDimension(
                dimensions=fence.dimensions,
                provider=fence.provider,
                model=fence.model,
                model_revision=fence.model_revision,
                endpoint_origin=authority.endpoint_origin,
                credential_source=authority.credential_source,
            )
        authority = await revalidate_execution_fence(self._revalidate, fence)
        resolved_config = ResolvedSemanticConfig(
            provider=resolved.provider,
            model=resolved.model,
            model_revision=resolved.model_revision,
            endpoint_origin=authority.endpoint_origin,
            credential_source=authority.credential_source,
            dimensions=resolved.dimensions,
        )
        await revalidate_execution_fence(self._revalidate, fence)
        return fence, resolved_config

    async def _list_refs(
        self,
        fence: SemanticExecutionFence,
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> tuple[NoteVersionRef, ...]:
        await revalidate_execution_fence(self._revalidate, fence)
        await _before_side_effect(before_side_effect)
        values = tuple(
            await self._note_reader.list_note_versions(
                fence.owner_user_id,
                fence.dataset_id,
                limit=self._settings.max_active_notes + 1,
            )
        )
        if len(values) > self._settings.max_active_notes:
            raise SemanticIndexingError("notes_semantic_active_note_limit_exceeded")
        refs = await asyncio.to_thread(
            lambda: tuple(sorted(values, key=lambda item: item.note_id))
        )
        if len({item.note_id for item in refs}) != len(refs):
            raise SemanticIndexingError("notes_semantic_snapshot_duplicate")
        if any(item.content_version < 1 for item in refs):
            raise SemanticIndexingError("notes_semantic_snapshot_version_invalid")
        return refs

    async def _read_snapshot(
        self,
        fence: SemanticExecutionFence,
        *,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> _SnapshotPlan:
        refs = await self._list_refs(
            fence,
            before_side_effect=before_side_effect,
        )
        seeds: list[SemanticSnapshotSeed] = []
        chunks_by_note: dict[str, tuple[SemanticChunkInput, ...]] = {}
        for ref in refs:
            await revalidate_execution_fence(self._revalidate, fence)
            await _before_side_effect(before_side_effect)
            snapshot = await self._note_reader.read_note_version(
                fence.owner_user_id,
                fence.dataset_id,
                ref.note_id,
                ref.content_version,
            )
            if (
                snapshot is None
                or snapshot.note_id != ref.note_id
                or snapshot.content_version != ref.content_version
            ):
                raise SemanticIndexingError("notes_semantic_snapshot_changed")
            try:
                fingerprint, chunks = await asyncio.to_thread(
                    _build_snapshot_note,
                    fence,
                    snapshot,
                    self._settings,
                )
            except SemanticContentError as exc:
                fingerprint = await asyncio.to_thread(
                    semantic_content_fingerprint,
                    snapshot.title,
                    snapshot.content,
                    snapshot.content_version,
                )
                seeds.append(
                    SemanticSnapshotSeed(
                        note_id=ref.note_id,
                        content_version=ref.content_version,
                        content_fingerprint=fingerprint,
                        state="excluded",
                        planned_chunk_count=0,
                        error_code=exc.code,
                    )
                )
                continue
            try:
                plan_semantic_embedding_batches(chunks, self._settings)
            except SemanticEmbeddingSystemError:
                raise SemanticIndexingError("notes_semantic_run_limit_exceeded") from None
            seeds.append(
                SemanticSnapshotSeed(
                    note_id=ref.note_id,
                    content_version=ref.content_version,
                    content_fingerprint=fingerprint,
                    state="pending",
                    planned_chunk_count=len(chunks),
                    error_code=None,
                )
            )
            chunks_by_note[ref.note_id] = chunks
        return _SnapshotPlan(refs, tuple(seeds), chunks_by_note)

    async def _publish_claimed_notes(
        self,
        fence: SemanticExecutionFence,
        config: ResolvedSemanticConfig,
        plan: _SnapshotPlan,
        *,
        run_budget: _RunBudget | None = None,
        before_side_effect: BeforeSemanticSideEffect | None = None,
    ) -> None:
        claim_limit = min(
            256,
            self._settings.max_active_notes,
            self._settings.max_provider_batch_inputs,
        )
        budget = run_budget or _RunBudget(self._settings)
        while True:
            claims = await asyncio.to_thread(
                self._store.claim_work_batch,
                dataset_id=fence.dataset_id,
                generation_id=fence.generation_id,
                kind=SemanticWorkKind.INDEX_NOTE,
                limit=claim_limit,
                now=self._clock(),
            )
            if not claims:
                return
            try:
                for claim in claims:
                    if claim.note_id is None:
                        raise SemanticIndexingError("notes_semantic_note_claim_invalid")
                    chunks = plan.chunks_by_note.get(claim.note_id)
                    if not chunks:
                        raise SemanticIndexingError("notes_semantic_note_claim_stale")
                    await revalidate_execution_fence(self._revalidate, fence)
                    try:
                        admitted = budget.admit(chunks)
                    except SemanticEmbeddingSystemError:
                        raise SemanticIndexingError(
                            "notes_semantic_run_limit_exceeded"
                        ) from None
                    try:
                        await _before_side_effect(before_side_effect)
                        batch = await self._embedder.embed_chunks(
                            chunks,
                            config,
                            user_id=fence.owner_user_id,
                        )
                    except SemanticNoteIndexingError as exc:
                        await revalidate_execution_fence(self._revalidate, fence)
                        await _before_side_effect(before_side_effect)
                        failed = await run_reconciled_transaction(
                            self._store.fail_claimed_note,
                            dataset_id=fence.dataset_id,
                            generation_id=fence.generation_id,
                            generation_fencing_token=fence.generation_fencing_token,
                            expected_configuration_revision=fence.configuration_revision,
                            work_id=claim.id,
                            claim_token=claim.claim_token,
                            work_fencing_token=claim.fencing_token,
                            claimed_dirty_generation=claim.dirty_generation,
                            note_id=claim.note_id,
                            error_code=exc.code,
                            now=self._clock(),
                        )
                        if not failed:
                            raise SemanticIndexingError(
                                "notes_semantic_note_claim_stale"
                            ) from exc
                        continue
                    if (
                        batch.provider != fence.provider
                        or batch.model != fence.model
                        or batch.model_revision != fence.model_revision
                        or batch.dimensions != fence.dimensions
                        or batch.endpoint_origin != fence.endpoint_origin
                        or batch.credential_source != fence.credential_source
                        or len(batch.vectors) != len(chunks)
                        or any(len(vector) != fence.dimensions for vector in batch.vectors)
                    ):
                        raise SemanticIndexingError(
                            "notes_semantic_embedding_identity_mismatch"
                        )
                    budget.reconcile_requests(
                        admitted,
                        batch.provider_request_count,
                    )
                    vectors = tuple(
                        SemanticVector(chunk.vector_id, tuple(vector))
                        for chunk, vector in zip(chunks, batch.vectors)
                    )
                    await self._publication.publish_note(
                        fence,
                        claim,
                        chunks,
                        vectors,
                        before_side_effect=before_side_effect,
                    )
            finally:
                for claim in claims:
                    if claim.claim_token:
                        await run_reconciled_transaction(
                            self._store.release_work_claim,
                            dataset_id=fence.dataset_id,
                            work_id=claim.id,
                            claim_token=claim.claim_token,
                            fencing_token=claim.fencing_token,
                            now=self._clock(),
                        )

    async def _record_systemic_failure(
        self,
        fence: SemanticExecutionFence,
        code: str,
    ) -> None:
        try:
            await revalidate_execution_fence(self._revalidate, fence)
        except SemanticIndexingError:
            return
        await asyncio.to_thread(
            self._store.fail_generation,
            dataset_id=fence.dataset_id,
            generation_id=fence.generation_id,
            generation_fencing_token=fence.generation_fencing_token,
            expected_configuration_revision=fence.configuration_revision,
            error_code=code,
            now=self._clock(),
        )


__all__ = [
    "InitialGenerationRequest",
    "NoteVersionRef",
    "SemanticGenerationBuilder",
    "SemanticNoteIndexingError",
    "VersionedNoteSnapshot",
]
