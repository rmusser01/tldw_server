"""App-managed and standalone Jobs worker for Notes semantic indexing."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.core.AuthNZ.permissions import NOTES_GRAPH_SEMANTIC_MANAGE
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticGenerationState,
    SemanticIndexingError,
    SemanticManifestPublication,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Notes_Graph.semantic_api import (
    load_semantic_settings,
    resolve_semantic_capabilities,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    SEMANTIC_CHUNKER_VERSION,
    SEMANTIC_NORMALIZATION_VERSION,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    NotesSemanticEmbedder,
    PendingSemanticConfig,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_indexing import (
    InitialGenerationRequest,
    NoteVersionRef,
    SemanticGenerationBuilder,
    VersionedNoteSnapshot,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_jobs import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
    SemanticJobCancelled,
    SemanticJobHandler,
    SemanticJobsError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_observability import (
    emit_semantic_audit_event,
    record_semantic_build_metrics,
    record_semantic_cancellation,
    record_semantic_denial,
    record_semantic_failure,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticAuthorityState,
    SemanticExecutionFence,
    SemanticPublicationService,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    create_semantic_vector_store,
)


def build_worker_config(*, worker_id: str) -> WorkerConfig:
    """Return the exact dedicated WorkerSDK policy for semantic Jobs."""

    return WorkerConfig(
        domain=JOB_DOMAIN,
        queue=JOB_QUEUE,
        worker_id=worker_id,
        lease_seconds=int(os.getenv("NOTES_SEMANTIC_LEASE_SECONDS", "180") or "180"),
        renew_threshold_seconds=15,
        renew_jitter_seconds=0,
        retry_on_exception=False,
        bind_completion_token=True,
    )


async def _open_owner_database(owner_user_id: str) -> Any:
    try:
        user_id = int(owner_user_id)
    except (TypeError, ValueError) as exc:
        raise SemanticIndexingError("notes_semantic_job_owner_invalid") from exc
    if user_id <= 0 or str(user_id) != owner_user_id:
        raise SemanticIndexingError("notes_semantic_job_owner_invalid")
    return await get_chacha_db_for_user_id(user_id, client_id=owner_user_id)


def _close_database(db: Any) -> None:
    release = getattr(db, "release_context_connection", None)
    close = release if callable(release) else getattr(db, "close_connection", None)
    if callable(close):
        close()


class _ProductionNoteReader:
    def __init__(self, db: Any) -> None:
        self._store = db.note_store

    async def list_note_versions(
        self,
        owner_user_id: str,
        dataset_id: str,
        *,
        limit: int,
    ) -> tuple[NoteVersionRef, ...]:
        del owner_user_id, dataset_id
        rows = await asyncio.to_thread(
            self._store.list_semantic_note_versions,
            limit=limit,
        )
        return tuple(NoteVersionRef(note_id, version) for note_id, version in rows)

    async def read_note_version(
        self,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        content_version: int,
    ) -> VersionedNoteSnapshot | None:
        del owner_user_id, dataset_id
        row = await asyncio.to_thread(
            self._store.get_semantic_note_version,
            note_id=note_id,
            content_version=content_version,
        )
        if row is None:
            return None
        return VersionedNoteSnapshot(
            note_id=str(row["id"]),
            title=row.get("title"),
            content=row.get("content"),
            content_version=int(row["version"]),
        )


def _compatibility_hash(
    *,
    provider: str,
    model: str,
    model_revision: str | None,
    vector_backend: str,
    dimensions: int,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "model_revision": model_revision,
        "vector_backend": vector_backend,
        "metric": "cosine",
        "resolved_dimensions": dimensions,
        "normalization_version": SEMANTIC_NORMALIZATION_VERSION,
        "chunker_version": SEMANTIC_CHUNKER_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


async def _build_vector_store(
    *,
    db: Any,
    owner_user_id: str,
    backend_name: str,
    settings: SemanticIndexSettings,
) -> Any:
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


class ProductionSemanticRuntime:
    """Concrete adapter over Task 6 indexing, publication, and cleanup services."""

    def __init__(
        self,
        *,
        db: Any,
        owner_user_id: str,
        dataset_id: str,
        configuration_revision: int,
        generation_id: str | None,
        root_job_id: str,
        settings: SemanticIndexSettings,
        vectors: Any | None = None,
    ) -> None:
        self._db = db
        self._store = db.note_semantic_store
        self._owner_user_id = owner_user_id
        self._dataset_id = dataset_id
        self._configuration_revision = configuration_revision
        self._generation_id = generation_id
        self._root_job_id = root_job_id
        self._vectors = vectors
        self._settings = settings
        self._publication: SemanticPublicationService | None = None
        self._builder: SemanticGenerationBuilder | None = None
        if vectors is not None:
            self._configure_services(vectors)

    def _configure_services(self, vectors: Any) -> None:
        self._vectors = vectors
        self._publication = SemanticPublicationService(
            store=self._store,
            vectors=vectors,
            revalidate=self._revalidate,
            clock=lambda: datetime.now(timezone.utc),
            receipt_factory=lambda: str(uuid4()),
            max_cleanup_vectors=self._settings.max_cleanup_vectors_per_run,
            max_vectors_per_publication=self._settings.max_chunks_per_note,
            backend=self._vector_backend(),
        )
        embedder = NotesSemanticEmbedder(
            dimension_cas=lambda _pending, _resolved: True,
            settings=self._settings,
        )
        self._builder = SemanticGenerationBuilder(
            store=self._store,
            note_reader=_ProductionNoteReader(self._db),
            embedder=embedder,
            vectors=vectors,
            revalidate=self._revalidate,
            compatibility_hash_for_dimension=lambda resolved: _compatibility_hash(
                provider=resolved.provider,
                model=resolved.model,
                model_revision=resolved.model_revision,
                vector_backend=self._vector_backend(),
                dimensions=resolved.dimensions,
            ),
            settings=self._settings,
            clock=lambda: datetime.now(timezone.utc),
            receipt_factory=lambda: str(uuid4()),
            backend=self._vector_backend(),
        )

    async def _ensure_services(self) -> None:
        if self._builder is not None and self._publication is not None:
            return
        config = self._store.get_configuration(self._dataset_id)
        if config is None or not config.vector_backend:
            raise SemanticIndexingError("notes_semantic_configuration_missing")
        vectors = await _build_vector_store(
            db=self._db,
            owner_user_id=self._owner_user_id,
            backend_name=config.vector_backend,
            settings=self._settings,
        )
        self._configure_services(vectors)

    def _services(self) -> tuple[SemanticGenerationBuilder, SemanticPublicationService]:
        if self._builder is None or self._publication is None:
            raise SemanticIndexingError("notes_semantic_worker_runtime_failed")
        return self._builder, self._publication

    def _vector_backend(self) -> str:
        config = self._store.get_configuration(self._dataset_id)
        return str(config.vector_backend or "") if config is not None else ""

    def _generation(self) -> Any:
        generation = None
        if self._generation_id is not None:
            generation = self._store.get_generation(self._dataset_id, self._generation_id)
        if generation is None:
            generation = self._store.get_generation_by_root_job_id(
                self._dataset_id,
                self._root_job_id,
            )
        if generation is None:
            raise SemanticIndexingError("notes_semantic_generation_missing")
        return generation

    def _fence(self) -> SemanticExecutionFence:
        config = self._store.get_configuration(self._dataset_id)
        generation = self._generation()
        if (
            config is None
            or config.configuration_revision != self._configuration_revision
            or generation.configuration_revision != self._configuration_revision
            or not generation.root_job_id
            or not config.capability_revision
            or not config.disclosure_hash
            or not config.provider
            or not config.model
            or not config.endpoint_origin_display
            or not config.endpoint_origin_revision
            or not config.vector_backend
            or config.model_revision != generation.model_revision
        ):
            raise SemanticIndexingError("notes_semantic_configuration_drift")
        return SemanticExecutionFence(
            owner_user_id=self._owner_user_id,
            dataset_id=self._dataset_id,
            generation_id=generation.id,
            generation_fencing_token=generation.root_job_id,
            configuration_revision=config.configuration_revision,
            capability_revision=config.capability_revision,
            disclosure_hash=config.disclosure_hash,
            provider=config.provider,
            model=config.model,
            model_revision=config.model_revision,
            endpoint_origin=config.endpoint_origin_display,
            credential_source="server_default",
            endpoint_origin_revision=config.endpoint_origin_revision,
            compatibility_hash=config.compatibility_hash,
            dimensions=config.dimensions,
            vector_backend=config.vector_backend,
        )

    async def _revalidate(self, fence: SemanticExecutionFence) -> SemanticAuthorityState:
        from tldw_Server_API.app.core.AuthNZ.rbac import user_has_permission

        user_exists = False
        try:
            users = await AuthnzUsersRepo.from_pool()
            user = await users.get_user_by_id(int(self._owner_user_id))
            user_exists = user is not None and bool(user.get("is_active", True))
        except (OSError, RuntimeError, TypeError, ValueError):
            user_exists = False
        if not user_exists:
            return SemanticAuthorityState(
                user_exists=False,
                owner_authorized=False,
                semantic_manage_allowed=False,
                desired_enabled=False,
                owner_user_id=fence.owner_user_id,
                dataset_id=fence.dataset_id,
                generation_id=fence.generation_id,
                generation_fencing_token=fence.generation_fencing_token,
                configuration_revision=fence.configuration_revision,
                capability_revision=fence.capability_revision,
                disclosure_hash=fence.disclosure_hash,
                provider=fence.provider,
                model=fence.model,
                model_revision=fence.model_revision,
                endpoint_origin=fence.endpoint_origin,
                credential_source=fence.credential_source,
                endpoint_origin_revision=fence.endpoint_origin_revision,
                endpoint_policy_allowed=False,
                compatibility_hash=fence.compatibility_hash,
                dimensions=fence.dimensions,
                vector_backend=fence.vector_backend,
                vector_capable=False,
            )
        config = self._store.get_configuration(self._dataset_id)
        generation = self._store.get_generation(self._dataset_id, fence.generation_id)
        current = resolve_semantic_capabilities(self._db, settings=self._settings)
        manage_allowed = False
        try:
            manage_allowed = await asyncio.to_thread(
                user_has_permission,
                int(self._owner_user_id),
                NOTES_GRAPH_SEMANTIC_MANAGE,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            manage_allowed = False
        return SemanticAuthorityState(
            user_exists=user_exists,
            owner_authorized=(
                str(self._store.owner_user_id) == self._owner_user_id
                and generation is not None
                and generation.owner_user_id == self._owner_user_id
                and generation.state in {SemanticGenerationState.STAGING, SemanticGenerationState.ACTIVE}
            ),
            semantic_manage_allowed=manage_allowed,
            desired_enabled=(config is not None and config.desired_state.value == "enabled"),
            owner_user_id=self._owner_user_id,
            dataset_id=self._dataset_id,
            generation_id=generation.id if generation is not None else "missing",
            generation_fencing_token=(str(generation.root_job_id or "") if generation is not None else ""),
            configuration_revision=(config.configuration_revision if config is not None else -1),
            capability_revision=current.capability_revision,
            disclosure_hash=current.disclosure_hash,
            provider=current.provider_label.lower(),
            model=current.model,
            model_revision=(
                current.model_revision
                if current.model_revision is not None
                else config.model_revision
                if config is not None
                else None
            ),
            endpoint_origin=current.endpoint_display or "",
            credential_source="server_default",
            endpoint_origin_revision=current.endpoint_origin_revision,
            endpoint_policy_allowed=current.endpoint_display is not None,
            compatibility_hash=(config.compatibility_hash if config is not None else None),
            dimensions=config.dimensions if config is not None else None,
            vector_backend=str(config.vector_backend or "") if config is not None else "",
            vector_capable=(
                current.indexing_available and config is not None and current.vector_backend == config.vector_backend
            ),
        )

    def _request(self) -> InitialGenerationRequest:
        fence = self._fence()
        return InitialGenerationRequest(
            fence=fence,
            embedding_config=PendingSemanticConfig(
                provider=fence.provider,
                model=fence.model,
                model_revision=fence.model_revision,
                endpoint_origin=fence.endpoint_origin,
                credential_source=fence.credential_source,
                consented=True,
                dimensions=fence.dimensions,
            ),
        )

    def _integrity_result(self, generation_id: str) -> dict[str, Any]:
        integrity = self._store.get_generation_integrity(
            self._dataset_id,
            generation_id,
        )
        return {
            "state": "completed",
            "indexed_notes": integrity.indexed_note_count,
            "excluded_notes": integrity.excluded_note_count,
            "failed_notes": integrity.failed_note_count,
            "published_chunks": integrity.published_chunk_count,
            "cleanup_complete": not self._store.has_pending_cleanup(self._dataset_id),
            "error_code": integrity.terminal_error_code,
        }

    async def recover(self, **kwargs: Any) -> dict[str, Any] | None:
        mode = str(kwargs["mode"])
        if mode == "delete":
            if not self._store.has_pending_cleanup(self._dataset_id):
                return {
                    "state": "completed",
                    "indexed_notes": 0,
                    "excluded_notes": 0,
                    "failed_notes": 0,
                    "published_chunks": 0,
                    "cleanup_complete": True,
                    "error_code": None,
                }
            return None
        generation = self._generation()
        if generation.state is SemanticGenerationState.ACTIVE:
            if mode in {"build", "rebuild"} or not self._store.has_pending_index_work(
                self._dataset_id,
                generation.id,
            ):
                return self._integrity_result(generation.id)
        return None

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        mode = str(kwargs["mode"])
        cancellation_requested = kwargs["cancellation_requested"]
        if mode != "delete" and not self._settings.indexing_enabled:
            record_semantic_denial("kill_switch")
            raise SemanticIndexingError("notes_semantic_indexing_disabled")
        if await cancellation_requested():
            raise SemanticJobCancelled()
        await self._ensure_services()
        builder, publication = self._services()

        async def before_side_effect() -> None:
            if await cancellation_requested():
                raise SemanticJobCancelled()

        if mode in {"build", "rebuild"}:
            receipt = await builder.build_initial_generation(
                self._request(),
                before_side_effect=before_side_effect,
            )
            return {
                "state": "completed",
                "indexed_notes": receipt.indexed_notes,
                "excluded_notes": receipt.excluded_notes,
                "failed_notes": receipt.failed_notes,
                "published_chunks": receipt.published_chunks,
                "cleanup_complete": not self._store.has_pending_cleanup(self._dataset_id),
                "error_code": None,
            }
        if mode in {"maintain", "retry_failed"}:
            generation = self._generation()
            if mode == "retry_failed":
                await asyncio.to_thread(
                    self._store.rearm_failed_notes,
                    dataset_id=self._dataset_id,
                    generation_id=generation.id,
                    limit=min(256, self._settings.max_active_notes),
                    now=datetime.now(timezone.utc),
                )
            integrity = await builder.maintain_generation(
                self._request(),
                before_side_effect=before_side_effect,
            )
            return self._integrity_result(integrity.generation_id)
        if mode == "delete":
            claims = await asyncio.to_thread(
                self._store.claim_generation_cleanup_batch,
                dataset_id=self._dataset_id,
                limit=100,
                now=datetime.now(timezone.utc),
            )
            for claim in claims:
                for _ in range(self._settings.max_retries + 1):
                    if await cancellation_requested():
                        raise SemanticJobCancelled()
                    if await publication.cleanup_generation(
                        claim,
                        before_side_effect=before_side_effect,
                    ):
                        break
            return {
                "state": "completed",
                "indexed_notes": 0,
                "excluded_notes": 0,
                "failed_notes": 0,
                "published_chunks": 0,
                "cleanup_complete": not self._store.has_pending_cleanup(self._dataset_id),
                "error_code": None,
            }
        raise SemanticIndexingError("notes_semantic_job_mode_invalid")

    async def cleanup_claim(self, claim: Any) -> bool:
        """Confirm one maintenance-owned delayed generation cleanup claim."""

        await self._ensure_services()
        _builder, publication = self._services()
        return await publication.cleanup_generation(claim)

    async def cleanup_obsolete_generation(self) -> bool:
        """Confirm one bounded v66 obsolete-vector cleanup batch."""

        generation = self._generation()
        marker = SemanticManifestPublication(
            note_id="maintenance",
            generation_id=generation.id,
            old_vector_ids=("maintenance",),
            new_vector_ids=(),
            dirty_generation=1,
            manifest_hash=None,
        )
        await self._ensure_services()
        _builder, publication = self._services()
        return await publication.cleanup_obsolete(self._fence(), marker)


async def build_production_runtime(**kwargs: Any) -> ProductionSemanticRuntime:
    """Recover exact root-job authority before constructing any vector adapter."""

    db = kwargs.pop("db")
    settings = kwargs.pop("settings")
    mode = str(kwargs.pop("mode"))
    config = db.note_semantic_store.get_configuration(kwargs["dataset_id"])
    if config is None or not config.vector_backend:
        raise SemanticIndexingError("notes_semantic_configuration_missing")
    if mode in {"build", "rebuild"}:
        admitted_revision = int(kwargs["configuration_revision"])
        generation = db.note_semantic_store.get_generation_by_root_job_id(
            kwargs["dataset_id"],
            kwargs["root_job_id"],
        )
        if generation is None:
            if config.configuration_revision != admitted_revision:
                raise SemanticIndexingError("notes_semantic_configuration_drift")
            try:
                generation = db.note_semantic_store.create_generation(
                    dataset_id=kwargs["dataset_id"],
                    configuration_revision=kwargs["configuration_revision"],
                    compatibility_hash=config.compatibility_hash,
                    dimension_state=config.dimension_state,
                    dimensions=config.dimensions,
                    root_job_id=kwargs["root_job_id"],
                    model_revision=config.model_revision,
                    now=datetime.now(timezone.utc),
                )
            except SemanticIndexingError as exc:
                if exc.code == "notes_semantic_run_cancelled":
                    raise SemanticJobCancelled() from None
                generation = db.note_semantic_store.get_generation_by_root_job_id(
                    kwargs["dataset_id"],
                    kwargs["root_job_id"],
                )
                if generation is None:
                    raise SemanticIndexingError("notes_semantic_generation_recovery_failed") from None
            except (OSError, RuntimeError, TypeError, ValueError):
                generation = db.note_semantic_store.get_generation_by_root_job_id(
                    kwargs["dataset_id"],
                    kwargs["root_job_id"],
                )
                if generation is None:
                    raise SemanticIndexingError("notes_semantic_generation_recovery_failed") from None
        desired_state = getattr(config.desired_state, "value", config.desired_state)
        exact_admission = (
            desired_state == "enabled"
            and config.configuration_revision == admitted_revision
            and generation.configuration_revision == admitted_revision
        )
        exact_dimension_transition = (
            desired_state == "enabled"
            and config.configuration_revision == admitted_revision + 1
            and generation.configuration_revision == config.configuration_revision
            and config.dimension_state.value == "resolved"
            and generation.dimension_state.value == "resolved"
            and config.dimensions == generation.dimensions
            and config.compatibility_hash == generation.compatibility_hash
            and config.model_revision == generation.model_revision
        )
        if generation.root_job_id != kwargs["root_job_id"] or not (exact_admission or exact_dimension_transition):
            raise SemanticIndexingError("notes_semantic_configuration_drift")
        kwargs["generation_id"] = generation.id
        kwargs["configuration_revision"] = config.configuration_revision
    elif mode in {"maintain", "retry_failed"}:
        generation_id = str(kwargs.get("generation_id") or "")
        generation = db.note_semantic_store.get_generation(
            kwargs["dataset_id"],
            generation_id,
        )
        if (
            generation is None
            or generation.configuration_revision != kwargs["configuration_revision"]
            or config.configuration_revision != kwargs["configuration_revision"]
            or generation.model_revision != config.model_revision
        ):
            raise SemanticIndexingError("notes_semantic_configuration_drift")
    return ProductionSemanticRuntime(
        db=db,
        settings=settings,
        **kwargs,
    )


async def _cancellation_requested(job: dict[str, Any], *, jobs: JobManager) -> bool:
    current = await asyncio.to_thread(
        jobs.get_job_or_archived_by_uuid,
        str(job.get("uuid") or ""),
        domain=JOB_DOMAIN,
        owner_user_id=str(job.get("owner_user_id") or ""),
    )
    return bool(current and (current.get("status") == "cancelled" or current.get("cancel_requested_at")))


async def handle_notes_semantic_index_job(
    job: dict[str, Any],
    *,
    jobs: JobManager,
    worker_id: str,
) -> dict[str, Any]:
    """Execute one semantic root Job against its owner-bound database."""

    owner = str(job.get("owner_user_id") or "")
    raw_payload = job.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    dataset_id = str(payload.get("dataset_id") or "")
    mode = str(payload.get("mode") or "build")
    operation = mode if mode in {"build", "rebuild", "maintain", "retry_failed", "delete"} else "build"
    started = time.perf_counter()
    backend = "unavailable"
    db = None

    def record_terminal(status: str) -> None:
        record_semantic_build_metrics(
            operation=operation,
            status=status,
            backend=backend,
            duration_seconds=time.perf_counter() - started,
            counts={
                "indexed": 0,
                "excluded": 0,
                "failed": 0,
                "dirty": 0,
                "pending": 0,
                "chunks": 0,
            },
        )

    try:
        db = await _open_owner_database(owner)
        config = db.note_semantic_store.get_configuration(dataset_id)
        configured_backend = str(getattr(config, "vector_backend", "") or "")
        if configured_backend in {"chromadb", "pgvector"}:
            backend = configured_backend
        settings = load_semantic_settings()
        handler = SemanticJobHandler(
            runtime_factory=lambda **kwargs: build_production_runtime(
                db=db,
                settings=settings,
                **kwargs,
            ),
            settings=settings,
        )
        result = await handler.handle(
            job,
            cancellation_requested=lambda: _cancellation_requested(job, jobs=jobs),
        )
        cleanup_complete = bool(result.get("cleanup_complete"))
        status = (
            "degraded"
            if int(result.get("excluded_notes") or 0)
            or int(result.get("failed_notes") or 0)
            or (mode == "delete" and not cleanup_complete)
            else "success"
        )
        counts = {
            "indexed": int(result.get("indexed_notes") or 0),
            "excluded": int(result.get("excluded_notes") or 0),
            "failed": int(result.get("failed_notes") or 0),
            "dirty": 0,
            "pending": 0,
            "chunks": int(result.get("published_chunks") or 0),
        }
        record_semantic_build_metrics(
            operation=operation,
            status=status,
            backend=backend,
            duration_seconds=time.perf_counter() - started,
            counts=counts,
        )
        if counts["failed"]:
            record_semantic_failure(
                component="provider",
                category="execution",
                backend=backend,
            )
        if mode != "delete" or cleanup_complete:
            try:
                await emit_semantic_audit_event(
                    owner_user_id=owner,
                    dataset_id=dataset_id,
                    event=(
                        "cleanup_completion"
                        if mode == "delete"
                        else "incremental_publication"
                        if mode in {"maintain", "retry_failed"}
                        else "generation_publication"
                    ),
                    status=status,
                    reason=("note_failed" if counts["failed"] else "note_excluded" if counts["excluded"] else "none"),
                    generation_id=(str(payload.get("generation_id")) if payload.get("generation_id") else None),
                    run_id=str(job.get("uuid") or "") or None,
                    counts=counts,
                )
            except Exception:  # noqa: BLE001 - publication is already authoritative
                logger.warning("Notes semantic worker audit persistence failed")
        return result
    except SemanticJobCancelled:
        record_terminal("cancelled")
        record_semantic_cancellation(operation)
        if all(job.get(key) is not None for key in ("id", "uuid", "lease_id")):
            await asyncio.to_thread(
                jobs.finalize_cancelled,
                int(job["id"]),
                reason="requested",
                expected_uuid=str(job["uuid"]),
                worker_id=worker_id,
                lease_id=str(job["lease_id"]),
            )
        raise
    except SemanticJobsError as exc:
        record_terminal("failed")
        record_semantic_failure(
            component="provider",
            category="unavailable",
            backend=backend,
        )
        raise SemanticIndexingError(exc.code) from None
    except SemanticIndexingError as exc:
        record_terminal("failed")
        category = (
            "configuration"
            if "configuration" in exc.code or "capability" in exc.code
            else "fence"
            if "fence" in exc.code
            else "unavailable"
            if "unavailable" in exc.code
            else "execution"
        )
        record_semantic_failure(
            component="provider" if "provider" in exc.code else "vector",
            category=category,
            backend=backend,
        )
        raise
    except Exception:  # noqa: BLE001 - WorkerSDK receives only an allowlisted code
        record_terminal("failed")
        record_semantic_failure(
            component="vector",
            category="unknown",
            backend=backend,
        )
        raise SemanticIndexingError("notes_semantic_worker_runtime_failed") from None
    finally:
        if db is not None:
            try:
                await asyncio.to_thread(_close_database, db)
            except Exception:  # noqa: BLE001 - WorkerSDK receives only a stable code
                raise SemanticIndexingError("notes_semantic_worker_runtime_failed") from None


def _build_sdk(*, jobs: JobManager, config: WorkerConfig) -> WorkerSDK:
    return WorkerSDK(jobs, config)


async def _run_worker(
    *,
    stop_event: asyncio.Event | None,
    handler: Any,
) -> None:
    worker_id = (os.getenv("NOTES_SEMANTIC_WORKER_ID") or f"notes-semantic-worker-{os.getpid()}").strip()
    jobs = JobManager()
    sdk = _build_sdk(jobs=jobs, config=build_worker_config(worker_id=worker_id))
    watcher: asyncio.Task[None] | None = None
    stopped = False

    def stop_sdk() -> None:
        nonlocal stopped
        if not stopped:
            stopped = True
            sdk.stop()

    if stop_event is not None:

        async def watch_stop() -> None:
            await stop_event.wait()
            stop_sdk()

        watcher = asyncio.create_task(watch_stop())
    logger.info("Notes semantic-index Jobs worker starting")
    try:
        await sdk.run(
            handler=lambda job: handler(job, jobs=jobs, worker_id=worker_id),
            job_type=JOB_TYPE,
            cancel_check=lambda job: _cancellation_requested(job, jobs=jobs),
        )
    finally:
        if stop_event is not None and stop_event.is_set():
            stop_sdk()
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher


async def run_notes_semantic_index_worker(
    stop_event: asyncio.Event | None = None,
) -> None:
    await _run_worker(
        stop_event=stop_event,
        handler=handle_notes_semantic_index_job,
    )


async def run_standalone_notes_semantic_index_worker(
    stop_event: asyncio.Event | None = None,
) -> None:
    await _run_worker(
        stop_event=stop_event,
        handler=handle_notes_semantic_index_job,
    )


__all__ = [
    "ProductionSemanticRuntime",
    "build_production_runtime",
    "build_worker_config",
    "handle_notes_semantic_index_job",
    "run_notes_semantic_index_worker",
    "run_standalone_notes_semantic_index_worker",
]
