"""Fail-closed Notes semantic cleanup for data-subject erasure."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticIndexingError,
)

from .semantic_observability import (
    emit_semantic_audit_event,
    record_semantic_dsr_metrics,
)
from .semantic_publication import (
    SemanticPublicationService,
    run_quiescent_operation,
)
from .semantic_settings import (
    DEFAULT_SEMANTIC_INDEX_SETTINGS,
    SemanticIndexSettings,
)
from .semantic_vectors import SemanticVectorError, create_semantic_vector_store

VectorStoreFactory = Callable[[str], Any | Awaitable[Any]]

_SUPPORTED_BACKENDS = frozenset({"chromadb", "pgvector"})
_ERROR_CODES = frozenset(
    {
        "notes_semantic_erasure_backend_unavailable",
        "notes_semantic_erasure_cleanup_failed",
        "notes_semantic_erasure_cleanup_unconfirmed",
        "notes_semantic_erasure_dataset_limit_exceeded",
        "notes_semantic_erasure_fence_lost",
        "notes_semantic_erasure_timeout",
    }
)


class SemanticErasureError(RuntimeError):
    """Stable, content-free semantic erasure failure."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_CODES:
            code = "notes_semantic_erasure_cleanup_failed"
        self.code = code
        self.failure_code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SemanticErasureResult:
    """Bounded aggregate returned to the Notes DSR category."""

    datasets: int
    cleaned_generations: int
    deleted_notes: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class _ReleasingSemanticAuthority:
    """Release the vector facade's executor-local authority connection."""

    def __init__(self, store: Any, db: Any) -> None:
        self._store = store
        self._db = db

    @property
    def owner_user_id(self) -> str:
        return str(self._store.owner_user_id)

    def get_generation(self, dataset_id: str, generation_id: str) -> Any:
        try:
            return self._store.get_generation(dataset_id, generation_id)
        finally:
            close = getattr(self._db, "close_connection", None)
            if callable(close):
                close()


class _QuiescentCleanupVectors:
    """Prevent cancellation from abandoning an in-flight vector deletion."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def delete_ids(
        self,
        dataset_id: str,
        generation_id: str,
        vector_ids: tuple[str, ...],
    ) -> Any:
        return await run_quiescent_operation(_resolve(self._delegate.delete_ids(dataset_id, generation_id, vector_ids)))

    async def delete_generation(self, dataset_id: str, generation_id: str) -> Any:
        return await run_quiescent_operation(_resolve(self._delegate.delete_generation(dataset_id, generation_id)))


def _metric(*, status: str, backend: str, error_code: str) -> None:
    normalized_backend = backend if backend in _SUPPORTED_BACKENDS else "unavailable"
    record_semantic_dsr_metrics(status=status, backend=normalized_backend)
    if status != "success":
        logger.bind(error_code=error_code).debug("Notes semantic erasure cleanup requires retry")


async def _audit(*, owner_user_id: str, status: str, reason: str) -> None:
    try:
        await emit_semantic_audit_event(
            owner_user_id=owner_user_id,
            dataset_id="all",
            event="dsr_cleanup",
            status=status,
            reason=reason,
        )
    except Exception:  # noqa: BLE001 - erasure state remains authoritative
        logger.warning("Notes semantic erasure audit persistence failed")


def _mapped_error(exc: SemanticIndexingError) -> SemanticErasureError:
    if exc.code == "notes_semantic_cleanup_unconfirmed":
        return SemanticErasureError("notes_semantic_erasure_cleanup_unconfirmed")
    if exc.code.endswith("fence_lost") or exc.code.endswith("finalization_fence_lost"):
        return SemanticErasureError("notes_semantic_erasure_fence_lost")
    if "backend" in exc.code or "unavailable" in exc.code:
        return SemanticErasureError("notes_semantic_erasure_backend_unavailable")
    return SemanticErasureError("notes_semantic_erasure_cleanup_failed")


class SemanticErasureCoordinator:
    """Fence, physically erase, and atomically finalize one Notes owner."""

    def __init__(
        self,
        *,
        db: Any,
        vector_store_factory: VectorStoreFactory | None = None,
        settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
        lease_seconds: int | None = None,
        max_datasets: int = 100,
        max_cleanup_steps: int = 10_000,
        clock: Callable[[], datetime] = _utc_now,
        close_database_on_exit: bool = False,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
            or timeout_seconds > 300
        ):
            raise ValueError("notes_semantic_erasure_timeout_invalid")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or poll_interval_seconds <= 0
            or poll_interval_seconds > 1
        ):
            raise ValueError("notes_semantic_erasure_poll_interval_invalid")
        if lease_seconds is None:
            try:
                lease_seconds = int(os.getenv("NOTES_SEMANTIC_LEASE_SECONDS", "180") or "180")
            except (TypeError, ValueError):
                lease_seconds = 86_400
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86_400:
            raise ValueError("notes_semantic_erasure_lease_invalid")
        if type(max_datasets) is not int or not 1 <= max_datasets <= 100:
            raise ValueError("notes_semantic_erasure_dataset_limit_invalid")
        if type(max_cleanup_steps) is not int or not 1 <= max_cleanup_steps <= 100_000:
            raise ValueError("notes_semantic_erasure_cleanup_limit_invalid")
        self._db = db
        self._store = db.note_semantic_store
        self._owner_user_id = str(self._store.owner_user_id)
        self._vector_store_factory = vector_store_factory
        self._settings = settings
        self._timeout_seconds = float(timeout_seconds)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._lease_seconds = lease_seconds
        self._max_datasets = max_datasets
        self._max_cleanup_steps = max_cleanup_steps
        self._clock = clock
        self._close_database_on_exit = close_database_on_exit
        self._deadline: float | None = None
        self._current_backend = "unavailable"
        self._executor: ThreadPoolExecutor | None = None

    def _store_executor(self) -> ThreadPoolExecutor:
        executor = self._executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="notes-semantic-erasure",
            )
            self._executor = executor
        return executor

    def _check_deadline(self) -> None:
        deadline = self._deadline
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            raise SemanticErasureError("notes_semantic_erasure_timeout")

    async def _store_call(
        self,
        function: Callable[..., Any],
        /,
        *,
        enforce_deadline: bool = True,
        check_after: bool = True,
        committed_result_wins: bool = False,
        **kwargs: Any,
    ) -> Any:
        if enforce_deadline:
            self._check_deadline()
        operation = asyncio.get_running_loop().run_in_executor(
            self._store_executor(),
            partial(function, **kwargs),
        )
        result = await run_quiescent_operation(
            operation,
            committed_result_wins=committed_result_wins,
        )
        if enforce_deadline and check_after:
            self._check_deadline()
        return result

    async def _default_vector_store(self, backend_name: str) -> Any:
        chroma_manager = None
        postgres_backend = None
        if backend_name == "chromadb":
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
            from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

            chroma_manager = await run_quiescent_operation(
                asyncio.to_thread(
                    ChromaDBManager,
                    user_id=self._owner_user_id,
                    user_embedding_config={
                        "USER_DB_BASE_DIR": str(DatabasePaths.get_user_db_base_dir()),
                    },
                )
            )
        elif backend_name == "pgvector":
            postgres_backend = getattr(self._db, "_backend", None)
        return await create_semantic_vector_store(
            backend_name,
            authority=_ReleasingSemanticAuthority(self._store, self._db),
            chroma_manager=chroma_manager,
            postgres_backend=postgres_backend,
            settings=self._settings,
        )

    async def _vectors(self, backend_name: str) -> Any:
        factory = self._vector_store_factory
        if factory is None:
            return await self._default_vector_store(backend_name)
        return await _resolve(factory(backend_name))

    async def _dataset_ids(self) -> tuple[str, ...]:
        datasets: list[str] = []
        after: str | None = None
        while len(datasets) < self._max_datasets:
            page = await self._store_call(
                self._store.list_maintenance_dataset_ids,
                limit=min(100, self._max_datasets - len(datasets)),
                after_dataset_id=after,
            )
            if not page:
                return tuple(datasets)
            datasets.extend(page)
            after = page[-1]
        overflow = await self._store_call(
            self._store.list_maintenance_dataset_ids,
            limit=1,
            after_dataset_id=after,
        )
        if overflow:
            raise SemanticErasureError("notes_semantic_erasure_dataset_limit_exceeded")
        return tuple(datasets)

    async def _fence_dataset(self, dataset_id: str) -> str | None:
        for _attempt in range(3):
            config = await self._store_call(
                self._store.get_configuration,
                dataset_id=dataset_id,
            )
            if config is None:
                return None
            backend_name = str(config.vector_backend or "").strip().lower()
            self._current_backend = backend_name
            if backend_name not in _SUPPORTED_BACKENDS:
                raise SemanticErasureError("notes_semantic_erasure_backend_unavailable")
            disabled = await self._store_call(
                self._store.disable_and_schedule_cleanup,
                dataset_id=dataset_id,
                expected_configuration_revision=config.configuration_revision,
                now=self._clock(),
            )
            if disabled is not None:
                return backend_name
        raise SemanticErasureError("notes_semantic_erasure_fence_lost")

    async def _drain_dataset(self, dataset_id: str, backend_name: str) -> int:
        self._current_backend = backend_name
        vectors = _QuiescentCleanupVectors(await self._vectors(backend_name))
        self._check_deadline()
        publication = SemanticPublicationService(
            store=self._store,
            vectors=vectors,
            revalidate=lambda _fence: None,
            clock=self._clock,
            receipt_factory=lambda: str(uuid4()),
            max_cleanup_vectors=self._settings.max_cleanup_vectors_per_run,
            store_call=self._store_call,
        )
        cleaned = 0
        steps = 0
        while True:
            self._check_deadline()
            now = self._clock()
            expired_before = now - timedelta(seconds=self._lease_seconds)
            await self._store_call(
                self._store.reclaim_expired_dataset_work,
                dataset_id=dataset_id,
                expired_before=expired_before,
                limit=100,
                now=now,
            )
            await self._store_call(
                self._store.reclaim_expired_obsolete_vector_claims,
                dataset_id=dataset_id,
                expired_before=expired_before,
                limit=self._settings.max_cleanup_vectors_per_run,
                now=now,
            )
            await self._store_call(
                self._store.rearm_exhausted_generation_cleanup,
                dataset_id=dataset_id,
                limit=100,
                now=now,
            )
            claims = await self._store_call(
                self._store.claim_generation_cleanup_batch,
                dataset_id=dataset_id,
                limit=1,
                now=now,
            )
            if not claims:
                pending = await self._store_call(
                    self._store.has_pending_cleanup,
                    dataset_id=dataset_id,
                )
                if not pending:
                    return cleaned
                await asyncio.sleep(self._poll_interval_seconds)
                continue
            claim = claims[0]
            try:
                while True:
                    steps += 1
                    if steps > self._max_cleanup_steps:
                        raise SemanticErasureError("notes_semantic_erasure_cleanup_failed")
                    if await publication.cleanup_generation(
                        claim,
                        before_side_effect=self._check_deadline,
                    ):
                        self._check_deadline()
                        cleaned += 1
                        break
            except BaseException:
                try:
                    await self._store_call(
                        self._store.release_work_claim,
                        enforce_deadline=False,
                        check_after=False,
                        dataset_id=claim.dataset_id,
                        work_id=claim.id,
                        claim_token=claim.claim_token or "",
                        fencing_token=claim.fencing_token,
                        now=self._clock(),
                    )
                except asyncio.CancelledError:
                    pass
                except Exception as release_exc:
                    raise SemanticErasureError("notes_semantic_erasure_cleanup_failed") from release_exc
                raise

    async def _erase(self) -> SemanticErasureResult:
        datasets = await self._dataset_ids()
        cleaned_generations = 0
        dataset_backends: dict[str, str] = {}
        for dataset_id in datasets:
            backend_name = await self._fence_dataset(dataset_id)
            if backend_name is not None:
                dataset_backends[dataset_id] = backend_name
        for dataset_id, backend_name in dataset_backends.items():
            cleaned_generations += await self._drain_dataset(
                dataset_id,
                backend_name,
            )
        self._check_deadline()
        deleted_notes = await self._store_call(
            self._store.finalize_owner_erasure,
            check_after=False,
            committed_result_wins=True,
            dataset_ids=datasets,
        )
        for backend_name in dataset_backends.values() or ["unavailable"]:
            _metric(status="success", backend=backend_name, error_code="none")
        await _resolve(
            _audit(
                owner_user_id=self._owner_user_id,
                status="success",
                reason="none",
            )
        )
        return SemanticErasureResult(
            datasets=len(datasets),
            cleaned_generations=cleaned_generations,
            deleted_notes=deleted_notes,
        )

    async def erase(self) -> SemanticErasureResult:
        """Run bounded erasure and preserve retry state on every failure."""

        self._deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        try:
            return await self._erase()
        except asyncio.CancelledError:
            raise
        except SemanticErasureError as exc:
            error = exc
        except SemanticIndexingError as exc:
            error = _mapped_error(exc)
        except SemanticVectorError:
            error = SemanticErasureError("notes_semantic_erasure_backend_unavailable")
        except Exception:  # noqa: BLE001 - DSR receives only a bounded failure code
            error = SemanticErasureError("notes_semantic_erasure_cleanup_failed")
        finally:
            self._deadline = None
            executor = self._executor
            if executor is not None:
                backend_type = getattr(
                    getattr(self._db, "backend_type", None),
                    "value",
                    "",
                )
                close_name = (
                    "close_all_connections"
                    if self._close_database_on_exit and backend_type == "sqlite"
                    else "close_connection"
                )
                close = getattr(self._db, close_name, None)
                try:
                    if callable(close):
                        operation = asyncio.get_running_loop().run_in_executor(
                            executor,
                            close,
                        )
                        await run_quiescent_operation(
                            operation,
                            committed_result_wins=True,
                        )
                except Exception:  # noqa: BLE001 - completion must not be masked
                    logger.warning("Notes semantic erasure database close failed")
                finally:
                    executor.shutdown(wait=True, cancel_futures=False)
                    self._executor = None
        reason = error.code.removeprefix("notes_semantic_erasure_")
        _metric(status="failed", backend=self._current_backend, error_code=reason)
        await _resolve(
            _audit(
                owner_user_id=self._owner_user_id,
                status="failed",
                reason=reason,
            )
        )
        raise error


__all__ = [
    "SemanticErasureCoordinator",
    "SemanticErasureError",
    "SemanticErasureResult",
]
