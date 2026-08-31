"""Fail-closed Notes semantic cleanup for data-subject erasure."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticIndexingError,
)

from .semantic_observability import (
    build_semantic_audit_event,
    build_semantic_metric_event,
)
from .semantic_publication import SemanticPublicationService
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _metric(*, status: str, backend: str, error_code: str) -> None:
    event = build_semantic_metric_event(
        operation="cleanup",
        status=status,
        backend=backend if backend in _SUPPORTED_BACKENDS else "unavailable",
        error_code=error_code,
        value=1,
    )
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

        increment_counter(event.name, event.value, dict(event.labels))
    except Exception:  # noqa: BLE001 - erasure must not depend on metrics availability
        logger.debug("Notes semantic erasure metric emission unavailable")


def _audit(*, status: str, reason: str) -> None:
    event = build_semantic_audit_event(
        event="cleanup_completion",
        status=status,
        reason=reason,
    )
    logger.bind(semantic_event=event.event, **dict(event.fields)).info(
        "Notes semantic erasure cleanup"
    )


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
                lease_seconds = int(
                    os.getenv("NOTES_SEMANTIC_LEASE_SECONDS", "180") or "180"
                )
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

    async def _default_vector_store(self, backend_name: str) -> Any:
        chroma_manager = None
        postgres_backend = None
        if backend_name == "chromadb":
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
            from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

            chroma_manager = await asyncio.to_thread(
                ChromaDBManager,
                user_id=self._owner_user_id,
                user_embedding_config={
                    "USER_DB_BASE_DIR": str(DatabasePaths.get_user_db_base_dir()),
                },
            )
        elif backend_name == "pgvector":
            postgres_backend = getattr(self._db, "_backend", None)
        return await create_semantic_vector_store(
            backend_name,
            authority=self._store,
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
            page = await asyncio.to_thread(
                self._store.list_maintenance_dataset_ids,
                limit=min(100, self._max_datasets - len(datasets)),
                after_dataset_id=after,
            )
            if not page:
                return tuple(datasets)
            datasets.extend(page)
            after = page[-1]
        overflow = await asyncio.to_thread(
            self._store.list_maintenance_dataset_ids,
            limit=1,
            after_dataset_id=after,
        )
        if overflow:
            raise SemanticErasureError(
                "notes_semantic_erasure_dataset_limit_exceeded"
            )
        return tuple(datasets)

    async def _fence_dataset(self, dataset_id: str) -> str | None:
        for _attempt in range(3):
            config = await asyncio.to_thread(self._store.get_configuration, dataset_id)
            if config is None:
                return None
            backend_name = str(config.vector_backend or "").strip().lower()
            if backend_name not in _SUPPORTED_BACKENDS:
                raise SemanticErasureError(
                    "notes_semantic_erasure_backend_unavailable"
                )
            disabled = await asyncio.to_thread(
                self._store.disable_and_schedule_cleanup,
                dataset_id=dataset_id,
                expected_configuration_revision=config.configuration_revision,
                now=self._clock(),
            )
            if disabled is not None:
                return backend_name
        raise SemanticErasureError("notes_semantic_erasure_fence_lost")

    async def _drain_dataset(self, dataset_id: str, backend_name: str) -> int:
        vectors = await self._vectors(backend_name)
        publication = SemanticPublicationService(
            store=self._store,
            vectors=vectors,
            revalidate=lambda _fence: None,
            clock=self._clock,
            receipt_factory=lambda: str(uuid4()),
            max_cleanup_vectors=self._settings.max_cleanup_vectors_per_run,
        )
        cleaned = 0
        steps = 0
        while True:
            now = self._clock()
            expired_before = now - timedelta(seconds=self._lease_seconds)
            await asyncio.to_thread(
                self._store.reclaim_expired_dataset_work,
                dataset_id=dataset_id,
                expired_before=expired_before,
                limit=100,
                now=now,
            )
            await asyncio.to_thread(
                self._store.reclaim_expired_obsolete_vector_claims,
                dataset_id=dataset_id,
                expired_before=expired_before,
                limit=self._settings.max_cleanup_vectors_per_run,
                now=now,
            )
            await asyncio.to_thread(
                self._store.rearm_exhausted_generation_cleanup,
                dataset_id=dataset_id,
                limit=100,
                now=now,
            )
            claims = await asyncio.to_thread(
                self._store.claim_generation_cleanup_batch,
                dataset_id=dataset_id,
                limit=100,
                now=now,
            )
            if not claims:
                pending = await asyncio.to_thread(
                    self._store.has_pending_cleanup,
                    dataset_id,
                )
                if not pending:
                    return cleaned
                await asyncio.sleep(self._poll_interval_seconds)
                continue
            for claim in claims:
                while True:
                    steps += 1
                    if steps > self._max_cleanup_steps:
                        raise SemanticErasureError(
                            "notes_semantic_erasure_cleanup_failed"
                        )
                    if await publication.cleanup_generation(claim):
                        cleaned += 1
                        break

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
        for dataset_id in datasets:
            await asyncio.to_thread(
                self._store.purge_semantic_dataset_for_erasure,
                dataset_id=dataset_id,
            )
        for backend_name in dataset_backends.values() or ["unavailable"]:
            _metric(status="success", backend=backend_name, error_code="none")
        _audit(status="success", reason="none")
        return SemanticErasureResult(
            datasets=len(datasets),
            cleaned_generations=cleaned_generations,
        )

    async def erase(self) -> SemanticErasureResult:
        """Run bounded erasure and preserve retry state on every failure."""

        try:
            return await asyncio.wait_for(
                self._erase(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            error = SemanticErasureError("notes_semantic_erasure_timeout")
        except SemanticErasureError as exc:
            error = exc
        except SemanticIndexingError as exc:
            error = _mapped_error(exc)
        except SemanticVectorError:
            error = SemanticErasureError(
                "notes_semantic_erasure_backend_unavailable"
            )
        except Exception:  # noqa: BLE001 - DSR receives only a bounded failure code
            error = SemanticErasureError("notes_semantic_erasure_cleanup_failed")
        finally:
            if self._close_database_on_exit:
                close = getattr(self._db, "close_all_connections", None)
                if callable(close):
                    await asyncio.to_thread(close)
        _metric(status="failed", backend="unavailable", error_code="cleanup_failed")
        _audit(status="failed", reason="cleanup_failed")
        raise error


__all__ = [
    "SemanticErasureCoordinator",
    "SemanticErasureError",
    "SemanticErasureResult",
]
