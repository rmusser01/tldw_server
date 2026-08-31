"""Bounded recovery, admission, and cleanup cadence for Notes semantic indexing."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.semantic_api import load_semantic_settings
from tldw_Server_API.app.core.Notes_Graph.semantic_jobs import (
    SemanticJobCommand,
    SemanticJobCoordinator,
    SemanticJobsError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.services.notes_semantic_index_worker import (
    build_production_runtime,
)

_MAINTENANCE_ERRORS = (
    AttributeError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class SemanticMaintenanceResult:
    claimed: int
    dirty_admitted: int
    failed_retries: int
    cleanup_confirmed: int


class SemanticMaintenanceCoordinator:
    """Share one hard claim budget across recovery, indexing, and cleanup."""

    def __init__(self, *, scopes: tuple[Any, ...], indexing_enabled: bool) -> None:
        self._scopes = scopes
        self._indexing_enabled = indexing_enabled

    async def run_pass(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> SemanticMaintenanceResult:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        claimed = dirty_admitted = failed_retries = cleanup_confirmed = 0
        dirty_keys: set[tuple[object, object, object, object]] = set()
        for scope in self._scopes:
            remaining = limit - claimed
            if remaining <= 0:
                break
            reclaimed = int(scope.reclaim_expired(limit=remaining, now=now))
            if not 0 <= reclaimed <= remaining:
                raise RuntimeError("notes_semantic_maintenance_budget_invalid")
            claimed += reclaimed

            if self._indexing_enabled:
                remaining = limit - claimed
                if remaining > 0:
                    dirty = tuple(scope.claim_dirty(limit=remaining, now=now))
                    if len(dirty) > remaining:
                        raise RuntimeError("notes_semantic_maintenance_budget_invalid")
                    claimed += len(dirty)
                    for claim in dirty:
                        key = (
                            getattr(claim, "owner_user_id", None),
                            getattr(claim, "dataset_id", None),
                            getattr(claim, "generation_id", None),
                            getattr(claim, "dirty_generation", None),
                        )
                        if key in dirty_keys:
                            continue
                        dirty_keys.add(key)
                        if scope.admit(mode="maintain", claim=claim):
                            dirty_admitted += 1

                remaining = limit - claimed
                if remaining > 0:
                    failed = tuple(scope.claim_failed(limit=remaining, now=now))
                    if len(failed) > remaining:
                        raise RuntimeError("notes_semantic_maintenance_budget_invalid")
                    claimed += len(failed)
                    for claim in failed:
                        if scope.admit(mode="retry_failed", claim=claim):
                            failed_retries += 1

            remaining = limit - claimed
            if remaining > 0:
                cleanup = tuple(scope.claim_cleanup(limit=remaining, now=now))
                if len(cleanup) > remaining:
                    raise RuntimeError("notes_semantic_maintenance_budget_invalid")
                claimed += len(cleanup)
                for claim in cleanup:
                    if await scope.cleanup_claim(claim):
                        cleanup_confirmed += 1
        return SemanticMaintenanceResult(
            claimed=claimed,
            dirty_admitted=dirty_admitted,
            failed_retries=failed_retries,
            cleanup_confirmed=cleanup_confirmed,
        )


async def run_maintenance_loop(
    runner: Any,
    stop_event: asyncio.Event,
    *,
    interval_seconds: float = 60,
    now: Any = lambda: datetime.now(timezone.utc),
) -> None:
    """Run one immediate pass, then wait interruptibly between passes."""

    while not stop_event.is_set():
        await runner.run_pass(now=now(), limit=100)
        if stop_event.is_set():
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


@dataclass(frozen=True, slots=True)
class _DirtyClaim:
    owner_user_id: str
    dataset_id: str
    generation_id: str
    dirty_generation: int


@dataclass(frozen=True, slots=True)
class _FailedClaim:
    owner_user_id: str
    dataset_id: str
    generation_id: str


@dataclass(frozen=True, slots=True)
class _ObsoleteCleanupClaim:
    generation_id: str


class _ProductionScope:
    def __init__(
        self,
        *,
        db: Any,
        jobs: JobManager,
        owner_user_id: str,
        dataset_id: str,
        settings: SemanticIndexSettings,
    ) -> None:
        self._db = db
        self._store = db.note_semantic_store
        self._jobs = jobs
        self._owner_user_id = owner_user_id
        self._dataset_id = dataset_id
        self._settings = settings

    def reclaim_expired(self, *, limit: int, now: datetime) -> int:
        lease_seconds = int(os.getenv("NOTES_SEMANTIC_LEASE_SECONDS", "180") or "180")
        return self._store.reclaim_expired_dataset_work(
            dataset_id=self._dataset_id,
            expired_before=now - timedelta(seconds=max(lease_seconds, 1)),
            limit=min(limit, 256),
            now=now,
        )

    def claim_dirty(self, *, limit: int, now: datetime) -> tuple[_DirtyClaim, ...]:
        del now
        rows = self._store.list_dirty_generation_watermarks(
            dataset_id=self._dataset_id,
            limit=min(limit, 100),
        )
        return tuple(
            _DirtyClaim(
                owner_user_id=self._owner_user_id,
                dataset_id=self._dataset_id,
                generation_id=generation_id,
                dirty_generation=watermark,
            )
            for generation_id, watermark in rows
        )

    def claim_failed(self, *, limit: int, now: datetime) -> tuple[_FailedClaim, ...]:
        del now
        rows = self._store.list_failed_generations(
            dataset_id=self._dataset_id,
            limit=min(limit, 100),
        )
        return tuple(
            _FailedClaim(
                owner_user_id=self._owner_user_id,
                dataset_id=self._dataset_id,
                generation_id=generation_id,
            )
            for generation_id in rows
        )

    def claim_cleanup(self, *, limit: int, now: datetime) -> tuple[Any, ...]:
        generation_claims = self._store.claim_generation_cleanup_batch(
            dataset_id=self._dataset_id,
            limit=min(limit, 100),
            now=now,
        )
        remaining = limit - len(generation_claims)
        if remaining <= 0:
            return generation_claims
        obsolete = self._store.list_obsolete_cleanup_generations(
            dataset_id=self._dataset_id,
            limit=min(remaining, 100),
        )
        return generation_claims + tuple(
            _ObsoleteCleanupClaim(generation_id=value) for value in obsolete
        )

    def admit(self, *, mode: str, claim: Any) -> bool:
        config = self._store.get_configuration(self._dataset_id)
        if config is None or config.desired_state.value != "enabled":
            return False
        generation_id = str(getattr(claim, "generation_id", "") or "")
        watermark = str(getattr(claim, "dirty_generation", "failed"))
        digest = hashlib.sha256(
            f"{mode}\0{self._dataset_id}\0{generation_id}\0{watermark}".encode()
        ).hexdigest()
        try:
            SemanticJobCoordinator(
                jobs=self._jobs,
                owner_user_id=self._owner_user_id,
            ).admit(
                SemanticJobCommand(
                    dataset_id=self._dataset_id,
                    configuration_revision=config.configuration_revision,
                    generation_id=generation_id,
                    mode=mode,
                ),
                idempotency_key=f"maintenance:{digest}",
            )
        except SemanticJobsError:
            return False
        return True

    async def _runtime(self, generation_id: str) -> Any:
        config = self._store.get_configuration(self._dataset_id)
        generation = self._store.get_generation(self._dataset_id, generation_id)
        if config is None or generation is None or not generation.root_job_id:
            raise RuntimeError("notes_semantic_cleanup_authority_missing")
        return await build_production_runtime(
            db=self._db,
            settings=self._settings,
            owner_user_id=self._owner_user_id,
            dataset_id=self._dataset_id,
            configuration_revision=config.configuration_revision,
            generation_id=generation_id,
            root_job_id=generation.root_job_id,
            mode="delete",
        )

    async def cleanup_claim(self, claim: Any) -> bool:
        generation_id = str(getattr(claim, "generation_id", "") or "")
        runtime = await self._runtime(generation_id)
        if isinstance(claim, _ObsoleteCleanupClaim):
            return await runtime.cleanup_obsolete_generation()
        for _ in range(self._settings.max_retries + 1):
            if await runtime.cleanup_claim(claim):
                return True
        return False


async def _open_owner_database(owner_user_id: str) -> Any:
    user_id = int(owner_user_id)
    return await get_chacha_db_for_user_id(user_id, client_id=owner_user_id)


def _close_database(db: Any) -> None:
    release = getattr(db, "release_context_connection", None)
    close = release if callable(release) else getattr(db, "close_connection", None)
    if callable(close):
        close()


class _MaintenanceRunner:
    def __init__(
        self,
        *,
        jobs: JobManager,
        users_repo: AuthnzUsersRepo,
        settings: SemanticIndexSettings,
    ) -> None:
        self._jobs = jobs
        self._users_repo = users_repo
        self._settings = settings

    async def run_pass(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> SemanticMaintenanceResult:
        aggregate = SemanticMaintenanceResult(0, 0, 0, 0)
        offset = 0
        total = 1
        while offset < total and aggregate.claimed < limit:
            users, total = await self._users_repo.list_users(offset=offset, limit=200)
            if not users:
                break
            for user in users:
                if aggregate.claimed >= limit:
                    break
                owner = str(user.get("id") or "")
                if not owner:
                    continue
                db = None
                try:
                    db = await _open_owner_database(owner)
                    datasets = await asyncio.to_thread(
                        db.note_semantic_store.list_maintenance_dataset_ids,
                        limit=100,
                    )
                    scopes = tuple(
                        _ProductionScope(
                            db=db,
                            jobs=self._jobs,
                            owner_user_id=owner,
                            dataset_id=dataset_id,
                            settings=self._settings,
                        )
                        for dataset_id in datasets
                    )
                    result = await SemanticMaintenanceCoordinator(
                        scopes=scopes,
                        indexing_enabled=self._settings.indexing_enabled,
                    ).run_pass(
                        now=now,
                        limit=limit - aggregate.claimed,
                    )
                    aggregate = SemanticMaintenanceResult(
                        claimed=aggregate.claimed + result.claimed,
                        dirty_admitted=aggregate.dirty_admitted + result.dirty_admitted,
                        failed_retries=aggregate.failed_retries + result.failed_retries,
                        cleanup_confirmed=(
                            aggregate.cleanup_confirmed + result.cleanup_confirmed
                        ),
                    )
                except _MAINTENANCE_ERRORS:
                    logger.warning("Notes semantic owner maintenance failed safely")
                finally:
                    if db is not None:
                        await asyncio.to_thread(_close_database, db)
            offset += 200
        return aggregate


async def run_notes_semantic_maintenance(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run semantic recovery immediately and then once per bounded interval."""

    stop = stop_event or asyncio.Event()
    jobs = await asyncio.to_thread(JobManager)
    users_repo = await AuthnzUsersRepo.from_pool()
    settings = load_semantic_settings()
    interval = float(os.getenv("NOTES_SEMANTIC_MAINTENANCE_INTERVAL_SECONDS", "60") or "60")
    logger.info("Notes semantic maintenance starting")
    await run_maintenance_loop(
        _MaintenanceRunner(jobs=jobs, users_repo=users_repo, settings=settings),
        stop,
        interval_seconds=max(interval, 1.0),
    )


__all__ = [
    "SemanticMaintenanceCoordinator",
    "SemanticMaintenanceResult",
    "run_maintenance_loop",
    "run_notes_semantic_maintenance",
]
