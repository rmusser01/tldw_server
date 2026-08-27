"""Independent maintenance lifecycle for Notes graph suggestions."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenancePassResult,
    MaintenanceScope,
    SuggestionMaintenance,
    run_maintenance_loop,
)


async def _open_owner_database(owner_user_id: str) -> Any:
    user_id = int(owner_user_id)
    return await get_chacha_db_for_user_id(user_id, client_id=str(user_id))


def _close_database(db: Any) -> None:
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
    elif hasattr(db, "close_connection"):
        db.close_connection()


async def _discover_scopes(
    *,
    users_repo: AuthnzUsersRepo,
    open_database: Any = _open_owner_database,
) -> tuple[tuple[MaintenanceScope, ...], tuple[Any, ...]]:
    """Discover durable suggestion scopes from authoritative users and owner stores."""

    scopes: list[MaintenanceScope] = []
    databases: list[Any] = []
    offset = 0
    page_size = 200
    total = 1
    try:
        while offset < total:
            users, total = await users_repo.list_users(offset=offset, limit=page_size)
            if not users:
                break
            for user in users:
                owner = str(user.get("id") or "")
                if not owner:
                    continue
                db = await open_database(owner)
                databases.append(db)
                store = db.note_graph_suggestion_store
                for dataset in store.list_maintenance_dataset_ids(limit=100):
                    scopes.append(MaintenanceScope(store, dataset))
            offset += page_size
        return tuple(scopes), tuple(databases)
    except Exception:
        for db in databases:
            try:
                _close_database(db)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("Notes graph suggestions maintenance database close skipped")
        raise


class _MaintenanceRunner:
    def __init__(self, *, jobs: JobManager, users_repo: AuthnzUsersRepo) -> None:
        self._jobs = jobs
        self._users_repo = users_repo

    async def run_pass(self, *, now: datetime) -> MaintenancePassResult:
        databases: tuple[Any, ...] = ()
        try:
            scopes, databases = await _discover_scopes(users_repo=self._users_repo)
            return SuggestionMaintenance(jobs=self._jobs, scopes=scopes).run_pass(now=now)
        except (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            logger.warning("Notes graph suggestions maintenance pass failed safely")
            return MaintenancePassResult(claimed=0, reconciled=0, released=0, cleaned=0)
        finally:
            for db in databases:
                try:
                    _close_database(db)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("Notes graph suggestions maintenance database close skipped")


async def run_notes_graph_suggestions_maintenance(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run provider-independent maintenance at startup and once per minute."""

    stop = stop_event or asyncio.Event()
    jobs = JobManager()
    users_repo = await AuthnzUsersRepo.from_pool()
    logger.info("Notes graph suggestions maintenance starting")
    await run_maintenance_loop(
        _MaintenanceRunner(jobs=jobs, users_repo=users_repo),
        stop,
    )


__all__ = ["run_notes_graph_suggestions_maintenance"]
