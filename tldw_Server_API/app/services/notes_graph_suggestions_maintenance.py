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
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenancePassResult,
    MaintenanceScope,
    SuggestionMaintenance,
    run_maintenance_loop,
)

_MAINTENANCE_ERRORS = (
    AttributeError,
    CharactersRAGDBError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def _open_owner_database(owner_user_id: str) -> Any:
    user_id = int(owner_user_id)
    return await get_chacha_db_for_user_id(user_id, client_id=str(user_id))


def _close_database(db: Any) -> None:
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
    elif hasattr(db, "close_connection"):
        db.close_connection()


class _MaintenanceRunner:
    def __init__(self, *, jobs: JobManager, users_repo: AuthnzUsersRepo) -> None:
        self._jobs = jobs
        self._users_repo = users_repo

    async def run_pass(self, *, now: datetime) -> MaintenancePassResult:
        claimed = reconciled = released = cleaned = 0
        remaining = 100
        offset = 0
        page_size = 200
        total = 1
        try:
            while offset < total:
                users, total = await self._users_repo.list_users(
                    offset=offset,
                    limit=page_size,
                )
                if not users:
                    break
                for user in users:
                    if remaining == 0:
                        return MaintenancePassResult(claimed, reconciled, released, cleaned)
                    owner = str(user.get("id") or "")
                    if not owner:
                        continue
                    db = None
                    try:
                        db = await _open_owner_database(owner)
                        store = db.note_graph_suggestion_store
                        scopes = tuple(
                            MaintenanceScope(store, dataset)
                            for dataset in store.list_maintenance_dataset_ids(limit=100)
                        )
                        owner_claimed = 0

                        def account_claims(count: int) -> None:
                            nonlocal claimed, owner_claimed, remaining
                            if count < 0 or count > remaining:
                                raise RuntimeError("notes_graph_maintenance_budget_invalid")
                            claimed += count
                            owner_claimed += count
                            remaining -= count

                        result = SuggestionMaintenance(
                            jobs=self._jobs,
                            scopes=scopes,
                        ).run_pass(
                            now=now,
                            limit=remaining,
                            on_claimed=account_claims,
                        )
                        if result.claimed != owner_claimed:
                            raise RuntimeError("notes_graph_maintenance_budget_invalid")
                        reconciled += result.reconciled
                        released += result.released
                        cleaned += result.cleaned
                        remaining -= min(remaining, result.cleaned)
                    except _MAINTENANCE_ERRORS:
                        logger.warning("Notes graph suggestions owner maintenance failed safely")
                    finally:
                        if db is not None:
                            try:
                                _close_database(db)
                            except _MAINTENANCE_ERRORS:
                                logger.debug(
                                    "Notes graph suggestions maintenance database close skipped"
                                )
                offset += page_size
        except _MAINTENANCE_ERRORS:
            logger.warning("Notes graph suggestions maintenance pass failed safely")
        return MaintenancePassResult(claimed, reconciled, released, cleaned)


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
