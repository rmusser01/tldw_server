from __future__ import annotations

import asyncio
import os
from sqlite3 import Error as SQLiteError

from loguru import logger

from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

_CHATBOOKS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _enumerate_user_ids() -> list[int]:
    """Get list of user IDs from user database directories."""
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _CHATBOOKS_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "chatbooks_cleanup: failed to resolve user db base dir"
        )
        return []

    uids: list[int] = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                uids.append(int(p.name))
            except (TypeError, ValueError):
                continue

    if not uids:
        try:
            uids = [DatabasePaths.get_single_user_id()]
        except _CHATBOOKS_NONCRITICAL_EXCEPTIONS:
            uids = []

    return sorted(set(uids))


def _build_chacha_db_for_user(user_id: int) -> CharactersRAGDB:
    """Build a per-user ChaChaNotes DB handle for cleanup tasks."""
    db_path = DatabasePaths.get_chacha_db_path(user_id)
    return CharactersRAGDB(db_path=str(db_path), client_id=str(user_id))


async def run_chatbooks_cleanup_loop(stop_event: asyncio.Event | None = None) -> None:
    """Run scheduled cleanup of expired chatbook exports."""
    interval_sec = int(os.getenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "3600") or "0")
    if interval_sec <= 0:
        logger.info("Chatbooks cleanup scheduler disabled (CHATBOOKS_CLEANUP_INTERVAL_SEC=0)")
        return

    logger.info(f"Starting chatbooks cleanup worker (every {interval_sec}s)")

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping chatbooks cleanup worker on shutdown signal")
            return
        try:
            deleted_exports = 0
            deleted_imports = 0
            for user_id in _enumerate_user_ids():
                db = _build_chacha_db_for_user(user_id)
                svc = ChatbookService(str(user_id), db, user_id_int=user_id)
                deleted_exports += svc.cleanup_expired_exports()
                deleted_imports += svc.cleanup_import_orphans()
            if deleted_exports:
                logger.info(f"Chatbooks cleanup removed {deleted_exports} expired export files")
            if deleted_imports:
                logger.info(f"Chatbooks cleanup removed {deleted_imports} orphaned import files")
        except _CHATBOOKS_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).warning("Chatbooks cleanup loop error")
        await asyncio.sleep(interval_sec)
