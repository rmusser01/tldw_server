"""
TTS history retention cleanup service.

Runs periodic cleanup based on:
  - TTS_HISTORY_RETENTION_DAYS
  - TTS_HISTORY_MAX_ROWS_PER_USER
  - TTS_HISTORY_PURGE_INTERVAL_HOURS
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Iterable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.Metrics import get_metrics_registry

_TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    BackendDatabaseError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _normalize_value(val: str | int | None) -> str | None:
    if val is None:
        return None
    text = str(val).strip()
    if not text or text.lower() in {"none", "null", "nil"}:
        return None
    return text


def _raw_setting(env_name: str, settings_attr: str, default: str | None = None) -> str | None:
    env_val = _normalize_value(os.getenv(env_name))
    if env_val is not None:
        return env_val
    settings_val = _normalize_value(getattr(settings, settings_attr, None))
    if settings_val is not None:
        return settings_val
    return _normalize_value(default)


def _int_setting(env_name: str, settings_attr: str, default: int) -> int:
    raw = _raw_setting(env_name, settings_attr, str(default))
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            f"tts_history_cleanup: invalid {env_name} value"
        )
        return default


def _resolve_cleanup_settings() -> tuple[int, int, int]:
    interval_hours = _int_setting(
        "TTS_HISTORY_PURGE_INTERVAL_HOURS",
        "TTS_HISTORY_PURGE_INTERVAL_HOURS",
        24,
    )
    retention_days = _int_setting(
        "TTS_HISTORY_RETENTION_DAYS",
        "TTS_HISTORY_RETENTION_DAYS",
        90,
    )
    max_rows = _int_setting(
        "TTS_HISTORY_MAX_ROWS_PER_USER",
        "TTS_HISTORY_MAX_ROWS_PER_USER",
        10000,
    )
    return interval_hours, retention_days, max_rows


def _enumerate_user_ids_from_fs() -> list[str]:
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "tts_history_cleanup: failed to resolve user db base dir"
        )
        return []
    uids: list[str] = []
    try:
        for p in base.iterdir():
            if p.is_dir():
                try:
                    int(p.name)
                    uids.append(p.name)
                except (TypeError, ValueError):
                    logger.debug("tts_history_cleanup: skipping non-int user dir")
    except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "tts_history_cleanup: failed to list user dirs"
        )
    if not uids:
        try:
            uids = [str(DatabasePaths.get_single_user_id())]
        except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                "tts_history_cleanup: single_user_id fallback failed"
            )
            uids = []
    return sorted(set(uids))


def _create_cleanup_db(db_path: str) -> Any:
    return create_media_database(
        client_id="tts_history_cleanup",
        db_path=db_path,
    )


def _purge_with_db(db: Any, user_ids: Iterable[str], retention_days: int, max_rows: int) -> int:
    removed_total = 0
    for uid in user_ids:
        try:
            removed = db.purge_tts_history_for_user(
                user_id=str(uid),
                retention_days=retention_days,
                max_rows=max_rows,
            )
            removed_total += removed
        except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                f"tts_history_cleanup: purge failed for user {uid}"
            )
    return removed_total


async def run_tts_history_cleanup_loop(stop_event: asyncio.Event | None = None) -> None:
    interval_hours, retention_days, max_rows = _resolve_cleanup_settings()

    if interval_hours <= 0 or (retention_days <= 0 and max_rows <= 0):
        logger.info("TTS history cleanup disabled by settings")
        return

    interval_sec = max(60, interval_hours * 3600)
    initial_delay = min(interval_sec, 60)
    if stop_event is not None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=initial_delay)
        if stop_event.is_set():
            return
    else:
        await asyncio.sleep(initial_delay)

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        removed_total = 0
        try:
            probe_db = _create_cleanup_db(
                str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
            )
            if probe_db.backend_type.name.lower() == "postgresql":
                user_ids = probe_db.list_tts_history_user_ids()
                removed_total = _purge_with_db(probe_db, user_ids, retention_days, max_rows)
            else:
                probe_db.close_connection()
                user_ids = _enumerate_user_ids_from_fs()
                for uid in user_ids:
                    db_path = DatabasePaths.get_media_db_path(uid)
                    db = _create_cleanup_db(str(db_path))
                    try:
                        removed_total += db.purge_tts_history_for_user(
                            user_id=str(uid),
                            retention_days=retention_days,
                            max_rows=max_rows,
                        )
                    finally:
                        db.close_connection()
                probe_db = None
            if removed_total:
                logger.info(f"TTS history cleanup removed={removed_total}")
        except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                "TTS history cleanup loop failed"
            )
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "tts_history_cleanup", "event": "cleanup_failed"},
                )
            except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for tts_history_cleanup")
        finally:
            try:
                if "probe_db" in locals() and probe_db is not None:
                    probe_db.close_connection()
            except _TTS_HISTORY_CLEANUP_NONCRITICAL_EXCEPTIONS:
                pass

        if stop_event is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        else:
            await asyncio.sleep(interval_sec)
