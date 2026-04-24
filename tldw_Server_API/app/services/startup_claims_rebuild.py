"""
Claims rebuild startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from loguru import logger

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def start_claims_rebuild_worker(app_settings: Mapping[str, Any]) -> Any | None:
    """Start the claims rebuild worker when enabled."""
    try:
        enabled = bool(app_settings.get("CLAIMS_REBUILD_ENABLED", False))
        if not enabled:
            logger.info("Claims rebuild worker disabled by settings")
            return None

        interval_sec = int(app_settings.get("CLAIMS_REBUILD_INTERVAL_SEC", 3600))
        policy = str(app_settings.get("CLAIMS_REBUILD_POLICY", "missing")).lower()

        async def _claims_rebuild_loop() -> None:
            logger.info(f"Starting claims rebuild worker (every {interval_sec}s, policy={policy})")
            service = _get_claims_rebuild_service()
            while True:
                try:
                    run_claims_rebuild_iteration(app_settings, service, policy=policy)
                except _STARTUP_GUARD_EXCEPTIONS as exc:
                    logger.warning(f"Claims rebuild loop error: {exc}")
                await asyncio.sleep(interval_sec)

        return asyncio.create_task(_claims_rebuild_loop())
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start claims rebuild worker: {exc}")
        return None


def run_claims_rebuild_iteration(
    app_settings: Mapping[str, Any],
    service: Any,
    *,
    policy: str,
) -> None:
    """Submit one claims rebuild batch for the configured default user."""
    with _claims_rebuild_db_session(app_settings) as (_, db_path, db):
        media_ids = _list_claims_rebuild_media_ids(
            db,
            policy=policy,
            stale_days=int(app_settings.get("CLAIMS_STALE_DAYS", 7)),
            compare_media_last_modified=False,
            limit=25,
        )
        for media_id in media_ids:
            service.submit(media_id=media_id, db_path=db_path)


@contextmanager
def _claims_rebuild_db_session(
    app_settings: Mapping[str, Any],
) -> Iterator[tuple[int, str, Any]]:
    """Yield one managed Media DB session for the claims rebuild worker loop."""
    user_id = int(app_settings.get("SINGLE_USER_FIXED_ID", "1"))
    db_path = str(_get_user_media_db_path(user_id))
    client_id = str(app_settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"))
    with _managed_media_database(
        client_id=client_id,
        db_path=db_path,
        initialize=False,
    ) as db:
        yield user_id, db_path, db


def _get_claims_rebuild_service() -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import (
        get_claims_rebuild_service as _get_claims_svc,
    )

    return _get_claims_svc()


def _get_user_media_db_path(user_id: int) -> Any:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import (
        get_user_media_db_path,
    )

    return get_user_media_db_path(user_id)


def _managed_media_database(*, client_id: str, db_path: str, initialize: bool) -> Any:
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        managed_media_database,
    )

    return managed_media_database(
        client_id=client_id,
        db_path=db_path,
        initialize=initialize,
    )


def _list_claims_rebuild_media_ids(
    db: Any,
    *,
    policy: str,
    stale_days: int,
    compare_media_last_modified: bool,
    limit: int,
) -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
        list_claims_rebuild_media_ids,
    )

    return list_claims_rebuild_media_ids(
        db,
        policy=policy,
        stale_days=stale_days,
        compare_media_last_modified=compare_media_last_modified,
        limit=limit,
    )
