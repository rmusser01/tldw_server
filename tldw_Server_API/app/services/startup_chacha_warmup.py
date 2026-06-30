"""
ChaChaNotes warm-up extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def warm_chacha_notes_on_startup(
    *,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        _reset_chacha_shutdown_state()
        if _is_single_user_mode():
            auth_settings = _get_auth_settings()
            single_user_id = int(getattr(auth_settings, "SINGLE_USER_FIXED_ID", 1))
            _schedule_warm_chacha_task(single_user_id, str(single_user_id))
            logger.info(
                f"App Startup: scheduled ChaChaNotes warm-up for single-user id={single_user_id}"
            )
        else:
            logger.debug("ChaChaNotes warm-up skipped (multi-user mode)")
    except startup_guard_exceptions as exc:
        logger.warning(f"ChaChaNotes warm-up scheduling failed: {exc}")


def _reset_chacha_shutdown_state() -> None:
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        reset_chacha_shutdown_state,
    )

    reset_chacha_shutdown_state()


def _is_single_user_mode() -> bool:
    from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode

    return bool(is_single_user_mode())


def _get_auth_settings() -> Any:
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    return get_settings()


def _schedule_warm_chacha_task(user_id: int, client_id: str) -> asyncio.Task[Any]:
    return asyncio.create_task(_warm_chacha_db_for_user(user_id, client_id))


async def _warm_chacha_db_for_user(user_id: int, client_id: str) -> None:
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        warm_chacha_db_for_user,
    )

    await warm_chacha_db_for_user(user_id, client_id)
