"""The "new sign-in" inbox alert (TASK-15649).

When a user signs in while another session of theirs is live (or one was just evicted by the
session cap), a ``new_sign_in`` notification is written to their inbox. It is best-effort and
runs off the request path: sign-in never depends on it.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from datetime import datetime, timedelta, timezone

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.device_label import device_label
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, UserNotificationRow
from tldw_Server_API.app.core.exceptions import DatabaseError as AuthnzDatabaseError
from tldw_Server_API.app.core.exceptions import StoragePathValidationError
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter

NEW_SIGN_IN_KIND = "new_sign_in"
NEW_SIGN_IN_TITLE = "New sign-in to your account"
NEW_SIGN_IN_SEVERITY = "warning"
NEW_SIGN_IN_SOURCE_DOMAIN = "auth_session"
NEW_SIGN_IN_LINK_URL = "/settings/sessions"
NEW_SIGN_IN_DEDUPE_BUCKET_SECONDS = 600
NEW_SIGN_IN_OPEN_AFTER = timedelta(hours=24)

SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS = (
    AuthnzDatabaseError,
    BackendDatabaseError,
    StoragePathValidationError,
    sqlite3.Error,
    OSError,
    KeyError,
    ValueError,
    TypeError,
    RuntimeError,
    TimeoutError,
    AttributeError,
)

_PENDING: set[asyncio.Task] = set()


def _install_id(device_id: str | None) -> str | None:
    # Lazy: interactive_session imports session_manager, which imports this module.
    from tldw_Server_API.app.core.AuthNZ.interactive_session import install_id_or_none

    return install_id_or_none(device_id)


def new_sign_in_message(*, user_agent: str | None, ip_address: str | None, now: datetime) -> str:
    where = ip_address[:64] if ip_address and ip_address != "unknown" else "an unknown address"
    return (
        f"{device_label(user_agent)} (as the browser reported it) signed in from {where} "
        f"at {now:%Y-%m-%d %H:%M} UTC. If this was not you, sign out your other sessions."
    )


def new_sign_in_dedupe_key(
    *, user_id: int, device_id: str | None, ip_address: str | None, user_agent: str | None, now: datetime
) -> str:
    group = _install_id(device_id) or f"{ip_address}|{user_agent}"
    bucket = int(now.timestamp()) // NEW_SIGN_IN_DEDUPE_BUCKET_SECONDS
    digest = hashlib.sha256(f"{user_id}|{group}|{bucket}".encode()).hexdigest()[:32]
    return f"{NEW_SIGN_IN_KIND}:{digest}"


def write_new_sign_in_alert(
    db: CollectionsDatabase,
    *,
    session_id: int,
    ip_address: str | None,
    user_agent: str | None,
    device_id: str | None,
    user_id: int,
    now: datetime | None = None,
) -> UserNotificationRow:
    now = now or datetime.now(timezone.utc)
    return db.create_user_notification(
        kind=NEW_SIGN_IN_KIND,
        title=NEW_SIGN_IN_TITLE,
        message=new_sign_in_message(user_agent=user_agent, ip_address=ip_address, now=now),
        severity=NEW_SIGN_IN_SEVERITY,
        source_job_id=str(session_id),
        source_domain=NEW_SIGN_IN_SOURCE_DOMAIN,
        link_type="settings_route",
        link_url=NEW_SIGN_IN_LINK_URL,
        dedupe_key=new_sign_in_dedupe_key(
            user_id=user_id, device_id=device_id, ip_address=ip_address, user_agent=user_agent, now=now
        ),
    )


def _write_in_user_db(user_id: int, **kwargs) -> None:
    with CollectionsDatabase.for_user(user_id=user_id) as db:
        write_new_sign_in_alert(db, user_id=user_id, **kwargs)


async def _alert_if_new_install(
    *,
    user_id: int,
    session_id: int,
    ip_address: str | None,
    user_agent: str | None,
    device_id: str | None,
    db_pool: DatabasePool,
    evicted_device_ids: tuple[str | None, ...],
) -> None:
    try:
        rows = await AuthnzSessionsRepo(db_pool).get_active_sessions_for_user(user_id)
        others = [r for r in rows if r["id"] != session_id]
        if not others and not evicted_device_ids:
            return
        new = _install_id(device_id)
        known = {_install_id(r.get("device_id")) for r in others} | {_install_id(d) for d in evicted_device_ids}
        if new is not None and new in known:
            return
        await asyncio.to_thread(
            _write_in_user_db,
            user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
        )
    except SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS as exc:
        log_counter("auth_sign_in_alert_failed")
        logger.bind(error_type=type(exc).__name__).warning("New sign-in alert not written")


def schedule_new_sign_in_alert(
    *,
    user_id: int,
    session_id: int,
    ip_address: str | None,
    user_agent: str | None,
    device_id: str | None,
    db_pool: DatabasePool,
    evicted_device_ids: tuple[str | None, ...] = (),
) -> None:
    """Fire-and-forget the alert; a no-op without a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(
        _alert_if_new_install(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
            db_pool=db_pool,
            evicted_device_ids=evicted_device_ids,
        )
    )
    _PENDING.add(task)
    task.add_done_callback(_PENDING.discard)


async def drain_pending_alerts(timeout: float = 5.0) -> None:
    """Tests only: wait for scheduled alerts.

    The tasks live on the TestClient portal's loop, so ``asyncio.gather`` would raise
    "attached to a different loop"; poll instead.
    """
    deadline = time.monotonic() + timeout
    while list(_PENDING) and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    if list(_PENDING):
        raise AssertionError("new sign-in alerts did not finish in time")
