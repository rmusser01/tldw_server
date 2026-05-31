"""
Storage quota enforcement dependency for upload endpoints.

Checks the authenticated user's org/team storage quota before allowing
file uploads.  When the user is already at or above their hard limit the
request is rejected with 413 (Payload Too Large).  A soft-limit warning
is communicated via the ``X-Storage-Warning`` response header.

Usage::

    @router.post("/process-videos", dependencies=[Depends(guard_storage_quota)])
    async def process_videos_endpoint(...): ...
"""
from __future__ import annotations

from contextlib import suppress
import os

from fastapi import Depends, HTTPException, Request, Response, UploadFile, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_profile_mode
from tldw_Server_API.app.core.Storage.quota_enforcement import check_storage_quota

_NONCRITICAL = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _is_enabled() -> bool:
    """Return True unless storage quota enforcement is explicitly disabled."""
    val = os.getenv("STORAGE_QUOTA_ENFORCEMENT", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


async def guard_storage_quota(
    request: Request,
    response: Response,
    current_user: User = Depends(get_request_user),
) -> None:
    """FastAPI dependency that blocks requests when storage quota is exceeded.

    Behaviour:
    * Disabled in single-user profile mode (local / desktop).
    * Disabled when ``STORAGE_QUOTA_ENFORCEMENT=0``.
    * Fail-open: if the quota check itself errors, the request proceeds.
    * Adds ``X-Storage-Warning`` header when the soft limit is reached.
    * Returns HTTP 413 when the hard limit is reached or remaining quota is 0.
    """
    if not _is_enabled():
        return

    # Skip enforcement for single-user / local deployments
    try:
        if is_single_user_profile_mode():
            return
    except _NONCRITICAL:
        pass

    # Resolve org_id from request state (set by auth middleware)
    org_id: int | None = None
    try:
        org_id = getattr(request.state, "org_id", None)
        if org_id is None and hasattr(current_user, "active_org_id"):
            org_id = current_user.active_org_id
        if org_id is None and hasattr(current_user, "org_ids") and current_user.org_ids:
            org_id = current_user.org_ids[0]
    except _NONCRITICAL:
        pass

    # Estimate upload size from Content-Length (best-effort; 0 if unavailable)
    upload_bytes: int = 0
    try:
        cl = request.headers.get("content-length")
        if cl is not None:
            upload_bytes = int(cl)
    except _NONCRITICAL:
        pass

    raw_user_id = getattr(current_user, "id", None)
    try:
        user_id = int(raw_user_id) if raw_user_id is not None else 0
    except (TypeError, ValueError):
        user_id = 0
        logger.warning("Storage quota guard falling back to anonymous user_id=0 for user {}", raw_user_id)

    try:
        db_pool = await get_db_pool()
        result = await check_storage_quota(
            user_id=user_id,
            file_size_bytes=upload_bytes,
            db_pool=db_pool,
            org_id=org_id,
        )
    except _NONCRITICAL as exc:
        # Fail-open: log and allow
        logger.warning("Storage quota guard failed (fail-open): {}", type(exc).__name__)
        return

    if not result.get("allowed", True):
        reason = result.get("reason", "Storage quota exceeded")
        used_mb = result.get("used_mb", 0)
        quota_mb = result.get("quota_mb", 0)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "storage_quota_exceeded",
                "message": reason,
                "used_mb": used_mb,
                "quota_mb": quota_mb,
                "remaining_mb": result.get("remaining_mb", 0),
            },
        )

    # Soft-limit warning header
    if "soft limit" in result.get("reason", "").lower():
        with suppress(*_NONCRITICAL):
            response.headers["X-Storage-Warning"] = result["reason"]
