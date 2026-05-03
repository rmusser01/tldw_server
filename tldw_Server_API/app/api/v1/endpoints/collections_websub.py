"""
WebSub (PubSubHubbub) push endpoints for Collections Feeds.

Two routers:
- ``router``: authenticated management endpoints (subscribe, unsubscribe, status)
- ``callback_router``: public callback endpoints (hub verification + push notification)
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query, Request, Response
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Watchlists_DB_Deps import get_watchlists_db_for_user
from tldw_Server_API.app.api.v1.schemas.collections_websub_schemas import (
    WebSubSubscribeRequest,
    WebSubSubscriptionResponse,
    WebSubUnsubscribeResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

_WEBSUB_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
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

# ---------------------------------------------------------------------------
# Authenticated management endpoints
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/collections/feeds", tags=["collections-websub"])


def _sub_to_response(sub) -> WebSubSubscriptionResponse:
    return WebSubSubscriptionResponse(
        id=sub.id,
        source_id=sub.source_id,
        hub_url=sub.hub_url,
        topic_url=sub.topic_url,
        state=sub.state,
        lease_seconds=sub.lease_seconds,
        verified_at=sub.verified_at,
        expires_at=sub.expires_at,
        last_push_at=sub.last_push_at,
        created_at=sub.created_at,
    )


@router.post(
    "/{feed_id}/websub/subscribe",
    response_model=WebSubSubscriptionResponse,
    summary="Subscribe a feed to WebSub push notifications",
)
async def websub_subscribe(
    feed_id: int = Path(..., ge=1),
    payload: WebSubSubscribeRequest = Body(WebSubSubscribeRequest()),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> WebSubSubscriptionResponse:
    # Validate source exists
    try:
        source = db.get_source(feed_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="feed_not_found") from None

    # Check for existing active subscription
    existing = db.get_websub_subscription_for_source(feed_id)
    if existing and existing.state in ("pending", "verified"):
        return _sub_to_response(existing)

    # Fetch feed once — validate and discover hub URL from the same response
    try:
        from tldw_Server_API.app.core.Watchlists.fetchers import discover_hub_url
        from tldw_Server_API.app.core.http_client import afetch

        resp = await afetch(method="GET", url=source.url, timeout=10.0)
        status = getattr(resp, "status_code", None) or getattr(resp, "status", 0)
        if not (200 <= int(status) < 300):
            raise HTTPException(status_code=502, detail="feed_fetch_failed")

        resp_text = ""
        resp_headers: dict[str, str] = {}
        try:
            resp_text = resp.text if hasattr(resp, "text") else (resp.content or b"").decode("utf-8", errors="replace")
            resp_headers = dict(getattr(resp, "headers", {}))
        except _WEBSUB_NONCRITICAL_EXCEPTIONS:
            pass

        hub_url, self_url = discover_hub_url(resp_text, resp_headers)
    except HTTPException:
        raise
    except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("WebSub hub discovery failed")
        raise HTTPException(status_code=502, detail="hub_discovery_failed") from exc

    if not hub_url:
        raise HTTPException(status_code=404, detail="no_websub_hub_found")

    topic_url = self_url or source.url

    # Validate hub URL
    from tldw_Server_API.app.core.Security.egress import is_url_allowed

    if not is_url_allowed(hub_url):
        raise HTTPException(status_code=403, detail="hub_url_blocked_by_egress_policy")

    # Generate token + secret, create DB record
    from tldw_Server_API.app.core.Watchlists.websub import (
        build_callback_url,
        generate_callback_token,
        generate_secret,
        send_subscribe_request,
    )

    token = generate_callback_token()
    secret = generate_secret()
    callback_url = build_callback_url(token, user_id=current_user.id)

    sub = db.create_websub_subscription(
        source_id=feed_id,
        hub_url=hub_url,
        topic_url=topic_url,
        callback_token=token,
        secret=secret,
        lease_seconds=payload.lease_seconds,
    )

    # Send subscribe request to hub (async, best-effort)
    try:
        result = await send_subscribe_request(
            hub_url=hub_url,
            callback_url=callback_url,
            topic_url=topic_url,
            secret=secret,
            lease_seconds=payload.lease_seconds,
        )
        if not result.get("ok"):
            logger.warning("WebSub: hub returned non-2xx for subscribe")
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        logger.warning("WebSub: subscribe request failed")

    # Re-fetch to get latest state
    try:
        sub = db.get_websub_subscription(sub.id)
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        pass

    return _sub_to_response(sub)


@router.delete(
    "/{feed_id}/websub",
    response_model=WebSubUnsubscribeResponse,
    summary="Unsubscribe a feed from WebSub push notifications",
)
async def websub_unsubscribe(
    feed_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> WebSubUnsubscribeResponse:
    sub = db.get_websub_subscription_for_source(feed_id)
    if not sub:
        raise HTTPException(status_code=404, detail="no_websub_subscription")

    # Send unsubscribe to hub
    from tldw_Server_API.app.core.Watchlists.websub import build_callback_url, send_unsubscribe_request

    try:
        callback_url = build_callback_url(sub.callback_token, user_id=sub.user_id)
        await send_unsubscribe_request(
            hub_url=sub.hub_url,
            callback_url=callback_url,
            topic_url=sub.topic_url,
            secret=sub.secret,
        )
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        logger.warning("WebSub: unsubscribe request failed")

    db.update_websub_subscription(sub.id, {"state": "unsubscribed"})

    return WebSubUnsubscribeResponse(message="unsubscribed", state="unsubscribed")


@router.get(
    "/{feed_id}/websub",
    response_model=WebSubSubscriptionResponse,
    summary="Get WebSub subscription status for a feed",
)
async def websub_status(
    feed_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> WebSubSubscriptionResponse:
    sub = db.get_websub_subscription_for_source(feed_id)
    if not sub:
        raise HTTPException(status_code=404, detail="no_websub_subscription")
    return _sub_to_response(sub)


# ---------------------------------------------------------------------------
# Public callback endpoints (no auth — called by hubs)
# ---------------------------------------------------------------------------

callback_router = APIRouter(prefix="/websub/callback", tags=["collections-websub"])


@callback_router.get(
    "/{user_id}/{callback_token}",
    response_class=Response,
    responses={
        200: {
            "description": "Plaintext WebSub hub challenge response.",
            "content": {
                "text/plain": {},
            },
        },
    },
    summary="WebSub hub verification challenge",
)
async def websub_verify_callback(
    user_id: int = Path(..., ge=1),
    callback_token: str = Path(...),
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_topic: str = Query(..., alias="hub.topic"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_lease_seconds: int | None = Query(None, alias="hub.lease_seconds"),
) -> Response:
    """Handle hub verification (subscribe/unsubscribe intent verification).

    The hub sends a GET with ``hub.mode``, ``hub.topic``, ``hub.challenge``,
    and optionally ``hub.lease_seconds``. We echo back ``hub.challenge`` as
    plain text with 200 if everything checks out.
    """
    sub = _lookup_subscription_by_token(callback_token, user_id=user_id)
    if not sub:
        raise HTTPException(status_code=404, detail="unknown_callback_token")

    # Verify topic matches
    if sub.topic_url != hub_topic:
        raise HTTPException(status_code=404, detail="topic_mismatch")

    db = WatchlistsDatabase(user_id=user_id)

    if hub_mode == "subscribe":
        from datetime import datetime, timedelta, timezone

        patch: dict[str, Any] = {"state": "verified", "verified_at": datetime.now(timezone.utc).isoformat()}
        if hub_lease_seconds is not None and hub_lease_seconds > 0:
            patch["lease_seconds"] = hub_lease_seconds
            patch["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=hub_lease_seconds)).isoformat()
        db.update_websub_subscription(sub.id, patch)
    elif hub_mode == "unsubscribe":
        db.update_websub_subscription(sub.id, {"state": "unsubscribed"})
    else:
        raise HTTPException(status_code=400, detail="unsupported_hub_mode")

    return Response(content=hub_challenge, media_type="text/plain", status_code=200)


@callback_router.post(
    "/{user_id}/{callback_token}",
    summary="WebSub push notification",
    response_class=Response,
    responses={
        200: {
            "description": "WebSub hub acknowledgement; no response body.",
        },
    },
)
async def websub_push_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: int = Path(..., ge=1),
    callback_token: str = Path(...),
) -> Response:
    """Receive a push notification from a WebSub hub.

    Validates HMAC signature, parses feed XML, and upserts items into Collections.
    Item processing runs in a background task so the hub gets a fast 200 ack.
    """
    sub = _lookup_subscription_by_token(callback_token, user_id=user_id)
    if not sub:
        raise HTTPException(status_code=404, detail="unknown_callback_token")

    if sub.state not in ("verified", "pending"):
        raise HTTPException(status_code=410, detail="subscription_not_active")

    body = await request.body()

    # Validate HMAC signature
    from tldw_Server_API.app.core.Watchlists.websub import verify_hub_signature

    signature_header = request.headers.get("X-Hub-Signature")
    if not verify_hub_signature(body, signature_header, sub.secret):
        logger.warning("WebSub: invalid signature")
        raise HTTPException(status_code=403, detail="invalid_signature")

    # Parse items from push XML
    from tldw_Server_API.app.core.Watchlists.websub import parse_push_items

    items = parse_push_items(body)
    logger.debug(f"WebSub: received {len(items)} items for source {sub.source_id}")

    # Schedule item processing + last_push_at update as a background task so
    # the 200 is returned immediately (hubs expect a fast ack).
    if items:
        background_tasks.add_task(_process_and_record_push, sub, items)
    else:
        background_tasks.add_task(_record_push_timestamp, sub)

    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_subscription_by_token(callback_token: str, *, user_id: int):
    """Look up a WebSub subscription by its unique callback token.

    The *user_id* is extracted from the callback URL path so we open the
    correct per-user database (fixing the previous hard-coded ``user_id=1``).
    """
    try:
        db = WatchlistsDatabase(user_id=user_id)
        return db.get_websub_subscription_by_token(callback_token)
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        return None


def _record_push_timestamp(sub) -> None:
    """Update ``last_push_at`` for a subscription (runs in background)."""
    from datetime import datetime, timezone

    try:
        db = WatchlistsDatabase(user_id=sub.user_id)
        db.update_websub_subscription(sub.id, {"last_push_at": datetime.now(timezone.utc).isoformat()})
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        pass


def _process_and_record_push(sub, items: list[dict[str, Any]]) -> None:
    """Process push items and update timestamp (runs in background)."""
    from datetime import datetime, timezone

    try:
        _process_push_items(sub, items)
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        logger.warning("WebSub: error processing push items")

    _record_push_timestamp(sub)


def _process_push_items(sub, items: list[dict[str, Any]]) -> int:
    """Upsert push items into the Collections content_items table.

    Reuses the same sanitize -> dedupe -> upsert path as polled items.
    """
    from tldw_Server_API.app.core.Collections.utils import hash_text_sha256, truncate_text, word_count
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    collections_db = CollectionsDatabase.for_user(int(sub.user_id))
    watchlists_db = WatchlistsDatabase(user_id=sub.user_id)

    count = 0
    for item in items:
        url = item.get("url") or ""
        title = item.get("title") or "Untitled"
        summary = item.get("summary") or ""
        published = item.get("published")
        guid = item.get("guid")

        # Dedupe: check if we've seen this item
        dedupe_key = guid or url
        if dedupe_key and watchlists_db.has_seen_item(sub.source_id, dedupe_key):
            continue

        # Sanitize HTML
        try:
            from tldw_Server_API.app.core.Watchlists.pipeline import _sanitize_feed_html
            summary = _sanitize_feed_html(summary) or ""
        except _WEBSUB_NONCRITICAL_EXCEPTIONS:
            pass

        metadata = {
            "source_id": sub.source_id,
            "origin": "feed",
            "websub_push": True,
        }

        try:
            collections_db.upsert_content_item(
                origin="feed",
                origin_type="rss",
                origin_id=sub.source_id,
                url=url,
                canonical_url=url,
                domain=None,
                title=title,
                summary=truncate_text(summary, 600),
                content_hash=hash_text_sha256(summary),
                word_count=word_count(summary),
                published_at=published,
                status="new",
                favorite=False,
                metadata=metadata,
                source_id=sub.source_id,
            )
            count += 1
        except _WEBSUB_NONCRITICAL_EXCEPTIONS:
            logger.debug("WebSub: upsert failed")
            continue

        # Mark seen
        if dedupe_key:
            try:
                watchlists_db.mark_seen_item(sub.source_id, dedupe_key)
            except _WEBSUB_NONCRITICAL_EXCEPTIONS:
                pass

    return count
