"""
WebSub (W3C PubSubHubbub) push support for Collections Feeds.

Provides:
- Token / secret generation for callback security
- Callback URL construction from WEBSUB_CALLBACK_BASE_URL env var
- Subscribe / unsubscribe HTTP requests to hubs
- HMAC signature verification (SHA-256 preferred, SHA-1 fallback)
- Push notification XML parsing (reuses ET patterns from fetchers)
- Lease renewal for expiring subscriptions
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from loguru import logger

_WEBSUB_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    ET.ParseError,
    DefusedXmlException,
)


# ---------------------------------------------------------------------------
# Token / secret helpers
# ---------------------------------------------------------------------------

def generate_callback_token() -> str:
    """Generate a cryptographically random URL-safe callback token."""
    return secrets.token_urlsafe(32)


def generate_secret() -> str:
    """Generate a cryptographically random HMAC secret (hex-encoded)."""
    return secrets.token_hex(32)


def build_callback_url(token: str, user_id: int | str | None = None) -> str:
    """Construct the full public callback URL for a given token.

    Requires ``WEBSUB_CALLBACK_BASE_URL`` env var (e.g. ``https://myserver.com/api/v1``).
    When *user_id* is provided the URL encodes it so that unauthenticated hub
    callbacks can resolve the correct per-user database.
    """
    base = os.getenv("WEBSUB_CALLBACK_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError(
            "WEBSUB_CALLBACK_BASE_URL environment variable is required for WebSub support"
        )
    if user_id is not None:
        return f"{base}/websub/callback/{user_id}/{token}"
    return f"{base}/websub/callback/{token}"


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe requests
# ---------------------------------------------------------------------------

_DEFAULT_LEASE_SECONDS = int(os.getenv("WEBSUB_DEFAULT_LEASE_SECONDS", "864000") or "864000")


async def send_subscribe_request(
    hub_url: str,
    callback_url: str,
    topic_url: str,
    secret: str,
    lease_seconds: int | None = None,
) -> dict[str, Any]:
    """Send a ``hub.mode=subscribe`` POST to the hub.

    Returns ``{"status": int, "ok": bool}`` from the hub's response.
    """
    from tldw_Server_API.app.core.Security.egress import is_url_allowed

    if not is_url_allowed(hub_url):
        return {"status": 403, "ok": False, "error": "hub_url_blocked_by_egress_policy"}

    lease = lease_seconds if lease_seconds is not None else _DEFAULT_LEASE_SECONDS
    form_data = urlencode({
        "hub.callback": callback_url,
        "hub.mode": "subscribe",
        "hub.topic": topic_url,
        "hub.secret": secret,
        "hub.lease_seconds": str(lease),
    })

    try:
        from tldw_Server_API.app.core.http_client import afetch

        resp = await afetch(
            method="POST",
            url=hub_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=form_data.encode("utf-8"),
            timeout=15.0,
        )
        status = getattr(resp, "status_code", None) or getattr(resp, "status", 500)
        return {"status": int(status), "ok": 200 <= int(status) < 300}
    except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"WebSub subscribe request to {hub_url} failed: {exc}")
        return {"status": 500, "ok": False, "error": str(exc)}


async def send_unsubscribe_request(
    hub_url: str,
    callback_url: str,
    topic_url: str,
    secret: str,
) -> dict[str, Any]:
    """Send a ``hub.mode=unsubscribe`` POST to the hub."""
    from tldw_Server_API.app.core.Security.egress import is_url_allowed

    if not is_url_allowed(hub_url):
        return {"status": 403, "ok": False, "error": "hub_url_blocked_by_egress_policy"}

    form_data = urlencode({
        "hub.callback": callback_url,
        "hub.mode": "unsubscribe",
        "hub.topic": topic_url,
        "hub.secret": secret,
    })

    try:
        from tldw_Server_API.app.core.http_client import afetch

        resp = await afetch(
            method="POST",
            url=hub_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=form_data.encode("utf-8"),
            timeout=15.0,
        )
        status = getattr(resp, "status_code", None) or getattr(resp, "status", 500)
        return {"status": int(status), "ok": 200 <= int(status) < 300}
    except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"WebSub unsubscribe request to {hub_url} failed: {exc}")
        return {"status": 500, "ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# HMAC signature verification
# ---------------------------------------------------------------------------

_SUPPORTED_ALGORITHMS = {
    "sha256": hashlib.sha256,
    "sha1": hashlib.sha1,
    "sha384": hashlib.sha384,
    "sha512": hashlib.sha512,
}


def verify_hub_signature(
    body: bytes,
    signature_header: str | None,
    secret: str,
) -> bool:
    """Validate the ``X-Hub-Signature`` header using HMAC.

    Supports ``sha256=<hex>``, ``sha1=<hex>``, ``sha384=<hex>``, ``sha512=<hex>``.
    Returns False if the header is missing or doesn't match.
    """
    if not signature_header:
        return False

    try:
        algo_name, _, hex_digest = signature_header.partition("=")
        algo_name = algo_name.strip().lower()
        hex_digest = hex_digest.strip()
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        return False

    hash_fn = _SUPPORTED_ALGORITHMS.get(algo_name)
    if hash_fn is None:
        logger.debug(f"WebSub: unsupported signature algorithm '{algo_name}'")
        return False

    expected = hmac.new(secret.encode("utf-8"), body, hash_fn).hexdigest()
    return hmac.compare_digest(expected, hex_digest)


# ---------------------------------------------------------------------------
# Push notification XML parsing
# ---------------------------------------------------------------------------

def parse_push_items(xml_bytes: bytes) -> list[dict[str, Any]]:
    """Parse feed XML from a hub push notification into normalized item dicts.

    Reuses the same ET + item extraction pattern as ``fetchers.fetch_rss_feed``.
    Returns ``[{"title", "url", "summary", "published", "guid"?}, ...]``.
    """
    try:
        text = xml_bytes.decode("utf-8", errors="replace")
        root = ET.fromstring(text)
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        return []

    def _find_text(node, names):
        for n in names:
            x = node.find(n)
            if x is not None and (x.text or "").strip():
                return x.text.strip()
        return None

    items_nodes = root.findall(".//item")
    if not items_nodes:
        items_nodes = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    atom_link_tag = "{http://www.w3.org/2005/Atom}link"
    atom_title_tag = "{http://www.w3.org/2005/Atom}title"
    atom_summary_tag = "{http://www.w3.org/2005/Atom}summary"
    atom_content_tag = "{http://www.w3.org/2005/Atom}content"
    atom_updated_tag = "{http://www.w3.org/2005/Atom}updated"
    atom_published_tag = "{http://www.w3.org/2005/Atom}published"
    atom_id_tag = "{http://www.w3.org/2005/Atom}id"

    items: list[dict[str, Any]] = []
    for it in items_nodes:
        title = _find_text(it, ["title", atom_title_tag]) or ""

        link = ""
        link_nodes = list(it.findall("link")) + list(it.findall(atom_link_tag))
        preferred_link = ""
        fallback_link = ""
        for node in link_nodes:
            candidate = (node.get("href") or (node.text or "")).strip()
            if not candidate:
                continue
            rel = (node.get("rel") or "").lower()
            if rel == "alternate" and not preferred_link:
                preferred_link = candidate
            elif rel not in {"self"} and not fallback_link:
                fallback_link = candidate
        link = preferred_link or fallback_link or _find_text(it, ["link", atom_link_tag]) or ""

        summary = _find_text(it, ["description", atom_summary_tag, atom_content_tag]) or ""
        published = _find_text(it, ["pubDate", atom_updated_tag, atom_published_tag]) or None
        guid = _find_text(it, ["guid", atom_id_tag]) or None

        rec: dict[str, Any] = {"title": title, "url": link or "", "summary": summary, "published": published}
        if guid:
            rec["guid"] = guid
        items.append(rec)

    return items


# ---------------------------------------------------------------------------
# Lease renewal
# ---------------------------------------------------------------------------

_RENEWAL_BEFORE_SECONDS = int(os.getenv("WEBSUB_RENEWAL_BEFORE_SECONDS", "3600") or "3600")
_RENEWAL_CHECK_INTERVAL = int(os.getenv("WEBSUB_RENEWAL_CHECK_INTERVAL", "1800") or "1800")


def _iter_user_ids_with_websub() -> list[int]:
    """Return user IDs that may have WebSub subscriptions.

    Scans the ``user_databases/`` directory for numeric subdirectories whose
    SQLite DB contains the ``feed_websub_subscriptions`` table.  Falls back to
    ``[1]`` when the directory cannot be listed (e.g. single-user mode).
    """
    from pathlib import Path

    try:
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

        base = DatabasePaths.get_base_directory()
        ids: list[int] = []
        for child in base.iterdir():
            if child.is_dir() and child.name.isdigit():
                ids.append(int(child.name))
        if ids:
            return sorted(ids)
    except _WEBSUB_NONCRITICAL_EXCEPTIONS:
        pass
    # Fallback: single-user
    return [int(os.getenv("SINGLE_USER_FIXED_ID", "1") or "1")]


async def renew_expiring_subscriptions(renew_before_seconds: int | None = None) -> int:
    """Find subscriptions expiring soon and re-subscribe to their hubs.

    Iterates over all user databases so multi-user subscriptions are renewed.
    Returns the number of subscriptions renewed.
    """
    before_sec = renew_before_seconds if renew_before_seconds is not None else _RENEWAL_BEFORE_SECONDS
    cutoff = (datetime.now(timezone.utc) + timedelta(seconds=before_sec)).isoformat()
    renewed = 0

    try:
        from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

        for user_id in _iter_user_ids_with_websub():
            try:
                db = WatchlistsDatabase.for_user(user_id)
                expiring = db.list_expiring_websub_subscriptions(cutoff)

                for sub in expiring:
                    try:
                        callback_url = build_callback_url(sub.callback_token, user_id=sub.user_id)
                        result = await send_subscribe_request(
                            hub_url=sub.hub_url,
                            callback_url=callback_url,
                            topic_url=sub.topic_url,
                            secret=sub.secret,
                            lease_seconds=sub.lease_seconds or _DEFAULT_LEASE_SECONDS,
                        )
                        if result.get("ok"):
                            db.update_websub_subscription(sub.id, {"state": "pending"})
                            renewed += 1
                            logger.debug(f"WebSub: renewed subscription {sub.id} for {sub.topic_url}")
                        else:
                            logger.warning(f"WebSub: renewal failed for sub {sub.id}: {result}")
                    except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
                        logger.warning(f"WebSub: renewal error for sub {sub.id}: {exc}")
            except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"WebSub: error scanning user {user_id}: {exc}")
    except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"WebSub: renew_expiring_subscriptions error: {exc}")

    return renewed


async def websub_renewal_loop() -> None:
    """Background loop that periodically renews expiring WebSub subscriptions.

    Designed to be launched via ``asyncio.create_task()`` and cancelled on shutdown.
    """
    import asyncio

    interval = _RENEWAL_CHECK_INTERVAL
    logger.info(f"WebSub renewal loop started (interval={interval}s, renew_before={_RENEWAL_BEFORE_SECONDS}s)")
    while True:
        try:
            await asyncio.sleep(interval)
            count = await renew_expiring_subscriptions()
            if count > 0:
                logger.info(f"WebSub: renewed {count} expiring subscription(s)")
        except asyncio.CancelledError:
            logger.info("WebSub renewal loop cancelled")
            break
        except _WEBSUB_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"WebSub renewal loop error: {exc}")
