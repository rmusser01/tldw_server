"""Podcast RSS publication adapter."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as DET
from defusedxml.common import DefusedXmlException
from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_workflow_file_path,
    resolve_workflow_file_uri,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import PodcastRSSPublishConfig

_PODCAST_RSS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
    DefusedXmlException,
)


class _InvalidRemoteSeedUrl(ValueError):
    """Raised when source_feed_url uses a disallowed scheme or form."""


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child(parent: ET.Element, tag: str) -> ET.Element | None:
    for child in list(parent):
        if _local_name(child.tag) == tag:
            return child
    return None


def _find_children(parent: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local_name(child.tag) == tag]


def _find_text(parent: ET.Element, tag: str) -> str | None:
    child = _find_child(parent, tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _upsert_text(parent: ET.Element, tag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    node = _find_child(parent, tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    node.text = text


def _to_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return 0.0
        with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
            return parsedate_to_datetime(raw).timestamp()
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
            return datetime.fromisoformat(normalized).timestamp()
        with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
            return float(raw)
    return 0.0


def _to_rfc2822(value: Any) -> str:
    dt: datetime | None = None
    if value is None:
        return format_datetime(datetime.now(timezone.utc))
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return format_datetime(datetime.now(timezone.utc))
        with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
            dt = parsedate_to_datetime(raw)
        if dt is None:
            normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
                dt = datetime.fromisoformat(normalized)
        if dt is None:
            return format_datetime(datetime.now(timezone.utc))
    else:
        return format_datetime(datetime.now(timezone.utc))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return format_datetime(dt.astimezone(timezone.utc))


def _episode_audio_ref(episode: dict[str, Any]) -> str | None:
    for key in ("enclosure_url", "audio_url", "audio_uri", "file_uri", "uri", "url"):
        value = episode.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _episode_guid(episode: dict[str, Any], audio_ref: str | None) -> str:
    for key in ("guid", "id", "episode_id"):
        value = episode.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    for value in (episode.get("link"), episode.get("url"), audio_ref):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    seed = "|".join(
        [
            str(episode.get("title") or "").strip(),
            str(episode.get("published_at") or episode.get("pub_date") or "").strip(),
            str(audio_ref or "").strip(),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"episode-{digest}"


def _build_episode_item(episode: dict[str, Any]) -> tuple[str, ET.Element]:
    title = str(episode.get("title") or "Untitled Episode").strip()
    description = str(
        episode.get("description")
        or episode.get("summary")
        or episode.get("content")
        or ""
    ).strip()
    link = str(episode.get("link") or episode.get("source_url") or "").strip() or None
    pub_date_raw = (
        episode.get("pub_date")
        or episode.get("published_at")
        or episode.get("published")
        or episode.get("created_at")
    )
    audio_ref = _episode_audio_ref(episode)
    guid = _episode_guid(episode, audio_ref)

    item = ET.Element("item")
    ET.SubElement(item, "title").text = title
    if link:
        ET.SubElement(item, "link").text = link
    guid_node = ET.SubElement(item, "guid")
    guid_node.text = guid
    guid_node.set("isPermaLink", "false")
    if description:
        ET.SubElement(item, "description").text = description
    ET.SubElement(item, "pubDate").text = _to_rfc2822(pub_date_raw)

    if audio_ref:
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", audio_ref)
        enclosure.set("type", str(episode.get("enclosure_type") or "audio/mpeg"))
        length = episode.get("enclosure_length") or episode.get("size_bytes") or 0
        enclosure.set("length", str(length))

    return guid, item


def _resolve_feed_path(feed_uri: str, context: dict[str, Any], config: dict[str, Any]) -> Path:
    if feed_uri.startswith("file://"):
        return resolve_workflow_file_uri(feed_uri, context, config)
    return resolve_workflow_file_path(feed_uri, context, config)


def _load_seed_xml(source_feed_url: str) -> ET.Element | None:
    parsed = urlparse(source_feed_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise _InvalidRemoteSeedUrl("source_feed_url must use http:// or https://")
    request = Request(source_feed_url, headers={"User-Agent": "tldw-workflows/0.2"})
    with urlopen(request, timeout=15) as response:  # nosec B310
        payload = response.read()
    if not payload:
        return None
    return DET.fromstring(payload)


@registry.register(
    "podcast_rss_publish",
    category="integration",
    description="Publish/update podcast RSS feed with dedupe and deterministic ordering",
    parallelizable=False,
    tags=["integration", "rss", "podcast"],
    config_model=PodcastRSSPublishConfig,
)
async def run_podcast_rss_publish_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Publish an episode entry to an RSS feed with dedupe and optimistic concurrency."""
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    feed_uri = str(config.get("feed_uri") or "").strip()
    if not feed_uri:
        return {"error": "missing_feed_uri", "published": False}

    episode_payload = config.get("episode")
    if isinstance(episode_payload, str):
        with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
            parsed = json.loads(episode_payload)
            if isinstance(parsed, dict):
                episode_payload = parsed
    if not isinstance(episode_payload, dict):
        return {"error": "missing_episode", "published": False}

    try:
        feed_path = _resolve_feed_path(feed_uri, context, config)
    except _PODCAST_RSS_NONCRITICAL_EXCEPTIONS:
        return {"error": "invalid_feed_uri", "published": False}

    feed_path.parent.mkdir(parents=True, exist_ok=True)

    source_feed_url = str(config.get("source_feed_url") or "").strip() or None
    allow_remote_fetch = bool(config.get("allow_remote_fetch", False))
    channel_cfg = config.get("channel") if isinstance(config.get("channel"), dict) else {}
    max_items = int(config.get("max_items", 200))
    max_items = max(1, min(max_items, 5000))
    expected_version = config.get("expected_version")
    if expected_version is not None:
        try:
            expected_version = int(expected_version)
        except _PODCAST_RSS_NONCRITICAL_EXCEPTIONS:
            return {"error": "invalid_expected_version", "published": False}
        if expected_version < 0:
            return {"error": "invalid_expected_version", "published": False}

    root: ET.Element
    if feed_path.exists():
        try:
            root = DET.parse(feed_path).getroot()
        except _PODCAST_RSS_NONCRITICAL_EXCEPTIONS:
            return {"error": "invalid_feed_xml", "published": False}
    elif source_feed_url:
        if not allow_remote_fetch:
            return {"error": "remote_fetch_disabled", "published": False}
        try:
            seeded = _load_seed_xml(source_feed_url)
        except _InvalidRemoteSeedUrl:
            return {"error": "invalid_remote_seed_url", "published": False}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            return {"error": "remote_seed_fetch_failed", "published": False}
        if seeded is None:
            return {"error": "remote_seed_empty", "published": False}
        root = seeded
    else:
        root = ET.Element("rss", attrib={"version": "2.0"})

    channel = _find_child(root, "channel")
    if channel is None:
        channel = ET.SubElement(root, "channel")

    _upsert_text(channel, "title", channel_cfg.get("title") or "tldw Podcast")
    _upsert_text(channel, "link", channel_cfg.get("link") or "https://localhost")
    _upsert_text(
        channel,
        "description",
        channel_cfg.get("description") or "Generated podcast feed",
    )
    _upsert_text(channel, "language", channel_cfg.get("language") or "en-us")
    _upsert_text(channel, "lastBuildDate", format_datetime(datetime.now(timezone.utc)))

    existing_items = _find_children(channel, "item")
    existing_by_guid: dict[str, ET.Element] = {}
    for item in existing_items:
        guid = _find_text(item, "guid") or _find_text(item, "link")
        if not guid:
            continue
        existing_by_guid[guid] = item

    current_version = len(existing_by_guid)
    if expected_version is not None and expected_version != current_version:
        return {
            "error": "version_conflict",
            "published": False,
            "expected_version": expected_version,
            "actual_version": current_version,
        }

    new_guid, new_item = _build_episode_item(episode_payload)
    replaced_existing = new_guid in existing_by_guid
    existing_by_guid[new_guid] = new_item

    ranked_items: list[tuple[str, float, ET.Element]] = []
    for guid, item in existing_by_guid.items():
        timestamp = _to_timestamp(_find_text(item, "pubDate"))
        ranked_items.append((guid, timestamp, item))
    ranked_items.sort(key=lambda row: (-row[1], row[0]))
    ranked_items = ranked_items[:max_items]

    for item in existing_items:
        with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
            channel.remove(item)
    for _, _, item in ranked_items:
        channel.append(item)

    tree = ET.ElementTree(root)
    with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
        ET.indent(tree, space="  ")

    tmp_file: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{feed_path.stem}_",
            suffix=".tmp",
            dir=str(feed_path.parent),
        )
        os.close(fd)
        tmp_file = Path(tmp_name)
        tree.write(tmp_file, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_file, feed_path)
    except _PODCAST_RSS_NONCRITICAL_EXCEPTIONS:
        logger.error("podcast_rss_publish write failed")
        return {"error": "feed_write_failed", "published": False}
    finally:
        if tmp_file is not None and tmp_file.exists():
            with contextlib.suppress(_PODCAST_RSS_NONCRITICAL_EXCEPTIONS):
                tmp_file.unlink()

    return {
        "published": True,
        "feed_uri": f"file://{feed_path}",
        "item_guid": new_guid,
        "item_count": len(ranked_items),
        "version": len(ranked_items),
        "replaced_existing_guid": replaced_existing,
        "source_feed_url": source_feed_url,
    }
