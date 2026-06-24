from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urlparse

from tldw_Server_API.app.core.Collections.utils import (
    is_supported_reading_url,
    normalize_reading_status,
)

try:
    from tldw_Server_API.app.core.Web_Scraping.url_utils import normalize_for_crawl
except Exception:  # pragma: no cover - import fallback for isolated tests
    normalize_for_crawl = None


@dataclass
class ReadingImportItem:
    url: str
    title: str | None
    tags: list[str]
    status: str | None
    favorite: bool
    notes: str | None
    read_at: str | None
    metadata: dict[str, Any]


_READING_STATUS_PRIORITY: dict[str, int] = {
    "saved": 0,
    "reading": 1,
    "archived": 2,
    "read": 3,
}


def detect_import_source(filename: str | None, raw_bytes: bytes) -> str:
    if filename:
        lowered = filename.lower()
        if lowered.endswith(".json"):
            return "pocket"
        if lowered.endswith(".csv"):
            return "instapaper"
    try:
        json.loads(raw_bytes.decode("utf-8"))
        return "pocket"
    except Exception:
        return "instapaper"


def parse_reading_import(raw_bytes: bytes, *, source: str, filename: str | None = None) -> list[ReadingImportItem]:
    if source == "auto":
        source = detect_import_source(filename, raw_bytes)
    if source == "pocket":
        return parse_pocket_export(raw_bytes)
    if source == "instapaper":
        return parse_instapaper_export(raw_bytes)
    raise ValueError("unsupported_import_source")


def parse_pocket_export(raw_bytes: bytes) -> list[ReadingImportItem]:
    text = raw_bytes.decode("utf-8", errors="replace")
    payload = json.loads(text)
    items_obj = payload.get("list")
    if isinstance(items_obj, list):
        items = items_obj
    elif isinstance(items_obj, dict):
        items = list(items_obj.values())
    else:
        items = []

    parsed: list[ReadingImportItem] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        url = entry.get("resolved_url") or entry.get("given_url") or entry.get("url")
        if not url:
            continue
        title = entry.get("resolved_title") or entry.get("given_title") or entry.get("title")
        tags = _normalize_tags(entry.get("tags"))
        status = _map_pocket_status(entry.get("status"))
        if status == "deleted":
            continue
        favorite = _truthy(entry.get("favorite"))
        read_at = _parse_epoch(entry.get("time_read"))
        if read_at:
            status = "read"
        notes = entry.get("excerpt") or entry.get("note") or None
        metadata: dict[str, Any] = {
            "import_source": "pocket",
        }
        added_at = _parse_epoch(entry.get("time_added"))
        if added_at:
            metadata["import_added_at"] = added_at
        parsed.append(
            ReadingImportItem(
                url=str(url),
                title=str(title) if title else None,
                tags=tags,
                status=status or "saved",
                favorite=favorite,
                notes=str(notes) if notes else None,
                read_at=read_at,
                metadata=metadata,
            )
        )
    return parsed


def parse_instapaper_export(raw_bytes: bytes) -> list[ReadingImportItem]:
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    parsed: list[ReadingImportItem] = []
    for row in reader:
        if not row:
            continue
        normalized = {(_normalize_header(k)): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
        url = normalized.get("url") or normalized.get("link")
        if not url:
            continue
        title = normalized.get("title")
        tags = _split_tags(normalized.get("tags"))
        notes = normalized.get("notes") or normalized.get("selection") or None
        folder = (normalized.get("folder") or normalized.get("state") or "").lower()
        status = "archived" if "archive" in folder else "saved"
        favorite = "star" in folder
        metadata: dict[str, Any] = {
            "import_source": "instapaper",
        }
        added_raw = normalized.get("added") or normalized.get("timestamp") or normalized.get("time")
        if added_raw:
            metadata["import_added_at"] = added_raw
        parsed.append(
            ReadingImportItem(
                url=str(url),
                title=str(title) if title else None,
                tags=tags,
                status=status,
                favorite=favorite,
                notes=str(notes) if notes else None,
                read_at=None,
                metadata=metadata,
            )
        )
    return parsed


def _normalize_header(value: str) -> str:
    return value.strip().lower()


def _split_tags(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _normalize_tags(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        tags = []
        for entry in value:
            if isinstance(entry, dict):
                tag = entry.get("tag") or entry.get("name")
                if tag:
                    tags.append(str(tag))
            elif entry:
                tags.append(str(entry))
        return sorted({t.strip().lower() for t in tags if t})
    if isinstance(value, dict):
        return sorted({str(k).strip().lower() for k in value if k})
    if isinstance(value, str):
        return _split_tags(value)
    return []


def _map_pocket_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text == "0":
        return "saved"
    if text == "1":
        return "archived"
    if text == "2":
        return "deleted"
    return "saved"


def _parse_epoch(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except Exception:
        return None
    if ivalue <= 0:
        return None
    return datetime.fromtimestamp(ivalue, tz=timezone.utc).replace(microsecond=0).isoformat()


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def _normalize_status(value: str | None) -> str:
    return normalize_reading_status(value)


def _merge_status(left: str | None, right: str | None) -> str:
    left_norm = _normalize_status(left)
    right_norm = _normalize_status(right)
    if _READING_STATUS_PRIORITY[right_norm] >= _READING_STATUS_PRIORITY[left_norm]:
        return right_norm
    return left_norm


def _normalize_import_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if normalize_for_crawl is None:
        return raw if is_supported_reading_url(raw) else ""
    try:
        normalized = normalize_for_crawl(raw, raw)
    except Exception:
        normalized = raw
    candidate = normalized or raw
    return candidate if is_supported_reading_url(candidate) else ""


def _title_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        slug = path.split("/")[-1]
        if slug:
            text = unquote(slug).replace("-", " ").replace("_", " ").strip()
            if text:
                return text[:200]
    if parsed.netloc:
        return parsed.netloc
    return None


def _merge_title(existing_title: str | None, new_title: str | None, normalized_url: str) -> str | None:
    if existing_title and not new_title:
        return existing_title
    if new_title and not existing_title:
        return new_title
    if not existing_title and not new_title:
        return None

    fallback = _title_from_url(normalized_url)
    if fallback and str(existing_title).strip().lower() == fallback.strip().lower():
        if str(new_title).strip().lower() != fallback.strip().lower():
            return new_title
    return existing_title


def normalize_import_items(items: Iterable[ReadingImportItem]) -> list[ReadingImportItem]:
    merged: dict[str, ReadingImportItem] = {}
    for item in items:
        normalized_url = _normalize_import_url(item.url)
        if not normalized_url:
            continue

        normalized_tags = sorted({str(tag).strip().lower() for tag in (item.tags or []) if str(tag).strip()})
        normalized_status = _normalize_status(item.status)
        normalized_title = str(item.title).strip() if item.title else None
        if not normalized_title:
            normalized_title = _title_from_url(normalized_url)

        normalized_item = ReadingImportItem(
            url=normalized_url,
            title=normalized_title,
            tags=normalized_tags,
            status=normalized_status,
            favorite=bool(item.favorite),
            notes=str(item.notes).strip() if item.notes else None,
            read_at=item.read_at,
            metadata={
                **(item.metadata or {}),
                "import_normalized_url": normalized_url,
            },
        )

        key = normalized_url.lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = normalized_item
            continue

        merged[key] = ReadingImportItem(
            url=normalized_url,
            title=_merge_title(existing.title, normalized_item.title, normalized_url),
            tags=sorted({*existing.tags, *normalized_item.tags}),
            status=_merge_status(existing.status, normalized_item.status),
            favorite=existing.favorite or normalized_item.favorite,
            notes=existing.notes or normalized_item.notes,
            read_at=existing.read_at or normalized_item.read_at,
            metadata={**existing.metadata, **normalized_item.metadata},
        )

    return list(merged.values())
