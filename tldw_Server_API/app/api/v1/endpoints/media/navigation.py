"""Media navigation tree endpoint for chapter/section browsing."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from collections import defaultdict
from typing import Any, Protocol

from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.media.document_outline import (
    MAX_OUTLINE_FILE_SIZE,
    _check_pymupdf_available,
    _extract_pdf_outline,
)
from tldw_Server_API.app.api.v1.endpoints.media_navigation_policy import (
    MEDIA_NAVIGATION_ROUTE_POLICY,
)
from tldw_Server_API.app.api.v1.schemas.media_navigation_schemas import (
    MediaNavigationContentQueryParams,
    MediaNavigationContentResponse,
    MediaNavigationNode,
    MediaNavigationQueryParams,
    MediaNavigationResponse,
    MediaNavigationStats,
    MediaNavigationTarget,
)
from tldw_Server_API.app.api.v1.utils.cache import cache_response, get_cached_response
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_document_version,
    get_latest_transcription,
    get_media_transcripts,
    lookup_section_by_heading,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.Storage import get_storage_backend
from tldw_Server_API.app.core.Storage.storage_interface import StorageError

router = APIRouter(tags=["Document Workspace"])

NAVIGATION_SOURCE_PRIORITY = (
    "pdf_outline",
    "generated_toc",
    "document_structure_index",
    "transcript_segment",
    "chunk_metadata",
)
NAVIGATION_CACHE_VERSION = 1
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MD_HINT_RE = re.compile(
    r"(^#{1,6}\s)|(^[-*+]\s)|(^\d+\.\s)|(```)|(\[[^\]]+\]\([^)]+\))",
    flags=re.MULTILINE,
)
_WS_RE = re.compile(r"\s+")
_MD_HEADING_LINE_RE = re.compile(
    r"(?m)^(?P<prefix>\s{0,3})(?P<marks>#{1,6})\s+(?P<title>[^\n#].*?)\s*$",
)
_NOISY_TITLE_REPEAT_SYMBOL_RE = re.compile(r"([^\w\s])\1{3,}")
_URL_PREFIX_RE = re.compile(r"^(?:https?://|www\.)", flags=re.IGNORECASE)
_TITLE_MULTI_ENTRY_SPLIT_RE = re.compile(
    r"\s+\d{1,4}\s+(?=(?:[A-Z]\.|[IVXLCDM]{1,8}\.|[0-9]{1,3}\.|Chapter\s+\d+|Letter\s+\w+)\s+)",
    flags=re.IGNORECASE,
)
_HEADING_STYLE_TITLE_RE = re.compile(
    r"^(?:[IVXLCDM]{1,8}\.|[A-Z]\.|[0-9]{1,3}\.|chapter\s+\d+|letter\s+\w+|introduction\b|conclusions?\b|appendix\b|acknowledg)",
    flags=re.IGNORECASE,
)
_TOC_MARKER_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?\s*(?:table\s+of\s+contents|contents)\s*(?:\*\*)?\s*$",
)
_TOC_ENTRY_RE = re.compile(r"^(?P<title>.+?)\s+(?P<page>\d{1,4})$")
_TOC_PRIMARY_LEVEL_RE = re.compile(
    r"^(?:[IVXLCDM]{1,8}\.|chapter\s+\d+|letter\s+\w+|introduction\b|conclusions?\b|appendix\b|acknowledg)",
    flags=re.IGNORECASE,
)
_TOC_SECONDARY_LEVEL_RE = re.compile(r"^[A-Z]\.\s+")
_TOC_TERTIARY_LEVEL_RE = re.compile(r"^\d+\.\s+")
_TOC_UNNUMBERED_TOP_LEVEL_RE = re.compile(
    r"^(?:acknowledg(?:ment|ement)s?|references?)\b",
    flags=re.IGNORECASE,
)
_ITALIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "we",
}


class MediaNavigationDb(Protocol):
    """Structural contract for DB sessions used by navigation helpers."""

    backend_type: Any

    def execute_query(self, query: str, params: Any = None) -> Any:
        """Execute a SQL query against the backing content store."""

    def get_media_by_id(
        self,
        media_id: int,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a media record by id."""

    def get_media_file(self, media_id: int, file_type: str) -> dict[str, Any] | None:
        """Fetch a stored media file record."""


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_postgres_backend(db: MediaNavigationDb) -> bool:
    backend_type = str(getattr(db, "backend_type", "")).lower()
    return "postgres" in backend_type


def _normalize_section_path(path_value: Any) -> list[str]:
    if path_value is None:
        return []
    if isinstance(path_value, str):
        raw = path_value.strip()
        if not raw:
            return []
        if " > " in raw:
            parts = [p.strip() for p in raw.split(" > ")]
        elif ">" in raw:
            parts = [p.strip() for p in raw.split(">")]
        elif "/" in raw:
            parts = [p.strip() for p in raw.split("/")]
        else:
            parts = [raw]
        return [p for p in parts if p]
    if isinstance(path_value, (list, tuple)):
        return [str(p).strip() for p in path_value if str(p).strip()]
    text = str(path_value).strip()
    return [text] if text else []


def _clean_navigation_title(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = html_lib.unescape(text)
    text = text.replace("\u00a0", " ")
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("`", "").replace("~", "")
    text = re.sub(r"(?<!\w)\*\*(.+?)\*\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)

    def _strip_italic_wrap(match: re.Match[str]) -> str:
        inner = str(match.group(1) or "")
        lowered = inner.lower()
        if re.fullmatch(r"[a-z]{2,3}", lowered) and lowered not in _ITALIC_STOPWORDS:
            return f"{lowered[0]}_{lowered[1:]}"
        return inner

    text = re.sub(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)", _strip_italic_wrap, text)
    text = re.sub(r"(?<=\w)_\s+(?=\w)", "_", text)
    text = re.sub(r"(?<=\s)_(?=\w)", "", text)
    text = re.sub(r"(?<=\w)_(?=\s)", "", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+-", "-", text)
    text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    if ">" in text:
        parts = [part.strip() for part in re.split(r"\s*>\s*", text) if part.strip()]
        if parts:
            text = parts[-1]
    split_match = _TITLE_MULTI_ENTRY_SPLIT_RE.search(text)
    if split_match:
        truncated = text[: split_match.start()].strip()
        if truncated:
            text = truncated
    cleaned = text.strip("•*|-_~:;,. ")
    return cleaned.strip()


def _is_noisy_navigation_title(title: str) -> bool:
    cleaned = _clean_navigation_title(title)
    if not cleaned:
        return True

    compact = "".join(ch for ch in cleaned if not ch.isspace())
    if len(compact) < 2 or len(compact) > 180:
        return True
    if _URL_PREFIX_RE.match(compact):
        return True
    if _NOISY_TITLE_REPEAT_SYMBOL_RE.search(compact):
        return True
    if cleaned.count("|") >= 2:
        return True

    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    symbols = sum(not ch.isalnum() for ch in compact)

    if letters == 0:
        return True
    if symbols / max(1, len(compact)) > 0.42:
        return True
    if letters > 0 and (letters / max(1, len(compact))) < 0.12 and digits > letters:
        return True
    words = [part for part in cleaned.split(" ") if part]
    return bool(len(words) > 16 and not _HEADING_STYLE_TITLE_RE.match(cleaned))


def _infer_generated_toc_level(title: str) -> int:
    normalized = _clean_navigation_title(title)
    if not normalized:
        return 1
    if _TOC_TERTIARY_LEVEL_RE.match(normalized):
        return 3
    if _TOC_SECONDARY_LEVEL_RE.match(normalized):
        return 2
    if _TOC_PRIMARY_LEVEL_RE.match(normalized):
        return 1
    return 1


def _is_plausible_generated_toc_title(title: str) -> bool:
    cleaned = _clean_navigation_title(title)
    if not cleaned:
        return False
    if _is_noisy_navigation_title(cleaned):
        return False

    alpha_chars = sum(ch.isalpha() for ch in cleaned)
    if alpha_chars < 3 and not _HEADING_STYLE_TITLE_RE.match(cleaned):
        return False

    words = [w for w in cleaned.split(" ") if w]
    if (
        len(words) <= 2
        and all(len(re.sub(r"[^A-Za-z]", "", w)) <= 1 for w in words)
        and not _TOC_SECONDARY_LEVEL_RE.match(cleaned)
    ):
        return False

    return not re.search(r"[=<>±∑∫]", cleaned)


def _sanitize_navigation_path_parts(path_parts: list[str]) -> list[str]:
    sanitized: list[str] = []
    for part in path_parts:
        cleaned = _clean_navigation_title(part)
        if not cleaned or _is_noisy_navigation_title(cleaned):
            continue
        if sanitized and sanitized[-1].lower() == cleaned.lower():
            continue
        sanitized.append(cleaned)
    return sanitized


def _is_sparse_pdf_outline(nodes: list[dict[str, Any]]) -> bool:
    if not nodes:
        return True
    node_count = len(nodes)
    max_level = max((_to_int(node.get("level")) or 1 for node in nodes), default=1)
    root_count = sum(1 for node in nodes if node.get("parent_id") is None)
    unique_pages = len(
        {_to_int(node.get("target_start")) for node in nodes if _to_int(node.get("target_start")) is not None}
    )
    if node_count <= 2 and max_level <= 1:
        return True
    return bool(node_count <= 3 and root_count == node_count and unique_pages <= 2)


def _parse_chunk_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    raw = value.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _node_sort_key(node: dict[str, Any]) -> tuple[int, str, str]:
    order = _to_int(node.get("order"))
    safe_order = order if order is not None and order >= 0 else 2_147_483_647
    title = str(node.get("title") or "").strip().lower()
    node_id = str(node.get("id") or "")
    return safe_order, title, node_id


def _preorder_with_path_labels(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not nodes:
        return []

    by_id = {str(node["id"]): node for node in nodes if node.get("id")}
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        parent_id = node.get("parent_id")
        parent_key = str(parent_id) if parent_id is not None and str(parent_id) in by_id else None
        children[parent_key].append(node)

    for key in list(children.keys()):
        children[key] = sorted(children[key], key=_node_sort_key)

    ordered: list[dict[str, Any]] = []
    visited: set[str] = set()

    def _visit(parent_id: str | None, prefix: str) -> None:
        siblings = children.get(parent_id, [])
        for idx, node in enumerate(siblings, start=1):
            node_id = str(node.get("id") or "")
            if not node_id or node_id in visited:
                continue
            visited.add(node_id)
            path_label = f"{prefix}.{idx}" if prefix else str(idx)
            node["path_label"] = path_label
            node["level"] = len(path_label.split("."))
            ordered.append(node)
            _visit(node_id, path_label)

    _visit(None, "")

    if len(visited) != len(by_id):
        leftovers = sorted(
            [node for node in nodes if str(node.get("id") or "") not in visited],
            key=_node_sort_key,
        )
        for node in leftovers:
            node_id = str(node.get("id") or "")
            if not node_id:
                continue
            next_index = len(ordered) + 1
            node["path_label"] = str(next_index)
            node["level"] = 1
            ordered.append(node)
            visited.add(node_id)

    return ordered


def _materialize_navigation_nodes(raw_nodes: list[dict[str, Any]]) -> list[MediaNavigationNode]:
    nodes: list[MediaNavigationNode] = []
    for raw in raw_nodes:
        try:
            nodes.append(MediaNavigationNode.model_validate(raw))
        except (TypeError, ValueError):
            logger.debug("Skipping invalid navigation node payload")
    return nodes


def _build_navigation_cache_key(
    media_id: int,
    media: dict[str, Any],
    params: MediaNavigationQueryParams,
) -> str:
    params_payload = {
        "include_generated_fallback": bool(params.include_generated_fallback),
        "max_depth": int(params.max_depth),
        "max_nodes": int(params.max_nodes),
        "parent_id": params.parent_id,
    }
    params_sig = hashlib.md5(
        json.dumps(params_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    media_version = _to_int(media.get("version")) or 0
    media_last_modified = str(media.get("last_modified") or "")
    return (
        f"cache:/api/v1/media/{media_id}/navigation:"
        f"cv:{NAVIGATION_CACHE_VERSION}:"
        f"mv:{media_version}:"
        f"lm:{media_last_modified}:"
        f"q:{params_sig}"
    )


def _serialize_nodes_for_version(nodes: list[MediaNavigationNode]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for node in nodes:
        serialized.append(
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "level": node.level,
                "title": node.title,
                "order": node.order,
                "path_label": node.path_label,
                "target_type": node.target_type,
                "target_start": node.target_start,
                "target_end": node.target_end,
                "target_href": node.target_href,
                "source": node.source,
            }
        )
    return serialized


def _compute_navigation_version(
    media_id: int,
    media: dict[str, Any],
    source_order_used: list[str],
    nodes: list[MediaNavigationNode],
) -> str:
    media_version = _to_int(media.get("version")) or 0
    payload = {
        "media_id": media_id,
        "media_version": media_version,
        "last_modified": str(media.get("last_modified") or ""),
        "source_order_used": source_order_used,
        "nodes": _serialize_nodes_for_version(nodes),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:10]
    return f"media_{media_id}:v{media_version}:{digest}"


def _filter_navigation_nodes(
    nodes: list[MediaNavigationNode],
    params: MediaNavigationQueryParams,
) -> tuple[list[MediaNavigationNode], int, bool, int]:
    filtered = [node for node in nodes if node.level <= params.max_depth]

    if params.parent_id:
        filtered = [node for node in filtered if node.parent_id == params.parent_id]
        filtered.sort(key=lambda node: (node.order, node.title.lower(), node.id))

    max_depth_seen = max((node.level for node in filtered), default=0)
    node_count = len(filtered)
    truncated = node_count > params.max_nodes
    if truncated:
        filtered = filtered[: params.max_nodes]
    return filtered, node_count, truncated, max_depth_seen


async def _extract_pdf_outline_nodes(
    media_id: int,
    db: MediaNavigationDb,
    media: dict[str, Any],
) -> list[dict[str, Any]]:
    media_type = str(media.get("type") or "").strip().lower()
    if media_type not in {"pdf", "document"}:
        return []
    if not _check_pymupdf_available():
        logger.debug("Navigation source pdf_outline skipped: PyMuPDF unavailable")
        return []

    try:
        file_record = db.get_media_file(media_id, "original")
    except Exception:
        logger.warning("Navigation source pdf_outline failed to fetch file record")
        return []

    if not file_record:
        return []

    storage_path = file_record.get("storage_path")
    mime_type = str(file_record.get("mime_type") or "").lower()
    is_pdf_mime = mime_type == "application/pdf"
    is_pdf_extension = str(storage_path or "").lower().endswith(".pdf")
    if not storage_path or (not is_pdf_mime and not is_pdf_extension):
        return []

    storage = get_storage_backend()
    try:
        if not await storage.exists(storage_path):
            return []
        try:
            file_size = await storage.get_size(storage_path)
            if file_size > MAX_OUTLINE_FILE_SIZE:
                logger.debug("Navigation source pdf_outline skipped due to file size")
                return []
        except FileNotFoundError:
            return []
        pdf_file = await storage.retrieve(storage_path)
    except StorageError:
        logger.warning("Navigation source pdf_outline storage access failed")
        return []

    try:
        entries, _total_pages = _extract_pdf_outline(pdf_file)
    except Exception:
        logger.warning("Navigation source pdf_outline extraction failed")
        return []

    if not entries:
        return []

    nodes: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    sibling_counts: dict[str | None, int] = defaultdict(int)

    for idx, entry in enumerate(entries):
        title = _clean_navigation_title(getattr(entry, "title", ""))
        level = _to_int(getattr(entry, "level", None)) or 1
        level = max(1, min(8, level))
        page = _to_int(getattr(entry, "page", None))
        if not title or _is_noisy_navigation_title(title) or page is None or page < 1:
            continue

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = stack[-1][1] if stack else None
        order = sibling_counts[parent_id]
        sibling_counts[parent_id] += 1
        node_id = f"pdf_outline:{idx}"
        nodes.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "level": level,
                "title": title,
                "order": order,
                "path_label": None,
                "target_type": "page",
                "target_start": page,
                "target_end": None,
                "target_href": None,
                "source": "pdf_outline",
                "confidence": 1.0,
            }
        )
        stack.append((level, node_id))

    return _preorder_with_path_labels(nodes)


def _extract_document_structure_nodes(
    media_id: int,
    db: MediaNavigationDb,
) -> list[dict[str, Any]]:
    bool_false = False if _is_postgres_backend(db) else 0
    query = """
        SELECT id, parent_id, level, title, start_char, end_char, order_index, path
        FROM DocumentStructureIndex
        WHERE media_id = ? AND deleted = ? AND kind IN ('section', 'header')
        ORDER BY COALESCE(level, 0) ASC, COALESCE(order_index, 2147483647) ASC, start_char ASC, id ASC
    """
    try:
        rows = db.execute_query(query, (media_id, bool_false)).fetchall() or []
    except Exception:
        logger.warning("Navigation source document_structure_index query failed")
        return []

    if not rows:
        return []

    prepared_rows: list[dict[str, Any]] = []
    row_ids: set[int] = set()
    for row in rows:
        row_dict = dict(row) if not isinstance(row, dict) else row
        row_id = _to_int(row_dict.get("id"))
        if row_id is None:
            continue
        row_ids.add(row_id)
        prepared_rows.append(row_dict)

    if not prepared_rows:
        return []

    nodes: list[dict[str, Any]] = []
    sibling_counts: dict[str | None, int] = defaultdict(int)

    for idx, row in enumerate(prepared_rows):
        row_id = _to_int(row.get("id"))
        start_char = _to_int(row.get("start_char"))
        end_char = _to_int(row.get("end_char"))
        if row_id is None or start_char is None or end_char is None or end_char <= start_char:
            continue

        path_parts = _sanitize_navigation_path_parts(_normalize_section_path(row.get("path")))
        title = str(row.get("title") or "").strip()
        if not title:
            title = path_parts[-1] if path_parts else f"Section {idx + 1}"
        title = _clean_navigation_title(title)
        if (not title or _is_noisy_navigation_title(title)) and path_parts:
            title = _clean_navigation_title(path_parts[-1])
        if not title or _is_noisy_navigation_title(title):
            continue

        parent_raw = _to_int(row.get("parent_id"))
        parent_id = f"dsi:{parent_raw}" if parent_raw is not None and parent_raw in row_ids else None

        order_raw = _to_int(row.get("order_index"))
        if order_raw is None or order_raw < 0:
            order = sibling_counts[parent_id]
        else:
            order = order_raw
        sibling_counts[parent_id] = max(sibling_counts[parent_id], order + 1)

        level = _to_int(row.get("level")) or 1
        level = max(1, min(8, level))
        nodes.append(
            {
                "id": f"dsi:{row_id}",
                "parent_id": parent_id,
                "level": level,
                "title": title,
                "order": order,
                "path_label": None,
                "target_type": "char_range",
                "target_start": start_char,
                "target_end": end_char,
                "target_href": None,
                "source": "document_structure_index",
                "confidence": 0.95,
            }
        )

    return _preorder_with_path_labels(nodes)


def _parse_colon_time_to_seconds(value: str) -> float | None:
    parts = [p.strip() for p in value.split(":")]
    if len(parts) not in {2, 3}:
        return None
    try:
        if len(parts) == 2:
            minutes = int(parts[0] or "0")
            seconds = float(parts[1] or "0")
            return max(0.0, (minutes * 60) + seconds)
        hours = int(parts[0] or "0")
        minutes = int(parts[1] or "0")
        seconds = float(parts[2] or "0")
        return max(0.0, (hours * 3600) + (minutes * 60) + seconds)
    except (TypeError, ValueError):
        return None


def _coerce_segment_seconds(value: Any, *, is_millis_hint: bool = False) -> float | None:
    if value is None:
        return None

    parsed: float | None = None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if ":" in raw:
            parsed = _parse_colon_time_to_seconds(raw)
        if parsed is None:
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                return None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None

    if parsed is None:
        return None
    if is_millis_hint:
        parsed = parsed / 1000.0
    if parsed < 0:
        return 0.0
    return parsed


def _normalize_transcript_segments(segments: Any) -> list[dict[str, Any]]:
    if not isinstance(segments, list):
        return []

    normalized: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue

        text_value = (
            seg.get("text")
            or seg.get("Text")
            or seg.get("transcription")
            or seg.get("transcript")
            or seg.get("composite")
            or ""
        )
        text = _WS_RE.sub(" ", str(text_value or "")).strip()

        start_any = seg.get("start")
        end_any = seg.get("end")
        if start_any is None:
            start_any = seg.get("start_seconds")
        if end_any is None:
            end_any = seg.get("end_seconds")
        if start_any is None:
            start_any = seg.get("start_time")
        if end_any is None:
            end_any = seg.get("end_time")
        if start_any is None:
            start_any = seg.get("Time_Start")
        if end_any is None:
            end_any = seg.get("Time_End")
        if start_any is None:
            start_any = seg.get("start_ms")
        if end_any is None:
            end_any = seg.get("end_ms")

        is_millis_hint = ("start_ms" in seg) or ("end_ms" in seg)
        start_seconds = _coerce_segment_seconds(start_any, is_millis_hint=is_millis_hint)
        end_seconds = _coerce_segment_seconds(end_any, is_millis_hint=is_millis_hint)
        if start_seconds is None:
            continue
        if end_seconds is not None and end_seconds <= start_seconds:
            end_seconds = None

        speaker_raw = seg.get("speaker")
        speaker = str(speaker_raw).strip() if speaker_raw is not None else None
        confidence = _to_float(seg.get("confidence"))
        normalized.append(
            {
                "index": idx,
                "text": text,
                "start": start_seconds,
                "end": end_seconds,
                "speaker": speaker,
                "confidence": confidence,
            }
        )

    return normalized


def _coerce_transcript_payload(raw_payload: Any) -> tuple[str, list[dict[str, Any]]]:
    if raw_payload is None:
        return "", []

    if isinstance(raw_payload, str):
        stripped = raw_payload.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
                return _coerce_transcript_payload(parsed)
            except (TypeError, ValueError):
                return raw_payload, []
        return raw_payload, []

    if isinstance(raw_payload, dict):
        text_val = raw_payload.get("text") or raw_payload.get("transcription") or raw_payload.get("transcript") or ""
        segments_any = raw_payload.get("segments") or raw_payload.get("Segments") or raw_payload.get("entries") or []
        segments = _normalize_transcript_segments(segments_any)
        text = str(text_val or "")
        if not text and segments:
            text = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
        return text, segments

    if isinstance(raw_payload, list):
        segments = _normalize_transcript_segments(raw_payload)
        text = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
        return text, segments

    return str(raw_payload), []


def _build_transcript_node_title(segment: dict[str, Any], ordinal: int) -> str:
    text = _WS_RE.sub(" ", str(segment.get("text") or "")).strip()
    if text:
        snippet = text[:84].rstrip()
        if len(text) > 84:
            snippet = f"{snippet}..."
    else:
        snippet = f"Segment {ordinal}"

    speaker = str(segment.get("speaker") or "").strip()
    if speaker:
        return f"{speaker}: {snippet}"
    return snippet


def _extract_transcript_segment_nodes(
    media_id: int,
    db: MediaNavigationDb,
    media: dict[str, Any],
) -> list[dict[str, Any]]:
    media_type = str(media.get("type") or "").strip().lower()
    if media_type not in {"audio", "video"}:
        return []

    try:
        transcripts = get_media_transcripts(db, media_id)
    except Exception:
        logger.warning("Navigation source transcript_segment query failed")
        return []
    if not transcripts:
        return []

    latest = transcripts[0] if isinstance(transcripts[0], dict) else dict(transcripts[0])
    _text, segments = _coerce_transcript_payload(latest.get("transcription"))
    if not segments:
        return []

    nodes: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        start_seconds = _to_float(segment.get("start"))
        if start_seconds is None:
            continue
        end_seconds = _to_float(segment.get("end"))
        if end_seconds is not None and end_seconds <= start_seconds:
            end_seconds = None

        confidence = _to_float(segment.get("confidence"))
        if confidence is None:
            confidence = 0.8

        nodes.append(
            {
                "id": f"transcript_segment:{idx}",
                "parent_id": None,
                "level": 1,
                "title": _build_transcript_node_title(segment, idx + 1),
                "order": idx,
                "path_label": None,
                "target_type": "time_range",
                "target_start": start_seconds,
                "target_end": end_seconds,
                "target_href": None,
                "source": "transcript_segment",
                "confidence": max(0.0, min(1.0, float(confidence))),
            }
        )

    return _preorder_with_path_labels(nodes)


def _chunk_path_to_node_id(path_parts: tuple[str, ...]) -> str:
    joined = " > ".join(path_parts)
    digest = hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"chunk_meta:{digest}"


def _extract_chunk_metadata_nodes(
    media_id: int,
    db: MediaNavigationDb,
) -> list[dict[str, Any]]:
    bool_false = False if _is_postgres_backend(db) else 0
    query = """
        SELECT chunk_index, start_char, end_char, metadata
        FROM UnvectorizedMediaChunks
        WHERE media_id = ? AND deleted = ? AND metadata IS NOT NULL
        ORDER BY chunk_index ASC, id ASC
    """
    try:
        rows = db.execute_query(query, (media_id, bool_false)).fetchall() or []
    except Exception:
        logger.warning("Navigation source chunk_metadata query failed")
        return []

    if not rows:
        return []

    sections: dict[tuple[str, ...], dict[str, Any]] = {}
    order_counter = 0

    for row in rows:
        row_dict = dict(row) if not isinstance(row, dict) else row
        start_char = _to_int(row_dict.get("start_char"))
        end_char = _to_int(row_dict.get("end_char"))
        if start_char is None or end_char is None or end_char <= start_char:
            continue

        metadata = _parse_chunk_metadata(row_dict.get("metadata"))
        path_parts = _sanitize_navigation_path_parts(
            _normalize_section_path(
                metadata.get("section_path") or metadata.get("ancestry_titles"),
            )
        )
        if not path_parts:
            continue

        for depth in range(1, len(path_parts) + 1):
            key = tuple(path_parts[:depth])
            parent_key = tuple(path_parts[: depth - 1]) if depth > 1 else None
            rec = sections.get(key)
            if rec is None:
                sections[key] = {
                    "key": key,
                    "parent_key": parent_key,
                    "title": str(path_parts[depth - 1]),
                    "order": order_counter,
                    "start_char": start_char,
                    "end_char": end_char,
                }
                order_counter += 1
            else:
                rec["start_char"] = min(int(rec["start_char"]), start_char)
                rec["end_char"] = max(int(rec["end_char"]), end_char)

    if not sections:
        return []

    nodes: list[dict[str, Any]] = []
    for rec in sorted(sections.values(), key=lambda item: (len(item["key"]), item["order"])):
        key = tuple(rec["key"])
        parent_key = rec.get("parent_key")
        parent_id = _chunk_path_to_node_id(tuple(parent_key)) if parent_key else None
        level = len(key)
        nodes.append(
            {
                "id": _chunk_path_to_node_id(key),
                "parent_id": parent_id,
                "level": level,
                "title": str(rec["title"]),
                "order": int(rec["order"]),
                "path_label": None,
                "target_type": "char_range",
                "target_start": int(rec["start_char"]),
                "target_end": int(rec["end_char"]),
                "target_href": None,
                "source": "chunk_metadata",
                "confidence": 0.7,
            }
        )

    return _preorder_with_path_labels(nodes)


def _extract_generated_heading_nodes(content: str) -> list[dict[str, Any]]:
    matches = list(_MD_HEADING_LINE_RE.finditer(content))
    if not matches:
        return []

    base_heading_level = min(len(m.group("marks")) for m in matches)
    nodes: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    sibling_counts: dict[str | None, int] = defaultdict(int)

    for idx, match in enumerate(matches):
        title = _clean_navigation_title(match.group("title"))
        if not title or _is_noisy_navigation_title(title):
            continue
        heading_level = len(match.group("marks"))
        level = max(1, min(8, (heading_level - base_heading_level) + 1))

        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        if end <= start:
            continue

        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = stack[-1][1] if stack else None
        order = sibling_counts[parent_id]
        sibling_counts[parent_id] += 1
        node_id = f"generated:heading:{idx}"
        nodes.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "level": level,
                "title": title,
                "order": order,
                "path_label": None,
                "target_type": "char_range",
                "target_start": start,
                "target_end": end,
                "target_href": None,
                "source": "generated",
                "confidence": 0.45,
            }
        )
        stack.append((level, node_id))

    return _preorder_with_path_labels(nodes)


def _extract_generated_toc_nodes(
    media_id: int,
    db: MediaNavigationDb,
    media: dict[str, Any],
) -> list[dict[str, Any]]:
    media_type = str(media.get("type") or "").strip().lower()
    if media_type in {"audio", "video"}:
        return []

    content = _get_media_text(media_id=media_id, media=media, db=db)
    if not content.strip():
        return []

    marker = _TOC_MARKER_RE.search(content[: min(len(content), 50_000)])
    if not marker:
        return []

    toc_window_end = min(len(content), marker.end() + 35_000)
    toc_segment = content[marker.end() : toc_window_end]

    raw_entries: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    last_page: int | None = None
    non_monotonic_count = 0

    for raw_line in toc_segment.splitlines():
        line = _clean_navigation_title(raw_line)
        if not line:
            continue

        line_match = _TOC_ENTRY_RE.match(line)
        if not line_match:
            if len(raw_entries) >= 6 and _HEADING_STYLE_TITLE_RE.match(line):
                # We likely crossed from the TOC block into body headings.
                break
            continue

        title = _clean_navigation_title(line_match.group("title"))
        page = _to_int(line_match.group("page"))
        if not title or page is None or page < 1:
            continue
        if title.lower() in {"contents", "table of contents"}:
            continue
        if not _is_plausible_generated_toc_title(title):
            continue

        dedupe_key = (title.lower(), page)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if last_page is not None and page < last_page:
            non_monotonic_count += 1
        last_page = page
        raw_entries.append((title, page))
        if len(raw_entries) >= 800:
            break

    if len(raw_entries) < 3:
        return []

    if non_monotonic_count > max(2, len(raw_entries) // 3):
        logger.debug(
            "Generated TOC parse rejected for media_id={} due to page-order quality (entries={}, non_monotonic={})",
            media_id,
            len(raw_entries),
            non_monotonic_count,
        )
        return []

    entries: list[tuple[str, int, int]] = []
    for title, page in raw_entries:
        level = _infer_generated_toc_level(title)
        if level == 2 and _TOC_SECONDARY_LEVEL_RE.match(title):
            prior_top_title = next(
                (entry_title for entry_title, _entry_page, entry_level in reversed(entries) if entry_level == 1),
                None,
            )
            if prior_top_title and _TOC_UNNUMBERED_TOP_LEVEL_RE.match(prior_top_title):
                # Appendix-style "A. ..." entries often follow unnumbered top-level items.
                level = 1
        entries.append((title, page, level))

    nodes: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    sibling_counts: dict[str | None, int] = defaultdict(int)

    for idx, (title, page, raw_level) in enumerate(entries):
        level = max(1, min(8, raw_level))
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = stack[-1][1] if stack else None
        order = sibling_counts[parent_id]
        sibling_counts[parent_id] += 1
        node_id = f"generated_toc:{idx}"
        nodes.append(
            {
                "id": node_id,
                "parent_id": parent_id,
                "level": level,
                "title": title,
                "order": order,
                "path_label": None,
                "target_type": "page",
                "target_start": page,
                "target_end": None,
                "target_href": None,
                "source": "generated_toc",
                "confidence": 0.85,
            }
        )
        stack.append((level, node_id))

    return _preorder_with_path_labels(nodes)


def _extract_generated_chunk_nodes(content: str) -> list[dict[str, Any]]:
    if not content.strip():
        return []

    nodes: list[dict[str, Any]] = []
    content_len = len(content)
    cursor = 0
    order = 0

    while cursor < content_len:
        while cursor < content_len and content[cursor].isspace():
            cursor += 1
        if cursor >= content_len:
            break

        start = cursor
        upper_bound = min(content_len, start + 2200)
        split_at = content.rfind("\n\n", start + 350, upper_bound)
        if split_at == -1:
            split_at = content.rfind(". ", start + 350, upper_bound)
            if split_at != -1:
                split_at += 1
        if split_at == -1 or split_at <= start:
            split_at = upper_bound

        end = max(start + 1, split_at)
        excerpt = _WS_RE.sub(" ", content[start:end]).strip()
        if not excerpt:
            cursor = end
            continue

        title = _clean_navigation_title(excerpt[:112].rstrip())
        if len(title) > 96:
            title = f"{title[:96].rstrip()}..."
        if not title or _is_noisy_navigation_title(title):
            title = f"Generated Section {order + 1}"

        nodes.append(
            {
                "id": f"generated:chunk:{order}",
                "parent_id": None,
                "level": 1,
                "title": title or f"Generated Section {order + 1}",
                "order": order,
                "path_label": None,
                "target_type": "char_range",
                "target_start": start,
                "target_end": end,
                "target_href": None,
                "source": "generated",
                "confidence": 0.3,
            }
        )
        order += 1
        cursor = end

    return _preorder_with_path_labels(nodes)


def _extract_generated_fallback_nodes(
    media_id: int,
    db: MediaNavigationDb,
    media: dict[str, Any],
) -> list[dict[str, Any]]:
    content = _get_media_text(media_id=media_id, media=media, db=db)
    if not content.strip():
        return []

    heading_nodes = _extract_generated_heading_nodes(content)
    if heading_nodes:
        return heading_nodes
    return _extract_generated_chunk_nodes(content)


async def _select_source_nodes(
    media_id: int,
    db: MediaNavigationDb,
    media: dict[str, Any],
    include_generated_fallback: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    source_order_used: list[str] = []
    sparse_pdf_candidate: list[dict[str, Any]] | None = None
    for source in NAVIGATION_SOURCE_PRIORITY:
        source_order_used.append(source)
        if source == "pdf_outline":
            nodes = await _extract_pdf_outline_nodes(media_id, db, media)
            if nodes and _is_sparse_pdf_outline(nodes):
                sparse_pdf_candidate = nodes
                logger.debug("Navigation source pdf_outline produced sparse structure; trying fallback sources")
                continue
        elif source == "generated_toc":
            nodes = _extract_generated_toc_nodes(media_id, db, media)
        elif source == "document_structure_index":
            nodes = _extract_document_structure_nodes(media_id, db)
        elif source == "transcript_segment":
            nodes = _extract_transcript_segment_nodes(media_id, db, media)
        else:
            nodes = _extract_chunk_metadata_nodes(media_id, db)
        if nodes:
            return nodes, source_order_used

    if include_generated_fallback:
        source_order_used.append("generated")
        generated_nodes = _extract_generated_fallback_nodes(media_id, db, media)
        if generated_nodes:
            return generated_nodes, source_order_used
        logger.debug("Generated fallback requested but produced no nodes")
    if sparse_pdf_candidate:
        return sparse_pdf_candidate, source_order_used
    return [], source_order_used


def _find_navigation_node(nodes: list[MediaNavigationNode], node_id: str) -> MediaNavigationNode | None:
    for node in nodes:
        if node.id == node_id:
            return node
    return None


def _get_media_text(media_id: int, media: dict[str, Any], db: MediaNavigationDb) -> str:
    try:
        latest_doc = get_document_version(
            db,
            media_id=media_id,
            version_number=None,
            include_content=True,
        )
    except Exception:
        logger.debug("Failed to fetch latest document version")
        latest_doc = None

    if latest_doc:
        content = latest_doc.get("content")
        if isinstance(content, str) and content:
            return content

    media_type = str(media.get("type") or "").strip().lower()
    if media_type in {"audio", "video"}:
        try:
            transcript = get_latest_transcription(db, media_id)
        except Exception:
            logger.debug("Failed to fetch latest transcript")
            transcript = None
        if isinstance(transcript, str) and transcript:
            return transcript

    media_content = media.get("content")
    return media_content if isinstance(media_content, str) else ""


def _slice_text(content: str, start: int | None, end: int | None) -> str:
    if not content:
        return ""
    text_len = len(content)
    safe_start = 0 if start is None else max(0, min(text_len, int(start)))
    safe_end = text_len if end is None else max(0, min(text_len, int(end)))
    if safe_end <= safe_start:
        return ""
    return content[safe_start:safe_end]


def _derive_content_span(
    node: MediaNavigationNode,
    all_nodes: list[MediaNavigationNode],
    media: dict[str, Any],
    db: MediaNavigationDb,
    media_id: int,
    content_length: int,
) -> tuple[int, int] | None:
    if content_length <= 0:
        return None

    if node.target_type == "char_range":
        start = _to_int(node.target_start)
        end = _to_int(node.target_end)
        if start is not None and end is not None and 0 <= start < end:
            return start, end
        return None

    if node.target_type == "page":
        current_page = _to_int(node.target_start)
        if current_page is None or current_page < 1:
            return None
        page_nodes = sorted(
            [
                _to_int(n.target_start)
                for n in all_nodes
                if n.target_type == "page" and _to_int(n.target_start) is not None
            ],
        )
        page_nodes = [p for p in page_nodes if p is not None and p >= 1]
        if page_nodes:
            max_page = max(page_nodes)
            if max_page > 0:
                next_page = next((p for p in page_nodes if p > current_page), max_page + 1)
                start = int(((current_page - 1) / max_page) * content_length)
                end = int(((next_page - 1) / max_page) * content_length)
                if end <= start:
                    end = min(content_length, start + max(128, content_length // max(max_page, 1)))
                if end > start:
                    return start, end

    if node.target_type == "time_range":
        duration = _to_float(media.get("duration"))
        start_sec = _to_float(node.target_start)
        end_sec = _to_float(node.target_end)
        if duration and duration > 0 and start_sec is not None and start_sec >= 0:
            start = int(max(0.0, min(1.0, start_sec / duration)) * content_length)
            if end_sec is not None and end_sec > start_sec:
                end = int(max(0.0, min(1.0, end_sec / duration)) * content_length)
            else:
                end = min(content_length, start + max(256, content_length // 20))
            if end > start:
                return start, end

    heading = node.title
    if node.target_type == "href" and node.target_href:
        anchor = node.target_href.lstrip("#").replace("-", " ").replace("_", " ").strip()
        if anchor:
            heading = anchor
    try:
        if heading:
            lookup = lookup_section_by_heading(db, media_id, heading)
        else:
            lookup = None
    except (DatabaseError, TypeError):
        logger.debug("Section heading lookup failed")
        lookup = None

    if lookup:
        start_lookup = _to_int(lookup[0])
        end_lookup = _to_int(lookup[1])
        if start_lookup is not None and end_lookup is not None and 0 <= start_lookup < end_lookup:
            return start_lookup, end_lookup

    return None


def _detect_intrinsic_format(text: str) -> str:
    if not text:
        return "plain"
    if _HTML_TAG_RE.search(text):
        return "html"
    if _MD_HINT_RE.search(text):
        return "markdown"
    return "plain"


def _html_to_plain(html_text: str) -> str:
    no_tags = _HTML_TAG_RE.sub("", html_text or "")
    return html_lib.unescape(no_tags)


def _plain_to_html(plain_text: str) -> str:
    escaped = html_lib.escape(plain_text or "")
    if not escaped:
        return "<p></p>"
    paragraphs = [p.replace("\n", "<br/>") for p in escaped.split("\n\n")]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


def _markdown_to_html(markdown_text: str) -> str:
    if not markdown_text:
        return "<p></p>"
    try:
        import markdown as markdown_lib

        rendered = markdown_lib.markdown(markdown_text, extensions=["extra", "sane_lists"])
    except Exception:
        rendered = _plain_to_html(markdown_text)
    return rendered or "<p></p>"


def _build_content_variants(selected_text: str) -> tuple[dict[str, str], list[str]]:
    source_format = _detect_intrinsic_format(selected_text)
    variants: dict[str, str] = {}
    intrinsic: list[str] = ["plain"]

    if source_format == "html":
        variants["html"] = selected_text
        variants["plain"] = _html_to_plain(selected_text)
        intrinsic = ["plain", "html"]
    elif source_format == "markdown":
        variants["markdown"] = selected_text
        variants["plain"] = selected_text
        intrinsic = ["plain", "markdown"]
    else:
        variants["plain"] = selected_text
        intrinsic = ["plain"]

    if "plain" not in variants:
        variants["plain"] = selected_text
    return variants, intrinsic


def _resolve_requested_content(
    requested_format: str,
    variants: dict[str, str],
    intrinsic_formats: list[str],
) -> tuple[str, str]:
    if requested_format == "auto":
        for candidate in ("html", "markdown", "plain"):
            if candidate in intrinsic_formats:
                return candidate, variants.get(candidate, variants.get("plain", ""))
        return "plain", variants.get("plain", "")

    if requested_format in intrinsic_formats and requested_format in variants:
        return requested_format, variants.get(requested_format, "")

    if requested_format == "html":
        if "html" not in variants:
            if "markdown" in variants:
                variants["html"] = _markdown_to_html(variants.get("markdown", ""))
            else:
                variants["html"] = _plain_to_html(variants.get("plain", ""))
        return "html", variants.get("html", "")

    if requested_format == "markdown":
        if "markdown" not in variants:
            if "html" in variants and "plain" not in variants:
                variants["plain"] = _html_to_plain(variants.get("html", ""))
            variants["markdown"] = variants.get("plain", "")
        return "markdown", variants.get("markdown", "")

    return "plain", variants.get("plain", "")


def _build_navigation_content_cache_key(
    media_id: int,
    media: dict[str, Any],
    node_id: str,
    navigation_version: str,
    params: MediaNavigationContentQueryParams,
) -> str:
    media_version = _to_int(media.get("version")) or 0
    return (
        f"cache:/api/v1/media/{media_id}/navigation/{node_id}/content:"
        f"cv:{NAVIGATION_CACHE_VERSION}:"
        f"mv:{media_version}:"
        f"nav:{navigation_version}:"
        f"fmt:{params.format}:"
        f"alts:{int(bool(params.include_alternates))}"
    )


@router.get(
    "/{media_id:int}/navigation",
    status_code=status.HTTP_200_OK,
    summary="Get Media Navigation Tree",
    response_model=MediaNavigationResponse,
    dependencies=[Depends(rbac_rate_limit(MEDIA_NAVIGATION_ROUTE_POLICY.rate_limit_resource))],
    responses={
        200: {"description": "Navigation tree retrieved"},
        404: {"description": "Media item not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Server error"},
    },
)
async def get_media_navigation(
    media_id: int = Path(..., description="The ID of the media item"),
    params: MediaNavigationQueryParams = Depends(),
    db: MediaNavigationDb = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MediaNavigationResponse:
    """Return normalized chapter/section navigation for a media item."""
    logger.debug(
        "Fetching media navigation for media_id={}, user_id={}, max_depth={}, max_nodes={}, parent_id={}",
        media_id,
        getattr(current_user, "id", "?"),
        params.max_depth,
        params.max_nodes,
        params.parent_id,
    )

    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as exc:
        logger.error("Database error fetching media for navigation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from exc

    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    cache_key = _build_navigation_cache_key(media_id, media, params)
    cached = get_cached_response(cache_key)
    if cached is not None:
        _etag, payload = cached
        try:
            return MediaNavigationResponse.model_validate(payload)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid cached navigation payload")

    raw_nodes, source_order_used = await _select_source_nodes(
        media_id=media_id,
        db=db,
        media=media,
        include_generated_fallback=bool(params.include_generated_fallback),
    )
    all_nodes = _materialize_navigation_nodes(raw_nodes)
    filtered_nodes, node_count, truncated, max_depth_seen = _filter_navigation_nodes(all_nodes, params)

    response = MediaNavigationResponse(
        media_id=media_id,
        available=bool(all_nodes),
        navigation_version=_compute_navigation_version(
            media_id=media_id,
            media=media,
            source_order_used=source_order_used,
            nodes=all_nodes,
        ),
        source_order_used=source_order_used,
        nodes=filtered_nodes,
        stats=MediaNavigationStats(
            returned_node_count=len(filtered_nodes),
            node_count=node_count,
            max_depth=max_depth_seen,
            truncated=truncated,
        ),
    )

    cache_response(cache_key, response.model_dump(mode="json"), media_id=media_id)
    return response


@router.get(
    "/{media_id:int}/navigation/{node_id}/content",
    status_code=status.HTTP_200_OK,
    summary="Get Media Navigation Node Content",
    response_model=MediaNavigationContentResponse,
    dependencies=[Depends(rbac_rate_limit(MEDIA_NAVIGATION_ROUTE_POLICY.rate_limit_resource))],
    responses={
        200: {"description": "Node content retrieved"},
        404: {"description": "Media item or navigation node not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Server error"},
    },
)
async def get_media_navigation_content(
    media_id: int = Path(..., description="The ID of the media item"),
    node_id: str = Path(..., min_length=1, description="Navigation node ID"),
    params: MediaNavigationContentQueryParams = Depends(),
    db: MediaNavigationDb = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MediaNavigationContentResponse:
    """Return content payload for a selected navigation node."""
    logger.debug(
        "Fetching media navigation content for media_id={}, node_id={}, user_id={}, format={}, include_alternates={}",
        media_id,
        node_id,
        getattr(current_user, "id", "?"),
        params.format,
        params.include_alternates,
    )

    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as exc:
        logger.error("Database error fetching media for navigation content")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from exc

    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    raw_nodes, source_order_used = await _select_source_nodes(
        media_id=media_id,
        db=db,
        media=media,
        include_generated_fallback=False,
    )
    all_nodes = _materialize_navigation_nodes(raw_nodes)
    navigation_version = _compute_navigation_version(
        media_id=media_id,
        media=media,
        source_order_used=source_order_used,
        nodes=all_nodes,
    )

    node = _find_navigation_node(all_nodes, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "NAVIGATION_NODE_NOT_FOUND",
                "message": f"Node '{node_id}' not found in current navigation set.",
                "media_id": media_id,
                "node_id": node_id,
                "navigation_version": navigation_version,
            },
        )

    cache_key = _build_navigation_content_cache_key(
        media_id=media_id,
        media=media,
        node_id=node_id,
        navigation_version=navigation_version,
        params=params,
    )
    cached = get_cached_response(cache_key)
    if cached is not None:
        _etag, payload = cached
        try:
            return MediaNavigationContentResponse.model_validate(payload)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid cached navigation content payload")

    full_text = _get_media_text(media_id=media_id, media=media, db=db)
    span = _derive_content_span(
        node=node,
        all_nodes=all_nodes,
        media=media,
        db=db,
        media_id=media_id,
        content_length=len(full_text),
    )
    if span is None:
        selected_text = full_text
    else:
        selected_text = _slice_text(full_text, span[0], span[1])
    if not selected_text:
        selected_text = full_text

    variants, intrinsic_formats = _build_content_variants(selected_text)
    resolved_format, resolved_content = _resolve_requested_content(
        requested_format=params.format,
        variants=variants,
        intrinsic_formats=intrinsic_formats,
    )

    alternate_content = None
    if params.include_alternates:
        alt_map = {fmt: variants[fmt] for fmt in intrinsic_formats if fmt != resolved_format and fmt in variants}
        alternate_content = alt_map or None

    response = MediaNavigationContentResponse(
        media_id=media_id,
        node_id=node.id,
        title=node.title,
        content_format=resolved_format,
        available_formats=intrinsic_formats,
        content=resolved_content,
        alternate_content=alternate_content,
        target=MediaNavigationTarget(
            target_type=node.target_type,
            target_start=node.target_start,
            target_end=node.target_end,
            target_href=node.target_href,
        ),
    )

    cache_response(cache_key, response.model_dump(mode="json"), media_id=media_id)
    return response
