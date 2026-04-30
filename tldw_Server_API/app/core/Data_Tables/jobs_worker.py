"""
Data Tables Jobs worker.

- Consumes core Jobs entries for data table generation/regeneration.
- domain = "data_tables"
- queue = os.getenv("DATA_TABLES_JOBS_QUEUE", "default")
- job_type = "data_table_generate"

Payload fields:
- table_id (required)
- table_uuid (optional)
- prompt (required)
- sources (required)
- column_hints (optional)
- model (optional)
- max_rows (optional)
- regenerate (optional)

Usage:
  python -m tldw_Server_API.app.core.Data_Tables.jobs_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from typing import Any, Callable

from cachetools import LRUCache
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    get_user_chacha_db_path,
    get_user_media_db_path,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    get_document_version,
    get_latest_transcription,
    get_media_by_id,
)
from tldw_Server_API.app.core.exceptions import DataTablesJobError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputNoPayloadError,
    StructuredOutputOptions,
    StructuredOutputParseError,
    parse_structured_output,
)
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

DATA_TABLES_DOMAIN = "data_tables"
DATA_TABLES_JOB_TYPE = "data_table_generate"

_ALLOWED_COLUMN_TYPES = {"text", "number", "date", "url", "boolean", "currency"}
_COLUMN_TYPE_ALIASES = {
    "string": "text",
    "str": "text",
    "integer": "number",
    "int": "number",
    "float": "number",
    "decimal": "number",
    "double": "number",
    "bool": "boolean",
    "link": "url",
    "uri": "url",
    "datetime": "date",
    "timestamp": "date",
    "money": "currency",
}

_DEFAULT_MAX_ROWS = int(os.getenv("DATA_TABLES_DEFAULT_MAX_ROWS", "200") or "200")
_MAX_ROWS_LIMIT = int(os.getenv("DATA_TABLES_MAX_ROWS", "2000") or "2000")
_MAX_SOURCE_CHARS = int(os.getenv("DATA_TABLES_MAX_SOURCE_CHARS", "12000") or "12000")
_MAX_TOTAL_SOURCE_CHARS = int(os.getenv("DATA_TABLES_MAX_TOTAL_SOURCE_CHARS", "60000") or "60000")
_MAX_SNAPSHOT_CHARS = int(os.getenv("DATA_TABLES_MAX_SNAPSHOT_CHARS", "8000") or "8000")
_MAX_PROMPT_CHARS = int(os.getenv("DATA_TABLES_MAX_PROMPT_CHARS", "24000") or "24000")
_CHAT_BATCH_SIZE = int(os.getenv("DATA_TABLES_CHAT_BATCH_SIZE", "250") or "250")
_CHAT_MAX_MESSAGES = int(os.getenv("DATA_TABLES_CHAT_MAX_MESSAGES", "1500") or "1500")
_LLM_MAX_TOKENS = int(os.getenv("DATA_TABLES_LLM_MAX_TOKENS", "2000") or "2000")
_LLM_TEMPERATURE = float(os.getenv("DATA_TABLES_LLM_TEMPERATURE", "0.2") or "0.2")
_LLM_TIMEOUT_SECONDS = int(os.getenv("DATA_TABLES_LLM_TIMEOUT", "300") or "300")

_PROMPT_TEMPLATE = """You are a data table generator.

Goal:
{prompt}

Column hints (use if helpful, keep order when possible):
{column_hints}

Sources:
{sources}

Return ONLY valid JSON in this exact format:
{{
  "columns": [
    {{
      "name": "Column name",
      "type": "text|number|date|url|boolean|currency",
      "description": "Optional description",
      "format": "Optional formatting hint"
    }}
  ],
  "rows": [
    ["cell1", "cell2", "cell3"]
  ]
}}

Rules:
- Column names must be unique.
- Each row array length must match the number of columns.
- Use null for unknown values.
- Dates must be ISO-8601 (YYYY-MM-DD) when possible.
- Do not exceed {max_rows} rows.
- Output JSON only, no markdown or commentary.
"""


_MAX_DB_CACHE_SIZE = max(1, int(os.getenv("DATA_TABLES_DB_CACHE_SIZE", "32") or "32"))
_MEDIA_DB_CACHE: LRUCache = LRUCache(maxsize=_MAX_DB_CACHE_SIZE)
_CHACHA_DB_CACHE: LRUCache = LRUCache(maxsize=_MAX_DB_CACHE_SIZE)
_MEDIA_DB_LOCK = threading.Lock()
_CHACHA_DB_LOCK = threading.Lock()
DATA_TABLES_DB_EXCEPTIONS = (
    sqlite3.Error,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)
DATA_TABLES_RUNTIME_EXCEPTIONS = (
    *DATA_TABLES_DB_EXCEPTIONS,
    json.JSONDecodeError,
    ChatConfigurationError,
)


def _close_media_db(user_id: str, db: Any) -> None:
    try:
        db.close_connection()
    except DATA_TABLES_DB_EXCEPTIONS:
        logger.warning("data_tables: failed to close media db")


def _close_chacha_db(user_id: str, db: CharactersRAGDB) -> None:
    try:
        db.close_connection()
    except DATA_TABLES_DB_EXCEPTIONS:
        logger.warning("data_tables: failed to close chacha db")


def _evict_lru_entry(cache: LRUCache, on_evict: Callable[[str, Any], None]) -> None:
    if len(cache) < cache.maxsize:
        return
    evicted_key, evicted_value = cache.popitem()
    on_evict(evicted_key, evicted_value)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_user_id(job: dict[str, Any], payload: dict[str, Any]) -> str:
    owner = payload.get("user_id") or job.get("owner_user_id")
    if owner is None or str(owner).strip() == "":
        return str(DatabasePaths.get_single_user_id())
    return str(owner)


def _normalize_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_max_rows(payload: dict[str, Any]) -> int:
    raw = payload.get("max_rows")
    requested = _coerce_int(raw, _DEFAULT_MAX_ROWS)
    if requested <= 0:
        requested = _DEFAULT_MAX_ROWS
    return max(1, min(requested, _MAX_ROWS_LIMIT))


def _resolve_model(provider: str, model: str | None, app_config: dict[str, Any]) -> str | None:
    if model:
        return model
    key = f"{provider.replace('-', '_').replace('.', '_')}_api"
    return (app_config.get(key) or {}).get("model")


def _get_adapter(provider: str) -> Any:
    registry = get_registry()
    adapter = registry.get_adapter(provider)
    if adapter is None:
        raise ChatConfigurationError(provider=provider, message="LLM adapter unavailable.")
    return adapter


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _normalize_column_type(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in _ALLOWED_COLUMN_TYPES:
        return text
    return _COLUMN_TYPE_ALIASES.get(text)


def _normalize_column_hints(raw: Any) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return hints
    for item in raw:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif hasattr(item, "dict"):
            data = item.dict()
        elif isinstance(item, dict):
            data = dict(item)
        else:
            continue
        name = str(data.get("name") or "").strip()
        if not name:
            continue
        hints.append(
            {
                "name": name,
                "type": data.get("type"),
                "description": data.get("description"),
                "format": data.get("format"),
            }
        )
    return hints


def _normalize_column_key(value: str) -> str:
    return str(value or "").strip().lower()


def _dedupe_column_names(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        base = str(name or "").strip() or "Column"
        key = _normalize_column_key(base)
        if key not in counts:
            counts[key] = 1
            output.append(base)
            continue
        counts[key] += 1
        output.append(f"{base} ({counts[key]})")
    return output


def _normalize_columns(
    raw_columns: Any,
    column_hints: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    if isinstance(raw_columns, list):
        for raw in raw_columns:
            if isinstance(raw, dict):
                name = raw.get("name") or raw.get("column") or raw.get("title")
                if name is None:
                    continue
                name = str(name).strip()
                if not name:
                    continue
                col_type = _normalize_column_type(raw.get("type") or raw.get("kind"))
                columns.append(
                    {
                        "name": name,
                        "type": col_type,
                        "description": raw.get("description"),
                        "format": raw.get("format"),
                    }
                )
            elif isinstance(raw, str):
                name = raw.strip()
                if name:
                    columns.append({"name": name, "type": None, "description": None, "format": None})

    hints = list(column_hints or [])
    if not columns and hints:
        for hint in hints:
            name = str(hint.get("name") or "").strip()
            if not name:
                continue
            columns.append(
                {
                    "name": name,
                    "type": hint.get("type"),
                    "description": hint.get("description"),
                    "format": hint.get("format"),
                }
            )

    if hints:
        hint_map = {_normalize_column_key(str(h.get("name") or "")): h for h in hints}
        used: set[str] = set()
        for col in columns:
            key = _normalize_column_key(col.get("name") or "")
            hint = hint_map.get(key)
            if hint:
                used.add(key)
                if not col.get("type") and hint.get("type"):
                    col["type"] = hint.get("type")
                if not col.get("description") and hint.get("description"):
                    col["description"] = hint.get("description")
                if not col.get("format") and hint.get("format"):
                    col["format"] = hint.get("format")
        for hint in hints:
            key = _normalize_column_key(hint.get("name") or "")
            if key in used:
                continue
            name = str(hint.get("name") or "").strip()
            if not name:
                continue
            columns.append(
                {
                    "name": name,
                    "type": hint.get("type"),
                    "description": hint.get("description"),
                    "format": hint.get("format"),
                }
            )

    if not columns:
        raise DataTablesJobError("columns_missing", retryable=False)

    deduped = _dedupe_column_names([col.get("name") or "" for col in columns])
    for col, name in zip(columns, deduped, strict=True):
        col["name"] = name
        col_type = _normalize_column_type(col.get("type"))
        col["type"] = col_type or "text"
    return columns


def _safe_cell_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def _normalize_rows(raw_rows: Any, columns: list[dict[str, Any]]) -> list[list[Any]]:
    if not isinstance(raw_rows, list):
        return []
    col_count = len(columns)
    col_lookup = {_normalize_column_key(col["name"]): idx for idx, col in enumerate(columns)}
    normalized: list[list[Any]] = []

    for raw in raw_rows:
        row_values: list[Any] | None = None
        if isinstance(raw, dict):
            row_values = [None] * col_count
            for key, value in raw.items():
                idx = col_lookup.get(_normalize_column_key(str(key)))
                if idx is None:
                    continue
                row_values[idx] = _safe_cell_value(value)
            if all(v is None for v in row_values):
                row_values = None
        elif isinstance(raw, (list, tuple)):
            row_values = [_safe_cell_value(v) for v in list(raw)]
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                row_values = [None] * col_count
                for key, value in parsed.items():
                    idx = col_lookup.get(_normalize_column_key(str(key)))
                    if idx is None:
                        continue
                    row_values[idx] = _safe_cell_value(value)
                if all(v is None for v in row_values):
                    row_values = None
            elif isinstance(parsed, list):
                row_values = [_safe_cell_value(v) for v in parsed]
        if row_values is None:
            continue
        if len(row_values) < col_count:
            row_values.extend([None] * (col_count - len(row_values)))
        elif len(row_values) > col_count:
            row_values = row_values[:col_count]
        normalized.append(row_values)

    return normalized


def _extract_json_payload(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        raise DataTablesJobError("llm_response_empty", retryable=False)
    text = str(raw).strip()
    if not text:
        raise DataTablesJobError("llm_response_empty", retryable=False)
    try:
        return parse_structured_output(
            text,
            options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
        )
    except (StructuredOutputNoPayloadError, StructuredOutputParseError) as exc:
        raise DataTablesJobError("llm_response_invalid_json", retryable=False) from exc


def _extract_text_from_snapshot(snapshot: Any) -> str | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, str):
        text = snapshot.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return snapshot
            return _extract_text_from_snapshot(parsed)
        return snapshot
    if isinstance(snapshot, dict):
        for key in ("text", "content", "summary"):
            value = snapshot.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(snapshot.get("chunks"), list):
            parts = []
            for chunk in snapshot.get("chunks") or []:
                if isinstance(chunk, dict):
                    text = chunk.get("chunk_text") or chunk.get("text") or chunk.get("content")
                else:
                    text = chunk
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)
        if isinstance(snapshot.get("documents"), list):
            parts = []
            for doc in snapshot.get("documents") or []:
                text = doc.get("content") or doc.get("text") if isinstance(doc, dict) else doc
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)
    if isinstance(snapshot, list):
        parts = [str(item).strip() for item in snapshot if str(item).strip()]
        return "\n".join(parts) if parts else None
    return None


def _normalize_rag_sources(raw_sources: Any) -> list[str]:
    if not raw_sources:
        return ["media_db"]
    if isinstance(raw_sources, str):
        sources = [raw_sources]
    elif isinstance(raw_sources, list):
        sources = raw_sources
    else:
        return ["media_db"]
    normalized: list[str] = []
    for src in sources:
        name = str(src).strip().lower()
        if name in {"media", "media_db"}:
            normalized.append("media_db")
        elif name in {"notes"}:
            normalized.append("notes")
        elif name in {"characters", "chats", "character_cards"}:
            normalized.append("characters")
        else:
            normalized.append(name)
    return normalized or ["media_db"]


def _normalize_rag_document(doc: Any) -> tuple[str, str, dict[str, Any], float]:
    if hasattr(doc, "content"):
        doc_id = str(getattr(doc, "id", "") or "")
        content = getattr(doc, "content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        score = float(getattr(doc, "score", 0.0) or 0.0)
        return doc_id, str(content), dict(metadata), score
    if isinstance(doc, dict):
        doc_id = str(doc.get("id") or doc.get("chunk_id") or doc.get("document_id") or "")
        content = doc.get("content") or doc.get("text") or doc.get("chunk_text") or ""
        metadata = doc.get("metadata") or {}
        score = float(doc.get("score") or doc.get("relevance_score") or 0.0)
        return doc_id, str(content), dict(metadata) if isinstance(metadata, dict) else {}, score
    return "", "", {}, 0.0


def _build_rag_snapshot(
    *,
    query: str,
    retrieval_params: dict[str, Any],
    documents: Sequence[Any],
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents, start=1):
        doc_id, content, metadata, score = _normalize_rag_document(doc)
        if not content:
            continue
        text = _truncate_text(content, _MAX_SNAPSHOT_CHARS)
        chunk: dict[str, Any] = {
            "chunk_id": doc_id or f"chunk_{idx}",
            "chunk_text": text,
            "chunk_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "score": score,
            "rank": idx,
        }
        media_id = metadata.get("media_id")
        if media_id is not None:
            chunk["media_id"] = media_id
        if metadata.get("title"):
            chunk["title"] = metadata.get("title")
        if metadata.get("source"):
            chunk["source"] = metadata.get("source")
        chunks.append(chunk)
    return {
        "query": query,
        "retrieval": retrieval_params,
        "chunks": chunks,
    }


def _build_sources_text(resolved: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    total_chars = 0
    for idx, src in enumerate(resolved, start=1):
        text = str(src.get("text") or "")
        if not text.strip():
            continue
        text = _truncate_text(text.strip(), _MAX_SOURCE_CHARS)
        header = f"[Source {idx}] type={src.get('source_type')} id={src.get('source_id')}"
        title = src.get("title")
        if title:
            header += f" title={title}"
        block = f"{header}\n{text}"
        total_chars += len(block)
        if total_chars > _MAX_TOTAL_SOURCE_CHARS:
            remaining = _MAX_TOTAL_SOURCE_CHARS - (total_chars - len(block))
            if remaining <= 0:
                break
            block = _truncate_text(block, remaining)
            parts.append(block)
            break
        parts.append(block)
    return "\n\n".join(parts)


def _render_column_hints(column_hints: Sequence[dict[str, Any]]) -> str:
    if not column_hints:
        return "None"
    lines = []
    for hint in column_hints:
        name = str(hint.get("name") or "").strip()
        if not name:
            continue
        hint_type = str(hint.get("type") or "").strip()
        desc = str(hint.get("description") or "").strip()
        fmt = str(hint.get("format") or "").strip()
        line = f"- {name}"
        if hint_type:
            line += f" ({hint_type})"
        if desc:
            line += f": {desc}"
        if fmt:
            line += f" [format: {fmt}]"
        lines.append(line)
    return "\n".join(lines) if lines else "None"


def _build_prompt(
    *,
    prompt: str,
    sources: list[dict[str, Any]],
    column_hints: Sequence[dict[str, Any]],
    max_rows: int,
) -> str:
    sources_text = _build_sources_text(sources)
    if not sources_text:
        raise DataTablesJobError("sources_empty", retryable=False)
    rendered_hints = _render_column_hints(column_hints)
    filled = _PROMPT_TEMPLATE.format(
        prompt=prompt,
        column_hints=rendered_hints,
        sources=sources_text,
        max_rows=max_rows,
    )
    if len(filled) <= _MAX_PROMPT_CHARS:
        return filled
    base = _PROMPT_TEMPLATE.format(
        prompt=prompt,
        column_hints=rendered_hints,
        sources="",
        max_rows=max_rows,
    )
    if len(base) > _MAX_PROMPT_CHARS:
        raise DataTablesJobError("prompt_too_long", retryable=False)
    allowed = max(0, _MAX_PROMPT_CHARS - len(base))
    sources_text = _truncate_text(sources_text, allowed)
    return _PROMPT_TEMPLATE.format(
        prompt=prompt,
        column_hints=rendered_hints,
        sources=sources_text,
        max_rows=max_rows,
    )


def _get_media_db(user_id: str) -> Any:
    with _MEDIA_DB_LOCK:
        cached = _MEDIA_DB_CACHE.get(user_id)
        if cached is not None:
            return cached
    db_path = get_user_media_db_path(user_id)
    db = create_media_database(client_id=str(user_id), db_path=db_path)
    with _MEDIA_DB_LOCK:
        cached = _MEDIA_DB_CACHE.get(user_id)
        if cached is not None:
            _close_media_db(user_id, db)
            return cached
        _evict_lru_entry(_MEDIA_DB_CACHE, _close_media_db)
        _MEDIA_DB_CACHE[user_id] = db
    return db


def _get_chacha_db(user_id: str) -> CharactersRAGDB:
    with _CHACHA_DB_LOCK:
        cached = _CHACHA_DB_CACHE.get(user_id)
        if cached is not None:
            return cached
    db_path = get_user_chacha_db_path(user_id)
    db = CharactersRAGDB(db_path=str(db_path), client_id=str(user_id))
    with _CHACHA_DB_LOCK:
        cached = _CHACHA_DB_CACHE.get(user_id)
        if cached is not None:
            _close_chacha_db(user_id, db)
            return cached
        _evict_lru_entry(_CHACHA_DB_CACHE, _close_chacha_db)
        _CHACHA_DB_CACHE[user_id] = db
    return db


def _extract_media_text(db: Any, media_id: int) -> str:
    media_item = get_media_by_id(db, media_id)
    if not media_item:
        raise DataTablesJobError(f"media_not_found:{media_id}", retryable=False)
    content = media_item.get("content")
    if isinstance(content, dict):
        text = content.get("content") or content.get("text") or ""
    elif isinstance(content, str):
        text = content
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                text = parsed.get("content") or parsed.get("text") or text
    else:
        text = ""

    if not text.strip():
        try:
            latest = get_document_version(db, media_id=media_id, version_number=None, include_content=True)
        except DATA_TABLES_DB_EXCEPTIONS:
            logger.debug("data_tables: get_document_version failed while extracting media text")
            latest = None
        if latest and latest.get("content"):
            text = str(latest.get("content") or "")
        else:
            try:
                fallback = get_latest_transcription(db, media_id)
            except DATA_TABLES_DB_EXCEPTIONS:
                logger.debug("data_tables: get_latest_transcription failed while extracting media text")
                fallback = None
            if fallback:
                text = fallback
    text = str(text or "").strip()
    if not text:
        raise DataTablesJobError(f"media_missing_content:{media_id}", retryable=False)
    return text


def _extract_chat_text(db: CharactersRAGDB, conversation_id: str) -> str:
    messages: list[dict[str, Any]] = []
    offset = 0
    while len(messages) < _CHAT_MAX_MESSAGES:
        batch = db.get_messages_for_conversation(
            conversation_id,
            limit=min(_CHAT_BATCH_SIZE, _CHAT_MAX_MESSAGES - len(messages)),
            offset=offset,
            order_by_timestamp="ASC",
            include_deleted=False,
        )
        if not batch:
            break
        messages.extend(batch)
        offset += len(batch)
        if len(batch) < _CHAT_BATCH_SIZE:
            break
    lines: list[str] = []
    for msg in messages:
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        sender = str(msg.get("sender") or "unknown").strip() or "unknown"
        lines.append(f"{sender}: {content}")
    text = "\n".join(lines).strip()
    if not text:
        raise DataTablesJobError(f"chat_missing_content:{conversation_id}", retryable=False)
    return text


async def _resolve_rag_query_source(
    *,
    query: str,
    media_db: Any,
    chacha_db: CharactersRAGDB,
    retrieval_params: dict[str, Any],
    user_id: str,
) -> tuple[str, dict[str, Any]]:
    sources = _normalize_rag_sources(retrieval_params.get("sources"))
    search_mode = str(retrieval_params.get("search_mode") or "hybrid")
    fts_level = str(retrieval_params.get("fts_level") or "chunk")
    top_k = _coerce_int(retrieval_params.get("top_k"), 10)
    min_score = _coerce_float(retrieval_params.get("min_score"), 0.0)

    result = await unified_rag_pipeline(
        query=query,
        sources=sources,
        media_db_path=str(media_db.db_path_str),
        notes_db_path=str(get_user_chacha_db_path(user_id)),
        character_db_path=str(get_user_chacha_db_path(user_id)),
        search_mode=search_mode,
        fts_level=fts_level,
        top_k=top_k,
        min_score=min_score,
        enable_generation=False,
        enable_citations=False,
        enable_monitoring=False,
        enable_observability=False,
        user_id=str(user_id),
        media_db=media_db,
        chacha_db=chacha_db,
    )

    documents = []
    if hasattr(result, "documents"):
        documents = list(result.documents or [])
    elif isinstance(result, dict):
        documents = list(result.get("documents") or [])
    snapshot = _build_rag_snapshot(query=query, retrieval_params=retrieval_params, documents=documents)
    text = _extract_text_from_snapshot(snapshot) or ""
    return text, snapshot


def _is_job_cancelled(jm: JobManager, job_id: int) -> bool:
    try:
        job = jm.get_job(job_id)
    except DATA_TABLES_DB_EXCEPTIONS:
        return False
    status = str(job.get("status") or "").lower() if job else ""
    return status == "cancelled"


async def _handle_job(job: dict[str, Any], jm: JobManager) -> dict[str, Any]:
    """Handle a single data tables job and persist the generated table."""
    payload = _normalize_payload(job.get("payload"))
    user_id = _normalize_user_id(job, payload)
    table_id = _coerce_int(payload.get("table_id"), 0)
    db: Any | None = None
    table_row: dict[str, Any] | None = None
    try:
        job_type = str(job.get("job_type") or payload.get("job_type") or "").strip().lower()
        if job_type and job_type != DATA_TABLES_JOB_TYPE:
            raise DataTablesJobError(f"unsupported_job_type:{job_type}", retryable=False)

        if table_id <= 0:
            raise DataTablesJobError("missing_table_id", retryable=False)
        prompt = str(payload.get("prompt") or "").strip()
        model = str(payload.get("model") or "").strip() or None

        job_id = _coerce_int(job.get("id"), 0)
        max_rows = _resolve_max_rows(payload)

        db = _get_media_db(user_id)
        chacha_db = _get_chacha_db(user_id)
        table_row = db.get_data_table(table_id, include_deleted=True, owner_user_id=user_id)
        if not table_row or int(table_row.get("deleted") or 0):
            raise DataTablesJobError("data_table_not_found", retryable=False)

        if not prompt:
            prompt = str(table_row.get("prompt") or "").strip()
        if not prompt:
            raise DataTablesJobError("missing_prompt", retryable=False)

        raw_hints = payload.get("column_hints")
        if raw_hints is None:
            raw_hints = table_row.get("column_hints_json")
            if isinstance(raw_hints, str):
                try:
                    raw_hints = json.loads(raw_hints)
                except json.JSONDecodeError:
                    raw_hints = None
        column_hints = _normalize_column_hints(raw_hints or [])

        sources = payload.get("sources")
        if not isinstance(sources, list):
            sources = []
        if not sources:
            stored = db.list_data_table_sources(table_id, owner_user_id=user_id)
            sources = [
                {
                    "source_type": row.get("source_type"),
                    "source_id": row.get("source_id"),
                    "title": row.get("title"),
                    "snapshot": row.get("snapshot_json"),
                    "retrieval_params": row.get("retrieval_params_json"),
                }
                for row in stored
            ]
        if not sources:
            raise DataTablesJobError("missing_sources", retryable=False)

        db.update_data_table(
            table_id,
            status="running",
            last_error=None,
            generation_model=model or table_row.get("generation_model"),
            prompt=prompt,
            owner_user_id=user_id,
        )
        jm.update_job_progress(job_id, progress_percent=5.0, progress_message="resolve_sources")

        if _is_job_cancelled(jm, job_id):
            db.update_data_table(table_id, status="cancelled", last_error=None, owner_user_id=user_id)
            return {"cancelled": True, "table_id": table_id}

        resolved_sources: list[dict[str, Any]] = []
        sources_db_payload: list[dict[str, Any]] = []

        for source in sources:
            if not isinstance(source, dict):
                continue
            source_type = str(source.get("source_type") or "").strip().lower()
            source_id = str(source.get("source_id") or "").strip()
            title = source.get("title")
            snapshot = source.get("snapshot")
            if isinstance(snapshot, str):
                text = snapshot.strip()
                if text.startswith("{") or text.startswith("["):
                    with contextlib.suppress(json.JSONDecodeError):
                        snapshot = json.loads(text)
            retrieval_params = source.get("retrieval_params") or {}
            if isinstance(retrieval_params, str):
                try:
                    retrieval_params = json.loads(retrieval_params)
                except json.JSONDecodeError:
                    retrieval_params = {}

            if not source_type or not source_id:
                continue

            text = _extract_text_from_snapshot(snapshot)
            updated_snapshot = snapshot

            if source_type == "chat":
                if not text:
                    text = _extract_chat_text(chacha_db, source_id)
            elif source_type == "document":
                if not text:
                    try:
                        media_id = int(source_id)
                    except (TypeError, ValueError) as exc:
                        raise DataTablesJobError("invalid_media_id", retryable=False) from exc
                    text = _extract_media_text(db, media_id)
            elif source_type == "rag_query":
                query = source_id
                if not text:
                    text, updated_snapshot = await _resolve_rag_query_source(
                        query=query,
                        media_db=db,
                        chacha_db=chacha_db,
                        retrieval_params=dict(retrieval_params or {}),
                        user_id=user_id,
                    )
                elif isinstance(updated_snapshot, dict):
                    updated_snapshot.setdefault("query", query)
                    updated_snapshot.setdefault("retrieval", retrieval_params)
            else:
                raise DataTablesJobError(f"unsupported_source_type:{source_type}", retryable=False)

            text = str(text or "").strip()
            resolved_sources.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "title": title,
                    "text": text,
                }
            )
            sources_db_payload.append(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "title": title,
                    "snapshot_json": updated_snapshot,
                    "retrieval_params_json": retrieval_params or None,
                }
            )

        jm.update_job_progress(job_id, progress_percent=30.0, progress_message="build_prompt")
        if _is_job_cancelled(jm, job_id):
            db.update_data_table(table_id, status="cancelled", last_error=None, owner_user_id=user_id)
            return {"cancelled": True, "table_id": table_id}

        llm_prompt = _build_prompt(
            prompt=prompt,
            sources=resolved_sources,
            column_hints=column_hints,
            max_rows=max_rows,
        )

        jm.update_job_progress(job_id, progress_percent=55.0, progress_message="llm_generate")
        if _is_job_cancelled(jm, job_id):
            db.update_data_table(table_id, status="cancelled", last_error=None, owner_user_id=user_id)
            return {"cancelled": True, "table_id": table_id}

        provider = (DEFAULT_LLM_PROVIDER or "openai").strip().lower()
        api_key, _debug = resolve_provider_api_key(provider, prefer_module_keys_in_tests=True)
        if provider_requires_api_key(provider) and not api_key:
            raise DataTablesJobError(f"missing_api_key:{provider}", retryable=False)

        messages_payload = [{"role": "user", "content": llm_prompt}]
        response_format = {"type": "json_object"}

        def _call_llm():
            adapter = _get_adapter(provider)
            app_config = load_and_log_configs() or {}
            model_to_use = _resolve_model(provider, model, app_config)
            if model_to_use is None:
                raise ChatConfigurationError(provider=provider, message="Model is required for provider.")
            return adapter.chat(
                {
                    "messages": messages_payload,
                    "api_key": api_key,
                    "model": model_to_use,
                    "temperature": _LLM_TEMPERATURE,
                    "max_tokens": _LLM_MAX_TOKENS,
                    "response_format": response_format,
                    "app_config": app_config,
                }
            )

        start = time.time()
        loop = asyncio.get_running_loop()
        llm_future = loop.run_in_executor(None, _call_llm)
        timeout = _LLM_TIMEOUT_SECONDS if _LLM_TIMEOUT_SECONDS > 0 else None
        try:
            if timeout is None:
                raw_response = await llm_future
            else:
                raw_response = await asyncio.wait_for(llm_future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            llm_future.cancel()
            try:
                await llm_future
            except asyncio.CancelledError:
                pass
            except DATA_TABLES_RUNTIME_EXCEPTIONS as cleanup_exc:
                logger.debug(
                    "data_tables worker: error awaiting cancelled LLM future: {}",
                    cleanup_exc,
                )
            jm.update_job_progress(job_id, progress_percent=55.0, progress_message="llm_timeout")
            raise DataTablesJobError("llm_timeout", retryable=True) from exc
        logger.info(
            "data_tables worker: LLM call completed in {:.1f}ms",
            (time.time() - start) * 1000.0,
        )

        content_text = extract_response_content(raw_response)
        payload_obj = _extract_json_payload(content_text if content_text is not None else raw_response)
        if not isinstance(payload_obj, dict):
            raise DataTablesJobError("llm_payload_invalid", retryable=False)

        raw_columns = payload_obj.get("columns") or payload_obj.get("schema")
        raw_rows = payload_obj.get("rows") or payload_obj.get("data") or payload_obj.get("records")

        jm.update_job_progress(job_id, progress_percent=75.0, progress_message="validate")
        if _is_job_cancelled(jm, job_id):
            db.update_data_table(table_id, status="cancelled", last_error=None, owner_user_id=user_id)
            return {"cancelled": True, "table_id": table_id}

        columns = _normalize_columns(raw_columns, column_hints=column_hints)
        rows = _normalize_rows(raw_rows, columns)

        if not rows:
            raise DataTablesJobError("rows_missing", retryable=False)
        if len(rows) > max_rows:
            rows = rows[:max_rows]

        column_records: list[dict[str, Any]] = []
        for idx, col in enumerate(columns):
            column_records.append(
                {
                    "column_id": str(uuid.uuid4()),
                    "name": col["name"],
                    "type": col["type"],
                    "description": col.get("description"),
                    "format": col.get("format"),
                    "position": idx,
                }
            )

        row_records: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            row_json = {
                column_records[c_idx]["column_id"]: _safe_cell_value(value)
                for c_idx, value in enumerate(row)
            }
            row_records.append(
                {
                    "row_index": idx,
                    "row_json": row_json,
                }
            )

        jm.update_job_progress(job_id, progress_percent=90.0, progress_message="persist")
        if _is_job_cancelled(jm, job_id):
            db.update_data_table(table_id, status="cancelled", last_error=None, owner_user_id=user_id)
            return {"cancelled": True, "table_id": table_id}

        with db.transaction():
            db.persist_data_table_generation(
                table_id,
                columns=column_records,
                rows=row_records,
                sources=sources_db_payload if sources_db_payload else None,
                status="ready",
                row_count=len(row_records),
                generation_model=model or table_row.get("generation_model"),
                last_error=None,
            )

        jm.update_job_progress(job_id, progress_percent=100.0, progress_message="finalize")
        return {
            "table_id": table_id,
            "table_uuid": payload.get("table_uuid") or table_row.get("uuid"),
            "row_count": len(row_records),
            "columns": [{"name": col["name"], "type": col["type"]} for col in column_records],
        }
    except DataTablesJobError as exc:
        if db is not None and table_id > 0:
            try:
                db.update_data_table(table_id, status="failed", last_error=str(exc), owner_user_id=user_id)
            except DATA_TABLES_DB_EXCEPTIONS as reset_exc:
                logger.debug(
                    "data_tables worker: failed to update table status for {}: {}",
                    table_id,
                    reset_exc,
                )
        raise
    except DATA_TABLES_RUNTIME_EXCEPTIONS as exc:
        if db is not None and table_id > 0:
            try:
                db.update_data_table(table_id, status="failed", last_error=str(exc), owner_user_id=user_id)
            except DATA_TABLES_DB_EXCEPTIONS as reset_exc:
                logger.debug(
                    "data_tables worker: failed to update table status for {}: {}",
                    table_id,
                    reset_exc,
                )
        raise DataTablesJobError(str(exc), retryable=False) from exc


async def run_data_tables_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the data tables jobs worker loop until stopped."""
    worker_id = (os.getenv("DATA_TABLES_JOBS_WORKER_ID") or f"data-tables-jobs-{os.getpid()}").strip()
    queue = (os.getenv("DATA_TABLES_JOBS_QUEUE") or "default").strip() or "default"
    cfg = WorkerConfig(
        domain=DATA_TABLES_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=_coerce_int(os.getenv("DATA_TABLES_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60),
        renew_jitter_seconds=_coerce_int(os.getenv("DATA_TABLES_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=_coerce_int(os.getenv("DATA_TABLES_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"), 10),
        backoff_base_seconds=_coerce_int(os.getenv("DATA_TABLES_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("DATA_TABLES_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("DATA_TABLES_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )
    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info(
        "Data Tables Jobs worker starting (queue={}, worker_id={})",
        queue,
        worker_id,
    )
    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        return await _handle_job(job, jm)

    try:
        await sdk.run(handler=_handler)
    finally:
        if stop_watcher_task is not None and not stop_watcher_task.done():
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher_task


if __name__ == "__main__":
    asyncio.run(run_data_tables_jobs_worker())
