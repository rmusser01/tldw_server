from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from html.parser import HTMLParser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from tldw_Server_API.app.core.Collections.embedding_queue import enqueue_embeddings_job_for_item
from tldw_Server_API.app.core.Collections.reading_importers import ReadingImportItem, normalize_import_items
from tldw_Server_API.app.core.Collections.utils import (
    hash_text_sha256,
    is_supported_reading_url,
    normalize_reading_status,
    truncate_text,
    truncate_text_hard,
    word_count,
)
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ContentItemRow
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.http_client import afetch
from tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils import (
    _enforce_max_bytes_from_headers,
    _resolve_max_bytes,
    _resolve_media_type_from_content_type,
    _resolve_media_type_from_suffix,
)
from tldw_Server_API.app.core.Web_Scraping.url_utils import normalize_for_crawl


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(f"Invalid integer for {name}: {raw!r}; using default={default}")
        return default
    if minimum is not None and value < minimum:
        logger.warning(f"Out-of-range integer for {name}: {value}; minimum={minimum}; using default={default}")
        return default
    return value


READING_DEFAULT_STATUS = "saved"
READING_ARCHIVE_MAX_BYTES = _env_int("READING_ARCHIVE_MAX_BYTES", 5 * 1024 * 1024, minimum=0)
READING_ARCHIVE_RETENTION_DAYS = _env_int("READING_ARCHIVE_RETENTION_DAYS", 30, minimum=0)
READING_CONTENT_METADATA_MAX_CHARS = _env_int("READING_CONTENT_METADATA_MAX_CHARS", 50_000, minimum=0)
READING_CLEAN_HTML_METADATA_MAX_CHARS = _env_int("READING_CLEAN_HTML_METADATA_MAX_CHARS", 50_000, minimum=0)

_READING_SERVICE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EgressPolicyError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    NetworkError,
    OSError,
    PermissionError,
    RetryExhaustedError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _contains_html_tag(raw: str) -> bool:
    """Return True when raw includes a tag-like "<a>" sequence."""
    if not raw:
        return False
    tag_started = False
    length = len(raw)
    for idx, ch in enumerate(raw):
        if tag_started:
            if ch == ">":
                return True
            continue
        if ch == "<" and idx + 1 < length:
            next_char = raw[idx + 1]
            if ("A" <= next_char <= "Z") or ("a" <= next_char <= "z"):
                tag_started = True
    return False


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _env_flag_enabled(raw: str | None) -> bool:
    value = str(raw or "").strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def _strip_html(raw: str) -> str:
    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self._parts: list[str] = []
            self._ignore_depth = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
            if tag.lower() in {"script", "style"}:
                self._ignore_depth += 1

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() in {"script", "style"} and self._ignore_depth > 0:
                self._ignore_depth -= 1

        def handle_data(self, data: str) -> None:
            if self._ignore_depth == 0:
                self._parts.append(data)

        def text(self) -> str:
            return "".join(self._parts)

    parser = _Stripper()
    parser.feed(str(raw or ""))
    parser.close()
    return parser.text()


def _safe_filename_fragment(raw: str, max_len: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip()).strip("._")
    if not text:
        text = "reading"
    return text[:max_len]


def _add_bounded_metadata_text(metadata: dict[str, object], key: str, value: str | None, limit: int) -> None:
    """Store bounded metadata text plus truncation diagnostics."""
    bounded, truncated, original_len = truncate_text_hard(value, limit)
    if bounded:
        metadata[key] = bounded
    if value:
        metadata[f"{key}_char_count"] = original_len
        if truncated:
            metadata[f"{key}_truncated"] = True
            metadata[f"{key}_max_chars"] = limit


@dataclass
class ReadingSaveResult:
    item: ContentItemRow
    media_id: int | None
    media_uuid: str | None
    created: bool
    archive_requested: bool = False
    archive_output_id: int | None = None
    archive_error: str | None = None


@dataclass
class ReadingImportResult:
    imported: int
    updated: int
    skipped: int
    errors: list[str]


class ReadingArchiveService:
    """Create reading archive artifacts for saved items."""

    def __init__(self, *, user_id: int, collections: CollectionsDatabase) -> None:
        self.user_id = user_id
        self.collections = collections

    @staticmethod
    def resolve_archive_retention() -> str | None:
        try:
            days = max(0, int(READING_ARCHIVE_RETENTION_DAYS))
        except (TypeError, ValueError):
            return None
        if days <= 0:
            return None
        expires = datetime.now(tz=timezone.utc) + timedelta(days=days)
        return expires.isoformat()

    async def create_archive_artifact(
        self,
        *,
        item_id: int,
        title: str,
        url: str | None,
        body_text: str,
        media_item_id: int | None,
    ) -> tuple[int | None, str | None]:
        text_value = (body_text or "").strip()
        if not text_value:
            return None, "archive_no_content"
        content = f"# {title}\n\n{url or ''}\n\n{text_value}\n"
        content_bytes = content.encode("utf-8")
        if READING_ARCHIVE_MAX_BYTES > 0 and len(content_bytes) > READING_ARCHIVE_MAX_BYTES:
            return None, "archive_too_large"

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_title = _safe_filename_fragment(title, max_len=40)
        filename = f"reading_archive_{item_id}_{safe_title}_{timestamp}.md"
        outputs_dir = DatabasePaths.get_user_outputs_dir(self.user_id)
        await asyncio.to_thread(outputs_dir.mkdir, parents=True, exist_ok=True)
        path = outputs_dir / filename
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")

        metadata = {
            "item_id": item_id,
            "url": url,
            "canonical_url": url,
            "source": "save_url",
            "format": "md",
            "title": title,
        }
        output_row = self.collections.create_output_artifact(
            type_="reading_archive",
            title=f"{title} (archive {timestamp})",
            format_="md",
            storage_path=filename,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
            media_item_id=media_item_id,
            retention_until=self.resolve_archive_retention(),
        )
        return int(output_row.id), None


class ReadingImportService:
    """Normalize and persist reading-list import items."""

    def __init__(self, *, collections: CollectionsDatabase, normalize_url: Any) -> None:
        self.collections = collections
        self._normalize_url = normalize_url

    def import_items(
        self,
        *,
        items: list[ReadingImportItem],
        merge_tags: bool = True,
        origin_type: str = "import",
    ) -> ReadingImportResult:
        skipped = sum(
            1
            for item in items
            if not is_supported_reading_url(str(getattr(item, "url", "") or "").strip())
        )
        normalized = normalize_import_items(items)
        imported = 0
        updated = 0
        errors: list[str] = []

        for item in normalized:
            url = item.url.strip() if item.url else ""
            if not url or not is_supported_reading_url(url):
                skipped += 1
                continue
            normalized_url = self._normalize_url(url, "reading_import")
            if not is_supported_reading_url(normalized_url):
                skipped += 1
                continue
            parsed = urlparse(normalized_url)
            domain = parsed.netloc.lower() if parsed.netloc else None
            status = normalize_reading_status(item.status)
            read_at = item.read_at
            if status == "read" and not read_at:
                read_at = _utcnow_iso()
            tags = item.tags or []
            metadata_payload = {"source": "reading_import"}
            metadata_payload.update(item.metadata or {})
            if normalized_url != url:
                metadata_payload.setdefault("import_original_url", url)
            try:
                row = self.collections.upsert_content_item(
                    origin="reading",
                    origin_type=origin_type,
                    origin_id=None,
                    url=normalized_url,
                    canonical_url=normalized_url,
                    domain=domain,
                    title=item.title or normalized_url,
                    summary=None,
                    notes=item.notes,
                    content_hash=None,
                    word_count=None,
                    published_at=None,
                    status=status,
                    favorite=item.favorite,
                    metadata=metadata_payload,
                    media_id=None,
                    job_id=None,
                    run_id=None,
                    source_id=None,
                    read_at=read_at,
                    tags=tags,
                    merge_tags=merge_tags,
                    preserve_existing_on_null=True,
                )
                if row.is_new:
                    imported += 1
                else:
                    updated += 1
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
                errors.append(f"{url}: {exc}")
        return ReadingImportResult(imported=imported, updated=updated, skipped=skipped, errors=errors)


class ReadingService:
    """Utilities for Reading List capture and updates."""

    def __init__(self, user_id: int | str) -> None:
        self.user_id = int(user_id)
        self.collections = CollectionsDatabase.for_user(self.user_id)
        self._archive_service = ReadingArchiveService(user_id=self.user_id, collections=self.collections)
        self._import_service = ReadingImportService(
            collections=self.collections,
            normalize_url=self._normalize_url,
        )

    @staticmethod
    def _normalize_url(value: str, source: str) -> str:
        try:
            normalized = normalize_for_crawl(value, source)
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS:
            return value
        return normalized or value

    @staticmethod
    async def _close_response(resp: Any) -> None:
        if resp is None:
            return
        close = getattr(resp, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
                return
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS:
                return
        close = getattr(resp, "close", None)
        if callable(close):
            try:
                close()
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS:
                return

    @staticmethod
    def _sanitize_html_content(raw: str) -> tuple[str, str | None]:
        """Sanitize HTML input and return (text, clean_html) when HTML is detected."""
        if not _contains_html_tag(raw):
            return raw, None
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator
            from tldw_Server_API.app.core.Web_Scraping.content import convert_html_to_markdown
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS:
            return raw, None
        try:
            validator = FileValidator()
            sanitized_html = validator.sanitize_html_content(raw)
            text = convert_html_to_markdown(sanitized_html)
            return text, sanitized_html
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS:
            return raw, None

    @staticmethod
    def _map_media_type_key(media_type_key: str | None) -> str | None:
        if not media_type_key:
            return None
        key = str(media_type_key).lower()
        if key == "xml":
            return "document"
        if key in {"pdf", "ebook", "document"}:
            return key
        return None

    async def _probe_url_metadata(self, url: str) -> dict[str, Any]:
        """Fetch headers to infer media type and enforce size limits."""
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        media_type_key = _resolve_media_type_from_suffix(suffix)
        resp = None
        content_type: str | None = None
        content_length: str | None = None
        resolved_url: str | None = None
        error: str | None = None
        status_code: int | None = None
        size_exceeded = False

        try:
            resp = await afetch(method="HEAD", url=url, timeout=10.0, allow_redirects=True)
            status_code = getattr(resp, "status_code", None)
            headers = getattr(resp, "headers", {}) or {}
            raw_content_type = headers.get("content-type") or headers.get("Content-Type") or ""
            content_type = raw_content_type.split(";", 1)[0].strip().lower() or None
            content_length = headers.get("content-length") or headers.get("Content-Length")
            resolved_url = str(getattr(resp, "url", "")) or url
            if isinstance(status_code, int) and status_code >= 400:
                error = f"head_status_{status_code}"
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            error = str(exc)
        finally:
            await self._close_response(resp)

        if content_type:
            media_type_key = _resolve_media_type_from_content_type(content_type) or media_type_key

        if content_length:
            try:
                max_bytes = _resolve_max_bytes(
                    max_bytes=None,
                    media_type_key=media_type_key,
                    effective_suffix=suffix,
                    content_type=content_type or "",
                )
                _enforce_max_bytes_from_headers(url, content_length, max_bytes)
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
                error = str(exc)
                size_exceeded = True

        return {
            "media_type_key": media_type_key,
            "content_type": content_type,
            "content_length": content_length,
            "resolved_url": resolved_url,
            "error": error,
            "status_code": str(status_code) if status_code is not None else None,
            "size_exceeded": size_exceeded,
        }

    async def _ingest_non_html(
        self,
        *,
        url: str,
        media_type_key: str | None,
        title_override: str | None,
    ) -> dict[str, Any]:
        """Route non-HTML URLs into the document ingestion pipeline."""
        media_type = self._map_media_type_key(media_type_key)
        if not media_type:
            return {
                "url": url,
                "canonical_url": url,
                "title": title_override or url,
                "content": "",
                "summary": None,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": media_type_key,
                "error": f"unsupported_content_type:{media_type_key or 'unknown'}",
            }

        try:
            from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
                resolve_default_transcription_model,
            )
            from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
                prepare_chunking_options_dict,
            )
            from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import TempDirManager
            from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import (
                process_document_like_item,
            )
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            return {
                "url": url,
                "canonical_url": url,
                "title": title_override or url,
                "content": "",
                "summary": None,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": media_type_key,
                "error": f"ingestion_import_failed:{exc}",
            }

        form_data = AddMediaForm(
            media_type=media_type,
            urls=[url],
            title=title_override,
            perform_analysis=False,
            transcription_model=resolve_default_transcription_model("whisper-large-v3"),
        )
        chunk_options = prepare_chunking_options_dict(form_data)
        db_path = str(DatabasePaths.get_media_db_path(self.user_id))

        loop = asyncio.get_running_loop()
        try:
            with TempDirManager(prefix="reading_ingest_") as temp_dir:
                result = await process_document_like_item(
                    item_input_ref=url,
                    processing_source=url,
                    media_type=media_type,
                    is_url=True,
                    form_data=form_data,
                    chunk_options=chunk_options,
                    temp_dir=temp_dir,
                    loop=loop,
                    db_path=db_path,
                    client_id=str(self.user_id),
                    user_id=self.user_id,
                )
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            return {
                "url": url,
                "canonical_url": url,
                "title": title_override or url,
                "content": "",
                "summary": None,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": media_type,
                "error": f"ingestion_failed:{exc}",
            }

        metadata = result.get("metadata") if isinstance(result, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        error = result.get("error") if isinstance(result, dict) else None
        title = title_override or metadata.get("title") or url
        author = metadata.get("author")
        content = result.get("content") if isinstance(result, dict) else None
        summary = None
        if isinstance(result, dict):
            summary = result.get("summary") or result.get("analysis")

        return {
            "url": url,
            "canonical_url": url,
            "title": title,
            "content": content or "",
            "summary": summary,
            "author": author,
            "published": None,
            "media_id": result.get("db_id") if isinstance(result, dict) else None,
            "media_uuid": result.get("media_uuid") if isinstance(result, dict) else None,
            "media_type": media_type,
            "error": error,
        }

    @staticmethod
    def _archive_default_enabled() -> bool:
        return _env_flag_enabled(os.getenv("READING_ARCHIVE_ON_SAVE_DEFAULT", "0"))

    @classmethod
    def _resolve_archive_requested(cls, archive_mode: str | None) -> bool:
        mode = str(archive_mode or "use_default").strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        return cls._archive_default_enabled()

    @staticmethod
    def _resolve_archive_retention() -> str | None:
        return ReadingArchiveService.resolve_archive_retention()

    async def _create_archive_artifact(
        self,
        *,
        item_id: int,
        title: str,
        url: str | None,
        body_text: str,
        media_item_id: int | None,
    ) -> tuple[int | None, str | None]:
        return await self._archive_service.create_archive_artifact(
            item_id=item_id,
            title=title,
            url=url,
            body_text=body_text,
            media_item_id=media_item_id,
        )

    async def save_url(
        self,
        *,
        url: str,
        tags: list[str] | None = None,
        status: str | None = None,
        favorite: bool = False,
        title_override: str | None = None,
        summary_override: str | None = None,
        content_override: str | None = None,
        notes: str | None = None,
        metadata: dict[str, object] | None = None,
        archive_mode: str = "use_default",
    ) -> ReadingSaveResult:
        """Fetch, dedupe, and persist a reading item."""
        normalized_status = normalize_reading_status(status, default=READING_DEFAULT_STATUS)
        tags = [t for t in (tags or []) if t]
        archive_requested = self._resolve_archive_requested(archive_mode)
        try:
            article = await self._fetch_article(
                url=url,
                title_override=title_override,
                content_override=content_override,
                summary_override=summary_override,
            )
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Reading article fetch failed for {url}: {exc}")
            article = {
                "url": url,
                "canonical_url": url,
                "title": title_override or url,
                "content": "",
                "summary": summary_override,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": None,
                "content_type": None,
                "error": str(exc),
            }

        title = article.get("title") or title_override or url
        content = article.get("content") or summary_override or ""
        summary = summary_override or article.get("summary") or truncate_text(content, limit=600)
        raw_canonical = article.get("canonical_url") or article.get("url") or url
        canonical_url = self._normalize_url(raw_canonical, url)
        published_at = article.get("published") or None
        media_id = article.get("media_id")
        media_uuid = article.get("media_uuid")
        content_word_count = word_count(content)
        fetch_error = article.get("error")

        metadata_payload: dict[str, object] = {
            "source": "reading_save",
            "tags": tags,
            "author": article.get("author"),
            "archive_mode": str(archive_mode or "use_default"),
            "archive_requested": archive_requested,
            "last_fetch_error": fetch_error,
        }
        if article.get("media_type"):
            metadata_payload["media_type"] = article.get("media_type")
        if article.get("content_type"):
            metadata_payload["content_type"] = article.get("content_type")
        if article.get("status_code"):
            metadata_payload["fetch_status"] = article.get("status_code")
        if media_uuid:
            metadata_payload["media_uuid"] = media_uuid
        if metadata:
            metadata_payload.update(metadata)
        _add_bounded_metadata_text(
            metadata_payload,
            "clean_html",
            str(article.get("clean_html") or "") if article.get("clean_html") else None,
            READING_CLEAN_HTML_METADATA_MAX_CHARS,
        )
        _add_bounded_metadata_text(metadata_payload, "text", content, READING_CONTENT_METADATA_MAX_CHARS)
        if content_word_count:
            metadata_payload["reading_time_seconds"] = max(1, int(content_word_count / 200 * 60))
        if fetch_error:
            metadata_payload["fetch_error"] = fetch_error
        else:
            metadata_payload["fetch_error"] = None

        item_row = self.collections.upsert_content_item(
            origin="reading",
            origin_type="manual",
            origin_id=None,
            url=url,
            canonical_url=canonical_url,
            domain=None,
            title=title,
            summary=summary,
            notes=notes,
            content_hash=hash_text_sha256(content),
            word_count=content_word_count,
            published_at=published_at,
            status=normalized_status,
            favorite=favorite,
            metadata=metadata_payload,
            media_id=media_id if isinstance(media_id, int) else None,
            job_id=None,
            run_id=None,
            source_id=None,
            read_at=None,
            tags=tags,
            merge_tags=True,
            preserve_existing_on_null=True,
        )

        archive_output_id: int | None = None
        archive_error: str | None = None
        if archive_requested:
            archive_text = content or summary or notes or ""
            if not archive_text and article.get("clean_html"):
                archive_text = _strip_html(str(article.get("clean_html") or ""))
            try:
                archive_output_id, archive_error = await self._create_archive_artifact(
                    item_id=int(item_row.id),
                    title=str(item_row.title or title or url),
                    url=item_row.canonical_url or item_row.url,
                    body_text=archive_text,
                    media_item_id=item_row.media_id,
                )
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
                archive_error = f"archive_failed:{exc}"
            meta_patch: dict[str, object] = {"archive_requested": True}
            if archive_output_id is not None:
                meta_patch["has_archive_copy"] = True
                meta_patch["archive_output_id"] = archive_output_id
                meta_patch["archive_error"] = None
            elif archive_error:
                meta_patch["archive_error"] = archive_error
            try:
                item_row = self.collections.update_content_item(item_row.id, metadata=meta_patch)
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
                update_error = f"archive_metadata_update_failed:{exc}"
                logger.warning(f"reading_archive_metadata_update_failed item_id={item_row.id}: {exc}")
                archive_error = f"{archive_error};{update_error}" if archive_error else update_error

        if item_row.is_new or item_row.content_changed:
            embedding_metadata = {
                "origin": "reading",
                "item_id": item_row.id,
                "url": url,
                "canonical_url": canonical_url,
                "title": title,
                "author": article.get("author"),
                "tags": tags,
            }
            try:
                await enqueue_embeddings_job_for_item(
                    user_id=self.user_id,
                    item_id=item_row.id,
                    content=content,
                    metadata=embedding_metadata,
                )
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Embedding enqueue failed for reading item {item_row.id}: {exc}")
            try:
                self.collections.reanchor_highlights_for_item(
                    item_row.id,
                    content_text=content,
                    content_hash=item_row.content_hash,
                )
                refreshed = self.collections.get_content_item(item_row.id)
                refreshed.is_new = item_row.is_new
                refreshed.content_changed = item_row.content_changed
                item_row = refreshed
            except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Highlight re-anchoring failed for item {item_row.id}: {exc}")

        return ReadingSaveResult(
            item=item_row,
            media_id=item_row.media_id,
            media_uuid=media_uuid,
            created=item_row.is_new,
            archive_requested=archive_requested,
            archive_output_id=archive_output_id,
            archive_error=archive_error,
        )

    async def _fetch_article(
        self,
        *,
        url: str,
        title_override: str | None,
        content_override: str | None,
        summary_override: str | None,
    ) -> dict[str, Any]:
        """Fetch, normalize, and sanitize content for reading list items."""
        if content_override is not None:
            content, clean_html = self._sanitize_html_content(content_override)
            return {
                "url": url,
                "canonical_url": url,
                "title": title_override or url,
                "content": content,
                "summary": summary_override,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": None,
                "content_type": None,
                "clean_html": clean_html,
            }

        probe = await self._probe_url_metadata(url)
        content_type = probe.get("content_type")
        media_type_key = probe.get("media_type_key")
        resolved_url = probe.get("resolved_url") or url
        if probe.get("size_exceeded"):
            return {
                "url": url,
                "canonical_url": self._normalize_url(resolved_url, url),
                "title": title_override or url,
                "content": "",
                "summary": summary_override,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": media_type_key,
                "content_type": content_type,
                "error": probe.get("error"),
                "status_code": probe.get("status_code"),
            }

        if media_type_key and str(media_type_key).lower() != "html":
            non_html = await self._ingest_non_html(
                url=url,
                media_type_key=media_type_key,
                title_override=title_override,
            )
            non_html["canonical_url"] = self._normalize_url(resolved_url, url)
            non_html["content_type"] = content_type
            non_html["status_code"] = probe.get("status_code")
            if probe.get("error") and not non_html.get("error"):
                non_html["error"] = probe.get("error")
            return non_html

        try:
            from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
            from tldw_Server_API.app.core.Web_Scraping.orchestration import scrape_article
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            return {
                "url": url,
                "canonical_url": self._normalize_url(resolved_url, url),
                "title": title_override or url,
                "content": "",
                "summary": summary_override,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": "html",
                "content_type": content_type,
                "error": f"scrape_import_failed:{exc}",
                "status_code": probe.get("status_code"),
            }

        try:
            data = await scrape_article(url)
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
            return {
                "url": url,
                "canonical_url": self._normalize_url(resolved_url, url),
                "title": title_override or url,
                "content": "",
                "summary": summary_override,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": "html",
                "content_type": content_type,
                "error": f"article_fetch_failed:{exc}",
                "status_code": probe.get("status_code"),
            }

        if not data or not data.get("extraction_successful"):
            error = data.get("error") if isinstance(data, dict) else "article_fetch_failed"
            return {
                "url": url,
                "canonical_url": self._normalize_url(resolved_url, url),
                "title": title_override or url,
                "content": "",
                "summary": summary_override,
                "author": None,
                "published": None,
                "media_id": None,
                "media_uuid": None,
                "media_type": "html",
                "content_type": content_type,
                "error": error,
                "status_code": probe.get("status_code"),
            }

        content = data.get("content") or ""
        try:
            content = ContentMetadataHandler.strip_metadata(content)  # type: ignore[attr-defined]
        except _READING_SERVICE_NONCRITICAL_EXCEPTIONS:
            pass
        content, clean_html = self._sanitize_html_content(content)
        canonical_raw = data.get("canonical_url") or data.get("url") or resolved_url or url

        return {
            "url": data.get("url") or url,
            "canonical_url": self._normalize_url(canonical_raw, url),
            "title": data.get("title") or title_override or url,
            "content": content,
            "summary": data.get("summary"),
            "author": data.get("author"),
            "published": data.get("date") or data.get("published"),
            "media_id": None,
            "media_uuid": None,
            "media_type": "html",
            "content_type": content_type,
            "clean_html": clean_html,
            "status_code": probe.get("status_code"),
        }

    def list_items(
        self,
        *,
        status: list[str] | None = None,
        tags: list[str] | None = None,
        favorite: bool | None = None,
        q: str | None = None,
        domain: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        size: int = 20,
        offset: int | None = None,
        limit: int | None = None,
        sort: str | None = None,
    ) -> tuple[list[ContentItemRow], int]:
        return self.collections.list_content_items(
            origin="reading",
            status=status,
            tags=tags,
            favorite=favorite,
            q=q,
            domain=domain,
            date_from=date_from,
            date_to=date_to,
            page=page,
            size=size,
            offset=offset,
            limit=limit,
            sort=sort,
        )

    def get_item(self, item_id: int) -> ContentItemRow:
        return self.collections.get_content_item(item_id)

    def update_item(
        self,
        item_id: int,
        *,
        status: str | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        title: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ContentItemRow:
        normalized_tags = tags if tags is None else [t for t in tags if t]
        normalized_status = normalize_reading_status(status) if status is not None else None
        metadata = metadata or {}
        read_at: str | None = None
        clear_read_at = False
        if status is not None:
            try:
                current = self.collections.get_content_item(item_id)
            except KeyError:
                raise
            if normalized_status == "read":
                if not current.read_at:
                    read_at = _utcnow_iso()
            else:
                if current.read_at:
                    clear_read_at = True

        return self.collections.update_content_item(
            item_id,
            status=normalized_status,
            favorite=favorite,
            tags=normalized_tags,
            notes=notes,
            metadata=metadata if metadata else None,
            title=title,
            read_at=read_at,
            clear_read_at=clear_read_at,
        )

    def delete_item(self, item_id: int) -> None:
        self.collections.delete_content_item(item_id)

    def import_items(
        self,
        *,
        items: list[ReadingImportItem],
        merge_tags: bool = True,
        origin_type: str = "import",
    ) -> ReadingImportResult:
        return self._import_service.import_items(
            items=items,
            merge_tags=merge_tags,
            origin_type=origin_type,
        )
