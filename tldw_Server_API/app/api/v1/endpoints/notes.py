# app/api/v1/endpoints/notes.py
#
#
# Imports
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import quote
from uuid import uuid4

#
# 3rd-party Libraries
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Header,  # Keep Header for expected_version
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep, get_request_user, RateLimiter, rbac_rate_limit, User

from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta

# Dependency to get user-specific ChaChaNotes_DB instance
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
    resolve_chacha_user_base_dir,
)

#
# Schemas for notes
from tldw_Server_API.app.api.v1.schemas.notes_schemas import (
    CollectionKeywordLinkResponse,
    CollectionKeywordLinksResponse,
    ConversationKeywordLinkResponse,
    ConversationKeywordLinksResponse,
    DetailResponse,
    KeywordCreate,
    KeywordCollectionCreate,
    KeywordCollectionResponse,
    KeywordCollectionsListResponse,
    KeywordCollectionUpdate,
    KeywordMergeRequest,
    KeywordMergeResponse,
    KeywordResponse,
    KeywordUpdate,
    KeywordsForNoteResponse,
    NoteBulkCreateItemResult,
    NoteBulkCreateRequest,
    NoteBulkCreateResponse,
    NoteAttachmentsListResponse,
    NoteAttachmentResponse,
    NoteCreate,
    NoteFolderCreate,
    NoteFolderResponse,
    NoteFoldersListResponse,
    NoteKeywordLinkResponse,
    NoteResponse,
    NotesExportRequest,
    NotesExportResponse,
    NotesImportRequest,
    NotesImportResponse,
    NotesImportFileResult,
    NotesForKeywordResponse,
    NotesListResponse,
    NoteUpdate,
    TitleSuggestRequest,
    TitleSuggestResponse,
)
from tldw_Server_API.app.api.v1.schemas.notes_studio import (
    NoteStudioDeriveRequest,
    NoteStudioDiagramRequest,
    NoteStudioDiagramResponse,
    NoteStudioRegenerateRequest,
    NoteStudioStateResponse,
)
from tldw_Server_API.app.api.v1.schemas.notes_moodboards import (
    MoodboardCreate,
    MoodboardListResponse,
    MoodboardNotesListResponse,
    MoodboardPinResponse,
    MoodboardResponse,
    MoodboardUpdate,
)
from tldw_Server_API.app.core.config import settings as core_settings

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (  # Corrected import path if needed
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename
from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service
from tldw_Server_API.app.core.Notes.studio_service import NotesStudioService
from tldw_Server_API.app.core.Notes_Tasks import NotesTaskService, TaskActor
from tldw_Server_API.app.core.Personalization import (
    build_note_bulk_import_activity,
    record_companion_activity_events_bulk,
    record_note_created,
    record_note_deleted,
    record_note_restored,
    record_note_updated,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    SyncServerOriginIdempotencyConflictError,
    SyncServerOriginMaterializationError,
    SyncServerOriginMutationNotSupportedError,
    capture_server_origin_mutation,
    get_active_server_origin_sync_service_for_user,
    server_origin_object_id,
    server_origin_stable_key,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Writing.note_title import TitleGenOptions, generate_note_title

#
#
#######################################################################################################################
#
# Functions:

_NOTES_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    HTTPException,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

router = APIRouter()

_NOTES_ATTACHMENTS_DIRNAME = "notes_attachments"
_NOTES_ATTACHMENT_META_SUFFIX = ".meta.json"
_NOTES_ATTACHMENT_MAX_FILENAME_LEN = 180
_NOTES_ATTACHMENT_DEFAULT_MAX_BYTES = 25 * 1024 * 1024
_NOTES_ATTACHMENT_ALLOWED_EXTENSIONS = {
    ".bmp",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".mp3",
    ".mp4",
    ".m4a",
    ".mov",
    ".ogg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".tar.gz",
    ".txt",
    ".wav",
    ".webm",
    ".webp",
    ".xlsx",
    ".xls",
    ".yaml",
    ".yml",
    ".zip",
}
def get_notes_task_service() -> NotesTaskService:
    """Provide the stateless notes task service for note-save reconciliation."""
    return NotesTaskService()


def _resolve_notes_attachment_max_bytes() -> int:
    try:
        value = int(core_settings.get("NOTES_ATTACHMENT_MAX_BYTES", _NOTES_ATTACHMENT_DEFAULT_MAX_BYTES))
        if value <= 0:
            return _NOTES_ATTACHMENT_DEFAULT_MAX_BYTES
        return value
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        return _NOTES_ATTACHMENT_DEFAULT_MAX_BYTES


_NOTES_ATTACHMENT_MAX_BYTES = _resolve_notes_attachment_max_bytes()

NoteStudioStateResponse.model_rebuild(_types_namespace={"NoteResponse": NoteResponse})
NoteStudioDiagramResponse.model_rebuild(_types_namespace={"NoteResponse": NoteResponse})


def _ensure_note_exists_or_404(db: CharactersRAGDB, note_id: str) -> None:
    note_data = db.get_note_by_id(note_id=note_id)
    if not note_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")


def _note_sync_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SyncServerOriginIdempotencyConflictError):
        envelope = exc.envelope
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "sync_server_origin_idempotency_conflict",
                "message": "The idempotency key was already used for a different note change.",
                "server_cursor": envelope.server_cursor,
                "apply_status": envelope.apply_status,
            },
        )
    if isinstance(exc, SyncServerOriginMaterializationError):
        envelope = exc.envelope
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "sync_server_origin_materialization_failed",
                "message": "Sync accepted the server-origin note change but projection apply failed.",
                "server_cursor": envelope.server_cursor,
                "apply_status": envelope.apply_status,
                "apply_error_code": envelope.apply_error_code,
                "apply_error_message": envelope.apply_error_message,
            },
        )
    if isinstance(exc, SyncServerOriginMutationNotSupportedError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                "dataset_id": exc.dataset.dataset_id,
                "domain": exc.domain,
            },
        )
    if isinstance(exc, SyncStoreError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "sync_server_origin_append_failed",
                "message": "Sync could not record the server-origin note change.",
            },
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error_code": "sync_server_origin_failed",
            "message": "Sync failed while recording the server-origin note change.",
        },
    )


def _note_keywords_sync_unsupported_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error_code": "sync_v2_keywords_not_supported",
            "message": "Sync v2 M1 does not support note keyword mutations.",
        },
    )


def _note_restore_sync_unsupported_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error_code": "sync_v2_note_restore_not_supported",
            "message": "Sync v2 M1 does not support restoring notes through this endpoint.",
        },
    )


def _note_payload_from_row(note: dict[str, Any]) -> dict[str, object]:
    return {
        "title": str(note.get("title") or ""),
        "content": str(note.get("content") or ""),
        "conversation_id": note.get("conversation_id"),
        "message_id": note.get("message_id"),
        "client_id": note.get("client_id"),
        "owner_user_id": note.get("owner_user_id") or note.get("client_id"),
    }


def _active_notes_sync_service(current_user: User) -> SyncV2Service | None:
    return get_active_server_origin_sync_service_for_user(str(current_user.id))


def _reconcile_note_tasks_after_save(
    *,
    db: CharactersRAGDB,
    note_data: dict[str, Any],
    current_user: User,
    task_service: NotesTaskService,
) -> None:
    """Synchronize markdown checklist tasks for a note row that was just saved."""
    note_id = str(note_data["id"])
    note_version = int(note_data["version"])
    try:
        task_service.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=note_version,
            content=str(note_data.get("content") or ""),
            actor=TaskActor(actor_type="user", actor_id=str(current_user.id)),
        )
    except Exception as exc:
        logger.warning(
            "Note task reconciliation failed after saved note {} version {}: {}",
            note_id,
            note_version,
            exc,
        )


def _safe_note_attachment_dirname(note_id: str) -> str:
    text = str(note_id or "").strip()
    if not text:
        return "note"
    safe = sanitize_filename(text, max_total_length=96).replace(" ", "_").strip("._")
    if safe and safe not in {".", ".."}:
        return safe
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"note_{digest}"


def _get_note_attachments_base_dir(user_id: int | str) -> Path:
    user_base_dir = DatabasePaths.get_user_base_directory(user_id)
    base_dir = (user_base_dir / _NOTES_ATTACHMENTS_DIRNAME).resolve()
    user_base_resolved = user_base_dir.resolve()
    try:
        base_dir.relative_to(user_base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid attachment storage path") from exc
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _get_note_attachments_dir(user_id: int | str, note_id: str, *, create: bool = False) -> Path:
    base_dir = _get_note_attachments_base_dir(user_id)
    note_dir = (base_dir / _safe_note_attachment_dirname(note_id)).resolve()
    try:
        note_dir.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid note attachment path") from exc
    if create:
        note_dir.mkdir(parents=True, exist_ok=True)
    return note_dir


def _sanitize_attachment_file_name(raw_name: str) -> str:
    input_name = str(raw_name or "").strip()
    if not input_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attachment filename is required")
    basename = Path(input_name).name
    if basename != input_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment filename")

    suffixes = [suffix.lower() for suffix in Path(basename).suffixes]
    extension = ""
    full_extension = "".join(suffixes)
    if full_extension and full_extension in _NOTES_ATTACHMENT_ALLOWED_EXTENSIONS:
        extension = full_extension
    elif suffixes and suffixes[-1] in _NOTES_ATTACHMENT_ALLOWED_EXTENSIONS:
        extension = suffixes[-1]
    if not extension:
        allowed = ", ".join(sorted(_NOTES_ATTACHMENT_ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported attachment type. Allowed extensions: {allowed}",
        )

    stem = basename[:-len(extension)] if len(extension) < len(basename) else "attachment"
    max_stem_len = max(1, _NOTES_ATTACHMENT_MAX_FILENAME_LEN - len(extension))
    safe_stem = sanitize_filename(stem, max_total_length=max_stem_len).replace(" ", "_").strip("._")
    if not safe_stem:
        safe_stem = "attachment"
    if len(safe_stem) > max_stem_len:
        safe_stem = safe_stem[:max_stem_len]
    return f"{safe_stem}{extension}"


def _resolve_unique_attachment_path(note_dir: Path, file_name: str) -> Path:
    candidate = (note_dir / file_name).resolve()
    try:
        candidate.relative_to(note_dir)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment filename") from exc
    if not candidate.exists():
        return candidate

    suffixes = Path(file_name).suffixes
    extension = "".join(suffixes) if suffixes else ""
    stem = file_name[:-len(extension)] if extension else file_name
    for idx in range(1, 1000):
        suffix = f"-{idx}"
        max_stem_len = max(1, _NOTES_ATTACHMENT_MAX_FILENAME_LEN - len(extension) - len(suffix))
        trimmed_stem = stem[:max_stem_len]
        next_name = f"{trimmed_stem}{suffix}{extension}"
        next_candidate = (note_dir / next_name).resolve()
        try:
            next_candidate.relative_to(note_dir)
        except ValueError:
            continue
        if not next_candidate.exists():
            return next_candidate
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Unable to allocate unique attachment filename")


def _attachment_metadata_path(file_path: Path) -> Path:
    return file_path.with_name(f"{file_path.name}{_NOTES_ATTACHMENT_META_SUFFIX}")


def _parse_uploaded_at(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text:
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(text)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except _NOTES_NONCRITICAL_EXCEPTIONS:
                return fallback
    return fallback


def _read_attachment_metadata(file_path: Path) -> dict[str, Any]:
    metadata_path = _attachment_metadata_path(file_path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_attachment_metadata(
    file_path: Path,
    *,
    original_file_name: str,
    content_type: Optional[str],
    size_bytes: int,
    uploaded_at: datetime,
) -> None:
    metadata_payload = {
        "original_file_name": original_file_name,
        "content_type": content_type,
        "size_bytes": int(size_bytes),
        "uploaded_at": uploaded_at.astimezone(timezone.utc).isoformat(),
    }
    metadata_path = _attachment_metadata_path(file_path)
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False), encoding="utf-8")


def _to_attachment_response(note_id: str, file_path: Path) -> dict[str, Any]:
    file_stat = file_path.stat()
    uploaded_fallback = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)
    metadata = _read_attachment_metadata(file_path)
    original_file_name = str(metadata.get("original_file_name") or file_path.name)
    content_type = metadata.get("content_type")
    if content_type is None:
        content_type = mimetypes.guess_type(file_path.name)[0]
    uploaded_at = _parse_uploaded_at(metadata.get("uploaded_at"), uploaded_fallback)
    size_bytes = metadata.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        size_bytes = int(file_stat.st_size)
    encoded_note_id = quote(str(note_id), safe="")
    encoded_file_name = quote(file_path.name, safe="")
    return {
        "file_name": file_path.name,
        "original_file_name": original_file_name,
        "content_type": content_type,
        "size_bytes": int(size_bytes),
        "uploaded_at": uploaded_at,
        "url": f"/api/v1/notes/{encoded_note_id}/attachments/{encoded_file_name}",
    }

# --- Title options helper -----------------------------------------------------
def _field_supplied(model_obj: Any, field_name: str) -> bool:
    """Return True if the incoming model explicitly supplied the field.

    Works across Pydantic v2 (model_fields_set) and v1 (__fields_set__).
    Falls back to checking a best-effort dump with exclude_unset.
    """
    try:
        s = getattr(model_obj, "model_fields_set", None)
        if isinstance(s, set):
            return field_name in s
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        s = getattr(model_obj, "__fields_set__", None)  # pydantic v1
        if isinstance(s, set):
            return field_name in s
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat as _dump
        data = _dump(model_obj, exclude_unset=True)
        return field_name in (data or {})
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        return False


def _build_title_opts(note_in: Any) -> TitleGenOptions:
    """Build TitleGenOptions from request payload with sane defaults and clamping.

    - Strategy: use client-provided value if supplied; otherwise fall back to default setting.
    - LLM gating: downgrade to heuristic when LLM strategies are disabled.
    - Max length: coerce to int, default 250, clamp to [min_len, max_len_bound].
    """
    # Resolve strategy honoring client intent when provided
    default_strategy = str(core_settings.get("NOTES_TITLE_DEFAULT_STRATEGY", "heuristic")).lower()
    if _field_supplied(note_in, "title_strategy"):
        strategy = getattr(note_in, "title_strategy", default_strategy) or default_strategy
    else:
        strategy = default_strategy

    # Apply LLM enabled gate after resolving strategy
    if strategy in ("llm", "llm_fallback") and not bool(core_settings.get("NOTES_TITLE_LLM_ENABLED", False)):
        strategy = "heuristic"

    # Resolve and clamp max length
    try:
        raw_len = getattr(note_in, "title_max_len", None)
        max_len_val = int(raw_len) if raw_len is not None else 250
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        max_len_val = 250
    try:
        max_bound = int(core_settings.get("NOTES_TITLE_MAX_LEN", 1000))
        if max_bound <= 0:
            max_bound = 1000
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        max_bound = 1000
    min_bound = 10
    # Clamp to API schema max for titles (NoteBase.title max_length=255).
    schema_max = 255
    max_bound = min(max_bound, schema_max)
    if max_bound < min_bound:
        max_bound = min_bound
    if max_len_val < min_bound:
        max_len_val = min_bound
    if max_len_val > max_bound:
        max_len_val = max_bound

    opts = TitleGenOptions()
    opts.strategy = strategy
    opts.max_len = max_len_val
    try:
        opts.language = getattr(note_in, "language", None)
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        opts.language = None
    return opts

# --- Note link validation -----------------------------------------------------
def _normalize_optional_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _validate_note_links(
    db: CharactersRAGDB,
    conversation_id: Optional[str],
    message_id: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    normalized_conversation_id = _normalize_optional_id(conversation_id)
    normalized_message_id = _normalize_optional_id(message_id)

    if normalized_conversation_id:
        conv = db.get_conversation_by_id(normalized_conversation_id)
        if not conv:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    message_conversation_id = None
    if normalized_message_id:
        message_conversation_id = db.get_message_conversation_id(normalized_message_id)
        if not message_conversation_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    if normalized_conversation_id and normalized_message_id:
        if message_conversation_id != normalized_conversation_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found in conversation",
            )

    return normalized_conversation_id, normalized_message_id


# --- CSV export helper --------------------------------------------------------
def _notes_csv_response(notes_data: list[dict[str, Any]], include_keywords: bool) -> StreamingResponse:
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    headers = ["id", "title", "content", "created_at", "last_modified", "version", "client_id"]
    if include_keywords:
        headers.append("keywords")
    writer.writerow(headers)
    for n in notes_data:
        row = [
            n.get("id"),
            n.get("title"),
            n.get("content"),
            n.get("created_at"),
            n.get("last_modified") or n.get("updated_at"),
            n.get("version"),
            n.get("client_id"),
        ]
        if include_keywords:
            kws = n.get("keywords") or []
            row.append(",".join([str(k.get("keyword")) for k in kws if isinstance(k, dict) and k.get("keyword") is not None]))
        writer.writerow(row)
    output.seek(0)
    from datetime import datetime as _dt
    from datetime import timezone
    headers_map = {"Content-Disposition": f"attachment; filename=notes_export_{_dt.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"}
    return StreamingResponse(output, media_type="text/csv; charset=utf-8", headers=headers_map)


def _parse_keyword_tokens_inline(value: str) -> list[str]:
    raw = value.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(","):
        token = part.strip().strip('"').strip("'")
        if token:
            out.append(token)
    return out


def _normalize_import_keywords(raw_keywords: Any) -> list[str]:
    if raw_keywords is None:
        return []
    tokens: list[str] = []
    if isinstance(raw_keywords, str):
        tokens.extend(_parse_keyword_tokens_inline(raw_keywords))
    elif isinstance(raw_keywords, list):
        for item in raw_keywords:
            text = _keyword_text_from_row(item)
            if text:
                tokens.append(text)
    elif isinstance(raw_keywords, dict):
        maybe = raw_keywords.get("keywords")
        if maybe is not None:
            return _normalize_import_keywords(maybe)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


def _fallback_title_from_filename(file_name: str | None) -> str | None:
    if not file_name:
        return None
    stem = Path(file_name).stem.strip()
    if not stem:
        return None
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem[:255] if stem else None


def _extract_json_note_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("notes", "data", "items", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
        if any(key in payload for key in ("title", "content", "id", "metadata", "keywords")):
            return [payload]
    raise InputError(  # noqa: TRY003
        "JSON import must be a note object, a note array, or an export wrapper containing notes/data/items/results."
    )


def _normalize_import_note_row(row: dict[str, Any], fallback_title: str | None = None) -> dict[str, Any]:
    raw_id = row.get("id")
    note_id = str(raw_id).strip() if raw_id is not None else None
    if note_id == "":
        note_id = None

    title = _normalize_keyword_text(row.get("title"))
    raw_content = row.get("content")
    content = "" if raw_content is None else str(raw_content)

    metadata = row.get("metadata")
    metadata_keywords = metadata.get("keywords") if isinstance(metadata, dict) else None
    keywords_provided = ("keywords" in row) or (metadata_keywords is not None)
    keywords = _normalize_import_keywords(row.get("keywords", metadata_keywords))

    if not title:
        title = _normalize_keyword_text(fallback_title)
    if not title:
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        title = first_line[:255] if first_line else None
    if not title:
        raise InputError("Imported note is missing a title and usable content.")  # noqa: TRY003
    if not content:
        content = title

    return {
        "id": note_id,
        "title": title[:255],
        "content": content,
        "keywords": keywords,
        "keywords_provided": keywords_provided,
    }


def _extract_markdown_frontmatter_keywords(text: str) -> tuple[list[str], str]:
    normalized = text.replace("\r\n", "\n").lstrip("\ufeff")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        return [], normalized

    end_index = None
    for idx in range(1, min(len(lines), 2000)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        return [], normalized

    keywords: list[str] = []
    frontmatter_lines = lines[1:end_index]
    idx = 0
    while idx < len(frontmatter_lines):
        stripped = frontmatter_lines[idx].strip()
        lower = stripped.lower()
        if lower.startswith("tags:") or lower.startswith("keywords:"):
            _, _, value = stripped.partition(":")
            inline_value = value.strip()
            if inline_value:
                keywords.extend(_parse_keyword_tokens_inline(inline_value))
            else:
                idx += 1
                while idx < len(frontmatter_lines):
                    nested = frontmatter_lines[idx].strip()
                    if not nested.startswith("-"):
                        idx -= 1
                        break
                    token = nested[1:].strip().strip('"').strip("'")
                    if token:
                        keywords.append(token)
                    idx += 1
        idx += 1

    deduped = _normalize_import_keywords(keywords)
    body = "\n".join(lines[end_index + 1 :])
    return deduped, body


def _parse_markdown_import_note(content: str, fallback_title: str | None = None) -> dict[str, Any]:
    keywords, body = _extract_markdown_frontmatter_keywords(content)
    lines = body.replace("\r\n", "\n").split("\n")

    title: str | None = None
    content_lines = lines
    first_non_empty_index = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        first_non_empty_index = idx
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            content_lines = lines[idx + 1 :]
        break

    if not title:
        title = _normalize_keyword_text(fallback_title)
    if not title and first_non_empty_index is not None:
        title = lines[first_non_empty_index].strip()[:255]
        content_lines = lines[first_non_empty_index + 1 :]
    if not title:
        title = "Imported note"

    note_content = "\n".join(content_lines).strip("\n")
    if not note_content:
        note_content = title

    return {
        "id": None,
        "title": title[:255],
        "content": note_content,
        "keywords": keywords,
        "keywords_provided": len(keywords) > 0,
    }


# --- Keyword attach helper ----------------------------------------------------
def _attach_keywords_bulk(db: CharactersRAGDB, notes_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    note_ids = [nd.get("id") for nd in notes_data if isinstance(nd, dict) and nd.get("id")]
    if not note_ids:
        return notes_data
    try:
        kw_map = db.get_keywords_for_notes(note_ids)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Bulk keyword lookup failed: {e}")
        return notes_data
    for nd in notes_data:
        if isinstance(nd, dict):
            nid = nd.get("id")
            if nid:
                nd["keywords"] = kw_map.get(nid, [])
    return notes_data


def _attach_folders_bulk(db: CharactersRAGDB, notes_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    note_ids = [nd.get("id") for nd in notes_data if isinstance(nd, dict) and nd.get("id")]
    if not note_ids:
        return notes_data
    try:
        folder_map = db.get_note_folders_for_notes(note_ids)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Bulk folder lookup failed: {e}")
        return notes_data
    for nd in notes_data:
        if isinstance(nd, dict):
            nid = nd.get("id")
            if nid:
                nd["folders"] = folder_map.get(nid, [])
    return notes_data


# --- Keyword attach helper ----------------------------------------------------
def _attach_keywords_inline(db: CharactersRAGDB, note_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        if note_dict and note_dict.get('id'):
            note_dict['keywords'] = db.get_keywords_for_note(note_id=note_dict['id'])
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to attach keywords to note {note_dict.get('id')}: {e}")
    return note_dict


def _attach_folders_inline(db: CharactersRAGDB, note_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        if note_dict and note_dict.get("id"):
            note_dict["folders"] = db.get_note_folders_for_note(note_id=note_dict["id"])
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to attach folders to note {note_dict.get('id')}: {e}")
    return note_dict


def _normalize_keyword_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _keyword_text_from_row(row: Any) -> Optional[str]:
    if isinstance(row, dict):
        for key in ("keyword", "keyword_text", "text"):
            if key in row:
                return _normalize_keyword_text(row.get(key))
    return _normalize_keyword_text(row)


def _get_or_create_keyword_row(db: CharactersRAGDB, keyword_text: Any) -> Optional[dict[str, Any]]:
    """Return existing keyword row or create one, handling concurrent creation."""
    text = _normalize_keyword_text(keyword_text)
    if not text:
        return None
    kw_row = db.get_keyword_by_text(text)
    if kw_row:
        return kw_row
    try:
        kw_id = db.add_keyword(text)
    except ConflictError:
        # Keyword may have been created concurrently; refetch and return if present.
        kw_row = db.get_keyword_by_text(text)
        if kw_row:
            return kw_row
        raise
    if kw_id is None:
        return None
    return db.get_keyword_by_id(kw_id)


def _sync_note_keywords(db: CharactersRAGDB, note_id: str, keywords: list[str]) -> dict[str, Any]:
    """Sync note keywords and return a summary of attachment failures.

    The note save itself should succeed even if one or more keyword operations fail.
    This summary allows clients to surface a partial-success warning.
    """
    desired: dict[str, str] = {}
    for kw in keywords:
        text = _normalize_keyword_text(kw)
        if not text:
            continue
        key = text.lower()
        if key in desired:
            continue
        desired[key] = text

    try:
        existing_rows = db.get_keywords_for_note(note_id=note_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as err:
        logger.warning(f"Keyword sync lookup failed for note {note_id}: {err}")
        existing_rows = []

    existing_by_key: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        text = _keyword_text_from_row(row)
        if not text:
            continue
        existing_by_key[text.lower()] = row

    desired_keys = set(desired.keys())
    failed_keywords: list[str] = []
    attached_count = 0

    for key, row in existing_by_key.items():
        if key in desired_keys:
            attached_count += 1
            continue
        kw_id = row.get("id") if isinstance(row, dict) else None
        if kw_id is None:
            continue
        try:
            db.unlink_note_from_keyword(note_id=note_id, keyword_id=int(kw_id))
        except _NOTES_NONCRITICAL_EXCEPTIONS as err:
            logger.warning(f"Keyword unlink failed for note {note_id}, keyword {kw_id}: {err}")

    for key, text in desired.items():
        if key in existing_by_key:
            continue
        try:
            kw_row = _get_or_create_keyword_row(db, text)
            if kw_row and kw_row.get("id") is not None:
                db.link_note_to_keyword(note_id=note_id, keyword_id=int(kw_row["id"]))
                attached_count += 1
            else:
                failed_keywords.append(text)
        except _NOTES_NONCRITICAL_EXCEPTIONS as err:
            failed_keywords.append(text)
            logger.warning(f"Keyword attach failed for '{text}' on note {note_id}: {err}")

    return {
        "failed_count": len(failed_keywords),
        "failed_keywords": failed_keywords,
        "requested_count": len(desired),
        "attached_count": attached_count,
    }


def _build_import_note_companion_event(
    *,
    db: CharactersRAGDB,
    note_id: str | int,
    operation: str,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reload an imported note and build a stable companion activity payload."""
    reloaded_note = db.get_note_by_id(str(note_id))
    if not reloaded_note:
        raise CharactersRAGDBError(
            f"Imported {str(operation).replace('_', ' ')} note could not be reloaded."
        )
    reloaded_note = _attach_keywords_inline(db, reloaded_note)
    reloaded_note = _attach_folders_inline(db, reloaded_note)
    return build_note_bulk_import_activity(
        note=reloaded_note,
        operation=operation,
        route="/api/v1/notes/import",
        surface="api.notes.import",
        patch=patch,
    )


def _normalize_keyword_tokens(tokens: Optional[list[str]]) -> list[str]:
    if not tokens:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token is None:
            continue
        text = str(token).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _attach_collection_keywords_inline(
    db: CharactersRAGDB,
    collection_dict: dict[str, Any],
) -> dict[str, Any]:
    try:
        collection_id_raw = collection_dict.get("id")
        if collection_id_raw is None:
            return collection_dict
        collection_id = int(collection_id_raw)
        collection_dict["keywords"] = db.get_keywords_for_collection(collection_id=collection_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as err:
        logger.warning(
            "Failed to attach keywords for collection {}: {}",
            collection_dict.get("id"),
            err,
        )
    return collection_dict


def _sync_collection_keywords(
    db: CharactersRAGDB,
    collection_id: int,
    keywords: list[str],
) -> dict[str, Any]:
    """Sync keyword links for a collection using desired keyword texts."""
    desired: dict[str, str] = {}
    for kw in keywords:
        text = _normalize_keyword_text(kw)
        if not text:
            continue
        key = text.lower()
        if key not in desired:
            desired[key] = text

    try:
        existing_rows = db.get_keywords_for_collection(collection_id=collection_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as err:
        logger.warning("Collection keyword lookup failed for {}: {}", collection_id, err)
        existing_rows = []

    existing_by_key: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        text = _keyword_text_from_row(row)
        if not text:
            continue
        existing_by_key[text.lower()] = row

    desired_keys = set(desired.keys())
    failed_keywords: list[str] = []
    linked_count = 0

    for key, row in existing_by_key.items():
        if key in desired_keys:
            linked_count += 1
            continue
        kw_id = row.get("id") if isinstance(row, dict) else None
        if kw_id is None:
            continue
        try:
            db.unlink_collection_from_keyword(collection_id=collection_id, keyword_id=int(kw_id))
        except _NOTES_NONCRITICAL_EXCEPTIONS as err:
            logger.warning(
                "Collection keyword unlink failed for collection {}, keyword {}: {}",
                collection_id,
                kw_id,
                err,
            )

    for key, text in desired.items():
        if key in existing_by_key:
            continue
        try:
            kw_row = _get_or_create_keyword_row(db, text)
            if kw_row and kw_row.get("id") is not None:
                db.link_collection_to_keyword(collection_id=collection_id, keyword_id=int(kw_row["id"]))
                linked_count += 1
            else:
                failed_keywords.append(text)
        except _NOTES_NONCRITICAL_EXCEPTIONS as err:
            failed_keywords.append(text)
            logger.warning(
                "Collection keyword link failed for collection {}, keyword '{}': {}",
                collection_id,
                text,
                err,
            )

    return {
        "failed_count": len(failed_keywords),
        "failed_keywords": failed_keywords,
        "requested_count": len(desired),
        "linked_count": linked_count,
    }


# --- Helper for Exception Handling (largely the same) ---
def handle_db_errors(e: Exception, entity_type: str = "resource"):
    if isinstance(e, HTTPException):  # If it's already an HTTPException, re-raise
        raise e

    logger_func = logger.warning  # Default to warning for known DB operational errors
    http_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR  # Default
    detail_message = f"An unexpected error occurred while processing your request for {entity_type}."

    if isinstance(e, InputError):
        http_status_code = status.HTTP_400_BAD_REQUEST
        detail_message = str(e)
    elif isinstance(e, ConflictError):
        http_status_code = status.HTTP_409_CONFLICT
        # Prioritize version mismatch and not-found semantics
        exception_message_str = str(e.args[0]) if e.args else str(e)  # Get the primary message
        lowered_msg = exception_message_str.lower()
        if "version mismatch" in lowered_msg:
            detail_message = "The resource has been modified since you last fetched it. Please refresh and try again."
        elif "not found" in lowered_msg or "soft-deleted" in lowered_msg or "soft deleted" in lowered_msg:
            http_status_code = status.HTTP_404_NOT_FOUND
            if "conversation" in lowered_msg or "message" in lowered_msg:
                detail_message = exception_message_str
            else:
                resource_name = entity_type or "resource"
                detail_message = f"{resource_name.capitalize()} not found."
        elif hasattr(e, 'entity') and e.entity and hasattr(e, 'entity_id') and e.entity_id:
            detail_message = f"A conflict occurred with {e.entity} (ID: {e.entity_id}). It might have been modified or deleted, or a unique constraint was violated."
        elif "already exists" in lowered_msg:
            detail_message = f"A {entity_type} with the provided identifier already exists."
        else:  # Generic conflict based on the exception's original message
            detail_message = exception_message_str
    elif isinstance(e, CharactersRAGDBError):  # General DB Error from our library
        logger_func = logger.error  # Log as error
        detail_message = f"A database error occurred while processing your request for {entity_type}."
    elif isinstance(e, ValueError):  # Catch generic ValueErrors that might not be InputError
        http_status_code = status.HTTP_400_BAD_REQUEST
        detail_message = str(e)
    else:  # Truly unexpected errors
        logger_func = logger.error

    logger_func(f"Error for {entity_type}: {type(e).__name__} - {str(e)}",
                exc_info=isinstance(e, (CharactersRAGDBError, Exception)) and not isinstance(e,
                                                                                             (InputError, ConflictError,
                                                                                              ValueError)))
    raise HTTPException(status_code=http_status_code, detail=detail_message)


_T = TypeVar("_T")


async def _run_db_call(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run synchronous Notes DB calls off the event loop thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


def _normalize_moodboard_payload(moodboard_row: dict[str, Any]) -> dict[str, Any]:
    """Ensure moodboard payloads expose stable optional fields for API responses."""
    payload = dict(moodboard_row)
    payload.setdefault("smart_rule", None)
    return payload


def _build_moodboard_update_data(moodboard_in: MoodboardUpdate) -> dict[str, Any]:
    """Serialize moodboard update inputs into DB-ready partial update fields."""
    data = moodboard_in.model_dump(exclude_unset=True, mode="json")
    if "smart_rule" in data and data["smart_rule"] is None:
        data["smart_rule"] = None
    return data


# --- Notes Endpoints ---

@router.get(
    "/health",
    summary="Notes service health",
    tags=["notes"],
    openapi_extra={"security": []},
)
async def notes_health() -> dict[str, Any]:
    """Unauthenticated health endpoint for Notes storage."""
    import os
    base_dir: Optional[Path] = None
    health = {
        "service": "notes",
        "status": "healthy",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "components": {}
    }
    storage_info: dict[str, Any] = {
        "base_dir": None,
        "db_path": None,
        "exists": False,
        "writable": False,
    }

    try:
        base_dir = resolve_chacha_user_base_dir()
        exists = base_dir.exists()
        writable = False
        if exists:
            try:
                test_path = base_dir / ".health_check"
                with open(test_path, "w") as f:
                    f.write("ok")
                os.remove(test_path)
                writable = True
            except _NOTES_NONCRITICAL_EXCEPTIONS:
                writable = False

        storage_info.update(
            {
                "base_dir": str(base_dir),
                "db_path": None,
                "exists": exists,
                "writable": writable,
            }
        )

        if not exists or not writable:
            health["status"] = "degraded"
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        health["status"] = "unhealthy"
        health["error"] = "Notes health check failed"
        if base_dir:
            storage_info["base_dir"] = str(base_dir)

    health["components"]["storage"] = storage_info
    return health

@router.post(
    "/",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new note",
    tags=["notes"]
)
async def create_note(
        request: Request,
        note_in: NoteCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),  # Use the user-specific DB instance
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        task_service: NotesTaskService = Depends(get_notes_task_service),
        _: None = Depends(rbac_rate_limit("notes.create")),
):
    try:
        # Centralized rate limit for notes.create
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.create")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.create",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})

        # The user context (user_id) is implicitly handled by `get_chacha_db_for_user`
        # The `db` instance is already specific to the authenticated user.
        safe_title_log = (note_in.title or "").strip()
        if len(safe_title_log) > 30:
            safe_title_log = safe_title_log[:30] + "..."
        logger.info(f"User (via DB instance client_id: {db.client_id}) creating note: Title='{safe_title_log}'")
        # Compute title (auto-generate if requested)
        effective_title = (note_in.title or "").strip()
        if not effective_title:
            if getattr(note_in, "auto_title", False):
                try:
                    opts = _build_title_opts(note_in)
                    effective_title = await asyncio.to_thread(
                        generate_note_title,
                        note_in.content,
                        options=opts,
                    )
                except _NOTES_NONCRITICAL_EXCEPTIONS as gen_err:
                    logger.warning(f"Auto-title generation failed, falling back: {gen_err}")
                    # Fallback to safe timestamped title
                    effective_title = await asyncio.to_thread(generate_note_title, note_in.content)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Title is required unless auto_title=true")

        conversation_id, message_id = _validate_note_links(
            db,
            note_in.conversation_id,
            note_in.message_id,
        )

        sync_service = _active_notes_sync_service(current_user)
        if sync_service is not None and _field_supplied(note_in, "keywords"):
            raise _note_keywords_sync_unsupported_error()
        idempotency_key = request.headers.get("Idempotency-Key")
        stable_key = server_origin_stable_key(
            source="server_api",
            domain="notes.note",
            operation="upsert",
            idempotency_key=idempotency_key,
        )
        note_id = (
            note_in.id
            or (server_origin_object_id("notes.note", idempotency_key) if sync_service is not None else None)
            or str(uuid4())
        )
        if sync_service is not None:
            try:
                capture_server_origin_mutation(
                    sync_service,
                    user_id=str(current_user.id),
                    domain="notes.note",
                    operation="upsert",
                    object_id=note_id,
                    payload={
                        "title": effective_title,
                        "content": note_in.content,
                        "conversation_id": conversation_id,
                        "message_id": message_id,
                        "client_id": str(current_user.id),
                        "owner_user_id": str(current_user.id),
                    },
                    source="server_api",
                    stable_key=stable_key,
                )
            except Exception as sync_exc:
                raise _note_sync_http_error(sync_exc) from sync_exc
        else:
            note_id = db.add_note(
                title=effective_title,
                content=note_in.content,
                note_id=note_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            if note_id is None:  # Should be caught by exceptions
                raise CharactersRAGDBError("Note creation failed to return an ID.")

        # Topic monitoring (non-blocking) for title and content
        try:
            mon = get_topic_monitoring_service()
            uid = getattr(db, 'client_id', None)
            src_id = str(note_id)
            if effective_title:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=effective_title,
                    source="notes.create",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                    source_id=src_id,
                )
            if note_in.content:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=note_in.content,
                    source="notes.create",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                    source_id=src_id,
                )
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            pass

        keyword_sync_summary: dict[str, Any] | None = None
        # Handle optional keywords without failing note creation on partial errors.
        try:
            kw_list = note_in.normalized_keywords if hasattr(note_in, 'normalized_keywords') else None
            if kw_list:
                keyword_sync_summary = _sync_note_keywords(db, note_id=note_id, keywords=kw_list)
        except _NOTES_NONCRITICAL_EXCEPTIONS as kw_outer_err:
            logger.warning(f"Keyword processing encountered an issue for note {note_id}: {kw_outer_err}")

        created_note_data = db.get_note_by_id(note_id=note_id)
        if not created_note_data:
            logger.error(
                f"Failed to retrieve note '{note_id}' immediately after creation for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Note created but could not be retrieved.")
        _reconcile_note_tasks_after_save(
            db=db,
            note_data=created_note_data,
            current_user=current_user,
            task_service=task_service,
        )
        # Attach keywords inline
        created_note_data = _attach_keywords_inline(db, created_note_data)
        created_note_data = _attach_folders_inline(db, created_note_data)
        if keyword_sync_summary and keyword_sync_summary.get("failed_count", 0) > 0:
            created_note_data["keyword_sync"] = {
                "failed_count": int(keyword_sync_summary.get("failed_count", 0)),
                "failed_keywords": list(keyword_sync_summary.get("failed_keywords", [])),
            }

        record_note_created(user_id=current_user.id, note=created_note_data)
        logger.info(f"Note '{note_id}' created successfully for user (DB client_id: {db.client_id}).")
        return created_note_data  # Pydantic will convert dict to NoteResponse (including keywords)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note")


@router.get(
    "/",
    response_model=NotesListResponse,
    summary="List all notes for the current user",
    tags=["notes"]
)
async def list_notes(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000, description="Number of notes to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        include_keywords: bool = Query(False, description="If true, include linked keywords inline per note"),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.list")),
):
    """Always returns a consistent object with a `notes` array and pagination fields."""
    try:
        # Rate limit: notes.list
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.list",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        logger.debug(f"User (DB client_id: {db.client_id}) listing notes: limit={limit}, offset={offset}")
        notes_data = db.list_notes(limit=limit, offset=offset)
        _attach_folders_bulk(db, notes_data)
        # Attach keywords inline for each note (optional for performance)
        if include_keywords:
            try:
                _attach_keywords_bulk(db, notes_data)
            except _NOTES_NONCRITICAL_EXCEPTIONS as outer_err:
                logger.warning(f"Attaching keywords for notes list failed: {outer_err}")
        # Lightweight total count
        total = None
        try:
            total = db.count_notes()
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            total = None
        # Back-compat aliases for list consumers
        return {
            "notes": notes_data,
            "items": notes_data,
            "results": notes_data,
            "count": len(notes_data),
            "limit": limit,
            "offset": offset,
            "total": total,
            "pagination": build_offset_pagination_meta(
                limit=limit,
                offset=offset,
                total=total,
                count=len(notes_data),
            ),
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes list")


@router.get(
    "/trash",
    response_model=NotesListResponse,
    summary="List soft-deleted notes for the current user",
    tags=["notes"]
)
async def list_deleted_notes(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000, description="Number of trashed notes to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        include_keywords: bool = Query(False, description="If true, include linked keywords inline per note"),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.list")),
):
    """Returns only soft-deleted notes with the same list payload shape as `/notes/`."""
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.list",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})

        logger.debug(
            f"User (DB client_id: {db.client_id}) listing deleted notes: limit={limit}, offset={offset}")
        notes_data = db.list_deleted_notes(limit=limit, offset=offset)
        _attach_folders_bulk(db, notes_data)
        if include_keywords:
            try:
                _attach_keywords_bulk(db, notes_data)
            except _NOTES_NONCRITICAL_EXCEPTIONS as outer_err:
                logger.warning(f"Attaching keywords for deleted notes list failed: {outer_err}")

        total = None
        try:
            total = db.count_deleted_notes()
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            total = None

        return {
            "notes": notes_data,
            "items": notes_data,
            "results": notes_data,
            "count": len(notes_data),
            "limit": limit,
            "offset": offset,
            "total": total,
            "pagination": build_offset_pagination_meta(
                limit=limit,
                offset=offset,
                total=total,
                count=len(notes_data),
            ),
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "deleted notes list")


async def _check_note_rate_limit(
        rate_limiter: RateLimiter,
        current_user: User,
        action: str,
) -> tuple[bool, dict[str, Any]]:
    """Check a notes rate limit and log fail-open limiter outages."""
    try:
        return await rate_limiter.check_user_rate_limit(int(current_user.id), action)
    except _NOTES_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Rate limiter check failed for notes action '{}' and user '{}': {}",
            action,
            current_user.id,
            exc,
        )
        return True, {}


@router.get(
    "/folders",
    response_model=NoteFoldersListResponse,
    summary="List note folders for the current user",
    tags=["notes"],
)
@router.get(
    "/folders/",
    response_model=NoteFoldersListResponse,
    summary="List note folders for the current user",
    tags=["notes"],
)
async def list_note_folders(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(1000, ge=1, le=1000, description="Number of folders to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.list")),
) -> NoteFoldersListResponse:
    """List active note folders for the authenticated user's notes UI."""
    try:
        allowed, meta = await _check_note_rate_limit(rate_limiter, current_user, "notes.list")
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )

        folders = await _run_db_call(db.list_note_folders, limit=limit, offset=offset)
        folder_items = [NoteFolderResponse.model_validate(folder) for folder in folders]
        return NoteFoldersListResponse(
            items=folder_items,
            folders=folder_items,
            count=len(folder_items),
        )
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note folders list")


@router.post(
    "/folders",
    response_model=NoteFolderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or reuse a note folder path",
    tags=["notes"],
)
@router.post(
    "/folders/",
    response_model=NoteFolderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or reuse a note folder path",
    tags=["notes"],
)
async def create_note_folder(
        folder_in: NoteFolderCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.create")),
) -> NoteFolderResponse | JSONResponse:
    """Create a note folder path or return the existing matching folder."""
    try:
        allowed, meta = await _check_note_rate_limit(rate_limiter, current_user, "notes.create")
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.create",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )

        existing = await _run_db_call(db.get_note_folder_by_path, folder_in.path)
        if existing is not None:
            existing_response = NoteFolderResponse.model_validate(existing)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=jsonable_encoder(existing_response),
            )
        try:
            folder = await _run_db_call(db.create_note_folder_path, folder_in.path)
        except ConflictError:
            existing_after_conflict = await _run_db_call(db.get_note_folder_by_path, folder_in.path)
            if existing_after_conflict is not None:
                existing_response = NoteFolderResponse.model_validate(existing_after_conflict)
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content=jsonable_encoder(existing_response),
                )
            raise
        return NoteFolderResponse.model_validate(folder)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note folder")


@router.get(
    "/search",
    response_model=list[NoteResponse],
    summary="Search notes for the current user",
    tags=["notes"]
)
@router.get(
    "/search/",
    response_model=list[NoteResponse],
    summary="Search notes for the current user",
    tags=["notes"]
)
async def search_notes_endpoint(  # Renamed to avoid conflict with imported search_notes
        query: Optional[str] = Query(None, min_length=1, description="Search term for notes"),
        tokens: Optional[list[str]] = Query(None, description="Keyword tokens to filter notes"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(10, ge=1, le=100, description="Number of results to return"),
        offset: int = Query(0, ge=0, description="Result offset for pagination"),
        include_keywords: bool = Query(False, description="If true, include linked keywords inline per note"),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.search")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.search")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.search",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        token_list = _normalize_keyword_tokens(tokens)
        query_term = query.strip() if query else ""
        if not query_term and not token_list:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="query or tokens is required")
        logger.debug(
            f"User (DB client_id: {db.client_id}) searching notes: query='{query_term}', limit={limit}, offset={offset}, tokens={token_list}")
        if token_list:
            notes_data = db.search_notes_with_keywords(
                search_term=query_term or None,
                keyword_tokens=token_list,
                limit=limit,
                offset=offset
            )
        else:
            notes_data = db.search_notes(search_term=query_term, limit=limit, offset=offset)
        _attach_folders_bulk(db, notes_data)
        # Attach keywords inline (optional)
        if include_keywords:
            try:
                _attach_keywords_bulk(db, notes_data)
            except _NOTES_NONCRITICAL_EXCEPTIONS as outer_err:
                logger.warning(f"Attaching keywords for notes search failed: {outer_err}")
        return notes_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes search")


@router.get(
    "/export",
    response_model=NotesExportResponse,
    summary="Export notes as JSON",
    tags=["notes"]
)
async def export_notes(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        q: Optional[str] = Query(None, description="Optional search query to filter notes"),
        limit: int = Query(1000, ge=1, le=10000, description="Max notes to export"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        include_keywords: bool = Query(False, description="If true, include linked keywords inline per note"),
        format: str = Query("json", description="Export format. Only json here; use /export.csv for CSV."),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.export")),
):
    """Simple JSON export for notes. If `q` is provided, uses FTS search; otherwise lists notes."""
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.export")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.export",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        if str(format).lower() != "json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="CSV export is available at /api/v1/notes/export.csv")
        total = None
        if q:
            notes_data = db.search_notes(search_term=q, limit=limit, offset=offset)
            try:
                total = db.count_notes_matching(q)
            except _NOTES_NONCRITICAL_EXCEPTIONS:
                total = None
        else:
            notes_data = db.list_notes(limit=limit, offset=offset)
            try:
                total = db.count_notes()
            except _NOTES_NONCRITICAL_EXCEPTIONS:
                total = None
        for nd in notes_data:
            if isinstance(nd, dict):
                nd.pop("bm25_score", None)
                nd.pop("rank", None)
        if include_keywords:
            _attach_keywords_bulk(db, notes_data)

        return {
            "notes": notes_data,
            "data": notes_data,
            "items": notes_data,
            "results": notes_data,
            "count": len(notes_data),
            "total": total,
            "limit": limit,
            "offset": offset,
            "pagination": build_offset_pagination_meta(
                limit=limit,
                offset=offset,
                total=total,
                count=len(notes_data),
            ),
            "exported_at": __import__("datetime").datetime.utcnow().isoformat()
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes export")


@router.get(
    "/export.csv",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Notes CSV export download.",
            "content": {"text/csv; charset=utf-8": {}},
        },
    },
    summary="Export notes as CSV",
    tags=["notes"]
)
async def export_notes_csv(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        q: Optional[str] = Query(None, description="Optional search query to filter notes"),
        limit: int = Query(1000, ge=1, le=10000, description="Max notes to export"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        include_keywords: bool = Query(False, description="If true, include linked keywords inline per note"),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.export")),
):
    """CSV export for notes. If `q` is provided, uses FTS search; otherwise lists notes."""
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.export")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.export",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        if q:
            notes_data = db.search_notes(search_term=q, limit=limit, offset=offset)
        else:
            notes_data = db.list_notes(limit=limit, offset=offset)
        for nd in notes_data:
            if isinstance(nd, dict):
                nd.pop("bm25_score", None)
                nd.pop("rank", None)
        if include_keywords:
            _attach_keywords_bulk(db, notes_data)
        return _notes_csv_response(notes_data, include_keywords)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes export (csv)")


@router.post(
    "/export",
    response_model=NotesExportResponse,
    summary="Export selected notes by ID",
    tags=["notes"]
)
async def export_notes_post(
        payload: NotesExportRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.export")),
):
    """Export notes by explicit IDs (parity with E2E scaffold)."""
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.export")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.export",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        note_ids = payload.note_ids
        include_keywords = bool(payload.include_keywords)
        fmt = str(payload.format).lower()
        if fmt != "json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="CSV export is available at /api/v1/notes/export.csv")

        results: list[dict[str, Any]] = []
        for nid in note_ids:
            try:
                nd = db.get_note_by_id(note_id=nid)
                if not nd:
                    continue
                if include_keywords:
                    nd["keywords"] = []
                results.append(nd)
            except _NOTES_NONCRITICAL_EXCEPTIONS as fetch_err:
                logger.debug(f"Skipping note ID '{nid}' during export: {fetch_err}")
                continue

        if include_keywords and results:
            _attach_keywords_bulk(db, results)

        return {
            "notes": results,
            "data": results,
            "items": results,
            "results": results,
            "count": len(results),
            "exported_at": __import__("datetime").datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes export (POST)")


@router.post(
    "/export.csv",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Selected notes CSV export download.",
            "content": {"text/csv; charset=utf-8": {}},
        },
    },
    summary="Export selected notes as CSV",
    tags=["notes"]
)
async def export_notes_post_csv(
        payload: NotesExportRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.export")),
):
    """CSV export for notes by explicit IDs."""
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.export")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.export",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        note_ids = payload.note_ids
        include_keywords = bool(payload.include_keywords)

        results: list[dict[str, Any]] = []
        for nid in note_ids:
            try:
                nd = db.get_note_by_id(note_id=nid)
                if not nd:
                    continue
                if include_keywords:
                    nd["keywords"] = []
                results.append(nd)
            except _NOTES_NONCRITICAL_EXCEPTIONS as fetch_err:
                logger.debug(f"Skipping note ID '{nid}' during CSV export: {fetch_err}")
                continue

        if include_keywords and results:
            _attach_keywords_bulk(db, results)

        return _notes_csv_response(results, include_keywords)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes export (POST csv)")


@router.post(
    "/import",
    response_model=NotesImportResponse,
    summary="Import notes from JSON or Markdown",
    tags=["notes"],
)
async def import_notes(
        payload: NotesImportRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        task_service: NotesTaskService = Depends(get_notes_task_service),
        _: None = Depends(rbac_rate_limit("notes.bulk_create")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.bulk_create")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.import",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        files: list[NotesImportFileResult] = []
        totals = {
            "detected_notes": 0,
            "created_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
        }
        companion_events: list[dict[str, Any]] = []
        sync_service = _active_notes_sync_service(current_user)

        for item in payload.items:
            file_result = NotesImportFileResult(
                file_name=item.file_name,
                source_format=item.format,
            )
            fallback_title = _fallback_title_from_filename(item.file_name)
            parsed_notes: list[dict[str, Any]] = []

            try:
                if item.format == "json":
                    raw_payload = json.loads(item.content)
                    raw_rows = _extract_json_note_rows(raw_payload)
                    file_result.detected_notes = len(raw_rows)
                    for row_index, row in enumerate(raw_rows, start=1):
                        try:
                            parsed_notes.append(_normalize_import_note_row(row, fallback_title=fallback_title))
                        except _NOTES_NONCRITICAL_EXCEPTIONS as row_err:
                            file_result.failed_count += 1
                            file_result.errors.append(f"Row {row_index}: {row_err}")
                else:
                    file_result.detected_notes = 1
                    parsed_notes = [_parse_markdown_import_note(item.content, fallback_title=fallback_title)]
            except json.JSONDecodeError as decode_err:
                file_result.failed_count += 1
                file_result.errors.append(f"JSON parse error: {decode_err.msg}")
            except _NOTES_NONCRITICAL_EXCEPTIONS as parse_err:
                file_result.failed_count += 1
                file_result.errors.append(f"Could not parse import content: {parse_err}")

            if sync_service is not None and any(
                parsed_note.get("keywords_provided") or parsed_note.get("keywords")
                for parsed_note in parsed_notes
            ):
                raise _note_keywords_sync_unsupported_error()

            for note_index, parsed_note in enumerate(parsed_notes, start=1):
                try:
                    imported_id = parsed_note.get("id")
                    existing_note = db.get_note_by_id(imported_id) if imported_id else None

                    if existing_note and payload.duplicate_strategy == "skip":
                        file_result.skipped_count += 1
                        continue

                    if existing_note and payload.duplicate_strategy == "overwrite":
                        update_patch = {
                            "title": parsed_note["title"],
                            "content": parsed_note["content"],
                        }
                        if sync_service is not None:
                            projected_note = dict(existing_note)
                            projected_note.update(update_patch)
                            try:
                                capture_server_origin_mutation(
                                    sync_service,
                                    user_id=str(current_user.id),
                                    domain="notes.note",
                                    operation="upsert",
                                    object_id=str(imported_id),
                                    payload=_note_payload_from_row(projected_note),
                                    source="server_api",
                                )
                            except Exception as sync_exc:
                                raise _note_sync_http_error(sync_exc) from sync_exc
                        else:
                            expected_version = int(existing_note.get("version", 1))
                            db.update_note(
                                note_id=str(imported_id),
                                update_data=update_patch,
                                expected_version=expected_version,
                            )
                        overwritten_note = db.get_note_by_id(str(imported_id))
                        if not overwritten_note:
                            raise CharactersRAGDBError("Import overwrite note could not be retrieved.")  # noqa: TRY003
                        _reconcile_note_tasks_after_save(
                            db=db,
                            note_data=overwritten_note,
                            current_user=current_user,
                            task_service=task_service,
                        )
                        if parsed_note.get("keywords_provided"):
                            _sync_note_keywords(
                                db,
                                note_id=str(imported_id),
                                keywords=parsed_note.get("keywords", []),
                            )
                        companion_events.append(
                            _build_import_note_companion_event(
                                db=db,
                                note_id=str(imported_id),
                                operation="import_overwrite",
                                patch=update_patch,
                            )
                        )
                        file_result.updated_count += 1
                        continue

                    create_with_id = None if payload.duplicate_strategy == "create_copy" else imported_id
                    if sync_service is not None:
                        created_note_id = str(create_with_id or uuid4())
                        try:
                            capture_server_origin_mutation(
                                sync_service,
                                user_id=str(current_user.id),
                                domain="notes.note",
                                operation="upsert",
                                object_id=created_note_id,
                                payload={
                                    "title": parsed_note["title"],
                                    "content": parsed_note["content"],
                                    "conversation_id": None,
                                    "message_id": None,
                                    "client_id": str(current_user.id),
                                    "owner_user_id": str(current_user.id),
                                },
                                source="server_api",
                            )
                        except Exception as sync_exc:
                            raise _note_sync_http_error(sync_exc) from sync_exc
                    else:
                        created_note_id = db.add_note(
                            title=parsed_note["title"],
                            content=parsed_note["content"],
                            note_id=create_with_id,
                        )
                    if not created_note_id:
                        raise CharactersRAGDBError("Import create returned no note ID.")  # noqa: TRY003
                    created_note = db.get_note_by_id(str(created_note_id))
                    if not created_note:
                        raise CharactersRAGDBError("Import created note could not be retrieved.")  # noqa: TRY003
                    _reconcile_note_tasks_after_save(
                        db=db,
                        note_data=created_note,
                        current_user=current_user,
                        task_service=task_service,
                    )
                    if parsed_note.get("keywords"):
                        _sync_note_keywords(
                            db,
                            note_id=str(created_note_id),
                            keywords=parsed_note.get("keywords", []),
                        )
                    companion_events.append(
                        _build_import_note_companion_event(
                            db=db,
                            note_id=str(created_note_id),
                            operation="import_create",
                        )
                    )
                    file_result.created_count += 1
                except ConflictError as conflict_err:
                    # If "create_copy" still conflicts (for example, stale imported ID edge case),
                    # retry once without imported ID before surfacing a failure.
                    if payload.duplicate_strategy == "create_copy":
                        try:
                            created_note_id = db.add_note(
                                title=parsed_note["title"],
                                content=parsed_note["content"],
                                note_id=None,
                            )
                            if not created_note_id:
                                raise CharactersRAGDBError("Import create-copy returned no note ID.")  # noqa: TRY003
                            created_note = db.get_note_by_id(str(created_note_id))
                            if not created_note:
                                raise CharactersRAGDBError("Import created note could not be retrieved.")  # noqa: TRY003
                            _reconcile_note_tasks_after_save(
                                db=db,
                                note_data=created_note,
                                current_user=current_user,
                                task_service=task_service,
                            )
                            if parsed_note.get("keywords"):
                                _sync_note_keywords(
                                    db,
                                    note_id=str(created_note_id),
                                    keywords=parsed_note.get("keywords", []),
                                )
                            companion_events.append(
                                _build_import_note_companion_event(
                                    db=db,
                                    note_id=str(created_note_id),
                                    operation="import_create",
                                )
                            )
                            file_result.created_count += 1
                            continue
                        except _NOTES_NONCRITICAL_EXCEPTIONS as retry_err:
                            file_result.failed_count += 1
                            file_result.errors.append(f"Note {note_index}: {retry_err}")
                            continue
                    file_result.failed_count += 1
                    file_result.errors.append(f"Note {note_index}: {conflict_err}")
                except HTTPException:
                    raise
                except _NOTES_NONCRITICAL_EXCEPTIONS as note_err:
                    file_result.failed_count += 1
                    file_result.errors.append(f"Note {note_index}: {note_err}")

            totals["detected_notes"] += file_result.detected_notes
            totals["created_count"] += file_result.created_count
            totals["updated_count"] += file_result.updated_count
            totals["skipped_count"] += file_result.skipped_count
            totals["failed_count"] += file_result.failed_count
            files.append(file_result)

        await _run_db_call(
            record_companion_activity_events_bulk,
            user_id=current_user.id,
            events=companion_events,
        )
        return NotesImportResponse(files=files, **totals)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes import")


@router.get(
    "/keywords",
    response_model=list[KeywordResponse],
    summary="List all keywords for the current user",
    tags=["Keywords (for Notes)"],
    include_in_schema=False,
)
async def list_keywords_endpoint_no_trailing_slash(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        include_note_counts: bool = Query(
            False,
            description="If true, include the active note count linked to each keyword"
        ),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.list")),
):
    # Keep this alias ahead of "/{note_id}" so "/notes/keywords" never binds
    # to get_note(note_id="keywords").
    return await _list_keywords_impl(
        db=db,
        limit=limit,
        offset=offset,
        include_note_counts=include_note_counts,
        rate_limiter=rate_limiter,
        current_user=current_user,
    )


@router.get(
    "/collections",
    response_model=KeywordCollectionsListResponse,
    summary="List keyword collections for the current user",
    tags=["Notes Collections"],
)
@router.get(
    "/collections/",
    response_model=KeywordCollectionsListResponse,
    summary="List keyword collections for the current user",
    tags=["Notes Collections"],
    include_in_schema=False,
)
async def list_keyword_collections_endpoint(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        include_keywords: bool = Query(
            False,
            description="If true, include linked keywords inline per collection."
        ),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.list")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        collections_data = db.list_keyword_collections(limit=limit, offset=offset)
        total = None
        try:
            total = db.count_keyword_collections()
        except _NOTES_NONCRITICAL_EXCEPTIONS as count_exc:
            logger.warning("Counting keyword collections failed: {}", count_exc)
        if include_keywords:
            collections_data = [
                _attach_collection_keywords_inline(db, dict(row))
                for row in collections_data
            ]
        response_total = total if total is not None else offset + len(collections_data)

        return {
            "collections": collections_data,
            "count": len(collections_data),
            "limit": limit,
            "offset": offset,
            "total": response_total,
            "pagination": build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(collections_data),
            ),
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.post(
    "/collections",
    response_model=KeywordCollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a keyword collection",
    tags=["Notes Collections"],
)
@router.post(
    "/collections/",
    response_model=KeywordCollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a keyword collection",
    tags=["Notes Collections"],
    include_in_schema=False,
)
async def create_keyword_collection_endpoint(
        collection_in: KeywordCollectionCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.create")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.create")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.create",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        collection_name = str(collection_in.name or "").strip()
        if not collection_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Collection name cannot be empty.")

        collection_id = db.add_keyword_collection(
            name=collection_name,
            parent_id=collection_in.parent_id,
        )
        if collection_id is None:
            raise CharactersRAGDBError("Collection creation failed to return an ID.")

        kw_list = collection_in.normalized_keywords or []
        if kw_list:
            _sync_collection_keywords(db, collection_id=int(collection_id), keywords=kw_list)

        collection_data = db.get_keyword_collection_by_id(collection_id=int(collection_id))
        if not collection_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Collection created but could not be retrieved."
            )

        return _attach_collection_keywords_inline(db, collection_data)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.get(
    "/collections/keyword-links",
    response_model=CollectionKeywordLinksResponse,
    summary="List collection-keyword links",
    tags=["Notes Collections"],
)
async def list_collection_keyword_links_endpoint(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(1000, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.list")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        cursor = db.execute_query(
            "SELECT collection_id, keyword_id FROM collection_keywords ORDER BY collection_id ASC, keyword_id ASC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()
        links = [
            {
                "collection_id": int(row["collection_id"]),
                "keyword_id": int(row["keyword_id"]),
            }
            for row in rows
        ]
        total = int(db.count_collection_keyword_links())
        pagination = build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(links),
        )
        return {
            "links": links,
            "count": len(links),
            "limit": limit,
            "offset": offset,
            "total": total,
            "pagination": pagination,
            "has_more": pagination.has_more,
            "next_offset": pagination.next_offset,
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.patch(
    "/collections/{collection_id}",
    response_model=KeywordCollectionResponse,
    summary="Update a keyword collection",
    tags=["Notes Collections"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse},
    },
)
async def update_keyword_collection_endpoint(
        collection_id: int,
        collection_in: KeywordCollectionUpdate,
        expected_version: int | None = Header(
            default=None,
            description="Expected collection version for optimistic locking.",
        ),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        task_service: NotesTaskService = Depends(get_notes_task_service),
        _: None = Depends(rbac_rate_limit("notes.update")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.update",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        current_collection = db.get_keyword_collection_by_id(collection_id=collection_id)
        if not current_collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")

        update_data: dict[str, Any] = {}
        if _field_supplied(collection_in, "name"):
            normalized_name = str(collection_in.name or "").strip()
            if not normalized_name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Collection name cannot be empty.")
            update_data["name"] = normalized_name
        if _field_supplied(collection_in, "parent_id"):
            update_data["parent_id"] = collection_in.parent_id

        keywords_supplied = _field_supplied(collection_in, "keywords")
        kw_list = collection_in.normalized_keywords if keywords_supplied else None

        if not update_data and not keywords_supplied:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No changes provided.")

        if update_data:
            version_to_use = (
                expected_version
                if expected_version is not None
                else int(current_collection.get("version", 1))
            )
            db.update_keyword_collection(
                collection_id=collection_id,
                update_data=update_data,
                expected_version=version_to_use,
            )

        if keywords_supplied:
            _sync_collection_keywords(db, collection_id=collection_id, keywords=kw_list or [])

        updated_collection = db.get_keyword_collection_by_id(collection_id=collection_id)
        if not updated_collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        return _attach_collection_keywords_inline(db, updated_collection)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft-delete a keyword collection",
    tags=["Notes Collections"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse},
    },
)
async def delete_keyword_collection_endpoint(
        collection_id: int,
        expected_version: int | None = Header(
            default=None,
            description="Expected collection version for optimistic locking.",
        ),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.delete")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.delete")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.delete",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        current_collection = db.get_keyword_collection_by_id(collection_id=collection_id)
        if not current_collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")

        version_to_use = (
            expected_version
            if expected_version is not None
            else int(current_collection.get("version", 1))
        )
        db.soft_delete_keyword_collection(
            collection_id=collection_id,
            expected_version=version_to_use,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.post(
    "/collections/{collection_id}/keywords/{keyword_id}",
    response_model=CollectionKeywordLinkResponse,
    summary="Link a keyword to a collection",
    tags=["Notes Collections"],
)
async def link_collection_to_keyword_endpoint(
        collection_id: int,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.link_keyword")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.link_keyword")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.link_keyword",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        collection = db.get_keyword_collection_by_id(collection_id=collection_id)
        if not collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        keyword = db.get_keyword_by_id(keyword_id=keyword_id)
        if not keyword:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found")

        linked = db.link_collection_to_keyword(collection_id=collection_id, keyword_id=keyword_id)
        msg = "Keyword linked to collection." if linked else "Link already exists or was created."
        return CollectionKeywordLinkResponse(success=True, message=msg)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.delete(
    "/collections/{collection_id}/keywords/{keyword_id}",
    response_model=CollectionKeywordLinkResponse,
    summary="Unlink a keyword from a collection",
    tags=["Notes Collections"],
)
async def unlink_collection_from_keyword_endpoint(
        collection_id: int,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.unlink_keyword")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.unlink_keyword")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.unlink_keyword",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        success = db.unlink_collection_from_keyword(collection_id=collection_id, keyword_id=keyword_id)
        msg = "Keyword unlinked from collection." if success else "Link not found or no action taken."
        return CollectionKeywordLinkResponse(success=success, message=msg)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.get(
    "/collections/{collection_id}/keywords",
    response_model=list[KeywordResponse],
    summary="List keywords linked to a collection",
    tags=["Notes Collections"],
)
@router.get(
    "/collections/{collection_id}/keywords/",
    response_model=list[KeywordResponse],
    summary="List keywords linked to a collection",
    tags=["Notes Collections"],
    include_in_schema=False,
)
async def list_keywords_for_collection_endpoint(
        collection_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.keywords.list")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.keywords.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.keywords.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        collection = db.get_keyword_collection_by_id(collection_id=collection_id)
        if not collection:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        return db.get_keywords_for_collection(collection_id=collection_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "collection")


@router.get(
    "/conversations/keyword-links",
    response_model=ConversationKeywordLinksResponse,
    summary="List conversation-keyword links",
    tags=["Notes Linking"],
)
async def list_conversation_keyword_links_endpoint(
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        ids: str | None = Query(
            default=None,
            description="Optional comma-separated conversation IDs.",
        ),
        limit: int = Query(1000, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.list")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        conversation_ids = []
        if ids:
            conversation_ids = [
                token.strip()
                for token in ids.split(",")
                if token and token.strip()
            ]

        links: list[dict[str, Any]] = []
        if conversation_ids:
            for conversation_id in conversation_ids:
                kw_rows = db.get_keywords_for_conversation(conversation_id=conversation_id)
                for kw in kw_rows:
                    kw_id = kw.get("id") if isinstance(kw, dict) else None
                    if kw_id is None:
                        continue
                    links.append(
                        {
                            "conversation_id": conversation_id,
                            "keyword_id": int(kw_id),
                        }
                    )
            total = len(links)
            links = links[offset: offset + limit]
        else:
            cursor = db.execute_query(
                "SELECT conversation_id, keyword_id FROM conversation_keywords ORDER BY conversation_id ASC, keyword_id ASC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
            links = [
                {
                    "conversation_id": str(row["conversation_id"]),
                    "keyword_id": int(row["keyword_id"]),
                }
                for row in rows
            ]
            total = int(db.count_conversation_keyword_links())
        pagination = build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(links),
        )
        return {
            "links": links,
            "count": len(links),
            "limit": limit,
            "offset": offset,
            "total": total,
            "pagination": pagination,
            "has_more": pagination.has_more,
            "next_offset": pagination.next_offset,
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "conversation-keyword link")


@router.post(
    "/conversations/{conversation_id}/keywords/{keyword_id}",
    response_model=ConversationKeywordLinkResponse,
    summary="Link a keyword to a conversation",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}},
)
async def link_conversation_to_keyword_endpoint(
        conversation_id: str,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.link_keyword")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.link_keyword")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.link_keyword",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        conv = db.get_conversation_by_id(conversation_id)
        if not conv:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        keyword = db.get_keyword_by_id(keyword_id)
        if not keyword:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found")

        linked = db.link_conversation_to_keyword(conversation_id=conversation_id, keyword_id=keyword_id)
        msg = "Keyword linked to conversation." if linked else "Link already exists or was created."
        return ConversationKeywordLinkResponse(success=True, message=msg)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "conversation-keyword link")


@router.delete(
    "/conversations/{conversation_id}/keywords/{keyword_id}",
    response_model=ConversationKeywordLinkResponse,
    summary="Unlink a keyword from a conversation",
    tags=["Notes Linking"],
)
async def unlink_conversation_from_keyword_endpoint(
        conversation_id: str,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.unlink_keyword")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.unlink_keyword")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.unlink_keyword",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        success = db.unlink_conversation_from_keyword(conversation_id=conversation_id, keyword_id=keyword_id)
        msg = "Keyword unlinked from conversation." if success else "Link not found or no action taken."
        return ConversationKeywordLinkResponse(success=success, message=msg)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "conversation-keyword link")


@router.get(
    "/conversations/{conversation_id}/keywords",
    response_model=list[KeywordResponse],
    summary="List keywords linked to a conversation",
    tags=["Notes Linking"],
)
@router.get(
    "/conversations/{conversation_id}/keywords/",
    response_model=list[KeywordResponse],
    summary="List keywords linked to a conversation",
    tags=["Notes Linking"],
    include_in_schema=False,
)
async def list_keywords_for_conversation_endpoint(
        conversation_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.keywords.list")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.keywords.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.keywords.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )

        conv = db.get_conversation_by_id(conversation_id=conversation_id)
        if not conv:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return db.get_keywords_for_conversation(conversation_id=conversation_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "conversation-keyword link")


@router.post(
    "/moodboards",
    response_model=MoodboardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a moodboard",
    tags=["Moodboards"],
)
@router.post(
    "/moodboards/",
    response_model=MoodboardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a moodboard",
    tags=["Moodboards"],
    include_in_schema=False,
)
async def create_moodboard(
    moodboard_in: MoodboardCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.create")),
) -> MoodboardResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.create")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.create",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        moodboard_id = await _run_db_call(
            db.add_moodboard,
            name=moodboard_in.name,
            description=moodboard_in.description,
            smart_rule=moodboard_in.smart_rule.model_dump(mode="json") if moodboard_in.smart_rule else None,
        )
        if moodboard_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Moodboard created but could not be retrieved.",
            )
        moodboard_row = await _run_db_call(db.get_moodboard_by_id, int(moodboard_id))
        if not moodboard_row:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Moodboard created but could not be retrieved.",
            )
        return _normalize_moodboard_payload(moodboard_row)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.get(
    "/moodboards",
    response_model=MoodboardListResponse,
    summary="List moodboards",
    tags=["Moodboards"],
)
@router.get(
    "/moodboards/",
    response_model=MoodboardListResponse,
    summary="List moodboards",
    tags=["Moodboards"],
    include_in_schema=False,
)
async def list_moodboards_endpoint(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_deleted: bool = Query(False),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.list")),
) -> MoodboardListResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        rows = await _run_db_call(db.list_moodboards, limit=limit, offset=offset, include_deleted=include_deleted)
        total = None
        try:
            total = await _run_db_call(db.count_moodboards, include_deleted=include_deleted)
        except _NOTES_NONCRITICAL_EXCEPTIONS as count_exc:
            logger.warning("Counting moodboards failed: {}", count_exc)
        payload = [_normalize_moodboard_payload(row) for row in rows]
        return MoodboardListResponse(
            items=payload,
            moodboards=payload,
            count=len(payload),
            limit=limit,
            offset=offset,
            total=total,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(payload),
            ),
        )
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.get(
    "/moodboards/{moodboard_id}",
    response_model=MoodboardResponse,
    summary="Get a moodboard",
    tags=["Moodboards"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}},
)
async def get_moodboard_endpoint(
    moodboard_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.get")),
) -> MoodboardResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.get")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.get",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        row = await _run_db_call(db.get_moodboard_by_id, moodboard_id=moodboard_id)
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Moodboard not found")
        return _normalize_moodboard_payload(row)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.patch(
    "/moodboards/{moodboard_id}",
    response_model=MoodboardResponse,
    summary="Update a moodboard",
    tags=["Moodboards"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse},
    },
)
async def update_moodboard_endpoint(
    moodboard_id: int,
    moodboard_in: MoodboardUpdate,
    expected_version: int = Header(..., description="Expected moodboard version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.update")),
) -> MoodboardResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.update",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        update_data = _build_moodboard_update_data(moodboard_in)
        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No data provided for update.")

        await _run_db_call(
            db.update_moodboard,
            moodboard_id=moodboard_id,
            update_data=update_data,
            expected_version=expected_version,
        )
        row = await _run_db_call(db.get_moodboard_by_id, moodboard_id=moodboard_id)
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Moodboard not found")
        return _normalize_moodboard_payload(row)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.delete(
    "/moodboards/{moodboard_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft-delete a moodboard",
    tags=["Moodboards"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse},
    },
)
async def delete_moodboard_endpoint(
    moodboard_id: int,
    expected_version: int = Header(..., description="Expected moodboard version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.delete")),
) -> Response:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.delete")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.delete",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        deleted = await _run_db_call(
            db.delete_moodboard,
            moodboard_id=moodboard_id,
            expected_version=expected_version,
        )
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Moodboard not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.post(
    "/moodboards/{moodboard_id}/notes/{note_id}",
    response_model=MoodboardPinResponse,
    summary="Pin a note to a moodboard",
    tags=["Moodboards"],
)
async def pin_note_to_moodboard_endpoint(
    moodboard_id: int,
    note_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.pin")),
) -> MoodboardPinResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.pin")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.pin",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        success = await _run_db_call(db.link_note_to_moodboard, moodboard_id=moodboard_id, note_id=note_id)
        return MoodboardPinResponse(success=success, moodboard_id=moodboard_id, note_id=note_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.delete(
    "/moodboards/{moodboard_id}/notes/{note_id}",
    response_model=MoodboardPinResponse,
    summary="Unpin a note from a moodboard",
    tags=["Moodboards"],
)
async def unpin_note_from_moodboard_endpoint(
    moodboard_id: int,
    note_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.unpin")),
) -> MoodboardPinResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.unpin")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.unpin",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        success = await _run_db_call(db.unlink_note_from_moodboard, moodboard_id=moodboard_id, note_id=note_id)
        return MoodboardPinResponse(success=success, moodboard_id=moodboard_id, note_id=note_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.get(
    "/moodboards/{moodboard_id}/notes",
    response_model=MoodboardNotesListResponse,
    summary="List moodboard notes",
    tags=["Moodboards"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}},
)
@router.get(
    "/moodboards/{moodboard_id}/notes/",
    response_model=MoodboardNotesListResponse,
    summary="List moodboard notes",
    tags=["Moodboards"],
    include_in_schema=False,
)
async def list_moodboard_notes_endpoint(
    moodboard_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("moodboards.notes.list")),
) -> MoodboardNotesListResponse:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "moodboards.notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for moodboards.notes.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )
        board = await _run_db_call(db.get_moodboard_by_id, moodboard_id=moodboard_id)
        if not board:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Moodboard not found")
        notes = await _run_db_call(db.list_moodboard_notes, moodboard_id=moodboard_id, limit=limit, offset=offset)
        total = await _run_db_call(db.count_moodboard_notes, moodboard_id=moodboard_id)
        return MoodboardNotesListResponse(
            items=notes,
            notes=notes,
            count=len(notes),
            limit=limit,
            offset=offset,
            total=total,
            pagination=build_offset_pagination_meta(
                limit=limit,
                offset=offset,
                total=total,
                count=len(notes),
            ),
        )
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "moodboard")


@router.get(
    "/{note_id}/studio",
    response_model=NoteStudioStateResponse,
    summary="Fetch Notes Studio state for a note",
    tags=["notes"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}},
)
async def get_note_studio_state_endpoint(
        note_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.get")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.get")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.get",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )

        studio_state = await NotesStudioService(db=db).get_note_studio_state(note_id=note_id)
        studio_state["note"] = _attach_keywords_inline(db, studio_state["note"])
        studio_state["note"] = _attach_folders_inline(db, studio_state["note"])
        return studio_state
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note studio")


@router.post(
    "/studio/derive",
    response_model=NoteStudioStateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a derived Notes Studio note from a source excerpt",
    tags=["notes"],
)
async def derive_note_studio_endpoint(
        studio_in: NoteStudioDeriveRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.create")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.create")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.create",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )

        studio_state = await NotesStudioService(db=db).derive_from_excerpt(
            source_note_id=studio_in.source_note_id,
            excerpt_text=studio_in.excerpt_text,
            template_type=studio_in.template_type,
            handwriting_mode=studio_in.handwriting_mode,
            provider=studio_in.provider,
            model=studio_in.model,
        )
        studio_state["note"] = _attach_keywords_inline(db, studio_state["note"])
        studio_state["note"] = _attach_folders_inline(db, studio_state["note"])
        record_note_created(user_id=current_user.id, note=studio_state["note"])
        return studio_state
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note studio")


@router.post(
    "/{note_id}/studio/regenerate",
    response_model=NoteStudioStateResponse,
    summary="Regenerate the Markdown companion from the current Notes Studio Markdown",
    tags=["notes"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}},
)
async def regenerate_note_studio_endpoint(
        note_id: str,
        regenerate_in: NoteStudioRegenerateRequest | None = Body(default=None),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.update")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.update",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )

        studio_state = await NotesStudioService(db=db).regenerate_note_markdown(
            note_id=note_id,
            current_markdown=regenerate_in.current_markdown if regenerate_in else None,
        )
        studio_state["note"] = _attach_keywords_inline(db, studio_state["note"])
        studio_state["note"] = _attach_folders_inline(db, studio_state["note"])
        record_note_updated(
            user_id=current_user.id,
            note=studio_state["note"],
            route=f"/api/v1/notes/{note_id}/studio/regenerate",
            action="regenerate",
            patch={"regenerated": True},
        )
        return studio_state
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note studio")


@router.post(
    "/{note_id}/studio/diagrams",
    response_model=NoteStudioDiagramResponse,
    summary="Store a Notes Studio diagram manifest",
    tags=["notes"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}},
)
async def update_note_studio_diagram_endpoint(
        note_id: str,
        diagram_in: NoteStudioDiagramRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.update")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.update",
                headers={"Retry-After": str(meta.get("retry_after", 60))},
            )

        studio_state = await NotesStudioService(db=db).update_diagram_manifest(
            note_id=note_id,
            diagram_type=diagram_in.diagram_type,
            source_section_ids=diagram_in.source_section_ids,
            provider=diagram_in.provider,
            model=diagram_in.model,
        )
        studio_state["note"] = _attach_keywords_inline(db, studio_state["note"])
        studio_state["note"] = _attach_folders_inline(db, studio_state["note"])
        return studio_state
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note studio")


@router.get(
    "/{note_id}",
    response_model=NoteResponse,
    summary="Get a specific note by ID",
    tags=["notes"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_note(
        note_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.get")),
):
    logger.debug(f"User (DB client_id: {db.client_id}) fetching note: ID='{note_id}'")
    try:  # Added try block here to catch DB errors during fetch
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.get")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.get",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        note_data = db.get_note_by_id(note_id=note_id, include_studio_summary=True)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:  # Catch DB errors from get_note_by_id
        handle_db_errors(e, "note")  # This will reraise appropriately

    if not note_data:
        logger.warning(f"Note ID '{note_id}' not found for user (DB client_id: {db.client_id}).")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")

    # If note_data is found, it's a dict from the DB. Pydantic will validate it on return.
    # No need for an explicit try-except for Pydantic here, FastAPI handles it.
    note_data = _attach_keywords_inline(db, note_data)
    note_data = _attach_folders_inline(db, note_data)
    return note_data


@router.post(
    "/{note_id}/attachments",
    response_model=NoteAttachmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an attachment for a note",
    tags=["notes"],
)
async def upload_note_attachment(
        note_id: str,
        file: UploadFile = File(...),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.attachments.upload")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.attachments.upload")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.attachments.upload",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        _ensure_note_exists_or_404(db, note_id)
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attachment filename is required")
        safe_file_name = _sanitize_attachment_file_name(file.filename)
        attachment_dir = _get_note_attachments_dir(current_user.id, note_id, create=True)
        target_path = _resolve_unique_attachment_path(attachment_dir, safe_file_name)
        payload = await file.read(_NOTES_ATTACHMENT_MAX_BYTES + 1)
        if len(payload) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attachment file is empty")
        if len(payload) > _NOTES_ATTACHMENT_MAX_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Attachment exceeds maximum size of {_NOTES_ATTACHMENT_MAX_BYTES} bytes",
            )
        target_path.write_bytes(payload)
        content_type = file.content_type or mimetypes.guess_type(target_path.name)[0]
        uploaded_at = datetime.now(timezone.utc)
        _write_attachment_metadata(
            target_path,
            original_file_name=file.filename,
            content_type=content_type,
            size_bytes=len(payload),
            uploaded_at=uploaded_at,
        )
        logger.info(
            "User {} uploaded attachment '{}' for note '{}'",
            current_user.id,
            target_path.name,
            note_id,
        )
        return _to_attachment_response(note_id, target_path)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note attachment")
    finally:
        try:
            await file.close()
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            pass


@router.get(
    "/{note_id}/attachments",
    response_model=NoteAttachmentsListResponse,
    summary="List attachments for a note",
    tags=["notes"],
)
async def list_note_attachments(
        note_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.attachments.list")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.attachments.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.attachments.list",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        _ensure_note_exists_or_404(db, note_id)
        attachment_dir = _get_note_attachments_dir(current_user.id, note_id, create=False)
        if not attachment_dir.exists():
            return {"note_id": note_id, "attachments": [], "count": 0}
        attachments: list[dict[str, Any]] = []
        for item in sorted(attachment_dir.iterdir(), key=lambda p: p.name.lower()):
            if not item.is_file():
                continue
            if item.name.endswith(_NOTES_ATTACHMENT_META_SUFFIX):
                continue
            attachments.append(_to_attachment_response(note_id, item))
        return {
            "note_id": note_id,
            "attachments": attachments,
            "count": len(attachments),
        }
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note attachment")


@router.get(
    "/{note_id}/attachments/{file_name}",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Raw note attachment file.",
            "content": {"application/octet-stream": {}},
        },
    },
    summary="Download an attachment for a note",
    tags=["notes"],
)
async def download_note_attachment(
        note_id: str,
        file_name: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.attachments.get")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.attachments.get")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.attachments.get",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        _ensure_note_exists_or_404(db, note_id)
        safe_name = _sanitize_attachment_file_name(file_name)
        if safe_name != file_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment filename")
        attachment_dir = _get_note_attachments_dir(current_user.id, note_id, create=False)
        file_path = (attachment_dir / safe_name).resolve()
        try:
            file_path.relative_to(attachment_dir)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment path") from exc
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found")

        metadata = _read_attachment_metadata(file_path)
        content_type = metadata.get("content_type") or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        download_name = str(metadata.get("original_file_name") or file_path.name)
        return FileResponse(
            path=str(file_path),
            filename=download_name,
            media_type=content_type,
            content_disposition_type="inline",
        )
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note attachment")


@router.delete(
    "/{note_id}/attachments/{file_name}",
    response_model=DetailResponse,
    summary="Delete an attachment from a note",
    tags=["notes"],
)
async def delete_note_attachment(
        note_id: str,
        file_name: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.attachments.delete")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.attachments.delete")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for notes.attachments.delete",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        _ensure_note_exists_or_404(db, note_id)
        safe_name = _sanitize_attachment_file_name(file_name)
        if safe_name != file_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment filename")
        attachment_dir = _get_note_attachments_dir(current_user.id, note_id, create=False)
        file_path = (attachment_dir / safe_name).resolve()
        try:
            file_path.relative_to(attachment_dir)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment path") from exc
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found")

        metadata_path = _attachment_metadata_path(file_path).resolve(strict=False)
        try:
            metadata_path.relative_to(attachment_dir)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment metadata path") from exc
        file_path.unlink(missing_ok=False)
        if metadata_path.exists():
            metadata_path.unlink(missing_ok=True)
        logger.info("User {} deleted attachment '{}' for note '{}'", current_user.id, safe_name, note_id)
        return DetailResponse(detail="Attachment deleted")
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note attachment")


async def _list_keywords_impl(
        *,
        db: CharactersRAGDB,
        limit: int,
        offset: int,
        include_note_counts: bool,
        rate_limiter: RateLimiter,
        current_user: User,
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.list",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        logger.debug(f"User (DB client_id: {db.client_id}) listing keywords: limit={limit}, offset={offset}")
        keywords_data = db.list_keywords(limit=limit, offset=offset)
        if include_note_counts and keywords_data:
            keyword_ids: list[int] = []
            for row in keywords_data:
                try:
                    keyword_ids.append(int(row.get("id")))
                except (TypeError, ValueError, AttributeError):
                    continue
            counts = db.get_note_counts_for_keywords(keyword_ids)
            for row in keywords_data:
                try:
                    keyword_id = int(row.get("id"))
                except (TypeError, ValueError, AttributeError):
                    continue
                row["note_count"] = counts.get(keyword_id, 0)
        return keywords_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keywords list")


@router.put(
    "/{note_id}",
    response_model=NoteResponse,
    summary="Update an existing note",
    tags=["notes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def update_note(
        note_id: str,
        note_in: NoteUpdate,
        expected_version: int = Header(..., description="The expected version of the note for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        task_service: NotesTaskService = Depends(get_notes_task_service),
        _: None = Depends(rbac_rate_limit("notes.update")),
):
    keywords_supplied = _field_supplied(note_in, "keywords")
    conversation_supplied = _field_supplied(note_in, "conversation_id")
    message_supplied = _field_supplied(note_in, "message_id")
    kw_list = note_in.normalized_keywords if keywords_supplied else None
    raw_data = note_in.model_dump(exclude_unset=True)
    update_data: dict[str, Any] = {}
    if "title" in raw_data and raw_data["title"] is not None:
        update_data["title"] = raw_data["title"]
    if "content" in raw_data and raw_data["content"] is not None:
        update_data["content"] = raw_data["content"]
    if conversation_supplied:
        update_data["conversation_id"] = raw_data.get("conversation_id")
    if message_supplied:
        update_data["message_id"] = raw_data.get("message_id")
    if "title" in update_data and isinstance(update_data["title"], str):
        stripped_title = update_data["title"].strip()
        if not stripped_title:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Title cannot be empty or whitespace.")
        update_data["title"] = stripped_title
    if not update_data and not keywords_supplied:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid fields provided for update.")
    try:
        current_note: Optional[dict[str, Any]] = None

        def _get_current_note() -> dict[str, Any]:
            nonlocal current_note
            if current_note is None:
                current_note = db.get_note_by_id(note_id=note_id)
            if not current_note:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
            return current_note

        # Rate limit: notes.update
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.update",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        if not update_data and keywords_supplied:
            current_note = _get_current_note()
            current_version = current_note.get("version")
            if current_version is not None and int(current_version) != int(expected_version):
                raise ConflictError(
                    f"Note ID {note_id} update failed: version mismatch (db has {current_version}, client expected {expected_version}).",
                    entity="notes",
                    entity_id=note_id,
                )
        if conversation_supplied or message_supplied:
            current_note = _get_current_note()
            current_conversation_id = current_note.get("conversation_id")
            effective_conversation_id = update_data.get("conversation_id") if conversation_supplied else current_conversation_id
            if message_supplied:
                effective_message_id = update_data.get("message_id")
            else:
                # Preserve the existing message_id so a conversation change is validated
                # against the current message linkage rather than silently clearing it.
                effective_message_id = current_note.get("message_id")
            validated_conversation_id, validated_message_id = _validate_note_links(
                db,
                effective_conversation_id,
                effective_message_id,
            )
            if conversation_supplied:
                update_data["conversation_id"] = validated_conversation_id
            if message_supplied:
                update_data["message_id"] = validated_message_id
        data_keys = list(update_data.keys())
        if keywords_supplied:
            data_keys.append("keywords")
        logger.info(
            f"User (DB client_id: {db.client_id}) updating note: ID='{note_id}', Version={expected_version}, DataKeys={data_keys}")
        # Topic monitoring (non-blocking) for updated fields
        try:
            mon = get_topic_monitoring_service()
            uid = getattr(db, 'client_id', None)
            src_id = str(note_id)
            if 'title' in update_data and update_data['title']:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=str(update_data['title']),
                    source="notes.update",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                    source_id=src_id,
                )
            if 'content' in update_data and update_data['content']:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=str(update_data['content']),
                    source="notes.update",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                    source_id=src_id,
                )
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            pass
        sync_service = _active_notes_sync_service(current_user)
        if sync_service is not None and keywords_supplied:
            raise _note_keywords_sync_unsupported_error()
        if update_data:
            if sync_service is not None:
                current_note = _get_current_note()
                current_version = current_note.get("version")
                if current_version is not None and int(current_version) != int(expected_version):
                    raise ConflictError(
                        f"Note ID {note_id} update failed: version mismatch (db has {current_version}, client expected {expected_version}).",
                        entity="notes",
                        entity_id=note_id,
                    )
                projected_note = dict(current_note)
                projected_note.update(update_data)
                try:
                    capture_server_origin_mutation(
                        sync_service,
                        user_id=str(current_user.id),
                        domain="notes.note",
                        operation="upsert",
                        object_id=note_id,
                        payload=_note_payload_from_row(projected_note),
                        source="server_api",
                    )
                except Exception as sync_exc:
                    raise _note_sync_http_error(sync_exc) from sync_exc
            else:
                success = db.update_note(
                    note_id=note_id,
                    update_data=update_data,
                    expected_version=expected_version
                )
                if not success:
                    raise CharactersRAGDBError("Note update reported non-success without specific exception.")

        keyword_sync_summary: dict[str, Any] | None = None
        if keywords_supplied:
            keyword_sync_summary = _sync_note_keywords(db, note_id=note_id, keywords=kw_list or [])

        updated_note_data = db.get_note_by_id(note_id=note_id)
        if not updated_note_data:
            logger.error(f"Note '{note_id}' not found after successful update for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found after update.")
        _reconcile_note_tasks_after_save(
            db=db,
            note_data=updated_note_data,
            current_user=current_user,
            task_service=task_service,
        )
        updated_note_data = _attach_keywords_inline(db, updated_note_data)
        updated_note_data = _attach_folders_inline(db, updated_note_data)
        if keyword_sync_summary and keyword_sync_summary.get("failed_count", 0) > 0:
            updated_note_data["keyword_sync"] = {
                "failed_count": int(keyword_sync_summary.get("failed_count", 0)),
                "failed_keywords": list(keyword_sync_summary.get("failed_keywords", [])),
            }
        record_note_updated(
            user_id=current_user.id,
            note=updated_note_data,
            route=f"/api/v1/notes/{note_id}",
            action="update",
            patch=raw_data,
        )
        logger.info(
            f"Note '{note_id}' updated successfully for user (DB client_id: {db.client_id}) to version {updated_note_data['version']}.")
        return updated_note_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note")


@router.patch(
    "/{note_id}",
    response_model=NoteResponse,
    summary="Partially update an existing note",
    tags=["notes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def patch_note(
        note_id: str,
        note_in: NoteUpdate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        expected_version: Optional[int] = Header(None, description="Optional expected version for optimistic locking"),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        task_service: NotesTaskService = Depends(get_notes_task_service),
        _: None = Depends(rbac_rate_limit("notes.update")),
):
    """PATCH variant that allows updates without an explicit expected-version header.
    If header is not provided, it fetches current version and applies the update."""
    keywords_supplied = _field_supplied(note_in, "keywords")
    conversation_supplied = _field_supplied(note_in, "conversation_id")
    message_supplied = _field_supplied(note_in, "message_id")
    kw_list = note_in.normalized_keywords if keywords_supplied else None
    raw_data = note_in.model_dump(exclude_unset=True)
    update_data: dict[str, Any] = {}
    if "title" in raw_data and raw_data["title"] is not None:
        update_data["title"] = raw_data["title"]
    if "content" in raw_data and raw_data["content"] is not None:
        update_data["content"] = raw_data["content"]
    if conversation_supplied:
        update_data["conversation_id"] = raw_data.get("conversation_id")
    if message_supplied:
        update_data["message_id"] = raw_data.get("message_id")
    if "title" in update_data and isinstance(update_data["title"], str):
        stripped_title = update_data["title"].strip()
        if not stripped_title:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Title cannot be empty or whitespace.")
        update_data["title"] = stripped_title
    if not update_data and not keywords_supplied:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid fields provided for update.")
    try:
        current_note: Optional[dict[str, Any]] = None

        def _get_current_note() -> dict[str, Any]:
            nonlocal current_note
            if current_note is None:
                current_note = db.get_note_by_id(note_id=note_id)
            if not current_note:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
            return current_note

        if expected_version is None:
            # Fallback to current version if not provided
            current = _get_current_note()
            expected_version = int(current.get("version", 1))
        elif not update_data and keywords_supplied:
            current = _get_current_note()
            current_version = current.get("version")
            if current_version is not None and int(current_version) != int(expected_version):
                raise ConflictError(
                    f"Note ID {note_id} update failed: version mismatch (db has {current_version}, client expected {expected_version}).",
                    entity="notes",
                    entity_id=note_id,
                )

        # Rate limit: notes.update
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.update",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        if conversation_supplied or message_supplied:
            current = _get_current_note()
            current_conversation_id = current.get("conversation_id")
            effective_conversation_id = update_data.get("conversation_id") if conversation_supplied else current_conversation_id
            if message_supplied:
                effective_message_id = update_data.get("message_id")
            else:
                # Preserve the existing message_id so a conversation change is validated
                # against the current message linkage rather than silently clearing it.
                effective_message_id = current.get("message_id")
            validated_conversation_id, validated_message_id = _validate_note_links(
                db,
                effective_conversation_id,
                effective_message_id,
            )
            if conversation_supplied:
                update_data["conversation_id"] = validated_conversation_id
            if message_supplied:
                update_data["message_id"] = validated_message_id
        data_keys = list(update_data.keys())
        if keywords_supplied:
            data_keys.append("keywords")
        logger.info(
            f"User (DB client_id: {db.client_id}) partially updating note: ID='{note_id}', Version={expected_version}, DataKeys={data_keys}")
        sync_service = _active_notes_sync_service(current_user)
        if sync_service is not None and keywords_supplied:
            raise _note_keywords_sync_unsupported_error()
        if update_data:
            if sync_service is not None:
                current = _get_current_note()
                current_version = current.get("version")
                if current_version is not None and int(current_version) != int(expected_version):
                    raise ConflictError(
                        f"Note ID {note_id} update failed: version mismatch (db has {current_version}, client expected {expected_version}).",
                        entity="notes",
                        entity_id=note_id,
                    )
                payload = _note_payload_from_row(current)
                payload.update(update_data)
                try:
                    capture_server_origin_mutation(
                        sync_service,
                        user_id=str(current_user.id),
                        domain="notes.note",
                        operation="upsert",
                        object_id=note_id,
                        payload=payload,
                        source="server_api",
                    )
                except Exception as sync_exc:
                    raise _note_sync_http_error(sync_exc) from sync_exc
            else:
                success = db.update_note(
                    note_id=note_id,
                    update_data=update_data,
                    expected_version=expected_version
                )
                if not success:
                    raise CharactersRAGDBError("Note update reported non-success without specific exception.")

        keyword_sync_summary: dict[str, Any] | None = None
        if keywords_supplied:
            keyword_sync_summary = _sync_note_keywords(db, note_id=note_id, keywords=kw_list or [])

        updated_note_data = db.get_note_by_id(note_id=note_id)
        if not updated_note_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found after update.")
        _reconcile_note_tasks_after_save(
            db=db,
            note_data=updated_note_data,
            current_user=current_user,
            task_service=task_service,
        )
        updated_note_data = _attach_keywords_inline(db, updated_note_data)
        updated_note_data = _attach_folders_inline(db, updated_note_data)
        if keyword_sync_summary and keyword_sync_summary.get("failed_count", 0) > 0:
            updated_note_data["keyword_sync"] = {
                "failed_count": int(keyword_sync_summary.get("failed_count", 0)),
                "failed_keywords": list(keyword_sync_summary.get("failed_keywords", [])),
            }
        record_note_updated(
            user_id=current_user.id,
            note=updated_note_data,
            route=f"/api/v1/notes/{note_id}",
            action="patch",
            patch=raw_data,
        )
        return updated_note_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note")


@router.delete(
    "/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft-delete a note",
    tags=["notes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def delete_note(
        note_id: str,
        expected_version: int = Header(..., description="The expected version of the note for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.delete")),
) -> Response:
    try:
        existing_note = db.get_note_by_id(note_id=note_id, include_deleted=True)
        was_active = bool(existing_note) and not bool(existing_note.get("deleted"))
        note_for_activity = (
            _attach_folders_inline(db, _attach_keywords_inline(db, dict(existing_note)))
            if was_active and existing_note
            else None
        )
        # Rate limit: notes.delete
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.delete")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.delete",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        logger.info(
            f"User (DB client_id: {db.client_id}) soft-deleting note: ID='{note_id}', Version={expected_version}")
        sync_service = _active_notes_sync_service(current_user)
        if sync_service is not None:
            if not existing_note:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
            current_version = existing_note.get("version")
            if current_version is not None and int(current_version) != int(expected_version):
                raise ConflictError(
                    f"Note ID {note_id} delete failed: version mismatch (db has {current_version}, client expected {expected_version}).",
                    entity="notes",
                    entity_id=note_id,
                )
            try:
                capture_server_origin_mutation(
                    sync_service,
                    user_id=str(current_user.id),
                    domain="notes.note",
                    operation="tombstone",
                    object_id=note_id,
                    payload={
                        "id": note_id,
                        "deleted": True,
                        "client_id": str(current_user.id),
                        "owner_user_id": str(current_user.id),
                    },
                    source="server_api",
                )
            except Exception as sync_exc:
                raise _note_sync_http_error(sync_exc) from sync_exc
        else:
            success = db.soft_delete_note(
                note_id=note_id,
                expected_version=expected_version
            )
            if not success:
                raise CharactersRAGDBError("Note soft delete reported non-success without specific exception.")
        if note_for_activity is not None:
            record_note_deleted(
                user_id=current_user.id,
                note=note_for_activity,
                deleted_version=expected_version + 1,
            )
        logger.info(
            f"Note '{note_id}' soft-deleted successfully (or was already deleted) for user (DB client_id: {db.client_id}).")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note")


@router.post(
    "/{note_id}/restore",
    response_model=NoteResponse,
    summary="Restore a soft-deleted note",
    tags=["notes"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def restore_note(
        note_id: str,
        expected_version: int = Query(..., description="The expected version of the note for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.restore")),
) -> NoteResponse:
    """
    Restores a soft-deleted note.

    Requires the `expected_version` query parameter for optimistic locking.
    Returns the restored note on success.
    """
    try:
        existing_note = db.get_note_by_id(note_id=note_id, include_deleted=True)
        was_deleted = bool(existing_note) and bool(existing_note.get("deleted"))
        # Rate limit: notes.restore
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.restore")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.restore",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})

        logger.info(
            f"User (DB client_id: {db.client_id}) restoring note: ID='{note_id}', Version={expected_version}")

        if _active_notes_sync_service(current_user) is not None:
            raise _note_restore_sync_unsupported_error()

        success = db.restore_note(
            note_id=note_id,
            expected_version=expected_version
        )
        if not success:
            raise CharactersRAGDBError("Note restore reported non-success without specific exception.")

        logger.info(
            f"Note '{note_id}' restored successfully for user (DB client_id: {db.client_id}).")

        # Fetch the restored note to return it
        restored_note = db.get_note_by_id(note_id)
        if not restored_note:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Note '{note_id}' not found after restore.")

        keywords = db.get_keywords_for_note(note_id)
        folders = db.get_note_folders_for_note(note_id)
        if was_deleted:
            restored_note_for_activity = dict(restored_note)
            restored_note_for_activity["keywords"] = list(keywords or [])
            restored_note_for_activity["folders"] = list(folders or [])
            record_note_restored(user_id=current_user.id, note=restored_note_for_activity)
        keyword_responses = [
            KeywordResponse(
                id=kw['id'],
                keyword=kw['keyword'],
                created_at=kw['created_at'],
                last_modified=kw['last_modified'],
                version=kw['version'],
                client_id=kw['client_id'],
                deleted=kw.get('deleted', False),
            )
            for kw in keywords
        ] if keywords else []

        return NoteResponse(
            id=str(restored_note['id']),
            title=restored_note.get('title', ''),
            content=restored_note.get('content', ''),
            created_at=restored_note.get('created_at'),
            last_modified=restored_note.get('last_modified'),
            version=restored_note.get('version', 1),
            client_id=restored_note.get('client_id', ''),
            deleted=bool(restored_note.get('deleted', False)),
            keywords=keyword_responses,
            folders=list(folders or []),
        )
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note")


# --- Keyword Endpoints (related to Notes) ---

@router.post(
    "/title/suggest",
    response_model=TitleSuggestResponse,
    summary="Suggest a title for provided content",
    tags=["notes"],
)
async def suggest_note_title(
        payload: TitleSuggestRequest,
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.title.suggest")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.title.suggest")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.title.suggest",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})

        opts = _build_title_opts(payload)
        title = await asyncio.to_thread(generate_note_title, payload.content, options=opts)
        return TitleSuggestResponse(title=title)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "title suggestion")

@router.post(
    "/bulk",
    response_model=NoteBulkCreateResponse,
    summary="Bulk create notes with optional keywords",
    tags=["notes"],
    dependencies=[Depends(rbac_rate_limit("notes.bulk_create"))]
)
async def bulk_create_notes(
        request: NoteBulkCreateRequest,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        task_service: NotesTaskService = Depends(get_notes_task_service),
):
    results: list[NoteBulkCreateItemResult] = []
    companion_events: list[dict[str, Any]] = []
    created = 0
    failed = 0
    # Enforce centralized per-request rate limit (notes.bulk_create)
    try:
        allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.bulk_create")
    except _NOTES_NONCRITICAL_EXCEPTIONS:
        allowed, meta = True, {}
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded for notes.bulk_create",
                            headers={"Retry-After": str(meta.get("retry_after", 60))})

    sync_service = _active_notes_sync_service(current_user)
    if sync_service is not None and any(_field_supplied(item, "keywords") for item in request.notes):
        raise _note_keywords_sync_unsupported_error()

    for item in request.notes:
        try:
            # Compute title per item
            effective_title = (getattr(item, 'title', None) or "").strip()
            if not effective_title:
                if getattr(item, "auto_title", False):
                    try:
                        opts = _build_title_opts(item)
                        effective_title = await asyncio.to_thread(
                            generate_note_title,
                            item.content,
                            options=opts,
                        )
                    except _NOTES_NONCRITICAL_EXCEPTIONS as gen_err:
                        logger.warning(f"[Bulk] Auto-title generation failed, falling back: {gen_err}")
                        effective_title = await asyncio.to_thread(generate_note_title, item.content)
                else:
                    raise InputError("Title is required for bulk item unless auto_title=true.")

            conversation_id, message_id = _validate_note_links(
                db,
                item.conversation_id,
                item.message_id,
            )

            note_id = item.id or str(uuid4())
            if sync_service is not None:
                try:
                    capture_server_origin_mutation(
                        sync_service,
                        user_id=str(current_user.id),
                        domain="notes.note",
                        operation="upsert",
                        object_id=note_id,
                        payload={
                            "title": effective_title,
                            "content": item.content,
                            "conversation_id": conversation_id,
                            "message_id": message_id,
                            "client_id": str(current_user.id),
                            "owner_user_id": str(current_user.id),
                        },
                        source="server_api",
                    )
                except Exception as sync_exc:
                    raise _note_sync_http_error(sync_exc) from sync_exc
            else:
                note_id = db.add_note(
                    title=effective_title,
                    content=item.content,
                    note_id=note_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
                if not note_id:
                    raise CharactersRAGDBError("Failed to create note (no ID returned)")

            # Topic monitoring (non-blocking) per item
            try:
                mon = get_topic_monitoring_service()
                uid = getattr(db, 'client_id', None)
                src_id = str(note_id)
                if effective_title:
                    mon.schedule_evaluate_and_alert(
                        user_id=str(uid) if uid else None,
                        text=effective_title,
                        source="notes.bulk_create",
                        scope_type="user",
                        scope_id=str(uid) if uid else None,
                        source_id=src_id,
                    )
                if getattr(item, 'content', None):
                    mon.schedule_evaluate_and_alert(
                        user_id=str(uid) if uid else None,
                        text=item.content,
                        source="notes.bulk_create",
                        scope_type="user",
                        scope_id=str(uid) if uid else None,
                        source_id=src_id,
                    )
            except _NOTES_NONCRITICAL_EXCEPTIONS:
                pass

            # Attach keywords if provided
            if sync_service is None:
                try:
                    kw_list = item.normalized_keywords if hasattr(item, 'normalized_keywords') else None
                    if kw_list:
                        for kw in kw_list:
                            try:
                                kw_row = _get_or_create_keyword_row(db, kw)
                                if kw_row and kw_row.get('id') is not None:
                                    db.link_note_to_keyword(note_id=note_id, keyword_id=int(kw_row['id']))
                            except _NOTES_NONCRITICAL_EXCEPTIONS as kw_err:
                                logger.warning(f"[Bulk] Keyword attach failed for '{kw}' on note {note_id}: {kw_err}")
                except _NOTES_NONCRITICAL_EXCEPTIONS as kw_outer_err:
                    logger.warning(f"[Bulk] Keyword processing issue for note {note_id}: {kw_outer_err}")

            nd = db.get_note_by_id(note_id=note_id)
            if not nd:
                raise CharactersRAGDBError("Created note could not be retrieved.")
            _reconcile_note_tasks_after_save(db=db, note_data=nd, current_user=current_user, task_service=task_service)
            nd = _attach_keywords_inline(db, nd)
            nd = _attach_folders_inline(db, nd)
            companion_events.append(
                build_note_bulk_import_activity(
                    note=nd,
                    operation="bulk_create",
                    route="/api/v1/notes/bulk",
                    surface="api.notes.bulk",
                )
            )
            results.append(NoteBulkCreateItemResult(success=True, note=nd))
            created += 1
        except _NOTES_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Bulk note create failed for title='{getattr(item, 'title', '')}': {e}")
            results.append(NoteBulkCreateItemResult(success=False, error=str(e)))
            failed += 1

    await _run_db_call(
        record_companion_activity_events_bulk,
        user_id=current_user.id,
        events=companion_events,
    )
    response_payload = NoteBulkCreateResponse(results=results, created_count=created, failed_count=failed)
    response_status = status.HTTP_200_OK if failed == 0 else status.HTTP_207_MULTI_STATUS
    return JSONResponse(content=jsonable_encoder(response_payload), status_code=response_status)


@router.post(
    "/keywords/",
    response_model=KeywordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new keyword",
    tags=["Keywords (for Notes)"]
)
async def create_keyword(
        keyword_in: KeywordCreate,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.create")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.create")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.create",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        logger.info(f"User (DB client_id: {db.client_id}) creating keyword: Text='{keyword_in.keyword}'")
        keyword_id = db.add_keyword(keyword_text=keyword_in.keyword)
        if keyword_id is None:
            raise CharactersRAGDBError("Keyword creation failed to return an ID.")

        created_keyword_data = db.get_keyword_by_id(keyword_id=keyword_id)
        if not created_keyword_data:
            logger.error(
                f"Failed to retrieve keyword '{keyword_id}' after creation for user (DB client_id: {db.client_id}).")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Keyword created but could not be retrieved.")
        logger.info(f"Keyword '{keyword_id}' created successfully for user (DB client_id: {db.client_id}).")
        return created_keyword_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/{keyword_id}",
    response_model=KeywordResponse,
    summary="Get a keyword by its ID",
    tags=["Keywords (for Notes)"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_keyword(
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.get")),
):
    logger.debug(f"User (DB client_id: {db.client_id}) fetching keyword by ID: {keyword_id}")
    try: # Added try block
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.get")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.get",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        keyword_data = db.get_keyword_by_id(keyword_id=keyword_id)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keyword")
        return

    if not keyword_data:
        logger.warning(f"Keyword ID '{keyword_id}' not found for user (DB client_id: {db.client_id}).")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found")
    return keyword_data


@router.get(
    "/keywords/text/{keyword_text}",
    response_model=KeywordResponse,
    summary="Get a keyword by its text content",
    tags=["Keywords (for Notes)"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_keyword_by_text(
        keyword_text: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.get")),
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching keyword by text: '{keyword_text}'")
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.get")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.get",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        keyword_data = db.get_keyword_by_text(keyword_text=keyword_text)
        if not keyword_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found")
        return keyword_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/",
    response_model=list[KeywordResponse],
    summary="List all keywords for the current user",
    tags=["Keywords (for Notes)"]
)
async def list_keywords_endpoint(  # Renamed to avoid conflict
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        include_note_counts: bool = Query(
            False,
            description="If true, include the active note count linked to each keyword"
        ),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.list")),
):
    return await _list_keywords_impl(
        db=db,
        limit=limit,
        offset=offset,
        include_note_counts=include_note_counts,
        rate_limiter=rate_limiter,
        current_user=current_user,
    )


@router.patch(
    "/keywords/{keyword_id}",
    response_model=KeywordResponse,
    summary="Rename a keyword",
    tags=["Keywords (for Notes)"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse},
    },
)
async def rename_keyword(
        keyword_id: int,
        keyword_in: KeywordUpdate,
        expected_version: int = Header(..., description="Expected keyword version for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.update")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.update")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for keywords.update",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        logger.info(
            "User (DB client_id: {}) renaming keyword {} to '{}'",
            db.client_id,
            keyword_id,
            keyword_in.keyword,
        )
        return db.rename_keyword(
            keyword_id=keyword_id,
            new_keyword_text=keyword_in.keyword,
            expected_version=expected_version,
        )
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keyword")


@router.post(
    "/keywords/{keyword_id}/merge",
    response_model=KeywordMergeResponse,
    summary="Merge keyword into another keyword",
    tags=["Keywords (for Notes)"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse},
    },
)
async def merge_keyword(
        keyword_id: int,
        merge_in: KeywordMergeRequest,
        expected_version: int = Header(..., description="Expected source keyword version for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.merge")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.merge")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for keywords.merge",
                headers={"Retry-After": str(meta.get("retry_after", 60))}
            )
        logger.info(
            "User (DB client_id: {}) merging keyword {} into {}",
            db.client_id,
            keyword_id,
            merge_in.target_keyword_id,
        )
        return db.merge_keywords(
            source_keyword_id=keyword_id,
            target_keyword_id=merge_in.target_keyword_id,
            expected_source_version=expected_version,
            expected_target_version=merge_in.expected_target_version,
        )
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keyword")


@router.delete(
    "/keywords/{keyword_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft-delete a keyword",
    tags=["Keywords (for Notes)"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": DetailResponse},
        status.HTTP_409_CONFLICT: {"model": DetailResponse}
    }
)
async def delete_keyword(
        keyword_id: int,
        expected_version: int = Header(..., description="The expected version of the keyword for optimistic locking"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.delete")),
) -> Response:
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.delete")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.delete",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        logger.info(
            f"User (DB client_id: {db.client_id}) soft-deleting keyword: ID='{keyword_id}', Version={expected_version}")
        success = db.soft_delete_keyword(
            keyword_id=keyword_id,
            expected_version=expected_version
        )
        if not success:
            raise CharactersRAGDBError("Keyword soft delete reported non-success without specific exception.")
        logger.info(
            f"Keyword '{keyword_id}' soft-deleted successfully (or was already deleted) for user (DB client_id: {db.client_id}).")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keyword")


@router.get(
    "/keywords/search/",
    response_model=list[KeywordResponse],
    summary="Search keywords for the current user",
    tags=["Keywords (for Notes)"]
)
async def search_keywords_endpoint(  # Renamed
        query: str = Query(..., min_length=1, description="Search term for keywords"),
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(10, ge=1, le=100),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.search")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.search")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.search",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        logger.debug(f"User (DB client_id: {db.client_id}) searching keywords: query='{query}', limit={limit}")
        keywords_data = db.search_keywords(search_term=query, limit=limit)
        return keywords_data
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keywords search")


# --- Note-Keyword Linking Endpoints ---
@router.post(
    "/{note_id}/keywords/{keyword_id}",
    response_model=NoteKeywordLinkResponse,
    summary="Link a note to a keyword",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def link_note_to_keyword_endpoint(
        note_id: str,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.link_keyword")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.link_keyword")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.link_keyword",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        if _active_notes_sync_service(current_user) is not None:
            raise _note_keywords_sync_unsupported_error()
        logger.info(f"User (DB client_id: {db.client_id}) linking note '{note_id}' to keyword '{keyword_id}'")
        # Check if note and keyword exist in the user's DB
        note_data = db.get_note_by_id(note_id)
        if not note_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note with ID '{note_id}' not found.")
        keyword_data = db.get_keyword_by_id(keyword_id)
        if not keyword_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Keyword with ID '{keyword_id}' not found.")

        success = db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)
        msg = "Note linked to keyword successfully." if success else "Link already exists or was created."
        return NoteKeywordLinkResponse(success=True, message=msg)  # True even if already exists
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note-keyword link")


@router.delete(
    "/{note_id}/keywords/{keyword_id}",
    response_model=NoteKeywordLinkResponse,
    summary="Unlink a note from a keyword",
    tags=["Notes Linking"]
)
async def unlink_note_from_keyword_endpoint(
        note_id: str,
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.unlink_keyword")),
):
    try:
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.unlink_keyword")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.unlink_keyword",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        if _active_notes_sync_service(current_user) is not None:
            raise _note_keywords_sync_unsupported_error()
        logger.info(f"User (DB client_id: {db.client_id}) unlinking note '{note_id}' from keyword '{keyword_id}'")
        success = db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)
        msg = "Note unlinked from keyword successfully." if success else "Link not found or no action taken."
        return NoteKeywordLinkResponse(success=success, message=msg)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "note-keyword unlink")


@router.get(
    "/{note_id}/keywords/",
    response_model=KeywordsForNoteResponse,
    summary="Get all keywords linked to a note",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_keywords_for_note_endpoint(
        note_id: str,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("notes.keywords.list")),
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching keywords for note '{note_id}'")
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "notes.keywords.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for notes.keywords.list",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        note_check = db.get_note_by_id(note_id=note_id)
        if not note_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note with ID '{note_id}' not found.")

        keywords_list = db.get_keywords_for_note(note_id=note_id)
        return KeywordsForNoteResponse(note_id=note_id, keywords=keywords_list)
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "keywords for note")


@router.get(
    "/keywords/{keyword_id}/notes/",
    response_model=NotesForKeywordResponse,
    summary="Get all notes linked to a keyword",
    tags=["Notes Linking"],
    responses={status.HTTP_404_NOT_FOUND: {"model": DetailResponse}}
)
async def get_notes_for_keyword_endpoint(
        keyword_id: int,
        db: CharactersRAGDB = Depends(get_chacha_db_for_user),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
        current_user: User = Depends(get_request_user),
        _: None = Depends(rbac_rate_limit("keywords.notes.list")),
):
    try:
        logger.debug(f"User (DB client_id: {db.client_id}) fetching notes for keyword '{keyword_id}'")
        try:
            allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), "keywords.notes.list")
        except _NOTES_NONCRITICAL_EXCEPTIONS:
            allowed, meta = True, {}
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail="Rate limit exceeded for keywords.notes.list",
                                headers={"Retry-After": str(meta.get("retry_after", 60))})
        keyword_check = db.get_keyword_by_id(keyword_id=keyword_id)
        if not keyword_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Keyword with ID '{keyword_id}' not found.")

        notes_list = db.get_notes_for_keyword(keyword_id=keyword_id, limit=limit, offset=offset)
        note_counts = db.get_note_counts_for_keywords([keyword_id])
        total = int(note_counts.get(int(keyword_id), len(notes_list)))
        pagination = build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(notes_list),
        )
        return NotesForKeywordResponse(
            keyword_id=keyword_id,
            notes=notes_list,
            count=len(notes_list),
            limit=limit,
            offset=offset,
            total=total,
            pagination=pagination,
        )
    except HTTPException:
        raise
    except _NOTES_NONCRITICAL_EXCEPTIONS as e:
        handle_db_errors(e, "notes for keyword")

#
# --- End of Notes and Keywords Endpoints ---
########################################################################################################################
