"""Preview helpers for hydrating OpenWebUI attachment references."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tldw_Server_API.app.core.DB_Management.OpenWebUI_DB import (
    load_openwebui_chat_file_rows_for_chats,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.config import get_ingestion_source_allowed_roots


MAX_PREVIEW_WARNING_ITEMS = 1000


@dataclass(frozen=True)
class OpenWebUIDataRoot:
    """Validated server-local OpenWebUI data root paths."""

    root_path: Path
    webui_db_path: Path
    uploads_path: Path


@dataclass(frozen=True)
class OpenWebUIHydrationScope:
    """Imported tldw scope to scan for preserved OpenWebUI references."""

    conversation_ids: tuple[str, ...] = ()
    openwebui_user_id: str | None = None


@dataclass(frozen=True)
class OpenWebUIHydrationReference:
    """One OpenWebUI file reference found on an imported message."""

    conversation_id: str
    message_id: str | None
    file_id: str
    raw_ref_index: int | None
    raw_ref: Any
    source: str
    source_chat_id: str | None = None
    source_message_id: str | None = None


@dataclass(frozen=True)
class OpenWebUIHydrationResolvedFile:
    """Path-resolution result for one OpenWebUI file row."""

    file_id: str | None
    filename: str | None
    path: Path | None
    status: str
    source: str | None = None
    file_kind: str | None = None
    mime_type: str | None = None
    warning_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenWebUIHydrationPreviewItem:
    """User-safe preview item for a reference warning or resolution state."""

    conversation_id: str | None
    message_id: str | None
    file_id: str | None
    status: str
    warning_code: str | None = None
    raw_ref_index: int | None = None
    source: str | None = None
    raw_ref_shape: str | None = None


@dataclass(frozen=True)
class OpenWebUIHydrationPreview:
    """Preview summary for OpenWebUI attachment hydration."""

    references: tuple[OpenWebUIHydrationReference, ...] = ()
    items: tuple[OpenWebUIHydrationPreviewItem, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenWebUIHydrationResult:
    """Final result placeholder for later hydration stages."""

    items: tuple[OpenWebUIHydrationPreviewItem, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_openwebui_data_root(root: str | Path, *, require_uploads: bool = False) -> OpenWebUIDataRoot:
    """Validate that an OpenWebUI data root is under configured allowed roots."""
    candidate = Path(root).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved_candidate = candidate.resolve(strict=False)

    allowed_roots = get_ingestion_source_allowed_roots(reload=True)
    if not allowed_roots:
        raise ValueError("No ingestion source allowed roots are configured.")

    safe_root: Path | None = None
    for allowed_root in allowed_roots:
        safe_root = resolve_safe_local_path(resolved_candidate, allowed_root)
        if safe_root is not None:
            break
    if safe_root is None:
        raise ValueError("OpenWebUI data root must resolve under one of the configured allowed roots.")
    if not safe_root.exists():
        raise ValueError("OpenWebUI data root does not exist.")
    if not safe_root.is_dir():
        raise ValueError("OpenWebUI data root is not a directory.")

    webui_db_path = resolve_safe_local_path(safe_root / "webui.db", safe_root)
    if webui_db_path is None or not webui_db_path.is_file():
        raise ValueError("OpenWebUI data root must contain webui.db.")

    uploads_path = resolve_safe_local_path(safe_root / "uploads", safe_root)
    if uploads_path is None:
        raise ValueError("OpenWebUI uploads path is not safe.")
    if require_uploads and not uploads_path.is_dir():
        raise ValueError("OpenWebUI data root must contain uploads when file bytes are needed.")

    return OpenWebUIDataRoot(
        root_path=safe_root,
        webui_db_path=webui_db_path,
        uploads_path=uploads_path,
    )


def resolve_openwebui_file_path(
    file_row: Mapping[str, Any],
    data_root: OpenWebUIDataRoot,
) -> OpenWebUIHydrationResolvedFile:
    """Resolve an OpenWebUI file row to a safe local source file path."""
    file_id = _row_text(file_row, "id")
    filename = _row_text(file_row, "filename")
    raw_path = _row_text(file_row, "path")

    if raw_path:
        resolved_from_path = _resolve_declared_file_path(raw_path, data_root)
        if resolved_from_path is None:
            return OpenWebUIHydrationResolvedFile(
                file_id=file_id,
                filename=filename,
                path=None,
                status="path_rejected",
                warning_codes=("path_rejected",),
            )
        if resolved_from_path.is_file():
            file_kind, mime_type = _classify_file_path(resolved_from_path)
            return OpenWebUIHydrationResolvedFile(
                file_id=file_id,
                filename=filename,
                path=resolved_from_path,
                status="resolved",
                source="file_path",
                file_kind=file_kind,
                mime_type=mime_type,
            )

    fallback = _resolve_uploads_id_filename_fallback(file_id=file_id, filename=filename, data_root=data_root)
    if fallback is not None:
        file_kind, mime_type = _classify_file_path(fallback)
        return OpenWebUIHydrationResolvedFile(
            file_id=file_id,
            filename=filename,
            path=fallback,
            status="resolved",
            source="uploads_id_filename",
            file_kind=file_kind,
            mime_type=mime_type,
        )

    return OpenWebUIHydrationResolvedFile(
        file_id=file_id,
        filename=filename,
        path=None,
        status="missing_file",
        warning_codes=("missing_file",),
    )


def extract_openwebui_hydration_references(
    chacha_db: Any,
    scope: OpenWebUIHydrationScope,
    *,
    openwebui_conn: sqlite3.Connection | None = None,
) -> OpenWebUIHydrationPreview:
    """Extract preserved OpenWebUI file references from imported message metadata."""
    references: list[OpenWebUIHydrationReference] = []
    items: list[OpenWebUIHydrationPreviewItem] = []
    seen_keys: set[tuple[str, str | None, str, str]] = set()

    for conversation_id in scope.conversation_ids:
        messages = _load_conversation_messages(chacha_db, conversation_id)
        message_ids = [str(message["id"]) for message in messages if message.get("id") is not None]
        metadata_by_message_id = _load_message_metadata_map(chacha_db, message_ids)
        source_message_to_tldw_id: dict[str, str] = {}

        for message_id in message_ids:
            import_meta = _openwebui_import_metadata(metadata_by_message_id.get(message_id))
            if not import_meta:
                continue
            source_message_id = _coerce_optional_text(import_meta.get("source_message_id"))
            if source_message_id:
                source_message_to_tldw_id[source_message_id] = message_id
            raw_refs = import_meta.get("attachment_refs")
            if not isinstance(raw_refs, list):
                raw_refs = []
            for raw_ref_index, raw_ref in enumerate(raw_refs):
                file_id = _file_id_from_reference(raw_ref)
                if file_id is None:
                    _append_preview_warning(
                        items,
                        OpenWebUIHydrationPreviewItem(
                            conversation_id=conversation_id,
                            message_id=message_id,
                            file_id=None,
                            status="unsupported_reference_shape",
                            warning_code="unsupported_reference_shape",
                            raw_ref_index=raw_ref_index,
                            source="message_metadata",
                            raw_ref_shape=type(raw_ref).__name__,
                        ),
                    )
                    continue
                key = (conversation_id, message_id, file_id, "message_metadata")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                references.append(
                    OpenWebUIHydrationReference(
                        conversation_id=conversation_id,
                        message_id=message_id,
                        file_id=file_id,
                        raw_ref_index=raw_ref_index,
                        raw_ref=raw_ref,
                        source="message_metadata",
                        source_message_id=source_message_id,
                    )
                )

        if openwebui_conn is not None:
            _extend_references_from_chat_file_fallback(
                references=references,
                seen_keys=seen_keys,
                openwebui_conn=openwebui_conn,
                chacha_db=chacha_db,
                conversation_id=conversation_id,
                source_message_to_tldw_id=source_message_to_tldw_id,
                openwebui_user_id=scope.openwebui_user_id,
            )

    return OpenWebUIHydrationPreview(
        references=tuple(references),
        items=tuple(items),
    )


def _resolve_declared_file_path(raw_path: str, data_root: OpenWebUIDataRoot) -> Path | None:
    candidate = Path(raw_path)
    roots = (data_root.root_path,)
    if candidate.is_absolute() or not candidate.parts or candidate.parts[0] != "uploads":
        roots = (data_root.root_path, data_root.uploads_path)
    for base_dir in roots:
        resolved = resolve_safe_local_path(candidate, base_dir)
        if resolved is not None:
            return resolved
    return None


def _resolve_uploads_id_filename_fallback(
    *,
    file_id: str | None,
    filename: str | None,
    data_root: OpenWebUIDataRoot,
) -> Path | None:
    if not file_id or not filename:
        return None
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename in {".", ".."}:
        return None
    fallback = data_root.uploads_path / f"{file_id}_{safe_filename}"
    resolved = resolve_safe_local_path(fallback, data_root.uploads_path)
    if resolved is None or not resolved.is_file():
        return None
    return resolved


def _classify_file_path(path: Path) -> tuple[str, str | None]:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return "unknown", None

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image", "image/gif"
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image", "image/webp"
    if header.startswith(b"%PDF"):
        return "document", "application/pdf"
    return "file", None


def _load_conversation_messages(chacha_db: Any, conversation_id: str) -> list[dict[str, Any]]:
    messages = chacha_db.get_messages_for_conversation(
        conversation_id,
        limit=1000,
        offset=0,
        order_by_timestamp="ASC",
        include_deleted=False,
    )
    return [message for message in messages if isinstance(message, dict)]


def _load_message_metadata_map(chacha_db: Any, message_ids: list[str]) -> dict[str, dict[str, Any]]:
    if hasattr(chacha_db, "get_message_metadata_map"):
        metadata_map = chacha_db.get_message_metadata_map(message_ids)
        if isinstance(metadata_map, dict):
            return metadata_map
    metadata_by_message_id: dict[str, dict[str, Any]] = {}
    if not hasattr(chacha_db, "get_message_metadata"):
        return metadata_by_message_id
    for message_id in message_ids:
        metadata = chacha_db.get_message_metadata(message_id)
        if isinstance(metadata, dict):
            metadata_by_message_id[message_id] = metadata
    return metadata_by_message_id


def _openwebui_import_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    extra = metadata.get("extra")
    if not isinstance(extra, dict):
        return None
    openwebui_import = extra.get("openwebui_import")
    if isinstance(openwebui_import, dict):
        return openwebui_import
    return None


def _extend_references_from_chat_file_fallback(
    *,
    references: list[OpenWebUIHydrationReference],
    seen_keys: set[tuple[str, str | None, str, str]],
    openwebui_conn: sqlite3.Connection,
    chacha_db: Any,
    conversation_id: str,
    source_message_to_tldw_id: dict[str, str],
    openwebui_user_id: str | None,
) -> None:
    source_chat_id = _source_chat_id_from_conversation_settings(chacha_db, conversation_id)
    if not source_chat_id:
        return
    rows = load_openwebui_chat_file_rows_for_chats(
        openwebui_conn,
        [source_chat_id],
        user_id=openwebui_user_id,
    )
    for row in rows:
        file_id = _row_text(row, "file_id")
        if not file_id:
            continue
        source_message_id = _row_text(row, "message_id")
        message_id = source_message_to_tldw_id.get(source_message_id) if source_message_id else None
        if source_message_id and message_id is None:
            continue
        key = (conversation_id, message_id, file_id, "chat_file")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        references.append(
            OpenWebUIHydrationReference(
                conversation_id=conversation_id,
                message_id=message_id,
                file_id=file_id,
                raw_ref_index=None,
                raw_ref=dict(row),
                source="chat_file",
                source_chat_id=source_chat_id,
                source_message_id=source_message_id,
            )
        )


def _source_chat_id_from_conversation_settings(chacha_db: Any, conversation_id: str) -> str | None:
    if not hasattr(chacha_db, "get_conversation_settings"):
        return None
    settings_record = chacha_db.get_conversation_settings(conversation_id)
    if not isinstance(settings_record, dict):
        return None
    settings = settings_record.get("settings")
    if not isinstance(settings, dict):
        return None
    import_meta = settings.get("openwebui_import")
    if not isinstance(import_meta, dict):
        return None
    metadata = import_meta.get("metadata")
    if isinstance(metadata, dict):
        row_id = _coerce_optional_text(metadata.get("row_id"))
        if row_id:
            return row_id
    return None


def _file_id_from_reference(raw_ref: Any) -> str | None:
    if isinstance(raw_ref, str):
        return _coerce_optional_text(raw_ref)
    if isinstance(raw_ref, dict):
        for key in ("id", "file_id", "fileId"):
            file_id = _coerce_optional_text(raw_ref.get(key))
            if file_id:
                return file_id
    return None


def _append_preview_warning(
    items: list[OpenWebUIHydrationPreviewItem],
    item: OpenWebUIHydrationPreviewItem,
) -> None:
    if len(items) < MAX_PREVIEW_WARNING_ITEMS:
        items.append(item)


def _row_text(row: Mapping[str, Any], key: str) -> str | None:
    try:
        value = row[key]
    except (KeyError, IndexError):
        return None
    return _coerce_optional_text(value)


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
