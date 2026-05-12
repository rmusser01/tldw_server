"""Preview helpers for hydrating OpenWebUI attachment references."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from tldw_Server_API.app.core.DB_Management.OpenWebUI_DB import (
    load_openwebui_chat_file_rows_for_chats,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import (
    open_safe_local_path,
    resolve_safe_local_path,
)
from tldw_Server_API.app.core.config import get_ingestion_source_allowed_roots


MAX_PREVIEW_WARNING_ITEMS = 1000


class _OpenWebUIHydrationMetadataError(RuntimeError):
    """Raised internally to roll back image hydration when metadata cannot be recorded."""


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
    job_id: str | None = None
    source_key: str | None = None
    message_image_position: int | None = None
    mime_type: str | None = None
    media_id: int | None = None
    media_file_id: str | None = None
    checksum: str | None = None
    storage_path: str | None = None
    processing_status: str | None = None


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


def classify_openwebui_file(path: Path) -> tuple[str, str | None]:
    """Classify a resolved OpenWebUI source file from conservative byte signatures."""
    return _classify_file_path(path)


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
    chat_file_contexts: dict[str, list[tuple[str, dict[str, str]]]] = {}

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
            source_chat_id = _source_chat_id_from_conversation_settings(chacha_db, conversation_id)
            if source_chat_id:
                chat_file_contexts.setdefault(source_chat_id, []).append(
                    (conversation_id, source_message_to_tldw_id)
                )

    if openwebui_conn is not None and chat_file_contexts:
        _extend_references_from_chat_file_fallback(
            references=references,
            seen_keys=seen_keys,
            openwebui_conn=openwebui_conn,
            chat_file_contexts=chat_file_contexts,
            openwebui_user_id=scope.openwebui_user_id,
        )

    return OpenWebUIHydrationPreview(
        references=tuple(references),
        items=tuple(items),
    )


def merge_openwebui_message_hydration_metadata(
    chacha_db: Any,
    message_id: str,
    item_update: dict[str, Any],
    *,
    job_id: str | None = None,
) -> bool:
    """Deep-merge one hydration item into message metadata without replacing import metadata."""
    current = chacha_db.get_message_metadata(message_id) or {}
    current_extra = current.get("extra") if isinstance(current, dict) else {}
    if not isinstance(current_extra, dict):
        current_extra = {}
    openwebui_import = current_extra.get("openwebui_import")
    if not isinstance(openwebui_import, dict):
        openwebui_import = {}
    merged_openwebui_import = dict(openwebui_import)

    hydration = merged_openwebui_import.get("hydration")
    if not isinstance(hydration, dict):
        hydration = {}
    merged_hydration = dict(hydration)
    items = merged_hydration.get("items")
    if not isinstance(items, list):
        items = []
    merged_items = [dict(item) for item in items if isinstance(item, dict)]

    source_key = _coerce_optional_text(item_update.get("source_key"))
    replaced = False
    if source_key:
        for index, existing_item in enumerate(merged_items):
            if existing_item.get("source_key") == source_key:
                merged_items[index] = {**existing_item, **item_update}
                replaced = True
                break
    if not replaced:
        merged_items.append(dict(item_update))

    merged_hydration["version"] = 1
    if job_id is not None:
        merged_hydration["last_job_id"] = job_id
    merged_hydration["items"] = merged_items
    merged_openwebui_import["hydration"] = merged_hydration

    new_extra = dict(current_extra)
    new_extra["openwebui_import"] = merged_openwebui_import
    return bool(
        chacha_db.add_message_metadata(
            message_id,
            tool_calls=current.get("tool_calls") if isinstance(current, dict) else None,
            extra=new_extra,
        )
    )


def hydrate_image_reference(
    chacha_db: Any,
    reference: OpenWebUIHydrationReference,
    resolved_file: OpenWebUIHydrationResolvedFile,
    *,
    job_id: str | None = None,
    max_image_bytes: int | None = None,
) -> OpenWebUIHydrationPreviewItem:
    """Hydrate one resolved image reference into a tldw message image slot."""
    message_id = reference.message_id
    if not message_id:
        return _image_result_item(reference, "message_missing", job_id=job_id)
    if resolved_file.path is None or resolved_file.status != "resolved":
        return _image_result_item(reference, resolved_file.status, job_id=job_id)

    image_limit = max_image_bytes if max_image_bytes is not None else _default_max_image_bytes()
    try:
        handle = open_safe_local_path(resolved_file.path, resolved_file.path.parent, mode="rb")
        if handle is None:
            return _image_result_item(reference, "missing_file", job_id=job_id)
        with handle:
            image_bytes = handle.read(image_limit + 1)
    except OSError:
        return _image_result_item(reference, "missing_file", job_id=job_id)

    if len(image_bytes) > image_limit:
        return _image_result_item(reference, "oversized", job_id=job_id)

    mime_type = _sniff_image_mime(image_bytes)
    if mime_type is None:
        return _image_result_item(reference, "unsupported_file_type", job_id=job_id)

    source_key = _source_key_for_reference(reference, image_bytes)
    existing_item = _existing_hydration_item(chacha_db, message_id, source_key)
    if existing_item is not None and existing_item.get("message_image_position") is not None:
        return _image_result_item(
            reference,
            "already_hydrated",
            job_id=job_id,
            source_key=source_key,
            message_image_position=int(existing_item["message_image_position"]),
            mime_type=str(existing_item.get("mime_type") or mime_type),
        )

    item_update = {
        "source_key": source_key,
        "source_file_id": reference.file_id,
        "source_message_id": reference.source_message_id,
        "status": "hydrated_image",
        "message_image_position": None,
        "mime_type": mime_type,
        "job_id": job_id,
    }
    try:
        if hasattr(chacha_db, "transaction"):
            with chacha_db.transaction():
                position = int(
                    chacha_db.append_message_image(
                        message_id,
                        image_bytes,
                        mime_type,
                        commit=False,
                    )
                )
                item_update["message_image_position"] = position
                if not merge_openwebui_message_hydration_metadata(
                    chacha_db,
                    message_id,
                    item_update,
                    job_id=job_id,
                ):
                    raise _OpenWebUIHydrationMetadataError("Failed to record OpenWebUI hydration metadata.")
        else:
            position = int(chacha_db.append_message_image(message_id, image_bytes, mime_type))
            item_update["message_image_position"] = position
            if not merge_openwebui_message_hydration_metadata(
                chacha_db,
                message_id,
                item_update,
                job_id=job_id,
            ):
                return _image_result_item(
                    reference,
                    "metadata_update_failed",
                    job_id=job_id,
                    source_key=source_key,
                    mime_type=mime_type,
                )
    except _OpenWebUIHydrationMetadataError:
        return _image_result_item(
            reference,
            "metadata_update_failed",
            job_id=job_id,
            source_key=source_key,
            mime_type=mime_type,
        )
    return _image_result_item(
        reference,
        "hydrated_image",
        job_id=job_id,
        source_key=source_key,
        message_image_position=position,
        mime_type=mime_type,
    )


def register_non_image_reference(
    chacha_db: Any,
    media_db: Any,
    reference: OpenWebUIHydrationReference,
    resolved_file: OpenWebUIHydrationResolvedFile,
    *,
    owner_user_id: int,
    storage_root: str | Path,
    job_id: str | None = None,
    process_supported_files: bool = False,
    processing_hook: Callable[..., Any] | None = None,
    run_dedupe_cache: MutableMapping[tuple[str, str], int] | None = None,
) -> OpenWebUIHydrationPreviewItem:
    """Register one resolved non-image OpenWebUI attachment in Media DB."""
    message_id = reference.message_id
    if not message_id:
        return _media_result_item(reference, "message_missing", job_id=job_id)
    if resolved_file.path is None or resolved_file.status != "resolved":
        return _media_result_item(reference, resolved_file.status, job_id=job_id)
    if resolved_file.file_kind == "image":
        return _media_result_item(reference, "unsupported_file_type", job_id=job_id)
    if not resolved_file.path.is_file():
        return _media_result_item(reference, "missing_file", job_id=job_id)

    try:
        checksum = _sha256_file(resolved_file.path)
    except Exception:
        return _media_result_item(
            reference,
            "media_registration_failed",
            job_id=job_id,
            source_key=f"openwebui:file:{reference.file_id}" if reference.file_id else None,
            warning_code="media_registration_failed",
            mime_type=resolved_file.mime_type,
        )
    source_key = _source_key_for_digest(reference, checksum)
    existing_item = _existing_hydration_item(chacha_db, message_id, source_key)
    if existing_item is not None and existing_item.get("media_id") is not None:
        return _media_result_item(
            reference,
            "already_registered_media",
            job_id=job_id,
            source_key=source_key,
            media_id=int(existing_item["media_id"]),
            media_file_id=_coerce_optional_text(existing_item.get("media_file_id")),
            checksum=_coerce_optional_text(existing_item.get("checksum")) or checksum,
            storage_path=_coerce_optional_text(existing_item.get("storage_path")),
            mime_type=_coerce_optional_text(existing_item.get("mime_type")) or resolved_file.mime_type,
            processing_status=_coerce_optional_text(existing_item.get("processing_status")),
        )

    source_file_id = _coerce_optional_text(reference.file_id) or _coerce_optional_text(resolved_file.file_id)
    filename = _safe_storage_filename(resolved_file.filename or resolved_file.path.name)
    mime_type = resolved_file.mime_type or _guess_mime_type(filename)
    media_url = _openwebui_media_url(
        owner_user_id=owner_user_id,
        source_file_id=source_file_id,
        checksum=checksum,
        job_id=job_id,
    )
    placeholder_content = _openwebui_placeholder_content(
        source_file_id=source_file_id,
        filename=filename,
        mime_type=mime_type,
        checksum=checksum,
    )
    safe_metadata = _openwebui_safe_metadata(
        reference=reference,
        source_file_id=source_file_id,
        filename=filename,
        mime_type=mime_type,
        checksum=checksum,
        job_id=job_id,
    )

    media_id = _deduped_media_id_from_run_cache(
        owner_user_id=owner_user_id,
        source_file_id=source_file_id,
        checksum=checksum,
        run_dedupe_cache=run_dedupe_cache,
    )
    try:
        if media_id is None:
            media_id, _, _ = media_db.add_media_with_keywords(
                url=media_url,
                title=filename,
                media_type=_media_type_for_file(resolved_file, mime_type, filename),
                content=placeholder_content,
                keywords=["openwebui", "attachment"],
                safe_metadata=json.dumps(safe_metadata, sort_keys=True),
                source_hash=checksum,
                visibility="personal",
                owner_user_id=owner_user_id,
            )
    except Exception:
        return _media_result_item(
            reference,
            "media_registration_failed",
            job_id=job_id,
            source_key=source_key,
            checksum=checksum,
            mime_type=mime_type,
            warning_code="media_registration_failed",
        )
    if media_id is None:
        return _media_result_item(
            reference,
            "media_registration_failed",
            job_id=job_id,
            source_key=source_key,
            checksum=checksum,
            mime_type=mime_type,
        )
    media_id = int(media_id)

    storage_path = None
    media_file_id = None
    try:
        media_file = media_db.get_media_file(media_id, "original")
        status = "already_registered_media" if media_file else "registered_media"
        if media_file:
            media_file_id = str(media_file["uuid"])
            storage_path = str(media_file["storage_path"])
        else:
            storage_path = _copy_openwebui_attachment_to_storage(
                source_path=resolved_file.path,
                storage_root=storage_root,
                owner_user_id=owner_user_id,
                media_id=media_id,
                filename=filename,
            )
            media_file_id = str(
                media_db.insert_media_file(
                    media_id=media_id,
                    file_type="original",
                    storage_path=storage_path,
                    original_filename=filename,
                    file_size=resolved_file.path.stat().st_size,
                    mime_type=mime_type,
                    checksum=checksum,
                )
            )
    except Exception:
        return _media_result_item(
            reference,
            "media_registration_failed",
            job_id=job_id,
            source_key=source_key,
            media_id=media_id,
            checksum=checksum,
            storage_path=storage_path,
            mime_type=mime_type,
            warning_code="media_registration_failed",
        )

    processing_status = "skipped"
    warning_code = None
    if process_supported_files:
        if processing_hook is None:
            processing_status = "not_configured"
        else:
            try:
                processing_hook(
                    media_db=media_db,
                    media_id=media_id,
                    media_file_id=media_file_id,
                    storage_path=storage_path,
                    owner_user_id=owner_user_id,
                )
                processing_status = "completed"
            except Exception:
                processing_status = "failed"
                warning_code = "processing_failed"

    item_update = {
        "source_key": source_key,
        "source_file_id": source_file_id,
        "source_message_id": reference.source_message_id,
        "status": "registered_media",
        "media_id": media_id,
        "media_file_id": media_file_id,
        "checksum": checksum,
        "storage_path": storage_path,
        "mime_type": mime_type,
        "job_id": job_id,
        "processing_status": processing_status,
    }
    if warning_code is not None:
        item_update["warning_code"] = warning_code
    try:
        metadata_saved = merge_openwebui_message_hydration_metadata(
            chacha_db,
            message_id,
            item_update,
            job_id=job_id,
        )
    except Exception:
        metadata_saved = False
    if not metadata_saved:
        return _media_result_item(
            reference,
            "metadata_update_failed",
            job_id=job_id,
            source_key=source_key,
            media_id=media_id,
            media_file_id=media_file_id,
            checksum=checksum,
            storage_path=storage_path,
            mime_type=mime_type,
            processing_status=processing_status,
        )
    if run_dedupe_cache is not None and not source_file_id:
        run_dedupe_cache[(str(owner_user_id), checksum)] = media_id

    return _media_result_item(
        reference,
        status,
        job_id=job_id,
        source_key=source_key,
        media_id=media_id,
        media_file_id=media_file_id,
        checksum=checksum,
        storage_path=storage_path,
        mime_type=mime_type,
        warning_code=warning_code,
        processing_status=processing_status,
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


def _sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _source_key_for_reference(reference: OpenWebUIHydrationReference, data: bytes) -> str:
    return _source_key_for_digest(reference, hashlib.sha256(data).hexdigest())


def _source_key_for_digest(reference: OpenWebUIHydrationReference, digest: str) -> str:
    if reference.file_id:
        return f"openwebui:file:{reference.file_id}"
    return f"openwebui:hash:{digest}"


def _existing_hydration_item(chacha_db: Any, message_id: str, source_key: str) -> dict[str, Any] | None:
    metadata = chacha_db.get_message_metadata(message_id) or {}
    openwebui_import = _openwebui_import_metadata(metadata)
    if not openwebui_import:
        return None
    hydration = openwebui_import.get("hydration")
    if not isinstance(hydration, dict):
        return None
    items = hydration.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("source_key") == source_key:
            return item
    return None


def _image_result_item(
    reference: OpenWebUIHydrationReference,
    status: str,
    *,
    job_id: str | None = None,
    source_key: str | None = None,
    message_image_position: int | None = None,
    mime_type: str | None = None,
) -> OpenWebUIHydrationPreviewItem:
    return OpenWebUIHydrationPreviewItem(
        conversation_id=reference.conversation_id,
        message_id=reference.message_id,
        file_id=reference.file_id,
        status=status,
        warning_code=None if status in {"hydrated_image", "already_hydrated"} else status,
        raw_ref_index=reference.raw_ref_index,
        source=reference.source,
        job_id=job_id,
        source_key=source_key,
        message_image_position=message_image_position,
        mime_type=mime_type,
    )


def _media_result_item(
    reference: OpenWebUIHydrationReference,
    status: str,
    *,
    job_id: str | None = None,
    source_key: str | None = None,
    media_id: int | None = None,
    media_file_id: str | None = None,
    checksum: str | None = None,
    storage_path: str | None = None,
    mime_type: str | None = None,
    warning_code: str | None = None,
    processing_status: str | None = None,
) -> OpenWebUIHydrationPreviewItem:
    return OpenWebUIHydrationPreviewItem(
        conversation_id=reference.conversation_id,
        message_id=reference.message_id,
        file_id=reference.file_id,
        status=status,
        warning_code=warning_code
        if warning_code is not None
        else None
        if status in {"registered_media", "already_registered_media"}
        else status,
        raw_ref_index=reference.raw_ref_index,
        source=reference.source,
        job_id=job_id,
        source_key=source_key,
        mime_type=mime_type,
        media_id=media_id,
        media_file_id=media_file_id,
        checksum=checksum,
        storage_path=storage_path,
        processing_status=processing_status,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    handle = open_safe_local_path(path, path.parent, mode="rb")
    if handle is None:
        raise ValueError("OpenWebUI attachment source path is unsafe.")
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_storage_filename(filename: str | None) -> str:
    safe_name = Path(filename or "attachment").name.strip().strip(".")
    if not safe_name:
        return "attachment"
    return safe_name.replace("/", "_").replace("\\", "_")


def _guess_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".txt", ".md", ".markdown"}:
        return "text/plain"
    if suffix in {".json"}:
        return "application/json"
    return "application/octet-stream"


def _media_type_for_file(
    resolved_file: OpenWebUIHydrationResolvedFile,
    mime_type: str,
    filename: str,
) -> str:
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        return "pdf"
    if resolved_file.file_kind in {"document", "file"}:
        return resolved_file.file_kind
    return "file"


def _openwebui_media_url(
    *,
    owner_user_id: int,
    source_file_id: str | None,
    checksum: str,
    job_id: str | None,
) -> str:
    safe_owner = quote(str(owner_user_id), safe="")
    if source_file_id:
        safe_source_file_id = quote(source_file_id, safe="")
        return f"openwebui://user/{safe_owner}/file/{safe_source_file_id}"
    safe_job_id = quote(job_id or "manual", safe="")
    return f"openwebui://user/{safe_owner}/run/{safe_job_id}/{checksum}"


def _openwebui_placeholder_content(
    *,
    source_file_id: str | None,
    filename: str,
    mime_type: str,
    checksum: str,
) -> str:
    return json.dumps(
        {
            "source": "openwebui",
            "source_file_id": source_file_id,
            "filename": filename,
            "mime_type": mime_type,
            "sha256": checksum,
        },
        sort_keys=True,
    )


def _openwebui_safe_metadata(
    *,
    reference: OpenWebUIHydrationReference,
    source_file_id: str | None,
    filename: str,
    mime_type: str,
    checksum: str,
    job_id: str | None,
) -> dict[str, Any]:
    return {
        "source": "openwebui",
        "source_file_id": source_file_id,
        "source_chat_id": reference.source_chat_id,
        "source_message_id": reference.source_message_id,
        "filename": filename,
        "mime_type": mime_type,
        "sha256": checksum,
        "hydration_job_id": job_id,
    }


def _deduped_media_id_from_run_cache(
    *,
    owner_user_id: int,
    source_file_id: str | None,
    checksum: str,
    run_dedupe_cache: MutableMapping[tuple[str, str], int] | None,
) -> int | None:
    if source_file_id or run_dedupe_cache is None:
        return None
    return run_dedupe_cache.get((str(owner_user_id), checksum))


def _copy_openwebui_attachment_to_storage(
    *,
    source_path: Path,
    storage_root: str | Path,
    owner_user_id: int,
    media_id: int,
    filename: str,
) -> str:
    root = Path(storage_root).expanduser().resolve()
    relative_path = Path(str(owner_user_id)) / "media" / str(media_id) / _safe_storage_filename(filename)
    target_path = resolve_safe_local_path(root / relative_path, root)
    if target_path is None:
        raise ValueError("OpenWebUI attachment storage path is unsafe.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_handle = open_safe_local_path(source_path, source_path.parent, mode="rb")
    if source_handle is None:
        raise ValueError("OpenWebUI attachment source path is unsafe.")
    with source_handle:
        target_handle = open_safe_local_path(target_path, root, mode="wb")
        if target_handle is None:
            raise ValueError("OpenWebUI attachment storage path is unsafe.")
        with target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
    return str(relative_path)


def _default_max_image_bytes() -> int:
    try:
        from tldw_Server_API.app.core.config import settings  # noqa: E402

        return int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
    except (ImportError, AttributeError, TypeError, ValueError):
        return 5 * 1024 * 1024


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
    chat_file_contexts: Mapping[str, list[tuple[str, dict[str, str]]]],
    openwebui_user_id: str | None,
) -> None:
    rows = load_openwebui_chat_file_rows_for_chats(
        openwebui_conn,
        tuple(chat_file_contexts.keys()),
        user_id=openwebui_user_id,
    )
    for row in rows:
        file_id = _row_text(row, "file_id")
        if not file_id:
            continue
        source_chat_id = _row_text(row, "chat_id")
        if not source_chat_id:
            continue
        contexts = chat_file_contexts.get(source_chat_id, [])
        source_message_id = _row_text(row, "message_id")
        for conversation_id, source_message_to_tldw_id in contexts:
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
