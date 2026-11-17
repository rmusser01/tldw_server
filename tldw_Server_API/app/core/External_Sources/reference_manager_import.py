"""Reference-manager import helpers for attachment ingestion and metadata sync."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_document_version,
    get_media_by_hash,
    get_media_repository,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.safe_metadata_write_ops import (
    update_version_safe_metadata_in_transaction,
)
from tldw_Server_API.app.core.External_Sources.connectors_service import (
    get_reference_item_binding,
    upsert_reference_item_binding,
)
from tldw_Server_API.app.core.External_Sources.reference_manager_dedupe import (
    build_metadata_fingerprint,
    rank_reference_item_match,
)
from tldw_Server_API.app.core.External_Sources.reference_manager_types import (
    NormalizedReferenceItem,
    ReferenceAttachmentCandidate,
)
from tldw_Server_API.app.core.Utils.metadata_utils import normalize_safe_metadata
from tldw_Server_API.app.core.exceptions import ReferenceImportError

_REFERENCE_CANONICAL_FIELDS = (
    "provider",
    "import_mode",
    "provider_item_key",
    "provider_library_id",
    "collection_key",
    "collection_name",
    "source_url",
    "doi",
    "title",
    "authors",
    "publication_date",
    "year",
    "journal",
    "abstract",
)


def _coerce_media_id(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _canonical_reference_safe_metadata(item: NormalizedReferenceItem) -> dict[str, Any]:
    canonical = {
        "provider": item.provider,
        "import_mode": item.import_mode,
        "provider_item_key": item.provider_item_key,
        "provider_library_id": item.provider_library_id,
        "collection_key": item.collection_key,
        "collection_name": item.collection_name,
        "source_url": item.source_url,
        "doi": item.doi,
        "title": item.title,
        "authors": item.authors,
        "publication_date": item.publication_date,
        "year": item.year,
        "journal": item.journal,
        "abstract": item.abstract,
    }
    return normalize_safe_metadata(
        {
            key: value
            for key, value in canonical.items()
            if value not in (None, "")
        }
    )


def _build_binding_metadata(
    item: NormalizedReferenceItem,
    *,
    selected_attachment: ReferenceAttachmentCandidate | None,
    safe_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": item.provider,
        "safe_metadata": safe_metadata,
    }
    provider_version = item.metadata.get("provider_version")
    if provider_version not in (None, ""):
        metadata["provider_version"] = provider_version
    if item.metadata:
        metadata["provider_metadata"] = dict(item.metadata)
    if selected_attachment is not None:
        metadata["selected_attachment"] = {
            "attachment_key": selected_attachment.attachment_key,
            "title": selected_attachment.title,
            "source_url": selected_attachment.source_url,
            "mime_type": selected_attachment.mime_type,
            "size_bytes": selected_attachment.size_bytes,
            "metadata": dict(selected_attachment.metadata or {}),
        }
    return metadata


def _carry_forward_selected_attachment(
    binding_metadata: dict[str, Any],
    same_provider_item: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(same_provider_item, dict):
        return binding_metadata
    raw_reference_metadata = same_provider_item.get("raw_reference_metadata")
    if not isinstance(raw_reference_metadata, dict):
        return binding_metadata
    selected_attachment = raw_reference_metadata.get("selected_attachment")
    if not isinstance(selected_attachment, dict) or not selected_attachment:
        return binding_metadata
    preserved_attachment = dict(selected_attachment)
    if isinstance(preserved_attachment.get("metadata"), dict):
        preserved_attachment["metadata"] = dict(preserved_attachment["metadata"])
    binding_metadata["selected_attachment"] = preserved_attachment
    return binding_metadata


def _select_best_attachment(
    attachments: list[ReferenceAttachmentCandidate],
) -> ReferenceAttachmentCandidate | None:
    if not attachments:
        return None
    return max(
        attachments,
        key=lambda attachment: (
            int(attachment.size_bytes or 0),
            len(str(attachment.title or "")),
            str(attachment.attachment_key),
        ),
    )


def _load_safe_metadata(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return dict(raw_value)
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return {}
    try:
        parsed = json.loads(raw_text)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _normalize_existing_safe_metadata_for_merge(existing_safe_metadata: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(existing_safe_metadata or {})
    invalid_key_groups = {
        "Invalid DOI format": ("doi", "DOI"),
        "Invalid PMID format": ("pmid", "PMID"),
        "Invalid PMCID format": ("pmcid", "PMCID"),
    }
    while True:
        try:
            return normalize_safe_metadata(candidate)
        except ValueError as exc:
            error_text = str(exc)
            matched_group = next(
                (
                    keys
                    for prefix, keys in invalid_key_groups.items()
                    if error_text.startswith(prefix)
                ),
                None,
            )
            if not matched_group:
                raise
            removed = False
            for key in matched_group:
                if key in candidate:
                    candidate.pop(key, None)
                    removed = True
            if not removed:
                raise
            logger.warning(
                "Ignoring malformed legacy safe-metadata identifier while merging reference metadata: {}",
                exc,
            )


def _provider_version(item: NormalizedReferenceItem) -> str | None:
    value = item.metadata.get("provider_version")
    return None if value in (None, "") else str(value)


def _metadata_fingerprint_candidate(row: dict[str, Any]) -> str | None:
    safe_metadata = _load_safe_metadata(row.get("safe_metadata"))
    return build_metadata_fingerprint(
        title=safe_metadata.get("title") or row.get("title"),
        authors=safe_metadata.get("authors"),
        year=safe_metadata.get("year"),
    )


def _find_metadata_fingerprint_match(
    media_db: Any,
    item: NormalizedReferenceItem,
) -> dict[str, Any] | None:
    fingerprint = build_metadata_fingerprint(
        title=item.title,
        authors=item.authors,
        year=item.year,
    )
    if not fingerprint or not str(item.title or "").strip():
        return None
    rows, _total = media_db.search_by_safe_metadata(
        text_query=str(item.title or "").strip(),
        per_page=25,
        group_by_media=True,
    )
    for row in rows:
        if _metadata_fingerprint_candidate(row) == fingerprint:
            return {"media_id": row.get("media_id")}
    return None


def _find_doi_match(media_db: Any, item: NormalizedReferenceItem) -> dict[str, Any] | None:
    if not item.doi:
        return None
    rows, _total = media_db.search_by_safe_metadata(
        filters=[{"field": "doi", "op": "eq", "value": item.doi}],
        per_page=1,
        group_by_media=True,
    )
    if not rows:
        return None
    return {"media_id": rows[0].get("media_id")}


def _find_file_hash_match(media_db: Any, content_hash: str | None) -> dict[str, Any] | None:
    if not content_hash:
        return None
    row = get_media_by_hash(media_db, content_hash)
    if not row:
        return None
    return {"media_id": row.get("id")}


def _ingest_reference_attachment(
    media_db: Any,
    *,
    item: NormalizedReferenceItem,
    selected_attachment: ReferenceAttachmentCandidate,
    content_text: str,
    safe_metadata: dict[str, Any],
) -> int:
    safe_metadata_json = json.dumps(safe_metadata, ensure_ascii=False)
    media_writer = get_media_repository(media_db)
    media_id, _media_uuid, _message = media_writer.add_media_with_keywords(
        url=(
            selected_attachment.source_url
            or item.source_url
            or f"{item.provider}://{item.provider_item_key}/{selected_attachment.attachment_key}"
        ),
        title=item.title or selected_attachment.title or item.provider_item_key,
        media_type="document",
        content=content_text or f"[empty content for {item.provider}:{item.provider_item_key}]",
        keywords=[],
        safe_metadata=safe_metadata_json,
        author=item.authors,
        ingestion_date=item.publication_date,
        overwrite=False,
    )
    if media_id is None:
        raise ReferenceImportError(
            f"Reference import did not return a media ID for {item.provider_item_key}"
        )
    return int(media_id)


def _merge_missing_reference_metadata(
    media_db: Any,
    *,
    media_id: int,
    incoming_safe_metadata: dict[str, Any],
) -> None:
    latest_version = get_document_version(media_db, media_id=media_id)
    if not latest_version or latest_version.get("id") is None:
        return
    existing_safe_metadata = _load_safe_metadata(latest_version.get("safe_metadata"))
    normalized_existing_safe_metadata = _normalize_existing_safe_metadata_for_merge(existing_safe_metadata)
    normalized_incoming_safe_metadata = normalize_safe_metadata(dict(incoming_safe_metadata or {}))
    merged_safe_metadata = dict(normalized_existing_safe_metadata)
    for key, value in normalized_incoming_safe_metadata.items():
        if value in (None, ""):
            continue
        if merged_safe_metadata.get(key) in (None, ""):
            merged_safe_metadata[key] = value
    if merged_safe_metadata == normalized_existing_safe_metadata:
        return
    with media_db.transaction() as connection:
        update_version_safe_metadata_in_transaction(
            db=media_db,
            dv_id=int(latest_version["id"]),
            safe_metadata_json=json.dumps(merged_safe_metadata, ensure_ascii=False),
            merged_metadata=merged_safe_metadata,
            connection=connection,
        )


async def _load_item_attachments(
    connector: Any,
    account: dict[str, Any],
    item: NormalizedReferenceItem,
) -> list[ReferenceAttachmentCandidate]:
    if item.attachments:
        return list(item.attachments)
    return list(
        await connector.list_item_attachments(
            account,
            item.provider_item_key,
        )
    )


async def _download_reference_attachment(
    connector: Any,
    account: dict[str, Any],
    attachment: ReferenceAttachmentCandidate,
) -> bytes:
    if hasattr(connector, "resolve_attachment_download"):
        raw_attachment = await connector.resolve_attachment_download(account, attachment)
    else:
        raw_attachment = await connector.download_file(
            account,
            attachment.attachment_key,
            mime_type=attachment.mime_type,
        )
    if not raw_attachment:
        raise ValueError(
            f"Attachment download returned no bytes for {attachment.provider}:{attachment.attachment_key}"
        )
    return raw_attachment


async def _resolve_reference_item_match(
    media_db: Any,
    *,
    item: NormalizedReferenceItem,
    same_provider_item: dict[str, Any] | None,
    doi_match: dict[str, Any] | None,
    content_hash: str | None,
) -> tuple[str | None, int | None]:
    match = rank_reference_item_match(
        item,
        same_provider_item=same_provider_item,
        doi_match=doi_match,
        hash_match=_find_file_hash_match(media_db, content_hash),
        metadata_match=_find_metadata_fingerprint_match(media_db, item),
    )
    return match.reason, match.media_id


async def sync_reference_manager_source(
    *,
    connectors_pool: Any,
    connector: Any,
    account: dict[str, Any],
    source: dict[str, Any],
    sync_state: dict[str, Any] | None,
    media_db: Any,
    job_id: str,
    convert_bytes_to_text: Callable[[bytes, str, str], Awaitable[str]],
) -> dict[str, Any]:
    """Import one collection page and keep the cursor stable on partial failures."""
    source_id = int(source["id"])
    provider = str(source.get("provider") or "")
    collection_key = str(source.get("remote_id") or "")
    current_cursor = str((sync_state or {}).get("cursor") or "").strip() or None
    next_cursor = current_cursor
    processed = 0
    imported = 0
    duplicates = 0
    metadata_only = 0
    failed = 0
    total = 0
    collection_name = str(source.get("path") or "").strip() or None

    items, page_cursor = await connector.list_collection_items(
        account,
        collection_key,
        collection_name=collection_name,
        cursor=current_cursor,
        page_size=100,
    )
    total = len(items or [])

    for item in items or []:
        try:
            safe_metadata = _canonical_reference_safe_metadata(item)
            async with connectors_pool.transaction() as db:
                same_provider_item = await get_reference_item_binding(
                    db,
                    source_id=source_id,
                    provider=provider,
                    provider_item_key=item.provider_item_key,
                )
            same_provider_media_id = _coerce_media_id(
                None if same_provider_item is None else same_provider_item.get("media_id")
            )
            if same_provider_media_id is not None:
                binding_metadata = _carry_forward_selected_attachment(
                    _build_binding_metadata(
                        item,
                        selected_attachment=None,
                        safe_metadata=safe_metadata,
                    ),
                    same_provider_item,
                )
                _merge_missing_reference_metadata(
                    media_db,
                    media_id=same_provider_media_id,
                    incoming_safe_metadata=safe_metadata,
                )
                async with connectors_pool.transaction() as db:
                    await upsert_reference_item_binding(
                        db,
                        source_id=source_id,
                        provider=provider,
                        provider_item_key=item.provider_item_key,
                        provider_library_id=item.provider_library_id,
                        collection_key=item.collection_key,
                        collection_name=item.collection_name,
                        provider_version=_provider_version(item),
                        provider_updated_at=None,
                        media_id=same_provider_media_id,
                        dedupe_match_reason="same_provider_item",
                        raw_reference_metadata=binding_metadata,
                    )
                duplicates += 1
                processed += 1
                continue

            doi_match = _find_doi_match(media_db, item)
            doi_media_id = _coerce_media_id(None if doi_match is None else doi_match.get("media_id"))
            if doi_media_id is not None:
                binding_metadata = _build_binding_metadata(
                    item,
                    selected_attachment=None,
                    safe_metadata=safe_metadata,
                )
                _merge_missing_reference_metadata(
                    media_db,
                    media_id=doi_media_id,
                    incoming_safe_metadata=safe_metadata,
                )
                async with connectors_pool.transaction() as db:
                    await upsert_reference_item_binding(
                        db,
                        source_id=source_id,
                        provider=provider,
                        provider_item_key=item.provider_item_key,
                        provider_library_id=item.provider_library_id,
                        collection_key=item.collection_key,
                        collection_name=item.collection_name,
                        provider_version=_provider_version(item),
                        provider_updated_at=None,
                        media_id=doi_media_id,
                        dedupe_match_reason="doi",
                        raw_reference_metadata=binding_metadata,
                    )
                duplicates += 1
                processed += 1
                continue

            attachments = await _load_item_attachments(connector, account, item)
            selected_attachment = _select_best_attachment(attachments)
            content_text = ""
            content_hash = None
            if selected_attachment is not None:
                raw_attachment = await _download_reference_attachment(
                    connector,
                    account,
                    selected_attachment,
                )
                attachment_name = (
                    str(selected_attachment.metadata.get("filename") or "").strip()
                    or selected_attachment.title
                    or f"{item.provider_item_key}.pdf"
                )
                content_text = await convert_bytes_to_text(
                    raw_attachment,
                    attachment_name,
                    str(selected_attachment.mime_type or "").lower(),
                )
                if content_text:
                    content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest()

            match_reason, matched_media_id = await _resolve_reference_item_match(
                media_db,
                item=item,
                same_provider_item=same_provider_item,
                doi_match=doi_match,
                content_hash=content_hash,
            )
            binding_metadata = _build_binding_metadata(
                item,
                selected_attachment=selected_attachment,
                safe_metadata=safe_metadata,
            )

            if matched_media_id is not None:
                _merge_missing_reference_metadata(
                    media_db,
                    media_id=int(matched_media_id),
                    incoming_safe_metadata=safe_metadata,
                )
                async with connectors_pool.transaction() as db:
                    await upsert_reference_item_binding(
                        db,
                        source_id=source_id,
                        provider=provider,
                        provider_item_key=item.provider_item_key,
                        provider_library_id=item.provider_library_id,
                        collection_key=item.collection_key,
                        collection_name=item.collection_name,
                        provider_version=_provider_version(item),
                        provider_updated_at=None,
                        media_id=int(matched_media_id),
                        dedupe_match_reason=match_reason,
                        raw_reference_metadata=binding_metadata,
                    )
                duplicates += 1
                processed += 1
                continue

            if selected_attachment is None:
                async with connectors_pool.transaction() as db:
                    await upsert_reference_item_binding(
                        db,
                        source_id=source_id,
                        provider=provider,
                        provider_item_key=item.provider_item_key,
                        provider_library_id=item.provider_library_id,
                        collection_key=item.collection_key,
                        collection_name=item.collection_name,
                        provider_version=_provider_version(item),
                        provider_updated_at=None,
                        media_id=None,
                        dedupe_match_reason="metadata_only",
                        raw_reference_metadata=binding_metadata,
                    )
                metadata_only += 1
                processed += 1
                continue

            media_id = _ingest_reference_attachment(
                media_db,
                item=item,
                selected_attachment=selected_attachment,
                content_text=content_text,
                safe_metadata=safe_metadata,
            )
            async with connectors_pool.transaction() as db:
                await upsert_reference_item_binding(
                    db,
                    source_id=source_id,
                    provider=provider,
                    provider_item_key=item.provider_item_key,
                    provider_library_id=item.provider_library_id,
                    collection_key=item.collection_key,
                    collection_name=item.collection_name,
                    provider_version=_provider_version(item),
                    provider_updated_at=None,
                    media_id=media_id,
                    dedupe_match_reason=None,
                    raw_reference_metadata=binding_metadata,
                )
            imported += 1
            processed += 1
        except Exception as exc:  # pragma: no cover - defensive counter path
            failed += 1
            logger.warning(
                "Reference-manager import failed for source_id={} provider={} item={}: {}",
                source_id,
                provider,
                getattr(item, "provider_item_key", "unknown"),
                exc,
            )

    if failed == 0:
        if page_cursor in (None, ""):
            next_cursor = None
        else:
            next_cursor = str(page_cursor).strip() or None

    return {
        "processed": processed,
        "total": total,
        "failed": failed,
        "imported": imported,
        "duplicates": duplicates,
        "metadata_only": metadata_only,
        "cursor": next_cursor,
    }
