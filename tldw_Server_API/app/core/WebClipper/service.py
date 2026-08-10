"""Canonical save and enrichment orchestration for browser web clips."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError as MediaConflictError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError as MediaDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    InputError as MediaInputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import MediaRepository
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    capture_note_upsert,
    stable_note_id,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename
from tldw_Server_API.app.core.WebClipper.schemas import (
    WebClipperAttachmentRecord,
    WebClipperEnrichmentPayload,
    WebClipperEnrichmentResponse,
    WebClipperSavedNote,
    WebClipperSaveRequest,
    WebClipperSaveResponse,
    WebClipperStatusResponse,
    WebClipperWorkspacePlacement,
    validate_capture_metadata_json_size,
)

_NOTES_ATTACHMENTS_DIRNAME = "notes_attachments"
_NOTES_ATTACHMENT_META_SUFFIX = ".meta.json"
_MAX_VISIBLE_BODY_INLINE = 12_000
_MAX_OCR_INLINE = 1_500
_MAX_VLM_INLINE = 1_000
_MAX_MACHINE_INLINE = 2_500
_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
_ALLOWED_ATTACHMENT_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/pdf",
        "application/json",
    }
)
_BLOCKED_ATTACHMENT_SUFFIXES: frozenset[str] = frozenset({".htm", ".html", ".svg"})
_FULL_EXTRACT_SPILLOVER_THRESHOLD = 20_000
_TRUNCATION_MARKER = "Truncated. Full extract preserved in clip metadata."
_ENRICHMENT_SECTION_START = "<!-- web-clipper-enrichment:start -->"
_ENRICHMENT_SECTION_END = "<!-- web-clipper-enrichment:end -->"
_WEB_CLIPPER_SOURCE_PREFIX = "web-clipper"
_WEB_CLIPPER_CAPTURE_ROUTING_KEY = "web_clipper_capture_v1"
_MAX_CAPTURED_AT_LENGTH = 128
_MAX_CLIP_ID_LENGTH = 256
WorkspaceSourceJobEnqueuer = Callable[[str, dict[str, Any]], None]


class _ClipIdCollisionError(ConflictError):
    """Raised when a WebClipper request attempts to claim a non-clipper note."""


@dataclass(slots=True)
class WebClipperService:
    """Focused orchestration for note-backed clip saves and enrichments."""

    db: CharactersRAGDB
    user_id: int | str
    media_db: Any | None = None
    promote_workspace_sources: bool = False
    workspace_source_job_enqueuer: WorkspaceSourceJobEnqueuer | None = None

    def save_clip(self, request: WebClipperSaveRequest) -> WebClipperSaveResponse:
        """Create or update the canonical note and optional workspace placement."""
        existing_document = self._document_for_save(request.clip_id)
        warnings: list[str] = []
        analysis_state = self._current_analysis_state(request.clip_id)
        self._record_requested_enhancements(analysis_state, request)

        primary_body = self._select_primary_body(request)
        visible_body, content_budget, generated_attachments = self._budget_visible_body(
            body=primary_body,
            fallback_full_extract=request.content.full_extract,
        )
        note_title = self._resolve_note_title(request)
        note_id = self._note_id_for_clip(request.clip_id, existing_document)
        coordinator = active_coordinator(self.db, user_id=self.user_id)
        try:
            if coordinator is not None:
                request_fingerprint = self._active_sync_request_fingerprint(
                    coordinator,
                    request,
                )
                replay = coordinator.replay_request_plan(
                    source="web-clipper",
                    idempotency_key=request_fingerprint,
                    request_fingerprint=request_fingerprint,
                    result_domain="notes.note",
                )
                existing_note = self.db.get_note_by_id(note_id)
                if replay is None and existing_note is not None and existing_document is None:
                    raise _ClipIdCollisionError(
                        "The canonical WebClipper note identity is already in use.",
                        entity="notes",
                        entity_id=note_id,
                    )
                capture_metadata = self._merge_capture_metadata(existing_document, request)
                note_content = self._render_note_content(
                    request=request,
                    visible_body=visible_body,
                    analysis=analysis_state,
                    capture_metadata=capture_metadata,
                )
                folder_paths: list[str] | None = None
                if request.note.folder_id is not None:
                    folder = next(
                        (row for row in self.db.list_note_folders() if int(row["id"]) == int(request.note.folder_id)),
                        None,
                    )
                    if folder is None:
                        warnings.append(f"Folder '{request.note.folder_id}' was not found.")
                    else:
                        folder_paths = [str(folder["path"])]
                persisted_note = capture_note_upsert(
                    coordinator,
                    note_id=note_id,
                    title=note_title,
                    content=note_content,
                    keywords=request.note.keywords,
                    folder_paths=folder_paths,
                    expected_version=(int(existing_note["version"]) if existing_note is not None else None),
                    source="web-clipper",
                    key=request_fingerprint,
                    request_fingerprint=request_fingerprint,
                    server_routing_metadata=self._capture_routing_metadata(capture_metadata),
                )
                if replay is not None:
                    note_content = str(persisted_note.get("content") or "")
                    captured_at = self._capture_date_from_replay(replay, note_id=note_id)
                    if not captured_at:
                        raise SyncStoreError("WebClipper replay metadata is incomplete")
                    capture_metadata["captured_at"] = captured_at
                    try:
                        capture_metadata = validate_capture_metadata_json_size(capture_metadata)
                    except ValueError as exc:
                        raise InputError(str(exc)) from exc
                current_analysis_state = self._current_analysis_state(request.clip_id)
                self._record_requested_enhancements(current_analysis_state, request)
                try:
                    self.db.upsert_note_clipper_document(
                        clip_id=request.clip_id,
                        note_id=note_id,
                        clip_type=request.clip_type,
                        source_url=request.source_url,
                        source_title=request.source_title,
                        capture_metadata=capture_metadata,
                        enrichments=current_analysis_state,
                        content_budget=content_budget,
                        source_note_version=int(persisted_note["version"]),
                    )
                except ConflictError:
                    raise
                except CharactersRAGDBError as exc:
                    raise SyncStoreError("WebClipper sidecar projection is incomplete") from exc
            else:
                with self.db.transaction() as conn:
                    existing_document = self._document_for_save(
                        request.clip_id,
                        conn=conn,
                    )
                    note_id = self._note_id_for_clip(request.clip_id, existing_document)
                    existing_note = self._fetch_note_row(note_id, conn=conn)
                    capture_metadata = self._merge_capture_metadata(existing_document, request)
                    note_content = self._render_note_content(
                        request=request,
                        visible_body=visible_body,
                        analysis=analysis_state,
                        capture_metadata=capture_metadata,
                    )
                    if existing_note is not None and existing_document is None:
                        raise _ClipIdCollisionError(
                            "The canonical WebClipper note identity is already in use.",
                            entity="notes",
                            entity_id=note_id,
                        )
                    if existing_note is None:
                        created_note_id = self.db.add_note(
                            title=note_title,
                            content=note_content,
                            note_id=note_id,
                            conn=conn,
                        )
                        if created_note_id is None:
                            raise CharactersRAGDBError("Failed to create canonical clip note.")  # noqa: TRY003
                    else:
                        updates: dict[str, Any] = {}
                        if str(existing_note.get("title") or "") != note_title:
                            updates["title"] = note_title
                        if str(existing_note.get("content") or "") != note_content:
                            updates["content"] = note_content
                        if updates:
                            self.db.update_note(
                                note_id=note_id,
                                update_data=updates,
                                expected_version=int(existing_note["version"]),
                                conn=conn,
                            )

                    persisted_note = self._fetch_note_row(note_id, conn=conn)
                    if persisted_note is None:
                        raise CharactersRAGDBError("Canonical clip note could not be reloaded after save.")  # noqa: TRY003

                    current_analysis_state = self._current_analysis_state(request.clip_id)
                    self._record_requested_enhancements(current_analysis_state, request)
                    self.db.upsert_note_clipper_document(
                        clip_id=request.clip_id,
                        note_id=note_id,
                        clip_type=request.clip_type,
                        source_url=request.source_url,
                        source_title=request.source_title,
                        capture_metadata=capture_metadata,
                        enrichments=current_analysis_state,
                        content_budget=content_budget,
                        source_note_version=int(persisted_note["version"]),
                        conn=conn,
                    )
        except _ClipIdCollisionError:
            raise
        except (CharactersRAGDBError, ConflictError) as exc:
            return WebClipperSaveResponse(
                clip_id=request.clip_id,
                status="failed",
                note=None,
                workspace_placement=None,
                attachments=[],
                warnings=[f"Canonical note save failed: {exc}"],
                note_id=note_id,
                workspace_placement_saved=False,
                workspace_placement_count=0,
            )

        note_id = str(persisted_note["id"])
        note_summary = self._build_note_summary(self._require_note(note_id))

        if coordinator is None:
            self._sync_keywords(note_id, request.note.keywords, warnings)
            self._sync_note_folder(note_id, request.note.folder_id, warnings)

        attachments = self._persist_attachments(
            note_id=note_id,
            attachments=[*request.attachments, *generated_attachments],
            warnings=warnings,
        )

        workspace_placement: WebClipperWorkspacePlacement | None = None
        workspace_requested = request.destination_mode in {"workspace", "both"}
        if workspace_requested and request.workspace is not None:
            try:
                workspace_placement = self._ensure_workspace_placement(
                    request=request,
                    note_summary=note_summary,
                    note_content=note_content,
                )
            except (CharactersRAGDBError, ConflictError, InputError) as exc:
                warnings.append(f"Workspace placement failed: {exc}")
            else:
                if self.promote_workspace_sources:
                    try:
                        workspace_source = self._ensure_workspace_source(
                            request=request,
                            note_summary=note_summary,
                            media_content=self._select_media_body(request),
                        )
                        self._notify_workspace_source_job(
                            workspace_id=request.workspace.workspace_id,
                            source=workspace_source,
                        )
                    except (CharactersRAGDBError, ConflictError, InputError) as exc:
                        warnings.append(f"Workspace source promotion failed: {exc}")

        status = self._derive_save_status(
            warnings=warnings,
            workspace_requested=workspace_requested,
            workspace_saved=workspace_placement is not None,
        )
        placement_count = 1 if workspace_placement is not None else 0
        final_content_budget = dict(content_budget)
        final_content_budget["last_save_status"] = status
        final_content_budget["warnings"] = list(warnings)
        final_content_budget["enhancement_requests"] = {
            "run_ocr": request.enhancements.run_ocr,
            "run_vlm": request.enhancements.run_vlm,
        }
        latest_analysis_state = self._current_analysis_state(request.clip_id)
        self._record_requested_enhancements(latest_analysis_state, request)
        try:
            self.db.upsert_note_clipper_document(
                clip_id=request.clip_id,
                note_id=note_id,
                clip_type=request.clip_type,
                source_url=request.source_url,
                source_title=request.source_title,
                capture_metadata=capture_metadata,
                enrichments=latest_analysis_state,
                content_budget=final_content_budget,
                source_note_version=note_summary.version,
            )
        except ConflictError:
            raise
        except CharactersRAGDBError as exc:
            if coordinator is not None:
                raise SyncStoreError("WebClipper sidecar projection is incomplete") from exc
            raise
        return WebClipperSaveResponse(
            clip_id=request.clip_id,
            status=status,
            note=note_summary,
            workspace_placement=workspace_placement,
            attachments=attachments,
            warnings=warnings,
            note_id=note_summary.id,
            workspace_placement_saved=workspace_placement is not None,
            workspace_placement_count=placement_count,
        )

    def get_clip_status(self, clip_id: str) -> WebClipperStatusResponse:
        """Return the canonical note, placements, attachments, and analysis for a clip."""
        clip_id = self._normalized_clip_id(clip_id)
        clip_document = self._require_clip_document(clip_id)
        note_id = str(clip_document["note_id"])
        note = self._require_note(note_id)
        placements = [
            self._placement_response_from_row(row) for row in self.db.list_note_clipper_workspace_placements(clip_id)
        ]
        attachments = self._list_attachments(note_id=note_id)
        analysis = self._normalize_json_dict(clip_document.get("analysis_json"))
        content_budget = self._normalize_json_dict(clip_document.get("content_budget_json"))
        status = str(content_budget.get("last_save_status") or "saved")
        if status not in {"saved", "saved_with_warnings", "partially_saved", "failed"}:
            status = "saved"
        return WebClipperStatusResponse(
            clip_id=clip_id,
            status=status,
            note=self._build_note_summary(note),
            workspace_placements=placements,
            attachments=attachments,
            analysis=analysis,
            content_budget=content_budget,
        )

    def persist_enrichment(
        self,
        clip_id: str,
        payload: WebClipperEnrichmentPayload,
    ) -> WebClipperEnrichmentResponse:
        """Persist OCR/VLM structured output and append inline summaries safely."""
        clip_id = self._normalized_clip_id(clip_id)
        if payload.clip_id != clip_id:
            raise InputError("clip_id path and payload clip_id must match.")  # noqa: TRY003

        clip_document = self._require_clip_document(clip_id)
        note_id = str(clip_document["note_id"])
        note = self._require_note(note_id)
        analysis = self._normalize_json_dict(clip_document.get("analysis_json"))

        trimmed_summary = self._truncate_inline_summary(
            payload.enrichment_type,
            payload.inline_summary,
        )
        analysis[payload.enrichment_type] = {
            "status": payload.status,
            "inline_summary": trimmed_summary,
            "structured_payload": payload.structured_payload,
            "error": payload.error,
            "source_note_version": payload.source_note_version,
        }

        inline_applied = False
        conflict_reason: str | None = None
        warnings: list[str] = []
        current_note_version = int(note["version"])

        if payload.status == "complete" and trimmed_summary:
            if current_note_version != payload.source_note_version:
                conflict_reason = "source_note_version_mismatch"
            else:
                next_content = self._render_note_content(
                    request=self._request_from_clip_state(clip_document=clip_document, note=note),
                    visible_body=self._visible_body_from_note(note),
                    analysis=analysis,
                    capture_metadata=self._normalize_json_dict(clip_document.get("capture_metadata_json")),
                )
                if next_content != str(note.get("content") or ""):
                    coordinator = active_coordinator(self.db, user_id=self.user_id)
                    if coordinator is not None:
                        capture_note_upsert(
                            coordinator,
                            note_id=note_id,
                            title=str(note.get("title") or ""),
                            content=next_content,
                            conversation_id=note.get("conversation_id"),
                            message_id=note.get("message_id"),
                            expected_version=current_note_version,
                            source="web-clipper-enrichment",
                        )
                    else:
                        self.db.update_note(
                            note_id=note_id,
                            update_data={"content": next_content},
                            expected_version=current_note_version,
                        )
                    note = self._require_note(note_id)
                    current_note_version = int(note["version"])
                inline_applied = True

        self.db.upsert_note_clipper_document(
            clip_id=clip_id,
            note_id=note_id,
            clip_type=str(clip_document.get("clip_type") or "clip"),
            source_url=clip_document.get("source_url"),
            source_title=clip_document.get("source_title"),
            capture_metadata=self._normalize_json_dict(clip_document.get("capture_metadata_json")),
            enrichments=analysis,
            content_budget=self._normalize_json_dict(clip_document.get("content_budget_json")),
            source_note_version=current_note_version,
        )

        return WebClipperEnrichmentResponse(
            clip_id=clip_id,
            enrichment_type=payload.enrichment_type,
            status=payload.status,
            source_note_version=current_note_version,
            inline_applied=inline_applied,
            inline_summary=trimmed_summary,
            conflict_reason=conflict_reason,
            warnings=warnings,
        )

    def _resolve_note_title(self, request: WebClipperSaveRequest) -> str:
        title = str(request.note.title or request.source_title or "").strip()
        if not title:
            raise InputError("Note title cannot be empty.")  # noqa: TRY003
        return title

    @staticmethod
    def _active_sync_request_fingerprint(coordinator: Any, request: WebClipperSaveRequest) -> str:
        """Hash normalized client intent without generated timestamps or versions."""

        def digest(value: str | None) -> str | None:
            if value is None:
                return None
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        workspace = request.workspace
        fields = {
            "clip_id": request.clip_id.strip(),
            "clip_type": request.clip_type.strip(),
            "source_url": request.source_url.strip(),
            "source_title": request.source_title.strip(),
            "destination_mode": request.destination_mode,
            "note": {
                "title": request.note.title.strip() if request.note.title else None,
                "comment": request.note.comment.strip() if request.note.comment else None,
                "folder_id": request.note.folder_id,
                "keywords": [keyword.strip() for keyword in request.note.keywords],
            },
            "workspace": (
                {
                    "workspace_id": workspace.workspace_id.strip(),
                    "default_review_state": workspace.default_review_state,
                }
                if workspace is not None
                else None
            ),
            "content": {
                "visible_body": digest(request.content.visible_body),
                "full_extract": digest(request.content.full_extract),
                "selected_text": digest(request.content.selected_text),
            },
            "attachments": [
                {
                    "slot": attachment.slot.strip(),
                    "file_name": attachment.file_name.strip() if attachment.file_name else None,
                    "media_type": attachment.media_type.strip(),
                    "text_content": digest(attachment.text_content),
                    "content_base64": digest(attachment.content_base64),
                    "source_url": attachment.source_url.strip() if attachment.source_url else None,
                }
                for attachment in request.attachments
            ],
            "enhancements": {
                "run_ocr": request.enhancements.run_ocr,
                "run_vlm": request.enhancements.run_vlm,
            },
            "capture_metadata": dict(request.capture_metadata),
        }
        return coordinator.request_fingerprint("web-clipper.save", fields)

    @staticmethod
    def _normalized_captured_at(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if (
            not normalized
            or normalized != value
            or len(normalized) > _MAX_CAPTURED_AT_LENGTH
            or any(character in normalized for character in ("\r", "\n", "\x00"))
        ):
            return None
        return normalized

    @staticmethod
    def _normalized_clip_id(value: object) -> str:
        normalized = str(value or "").strip()
        if not normalized or len(normalized) > _MAX_CLIP_ID_LENGTH:
            raise InputError("clip_id must contain 1 to 256 non-whitespace characters")
        return normalized

    @classmethod
    def _capture_routing_metadata(
        cls,
        capture_metadata: Mapping[str, object],
    ) -> dict[str, object]:
        captured_at = cls._normalized_captured_at(capture_metadata.get("captured_at"))
        if captured_at is None:
            raise InputError("WebClipper captured_at metadata is invalid")
        return {
            _WEB_CLIPPER_CAPTURE_ROUTING_KEY: {"captured_at": captured_at},
        }

    @classmethod
    def _capture_date_from_replay(cls, replay: Any, *, note_id: str) -> str | None:
        note_steps = [step for step in replay.steps if step.domain == "notes.note" and step.object_id == note_id]
        if len(note_steps) != 1:
            return None
        metadata = note_steps[0].routing_metadata.get(_WEB_CLIPPER_CAPTURE_ROUTING_KEY)
        if not isinstance(metadata, Mapping):
            return None
        return cls._normalized_captured_at(metadata.get("captured_at"))

    def _note_id_for_clip(
        self,
        clip_id: str,
        document: dict[str, Any] | None,
    ) -> str:
        """Resolve the private canonical note identity for a public clip id."""
        if document is not None:
            return str(document["note_id"])
        return stable_note_id(
            "web-clipper",
            f"{self.user_id}\0{self._normalized_clip_id(clip_id)}",
        )

    def _document_for_save(
        self,
        clip_id: str,
        *,
        conn: Any | None = None,
    ) -> dict[str, Any] | None:
        document = self.db._fetch_note_clipper_document_row(
            column="clip_id",
            value=self._normalized_clip_id(clip_id),
            conn=conn,
            include_deleted=True,
        )
        if document is not None and bool(document.get("deleted")):
            raise _ClipIdCollisionError(
                "The clip is deleted; restore it before saving again.",
                entity="note_clipper_documents",
                entity_id=self._normalized_clip_id(clip_id),
            )
        return document

    def _record_requested_enhancements(
        self,
        analysis: dict[str, Any],
        request: WebClipperSaveRequest,
    ) -> None:
        if request.enhancements.run_ocr:
            analysis.setdefault(
                "ocr",
                {
                    "status": "pending",
                    "inline_summary": None,
                    "structured_payload": {},
                    "error": None,
                },
            )
        if request.enhancements.run_vlm:
            analysis.setdefault(
                "vlm",
                {
                    "status": "pending",
                    "inline_summary": None,
                    "structured_payload": {},
                    "error": None,
                },
            )

    @staticmethod
    def _normalize_json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _merge_capture_metadata(
        self,
        existing_document: dict[str, Any] | None,
        request: WebClipperSaveRequest,
    ) -> dict[str, Any]:
        merged = self._normalize_json_dict(
            existing_document.get("capture_metadata_json") if existing_document else None
        )
        merged.update(request.capture_metadata)
        merged.setdefault("captured_at", datetime.now(timezone.utc).isoformat())
        captured_at = self._normalized_captured_at(merged.get("captured_at"))
        if captured_at is None:
            raise InputError("WebClipper captured_at metadata is invalid")
        merged["captured_at"] = captured_at
        try:
            return validate_capture_metadata_json_size(merged)
        except ValueError as exc:
            raise InputError(str(exc)) from exc  # noqa: TRY003

    def _current_analysis_state(self, clip_id: str) -> dict[str, Any]:
        current_document = self.db.get_note_clipper_document_by_clip_id(clip_id)
        if current_document is None:
            return {}
        return self._normalize_json_dict(current_document.get("analysis_json"))

    def _select_primary_body(self, request: WebClipperSaveRequest) -> str:
        for candidate in (
            request.content.visible_body,
            request.content.selected_text,
            request.content.full_extract,
        ):
            text = str(candidate or "").strip()
            if text:
                return text
        return ""

    def _select_media_body(self, request: WebClipperSaveRequest) -> str:
        for candidate in (
            request.content.full_extract,
            request.content.selected_text,
            request.content.visible_body,
        ):
            text = str(candidate or "").strip()
            if text:
                return text
        return ""

    def _budget_visible_body(
        self,
        *,
        body: str,
        fallback_full_extract: str | None,
    ) -> tuple[str, dict[str, Any], list[WebClipperSaveRequest.AttachmentPayload]]:
        normalized_body = str(body or "")
        full_extract = str(fallback_full_extract or normalized_body or "")
        truncated = len(normalized_body) > _MAX_VISIBLE_BODY_INLINE
        visible_body = normalized_body
        if truncated:
            visible_body = f"{self._truncate_on_paragraph_boundary(normalized_body)}\n{_TRUNCATION_MARKER}"
        content_budget = {
            "visible_body": visible_body,
            "visible_body_length": len(normalized_body),
            "visible_body_truncated": truncated,
            "visible_body_limit": _MAX_VISIBLE_BODY_INLINE,
            "full_extract_length": len(full_extract),
            "full_extract_spillover": full_extract if len(full_extract) > _FULL_EXTRACT_SPILLOVER_THRESHOLD else None,
            "ocr_inline_limit": _MAX_OCR_INLINE,
            "vlm_inline_limit": _MAX_VLM_INLINE,
            "machine_inline_limit": _MAX_MACHINE_INLINE,
            "generated_attachment_slots": [],
        }
        return visible_body, content_budget, []

    def _truncate_on_paragraph_boundary(self, text: str) -> str:
        if len(text) <= 11_995:
            return text.rstrip()
        paragraph_boundary = text.rfind("\n\n", 0, _MAX_VISIBLE_BODY_INLINE)
        if paragraph_boundary >= 9_000:
            return text[:paragraph_boundary].rstrip()
        line_boundary = text.rfind("\n", 0, _MAX_VISIBLE_BODY_INLINE)
        if line_boundary >= 9_000:
            return text[:line_boundary].rstrip()
        return text[:11_995].rstrip()

    def _render_note_content(
        self,
        *,
        request: WebClipperSaveRequest,
        visible_body: str,
        analysis: dict[str, Any],
        capture_metadata: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        comment = str(request.note.comment or "").strip()
        if comment:
            parts.append(comment)
        capture_date = str(capture_metadata.get("captured_at") or "").strip()
        parts.append(
            "\n".join(
                [
                    "Source:",
                    f"Title: {request.source_title}",
                    f"URL: {request.source_url}",
                    f"Capture date: {capture_date}",
                    f"Clip type: {request.clip_type}",
                ]
            )
        )
        if visible_body:
            parts.append(visible_body)
        enrichment_section = self._build_enrichment_section(analysis)
        content = "\n\n".join(parts).strip()
        if enrichment_section:
            if content:
                return f"{content}\n\n{enrichment_section}".strip()
            return enrichment_section
        return content

    def _build_enrichment_section(self, analysis: dict[str, Any]) -> str:
        remaining_budget = _MAX_MACHINE_INLINE
        lines: list[str] = []
        for enrichment_type, per_type_limit in (("ocr", _MAX_OCR_INLINE), ("vlm", _MAX_VLM_INLINE)):
            entry = analysis.get(enrichment_type)
            if not isinstance(entry, dict):
                continue
            summary = str(entry.get("inline_summary") or "").strip()
            if not summary:
                continue
            summary = summary[: min(per_type_limit, remaining_budget)].strip()
            if not summary:
                continue
            remaining_budget -= len(summary)
            lines.extend([f"### {enrichment_type.upper()}", summary])
            if remaining_budget <= 0:
                break
        if not lines:
            return ""
        section_body = "\n\n".join(["## Web Clipper Analysis", *lines]).strip()
        return f"{_ENRICHMENT_SECTION_START}\n{section_body}\n{_ENRICHMENT_SECTION_END}"

    def _visible_body_from_note(self, note: dict[str, Any]) -> str:
        content = str(note.get("content") or "")
        without_machine = self._strip_machine_section(content)
        parts = [part.strip() for part in without_machine.split("\n\n") if part.strip()]
        source_index = next(
            (index for index, part in enumerate(parts) if part.startswith("Source:")),
            None,
        )
        if source_index is not None:
            body_parts = parts[source_index + 1 :]
            if body_parts:
                return "\n\n".join(body_parts).strip()
            return ""
        if parts:
            return "\n\n".join(parts).strip()
        return ""

    def _request_from_clip_state(
        self,
        *,
        clip_document: dict[str, Any],
        note: dict[str, Any],
    ) -> WebClipperSaveRequest:
        content_budget = self._normalize_json_dict(clip_document.get("content_budget_json"))
        capture_metadata = self._normalize_json_dict(clip_document.get("capture_metadata_json"))
        return WebClipperSaveRequest(
            clip_id=str(clip_document["clip_id"]),
            clip_type=str(clip_document.get("clip_type") or "clip"),
            source_url=str(clip_document.get("source_url") or ""),
            source_title=str(clip_document.get("source_title") or note.get("title") or ""),
            destination_mode="note",
            note=WebClipperSaveRequest.NotePayload(
                title=str(note.get("title") or ""),
                comment=self._extract_comment_from_note(note),
                keywords=[keyword["keyword"] for keyword in self.db.get_keywords_for_note(str(note["id"]))],
            ),
            content=WebClipperSaveRequest.ContentPayload(
                visible_body=str(content_budget.get("visible_body") or self._visible_body_from_note(note)),
                full_extract=None,
                selected_text=None,
            ),
            capture_metadata=capture_metadata,
        )

    def _active_deleted_value(self) -> bool | int:
        return False if self.db.backend_type == BackendType.POSTGRESQL else 0

    def _extract_comment_from_note(self, note: dict[str, Any]) -> str | None:
        content = self._strip_machine_section(str(note.get("content") or ""))
        blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
        if len(blocks) >= 3 and not blocks[0].startswith("Source:"):
            return blocks[0]
        if len(blocks) >= 2 and not blocks[0].startswith("Source:") and blocks[1].startswith("Source:"):
            return blocks[0]
        return None

    def _sync_keywords(self, note_id: str, keywords: list[str], warnings: list[str]) -> None:
        normalized_keywords: list[str] = []
        seen_keywords: set[str] = set()
        for keyword in keywords:
            text = str(keyword or "").strip()
            if not text:
                continue
            if text in seen_keywords:
                continue
            seen_keywords.add(text)
            normalized_keywords.append(text)

        existing_keywords = self.db.get_keywords_for_note(note_id)
        existing_ids_by_text = {str(row["keyword"]): int(row["id"]) for row in existing_keywords}
        desired_keyword_ids: set[int] = set()

        for text in normalized_keywords:
            try:
                if text in existing_ids_by_text:
                    keyword_id = existing_ids_by_text[text]
                else:
                    existing_keyword = self.db.get_keyword_by_text(text)
                    if existing_keyword is not None:
                        keyword_id = int(existing_keyword["id"])
                    else:
                        keyword_id = self.db.add_keyword(text)
                        if keyword_id is None:
                            warnings.append(f"Keyword '{text}' could not be created.")
                            continue
                desired_keyword_ids.add(keyword_id)
                if keyword_id not in existing_ids_by_text.values():
                    self.db.link_note_to_keyword(note_id, keyword_id)
            except CharactersRAGDBError as exc:
                warnings.append(f"Keyword '{text}' could not be linked: {exc}")

        for existing_keyword in existing_keywords:
            keyword_id = int(existing_keyword["id"])
            if keyword_id in desired_keyword_ids:
                continue
            keyword_text = str(existing_keyword["keyword"])
            try:
                self.db.unlink_note_from_keyword(note_id, keyword_id)
            except CharactersRAGDBError as exc:
                warnings.append(f"Keyword '{keyword_text}' could not be removed: {exc}")

    def _sync_note_folder(self, note_id: str, folder_id: int | None, warnings: list[str]) -> None:
        if folder_id is None:
            return
        deleted_value = self._active_deleted_value()
        with self.db.transaction() as conn:
            folder_row = conn.execute(
                "SELECT id FROM note_folders WHERE id = ? AND deleted = ?",
                (folder_id, deleted_value),
            ).fetchone()
            if folder_row is None:
                warnings.append(f"Folder '{folder_id}' was not found.")
                return
            conn.execute("DELETE FROM note_folder_memberships WHERE note_id = ?", (note_id,))
            prepared_query, prepared_params = self.db._prepare_backend_statement(
                "INSERT INTO note_folder_memberships(note_id, folder_id, created_at) VALUES (?, ?, ?)",
                (
                    note_id,
                    folder_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.execute(prepared_query, prepared_params or ())

    def _persist_attachments(
        self,
        *,
        note_id: str,
        attachments: list[WebClipperSaveRequest.AttachmentPayload],
        warnings: list[str],
    ) -> list[WebClipperAttachmentRecord]:
        persisted: list[WebClipperAttachmentRecord] = []
        seen_slots: set[str] = set()
        for attachment in attachments:
            if attachment.slot in seen_slots:
                continue
            seen_slots.add(attachment.slot)
            try:
                persisted.append(self._persist_attachment(note_id=note_id, attachment=attachment))
            except (CharactersRAGDBError, InputError) as exc:
                warnings.append(f"Attachment '{attachment.slot}' could not be saved: {exc}")
        return persisted

    def _persist_attachment(
        self,
        *,
        note_id: str,
        attachment: WebClipperSaveRequest.AttachmentPayload,
    ) -> WebClipperAttachmentRecord:
        attachment_dir = self._get_note_attachments_dir(note_id, create=True)
        safe_file_name = self._deterministic_attachment_file_name(
            slot=attachment.slot,
            file_name=attachment.file_name,
        )
        target_path = (attachment_dir / safe_file_name).resolve()
        try:
            target_path.relative_to(attachment_dir)
        except ValueError as exc:
            raise InputError("Invalid attachment path.") from exc  # noqa: TRY003

        resolved_media_type = attachment.media_type or mimetypes.guess_type(safe_file_name)[0]
        if any(suffix.lower() in _BLOCKED_ATTACHMENT_SUFFIXES for suffix in Path(safe_file_name).suffixes):
            raise InputError("Attachment file type is not allowed.")  # noqa: TRY003
        resolved_media_type = str(resolved_media_type or "").split(";", maxsplit=1)[0].strip().lower() or None
        if resolved_media_type is None or resolved_media_type not in _ALLOWED_ATTACHMENT_MEDIA_TYPES:
            raise InputError(  # noqa: TRY003
                f"Attachment media type '{resolved_media_type}' is not allowed. "
                f"Allowed types: {', '.join(sorted(_ALLOWED_ATTACHMENT_MEDIA_TYPES))}."
            )

        if attachment.content_base64:
            try:
                raw_bytes = base64.b64decode(attachment.content_base64, validate=True)
            except (ValueError, TypeError) as exc:
                raise InputError("Attachment content_base64 is invalid.") from exc  # noqa: TRY003
        elif attachment.text_content is not None:
            raw_bytes = attachment.text_content.encode("utf-8")
        else:
            raw_bytes = b""

        if len(raw_bytes) > _MAX_ATTACHMENT_BYTES:
            raise InputError(f"Attachment exceeds maximum size of {_MAX_ATTACHMENT_BYTES} bytes")  # noqa: TRY003

        try:
            target_path.write_bytes(raw_bytes)
            uploaded_at = datetime.now(timezone.utc)
            metadata = {
                "slot": attachment.slot,
                "original_file_name": attachment.file_name or safe_file_name,
                "content_type": resolved_media_type,
                "size_bytes": len(raw_bytes),
                "uploaded_at": uploaded_at.isoformat(),
                "source_url": attachment.source_url,
            }
            self._attachment_metadata_path(target_path).write_text(
                json.dumps(metadata, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise CharactersRAGDBError(f"Attachment persistence failed for slot '{attachment.slot}'.") from exc  # noqa: TRY003
        return self._attachment_response(note_id=note_id, file_path=target_path)

    def _ensure_workspace_placement(
        self,
        *,
        request: WebClipperSaveRequest,
        note_summary: WebClipperSavedNote,
        note_content: str,
    ) -> WebClipperWorkspacePlacement:
        if request.workspace is None:
            raise InputError("workspace payload is required for workspace saves.")  # noqa: TRY003
        existing_rows = self.db.list_note_clipper_workspace_placements(request.clip_id)
        for row in existing_rows:
            if str(row.get("workspace_id")) == request.workspace.workspace_id:
                workspace_note_id = row.get("workspace_note_id")
                if workspace_note_id is None:
                    raise CharactersRAGDBError("Workspace placement is missing a workspace note id.")  # noqa: TRY003
                self._sync_workspace_note(
                    workspace_id=request.workspace.workspace_id,
                    workspace_note_id=int(workspace_note_id),
                    title=note_summary.title,
                    content=note_content,
                    keywords=request.note.keywords,
                )
                self.db.upsert_note_clipper_workspace_placement(
                    clip_id=request.clip_id,
                    workspace_id=request.workspace.workspace_id,
                    workspace_note_id=workspace_note_id,
                    source_note_id=note_summary.id,
                    source_note_version=note_summary.version,
                )
                return self._placement_response_from_row(
                    self.db.list_note_clipper_workspace_placements(request.clip_id)[0]
                    if len(existing_rows) == 1
                    else next(
                        placement
                        for placement in self.db.list_note_clipper_workspace_placements(request.clip_id)
                        if str(placement.get("workspace_id")) == request.workspace.workspace_id
                    )
                )

        workspace_note = self.db.add_workspace_note(
            request.workspace.workspace_id,
            {
                "title": note_summary.title,
                "content": note_content,
                "keywords": request.note.keywords,
            },
        )
        placement_row = self.db.upsert_note_clipper_workspace_placement(
            clip_id=request.clip_id,
            workspace_id=request.workspace.workspace_id,
            workspace_note_id=int(workspace_note["id"]),
            source_note_id=note_summary.id,
            source_note_version=note_summary.version,
        )
        return self._placement_response_from_row(placement_row)

    def _ensure_workspace_source(
        self,
        *,
        request: WebClipperSaveRequest,
        note_summary: WebClipperSavedNote,
        media_content: str,
    ) -> dict[str, Any]:
        if request.workspace is None:
            raise InputError("workspace payload is required for workspace source promotion.")  # noqa: TRY003
        if self.media_db is None:
            raise CharactersRAGDBError("Media DB is unavailable for workspace source promotion.")  # noqa: TRY003

        content = media_content.strip()
        if not content:
            content = "No readable page text was captured for this browser clip."

        media_id = self._upsert_workspace_clip_media(
            request=request,
            title=note_summary.title,
            content=content,
            workspace_id=request.workspace.workspace_id,
        )
        source_id = self._workspace_source_id(request.clip_id)
        existing_sources = self.db.list_workspace_sources(request.workspace.workspace_id)
        source_data = {
            "id": source_id,
            "media_id": media_id,
            "title": note_summary.title,
            "source_type": "web_clip",
            "url": request.source_url,
            "position": len(existing_sources),
            "selected": True,
        }
        if request.workspace.default_review_state == "needs_review":
            source_data["review_state"] = "needs_review"
        return self.db.add_workspace_source(request.workspace.workspace_id, source_data)

    def _upsert_workspace_clip_media(
        self,
        *,
        request: WebClipperSaveRequest,
        title: str,
        content: str,
        workspace_id: str,
    ) -> int:
        metadata = {
            "source": "web_clipper",
            "clip_id": request.clip_id,
            "clip_type": request.clip_type,
            "source_url": request.source_url,
            "source_title": request.source_title,
            "workspace_id": workspace_id,
            "capture_metadata": request.capture_metadata,
        }
        chunks = [
            {
                "text": content,
                "start_char": 0,
                "end_char": len(content),
                "chunk_type": "web_clip",
                "metadata": {
                    "clip_id": request.clip_id,
                    "source_url": request.source_url,
                    "source": "web_clipper",
                },
            }
        ]
        try:
            media_id, _media_uuid, _message = MediaRepository.from_legacy_db(self.media_db).add_text_media(
                url=f"{_WEB_CLIPPER_SOURCE_PREFIX}://{quote(request.clip_id, safe='')}",
                title=title,
                media_type=request.clip_type or "web_clip",
                content=content,
                keywords=request.note.keywords,
                safe_metadata=json.dumps(metadata, ensure_ascii=False),
                source_hash=self._workspace_source_id(request.clip_id),
                overwrite=True,
                chunks=chunks,
                owner_user_id=self._coerce_owner_user_id(),
            )
        except (MediaConflictError, MediaDatabaseError, MediaInputError) as exc:
            raise CharactersRAGDBError("Media DB workspace clip persistence failed.") from exc  # noqa: TRY003
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise CharactersRAGDBError("Media DB workspace clip persistence failed.") from exc  # noqa: TRY003

        if media_id is None:
            raise CharactersRAGDBError("Media DB did not return a media id for the workspace clip.")  # noqa: TRY003
        return int(media_id)

    def _coerce_owner_user_id(self) -> int | None:
        try:
            return int(self.user_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _workspace_source_id(clip_id: str) -> str:
        return f"{_WEB_CLIPPER_SOURCE_PREFIX}:{clip_id}"

    def _notify_workspace_source_job(self, *, workspace_id: str, source: dict[str, Any]) -> None:
        if self.workspace_source_job_enqueuer is None:
            return
        try:
            self.workspace_source_job_enqueuer(workspace_id, source)
        except Exception as exc:
            raise CharactersRAGDBError("Workspace source job enqueue failed.") from exc  # noqa: TRY003

    def _sync_workspace_note(
        self,
        *,
        workspace_id: str,
        workspace_note_id: int,
        title: str,
        content: str,
        keywords: list[str],
    ) -> None:
        workspace_note = self.db.execute_query(
            "SELECT * FROM workspace_notes WHERE workspace_id = ? AND id = ? AND deleted = ?",
            (workspace_id, workspace_note_id, self._active_deleted_value()),
        ).fetchone()
        if workspace_note is None:
            raise ConflictError(
                f"Workspace note '{workspace_note_id}' not found.",
                entity="workspace_notes",
                entity_id=str(workspace_note_id),
            )

        desired_keywords_json = json.dumps(keywords)
        updates: dict[str, Any] = {}
        if str(workspace_note["title"] or "") != title:
            updates["title"] = title
        if str(workspace_note["content"] or "") != content:
            updates["content"] = content
        if str(workspace_note["keywords_json"] or "[]") != desired_keywords_json:
            updates["keywords_json"] = desired_keywords_json
        if not updates:
            return

        self.db.update_workspace_note(
            workspace_id,
            workspace_note_id,
            updates,
            expected_version=int(workspace_note["version"]),
        )

    def _derive_save_status(
        self,
        *,
        warnings: list[str],
        workspace_requested: bool,
        workspace_saved: bool,
    ) -> str:
        if workspace_requested and not workspace_saved:
            return "partially_saved"
        if warnings:
            return "saved_with_warnings"
        return "saved"

    def _build_note_summary(self, note: dict[str, Any]) -> WebClipperSavedNote:
        return WebClipperSavedNote(
            id=str(note["id"]),
            title=str(note.get("title") or ""),
            version=int(note.get("version") or 1),
        )

    def _placement_response_from_row(self, row: dict[str, Any]) -> WebClipperWorkspacePlacement:
        return WebClipperWorkspacePlacement(
            workspace_id=str(row["workspace_id"]),
            workspace_note_id=int(row["workspace_note_id"]),
            source_note_id=str(row.get("source_note_id") or row.get("clip_id") or ""),
            source_note_version=int(row["source_note_version"]) if row.get("source_note_version") is not None else None,
        )

    def _require_note(self, note_id: str) -> dict[str, Any]:
        note = self.db.get_note_by_id(note_id)
        if note is None:
            raise ConflictError("Canonical note not found.", entity="notes", entity_id=note_id)  # noqa: TRY003
        return note

    def _require_clip_document(self, clip_id: str) -> dict[str, Any]:
        document = self.db.get_note_clipper_document_by_clip_id(clip_id)
        if document is None:
            raise ConflictError("Clip document not found.", entity="note_clipper_documents", entity_id=clip_id)  # noqa: TRY003
        return document

    def _fetch_note_row(self, note_id: str, *, conn: Any) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ? AND deleted = ?",
            (note_id, self._active_deleted_value()),
        ).fetchone()
        return dict(row) if row else None

    def _truncate_inline_summary(self, enrichment_type: str, inline_summary: str | None) -> str | None:
        text = str(inline_summary or "").strip()
        if not text:
            return None
        per_type_limit = _MAX_OCR_INLINE if enrichment_type == "ocr" else _MAX_VLM_INLINE
        return text[: min(per_type_limit, _MAX_MACHINE_INLINE)].strip()

    def _strip_machine_section(self, content: str) -> str:
        start = content.find(_ENRICHMENT_SECTION_START)
        end = content.find(_ENRICHMENT_SECTION_END)
        if start == -1 or end == -1 or end < start:
            return content.strip()
        prefix = content[:start].rstrip()
        suffix = content[end + len(_ENRICHMENT_SECTION_END) :].strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}".strip()
        return (prefix or suffix).strip()

    def _get_note_attachments_dir(self, note_id: str, *, create: bool) -> Path:
        user_base_dir = DatabasePaths.get_user_base_directory(self.user_id)
        base_dir = (user_base_dir / _NOTES_ATTACHMENTS_DIRNAME).resolve()
        user_base_resolved = user_base_dir.resolve()
        try:
            base_dir.relative_to(user_base_resolved)
        except ValueError as exc:
            raise CharactersRAGDBError("Invalid attachment storage path.") from exc  # noqa: TRY003
        if create:
            base_dir.mkdir(parents=True, exist_ok=True)

        note_dir_name = self._safe_note_attachment_dirname(note_id)
        note_dir = (base_dir / note_dir_name).resolve()
        try:
            note_dir.relative_to(base_dir)
        except ValueError as exc:
            raise CharactersRAGDBError("Invalid note attachment path.") from exc  # noqa: TRY003
        if create:
            note_dir.mkdir(parents=True, exist_ok=True)
        return note_dir

    def _safe_note_attachment_dirname(self, note_id: str) -> str:
        text = str(note_id or "").strip()
        if not text:
            return "note"
        safe = sanitize_filename(text, max_total_length=96).replace(" ", "_").strip("._")
        if safe and safe not in {".", ".."}:
            return safe
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"note_{digest}"

    def _deterministic_attachment_file_name(self, *, slot: str, file_name: str | None) -> str:
        raw_name = str(file_name or slot or "").strip()
        if not raw_name:
            raise InputError("Attachment filename is required.")  # noqa: TRY003
        basename = Path(raw_name).name
        if basename != raw_name:
            raise InputError("Invalid attachment filename.")  # noqa: TRY003
        suffix = "".join(Path(basename).suffixes)
        if not suffix:
            suffix = mimetypes.guess_extension(mimetypes.guess_type(basename)[0] or "") or ".bin"
        safe_slot = (
            sanitize_filename(str(slot or ""), max_total_length=max(1, 180 - len(suffix))).replace(" ", "_").strip("._")
        )
        if not safe_slot:
            safe_slot = "attachment"
        return f"{safe_slot}{suffix.lower()}"

    def _attachment_metadata_path(self, file_path: Path) -> Path:
        return file_path.with_name(f"{file_path.name}{_NOTES_ATTACHMENT_META_SUFFIX}")

    def _list_attachments(self, *, note_id: str) -> list[WebClipperAttachmentRecord]:
        attachment_dir = self._get_note_attachments_dir(note_id, create=False)
        if not attachment_dir.exists():
            return []
        attachments: list[WebClipperAttachmentRecord] = []
        for item in sorted(attachment_dir.iterdir(), key=lambda path: path.name.lower()):
            if not item.is_file() or item.name.endswith(_NOTES_ATTACHMENT_META_SUFFIX):
                continue
            attachments.append(self._attachment_response(note_id=note_id, file_path=item))
        return attachments

    def _attachment_response(self, *, note_id: str, file_path: Path) -> WebClipperAttachmentRecord:
        metadata = {}
        metadata_path = self._attachment_metadata_path(file_path)
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (TypeError, ValueError, OSError):
                metadata = {}
        file_stat = file_path.stat()
        uploaded_at_raw = metadata.get("uploaded_at")
        uploaded_at = self._parse_uploaded_at(uploaded_at_raw, file_stat.st_mtime)
        encoded_note_id = quote(str(note_id), safe="")
        encoded_file_name = quote(file_path.name, safe="")
        return WebClipperAttachmentRecord(
            slot=str(metadata.get("slot") or Path(file_path.name).stem),
            file_name=file_path.name,
            original_file_name=str(metadata.get("original_file_name") or file_path.name),
            content_type=metadata.get("content_type") or mimetypes.guess_type(file_path.name)[0],
            size_bytes=int(metadata.get("size_bytes") or file_stat.st_size),
            uploaded_at=uploaded_at,
            url=f"/api/v1/notes/{encoded_note_id}/attachments/{encoded_file_name}",
        )

    def _parse_uploaded_at(self, value: Any, fallback_timestamp: float) -> datetime:
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
                except ValueError:
                    pass
        return datetime.fromtimestamp(fallback_timestamp, tz=timezone.utc)
