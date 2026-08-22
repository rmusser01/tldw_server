"""Shared content-kind negotiation, projection, and mutation seam for Slides."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any

from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    InputError,
    PresentationRow,
    PresentationSummaryRow,
    SlidesDatabase,
    bind_validated_standalone_source,
    decode_presentation_version_payload,
)
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationResult,
)
from tldw_Server_API.app.core.Slides.standalone_html_validation_pool import (
    StandaloneHtmlValidationPool,
)

CONTENT_KIND_HEADER = "X-Slides-Accept-Content-Kinds"
STRUCTURED_SLIDES = "structured_slides"
STANDALONE_HTML = "standalone_html"
KNOWN_CONTENT_KINDS = frozenset({STRUCTURED_SLIDES, STANDALONE_HTML})
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PresentationServiceError(RuntimeError):
    """Fixed source-free domain error suitable for REST and MCP mapping."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int,
        operation: str | None = None,
        content_kind: str | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.operation = operation
        self.content_kind = content_kind
        super().__init__(code)


def parse_accepted_content_kinds(raw: str | None) -> frozenset[str]:
    """Parse the closed additive representation signal exactly once."""
    if raw is None:
        return frozenset({STRUCTURED_SLIDES})
    parts = raw.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise PresentationServiceError("invalid_content_kind_header", status_code=400)
    tokens = [part.strip() for part in parts]
    if any(_TOKEN_RE.fullmatch(token) is None for token in tokens):
        raise PresentationServiceError("invalid_content_kind_header", status_code=400)
    accepted = KNOWN_CONTENT_KINDS.intersection(tokens)
    if not accepted:
        raise PresentationServiceError("invalid_content_kind_header", status_code=400)
    return frozenset(accepted)


def merge_vary_header(
    headers: MutableMapping[str, str],
    token: str = CONTENT_KIND_HEADER,
) -> None:
    """Merge one Vary token without dropping auth/origin variation."""
    current = headers.get("Vary", "")
    values = [item.strip() for item in current.split(",") if item.strip()]
    if token.lower() not in {item.lower() for item in values}:
        values.append(token)
    headers["Vary"] = ", ".join(values)


def merge_auth_vary_header(headers: MutableMapping[str, str]) -> None:
    """Vary source-bearing responses across every supported auth carrier."""
    for token in ("Authorization", "X-API-KEY", "Cookie"):
        merge_vary_header(headers, token)


def _json_value(raw: str | None, *, default: Any = None) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _common_detail(row: PresentationRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "description": row.description,
        "theme": row.theme,
        "source_type": row.source_type,
        "source_ref": _json_value(row.source_ref, default=row.source_ref),
        "source_query": row.source_query,
        "created_at": row.created_at,
        "last_modified": row.last_modified,
        "deleted": bool(row.deleted),
        "client_id": row.client_id,
        "version": int(row.version),
        "content_kind": row.content_kind,
    }


def presentation_detail(row: PresentationRow) -> dict[str, Any]:
    """Project one complete row into its active discriminated representation."""
    common = _common_detail(row)
    if row.content_kind == STANDALONE_HTML:
        return {
            **common,
            "html_document": row.html_document,
            "html_sha256": row.html_sha256,
            "html_bytes": row.html_bytes,
            "html_slide_count": row.html_slide_count,
            "generation_provenance": _json_value(row.generation_provenance_json, default={}),
        }
    if row.content_kind != STRUCTURED_SLIDES:
        raise PresentationServiceError("operation_not_supported_for_content_kind", status_code=409)
    slides = _json_value(row.slides, default=[])
    if not isinstance(slides, list):
        raise PresentationServiceError("presentation_payload_invalid", status_code=500)
    return {
        **common,
        "marp_theme": row.marp_theme,
        "template_id": row.template_id,
        "visual_style_id": row.visual_style_id,
        "visual_style_scope": row.visual_style_scope,
        "visual_style_name": row.visual_style_name,
        "visual_style_version": row.visual_style_version,
        "visual_style_snapshot": _json_value(row.visual_style_snapshot),
        "settings": _json_value(row.settings),
        "studio_data": _json_value(row.studio_data),
        "slides": slides,
        "custom_css": row.custom_css,
    }


def presentation_summary(row: PresentationSummaryRow) -> dict[str, Any]:
    """Project one source-free summary row into a discriminated representation."""
    common = {
        "id": row.id,
        "title": row.title,
        "description": row.description,
        "theme": row.theme,
        "created_at": row.created_at,
        "last_modified": row.last_modified,
        "deleted": bool(row.deleted),
        "version": int(row.version),
        "content_kind": row.content_kind,
        "provenance": {
            "source_kind": row.source_kind,
            "provider": row.provider,
            "model": row.model,
        },
    }
    if row.content_kind == STANDALONE_HTML:
        return {
            **common,
            "html_slide_count": row.html_slide_count,
            "html_bytes": row.html_bytes,
        }
    return {**common, "slide_count": row.slide_count}


def snapshot_detail(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project one authorized full snapshot, defaulting legacy rows to structured."""
    content_kind = payload.get("content_kind", STRUCTURED_SLIDES)
    common = {
        "id": payload.get("id") or payload.get("presentation_id"),
        "title": payload.get("title"),
        "description": payload.get("description"),
        "theme": payload.get("theme") or "black",
        "source_type": payload.get("source_type"),
        "source_ref": (
            _json_value(payload.get("source_ref"), default=payload.get("source_ref"))
            if isinstance(payload.get("source_ref"), str)
            else payload.get("source_ref")
        ),
        "source_query": payload.get("source_query"),
        "created_at": payload.get("created_at") or payload.get("last_modified"),
        "last_modified": payload.get("last_modified") or payload.get("created_at"),
        "deleted": bool(payload.get("deleted")),
        "client_id": payload.get("client_id") or "",
        "version": int(payload.get("version") or 0),
        "content_kind": content_kind,
    }
    if not common["id"] or not common["title"] or not common["created_at"]:
        raise PresentationServiceError("version_payload_invalid", status_code=500)
    if content_kind == STANDALONE_HTML:
        return {
            **common,
            "html_document": payload.get("html_document"),
            "html_sha256": payload.get("html_sha256"),
            "html_bytes": payload.get("html_bytes"),
            "html_slide_count": payload.get("html_slide_count"),
            "generation_provenance": _json_value(payload.get("generation_provenance_json"), default={}),
        }
    slides = payload.get("slides", [])
    if isinstance(slides, str):
        slides = _json_value(slides, default=[])
    return {
        **common,
        "marp_theme": payload.get("marp_theme"),
        "template_id": payload.get("template_id"),
        "visual_style_id": payload.get("visual_style_id"),
        "visual_style_scope": payload.get("visual_style_scope"),
        "visual_style_name": payload.get("visual_style_name"),
        "visual_style_version": payload.get("visual_style_version"),
        "visual_style_snapshot": (
            _json_value(payload.get("visual_style_snapshot"))
            if isinstance(payload.get("visual_style_snapshot"), str)
            else payload.get("visual_style_snapshot")
        ),
        "settings": (
            _json_value(payload.get("settings"))
            if isinstance(payload.get("settings"), str)
            else payload.get("settings")
        ),
        "studio_data": (
            _json_value(payload.get("studio_data"))
            if isinstance(payload.get("studio_data"), str)
            else payload.get("studio_data")
        ),
        "slides": slides,
        "custom_css": payload.get("custom_css"),
    }


class PresentationService:
    """The sole content-kind-aware domain seam used by Slides transports."""

    def __init__(
        self,
        db: SlidesDatabase,
        *,
        validation_pool: StandaloneHtmlValidationPool | None = None,
    ) -> None:
        self.db = db
        self.validation_pool = validation_pool

    def _require_validation_pool(self) -> StandaloneHtmlValidationPool:
        if self.validation_pool is None:
            raise PresentationServiceError("validator_unavailable", status_code=503)
        return self.validation_pool

    @staticmethod
    def operation_not_supported(content_kind: str, operation: str) -> PresentationServiceError:
        """Build one bounded structured operation error."""
        bounded_operation = operation if _TOKEN_RE.fullmatch(operation) and len(operation) <= 64 else "unknown"
        bounded_kind = (
            content_kind
            if isinstance(content_kind, str) and len(content_kind) <= 64 and _TOKEN_RE.fullmatch(content_kind)
            else "unknown"
        )
        return PresentationServiceError(
            "operation_not_supported_for_content_kind",
            status_code=409,
            operation=bounded_operation,
            content_kind=bounded_kind,
        )

    def guard_target(
        self,
        presentation_id: str,
        accepted_content_kinds: Iterable[str],
        *,
        include_deleted: bool = False,
    ):
        row = self.db.get_presentation_kind(presentation_id, include_deleted=include_deleted)
        if row.content_kind not in set(accepted_content_kinds):
            raise PresentationServiceError("content_kind_not_accepted", status_code=406)
        return row

    @staticmethod
    def require_operation(
        content_kind: str,
        operation: str,
        *,
        export_format: str | None = None,
    ) -> None:
        if content_kind == STRUCTURED_SLIDES:
            if operation in {"html_source", "draft_attachment"} or export_format == "html":
                raise PresentationService.operation_not_supported(content_kind, operation)
            return
        if content_kind != STANDALONE_HTML:
            raise PresentationService.operation_not_supported(content_kind, operation)
        allowed = {"read", "versions", "delete", "restore", "html_source", "draft_attachment"}
        if operation == "export" and export_format in {"html", "json"}:
            return
        if operation not in allowed:
            raise PresentationService.operation_not_supported(content_kind, operation)

    @staticmethod
    def require_generic_create(content_kind: str) -> None:
        if content_kind == STANDALONE_HTML:
            raise PresentationServiceError("standalone_html_creation_requires_generation", status_code=409)
        if content_kind != STRUCTURED_SLIDES:
            raise PresentationService.operation_not_supported(content_kind, "create")

    def list_summaries(self, *, accepted_content_kinds: Iterable[str], **kwargs: Any):
        return self.db.list_presentation_summaries(accepted_content_kinds=accepted_content_kinds, **kwargs)

    def search_summaries(self, *, accepted_content_kinds: Iterable[str], **kwargs: Any):
        return self.db.search_presentation_summaries(accepted_content_kinds=accepted_content_kinds, **kwargs)

    def get_detail(
        self,
        presentation_id: str,
        accepted_content_kinds: Iterable[str],
        *,
        include_deleted: bool = False,
    ) -> PresentationRow:
        self.guard_target(
            presentation_id,
            accepted_content_kinds,
            include_deleted=include_deleted,
        )
        return self.db.get_presentation_by_id(presentation_id, include_deleted=include_deleted)

    def create_standalone_for_worker(
        self,
        *,
        presentation_id: str,
        html_document: str | bytes,
        validation_result: StandaloneHtmlValidationResult,
        generation_job_uuid: str,
        generation_provenance: Mapping[str, Any],
    ) -> PresentationRow:
        derived = validation_result
        source = bind_validated_standalone_source(html_document, derived)
        provenance_json = json.dumps(
            dict(generation_provenance),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return self.db.create_presentation(
            presentation_id=presentation_id,
            title=derived.title,
            description=None,
            theme="black",
            marp_theme=None,
            settings=None,
            studio_data=None,
            slides="[]",
            slides_text=derived.indexable_text,
            source_type=str(generation_provenance.get("source_kind") or "prompt"),
            source_ref=None,
            source_query=None,
            custom_css=None,
            content_kind=STANDALONE_HTML,
            html_document=source,
            html_sha256=derived.html_sha256,
            html_bytes=derived.html_bytes,
            html_slide_count=derived.slide_count,
            generation_job_uuid=generation_job_uuid,
            generation_provenance_json=provenance_json,
        )

    async def save_html_source(
        self,
        *,
        presentation_id: str,
        html_document: str | bytes,
        expected_version: int,
    ) -> PresentationRow:
        current = self.db.get_presentation_kind(presentation_id)
        if current.content_kind != STANDALONE_HTML:
            raise InputError("operation_not_supported_for_content_kind")
        if current.version != expected_version:
            identity = self.db.get_presentation_source_identity(presentation_id)
            if isinstance(html_document, bytes):
                candidate_bytes = html_document
            elif isinstance(html_document, str):
                try:
                    candidate_bytes = html_document.encode("utf-8")
                except UnicodeEncodeError:
                    candidate_bytes = b""
            else:
                candidate_bytes = b""
            if (
                candidate_bytes
                and identity.id == current.id
                and identity.content_kind == STANDALONE_HTML
                and identity.version == current.version
                and identity.html_bytes == len(candidate_bytes)
                and isinstance(identity.html_sha256, str)
                and hashlib.sha256(candidate_bytes).hexdigest() == identity.html_sha256
            ):
                row = self.db.get_presentation_by_id(presentation_id, include_deleted=False)
                if (
                    row.version == current.version
                    and row.content_kind == STANDALONE_HTML
                    and row.title == identity.title
                    and row.html_document is not None
                    and row.html_document.encode("utf-8") == candidate_bytes
                    and self.db.presentation_row_invariant_holds(row)
                ):
                    return row
            raise ConflictError(
                "version_conflict",
                entity="presentations",
                identifier=presentation_id,
            )
        derived = await self._require_validation_pool().validate(html_document)
        return self.db.save_standalone_html_source(
            presentation_id=presentation_id,
            html_document=html_document,
            validation_result=derived,
            expected_version=expected_version,
        )

    def prepare_draft_attachment(
        self,
        *,
        presentation_id: str,
        html_document: bytes,
    ) -> bytes:
        """Validate only the bounded recovery-echo transport invariants."""
        kind = self.db.get_presentation_kind(presentation_id)
        if kind.content_kind != STANDALONE_HTML:
            raise self.operation_not_supported(kind.content_kind, "draft_attachment")
        if not isinstance(html_document, bytes) or len(html_document) > 1_048_576:
            raise PresentationServiceError("standalone_html_storage_limit", status_code=413)
        try:
            decoded = html_document.decode("utf-8")
        except UnicodeDecodeError:
            raise PresentationServiceError("standalone_html_invalid_document", status_code=422) from None
        if "\x00" in decoded:
            raise PresentationServiceError("standalone_html_invalid_document", status_code=422)
        return html_document

    async def restore_version(
        self,
        *,
        presentation_id: str,
        version: int,
        expected_version: int,
        structured_restore: Callable[[Mapping[str, Any]], PresentationRow] | None = None,
    ) -> PresentationRow:
        current_kind = self.db.get_presentation_kind(presentation_id, include_deleted=True)
        kind = current_kind.content_kind
        if kind == STRUCTURED_SLIDES:
            if structured_restore is None:
                raise PresentationServiceError("structured_restore_handler_required", status_code=500)
            version_row = self.db.get_presentation_version(
                presentation_id=presentation_id,
                version=version,
            )
            payload = decode_presentation_version_payload(version_row.payload_json)
            if payload.get("content_kind", STRUCTURED_SLIDES) != kind:
                raise InputError("version_content_kind_mismatch")
            return structured_restore(payload)
        if kind != STANDALONE_HTML:
            raise self.operation_not_supported(kind, "restore")
        if current_kind.version != expected_version:
            raise ConflictError(
                "version_conflict",
                entity="presentations",
                identifier=presentation_id,
            )
        version_row = self.db.get_presentation_version(
            presentation_id=presentation_id,
            version=version,
        )
        payload = decode_presentation_version_payload(version_row.payload_json)
        if payload.get("content_kind") != STANDALONE_HTML:
            raise InputError("version_content_kind_mismatch")
        source = payload.get("html_document")
        if not isinstance(source, str):
            raise InputError("version_payload_invalid")
        derived = await self._require_validation_pool().validate(source)
        return self.db.restore_standalone_html_version(
            presentation_id=presentation_id,
            version=version,
            expected_version=expected_version,
            html_document=source,
            validation_result=derived,
            expected_payload_json=version_row.payload_json,
        )

    async def validate_saved_standalone(self, row: PresentationRow) -> StandaloneHtmlValidationResult:
        """Revalidate an exact persisted source before standalone export."""
        source = row.html_document
        if (
            row.content_kind != STANDALONE_HTML
            or not isinstance(source, str)
            or not self.db.presentation_row_invariant_holds(row)
        ):
            raise PresentationServiceError("standalone_html_response_invalid", status_code=500)
        derived = await self._require_validation_pool().validate(source)
        if (
            row.title,
            row.html_sha256,
            row.html_bytes,
            row.html_slide_count,
            row.slides_text,
        ) != (
            derived.title,
            derived.html_sha256,
            derived.html_bytes,
            derived.slide_count,
            derived.indexable_text,
        ):
            raise PresentationServiceError("standalone_html_response_invalid", status_code=500)
        return derived

    async def restore_presentation(
        self,
        *,
        presentation_id: str,
        expected_version: int,
    ) -> PresentationRow:
        """Undelete structured content directly or validated standalone source."""
        kind = self.db.get_presentation_kind(presentation_id, include_deleted=True)
        if kind.content_kind == STRUCTURED_SLIDES:
            return self.db.restore_presentation(presentation_id, expected_version)
        if kind.content_kind != STANDALONE_HTML:
            raise self.operation_not_supported(kind.content_kind, "restore")
        if kind.version != expected_version:
            raise ConflictError(
                "version_conflict",
                entity="presentations",
                identifier=presentation_id,
            )
        row = self.db.get_presentation_by_id(presentation_id, include_deleted=True)
        if row.version != expected_version:
            raise ConflictError(
                "version_conflict",
                entity="presentations",
                identifier=presentation_id,
            )
        if row.deleted != 1:
            raise InputError("operation_not_supported_for_content_kind")
        derived = await self.validate_saved_standalone(row)
        return self.db.restore_validated_standalone_presentation(
            presentation_id=presentation_id,
            html_document=row.html_document,
            validation_result=derived,
            expected_version=expected_version,
        )

    def delete_presentation(
        self,
        *,
        presentation_id: str,
        expected_version: int,
    ) -> PresentationRow | dict[str, Any]:
        kind = self.db.get_presentation_kind(presentation_id)
        if kind.content_kind == STANDALONE_HTML:
            deleted = self.db.soft_delete_presentation(presentation_id, expected_version)
            return {
                "id": deleted.id,
                "content_kind": deleted.content_kind,
                "deleted_at": deleted.last_modified,
            }
        return self.db.soft_delete_presentation(presentation_id, expected_version)


__all__ = [
    "CONTENT_KIND_HEADER",
    "KNOWN_CONTENT_KINDS",
    "PresentationService",
    "PresentationServiceError",
    "STANDALONE_HTML",
    "STRUCTURED_SLIDES",
    "merge_auth_vary_header",
    "merge_vary_header",
    "parse_accepted_content_kinds",
    "presentation_detail",
    "presentation_summary",
    "snapshot_detail",
]
