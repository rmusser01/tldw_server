from __future__ import annotations

from typing import Any

from .models import AccessScope, SyncSourceRequest
from .settings import DocsSettings
from .source_utils import redacted_url_for_display
from .store.sqlite import DocsCatalogStore

_SYNC_MODES = {"dry_run", "apply"}
_STALE_POLICIES = {"report", "tombstone"}
_URL_SOURCE_TYPES = {"url_page", "url_sitemap"}
_ZERO_COUNTS = {
    "created": 0,
    "updated": 0,
    "unchanged": 0,
    "missing": 0,
    "tombstoned": 0,
    "failed": 0,
    "skipped": 0,
}


class DocsSourceSyncService:
    def __init__(
        self,
        *,
        settings: DocsSettings,
        store: DocsCatalogStore,
        resolver: object | None = None,
        transport: object | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.resolver = resolver
        self.transport = transport

    def sync_source(self, *, scope: AccessScope, request: SyncSourceRequest) -> dict[str, Any]:
        if not self.settings.enable_source_sync:
            return {"status": "denied", "reason_code": "source_sync_disabled"}

        validation = _validate_request(request)
        if validation is not None:
            return validation

        source = self._get_source(scope=scope, request=request)
        if source is None:
            return {"status": "denied", "reason_code": "source_not_found"}
        if not source["sync_enabled"]:
            return _response(
                status="denied",
                reason_code="source_sync_disabled",
                source=source,
                request=request,
            )
        if source["source_type"] == "url_sitemap" and not self.settings.sitemap_sync_enabled:
            return _response(
                status="denied",
                reason_code="sitemap_sync_disabled",
                source=source,
                request=request,
            )

        return _response(
            status="skipped",
            reason_code="source_sync_unsupported_type",
            source=source,
            request=request,
        )

    def _get_source(self, *, scope: AccessScope, request: SyncSourceRequest) -> dict[str, Any] | None:
        if request.source_id is not None:
            return self.store.get_source(scope=scope, source_id=int(request.source_id))
        return self.store.get_source(scope=scope, canonical_uri=str(request.source_uri or "").strip())


def _validate_request(request: SyncSourceRequest) -> dict[str, Any] | None:
    has_id = request.source_id is not None
    has_uri = bool(str(request.source_uri or "").strip())
    if has_id == has_uri:
        return {"status": "denied", "reason_code": "source_selector_invalid"}

    if request.source_id is not None and (
        isinstance(request.source_id, bool) or not isinstance(request.source_id, int) or request.source_id <= 0
    ):
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "source_id"}
    if request.max_documents is not None and (
        isinstance(request.max_documents, bool)
        or not isinstance(request.max_documents, int)
        or request.max_documents <= 0
    ):
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "max_documents"}
    if request.max_pages is not None and (
        isinstance(request.max_pages, bool) or not isinstance(request.max_pages, int) or request.max_pages <= 0
    ):
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "max_pages"}

    if str(request.mode).strip().lower() not in _SYNC_MODES:
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "mode"}
    if str(request.stale_policy).strip().lower() not in _STALE_POLICIES:
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "stale_policy"}
    if not isinstance(request.force, bool):
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "force"}

    return None


def _response(
    *,
    status: str,
    reason_code: str,
    source: dict[str, Any],
    request: SyncSourceRequest,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "source": _source_summary(source),
        "mode": str(request.mode).strip().lower(),
        "stale_policy": str(request.stale_policy).strip().lower(),
        "force": request.force,
        "counts": dict(_ZERO_COUNTS),
        "warnings": [],
    }


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    public_source = source_summary_for_tool_response(source)
    return {
        "id": public_source["id"],
        "source_type": public_source["source_type"],
        "canonical_uri": public_source["canonical_uri"],
        "display_name": public_source["display_name"],
        "display_uri": public_source.get("display_uri", public_source["canonical_uri"]),
        "redacted_source_url": public_source.get("redacted_source_url"),
        "sync_enabled": public_source["sync_enabled"],
        "document_count": public_source["document_count"],
    }


def source_summary_for_tool_response(source: dict[str, Any]) -> dict[str, Any]:
    if source["source_type"] not in _URL_SOURCE_TYPES:
        return dict(source)

    redacted_uri = _redacted_source_uri(source)
    public_source = dict(source)
    public_source.pop("source_url", None)
    public_source["canonical_uri"] = redacted_uri
    public_source["display_uri"] = redacted_uri
    public_source["redacted_source_url"] = redacted_uri
    return public_source


def _redacted_source_uri(source: dict[str, Any]) -> str:
    redacted_uri = source.get("redacted_source_url")
    if isinstance(redacted_uri, str) and redacted_uri.strip():
        return redacted_uri.strip()
    raw_uri = source.get("canonical_uri") or source.get("source_url") or ""
    return redacted_url_for_display(str(raw_uri))
