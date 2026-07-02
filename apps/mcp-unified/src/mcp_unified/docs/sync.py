from __future__ import annotations

from typing import Any

from .models import AccessScope, SyncSourceRequest
from .settings import DocsSettings
from .store.sqlite import DocsCatalogStore

_SYNC_MODES = {"dry_run", "apply"}
_STALE_POLICIES = {"report", "tombstone"}
_ZERO_COUNTS = {"created": 0, "updated": 0, "unchanged": 0, "failed": 0, "skipped": 0}


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

    if str(request.mode) not in _SYNC_MODES:
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "mode"}
    if str(request.stale_policy) not in _STALE_POLICIES:
        return {"status": "denied", "reason_code": "source_sync_request_invalid", "field": "stale_policy"}

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
        "mode": str(request.mode),
        "stale_policy": str(request.stale_policy),
        "counts": dict(_ZERO_COUNTS),
        "warnings": [],
    }


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source["id"],
        "source_type": source["source_type"],
        "canonical_uri": source["canonical_uri"],
        "display_name": source["display_name"],
        "sync_enabled": source["sync_enabled"],
        "document_count": source["document_count"],
    }
