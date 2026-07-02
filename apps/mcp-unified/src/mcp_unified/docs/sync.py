from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from .acquisition.service import DocsAcquisitionService
from .errors import DocsError
from .importers.base import ParsedDocument, chunks_from_text
from .importers.local import DocsImportService
from .models import AccessScope, SyncSourceRequest
from .settings import DocsSettings
from .source_utils import file_uri_for_path, redacted_url_for_display
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
        self.importer = DocsImportService(settings=settings, store=store)
        self.acquisition = DocsAcquisitionService(
            settings=settings,
            store=store,
            resolver=resolver,
            transport=transport,
        )

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

        if source["source_type"] == "local_file":
            return self._sync_local_file(scope=scope, source=source, request=request)
        if source["source_type"] == "local_directory":
            return self._sync_local_directory(scope=scope, source=source, request=request)
        if source["source_type"] == "url_page":
            return self._sync_url_page(scope=scope, source=source, request=request)
        return self._denied(source=source, request=request, reason_code="source_sync_unsupported_type")

    def _get_source(self, *, scope: AccessScope, request: SyncSourceRequest) -> dict[str, Any] | None:
        if request.source_id is not None:
            return self.store.get_source(scope=scope, source_id=int(request.source_id))
        return self.store.get_source(scope=scope, canonical_uri=str(request.source_uri or "").strip())

    def _sync_local_file(
        self,
        *,
        scope: AccessScope,
        source: dict[str, Any],
        request: SyncSourceRequest,
    ) -> dict[str, Any]:
        try:
            target = self._source_path(source)
            resolved = self.importer._assert_allowed_path(target)
        except DocsError as exc:
            return self._failed(source=source, request=request, reason_code=exc.code, warning=exc.message)

        files = [resolved] if resolved.is_file() else []
        return self._sync_local_paths(scope=scope, source=source, request=request, files=files)

    def _sync_local_directory(
        self,
        *,
        scope: AccessScope,
        source: dict[str, Any],
        request: SyncSourceRequest,
    ) -> dict[str, Any]:
        try:
            target = self._source_path(source)
            resolved = self.importer._assert_allowed_path(target)
            files = self.importer._iter_import_files(resolved) if resolved.is_dir() else []
        except DocsError as exc:
            return self._failed(source=source, request=request, reason_code=exc.code, warning=exc.message)

        return self._sync_local_paths(scope=scope, source=source, request=request, files=files)

    def _sync_local_paths(
        self,
        *,
        scope: AccessScope,
        source: dict[str, Any],
        request: SyncSourceRequest,
        files: list[Path],
    ) -> dict[str, Any]:
        mode = str(request.mode).strip().lower()
        stale_policy = str(request.stale_policy).strip().lower()
        counts = dict(_ZERO_COUNTS)
        items: list[dict[str, Any]] = []
        warnings: list[str] = []
        source_id = int(source["id"])
        links = self.store.source_document_links(scope=scope, source_id=source_id)
        links_by_uri = {str(link["source_item_uri"]): link for link in links}
        current_uris = {file_uri_for_path(file_path) for file_path in files}
        stale_links = [
            link
            for link in links
            if str(link["source_item_uri"]) not in current_uris and link.get("status") != "tombstoned"
        ]
        limit = self._document_limit(request)
        sync_item_count = len(files) + len(stale_links)
        if sync_item_count > limit:
            return _sync_response(
                status="denied",
                reason_code="source_sync_limit_exceeded",
                source=source,
                request=request,
                counts=dict(_ZERO_COUNTS),
                items=[],
                warnings=[f"sync_item_count={sync_item_count} exceeds max_documents={limit}"],
            )
        default_keywords = _metadata_string_tuple(source.get("metadata"), "default_keywords")
        default_collections = _metadata_string_tuple(source.get("metadata"), "default_collections")

        for file_path in files:
            item_uri = file_uri_for_path(file_path)
            try:
                parsed, chunks, content_hash = self._parsed_local_sync_item(file_path)
            except DocsError as exc:
                counts["failed"] += 1
                warnings.append(exc.code)
                items.append(
                    {
                        "source_item_uri": item_uri,
                        "status": "failed",
                        "reason_code": exc.code,
                    }
                )
                continue

            link = links_by_uri.get(item_uri)
            item_status = self._local_item_status(
                scope=scope,
                parsed=parsed,
                link=link,
                content_hash=content_hash,
                force=request.force,
            )
            counts[item_status] += 1
            document_id = int(link["document_id"]) if link is not None else None

            if mode == "apply":
                if item_status in {"created", "updated"}:
                    document_id = self.store.upsert_document_for_sync(
                        scope=scope,
                        title=parsed.title,
                        document_type=parsed.document_type,
                        canonical_uri=parsed.canonical_uri,
                        source_path=parsed.source_path,
                        source_url=parsed.source_url,
                        text=parsed.text,
                        sections=[asdict(section) for section in parsed.sections],
                        chunks=chunks,
                        source_default_keywords=default_keywords,
                        source_default_collections=default_collections,
                        metadata=_local_sync_metadata(parsed),
                    )
                    self.store.link_source_document(
                        scope=scope,
                        source_id=source_id,
                        document_id=document_id,
                        source_item_uri=item_uri,
                        status="active",
                        last_hash=content_hash,
                        metadata={"importer": "local"},
                    )
                elif link is not None and link.get("last_hash") != content_hash:
                    self.store.link_source_document(
                        scope=scope,
                        source_id=source_id,
                        document_id=int(link["document_id"]),
                        source_item_uri=item_uri,
                        status="active",
                        last_hash=content_hash,
                        metadata=link.get("metadata") or {"importer": "local"},
                    )

            items.append(
                {
                    "source_item_uri": item_uri,
                    "status": item_status,
                    "document_id": document_id,
                    "canonical_uri": parsed.canonical_uri,
                    "title": parsed.title,
                }
            )

        for link in stale_links:
            item_uri = str(link["source_item_uri"])
            if stale_policy == "tombstone":
                counts["tombstoned"] += 1
                item_status = "tombstoned"
                if mode == "apply":
                    self.store.tombstone_source_item(
                        scope=scope,
                        source_id=source_id,
                        source_item_uri=item_uri,
                    )
            else:
                counts["missing"] += 1
                item_status = "missing"
            items.append(
                {
                    "source_item_uri": item_uri,
                    "status": item_status,
                    "document_id": int(link["document_id"]),
                    "canonical_uri": link.get("canonical_uri"),
                    "title": link.get("title"),
                }
            )

        run_status = "partial" if counts["failed"] else "completed"
        reason_code = "partial_failure" if counts["failed"] else "ok"
        if mode == "apply":
            self.store.record_sync_run(
                scope=scope,
                source_id=source_id,
                mode=mode,
                status=run_status,
                requested_limits={
                    "max_documents": request.max_documents,
                    "max_pages": request.max_pages,
                    "effective_max_documents": self._document_limit(request),
                },
                counts=counts,
                warnings=warnings,
                error_code=None if reason_code == "ok" else reason_code,
                metadata={"source_type": source["source_type"]},
            )
            source = self.store.get_source(scope=scope, source_id=source_id) or source

        return _sync_response(
            status=run_status,
            reason_code=reason_code,
            source=source,
            request=request,
            counts=counts,
            items=items,
            warnings=warnings,
        )

    def _sync_url_page(
        self,
        *,
        scope: AccessScope,
        source: dict[str, Any],
        request: SyncSourceRequest,
    ) -> dict[str, Any]:
        if not self.settings.enable_web_acquisition:
            return _response(
                status="denied",
                reason_code="web_acquisition_disabled",
                source=source,
                request=request,
            )

        source_url = str(source.get("source_url") or "").strip()
        if not source_url:
            return self._failed(
                source=source,
                request=request,
                reason_code="source_url_missing",
                warning="URL source does not include a source URL.",
            )

        decision = self.acquisition.policy.evaluate(source_url)
        if decision.status != "allowed":
            reason_code = "approval_required" if decision.status == "approval_required" else "source_policy_denied"
            return _response(
                status="denied",
                reason_code=reason_code,
                source=source,
                request=request,
            )

        limit = self._document_limit(request)
        if limit < 1:
            return _sync_response(
                status="denied",
                reason_code="source_sync_limit_exceeded",
                source=source,
                request=request,
                counts=dict(_ZERO_COUNTS),
                items=[],
                warnings=[f"sync_item_count=1 exceeds max_documents={limit}"],
            )

        fetched_document = self.acquisition._fetch_parsed_url(url=source_url)
        if fetched_document["status"] != "fetched":
            reason_code = str(fetched_document.get("reason_code") or "fetch_failed")
            if fetched_document["status"] == "approval_required":
                return _response(
                    status="denied",
                    reason_code="approval_required",
                    source=source,
                    request=request,
                )
            if fetched_document["status"] == "denied":
                return _response(
                    status="denied",
                    reason_code=reason_code,
                    source=source,
                    request=request,
                )
            return self._failed(source=source, request=request, reason_code=reason_code, warning=reason_code)

        mode = str(request.mode).strip().lower()
        source_id = int(source["id"])
        parsed = fetched_document["parsed"]
        fetched = fetched_document["fetch"]
        chunks, content_hash = self._parsed_url_sync_item(parsed)
        links = self.store.source_document_links(scope=scope, source_id=source_id)
        link = _url_page_link(links=links, source_url=source_url, parsed_uri=parsed.canonical_uri)
        stale_active_links = _stale_url_page_links(links=links, parsed_uri=parsed.canonical_uri)
        item_status = (
            "updated"
            if stale_active_links
            else self._local_item_status(
                scope=scope,
                parsed=parsed,
                link=link,
                content_hash=content_hash,
                force=request.force,
            )
        )
        counts = dict(_ZERO_COUNTS)
        counts[item_status] = 1
        default_keywords = _metadata_string_tuple(source.get("metadata"), "default_keywords")
        default_collections = _metadata_string_tuple(source.get("metadata"), "default_collections")
        document_id = int(link["document_id"]) if link is not None else None

        if mode == "apply":
            if item_status in {"created", "updated"}:
                document_id = self.store.upsert_document_for_sync(
                    scope=scope,
                    title=parsed.title,
                    document_type=parsed.document_type,
                    canonical_uri=parsed.canonical_uri,
                    source_path=parsed.source_path,
                    source_url=parsed.source_url,
                    text=parsed.text,
                    sections=[asdict(section) for section in parsed.sections],
                    chunks=chunks,
                    source_default_keywords=default_keywords,
                    source_default_collections=default_collections,
                    metadata=_url_sync_metadata(parsed=parsed, fetched=fetched),
                )
                self.store.link_source_document(
                    scope=scope,
                    source_id=source_id,
                    document_id=document_id,
                    source_item_uri=parsed.canonical_uri,
                    status="active",
                    last_hash=content_hash,
                    metadata={"importer": "url"},
                )
                for stale_link in stale_active_links:
                    self.store.tombstone_source_item(
                        scope=scope,
                        source_id=source_id,
                        source_item_uri=str(stale_link["source_item_uri"]),
                    )
            elif link is not None and link.get("last_hash") != content_hash:
                self.store.link_source_document(
                    scope=scope,
                    source_id=source_id,
                    document_id=int(link["document_id"]),
                    source_item_uri=str(link["source_item_uri"]),
                    status="active",
                    last_hash=content_hash,
                    metadata=link.get("metadata") or {"importer": "url"},
                )

            self.store.record_sync_run(
                scope=scope,
                source_id=source_id,
                mode=mode,
                status="completed",
                requested_limits={
                    "max_documents": request.max_documents,
                    "max_pages": request.max_pages,
                    "effective_max_documents": self._document_limit(request),
                },
                counts=counts,
                warnings=list(parsed.warnings),
                error_code=None,
                metadata={"source_type": source["source_type"]},
            )
            source = self.store.get_source(scope=scope, source_id=source_id) or source

        return _sync_response(
            status="completed",
            reason_code="ok",
            source=source,
            request=request,
            counts=counts,
            items=[
                {
                    "source_item_uri": redacted_url_for_display(parsed.canonical_uri),
                    "status": item_status,
                    "document_id": document_id,
                    "canonical_uri": redacted_url_for_display(parsed.canonical_uri),
                    "title": parsed.title,
                }
            ],
            warnings=list(parsed.warnings),
        )

    def _denied(self, *, source: dict[str, Any], request: SyncSourceRequest, reason_code: str) -> dict[str, Any]:
        return _response(status="skipped", reason_code=reason_code, source=source, request=request)

    def _failed(
        self,
        *,
        source: dict[str, Any],
        request: SyncSourceRequest,
        reason_code: str,
        warning: str,
    ) -> dict[str, Any]:
        counts = dict(_ZERO_COUNTS)
        counts["failed"] = 1
        return _sync_response(
            status="failed",
            reason_code=reason_code,
            source=source,
            request=request,
            counts=counts,
            items=[],
            warnings=[warning],
        )

    def _source_path(self, source: dict[str, Any]) -> Path:
        source_path = source.get("source_path")
        if source_path:
            return Path(str(source_path))
        canonical_uri = str(source.get("canonical_uri") or "")
        if canonical_uri.startswith("file://"):
            return Path(canonical_uri.removeprefix("file://"))
        raise DocsError(
            code="source_path_missing",
            message="Local source does not include a source path.",
            details={"source_id": source.get("id")},
        )

    def _document_limit(self, request: SyncSourceRequest) -> int:
        limits = [self.settings.max_sync_documents, self.settings.max_sync_run_items]
        if request.max_documents is not None:
            limits.append(int(request.max_documents))
        return min(limits)

    def _parsed_local_sync_item(self, file_path: Path) -> tuple[ParsedDocument, list[dict[str, str]], str]:
        parsed = self.importer._parse_file(file_path)
        chunks = [
            {
                "text": chunk,
                "citation": f"{file_path.name}:{idx + 1}",
            }
            for idx, chunk in enumerate(chunks_from_text(parsed.text))
        ]
        content_hash = sha256(parsed.text.encode("utf-8")).hexdigest()
        return parsed, chunks, content_hash

    def _parsed_url_sync_item(self, parsed: ParsedDocument) -> tuple[list[dict[str, str]], str]:
        chunks = [
            {
                "text": chunk,
                "citation": f"{parsed.source_url or parsed.canonical_uri}#{idx + 1}",
            }
            for idx, chunk in enumerate(chunks_from_text(parsed.text))
        ]
        content_hash = sha256(parsed.text.encode("utf-8")).hexdigest()
        return chunks, content_hash

    def _local_item_status(
        self,
        *,
        scope: AccessScope,
        parsed: ParsedDocument,
        link: dict[str, Any] | None,
        content_hash: str,
        force: bool,
    ) -> str:
        if link is None:
            return "created"
        if force:
            return "updated"
        previous_hash = link.get("last_hash") or _existing_content_hash(self.store, scope, parsed.canonical_uri)
        if previous_hash == content_hash and link.get("status") == "active":
            return "unchanged"
        return "updated"


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
        "items": [],
        "warnings": [],
    }


def _sync_response(
    *,
    status: str,
    reason_code: str,
    source: dict[str, Any],
    request: SyncSourceRequest,
    counts: dict[str, int],
    items: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "source": _source_summary(source),
        "mode": str(request.mode).strip().lower(),
        "stale_policy": str(request.stale_policy).strip().lower(),
        "force": request.force,
        "counts": {key: int(counts.get(key, 0)) for key in _ZERO_COUNTS},
        "items": items,
        "warnings": warnings,
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


def _metadata_string_tuple(metadata: object, key: str) -> tuple[str, ...]:
    if not isinstance(metadata, dict):
        return ()
    value = metadata.get(key) or ()
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        values = (value,)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _existing_content_hash(store: DocsCatalogStore, scope: AccessScope, canonical_uri: str) -> str | None:
    try:
        document = store.get_document(scope, canonical_uri, mode="snippet")
    except DocsError as exc:
        if exc.code == "document_not_found":
            return None
        raise
    value = document.get("content_hash")
    return str(value) if value else None


def _url_page_link(
    *,
    links: list[dict[str, Any]],
    source_url: str,
    parsed_uri: str,
) -> dict[str, Any] | None:
    for link in links:
        if str(link.get("source_item_uri") or "") == parsed_uri:
            return link
    for link in links:
        if str(link.get("source_item_uri") or "") == source_url:
            return link
    active_links = [link for link in links if link.get("status") == "active"]
    if len(active_links) == 1:
        return active_links[0]
    return None


def _stale_url_page_links(*, links: list[dict[str, Any]], parsed_uri: str) -> list[dict[str, Any]]:
    return [
        link
        for link in links
        if link.get("status") == "active" and str(link.get("source_item_uri") or "") != parsed_uri
    ]


def _local_sync_metadata(parsed: ParsedDocument) -> dict[str, Any]:
    return {
        "importer": "local",
        "extraction_method": parsed.extraction_method,
        "warnings": list(parsed.warnings),
    }


def _url_sync_metadata(*, parsed: ParsedDocument, fetched: Any) -> dict[str, Any]:
    return {
        "importer": "url",
        "extraction_method": parsed.extraction_method,
        "fetch_status_code": fetched.status_code,
        "redirect_count": len(fetched.redirects),
        "warnings": list(parsed.warnings),
    }
