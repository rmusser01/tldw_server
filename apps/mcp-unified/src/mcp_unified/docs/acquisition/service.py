from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any

from loguru import logger

from ..errors import DocsError
from ..importers.base import chunks_from_text
from ..models import AccessScope
from ..settings import DocsSettings
from ..source_utils import redacted_url_for_display, source_defaults_metadata, url_has_query
from ..store.sqlite import DocsCatalogStore
from .extract import extract_fetched_document
from .fetcher import URLFetcher
from .policy import SourcePolicy


class DocsAcquisitionService:
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
        self.policy = _policy_from_settings(settings)
        self.fetcher = URLFetcher(settings=settings, policy=self.policy, resolver=resolver, transport=transport)

    def ingest_url(
        self,
        *,
        scope: AccessScope,
        url: str,
        keywords: Iterable[str] = (),
        collection_names: Iterable[str] = (),
        title_override: str | None = None,
    ) -> dict[str, Any]:
        if not self.settings.enable_web_acquisition:
            logger.info("Docs URL ingestion disabled; skipping requested URL")
            return {
                "status": "capability_disabled",
                "reason_code": "capability_disabled",
                "final_url": None,
                "redirects": [],
            }

        fetched = self.fetcher.fetch(url)
        if fetched.status != "fetched":
            logger.info(
                "Docs URL fetch did not produce ingestable content: status={} reason={} final_url={} redirects={}",
                fetched.status,
                fetched.reason,
                fetched.final_url,
                len(fetched.redirects),
            )
            return {
                "status": fetched.status,
                "reason_code": fetched.reason,
                "final_url": fetched.final_url,
                "redirects": [asdict(item) for item in fetched.redirects],
                "safe_argument_hash": fetched.safe_argument_hash,
            }

        content_type = fetched.headers.get("content-type", "text/html")
        document_url = fetched.canonical_url or fetched.final_url or url
        parsed = extract_fetched_document(url=document_url, content_type=content_type, body=fetched.body)
        if title_override:
            parsed = replace(parsed, title=title_override)
        if not parsed.text.strip():
            return {
                "status": "failed",
                "reason_code": "extract_empty",
                "final_url": fetched.final_url,
                "redirects": [asdict(item) for item in fetched.redirects],
            }

        previous_hash = _existing_content_hash(self.store, scope, parsed.canonical_uri)
        new_hash = sha256(parsed.text.encode("utf-8")).hexdigest()
        keyword_tuple = tuple(keywords)
        collection_tuple = tuple(collection_names)
        chunks = [
            {"text": chunk, "citation": f"{parsed.source_url or parsed.canonical_uri}#{index + 1}"}
            for index, chunk in enumerate(chunks_from_text(parsed.text))
        ]
        document_id = self.store.upsert_document(
            scope=scope,
            title=parsed.title,
            document_type=parsed.document_type,
            canonical_uri=parsed.canonical_uri,
            source_path=parsed.source_path,
            source_url=parsed.source_url,
            text=parsed.text,
            sections=[asdict(section) for section in parsed.sections],
            chunks=chunks,
            keywords=keyword_tuple,
            collection_names=collection_tuple,
            metadata={
                "importer": "url",
                "extraction_method": parsed.extraction_method,
                "fetch_status_code": fetched.status_code,
                "redirect_count": len(fetched.redirects),
                "warnings": list(parsed.warnings),
            },
        )
        warnings = list(parsed.warnings)
        source: dict[str, Any] | None = None
        can_persist_query_source = self.settings.persist_url_query_strings or not url_has_query(parsed.canonical_uri)
        if can_persist_query_source:
            redacted_source_url = redacted_url_for_display(parsed.canonical_uri)
            source_id = self.store.upsert_source(
                scope=scope,
                source_type="url_page",
                canonical_uri=parsed.canonical_uri,
                display_name=parsed.title,
                source_path=None,
                source_url=parsed.canonical_uri,
                redacted_source_url=redacted_source_url,
                policy_profile=self.settings.web_source_profile,
                sync_enabled=True,
                metadata=source_defaults_metadata(
                    keywords=keyword_tuple,
                    collection_names=collection_tuple,
                ),
            )
            self.store.link_source_document(
                scope=scope,
                source_id=source_id,
                document_id=document_id,
                source_item_uri=parsed.canonical_uri,
                status="active",
                last_hash=new_hash,
                metadata={"importer": "url"},
            )
            source = self.store.get_source(scope=scope, source_id=source_id)
        else:
            warnings.append("url_query_not_persisted")
        status = "created" if previous_hash is None else "unchanged" if previous_hash == new_hash else "updated"
        return {
            "status": status,
            "reason_code": "ok",
            "document": {
                "id": document_id,
                "title": parsed.title,
                "canonical_uri": parsed.canonical_uri,
                "source_url": parsed.source_url,
                "chunks": len(chunks),
                "extraction_method": parsed.extraction_method,
            },
            "fetch": {
                "final_url": fetched.final_url,
                "status_code": fetched.status_code,
                "redirects": [asdict(item) for item in fetched.redirects],
                "content_type": content_type,
                "bytes": len(fetched.body),
            },
            "source": source,
            "warnings": warnings,
        }


def _policy_from_settings(settings: DocsSettings) -> SourcePolicy:
    return SourcePolicy(
        web_source_profile=settings.web_source_profile,
        preapproved_domains=settings.preapproved_domains,
        allowed_url_prefixes=settings.allowed_url_prefixes,
        denied_domains=settings.denied_domains,
        allow_arbitrary_public_domains=settings.allow_arbitrary_public_domains,
    )


def _existing_content_hash(store: DocsCatalogStore, scope: AccessScope, canonical_uri: str) -> str | None:
    try:
        document = store.get_document(scope, canonical_uri, mode="snippet")
    except DocsError as exc:
        if exc.code == "document_not_found":
            return None
        raise
    value = document.get("content_hash")
    return str(value) if value else None


__all__ = ["DocsAcquisitionService"]
