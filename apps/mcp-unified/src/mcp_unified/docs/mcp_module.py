from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .acquisition.extract import available_extractors
from .acquisition.service import DocsAcquisitionService
from .importers.local import DocsImportService
from .models import AccessScope, ContextRequest, SearchFilters, SearchRequest
from .retrieval.aliases import DocsAliasResolver
from .retrieval.context import DocsContextBuilder
from .retrieval.search import DocsRetrievalService
from .settings import DocsSettings
from .store.sqlite import DocsCatalogStore

WRITE_CATEGORIES = {"ingestion", "management"}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    category: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
        "metadata": {"category": category, "readOnlyHint": category not in WRITE_CATEGORIES},
    }


def _web_policy_status(settings: DocsSettings) -> dict[str, Any]:
    return {
        "allow_arbitrary_public_domains": settings.allow_arbitrary_public_domains,
        "preapproved_domains": list(settings.preapproved_domains),
        "allowed_url_prefixes": list(settings.allowed_url_prefixes),
        "denied_domains": list(settings.denied_domains),
        "max_url_redirects": settings.max_url_redirects,
        "max_url_body_bytes": settings.max_url_body_bytes,
        "url_request_timeout_seconds": settings.url_request_timeout_seconds,
        "allowed_content_types": list(settings.allowed_content_types),
        "url_user_agent": settings.url_user_agent,
        "respect_robots": settings.respect_robots,
    }


def _source_sync_status(settings: DocsSettings) -> dict[str, Any]:
    return {
        "enabled": settings.enable_source_sync,
        "max_sync_documents": settings.max_sync_documents,
        "max_sync_pages": settings.max_sync_pages,
        "max_sync_run_items": settings.max_sync_run_items,
        "default_stale_policy": settings.default_stale_policy,
        "sitemap_sync_enabled": settings.sitemap_sync_enabled,
        "persist_url_query_strings": settings.persist_url_query_strings,
    }


class DocsMCPToolProvider:
    def __init__(self, *, settings: DocsSettings, store: DocsCatalogStore | None = None) -> None:
        self.settings = settings
        self.store = store or DocsCatalogStore(settings.db_path)
        self.store.migrate()
        self.retrieval = DocsRetrievalService(self.store)
        self.context = DocsContextBuilder(self.retrieval)
        self.aliases = DocsAliasResolver(self.store)
        self.importer = DocsImportService(settings=settings, store=self.store)
        self.acquisition = (
            DocsAcquisitionService(settings=settings, store=self.store) if settings.enable_web_acquisition else None
        )

    def tool_definitions(self) -> list[dict[str, Any]]:
        tools = [
            _tool(
                "docs.search",
                "Search the local docs corpus.",
                {"query": {"type": "string"}, "limit": {"type": "integer"}},
                ["query"],
                "search",
            ),
            _tool(
                "docs.get",
                "Get a document, section, or chunk.",
                {"id": {"type": "string"}, "mode": {"type": "string"}},
                ["id"],
                "retrieval",
            ),
            _tool(
                "docs.context",
                "Build a bounded RAG context pack.",
                {
                    "query": {"type": "string"},
                    "max_chunks": {"type": "integer"},
                    "max_characters": {"type": "integer"},
                },
                ["query"],
                "retrieval",
            ),
            _tool(
                "docs.resolve",
                "Resolve a document, collection, source, keyword, or package-like docs name.",
                {"name": {"type": "string"}},
                ["name"],
                "retrieval",
            ),
            _tool(
                "docs.list",
                "List docs corpus records.",
                {"kind": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}},
                ["kind"],
                "retrieval",
            ),
            _tool("docs.status", "Report docs corpus health and capability status.", {}, [], "retrieval"),
            _tool(
                "docs.import_path",
                "Import local files under configured trusted roots.",
                {
                    "path": {"type": "string"},
                    "keywords": {"type": "array"},
                    "collections": {"type": "array"},
                },
                ["path"],
                "ingestion",
            ),
            _tool("docs.collections.list", "List collections.", {}, [], "retrieval"),
            _tool(
                "docs.collections.create",
                "Create a collection.",
                {"name": {"type": "string"}, "description": {"type": "string"}},
                ["name"],
                "management",
            ),
            _tool(
                "docs.collections.update",
                "Update a collection.",
                {"name": {"type": "string"}, "description": {"type": "string"}},
                ["name"],
                "management",
            ),
            _tool(
                "docs.collections.set_membership",
                "Set collection membership.",
                {
                    "collection": {"type": "string"},
                    "document_id": {"type": "integer"},
                    "action": {"type": "string"},
                },
                ["collection", "document_id", "action"],
                "management",
            ),
            _tool("docs.keywords.list", "List keywords.", {}, [], "retrieval"),
            _tool(
                "docs.keywords.apply",
                "Apply keywords to a document.",
                {"document_id": {"type": "integer"}, "keywords": {"type": "array"}},
                ["document_id", "keywords"],
                "management",
            ),
            _tool(
                "resolve-library-id",
                "Context7-compatible library id resolver backed by docs collections.",
                {"libraryName": {"type": "string"}},
                ["libraryName"],
                "retrieval",
            ),
            _tool(
                "get-library-docs",
                "Context7-compatible docs retrieval backed by docs.context.",
                {
                    "context7CompatibleLibraryID": {"type": "string"},
                    "topic": {"type": "string"},
                    "tokens": {"type": "integer"},
                },
                ["context7CompatibleLibraryID"],
                "retrieval",
            ),
        ]
        if self.acquisition is not None:
            tools.append(
                _tool(
                    "docs.ingest_url",
                    "Fetch and ingest one approved HTTP or HTTPS page into the local docs corpus.",
                    {
                        "url": {"type": "string"},
                        "keywords": {"type": "array"},
                        "collections": {"type": "array"},
                        "title": {"type": "string"},
                    },
                    ["url"],
                    "ingestion",
                )
            )
        return tools

    def execute(self, tool_name: str, arguments: dict[str, Any] | None, *, scope: AccessScope) -> Any:
        args = dict(arguments or {})
        if tool_name == "docs.status":
            status = self.store.status()
            status["web_acquisition_enabled"] = self.settings.enable_web_acquisition
            status["web_acquisition_available"] = self.acquisition is not None
            status["web_extractors"] = available_extractors() if self.acquisition is not None else []
            status["web_source_profile"] = self.settings.web_source_profile
            status["web_policy"] = _web_policy_status(self.settings)
            status["source_sync"] = _source_sync_status(self.settings)
            status["web_acquisition_unavailable_reason"] = (
                None if self.acquisition is not None else "web_acquisition_disabled"
            )
            return status
        if tool_name == "docs.search":
            query = _required_str(args, "query")
            return self.retrieval.search(
                scope=scope,
                request=SearchRequest(
                    query=query,
                    filters=_filters_from_args(args),
                    limit=int(args.get("limit", 10)),
                    offset=int(args.get("offset", 0)),
                    snippet_length=int(args.get("snippet_length", 300)),
                ),
            )
        if tool_name == "docs.context":
            query = _required_str(args, "query")
            return self.context.build(
                scope=scope,
                request=ContextRequest(
                    query=query,
                    filters=_filters_from_args(args),
                    max_chunks=int(args.get("max_chunks", 8)),
                    max_documents=int(args.get("max_documents", 4)),
                    max_characters=int(args.get("max_characters", 12_000)),
                ),
            )
        if tool_name == "docs.resolve":
            return self.aliases.resolve(scope=scope, name=_required_str(args, "name"))
        if tool_name == "resolve-library-id":
            return self.aliases.resolve_library_id(scope=scope, library_name=_required_str(args, "libraryName"))
        if tool_name == "get-library-docs":
            collection = _required_str(args, "context7CompatibleLibraryID")
            topic = str(args.get("topic") or collection)
            token_budget = int(args.get("tokens", 3000))
            result = self.context.build(
                scope=scope,
                request=ContextRequest(
                    query=topic,
                    filters=SearchFilters(collection=collection),
                    max_characters=max(1, token_budget) * 4,
                ),
            )
            result["canonical_tool"] = "docs.context"
            return result
        if tool_name == "docs.import_path":
            path = _required_str(args, "path")
            return self.importer.import_path(
                scope=scope,
                path=Path(path),
                keywords=tuple(str(item) for item in args.get("keywords") or ()),
                collection_names=tuple(str(item) for item in args.get("collections") or ()),
            )
        if tool_name == "docs.ingest_url":
            if self.acquisition is None:
                return {"status": "capability_disabled", "reason_code": "web_acquisition_disabled"}
            url = _optional_str(args.get("url"))
            if url is None:
                raise ValueError("url is required")
            return self.acquisition.ingest_url(
                scope=scope,
                url=url,
                keywords=tuple(str(item) for item in args.get("keywords") or ()),
                collection_names=tuple(str(item) for item in args.get("collections") or ()),
                title_override=_optional_str(args.get("title")),
            )
        return self._execute_management_or_list(tool_name=tool_name, args=args, scope=scope)

    def _execute_management_or_list(self, *, tool_name: str, args: dict[str, Any], scope: AccessScope) -> Any:
        if tool_name == "docs.get":
            return self.retrieval.get(
                scope=scope, target=_required_str(args, "id"), mode=str(args.get("mode") or "snippet")
            )
        if tool_name == "docs.list":
            kind = _required_str(args, "kind")
            limit = int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            if kind == "documents":
                return self.retrieval.list_documents(scope=scope, limit=limit, offset=offset)
            if kind == "collections":
                return self.retrieval.list_collections(scope=scope)
            if kind == "keywords":
                return self.retrieval.list_keywords(scope=scope)
            if kind == "sources":
                return {"sources": [], "warnings": [{"code": "sources_not_populated_in_stage1"}]}
            raise ValueError(f"Unsupported docs.list kind: {kind}")
        if tool_name == "docs.collections.list":
            return self.retrieval.list_collections(scope=scope)
        if tool_name == "docs.collections.create":
            name = _required_str(args, "name")
            description = str(args.get("description") or "")
            collection_id = self.store.create_collection(scope=scope, name=name, description=description)
            return {"status": "created", "id": collection_id, "name": name}
        if tool_name == "docs.collections.update":
            name = _required_str(args, "name")
            description = str(args.get("description") or "")
            updated = self.store.update_collection(scope=scope, name=name, description=description)
            return {"status": "updated" if updated else "unchanged", "name": name}
        if tool_name == "docs.collections.set_membership":
            collection = _required_str(args, "collection")
            document_id = _required_int(args, "document_id")
            action = _required_str(args, "action").lower()
            result = self.store.set_collection_membership(
                scope=scope,
                collection=collection,
                document_id=document_id,
                action=action,
            )
            return {"status": result, "collection": collection, "document_id": document_id}
        if tool_name == "docs.keywords.list":
            return self.retrieval.list_keywords(scope=scope)
        if tool_name == "docs.keywords.apply":
            document_id = _required_int(args, "document_id")
            if "keywords" not in args:
                raise ValueError("keywords is required")
            keywords = tuple(str(item) for item in args.get("keywords") or ())
            self.store.apply_keywords(scope=scope, document_id=document_id, keywords=keywords)
            return {"status": "updated", "document_id": document_id, "keywords": list(keywords)}
        raise ValueError(f"Unknown docs tool: {tool_name}")


def _filters_from_args(args: dict[str, Any]) -> SearchFilters:
    raw_filters = args.get("filters") or {}
    if not isinstance(raw_filters, dict):
        raw_filters = {}

    def value(name: str) -> Any:
        return raw_filters.get(name, args.get(name))

    keywords = value("keywords") or ()
    if isinstance(keywords, str):
        keyword_tuple = (keywords,)
    elif isinstance(keywords, Iterable):
        keyword_tuple = tuple(str(item) for item in keywords)
    else:
        keyword_tuple = (str(keywords),)

    return SearchFilters(
        collection=_optional_str(value("collection")),
        keywords=keyword_tuple,
        document_type=_optional_str(value("document_type")),
        uri_prefix=_optional_str(value("uri_prefix")),
        package=_optional_str(value("package")),
        version=_optional_str(value("version")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_str(args: dict[str, Any], name: str) -> str:
    value = _optional_str(args.get(name))
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _required_int(args: dict[str, Any], name: str) -> int:
    if name not in args:
        raise ValueError(f"{name} is required")
    return int(args[name])
