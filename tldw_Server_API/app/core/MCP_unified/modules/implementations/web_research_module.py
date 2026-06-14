"""Read-only ``web.research`` MCP tool composing ``web.search`` + ``web.fetch``.

The tool runs one search query and then fetches + extracts the top N results
into a single bounded research bundle. It does not re-implement search or fetch:
it drives a :class:`WebSearchModule` and :class:`WebFetchModule` through their
existing ``execute_tool`` contracts, so the per-provider and per-hop outbound
(SSRF/egress) policies, byte/content bounds, and structured error codes are all
inherited. Individual fetch failures are tolerated and recorded per source.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit

from loguru import logger

from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_tool_eval_metadata,
)

from ..base import ModuleConfig, create_tool_definition
from .web_tool_base import WebToolBase, WebToolError

_TOOL_RESEARCH = "web.research"
_TOOL_PROMPT_VERSION = "2026.06.14"

_ALLOWED_ARGS = {
    "query",
    "engine",
    "max_results",
    "fetch_top_n",
    "format",
    "max_bytes",
    "site_whitelist",
    "site_blacklist",
}
_FORMATS = {"markdown", "text", "html"}

_DEFAULT_MAX_RESULTS = 5
_MAX_MAX_RESULTS = 25
_DEFAULT_FETCH_TOP_N = 3
_MAX_FETCH_TOP_N = 10
_FETCH_CONCURRENCY = 3


class WebToolModule(Protocol):
    """Minimal surface needed from the composed search/fetch modules."""

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], context: Any | None = ...
    ) -> Any: ...


# (url, context) -> "allow" | "ask" | "deny". When wired by the gateway, this lets
# web.research honor the profile's WebFetch(<domain>) permission rules for each
# sub-fetched URL — which the gateway cannot enforce itself because the top-level
# web.research call carries no per-URL `url` subject. Absent a check, the
# always-on outbound (SSRF/egress) policy inside web.fetch still applies.
PermissionCheck = Callable[[str, Any], str]


def _safe_host(url: str) -> str:
    """Return just the host for log context, never the path/query (may hold secrets)."""
    try:
        return urlsplit(url).hostname or "unknown"
    except ValueError:
        return "unknown"


class WebResearchModule(WebToolBase):
    """Single read-only ``web.research`` tool composing search + fetch."""

    _ACTION_FAMILY = "web_research"
    _RESULT_KIND = "bounded_web_research_bundle"
    _TOOL_PROMPT_VERSION = _TOOL_PROMPT_VERSION

    def __init__(
        self,
        config: ModuleConfig,
        *,
        search_module: WebToolModule | None = None,
        fetch_module: WebToolModule | None = None,
        permission_check: PermissionCheck | None = None,
    ) -> None:
        super().__init__(config)
        self._search = search_module or self._default_search_module()
        self._fetch = fetch_module or self._default_fetch_module()
        self._permission_check = permission_check

    @staticmethod
    def _default_search_module() -> WebToolModule:
        from .web_search_module import WebSearchModule

        return WebSearchModule(ModuleConfig(name="WebSearch"))

    @staticmethod
    def _default_fetch_module() -> WebToolModule:
        from .web_fetch_module import WebFetchModule

        return WebFetchModule(ModuleConfig(name="WebFetch"))

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {
            "initialized": True,
            "search_module": self._search is not None,
            "fetch_module": self._fetch is not None,
        }

    async def get_tools(self) -> list[dict[str, Any]]:
        tool = create_tool_definition(
            name=_TOOL_RESEARCH,
            description=(
                "Search the web and fetch + extract the top results into one "
                "bounded research bundle. Composes web.search and web.fetch; "
                "subject to outbound (SSRF/egress) policy and external-network "
                "permission."
            ),
            parameters={
                "properties": {
                    "query": {"type": "string", "description": "The research query."},
                    "engine": {"type": "string", "description": "Search provider; defaults to configured."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": _MAX_MAX_RESULTS},
                    "fetch_top_n": {"type": "integer", "minimum": 0, "maximum": _MAX_FETCH_TOP_N},
                    "format": {"type": "string", "enum": sorted(_FORMATS)},
                    "max_bytes": {"type": "integer", "minimum": 1},
                    "site_whitelist": {"type": "array", "items": {"type": "string"}},
                    "site_blacklist": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
            },
            metadata={
                "category": "web",
                "readOnlyHint": True,
                "uses_network": True,
                "capabilities": ["research.web", "external.network"],
                **build_tool_eval_metadata(
                    tool_prompt_id=f"mcp.{_TOOL_RESEARCH}.v1",
                    tool_prompt_version=_TOOL_PROMPT_VERSION,
                    task_families=["web_research", "citation_collection"],
                    expected_result_kind="bounded_web_research_bundle",
                    success_signals=[
                        "enforced_outbound_policy",
                        "bounded_sources",
                        "tolerated_partial_fetch_failures",
                    ],
                ),
            },
        )
        tool["inputSchema"]["additionalProperties"] = False
        return [tool]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if tool_name != _TOOL_RESEARCH:
            return self._structured_error(tool_name, "unknown_tool", "Unknown web tool.", context=context)

        # The MCP protocol layer already runs sanitize_input on arguments before
        # execute_tool, so we do not re-sanitize here.
        try:
            params = self._validate(arguments or {})
        except WebToolError as exc:
            return self._structured_error(tool_name, exc.reason_code, exc.message, context=context)

        search_args: dict[str, Any] = {
            "query": params["query"],
            "result_count": params["max_results"],
        }
        if params["engine"] is not None:
            search_args["engine"] = params["engine"]
        if params["site_whitelist"] is not None:
            search_args["site_whitelist"] = params["site_whitelist"]
        if params["site_blacklist"] is not None:
            search_args["site_blacklist"] = params["site_blacklist"]

        try:
            search_result = await self._search.execute_tool("web.search", search_args, context)
        except Exception as exc:  # noqa: BLE001 - a search crash maps to a structured error.
            logger.bind(stage="web.research").opt(exception=exc).error("web.research search stage failed")
            return self._structured_error(tool_name, "search_failed", "Web search stage failed.", context=context)

        if not isinstance(search_result, dict) or not search_result.get("ok"):
            reason_code, message = self._delegated_error(search_result, "search_failed")
            return self._structured_error(tool_name, reason_code, message, context=context)

        results = search_result.get("results")
        if not isinstance(results, list):
            results = []
        fetch_targets = self._fetch_targets(results, params["fetch_top_n"])
        fetched = await self._fetch_sources(fetch_targets, params, context)

        sources, fetched_count, any_fetch_truncated = self._assemble_sources(results, fetched)
        truncated = bool(search_result.get("truncated")) or any_fetch_truncated

        return {
            "ok": True,
            "query": params["query"],
            "engine": search_result.get("engine"),
            "result_count": len(results),
            "fetched_count": fetched_count,
            "truncated": truncated,
            "sources": sources,
            "eval": self._eval_metadata(
                _TOOL_RESEARCH, reason_code=None, truncated=truncated, context=context
            ),
        }

    # ---- orchestration -------------------------------------------------

    @staticmethod
    def _fetch_targets(results: list[Any], fetch_top_n: int) -> list[str]:
        targets: list[str] = []
        for entry in results:
            if len(targets) >= fetch_top_n:
                break
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            if isinstance(url, str) and url.strip():
                targets.append(url.strip())
        return targets

    async def _fetch_sources(
        self, urls: list[str], params: dict[str, Any], context: Any | None
    ) -> dict[str, dict[str, Any]]:
        if not urls:
            return {}
        # Deduplicate so a repeated URL is fetched (and policy-checked) once.
        unique_urls = list(dict.fromkeys(urls))

        results: dict[str, dict[str, Any]] = {}
        to_fetch: list[str] = []
        for url in unique_urls:
            decision = self._check_permission(url, context)
            if decision == "deny":
                results[url] = {"ok": False, "reason_code": "permission_denied"}
            elif decision == "ask":
                results[url] = {"ok": False, "reason_code": "permission_required"}
            else:
                to_fetch.append(url)

        semaphore = asyncio.Semaphore(_FETCH_CONCURRENCY)

        async def _fetch_one(url: str) -> tuple[str, Any]:
            fetch_args: dict[str, Any] = {"url": url, "format": params["format"]}
            if params["max_bytes"] is not None:
                fetch_args["max_bytes"] = params["max_bytes"]
            async with semaphore:
                try:
                    return url, await self._fetch.execute_tool("web.fetch", fetch_args, context)
                except Exception as exc:  # noqa: BLE001 - a fetch crash must not fail the bundle.
                    logger.bind(stage="web.research", host=_safe_host(url)).opt(exception=exc).warning(
                        "web.research sub-fetch raised"
                    )
                    return url, {"ok": False, "reason_code": "fetch_failed"}

        pairs = await asyncio.gather(*(_fetch_one(url) for url in to_fetch))
        results.update(dict(pairs))
        return results

    def _check_permission(self, url: str, context: Any | None) -> str:
        """Evaluate the optional per-URL permission hook, failing closed on error."""
        if self._permission_check is None:
            return "allow"
        try:
            decision = self._permission_check(url, context)
        except Exception as exc:  # noqa: BLE001 - a check error must not silently allow.
            logger.bind(stage="web.research", host=_safe_host(url)).opt(exception=exc).warning(
                "web.research permission check raised; denying"
            )
            return "deny"
        return decision if decision in {"allow", "ask", "deny"} else "allow"

    def _assemble_sources(
        self, results: list[Any], fetched: dict[str, dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, bool]:
        sources: list[dict[str, Any]] = []
        fetched_count = 0
        any_truncated = False
        rank = 0
        for entry in results:
            if not isinstance(entry, dict):
                continue
            rank += 1
            url = entry.get("url")
            url = url.strip() if isinstance(url, str) else ""
            metadata = entry.get("metadata")
            source: dict[str, Any] = {
                # Citation-oriented fields: rank for ordering, domain for
                # attribution, and the search provider's own metadata
                # (author/date/source/...) carried through for the caller.
                "rank": rank,
                "title": entry.get("title"),
                "url": url,
                "domain": _safe_host(url) if url else None,
                "snippet": entry.get("content"),
                "search_metadata": metadata if isinstance(metadata, dict) else {},
                "fetched": False,
            }
            fetch_result = fetched.get(url) if url else None
            if isinstance(fetch_result, dict):
                if fetch_result.get("ok"):
                    source["fetched"] = True
                    source["final_url"] = fetch_result.get("final_url") or url
                    source["status_code"] = fetch_result.get("status_code")
                    source["content_type"] = fetch_result.get("content_type")
                    source["content"] = fetch_result.get("content")
                    source["retrieved_at"] = datetime.now(UTC).isoformat()
                    if fetch_result.get("truncated"):
                        any_truncated = True
                    fetched_count += 1
                else:
                    source["reason_code"] = fetch_result.get("reason_code", "fetch_failed")
            sources.append(source)
        return sources, fetched_count, any_truncated

    @staticmethod
    def _delegated_error(result: Any, fallback: str) -> tuple[str, str]:
        if isinstance(result, dict):
            return (
                str(result.get("reason_code") or fallback),
                str(result.get("message") or "Delegated web tool failed."),
            )
        return fallback, "Delegated web tool returned an unexpected payload."

    # ---- validation ----------------------------------------------------

    def _validate(self, args: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(args) - _ALLOWED_ARGS)
        if unknown:
            raise WebToolError("invalid_arguments", f"unknown arguments: {', '.join(unknown)}")

        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise WebToolError("invalid_arguments", "query is required")
        query = query.strip()

        engine = args.get("engine")
        if engine is not None and (not isinstance(engine, str) or not engine.strip()):
            raise WebToolError("invalid_arguments", "engine must be a non-empty string")

        max_results = self._bounded_int(
            args, "max_results", default=_DEFAULT_MAX_RESULTS, minimum=1, maximum=_MAX_MAX_RESULTS
        )
        fetch_top_n = self._bounded_int(
            args, "fetch_top_n", default=_DEFAULT_FETCH_TOP_N, minimum=0, maximum=_MAX_FETCH_TOP_N
        )
        # Never fetch more than the search will return.
        fetch_top_n = min(fetch_top_n, max_results)

        fmt = args.get("format", "markdown")
        if fmt not in _FORMATS:
            raise WebToolError("invalid_arguments", "format must be one of markdown, text, html")

        max_bytes = args.get("max_bytes")
        if max_bytes is not None:
            if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
                raise WebToolError("invalid_arguments", "max_bytes must be a positive integer")

        site_whitelist = self._validate_domain_list(args, "site_whitelist")
        site_blacklist = self._validate_domain_list(args, "site_blacklist")

        return {
            "query": query,
            "engine": engine.strip() if isinstance(engine, str) else None,
            "max_results": max_results,
            "fetch_top_n": fetch_top_n,
            "format": fmt,
            "max_bytes": max_bytes,
            "site_whitelist": site_whitelist,
            "site_blacklist": site_blacklist,
        }

    @staticmethod
    def _bounded_int(args: dict[str, Any], name: str, *, default: int, minimum: int, maximum: int) -> int:
        value = args.get(name)
        if value is None:
            return default
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise WebToolError("invalid_arguments", f"{name} must be an integer >= {minimum}")
        if value > maximum:
            raise WebToolError("invalid_arguments", f"{name} exceeds maximum ({maximum})")
        return value
