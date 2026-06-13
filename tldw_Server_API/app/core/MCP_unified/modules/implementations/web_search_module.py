"""Read-only ``web.search`` MCP tool wrapping the multi-provider web search API.

The tool runs a query against a configured search provider and returns a bounded
list of normalized results. The provider call itself enforces the centralized
outbound (SSRF/egress) policy inside ``WebSearch_APIs.perform_websearch``; this
module adds argument validation, result bounding, and structured error mapping.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_execution_eval_metadata,
    build_tool_eval_metadata,
)

from ..base import BaseModule, ModuleConfig, create_tool_definition

_TOOL_SEARCH = "web.search"
_TOOL_PROMPT_VERSION = "2026.06.12"

_ALLOWED_ARGS = {
    "query",
    "engine",
    "result_count",
    "content_country",
    "search_lang",
    "output_lang",
    "safesearch",
    "site_whitelist",
    "site_blacklist",
    "date_range",
}

# Curated provider allow-list. Mirrors the engines dispatched by
# ``WebSearch_APIs.perform_websearch`` that return normalized results.
_ALLOWED_ENGINES = frozenset(
    {
        "google",
        "duckduckgo",
        "brave",
        "kagi",
        "serper",
        "tavily",
        "exa",
        "firecrawl",
        "searx",
        "yandex",
        "baidu",
    }
)
_DEFAULT_ENGINE = "duckduckgo"

_DEFAULT_RESULT_COUNT = 10
_MAX_RESULT_COUNT = 25
_MAX_RESULT_CONTENT_CHARS = 4000

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_SANITIZE_DEPTH = 20


class WebSearchBackend(Protocol):
    """Injectable search backend so the module is unit-testable without network."""

    def search(self, **kwargs: Any) -> dict[str, Any]: ...


class PerformWebSearchBackend:
    """Default backend delegating to ``WebSearch_APIs.perform_websearch``."""

    def search(self, **kwargs: Any) -> dict[str, Any]:
        from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
            perform_websearch,
        )

        return perform_websearch(
            kwargs["engine"],
            kwargs["query"],
            kwargs.get("content_country") or "US",
            kwargs.get("search_lang") or "en",
            kwargs.get("output_lang") or "en",
            kwargs["result_count"],
            date_range=kwargs.get("date_range"),
            safesearch=kwargs.get("safesearch"),
            site_whitelist=kwargs.get("site_whitelist"),
            site_blacklist=kwargs.get("site_blacklist"),
        )


class _WebSearchError(Exception):
    """Internal control-flow error carrying a structured reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class WebSearchModule(BaseModule):
    """Single read-only ``web.search`` tool over the multi-provider search API."""

    def __init__(
        self,
        config: ModuleConfig,
        *,
        backend: WebSearchBackend | None = None,
    ) -> None:
        super().__init__(config)
        self._backend: WebSearchBackend = backend or PerformWebSearchBackend()

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True, "backend": self._backend is not None}

    async def get_tools(self) -> list[dict[str, Any]]:
        tool = create_tool_definition(
            name=_TOOL_SEARCH,
            description=(
                "Search the web with a configured provider and return a bounded "
                "list of normalized results. Subject to outbound (SSRF/egress) "
                "policy and external-network permission."
            ),
            parameters={
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "engine": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_ENGINES),
                        "description": "Search provider; defaults to the configured provider.",
                    },
                    "result_count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_RESULT_COUNT,
                    },
                    "content_country": {"type": "string"},
                    "search_lang": {"type": "string"},
                    "output_lang": {"type": "string"},
                    "safesearch": {"type": "string"},
                    "site_whitelist": {"type": "array", "items": {"type": "string"}},
                    "site_blacklist": {"type": "array", "items": {"type": "string"}},
                    "date_range": {"type": "string"},
                },
                "required": ["query"],
            },
            metadata={
                "category": "web",
                "readOnlyHint": True,
                "uses_network": True,
                "capabilities": ["research.web", "external.network"],
                **build_tool_eval_metadata(
                    tool_prompt_id=f"mcp.{_TOOL_SEARCH}.v1",
                    tool_prompt_version=_TOOL_PROMPT_VERSION,
                    task_families=["web_research", "citation_collection"],
                    expected_result_kind="bounded_web_search_results",
                    success_signals=[
                        "enforced_outbound_policy",
                        "bounded_results",
                        "normalized_result_shape",
                    ],
                ),
            },
        )
        tool["inputSchema"]["additionalProperties"] = False
        return [tool]

    def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
        """Permissive sanitizer for web search inputs.

        The base implementation rejects strings containing SQL-injection
        substrings such as ``--``, ``/*`` and ``*/``. Those appear in legitimate
        search queries (e.g. ``pip install --no-cache-dir``) and punycode domain
        filters (e.g. ``xn--...``). The MCP protocol layer runs
        ``module.sanitize_input`` on every tool call before execution, so without
        this override such inputs would be rejected with a protocol
        ``InvalidParams`` error instead of being searched. We therefore strip
        only NUL/control characters and keep a depth guard.
        """
        if _depth > _MAX_SANITIZE_DEPTH:
            raise ValueError("Input too deeply nested")
        if isinstance(input_data, str):
            return _CONTROL_CHARS_RE.sub("", input_data)
        if isinstance(input_data, dict):
            return {key: self.sanitize_input(value, _depth + 1) for key, value in input_data.items()}
        if isinstance(input_data, list):
            return [self.sanitize_input(value, _depth + 1) for value in input_data]
        return input_data

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if tool_name != _TOOL_SEARCH:
            return self._error(tool_name, "unknown_tool", "Unknown web tool.", context=context)

        args = self.sanitize_input(arguments or {})
        try:
            params = self._validate(args)
        except _WebSearchError as exc:
            return self._error(tool_name, exc.reason_code, exc.message, context=context)

        try:
            payload = await asyncio.to_thread(self._backend.search, **params)
        except Exception as exc:  # noqa: BLE001 - provider/network errors are mapped.
            logger.bind(stage="web.search", engine=params["engine"]).opt(exception=exc).warning(
                "web.search backend error"
            )
            return self._error(tool_name, "search_failed", "Web search failed.", context=context)

        try:
            results, total, truncated = self._normalize(payload, params["result_count"])
        except _WebSearchError as exc:
            return self._error(tool_name, exc.reason_code, exc.message, context=context)

        return {
            "ok": True,
            "engine": params["engine"],
            "query": params["query"],
            "result_count": len(results),
            "total_results_found": total,
            "truncated": truncated,
            "results": results,
            "eval": self._eval_metadata(
                _TOOL_SEARCH, reason_code=None, truncated=truncated, context=context
            ),
        }

    # ---- validation ----------------------------------------------------

    def _validate(self, args: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(args) - _ALLOWED_ARGS)
        if unknown:
            raise _WebSearchError("invalid_arguments", f"unknown arguments: {', '.join(unknown)}")

        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise _WebSearchError("invalid_arguments", "query is required")
        query = query.strip()

        engine = args.get("engine")
        if engine is None:
            engine = _DEFAULT_ENGINE
        elif not isinstance(engine, str) or engine.lower() not in _ALLOWED_ENGINES:
            raise _WebSearchError("invalid_engine", "engine is not a supported provider")
        else:
            engine = engine.lower()

        result_count = args.get("result_count")
        if result_count is None:
            result_count = _DEFAULT_RESULT_COUNT
        elif not isinstance(result_count, int) or isinstance(result_count, bool) or result_count <= 0:
            raise _WebSearchError("invalid_arguments", "result_count must be a positive integer")
        elif result_count > _MAX_RESULT_COUNT:
            raise _WebSearchError("invalid_arguments", f"result_count exceeds maximum ({_MAX_RESULT_COUNT})")

        # Strip optional string fields; normalize empty/whitespace to None so the
        # backend applies its own defaults instead of receiving blank values.
        str_fields: dict[str, str | None] = {}
        for str_field in ("content_country", "search_lang", "output_lang", "safesearch", "date_range"):
            value = args.get(str_field)
            if value is None:
                str_fields[str_field] = None
                continue
            if not isinstance(value, str):
                raise _WebSearchError("invalid_arguments", f"{str_field} must be a string")
            stripped = value.strip()
            str_fields[str_field] = stripped or None

        site_whitelist = self._validate_domain_list(args, "site_whitelist")
        site_blacklist = self._validate_domain_list(args, "site_blacklist")

        return {
            "engine": engine,
            "query": query,
            "result_count": result_count,
            "content_country": str_fields["content_country"],
            "search_lang": str_fields["search_lang"],
            "output_lang": str_fields["output_lang"],
            "safesearch": str_fields["safesearch"],
            "site_whitelist": site_whitelist,
            "site_blacklist": site_blacklist,
            "date_range": str_fields["date_range"],
        }

    @staticmethod
    def _validate_domain_list(args: dict[str, Any], name: str) -> list[str] | None:
        value = args.get(name)
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise _WebSearchError("invalid_arguments", f"{name} must be a list of non-empty strings")
        # An empty (or all-whitespace) list would be read by some providers as
        # "exclude everything"; normalize it to None instead.
        stripped = [item.strip() for item in value]
        return stripped or None

    # ---- normalization -------------------------------------------------

    def _normalize(
        self, payload: Any, result_count: int
    ) -> tuple[list[dict[str, Any]], int, bool]:
        if not isinstance(payload, dict):
            raise _WebSearchError("search_failed", "Search backend returned an unexpected payload.")

        error_text = payload.get("processing_error") or payload.get("error")
        if error_text:
            if "outbound policy" in str(error_text).lower():
                raise _WebSearchError("outbound_policy_denied", "Outbound policy denied the search provider.")
            raise _WebSearchError("search_failed", "Web search failed.")

        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list):
            raise _WebSearchError("search_failed", "Search backend returned malformed results.")

        # The response is bounded when the provider returned more results than
        # requested or any single result's content had to be clipped.
        truncated = len(raw_results) > result_count
        results: list[dict[str, Any]] = []
        for entry in raw_results[:result_count]:
            if not isinstance(entry, dict):
                continue
            normalized, content_clipped = self._normalize_entry(entry)
            truncated = truncated or content_clipped
            results.append(normalized)

        total = payload.get("total_results_found")
        if not isinstance(total, int):
            total = len(results)
        return results, total, truncated

    def _normalize_entry(self, entry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        title = self._coerce_str(entry.get("title"))
        url = self._coerce_str(entry.get("url"))
        content = self._coerce_str(entry.get("content"))
        content_clipped = len(content) > _MAX_RESULT_CONTENT_CHARS
        if content_clipped:
            content = content[:_MAX_RESULT_CONTENT_CHARS]
        metadata = entry.get("metadata")
        return (
            {
                "title": title,
                "url": url,
                "content": content,
                "metadata": metadata if isinstance(metadata, dict) else {},
            },
            content_clipped,
        )

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if isinstance(value, str):
            return value
        return "" if value is None else str(value)

    # ---- result helpers ------------------------------------------------

    def _error(
        self,
        tool_name: str,
        reason_code: str,
        message: str,
        *,
        context: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "reason_code": reason_code,
            "message": message,
            "eval": self._eval_metadata(tool_name, reason_code=reason_code, context=context),
        }

    def _eval_metadata(
        self,
        tool_name: str,
        *,
        reason_code: str | None,
        truncated: bool = False,
        context: Any | None,
    ) -> dict[str, Any]:
        return build_execution_eval_metadata(
            tool_name=tool_name,
            tool_prompt_id=f"mcp.{tool_name}.v1",
            tool_prompt_version=_TOOL_PROMPT_VERSION,
            action_family="web_search",
            result_kind="bounded_web_search_results",
            profile_id=self._profile_id_from_context_metadata(context),
            path_filter_used=False,
            truncated=truncated,
            reason_code=reason_code,
        )

    @staticmethod
    def _profile_id_from_context_metadata(context: Any | None) -> str | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        for key in ("profile_id", "selected_profile_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
