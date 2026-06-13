from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_search_module import (
    WebSearchModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.asyncio


class _FakeBackend:
    """Injectable search backend so unit tests never hit a provider/network."""

    def __init__(self, result: dict[str, Any] | None = None, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        assert self.result is not None  # nosec B101
        return self.result


def _module(backend: _FakeBackend) -> WebSearchModule:
    return WebSearchModule(ModuleConfig(name="WebSearch"), backend=backend)


def _ok_payload(*, engine: str = "duckduckgo", query: str = "q", results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "search_engine": engine,
        "search_query": query,
        "results": results
        if results is not None
        else [
            {
                "title": "Result One",
                "url": "https://example.com/one",
                "content": "First snippet of content.",
                "metadata": {"source": "example.com"},
            },
            {
                "title": "Result Two",
                "url": "https://example.org/two",
                "content": "Second snippet of content.",
                "metadata": {"source": "example.org"},
            },
        ],
        "total_results_found": 2,
        "processing_error": None,
        "error": None,
    }


async def test_get_tools_exposes_web_search_contract() -> None:
    module = _module(_FakeBackend())
    tools = await module.get_tools()
    assert [tool["name"] for tool in tools] == ["web.search"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["query"]  # nosec B101
    assert tool["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101


async def test_happy_path_returns_normalized_results() -> None:
    backend = _FakeBackend(_ok_payload(query="python asyncio"))
    module = _module(backend)
    result = await module.execute_tool(
        "web.search", {"query": "python asyncio", "engine": "duckduckgo"}
    )
    assert result["ok"] is True  # nosec B101
    assert result["engine"] == "duckduckgo"  # nosec B101
    assert result["query"] == "python asyncio"  # nosec B101
    assert result["result_count"] == 2  # nosec B101
    assert result["total_results_found"] == 2  # nosec B101
    assert [item["url"] for item in result["results"]] == [
        "https://example.com/one",
        "https://example.org/two",
    ]  # nosec B101
    assert backend.calls[0]["query"] == "python asyncio"  # nosec B101
    assert backend.calls[0]["engine"] == "duckduckgo"  # nosec B101


async def test_result_content_is_bounded() -> None:
    long = "x" * 10_000
    backend = _FakeBackend(
        _ok_payload(
            results=[
                {"title": "T", "url": "https://example.com", "content": long, "metadata": {}}
            ]
        )
    )
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    assert result["ok"] is True  # nosec B101
    assert len(result["results"][0]["content"]) < len(long)  # nosec B101


async def test_results_capped_to_requested_count() -> None:
    many = [
        {"title": f"T{i}", "url": f"https://example.com/{i}", "content": "c", "metadata": {}}
        for i in range(5)
    ]
    backend = _FakeBackend(_ok_payload(results=many))
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q", "result_count": 2})
    assert result["ok"] is True  # nosec B101
    assert result["result_count"] == 2  # nosec B101
    assert len(result["results"]) == 2  # nosec B101


async def test_empty_query_is_rejected() -> None:
    backend = _FakeBackend()
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "   "})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101
    assert backend.calls == []  # nosec B101


async def test_invalid_engine_is_rejected() -> None:
    backend = _FakeBackend()
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q", "engine": "askjeeves"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_engine"  # nosec B101
    assert backend.calls == []  # nosec B101


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "q", "result_count": 0},
        {"query": "q", "result_count": -1},
        {"query": "q", "result_count": 1000},
        {"query": "q", "result_count": "five"},
        {"query": 123},
        {"query": "q", "site_whitelist": "example.com"},
        {"query": "q", "depth": 2},
    ],
)
async def test_argument_validation(arguments: dict[str, Any]) -> None:
    backend = _FakeBackend()
    module = _module(backend)
    result = await module.execute_tool("web.search", arguments)
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101
    assert backend.calls == []  # nosec B101


async def test_processing_error_maps_to_search_failed() -> None:
    backend = _FakeBackend({"results": [], "processing_error": "Error performing web search"})
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "search_failed"  # nosec B101


async def test_outbound_policy_error_maps_to_denied() -> None:
    backend = _FakeBackend(
        {"results": [], "processing_error": "Error performing web search: Blocked by outbound policy: ssrf"}
    )
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "outbound_policy_denied"  # nosec B101


async def test_backend_exception_maps_to_search_failed() -> None:
    backend = _FakeBackend(exc=RuntimeError("boom"))
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "search_failed"  # nosec B101


async def test_unknown_tool_returns_structured_error() -> None:
    backend = _FakeBackend()
    module = _module(backend)
    result = await module.execute_tool("web.other", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "unknown_tool"  # nosec B101


async def test_site_filters_passed_to_backend() -> None:
    backend = _FakeBackend(_ok_payload())
    module = _module(backend)
    result = await module.execute_tool(
        "web.search",
        {
            "query": "q",
            "site_whitelist": ["example.com"],
            "site_blacklist": ["spam.example"],
        },
    )
    assert result["ok"] is True  # nosec B101
    assert backend.calls[0]["site_whitelist"] == ["example.com"]  # nosec B101
    assert backend.calls[0]["site_blacklist"] == ["spam.example"]  # nosec B101


async def test_eval_metadata_records_profile_from_context() -> None:
    backend = _FakeBackend(_ok_payload())
    module = _module(backend)
    context = RequestContext(request_id="req-search", metadata={"profile_id": "deep-researcher"})
    result = await module.execute_tool("web.search", {"query": "q"}, context)
    assert result["ok"] is True  # nosec B101
    assert result["eval"]["profile_id"] == "deep-researcher"  # nosec B101


async def test_sanitize_input_allows_sql_like_query_and_punycode() -> None:
    # The protocol layer runs module.sanitize_input before execution; it must not
    # reject queries with '--' (CLI flags) or punycode 'xn--' domain filters.
    module = _module(_FakeBackend())
    cleaned = module.sanitize_input(
        {
            "query": "pip install --no-cache-dir",
            "site_whitelist": ["xn--80ak6aa92e.com"],
        }
    )
    assert cleaned["query"] == "pip install --no-cache-dir"  # nosec B101
    assert cleaned["site_whitelist"] == ["xn--80ak6aa92e.com"]  # nosec B101


async def test_sanitize_input_strips_control_characters() -> None:
    module = _module(_FakeBackend())
    cleaned = module.sanitize_input({"query": "a\x00b\x07c"})
    assert cleaned["query"] == "abc"  # nosec B101


async def test_sql_like_query_executes_successfully() -> None:
    backend = _FakeBackend(_ok_payload(query="pip install --no-cache-dir"))
    module = _module(backend)
    result = await module.execute_tool(
        "web.search", {"query": "pip install --no-cache-dir"}
    )
    assert result["ok"] is True  # nosec B101
    assert backend.calls[0]["query"] == "pip install --no-cache-dir"  # nosec B101


async def test_truncated_reported_when_results_exceed_count() -> None:
    many = [
        {"title": f"T{i}", "url": f"https://example.com/{i}", "content": "c", "metadata": {}}
        for i in range(5)
    ]
    backend = _FakeBackend(_ok_payload(results=many))
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q", "result_count": 2})
    assert result["truncated"] is True  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


async def test_truncated_reported_when_content_clipped() -> None:
    backend = _FakeBackend(
        _ok_payload(
            results=[
                {"title": "T", "url": "https://example.com", "content": "x" * 9000, "metadata": {}}
            ]
        )
    )
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    assert result["truncated"] is True  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


async def test_not_truncated_when_within_bounds() -> None:
    backend = _FakeBackend(_ok_payload())
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    assert result["truncated"] is False  # nosec B101
    assert result["eval"]["truncated"] is False  # nosec B101


async def test_result_fields_coerced_to_strings() -> None:
    backend = _FakeBackend(
        _ok_payload(
            results=[
                {"title": None, "url": 12345, "content": None, "metadata": "not-a-dict"}
            ]
        )
    )
    module = _module(backend)
    result = await module.execute_tool("web.search", {"query": "q"})
    entry = result["results"][0]
    assert entry["title"] == ""  # nosec B101
    assert entry["url"] == "12345"  # nosec B101
    assert entry["content"] == ""  # nosec B101
    assert entry["metadata"] == {}  # nosec B101


async def test_empty_site_list_normalized_to_none() -> None:
    backend = _FakeBackend(_ok_payload())
    module = _module(backend)
    result = await module.execute_tool(
        "web.search", {"query": "q", "site_whitelist": []}
    )
    assert result["ok"] is True  # nosec B101
    assert backend.calls[0]["site_whitelist"] is None  # nosec B101


async def test_blank_optional_strings_normalized_to_none() -> None:
    backend = _FakeBackend(_ok_payload())
    module = _module(backend)
    result = await module.execute_tool(
        "web.search", {"query": "q", "content_country": "   ", "safesearch": ""}
    )
    assert result["ok"] is True  # nosec B101
    assert backend.calls[0]["content_country"] is None  # nosec B101
    assert backend.calls[0]["safesearch"] is None  # nosec B101
