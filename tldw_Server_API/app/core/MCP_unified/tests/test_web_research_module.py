from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_research_module import (
    WebResearchModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.asyncio


class _FakeSearchModule:
    """Stands in for WebSearchModule.execute_tool."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def execute_tool(self, name: str, args: dict[str, Any], context: Any | None = None) -> Any:
        self.calls.append({"name": name, "args": dict(args)})
        return self.result


class _FakeFetchModule:
    """Stands in for WebFetchModule.execute_tool; returns one canned result per url."""

    def __init__(self, by_url: dict[str, dict[str, Any]] | None = None, default: dict[str, Any] | None = None) -> None:
        self.by_url = by_url or {}
        self.default = default
        self.calls: list[str] = []

    async def execute_tool(self, name: str, args: dict[str, Any], context: Any | None = None) -> Any:
        url = args.get("url")
        self.calls.append(url)
        if url in self.by_url:
            return self.by_url[url]
        if self.default is not None:
            return dict(self.default, url=url, final_url=url)
        raise AssertionError(f"unexpected fetch url: {url}")


def _module(
    search: _FakeSearchModule,
    fetch: _FakeFetchModule,
    *,
    permission_check: Any | None = None,
) -> WebResearchModule:
    return WebResearchModule(
        ModuleConfig(name="WebResearch"),
        search_module=search,
        fetch_module=fetch,
        permission_check=permission_check,
    )


def _search_ok(
    *,
    engine: str = "duckduckgo",
    query: str = "q",
    results: list[dict[str, Any]] | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    if results is None:
        results = [
            {"title": "One", "url": "https://example.com/1", "content": "snippet one", "metadata": {}},
            {"title": "Two", "url": "https://example.org/2", "content": "snippet two", "metadata": {}},
            {"title": "Three", "url": "https://example.net/3", "content": "snippet three", "metadata": {}},
        ]
    return {
        "ok": True,
        "engine": engine,
        "query": query,
        "result_count": len(results),
        "total_results_found": len(results),
        "truncated": truncated,
        "results": results,
        "eval": {"truncated": truncated},
    }


def _fetch_ok(*, content: str = "page content", status_code: int = 200, truncated: bool = False) -> dict[str, Any]:
    return {
        "ok": True,
        "url": "https://example.com/x",
        "final_url": "https://example.com/x",
        "status_code": status_code,
        "content_type": "text/html",
        "title": "Page",
        "format": "markdown",
        "content": content,
        "bytes_fetched": len(content),
        "truncated": truncated,
        "eval": {},
    }


def _error(reason_code: str) -> dict[str, Any]:
    return {"ok": False, "reason_code": reason_code, "message": "boom", "eval": {}}


async def test_get_tools_exposes_web_research_contract() -> None:
    module = _module(_FakeSearchModule(_search_ok()), _FakeFetchModule(default=_fetch_ok()))
    tools = await module.get_tools()
    assert [tool["name"] for tool in tools] == ["web.research"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["query"]  # nosec B101
    assert tool["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101


async def test_happy_path_searches_then_fetches_top_n() -> None:
    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool(
        "web.research", {"query": "python", "max_results": 3, "fetch_top_n": 2}
    )
    assert result["ok"] is True  # nosec B101
    assert result["query"] == "python"  # nosec B101
    assert result["engine"] == "duckduckgo"  # nosec B101
    assert result["result_count"] == 3  # nosec B101
    assert result["fetched_count"] == 2  # nosec B101
    # search received result_count=max_results
    assert search.calls[0]["args"]["result_count"] == 3  # nosec B101
    # only the top 2 urls were fetched, in order
    assert fetch.calls == ["https://example.com/1", "https://example.org/2"]  # nosec B101
    # sources preserve search order; fetched ones carry content
    assert [s["url"] for s in result["sources"]] == [
        "https://example.com/1",
        "https://example.org/2",
        "https://example.net/3",
    ]  # nosec B101
    assert result["sources"][0]["fetched"] is True  # nosec B101
    assert result["sources"][0]["snippet"] == "snippet one"  # nosec B101
    assert result["sources"][0]["content"] == "page content"  # nosec B101
    # the third result was not fetched (beyond fetch_top_n)
    assert result["sources"][2]["fetched"] is False  # nosec B101


async def test_search_error_short_circuits() -> None:
    search = _FakeSearchModule(_error("search_failed"))
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "search_failed"  # nosec B101
    assert fetch.calls == []  # nosec B101


async def test_individual_fetch_failure_is_tolerated() -> None:
    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(
        by_url={
            "https://example.com/1": _error("outbound_policy_denied"),
            "https://example.org/2": _fetch_ok(content="ok two"),
        }
    )
    module = _module(search, fetch)
    result = await module.execute_tool(
        "web.research", {"query": "q", "fetch_top_n": 2}
    )
    assert result["ok"] is True  # nosec B101
    assert result["fetched_count"] == 1  # nosec B101
    src0 = next(s for s in result["sources"] if s["url"] == "https://example.com/1")
    assert src0["fetched"] is False  # nosec B101
    assert src0["reason_code"] == "outbound_policy_denied"  # nosec B101
    src1 = next(s for s in result["sources"] if s["url"] == "https://example.org/2")
    assert src1["fetched"] is True  # nosec B101
    assert src1["content"] == "ok two"  # nosec B101


async def test_fetch_top_n_clamped_to_results() -> None:
    search = _FakeSearchModule(_search_ok())  # 3 results
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool(
        "web.research", {"query": "q", "max_results": 3, "fetch_top_n": 9}
    )
    assert result["ok"] is True  # nosec B101
    # fetch_top_n (9) clamped to the 3 available results
    assert len(fetch.calls) == 3  # nosec B101
    assert result["fetched_count"] == 3  # nosec B101


async def test_fetch_top_n_zero_returns_search_only() -> None:
    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool(
        "web.research", {"query": "q", "fetch_top_n": 0}
    )
    assert result["ok"] is True  # nosec B101
    assert fetch.calls == []  # nosec B101
    assert result["fetched_count"] == 0  # nosec B101
    assert all(s["fetched"] is False for s in result["sources"])  # nosec B101


async def test_truncated_propagates_from_search() -> None:
    search = _FakeSearchModule(_search_ok(truncated=True))
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 1})
    assert result["truncated"] is True  # nosec B101


async def test_truncated_propagates_from_fetch() -> None:
    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(default=_fetch_ok(truncated=True))
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 1})
    assert result["truncated"] is True  # nosec B101


async def test_results_without_url_are_skipped_for_fetch() -> None:
    results = [
        {"title": "No URL", "url": "", "content": "snippet", "metadata": {}},
        {"title": "Has URL", "url": "https://example.com/ok", "content": "snippet", "metadata": {}},
    ]
    search = _FakeSearchModule(_search_ok(results=results))
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 2})
    assert fetch.calls == ["https://example.com/ok"]  # nosec B101
    assert result["fetched_count"] == 1  # nosec B101


async def test_sanitize_input_allows_sql_like_query() -> None:
    module = _module(_FakeSearchModule(_search_ok()), _FakeFetchModule(default=_fetch_ok()))
    cleaned = module.sanitize_input({"query": "pip install --no-cache-dir"})
    assert cleaned["query"] == "pip install --no-cache-dir"  # nosec B101


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "   "},
        {"query": 123},
        {"query": "q", "max_results": 0},
        {"query": "q", "max_results": 1000},
        {"query": "q", "fetch_top_n": -1},
        {"query": "q", "fetch_top_n": 99},
        {"query": "q", "format": "pdf"},
        {"query": "q", "depth": 2},
    ],
)
async def test_argument_validation(arguments: dict[str, Any]) -> None:
    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", arguments)
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101
    assert search.calls == []  # nosec B101


async def test_invalid_engine_surfaces_from_search() -> None:
    # Engine-value validation is delegated to web.search; its error is surfaced.
    search = _FakeSearchModule(_error("invalid_engine"))
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "engine": "askjeeves"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_engine"  # nosec B101
    assert fetch.calls == []  # nosec B101


async def test_unknown_tool_returns_structured_error() -> None:
    module = _module(_FakeSearchModule(_search_ok()), _FakeFetchModule(default=_fetch_ok()))
    result = await module.execute_tool("web.other", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "unknown_tool"  # nosec B101


async def test_eval_metadata_records_profile_from_context() -> None:
    module = _module(_FakeSearchModule(_search_ok()), _FakeFetchModule(default=_fetch_ok()))
    context = RequestContext(request_id="req-research", metadata={"profile_id": "deep-researcher"})
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 1}, context)
    assert result["ok"] is True  # nosec B101
    assert result["eval"]["profile_id"] == "deep-researcher"  # nosec B101


async def test_search_exception_maps_to_search_failed() -> None:
    class _RaisingSearch:
        calls: list[Any] = []

        async def execute_tool(self, name: str, args: dict[str, Any], context: Any | None = None) -> Any:
            raise RuntimeError("boom")

    fetch = _FakeFetchModule(default=_fetch_ok())
    module = WebResearchModule(
        ModuleConfig(name="WebResearch"), search_module=_RaisingSearch(), fetch_module=fetch
    )
    result = await module.execute_tool("web.research", {"query": "q"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "search_failed"  # nosec B101
    assert fetch.calls == []  # nosec B101


async def test_non_list_results_handled() -> None:
    search = _FakeSearchModule(
        {"ok": True, "engine": "duckduckgo", "query": "q", "results": "not-a-list", "truncated": False, "eval": {}}
    )
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q"})
    assert result["ok"] is True  # nosec B101
    assert result["result_count"] == 0  # nosec B101
    assert fetch.calls == []  # nosec B101


async def test_duplicate_urls_fetched_once() -> None:
    dup = [
        {"title": "A", "url": "https://example.com/x", "content": "s", "metadata": {}},
        {"title": "B", "url": "https://example.com/x", "content": "s", "metadata": {}},
    ]
    search = _FakeSearchModule(_search_ok(results=dup))
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 2})
    assert fetch.calls == ["https://example.com/x"]  # nosec B101 - fetched once
    assert result["ok"] is True  # nosec B101


async def test_permission_check_denies_source_before_fetch() -> None:
    def _check(url: str, context: Any) -> str:
        return "deny" if "example.com" in url else "allow"

    search = _FakeSearchModule(_search_ok())  # example.com/1, example.org/2, example.net/3
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch, permission_check=_check)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 3})
    denied = next(s for s in result["sources"] if s["url"] == "https://example.com/1")
    assert denied["fetched"] is False  # nosec B101
    assert denied["reason_code"] == "permission_denied"  # nosec B101
    # The denied URL was never fetched; the allowed ones were.
    assert "https://example.com/1" not in fetch.calls  # nosec B101
    assert "https://example.org/2" in fetch.calls  # nosec B101


async def test_permission_check_ask_marks_required() -> None:
    def _check(url: str, context: Any) -> str:
        return "ask"

    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch, permission_check=_check)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 2})
    assert fetch.calls == []  # nosec B101
    assert all(s["fetched"] is False for s in result["sources"])  # nosec B101
    assert result["sources"][0]["reason_code"] == "permission_required"  # nosec B101


async def test_permission_check_failure_denies() -> None:
    def _check(url: str, context: Any) -> str:
        raise RuntimeError("policy backend down")

    search = _FakeSearchModule(_search_ok())
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch, permission_check=_check)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 1})
    assert fetch.calls == []  # nosec B101 - fail closed
    assert result["sources"][0]["reason_code"] == "permission_denied"  # nosec B101


async def test_sources_carry_citation_fields() -> None:
    results = [
        {
            "title": "Doc One",
            "url": "https://example.com/one",
            "content": "snippet one",
            "metadata": {"author": "A. Writer", "date_published": "2026-01-02", "source": "example.com"},
        }
    ]
    search = _FakeSearchModule(_search_ok(results=results))
    fetch = _FakeFetchModule(
        default={
            "ok": True,
            "final_url": "https://example.com/one-final",
            "status_code": 200,
            "content_type": "text/html",
            "content": "page",
            "truncated": False,
            "eval": {},
        }
    )
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 1})
    src = result["sources"][0]
    assert src["rank"] == 1  # nosec B101
    assert src["domain"] == "example.com"  # nosec B101
    assert src["search_metadata"]["author"] == "A. Writer"  # nosec B101
    assert src["final_url"] == "https://example.com/one"  # nosec B101 - fake echoes url
    assert src["content_type"] == "text/html"  # nosec B101
    assert isinstance(src["retrieved_at"], str) and src["retrieved_at"]  # nosec B101


async def test_source_rank_is_one_based_and_sequential() -> None:
    search = _FakeSearchModule(_search_ok())  # 3 results
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 3})
    assert [s["rank"] for s in result["sources"]] == [1, 2, 3]  # nosec B101


async def test_unfetched_source_omits_retrieval_fields() -> None:
    search = _FakeSearchModule(_search_ok())  # 3 results
    fetch = _FakeFetchModule(default=_fetch_ok())
    module = _module(search, fetch)
    result = await module.execute_tool("web.research", {"query": "q", "fetch_top_n": 1})
    unfetched = result["sources"][2]
    assert unfetched["fetched"] is False  # nosec B101
    assert "retrieved_at" not in unfetched  # nosec B101
    assert "final_url" not in unfetched  # nosec B101
    # citation fields available even without a fetch
    assert unfetched["rank"] == 3  # nosec B101
    assert unfetched["domain"] == "example.net"  # nosec B101
