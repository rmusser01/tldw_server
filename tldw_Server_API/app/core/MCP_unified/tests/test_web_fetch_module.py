from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import (
    web_fetch_module as wfm,
)
from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_fetch_module import (
    WebFetchModule,
    WebFetchResponse,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.asyncio


class _FakeClient:
    """Injectable fetcher so unit tests never touch the network."""

    def __init__(self, response: WebFetchResponse | None = None, exc: BaseException | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        max_bytes: int,
        user_agent: str,
    ) -> WebFetchResponse:
        self.calls.append(
            {
                "url": url,
                "timeout_seconds": timeout_seconds,
                "max_bytes": max_bytes,
                "user_agent": user_agent,
            }
        )
        if self.exc is not None:
            raise self.exc
        assert self.response is not None  # nosec B101
        return self.response


def _module(client: _FakeClient) -> WebFetchModule:
    return WebFetchModule(ModuleConfig(name="WebFetch"), client=client)


def _allow_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _decide(url: str, **kwargs: Any) -> Any:
        return SimpleNamespace(
            allowed=True,
            reason="allowed",
            mode="compat",
            stage=kwargs.get("stage", ""),
            source=kwargs.get("source", ""),
            details=None,
        )

    monkeypatch.setattr(wfm, "decide_web_outbound_policy", _decide)


def _deny_policy(monkeypatch: pytest.MonkeyPatch, *, reason: str = "ssrf_private_ip") -> list[str]:
    seen: list[str] = []

    async def _decide(url: str, **kwargs: Any) -> Any:
        seen.append(url)
        return SimpleNamespace(
            allowed=False,
            reason=reason,
            mode="strict",
            stage=kwargs.get("stage", ""),
            source=kwargs.get("source", ""),
            details=None,
        )

    monkeypatch.setattr(wfm, "decide_web_outbound_policy", _decide)
    return seen


def _html_response(
    body: bytes = b"",
    *,
    final_url: str = "https://example.com/post",
    status_code: int = 200,
    content_type: str = "text/html; charset=utf-8",
    truncated: bool = False,
) -> WebFetchResponse:
    return WebFetchResponse(
        final_url=final_url,
        status_code=status_code,
        content_type=content_type,
        body=body,
        truncated=truncated,
    )


_RICH_HTML = (
    b"<html><head><title>Example Title</title></head>"
    b"<body><article><h1>Heading</h1>"
    b"<p>Example article body paragraph one with enough text to extract.</p>"
    b"<p>Example article body paragraph two continuing the readable content.</p>"
    b"</article></body></html>"
)


async def test_get_tools_exposes_web_fetch_contract() -> None:
    module = _module(_FakeClient())
    tools = await module.get_tools()
    assert [tool["name"] for tool in tools] == ["web.fetch"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["url"]  # nosec B101
    assert tool["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101


async def test_invalid_scheme_returns_invalid_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient()
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "ftp://example.com/x"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_url"  # nosec B101
    assert client.calls == []  # nosec B101


async def test_outbound_policy_denial_blocks_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _deny_policy(monkeypatch)
    client = _FakeClient()
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://169.254.169.254/latest"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "outbound_policy_denied"  # nosec B101
    assert seen == ["https://169.254.169.254/latest"]  # nosec B101
    assert client.calls == []  # nosec B101


async def test_happy_path_html_extracted_to_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client)
    result = await module.execute_tool(
        "web.fetch",
        {"url": "https://example.com/post", "format": "markdown"},
    )
    assert result["ok"] is True  # nosec B101
    assert result["status_code"] == 200  # nosec B101
    assert result["final_url"] == "https://example.com/post"  # nosec B101
    assert result["format"] == "markdown"  # nosec B101
    assert "Example article body" in result["content"]  # nosec B101
    assert result["title"] == "Example Title"  # nosec B101
    assert result["bytes_fetched"] == len(_RICH_HTML)  # nosec B101
    assert result["truncated"] is False  # nosec B101
    assert client.calls and client.calls[0]["url"] == "https://example.com/post"  # nosec B101


async def test_byte_cap_truncation_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML, truncated=True))
    module = _module(client)
    result = await module.execute_tool(
        "web.fetch",
        {"url": "https://example.com/post", "max_bytes": 16},
    )
    assert result["ok"] is True  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert client.calls[0]["max_bytes"] == 16  # nosec B101


async def test_text_plain_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    body = b"plain readable body content"
    client = _FakeClient(
        _html_response(body, content_type="text/plain; charset=utf-8")
    )
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/raw.txt"})
    assert result["ok"] is True  # nosec B101
    assert result["content"] == "plain readable body content"  # nosec B101


async def test_unsupported_content_type_returns_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(b"\x89PNG\r\n", content_type="image/png"))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/logo.png"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "empty_content"  # nosec B101


async def test_client_exception_maps_to_fetch_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(exc=TimeoutError("boom"))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/slow"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "fetch_failed"  # nosec B101


async def test_http_error_status_maps_to_fetch_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(b"not found", status_code=404, content_type="text/html"))
    module = _module(client)
    result = await module.execute_tool("web.fetch", {"url": "https://example.com/missing"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "fetch_failed"  # nosec B101
    assert result["status_code"] == 404  # nosec B101


async def test_unknown_argument_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    module = _module(_FakeClient())
    result = await module.execute_tool(
        "web.fetch", {"url": "https://example.com", "depth": 3}
    )
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101


@pytest.mark.parametrize(
    "arguments",
    [
        {"url": "https://example.com", "max_bytes": -1},
        {"url": "https://example.com", "timeout_seconds": 0},
        {"url": "https://example.com", "max_bytes": 10**9},
        {"url": "https://example.com", "format": "pdf"},
        {"url": 123},
    ],
)
async def test_bad_bounds_are_rejected(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    _allow_policy(monkeypatch)
    module = _module(_FakeClient())
    result = await module.execute_tool("web.fetch", arguments)
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "invalid_arguments"  # nosec B101


async def test_unknown_tool_returns_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    module = _module(_FakeClient())
    result = await module.execute_tool("web.other", {"url": "https://example.com"})
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "unknown_tool"  # nosec B101


async def test_eval_metadata_records_profile_from_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_policy(monkeypatch)
    client = _FakeClient(_html_response(_RICH_HTML))
    module = _module(client)
    context = RequestContext(request_id="req-web", metadata={"profile_id": "deep-researcher"})
    result = await module.execute_tool(
        "web.fetch", {"url": "https://example.com/post"}, context
    )
    assert result["ok"] is True  # nosec B101
    assert result["eval"]["profile_id"] == "deep-researcher"  # nosec B101
