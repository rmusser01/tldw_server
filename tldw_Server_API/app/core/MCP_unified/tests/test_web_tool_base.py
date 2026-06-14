from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_tool_base import (
    WebToolBase,
    WebToolError,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.asyncio


class _SampleWebTool(WebToolBase):
    _ACTION_FAMILY = "sample_web"
    _RESULT_KIND = "sample_result"
    _TOOL_PROMPT_VERSION = "2026.01.01"

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        return None


def _tool() -> _SampleWebTool:
    return _SampleWebTool(ModuleConfig(name="Sample"))


async def test_sanitize_input_allows_sql_like_and_punycode_strips_control() -> None:
    tool = _tool()
    cleaned = tool.sanitize_input(
        {"query": "pip install --no-cache-dir", "domains": ["xn--80ak6aa92e.com"], "ctl": "a\x00b\x07c"}
    )
    assert cleaned["query"] == "pip install --no-cache-dir"  # nosec B101
    assert cleaned["domains"] == ["xn--80ak6aa92e.com"]  # nosec B101
    assert cleaned["ctl"] == "abc"  # nosec B101


async def test_sanitize_input_depth_guard() -> None:
    tool = _tool()
    deep: Any = "x"
    for _ in range(25):
        deep = {"k": deep}
    with pytest.raises(ValueError):
        tool.sanitize_input(deep)


async def test_structured_error_shape_and_extra_fields() -> None:
    tool = _tool()
    err = tool._structured_error("sample.tool", "bad_thing", "it broke", status_code=503)
    assert err["ok"] is False  # nosec B101
    assert err["reason_code"] == "bad_thing"  # nosec B101
    assert err["message"] == "it broke"  # nosec B101
    assert err["status_code"] == 503  # nosec B101
    assert err["eval"]["reason_code"] == "bad_thing"  # nosec B101


async def test_structured_error_omits_none_extras() -> None:
    tool = _tool()
    err = tool._structured_error("sample.tool", "bad", "msg", status_code=None)
    assert "status_code" not in err  # nosec B101


async def test_eval_metadata_uses_class_identity_and_profile() -> None:
    tool = _tool()
    context = RequestContext(request_id="r", metadata={"profile_id": "deep-researcher"})
    meta = tool._eval_metadata("sample.tool", reason_code=None, truncated=True, context=context)
    assert meta["profile_id"] == "deep-researcher"  # nosec B101
    assert meta["truncated"] is True  # nosec B101


async def test_validate_domain_list() -> None:
    tool = _tool()
    assert tool._validate_domain_list({}, "site_whitelist") is None  # nosec B101
    assert tool._validate_domain_list({"site_whitelist": []}, "site_whitelist") is None  # nosec B101
    assert tool._validate_domain_list({"site_whitelist": ["  a.com "]}, "site_whitelist") == ["a.com"]  # nosec B101
    with pytest.raises(WebToolError):
        tool._validate_domain_list({"site_whitelist": ["ok", 5]}, "site_whitelist")
