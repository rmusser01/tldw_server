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


async def test_sanitize_input_allows_sql_like_substrings() -> None:
    cleaned = _tool().sanitize_input({"query": "pip install --no-cache-dir"})
    assert cleaned["query"] == "pip install --no-cache-dir"  # nosec B101


async def test_sanitize_input_preserves_punycode_in_lists() -> None:
    cleaned = _tool().sanitize_input({"domains": ["xn--80ak6aa92e.com"]})
    assert cleaned["domains"] == ["xn--80ak6aa92e.com"]  # nosec B101


async def test_sanitize_input_strips_control_characters() -> None:
    cleaned = _tool().sanitize_input({"ctl": "a\x00b\x07c"})
    assert cleaned["ctl"] == "abc"  # nosec B101


async def test_sanitize_input_depth_guard() -> None:
    deep: Any = "x"
    for _ in range(25):
        deep = {"k": deep}
    with pytest.raises(ValueError):
        _tool().sanitize_input(deep)


async def test_structured_error_is_not_ok() -> None:
    err = _tool()._structured_error("sample.tool", "bad_thing", "it broke")
    assert err["ok"] is False  # nosec B101


async def test_structured_error_carries_reason_and_message() -> None:
    err = _tool()._structured_error("sample.tool", "bad_thing", "it broke")
    assert (err["reason_code"], err["message"]) == ("bad_thing", "it broke")  # nosec B101


async def test_structured_error_merges_extra_fields() -> None:
    err = _tool()._structured_error("sample.tool", "bad", "msg", status_code=503)
    assert err["status_code"] == 503  # nosec B101


async def test_structured_error_omits_none_extras() -> None:
    err = _tool()._structured_error("sample.tool", "bad", "msg", status_code=None)
    assert "status_code" not in err  # nosec B101


async def test_structured_error_extra_cannot_overwrite_core_fields() -> None:
    # ``ok`` and ``eval`` are the result keys reachable via **extra; the guard
    # must keep the caller from clobbering them.
    err = _tool()._structured_error("sample.tool", "bad", "msg", ok=True, eval="HIJACKED")
    assert err["ok"] is False  # nosec B101
    assert err["eval"] != "HIJACKED"  # nosec B101


async def test_eval_metadata_reads_profile_from_context() -> None:
    context = RequestContext(request_id="r", metadata={"profile_id": "deep-researcher"})
    meta = _tool()._eval_metadata("sample.tool", reason_code=None, truncated=False, context=context)
    assert meta["profile_id"] == "deep-researcher"  # nosec B101


async def test_eval_metadata_propagates_truncated() -> None:
    meta = _tool()._eval_metadata("sample.tool", reason_code=None, truncated=True, context=None)
    assert meta["truncated"] is True  # nosec B101


async def test_validate_domain_list_absent_is_none() -> None:
    assert _tool()._validate_domain_list({}, "site_whitelist") is None  # nosec B101


async def test_validate_domain_list_empty_is_none() -> None:
    assert _tool()._validate_domain_list({"site_whitelist": []}, "site_whitelist") is None  # nosec B101


async def test_validate_domain_list_strips_entries() -> None:
    assert _tool()._validate_domain_list({"site_whitelist": ["  a.com "]}, "site_whitelist") == ["a.com"]  # nosec B101


async def test_validate_domain_list_rejects_non_strings() -> None:
    with pytest.raises(WebToolError):
        _tool()._validate_domain_list({"site_whitelist": ["ok", 5]}, "site_whitelist")
