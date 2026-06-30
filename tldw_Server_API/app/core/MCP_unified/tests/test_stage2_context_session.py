"""Stage 2 plumbing tests: RequestContext db_paths and safe config parsing."""

from typing import Dict, Any

from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
from tldw_Server_API.app.core.MCP_unified.server import MCPServer


def test_request_context_keeps_db_paths_empty_without_explicit_mapping() -> None:
    ctx = RequestContext(request_id="rx1", user_id="1", client_id="test")
    assert ctx.db_paths == {}


def test_safe_config_merge_allowlist_and_clamp() -> None:
    srv = MCPServer()
    base: Dict[str, Any] = {"snippet_length": 300, "chars_per_token": 4}
    incoming: Dict[str, Any] = {
        "snippet_length": 9999,  # should clamp
        "chars_per_token": 0,    # should clamp to >=1
        "aliasMode": True,
        "compactShape": False,
        "order_by": "recent",
        "maxSessionUris": 999999,
        "ignored": "x",
    }
    merged = srv._merge_safe_config(base, incoming)
    assert merged["snippet_length"] <= 2000
    assert merged["chars_per_token"] >= 1
    assert merged["aliasMode"] is True
    assert merged["compactShape"] is False
    assert merged["order_by"] == "recent"
    assert merged["maxSessionUris"] <= 5000
    assert "ignored" not in merged
