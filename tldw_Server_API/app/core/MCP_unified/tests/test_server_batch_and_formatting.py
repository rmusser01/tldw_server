import os
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.MCP_unified import config as config_module
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import reset_module_registry
from tldw_Server_API.app.core.MCP_unified.monitoring import metrics as metrics_module
from tldw_Server_API.app.core.MCP_unified.protocol import MCPRequest
from tldw_Server_API.app.core.MCP_unified.protocol_types import AuthenticatedExecutionScope
from tldw_Server_API.app.core.MCP_unified.server import MCPServer


class _AllowAll:
    async def check_permission(self, *args, **kwargs):
        return True


def _ensure_env() -> None:
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("MCP_JWT_SECRET", "x" * 64)
    os.environ.setdefault("MCP_API_KEY_SALT", "s" * 64)


class DictResultModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "dict.echo",
                "description": "Return a dict payload",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            }
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        if tool_name == "dict.echo":
            return {"ok": True, "x": arguments.get("x")}
        raise ValueError(tool_name)


@pytest.mark.asyncio
async def test_http_transports_propagate_only_explicit_authenticated_scope():
    _ensure_env()
    server = MCPServer()
    contexts: list[Any] = []

    async def _ping(params: dict[str, Any], context: Any) -> dict[str, bool]:
        contexts.append(context)
        return {"ok": True}

    server.protocol.handlers["ping"] = _ping  # type: ignore[assignment]
    trusted_scope = AuthenticatedExecutionScope(active_org_id=7, active_team_id=11)

    await server.handle_http_request(
        MCPRequest(method="ping", params={}, id="scoped-request"),
        user_id="41",
        metadata={"org_id": 999, "team_id": 998},
        server_auth_scope=trusted_scope,
    )
    assert contexts[-1].server_auth_scope is trusted_scope

    await server.handle_http_batch(
        [MCPRequest(method="ping", params={}, id="scoped-batch")],
        user_id="41",
        metadata={"org_id": 999, "team_id": 998},
        server_auth_scope=trusted_scope,
    )
    assert contexts[-1].server_auth_scope is trusted_scope

    await server.handle_http_request(
        MCPRequest(method="ping", params={}, id="metadata-only"),
        user_id="41",
        metadata={"active_org_id": 999, "active_team_id": 998},
    )
    assert contexts[-1].server_auth_scope is None

    await server.shutdown()


@pytest.mark.asyncio
async def test_tools_call_dict_result_is_json_content():
    _ensure_env()
    server = MCPServer()
    server.protocol.rbac_policy = _AllowAll()

    await server.module_registry.register_module("dictmod", DictResultModule, ModuleConfig(name="dictmod"))

    req = MCPRequest(
        method="tools/call",
        params={"name": "dict.echo", "arguments": {"x": 7}},
        id="r1",
    )
    resp = await server.handle_http_request(req, user_id="1")

    assert resp.error is None
    assert isinstance(resp.result, dict)
    content = resp.result.get("content")
    assert isinstance(content, list) and content
    assert content[0].get("type") == "json"

    # The subject is that a dict result travels as type "json" rather than stringified,
    # and that the module's own fields arrive intact.
    payload = content[0].get("json")
    assert isinstance(payload, dict)
    assert payload["ok"] is True
    assert payload["x"] == 7

    # This was an exact-equality check against {"ok": True, "x": 7}, which broke when
    # execution eval metadata was attached to tool results. Exact matching cannot work
    # here at all now -- eval carries duration_ms, which differs every run. Assert the
    # stable identity fields and ignore the timing. See TASK-13358.
    eval_meta = payload["eval"]
    assert eval_meta["tool_name"] == "dict.echo"
    assert eval_meta["result_kind"] == "dict_result"
    assert eval_meta["action_family"] == "dict"
    assert eval_meta["truncated"] is False
    assert isinstance(eval_meta["duration_ms"], (int, float))

    # And nothing beyond the module's fields plus eval, so a future addition is a
    # deliberate change to this assertion rather than a silent one.
    assert set(payload) == {"ok", "x", "eval"}, sorted(payload)

    await server.shutdown()
    await reset_module_registry()


@pytest.mark.asyncio
async def test_safe_config_clamped_without_session():
    _ensure_env()
    server = MCPServer()

    async def _ping(params: dict[str, Any], context):
        return {"safe_config": context.metadata.get("safe_config")}

    server.protocol.handlers["ping"] = _ping  # type: ignore[assignment]

    req = MCPRequest(method="ping", params={}, id="p1")
    resp = await server.handle_http_request(
        req,
        user_id="1",
        metadata={
            "safe_config": {
                "snippet_length": 9999,
                "max_tokens": 999999,
                "aliasMode": "yes",
                "unknown": "nope",
            }
        },
    )

    assert resp.error is None
    safe_cfg = (resp.result or {}).get("safe_config")
    assert safe_cfg["snippet_length"] == 2000
    assert safe_cfg["max_tokens"] == 200000
    assert "unknown" not in safe_cfg
    assert "aliasMode" not in safe_cfg

    await server.shutdown()


@pytest.mark.asyncio
async def test_batch_session_semantics_enforced_and_seen_uris_saved():
    _ensure_env()
    server = MCPServer()

    async def _ping(params: dict[str, Any], context):
        seen = context.metadata.get("seen_uris")
        if not isinstance(seen, list):
            seen = []
            context.metadata["seen_uris"] = seen
        seen.append("media://1")
        return {"ok": True}

    server.protocol.handlers["ping"] = _ping  # type: ignore[assignment]

    # Bind session to user 1
    req = MCPRequest(method="ping", params={}, id="bind")
    await server.handle_http_request(req, user_id="1", metadata={"session_id": "sess"})

    # Different user should be rejected for same session
    with pytest.raises(HTTPException) as exc:
        await server.handle_http_batch(
            [MCPRequest(method="ping", params={}, id="b1")],
            user_id="2",
            metadata={"session_id": "sess"},
        )
    assert exc.value.status_code == 403

    # Same user should update seen_uris
    resp = await server.handle_http_batch(
        [MCPRequest(method="ping", params={}, id="b2")],
        user_id="1",
        metadata={"session_id": "sess"},
    )
    assert isinstance(resp, list)
    sess = server.sessions.get("sess")
    assert sess is not None
    assert "media://1" in sess.uris_index

    await server.shutdown()


@pytest.mark.asyncio
async def test_metrics_collection_respects_config_toggle(monkeypatch):
    _ensure_env()
    monkeypatch.setenv("MCP_METRICS_ENABLED", "false")
    config_module.get_config.cache_clear()  # type: ignore[attr-defined]

    # Reset metrics collector singleton for clean state
    metrics_module._metrics_collector = None

    server = MCPServer()
    await server.initialize()

    collector = metrics_module.get_metrics_collector()
    assert collector._collection_task is None

    await server.shutdown()
    # Clear cached config so subsequent tests aren't pinned to metrics disabled.
    try:
        config_module.get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None
