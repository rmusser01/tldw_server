from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core.MCP_unified.protocol import (
    ApprovalRequiredError,
    MCPProtocol,
    MCPRequest,
    RequestContext,
)
from tldw_Server_API.app.core.MCP_unified.server import MCPServer


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pytest.mark.asyncio
async def test_protocol_error_log_redacts_tokens():
    proto = MCPProtocol()
    # Replace a handler to force an exception which includes secrets
    async def boom(params, context):
        raise Exception("Authorization: Bearer secret.token token=mysecret refresh_token=abc")
    proto.handlers["ping"] = boom

    captured = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="DEBUG")
    try:
        req = MCPRequest(method="ping", id="redact1")
        ctx = RequestContext(request_id="r", user_id="u", client_id="c")
        await proto.process_request(req, ctx)
    except Exception:
        # process_request swallows and returns MCPResponse; shouldn't raise
        _ = None
    finally:
        logger.remove(sink_id)

    # Ensure masked in logs
    text = "\n".join(str(x) for x in captured)
    _ensure("Bearer ****" in text, f"Bearer token was not redacted in logs: {text!r}")
    _ensure("token=****" in text, f"Token parameter was not redacted in logs: {text!r}")
    _ensure("refresh_token=****" in text, f"Refresh token was not redacted in logs: {text!r}")
    _ensure(
        "mysecret" not in text and "secret.token" not in text and "abc" not in text,
        f"Secret values leaked into logs: {text!r}",
    )


@pytest.mark.asyncio
async def test_protocol_generic_handler_exception_returns_masked_internal_error():
    proto = MCPProtocol()

    async def boom(params, context):
        raise Exception("Authorization: Bearer secret.token token=mysecret")

    proto.handlers["ping"] = boom
    response = await proto.process_request(
        MCPRequest(method="ping", id="redact2"),
        RequestContext(request_id="r2", user_id="u", client_id="c"),
    )

    _ensure(response.error is not None, f"Expected error response, got: {response!r}")
    _ensure(response.error.message == "Internal error", f"Unexpected error payload: {response!r}")


@pytest.mark.asyncio
async def test_protocol_secret_bearing_noncritical_exception_returns_internal_error_in_debug_mode(monkeypatch):
    proto = MCPProtocol()

    async def boom(params, context):
        raise RuntimeError("Authorization: Bearer secret.token token=mysecret")

    proto.handlers["ping"] = boom
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.protocol.get_config",
        lambda: SimpleNamespace(debug_mode=True),
    )
    captured = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="DEBUG")

    try:
        response = await proto.process_request(
            MCPRequest(method="ping", id="redact3"),
            RequestContext(request_id="r3", user_id="u", client_id="c"),
        )
    finally:
        logger.remove(sink_id)

    _ensure(response.error is not None, f"Expected error response, got: {response!r}")
    _ensure(response.error.message == "Internal error", f"Unexpected error payload: {response!r}")
    text = "\n".join(captured)
    _ensure("secret.token" not in text and "mysecret" not in text, f"Secret values leaked into runtime error logs: {text!r}")


@pytest.mark.asyncio
async def test_protocol_secret_bearing_prespan_exception_returns_internal_error_in_debug_mode(monkeypatch):
    proto = MCPProtocol()

    async def ping(params, context):
        return {"ok": True}

    async def boom_auth(request, context):
        raise RuntimeError("Authorization: Bearer secret.token token=mysecret")

    proto.handlers["ping"] = ping
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.protocol.get_config",
        lambda: SimpleNamespace(debug_mode=True),
    )
    monkeypatch.setattr(proto, "_check_authorization", boom_auth)
    captured = []
    sink_id = logger.add(lambda m: captured.append(str(m)), level="DEBUG")

    try:
        response = await proto.process_request(
            MCPRequest(method="ping", id="redact4"),
            RequestContext(request_id="r4", user_id="u", client_id="c"),
        )
    finally:
        logger.remove(sink_id)

    _ensure(response.error is not None, f"Expected error response, got: {response!r}")
    _ensure(response.error.message == "Internal error", f"Unexpected error payload: {response!r}")
    text = "\n".join(captured)
    _ensure("secret.token" not in text and "mysecret" not in text, f"Secret values leaked into prespan logs: {text!r}")


@pytest.mark.asyncio
async def test_protocol_span_telemetry_sanitizer_does_not_change_authorization_mapping(monkeypatch):
    proto = MCPProtocol()

    async def needs_approval(params, context):
        del params, context
        raise ApprovalRequiredError("Approval required", approval={"ticket": "abc"})

    async def _allow(_request, _context):
        return True

    proto.handlers["approval.test"] = needs_approval
    monkeypatch.setattr(proto, "_check_authorization", _allow)
    monkeypatch.setattr(proto, "_sanitize_exception_for_telemetry", lambda exc: RuntimeError("telemetry-only"))

    response = await proto.process_request(
        MCPRequest(method="approval.test", id="approval-1"),
        RequestContext(request_id="approval-req", user_id="u", client_id="c"),
    )

    _ensure(response.error is not None, f"Expected error response, got: {response!r}")
    _ensure(response.error.message == "Approval required", f"Authorization mapping changed unexpectedly: {response!r}")
    _ensure(response.error.data == {"approval": {"ticket": "abc"}}, f"Approval payload was lost: {response!r}")


def test_protocol_sanitize_exception_for_telemetry_whitelists_safe_attrs():
    proto = MCPProtocol()

    class _UnsafeError(RuntimeError):
        pass

    exc = _UnsafeError("Authorization: Bearer secret.token token=mysecret")
    exc.code = "E123"
    exc.name = "demo"
    exc.lineno = 77
    exc.unsafe_secret = "should-not-copy"  # nosec B105 - sentinel value for redaction-copy test

    sanitized = proto._sanitize_exception_for_telemetry(exc)

    _ensure("secret.token" not in str(sanitized), f"Secret value leaked into sanitized exception: {sanitized!r}")
    _ensure("mysecret" not in str(sanitized), f"Token value leaked into sanitized exception: {sanitized!r}")
    _ensure(getattr(sanitized, "code", None) == "E123", f"Safe code attr missing from sanitized exception: {sanitized.__dict__!r}")
    _ensure(getattr(sanitized, "name", None) == "demo", f"Safe name attr missing from sanitized exception: {sanitized.__dict__!r}")
    _ensure(getattr(sanitized, "lineno", None) == 77, f"Safe lineno attr missing from sanitized exception: {sanitized.__dict__!r}")
    _ensure(not hasattr(sanitized, "unsafe_secret"), f"Unsafe attrs should not be copied to sanitized exception: {sanitized.__dict__!r}")
    _ensure(getattr(sanitized, "_mcp_masked_secret", False) is True, f"Sanitized exception should be marked as masked: {sanitized.__dict__!r}")


class FakeWebSocket:
    def __init__(self, headers: dict, client_host: str = "127.0.0.1"):
        from types import SimpleNamespace
        self.headers = headers
        self.client = SimpleNamespace(host=client_host)
        self.accepted = False
        self.closed = False
        self.close_args = None
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_args = (code, reason)

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        # Force disconnect immediately after accept if needed
        from fastapi import WebSocketDisconnect
        raise WebSocketDisconnect()


@pytest.mark.asyncio
async def test_server_ws_auth_debug_log_redacts_tokens(monkeypatch):
    server = MCPServer()
    server.config.ws_allowed_origins = ["*"]
    server.config.ws_auth_required = True

    # Monkeypatch MCP JWT verification to raise an error including tokens
    def bad_verify(token: str):
        raise HTTPException(status_code=401, detail=f"MCP JWT auth failed: Bearer {token} token=mytok refresh_token=abc")
    server.jwt_manager.verify_token = bad_verify  # type: ignore

    # Provide a Bearer header to trigger both AuthNZ failure and MCP JWT fallback
    ws = FakeWebSocket(headers={
        "origin": "https://any.com",
        "Authorization": "Bearer supersecrettoken",
    })

    captured = []
    sink_id = logger.add(lambda m: captured.append(m.record.get("message", "")), level="DEBUG")
    try:
        await server.handle_websocket(ws, client_id="c1")
    finally:
        logger.remove(sink_id)

    text = "\n".join(str(x) for x in captured)
    # Expect debug log line with masked tokens
    _ensure("MCP JWT auth failed" in text, f"Missing websocket auth failure log: {text!r}")
    _ensure("Bearer ****" in text, f"Bearer token was not redacted in websocket logs: {text!r}")
    _ensure(
        "token=****" in text and "refresh_token=****" in text,
        f"Token parameters were not redacted in websocket logs: {text!r}",
    )
    _ensure(
        "supersecrettoken" not in text and "mytok" not in text and "abc" not in text,
        f"Secret websocket auth values leaked into logs: {text!r}",
    )
