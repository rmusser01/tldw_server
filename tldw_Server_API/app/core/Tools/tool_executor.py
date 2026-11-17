from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPRequest, RequestContext
except Exception:  # pragma: no cover - optional import in minimal environments
    get_mcp_server = None  # type: ignore
    MCPRequest = None  # type: ignore
    RequestContext = None  # type: ignore


class ToolExecutionError(Exception):
    pass


class ToolApprovalRequiredError(ToolExecutionError):
    """Execution refusal carrying a structured MCP approval demand.

    The MCP protocol converts ``ApprovalRequiredError`` (raised by
    tool_execution/security.py and hooks.py) into an AUTHORIZATION_ERROR
    response whose ``data`` holds the approval request payload
    (``{"approval": {...}}``). Plain ``ToolExecutionError`` drops that
    payload; this subclass preserves it (``.approval``) so unattended
    callers — the scheduled read-only envelope — can classify the refusal
    as an approval demand instead of an ordinary execution failure.
    """

    def __init__(self, message: str, approval: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.approval = approval if isinstance(approval, dict) else {}


class ToolExecutor:
    """Thin wrapper around MCP Unified protocol for server-side tool execution.

    This keeps Chat and API endpoints decoupled from MCP internals and provides
    a stable surface for argument/permission checks.
    """

    def __init__(self) -> None:
        if get_mcp_server is None:
            raise RuntimeError("MCP Unified module not available")
        self.server = get_mcp_server()

    def _make_context(
        self,
        *,
        user_id: str | None,
        client_id: str | None,
        request_id: str | None = None,
        admin_override: bool = False,
        allowed_tools: list[str] | None = None,
    ) -> Any:
        if RequestContext is None:
            raise RuntimeError("MCP RequestContext unavailable")
        meta = {"admin_override": bool(admin_override)}
        if allowed_tools is not None:
            meta["allowed_tools"] = allowed_tools
        return RequestContext(
            request_id=str(request_id or "tools"),
            user_id=str(user_id) if user_id else None,
            client_id=str(client_id or "api_client"),
            metadata=meta,
        )

    async def list_tools(self, *, user_id: str | None, client_id: str | None) -> dict[str, Any]:
        ctx = self._make_context(user_id=user_id, client_id=client_id)
        req = MCPRequest(method="tools/list", params={})
        resp = await self.server.protocol.process_request(req, ctx)
        if getattr(resp, "error", None):
            raise ToolExecutionError(str(resp.error))
        return getattr(resp, "result", {}) or {}

    async def execute(
        self,
        *,
        user_id: str | None,
        client_id: str | None,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        validate_only: bool = False,
        allowed_tools: list[str] | None = None,
        approval_token: str | None = None,
        approval_context: dict[str, Any] | None = None,
        approval_verifier: Callable[..., tuple[bool, str | None]] | None = None,
    ) -> dict[str, Any]:
        if approval_token is not None:
            if approval_verifier is None or not isinstance(approval_context, dict):
                raise ToolExecutionError("Approval verification unavailable")
            ok, error = approval_verifier(token=approval_token, **approval_context)
            if not ok:
                raise ToolExecutionError(f"Approval denied: {error or 'invalid token'}")

        if validate_only:
            # Lightweight permission probe via tools/list; skips actual execution
            listing = await self.list_tools(user_id=user_id, client_id=client_id)
            can = False
            for t in listing.get("tools", []) or []:
                if isinstance(t, dict) and t.get("name") == tool_name:
                    can = bool(t.get("canExecute"))
                    break
            if not can:
                raise ToolExecutionError("Permission denied for tool or tool not found")
            return {"validated": True}

        ctx = self._make_context(user_id=user_id, client_id=client_id, allowed_tools=allowed_tools)
        req = MCPRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments or {},
                "idempotencyKey": idempotency_key,
            },
        )
        resp = await self.server.protocol.process_request(req, ctx)
        if getattr(resp, "error", None):
            err = getattr(resp.error, "message", str(resp.error))  # type: ignore[attr-defined]
            data = getattr(resp.error, "data", None)  # type: ignore[attr-defined]
            # Approval-demand signal: the protocol maps ApprovalRequiredError
            # to AUTHORIZATION_ERROR with data={"approval": {...}} — that
            # mapping (not the bare code, which plain permission denials
            # share) is what distinguishes an interactive-approval demand
            # from an ordinary authorization failure. Preserve it.
            if isinstance(data, dict) and isinstance(data.get("approval"), dict):
                raise ToolApprovalRequiredError(err, approval=data["approval"])
            raise ToolExecutionError(err)
        return getattr(resp, "result", {}) or {}
