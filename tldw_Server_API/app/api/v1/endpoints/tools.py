from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, RequirePermission, User

from tldw_Server_API.app.api.v1.schemas.tools import (
    ExecuteToolRequest,
    ExecuteToolResult,
    ToolInfo,
    ToolListResponse,
)
from tldw_Server_API.app.core.Tools.tool_executor import ToolExecutionError, ToolExecutor

router = APIRouter()
_TOOL_ENDPOINT_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@router.get("/tools", response_model=ToolListResponse, summary="List available tools for current user")
async def list_tools_endpoint(current_user: User = Depends(get_request_user)) -> ToolListResponse:
    try:
        executor = ToolExecutor()

        out = await executor.list_tools(
            user_id=str(current_user.id),
            client_id=str(current_user.username or current_user.id),
        )

        def _to_tool_info(t: dict[str, Any]) -> ToolInfo | None:
            try:
                return ToolInfo(**t)
            except _TOOL_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                logger.warning("Failed to parse tool info, falling back to best-effort mapping")
                # Best-effort mapping from provided dict without re-fetching
                try:
                    return ToolInfo(
                        name=str(t.get("name", "")),
                        description=t.get("description"),
                        module=t.get("module"),
                        canExecute=bool(t.get("canExecute")),
                    )
                except _TOOL_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                    logger.error("Failed to best-effort map tool info")
                    return None

        raw_tools = (out.get("tools") or []) if isinstance(out, dict) else []
        tools_maybe = [_to_tool_info(t) for t in raw_tools]
        tools: list[ToolInfo] = [ti for ti in tools_maybe if ti is not None]
        return ToolListResponse(tools=tools)
    except _TOOL_ENDPOINT_NONCRITICAL_EXCEPTIONS as e:
        logger.error("tools.list failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list tools") from e


@router.post(
    "/tools/execute",
    response_model=ExecuteToolResult,
    summary="Execute a tool via the server",
    dependencies=[
        Depends(RequirePermission("tools.execute:*")),
    ],
)
async def execute_tool_endpoint(
    req: ExecuteToolRequest,
    current_user: User = Depends(get_request_user),
) -> ExecuteToolResult:
    try:
        executor = ToolExecutor()
        if req.dry_run:
            await executor.execute(
                user_id=str(current_user.id),
                client_id=str(current_user.username or current_user.id),
                tool_name=req.tool_name,
                arguments=req.arguments,
                idempotency_key=req.idempotency_key,
                validate_only=True,
            )
            return ExecuteToolResult(ok=True, result={"validated": True}, module=None)

        result = await executor.execute(
            user_id=str(current_user.id),
            client_id=str(current_user.username or current_user.id),
            tool_name=req.tool_name,
            arguments=req.arguments,
            idempotency_key=req.idempotency_key,
        )
        return ExecuteToolResult(ok=True, result=result.get("result", result), module=result.get("module"))
    except ToolExecutionError as te:
        logger.warning("tools_execute_denied_or_invalid")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(te)) from te
    except _TOOL_ENDPOINT_NONCRITICAL_EXCEPTIONS as e:
        logger.error("tools.execute failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Tool execution failed") from e
