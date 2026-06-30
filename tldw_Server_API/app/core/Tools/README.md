# Tools

The Tools module is the server-side bridge for invoking MCP Unified tools from
API handlers and services without making those callers speak MCP protocol
directly. It wraps tool discovery, permission preflight, idempotency keys, and
tool-call error mapping around MCP Unified request contexts.

## Start Here

- Core wrapper: `tool_executor.py`.
- API surfaces: `app/api/v1/endpoints/tools.py` for generic tool routes and
  `app/api/v1/endpoints/mcp_unified_endpoint.py` for MCP JSON-RPC, WebSocket,
  and `/mcp/tools/execute` flows.
- Related core module: `app/core/MCP_unified/`.
- Tests: `tests/Tools/`, `tests/AuthNZ_Unit/test_tools_permissions_claims.py`,
  `tests/MCP/test_mcp_tools_execute_authz.py`, and Chat tool auto-execution
  tests under `tests/Chat/unit/`.

## Responsibilities

- List MCP tools available to a caller and surface `canExecute` metadata.
- Execute a named tool with arguments, caller context, and optional idempotency
  key.
- Provide validation-only preflight so UI and chat flows can fail before a
  mutating tool call.
- Raise `ToolExecutionError` for denied calls or MCP error responses so endpoint
  layers can map them consistently.

## Module Map

- `tool_executor.py` defines `ToolExecutor` and `ToolExecutionError`.

## How It Connects

- `Chat/tool_auto_exec.py` uses this module when chat auto-executes model tool
  calls.
- MCP Unified owns the actual tool registry, module registry, RBAC decisions,
  and execution runtime.
- AuthNZ permissions and MCP Hub policy decide whether the caller can execute a
  tool; this package should not duplicate that policy.

## Extension Points

- Add domain-specific convenience helpers in the consuming module, not here, when
  a workflow needs a narrower wrapper around `execute(...)`.
- Keep new arguments JSON-serializable because MCP requests cross protocol and
  worker boundaries.

## Testing

- Route behavior and sanitized error logs: `tests/Tools/test_tools_routes.py` and
  `tests/Tools/test_tools_endpoint_error_logs.py`.
- Permission checks: `tests/Tools/test_tools_permissions.py`,
  `tests/AuthNZ_Unit/test_tools_permissions_claims.py`, and
  `tests/MCP/test_mcp_tools_execute_authz.py`.
- Chat integration: `tests/Chat/unit/test_tool_auto_exec.py` and
  `tests/Chat/unit/test_chat_service_tool_autoexec.py`.

## Gotchas

- Do not bypass MCP Unified `canExecute` checks. If a caller only needs to know
  whether an action is available, use validation-only execution.
- Preserve caller identity and request metadata in `RequestContext`; audit and
  policy layers rely on it.
