# Tools API (Server Tool Executor)

This document describes the REST surface for the server-side tool executor that integrates with the existing MCP Unified server. It lets authenticated clients discover available tools and execute them through a simple, controlled API while inheriting MCP’s RBAC, validation, and safety policies.

Status: Experimental (route-gated). Must be explicitly enabled.

## Enabling the API

Routes are controlled by the route policy in `config.txt` (section `[API-Routes]`) or environment variables.

- Marked experimental: the `tools` route is disabled when `stable_only=true`.
- Enable via config:

  ```ini
  [API-Routes]
  enable = tools
  # or
  stable_only = false
  ```
- The `ROUTES_STABLE_ONLY`, `ROUTES_DISABLE`, `ROUTES_ENABLE` and `ROUTES_EXPERIMENTAL` environment variables are route-toggle overrides, honoured only in a test runtime (pytest or `TEST_MODE`). In production, `[API-Routes]` in `config.txt` is authoritative.

After restart, the server includes the tools router under `/api/v1`.

## Authentication & Permissions

- Auth: same as other API sections - either `X-API-KEY` (single-user) or `Authorization: Bearer <JWT>` (multi-user).
- RBAC:
  - Listing tools requires being authenticated (any user); MCP filters visibility and returns a `canExecute` flag per tool.
  - Executing tools requires both:
    - Endpoint permission: `tools.execute:*` (defense-in-depth)
    - MCP protocol permission: `tools.execute:{tool_name}` (or wildcard) for per-tool control
- Write-capable tools may be completely disabled by `MCP_DISABLE_WRITE_TOOLS=1` (see MCP Unified admin docs).

## Endpoints

### GET /api/v1/tools - List tools

Returns the catalog of tools visible to the current user. Each entry includes `canExecute` indicating whether the caller has permission to execute that tool via MCP.

Response body:

```json
{
  "tools": [
    {
      "name": "media.search",
      "description": "Search the media library",
      "module": "media",
      "inputSchema": {"type":"object","properties": {"query":{"type":"string"}}, "required": ["query"]},
      "canExecute": true
    }
  ]
}
```

Example cURL:

```bash
curl -sS -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://127.0.0.1:8000/api/v1/tools | jq
```

Errors: `401` when unauthenticated, `500` if the MCP server is not initialized.

### POST /api/v1/tools/execute - Execute a tool

Body schema:

```json
{
  "tool_name": "media.search",
  "arguments": {"query": "qwen2"},
  "idempotency_key": "optional-key",
  "dry_run": false
}
```

- `dry_run=true` performs a permission/validation probe (no execution) and returns `{ ok: true, result: { validated: true } }` on success.
- For write-capable tools, pass a stable `idempotency_key` to dedupe retries.

Response (success):

```json
{
  "ok": true,
  "result": {"items": []},
  "module": "media"
}
```

Errors:
- `403`: permission denied (either endpoint or MCP per-tool permission)
- `500`: tool not found/malformed request or MCP server not initialized

## Behavior & Policies (inherit from MCP)

- Input validation: modules may declare JSON schemas; protocol enforces `validate_input_schema` when enabled.
- Write tools: can be disabled by policy; require module’s `validate_tool_arguments` override; idempotency support is available.
- Rate limiting and circuit breakers: applied at the MCP module layer; per-category (e.g., `ingestion` vs `read`) when metadata is present.
- Audit & metrics: MCP protocol emits audit logs and metrics for tool calls.

## Tips

- Start with `GET /api/v1/tools` to see `canExecute` flags for your user.
- Use `dry_run=true` to validate arguments and permission before executing a write tool.
- For multi-tenant deployments, grant `tools.execute:{tool}` (or wildcard) to the appropriate roles using the Admin RBAC endpoints.

## Related Docs

- MCP Unified → Developer Guide: `Docs/MCP/Unified/Developer_Guide.md`
- MCP Tool Catalogs: `Docs/MCP/mcp_tool_catalogs.md`
- Adding tools: `Docs/MCP/Unified/Adding_Tools.md`
