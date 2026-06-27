# Unified MCP Operator Cheatsheet

Compact reference for repeat MCP work. For the first successful setup path, use `User_Guide.md`.

## Shell Variables

```bash
export MCP_BASE="http://127.0.0.1:8000/api/v1/mcp"
export MCP_WS="ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=operator"
export MCP_TOKEN="<jwt-token>"
export MCP_API_KEY="$SINGLE_USER_API_KEY"
export MCP_AUTH_HEADER="Authorization: Bearer $MCP_TOKEN"
# Single-user mode alternative:
# export MCP_AUTH_HEADER="X-API-KEY: $MCP_API_KEY"
```

## Initialize And Reuse A Session

```bash
curl -i \
  -H "$MCP_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"init-1","method":"initialize","params":{"clientInfo":{"name":"operator-cli","version":"1.0.0"}}}' \
  "$MCP_BASE/request"
```

Copy the returned `mcp-session-id` response header when you want request continuity:

```bash
export MCP_SESSION="<returned-mcp-session-id>"
```

Use it on later calls:

```bash
curl \
  -H "$MCP_AUTH_HEADER" \
  -H "mcp-session-id: $MCP_SESSION" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"list-1","method":"tools/list","params":{}}' \
  "$MCP_BASE/request"
```

## List Tools

Strict catalog discovery is the default for unresolved catalog filters. A misspelled catalog returns an empty list with `_meta.catalog.status=unresolved` instead of silently widening discovery.

```bash
curl \
  -H "$MCP_AUTH_HEADER" \
  "$MCP_BASE/tools?catalog=research"
```

Check visible catalog names:

```bash
curl -H "$MCP_AUTH_HEADER" "$MCP_BASE/tool_catalogs"
```

Use fail-open only for migration diagnosis:

```bash
curl \
  -H "$MCP_AUTH_HEADER" \
  "$MCP_BASE/tools?catalog=research&catalog_fail_open=true"
```

## Call A Tool

HTTP convenience endpoint:

```bash
curl \
  -H "$MCP_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"media.search","arguments":{"query":"retrieval","limit":5}}' \
  "$MCP_BASE/tools/execute"
```

JSON-RPC:

```bash
curl \
  -H "$MCP_AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"call-1","method":"tools/call","params":{"name":"media.search","arguments":{"query":"retrieval","limit":5}}}' \
  "$MCP_BASE/request"
```

## Batch

```bash
curl \
  -H "$MCP_AUTH_HEADER" \
  -H "mcp-session-id: $MCP_SESSION" \
  -H "Content-Type: application/json" \
  -d '[{"jsonrpc":"2.0","id":"tools-1","method":"tools/list","params":{}},{"jsonrpc":"2.0","id":"health-1","method":"modules/health","params":{}}]' \
  "$MCP_BASE/request/batch"
```

## WebSocket

Use subprotocol auth for browser or persistent clients. This sends `Sec-WebSocket-Protocol: bearer,<token>`.

```javascript
const ws = new WebSocket(
  "ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=operator",
  ["bearer", "<jwt-token>"],
);

ws.onopen = () => {
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: "tools-1",
    method: "tools/list",
    params: {},
  }));
};
```

## Status, Health, Metrics

```bash
curl "$MCP_BASE/status"
curl "$MCP_BASE/health"
curl -H "$MCP_AUTH_HEADER" "$MCP_BASE/metrics"
curl -H "$MCP_AUTH_HEADER" "$MCP_BASE/metrics/prometheus"
curl -H "$MCP_AUTH_HEADER" "$MCP_BASE/modules/health"
```

Read `problem_modules`, `config_warnings`, and `surface` in `/status` before debugging individual tools.

## Client Wizard

Dry-run a client config without writing files:

```bash
python -m tldw_Server_API.cli.wizard mcp add \
  --client cursor \
  --api-key-env SINGLE_USER_API_KEY \
  --dry-run
```

Write the config and verify readiness:

```bash
python -m tldw_Server_API.cli.wizard mcp add \
  --client cursor \
  --api-key "$SINGLE_USER_API_KEY" \
  --verify
```

## Common Failures

| Symptom | Likely Cause | Next Action |
| --- | --- | --- |
| `401` or `403` | Missing, expired, or under-scoped credential | Use `Authorization: Bearer`, `X-API-KEY`, or rerun wizard with `--api-key` / `--api-key-env`; ask an admin for missing permissions. |
| `503 Server not initialized` on `/health` | Health was checked before initialization | Call `/status` or send `initialize`, then retry `/health`. |
| Empty `tools/list` | No module surface, no permission, or unresolved catalog | Check `/status`, `/tool_catalogs`, and `_meta.catalog.status`; remove or fix the catalog filter. |
| `_meta.catalog.status=unresolved` | Catalog name/id is not visible to this principal | List visible catalogs, fix spelling/scope, or remove the filter. |
| `invalid_safe_config` | `config` query value is not valid base64url JSON | Re-encode JSON config or omit the query parameter. |
| `problem_modules` present | Module dependency/config failure | Follow each module `next_action`; restart only after config is corrected. |
| Wizard says `configured_but_not_ready` | Client file was written with placeholder credentials | Set `SINGLE_USER_API_KEY`, pass `--api-key`, or pass `--api-key-env`, then rerun with `--verify`. |
