# MCP unified server

This document describes the embedded unified MCP server: WebSocket and HTTP transports, the protocol dispatch, the two-phase prepared tool call (security pipeline then runtime execution), the module registry with circuit breakers and failover, prompt-catalog continuation cursors, and federation. The standalone gateway is planned but not shipped — unified MCP is embedded in the server today.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Server | `app/core/MCP_unified/server.py` | `MCPServer` — initialize, default-module registration, WebSocket handling, session/connection cleanup loops |
| Protocol | `app/core/MCP_unified/protocol.py` | `MCPProtocol` — method dispatch (`initialize`, `tools/list`, `tools/call`, `resources/list|read`, `prompts/list|get`), prepared-call delegation, RBAC method mapping; protocol version `2024-11-05`; strict tool-name regex |
| Execution security | `MCP_unified/tool_execution/security.py` | `ToolExecutionSecurity.prepare_tool_call` — validation/governance pipeline producing a sealed `PreparedToolCall` |
| Execution runtime | `MCP_unified/tool_execution/runtime.py` | `ToolExecutionRuntime.execute_prepared_tool_call` — rate-limit admission, module admission, observers |
| Idempotency | `tool_execution/idempotency.py` | Redis-backed with a degraded-mode metric |
| Modules | `MCP_unified/modules/{base,registry}.py` + `implementations/` (35+) | `BaseModule` (tools/resources/prompts, context-aware variants, per-module circuit breaker), `ModuleRegistry` routing maps + `execute_with_failover` |
| Continuation | `implementations/prompts_catalog.py` | encoded continuation cursors (page clamp default 50/max 100), identifier-stripped for anonymity |
| Config | `MCP_unified/config.py` | `MCPConfig(BaseSettings)` — every `MCP_*` key |
| Transport/security | `MCP_unified/{auth,security,jsonrpc_transport,persona_scope,tool_observability}.py` | JWT manager, RBAC, rate limiter, IP filtering, JSON-RPC parsing |
| Federation | `MCP_unified/external_servers/` | external server manager, config schema, transports |
| Routes | `app/api/v1/endpoints/mcp_unified_endpoint.py` | `/api/v1/mcp` surface: `POST /request` (+batch), `WS /ws`, status/metrics, `POST /tools/execute`, modules/health, resources, prompts, auth token/refresh, health, catalog |

## Tool call through the gateway (dataflow)

1. Transport: WebSocket `/api/v1/mcp/ws` or HTTP `POST /api/v1/mcp/request` (`initialize` mints the `mcp-session-id` header; anonymous `tools/list` over HTTP is 403-filtered; batch supported).
2. `tools/call` dispatch — with a special guard rejecting `slides.*` tools on unguarded WebSockets — delegates to tool execution.
3. **Prepare** (security): strict name regex → context allow → effective policy → policy allow → external-access evaluation → RBAC module/tool permission → input-schema validation (flag-gated) → path-scope enforcement → runtime approval → idempotency normalization. The result is a `PreparedToolCall` sealed with a 32-byte secret.
4. **Execute** (runtime): rate-limit admission → module circuit-breaker admission → registry routing → the module's `execute_tool` with failover to a replica when a module is unhealthy.
5. Observers record tool-use events, metrics, and eval-profile metadata.
6. Resources route by URI; prompts aggregate context-aware listings with `nextCursor` continuations — cursors carry no prompt identifiers (anonymity) and each continuation must re-authenticate.
7. Module set is config-driven: the modules YAML (`Config_Files/mcp_modules.yaml`), `MCP_MODULES` env pairs, per-module flags; test mode auto-enables the media module.

## Config keys (selection; `MCPConfig` is authoritative)

`MCP_SERVER_NAME/_VERSION`, `MCP_JWT_SECRET/_ALGORITHM/_ACCESS_EXPIRE/_REFRESH_EXPIRE`, `MCP_VALIDATE_INPUT_SCHEMA` (true), `MCP_DISABLE_WRITE_TOOLS`, `MCP_IDEMPOTENCY_*`, `MCP_REDIS_*`, `MCP_RATE_LIMIT_*`, `MCP_WS_MAX_CONNECTIONS[_PER_IP]`, `MCP_WS_MAX_MESSAGE_SIZE` (1 MiB), `MCP_WS_AUTH_REQUIRED/_ALLOWED_ORIGINS/_ALLOW_QUERY_AUTH/_IDLE_TIMEOUT_SECONDS`, `MCP_ALLOWED_IPS/_BLOCKED_IPS`, `MCP_TRUST_X_FORWARDED` + trusted-proxy keys, `MCP_HTTP_MAX_BODY_BYTES` (512 KiB), client-cert and CORS keys, `MCP_MODULE_TIMEOUT` (30) / `_MAX_RETRIES` (3), `MCP_MODULES_CONFIG` / `MCP_MODULES` / `MCP_ENABLE_MEDIA_MODULE`.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Invalid/missing tool name | JSON-RPC `InvalidParams` |
| Not allowed by context/policy/RBAC | `PermissionError` (anonymous tools/list → HTTP 403 with hint) |
| Module unhealthy / circuit open | Typed breaker error; failover via the registry |
| Duplicate idempotency key in flight | Wait-then-reuse (degraded mode records a metric) |
| `slides.*` over unguarded WS | Structured failure `slides_websocket_guard_required` |
| Explicit JSON-RPC null ids | Sentinel round-trip preserving notification semantics |

## Verified gotchas

1. `MCPProtocol.__setattr__` live-syncs overridden/monkeypatched methods into the tool-execution sub-objects — subclass or hot-patch carefully.
2. Continuation cursors exist for **prompts** only; `tools/list` does not paginate.
3. Gateway-named tests and a `docker/` sketch exist, but the standalone gateway is not shipped — use the embedded path.
4. A lazy JWT-manager proxy exists purely to avoid import cycles.

## Governing decisions

ADR-032 (`Docs/ADR/032-mcp-unified-multi-revision-stdio-protocol.md`), ADR-033 (stdio contract hardening), ADR-052 (MCP admin claims — deliberately narrower than platform admin). References: `Docs/MCP/Unified/` guide family, `app/core/MCP_unified/README.md` (current). The chatbook counterpart is `mcp-hub.md` in the tldw_chatbook repo's architecture reference.

## Related docs

- [authnz.md](./authnz.md) — the admin predicate and principals
- [jobs-background.md](./jobs-background.md) — background execution conventions
