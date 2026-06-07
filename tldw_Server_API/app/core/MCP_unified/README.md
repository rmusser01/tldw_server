# MCP Unified Module - Production Ready

## 1. Descriptive of Current Feature Set

- Purpose: Secure, production-ready Model Context Protocol (MCP) server with HTTP + WebSocket transport, JWT/RBAC, rate limiting, idempotency, module system, and Prometheus metrics.
- Capabilities:
  - Protocol: JSON-RPC 2.0 over WS/HTTP, tool execution, modules registry, resources/prompts discovery.
  - Security: Auth modes (AuthNZ JWT, MCP JWT, API keys), RBAC, rate limits, input validation, optional mTLS via proxy.
  - Operations: Health, status, module health, metrics (JSON + Prometheus), circuit breakers, runtime tuning.
  - Deployment: Env-first config, Redis-backed limiters, Postgres/SQLite backends.
- Inputs/Outputs:
  - Inputs: JSON-RPC requests (HTTP or WS) encapsulated as `MCPRequest`.
  - Outputs: `MCPResponse` for single/batch requests; JSON metrics/status payloads; Prometheus text.
- Related Endpoints (selected; mounted under `/api/v1/mcp`):
  - WebSocket `/ws`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:206
  - POST `/request`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:252
  - POST `/request/batch`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:366
  - GET `/status`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:453
  - GET `/metrics`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:476
  - GET `/metrics/prometheus`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:499
  - POST `/tools/execute`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:622
  - GET `/modules` and `/modules/health`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:711, 757
  - GET `/resources` and `/prompts`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:792, 838
  - POST `/auth/token` and `/auth/refresh`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:886, 968
  - GET `/health`: tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:1014
- Related Schemas/Types:
  - `MCPRequest`, `MCPResponse`, `RequestContext`: tldw_Server_API/app/core/MCP_unified/protocol.py:58, 90, 106
  - Server facade: `MCPServer`: tldw_Server_API/app/core/MCP_unified/server.py:108

## 2. Technical Details of Features

- Architecture & Components
  - Core server (`server.py`), protocol (`protocol.py`), auth/RBAC/rate limiter modules, module system (`modules/`), monitoring (`monitoring/metrics.py`).
  - Circuit breakers and runtime controls per module (concurrency limits, backoff, idempotency cache).
  - WS hardening: header/subprotocol auth, origin allowlist, query auth disabled by default.
- Data & Storage
  - Optional Postgres + Redis; SQLite defaults supported for local/offline.
- Configuration
  - Env-first configuration; secure defaults; test/development knobs gated.
- Security
  - JWT-based auth (AuthNZ tokens preferred), API keys, RBAC, schema validation, and SSRF/egress enforcement delegated where applicable.
- Observability
  - JSON metrics + Prometheus endpoint; health/status routes; request/operation counters.

## 3. Developer-Related/Relevant Information for Contributors

- Quick Start & Environment: See “🚀 Quick Start” below for env vars, dependencies, running tests, and starting the server.
- Folder Structure: See “📁 Directory Structure”.
- Endpoints: See “📊 API Endpoints” with WS/HTTP examples and auth modes.
- Monitoring & Health: See “📈 Monitoring”.
- Security Hardening: See “🔐 Production Hardening” and “🛡️ Security Checklist”.
- Adding Modules: See “➕ Adding Modules (Autoload)” and authoring guide in Docs.
- Tests: `tldw_Server_API/app/core/MCP_unified/tests/*` (unit/integration/security). Run with `pytest -m ...` markers.

---

## Overview
A secure, production-ready Model Context Protocol implementation that consolidates MCP v1 and v2 with enterprise-grade features.

## Managed External Credential Brokering

Managed external MCP servers now follow a brokered runtime model instead of static secret hydration.

- Runtime registry payloads stay auth-neutral. Managed server configs are loaded with `auth.mode=none`, and durable headers/env are not baked into adapter config.
- Effective policy and MCP Hub credential bindings are resolved at call time.
- Managed secret refs and legacy encrypted slot secrets are translated into transient execution headers/env only for the active tool call.
- Transport adapters must not persist brokered secret values in long-lived state, config snapshots, telemetry, or logs.

Current enterprise support matrix for this path:

- OIDC-backed AuthNZ federation is the supported identity source in phase 1.
- Local encrypted secret refs (`local_encrypted_v1`) are the first managed backend.
- Brokered credentials are applied per execution for managed websocket and stdio external MCP servers.

## ✅ What's Been Built

### Core Components
- **Secure Configuration** (`config.py`) - All secrets from environment variables
- **MCP Server** (`server.py`) - WebSocket and HTTP support with connection management
- **Protocol Handler** (`protocol.py`) - Full JSON-RPC 2.0 implementation
- **Module System** (`modules/`) - Extensible module architecture with health checks

### Security Layer (All vulnerabilities fixed!)
- **JWT Authentication** (`auth/jwt_manager.py`) - No hardcoded secrets, token rotation
- **RBAC** (`auth/rbac.py`) - Fine-grained permissions with role inheritance
- **Rate Limiting** (`auth/rate_limiter.py`) - Token bucket and sliding window algorithms

### Production Features
- **Health Monitoring** - Automatic health checks with circuit breakers
- **Metrics Collection** (`monitoring/metrics.py`) - Prometheus-compatible metrics
- **Connection Pooling** - Efficient resource management
- **Graceful Degradation** - Circuit breaker pattern for resilience

## 🚀 Quick Start

### 1. Set Required Environment Variables

```bash
# Generate secure secrets
export MCP_JWT_SECRET=$(openssl rand -base64 32)
export MCP_API_KEY_SALT=$(openssl rand -base64 32)

# Optional configuration
export MCP_LOG_LEVEL=INFO
export MCP_RATE_LIMIT_RPM=60
export MCP_DATABASE_URL=sqlite+aiosqlite:///./Databases/mcp_unified.db
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn loguru pydantic PyJWT passlib bcrypt aiosqlite
```

### 3. Run Tests

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/ -v
```

### 4. Start Server

```python
from tldw_Server_API.app.core.MCP_unified import get_mcp_server

server = get_mcp_server()
await server.initialize()
```

## 📁 Directory Structure

```
MCP_unified/
├── __init__.py              # Main exports
├── config.py                # Secure configuration
├── server.py                # MCP server
├── protocol.py              # Protocol handler
├── auth/                    # Authentication & authorization
│   ├── jwt_manager.py       # JWT management (no hardcoded secrets!)
│   ├── rbac.py             # Role-based access control
│   └── rate_limiter.py    # Rate limiting
├── modules/                 # Module system
│   ├── base.py             # Base module interface
│   ├── registry.py         # Module registry
│   └── implementations/    # Module implementations
├── monitoring/              # Observability
│   └── metrics.py          # Metrics collection
└── tests/                   # Test suite
    └── test_basic_functionality.py

```

## 🔒 Security Features

### No Hardcoded Secrets
- All sensitive configuration from environment variables
- Validation on startup to prevent use of default values
- Secure random generation if not provided (with warnings)

### Authentication & Authorization
- JWT with access and refresh tokens
- Token rotation for enhanced security
- Fine-grained RBAC with permission inheritance
- API key management with PBKDF2 hashing

### WebSocket Hardening
- Require WS authentication in production (`MCP_WS_AUTH_REQUIRED=true`)
- Explicitly set allowed origins (`MCP_WS_ALLOWED_ORIGINS=http://your-ui:port`)

### Client Certificate (mTLS) via Reverse Proxy
- Enforce client certs by enabling `MCP_CLIENT_CERT_REQUIRED=true`
- Configure the header asserted by your proxy (default `x-ssl-client-verify`)
- Set `MCP_CLIENT_CERT_HEADER_VALUE` to the exact success sentinel (e.g., `SUCCESS`)
- Only trusted proxies may assert this header - set `MCP_TRUSTED_PROXY_IPS` to your proxy CIDRs
- X-Forwarded-For is honored only from trusted proxies (`MCP_TRUST_X_FORWARDED=true` if desired)

Note: In test mode (`TEST_MODE=true`) the harness peer is treated as trusted for convenience.

### Rate Limiting
- Multiple algorithms (Token Bucket, Sliding Window)
- Per-user and per-endpoint limits
- Redis support for distributed deployments
- Automatic cleanup of old entries

### Input Validation
- Pydantic models for all inputs
- SQL injection prevention
- XSS protection
- Request size limits

### SSRF Protection (Media Ingestion)
- Media module only accepts `http/https` URLs
- Rejects `.local` and hosts resolving to loopback/private/link-local/reserved/multicast IPs
- Enforce allowed ports (`80,443` by default)
- Optional allowlist per deployment via module settings:
  - `allowed_domains`: ["example.com", "cdn.example.com"]
  - `allowed_ports`: [80, 443]
  - `blocked_domains`: ["unwanted.tld"]

## 🎯 Key Improvements Over Original

| Feature | Original MCP v1/v2 | Unified MCP |
|---------|-------------------|-------------|
| JWT Secret | Hardcoded in code | Environment variable |
| Rate Limiting | Basic/None | Advanced with Redis support |
| Health Checks | None | Automatic with caching |
| Circuit Breakers | None | Built-in with configurable thresholds |
| Metrics | None | Prometheus-compatible |
| Input Validation | Basic | Comprehensive with Pydantic |
| Error Handling | Generic | Detailed with proper codes |
| Testing | Minimal | Comprehensive test suite |

## 📊 API Endpoints

### Authentication

MCP Unified supports multiple authentication methods:

- AuthNZ JWT (preferred): `Authorization: Bearer <AuthNZ access token>`
- MCP JWT (back-compat): `Authorization: Bearer <MCP JWT>`
- API Key (HTTP): `X-API-KEY: <api_key>`
- API Key (WebSocket): query param `api_key=<api_key>`

When using API keys, RequestContext.metadata includes `org_id` and `team_id` (if present on the key) so modules can scope behavior.

### WebSocket
```
ws://localhost:8000/api/v1/mcp/ws?client_id=<id>&token=<jwt>
# or
ws://localhost:8000/api/v1/mcp/ws?client_id=<id>&api_key=<api_key>
# Recommended (headers/subprotocol):
#   Authorization: Bearer <token>
#   Sec-WebSocket-Protocol: bearer,<token>
```

### HTTP Endpoints
- `POST /api/v1/mcp/request` - Process MCP request
- `GET /api/v1/mcp/status` - Server status
- `GET /api/v1/mcp/metrics` - Server metrics (admin only)
- `GET /api/v1/mcp/tools` - List available tools (auth required; RBAC-filtered)
- `POST /api/v1/mcp/tools/execute` - Execute tool (auth required)
- `POST /api/v1/mcp/auth/token` - Issue demo MCP tokens when explicitly enabled
- `POST /api/v1/mcp/auth/refresh` - Rotate refresh tokens (JSON body required)
- `GET /api/v1/mcp/health` - Health check

#### Refresh Token Contract
- `POST /api/v1/mcp/auth/refresh` accepts refresh credentials in the request body:
  - `{"refresh_token":"<token>","token_id":"<optional-token-id>"}`
- Query-string refresh token transport is rejected to avoid leaking secrets via URLs/logs.

#### Tool Discovery & Catalogs
- Reduce discovery size by grouping tools into catalogs (global, org, team).
- Filtering:
  - HTTP: `GET /api/v1/mcp/tools?catalog=<name>` or `?catalog_id=<id>`
  - JSON-RPC: `tools/list` with `{ catalog?: string, catalog_id?: number }`
- Name resolution respects caller context with precedence `team > org > global`; `catalog_id` takes precedence.
- Responses include `canExecute` per tool; catalog membership does not grant execution rights.
- See `Docs/MCP/mcp_tool_catalogs.md` for admin/manager APIs to create/manage catalogs.

## 🛡️ Production Checklist

- Set secure secrets: `MCP_JWT_SECRET`, `MCP_API_KEY_SALT`
- Enforce WS auth: `MCP_WS_AUTH_REQUIRED=true`
- Configure `MCP_WS_ALLOWED_ORIGINS`
- Keep WS query-parameter auth disabled (default): `MCP_WS_ALLOW_QUERY_AUTH=0`; use headers/subprotocol instead
- If using mTLS via proxy: set `MCP_CLIENT_CERT_REQUIRED=true`, `MCP_CLIENT_CERT_HEADER_VALUE`, and `MCP_TRUSTED_PROXY_IPS`
- Keep rate limiting enabled; configure Redis for multi-instance
- Do not use wildcard CORS in production

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tldw_Server_API/app/core/MCP_unified/tests/ -v

# Specific test categories
pytest -m unit        # Unit tests
pytest -m integration # Integration tests
pytest -m security    # Security tests
```

## 📝 Module Development

Create a new module by extending `BaseModule`:

```python
from tldw_Server_API.app.core.MCP_unified.modules import BaseModule

class MyModule(BaseModule):
    async def on_initialize(self):
        # Initialize resources
        pass

    async def check_health(self) -> Dict[str, bool]:
        return {"service": True}

    async def get_tools(self) -> List[Dict[str, Any]]:
        return [...]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        # Execute tool with circuit breaker
        return await self.execute_with_circuit_breaker(
            self._do_work, arguments
        )
```

## ➕ Adding Modules (Autoload)

Modules can be autoloaded via YAML or environment variables:

- YAML (default path `tldw_Server_API/Config_Files/mcp_modules.yaml`):
```
modules:
  - id: media
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule
    enabled: true
    name: Media
    settings:
      # Per-user default (single-user mode example): <USER_DB_BASE_DIR>/1/<media-db-file>
      # For multi-user setups, run a module instance per user or pass user-specific db_path at runtime.
      db_path: <USER_DB_BASE_DIR>/1/<media-db-file>
    # Optional runtime controls
    # Limit concurrent operations per module instance
    max_concurrent: 16
    # Circuit breaker tuning
    circuit_breaker_threshold: 3
    circuit_breaker_timeout: 30
    circuit_breaker_backoff_factor: 2.0
    circuit_breaker_max_timeout: 180
```

`USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to `Databases/user_databases/` under the project root). Override via environment variable or `Config_Files/config.txt` as needed.

- Environment variable (comma-separated list):
```
export MCP_MODULES="example=tldw_Server_API.app.core.MCP_unified.modules.implementations.template_module:TemplateModule"
```

- Optional quick-start flag:
```
export MCP_ENABLE_MEDIA_MODULE=true
```

Tool results include the serving module:
```
{"content": [...], "module": "Media", "tool": "search_media"}
```

See `Docs/MCP/Unified/Modules.md` for a complete guide.

## Tool Observability And Evaluation Metadata

MCP Unified attaches a shared, non-sensitive evaluation contract across the tool
surface so operators can compare tool prompt variants, profile grants, model
tool-use quality, and external-server behavior without scraping raw tool output.

Tool definitions include `metadata.eval` with:

- `tool_prompt_id`: stable identifier, defaulting to `mcp.<tool-name>.v1`
- `tool_prompt_version`: version of the built-in or operator-supplied tool prompt
- `task_families`: coarse evaluation category derived from explicit metadata or tool name
- `expected_result_kind`: expected structured result family
- `success_signals`: safe rubric hints such as `avoided_mutation`
- `prompt_variant`: `builtin`, `alias`, `external_federated`, or an operator variant

Execution responses include top-level `eval` metadata. Structured JSON tool
results that do not already provide their own `eval` block also receive an
embedded copy. These fields are intentionally limited to scalar values such as
tool name, prompt id/version, action family, result kind, optional profile id,
truncation/path-filter flags, reason code, and duration. They must not contain
raw arguments, raw file contents, diffs, secrets, absolute local paths, or user
email addresses. Profile IDs are accepted only as short safe labels using
letters, numbers, `_`, `.`, or `-`; unsafe values are omitted.

Module authors should use `create_tool_definition()` for new tools. It merges
safe explicit `metadata.eval` fields over inferred defaults and drops unknown or
non-string scalar fields.
Manual descriptors and federated external tools are normalized at catalog and
protocol boundaries so standalone gateway and hosted MCP callers see the same
contract.

## Git Read-Only Inspection Module

The optional native Git inspection module is enabled only when explicitly
configured by the operator:

```bash
export MCP_ENABLE_GIT_MODULE=true
```

When enabled, the module exposes these read-only tools:

- `git.status`
- `git.diff`
- `git.log`
- `git.blame`
- `git.branches`
- `git.conflicts.list`
- `git.conflicts.read`

The module is bound to the active workspace repository root. Callers cannot
supply an alternate repository path. The tools do not expose checkout, add,
commit, merge, rebase, stash, reset, clean, push, pull, arbitrary Git argv, or
host shell execution.

Git responses are intentionally privacy- and safety-bounded. Ignored files are
excluded from status output; author emails are omitted from log and blame
output; and external diff/textconv processing is disabled for diff reads.
Responses are bounded by tool limits and include non-sensitive evaluation
metadata through the shared MCP tool observability contract.

## 🧭 Phase-1 Virtual CLI Runtime

Phase-1 adds a governed virtual CLI foundation to MCP Unified.

- New module IDs: `filesystem`, `knowledge`, `run_command`
- New tool names: `fs.list`, `fs.read_text`, `fs.write_text`, `run`
- Typed MCP tools remain directly available; `run` is an additive orchestration surface.

`run` is not a raw host shell. It compiles command steps into policy-checked MCP tool calls (`prepare_tool_call` / `execute_prepared_tool_call`) so approvals, RBAC, path scope, validation, and idempotency all still apply.

`bash` and `shell` may also appear as compatibility aliases for `run`. They use the same `command` argument and the same governed runtime; they are not host shell execution surfaces.

Phase-1 command families:

- Pure transforms (no MCP backend call): `grep`, `head`, `tail`, `json`
- MCP-backed adapters:
  - `ls` -> `fs.list`
  - `cat` -> `fs.read_text`
  - `write` -> `fs.write_text`
  - `stat` -> `fs.stat`
  - `glob`, `find` -> `fs.glob`
  - `rg`, `grep-files` -> `fs.grep`
  - `knowledge` -> `knowledge.search`, `knowledge.get`
  - `media` -> `media.search`, `media.get`
  - `mcp` -> `mcp.modules.list`, `mcp.tools.list`
  - `sandbox` -> `sandbox.run`

Command aliases are policy-filtered by their backing MCP tools. For example, `rg` and `grep-files` are visible only when `fs.grep` is executable in the active profile. Plain `grep` remains a pure stdin filter for pipelines such as `cat app.log | grep ERROR`.

Default `run_command` runtime settings in module inventory:

- `spill_dir: ${MCP_RUN_COMMAND_SPILL_DIR:-.mcp/spills}`
- `spill_threshold_bytes: 65536`
- `preview_line_limit: 200`
- `preview_byte_limit: 51200`

Targeted validation commands for this phase:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_parser.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_execution.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_presentation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  tldw_Server_API/tests/MCP/test_mcp_tools_execute_authz.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py \
  -v
```

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified -f json -o /tmp/bandit_mcp_virtual_cli_phase1.json
```

## 🚢 Production Deployment

### Environment Variables (Required)
```bash
MCP_JWT_SECRET=<strong-random-secret>
MCP_API_KEY_SALT=<strong-random-secret>
```

### Environment Variables (Optional)
```bash
MCP_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/mcp
RG_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
MCP_RATE_LIMIT_ENABLED=true
MCP_RATE_LIMIT_RPM_INGESTION=30         # Optional per-category rate; default = RPM
MCP_RATE_LIMIT_BURST_INGESTION=5
MCP_RATE_LIMIT_RPM_READ=120
MCP_METRICS_ENABLED=true
MCP_LOG_LEVEL=INFO
MCP_WS_AUTH_REQUIRED=1                  # Require authenticated WS (prod hardening)
MCP_WS_ALLOWED_ORIGINS=https://your-ui.example.com  # Enforce WS Origin; comma-separated list
MCP_WS_ALLOW_QUERY_AUTH=0               # Disable ?token= / ?api_key= query auth (use headers/subprotocol)
```

## 🔐 Production Hardening

Recommended hardening steps for Internet-exposed deployments:

- Require WS auth and enforce allowed origins
  - Set `MCP_WS_AUTH_REQUIRED=1`
  - Set `MCP_WS_ALLOWED_ORIGINS=https://your-ui.example.com` (comma-separated if multiple)
  - Prefer header-based auth for WS: `Authorization: Bearer <token>` or `X-API-KEY`
  - Optional: Subprotocol auth: `Sec-WebSocket-Protocol: bearer,<token>`
- Disable query-string authentication for WS
  - `MCP_WS_ALLOW_QUERY_AUTH=0` (default). If a client passes `?token=` or `?api_key=`, the server ignores it and logs a warning.
- Rate limiting
  - For multi-node rate-limit consistency, use RG Redis backend: `RG_BACKEND=redis` and shared `REDIS_URL=redis://...`
  - Keep `MCP_RATE_LIMIT_ENABLED=1` to enforce MCP route-map policies.
- Restrict module autoloads
  - Only classes under `tldw_Server_API.app.core.MCP_unified.modules.implementations` are allowed when auto-loading.
- Write tools safety & validation (Security knobs)
  - `MCP_DISABLE_WRITE_TOOLS=0|1` - If set to `1`, the protocol blocks all write-capable tools (category `ingestion`/`management`).
  - `MCP_VALIDATE_INPUT_SCHEMA=0|1` - Validate tool `inputSchema` at the protocol layer (required fields, primitive types, unknown fields).
  - `MCP_IDEMPOTENCY_TTL_SECONDS` - TTL for protocol-level idempotency cache for write tools (default: 300s).
  - `MCP_IDEMPOTENCY_CACHE_SIZE` - Max entries for idempotency cache (LRU, default: 512).
  - Client hint: pass `idempotencyKey` in JSON-RPC `tools/call` params to dedupe writes.
  - Idempotency keys are bound to the first request argument fingerprint. Reusing a key with different arguments returns `INVALID_PARAMS` instead of replaying stale payloads.
- Demo auth (dev only)
  - `MCP_ENABLE_DEMO_AUTH` is for development/testing. If enabled in non-debug environments, the server logs a loud warning.
  - `MCP_DEMO_AUTH_SECRET` must be set to a strong value; the token endpoint also requires loopback/private clients and debug/test mode.

### Security Knobs (Quick Reference)

| Knob | Env/Config | Default | Purpose |
|---|---|---|---|
| WebSocket auth required | `MCP_WS_AUTH_REQUIRED` | `1` | Enforce `Authorization`/`X-API-KEY` headers for WS clients |
| WebSocket allowed origins | `MCP_WS_ALLOWED_ORIGINS` | *(empty)* | Comma-separated Origin allowlist to prevent UI spoofing |
| WebSocket query auth | `MCP_WS_ALLOW_QUERY_AUTH` | `0` | Reject `?token=`/`?api_key=` query parameters (set `1` only for legacy clients) |
| WebSocket idle timeout | `MCP_WS_IDLE_TIMEOUT_SECONDS` | `300` | Close idle WS sessions after N seconds |
| WebSocket session rate cap | `MCP_WS_SESSION_RATE_COUNT` / `MCP_WS_SESSION_RATE_WINDOW_SECONDS` | `120 / 60` | Sliding-window JSON-RPC rate limits per session |
| Disable write tools | `MCP_DISABLE_WRITE_TOOLS` | `0` | Hard block write-capable tools (ingestion/management categories) |
| Input schema validation | `MCP_VALIDATE_INPUT_SCHEMA` | `1` | Enforce required fields, primitive types, unknown-field rejection |
| Request size guard | `MCP_HTTP_MAX_BODY_BYTES` | `524288` | Reject oversized HTTP payloads (bytes) |
| IP allow/deny lists | `MCP_ALLOWED_IPS` / `MCP_BLOCKED_IPS` / `MCP_TRUSTED_PROXY_IPS` | *(empty)* | Defense-in-depth for client networks and the proxies whose X-Forwarded-For headers are trusted |
| Client certificates | `MCP_CLIENT_CERT_REQUIRED`, `MCP_CLIENT_CERT_HEADER`, `MCP_CLIENT_CERT_HEADER_VALUE` | `0`, `x-ssl-client-verify`, *(empty)* | Require mTLS headers from reverse proxy (e.g., NGINX, ALB) |
| Idempotency cache TTL | `MCP_IDEMPOTENCY_TTL_SECONDS` | `300` | Time window for write-tool dedupe |
| Idempotency cache size | `MCP_IDEMPOTENCY_CACHE_SIZE` | `512` | LRU size for idempotency cache entries |

### WebSocket Session Policy Knobs

Configure WS behavior to protect the server from idle and bursty sessions:

- `MCP_WS_IDLE_TIMEOUT_SECONDS` (default: 300) - If no activity for this many seconds, the server closes the WS with code 1001 (Idle timeout).
- `MCP_WS_SESSION_RATE_COUNT` (default: 120) - Max JSON-RPC requests allowed per session over the configured window.
- `MCP_WS_SESSION_RATE_WINDOW_SECONDS` (default: 60) - Sliding window in seconds used for per-session rate counting.

Notes
- When the session rate is exceeded, the server sends a JSON-RPC error (-32002) and closes the connection with code 1013 (session rate limit exceeded).
- The server emits Prometheus counters for WS session closures by reason (idle/session_rate): `mcp_ws_session_closures_total{reason="..."}`.

## 🔧 Rate Limits

MCP supports global and per-category (tool-driven) rate limits.

- Global limiter: configured via MCP_RATE_LIMIT_ENABLED, MCP_RATE_LIMIT_RPM, MCP_RATE_LIMIT_BURST.
- Distributed deployments: use shared RG Redis backend via `RG_BACKEND=redis` + `REDIS_URL`.
- Per-category limiters:
  - Categories: free-form labels; project recognizes at least ‘ingestion’ and ‘read’.
  - Category RPM/bursts via env:
    - MCP_RATE_LIMIT_RPM_INGESTION, MCP_RATE_LIMIT_BURST_INGESTION
    - MCP_RATE_LIMIT_RPM_READ (burst falls back to global burst)
  - Tool → category mapping:
    - JSON env (MCP_TOOL_CATEGORY_MAP)
    - YAML file (MCP_TOOL_CATEGORY_MAP_FILE)

Examples

- JSON env mapping:
```bash
export MCP_TOOL_CATEGORY_MAP='{"ingest_media":"ingestion","media.search":"read","mock_ingest":"ingestion"}'
```

- YAML mapping file (recommended):
```yaml
# tldw_Server_API/Config_Files/mcp_tool_categories.yaml
ingest_media: ingestion
update_media: ingestion
delete_media: ingestion

media.search: read
knowledge.search: read
notes.search: read
```
Use with:
```bash
export MCP_TOOL_CATEGORY_MAP_FILE=tldw_Server_API/Config_Files/mcp_tool_categories.yaml
```

Notes
- Config mapping takes precedence over the heuristic fallback (which classifies ingest_media/update_media/delete_media as ‘ingestion’).
- If Redis is enabled, per-category limiters also use Redis; otherwise in-memory token buckets are used.

See also: Ops tuning guide at Docs/Deployment/Operations/MCP_Rate_Limits_Tuning.md

### Docker Deployment
```bash
docker build -f docker/Dockerfile -t mcp-unified .
docker run -d \
  -e MCP_JWT_SECRET=$MCP_JWT_SECRET \
  -e MCP_API_KEY_SALT=$MCP_API_KEY_SALT \
  -p 8000:8000 \
  mcp-unified
```

## 📈 Monitoring

### Prometheus Metrics
Scrape MCP metrics at `GET /api/v1/mcp/metrics/prometheus` (text exposition format):
- Request rates and latencies (per MCP method)
- Module operation metrics (per module)
- Connection statistics (WebSocket)
- Rate limit hits
- Cache hit/miss rates
- System resource usage
 - Validation metrics:
   - `mcp_tool_invalid_params_total{module,tool}` - schema/validator failures
   - `mcp_tool_validator_missing_total{module,tool}` - write tools missing custom validators
 - Idempotency metrics:
   - `mcp_idempotency_hits_total{module,tool}` - protocol-level idempotency cache hits
   - `mcp_idempotency_misses_total{module,tool}` - protocol-level idempotency cache misses

Security: The Prometheus endpoint is gated by `RequirePermission(SYSTEM_LOGS)` on `AuthPrincipal` (admin-style principals also pass). Unauthenticated scraping is not supported; use credentials or an auth proxy for Prometheus.

### Health Checks
- `/api/v1/mcp/health` - Overall health
- `/api/v1/mcp/modules/health` - Module-specific health

## ⚙️ Module Runtime Controls

Tune module behavior and resilience without changing code.

- Concurrency guard (per module)
  - `ModuleConfig.max_concurrent` - Maximum concurrent operations per module (default: 20). Set to 0 to disable the guard.
- Circuit breaker backoff (per module)
  - `ModuleConfig.circuit_breaker_threshold` - Failures before opening (default: 5)
  - `ModuleConfig.circuit_breaker_timeout` - Initial open window in seconds (default: 60)
  - `ModuleConfig.circuit_breaker_backoff_factor` - Multiplier applied when re-opening after half-open failure (default: 2.0)
  - `ModuleConfig.circuit_breaker_max_timeout` - Cap for backoff window (default: 300)

How it works
- When the breaker opens and the timeout elapses, the next call enters half-open state (one probe).
- If the probe succeeds, the breaker heals and the timeout resets to baseline.
- If the probe fails, the breaker re-opens with an exponentially increased timeout (capped).

## 🛡️ Security Checklist

- ✅ No hardcoded secrets
- ✅ JWT authentication with rotation
- ✅ RBAC with fine-grained permissions
- ✅ Rate limiting protection
- ✅ Input validation and sanitization
- ✅ SQL injection prevention
- ✅ XSS protection
- ✅ CORS configuration
- ✅ Audit logging
- ✅ Secure password hashing (bcrypt)

## 📚 Documentation

- Developer Guide: `Docs/MCP/Unified/Developer_Guide.md`
- System Admin Guide: `Docs/MCP/Unified/System_Admin_Guide.md`
- User Guide: `Docs/MCP/Unified/User_Guide.md`
- Module Authoring: `Docs/MCP/Unified/Modules.md`
- Documentation Ingestion Playbook: `Docs/MCP/Unified/Documentation_Ingestion_Playbook.md`
- Context search design (FTS-first): `Docs/User_Guides/WebUI_Extension/context_mcp_search.md`
- API documentation available at `/docs` when server is running

## 🤝 Contributing

1. Follow existing patterns and conventions
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass
5. No hardcoded secrets or credentials

## 📄 License

Part of tldw_server project - see main LICENSE file.
### Authorization (RBAC)

MCP Unified now uses the project's AuthNZ RBAC (roles, permissions, overrides). Tool execution uses fine-grained permissions:

- Per-tool permission: `tools.execute:<tool_name>`
- Wildcard permission: `tools.execute:*`

Admin endpoints for managing tool permissions:

- List: `GET /api/v1/admin/permissions/tools`
- Create: `POST /api/v1/admin/permissions/tools` with `{ "tool_name": "*" | "<name>", "description": "..." }`
- Delete: `DELETE /api/v1/admin/permissions/tools/{perm_name}`

Grant/revoke tool permissions to roles:

- Grant: `POST /api/v1/admin/roles/{role_id}/permissions/tools` with `{ "tool_name": "*" | "<name>" }`
- Revoke: `DELETE /api/v1/admin/roles/{role_id}/permissions/tools/{tool_name}`

Example: seed wildcard and grant to a role

```bash
# Create wildcard permission (if not present)
curl -X POST http://127.0.0.1:8000/api/v1/admin/permissions/tools \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"*","description":"Allow executing all tools"}'

# Grant wildcard to role (replace 1 with your role id)
curl -X POST http://127.0.0.1:8000/api/v1/admin/roles/1/permissions/tools \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"*"}'
```
