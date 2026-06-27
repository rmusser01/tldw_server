# MCP Unified Extraction — Design Doc / PRD

Author: tldw_server team
Status: Draft (v0.1)
Target Version: v0.2.x

> **Status note:** This PRD describes planned extraction work. It is not an installation guide for a shipped standalone package or gateway.

## 1) Summary
Extract the existing MCP Unified server into a reusable, installable module without removing it from tldw_server. The module will expose FastAPI routers (HTTP + WebSocket), clean interfaces for AuthNZ, rate limiting, metrics, and pluggable tools/modules. The tldw_server app will include these routers and provide adapters, preserving current behavior and API shapes.

High-level outcome: keep today’s endpoints and capabilities working in-place while enabling easy reuse by other projects via `pip install` or vendoring the subpackage.

## 2) Goals
- Create an installable `mcp_unified` subpackage with minimal external assumptions.
- Expose `FastAPI` routers for HTTP and WebSocket, mountable via `include_router`.
- Define adapter interfaces (AuthNZ, RateLimiter, RBAC/Policy, Metrics, Stream factory) and refactor MCP to depend on these.
- Preserve current API contracts and behavior in tldw_server, including:
  - WebSocket endpoint and JSON-RPC behavior: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:38`.
  - HTTP request endpoints (`/mcp/request`, `/mcp/request/batch`, discovery, tools, metrics).
  - Security posture (JWT/API key, RBAC, origin/IP guards), rate limits, metrics.
- Keep module registry and tool system working, with a documented plugin path.
- Provide a clear quick-start and adapter examples for FastAPI apps.

## 3) Non-Goals
- Rewriting MCP protocol or tool schemas.
- Changing wire payloads for JSON-RPC or domain messages.
- Removing the MCP server from tldw_server; we only extract and reuse.
- Large functional changes to AuthNZ or DB schemas.

## 4) Current State (as of v0.1.0)
- Core server lives in `tldw_Server_API/app/core/MCP_unified` with:
  - Server, protocol, auth, security, monitoring, and modules runtime.
  - Tests under `tldw_Server_API/app/core/MCP_unified/tests`.
  - Endpoints in `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` mount the server.
- Couplings to broader app:
  - AuthNZ: `jwt_service`, `api_key_manager`, single-user settings.
  - DB pool (seeding permissions) and `DatabasePaths` (default media module path).
  - Streaming helper: `tldw_Server_API.app.core.Streaming.streams.WebSocketStream`.
  - Metrics collector is internal, but can be made swappable.

## 5) Problem / Motivation
- External users cannot easily adopt the MCP runtime without pulling the whole repo.
- Cross-module imports complicate reuse and testing.
- Clear adapter seams improve maintainability, testing, and optional separation as a service.

## 6) Proposed Solution Overview
- Extract an installable subpackage `mcp_unified/` that exposes:
  - `get_mcp_router(settings, adapters...) -> APIRouter` (HTTP + WS under `/mcp`).
  - `create_app(settings, adapters...) -> FastAPI` (optional standalone).
- Introduce minimal interfaces (protocols) within `mcp_unified` and refactor code to depend on them.
- Provide tldw_server adapters that implement these interfaces using existing AuthNZ, rate limiting, and streaming.
- Maintain a “shim” module inside tldw_server that re-exports symbols to avoid breaking imports.

## 7) Architecture & Interfaces

### 7.1 Package Layout (new)
```
mcp_unified/
├── mcp_unified/
│   ├── __init__.py
│   ├── settings.py               # MCPSettings (Pydantic BaseSettings)
│   ├── interfaces/
│   │   ├── auth.py               # AuthBackend, RBACPolicy, ApiKeyInfo, TokenInfo
│   │   ├── rate_limiter.py       # RateLimiter, RateLimitExceeded
│   │   ├── metrics.py            # MetricsCollector
│   │   ├── streams.py            # StreamFactory (WS lifecycle abstraction)
│   │   └── ip_access.py          # IPAccessController
│   ├── core/
│   │   ├── protocol.py           # JSON-RPC protocol (moved/refactored)
│   │   ├── server.py             # MCPServer class (deps injected)
│   │   ├── modules/              # Module registry + default implementations
│   │   └── security/             # Request guards, cert checks
│   ├── api/
│   │   └── router.py             # get_mcp_router(...)
│   ├── tests/                    # Unit tests for module with no-op adapters
│   └── py.typed
└── pyproject.toml                # Installable package
```

### 7.2 Core Settings
- `MCPSettings` captures config now in `config.py` and endpoint code:
  - WS: `ws_max_connections`, `ws_max_connections_per_ip`, `ws_allowed_origins`,
    `ws_allow_query_auth`, `ws_auth_required`, `ws_ping_interval`, `ws_idle_timeout_seconds`.
  - HTTP: `idempotency_window`, `max_batch`, security toggles.
  - Debug flags used today.

### 7.3 Interfaces (Protocols)
Minimal shapes shown for clarity; concrete types live in `mcp_unified.interfaces.*`.

- AuthBackend
  - `async def decode_access_token(token: str) -> TokenInfo | None`
  - `async def validate_api_key(api_key: str, ip_address: str | None) -> ApiKeyInfo | None`
  - `def single_user_defaults() -> tuple[user_id: str | None, roles: list[str]]` (optional)

- RBACPolicy
  - `def has_permission(user: TokenInfo | None, permission: str) -> bool`
  - `def effective_roles(user: TokenInfo | None) -> list[str]`

- RateLimiter
  - `async def check(category: str, key: str, cost: int = 1) -> None`
  - Raises `RateLimitExceeded(retry_after: int)` on violation.

- MetricsCollector
  - `def update_connection_count(kind: str, count: int) -> None`
  - `def record_ws_rejection(reason: str, bucket: str | None = None) -> None`
  - `def get_prometheus_metrics() -> str`

- StreamFactory
  - `def create(websocket, heartbeat_interval_s: float | None, idle_timeout_s: float | None, labels: dict[str, str]) -> WebSocketStreamLike`
  - Returned object exposes `await start()`, plus underlying `send_json`/`close` via WebSocket.

- IPAccessController
  - `def allow(ip: str | None) -> bool`

These enable removal of direct imports of `tldw_Server_API.app.core.AuthNZ`, DB seeding, or project-specific streaming code.

### 7.4 Router API
- `get_mcp_router(settings: MCPSettings, *, auth: AuthBackend, rbac: RBACPolicy, limiter: RateLimiter, metrics: MetricsCollector | None = None, streams: StreamFactory | None = None, ip_access: IPAccessController | None = None) -> APIRouter`
  - Returns a router with:
    - `GET /mcp/status`, `GET /mcp/metrics`, `GET /mcp/metrics/prometheus`
    - `POST /mcp/request`, `POST /mcp/request/batch`
    - `GET /mcp/tools`, `POST /mcp/tools/call`, and existing discovery endpoints
    - `WS  /mcp/ws`
  - The router uses injected adapters; no global singletons required.

### 7.5 Server Instance & Lifespan
- `MCPServer(settings, adapters...)` constructed by the router; lifespan wired via FastAPI dependency or startup event.
- `create_app(...)` helper can build a standalone FastAPI app for separate deployment.

### 7.6 Module/Tool Registry
- Keep the current module registry and tool execution API, but:
  - Remove references to `DatabasePaths` defaults; instead expose settings for default module paths.
  - Keep YAML/env-based module autoload but allow host apps to pass a resolver callback for module configs.

### 7.7 Security & Guards
- Keep client-certificate and origin/IP checks but run them through injected `IPAccessController` and request-guard helpers.
- Maintain demo auth feature flags; default OFF unless debug/testing.

## 8) Backward Compatibility
- Import shim: retain `tldw_Server_API.app.core.MCP_unified` as a thin wrapper that re-exports from `mcp_unified`.
- Endpoint compatibility: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` continues to exist but delegates to the new router factory, maintaining route paths and schemas.
- Env var mapping: maintain current names; `MCPSettings` reads the same env vars, with deprecation warnings only if needed.
- Tests: keep integration tests in the main repo mounting the new router; unit tests live in the `mcp_unified` package.

## 9) Migration Plan (Phased)

Phase 0 — Inventory & guardrails (0.5d)
- Catalog all imports in MCP server that reference AuthNZ, DB, Streaming, and settings.
- Pin acceptance criteria and set CI gates.

Phase 1 — Introduce interfaces in-place (1–2d)
- Add `interfaces/` in current MCP directory and refactor server to call adapters instead of direct imports (via default adapters that wrap existing implementations).
- Ensure existing tests pass.

Phase 2 — Create `mcp_unified` subpackage (1d)
- Copy/refactor MCP code into `mcp_unified/` with identical functionality.
- Provide `get_mcp_router(...)` and optional `create_app(...)`.

Phase 3 — Wire tldw_server to extracted module (0.5–1d)
- Replace direct endpoint imports with the new router factory while keeping the same paths under `/api/v1/mcp`.
- Implement adapters using existing AuthNZ, rate limiter, metrics, and streaming helpers.

Phase 4 — Tests & docs (1–2d)
- Move/add unit tests under `mcp_unified/tests` with no-op adapters.
- Keep integration tests under `tldw_Server_API/tests` validating WebSocket + HTTP flows.
- Write quick-start docs and adapter examples.

Phase 5 — Optional packaging & release (0.5–1d)
- Add `pyproject.toml`, license notice, and publish as an internal package (local or PyPI later).

## 10) Testing Strategy
- Unit tests in `mcp_unified/tests`:
  - HTTP mapping: request/response behavior, error codes, auth adapter behavior with stubs.
  - WS lifecycle: ping/idle handling via a stub `StreamFactory`.
  - Rate limiting: simulate quota exceed and `retry_after` propagation.
  - RBAC: permission denials map to HTTP 403 with hints for tool calls.
- Integration tests in tldw_server:
  - Keep coverage for `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` mounting the router.
  - WebSocket smoke test and exception close codes.
  - Tool discovery, catalogs, and batch requests.
  - Prometheus endpoint security rules.

## 11) Security Considerations
- No relaxation of auth checks: priority is still AuthNZ JWT, then MCP JWT, then API key.
- Origin/IP guards and client-certificate enforcement remain intact via adapters.
- Explicit config gates for demo auth and public Prometheus endpoint.
- Secrets never logged; masking preserved.

## 12) Performance & Resource Limits
- Preserve current connection caps and metrics.
- Ensure adapters do not add overhead in hot paths; favor direct function calls.
- Rate limiter semantics unchanged; adapter can wrap RG or other custom implementations.

## 13) Deployment & Packaging
- In-repo subpackage with its own `pyproject.toml` and `py.typed`.
- Root `pyproject.toml` can add an extra `.[mcp]` that depends on `mcp_unified @ file://...` for local dev.
- Optional: publish to an internal index for broader reuse.

## 14) Acceptance Criteria
- tldw_server builds and runs with the extracted module mounted at the same paths.
- All existing MCP-related tests pass:
  - `tldw_Server_API/app/core/MCP_unified/tests/` WS/HTTP tests.
- External FastAPI app can:
  - Install `mcp_unified`, construct `get_mcp_router(...)`, and successfully call `/mcp/status`, `/mcp/request`, and open `/mcp/ws` with stub adapters.
- No breaking changes to existing env vars or route schemas.

## 15) Risks & Mitigations
- Hidden imports back into tldw_server: mitigate by CI checks and code review that enforce adapter boundaries.
- Rate limiter/router decoration coupling: use callable adapter instead of decorators bound to `app`.
- DB seeding for wildcard permission: move behind adapter or remove from server; document seeding in host app.
- Streaming lifecycle dependency: inject `StreamFactory` that wraps current `WebSocketStream`.
- Config drift: centralize in `MCPSettings` and avoid reading envs in multiple places.

## 16) Open Questions
- Should we keep the singleton `get_mcp_server()` pattern or construct per-router instance? Proposal: prefer per-router instance; keep a lightweight singleton in the shim for back-compat only.
- Where should module default configs live? Proposal: allow host to pass module loader callback; keep YAML/env support behind a flag.
- Do we want to publish as a separate package now or after internal stabilization?

## 17) Effort Estimate
- Extraction and adapters: 2–4 days.
- Tests and docs polish: 3–5 days.
- Optional packaging/publishing: 1–2 days.

## 18) Developer Experience (DevX)
- Mounting example in tldw_server (`tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` continues to exist, but internally calls the new router):

```python
# main.py (excerpt)
from fastapi import FastAPI
from mcp_unified.api.router import get_mcp_router
from tldw_Server_API.app.core.AuthNZ.adapters import McpAuthAdapter, McpRBACAdapter
from tldw_Server_API.app.core.Metrics.adapters import McpMetricsAdapter
from tldw_Server_API.app.core.Streaming.adapters import McpStreamFactory
from tldw_Server_API.app.core.RateLimit.adapters import McpRateLimiter
from tldw_Server_API.app.core.Security.adapters import McpIPAccess
from tldw_Server_API.app.core.MCP_unified.config_adapter import load_mcp_settings

app = FastAPI()
app.include_router(
    get_mcp_router(
        settings=load_mcp_settings(),
        auth=McpAuthAdapter(),
        rbac=McpRBACAdapter(),
        limiter=McpRateLimiter(),
        metrics=McpMetricsAdapter(),
        streams=McpStreamFactory(),
        ip_access=McpIPAccess(),
    ),
    prefix="/api/v1",
)
```

## 19) PR / Implementation Plan Checklist
- [ ] Add design-approved interfaces and default adapters in-place behind current MCP to reduce churn.
- [ ] Create `mcp_unified/` subpackage with `get_mcp_router(...)`.
- [ ] Add tldw_server adapters mapping to existing AuthNZ, rate limiter, metrics, and streaming.
- [ ] Replace endpoint wiring to use the router factory; ensure route paths unchanged.
- [ ] Keep a shim re-exporting symbols for import back-compat.
- [ ] Migrate/add tests and docs; update quick-start guides.
- [ ] Validate with external minimal FastAPI sample app.

---

Appendix A — References (code pointers)
- Current endpoint router: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py:38`
- MCP server: `tldw_Server_API/app/core/MCP_unified/server.py`
- Protocol: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Auth (current): `tldw_Server_API/app/core/MCP_unified/auth/*` and AuthNZ services
- Streaming helper: `tldw_Server_API/app/core/Streaming/streams.py`
- Metrics: `tldw_Server_API/app/core/MCP_unified/monitoring/metrics.py`
