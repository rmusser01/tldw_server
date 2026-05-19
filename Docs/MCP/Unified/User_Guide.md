# MCP Unified - User Guide

> Part of the MCP Unified documentation set. See `Docs/MCP/Unified/README.md` for the full guide index.

This guide is for users and integrators who want to run MCP Unified, connect clients, configure modules, and extend MCP with new modules/tools.

## 1. What MCP Unified Is

MCP Unified is the TLDW server's production Model Context Protocol surface. It supports:

- JSON-RPC over HTTP and WebSocket
- Tool discovery and execution
- Module-based tool loading
- AuthNZ JWT / MCP JWT / API key auth paths
- RBAC-aware tool permissions
- Health and metrics endpoints

Main base path: `http://127.0.0.1:8000/api/v1/mcp`

## 2. Getting Started

### Prerequisites

- TLDW repo checked out locally
- Virtual environment created
- Dependencies installed
- TLDW server configured (including AuthNZ if required)

### Step 1: Start the server

From repo root:

```bash
source .venv/bin/activate
python -m uvicorn tldw_Server_API.app.main:app --reload
```

### Step 2: Confirm MCP is reachable

`/status` will initialize MCP server state if needed:

```bash
curl http://127.0.0.1:8000/api/v1/mcp/status
```

Then check health:

```bash
curl http://127.0.0.1:8000/api/v1/mcp/health
```

### Step 3: Authenticate

Recommended in production: use an AuthNZ access token.

```bash
curl -H "Authorization: Bearer <authnz_access_token>" \
  http://127.0.0.1:8000/api/v1/mcp/tools
```

HTTP API key option:

```bash
curl -H "X-API-KEY: <api_key>" \
  http://127.0.0.1:8000/api/v1/mcp/tools
```

Demo token endpoint (`POST /auth/token`) exists only for debug/test workflows and is disabled unless explicitly enabled with:

- `MCP_ENABLE_DEMO_AUTH=1`
- `MCP_DEMO_AUTH_SECRET=<strong-secret>`

## 3. Endpoint Quick Reference

| Endpoint | Method | Purpose | Auth |
|---|---|---|---|
| `/api/v1/mcp/ws` | WS | Full MCP over WebSocket | Usually required (`MCP_WS_AUTH_REQUIRED=1`) |
| `/api/v1/mcp/request` | POST | Single JSON-RPC request | Method-dependent |
| `/api/v1/mcp/request/batch` | POST | Batch JSON-RPC requests | Method-dependent |
| `/api/v1/mcp/tools` | GET | List tools (RBAC filtered) | Recommended |
| `/api/v1/mcp/tools/execute` | POST | Execute one tool via HTTP facade | Required |
| `/api/v1/mcp/modules` | GET | List loaded modules | Recommended |
| `/api/v1/mcp/modules/health` | GET | Module health details | `system.logs` or admin |
| `/api/v1/mcp/resources` | GET | List MCP resources | Recommended |
| `/api/v1/mcp/prompts` | GET | List MCP prompts | Recommended |
| `/api/v1/mcp/tool_catalogs` | GET | List visible tool catalogs | Required |
| `/api/v1/mcp/metrics` | GET | MCP metrics (JSON) | `system.logs` or admin |
| `/api/v1/mcp/metrics/prometheus` | GET | Prometheus scrape output | `system.logs` or admin |
| `/api/v1/mcp/status` | GET | Server status summary | Not required |
| `/api/v1/mcp/health` | GET | Health probe | Not required |

## 4. Using MCP Over HTTP

### JSON-RPC request endpoint

Send MCP requests to:

- `POST /api/v1/mcp/request`
- `POST /api/v1/mcp/request/batch`

Example `initialize`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/mcp/request \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
      "clientInfo": { "name": "example-client", "version": "1.0.0" }
    },
    "id": 1
  }'
```

Example `tools/list`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/mcp/request \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
  }'
```

Example `tools/call`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/mcp/request \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "media.search",
      "arguments": { "query": "retrieval augmented generation", "limit": 5 }
    },
    "id": 3
  }'
```

### Convenience HTTP endpoints

- `GET /api/v1/mcp/tools`
- `POST /api/v1/mcp/tools/execute`
- `GET /api/v1/mcp/modules`
- `GET /api/v1/mcp/resources`
- `GET /api/v1/mcp/prompts`

The convenience endpoints map to MCP operations and keep the same RBAC behavior.

### Sessions and safe config

`/request` and `/request/batch` support:

- `mcp-session-id` header
- `config` query parameter (base64 JSON safe config)

If `initialize` is sent without `mcp-session-id`, the server can return one in the response header. Reuse it on later requests.

## 5. Using MCP Over WebSocket

Endpoint:

```text
ws://127.0.0.1:8000/api/v1/mcp/ws
```

Recommended auth:

- `Authorization: Bearer <token>` header
- `X-API-KEY: <api_key>` header
- `Sec-WebSocket-Protocol: bearer,<token>` subprotocol form

Query auth (`?token=` or `?api_key=`) is disabled by default. Enable only for legacy clients with:

- `MCP_WS_ALLOW_QUERY_AUTH=1`

Minimal JavaScript example:

```javascript
const token = "<token>";
const ws = new WebSocket(
  "ws://127.0.0.1:8000/api/v1/mcp/ws?client_id=web-client",
  ["bearer", token]
);

ws.onopen = () => {
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    method: "initialize",
    params: { clientInfo: { name: "web-client", version: "1.0.0" } },
    id: 1
  }));
};
```

## 6. Configure MCP Unified

### Common environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MCP_JWT_SECRET` | auto-generated if missing | MCP JWT signing secret (set explicitly in production) |
| `MCP_API_KEY_SALT` | auto-generated if missing | API key hashing salt |
| `MCP_DATABASE_URL` | `sqlite+aiosqlite:///./Databases/mcp_unified.db` | MCP metadata storage |
| `MCP_LOG_LEVEL` | `INFO` | MCP logging level |
| `MCP_RATE_LIMIT_ENABLED` | `true` | Enables MCP rate limiting |
| `MCP_RATE_LIMIT_RPM` | `60` | Requests/minute baseline |
| `MCP_RATE_LIMIT_BURST` | `10` | Burst capacity |
| `MCP_HTTP_MAX_BODY_BYTES` | `524288` | HTTP payload size guard |
| `MCP_VALIDATE_INPUT_SCHEMA` | `true` | Schema validation for tool arguments |
| `MCP_DISABLE_WRITE_TOOLS` | `false` | Global write-tool kill switch |
| `MCP_IDEMPOTENCY_TTL_SECONDS` | `300` | Write idempotency cache TTL |
| `MCP_WS_AUTH_REQUIRED` | `true` | Require auth on WS |
| `MCP_WS_ALLOWED_ORIGINS` | local defaults | WS origin allowlist |
| `MCP_ALLOWED_IPS` | `127.0.0.1,::1` | Allowed client IPs/CIDRs |
| `MCP_BLOCKED_IPS` | empty | Explicit deny list |

For full environment coverage, see `Docs/Operations/Env_Vars.md`.

## 7. Configure Modules

### YAML file (recommended)

Default module config path:

- `tldw_Server_API/Config_Files/mcp_modules.yaml`

Override path:

- `MCP_MODULES_CONFIG=/path/to/mcp_modules.yaml`

Example:

```yaml
modules:
  - id: media
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule
    enabled: true
    name: Media
    version: "1.0.0"
    department: media
    timeout_seconds: 30
    max_retries: 3
    max_concurrent: 16
    circuit_breaker_threshold: 3
    circuit_breaker_timeout: 30
    circuit_breaker_backoff_factor: 2.0
    circuit_breaker_max_timeout: 180
    settings:
      db_path: Databases/user_databases/1/<content-db>.db
      cache_ttl: 300
```

Replace `<content-db>.db` with your configured per-user content DB filename.

### Environment variable registration (quick dev path)

```bash
export MCP_MODULES="my_module=tldw_Server_API.app.core.MCP_unified.modules.implementations.my_module:MyModule"
```

Optional default convenience flags:

- `MCP_ENABLE_MEDIA_MODULE=true`
- `MCP_ENABLE_SANDBOX_MODULE=true`

### Module autoload safety rule

Autoloaded module classes must be under:

- `tldw_Server_API.app.core.MCP_unified.modules.implementations`

Classes outside that namespace are ignored.

## 8. Add a New Module

### Step 1: Create module file

Create a new implementation under:

- `tldw_Server_API/app/core/MCP_unified/modules/implementations/`

Use `template_module.py` as the starting pattern:

```python
from typing import Any

from ..base import BaseModule, create_tool_definition


class MyModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="my.echo",
                description="Echo message",
                parameters={
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                metadata={"category": "read"},
            )
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        if tool_name == "my.echo":
            return {"text": str(arguments.get("message", ""))}
        raise ValueError(f"Unknown tool: {tool_name}")
```

### Step 2: Register module in YAML

Add your module to `tldw_Server_API/Config_Files/mcp_modules.yaml`:

```yaml
modules:
  - id: my_module
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.my_module:MyModule
    enabled: true
    name: My Module
    department: custom
    settings: {}
```

### Step 3: Restart and verify

Restart server, then verify:

```bash
curl -H "Authorization: Bearer <token>" \
  http://127.0.0.1:8000/api/v1/mcp/modules

curl -H "Authorization: Bearer <token>" \
  http://127.0.0.1:8000/api/v1/mcp/tools
```

If module loading fails, check server logs for import path or class resolution errors.

## 9. Add Tools to an Existing Module

When extending an existing module:

1. Add tool schema in `get_tools()`
2. Handle dispatch in `execute_tool(...)`
3. Mark tool category in metadata (`read`, `ingestion`, `management`, etc.)
4. For write tools, add strong validation in `validate_tool_arguments(...)`
5. Add/adjust RBAC permission grants (`tools.execute:<tool_name>`)

Detailed workflow: `Docs/MCP/Unified/Adding_Tools.md`

## 10. Permissions and Tool Catalogs

### Tool execution permissions

Execution is controlled by per-tool permissions:

- `tools.execute:<tool_name>`
- `tools.execute:*` (wildcard)

`POST /api/v1/mcp/tools/execute` requires authenticated context and permission.

### Tool catalogs

Catalogs reduce discovery noise by grouping tools. Use filters:

- `GET /api/v1/mcp/tools?catalog=<name>`
- `GET /api/v1/mcp/tools?catalog_id=<id>`
- `GET /api/v1/mcp/tool_catalogs` (visible catalogs for current principal)

Catalog membership affects discovery, not execution rights.

Reference: `Docs/MCP/mcp_tool_catalogs.md`

## 11. Kanban Workflow Control (Safe Orchestrators)

The Kanban MCP module now exposes a workflow control plane intended for safe orchestrator loops.

### Canonical state model

- `card.workflow_status` is the canonical orchestrator state (`kanban_card_workflow_state.workflow_status_key`).
- Card list placement is a projection side effect (`auto_move_list_id`) and can be strict (`strict_projection=true`) or best-effort (`false`).
- Do not infer workflow status from list placement.

### Workflow tools

- `kanban.workflow.policy.get`
- `kanban.workflow.policy.upsert`
- `kanban.workflow.statuses.list`
- `kanban.workflow.transitions.list`
- `kanban.workflow.task.state.get`
- `kanban.workflow.task.state.patch`
- `kanban.workflow.task.claim`
- `kanban.workflow.task.release`
- `kanban.workflow.task.transition`
- `kanban.workflow.task.approval.decide`
- `kanban.workflow.task.events.list`
- `kanban.workflow.control.pause` (admin)
- `kanban.workflow.control.resume` (admin)
- `kanban.workflow.control.drain` (admin)
- `kanban.workflow.recovery.list_stale_claims`
- `kanban.workflow.recovery.force_reassign` (admin)

### Write safety contract

For orchestrator writes, use optimistic concurrency and idempotency consistently:

- Always pass `expected_version` on status mutation calls.
- Always pass a unique `idempotency_key` per intent.
- Pass `correlation_id` for traceability across multi-step runs (required on transition, approval, and force-reassign).
- Re-read state (`kanban.workflow.task.state.get`) before retries.

### Stable conflict codes

Workflow conflict responses surface machine-readable codes in `detail.code`:

- `version_conflict`
- `lease_required`
- `lease_mismatch`
- `policy_paused`
- `transition_not_allowed`
- `approval_required`
- `projection_failed`
- `idempotency_conflict`

Recommended handling:

- `version_conflict`: refresh state and retry once with new `expected_version`.
- `lease_required`: claim lease, then retry transition.
- `policy_paused`: stop writes; wait for admin resume.
- `transition_not_allowed`/`approval_required`: treat as logic/state errors, not transient retries.
- `projection_failed`: repair target list mapping or disable strict projection if appropriate.

### Recommended loop (explicit, not implicit)

There is no built-in autonomous multi-stage agent loop. The safe pattern is explicit orchestration:

1. Get state (`kanban.workflow.task.state.get`).
2. Claim lease (`kanban.workflow.task.claim`) when required by policy edge.
3. Transition (`kanban.workflow.task.transition`) with `expected_version`, `idempotency_key`, and `correlation_id`.
4. If `approval_state=awaiting_approval`, decide (`kanban.workflow.task.approval.decide`) explicitly.
5. Inspect audit trail (`kanban.workflow.task.events.list`) for deterministic recovery.
6. Release lease (`kanban.workflow.task.release`) when work completes.

### Control and recovery

- Use `pause`/`resume`/`drain` to gate orchestrator writes during incidents, migrations, or maintenance.
- Use stale-claim listing plus force-reassign to recover orphaned leases.
- Keep privileged operations behind admin-only identities.

## 12. Troubleshooting

### `503 Server not initialized` on `/health`

- Call `/api/v1/mcp/status` first
- Confirm server startup logs

### `401 Authentication required` on tool execution

- Pass `Authorization: Bearer <token>` or `X-API-KEY`
- Check token/key validity and auth mode

### `403 Permission denied` when calling a tool

- Role likely lacks `tools.execute:<tool_name>`
- Grant per-tool or wildcard permission

### Tool missing from `/tools`

- Confirm module is `enabled: true` in YAML
- Confirm class path is under allowed implementations namespace
- Restart server after config/module changes

### WebSocket closes immediately

- If close reason is auth related, verify token/key and `MCP_WS_AUTH_REQUIRED`
- If close reason is origin/IP related, review `MCP_WS_ALLOWED_ORIGINS`, `MCP_ALLOWED_IPS`, and proxy setup

## 13. Next Docs

- Architecture and internals: `Docs/MCP/Unified/Developer_Guide.md`
- Deployment and hardening: `Docs/MCP/Unified/System_Admin_Guide.md`
- Module authoring: `Docs/MCP/Unified/Modules.md`
- Native CodeGraph module: `Docs/MCP/Unified/CodeGraph.md`
- YAML module config details: `Docs/MCP/Unified/Using_Modules_YAML.md`
- External federation module: `Docs/MCP/Unified/External_Federation.md`
- Client snippets: `Docs/MCP/Unified/Client_Snippets.md`
