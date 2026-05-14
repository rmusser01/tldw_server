# Agent Client Protocol (ACP) Module

This document is the authoritative contributor and operator reference for the
current ACP server + runner flow. It summarizes the implemented architecture,
route contracts, setup requirements, runtime caveats, and troubleshooting path
for the ACP module.

## Documentation Map

- Product/design record:
  [ACP_Agent_Orchestration_PRD.md](../Product/ACP_Agent_Orchestration_PRD.md)
- Operator and contributor guide: this document.
- Release readiness and evidence checklist:
  [ACP_Production_Readiness.md](ACP_Production_Readiness.md)
- Downstream-agent compatibility matrix and certification contract:
  [ACP_Compatibility_Matrix.md](ACP_Compatibility_Matrix.md)
- Downstream-agent certification checklist and smoke manifest:
  [ACP_Certification_Checklist.md](ACP_Certification_Checklist.md)
- Governance, RBAC, approval, and audit details:
  [ACP_Governance_Audit.md](ACP_Governance_Audit.md)

## Status Summary

- Server-side ACP client + endpoints are wired and available behind `/api/v1/acp/*`.
- **WebSocket endpoint** for real-time session streaming at `/api/v1/acp/sessions/{session_id}/stream`.
- **Permission UI flow** - Permission requests are sent to connected WebSocket clients for approval.
- ACP runner exists in `tools/tldw-agent` and proxies to a downstream ACP agent.
- Downstream-agent support claims are governed by
  [ACP_Compatibility_Matrix.md](ACP_Compatibility_Matrix.md). Stub-agent
  protocol coverage is not the same as live Codex, Claude Code, OpenCode, or
  custom-agent certification.
- Session lifecycle is supported: `session/new`, `session/prompt`, `session/cancel`,
  and `_tldw/session/close`.
- Downstream capabilities are reflected in `initialize`.
- Terminal tooling is allowlisted by config; file read/write is scoped to workspace.
- Tests added for the ACP runner (Go), server endpoints, and WebSocket (pytest).
- Smoke test validated via stub agent.

## Production Readiness Tracking

The ACP production readiness matrix and closeout checklist are the release gate
for the #1471 productionization epic. Keep that matrix in sync with the child
issues as work lands, and treat this document as the route/setup/troubleshooting
source that supports the matrix.

## Architecture

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   WebUI/Client  │────▶│  tldw_server    │────▶│   tldw-agent    │
│                 │◀────│  (FastAPI)      │◀────│   (Go Runner)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               │                        ▼
                               │               ┌─────────────────┐
                               │               │ Downstream Agent│
                               │               │ (Claude Code/   │
                               │               │  Codex/Custom)  │
                               │               └─────────────────┘
                               │
                        REST + WebSocket
```

## Module Layout

### Server (tldw_server2)

**Core client:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/stdio_client.py` - JSON-RPC stdio communication
- `tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py` - Session management, WebSocket registry, permission handling
- `tldw_Server_API/app/core/Agent_Client_Protocol/config.py` - Configuration loading

**API schemas:**
- `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py` - Pydantic models for REST and WebSocket messages

**API endpoints:**
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` - REST and WebSocket endpoints

### Runner (tldw-agent)

- `tools/tldw-agent/internal/acp/conn.go` - Connection management
- `tools/tldw-agent/internal/acp/runner.go` - ACP runner logic
- `tools/tldw-agent/internal/acp/terminal.go` - Terminal tool handling
- `tools/tldw-agent/internal/acp/stdio.go` - Stdio communication
- `tools/tldw-agent/internal/acp/types.go` - Type definitions
- `tools/tldw-agent/cmd/tldw-agent-acp/main.go` - Runner entrypoint

### Frontend (apps/packages/ui)

- `src/services/acp/types.ts` - TypeScript type definitions
- `src/services/acp/client.ts` - REST and WebSocket client
- `src/services/acp/constants.ts` - Tool tiers and configuration
- `src/services/acp/readiness.ts` - Shared ACP health normalization and setup gap mapping
- `src/hooks/useACPSession.tsx` - React hook for session management
- `src/store/acp-sessions.ts` - Zustand store for session state
- `src/components/Option/ACPPlayground/` - ACP Playground UI components
- `src/components/Option/AgentRegistry/` - Agent health, transport, and launch surface
- `src/components/Option/AgentTasks/` - Project/task/run/review surface with ACP setup guidance and run diagnostics

### Test Assets

- `Helper_Scripts/acp_stub_agent.py` - Stub agent for smoke testing
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py` - REST endpoint tests
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py` - WebSocket tests
- `tools/tldw-agent/internal/acp/runner_test.go` - Runner tests (Go)
- `tools/tldw-agent/internal/acp/terminal_test.go` - Terminal tests (Go)

## Endpoints

### REST Endpoints

|Endpoint|Method|Description|
|---|---|---|
|`/api/v1/acp/health`|GET|Runner, route, downstream-agent, and API-key health|
|`/api/v1/acp/setup-guide`|GET|Actionable setup guidance for runner and agents|
|`/api/v1/acp/agents`|GET|Configured ACP agent list and default agent|
|`/api/v1/acp/agents/health`|GET|Cached/on-demand monitored agent health|
|`/api/v1/acp/agents/register`|POST|Admin-only dynamic agent registration|
|`/api/v1/acp/agents/{agent_type}`|PUT/DELETE|Admin-only dynamic agent update or removal|
|`/api/v1/acp/sessions/new`|POST|Create a new ACP session|
|`/api/v1/acp/sessions/prompt`|POST|Send a prompt to a session|
|`/api/v1/acp/sessions/cancel`|POST|Cancel the current operation|
|`/api/v1/acp/sessions/close`|POST|Close and cleanup a session|
|`/api/v1/acp/sessions/{session_id}/teardown`|POST|Force teardown and reconciliation for a session|
|`/api/v1/acp/sessions/{session_id}/reconciliation`|GET|Read teardown/reconcile state|
|`/api/v1/acp/sessions/{session_id}/reconcile`|POST|Attempt server-side reconciliation for a session|
|`/api/v1/acp/sessions/{session_id}/updates`|GET|Poll for session updates|
|`/api/v1/acp/sessions/{session_id}/detail`|GET|Session metadata, usage, messages, and lineage|
|`/api/v1/acp/sessions/{session_id}/events`|GET|Persisted session event/message timeline|
|`/api/v1/acp/sessions/{session_id}/events/stream`|GET|SSE stream of persisted ACP session events|
|`/api/v1/acp/sessions/{session_id}/artifacts`|GET|Artifacts emitted in session messages|
|`/api/v1/acp/sessions/{session_id}/diagnostics`|GET|Normalized session diagnostics and reconciliation state|
|`/api/v1/acp/sessions/{session_id}/audit`|GET|Sanitized ACP audit trail for the session|
|`/api/v1/acp/sessions/{session_id}/fork`|POST|Fork a resumable ACP session from message history|
|`/api/v1/acp/sessions/prompt-async`|POST|Submit an ACP prompt to Scheduler `acp_run`|
|`/api/v1/acp/tasks/{task_id}`|GET|Poll async ACP task status/result|
|`/api/v1/acp/runs`|GET|List ACP run history|
|`/api/v1/acp/runs/aggregate`|GET|Aggregate ACP usage and cost data|
|`/api/v1/admin/acp/execution-health/summary`|GET|Admin summary of ACP session health, failure buckets, retention/redaction posture, and compatibility evidence|
|`/api/v1/acp/sessions/{session_id}/rollback`|POST|Rollback a sandbox-backed session to a checkpoint|
|`/api/v1/acp/sessions/{session_id}/checkpoints`|GET|List available session checkpoints|
|`/api/v1/agent-orchestration/workspaces`|GET/POST|List or create ACP workspaces|
|`/api/v1/agent-orchestration/workspaces/{workspace_id}`|GET/PUT/DELETE|Read, update, or delete an ACP workspace|
|`/api/v1/agent-orchestration/workspaces/{workspace_id}/health`|GET|On-demand workspace health check|
|`/api/v1/agent-orchestration/workspaces/health/refresh-all`|POST|Refresh workspace health for the current user|
|`/api/v1/agent-orchestration/workspaces/{workspace_id}/mcp-servers`|GET/POST|List or add workspace MCP servers|
|`/api/v1/agent-orchestration/workspaces/{workspace_id}/mcp-servers/{server_id}`|DELETE|Remove a workspace MCP server|
|`/api/v1/agent-orchestration/workspaces/discover`|POST|Discover candidate workspaces under an allowlisted root|
|`/api/v1/agent-orchestration/projects`|GET/POST|List or create agent projects|
|`/api/v1/agent-orchestration/projects/{project_id}`|GET/DELETE|Read or delete an agent project|
|`/api/v1/agent-orchestration/projects/{project_id}/tasks`|GET/POST|List or create project tasks|
|`/api/v1/agent-orchestration/tasks/{task_id}`|GET|Task detail with reviews and enriched run drill-through|
|`/api/v1/agent-orchestration/tasks/{task_id}/run`|POST|Dispatch a task run through ACP|
|`/api/v1/agent-orchestration/tasks/{task_id}/review`|POST|Submit manual review decision|
|`/api/v1/acp/schedules`|GET/POST|List or create recurring ACP schedules|
|`/api/v1/acp/schedules/{schedule_id}`|PUT/DELETE|Update or delete an ACP schedule|
|`/api/v1/acp/triggers`|GET/POST|List or create ACP triggers|
|`/api/v1/acp/triggers/{trigger_id}`|GET/PUT/DELETE|Read, update, or delete an ACP trigger|
|`/api/v1/acp/triggers/webhook/{trigger_id}`|POST|Inbound webhook trigger receiver|

### Orchestration Run History Drill-Through

`GET /api/v1/agent-orchestration/tasks/{task_id}` returns task detail with
`runs[]` and `reviews[]`. Run entries keep the original orchestration run fields
and add an operator-facing drill-through contract:

```json
{
  "id": 1,
  "task_id": 10,
  "session_id": "session-abc",
  "status": "completed",
  "session": {
    "session_id": "session-abc",
    "available": true,
    "status": "closed",
    "agent_type": "codex",
    "message_count": 2,
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    "links": {
      "detail": "/api/v1/acp/sessions/session-abc/detail",
      "events": "/api/v1/acp/sessions/session-abc/events",
      "events_stream": "/api/v1/acp/sessions/session-abc/events/stream",
      "artifacts": "/api/v1/acp/sessions/session-abc/artifacts",
      "diagnostics": "/api/v1/acp/sessions/session-abc/diagnostics",
      "audit": "/api/v1/acp/sessions/session-abc/audit",
      "updates": "/api/v1/acp/sessions/session-abc/updates",
      "usage": "/api/v1/acp/sessions/session-abc/usage"
    }
  },
  "history": {
    "event_count": 2,
    "audit_event_count": 1,
    "artifact_count": 1,
    "diagnostic_count": 0,
    "tool_call_count": 1,
    "stop_reason": "end",
    "prompt": {"role": "user", "preview": "Task prompt text"},
    "result": {"role": "assistant", "preview": "Done"},
    "artifacts": [{"id": "artifact-1", "type": "summary"}],
    "diagnostics": []
  },
  "failure_context": null,
  "review_decision": null
}
```

Frontend surfaces should use the `session.links` fields instead of constructing
raw ACP URLs. When the linked session record has been cleaned up or cannot be
loaded, `session.available` is `false` but links remain present for operators.
Failed runs prefer normalized session diagnostics for `failure_context`; if no
session diagnostic exists, the orchestration run error is exposed as the fallback
failure source. Reviewer runs include a `review_decision` summary when a matching
durable review row exists.

### Retention And Redaction Policy

ACP session history has two supported read modes: authenticated full-fidelity
operator drill-through for incident reconstruction, and explicit redacted views
for support-safe sharing. The current release posture is:

- Session metadata and normalized message text are persisted in
  `acp_sessions.db`; prompt and assistant raw payloads are also retained so
  session detail, event replay, artifacts, diagnostics, forking, and run-history
  inspection can reconstruct what happened.
- `ACP_SESSION_TTL_SECONDS` and `ACP_MAX_SESSION_DURATION_SECONDS` close active
  sessions through the ACP session cleanup task. `ACP_SESSION_RETENTION_DAYS`
  then hard-deletes closed/error sessions older than the retention window, with
  message rows removed by the session table cascade. Active sessions are not
  hard-deleted until duration limits have closed them.
- `GET /api/v1/acp/sessions/{session_id}/detail` and
  `GET /api/v1/acp/sessions/{session_id}/events` return full-fidelity
  authenticated session history by default, including stored message content and
  raw payload fields. Add `?redacted=true` for support-safe views that preserve
  operational shape such as roles, timestamps, event order, and normalized reason
  codes while replacing transcript content and raw payloads with `[redacted]`.
- `/artifacts` returns artifact dictionaries emitted in session messages. The
  default response is full fidelity for authorized operators. Add
  `?redacted=true` to preserve useful artifact context such as IDs, types, and
  non-sensitive metadata while scrubbing embedded content, secret-looking values,
  and local filesystem paths.
- `/diagnostics` normalizes failure reason codes and redacts diagnostic messages
  that look like API keys, bearer tokens, Slack bot tokens, or OpenAI-style
  secret keys. It also truncates long diagnostic text.
- `/audit` returns ACP audit events after audit metadata has been sanitized.
  The audit sanitizer redacts sensitive metadata keys such as prompts,
  messages, content, command arguments, `cwd`, environment values, MCP server
  definitions, API keys, and authorization tokens. It also redacts strings with
  common secret markers and truncates long string values.
- `ACP_AUDIT_RETENTION_DAYS` is enforced by ACP retention maintenance at store
  startup and by the periodic cleanup task. The maintenance pass flushes pending
  audit events before purging old audit rows.
- Workspace `env_vars` and runner environment configuration are operational
  configuration. They may be stored or forwarded as plaintext in orchestration
  metadata and process environment. Use external secret managers or host-level
  environment injection for real secrets.

Current policy classification:

| Surface | Status | Release implication |
| --- | --- | --- |
| Session detail and event history | Compliant | Owner-scoped full-fidelity drill-through is supported by default; `?redacted=true` provides support-safe transcript/event views. |
| Session artifacts | Compliant | Authorized full-fidelity artifact drill-through is supported by default; `?redacted=true` scrubs sensitive artifact payloads while preserving IDs, types, and safe metadata. |
| Diagnostics | Compliant | Failure reason codes are normalized, diagnostic text is secret-pattern redacted and truncated, and release notes may claim sanitized diagnostics. |
| Audit metadata | Compliant | Sensitive metadata keys, common secret markers, and long string values are sanitized before audit events are returned. |
| Session TTL and max-duration cleanup | Compliant | Active sessions are closed by configured duration limits; closed/error sessions older than `ACP_SESSION_RETENTION_DAYS` are hard-deleted with message cascade cleanup. |
| Automatic audit retention enforcement | Compliant | `ACP_AUDIT_RETENTION_DAYS` is enforced by ACP retention maintenance at startup and during the periodic cleanup task. |
| Workspace environment and runner env vars | Partial | Operational environment configuration can be stored or forwarded as plaintext; real secrets must come from host-level injection or an external secret manager. |
| Redacted transcript and artifact views | Compliant | Session detail, event, and artifact endpoints accept `?redacted=true` for support-safe output. |

Release notes may claim authenticated ACP session drill-through, bounded run
previews, sanitized audit metadata, sanitized diagnostics, automatic ACP
session/audit retention maintenance, and opt-in redacted session/event/artifact
views. Do not claim that the default drill-through endpoints are redacted; they
remain intentionally full fidelity for authorized operators.

### Admin Execution-Health Summary

`GET /api/v1/admin/acp/execution-health/summary` provides the backend-owned
admin reporting contract for ACP release readiness and future admin UI display.
The endpoint aggregates existing ACP session history and configured agent
metadata. It does not introduce a separate observability store.

The response includes:

- `sessions.total` and `sessions.by_status` for sessions considered in the
  requested `range_days` lookback.
- `failure_buckets.setup_blockers` for unconfigured or blocked agent setup
  state, `runner_session_failures` for errored sessions and runner/session
  failures, `reviewer_rejections` and `reviewer_failures` for review-loop
  outcomes, `governance_denials` for policy or permission-denial outcomes,
  `structured_completion_failures` for missing or invalid completion signals,
  `sandbox_runtime_errors` for sandbox or runtime launch/execution failures,
  and `retention_redaction_actions` for observed retention or redaction
  lifecycle actions.
- `setup_health` dimensions for `agent`, `workspace`, `sandbox_runtime`,
  `mcp_injection`, and `scheduler_trigger_path`. Each dimension reports a
  status, normalized blockers, and an evidence count so operators can separate
  known blockers from dimensions that are simply not observed in the summary
  window.
- `agents[]` entries with per-agent setup and compatibility posture, including
  `support_state`, `verification_level`, `setup_blocked`, and
  `primary_blocker`.
- `compatibility.by_support_state`, `documented_unverified_agents`, and
  `live_certification_required` so admin reporting can distinguish documented
  candidates from live-certified agents without overstating support.
- `retention` and `redaction` summaries that mirror the configured ACP
  retention policy and the support-safe redaction posture for details, events,
  artifacts, diagnostics, and audit metadata.

The initial contract intentionally stays summary-level. Operator drill-through
continues to use the existing session detail, events, artifacts, diagnostics,
audit, run-history, and task-detail endpoints listed above.

### Frontend Setup And Diagnostics Surfaces

Agent Tasks, Agent Registry, and ACP Playground share the same browser transport
and auth helpers so hosted and direct-backend deployments resolve ACP requests
consistently. Agent Tasks also calls `/api/v1/acp/health` and normalizes the
response through `src/services/acp/readiness.ts` before showing first-run setup
guidance. The setup banner should point operators to Agent Registry for runner
and agent configuration, and to ACP Playground for direct connection checks.

Task cards expose an inspect action backed by
`GET /api/v1/agent-orchestration/tasks/{task_id}`. The diagnostics modal should
render run status, review counts, session IDs, normalized failure context,
result previews, reviewer decisions, and the server-provided `session.links`
targets for diagnostics, artifacts, and audit history. Frontend code should not
ask users to manually copy task, run, or session IDs for normal diagnose flows.

### Schedules And Webhook Triggers

ACP schedules are stored in the shared `workflow_schedules` table with
`acp_config_json` set. The recurring scheduler owns cron registration, then
submits due ACP work to the core Scheduler as `handler="acp_run"` on the `acp`
queue. Plain workflow schedules continue to route to `workflow_run`.

Schedule operator state:

- `last_status="pending"` when a due run starts handoff to the Scheduler.
- `last_status="queued"` after the `acp_run` task is accepted.
- `last_status="error"` when Scheduler submission fails.
- `last_status="skipped_disabled"` when a stale APScheduler job fires after the
  schedule has been disabled.
- `next_run_at` is exposed in schedule responses so operators can see the next
  planned fire time.

Concurrency is explicit per ACP schedule:

| `concurrency_mode` | APScheduler behavior | Use when |
| --- | --- | --- |
| `skip` | `max_instances=1`, coalescing enabled by default | Only one run should be active and missed fires should collapse. |
| `queue` | `max_instances=3`, coalescing disabled when requested | A few overlapping runs are acceptable. |

Webhook triggers are managed through `/api/v1/acp/triggers` and inbound webhooks
arrive at `/api/v1/acp/triggers/webhook/{trigger_id}`. The inbound endpoint is
not authenticated by user session; it relies on provider-specific HMAC
verification and replay controls. Trigger secrets are encrypted at rest through
`ACP_TRIGGER_ENCRYPTION_KEY`, CRUD responses strip stored encrypted secrets, and
webhook errors are sanitized to stable public error codes.

### WebSocket Endpoint

**URL:** `WS /api/v1/acp/sessions/{session_id}/stream`

**Query Parameters:**
- `token` - JWT access token (multi-user mode)
- `api_key` - API key (single-user mode)

**Server → Client Messages:**

|Type|Description|
|---|---|
|`connected`|Connection established, includes agent capabilities|
|`update`|Real-time update from agent session|
|`permission_request`|Tool execution requires approval|
|`error`|Error occurred|
|`prompt_complete`|Prompt execution completed|

**Client → Server Messages:**

|Type|Description|
|---|---|
|`permission_response`|Approve or deny a permission request|
|`cancel`|Cancel the current operation|
|`prompt`|Send a new prompt|

### Permission Request Example

```json
{
  "type": "permission_request",
  "request_id": "uuid",
  "session_id": "session-id",
  "tool_name": "fs.write",
  "tool_arguments": {"path": "/file.txt", "content": "..."},
  "tier": "batch",
  "timeout_seconds": 300
}
```

### Permission Response Example

```json
{
  "type": "permission_response",
  "request_id": "uuid",
  "approved": true,
  "batch_approve_tier": "batch"
}
```

## Permission Tiers

Tools are classified into permission tiers based on their risk level:

| Tier | Description | Examples |
|------|-------------|----------|
| `auto` | Auto-approved (read-only) | `fs.read`, `git.status`, `search.grep` |
| `batch` | Approve multiple at once | `fs.write`, `git.commit`, `git.add` |
| `individual` | Review each one | `fs.delete`, `exec.run`, `git.push` |

### Tier Determination Heuristics

The server automatically determines the permission tier based on the tool name using pattern matching:

**Auto tier** (read-only operations - auto-approved):
- Patterns: `read`, `get`, `list`, `search`, `find`, `view`, `show`, `glob`, `grep`, `status`
- Example: `fs.readFile` → `auto` (contains "read")

**Individual tier** (destructive operations - require individual approval):
- Patterns: `delete`, `remove`, `exec`, `run`, `shell`, `bash`, `terminal`, `push`, `force`
- Example: `git.push` → `individual` (contains "push")

**Batch tier** (default for write operations):
- Any tool that doesn't match auto or individual patterns
- Example: `fs.write` → `batch` (no special pattern match)

Pattern matching is case-insensitive and checks if the tool name contains any of the patterns.

## Governance Integration

ACP now uses a shared governance coordinator for both prompt and permission flows.

Contract details:
- Prompt checks and permission checks go through `ACPGovernanceCoordinator`.
- Permission outcome is unified to one path: `approve`, `deny`, or `prompt`.
- Governance `require_approval` is merged into the same approval prompt path as tiered ACP approvals, preventing duplicate prompts.
- Governance deny decisions raise `ACPGovernanceDeniedError` with structured governance metadata.

Compatibility and migration notes:
- MCP wire compatibility is unchanged; governance metadata is additive on MCP errors.
- ACP moves toward the unified governance contract and deprecates legacy split approval behavior.
- Rollout configuration is shared via `GOVERNANCE_ROLLOUT_MODE` (`off`, `shadow`, `enforce`).

## Configuration

### Server config.txt

Enable ACP routes in `tldw_Server_API/Config_Files/config.txt`:

```ini
[API-Routes]
stable_only = true
enable = tools, jobs, acp

[ACP]
runner_command = go
runner_args = ["run", "./cmd/tldw-agent-acp"]
runner_cwd = tools/tldw-agent
runner_env = HOME=./acp_runner_home,PYTHONUNBUFFERED=1
startup_timeout_ms = 10000
```

Relative `HOME` values in `runner_env` are resolved against `tldw_Server_API/Config_Files`, so the checked-in example stays portable across machines.

### Environment Overrides

```bash
ACP_RUNNER_COMMAND=/path/to/runner
ACP_RUNNER_ARGS='["--flag","value"]'
ACP_RUNNER_ENV='HOME=/abs/path,PYTHONUNBUFFERED=1'
ACP_RUNNER_CWD=/abs/path/to/runner/dir
ACP_RUNNER_STARTUP_TIMEOUT_MS=10000
ACP_SESSION_TTL_SECONDS=86400
ACP_MAX_SESSION_DURATION_SECONDS=14400
ACP_SESSION_RETENTION_DAYS=30
ACP_AUDIT_RETENTION_DAYS=30
```

### Workspace Roots And Session Environment

Agent orchestration workspaces must live under an explicit allowlist before they
can be attached to projects or used as run working directories. Configure the
allowlist in `config.txt` or the environment:

```ini
[ACP-WORKSPACE]
allowed_base_paths = /Users/me/projects,/srv/acp-workspaces
```

```bash
ACP_WORKSPACE_ALLOWED_BASE_PATHS=/Users/me/projects:/srv/acp-workspaces
```

Workspace root validation fails closed with stable error codes:

| Code | HTTP status | Operator action |
| --- | --- | --- |
| `workspace_root_not_absolute` | `400` | Submit an absolute path. |
| `workspace_roots_not_configured` | `503` | Set `ACP-WORKSPACE.allowed_base_paths` or `ACP_WORKSPACE_ALLOWED_BASE_PATHS`. |
| `workspace_root_not_allowed` | `403` | Move the workspace under an allowed base path or update the allowlist. |

When an orchestration project is bound to a workspace, dispatch creates ACP
sessions with:

- `cwd` resolved inside the workspace root.
- enabled workspace MCP servers converted to `mcpServers`.
- workspace `env_vars` forwarded as per-session `env`.

For product-level workspace membership, ACP execution workspaces attach to the
canonical workspace model rather than defining a separate product workspace.
See `../Design/ACP_Workspace_Integration_Decision_2026_05.md` for the bridge
contract between `/api/v1/workspaces` and `/api/v1/agent-orchestration`.
Use `POST /api/v1/agent-orchestration/workspaces/canonical-bridge` to find or
create the ACP execution workspace for a canonical workspace. The request must
include the canonical workspace ID and an absolute `root_path` under the
configured ACP workspace allowlist; the response includes a `canonical_workspace`
bridge object with the canonical ID, source, link status, and ACP execution
workspace ID.

For the standard runner, per-session env is sent with `session/new`. For sandbox
mode, per-session env is merged over `[ACP-SANDBOX].agent_env` and passed to the
entrypoint as `ACP_AGENT_ENV_JSON`; only the per-session env is also included on
the sandboxed `session/new` request.

## ACP Sandbox Mode (Container/VM)

ACP sandbox mode runs `tldw-agent-acp` inside a sandbox container and exposes a web SSH proxy.

### Install ACP Dependencies

```bash
pip install -e ".[acp]"
```

### Build the ACP Image

With the in-repo runner under `tools/tldw-agent`, build from the repository root:

```bash
# From tldw_server2/
docker build -f Dockerfiles/ACP/Dockerfile \
  --build-arg TLDW_SERVER_DIR=. \
  --build-arg TLDW_AGENT_DIR=tools/tldw-agent \
  -t tldw/acp-agent:latest .
```

### Config

Enable ACP sandbox mode and set the agent command:

```ini
[ACP-SANDBOX]
enabled = true
runtime = docker
base_image = tldw/acp-agent:latest
network_policy = allow_all
agent_command = claude
agent_args = ["code"]
session_retention_days = 30
audit_retention_days = 30
```

`agent_command` must be the downstream coding agent executable (`claude`, `codex`, `opencode`, etc).
Do not set it to `tldw-agent-acp` (that recursively launches the runner and fails with `resource temporarily unavailable`).

### Required Env

```bash
ACP_SANDBOX_ENABLED=1
ACP_SANDBOX_AGENT_COMMAND=claude
SANDBOX_ENABLE_EXECUTION=1
SANDBOX_BACKGROUND_EXECUTION=1
SANDBOX_DOCKER_BIND_WORKSPACE=1
```

### Notes

- Each ACP session starts a dedicated sandbox run.
- The container exposes SSH on a host port (local only) and the UI connects via WS proxy.
- Docker, Lima, and VZ runtimes depend on host prerequisites. If the selected
  runtime is unavailable or not opted in, session creation should fail before
  launching the downstream agent with an operator-facing runtime reason code.
- Sandbox mode ignores host `cwd` and uses `/workspace` inside the runtime.
  Workspace setup guidance should make the bind/mount requirement explicit for
  the selected backend.

## Agent Configurations

The runner launches the downstream ACP agent based on:
`~/.tldw-agent/config.yaml` (or the HOME specified in runner_env)

### Compatibility Status

Configuration examples in this section document candidate downstream-agent
profiles. They are not support claims by themselves. Before release notes,
Agent Registry, or setup surfaces describe a named agent as supported, update
`ACP_Compatibility_Matrix.md` with the agent version, host/runtime profile,
support state, verification level, capability-check results, evidence command,
caveats, and follow-up issue.

Use `documented_unverified` for agents that have a documented command profile
but no current live evidence. Use `unsupported` only when the agent is proven not
to satisfy the ACP stdio/protocol contract; missing binaries, credentials,
workspace allowlists, or host runtimes are setup caveats rather than protocol
incompatibilities.

### Complete Configuration Example

```yaml
# ~/.tldw-agent/config.yaml
# Complete configuration example with all available options

# Agent configuration - defines which downstream ACP agent to launch
agent:
  # Command to execute (required)
  # Can be an absolute path or command in PATH
  command: "claude"

  # Command-line arguments (optional, default: [])
  args: ["code"]

  # Environment variables for the agent process (optional)
  # Use ${VAR_NAME} to reference existing environment variables
  env:
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
    # Add any additional env vars needed by your agent
    # SOME_CONFIG: "value"

# Workspace configuration (optional)
workspace:
  # Allowed root directories for file operations
  # File operations outside these roots will be blocked
  allowed_roots:
    - "/home/user/projects"
    - "/tmp/sandbox"

# Terminal configuration (optional)
terminal:
  # Whether terminal tools are enabled
  enabled: true

  # Allowlist of permitted command patterns
  # Commands not matching any pattern are blocked
  allowed_commands:
    - "git *"           # Allow all git commands
    - "npm *"           # Allow npm commands
    - "python *.py"     # Allow running Python scripts
    - "ls *"            # Allow listing directories
    - "cat *"           # Allow reading files

# Logging configuration (optional)
logging:
  # Log level: debug, info, warn, error
  level: "info"

  # Log file path (optional, logs to stderr if not set)
  # file: "/var/log/tldw-agent/agent.log"
```

### Claude Code

```yaml
agent:
  command: "claude"
  args: ["code"]
  env:
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
```

### Codex CLI

```yaml
agent:
  command: "codex"
  args: []
  env:
    OPENAI_API_KEY: "${OPENAI_API_KEY}"
```

### Custom ACP Agent

```yaml
agent:
  command: "/path/to/custom-agent"
  args: ["--stdio"]
  env: {}
```

## Frontend Integration

### Using the useACPSession Hook

```typescript
import { useACPSession } from "@/hooks/useACPSession"

const {
  state,              // Session state
  isConnected,        // Whether connected
  updates,            // List of updates
  pendingPermissions, // Pending permission requests
  connect,            // Connect to session
  disconnect,         // Disconnect from session
  sendPrompt,         // Send a prompt
  approvePermission,  // Approve a permission
  denyPermission,     // Deny a permission
  cancel,             // Cancel current operation
} = useACPSession({
  sessionId: "session-id",
  autoConnect: true,
  onUpdate: (update) => console.log(update),
  onPermissionRequest: (request) => console.log(request),
})
```

### Using the Zustand Store

```typescript
import { useACPSessionsStore } from "@/store/acp-sessions"

const sessions = useACPSessionsStore((s) => s.getSessions())
const activeSession = useACPSessionsStore((s) =>
  s.activeSessionId ? s.getSession(s.activeSessionId) : undefined
)
const createSession = useACPSessionsStore((s) => s.createSession)
```

## Testing

### Server Endpoint Tests

```bash
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -v
```

### WebSocket Tests

```bash
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py -v
```

### Runner Tests (Go)

```bash
cd tools/tldw-agent
./scripts/verify-local-build.sh
```

### Certification Smoke Manifest

Use the certification helper when updating downstream-agent compatibility
claims. The `stub-smoke` profile reuses the in-repo backend, runner, and mocked
browser gates; the `live-e2e` profile documents the operator-supplied runtime
state needed before claiming support for a named downstream agent.

```bash
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile stub-smoke --format json
python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --format json
```

Follow [ACP_Certification_Checklist.md](ACP_Certification_Checklist.md) before
changing support states in
[ACP_Compatibility_Matrix.md](ACP_Compatibility_Matrix.md).

## Behavior Summary

### Server ACP Client

- Spawns the ACP runner via stdio and JSON-RPC line framing.
- Maintains a per-process client and queues session updates.
- **WebSocket Registry**: Tracks connected WebSocket clients per session.
- **Permission Flow**: Permission requests are broadcast to connected WebSocket clients.
  - If no WebSocket is connected, permissions are auto-cancelled after 5 minutes.
  - Auto-approve for `auto` tier tools.
  - Batch approval option for `batch` tier tools.
- Supports env/config overrides for runner command/args/env/cwd.

### ACP Runner

- One downstream process per ACP session.
- Validates workspace roots and keeps file ops inside workspace.
- Handles allowlisted `terminal/*` tools with command templates and argument guards.
- Forwards `session/request_permission` upstream.
- Caches downstream capabilities and reflects them in `initialize`.

## Known Constraints

- ACP routes are gated by `stable_only` unless explicitly enabled.
- ACP runner uses stdio; ensure the runner executable is available in PATH or
  configured explicitly.
- Permission timeout is 5 minutes; requests are auto-cancelled if not responded to.
- WebSocket reconnection uses exponential backoff (max 10 attempts).

## Troubleshooting

### Connection Issues

1. **Check ACP is enabled**: Verify `enable = acp` in config.txt `[API-Routes]` section.
2. **Check runner path**: Ensure `runner_command` points to a valid executable.
3. **Check authentication**: Verify JWT token or API key is valid.

### Permission Requests Not Appearing

1. **WebSocket connected**: Ensure the WebSocket connection is established before sending prompts.
2. **Tool tier**: `auto` tier tools are auto-approved and won't trigger permission requests.

### Agent Not Responding

1. **Check agent config**: Verify `~/.tldw-agent/config.yaml` has correct agent configuration.
2. **Check API keys**: Ensure ANTHROPIC_API_KEY or OPENAI_API_KEY is set for the agent.
3. **Check logs**: Review server logs for ACP-related errors.
