# ACP Governance And Audit Coverage

This note tracks the governance, authorization, and audit evidence for ACP
productionization issue
[#1476](https://github.com/rmusser01/tldw_server/issues/1476). It is a focused
companion to the
[ACP production readiness matrix](ACP_Production_Readiness.md).

## Policy Authority

ACP tool permission decisions use the MCP Hub/runtime policy stack as the
authoritative policy source:

- MCP Hub policy resolution produces the effective allowed tools, approval
  mode, approval summaries, and provenance summaries.
- `ACPRuntimePolicyService` builds the per-session runtime policy snapshot.
- ACP session creation persists the snapshot version, fingerprint, refreshed
  timestamp, policy summary, and provenance summary on the session record.
- Runner permission handling consults the runtime policy snapshot before a
  tool decision is surfaced to the client.
- Audit metadata records the policy snapshot fingerprint and decision
  provenance, not raw policy documents, MCP server configs, command arguments,
  cwd values, prompts, or environment variables.

## Route Authorization Coverage

REST endpoints use `TokenScopeGuard(..., require_if_present=True, endpoint_id=...)`
plus `get_request_user`. WebSocket endpoints use explicit API-key/JWT
authentication with write scope and session ownership checks.

| Surface | Authorization gate |
| --- | --- |
| `/api/v1/acp/health`, `/api/v1/acp/setup-guide` | `acp.health`, `acp.setup_guide` |
| `/api/v1/acp/agents`, `/api/v1/acp/agents/health` | `acp.agents.list`, `acp.agents.health` |
| `/api/v1/acp/agents/register`, `/api/v1/acp/agents/{agent_type}` | `acp.agents.register` or `acp.agents.manage` plus admin role checks |
| `/api/v1/acp/sessions/new`, prompt, cancel, close, teardown, reconcile, fork, rollback | `acp.sessions.manage` |
| `/api/v1/acp/sessions/{session_id}/updates`, detail, usage, events, diagnostics, artifacts, audit | `acp.sessions.read` plus session ownership checks |
| `/api/v1/acp/sessions/{session_id}/stream` | WebSocket auth with write scope plus `_require_session_access` |
| `/api/v1/acp/sessions/{session_id}/ssh` | WebSocket auth with write scope plus `_require_session_access` before SSH info is requested |
| `/api/v1/acp/sessions/prompt-async` and `/api/v1/acp/tasks/{task_id}` | `acp.sessions.prompt_async` and `acp.tasks.status`; task status verifies task owner metadata |
| `/api/v1/acp/runs`, `/api/v1/acp/runs/aggregate` | `acp.runs.list`, `acp.runs.aggregate`; queries are scoped to the authenticated user |
| `/api/v1/agent-orchestration/workspaces/*` | `agent_orchestration.workspaces.read` or `.manage` |
| `/api/v1/agent-orchestration/projects/*` | `agent_orchestration.projects.read` or `.manage` |
| `/api/v1/agent-orchestration/tasks/*` | `agent_orchestration.tasks.read` or `.manage` |

## Audit Events

ACP audit events are best-effort persisted to `ACP_Audit_DB` and kept in the
session hot cache for session audit views. Metadata is sanitized before it is
stored.

| Event | Purpose | Sensitive payload policy |
| --- | --- | --- |
| `agent_registered`, `agent_updated`, `agent_deregistered` | Tracks dynamic agent registry control-plane changes. | Records agent identifiers and non-secret shape only; command, args, env, and API-key names are not recorded. |
| `session_created` | Tracks ACP session creation and initial policy snapshot state. | Records agent/session identifiers, tenancy IDs, MCP server count, and policy fingerprint; cwd and MCP server config are not recorded. |
| `prompt`, `prompt_failed`, `prompt_blocked` | Tracks prompt control-plane outcomes. | Records counts and reason codes, not prompt text. |
| `permission_response` | Tracks operator approval/denial decisions. | Records request id, approval state, batch tier, policy fingerprint, and provenance metadata from pending permission state. |
| `cancel`, `close`, `teardown`, `reconcile`, `rollback` | Tracks session lifecycle control actions. | Failure messages are sanitized before persistence. |
| `orchestration_dispatch_started`, `orchestration_task_completed` | Tracks task dispatch and structured completion. | Records task/run identifiers, status, completion state, and artifact count; task descriptions and completion summaries are not recorded. |
| `orchestration_review_started`, `orchestration_review_decision` | Tracks reviewer-agent and manual review gates. | Records reviewer identity, approval state, reason code, review count, and feedback presence; raw reviewer feedback is not recorded. |
| `orchestration_task_requeued`, `orchestration_task_triaged`, `orchestration_task_finalized` | Tracks final post-review state transitions. | Records reason codes and state metadata only. |

## Remaining Caveats

- Audit persistence is best effort. The session hot cache is useful for recent
  session views, but durable closeout evidence should use the SQLite audit DB
  once flushed.
- Runtime policy snapshots intentionally expose summaries and fingerprints.
  Full resolved policy documents remain internal implementation state.
- Live downstream agent binaries may have their own local logs; #1475 should
  decide how much of that history is surfaced through run-history drill-through.
