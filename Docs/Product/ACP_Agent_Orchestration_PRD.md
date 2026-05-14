# ACP Agent Orchestration and Third-Party Agent Support PRD

Author: tldw_server team
Status: Current implementation record, refreshed from draft v0.1 on 2026-05-10
Parent epic: [#1471](https://github.com/rmusser01/tldw_server/issues/1471)
Operational doc: [Agent_Client_Protocol.md](../Development/Agent_Client_Protocol.md)
Release checklist: [ACP_Production_Readiness.md](../Development/ACP_Production_Readiness.md)

## 1) Summary

ACP support now covers the core product loop proposed by the original draft:
configured agents are discoverable, users can create projects and tasks,
dispatch task runs through ACP sessions, enforce structured completion signals,
optionally run reviewer agents, and inspect run history without reading server
logs.

The productionization track moved the plan from a pi-agent-specific proposal to
a configurable ACP agent platform. The server owns orchestration state,
governance, audit, scheduling, workspace constraints, and frontend setup and
diagnostic surfaces. The runner remains an external process boundary and can
launch configured downstream agents such as Codex, Claude Code, OpenCode, or a
custom ACP-compatible agent.

## 2) Documentation Authority

This PRD is the product and design record. It describes what the ACP feature is
for, which parts are shipped, which parts are partial, and which older draft
claims are superseded.

[Agent_Client_Protocol.md](../Development/Agent_Client_Protocol.md) is the
authoritative contributor and operator guide for setup, endpoint behavior,
configuration, governance, sandbox/workspace operation, scheduling, frontend
integration, and troubleshooting.

[ACP_Production_Readiness.md](../Development/ACP_Production_Readiness.md) is
the release-readiness checklist. It tracks the #1471 child issues, verification
commands, release gates, runtime caveats, and final closeout evidence.

[ACP_Governance_Audit.md](../Development/ACP_Governance_Audit.md) summarizes
the current governance, RBAC, approval, and audit model.

[ACP_Workspace_Integration_Decision_2026_05.md](../Design/ACP_Workspace_Integration_Decision_2026_05.md)
records the maturity decision for connecting ACP projects, tasks, runs,
reviews, and diagnostics to the canonical workspace model without creating a
parallel ACP-only workspace product.

## 3) Current Implementation Status

| Area | Status | Notes |
| --- | --- | --- |
| ACP session lifecycle | Shipped | REST, WebSocket, SSE/event, prompt, cancel, close, teardown, reconcile, fork, rollback/checkpoint, run-history, diagnostics, artifacts, audit, and async prompt routes exist under `/api/v1/acp/*`. |
| Agent registry and setup | Shipped | `/api/v1/acp/agents`, `/api/v1/acp/agents/health`, `/api/v1/acp/health`, and `/api/v1/acp/setup-guide` expose configured agents, runner health, setup gaps, and API-key status. |
| Project/task orchestration | Shipped | `/api/v1/agent-orchestration/*` owns workspaces, projects, tasks, dependencies, dispatch, reviews, task detail, and enriched run history. |
| Structured completion signals | Shipped by [#1479](https://github.com/rmusser01/tldw_server/issues/1479) | Runs validate machine-readable completion state before review/finalization and persist failure reason codes. |
| Reviewer-agent loop | Shipped by [#1478](https://github.com/rmusser01/tldw_server/issues/1478) | Reviewer runs are durable, decisions are audited, rejections can retry, and repeated rejection moves tasks to triage. |
| Run history and drill-through | Shipped by [#1475](https://github.com/rmusser01/tldw_server/issues/1475) | Task detail includes runs, reviews, session links, history counts, failure context, prompt/result previews, and reviewer decisions. |
| Governance, RBAC, and audit | Shipped by [#1476](https://github.com/rmusser01/tldw_server/issues/1476) | ACP control surfaces use token scope guards where applicable, prompt/permission flows use shared governance coordination, and audit events are sanitized. |
| Workspace and sandbox readiness | Shipped with runtime caveats by [#1477](https://github.com/rmusser01/tldw_server/issues/1477) | Workspace roots fail closed; workspace MCP servers and env flow into sessions; sandbox mode merges configured and per-session env. Docker/Lima/VZ runtime verification remains host-specific. |
| Schedules and triggers | Shipped by [#1474](https://github.com/rmusser01/tldw_server/issues/1474) | ACP schedules route through APScheduler to the core Scheduler `acp_run` handler; triggers sanitize secrets and expose operator state. |
| Admin execution-health reporting | Backend contract added under [#1537](https://github.com/rmusser01/tldw_server/issues/1537) | `/api/v1/admin/acp/execution-health/summary` summarizes ACP sessions, failure buckets, setup blockers, retention/redaction posture, and downstream-agent compatibility evidence for admin display. |
| Frontend setup/run/diagnose UX | Shipped by [#1473](https://github.com/rmusser01/tldw_server/issues/1473) | Agent Tasks, Agent Registry, and ACP Playground share connection/auth handling; Agent Tasks shows setup gaps and task diagnostics without manual ID copying. |
| Production readiness closeout | Remaining under [#1472](https://github.com/rmusser01/tldw_server/issues/1472) | Final release signoff still needs the readiness matrix closeout, broader live-backend E2E, Go runner verification, and accepted runtime caveats. |

## 4) Goals

- Let users configure and discover ACP agents without changing the API surface
  per agent.
- Let users create projects and tasks, express dependencies and success
  criteria, and dispatch runs to a selected ACP agent.
- Require structured completion signals before a run can be treated as complete.
- Support reviewer-agent or manual review gates with retry and triage behavior.
- Preserve useful run history, artifacts, diagnostics, and audit records.
- Keep privileged tools, prompts, approvals, workspaces, and sandbox execution
  governed by server-side policy and authenticated route checks.
- Give first-time users a clear setup path and regular users a task
  run/review/diagnose workflow in the WebUI.

## 5) Non-Goals And Superseded Draft Claims

- The original draft route names are superseded. Use
  `/api/v1/agent-orchestration/*` instead of proposed top-level
  `/api/v1/agent_projects`, `/api/v1/agent_tasks`, and `/api/v1/agent_agents`.
- The first-party pi-agent-only framing is superseded by the configurable ACP
  agent registry. A first-party agent may still be configured, but it is not the
  only supported execution model.
- A full Kanban board remains out of scope for this productionization slice.
  Agent Tasks provides project/task/run/review operations, not a full board.
- ACP does not replace MCP, Jobs, Scheduler, or Workflows. ACP uses the core
  Scheduler for async/background `acp_run` work and can use workspace MCP server
  configuration when launching sessions.
- Registry-based agent setup is config/admin owned. This is not a public agent
  marketplace or installer flow.

## 6) Personas

- Admin or power user: configures agents, runner settings, workspaces, sandbox
  behavior, schedules, triggers, governance, and API keys.
- Analyst or researcher: creates projects and tasks, dispatches agent runs,
  reviews outcomes, and follows diagnostic links when something fails.
- Developer or operator: integrates external ACP agents, validates runner
  behavior, troubleshoots route/config/auth issues, and reviews audit history.

## 7) User Stories

1. As an admin, I can inspect ACP health and setup gaps before dispatching work.
2. As a user, I can create a project with tasks and dependencies and run a task
   with a configured ACP agent.
3. As a user, I can require a reviewer agent or submit a manual review before a
   task becomes complete.
4. As a user, I can inspect a failed run from Agent Tasks and open the linked
   session diagnostics, artifacts, and audit history.
5. As an operator, I can schedule recurring ACP runs or trigger them through a
   webhook while preserving owner, status, and failure visibility.
6. As a developer, I can configure a custom ACP-compatible agent and verify it
   through the registry and ACP Playground.

## 8) Stable Route Contract

All paths below are mounted below `/api/v1`.

### ACP Core

- `GET /acp/health`
- `GET /acp/setup-guide`
- `GET /acp/agents`
- `GET /acp/agents/health`
- `POST /acp/agents/register`
- `PUT /acp/agents/{agent_type}`
- `DELETE /acp/agents/{agent_type}`
- `POST /acp/sessions/new`
- `POST /acp/sessions/prompt`
- `POST /acp/sessions/cancel`
- `POST /acp/sessions/close`
- `POST /acp/sessions/{session_id}/teardown`
- `GET /acp/sessions/{session_id}/reconciliation`
- `POST /acp/sessions/{session_id}/reconcile`
- `GET /acp/sessions/{session_id}/updates`
- `GET /acp/sessions/{session_id}/detail`
- `GET /acp/sessions/{session_id}/events`
- `GET /acp/sessions/{session_id}/events/stream`
- `GET /acp/sessions/{session_id}/artifacts`
- `GET /acp/sessions/{session_id}/diagnostics`
- `GET /acp/sessions/{session_id}/audit`
- `POST /acp/sessions/{session_id}/fork`
- `POST /acp/sessions/{session_id}/prompt-async`
- `GET /acp/tasks/{task_id}`
- `GET /acp/runs`
- `GET /acp/runs/aggregate`
- `POST /acp/sessions/{session_id}/rollback`
- `GET /acp/sessions/{session_id}/checkpoints`
- `WS /acp/sessions/{session_id}/stream`
- `WS /acp/sessions/{session_id}/ssh`

### Admin Reporting

- `GET /admin/acp/execution-health/summary`

### Agent Orchestration

- `POST /agent-orchestration/workspaces`
- `GET /agent-orchestration/workspaces`
- `GET /agent-orchestration/workspaces/{workspace_id}`
- `PUT /agent-orchestration/workspaces/{workspace_id}`
- `DELETE /agent-orchestration/workspaces/{workspace_id}`
- `GET /agent-orchestration/workspaces/{workspace_id}/health`
- `POST /agent-orchestration/workspaces/health/refresh-all`
- `GET /agent-orchestration/workspaces/{workspace_id}/mcp-servers`
- `POST /agent-orchestration/workspaces/{workspace_id}/mcp-servers`
- `DELETE /agent-orchestration/workspaces/{workspace_id}/mcp-servers/{server_id}`
- `POST /agent-orchestration/workspaces/discover`
- `POST /agent-orchestration/projects`
- `GET /agent-orchestration/projects`
- `GET /agent-orchestration/projects/{project_id}`
- `DELETE /agent-orchestration/projects/{project_id}`
- `POST /agent-orchestration/projects/{project_id}/tasks`
- `GET /agent-orchestration/projects/{project_id}/tasks`
- `GET /agent-orchestration/tasks/{task_id}`
- `POST /agent-orchestration/tasks/{task_id}/run`
- `POST /agent-orchestration/tasks/{task_id}/review`

### Schedules And Triggers

- `POST /acp/schedules`
- `GET /acp/schedules`
- `PUT /acp/schedules/{schedule_id}`
- `DELETE /acp/schedules/{schedule_id}`
- `POST /acp/triggers`
- `GET /acp/triggers`
- `GET /acp/triggers/{trigger_id}`
- `PUT /acp/triggers/{trigger_id}`
- `DELETE /acp/triggers/{trigger_id}`
- `POST /acp/triggers/webhook/{trigger_id}`

## 9) Architecture

### Server

The FastAPI backend owns route authorization, task state, run history,
governance, audit, workspace validation, schedules, triggers, and session
metadata. Agent orchestration is intentionally server-owned so UI and API
clients consume the same durable state.

### Runner And Agents

The server talks to an ACP runner client over the configured runner boundary.
The standard runner launches downstream ACP-compatible agents and proxies
session lifecycle and prompt calls. Sandbox mode runs the ACP runner inside a
configured container/VM backend and exposes additional SSH/checkpoint behavior
where supported.

### Storage

Current state is stored in dedicated ACP/session/orchestration tables and
supporting scheduler/trigger storage rather than the originally proposed
single minimal table sketch. Task detail enriches orchestration run rows with
ACP session records when available.

### Background Work

Async prompts and recurring schedules enqueue `handler="acp_run"` on the core
Scheduler `acp` queue. Use Jobs for future user-facing/admin queue systems that
need pause/resume/drain semantics beyond recurring cron registration.

### Frontend

The WebUI uses shared ACP connection/auth helpers for Agent Tasks, Agent
Registry, and ACP Playground. Agent Tasks is the project/task/run/review surface;
Agent Registry is the setup and health surface; ACP Playground is the direct
session experimentation and diagnostics surface.

## 10) Data And State Model

Projects contain tasks. Tasks can reference dependencies, success criteria,
selected agent type, reviewer agent type, maximum review attempts, metadata,
run history, and review rows. Runs store selected agent, status, session ID,
timing, error/result summaries, token usage, and completion signal metadata.

Task status uses the production workflow:

1. `todo`: ready to run after dependencies complete.
2. `in_progress`: agent run active or awaiting retry.
3. `review`: primary agent signaled structured completion and review is needed.
4. `complete`: reviewer approved or no reviewer was required.
5. `triage`: fatal error or reviewer rejection after max attempts.

## 11) Governance And Security

- ACP control routes use AuthNZ dependencies and route-specific scope guards
  where applicable.
- Agent registration, update, and removal require admin privileges.
- Prompt and permission flows use shared ACP governance coordination.
- Permission, prompt, review, task, session, schedule, trigger, and diagnostic
  operations record sanitized audit events where applicable.
- Workspace roots are allowlisted. Missing workspace configuration is a
  fail-closed setup error, not implicit permission to run anywhere.
- Webhook trigger secrets are encrypted at rest and sanitized in responses.
- Per-session workspace env vars are operational configuration, not a secret
  vault. Prefer external secret managers for durable secrets.

## 12) Metrics And Observability

The current implementation exposes enough state to derive:

- task completion, rejection, retry, and triage rates
- average runs and reviews per task
- run duration and failure reasons
- token usage and run cost aggregates from ACP run history
- ACP execution-health buckets for setup blockers, runner/session failures,
  reviewer outcomes, governance denials, retention/redaction posture, and
  documented-unverified compatibility
- schedule queued, skipped, disabled, and error states
- audit event volume by action and session

## 13) Remaining Work Before Production Signoff

These items remain under the #1471/#1472 closeout rather than this PRD refresh:

- Run the final live-backend ACP browser E2E against a seeded local backend and
  API key, or document accepted release skips.
- Run the Go runner build/test gate in `tools/tldw-agent`.
- Decide and document artifact retention and transcript redaction policy before
  release notes claim production retention behavior.
- Verify any sandbox backend selected as default on the target host runtime.
- Keep `ACP_Production_Readiness.md` current with final evidence for each child
  issue and close #1472 only after the matrix is fully resolved.

## 14) Verification References

Use the readiness matrix command catalog for current verification commands. At
minimum, a release candidate needs:

- focused backend ACP protocol and orchestration pytest suites
- focused frontend ACP Vitest suite
- browser E2E for setup/run/diagnose flows
- Go runner build and test verification
- Bandit on touched backend Python paths
- `git diff --check`

## 15) Change History

- Draft v0.1: original agent orchestration and pi-agent proposal.
- 2026-05-10 refresh: aligned the PRD with the current ACP productionization
  implementation, marked superseded route and pi-agent assumptions, linked the
  #1471 child issue map, and moved operator detail to the operational docs and
  readiness matrix.
