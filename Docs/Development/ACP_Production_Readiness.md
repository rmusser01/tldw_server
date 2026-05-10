# ACP Production Readiness Matrix

This document is the release-readiness checklist for ACP productionization. It
is seeded from GitHub issue
[#1472](https://github.com/rmusser01/tldw_server/issues/1472) under epic
[#1471](https://github.com/rmusser01/tldw_server/issues/1471).

## Status

This matrix has been refreshed after the #1471 productionization child issues
landed. It is the current ACP readiness control document, with final closeout
evidence recorded below. Remaining live-backend and host-runtime caveats are
called out explicitly and should be resolved before release notes claim fully
verified production deployment on a specific host.

## Issue Map

| Issue | Workstream | Readiness role |
| --- | --- | --- |
| [#1472](https://github.com/rmusser01/tldw_server/issues/1472) | Production readiness matrix and release checklist | Seed this document first; close it last after all rows have current evidence. |
| [#1479](https://github.com/rmusser01/tldw_server/issues/1479) | Structured completion signals | Proves orchestration runs can finish, fail, and time out with durable machine-readable state. |
| [#1478](https://github.com/rmusser01/tldw_server/issues/1478) | Reviewer-agent loop and triage history | Proves multi-agent review decisions are tracked and auditable. |
| [#1476](https://github.com/rmusser01/tldw_server/issues/1476) | Governance, permissions, RBAC, and audit coverage | Proves privileged actions are policy-controlled and recorded. |
| [#1475](https://github.com/rmusser01/tldw_server/issues/1475) | Run history, artifacts, and session drill-through | Proves operators can inspect what happened after a run completes. |
| [#1477](https://github.com/rmusser01/tldw_server/issues/1477) | Sandbox and workspace production readiness | Proves workspace roots, runtime isolation, and optional sandbox backends are safe to operate. |
| [#1474](https://github.com/rmusser01/tldw_server/issues/1474) | Schedules, triggers, and background runs | Proves unattended runs are controlled, observable, and recoverable. |
| [#1473](https://github.com/rmusser01/tldw_server/issues/1473) | Agent Tasks, Playground, and Registry UX | Proves the production UI supports setup, execution, review, and troubleshooting. |
| [#1480](https://github.com/rmusser01/tldw_server/issues/1480) | PRD and operational docs refresh | Brings design, operator, and user-facing docs in sync with the verified implementation. |

Recommended order: seed #1472, then complete #1479, #1478, #1476, #1475,
#1477, #1474, #1473, #1480, and finally close #1472.

## Readiness Matrix

| Surface | Owner modules | Required evidence | Verification commands | Pass/fail gate | Runtime caveats |
| --- | --- | --- | --- | --- | --- |
| ACP REST, WebSocket, SSE, and session lifecycle | `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`, `tldw_Server_API/app/core/Agent_Client_Protocol/` | Session create, prompt, cancel, close, reconnect, streaming, and error paths are covered. | `source .venv/bin/activate`; `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol -q` | Pass when focused ACP pytest suite is green and failures distinguish user errors from server faults. | Stub-agent tests are sufficient for base protocol; live downstream agents should be verified before release notes claim support. |
| Agent orchestration tasks | `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`, `tldw_Server_API/app/core/Agent_Orchestration/`, orchestration DB helpers | Tasks persist status, attempts, outputs, completion reason, and retry/timeout state. | `source .venv/bin/activate`; `python -m pytest tldw_Server_API/tests/Agent_Orchestration -q` | Pass when task state transitions are durable and completion signals satisfy #1479. | Background-worker tests may need runtime feature flags if a local worker pool is not enabled by default. |
| Structured completion and failure semantics | ACP schemas, orchestration service, run handlers, session store | Every run records terminal state, reason, timestamps, and retriable/non-retriable classification. | Focus the new/changed tests from #1479 plus `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_run_handler.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_service.py -q` | Pass when UI and API can consume the same terminal state without text parsing. | This is a blocker for reviewer loops, schedules, and production run history. |
| Reviewer-agent loop and triage history | Orchestration service, task APIs, run/history DB tables | Review requests, reviewer responses, decisions, follow-up tasks, and overrides are durable. | Add #1478 focused pytest coverage, then include it in the orchestration suite. | Pass when a reviewer decision can be traced from task request to final run history. | Do not count manual GitHub comments as triage history unless they are linked from durable ACP state. |
| Governance, permissions, RBAC, and audit | ACP permission helpers, governance coordinator, audit DB, AuthNZ dependencies, `Docs/Development/ACP_Governance_Audit.md` | Privileged tool requests are policy-checked, approval-gated where required, and audit logged. | `source .venv/bin/activate`; `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_governance_coordinator.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_permissions_helpers.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_controls.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q` | Pass when deny, approve, timeout, and escalation paths have audit evidence and RBAC coverage from #1476. | Single-user API-key mode has route/audit coverage for ACP control surfaces; multi-user JWT/API-key mode keeps scope checks through `TokenScopeGuard` and WebSocket required scopes. |
| Sandbox and workspace isolation | Runtime policy service, sandbox runner client, workspace APIs, config validation | Workspace roots fail with stable actionable error payloads; workspace MCP servers and `env_vars` are passed into orchestration ACP sessions; standard runner sessions send per-session `env`; sandbox sessions merge configured agent env with per-session env through `ACP_AGENT_ENV_JSON`. | `source .venv/bin/activate`; `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_runtime_policy_service.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py tldw_Server_API/tests/Agent_Orchestration/test_workspace_api_helpers.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q` | Pass when local, disabled-sandbox, and configured-sandbox paths fail closed and #1477 documents host requirements. UI rendering of unavailable workspace/sandbox states is tracked separately by #1473. | Docker/Lima/VZ backends are optional unless configured as defaults. Missing host runtime is a documented skip, not a production pass; runtime-specific bind/mount setup must be covered before claiming live sandbox support. |
| Schedules, triggers, and unattended background runs | ACP schedules/triggers endpoints, `workflows_scheduler.py`, ACP trigger manager, Scheduler `acp_run` handler | ACP schedules route to `acp_run` while workflow schedules continue to route to `workflow_run`; owner IDs are preserved; schedule responses expose `last_status`, `next_run_at`, and concurrency controls; disabled stale jobs record `skipped_disabled`; submit failures record `error`; webhook trigger secrets remain encrypted and sanitized. | `source .venv/bin/activate`; `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_schedules.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_triggers_endpoint.py tldw_Server_API/tests/Agent_Client_Protocol/test_webhook_triggers.py -q` | Pass when #1474 proves unattended execution has owner, queue/skip concurrency, skipped/error visibility, and trigger security coverage. | Current recurring ACP use cases are Scheduler-owned through APScheduler -> Scheduler `acp_run`. Use Jobs only for future user-visible/admin-controlled queues that need pause/resume/drain semantics beyond cron registration. |
| Run history, artifacts, and session drill-through | Session store, artifact APIs, `GET /api/v1/agent-orchestration/tasks/{task_id}`, WebUI session detail surfaces | Operators can open a task run and follow structured links to ACP session detail, events, artifacts, diagnostics, audit, updates, and usage. Task detail run entries expose prompt/result previews, stop reason, tool-call/artifact/diagnostic/audit counts, failure context, and reviewer decision summaries where available. | Backend tests from #1475 plus Stage 1 retention/redaction tests and UI tests for ACP Playground, Agent Tasks, and Agent Registry. | Pass when every production run has a useful post-run audit trail without reading server logs or constructing raw session URLs in frontend code, and support-safe redacted views exist for detail/events/artifacts. | Session detail/events/artifacts are authenticated full-fidelity drill-through surfaces by default and support `?redacted=true` for support-safe views. Diagnostics and audit metadata are sanitized. `ACP_SESSION_RETENTION_DAYS` and `ACP_AUDIT_RETENTION_DAYS` are enforced by ACP retention maintenance. Workspace environment metadata can still be operational plaintext and should not be treated as a secret store. |
| Frontend ACP UX | `apps/packages/ui/src/components/Option/ACPPlayground/`, `AgentTasks/`, `AgentRegistry/`, ACP client/store | Agent Tasks now uses shared ACP auth/transport helpers, shows actionable ACP health/setup gaps, links first-time users to Registry/Playground, and consumes enriched task detail runs with session diagnostics/artifact/audit links. Registry health normalization shares the same ACP readiness helper. | `cd apps/packages/ui`; `./node_modules/.bin/vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx --maxWorkers=1 --no-file-parallelism` | Pass when #1473 proves first-run setup, shared auth, task execution/review visibility, failed run diagnostics, and cross-surface navigation. | This slice covers focused contracts and the Agent Tasks diagnose path. Broader denied-permission/reconnect recovery remains covered by existing ACP Playground component tests and should be included in final release E2E. |
| Browser E2E ACP flows | `apps/tldw-frontend/e2e/workflows/tier-3-automation/` | ACP Playground, Agent Registry, and Agent Tasks work in the real app shell; Agent Tasks has a mocked setup/run/diagnose journey that does not require a live downstream agent. | `cd apps/tldw-frontend`; `TLDW_WEB_URL=http://localhost:18080 TLDW_WEB_CMD='bun run dev -- -p 18080' ./node_modules/.bin/playwright test e2e/workflows/tier-3-automation/agent-tasks.spec.ts --grep "guide ACP setup" --reporter=line` plus the full live-backend ACP tier-3 command before release. | Pass when E2E covers setup, start run, observe status, inspect run/session, and recover from failure. | #1501 expands deterministic denial/reconnect/recovery evidence across backend and frontend tests. Full live browser E2E for a real downstream-agent permission denial still requires a seeded backend, API key, installed ACP-compatible downstream agent, and provider credentials. |
| Go ACP runner | `tools/tldw-agent/internal/acp/`, `tools/tldw-agent/cmd/tldw-agent-acp/`, `tools/tldw-agent/cmd/tldw-agent-host/` | Runner builds, protocol tests pass, and host command wiring is verified. | `cd tools/tldw-agent`; `./scripts/verify-local-build.sh` | Pass when both runner binaries build and `go test ./...` is green. | Downstream agent binaries and API keys are external prerequisites for live-agent verification. |
| Security validation | Touched ACP backend paths, runner command execution paths, frontend E2E harness only when modified | No new security findings in touched code; high-risk command and path behavior reviewed. | `source .venv/bin/activate`; `python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_acp_<task>.json` | Pass when Bandit is clean or findings are documented as pre-existing/false positives with rationale. | Bandit is Python-only. Do not treat TypeScript parse errors as security coverage. |
| Documentation and operator guidance | `Docs/Product/ACP_Agent_Orchestration_PRD.md`, `Docs/Development/Agent_Client_Protocol.md`, `Docs/Development/ACP_Production_Readiness.md`, release notes | #1480 refresh makes the PRD a current product/design record, marks shipped/partial/superseded/remaining scope, records stable ACP route families, and makes `Agent_Client_Protocol.md` the authoritative contributor/operator guide. | `git diff --check`; `rg -n "agent_projects|agent_tasks|agent_agents|pi-agent" Docs/Product/ACP_Agent_Orchestration_PRD.md Docs/Development/Agent_Client_Protocol.md Docs/Development/ACP_Production_Readiness.md` followed by targeted read review of matches. | Pass when #1480 aligns PRD, development docs, operator steps, route inventory, child-issue links, and release checklist with actual implementation. | Docs are release blockers if they describe draft behavior as shipped behavior or omit required operator caveats. This row is docs-only; Bandit is not applicable unless backend Python changes in the same slice. |

## Command Catalog

Use the narrow command for a child issue while developing it, then run the wider
gate before marking the row complete.

### Backend Protocol And Orchestration

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol -q
python -m pytest tldw_Server_API/tests/Agent_Orchestration -q
```

### Focused ACP Smoke

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_e2e_smoke.py \
  tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py \
  -q
```

### Frontend Unit Contracts

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx \
  src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx \
  src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

### Browser E2E

```bash
cd apps/tldw-frontend
TLDW_WEB_URL=http://localhost:18080 \
TLDW_WEB_CMD='bun run dev -- -p 18080' \
./node_modules/.bin/playwright test \
  e2e/workflows/tier-3-automation/agent-tasks.spec.ts \
  --grep "guide ACP setup" \
  --reporter=line

# Live backend closeout gate:
TLDW_E2E_SERVER_URL=127.0.0.1:8000 \
TLDW_E2E_API_KEY=<local-api-key> \
bunx playwright test \
  e2e/workflows/tier-3-automation/acp-playground.spec.ts \
  e2e/workflows/tier-3-automation/agent-registry.spec.ts \
  e2e/workflows/tier-3-automation/agent-tasks.spec.ts \
  --reporter=line
```

### Go Runner

```bash
cd tools/tldw-agent
./scripts/verify-local-build.sh
```

### Security

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_acp_<task>.json
```

## Optional Runtime Caveats

- Docker sandbox verification requires the ACP sandbox dependencies and a local
  Docker runtime. If Docker is unavailable, mark the Docker row blocked or
  skipped with host evidence instead of marking it passed.
- Lima and Apple Virtualization Framework coverage is host-specific. These can
  be optional rows unless one of them is selected as the default production
  runtime.
- Workspace creation and dispatch require `ACP-WORKSPACE.allowed_base_paths` or
  `ACP_WORKSPACE_ALLOWED_BASE_PATHS`. A missing allowlist is a configuration
  failure, not an implicit permission to run anywhere on the host.
- Workspace `env_vars` are stored as plaintext orchestration metadata and passed
  to ACP session creation. Treat them as operational configuration, not a secret
  vault; prefer external secret managers where available.
- Live downstream agents need their own ACP stdio-compatible entrypoints, API
  keys, and workspace permissions. Stub-agent protocol tests are not a
  substitute for live-agent verification before release notes claim downstream
  agent support.
- Until #1504 records a real downstream-agent create/prompt/cancel run, release
  notes must describe ACP downstream-agent support as protocol/runner
  validation only. Binary presence for tools such as Claude Code or Codex is not
  enough unless the configured command is verified to speak ACP stdio.
- ACP session detail, event, and artifact endpoints are owner-scoped
  full-fidelity drill-through surfaces by default. Use `?redacted=true` on those
  endpoints for support-safe output that scrubs transcript content, raw payloads,
  secret-looking values, and local filesystem paths while preserving operational
  context such as roles, timestamps, reason codes, and artifact IDs.
- Browser E2E requires a running backend, deterministic auth state, and a known
  API key. If a child issue only changes backend internals, browser E2E can be
  deferred until the UX and closeout rows.

## Evidence Log Template

Use this format in #1472 comments and in child issue closeout notes.

```text
Surface:
Issue:
Commit/branch:
Verification:
Result:
Caveats:
Follow-up:
```

## Final Closeout Evidence

Recorded on 2026-05-10 from branch `codex/acp-productionization-1472-1479`.

Final GitHub evidence:

- #1472 closeout evidence:
  https://github.com/rmusser01/tldw_server/issues/1472#issuecomment-4414362913
- #1471 parent epic evidence map:
  https://github.com/rmusser01/tldw_server/issues/1471#issuecomment-4414364214

| Gate | Command | Result | Caveats |
| --- | --- | --- | --- |
| Backend ACP and orchestration suites | `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol tldw_Server_API/tests/Agent_Orchestration -q` | 969 passed, 18 warnings in 154.84s. | Warnings are existing pytest/runtime warnings; no failures. |
| Frontend ACP unit contracts | `cd apps/packages/ui && ./node_modules/.bin/vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx --maxWorkers=1 --no-file-parallelism` | 3 files passed, 9 tests passed. | Uses focused contract coverage for Agent Tasks, Agent Registry, and ACP Playground. |
| Browser ACP E2E | `cd apps/tldw-frontend && TLDW_WEB_URL=http://localhost:18081 TLDW_WEB_CMD='bun run dev -- -p 18081' ./node_modules/.bin/playwright test e2e/workflows/tier-3-automation/agent-tasks.spec.ts --grep "guide ACP setup" --reporter=line` | 1 passed in 20.5s. | Deterministic mocked setup/run/diagnose path. Full live-backend ACP tier-3 gate still needs a seeded backend and API key on the release host. |
| Go ACP runner | `cd tools/tldw-agent && ./scripts/verify-local-build.sh` | Runner host and ACP binaries built; `go test ./...` passed. | Downstream live-agent binaries and provider API keys are external prerequisites. |
| Security validation | `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/acp_schedules.py tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py tldw_Server_API/app/core/Agent_Orchestration/completion_signals.py tldw_Server_API/app/services/workflows_scheduler.py -f json -o /tmp/bandit_acp_closeout_1472.json` | `results=[]`, `errors=[]`, 7,357 LOC scanned. | `sandbox_runner_client.py` has 2 documented skipped tests from inline `nosec` comments. |
| Documentation formatting and stale-artifact review | `git diff --check`; targeted `rg` scans for escaped patch artifacts, draft status, stale runner paths, and superseded route names. | Formatting clean; no escaped patch artifacts or draft status; legacy sibling runner paths removed. | Superseded old route names intentionally remain only in the PRD superseded-claims section. |

## ACP Release Signoff Addendum

Issue #1501 expands release evidence for denial, reconnect, and recovery paths
without depending on a live downstream ACP agent in CI. Treat this as
deterministic release evidence plus an explicit live-agent caveat, not as proof
that every installed third-party agent can produce the same permission and
recovery events.

Supported downstream ACP agents for this release host:

| Agent | Version | Release status | Evidence |
| --- | --- | --- | --- |
| None certified | N/A | No live downstream ACP stdio agent is supported or claimed for this release host yet. Candidate local tools were inventoried during #1504, but no installed binary plus provider credentials exposed a verifiable ACP stdio downstream-agent path. | #1504 / #1508 |

| Flow | Automated evidence | Release posture |
| --- | --- | --- |
| Permission denial | Backend coverage includes `test_acp_websocket.py` permission denial, `test_acp_session_management.py` permission-response audit metadata, `test_acp_governance_coordinator.py` governance/policy deny paths, and `test_acp_integration_stub.py` stub-agent policy denial. Frontend coverage includes `ACPPermissionModal.test.tsx` deny action wiring and `useACPSession.test.tsx` WebSocket denial payload/queue cleanup. | Verified for backend policy/audit behavior and frontend denial controls. Live browser denial against a real downstream ACP stdio agent remains release-caveated until #1504 has a real compatible agent and provider credentials. |
| Reconnect and session replay | Backend coverage includes `test_ws_reconnect.py`, `test_sse_consumer.py`, `test_replay_utils.py`, `test_event_bus.py`, and `test_acp_integration_persistence.py`. Frontend coverage includes `ACPChatPanel.test.tsx` manual reconnect affordance and `useACPSession.test.tsx` retry/backoff progress for transient WebSocket closes plus non-retry handling for fatal close codes. | Verified for persisted session state, event replay helpers, SSE/WebSocket catch-up, manual reconnect affordance, and client retry state. Live browser reconnection remains dependent on the seeded backend and active session state used by the final release host. |
| Failed-run recovery and diagnosis | Backend/orchestration coverage includes `test_orchestration_api.py` failed-session diagnostics and task run drill-through. Browser coverage includes `agent-tasks.spec.ts` mocked setup/run/diagnose flow with diagnostics, artifact, audit links, failure context, and reviewer decision summary. | Verified for deterministic task diagnosis and recovery navigation. Full live recovery still depends on #1505 seeded-backend E2E and #1504 downstream-agent availability. |

The accepted #1501 release caveat is that CI-stable coverage uses backend
stubs, component tests, hook tests, and mocked browser routes for deterministic
paths. Release notes should avoid claiming live downstream-agent
permission-denial or reconnect behavior until a configured ACP-compatible agent
is verified on the release host.

## Closeout Checklist

- [x] #1479 is complete, and structured completion status is consumed by API and UI surfaces.
- [x] #1478 is complete, and reviewer decisions are durable and auditable.
- [x] #1476 is complete, and permission/RBAC/audit evidence covers allow, deny, timeout, and escalation paths.
- [x] #1475 is complete, and run history plus artifacts make completed runs inspectable without server logs.
- [x] #1477 is complete, and sandbox/workspace runtime requirements are documented with fail-closed tests.
- [x] #1474 is complete, and schedules/triggers have owner, quota, pause/resume, cancellation, and failure visibility.
- [x] #1473 is complete, and ACP Playground, Agent Tasks, and Agent Registry cover setup, healthy, failed, and recovery states.
- [x] #1480 is complete, and PRD/operator/user docs match the verified implementation.
- [x] Backend ACP and orchestration pytest suites are green or have documented accepted skips.
- [x] Frontend ACP Vitest and browser E2E gates are green or have documented accepted skips.
- [x] Go runner build/test verification is green.
- [x] Bandit has been run on touched backend Python paths with no new unresolved findings.
- [x] Epic #1471 links to final evidence for each readiness row.
