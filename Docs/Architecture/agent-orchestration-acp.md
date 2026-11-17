# Agent orchestration and ACP

This document describes the Agent Client Protocol surface (runner sessions, the layered permission stack, hardening) and the orchestration layer above it (workspaces, projects, tasks, runs, the completion-signal contract, reviewer gates, artifact promotion).

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Runner client | `app/core/Agent_Client_Protocol/runner_client.py` | `ACPRunnerClient` (JSON-RPC over the runner), `ACPGovernanceCoordinator`, runtime-policy resolution merged with governance outcomes, a 300 s permission timeout |
| Config | `Agent_Client_Protocol/config.py` | runner command/args/env/cwd, permission-policy templates (read-only/developer/admin/lockdown), validation |
| Hardening | `Agent_Client_Protocol/hardening.py` | launch-input validation: absolute cwd inside allowlisted roots, session-env and inline-stdio MCP opt-ins, SSRF checks on MCP HTTP URLs; 256 KiB update / 1 MiB stream buffer caps |
| Registry | `Agent_Client_Protocol/agent_registry.py` | `agents.yaml`-driven registry with entrypoint classification |
| Events | `events.py`, `event_bus.py`, `consumers/` | typed event kinds (thinking, tool_call, permission_request, …); audit, checkpoint (bounded 50), metrics, SSE, WS broadcast consumers |
| Permissions | `governance_filter.py`, `tool_gate.py`, `permission_decision_service.py`, `permission_tiers.py` | the tier system, persisted allow/deny decisions (global then session scope, fnmatch) |
| Adapters | `adapters/mcp_{adapter,runners}.py`, transports (sse/stdio/streamable_http) | agent-driven and LLM-driven (ReAct) MCP runners behind the ToolGate |
| Sandbox | `sandbox_runner_client.py`, `sandbox_bridge.py` | optional docker/firecracker/lima runtimes; a WS SSH endpoint |
| Endpoints | `app/api/v1/endpoints/agent_client_protocol.py` (prefix `/acp`) | sessions new/prompt/cancel/close, detail/usage/updates/events(+SSE)/artifacts/diagnostics/audit/checkpoints, teardown/reconcile/fork/rollback, WS stream + SSH, `prompt-async` → Scheduler handler `acp_run` |
| Session persistence | `app/core/DB_Management/ACP_Sessions_DB.py` + `app/services/admin_acp_sessions_service.py` | the shared admin-side sessions store (ADR-016) |
| Runtime policy | `app/services/acp_runtime_policy_service.py` | MCP-Hub-derived per-session policy snapshots, persisted with the session |
| Orchestration | `app/core/Agent_Orchestration/{models,db_factory,completion_signals,artifact_promotion}.py` + endpoints (prefix `/agent-orchestration`) | task/run state machines, per-user orchestration DBs, the completion/review marker contracts (≤20 artifacts, 1 MB JSON, 20k text), canonical-workspace artifact promotion |

## Session lifecycle (dataflow)

1. `POST /acp/sessions/new`: quota check (default 5 concurrent/user) → runner selection (sandbox manager when enabled, else the local singleton) → **hardening validation** → JSON-RPC `session/new` (progressive retry dropping `agentType`, then `mcpServers` on invalid-params) → registry-resolved agent/model metadata → session persisted + the initial runtime-policy snapshot built (snapshot failure marks the state but keeps the session usable).
2. `POST /acp/sessions/prompt`: token-quota and budget pre-checks → **governance first** (a deny under `enforce` → 403; other ACP errors → 502).
3. `session/update` notifications queue (bounded) for polling and WS broadcast; consumers fan out from the event bus.

## Tool permission decision (dataflow — the layered stack)

1. The runner requests permission; the tool's tier is determined (batch-tier approvals remembered).
2. **Runtime policy** (default-deny): a missing snapshot denies (`policy_snapshot_unavailable`); `denied_tools` fnmatch denies; a tool not matching the allow set denies (`tool_not_allowed_by_policy`); tier overrides and approval modes can force a human prompt.
3. **Governance** check with the tool name as category; rollout `off` short-circuits allow.
4. **Merge**: runtime deny wins; governance deny under `enforce` denies; `require_approval` or a runtime prompt → prompt; else approve.
5. A prompt with no live WebSocket connection — or the 300 s timeout — resolves **cancelled** (a denial). This makes headless API sessions auto-tier-only.

## Orchestrated task run (dataflow)

1. Guards: dependency complete, no running run; cwd confined to the validated workspace root.
2. Quota → task `inprogress` → ACP session created with the workspace's enabled MCP servers and env → run row → prompt carries the **completion contract**: exactly one `<acp-task-completion>` marker (or the direct field). Missing/malformed/multiple markers fail the run (`completion_signal_invalid`) and send the task to **triage**.
3. Reviewer gate (when set): a second ACP session must emit an `<acp-review-decision>` marker; a failed review records a rejection, pushing the task back to `inprogress` until `max_review_attempts` (3), then triage. Approved → `complete`.
4. **Artifact promotion** into the canonical workspace (user ChaChaNotes) only on accepted decisions, only for promotable types, with task/run/session lineage embedded and optimistic-versioned updates; per-artifact failures become `errors` entries, never task aborts. Every state change audits.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Empty `runner_command` / cwd outside roots | Sessions cannot launch → 502 (roots default to the runner cwd) |
| Governance deny (enforce) on prompt | 403; on orchestrated runs: run failed, task triaged |
| Permission prompt with no WS / timeout | Cancelled (= denial) |
| Quotas (sessions/tokens/budget) | 429 with typed details |
| Invalid completion/review marker; reviewer crash | Run failed, task triaged, 502 |
| Workspace root unconfigured / not absolute / outside allowlist | 503 / 400 / 403 |
| Runner lacks close method | Falls back through the custom close RPCs |

## Config keys (selection)

`[ACP]`: `runner_command`/`runner_args` — the shipped config composes the default launch `go run ./cmd/tldw-agent-acp`; the **code** default for `runner_command` is empty (with a loud warning), so sessions are disabled until configured — plus `runner_env/cwd/binary_path` (relative paths resolve against **Config_Files**, not the process cwd), `startup_timeout_ms`; env `ACP_RUNNER_*`, `ACP_ALLOWED_SESSION_CWD_ROOTS`, `ACP_ALLOW_SESSION_ENV`, `ACP_ALLOW_INLINE_STDIO_MCP`, `ACP_ALLOW_PRIVATE_MCP_HTTP`. `[ACP-WORKSPACE]`: `allowed_base_paths`, discovery depth/patterns. `[ACP-SANDBOX]`: `enabled` (false), runtime, image, network policy (`deny_all`), egress allowlist, SSH range, TTL/concurrency/token/duration caps, retention. Also `ACP_SESSIONS_DB_PATH`, buffer caps, and the first-run rollout experiment knobs.

## Verified gotchas

1. Runtime policy is default-deny for tools; the **governance** layer fails *open* (coordinator unavailable ⇒ no governance constraint) — the two layers have opposite failure postures by design.
2. A governance deny only blocks under `enforce`; `require_approval` escalates to a prompt in both shadow and enforce; shadow never blocks.
3. The completion/review markers are exact automation contracts — prompt-text changes must be tested against the validators.
4. `ACPWorkspace.env_vars` are plaintext in per-user SQLite — not for secrets.
5. Orchestration dispatch is synchronous end-to-end inside the HTTP request; the async escape is `/acp/sessions/prompt-async` (Scheduler `acp` queue — deliberately not Jobs; ADR-041 keeps *scheduled* automated agent execution uncertified).
6. Promotion requires the workspace to carry the canonical-bridge id; unbridged workspaces skip with `missing_workspace`.
7. The chatbook's "ACP" is a local runtime launcher, not this surface; the server-seam client method for it has no production callers today.

## Governing decisions and docs

ADR-016 (`Docs/ADR/016-acp-session-and-orchestration-persistence.md` — the shared-sessions vs per-user-orchestration split), ADR-041 (scheduled agent execution feasibility), ADR-010 (sandbox runtime), ADR-052 (adjacent). Current: both module READMEs, `Docs/Development/Agent_Client_Protocol.md`, the ADR production-readiness family. Tests: `tests/Agent_Client_Protocol/` (72 test files), `tests/Agent_Orchestration/`.

## Related docs

- [governance-billing.md](./governance-billing.md) — the governance half of the permission stack
- [mcp-unified.md](./mcp-unified.md) — the MCP the adapters speak
- [jobs-background.md](./jobs-background.md) — the Scheduler path for async prompts
