# Sandbox & ACP Competitive Improvements Design

**Date:** 2026-04-05
**Status:** Draft (reviewed)
**Inspired by:** ClaudeCodeUI, Codexia, 1code.dev

## Context

Review of three competitor/reference apps (ClaudeCodeUI, Codexia, 1code.dev) identified 16 high-leverage improvements for tldw_server's Sandbox and ACP modules. The goal is a balanced mix of production hardening and new capabilities across four areas.

**Out of scope:** vz_linux/vz_macos real execution (already in flight on `codex/vz-linux-real-execution-dev` branch with vsock transport, debian builder, kernel artifact extraction, and rootfs packer).

## Review Notes

### First Review Pass
1. **Effort estimates revised upward** — realistic total ~3,620 lines (1.7x original)
2. **Security hardening required** for worktree runner (1.1) and webhook triggers (3.5)
3. **WSBroadcaster architecture** requires per-connection replay, not just a param change (2.2)
4. **Shared replay abstraction** needed between SSE (2.1) and WS reconnect (2.2)
5. **Endpoint file decomposition** — split `agent_client_protocol.py` (2,552 lines, 25 routes) before adding ~12 endpoints
6. **Database migrations** needed for items 3.3 and 4.2 (schema version bumps)
7. **Permission items 4.1 and 4.2 moved to same phase** to avoid double-touching governance code
8. **Observability, health checks, graceful shutdown** added as cross-cutting concerns

### Second Review Pass (deep-dive into actual codebase)
9. **Item 1.2 scope is narrower** — orchestrator `_sessions` is already an L1 cache over the store; session CRUD methods already exist on all 3 backends; only run queue needs migration. Revised from 650 to 400 lines.
10. **Item 2.2 can use existing `bus.snapshot(from_sequence)`** — no need to add `get_events_from()` method; `SessionEventBus.snapshot()` already returns events >= a sequence.
11. **Item 4.3 has no `sandbox_bridge.py`** — snapshot/restore exposed via `SandboxService` directly (line 1910+ of `service.py`); `CheckpointConsumer` must handle active-run guard.
12. **Item 1.3 warm pool needs `docker exec` pattern** — command is baked into `docker create`; pre-created containers must use `sleep infinity` entrypoint + `docker exec` for actual commands.
13. **Item 4.2 requires `PermissionDecisionService` injection** — `GovernanceFilter.__init__()` has no DB handle; inject a service class rather than passing DB directly.
14. **Item 4.5 optimal insertion point** is `_prepare_acp_runtime_prompt()` at line 597 of `agent_client_protocol.py` (already a preprocessing stage). Use tuple return for `_render_prompt()` to minimize interface changes.
15. **Item 3.1 handler registration** is import-side-effect — needs bare import like `from ... import acp as _ensure_acp_handlers`.

---

## Area 1: Sandbox Runtimes & Isolation

### 1.1 Git Worktree Runner ("sandbox lite")

**Inspiration:** 1code.dev (worktree per chat), Codexia (git worktree management)

**What:** New `RuntimeType.worktree` that creates an isolated git worktree per session. No Docker dependency. On macOS, optionally layered with Seatbelt for filesystem restriction.

**How:**
- Add `worktree` to `RuntimeType` enum in `Sandbox/models.py`
- New `runners/worktree_runner.py` (~400 lines):
  - `create_session()`: `git worktree add <temp_path> --detach` from user's repo
  - `run()`: execute commands in the worktree directory; on macOS, delegate to `SeatbeltRunner` with worktree-specific profile as writable root; on Linux, require `unshare` namespace isolation (refuse to run unconfined — no "warning-only" mode)
  - `destroy_session()`: `git worktree remove <path>` + cleanup
- Add worktree preflight to `runtime_capabilities.py` (checks: git repo exists, git version >= 2.15)
- Policy: trust level defaults to `trusted` (host-process, no VM guarantee); `standard` requires Seatbelt (macOS) or `unshare` (Linux)
- Integrate into `service.py` — requires boilerplate in `start_run_scaffold()` (if/elif chain at ~line 1337), `cancel_run()` (~line 1747), and `collect_runtime_preflights()`

**Security hardening (post-review):**
- Validate repo path against an allowlist of permitted directories (prevent traversal to `/etc`, `~/.ssh`, etc.)
- On Linux without `unshare` capability, refuse to start with clear error rather than running unconfined
- Seatbelt delegation requires a worktree-specific profile (existing `seatbelt_policy.py` profiles don't cover this)
- Restrict environment variable passthrough (no `HOME`, `SSH_AUTH_SOCK`, etc.)

**Files:**
- `tldw_Server_API/app/core/Sandbox/models.py` — add enum value
- `tldw_Server_API/app/core/Sandbox/runners/worktree_runner.py` — new (~400 lines)
- `tldw_Server_API/app/core/Sandbox/runners/seatbelt_policy.py` — add worktree profile
- `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py` — add preflight
- `tldw_Server_API/app/core/Sandbox/service.py` — route worktree runtime (3 insertion points)

### 1.2 Orchestrator State Durability

**Inspiration:** Production hardening (orchestrator self-documents as "not production-grade")

**What:** Migrate in-memory session dict and run queue to existing store backends (SQLite/PostgreSQL APIs already exist).

**How:**

**Second-pass finding:** The `_sessions` dict is already an **L1 cache** over the store — `get_session()` validates against the store before returning (lines 282-307 of `orchestrator.py`). Only 9 methods touch it. The session CRUD methods (`put_session`, `get_session`, `delete_session`) already exist on all 3 store backends. The store also already has run claim/admit methods (`try_claim_run`, `renew_run_claim`, `release_run_claim`, `try_admit_run_start`).

**Actual gap is narrower than originally stated:**
- The orchestrator's **run queue** (`self._queue`) is in-memory — this needs migration to the store. Add `enqueue_run()` and `dequeue_run()` abstract methods + 3 backend implementations.
- The orchestrator's `_sessions` dict only needs to become a **proper write-through cache** — ensure all 9 write paths persist to the store (some may already do so; verify each).
- The idempotency tracking may also need store-backed migration if it's currently in-memory.

**Revised effort:** ~400 lines (down from 650 — session CRUD and claim methods already exist; only run queue and cache write-through logic needed).

**Files:**
- `tldw_Server_API/app/core/Sandbox/orchestrator.py` — audit 9 methods touching `_sessions`, ensure write-through to store; migrate `_queue` to store
- `tldw_Server_API/app/core/Sandbox/store.py` — add `enqueue_run`/`dequeue_run` to abstract class + 3 backends

### 1.3 Docker Warm Container Pool

**Inspiration:** 1code.dev (E2B instant sandboxes)

**What:** Pre-create N idle Docker containers for common base images. Claim on session start for sub-second startup.

**How:**
- New `Sandbox/pool.py` (~250 lines, revised up):
  - `DockerWarmPool`: background thread maintains N idle containers
  - `claim(image) -> container_id | None` (fallback to cold start)
  - `release(container_id)`: return to pool or destroy if tainted
  - Config: `SANDBOX_WARM_POOL_SIZE` (default 3), `SANDBOX_WARM_POOL_IMAGES`
- Modify `DockerRunner.start_run()` to accept optional pre-created container ID

**Second-pass finding — command injection issue:** In `docker_runner.py`, the command/entrypoint is baked into the `docker create` call (lines 405-413). Pre-created containers can't have their command changed after creation. Two options:
- **(a) `docker exec` pattern:** Pre-create containers with a long-running `sleep` entrypoint, then use `docker exec` to run the actual command. This avoids the baked-command problem but changes the execution model.
- **(b) Image-matched pool:** Pre-create containers with a generic entrypoint (`/bin/sh -c "sleep infinity"`), claim one, then `docker exec` the user's command inside it. File injection (lines 472-524) still works via `docker cp`.
- **Recommended: option (b)** — matches the existing file injection pattern and avoids modifying the create flow.

**Files:**
- `tldw_Server_API/app/core/Sandbox/pool.py` — new (~250 lines)
- `tldw_Server_API/app/core/Sandbox/runners/docker_runner.py` — add `docker exec` path when pooled container provided

---

## Area 2: Agent Communication & Streaming

### 2.1 SSE Consumer + Endpoint

**Inspiration:** 1code.dev (11 typed SSE events, Vercel AI SDK compatible)

**What:** SSE as an alternative transport for ACP agent events, using existing `AgentEventKind` (13 types) as SSE event names.

**How:**
- New `consumers/sse_consumer.py` (~100 lines):
  - `SSEConsumer(EventConsumer)` with `asyncio.Queue`
  - `async def iter_sse_lines()` generator: `event: {kind}\ndata: {json}\n\n`
  - Supports `from_sequence` for reconnect via `SessionEventBus.subscribe()`
- New endpoint `GET /api/v1/acp/sessions/{session_id}/events`:
  - Returns `StreamingResponse(media_type="text/event-stream")`
  - Query param `last_event_id` → `from_sequence`
  - 15-second heartbeat keepalive

**Files:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/sse_consumer.py` — new
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add SSE endpoint

### 2.2 WebSocket Reconnect with Catch-Up

**Inspiration:** ClaudeCodeUI (WS reconnect with message catch-up)

**What:** Accept `last_sequence` on WS connect; replay buffered events before switching to live.

**Architecture note (post-review):** The current `WSBroadcaster` uses a **single** bus subscription (`consumer_id = "ws_broadcaster"`) shared by all connections. Per-connection replay requires either:
- (a) Multiple bus subscriptions (one per connection), or
- (b) A per-connection replay mechanism within WSBroadcaster that reads from the ring buffer directly

Option (b) is preferred — add a `_replay_to_connection()` method that reads from `SessionEventBus.get_events_from(sequence)` and sends buffered events to the new connection before adding it to the live broadcast set. This avoids N bus subscriptions.

**Shared replay abstraction (post-review):** Items 2.1 (SSE) and 2.2 (WS) both need catch-up replay. Extract a common `ReplayMixin` or helper function `replay_events_from(bus, sequence, sink)` that both `SSEConsumer` and `WSBroadcaster` use. Prevents duplicated replay logic with subtly different behaviors.

**How:**

**Second-pass finding:** `SessionEventBus` already has `snapshot(from_sequence: int = 0) -> list[AgentEvent]` at line 113 — this returns buffered events >= `from_sequence`. No need to add `get_events_from()`; use existing `snapshot()` instead. Buffer size is already configurable via `max_buffer` param in `__init__()` (default 10,000 at line 23).

- Use existing `bus.snapshot(from_sequence)` in `_replay_to_connection()` — no event_bus changes needed
- Add `_replay_to_connection(ws, from_sequence)` to `WSBroadcaster`
- Extract shared `replay_events(bus, from_sequence, emit_fn)` helper for SSE + WS reuse
- Accept `last_sequence` query param on WS connect endpoint
- Expose `ACP_EVENT_BUFFER_SIZE` as a config option that gets passed to `SessionEventBus(max_buffer=...)` at construction time

**Effort:** ~130 lines

**Files:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py` — add `_replay_to_connection()` using `bus.snapshot()`
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/replay_utils.py` — new shared replay helper
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — accept `last_sequence` query param on WS connect

### 2.3 Event Sink Unification (Sandbox → ACP Events)

**Inspiration:** Codexia (event sink abstraction decoupling business logic from broadcasting)

**What:** Bridge RunStreamHub frames into `AgentEvent` objects on the `SessionEventBus` — one streaming pipeline for all consumers.

**How:**
- New `Sandbox/event_bridge.py` (~100 lines):
  - `SandboxEventBridge` subscribes to `RunStreamHub` for a given run_id
  - Translates raw frames: stdout/stderr → `TERMINAL_OUTPUT`, file change → `FILE_CHANGE`, exit → `COMPLETION`, error → `ERROR`
  - Publishes onto `SessionEventBus` with consistent sequencing
- Eliminates need for clients to speak two different streaming protocols

**Note (post-review):** `RunStreamHub` already has `subscribe()` and `subscribe_with_buffer()` methods (lines 95-148 of `streams.py`). No modification to `streams.py` needed — the bridge just consumes from the existing API.

**Files:**
- `tldw_Server_API/app/core/Sandbox/event_bridge.py` — new (consumes existing `RunStreamHub.subscribe()` API)

---

## Area 3: Automation & Scheduling

### 3.1 `acp_run` Scheduler Handler

**Inspiration:** Codexia (cron-based agent tasks with run history)

**What:** New task handler registered in the existing Scheduler, following the `workflow_run` pattern.

**How:**
- New `Scheduler/handlers/acp.py` (~100 lines):
  ```python
  @task(name="acp_run", max_retries=1, timeout=7200, queue="acp")
  async def acp_run(payload: dict) -> dict:
      # 1. Get runner client
      # 2. Create session (cwd, agent_type, user_id)
      # 3. Send prompt
      # 4. Close session
      # 5. Return {session_id, result, usage, duration_ms}
  ```
- Payload: `{user_id, agent_type, cwd, prompt, model, token_budget, persona_id, workspace_id, sandbox_enabled}`
- Reuses session lifecycle from `acp_stage` adapter

**Registration note (post-review):** The handler must be explicitly imported at startup for registration (see `workflows_scheduler.py` lines 38-43 for the pattern). Add import in the scheduler initialization path.

**Files:**
- `tldw_Server_API/app/core/Scheduler/handlers/acp.py` — new
- `tldw_Server_API/app/services/workflows_scheduler.py` — add handler import for registration
- Pattern reference: `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`

### 3.2 Async Fire-and-Forget API

**Inspiration:** 1code.dev (async HTTP API with status polling)

**What:** Submit ACP prompts asynchronously; poll for results.

**How:**
- `POST /api/v1/acp/sessions/prompt-async` → submits `acp_run` to Scheduler → returns `{task_id, poll_url}`
- `GET /api/v1/acp/tasks/{task_id}` → returns `{status, result, usage, error}` from Scheduler
- Optional: emit `COMPLETION` event on `SessionEventBus` for push notification via WS/SSE

**Files:**
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add 2 endpoints

### 3.3 ACP Schedule CRUD

**Inspiration:** Codexia (automation scheduler with daily/interval/cron modes)

**What:** CRUD endpoints for recurring ACP agent schedules, built on existing `WorkflowsScheduler`.

**How:**
- Extend `workflow_schedules` table with nullable `acp_config_json` column
- When present, scheduler emits `acp_run` instead of `workflow_run`
- 4 endpoints: `POST/GET/PUT/DELETE /api/v1/acp/schedules`
- Leverages existing cron parsing, concurrency modes, jitter, coalesce, misfire grace

**Migration note (post-review):** Requires ALTER TABLE on `workflow_schedules` in both SQLite and PostgreSQL schemas (hardcoded at lines 33-92 of `Workflows_Scheduler_DB.py`). Must add explicit guard in `_load_all()` to skip ACP-configured schedules when emitting `workflow_run` tasks — without this, ACP schedules will incorrectly fire as `workflow_run`.

**Effort:** ~450 lines (revised from 200 — dual-schema migration + APScheduler integration + guard logic)

**Files:**
- `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py` — add column + migration
- `tldw_Server_API/app/services/workflows_scheduler.py` — route acp_config schedules + guard `_load_all()`
- `tldw_Server_API/app/api/v1/endpoints/acp_schedules.py` — new file (split from main endpoint file)

### 3.4 Run History + Cost Tracking API

**Inspiration:** Codexia (run history with token counts)

**What:** Query and aggregate existing session data — no new tables needed.

**How:**
- `GET /api/v1/acp/runs` — query `SessionRecord` with date range, agent_type, status filters
- `GET /api/v1/acp/runs/aggregate` — sum token usage via `compute_token_cost()`

**Files:**
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add 2 endpoints
- `tldw_Server_API/app/services/admin_acp_sessions_service.py` — add query methods

### 3.5 Webhook/Event Trigger System

**Inspiration:** 1code.dev (GitHub/Slack triggers via @mentions), Codexia (event-driven automation)

**What:** Inbound webhook receiver that translates external events into ACP agent executions.

**How:**
- New `Agent_Client_Protocol/triggers.py` (~500 lines, revised up):
  - `ACPTriggerManager` routes inbound events to `acp_run` task submissions
  - Methods: `on_github_webhook()`, `on_slack_event()`, `on_generic_webhook()`
  - Configurable routing rules: event type → agent config + prompt template
- New endpoint: `POST /api/v1/acp/triggers/webhook/{trigger_id}`
  - Validates HMAC signature — **write from scratch** (existing `webhook.py` adapter signs outbound, not inbound)
  - Provider-specific verification: GitHub uses `X-Hub-Signature-256`, Slack uses its own signing scheme
  - Enqueues `acp_run` with trigger metadata
  - Returns `{task_id}` for tracking
- Trigger CRUD: `POST/GET/PUT/DELETE /api/v1/acp/triggers`
  - Each trigger: `{id, name, source_type, secret_encrypted, agent_config, prompt_template, enabled}`

**Security hardening (post-review):**
- Rate limiting on inbound webhook endpoint (per trigger_id, e.g., 60 req/min) to prevent flood attacks
- Webhook secrets **encrypted at rest** in the database, not plaintext
- Replay attack prevention: reject requests with timestamps older than 5 minutes
- Trigger-to-user mapping: each trigger has an `owner_user_id`; `acp_run` runs as that user
- Timing-safe HMAC comparison (`hmac.compare_digest`)

**Files:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/triggers.py` — new (~500 lines)
- `tldw_Server_API/app/api/v1/endpoints/acp_triggers.py` — new file (split from main endpoint file)
- Pattern reference: `tldw_Server_API/app/core/Workflows/adapters/integration/webhook.py` (outbound signing only)

---

## Area 4: Permission & Governance UX

### 4.0 Permission Unification: GovernanceFilter → MCPHub Integration

**Motivation:** Currently there are two **parallel, independent** permission systems that never converge:

| System | Scope | Data Source | Decision Point | Persona/Workspace Aware |
|--------|-------|-------------|----------------|------------------------|
| **ACP Tier System** | Per tool call | In-memory `_permission_policies` dict (lost on restart) | `GovernanceFilter.process()` | No |
| **MCPHub Policy Resolver** | Per session | Database-backed assignments, profiles, scopes, overrides | `ACPRuntimePolicyService.build_snapshot()` | Yes (full hierarchy) |

A tool blocked by MCPHub policy can still be auto-approved by ACP tiers. ACP tier decisions have no persona/workspace/org context. Adding new permission features (4.1-4.4) on top of the ACP tier system would create a **third layer**, making divergence worse.

**What:** Make `GovernanceFilter` consult the `ACPRuntimePolicySnapshot` (which comes from MCPHub) for per-tool-call decisions. MCPHub becomes the single authority; ACP tiers become a fallback for sessions without an MCPHub snapshot.

**How:**
1. **Inject `ACPRuntimePolicySnapshot` into `GovernanceFilter`** — Pass the session's snapshot (already built by `ACPRuntimePolicyService.build_snapshot()`) to the filter at session creation.
2. **Check MCPHub policy first in `GovernanceFilter.process()`** — Before calling `determine_permission_tier()`:
   - If tool_name matches `snapshot.denied_tools` → immediately deny (no human prompt)
   - If tool_name matches `snapshot.allowed_tools` → auto-approve
   - If snapshot has `tool_tier_overrides` (new field, see 4.1) → use that tier
   - If snapshot has `approval_mode` → respect it (ask_every_time, ask_outside_profile, etc.)
   - Fall through to existing `determine_permission_tier()` only if no MCPHub decision
3. **Persist ACP permission policies to database** — Move `_permission_policies` from in-memory dict to `ACPSessionsDB` so they survive restarts. These become a supplementary layer under MCPHub.
4. **Pass persona/workspace context** — `GovernanceFilter` receives session metadata (persona_id, workspace_id) so MCPHub rules can be context-aware per tool call.

**Decision hierarchy (unified):**
```
Tool Call → GovernanceFilter.process()
  1. Check MCPHub snapshot: denied_tools → DENY
  2. Check MCPHub snapshot: allowed_tools → AUTO
  3. Check MCPHub snapshot: tool_tier_overrides (4.1) → use tier
  4. Check MCPHub snapshot: approval_mode → apply mode
  5. Check persisted permission decisions (4.2) → apply if matched
  6. Check admin ACP permission policies (from DB) → apply if matched
  7. Fallback: heuristic rules (permission_tiers.py) → auto/batch/individual
```

**Files:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/governance_filter.py` — accept snapshot + session metadata in `__init__()`; check MCPHub policy at line 94 before tier classification
- `tldw_Server_API/app/services/acp_runtime_policy_service.py` — add `tool_tier_overrides` to snapshot schema
- `tldw_Server_API/app/services/admin_acp_sessions_service.py` — migrate `_permission_policies` dict to `ACPSessionsDB`
- `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py` — add `permission_policies` table (replaces in-memory dict)

**Effort:** ~300 lines

---

### 4.1 Bash Prefix Matching

**Inspiration:** ClaudeCodeUI (`Bash(git:*)`, `Bash(npm:*)` prefix matching)

**What:** Granular tool permission overrides using fnmatch patterns.

**How (builds on 4.0 unification):**
- Add `tool_tier_overrides` dict to the MCPHub policy document schema and `ACPRuntimePolicySnapshot`
- GovernanceFilter checks `snapshot.tool_tier_overrides` at step 3 of the unified decision hierarchy
- Config example: `{"Bash(git:*)": "auto", "Bash(npm:*)": "batch", "Bash(rm:*)": "individual"}`
- Overrides are set via MCPHub policy assignments (per persona, per workspace, per org) — inheriting full scope hierarchy
- `runner_client.py`'s `_resolve_runtime_permission_outcome()` at line ~78 also gains `tool_tier_overrides` for consistency

**Files:**
- `tldw_Server_API/app/services/mcp_hub_policy_resolver.py` — add `tool_tier_overrides` to resolved policy document
- `tldw_Server_API/app/services/acp_runtime_policy_service.py` — propagate to snapshot
- `tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py` — extend `_resolve_runtime_permission_outcome()`

### 4.2 Permission Decision Persistence ("Remember")

**Inspiration:** ClaudeCodeUI ("Remember" button for persisting permission decisions)

**What:** Persist user permission decisions so they don't get re-prompted for the same tool.

**How:**
- New `permission_decisions` table in `ACPSessionsDB`:
  ```sql
  CREATE TABLE permission_decisions (
      id TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL,
      tool_pattern TEXT NOT NULL,
      decision TEXT NOT NULL,  -- 'allow' | 'deny'
      scope TEXT NOT NULL DEFAULT 'session',  -- 'session' | 'global'
      session_id TEXT,
      created_at TEXT NOT NULL,
      expires_at TEXT,
      reason TEXT
  );
  ```
- Modify `GovernanceFilter.process()`: check persisted decisions before holding tool calls
- Modify `GovernanceFilter.on_permission_response()`: persist when `remember=true`
- CRUD endpoints: `GET/POST/DELETE /api/v1/acp/permissions`

**Builds on 4.0 unification:** `GovernanceFilter` now receives session metadata and `PermissionDecisionService` via its expanded `__init__()` (from item 4.0). Persisted decisions are checked at step 5 of the unified decision hierarchy — after MCPHub snapshot checks but before admin policies and heuristic fallback.

**`PermissionDecisionService`** (~60 lines) wraps `ACPSessionsDB` with methods:
- `check(user_id, tool_name, session_id=None) -> str|None` — return persisted decision if matched
- `persist(user_id, tool_pattern, decision, scope, session_id=None)` — save decision
- `list_for_user(user_id) -> list[dict]` — list persisted decisions
- `revoke(decision_id)` — remove a decision

Persisted decisions respect scope:
- `scope="session"` — only applies within the originating session
- `scope="global"` — applies across all sessions for that user
- Both scopes are persona-aware (decisions stored with persona_id from session metadata)

**Migration note:** Requires schema version bump in `ACP_Sessions_DB.py` (currently `_SCHEMA_VERSION = 7` at line 21).

**Files:**
- `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py` — add table + bump schema version to 8
- `tldw_Server_API/app/core/Agent_Client_Protocol/governance_filter.py` — accept `PermissionDecisionService` in `__init__()`, check at line 94, persist in `on_permission_response()` at line 157
- `tldw_Server_API/app/core/Agent_Client_Protocol/permission_decision_service.py` — new (~60 lines)
- `tldw_Server_API/app/api/v1/endpoints/acp_permissions.py` — new file (split from main endpoint file)

### 4.3 Checkpoint-Based Rollback

**Inspiration:** 1code.dev (rollback from any message point)

**What:** Auto-snapshot sandbox state before file mutations; enable rollback to any checkpoint.

**How:**
- New `consumers/checkpoint_consumer.py` (~200 lines):
  - `CheckpointConsumer(EventConsumer)` subscribes to `SessionEventBus`
  - Auto-snapshots before `FILE_CHANGE` events (configurable: every N tool calls, or on explicit request)
  - Labels snapshots with sequence number: `seq_{N}_pre_{tool_name}`
- New endpoint: `POST /api/v1/acp/sessions/{session_id}/rollback`
  - Accepts `{to_sequence: N}` or `{to_snapshot_id: "..."}`
  - Restores via existing `SnapshotManager.restore_snapshot()`
  - Emits `LIFECYCLE` event marking the rollback
- **Scoped to sandbox-backed sessions only** (where `SnapshotManager` can checkpoint filesystem)

**Second-pass correction:** There is no `sandbox_bridge.py` file. Snapshot/restore is exposed via `SandboxService` directly:
- `SandboxService.create_snapshot(session_id) -> dict` (line 1910 of `service.py`) — wraps with workspace lock and active-run check
- `SandboxService.restore_snapshot(session_id, snapshot_id) -> bool` — same guards
- The `CheckpointConsumer` must hold a reference to `SandboxService` (not a bridge). The consumer calls `service.create_snapshot(session_id)` on each `FILE_CHANGE` event.
- **Important:** `create_snapshot()` calls `_ensure_no_active_session_runs()` — the consumer must handle the case where a run is active (skip snapshot, log warning).

**Files:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/checkpoint_consumer.py` — new
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — add rollback endpoint
- Existing: `tldw_Server_API/app/core/Sandbox/snapshots.py`, `tldw_Server_API/app/core/Sandbox/service.py` (lines 1910+)

### 4.4 Permission Policy Templates

**Inspiration:** Codexia (three-level permission model)

**What:** 4-5 preset policy configurations as JSON base layers.

**How:**
- Presets: `read-only`, `developer`, `admin`, `lockdown`
  - **read-only**: all write/exec tools → `individual`, read tools → `auto`
  - **developer**: git/npm → `auto`, file writes → `batch`, shell exec → `individual`
  - **admin**: everything `auto` except destructive operations
  - **lockdown**: everything `individual`
- Loaded as base layers in `ACPRuntimePolicyService.build_snapshot()`
- Merged with user customizations (user overrides take precedence)

**Files:**
- `tldw_Server_API/app/services/acp_runtime_policy_service.py` — add template loading
- `tldw_Server_API/app/core/Agent_Client_Protocol/config.py` — add template definitions

### 4.5 MCP Tool @Mentions

**Inspiration:** 1code.dev (MCP tool @mentions in prompts for natural language tool invocation)

**What:** Parse `@tool_name` in prompts and translate to structured tool hints in ACP message metadata.

**How:**
- Prompt preprocessor in `Agent_Client_Protocol/prompt_utils.py` (~180 lines, revised up):
  - Regex scan for `@tool_name` patterns
  - Resolve against registered MCP tools (from MCP Hub registry) — requires async DB lookup, so preprocessor must be `async`
  - Replace with clean text and add `tool_hints: [tool_name, ...]` to message metadata
  - Cache resolved tool names per-session to avoid repeated DB lookups
  - Unresolved @mentions left as-is with a warning in response metadata

**Second-pass finding — optimal insertion point:** The HTTP-layer function `_prepare_acp_runtime_prompt()` at line 597 of `agent_client_protocol.py` is the ideal place to wire the preprocessor. It already preprocesses prompts (injecting bootstrap context) before governance checks. This keeps @mention resolution in the HTTP layer where user context is available.

**Return type impact on `_render_prompt()`:** The `acp.py` adapter's `_render_prompt()` returns `list[dict[str, Any]]` which is directly passed to `runner.prompt(session_id, prompt_payload)` at line 490. Changing the return type to a `RenderedPrompt` dataclass requires updating this call site. **Alternative:** Keep `_render_prompt()` return type unchanged, and have the preprocessor operate on the list in-place while returning tool_hints as a separate value via a tuple `(messages, tool_hints)`. This minimizes interface changes.

**Files:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/prompt_utils.py` — new (~180 lines)
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` — wire into `_prepare_acp_runtime_prompt()` at line 597
- `tldw_Server_API/app/core/Workflows/adapters/integration/acp.py` — wire preprocessor, minimal return type change (tuple instead of dataclass)

---

## Implementation Sequence

### Phase 0: Prep (before starting)
- **Split `agent_client_protocol.py`** (2,552 lines, 25 routes) into sub-modules before adding ~12 endpoints:
  - `acp_schedules.py` — schedule CRUD
  - `acp_triggers.py` — webhook trigger endpoints
  - `acp_permissions.py` — permission CRUD
  - Keep core session/prompt endpoints in the main file

### Phase 1: Foundations (Week 1)
- **4.0** Permission unification: GovernanceFilter → MCPHub integration (MUST come before 4.1/4.2 — they build on it)
- **1.2** Orchestrator state durability
- **2.2** WebSocket reconnect with catch-up + shared replay abstraction

### Phase 1b: Permission Features (Week 1-2, after 4.0)
- **4.1** Bash prefix matching (builds on 4.0 unified hierarchy)
- **4.2** Permission decision persistence (builds on 4.0 expanded GovernanceFilter)

### Phase 2: Streaming & Async (Week 2)
- **2.1** SSE consumer + endpoint (reuses shared replay from 2.2)
- **2.3** Event sink unification
- **3.1** `acp_run` scheduler handler
- **3.2** Async fire-and-forget API

### Phase 3: Automation (Week 3)
- **3.3** ACP schedule CRUD
- **3.4** Run history + cost tracking API
- **3.5** Webhook/event trigger system

### Phase 4: Sandbox & Permissions (Week 4)
- **1.1** Git worktree runner
- **4.4** Permission policy templates
- **4.5** MCP tool @mentions

### Phase 5: Advanced (Week 5)
- **1.3** Docker warm container pool
- **4.3** Checkpoint-based rollback

## Cross-Cutting Concerns (apply throughout)

**Observability:** Add `increment_counter`/`observe_histogram` metrics (from existing `Metrics` module) for warm pool claims, scheduled runs, webhook triggers, permission decisions, and rollbacks.

**Health checks:** Register new background components (warm pool thread, trigger manager) with existing `health_monitor.py`.

**Graceful shutdown:** Items 1.3 (warm pool background thread) and 3.3 (APScheduler integration) need shutdown hooks to clean up on process exit.

**Config validation:** Add startup validation for new env vars (`SANDBOX_WARM_POOL_SIZE`, `SANDBOX_WARM_POOL_IMAGES`, `ACP_EVENT_BUFFER_SIZE`) following the pattern of `validate_acp_config()` at `config.py` line 89.

---

## Verification Plan

### Per-item testing
- Each new runner/consumer/handler gets a dedicated test file in the corresponding `tests/` directory
- Follow existing patterns: `tests/sandbox/test_*.py`, `tests/Agent_Client_Protocol/test_*.py`

### Integration testing
- **Worktree runner**: create session in a test repo, run `echo hello > test.txt`, verify file exists in worktree but not in main working copy, destroy session, verify worktree removed
- **SSE endpoint**: connect SSE client, send prompt, verify typed events received in order with sequence numbers
- **WebSocket reconnect**: connect, receive events, disconnect, reconnect with `last_sequence`, verify replay
- **Async API**: submit prompt-async, poll until complete, verify result matches synchronous execution
- **Scheduled runs**: create cron schedule, advance time, verify `acp_run` task enqueued and executed
- **Webhook triggers**: POST to trigger endpoint with HMAC, verify `acp_run` enqueued with correct payload
- **Permission persistence**: approve with `remember=true`, submit same tool call, verify auto-approved
- **Rollback**: execute 3 tool calls, rollback to sequence 1, verify sandbox state matches post-call-1
- **MCP @mentions**: send prompt with `@search_web`, verify `tool_hints` in message metadata

### End-to-end scenario
1. Create a scheduled ACP agent that runs daily via cron (3.3)
2. Agent runs in a git worktree sandbox (1.1) with developer permission template (4.4)
3. Stream events via SSE (2.1) with reconnect support (2.2)
4. Permission decisions persisted via "remember" (4.2) with bash prefix rules (4.1)
5. If something goes wrong, rollback to checkpoint (4.3)
6. External GitHub webhook triggers ad-hoc runs (3.5)
7. Cost tracked and queryable (3.4)

---

## Summary (revised estimates — second-pass review + unification)

| # | Item | Area | New files | ~Lines | Key risk / notes |
|---|------|------|-----------|--------|------------------|
| 0 | Endpoint file split | Prep | 3 | 200 | Prerequisite for all endpoint additions |
| **4.0** | **Permission unification** | **Permissions** | **1** | **300** | **GovernanceFilter → MCPHub; migrate in-memory policies to DB; single authority** |
| 1.1 | Git worktree runner | Sandbox | 2 | 400 | Security: no unconfined Linux execution; needs seatbelt profile |
| 1.2 | Orchestrator durability | Sandbox | 0 | 400 | Narrower gap: sessions already cached over store; only run queue needs migration |
| 1.3 | Docker warm pool | Sandbox | 1 | 250 | Must use `docker exec` pattern (command baked into `docker create`) |
| 2.1 | SSE consumer + endpoint | Streaming | 1 | 140 | Reuses shared replay + existing `bus.snapshot()` |
| 2.2 | WS reconnect catch-up | Streaming | 1 | 130 | Use existing `bus.snapshot(from_sequence)`, no event_bus changes |
| 2.3 | Event sink unification | Streaming | 1 | 100 | No `streams.py` changes needed |
| 3.1 | `acp_run` handler | Automation | 1 | 100 | Registration via bare import side-effect |
| 3.2 | Async API | Automation | 0 | 80 | — |
| 3.3 | Schedule CRUD | Automation | 1 | 450 | Guard `_load_all()` against ACP schedules firing as `workflow_run` |
| 3.4 | Run history + costs | Automation | 0 | 100 | — |
| 3.5 | Webhook triggers | Automation | 2 | 500 | Multi-provider HMAC, encrypted secrets, rate limiting |
| 4.1 | Bash prefix matching | Permissions | 0 | 50 | Builds on 4.0; add `tool_tier_overrides` to MCPHub policy + snapshot |
| 4.2 | Permission persistence | Permissions | 2 | 260 | Builds on 4.0; `PermissionDecisionService` + persona-aware scoping |
| 4.3 | Checkpoint rollback | Permissions | 1 | 200 | No `sandbox_bridge.py`; use `SandboxService` directly; handle active-run guard |
| 4.4 | Policy templates | Permissions | 0 | 100 | — |
| 4.5 | MCP @mentions | Permissions | 1 | 180 | Wire into `_prepare_acp_runtime_prompt()` at line 597 |
| | **Total** | | **18 new** | **~3,940** | +300 for unification |
