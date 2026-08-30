# Canonical Admin Webhook Delivery Substrate And Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver upstream PR 2 of the approved canonical outgoing-webhook design: encrypted synthetic event capture and bounded fanout, recoverable AuthNZ/Jobs delivery handshakes, a secure status-only attempt executor, supported Jobs disposition semantics, synchronous tests, manual redelivery, history, retention, metrics, and health.

**Architecture:** AuthNZ remains the operator-facing source of truth for immutable events, delivery state, append-only attempts, idempotency, and runtime heartbeat evidence. Jobs remains the sole lease and automatic retry scheduler. A delivery service writes events and bounded fanout transactionally; a reconciler bridges AuthNZ and Jobs with idempotent claims; a lease-aware worker reserves one durable attempt before invoking a shared executor; and durable prepared-disposition tokens repair every cross-database crash window without an unintended HTTP request. The existing peer-verified one-hop Security transport gains a status-only mode, while the canonical API exposes persisted test, redelivery, history, and health contracts without carrying SQL or network logic.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, aiosqlite/SQLite, asyncpg/PostgreSQL, the repository AES-GCM webhook key ring, generic Jobs/WorkerSDK, httpcore-based `Security/http_hop.py`, pytest, Ruff, Bandit.

**Spec:** `Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md`

**Backlog task:** `TASK-13111`

**Dependency:** Satisfied. PR #2806 / `TASK-13014` merged into `dev` as `24f79419061ba85e9273b38a05431d6fd46ca40f`; tracking PR #2828 then merged as `9fd2246157ce8a32ae6a6691a75efab788229f77`.

## Global Constraints

- This plan implements only upstream PR 2, "Delivery Substrate And Recovery." Durable user/incident producers, incident file-marker capture, final legacy-handler deletion, final canonical route activation, operational admin UI completion, and public release remain PR 3 work.
- PR 2 implementation starts from a fresh branch containing final reviewed PR 1 head `f37d4c448ace69b56e208ca1f9bda94c571d86f3` and the merged `TASK-13014` closeout. Before implementation, fetch `origin/dev` and prove that reviewed head remains an ancestor.
- `TLDW_ADMIN_WEBHOOKS_MODE` continues to default to `off`. Tests may construct mode `on` explicitly. No deployment or release configuration enables canonical mode in PR 2.
- `tldw_Server_API/app/services/admin_webhooks_service.py` and `tldw_Server_API/app/services/jobs_webhooks_service.py` are unrelated legacy implementations. Canonical modules must not import them, reuse their direct SQL, raw/generic HTTP behavior, global URL, cursor file, or outbox semantics. They remain isolated until PR 3 removes temporary compatibility behavior.
- Jobs payloads contain exactly `{"delivery_id":"<opaque delivery UUID>"}`. Jobs rows never contain target URLs, secrets, event bodies, signatures, response bodies, or ordinary response headers.
- Canonical Jobs identity is fixed to domain `admin_webhooks`, queue `delivery`, job type `admin_webhook_delivery`, and idempotency key `admin-webhook-delivery:<delivery_id>`. The queue is registered in `JobManager.DOMAIN_ALLOWED_QUEUES`; canonical code uses typed admission and lookup-only facades rather than interpreting `create_job()` exceptions. Typed admission and legacy `create_job()` share one complete validation, transformation, persistence, metric, event/audit, gauge, and invariant pipeline; only terminal result mapping differs.
- AuthNZ owns event, delivery, attempt, idempotency, and component-heartbeat SQL. The router owns no SQL; the repository performs no HTTP; the worker accepts no caller payload or secret from Jobs.
- Event bodies are deterministic compact UTF-8 JSON, reject NaN/Infinity, are at most 65,536 decrypted bytes, and are encrypted with `WebhookKeyRing.encrypt_event_body()` before persistence. Exact body bytes are reused for every delivery of one event.
- Event, delivery, and attempt IDs are lowercase canonical UUID strings generated with `uuid.uuid4()` by the service before their transaction. Claim, test-attempt, and disposition tokens are independent 32-random-byte lowercase hex values and are never reused across purposes.
- Event and matching automatic deliveries commit in one AuthNZ transaction. Fanout performs one bounded matching-registration query and one batch insert, not one registration query or transaction per delivery. The active-registration hard ceiling remains 1,000.
- Automatic delivery uniqueness is `(event_id, webhook_id)` only for `kind='automatic'`. Manual and test deliveries use new opaque IDs and do not weaken that partial unique index.
- AuthNZ delivery states remain exactly `pending`, `enqueue_claimed`, `queued`, `processing`, `retry_wait`, `succeeded`, `dead`, `canceled`, and `superseded`. Terminal states are monotonic.
- Attempt states remain exactly `processing`, `succeeded`, `retryable`, `failed`, `canceled`, `superseded`, and `outcome_unknown`. Attempt rows are append-only after their one conditional terminalization; they are never deleted independently or rewritten to claim an ambiguous request was unsent.
- One initial network attempt plus three retries is a hard four-attempt I/O ceiling. `outcome_unknown` consumes a slot. The webhook Jobs job uses `max_retries=3` and a validated quarantine threshold of 5, above the four-attempt safety cap.
- Jobs is the only automatic retry clock. Retry delays are exactly 60, 300, and 1,800 seconds, except a valid 429/503 `Retry-After` may increase the current delay up to 1,800 seconds. AuthNZ recovery reuses the already-recorded delay and never runs a second exponential calculation.
- Automatic and manual queued work expires after 72 hours. Before attempt reservation, both remaining Jobs lease and remaining delivery lifetime must exceed the registration timeout plus a 30-second terminal-commit margin. A failed horizon check sends nothing and consumes no attempt.
- A processing attempt becomes recoverable only after `started_at + timeout_seconds + 90 seconds`. A replacement lease defers without an attempt until that time, then records `outcome_unknown` and a prepared retry or terminal disposition; it never immediately sends.
- The enqueue handshake is AuthNZ claim, idempotent Jobs create/read with key `admin-webhook-delivery:<delivery_id>`, conditional Jobs-ID attach, then `queued`. Every crash point must converge without duplicate automatic delivery or duplicate Jobs row.
- Every post-attempt or cancellation Jobs disposition has a random opaque application token persisted in AuthNZ before Jobs mutation. Retry/defer also persist the absolute original not-before timestamp; retry retains its bounded operator-visible delay. Jobs records a bounded reserved result marker containing only token, kind, delivery ID, attempt ID when present, and the same not-before timestamp when scheduled. A repeated token is idempotent and reuses that absolute schedule rather than starting a new delay; no token, target, payload, or secret appears in logs.
- One narrow exception handles AuthNZ loss after Jobs acquisition but before attempt reservation: the prepared worker applies an infrastructure-only Jobs `defer` with a fresh token and no caller-computed schedule, leaves AuthNZ `queued` unchanged, and emits no acknowledgement. This path is legal only while no attempt is reserved, sends no HTTP, and changes no retry/failure/quarantine counter, so loss/reapplication cannot misrepresent or duplicate a request.
- Prepared dispositions are exactly `complete`, `retry`, `fail`, `cancel`, and no-attempt `defer`. `WorkerSDK` applies one prepared disposition under the current lease instead of also invoking its default completion/failure path.
- Acquisition preflight exceptions fail closed for the new prepared-worker path. Infrastructure failure after acquisition returns a bounded `defer` without increasing attempt, retry, failure-streak, or quarantine counters.
- Every infrastructure-failure no-attempt defer carries no absolute schedule. On first application, the Jobs backend atomically calculates `not_before_at = database_now + 30 seconds`, persists it in the reserved-result marker, and schedules the row from that value; exact-token replay returns and reuses the stored timestamp. Stale-attempt recovery is a separate no-acknowledgement defer carrying the explicit AuthNZ-derived deterministic stale timestamp, never an infrastructure-delay calculation.
- Existing `WorkerSDK.run()` behavior and existing workers remain backward compatible. Canonical delivery uses a new typed prepared-worker entry point and lease context rather than changing existing handler arity or making current fail-open guards globally fail closed.
- Delivery attempts use `Security/http_hop.py`; canonical code never creates raw `httpx` clients. Status-only mode repeats URL policy and DNS checks, pins an approved peer, preserves hostname TLS/Host semantics, ignores ambient proxies, disables redirects, bounds headers/time, does not iterate or buffer the response body, and exposes only status, latency, and parsed bounded `Retry-After`.
- Full target URL, secret, body, signature, ordinary receiver headers/body, exception text, and incident narrative never enter delivery history, Jobs, metrics, audit, logs, or API responses. Dataclasses holding decrypted target/secret/body use `repr=False`.
- Success is any 2xx. Network errors, timeout, 408, 429, and 5xx retry automatically; redirects and other 4xx fail terminally. Tests perform one attempt and never use Jobs retry APIs.
- Disable, rotation, delivery-configuration mutation, and delete conditionally terminate work that has not crossed the pre-I/O reservation boundary and persist a cancel disposition for attached Jobs work. They do not overwrite a real in-flight outcome; a success after a configuration race remains success with `completed_after_config_change=true`.
- Synchronous test claims idempotency, validates ETag/reviewed versions, inserts a `webhook.test` event, test delivery, and attempt sequence one directly in `processing`, and marks first canonical activity in one transaction. It creates no Jobs job and can never retry implicitly.
- Manual redelivery references an existing historical event, snapshots current active configuration, requires ETag plus reviewed delivery configuration, requires explicit changed-configuration confirmation when applicable, creates a new delivery, and enters the same enqueue handshake.
- Exact test/redelivery idempotency replay is evaluated before current preconditions. Same-key/different-request conflicts; a processing test replay returns 202 without I/O; a terminal replay returns bounded stored metadata without decryption or I/O.
- Delivery history exposes stable IDs, kinds, states, attempt counts, status class/code, latency, bounded reason codes, version snapshots, timestamps, expiry, and redelivery linkage only. It never exposes event body, URL, secret, signature, receiver body, or ordinary headers.
- Runtime health is durable across separate API/worker processes. An additive AuthNZ runtime-heartbeat table stores bounded component/instance IDs, readiness, stable reason code, and timestamps; it stores no hostnames, URLs, payloads, secrets, or free text.
- Terminal deliveries and dependent events are retained for 30 days. Retention uses bounded batches, removes expired idempotency rows, never deletes nonterminal work, and does not purge registration tombstones until existing admission-bound eligibility is satisfied.
- Metrics use only closed low-cardinality labels: state, kind, event type, reason code, status class, component, and backend. Webhook/delivery/event/attempt IDs and target hostname are log fields where needed, never metric labels.
- Synthetic event capture and the first delivery attempt use `mark_first_canonical_activity()` transactionally. Manual-redelivery creation also closes the rollback window in its creation transaction using the schema's existing closed `delivery_attempt` activity category before enqueue; it does not insert an attempt row at that point. Replays, conflicts, rejected requests, no-attempt deferrals, and recovery acknowledgements do not create a new activity marker.
- SQLite and PostgreSQL must have equivalent constraints and repository behavior. AuthNZ/Jobs recovery tests cover SQLite/SQLite, SQLite/PostgreSQL, PostgreSQL/SQLite, and PostgreSQL/PostgreSQL with zero PostgreSQL skips in the required gate. Every pytest command runs with `RUN_JOBS=1` so Jobs coverage cannot silently skip.
- All implementation commits update `TASK-13111`. Touched Python passes focused pytest, Ruff, Bandit, and `git diff --check`; the final evidence file records exact counts and environment.
- Start implementation in a fresh isolated worktree from current `origin/dev`; never copy or stage unrelated files from older worktrees.

---

## Delivery Stages

1. Add the missing durable recovery token/heartbeat schema and freeze delivery contracts.
2. Implement equivalent event, delivery, attempt, claim, disposition, heartbeat, and retention repositories.
3. Add supported Jobs prepared-disposition and lease-horizon operations without regressing existing workers.
4. Build status-only egress, deterministic signing, and the shared one-attempt executor.
5. Implement synthetic capture, lifecycle cancellation, enqueue/disposition recovery, and the Jobs worker.
6. Expose synchronous test, manual redelivery, delivery history, health, and bounded runtime services.
7. Prove all backend/crash/security gates, document operation, and submit one PR 2 review unit.

## File Map

**Create**

- `tldw_Server_API/app/core/Admin_Webhooks/delivery.py` - delivery records, synthetic capture, test/redelivery commands, history, and lifecycle orchestration without HTTP or Jobs SQL.
- `tldw_Server_API/app/core/Admin_Webhooks/executor.py` - deterministic body/signature headers, retry classification, and one bounded network attempt.
- `tldw_Server_API/app/core/Admin_Webhooks/reconciler.py` - enqueue claims, Jobs attach, pending-disposition/cancellation repair, stale-attempt recovery, expiry, and lost-ack convergence.
- `tldw_Server_API/app/core/Admin_Webhooks/worker.py` - prepared Jobs handler, preflight, attempt reservation, lease horizon, executor invocation, and AuthNZ outcome commit.
- `tldw_Server_API/app/core/Admin_Webhooks/observability.py` - closed metric adapter, durable component heartbeats, health aggregation, and sanitized status snapshot.
- `tldw_Server_API/app/services/admin_webhook_delivery_runtime.py` - stop-event runtime supervising worker, reconciler, heartbeat, and retention loops.
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_enqueue_reconciler.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_executor.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_worker.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py`
- `tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py`
- `tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py`
- `tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py`
- `tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py`
- `Docs/Admin_Webhooks_Delivery_Runbook.md` - worker, reconciler, backlog, dead delivery, test, redelivery, disable, and retention operations available in PR 2.
- `Docs/Evidence/Admin_Webhooks_PR2_Verification.md` - exact PR 2 gate evidence.

**Modify**

- `tldw_Server_API/app/core/Admin_Webhooks/__init__.py` - export only reviewed delivery public types.
- `tldw_Server_API/app/core/Admin_Webhooks/domain.py` - stable delivery/test/redelivery errors and status records.
- `tldw_Server_API/app/core/Admin_Webhooks/config.py` - fixed protocol invariants plus bounded runtime cadence/claim/heartbeat settings.
- `tldw_Server_API/app/core/Admin_Webhooks/control_plane.py` - transactional cancel/supersede integration and real delivery-capability health.
- `tldw_Server_API/app/core/Admin_Webhooks/crypto.py` - retain contextual event-body helpers and add only validation needed by persisted delivery reads.
- `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py` - all PR 2 AuthNZ SQL and unit-of-work compare-and-set operations.
- `tldw_Server_API/app/core/AuthNZ/migrations.py` - additive SQLite migration 095 for pending-disposition tokens/not-before timestamps, per-attempt request timeout, runtime heartbeats, and recovery indexes.
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` - equivalent idempotent PostgreSQL ensure path and indexes.
- `tldw_Server_API/app/core/Jobs/operations/contracts.py` - typed prepared dispositions, exact delay/defer, lease horizon, and results.
- `tldw_Server_API/app/core/Jobs/migrations.py` - additive/default-compatible SQLite per-job lease-recovery policy and quarantine threshold.
- `tldw_Server_API/app/core/Jobs/pg_migrations.py` - equivalent PostgreSQL Jobs schema extension.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py` - persist validated per-job controls at admission.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py` - SQLite prepared disposition and lease-horizon transitions.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py` - export supported SQLite operations.
- `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py` - persist validated per-job controls at admission.
- `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py` - PostgreSQL prepared disposition and lease-horizon transitions.
- `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py` - export supported PostgreSQL operations.
- `tldw_Server_API/app/core/Jobs/manager.py` - backend-neutral facade for exact prepared disposition and observable lease horizon.
- `tldw_Server_API/app/core/Jobs/worker_sdk.py` - backward-compatible prepared-worker loop and lease context.
- `tldw_Server_API/app/core/Security/http_hop.py` - status-only response mode on the existing peer-verified transport.
- `tldw_Server_API/app/api/v1/schemas/admin_webhooks.py` - test, redelivery, delivery-history, attempt, and health schemas.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py` - canonical test/redelivery/history routes and expanded sanitized status.
- `tldw_Server_API/app/services/startup_optional_workers.py` - declarative canonical delivery runtime worker spec enabled only by validated canonical mode/runtime configuration.
- `tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py` - 094-to-095 upgrade and fresh-install assertions.
- `tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py` - equivalent PostgreSQL upgrade/fresh ensure assertions.
- `tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py` - cancellation/supersession/config-race and activation-health behavior.
- `tldw_Server_API/tests/Admin_Webhooks/test_api.py` - canonical error, authorization, audit, cache, idempotency, and static-route ordering coverage.
- `tldw_Server_API/tests/Admin_Webhooks/test_openapi.py` - reviewed PR 2 schema/route delta.
- `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py` - typed invariant tests.
- `tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py` - fresh/upgrade Jobs control columns.
- `tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py` - equivalent PostgreSQL Jobs migration.
- `tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py` - preserve existing SQLite auxiliary schema/index compatibility while extending jobs.
- `tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py` - equivalent PostgreSQL compatibility assertions.
- `tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py` - default behavior plus no-attempt policy.
- `tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py` - equivalent PostgreSQL lease recovery.
- `tldw_Server_API/tests/Jobs/test_worker_sdk.py` - regression proof for unchanged legacy `run()` behavior.
- `tldw_Server_API/tests/Security/test_http_hop_contract.py` - status-only public contract.
- `tldw_Server_API/tests/Security/test_http_hop_transport.py` - DNS pin/TLS/redirect/proxy behavior.
- `tldw_Server_API/tests/Security/test_http_hop_streaming.py` - prove status-only closes without body iteration/buffering.
- `tldw_Server_API/tests/Services/test_startup_optional_workers.py` - mode-gated runtime registration and shutdown.
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json` - record the reviewed canonical PR 2 API delta.
- `backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md` - execution notes, evidence, blockers, and PR state.

### Task 0: Merged Dependency Gate, Baseline, And Task Activation

**Files:**
- Modify: `backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md`

**Interfaces:**
- Consumes: merged PR #2806, current `origin/dev`, approved design, and this plan.
- Produces: a clean implementation branch based on current `origin/dev` with `TASK-13111` In Progress and baseline evidence.

- [ ] **Step 1: Prove PR 1 is merged before touching implementation files**

```bash
gh pr view 2806 --repo rmusser01/tldw_server --json state,mergedAt,mergeCommit,url
gh pr view 2828 --repo rmusser01/tldw_server --json state,mergedAt,mergeCommit,url
git fetch origin dev
git merge-base --is-ancestor f37d4c448ace69b56e208ca1f9bda94c571d86f3 origin/dev
```

Expected: both PR states are `MERGED`, `mergedAt` and `mergeCommit` are non-null, and `merge-base` exits 0. If any assertion fails, stop before touching runtime code.

- [ ] **Step 2: Create a fresh implementation branch and verify migration allocation**

```bash
REPO_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
git -C "$REPO_ROOT" check-ignore -q .worktrees
git -C "$REPO_ROOT" worktree add \
  "$REPO_ROOT/.worktrees/admin-webhooks-delivery-substrate" \
  -b codex/admin-webhooks-delivery-substrate origin/dev
cd "$REPO_ROOT/.worktrees/admin-webhooks-delivery-substrate"
git log -1 --format='%H %s'
git status --short
rg -n "migration_09[0-9]|Migration 09[0-9]" tldw_Server_API/app/core/AuthNZ/migrations.py
```

Expected: the implementation branch equals current `origin/dev`, the worktree is clean, migration 094 is the latest allocation, and migration 095 is unused. If another merged change claims 095, update this plan, both migration test files, and `TASK-13111` to the next free number before implementation.

- [ ] **Step 3: Mark the task active and attach this plan**

```bash
backlog task edit 13111 -s "In Progress" --plan $'1. Add durable recovery/health schema and delivery contracts.\n2. Implement AuthNZ persistence and Jobs prepared dispositions.\n3. Add status-only egress, executor, reconciler, and worker.\n4. Expose test, redelivery, history, retention, metrics, and health.\n5. Run all backend, crash, security, and review gates.\nDetailed plan: Docs/superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md'
backlog task 13111 --plain
```

Expected: `TASK-13111` is In Progress, depends on completed `TASK-13014`, and links the design and detailed plan.

- [ ] **Step 4: Run the pre-change baseline**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py
```

Expected: record exact pass/fail/skip counts in `TASK-13111`. PostgreSQL skips are acceptable only for this local baseline; the final required PostgreSQL gate permits zero skips.

- [ ] **Step 5: Commit only the activated task metadata if it changed**

```bash
git add "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "chore(backlog): start admin webhook delivery task"
```

### Task 1: Freeze Delivery Contracts And Add Recovery/Heartbeat Schema

**Files:**
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/domain.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/config.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/__init__.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py`

**Interfaces:**
- Consumes: PR 1 schema version 1 and registration contracts.
- Produces: stable delivery enums/records/settings plus additive migration 095 with `pending_jobs_disposition_token`, `pending_jobs_disposition_not_before_at`, `admin_webhook_delivery_attempts.request_timeout_seconds`, and durable per-instance runtime heartbeat rows. The canonical migration-state `schema_version` remains 1 because PR 1 intentionally constrains that full canonical schema contract to 1; PR 2 readiness probes its additive extension explicitly.

- [ ] **Step 1: Write failing domain/config tests**

Cover exact enum values, terminal-state sets, retry schedule `(60, 300, 1800)`, max attempts 4, quarantine threshold 5, fixed infrastructure-defer delay 30 seconds, 72-hour expiry, 30-day retention, 30-second commit margin, 90-second stale-attempt margin, claim TTL default 60 seconds bounded 5-300, loop interval default 1 second bounded 1-60, heartbeat interval default 10 seconds bounded 1-60, heartbeat freshness default 30 seconds and strictly greater than the interval, and invalid booleans/integers without leaking raw environment values.

```python
def test_delivery_protocol_invariants_are_not_operator_expandable() -> None:
    settings = AdminWebhookSettings.from_environment({})
    assert settings.delivery_retry_delays_seconds == (60, 300, 1800)
    assert settings.delivery_max_attempts == 4
    assert settings.jobs_quarantine_threshold == 5
    assert settings.delivery_expiry_seconds == 72 * 60 * 60
    assert settings.delivery_retention_days == 30
```

- [ ] **Step 2: Write failing SQLite and PostgreSQL migration tests**

Assert fresh install and 094 upgrade produce:

```text
admin_webhook_deliveries.pending_jobs_disposition_token TEXT NULL
admin_webhook_deliveries.pending_jobs_disposition_not_before_at TEXT NULL
admin_webhook_delivery_attempts.request_timeout_seconds INTEGER NULL
admin_webhook_runtime_heartbeats(
  component, instance_id, ready, reason_code,
  heartbeat_at, last_success_at, created_at, updated_at,
  PRIMARY KEY(component, instance_id)
)
```

New PR 2 attempt rows require timeout 1-30; nullable upgrade rows are tolerated only for a database that somehow contains pre-PR-2 attempts and recover conservatively with the fixed 30-second maximum. `component` accepts only `worker`, `reconciler`, or `retention`; IDs/reasons are bounded; `ready` is boolean-equivalent; timestamps are required as specified. Add equivalent recovery indexes for pending/enqueue-claimed/disposition work and heartbeat freshness. Test rerun idempotence, no row loss, and preservation of migration-state `schema_version=1`. SQLite 094 data must survive 095; PostgreSQL ensure must work on both empty and existing schemas.

- [ ] **Step 3: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py
```

Expected: failures identify missing delivery contracts, migration 095, disposition token/not-before/attempt-timeout fields, heartbeat table, and explicit PR 2 schema-extension preflight.

- [ ] **Step 4: Implement closed delivery types and settings**

Add `DeliveryKind`, `DeliveryState`, `AttemptState`, `JobsDispositionKind`, `DeliveryReasonCode`, `EventSourceKind`, `WebhookEvent`, `WebhookDelivery`, `WebhookDeliveryAttempt`, `DeliveryHistoryPage`, `DeliveryRuntimeComponent`, `DeliveryRuntimeHeartbeat`, and `DeliveryHealthSnapshot`. Extend `WebhookErrorCode` only with fixed test/redelivery/history/recovery errors used by later tasks. Keep all sensitive byte/string fields out of these public metadata records.

Protocol constants are fixed properties, not arbitrary environment overrides. Only bounded operational cadence/claim/heartbeat values are configurable. `AdminWebhookSettings.from_environment()` remains deterministic from the supplied mapping.

- [ ] **Step 5: Implement additive migration 095 on both backends**

SQLite adds the nullable disposition and attempt-timeout columns after checking `PRAGMA table_info` and creates the heartbeat table/indexes transactionally. PostgreSQL uses `ADD COLUMN IF NOT EXISTS` plus equivalent checks/indexes. Neither backend rewrites `admin_webhook_migration_state` or changes its constrained canonical `schema_version=1`. Add `AdminWebhookRepository.delivery_schema_ready()` to inspect the required column/table/index contract without leaking SQL into services. Because SQLite cannot safely add the new cross-column checks to an existing table, enforce and test repository invariants on both backends: pending disposition and token are both null or both non-null; retry/defer require an absolute not-before time; complete/fail/cancel require it null; only retry uses the existing bounded delay column; every newly reserved attempt has timeout 1-30.

- [ ] **Step 6: Run GREEN and static checks**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/Admin_Webhooks/config.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py
```

- [ ] **Step 7: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Added migration-095 recovery token/runtime heartbeat contract while preserving canonical schema version 1; focused migration/domain tests pass on recorded backends."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/Admin_Webhooks/config.py \
  tldw_Server_API/app/core/Admin_Webhooks/__init__.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/tests/Admin_Webhooks/test_domain_config_catalog.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): define delivery recovery schema"
```

### Task 2: Implement Event, Delivery, Attempt, And Runtime Repositories

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/crypto.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`

**Interfaces:**
- Consumes: canonical schema version 1 plus the verified migration-095 delivery extension, `WebhookKeyRing`, active registration snapshots, generated opaque IDs, and a transaction clock.
- Produces: backend-equivalent repository/UoW methods for encrypted capture/fanout, history, enqueue claims, attempt reservation/outcome, disposition tokens, heartbeat, expiry, and retention.

- [ ] **Step 1: Write a shared repository contract suite and thin backend fixtures**

Place backend-neutral assertions in helpers imported by both SQLite and PostgreSQL test modules. Define and test these exact repository records:

```text
EventInsert
StoredWebhookEvent
EventCaptureResult
StoredWebhookDelivery
DeliveryBundle
EnqueueClaim
AttemptReservation
AttemptCompletion
PendingJobsDisposition
RuntimeHeartbeatWrite
RetentionBatchResult
```

Cover canonical timestamp normalization, opaque ID bounds, enum decoding, malformed-row failure, disposition token/not-before invariants, encrypted-body identity binding, 65,536-byte acceptance, 65,537-byte rejection before SQL, duplicate aggregate/command source returning the existing event/fanout rather than duplicating it, and cross-event envelope substitution rejection.

The UoW entry point is `capture_event_and_expand(event, delivery_id_factory, expires_at)`. The service supplies a UUID factory; after one matching-registration query, the repository calls it once per matched registration and performs one batch insert. A source-conflict path calls it zero times.

- [ ] **Step 2: Write fanout and history RED tests**

Instrument the connection adapter to prove `capture_event_and_expand()` performs one matching-registration query and one batch insert for 25 active matches. Assert inactive, deleted, unsubscribed, and `secret_rotation_required` registrations are excluded; each delivery snapshots current delivery/secret versions and expires at `created_at + 72h`; duplicate producer calls return the same event and automatic rows. Verify manual/test rows can coexist with the automatic partial unique key.

History tests require deterministic `created_at DESC, id DESC` ordering, pagination bounds, append-only attempt ordering, no ciphertext/body/URL/secret fields in returned metadata, and registration-scoped lookup that cannot read another registration's delivery.

- [ ] **Step 3: Write claim/reservation/disposition/recovery RED tests**

Cover the exact compare-and-set methods:

```text
claim_pending_delivery(claim_token, claimed_until, now)
attach_jobs_job(delivery_id, claim_token, jobs_job_id, now)
release_expired_enqueue_claim(delivery_id, expected_token, now)
reserve_jobs_attempt(delivery_id, jobs_job_id, lease_id, attempt_id, request_timeout_seconds, now, required_horizon)
reserve_test_attempt(test_attempt_token, delivery_id, attempt_id, request_timeout_seconds, started_at)
finish_attempt_and_prepare_disposition(attempt_token, outcome, disposition_token, not_before_at)
acknowledge_jobs_disposition(delivery_id, disposition_token, jobs_state)
close_stale_attempt_as_unknown(delivery_id, attempt_id, stale_before)
cancel_registration_work(webhook_id, cutoff_versions, reason, disposition_token_factory, now)
expire_delivery(delivery_id, expected_state, now)
upsert_runtime_heartbeat(write)
purge_retained_rows(now, retention_cutoff, batch_size)
```

Prove stale claim/lease/attempt/disposition tokens lose without mutation; attempt numbers are monotonic 1-4; attempt five is rejected as `attempt_budget_exhausted`; each new attempt persists the exact reviewed timeout 1-30; stale recovery uses that persisted timeout (or conservative 30 only for nullable upgrade residue); acknowledgement of an attempt disposition marks both the exact delivery token and matching append-only attempt `jobs_disposition_applied`, while cancellation/defer without an attempt changes only the delivery; reservation marks `first_canonical_activity_kind='delivery_attempt'` only when no earlier activity exists; no-attempt paths do not mark it; terminal delivery states cannot regress; and pending disposition/token nullability is enforced in application code on both backends.

- [ ] **Step 4: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py
```

Expected: collection or attribute failures for the missing PR 2 repository contracts.

- [ ] **Step 5: Implement one repository API over backend-specific adapters**

Keep all SQL in `admin_webhooks_repository.py`. Reuse `_ConnectionAdapter` parameter conversion and transaction ownership. Matching subscriptions use one backend-specific set query (`json_each` on SQLite; JSONB/JSON containment on PostgreSQL) and one bounded batch insert. Reads select explicit columns rather than `SELECT *`. Mutations always include state/token/version predicates and verify affected rows.

The repository treats ciphertext as a bounded `ProtectedValue` and never owns key-ring or decryption logic. Its UoW inserts the event and deliveries and marks first canonical activity as `event_capture` in the same transaction. On a source-unique conflict it returns the existing protected event plus existing fanout without generating rows; the delivery service in Task 6 contextually decrypts and verifies that existing event before treating the operation as an idempotent replay.

- [ ] **Step 6: Implement bounded readback, retention, and runtime-health queries**

Retention order is expired idempotency, eligible terminal deliveries, now-orphaned events, stale heartbeat instances, then existing eligible registration tombstones. Batch size is bounded to 200. Event deletion requires every dependent delivery terminal and the latest terminal time older than 30 days. No nonterminal row is selected or cascaded.

- [ ] **Step 7: Run GREEN, backend parity, and SQL-sensitive checks**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/Admin_Webhooks/crypto.py
```

- [ ] **Step 8: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Implemented dual-backend encrypted event/fanout, delivery/attempt CAS, disposition, heartbeat, history, expiry, and retention repositories; parity tests recorded."
git add \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/Admin_Webhooks/crypto.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): persist delivery state and attempts"
```

### Task 3: Add Supported Jobs Prepared-Disposition Operations

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Modify: `tldw_Server_API/app/core/Jobs/migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/pg_migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py`

**Interfaces:**
- Consumes: one canonical Jobs row, current worker/lease identity for processing transitions, an opaque disposition token, closed disposition kind, and exact bounded schedule. A trusted canonical reconciler may apply tokenized cancel only to queued work without a lease.
- Produces: typed, backend-neutral lease-horizon and prepared-disposition operations with identical SQLite/PostgreSQL behavior and bounded durable result evidence.

- [ ] **Step 1: Write contract RED tests for these public types**

```text
PreparedDispositionKind(COMPLETE, RETRY, FAIL, CANCEL, DEFER)
PreparedDispositionOrigin(AUTHNZ, INFRASTRUCTURE, RECOVERY)
ExpiredLeasePolicy(CONSUME_RETRY, REQUEUE_NO_ATTEMPT)
PreparedJobDisposition
ApplyPreparedDispositionCommand
PreparedDispositionResult
EnsureLeaseHorizonCommand
LeaseHorizonResult
FindJobByIdentityCommand
JobIdentityLookupState(ACTIVE, ARCHIVED, MISSING, CONFLICT)
JobIdentityLookupResult
```

Validate token length/charset, delivery/attempt ID bounds, allowed fields by kind, delay 1-1800 only for retry, stable reason code bounds, deep-copy/frozen result metadata, and no arbitrary exception text. AuthNZ retry requires an absolute timezone-aware `not_before_at`; `origin=INFRASTRUCTURE` is legal only for a no-attempt defer with no caller-provided timestamp and never requests an AuthNZ acknowledgement; `origin=RECOVERY` is legal only for a no-attempt defer with an explicit timezone-aware AuthNZ-derived stale timestamp and also requests no acknowledgement. Complete/retry/fail/cancel use `origin=AUTHNZ`. `CreateJobCommand` gains `expired_lease_policy=CONSUME_RETRY` and nullable positive `quarantine_threshold`; existing callers retain those defaults. Prepared retry reads the threshold captured on the Jobs row rather than accepting a conflicting per-attempt override. Factory methods are `complete()`, `retry()`, `fail()`, `cancel()`, `infrastructure_defer()`, and `recovery_defer_until()`.

- [ ] **Step 2: Write additive Jobs schema/admission RED tests**

Fresh and upgraded SQLite/PostgreSQL jobs tables add:

```text
expired_lease_policy = 'consume_retry' | 'requeue_no_attempt' NOT NULL DEFAULT 'consume_retry'
quarantine_threshold INTEGER NULL, positive when present
```

Existing rows backfill the default and preserve current behavior. Admission persists requested controls atomically and idempotent-existing create verifies they match. Canonical enqueue uses `requeue_no_attempt` and threshold 5; invalid policies/thresholds fail before SQL. Archive need not copy these nonterminal controls, but terminal `result` must retain prepared-disposition proof.

Register `delivery` as the built-in queue for domain `admin_webhooks`. Refactor one shared internal admission pipeline used by both public facades. It performs the existing queue/job-type policy, fair-share checks and priority transformation, secret scan/redaction/rejection, optional payload encryption, JSON size handling, trace/UUID initialization, backend admission, metrics, durable event/audit emission, gauges, and invariant checks exactly once. `JobManager.admit_job()` accepts the same validated keyword contract as `create_job()` plus the two persisted execution controls and returns the existing `AdmissionResult`; `create_job()` delegates to it and preserves its current dict return and exception mapping. Add parity tests proving both paths perform identical validation, transformations, inserted/existing side effects, and invariant checks, with typed `ADMISSION_REJECTED` as the only mapping difference and no side effects on rejection. Add lookup-only active/archive queries by exact domain, queue, job type, and idempotency key. Missing lookup returns `MISSING` without inserting a row; multiple or identity-inconsistent matches return `CONFLICT`.

Extend the existing migration-compatibility assertions rather than replacing them: all auxiliary Jobs tables/indexes remain present, rerunning ensure functions remains idempotent, and old rows/callers observe the exact default policy and null threshold on both backends.

- [ ] **Step 3: Write backend RED tests for exact state transitions**

For both backends prove:

- complete: `processing -> completed` under matching lease;
- retry: `processing -> queued`, `retry_count += 1`, `available_at = max(database_now, original_not_before_at)`, and no generic backoff/jitter;
- fail: `processing -> failed` without retry;
- cancel: matching-lease `processing -> cancelled`, or trusted canonical reconciler `queued -> cancelled` without a lease, with stable reason;
- recovery defer: `processing -> queued` at `max(database_now, explicit_stale_not_before_at)` with retry count, failure streak, and quarantine counters unchanged;
- first application of `infrastructure_defer` atomically computes, persists, and schedules exactly `database_now + 30 seconds`; caller/application clocks cannot influence it, and exact-token replay reuses that stored absolute timestamp;
- `recovery_defer_until` schedules from its explicit AuthNZ-derived stale timestamp without substituting the infrastructure delay and retains that same timestamp on exact-token replay;
- normal retryable attempts do not quarantine before the AuthNZ four-attempt cap when threshold is 5;
- same token/kind/delivery is idempotent and returns `already_applied=true`;
- lookup-only identity returns the one matching active or archived job and never creates work;
- a previously applied AuthNZ retry token observed after the job has been reacquired under a newer processing lease reports the prior application plus current processing state and performs no transition; canonical pre-reservation recovery must acknowledge that token and continue the current lease rather than treating it as finalization;
- a previously applied infrastructure/recovery defer marker observed after reacquisition is historical no-acknowledgement evidence: canonical handling never invokes an AuthNZ callback, never reapplies it, and continues only under the current lease; it does not conflict with a later exact AuthNZ disposition for that current lease;
- same token with different facts is a backend conflict;
- stale worker/lease cannot apply a new token;
- unleased complete/retry/fail/defer is always rejected; unleased cancel cannot terminalize processing work and may touch only the expected canonical domain/queue/type/payload;
- reserved result evidence contains only schema version, token, kind, origin, delivery ID, optional attempt ID, optional original not-before timestamp, and applied timestamp;
- counters/outbox events reflect the one real Jobs transition exactly once.

`ensure_lease_horizon()` must atomically extend but never shorten a matching processing lease, return the observed `leased_until`, reject stale leases, and obey `JOBS_LEASE_MAX_SECONDS`.

Expired lease recovery for `requeue_no_attempt` changes `processing -> queued`, clears lease/acquisition fields, and leaves retry count, failure streak, quarantine state, and max retries unchanged. The ordinary acquisition path and integrity sweeper both honor the persisted policy. Existing/default `consume_retry` behavior and its current events/counters remain byte-for-byte compatible.

- [ ] **Step 4: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py
```

- [ ] **Step 5: Implement schema, admission, and lifecycle operations atomically**

Add the Jobs columns through existing idempotent migration conventions and thread them through the shared facade pipeline before backend admission. Do not route prepared retry through the existing generic exponential calculation. Each backend transaction locks/reads the current row, recognizes an already-applied reserved marker, validates the current lease for a new token, performs one state/counter/outbox update, and returns typed evidence. On a new infrastructure-defer token, that transaction obtains the Jobs database timestamp, derives the one absolute 30-second schedule, and stores it with the marker; it never accepts an application-clock substitute. PostgreSQL uses database timestamps and row locking; SQLite uses one immediate transaction and its injected database-clock convention already used by lifecycle operations. Refactor expired-lease recovery narrowly enough that acquisition and integrity-sweep callers use the persisted row policy; do not special-case the `admin_webhooks` domain in SQL. Implement lookup-only identity through the same backend operation boundary, including archived lookup and conflict detection, without reusing create as a probe.

- [ ] **Step 6: Add narrow `JobManager` facades**

Expose:

```text
admit_job(
    *,
    domain: str,
    queue: str,
    job_type: str,
    payload: dict[str, Any],
    owner_user_id: str | None,
    project_id: int | None = None,
    batch_group: str | None = None,
    priority: int = 5,
    max_retries: int = 3,
    available_at: datetime | None = None,
    idempotency_key: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    expired_lease_policy: ExpiredLeasePolicy = CONSUME_RETRY,
    quarantine_threshold: int | None = None,
) -> AdmissionResult
find_job_by_identity(command: FindJobByIdentityCommand) -> JobIdentityLookupResult
ensure_lease_horizon(command: EnsureLeaseHorizonCommand) -> LeaseHorizonResult
apply_prepared_disposition(command: ApplyPreparedDispositionCommand) -> PreparedDispositionResult
```

The facade selects the backend, preserves RLS/domain checks, refuses prepared disposition for jobs outside the requested canonical domain/queue/type/payload, and does not expose `_connect()` to canonical webhook code. A single private pipeline backs both admission facades; `admit_job()` preserves typed created/existing/backend-rejected outcomes while `create_job()` only maps that result to its existing dict/exception contract. Pre-admission validation exceptions remain identical. No validation, transformation, encryption, limit, trace, metric, audit/event, gauge, or invariant behavior may diverge. `find_job_by_identity()` is read-only and fails closed on ambiguity. The unleased disposition form is accepted only for `cancel` against a currently queued canonical delivery job; every processing transition still requires matching worker and lease IDs.

- [ ] **Step 7: Run GREEN plus existing lifecycle regression**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py
../../.venv/bin/ruff check tldw_Server_API/app/core/Jobs
```

- [ ] **Step 8: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Added backend-neutral exact prepared disposition/no-attempt defer and observable lease-horizon operations with SQLite/PostgreSQL parity."
git add \
  tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/app/core/Jobs/migrations.py \
  tldw_Server_API/app/core/Jobs/pg_migrations.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/admission.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(jobs): support prepared worker dispositions"
```

### Task 4: Add A Backward-Compatible Prepared Worker Loop

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Create: `tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py`
- Modify: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`

**Interfaces:**
- Consumes: `PreparedJobHandler(job, WorkerExecutionContext)`, fail-closed async pre-acquire guard, closed handler-error disposition factory, existing `JobManager`, and prepared-disposition callbacks.
- Produces: `WorkerSDK.run_prepared()` with observable renewal loss, horizon enforcement, one typed finalizer, and unchanged `WorkerSDK.run()` semantics.

- [ ] **Step 1: Write RED tests for the new execution contract**

Define:

```text
WorkerLeaseSnapshot(worker_id, lease_id, leased_until, renewal_lost)
WorkerExecutionContext.snapshot()
WorkerExecutionContext.ensure_lease_horizon(seconds)
WorkerExecutionContext.renewal_lost
PreparedJobHandler
PreparedDispositionCallback
```

Test pre-acquire false and exception paths never call `acquire_next_job`; post-acquisition handler returns each disposition; SDK invokes `apply_prepared_disposition()` exactly once and never `complete_job()`/`fail_job()`; a rejected apply invokes only `on_disposition_rejected`; successful/idempotent `origin=AUTHNZ` apply invokes bounded `on_disposition_applied`; successful/idempotent infrastructure/recovery apply never invokes that callback; and callback timeout/error cannot trigger a second Jobs transition. A handler exception or malformed result calls the injected error-disposition factory with only the exception class, returns a timestamp-free `infrastructure_defer()`, lets `apply_prepared_disposition()` derive the absolute schedule from the Jobs database clock, and never serializes/logs exception text.

Test auto-renew updates the observable lease snapshot. Renewal false/exception sets `renewal_lost` permanently and stops renewal. `ensure_lease_horizon()` returns false on stale lease/backend failure without hiding it. Cancellation during handler still cancels renewal and exits cleanly.

- [ ] **Step 2: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py
```

- [ ] **Step 3: Implement `run_prepared()` over shared private loop primitives**

Keep existing one-argument `JobHandler` and `run()` public behavior intact. Add a separate two-argument prepared handler path. Pre-acquire guard exceptions fail closed only in `run_prepared()`. Require an actual `PreparedJobDisposition`; a `dict`, `None`, handler exception, or malformed value is passed to the required closed error-disposition factory, never treated as default success/failure. The canonical factory returns a timestamp-free infrastructure-only no-attempt defer; neither the SDK nor handler reads a database clock to construct it.

The prepared path starts renewal only after acquisition checks, passes one mutable-internal/read-only-public lease context, and applies exactly one prepared disposition. It runs one bounded acknowledgement callback only for `origin=AUTHNZ`; infrastructure and stale-recovery defers have no cross-database acknowledgement. It never invokes default completion/failure after a prepared handler result.

- [ ] **Step 4: Run GREEN and prove legacy behavior**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py
```

- [ ] **Step 5: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Added fail-closed prepared WorkerSDK path with observable renewal/horizon state and no default double finalization; legacy run regression remains green."
git add \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(jobs): add prepared worker execution path"
```

### Task 5: Add Status-Only Egress And The Shared Attempt Executor

**Files:**
- Modify: `tldw_Server_API/app/core/Security/http_hop.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/executor.py`
- Modify: `tldw_Server_API/tests/Security/test_http_hop_contract.py`
- Modify: `tldw_Server_API/tests/Security/test_http_hop_transport.py`
- Modify: `tldw_Server_API/tests/Security/test_http_hop_streaming.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_executor.py`

**Interfaces:**
- Consumes: one validated/decrypted target, exact encrypted-event plaintext bytes, signing secret, registration timeout, event/delivery/attempt metadata, and injected clock/egress callable.
- Produces: `request_http_hop_status()` and `DeliveryAttemptExecutor.execute()` returning bounded status/latency/reason/retry evidence only.

- [ ] **Step 1: Write status-only HTTP contract RED tests**

Add `StatusOnlyHTTPHopResponse` and `request_http_hop_status(request, *, resolver, clock)` contract tests. The request still uses `NormalizedHTTPHopRequest`; status-only is an explicit response mode or wrapper that cannot be selected accidentally by existing bounded-body callers.

Assert status-only results expose only `status_code`, `latency_ms`, and `retry_after_seconds`. They do not expose `headers`, `body`, response stream, target, or resolved peer details to webhook callers. Duplicate/malformed/non-ASCII `Retry-After` produces `None`; delta seconds and RFC-compliant dates use standard-library parsing and clamp to 1-1,800 seconds; only 429/503 may return it.

- [ ] **Step 2: Write transport/no-buffer RED tests**

Use a fake response stream whose iterator raises a unique canary exception. Status-only must return/close without invoking that iterator, even for large/chunked/compressed bodies. Header and raw parser limits still apply before return. Verify DNS pinning, connected-peer validation, hostname TLS/SNI, Host semantics, no redirects, ignored proxy environment, HTTP rejection except the already-validated non-production override, timeout bounded to 30 seconds, and closure on success/error/cancellation.

- [ ] **Step 3: Write executor RED tests with a published static vector**

Use this exact independent vector:

```text
secret: whsec_1111111111111111111111111111111111111111111111111111111111111111
timestamp: 1787443200
body: {"api_version":"2026-07-01","created_at":"2026-08-23T00:00:00Z","data":{"synthetic":true},"id":"00000000-0000-4000-8000-000000000001","type":"user.created"}
signature: v1=294bc280642cfd89fd011f606fbbe39633a77372db8ae9efd4281b2a3e509811
```

Assert exact headers, body bytes, test header only for `kind=test`, regenerated timestamp/signature per attempt, stable event/delivery IDs, constant deterministic body, and no header/body mutation by the egress adapter.

Cover classification: 2xx success; network/timeout/408/429/5xx retryable; 3xx and all other 4xx terminal. For retry attempt indexes 1-3, select 60/300/1,800 and raise only 429/503 delay to a valid bounded receiver value. Attempt 4 returns terminal `attempt_budget_exhausted`. Map every transport exception to a stable closed reason without exception text.

- [ ] **Step 4: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py
```

- [ ] **Step 5: Implement status-only mode by extending the existing one-hop transport**

Reuse `_PinnedBackend`, peer verification, TLS context, header guard, request validation, resolver, and total-timeout boundary. After final bounded response headers arrive, parse only permitted `Retry-After`, close the response context, and do not call `_read_decoded_body()` or expose `_response_headers()`. Existing `request_http_hop()` behavior and tests remain unchanged.

- [ ] **Step 6: Implement `DeliveryAttemptExecutor` as a one-I/O pure boundary**

`DeliveryAttemptExecutor` receives dependencies in its constructor and has no repository or Jobs access. It revalidates canonical URL syntax and `evaluate_platform_webhook_url_policy()` at attempt time, decrypts nothing itself, builds exact headers, calls status-only egress once, measures monotonic latency, and returns `AttemptExecutionResult`. Sensitive input dataclass fields use `repr=False`; logs are emitted by the caller with sanitized IDs/hostname and closed reason only.

- [ ] **Step 7: Run GREEN, security regressions, Ruff, and Bandit**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/core/Admin_Webhooks/executor.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/core/Admin_Webhooks/executor.py
```

- [ ] **Step 8: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Extended peer-verified HTTP hop with no-buffer status-only mode and added deterministic signed one-attempt executor; transport/security vectors pass."
git add \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/core/Admin_Webhooks/executor.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): execute status-only signed attempts"
```

### Task 6: Implement Synthetic Capture And Registration-Work Lifecycle

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/delivery.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/control_plane.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/__init__.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`

**Interfaces:**
- Consumes: validated synthetic event commands, registration mutation transactions, repository UoW, key-ring load result, mandatory audit sink, and injected IDs/clock.
- Produces: `AdminWebhookDeliveryService.capture_synthetic_event()` for tests/internal proof and transactional cancel/supersede behavior attached to control-plane mutations.

- [ ] **Step 1: Write RED tests for synthetic capture**

Define `CaptureSyntheticEventCommand` with server-owned event type/source identity/body; it is not an API schema and accepts no arbitrary receiver headers or target. Test aggregate and command source identities, deterministic canonical JSON, 64 KiB boundary, encryption/key-rotation gate, duplicate source replay, matching fanout, no-match event persistence, same-transaction rollback, `event_capture` first-activity marking, and mandatory accepted/failed audit with only event type, event ID, fanout count, actor, request ID, outcome, and reason code.

- [ ] **Step 2: Write RED tests for registration lifecycle effects**

Within the existing patch/rotate/delete transaction assert:

- active true to false terminates unstarted automatic/manual work as `canceled_disabled`;
- target/events/timeout changes supersede old-version unstarted work as `superseded_config`;
- secret rotation cancels old-version unstarted work as `canceled_secret_rotation`;
- soft delete cancels unstarted work as `canceled_deleted`;
- description-only/no-op/rejected/stale/replayed mutations do not touch deliveries;
- pending/enqueue-claimed work without Jobs becomes terminal directly;
- queued/retry-wait work with Jobs becomes terminal and records a cancel disposition/token;
- processing work remains processing; later success records `completed_after_config_change=true` and later pre-I/O checks select the specific lifecycle reason;
- mutation rollback also rolls back every delivery/disposition change.

- [ ] **Step 3: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py
```

- [ ] **Step 4: Implement delivery service composition and synthetic capture**

`AdminWebhookDeliveryService` owns key/precondition/idempotency/audit orchestration but delegates SQL to the repository and cryptography to `WebhookKeyRing`. `capture_synthetic_event()` is an internal service method used only by PR 2 tests and controlled composition; do not mount an arbitrary event-emission route or CLI. It resolves a writable primary key before transaction, encrypts exact bytes with event identity, invokes one UoW capture/fanout operation, and contextually decrypts/verifies any source-conflict event before accepting it as replay.

- [ ] **Step 5: Integrate conditional work lifecycle into control-plane transactions**

Extend `AdminWebhookControlPlane.patch()`, `rotate_secret()`, and `delete()` only after the effective mutation and version deltas are known. Generate cancellation disposition tokens before the transaction and reuse them on transaction retries. Preserve the existing accepted/no-op audit boundary and ETag/idempotency semantics.

- [ ] **Step 6: Run GREEN and existing control-plane regressions**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/control_plane.py
```

- [ ] **Step 7: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Added internal encrypted synthetic capture/fanout and transactional registration-work cancellation/supersession without producer or UI activation."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/control_plane.py \
  tldw_Server_API/app/core/Admin_Webhooks/__init__.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): capture synthetic events and retire stale work"
```

### Task 7: Implement The Recoverable Enqueue Handshake

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/reconciler.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_enqueue_reconciler.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py`

**Interfaces:**
- Consumes: AuthNZ repository, a narrow `JobsDeliveryQueue` adapter over typed `JobManager.admit_job()`, `find_job_by_identity()`, and known-ID reads, random claim token, injected clock, and one delivery ID.
- Produces: `AdminWebhookReconciler.reconcile_enqueue_once()` and idempotent attach/recovery for all AuthNZ/Jobs backend combinations.

**Implementation ruling:** The Task 2 repository surface cannot discover stale
enqueue claims, release a still-live claim after a transient Jobs failure,
terminalize a claimed identity conflict, or safely link and cancel a Jobs row
created during a terminal lifecycle race. Extend the repository without a
schema change. `claim_pending_delivery()` must atomically select pending rows
(including rows that have reached delivery expiry) or take over only expired
enqueue claims, preserve an already-terminal state during stale-claim takeover,
and order candidates by expiry, creation time, and ID. It must never steal an
unexpired claim. After the initial claim commit, the reconciler must open a
second AuthNZ transaction, lock and revalidate the exact owned claim, and keep
that row lock through idempotent Jobs admission plus exact AuthNZ attach. If a
lifecycle mutation committed first, revalidation observes terminal work and
uses lookup-only recovery without admission; if it commits later, it waits for
attach and then persists normal cancellation. This is one AuthNZ lock
transaction around an idempotent Jobs call, not a distributed transaction.

Add exact-token compare-and-set operations that (1) release a known-safe
transient rejection to `pending`, or to `dead:delivery_expired` if its delivery
lifetime elapsed and Jobs admission is known not to have created a row; (2)
fail an owned nonterminal claim as `dead:jobs_identity_conflict`; and (3)
recover an owned terminal/expired claim without losing the orphan-cancellation
coordinate. Lookup-proven missing Jobs identity clears the claim immediately.
A known canonical Jobs row is atomically attached with one tokenized cancel
disposition while retaining a recoverable claim until Jobs cancellation is
observed and exact AuthNZ acknowledgement atomically applies the disposition
and clears the claim. Retries reuse the persisted Jobs ID and disposition token;
ambiguous admission or lookup outcomes retain the claim for takeover after its
lease expires. Preserve an existing terminal state/reason, never make it
runnable, and require `attach_jobs_job()` to reject a delivery whose lifetime
elapsed. Prove these repository contracts on SQLite and required PostgreSQL
before the reconciler uses them.

- [ ] **Step 1: Define a narrow queue adapter and write unit RED tests**

The adapter exposes only:

```text
admit_delivery_job(delivery_id, expires_at) -> JobsDeliveryAdmission
find_delivery_job_by_identity(delivery_id) -> JobsDeliveryRecord | None
get_delivery_job(jobs_job_id) -> JobsDeliveryRecord | None
apply_queued_cancel(jobs_job_id, delivery_id, disposition_token, reason_code) -> PreparedDispositionResult
```

Admission uses the registered domain `admin_webhooks`, queue `delivery`, job type `admin_webhook_delivery`, payload containing only delivery ID, idempotency key `admin-webhook-delivery:<delivery_id>`, `max_retries=3`, `expired_lease_policy=requeue_no_attempt`, `quarantine_threshold=5`, and no owner. Preserve typed admission outcomes. Validate an idempotent existing row has the expected domain/queue/type/payload plus those immutable execution controls; any mismatch fails closed as a Jobs conflict. Identity lookup uses those same fixed fields but never inserts a row.

- [ ] **Step 2: Write every enqueue crash-window test**

Inject a crash at these boundaries and rerun reconciliation:

1. before AuthNZ claim commit;
2. after claim commit and before Jobs create;
3. after Jobs create and before create response;
4. after Jobs create/read and before AuthNZ attach;
5. after AuthNZ attach and before claim clear/queued commit;
6. after queued commit and before loop acknowledgement.

For each boundary assert one automatic delivery, one Jobs row/idempotency key, one attached Jobs ID, no stale claim, and final `queued`. An unexpired foreign claim is not stolen; an expired claim is recovered conditionally. A missing Jobs row after claim is recreated/read through the same idempotent create command. A mismatched existing Jobs payload makes delivery `dead:jobs_identity_conflict` without sending.

Transient Jobs database errors and typed `ADMISSION_REJECTED` outcomes clear only the matching AuthNZ claim back to `pending`, record a closed enqueue-failure metric, and retry on a later reconciler pass without creating an attempt or terminalizing delivery. Queue pause/drain remains an acquisition control and is not invented as an admission result. A backend/schema conflict, ambiguous identity lookup, or idempotent-existing identity mismatch is permanent and records `dead:jobs_identity_conflict`. Repeated transient rejection remains bounded by the 72-hour delivery expiry.

- [ ] **Step 3: Add cancellation/expiry races**

If delivery becomes canceled/superseded/dead before attach, do not mark it queued. Use lookup-only identity recovery: when no Jobs row exists, finish without creating one; when a matching queued row exists, attach identity only as needed and persist/apply tokenized cancellation; when the row is processing, leave the cancel disposition pending for the lease holder and never claim the request was unsent. If it expires before Jobs create, terminalize `dead:delivery_expired` without creating a job. If it expires after Jobs create, use the same lookup-only branch to cancel the orphan; never call admission as a cancellation probe and never make terminal AuthNZ work runnable. Reconciler heartbeat failure does not mutate delivery state.

Prove the pre-admission revalidation race by terminalizing immediately after
the initial claim commit for every AuthNZ/Jobs backend pair and asserting that
admission is never called. Also crash after terminal-orphan preparation but
before Jobs cancellation, and after Jobs cancellation but before AuthNZ
acknowledgement; after claim expiry, each rerun must reuse the persisted Jobs ID
and disposition token, avoid admission, and converge to an applied disposition
with no enqueue claim. A processing Jobs row keeps the pending cancel and a
recoverable claim for its lease holder or later recovery.

- [ ] **Step 4: Parameterize the four-backend enqueue matrix**

Use repository fixtures, not production data:

```text
AuthNZ SQLite / Jobs SQLite
AuthNZ SQLite / Jobs PostgreSQL
AuthNZ PostgreSQL / Jobs SQLite
AuthNZ PostgreSQL / Jobs PostgreSQL
```

Run all six crash boundaries, lookup-only missing/existing orphan branches, and cancellation/expiry cases for each pair. The PostgreSQL fixture uses `TLDW_TEST_POSTGRES_REQUIRED=1` and `RUN_JOBS=1` in the required gate.

- [ ] **Step 5: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_enqueue_reconciler.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py -k enqueue
```

- [ ] **Step 6: Implement one bounded reconciliation iteration**

Each iteration claims at most 100 rows ordered by expiry/creation/ID, handles one delivery transaction at a time, uses no distributed transaction, records closed metrics/reasons, and yields between batches. Claim tokens and Jobs identity are never logged. The loop caller owns cadence and heartbeat; the reconciler method is deterministic and independently testable.

- [ ] **Step 7: Run GREEN and verify no direct Jobs SQL**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_enqueue_reconciler.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py -k enqueue
if rg -n "_connect\(|SELECT .* FROM jobs|UPDATE jobs|INSERT INTO jobs" \
  tldw_Server_API/app/core/Admin_Webhooks/reconciler.py; then
  printf 'canonical reconciler bypasses JobManager\n' >&2
  exit 1
fi
```

- [ ] **Step 8: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Implemented recoverable idempotent AuthNZ-to-Jobs enqueue claims; every crash window converges across the four backend combinations."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/reconciler.py \
  tldw_Server_API/tests/Admin_Webhooks/test_enqueue_reconciler.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): reconcile delivery enqueue"
```

### Task 8: Implement Attempt Reservation, Worker, And Disposition Recovery

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/worker.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/reconciler.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/domain.py`
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_worker.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py`

**Interfaces:**
- Consumes: one acquired canonical Jobs row, `WorkerExecutionContext`, AuthNZ delivery bundle, webhook key ring, shared executor, repository CAS operations, and pending-disposition queue adapter.
- Produces: `AdminWebhookPreparedHandler.__call__()` plus disposition acknowledgement/recovery that never double-finalizes or performs I/O while repair is pending.

**Implementation ruling:** The Task 8 file list omitted load-bearing contracts.
Extend the existing AuthNZ repository and domain types without a schema change.
The final pre-I/O transaction must follow the control plane's
registration-before-delivery lock order, revalidate the exact attached Jobs ID,
active/tombstone/config/secret snapshot, hard attempt budget, and
`expires_at > required_horizon`, and then do exactly one of: reserve one
append-only attempt, or atomically persist the specific no-I/O terminal state
and one pending Jobs disposition. The current `required_horizon` argument must
become an enforced predicate rather than validation-only evidence.

Before a stale attempt's persisted `started_at + timeout_seconds + 90 seconds`,
return only `recovery_defer_until(stale_at)` and mutate no AuthNZ row. At or
after that boundary, one transaction conditionally marks the exact attempt
`outcome_unknown`, consumes its slot, and persists either the next exact retry
schedule or terminal `attempt_budget_exhausted`; it never proceeds directly to
HTTP. A retryable real result that loses a configuration race remains
append-only `retryable` evidence, but its delivery and pending Jobs disposition
use the specific cancel/supersede reason and carry no retry schedule.

Extend the generic prepared-disposition contract only as needed to allow an
AuthNZ `fail` without `attempt_id` for a no-attempt terminal such as delivery
expiry or an already-exhausted hard budget. Keep complete/retry attempt-bound,
and keep ordinary post-attempt fail attempt-bound in canonical worker code.
Use one strict bounded Jobs-marker projection and exact disposition fingerprint
comparison for lost-ack recovery; do not duplicate partial/ad-hoc marker JSON
checks in the worker and reconciler. Add a bounded ordered repository scan for
pending dispositions so the reconciler can repair exact Jobs/AuthNZ mirrors and
queued cancellation without admission or direct Jobs SQL. Prove these additive
repository/domain/Jobs contracts on SQLite and required PostgreSQL before the
worker consumes them. If wrong, the cost is localized Task 8 contract rework;
no migration or public API change is authorized.

**Review ruling (fix round 1):** A `processing` delivery always takes the
persisted-attempt branch before lifecycle, expiry, or budget terminalization.
Before its exact stale boundary, return only `recovery_defer_until(stale_at)`;
the generic no-attempt terminal helper must reject a processing delivery or any
non-null current attempt so it cannot orphan append-only evidence. At or after
the boundary, only the exact stale-attempt recovery transaction may close it.

Canonical `reserve_jobs_attempt()` callers must supply the reviewed config and
secret versions plus one valid disposition token. These coordinates are not
optional compatibility defaults: a terminal reservation must always atomically
persist its exact pending Jobs disposition. Pending cancellation recovery may
replace a historical retry or defer marker only when the exact canonical Jobs
row is currently queued; apply the new typed cancel monotonically, do not
acknowledge or replay the historical marker, then acknowledge only the new
AuthNZ cancel token.

The review claim that worker code can schedule a fourth retry is not a
production defect: the shared executor owns retry classification and converts
every retry-class result on attempt four to terminal
`attempt_budget_exhausted`, while reservation independently prevents a fifth
I/O. Do not duplicate that mapping in the worker. Add a worker-plus-real-
executor integration test that proves the terminal fourth result and no fifth
request. Expand the real SQLite/PostgreSQL matrix to exercise all six crash
boundaries for complete/retry/fail/cancel outcomes, both historical defer
origins, queued cancellation over historical markers, hard-cap I/O, and exact
late-writer rejection across all four AuthNZ/Jobs backend pairs. If this ruling
is wrong, the cost is localized worker/reconciler/repository contract rework;
no schema or public surface change is authorized.

- [x] **Step 1: Write the worker decision-table RED tests**

Before any executor call, assert this order:

1. validate Jobs payload/domain/queue/type and load delivery bundle; if AuthNZ is unavailable before any reservation, return timestamp-free `infrastructure_defer()` and leave AuthNZ unchanged;
2. if AuthNZ has an unapplied pending disposition, compare it with the acquired Jobs row's bounded reserved-result marker: when the marker is absent or is historical infrastructure/recovery evidence from an earlier lease, return the original AuthNZ disposition and absolute not-before time without I/O; when the same AuthNZ token was already applied before a lost acknowledgement and this job has since been reacquired, acknowledge the exact prior token in AuthNZ and continue the current lease rather than returning an already-applied token; a different AuthNZ-origin marker is a fail-closed conflict;
3. if delivery is terminal or lifecycle/config/secret/tombstone/active checks fail, commit/return the matching cancel/fail disposition without I/O;
4. if another processing attempt is not stale, return `recovery_defer_until(stale_at)` with the explicit AuthNZ-derived deterministic stale timestamp;
5. if stale, close it `outcome_unknown` using its persisted request timeout, consume its slot, and return a scheduled retry/terminal disposition without I/O;
6. contextually decrypt/validate target, secret, and exact event bytes and run the first delivery-time egress policy check without receiver HTTP I/O;
7. prove delivery lifetime and call `ensure_lease_horizon(timeout + 30)`;
8. reserve the next append-only attempt with the reviewed timeout under Jobs/lease/current-config/secret/active/tombstone/expiry predicates; this commit is the final pre-I/O configuration boundary;
9. immediately invoke the executor exactly once using the already-reviewed decrypted snapshot; the executor repeats DNS/egress/peer policy, but does not reload a potentially changed registration;
10. conditionally finish the same attempt and persist one pending prepared disposition before return.

Key/mode/database/policy/lease-horizon infrastructure failures before reservation return timestamp-free `infrastructure_defer()`; the Jobs apply transaction computes and persists `jobs_database_now + 30 seconds`. The stale-attempt path instead carries its explicit AuthNZ-derived stale timestamp and never substitutes the infrastructure schedule. Delivery expiry before reservation returns terminal fail. A lease loss after reservation permits the real executor outcome to be attempted conditionally; stale token rejection leaves later recovery to mark `outcome_unknown`.

- [x] **Step 2: Write append-only and hard-budget tests**

Prove sequence 1-4 across retries and lease losses; an interrupted slot becomes `outcome_unknown`; a fifth executor call is impossible; reaching four ambiguous/retryable slots commits `dead:attempt_budget_exhausted`; pending disposition replay consumes no new attempt; and replacement lease before staleness never overlaps a request.

- [x] **Step 3: Write post-attempt crash-window tests**

For AuthNZ-origin complete/retry/fail/cancel dispositions, inject crashes:

1. before attempt reservation commit;
2. after reservation commit/before I/O;
3. after receiver result/before AuthNZ outcome commit;
4. after AuthNZ outcome/disposition commit/before Jobs apply;
5. after Jobs apply/before AuthNZ acknowledgement;
6. after AuthNZ acknowledgement/before worker-loop return.

Expected semantics are explicit: boundaries 2-3 eventually record `outcome_unknown`; boundary 4 applies the stored disposition with no new request; boundary 5 proves Jobs reserved result/state and marks the matching AuthNZ token applied; boundary 6 is an idempotent no-op. For retry boundary 5, advance through the original not-before time and reacquire the row before AuthNZ acknowledgement: the new handler must acknowledge the already-applied retry token, clear the old pending state, continue under the current lease, and produce at most one next attempt. It must not reapply the old transition, increment retry/quarantine counters twice, extend the original schedule, or leave the processing lease unresolved. Complete/fail/cancel terminal lost acknowledgements are repaired from the terminal Jobs marker without reacquisition.

Separately crash after Jobs applies each no-acknowledgement `infrastructure_defer` and `recovery_defer_until` but before the worker loop returns. Assert no AuthNZ acknowledgement callback is invoked, the one stored schedule/token survives, and later reacquisition treats that marker as historical evidence: it neither reapplies nor acknowledges the old defer, does not change retry/failure/quarantine counters, and continues only under the new lease. Cover every AuthNZ-origin disposition and both no-acknowledgement defer origins across all four backend pairs.

- [x] **Step 4: Write cancellation/configuration/in-flight race tests**

Disable, rotate, update, or delete winning before reservation sends nothing and records its specific terminal reason. Winning after reservation cannot erase the attempt. A 2xx result remains succeeded with `completed_after_config_change=true`. A terminal receiver/network classification remains `dead` with its real reason and the same flag. A retryable result is retained on the append-only attempt, but no retry is scheduled against changed/disabled/deleted configuration; the delivery terminalizes with the specific canceled/superseded lifecycle reason and `completed_after_config_change=true`. A late worker cannot overwrite a replacement attempt or recovered terminal row.

- [x] **Step 5: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_worker.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py -k 'disposition or attempt or cancellation'
```

- [x] **Step 6: Implement the prepared handler and bounded callbacks**

The handler returns only `PreparedJobDisposition`. It never calls Jobs finalizers directly. `on_disposition_applied` conditionally acknowledges only an exact `origin=AUTHNZ` token; rejection leaves it pending, while infrastructure/recovery origins never invoke this callback. Before attempt reservation, the handler reconciles an acquired row's reserved-result marker with AuthNZ: an exact already-applied AuthNZ retry token is acknowledged and execution continues under that current lease; an unapplied AuthNZ token is returned once when the marker is absent or only historical no-acknowledgement evidence; a historical infrastructure/recovery defer marker is neither acknowledged nor reapplied; and a conflicting AuthNZ identity/token fails closed with timestamp-free `infrastructure_defer()`. Extend `AdminWebhookReconciler` to use known-ID and lookup-only identity reads, acknowledge exact AuthNZ lost-ack matches, cancel orphan Jobs rows without creating them, and repair Jobs terminal/AuthNZ nonterminal mirrors monotonically.

- [x] **Step 7: Run GREEN and the prepared SDK integration**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_worker.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Admin_Webhooks/worker.py \
  tldw_Server_API/app/core/Admin_Webhooks/reconciler.py
```

- [x] **Step 8: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Implemented lease-aware one-attempt worker and durable complete/retry/fail/cancel/defer recovery; crash and hard-attempt matrices pass without extra I/O."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/worker.py \
  tldw_Server_API/app/core/Admin_Webhooks/reconciler.py \
  tldw_Server_API/tests/Admin_Webhooks/test_worker.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): recover prepared delivery outcomes"
```

### Task 9: Implement Persisted Synchronous Test Attempts

**Files:**
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/delivery.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/reconciler.py`
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`

**Interfaces:**
- Consumes: `TestWebhookCommand`, current registration ETag/config version, idempotency scope/key, generated event/delivery/attempt/token IDs, mandatory audit, key ring, repository, and shared executor.
- Produces: `AdminWebhookDeliveryService.test_webhook()` with exactly one persisted synchronous attempt and interrupted-test recovery.

**Implementation ruling:** The written file list omitted load-bearing repository
and dual-backend proof. No schema change is required. Add a non-mutating exact
idempotency lookup so an existing conflict, processing replay, or terminal
replay is resolved before current registration, migration, or key preconditions.
An in-progress lookup exposes paired test delivery/attempt coordinates only
after the start transaction has durably attached them; a new uncommitted claim
without coordinates is not an externally useful processing replay.

For a new request, contextually decrypt and policy-check one reviewed in-memory
target/secret snapshot before the start transaction. In that transaction claim
idempotency, lock migration/key state, then the exact registration, and recheck
revision, reviewed delivery-config version, tombstone, target/secret versions,
and the same primary-key snapshot. Insert exactly one protected
`webhook.test` command-source event without subscription fanout, one `kind=test`
delivery, and attempt sequence one with its random test token; attach the paired
idempotency coordinates and commit delivery/attempt directly `processing`.
Intermediate pending state may exist only inside that transaction. Tighten the
test reservation contract to attempt one only and return explicit start
ownership. Only that owner may call the shared executor, exactly once.

Do not use the generic token-only Jobs completion branch for a synchronous
test. Add an exact repository operation that conditionally matches delivery ID,
attempt ID, and test token, closes the attempt/delivery with the real receiver
classification, and completes the same idempotency record with bounded response
metadata in one transaction. A test retry-class result is terminal dead with
its actual reason and no retry disposition; success is succeeded. Extend the
closed completion shape only as required for test
`outcome_unknown + dead + no Jobs disposition`; existing Jobs callers still
require their exact disposition.

Add a bounded ordered stale-test candidate read and exact recovery operation.
Before persisted `started_at + request_timeout_seconds + 90 seconds`, mutate
nothing. At or after it, atomically mark the exact attempt `outcome_unknown`,
the delivery `dead:test_attempt_interrupted`, and the idempotency result
terminal; the same exact token CAS rejects a late completion. Recovery performs
no HTTP, Jobs, or retry work. Processing and terminal replays load the exact
stored delivery/attempt by their paired idempotency coordinates without
decryption or mutable-current-registration reconstruction.

Define bounded internal test command/result/audit contracts in `delivery.py`.
The result carries exact delivery/attempt, `idempotent_replay`, `in_progress`,
and bounded retry guidance for Task 10. Mandatory `accepted` audit runs before
the start commit. A correlated completion/failure audit is attempted only after
durable completion; its failure is non-blocking and cannot roll back, rewrite,
or hide the persisted receiver outcome. Prove start/replay/completion/stale
contracts on SQLite and required PostgreSQL, including inactive registration,
all key/revision/config races, exact late-writer rejection, no secret/target
leakage, and explicit zero Jobs calls. If wrong, the cost is localized Task 9
repository/service/reconciler rework; no migration or public API change is
authorized.

- [x] **Step 1: Write RED tests for the transactional start boundary**

An exact idempotency replay lookup happens before current registration/key preconditions. For a new request, contextually decrypt and validate the reviewed target and secret, construct the exact deterministic event bytes plus protected persistence value, and run the first delivery-time URL/DNS policy check without receiver HTTP before opening the start transaction.

One AuthNZ transaction must then claim idempotency, recheck migration/key/rotation/revision/configuration/tombstone predicates and the exact reviewed key/version snapshot, create a `webhook.test` command-source event, create `kind=test` delivery, insert attempt 1 with its request timeout, and commit both delivery/attempt directly `processing`. That commit is the final pre-I/O boundary and returns explicit start ownership; a concurrent replay/conflict never invokes the executor. Assert no committed pending test row, no Jobs adapter call, test allowed while inactive, and event/delivery/attempt identities stored in idempotency.

Missing/stale ETag, stale reviewed config, deleted registration, unavailable key, rotation, malformed idempotency, same-key/different-request, and database busy return existing closed errors and perform no I/O.

- [x] **Step 2: Write RED tests for replay and recovery**

Processing exact replay returns 202 with original delivery ID, stable retry guidance, and no executor call. Terminal exact replay returns stored bounded result with `idempotent_replay=true` and no decrypt/I/O. A process crash after start is recovered only after `started_at + timeout + 90s`; attempt becomes `outcome_unknown`, delivery becomes `dead:test_attempt_interrupted`, idempotency completes with bounded metadata, and late completion token loses. Tests never retry or create Jobs rows.

- [x] **Step 3: Write RED tests for one executor outcome**

Assert `X-TLDW-Webhook-Test: true`, attempt sequence 1, 2xx success, retry-class HTTP/network outcomes become terminal dead with their actual reason, no pending Jobs disposition, no second attempt, and mandatory accepted audit before start commit. Completion audit is bounded and cannot rewrite durable outcome.

Add deterministic configuration-race tests. Rotation, configuration mutation, or deletion before the start commit fails its compare-and-set and sends nothing. A change after the committed reservation cannot erase the real attempt: success remains succeeded, while any failed test remains dead with its actual receiver classification; both record `completed_after_config_change=true` and neither retries.

- [x] **Step 4: Run RED**

```bash
RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py
```

- [x] **Step 5: Implement test start, execute, complete, and stale recovery**

Generate every identity before the transaction so a retried transaction reuses it. Return replay/conflict before decrypting whenever a stored idempotency record permits it. For a new operation, hold the reviewed decrypted target/secret and exact event bytes only in `repr=False` in-memory values, run the non-I/O egress preflight, and use transaction predicates to prove the same snapshot is still current while reserving attempt one. Only the transaction result that owns the new start may call `DeliveryAttemptExecutor`, immediately after commit, exactly once with that snapshot; conditionally finish by test token and preserve post-reservation configuration races. Extend the reconciler with a separate bounded stale-test pass that uses the persisted timeout and never schedules Jobs or HTTP.

- [x] **Step 6: Run GREEN and no-Jobs proof**

```bash
RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py
if rg -n "create_or_get_delivery_job|create_job\(" \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py | rg -v "assert_not_called|raises"; then
  printf 'review test path for an unintended Jobs create\n' >&2
  exit 1
fi
```

- [x] **Step 7: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Implemented persisted one-attempt synchronous tests with precondition/idempotency replay and interrupted-attempt recovery; no Jobs/retry path is reachable."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/reconciler.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): persist synchronous test attempts"
```

**Review ruling (fix round 1):** The independent review correctly identified
four evidence/behavior gaps. Expand the shared SQLite/PostgreSQL contract to
prove post-commit 204, retryable HTTP, and retryable network outcomes across
semantic configuration, signing-secret, and deletion races; prove every
pre-commit target/secret version, ciphertext, key-ID, and active-primary race
rolls back the entire start with zero receiver I/O; commit stale recovery before
an independent late-completion transaction proves the exact token CAS loses;
and implement the global correlated-audit protocol when transaction exit fails
after the mandatory `accepted` audit. That last path must attempt a bounded
`failed` audit with the same request, webhook, delivery, and attempt identity,
roll back all AuthNZ rows, and never invoke the executor; failure of the
follow-up audit must not mask the original commit error.

The review suggestion that envelope ciphertext or key-ID rewrites alone set
`completed_after_config_change=true` is rejected as stated. Dedicated
encryption-at-rest rotation preserves the receiver-visible target, signing
secret, timeout, and event configuration and does not advance their semantic
versions. The flag remains tied to registration revision/lifecycle,
`delivery_config_version`, and signing-secret version changes. Add an explicit
post-start re-encryption regression proving the real receiver result is retained
with `completed_after_config_change=false`; this prevents operational key
maintenance from being mislabeled as a webhook configuration change. Start-time
snapshot CAS remains stricter and must reject an intervening envelope/key rewrite
before I/O.

**Completion:** Task 9 is complete at `30ca1f3958525f6f4d859990288d5d0521651749`
after one fix round and a clean independent re-review. The final required
PostgreSQL regression gate passed 260 tests with zero skips; the independent
reviewer reran the focused SQLite and required-PostgreSQL selections at four
passes each, with no Critical, Important, or Minor findings.

### Task 10: Expose Manual Redelivery, Test, History, And Audit APIs

**Files:**
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/audit.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/delivery.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/domain.py`
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_webhooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_audit.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_api.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_openapi.py`
- Modify: `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

**Interfaces:**
- Consumes: current registration ETag/config version, historical delivery/event, idempotency key, platform-admin principal, mandatory mutation audit, delivery service, and read-audit adapter.
- Produces: canonical `GET /{webhook_id}/deliveries`, `POST /{webhook_id}/test`, and `POST /{webhook_id}/deliveries/{delivery_id}/redeliver` contracts.

**Preflight ruling:** The original file list omitted repository/domain history
projection work and a closed audit contract required by the approved API. Add a
dedicated sanitized history item containing `WebhookDelivery`, bounded event
type, `completed_after_config_change`, and ordered attempts. Page deliveries
newest-first, then load attempts for all page delivery IDs in one set-based
query; per-delivery attempt queries are forbidden. Count, page, and attempt
reads remain bounded and backend-neutral. They never load or decrypt event
bodies, registration targets, or secrets, and history remains available when
the key ring is unavailable and for a retained soft-deleted registration. A
missing registration returns the same closed 404 without disclosing whether
unrelated delivery IDs exist.

Manual redelivery reuses the existing idempotency table without a migration.
Add a closed `redelivery_delivery_id` response-metadata key, validate it as a
canonical UUID when decoding, and project it through a typed replay field.
The scoped `delivery_id` remains the source delivery ID; the paired test
coordinate columns remain test-only. Exact replay/conflict lookup occurs before
migration, key, current-registration, source-delivery, event-decryption, and
policy checks. A valid replay loads the exact created manual delivery through
the stored replay coordinate, verifies webhook/source linkage, emits one
mandatory `no_op` audit (never a second `accepted` audit), and returns without
decryption, first-activity rewrite, or Jobs work. Malformed or missing persisted
replay coordinates fail closed as delivery unavailable.

Do not widen `MutationAudit` with optional unrelated fields. Add a separate
closed typed delivery-mutation audit record and mandatory emitter for exactly
`admin_webhook.test` and `admin_webhook.redeliver`. Its action-specific field
matrix permits only bounded actor/request/webhook IDs, canonical source/new
delivery and test-attempt IDs when allocated, bounded target hostname when
known, source/current config versions, changed-config boolean, bounded status,
closed outcome, and closed domain/delivery reason. Accepted and test-completion
records require their exact allocated coordinates; an early denial/failure may
omit coordinates that were never allocated. Bridge Task 9's internal test audit
through this emitter. After actor establishment, a deterministic service
failure before any internal audit emits exactly one denied/failed record;
framework validation/auth failures before actor establishment do not fabricate
one. An accepted audit is pre-commit. If commit fails after accepted, attempt
one correlated failed audit with the same coordinates and preserve the original
error if that follow-up fails. No URL/path/query, event body, secret, receiver
content, header map, free text, exception text, key/Jobs/test token, or
idempotency material is accepted by the audit type or emitter.

No schema/migration, direct Jobs admission, PR 3 producer/UI/legacy removal, or
Task 11 runtime activation is authorized by this task.

- [x] **Step 1: Write schema RED tests**

Add strict Pydantic models:

```text
WebhookTestRequest(delivery_config_version)
WebhookTestResponse(delivery, attempt, idempotent_replay, in_progress)
WebhookRedeliveryRequest(delivery_config_version, confirm_changed_configuration)
WebhookRedeliveryResponse(delivery, idempotent_replay)
WebhookDeliveryAttemptResponse
WebhookDeliveryResponse
WebhookDeliveryHistoryItemResponse(delivery, attempts)
WebhookDeliveryListResponse(items, total, limit, offset)
```

Request bodies use strict fields, forbid extras and explicit null, and do not
coerce booleans/integers. Response models forbid extras and permit null only for
semantically absent terminal/attempt evidence. IDs, event types, and reason
codes are bounded. Delivery responses contain event ID/type, kind/state,
delivery/secret version snapshots, attempt count, bounded outcome metadata,
expiry/timestamps, redelivery linkage, and
`completed_after_config_change`. History items add only ordered attempts.
Delivery responses never contain event data/ciphertext/key IDs, target URL or
display/path/query, secret/signature/request headers, response body or ordinary
response headers, Jobs IDs/leases/tokens, test tokens, or idempotency material.
Attempt responses expose only ID, sequence/state, request-timeout snapshot,
status/latency/reason/requested retry delay, and start/finish timestamps.

- [x] **Step 2: Write manual-redelivery service RED tests**

Require active non-deleted current registration, available key, completed
migration, strong ETag, reviewed current delivery-config version, source
delivery belonging to that registration, and an existing decryptable historical
event. New work claims idempotency, locks migration/key state and then the
current registration, revalidates revision/config/activity and source ownership,
loads the source event under the same transaction, and verifies historical event
decryption/integrity without rewriting it. Create a new `kind=manual` delivery
with current config/secret versions, same event ID, new delivery ID,
`redelivery_of_id`, pending state, 72-hour expiry, typed replay coordinate,
completed idempotency record, and
`mark_first_canonical_activity("delivery_attempt")` in that transaction. This
marker intentionally closes the structural legacy-restore window at accepted
redelivery creation while retaining the schema's existing closed activity
categories; it does not insert an attempt row before Jobs acquisition.

When original and current delivery-config versions differ,
`confirm_changed_configuration` must be true or return
`428 admin_webhook_redelivery_confirmation_required`. A secret-version-only
change does not trigger this confirmation. Audit records the typed
`redelivery_to_changed_config` fact and source/current versions. Exact replay
follows the preflight ruling above. Same key/different source/config/
confirmation/conditional ETag conflicts, and rejected preconditions leave the
delivery, idempotency claim, and activity marker unchanged. Concurrency creates
one row. An accepted-audit failure rolls back all AuthNZ writes. A commit failure
after accepted attempts one correlated failed audit without masking the original
failure. The service and route never call Jobs; the existing reconciler later
admits the pending row.

- [x] **Step 3: Write route/auth/audit/error RED tests**

Assert platform-admin authorization on all three routes and numeric user-backed
principal for test/redelivery. History reads retained tombstones without key
material and use bounded best-effort access audit with result count;
test/redelivery use the mandatory delivery-mutation audit bridge. Validate
401/403/404/409/412/422/428/429/500/503 fixed envelopes,
`Cache-Control: no-store`, normalized `X-Request-ID`, no secret canaries, and no
default FastAPI validation body.

Route semantics:

- history defaults `limit=50`, bounds 1-100, offset 0-1,000, returns newest first;
- terminal test returns 200; an exact still-processing replay returns 202; both use the same bounded response schema;
- accepted redelivery returns 202 and the pending/queued delivery metadata;
- source delivery not owned by the path registration returns the same closed not-found response as a missing delivery;
- test/redelivery require `Idempotency-Key`; both require `If-Match` and reviewed version in body.
- test `Retry-After` is present only for an in-progress replay and is a bounded decimal value;
- history does not require a numeric user principal, migration key, or active registration.

Declare static `/catalog` and `/status`, then collection routes, then nested
`/deliveries` and action routes, then `/{webhook_id}`. Add the existing Task 10
error codes to the closed route map. OpenAPI advertises only canonical error
envelopes and no arbitrary payload/header inputs. Production composition uses
the application-scoped AuthNZ pool, environment-validated settings, loaded key
ring, UUID/token factories, UTC clock, and the same status-only peer-verified
`DeliveryAttemptExecutor` implementation as the worker. It remains dependency-
overrideable in tests, imports no legacy webhook service, and starts no worker.

- [x] **Step 4: Run RED**

```bash
RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py
```

- [x] **Step 5: Implement service commands and thin route composition**

Implement the dedicated domain history item, set-based repository history read,
typed redelivery replay coordinate, atomic manual-redelivery command/result,
closed delivery-mutation audit record/emitter, Task 9 test-audit bridge, and thin
route composition. Routes parse/sanitize, authorize, build commands/audit sinks,
call the service, serialize explicit response models, and preserve no-store/
request-ID/Retry-After headers. They contain no SQL or outbound HTTP. Use
FastAPI dependency injection for service/executor fakes; production composition
uses the application-scoped AuthNZ pool and key ring. Do not add a migration or
touch Jobs admission/runtime wiring.

- [x] **Step 6: Refresh and review OpenAPI fingerprint**

```bash
make openapi-fingerprint
make openapi-drift-check
git diff -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected: only the three reviewed PR 2 route families and their delivery/audit
schemas differ. Status, durable producers, runtime activation, and legacy routes
remain unchanged.

- [x] **Step 7: Run GREEN and leak assertions**

```bash
RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Admin_Webhooks/audit.py \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
../../.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/Admin_Webhooks/audit.py \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
/Users/macbook-dev/.local/bin/python3.10 -m py_compile \
  tldw_Server_API/app/core/Admin_Webhooks/audit.py \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
if rg -n "JobManager|admit_job|create_job|jobs_webhooks_task|admin_webhook_delivery_runtime" \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py; then exit 1; fi
git diff --check
```

The required PostgreSQL selection must run with zero skips. Add explicit
SQLite/PostgreSQL proof for set-based history attempts, exact redelivery replay,
concurrency, changed-config confirmation, foreign source, rollback, accepted-
audit/commit-failure behavior, keyless replay/history, malformed replay
coordinates, and zero direct Jobs work. Review warning provenance and run
no-leak scans over schemas, OpenAPI, audit metadata, repr/log surfaces, and API
responses.

- [x] **Step 8: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Exposed canonical persisted test, manual redelivery, and sanitized history APIs with ETag/idempotency/audit/OpenAPI contracts; no PR 3 UI or producer work included."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/audit.py \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/Admin_Webhooks/domain.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  tldw_Server_API/tests/Admin_Webhooks/test_redelivery_history_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_audit.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): expose delivery operations and history"
```

**Review 1 and fix-round 1 ruling:** Independent review of
`db91e7fb46c7d467b61c41de6fedf26ead11a992..d083aaca14dfb4c3c876070f9130b0df23a33d09`
found no Critical issues, six Important implementation defects, and one Minor
report-only RED-chronology gap. All six implementation findings are accepted;
controller verification adds a seventh Important mapping defect. Task 10 is not
complete at this head.

1. The approved same-key/different-source conflict must coexist with a full
   source-bound persisted scope. Derive redelivery's lookup digest from a stable
   actor/operation/webhook key-family scope that omits only the source coordinate,
   while storing and comparing the full canonical route plus source delivery in
   the idempotency row and request fingerprint. Thus exact same-source replay
   matches, a second source under the same actor/operation/webhook and raw key
   reaches scope mismatch and returns 409, and another webhook remains a separate
   family. Do not change generic idempotency semantics or schema.
2. Decode redelivery idempotency rows through an exact action-specific state
   matrix. In-progress redelivery has no response/result coordinates. Completed
   redelivery has status 202, exactly one canonical
   `redelivery_delivery_id` metadata key, and no generic resource/version,
   secret/replay-secret, or test coordinates. Any other shape fails before
   registration/key reads as delivery unavailable.
3. History SQL and mappers select only the public delivery/history columns and
   public attempt columns. They must never select, instantiate, validate, or
   retain Jobs IDs/leases, enqueue claims, disposition tokens/state, test tokens,
   idempotency material, protected values, or full internal stored-delivery
   records. Add SQL/query and malformed-hidden-column proof, not response-only
   no-leak assertions.
4. Registration existence/count, delivery page, and set-based attempts must come
   from one backend-correct read snapshot (or one equivalent statement) so a
   concurrent commit cannot produce mismatched totals/deliveries/attempts.
   Preserve bounded query count and avoid N+1 behavior on SQLite and PostgreSQL.
5. OpenAPI must mark `If-Match` required while runtime omission still reaches the
   service's canonical 428 response, constrain `Idempotency-Key` to the exact
   16-255 safe-character contract, and document `X-Request-ID` plus
   `Cache-Control` on all three success families and `Retry-After` on test 202
   only. Add generated-schema assertions and refresh the fingerprint.
6. Remove the Task 10 broad `except Exception` from `AdminWebhookRoute`; unknown
   programming failures remain owned by the global sanitized 500 handler and its
   telemetry. Add regression proof covering an existing route and a Task 10
   route.
7. Map repository `NOT_FOUND` from `list_delivery_history()` to public
   `WebhookErrorCode.NOT_FOUND`, preserving the required 404 and best-effort
   denied read audit. Do not broaden the shared capture-error mapping in a way
   that changes unrelated service semantics.

Write deterministic RED tests for all seven defects before production fixes,
then rerun the complete Task 10 required-PostgreSQL zero-skip gate, focused
SQLite/PostgreSQL race/malformed/query-shape tests, Task 9 regressions, OpenAPI
drift, Ruff, Bandit, Python 3.10, scope/no-leak scans, and diff checks. The Minor
chronology limitation remains explicitly documented; do not fabricate historical
evidence or add a test-only commit solely to rewrite the initial sequence.

**Fix round 1 completion:** Six of the seven accepted defects were closed with
deterministic pre-production RED coverage; independent re-review left the
single-item public-history projection defect open. The required Task 10 gate
passed 138 tests with PostgreSQL required and zero skips; Task 9 regressions
passed 18 tests with zero skips, and event-expansion regressions passed 24.
OpenAPI fingerprint/drift, Ruff, reviewed Bandit, Python 3.10 compilation,
scope/no-leak scans, warning provenance, and diff checks passed. No schema,
migration, direct Jobs admission, runtime activation, producer/UI, legacy
service import, or Task 11 work was added. Evidence is recorded in
`.superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-10-fix-1-report.md`.

**Re-review 1 and fix round 2 completion:** Independent re-review found one
remaining Important privacy/query-shape defect in the live single-item history
path used by exact redelivery replay, plus its unused attempt `created_at`
projection. Fix round 2 replaced the internal delivery/attempt projections and
mappers with the public allowlists and sanitized mappers under one read
snapshot, preserving two bounded ordered queries and ownership semantics.
Dual-backend RED failed all six new contracts; focused GREEN passed six with
PostgreSQL required and zero skips, and the complete Task 10 gate passed 144
tests with zero skips. Ruff, Python 3.10 compilation, reviewed Bandit,
query-shape/no-leak scans, self-review, and diff checks passed. Evidence is in
`.superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-10-fix-2-report.md`.
Scoped independent re-review found both remaining findings addressed with no
new Critical, Important, or Minor breakage. Task 10 is complete at
`928adf14edcc3db00b8b7393bad60094e4863b9d`.

### Task 11: Add Durable Health, Metrics, Retention, And Runtime Wiring

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/observability.py`
- Create: `tldw_Server_API/app/services/admin_webhook_delivery_runtime.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/control_plane.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/reconciler.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_webhooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py`
- Modify: `tldw_Server_API/app/services/startup_optional_workers.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_api.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_optional_workers.py`
- Create: `Docs/Admin_Webhooks_Delivery_Runbook.md`

**Interfaces:**
- Consumes: repository health queries/heartbeats, Jobs preflight, key/migration/mode state, worker/reconciler iterations, metrics registry, and stop event.
- Produces: durable `DeliveryCapabilityStatus`, sanitized status API expansion, bounded retention, low-cardinality metrics, and one supervised optional runtime.

**Preflight ruling:** The local file list omits load-bearing domain, repository,
metrics-integration, backend-contract, and OpenAPI work. Task 11 may also modify
`domain.py`, `admin_webhooks_repository.py`, `delivery.py`, `worker.py`, their
focused tests and SQLite/PostgreSQL wrappers, `test_openapi.py`, and the frontend
OpenAPI fingerprint. It may not add a schema/migration, change Jobs core SQL or
WorkerSDK contracts, enable deployment configuration, connect PR 3 producers or
UI, import either legacy webhook service, or perform direct Jobs SQL.

Add one closed `DeliveryCapabilityStatus` projection with exact canonical
schema version/readiness, delivery-extension readiness, migration/key/Jobs and
fixed queue/type readiness, per-component heartbeat readiness/reason/age,
closed nonterminal backlog counts, oldest nonterminal age, retention status,
and the final activation capability boolean. No instance ID or sensitive field
is public. AuthNZ health facts come from one bounded backend-correct read
snapshot. A component is ready when any fresh ready instance exists; otherwise
use the freshest bounded row to report its closed unready reason, a stale row
reports `heartbeat_stale`, and no row reports the component-specific unavailable
reason. Retention readiness is visible but is not an acquisition prerequisite.

Avoid the startup fixed point where worker readiness requires its own heartbeat.
The capability has a foundational/acquisition preflight that checks current
schema, migration, key, Jobs access, exact canonical queue/type registration,
and fresh reconciler evidence, but not worker or retention heartbeat. The
worker pre-acquire guard uses that result and writes its own ready/unready
heartbeat. Full activation/API readiness additionally requires a fresh ready
worker heartbeat. Reconciler and retention loops continue independently when
their own required dependencies permit recovery.

Metrics are real integrations, not definitions alone. Use a typed, fail-open
adapter with a fixed `admin_webhooks_` metric-name and label-schema registry;
callers cannot supply arbitrary names or label keys/values. Counter observations
happen only after the corresponding durable commit, gauges use one current
health snapshot, and metric failures never alter durable control flow. Add
observer seams only where needed in control plane, delivery service, reconciler,
and worker; derive SSRF-denial metrics from the worker's closed executor outcome
rather than adding transport leakage. Labels remain limited to the global
closed state/kind/event-type/reason/status-class/component/backend catalog.

The Task 11 retention order supersedes Task 2's provisional order: eligible
terminal deliveries, newly orphaned events, expired idempotency, stale runtime
heartbeats, then eligible tombstones, with one total 1-200 row budget and
deterministic continuation. Repeated partial batches must drain every finite
eligible category. Runtime expiry uses a separate bounded reconciler operation:
exclude live processing/current-attempt rows; terminalize due unattached work
atomically; and for attached Jobs work persist one exact cancel disposition
token before the existing lookup/apply/ack repair path. It never mutates Jobs
inside AuthNZ SQL and never sends HTTP.

The canonical lifecycle spec is named exactly
`admin_webhook_delivery_runtime_task` and calls only the new runtime. It is
enabled only for validated `mode=on` plus canonical route selection. The
isolated `jobs_webhooks_task` implementation is not imported, called, or used as
an alias by canonical code; preserve its PR 2 off/compatibility behavior, but it
must not be simultaneously enabled as the canonical admin-webhook runtime.
Refresh and review the OpenAPI fingerprint for the bounded status expansion.
All new repository contracts run on SQLite and required PostgreSQL with zero
skips.

- [x] **Step 1: Write observability/health RED tests**

Health aggregates canonical schema version, explicit migration-095 delivery-extension readiness, migration completion, key availability/primary match, Jobs database access, exact queue/type registration, freshest ready worker heartbeat, freshest ready reconciler heartbeat, retention heartbeat, backlog counts by state, and oldest pending age. Heartbeats are fresh only within configured freshness; stale/unready rows report stable reason codes. Multiple instances choose the freshest ready evidence without deleting another live instance.

`delivery_capability_ready` requires canonical schema version 1, successful `delivery_schema_ready()` extension preflight, completed migration, key match, Jobs preflight, and fresh worker/reconciler heartbeats. Retention staleness degrades status but does not by itself permit acquisition. Activating a registration requires capability ready; metadata read/disable/delete/history remain available in degraded states under existing key rules.

- [x] **Step 2: Write metrics RED tests**

Register/increment only the approved families: registrations/admission denials, events/fanout, enqueue claim/recovery/failure, delivery state/reason/status class, attempts/latency/retries/expiry, backlog/oldest age, heartbeat age/readiness, retention deletion counts, key/migration errors, and SSRF denials. Assert label schemas are closed and no ID, hostname, URL, email, narrative, payload, secret, signature, exception string, or response content is accepted as a label.

- [x] **Step 3: Write retention RED tests**

At 29d23h59m terminal metadata remains. At 30d, a bounded batch removes terminal deliveries only when eligible, then orphan events, expired idempotency, stale heartbeat instances, and eligible tombstones. Nonterminal pending/claimed/queued/processing/retry rows never delete; terminal time, not creation/expiry, starts retention. A partial batch resumes without starvation. Failed retention writes publish an unready heartbeat/reason and leave rows intact.

- [x] **Step 4: Write runtime/startup RED tests**

`run_admin_webhook_delivery_runtime(stop_event)` supervises three independent loops: prepared Jobs worker, reconciler (enqueue/disposition/stale/expiry), and retention. Each writes its own heartbeat and isolates bounded failures without silently marking itself ready. Stop cancels/awaits every child and flushes a final unready heartbeat where possible.

`provide_optional_worker_specs()` registers exactly one `admin_webhook_delivery_runtime_task` only when validated canonical route selection is canonical and mode is `on`; default off, migrate, legacy compatibility, and invalid settings do not start it. Startup never starts or aliases legacy `jobs_webhooks_task`. A key/JOBS preflight failure may start the runtime for observable recovery but the worker pre-acquire guard stays closed and heartbeat reports the reason.

- [x] **Step 5: Run RED**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py
```

- [x] **Step 6: Implement observability and real delivery capability composition**

Use the repository metrics registry through a narrow adapter that validates names/labels before forwarding. `AdminWebhookDeliveryCapability.status(now)` is async and returns one sanitized snapshot; update activation and status paths to await it. `UnavailableDeliveryCapability` remains for off/migrate/test construction and returns fixed unavailable facts.

- [x] **Step 7: Implement bounded loops and startup spec**

The runtime builds separate repository handles as required by pool/thread safety, one `JobManager` adapter, one shared executor, and unique random instance IDs. Worker acquisition preflight calls current health dependencies each cycle. Reconciler and retention iterations are bounded and sleep interruptibly. Do not add ad hoc startup tasks outside the declarative lifecycle worker catalog.

- [x] **Step 8: Write the PR 2 delivery runbook**

Document default-off status, exact environment/cadence bounds, mode/key/Jobs preflight, worker/reconciler/retention heartbeat interpretation, queue/domain/type, backlog and oldest-age triage, dead reason codes, attempt ambiguity/at-least-once semantics, test behavior, manual changed-config redelivery, disabling and in-flight limits, 72-hour expiry, 30-day retention, retry schedule, receiver deduplication/signature vector, no-buffer SSRF controls, forward-fix procedure, and explicit statement that user/incident producers and operational UI activation wait for PR 3.

- [x] **Step 9: Run GREEN and shutdown/status regressions**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Admin_Webhooks/observability.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/startup_optional_workers.py
```

- [x] **Step 10: Update the task and commit**

```bash
backlog task edit 13111 --append-notes "Added durable worker/reconciler/retention health, bounded metrics/retention, mode-gated lifecycle runtime, and PR 2 operations runbook."
git add \
  tldw_Server_API/app/core/Admin_Webhooks/observability.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/core/Admin_Webhooks/control_plane.py \
  tldw_Server_API/app/core/Admin_Webhooks/reconciler.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py \
  tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py \
  tldw_Server_API/tests/Admin_Webhooks/test_api.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  Docs/Admin_Webhooks_Delivery_Runbook.md \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "feat(admin-webhooks): operate delivery runtime safely"
```

Task 11 implemented the typed async delivery capability/status projection, one
backend-correct bounded health snapshot, fixed fail-open metrics adapter,
deterministic bounded retention and expiry recovery, and three independently
supervised runtime loops behind the exact canonical startup gate. The sanitized
status API and reviewed OpenAPI fingerprint were updated, and the PR 2 delivery
runbook was added. Strict RED evidence covered the health/acquisition fixed
point, retention order/fairness, startup gating, closed metric labels, and each
added integration boundary. Final focused Task 11 verification passed 137
tests; the broader relevant regression gate passed 313 tests; the four-backend
recovery matrix passed 68 tests; and the complete SQLite and required
PostgreSQL repository contracts passed 34 tests each, all with zero skips.
Ruff, Python 3.10 compilation, OpenAPI drift, reviewed Bandit, shutdown/status,
scope/security scans, and diff checks passed. Evidence and warning triage are in
`.superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-11-report.md`.

#### Task 11 Fix Round 1: Close Independent Runtime Review Findings

**FIX_BASE:** `99098c213930d36c0b983493099954a659abc4fb`

**Review:**
`.superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-11-review-1.md`

The independent review found no Critical issues and six accepted Important
issues. Task 12 remains blocked until every item below has strict RED evidence,
required PostgreSQL proof, a fix commit, and a clean independent re-review.

- [x] Preserve the created-but-unattached Jobs recovery coordinate through
  expiry. Blind repository expiry must not terminalize an `enqueue_claimed` row
  with no attached `jobs_job_id`; enqueue reconciliation owns lookup-only
  discovery and exact terminal cancellation after `BEFORE_AUTHNZ_ATTACH`.
  Exercise crash, live/expired claim, expiry/reconciler interleavings, and
  concurrent repair across all four AuthNZ/Jobs backend pairs. Prove one Jobs
  row, one exact persisted cancel token, eventual cancellation/acknowledgement,
  no HTTP attempt, and no stranded claim or Jobs work.

- [x] Replace the one-shot Jobs runtime objects with a runtime-local bounded
  refresh boundary. Each supervised worker start receives a fresh `JobManager`,
  `WorkerSDK`, Jobs worker ID, and handler generation; stopping one SDK must not
  poison the next generation. A construction failure installs only closed
  unavailable queue/probe delegates, retries interruptibly at the configured
  cadence without tight spin, and atomically promotes a complete healthy
  generation so capability and reconciler delegates recover without process
  restart. Retention remains independent. Test unexpected SDK exit followed by
  a live second generation, initial fail-then-success Jobs construction,
  resumed enqueue/acquisition, truthful heartbeat transitions, and exact
  stop/await ownership for every child.

- [x] Reject clock-skewed future heartbeat evidence. Apply one explicit small
  maximum-future-skew bound to both SQLite and PostgreSQL snapshot queries;
  future-only rows report the closed `heartbeat_stale` reason and cannot satisfy
  acquisition or activation. A genuinely fresh ready instance still wins over
  a future invalid ready or unready row. Add dual-backend precedence, boundary,
  and future-ready/future-unready tests.

- [x] Compose factual degraded public status instead of substituting fabricated
  schema facts. When AuthNZ is readable, always use its bounded health snapshot,
  combine it with a closed unavailable Jobs probe as needed, and apply `off` or
  `migrate` as a mode gate using the existing exact `mode_off` or `mode_migrate`
  reason. Jobs failure reports `jobs_unavailable`. Reserve fixed fallback for a
  genuine delivery-health read failure, preserve already-known migration/key
  facts, and report `database_unavailable`. Add real control-plane/API tests for
  every mode and dependency failure, internally consistent facts, and exact
  reason precedence.

- [x] Complete metrics integration at durable product boundaries: synchronous
  test completion owned by the start caller; worker no-I/O terminal commits;
  expiry/terminal recovery paths; and registration gauges initialized and
  refreshed from one bounded current-count snapshot. Include automatic,
  manual, and test kinds plus delivery state/reason/status class, latency,
  retry/expiry, and SSRF denial where applicable. Metrics remain fail-open and
  best effort: prove one emission for an owned committed transition and none on
  rollback, stale CAS, or idempotent replay, but do not claim crash-proof
  exactly-once telemetry without a durable outbox. Registry failures must never
  change durable behavior.

- [x] Update the runbook and Task 11 evidence with enqueue-crash expiry repair,
  in-process Jobs generation recovery, future-heartbeat handling, degraded
  status reason precedence, and the best-effort telemetry boundary. Re-run the
  focused Task 11 gate, complete SQLite/PostgreSQL repository contracts, the
  four-backend recovery matrix, worker/reconciler/status/startup regressions,
  OpenAPI drift review, Ruff, reviewed Bandit, Python 3.10 compilation,
  direct-Jobs-SQL/legacy-import/sensitive-label/scope scans, warning triage, and
  diff checks with zero required PostgreSQL skips.

Fix Round 1 closed all six accepted Important findings with strict focused RED
before each production boundary. Final verification passed 226 focused
status/startup/shutdown/OpenAPI/worker/reconciler/control/API/test/redelivery
tests, 72 complete SQLite/PostgreSQL repository contracts, and all 76
four-backend recovery cases with required PostgreSQL and zero skips. The
project-interpreter OpenAPI drift check, Ruff, Python 3.10 compilation, reviewed
Bandit, direct-Jobs-SQL/legacy-import/sensitive-label/scope scans, warning
triage, base-to-head self-review, and diff checks passed. Bandit reported no
High findings; its 43 Medium/Low-confidence reports are fixed SQL fragments
with bound values, and its 14 Low findings are the required fail-open observer
boundaries. Evidence is recorded in
`.superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-11-fix-1-report.md`.

Independent re-review of
`99098c213930d36c0b983493099954a659abc4fb..fe3290f32d4b411e2d56f41b9b473e3bd788f95e`
closed all six original findings and found no new Critical, Important, or Minor
defect. Controller verification independently reproduced all eight
four-backend crash-expiry cases plus the eight runtime-recovery,
future-heartbeat, and degraded-status contracts with required PostgreSQL and
zero skips. Task 11 is complete and Task 12 may begin. Re-review evidence is in
`.superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-11-fix-1-re-review.md`.

#### Task 11 Fix Round 2: Restore Aggregate Direct-Marker Compliance

Task 12's first clean serial aggregate gate stopped on
`test_each_webhook_pr_test_has_one_direct_accepted_marker`: 41 real PR 2 tests
relied on module-level markers instead of the repository's deliberately
required direct accepted marker, and the audit's unrestricted `ast.walk()`
also misclassified `_FakeDeliveryService.test_webhook()` as a collectable test.
This is test-selection compliance debt, not evidence of a production-runtime
defect. The earlier overlapping verifier runs are discarded and Task 12 remains
blocked.

Fix Round 2 is test-only. Preserve the direct-marker rule; make the audit mirror
pytest collection by checking top-level test functions and test methods only in
collectable `Test*` classes, while excluding helper methods in non-test classes.
Give every affected real test exactly one direct `unit` or `integration` marker
matching its existing module classification, and remove redundant module-level
classification only after every test is directly classified. Strict RED is the
clean serial failure with 42 violations. GREEN requires the exact audit, every
affected module, and the complete Task 12 Step 1 union to pass serially with
`TLDW_TEST_POSTGRES_REQUIRED=1`, explicit timeout and `-ra`, zero skips, followed
by an independent scoped review. No production file may change. Task 12 may
restart only after that review is clean.

The clean post-marker Step 1 union collected 1,489 tests and then exposed one
separate stale Task 10 assertion at 46%:
`test_canonical_selection_excludes_legacy_delivery_routes` still required the
canonical synchronous-test and delivery-history method/path pairs to be absent.
That expectation predates Task 10 and contradicts this plan's approved
canonical `POST /{webhook_id}/test`, `GET /{webhook_id}/deliveries`, and
`POST /{webhook_id}/deliveries/{delivery_id}/redeliver` API. Focused
reproduction failed one of one with zero skips. Extend this same test-only fix
round to rename/correct that route-selection contract, require all three PR 2
canonical method/path pairs, and continue to require the uniquely legacy
incident-notify path absent. Do not modify route composition or any production
file. Rerun the focused route-selection module and the complete serial Step 1
union after correction; the partial aggregate run is not GREEN evidence.

Fix Round 2 implementation and local verification are complete at the
pre-commit tree. Strict RED reported 42 direct-marker violations; focused GREEN
reported 1 exact audit pass and 277 affected-module passes, all with zero skips.
The stale route contract reproduced 1 failure with zero skips, then the corrected
node and complete route module passed 1 and 10 tests respectively. The
controller-owned cache-cleared, host-enabled Step 1 union passed all 1,489 tests
with zero skips, 2,654 warnings, seed 20260829, and duration 996.39s (0:16:36).
Focused Ruff, Python 3.10 compilation, `git diff --check`, exact changed-path
allowlisting, and non-pytest AST marker review passed. No production, schema,
OpenAPI, runtime, or public API file changed. Fix Round 2 is ready for independent
scoped review; Task 12 remains blocked until that review is clean.

### Task 12: Run The Complete PR 2 Verification And Security Gates

**Files:**
- Create: `Docs/Evidence/Admin_Webhooks_PR2_Verification.md`
- Modify: `backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md`

**Interfaces:**
- Consumes: all PR 2 implementation, test output, OpenAPI delta, and approved design gates.
- Produces: reproducible evidence proving backend parity, crash convergence, no-buffer egress, no producer/UI activation, static analysis, and review readiness.

**Preflight ruling:** Verification is evidence-only. Task 12 may create the
evidence artifact and update this plan, the OpenAPI fingerprint only when the
authoritative exporter changes it, and `TASK-13111`; it may not modify
production code or tests. A genuine failure blocks Task 12 and returns to the
owning implementation task for a RED-first fix and review cycle.

Pin scope evidence to merge base
`1ad2f1e5b30c49ea75396e4b713496b73e875fec` through the verified Task 11
closure head, while also recording the observed `origin/dev` object at run
time. Do not fetch, rebase, merge, push, or rewrite history during verification.
Use `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python`, add
`TLDW_TEST_POSTGRES_REQUIRED=1`, `-ra`, and an explicit per-test timeout to
every pytest gate, including the nominal SQLite/security gates, so no
PostgreSQL contract can silently skip. Large opaque commands may be split into
named modules while preserving the exact union; record every command, count,
warning, skip, duration, seed where relevant, and aggregate total.

The plain host `make openapi-fingerprint` target currently selects Python 3.9
and fails on the repository's existing `dataclass(slots=True)` usage. Record
that environment mismatch once; the authoritative fingerprint and drift gates
use `CI_LOCAL_PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python`
with `make openapi-fingerprint` and `make openapi-drift-check`. Review the
fingerprint delta before staging it and leave it untouched when unchanged.

Raw Bandit is expected to return the already reviewed fixed-query B608 and
intentional fail-open observer findings. Capture and classify every hit,
compare it to the established Task 11 baseline, fail on any unreviewed or High
finding, add no suppression, and describe the reviewed result rather than
claiming a misleading raw exit-zero. Query the PostgreSQL server version via
the project driver because no host `psql` binary is installed, and never place
test credentials, DSNs, or private connection details in the evidence file.

- [ ] **Step 1: Run the complete SQLite/API/security matrix**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Security/test_egress.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py
```

Expected: all selected tests pass. Record exact count, warnings, duration, and any skips. A skip in an expected SQLite/security/runtime path blocks review.

- [ ] **Step 2: Run required PostgreSQL and four-backend crash matrix with zero skips**

```bash
RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_migration_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py
```

Expected: all four AuthNZ/Jobs pairs and every enqueue/disposition/cancel crash boundary pass with zero skips. PostgreSQL unavailability blocks review; do not convert it to a documented skip.

- [ ] **Step 3: Run deterministic protocol and security-focused gates separately**

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_executor.py \
  tldw_Server_API/tests/Admin_Webhooks/test_worker.py \
  tldw_Server_API/tests/Admin_Webhooks/test_test_delivery.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py
```

The evidence maps tests to DNS change, private/reserved ranges, redirects, proxies, TLS verification, timeout, response no-buffering, URL redaction, retry classification, `Retry-After`, exact signature vector, no overlapping attempt, hard four-attempt cap, lost lease, late completion rejection, and all disposition recovery paths.

- [ ] **Step 4: Run Ruff, Bandit, diff, and sensitive-data scans**

```bash
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/Jobs/operations \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
../../.venv/bin/python -m bandit -q -r \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/Jobs/migrations.py \
  tldw_Server_API/app/core/Jobs/pg_migrations.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
git diff --check origin/dev...
if rg -n "logger\..*(url|secret|signature|payload|response|ciphertext)|labels=.*(id|host|url|email|secret|payload)" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py; then
  printf 'review possible sensitive delivery telemetry\n' >&2
  exit 1
fi
```

Expected: Ruff/Bandit/diff pass. Review every scan hit; only fixed field names in tests or explicit redaction guards may remain, and each is recorded in evidence. Do not suppress a real sensitive value.

- [ ] **Step 5: Prove PR 3 exclusions and legacy isolation**

```bash
git diff --name-only origin/dev... | rg '(^|/)(admin-ui|users|incidents|admin_system_ops_service|admin_webhooks_service|jobs_webhooks_service)' || true
rg -n "services\.(admin_webhooks_service|jobs_webhooks_service)|from .*admin_webhooks_service|from .*jobs_webhooks_service" \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
```

Expected: no admin UI, user/incident producer, legacy service, or generic Jobs-webhook file is changed/imported. If the first command reports a path, inspect it and remove PR 3 scope before review. The second command returns no match.

- [ ] **Step 6: Re-run and review OpenAPI drift**

```bash
make openapi-fingerprint
make openapi-drift-check
git diff -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected: fingerprint is current and the human-reviewed delta contains only PR 2 test/redelivery/history/status contracts.

- [ ] **Step 7: Write the evidence artifact**

`Docs/Evidence/Admin_Webhooks_PR2_Verification.md` records branch/base/head commits, Python/PostgreSQL versions, exact commands and counts, all four backend pairs, crash-point mapping, signature vector, no-buffer proof, static/security output, OpenAPI review, exclusions, known warnings, and links to `TASK-13111`, the design, plan, and PR. Never include DSNs, tokens, URLs with paths/query, secrets, payloads beyond the published synthetic vector, or receiver content.

- [ ] **Step 8: Commit verification evidence**

```bash
backlog task edit 13111 --append-notes "PR 2 verification complete: exact counts and all four backend/crash/security gates recorded in Docs/Evidence/Admin_Webhooks_PR2_Verification.md."
git add \
  Docs/Evidence/Admin_Webhooks_PR2_Verification.md \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json \
  "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "docs(admin-webhooks): record delivery substrate verification"
```

### Task 13: Request Review And Open PR 2 Without Merging It

**Files:**
- Modify: `backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md`
- Review: every file changed from `origin/dev`.

**Interfaces:**
- Consumes: fully verified implementation branch and evidence.
- Produces: one reviewable PR 2 against `dev`, linked Backlog history, and no local/remote merge action without user choice.

- [ ] **Step 1: Perform a fresh diff and scope review**

```bash
git status --short
git log --oneline origin/dev..HEAD
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/core/Jobs \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py
```

Expected: only PR 2 files are tracked; no unrelated files and no producer/UI/final activation work appear.

- [ ] **Step 2: Invoke `superpowers:requesting-code-review`**

Provide the reviewer the approved design, this plan, `TASK-13111`, PR 1 merge commit, full diff range, and evidence file. Require findings first, ordered by severity, with file/line references and explicit attention to cross-database idempotency, lease loss, attempt overlap, SSRF/body buffering, retry authority, stale-screen replay, terminal monotonicity, sensitive data, and missing PostgreSQL tests.

- [ ] **Step 3: Address review findings rigorously**

Use `superpowers:receiving-code-review` before changing code. Reproduce each valid issue with a failing test, implement the smallest fix, rerun focused and impacted gates, update evidence/task notes, and commit the fix. Document technically invalid suggestions with concrete repository/design evidence rather than applying them blindly.

- [ ] **Step 4: Push the implementation branch and create PR 2**

```bash
git push -u origin codex/admin-webhooks-delivery-substrate
gh pr create --repo rmusser01/tldw_server --base dev \
  --head codex/admin-webhooks-delivery-substrate \
  --title "feat(admin-webhooks): add canonical delivery substrate" \
  --body-file /tmp/admin-webhooks-pr2-body.md
```

The PR body states scope/exclusions, architecture, recovery semantics, test counts, all four backend pairs, security/no-buffer proof, at-least-once behavior, default-off/no-release gate, evidence path, design/plan/task links, and PR 1 dependency. Create `/tmp/admin-webhooks-pr2-body.md` with `apply_patch`; do not include secrets or private infrastructure details.

- [ ] **Step 5: Attach the PR and leave the task open through review**

```bash
PR_URL="$(gh pr view --repo rmusser01/tldw_server --json url --jq .url)"
backlog task edit 13111 \
  --ref "https://github.com/rmusser01/tldw_server/pull/2806" \
  --ref "https://github.com/rmusser01/tldw_server/pull/2828" \
  --ref "$PR_URL" \
  --append-notes "PR 2 opened against dev after full verification and code review. Keep In Progress until review feedback is resolved and the user confirms merge."
git add "backlog/tasks/task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md"
git diff --cached --check
git commit -m "chore(backlog): link admin webhook delivery PR"
git push
```

- [ ] **Step 6: Stop before integration**

Report PR URL, head commit, checks, review findings/resolutions, evidence path, default-off state, and PR 3 exclusions. Do not merge, force-push after review starts, delete the branch/worktree, mark `TASK-13111` Done, or begin durable producers until the user explicitly chooses integration and PR 2 is merged.

---

## Spec-To-Test Traceability

| Approved PR 2 gate | Primary proof |
| --- | --- |
| One automatic delivery per matching registration; set-based fanout | `test_event_expansion.py`, dual-backend repository tests |
| Encrypted body/key rotation/64 KiB | delivery repository and crypto tests |
| Four AuthNZ/Jobs backend combinations at every crash point | `test_recovery_backend_matrix.py` with required PostgreSQL |
| Append-only sequencing/stale unknown/no extra request | `test_worker.py`, recovery matrix |
| Hard four-attempt cap across lease loss | `test_worker.py` |
| Exact Jobs retry/quarantine/lease/defer/fail-closed acquisition | Jobs prepared-operation and prepared-SDK tests |
| Prepared complete/retry/fail/cancel/defer; no double finalization | `test_worker_sdk_prepared.py`, recovery matrix |
| Renewal loss, stale deferral, no overlap, late token rejection | `test_worker.py` |
| Pre-I/O lease/expiry horizon | `test_worker.py`, Jobs lease-horizon tests |
| Configuration/disable/rotation/delete/in-flight races | control-plane and worker tests |
| Manual redelivery, changed-config confirmation | redelivery/history API tests |
| 72-hour expiry and 30-day retention | worker/reconciler/runtime tests |
| Deterministic body and signature vector | `test_executor.py` and runbook |
| SSRF/DNS/proxy/redirect/TLS/timeout/no-buffer/redaction | Security HTTP-hop plus executor tests |
| Retry classification and bounded `Retry-After` | executor and status-only contract tests |
| Synchronous direct-processing/replay/interruption/no Jobs | `test_test_delivery.py` |
| Worker/reconciler health and backlog preflight | retention/health/runtime, control-plane, API tests |
| Transactional first-canonical-activity marking for capture/reservation | repository, event-expansion, worker, and synchronous-test tests |
| Bandit and sensitive-data review | Task 12 evidence |

## Execution Handoff

The dependency gate is satisfied: PR #2806 and tracking PR #2828 are merged, and current `origin/dev` contains final reviewed PR 1 head `f37d4c448ace69b56e208ca1f9bda94c571d86f3`. After this planning artifact is merged, create a fresh isolated implementation worktree from current `origin/dev` and use `superpowers:subagent-driven-development` for task-by-task execution and two-stage review, or `superpowers:executing-plans` for sequential checkpoints. Keep runtime implementation out of the planning PR.
