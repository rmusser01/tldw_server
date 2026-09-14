# Canonical Admin Webhook Durable Producers And Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task by task. Keep `TASK-13145` current after every reviewable commit.

**Goal:** Deliver upstream PR 3 of the approved canonical outgoing-webhook design: six durable production event sources, encrypted incident recovery markers, final canonical-only routing, a complete operational admin UI, controlled-receiver proof, and the documentation/evidence required to enable a private beta safely.

**Architecture:** Shared event-envelope code prepares deterministic, encrypted `EventInsert` records for both synthetic and production capture. User producers join the caller-owned AuthNZ transaction through an explicit webhook unit of work. File-backed incident mutations atomically persist the incident and an encrypted pending marker under the existing `system_ops.json` lock; a dedicated reconciler converts markers into canonical database events and removes each marker only after commit. The final runtime deletes the legacy admin-webhook control path, always mounts the canonical router, and retains `route_selection: "canonical"` only as a response-compatibility field. The admin UI becomes canonical-only and adds delivery history, test, redelivery, and exact incident-notification preview workflows.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, aiosqlite/SQLite, asyncpg/PostgreSQL, AES-GCM webhook key ring, JSON atomic file storage, Next.js 16, React 19, TypeScript 5.9, Vitest, Playwright, pytest, Ruff, Bandit.

**Spec:** `Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md`

**Backlog task:** `TASK-13145`

**Dependency:** Satisfied by PR #2842, merged into `dev` as `7b1450c927de9001975fe50694f37d91eb4ef8d6`.

## Decisions And Constraints

- This is one review unit, PR 3. Do not enable canonical mode in deployment configuration or perform production rollout in the implementation PR.
- The event subscription catalog remains exactly `user.created`, `user.deleted`, `incident.created`, `incident.updated`, `incident.resolved`, and `incident.notify`. `webhook.test` remains reserved and cannot be subscribed to.
- Event bodies are deterministic compact UTF-8 JSON and fail closed above 65,536 bytes. They are never truncated.
- Routine user payload data is exactly stable user ID, lifecycle status, resource version, and lifecycle timestamps. It excludes username, email, password material, sessions, API keys, profile text, billing, organization, and invitation data.
- Routine incident payload data is exactly incident ID, state, severity, integer resource version, and timestamps. It excludes title, summary, tags, timeline messages, evidence, assignee, root cause, impact, runbook URL, and action items.
- `incident.notify` is a separate explicit webhook command, not an implicit consequence of stakeholder email. Its payload may add only the operator-reviewed narrative shown verbatim in the confirmation dialog. Recipient email addresses never enter the webhook event.
- A status transition to `resolved` emits `incident.resolved`; every other effective incident mutation, including timeline append, emits `incident.updated`. A no-op update emits nothing. Incident deletion has no event in this catalog.
- User source identity is a command ID generated before the source transaction and retained for all retries within that command. Incident create/update/resolve identity is `(event_type, incident, incident_id, version)`. Incident notify identity is a scoped digest of one caller-generated idempotency key.
- In mode `on`, a writable primary key and complete migration state are required before starting a source mutation. The transaction or locked file write rechecks writable state as close to commit as its storage boundary permits. Failure aborts the source mutation; no plaintext fallback exists.
- In `off` and `migrate`, ordinary user/incident behavior remains available but production webhook capture is disabled. This is the explicit availability-over-delivery mode decision in the approved design.
- User event and automatic-delivery rows commit in the same AuthNZ transaction as the user insert/deactivation. The source mutation cannot commit without its event in mode `on`.
- Incident record/version and encrypted marker publish in one existing atomic `system_ops.json` save. The marker contains no plaintext body and no receiver data.
- Marker reconciliation is at-least-once and source-deduplicated: insert/expand commits first; marker removal follows under the file lock. A crash may repeat reconciliation but cannot create a second canonical event or automatic delivery.
- The existing key-rotation pending-marker support is retained and extended by tests. Corrupt or undecryptable markers are never discarded or skipped silently; reconciliation degrades visibly and leaves the source file unchanged.
- `TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT=true` becomes a startup configuration error. `false` may be accepted temporarily so old deployment files fail only when they still request legacy behavior.
- `WebhookRouteSelection` and conditional mounting are removed. `GET /admin/webhooks/status` retains `route_selection` narrowed to literal `canonical` for one compatibility cycle.
- Remove the legacy webhook CRUD/test/history and incident-notify-webhooks handlers from `admin_ops`, remove their `system_ops.json` writers, remove `admin_webhooks_service.py`, and remove admin UI legacy fallback. Do not remove `jobs_webhooks_service.py`; it is a separate global Jobs event integration and outside this design.
- Existing legacy source fields remain readable only by migration/restore tooling. Normal store initialization and runtime mutations must not recreate `webhooks` or `webhook_deliveries` fields after sanitization.
- Canonical `on` remains an operator rollout decision. The PR adds a two-phase activation check: `predeploy` validates static dependencies while mode remains `migrate`; `live` validates fresh worker/reconciler/retention heartbeats and backlog after one no-traffic canary starts in `on`. It does not make transient worker loss crash the API because already-active registrations must continue queueing durable events during a worker outage.
- All browser idempotency keys remain memory-only. Test, redelivery, and incident-notify retry the same normalized command/key only after ambiguous transport failure; a new operator action receives a new key.
- No payload editor, custom headers, wildcards, response bodies, full target URLs, secrets, or replay-capable keys are added to the UI, API history, logs, metrics, or audit records.

## File Map

**Create**

- `tldw_Server_API/app/core/Admin_Webhooks/events.py` - deterministic event snapshots, canonical body construction, protected `EventInsert` preparation, and source replay verification shared by synthetic and production capture.
- `tldw_Server_API/app/core/Admin_Webhooks/producer.py` - mode/key preflight, production event contexts, payload allowlists, transaction-bound capture, and incident marker construction.
- `tldw_Server_API/app/core/Admin_Webhooks/incident_reconciler.py` - strict pending-marker reads, idempotent database capture, and post-commit marker removal.
- `tldw_Server_API/tests/Admin_Webhooks/test_production_event_contracts.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_user_producers_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_user_producers_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_incident_producers.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_sqlite.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_postgres.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_final_activation.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_controlled_receiver_e2e.py`
- `Docs/Admin_Webhooks_Receiver_Guide.md`
- `Docs/Evidence/Admin_Webhooks_PR3_Verification.md`

**Modify**

- `tldw_Server_API/app/core/Admin_Webhooks/delivery.py` - use shared event preparation/replay code; keep synthetic capture as a test seam, not a production API.
- `tldw_Server_API/app/core/Admin_Webhooks/domain.py` - final canonical-only status contract and bounded incident-marker validation where needed.
- `tldw_Server_API/app/core/Admin_Webhooks/config.py` - remove route selection and reject requested legacy compatibility.
- `tldw_Server_API/app/core/Admin_Webhooks/__init__.py` - export only reviewed production event interfaces.
- `tldw_Server_API/app/core/Admin_Webhooks/control_plane.py` - canonical-only status and activation check inputs.
- `tldw_Server_API/app/core/Admin_Webhooks/key_rotation.py` - preserve strict marker re-encryption/readback with the production marker collection.
- `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py` - caller-connection unit-of-work factory and source read/capture operations used by producers/reconciliation.
- `tldw_Server_API/app/services/registration_service.py` - transactional `user.created` capture for every registration path.
- `tldw_Server_API/app/services/admin_users_service.py` - service-owned transaction and transactional `user.deleted` capture.
- `tldw_Server_API/app/services/admin_system_ops_service.py` - incident versions, atomic encrypted marker writes, and removal of legacy webhook runtime writers.
- `tldw_Server_API/app/services/admin_webhook_delivery_runtime.py` - pending-incident reconciliation stage and health behavior.
- `tldw_Server_API/app/services/startup_optional_workers.py` - canonical-only runtime enablement; retain the unrelated Jobs webhook worker independently.
- `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py` - mount status and canonical routes unconditionally.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_webhooks.py` - final canonical webhook control/delivery routes and canonical-only status.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py` - await durable incident mutations, expose the canonical producer-backed incident-notify command, and remove legacy webhook handlers/imports.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_user.py` - pass request identity and use the service-owned deletion transaction.
- `tldw_Server_API/app/api/v1/schemas/admin_webhooks.py` - canonical-only status literal and incident notify command response if colocated here.
- `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` - incident version plus bounded notify-webhook request/response contract.
- `tldw_Server_API/cli/commands/admin_webhooks.py` - read-only activation-check command.
- `tldw_Server_API/tests/Admin_Webhooks/test_api.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_openapi.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_admin_webhooks_cli.py`
- `tldw_Server_API/tests/Admin/test_incidents_service.py`
- `tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py`
- `tldw_Server_API/tests/Admin/test_admin_account_audit_events.py`
- `tldw_Server_API/tests/Admin/test_admin_users_service_sanitizers.py`
- `tldw_Server_API/tests/Admin/test_admin_user_endpoint_sanitizers.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_registration_role_membership_postgres.py`
- `tldw_Server_API/tests/Services/test_startup_optional_workers.py`
- `admin-ui/types/webhooks.ts` - full canonical delivery, attempt, test, redelivery, and status types.
- `admin-ui/types/incidents.ts` - incident version and notify-webhook command types.
- `admin-ui/lib/api-client.ts` - canonical history/test/redelivery and incident-notify methods; delete legacy client/detection.
- `admin-ui/app/webhooks/webhook-controller-shared.ts` - canonical status and retry/redelivery helpers.
- `admin-ui/app/webhooks/use-webhook-control-plane.ts` - canonical-only status/catalog/list/history reads.
- `admin-ui/app/webhooks/use-webhooks-page-controller.ts` - test, history, redelivery, and warning workflows; remove legacy state/actions.
- `admin-ui/app/webhooks/use-webhook-secret-commands.ts` - preserve create/rotate secret lifecycle and share memory-only command cleanup where appropriate.
- `admin-ui/app/webhooks/page.tsx` - canonical operational UI and delivery detail view.
- `admin-ui/app/incidents/page.tsx` - exact outbound webhook narrative preview and explicit confirmation.
- `admin-ui/app/webhooks/__tests__/page.test.tsx`
- `admin-ui/lib/api-client-webhooks.test.ts`
- `admin-ui/tests/e2e/webhooks-control-plane.spec.ts`
- `admin-ui/tests/e2e/real-backend/webhooks.spec.ts`
- `admin-ui/app/incidents/__tests__/page.test.tsx`
- Incident page/API client tests covering preview and same-command retry.
- `Docs/Admin_Webhooks_Control_Plane.md`
- `Docs/Admin_Webhooks_Migration_Runbook.md`
- `Docs/Admin_Webhooks_Key_Rotation_Runbook.md`
- `Docs/Admin_Webhooks_Delivery_Runbook.md`
- `Docs/RELEASE_NOTES.md`
- `backlog/tasks/task-13145 - Implement-canonical-admin-webhook-durable-producers-and-final-activation.md`

**Delete**

- `tldw_Server_API/app/services/admin_webhooks_service.py`
- `tldw_Server_API/tests/Admin/test_admin_webhooks_service.py`
- `tldw_Server_API/tests/Admin/test_admin_ops_webhooks_reports.py`
- `tldw_Server_API/tests/Admin_Webhooks/test_route_selection.py`

## Delivery Stages

1. Freeze shared event and production payload contracts.
2. Join user create/delete producers to their AuthNZ source transactions.
3. Add incident versions and atomic encrypted source markers.
4. Reconcile incident markers into canonical events with crash convergence.
5. Remove legacy runtime selection and add final activation checks.
6. Complete the canonical admin Webhooks and incident-notify UI.
7. Prove controlled-receiver, backend, security, documentation, and rollout gates.

### Task 0: Integration Baseline And Task Activation

**Files:**
- Modify: `backlog/tasks/task-13145 - Implement-canonical-admin-webhook-durable-producers-and-final-activation.md`

- [ ] Fetch and prove the merged dependency:

```bash
git fetch origin dev
git merge-base --is-ancestor 7b1450c927de9001975fe50694f37d91eb4ef8d6 origin/dev
git status --short
```

- [ ] Mark `TASK-13145` In Progress and attach this plan with Backlog CLI.

- [ ] Run and record the pre-change baseline:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Admin/test_incidents_service.py \
  tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py \
  tldw_Server_API/tests/Admin/test_admin_account_audit_events.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py
cd admin-ui && bun run test -- app/webhooks lib/api-client-webhooks.test.ts && bun run typecheck
```

Expected: record exact pass/fail/skip counts. PostgreSQL skips are allowed only in this local baseline, not the final required PostgreSQL gate.

- [ ] Commit task activation and plan only:

```bash
git add backlog/tasks/task-13145\ -\ Implement-canonical-admin-webhook-durable-producers-and-final-activation.md \
  Docs/superpowers/plans/2026-08-31-canonical-admin-webhook-durable-producers-activation.md
git diff --cached --check
git commit -m "docs(admin-webhooks): plan durable producers and activation"
```

### Task 1: Shared Event Preparation And Privacy Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/events.py`
- Create: `tldw_Server_API/app/core/Admin_Webhooks/producer.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_production_event_contracts.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/delivery.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py`
- Modify: `tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ProductionEventPreparation:
    event_id: str
    created_at: datetime
    source_component: str
    source_request_id: str | None

class AdminWebhookEventProducer:
    async def begin_capture(...) -> ProductionEventPreparation | None: ...
    async def capture_in_transaction(
        self,
        preparation: ProductionEventPreparation,
        *,
        tx: AdminWebhookUnitOfWork,
        event_type: str,
        source_kind: EventSourceKind,
        aggregate_type: str | None,
        aggregate_id: str | None,
        aggregate_version: str | None,
        source_command_id: str | None,
        data: Mapping[str, object],
    ) -> EventCaptureResult: ...
```

`begin_capture()` returns `None` in `off`/`migrate`, and in `on` it validates migration/key state and generates event ID/timestamp before the source transaction or locked file mutation. Source identity is bound only after the source record/version is known, which keeps incident transition classification inside the file lock. `capture_in_transaction()` locks and revalidates migration/key state, allowlist-validates the event-specific payload, encrypts the canonical body, calls the existing set-based `capture_event_and_expand()`, verifies source replay byte-for-byte, and relies on that transaction's `event_capture` activity mark.

- [ ] Write failing tests for all six exact payload shapes, forbidden fields, timestamps/resource versions, deterministic bytes, invalid JSON, NaN/Infinity, depth bounds, 65,536-byte acceptance, 65,537-byte rejection, encryption at rest, source mismatch, and duplicate replay.
- [ ] Move `_snapshot_json_object`, `_canonical_event_body`, stored-body validation, and replay comparison into `events.py` without changing PR 2 behavior. Keep transport and delivery lifecycle code in `delivery.py`.
- [ ] Add `AdminWebhookRepository.unit_of_work(connection)` so source services do not instantiate backend adapters or issue webhook SQL themselves.
- [ ] Implement event-specific builder functions instead of accepting arbitrary producer dictionaries. Tests must prove no user/incident model dump can leak fields accidentally.
- [ ] Run focused tests and static checks:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_production_event_contracts.py \
  tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py \
  tldw_Server_API/tests/Admin_Webhooks/test_event_expansion.py
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Admin_Webhooks/events.py \
  tldw_Server_API/app/core/Admin_Webhooks/producer.py \
  tldw_Server_API/app/core/Admin_Webhooks/delivery.py \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py
```

- [ ] Commit: `feat(admin-webhooks): add production event contracts`

### Task 2: Transactional User Producers

**Files:**
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_user_producers_sqlite.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_user_producers_postgres.py`
- Modify: `tldw_Server_API/app/services/registration_service.py`
- Modify: `tldw_Server_API/app/services/admin_users_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_user.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_account_audit_events.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_users_service_sanitizers.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_user_endpoint_sanitizers.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_registration_role_membership_postgres.py`

- [ ] Write failing SQLite tests proving `user.created` commits with user, role, audit, event, first-activity marker, and one delivery per matching active registration in one transaction. Assert the decrypted body has only the approved user fields.
- [ ] Add rollback tests for unavailable key, key-primary mismatch, active rotation, migration incomplete, oversized/invalid payload, fanout failure, and mandatory source-transaction failure. Assert zero user/event/delivery rows and directory cleanup where relevant.
- [ ] Generate one command ID before `RegistrationService.register_user()` opens its transaction. After the versioned insert and role/membership writes, read the committed-in-transaction profile version and capture `user.created` through the connection-bound webhook unit of work before returning.
- [ ] Preserve mode-off/migrate registration behavior with no event. Do not add an event only to the admin create endpoint; all paths through `RegistrationService` must use the same producer boundary.
- [ ] Refactor admin deactivation so `admin_users_service.delete_user()` generates its command ID and performs guardrail read, versioned `is_active=false` update, profile-version read, `user.deleted` capture, and existing account audit under a service-owned transaction. Remove the route-owned `get_db_transaction` dependency for this endpoint.
- [ ] Treat an already-inactive target as an effective no-op or stable conflict according to the existing user contract; it must not create another `user.deleted` event. Document the chosen existing-compatible response in tests.
- [ ] Prove source-command replay returns the original event/fanout, same-source/different-body fails, and only the approved stable ID/status/version/timestamps are present.
- [ ] Run SQLite and PostgreSQL producer tests. PostgreSQL final evidence permits zero skips:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_user_producers_sqlite.py \
  tldw_Server_API/tests/Admin/test_admin_account_audit_events.py
RUN_JOBS=1 ADMIN_WEBHOOKS_TEST_POSTGRES=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_user_producers_postgres.py
```

- [ ] Commit: `feat(admin-webhooks): capture transactional user events`

### Task 3: Atomic Incident Versions And Encrypted Markers

**Files:**
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_incident_producers.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/producer.py`
- Modify: `tldw_Server_API/app/core/Admin_Webhooks/domain.py`
- Modify: `tldw_Server_API/app/services/admin_system_ops_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Modify: `tldw_Server_API/tests/Admin/test_incidents_service.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py`

- [ ] Add an integer `version >= 1` to normalized incidents. Existing records without a version normalize to 1; a write persists the normalized version. Create starts at 1 and each effective update/timeline append increments exactly once.
- [ ] Write failing tests proving each create/update/resolve mutation writes the incident and one `PendingIncidentWebhookMarker` in the same `_locked_store(write=True)` atomic save. Assert aggregate identity uses incident ID/version and the marker record contains ciphertext/key ID but no approved payload plaintext, title, summary, tags, timeline, notes, or receiver data.
- [ ] Make incident mutation preparation fail before `_locked_store` when mode-on migration/key state is invalid. Inject atomic-write failure and prove the previous file remains byte-for-byte unchanged.
- [ ] Emit `incident.resolved` only on an effective transition into resolved; otherwise emit `incident.updated`. No-op patches emit neither a version bump nor marker. Reopening a resolved incident emits updated.
- [ ] Replace the legacy direct-dispatch endpoint with one durable, idempotent `POST /admin/incidents/{incident_id}/notify-webhooks` command. Require `Idempotency-Key`, accept only a bounded optional narrative, derive a stable scoped command source ID, and return event/command acceptance metadata without delivery counts. Before appending a marker, resolve the source against both pending file markers and already-reconciled database events; same source/same body is replay and same source/different body is conflict in either state.
- [ ] Build the exact notify body before confirmation in both API tests and UI fixtures. Persist only the encrypted body in the marker; recipient addresses and stakeholder email results are absent. Keep `/incidents/{id}/notify` stakeholder email behavior separate.
- [ ] Prove same key/same request is one command, same key/different narrative is a stable conflict, and a response-loss retry cannot create a second marker/event.
- [ ] Extend marker key-rotation tests for primary/previous readback, rewrite, substitution rejection, malformed collection, duplicate IDs, and key loss. Never drop an unreadable marker.
- [ ] Run:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_incident_producers.py \
  tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py \
  tldw_Server_API/tests/Admin/test_incidents_service.py \
  tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py
```

- [ ] Commit: `feat(admin-webhooks): persist encrypted incident markers`

### Task 4: Incident Marker Reconciliation And Runtime Recovery

**Files:**
- Create: `tldw_Server_API/app/core/Admin_Webhooks/incident_reconciler.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_sqlite.py`
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_postgres.py`
- Modify: `tldw_Server_API/app/services/admin_webhook_delivery_runtime.py`
- Modify: runtime/recovery/observability tests.

**Interface:**

```python
class PendingIncidentEventReconciler:
    async def reconcile_once(self, *, limit: int = 100) -> int: ...
```

- [ ] Read markers strictly under the file lock into immutable validated records, then release the lock before database work. Bound each pass to 100 markers and use deterministic `(created_at, event_id)` order without promising receiver order.
- [ ] For each marker, decrypt and validate its canonical body and source identity, then capture/expand in one AuthNZ transaction. Existing source identity is an idempotent read and must pass exact-body replay verification.
- [ ] After database commit, reacquire the file lock, reread strictly, and remove only the exact marker whose full stored record still matches. Atomic-save failure leaves the marker for retry.
- [ ] Cover crash points: before DB transaction, after event insert, after DB commit/before removal, after in-memory removal/before atomic save, and after removal. Every restart converges to one event and one automatic delivery per registration.
- [ ] Cover a marker replaced while reconciliation is in flight; compare-and-remove must preserve the replacement. Cover corrupt, oversized, undecryptable, and source-conflicting markers as fail-closed with no file mutation.
- [ ] Add `reconcile_once()` before enqueue/disposition repair in `_run_reconciler_loop`; a marker failure degrades that iteration but does not execute unsafe deletion or stop worker/retention peers. Record bounded metrics/health only, with no event body or incident narrative labels/logs.
- [ ] Run both backends and runtime tests:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_sqlite.py \
  tldw_Server_API/tests/Admin_Webhooks/test_retention_health_runtime.py
RUN_JOBS=1 ADMIN_WEBHOOKS_TEST_POSTGRES=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_postgres.py
```

- [ ] Commit: `feat(admin-webhooks): reconcile durable incident events`

### Task 5: Final Canonical Routing And Activation Check

**Files:**
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_final_activation.py`
- Modify: config/domain/control-plane/router/startup/CLI/OpenAPI tests listed in the file map.
- Delete: legacy admin webhook service and runtime tests listed in the file map.

- [ ] Write failing route-inventory tests that enumerate every `/api/v1/admin/webhooks` and `/notify-webhooks` method/path and assert exactly one matching route, platform-admin protection, and canonical response schemas. Assert no string-ID legacy route and no legacy payload field names remain in OpenAPI.
- [ ] Remove `WebhookRouteSelection`, `_mount_admin_webhook_routes()` selection logic, `legacy_webhooks_router`, legacy admin webhook endpoint imports, system-ops webhook CRUD/delivery functions, and admin webhook direct-dispatch service.
- [ ] Always mount status plus canonical routes. Keep `route_selection: Literal["canonical"]` in status for compatibility. Reject `TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT=true` with a fixed sanitized startup error.
- [ ] Stop normal system-ops store initialization from recreating `webhooks`/`webhook_deliveries`; preserve strict legacy importer/restore reads for existing files.
- [ ] Add `admin-webhooks activation-check --phase predeploy|live` as a read-only command that reports only closed readiness fields and changes no mode or data. `predeploy` requires schema, migration, key primary, Jobs database/queue/type, limits, and no incompatible legacy selection while still in `migrate`. `live` runs only against the no-traffic `on` canary and additionally requires fresh worker/reconciler/retention heartbeats plus the configured backlog-age threshold. A phase mismatch exits nonzero.
- [ ] Preserve transient-degradation behavior after activation: existing active registrations remain active and production events remain durable when delivery workers go offline; only new activation is blocked.
- [ ] Remove temporary warning/copy about selecting compatibility mode from API, admin UI fixtures, runbooks, environment examples, and tests.
- [ ] Run:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_final_activation.py \
  tldw_Server_API/tests/Admin_Webhooks/test_openapi.py \
  tldw_Server_API/tests/Admin_Webhooks/test_admin_webhooks_cli.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py
```

- [ ] Commit: `feat(admin-webhooks): activate canonical-only routes`

### Task 6: Canonical API Client And Operational Webhooks UI

**Files:**
- Modify: all `admin-ui` webhook files/tests in the file map.

- [ ] Extend strict client validators/types for full `delivery` status, delivery/attempt history, persisted test responses including 202/`Retry-After`, and 202 manual redelivery. Invalid success bodies remain visible contract errors.
- [ ] Delete `legacyWebhookApi`, `detectWebhookApi`, `LegacyWebhookView`, legacy delivery DTOs, mode detection, legacy UI state/actions, and 404/malformed-status fallback. A status failure is an actionable failure, never a downgrade.
- [ ] Load canonical status first. In `off`/`migrate`, show exact mode guidance without issuing blocked catalog/list calls. In `on`, show key, worker, reconciler, retention, backlog counts, oldest age, and activation reason using closed values.
- [ ] Add per-registration expandable delivery history with kind/state, event type, attempt sequence, status code/class, latency, bounded reason, retry delay, versions, timestamps, expiry, and redelivery linkage. Do not expose payload, destination URL, request headers, signature, or receiver content.
- [ ] Add canonical test using current ETag/config version and a fresh memory-only idempotency key. Ambiguous retry and in-progress polling reuse the same command; a deliberate new test uses a new key. Show test-versus-automatic header semantics in labels only where the API contract supports them.
- [ ] Add manual redelivery from an eligible history row. Fetch the current registration/ETag immediately before confirmation. When versions differ, display old/current versions and current redacted hostname, require explicit changed-configuration confirmation, and set `confirm_changed_configuration=true` only then.
- [ ] Add warnings before disable, rotate, delete, and configuration-changing updates that queued work may be canceled/superseded but an in-flight HTTP request cannot be recalled.
- [ ] Extend synchronous `pagehide`/persisted-`pageshow` cleanup to every sensitive command key/request and one-time secret reference. Assert no key or secret enters local/session storage, URL, logs, or rendered state after cleanup.
- [ ] Keep row/button dimensions stable and use existing Lucide icons/tooltips. Delivery details are an unframed expandable table section, not nested cards.
- [ ] Run:

```bash
cd admin-ui
bun run test -- app/webhooks lib/api-client-webhooks.test.ts
bun run typecheck
bun run lint
bun run build
bunx playwright test tests/e2e/webhooks-control-plane.spec.ts --project=chromium
```

- [ ] Commit: `feat(admin-ui): complete canonical webhook operations`

### Task 7: Incident Notify Preview UI

**Files:**
- Modify: `admin-ui/app/incidents/page.tsx`
- Modify: `admin-ui/lib/api-client.ts`
- Modify: `admin-ui/types/incidents.ts`
- Modify/Create: focused incident page and E2E tests.

- [ ] Remove the dead `notifyIncident()` call that posts an empty body to the stakeholder-email route.
- [ ] Add a distinct "Queue webhook notification" command. The dialog shows the exact outbound event preview: incident ID, status, severity, version, timestamp semantics, and the optional bounded narrative. It shows no email recipient data or hidden incident fields.
- [ ] Require explicit confirmation after preview. Generate one memory-only idempotency key per operator command and reuse it only for ambiguous transport retry. Disable close/navigation behavior consistently while command outcome is ambiguous, then clear synchronously on `pagehide`.
- [ ] Keep stakeholder email notification separate and unchanged except for copy that distinguishes email from outgoing webhook delivery.
- [ ] Test preview/body equality, narrative bounds, cancel/no request, same-command retry, same-key conflict, no secret/key storage, and accessible focus/error states.
- [ ] Run focused tests, typecheck, lint, and Playwright incident workflow.
- [ ] Commit: `feat(admin-ui): add incident webhook preview command`

### Task 8: Controlled Receiver And Backend Activation Matrix

**Files:**
- Create: `tldw_Server_API/tests/Admin_Webhooks/test_controlled_receiver_e2e.py`
- Modify: `admin-ui/tests/e2e/real-backend/webhooks.spec.ts`
- Modify: `admin-ui/package.json`

- [ ] Start a local controlled HTTPS receiver through the existing safe test transport seam. Capture exact bytes and approved webhook headers only in test memory.
- [ ] Prove all six automatic producers, synchronous `webhook.test`, and manual redelivery produce valid HMAC signatures and canonical IDs/timestamps. Assert automatic duplicates retain one event/delivery identity and receiver deduplicates by event/delivery identifiers.
- [ ] Prove retry classification and exact retry schedule with one injected 503 then success; prove no retry for terminal 4xx; prove test performs one attempt only; prove redelivery has a new delivery ID and historical event ID.
- [ ] Run the same producer/capture/expansion assertions on SQLite and PostgreSQL AuthNZ. Run all four supported AuthNZ/Jobs backend combinations for enqueue/delivery recovery with zero PostgreSQL skips.
- [ ] Run real-backend admin UI flow: create inactive registration, store secret, enable after healthy status, create user and incident sources, inspect history, test, redeliver, rotate, disable, and confirm redacted UI.
- [ ] Run the security regressions for SSRF, DNS rebinding, redirects, proxies, TLS, no response buffering, payload-size boundaries, and log/audit redaction.
- [ ] Commit: `test(admin-webhooks): prove production delivery activation`

### Task 9: Receiver Docs, Runbooks, Evidence, And Release Gate

**Files:**
- Create: `Docs/Admin_Webhooks_Receiver_Guide.md`
- Create: `Docs/Evidence/Admin_Webhooks_PR3_Verification.md`
- Modify: webhook runbooks/control-plane/release notes and `TASK-13145`.

- [ ] Document the public event envelope, exact six-event payload schemas, HMAC verification, timestamp tolerance, constant-time comparison, event/delivery deduplication, at-least-once delivery, unordered events, retry schedule, test headers, and manual redelivery semantics. Include examples with synthetic IDs and no real credentials.
- [ ] Update migration/key/delivery runbooks for canonical-only routes, pending marker backup/restore/readback, key-loss behavior, dead delivery inspection, retention, activation check, and safe disable.
- [ ] Document rollout: remain `migrate`; complete import/readback; rotate imported secrets; provision key ring; pass `activation-check --phase predeploy`; deploy one no-traffic canary in `on`; pass `activation-check --phase live`; perform controlled automatic/test/redelivery probes; then expand. Record that rollback after first `event_capture` is disable-and-forward-fix, never return to the legacy writer.
- [ ] Record exact commands, versions, commit SHA, pass/fail/skip counts, PostgreSQL fixture identity class (never credentials), controlled receiver results, UI screenshots where useful, and known residual risks in PR 3 evidence.
- [ ] Run the aggregate gate:

```bash
RUN_JOBS=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks \
  tldw_Server_API/tests/Admin/test_incidents_service.py \
  tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py \
  tldw_Server_API/tests/Admin/test_admin_account_audit_events.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py
RUN_JOBS=1 ADMIN_WEBHOOKS_TEST_POSTGRES=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Admin_Webhooks/test_user_producers_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_incident_marker_reconciler_postgres.py \
  tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py
../../.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/registration_service.py \
  tldw_Server_API/app/services/admin_users_service.py \
  tldw_Server_API/app/services/admin_system_ops_service.py \
  tldw_Server_API/app/api/v1/endpoints/admin \
  tldw_Server_API/app/api/v1/schemas/admin_webhooks.py \
  tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/cli/commands/admin_webhooks.py
../../.venv/bin/python -m bandit -q -r \
  tldw_Server_API/app/core/Admin_Webhooks \
  tldw_Server_API/app/services/admin_webhook_delivery_runtime.py \
  tldw_Server_API/app/services/registration_service.py \
  tldw_Server_API/app/services/admin_users_service.py \
  tldw_Server_API/app/services/admin_system_ops_service.py
cd admin-ui && bun run test && bun run typecheck && bun run lint && bun run build
git diff --check
```

- [ ] Request independent code review. Resolve every technically valid finding and record rejected findings with evidence in `TASK-13145`.
- [ ] Complete `TASK-13145`, attach evidence and PR URL, and commit: `docs(admin-webhooks): record PR 3 activation evidence`.

## Final Acceptance Checklist

- [ ] Six production producers emit only approved encrypted payloads with stable source identity.
- [ ] User source mutation and event/fanout are one AuthNZ transaction on SQLite and PostgreSQL.
- [ ] Incident source mutation/version and encrypted marker are one atomic file save.
- [ ] Every marker crash window converges to one canonical event and bounded fanout.
- [ ] `event_capture` closes the structural legacy-restore window transactionally.
- [ ] Canonical router is the sole admin webhook runtime; legacy admin handlers and UI fallback are absent.
- [ ] Status compatibility says only `route_selection: canonical`; requested legacy mode fails at startup.
- [ ] Admin UI covers create/edit/secret/enable/disable/delete/status/history/test/redelivery and incident notify preview without secret or key persistence.
- [ ] Controlled receiver proves signatures, duplicates, retry, test, automatic producers, and manual redelivery.
- [ ] SQLite/PostgreSQL, security, UI test/typecheck/lint/build, Bandit, docs, and evidence gates pass.
- [ ] Deployment remains disabled until the separately executed private-beta activation runbook completes.
