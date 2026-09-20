# Claims Jobs Operational Control Plane Design

## Goal

Move Claims background work onto the existing core Jobs module in stages, starting with rebuild and notification delivery, then expanding to all admin-visible Claims background work and recurring orchestration.

Backlog task: TASK-9935.

## Design Principles

Jobs owns background-work mechanics: persistence, queues, leases, retries, backoff, pause/resume/drain, quarantine, admin controls, lifecycle metrics, events, and worker SDK behavior.

Claims owns domain contracts only: job type names, JSON payload validation, enqueue helpers, idempotency keys, handler dispatch, and business logic for one unit of Claims work.

The refactor must not recreate queue handling in Claims. Claims may summarize Jobs state for domain dashboards, but queue controls stay in the existing Jobs admin surface.

Public Claims APIs should keep their current behavior. The refactor changes how background work is executed and observed, not the external request/response contracts.

## Staged Rollout

### Stage 1: Rebuild And Notification Jobs

Stage 1 migrates the smallest production background surface:

- Claims rebuild for one media item.
- Review notification delivery.
- Alert delivery.

`claims_service.py` remains the API-facing facade. When Jobs mode is enabled, service code enqueues Jobs and does not start local daemon execution. When Jobs mode is disabled, the existing bounded local rebuild, review-notification, and alert-delivery compatibility paths remain available temporarily.

Stage 1 adds:

- `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`
- `tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py`
- `tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py`
- `tldw_Server_API/app/services/claims_jobs_worker.py`

The new worker uses `WorkerSDK`. It does not implement a custom lease loop.

### Stage 2: Remaining Admin-Visible Claims Work

Stage 2 adds Jobs coverage for the rest of Claims background work that users or admins need to observe:

- Analytics exports.
- Review metrics aggregation.
- Cluster rebuilds.
- Any remaining Claims background task with retries, status, or admin controls.

Stage 2 makes the shared Claims job error/result conventions mandatory for all Claims job handlers.

### Stage 3: Full Claims Control Plane

Stage 3 completes the operational model:

- Scheduler/APScheduler owns recurring enqueue decisions.
- Jobs owns execution, status, retries, leases, and admin behavior.
- Claims dashboards combine Jobs summaries with Claims DB domain outcomes.
- Old in-memory rebuild health and local daemon-dispatch paths are removed after parity is proven.

Scheduler is used only for recurring orchestration decisions. It does not replace Jobs as the durable execution/status system for user/admin-visible Claims work.

## Module Responsibilities

`claims_job_contracts.py` defines:

- `CLAIMS_JOBS_DOMAIN = "claims"`
- Job type constants.
- Payload version constants.
- Payload validators.
- Result shape helpers.
- Claims job error types that expose the attributes consumed by `WorkerSDK`: `retryable`, `failure_code`, and optional `backoff_seconds`.

`claims_jobs.py` defines:

- Queue name resolution.
- Jobs-enabled flag resolution.
- Enqueue helpers that call `JobManager.create_job(...)`.
- Idempotency-key builders.

`claims_job_handlers.py` defines:

- Handler dispatch by `job_type`.
- Business handlers for one Claims job.
- Mapping domain outcomes to small, non-sensitive result dictionaries.

`app/services/claims_jobs_worker.py` defines:

- Worker startup entrypoint.
- `WorkerConfig` using `domain="claims"` and the configured queue.
- `WorkerSDK` integration.
- Stop-event wiring for app shutdown.

`claims_service.py` keeps the public facade role. It chooses local compatibility dispatch or Jobs enqueue based on configuration, and it does not own Jobs mechanics.

## Stage 1 Job Types

Stage 1 uses `domain="claims"` with these job types:

- `claims_rebuild_media`
- `claims_deliver_review_notification`
- `claims_deliver_alert`

Payloads include `version=1`, `owner_user_id`, stable IDs, and no raw DB paths or message bodies.

### `claims_rebuild_media`

Payload:

```json
{
  "version": 1,
  "owner_user_id": "1",
  "media_id": 123
}
```

Idempotency key:

```text
claims:rebuild:{owner_user_id}:{media_id}
```

Handler behavior:

- Resolve the user media DB path from `owner_user_id`.
- Validate that the media row belongs to the owner.
- Run the existing strict rebuild logic.
- Complete with a small result such as `{ "outcome": "ok", "media_id": 123, "deleted": 2, "inserted": 3 }`.
- Complete as skipped when the media row is missing or no longer eligible.
- Fail retryably for transient DB locks or provider/network failures.
- Fail non-retryably for invalid payloads or owner-scope violations.

### `claims_deliver_review_notification`

Payload:

```json
{
  "version": 1,
  "owner_user_id": "1",
  "notification_ids": [10, 11]
}
```

Idempotency key:

```text
claims:notify_review:{owner_user_id}:{sha256(sorted_notification_ids)}
```

Handler behavior:

- Resolve the user media DB path from `owner_user_id`.
- Reload notification rows by ID.
- Drop already-delivered IDs before attempting delivery.
- Complete as skipped when settings are disabled or no undelivered notifications remain.
- Deliver through existing review notification helpers.
- Mark delivered rows in the Claims DB only after successful delivery.
- Fail retryably for transient delivery errors.
- Fail non-retryably for invalid payloads or owner-scope violations.

### `claims_deliver_alert`

Payload:

```json
{
  "version": 1,
  "owner_user_id": "1",
  "event_id": 555,
  "alert_id": 9,
  "channel": "webhook"
}
```

Allowed channels are `slack`, `webhook`, and `email` only if the existing alert email path is migrated in the same stage.

Idempotency key:

```text
claims:alert:{owner_user_id}:{event_id}:{alert_id}:{channel}
```

Handler behavior:

- Resolve the user media DB path from `owner_user_id`.
- Reload the monitoring event and alert settings from the Claims DB.
- Skip if an attempt for `{event_id, alert_id, channel}` already succeeded.
- Persist delivery attempts before and after delivery.
- Complete as skipped when the alert is disabled, channel is disabled, or a previous attempt already succeeded.
- Fail retryably for transient delivery errors.
- Fail non-retryably for invalid payloads, unsupported channels, missing owner scope, or unsafe state.

Alert delivery must not move to Jobs until the Claims DB can persist delivery attempts keyed by `{event_id, alert_id, channel}`. If the current schema cannot do this, Stage 1 must first add a minimal delivery-attempt table or equivalent Media DB helper, then migrate alert delivery onto Jobs.

## Payload And Data Safety

Jobs payloads must not include:

- Raw DB paths.
- Webhook URLs.
- Email recipients.
- Full claim text.
- Notification bodies.
- Alert event payloads.
- Secrets or provider keys.

Workers derive DB paths from `owner_user_id` using existing user database helpers. Jobs must use the real media/settings owner. If a call site cannot resolve a real owner, enqueue should fail before job creation or the handler should fail non-retryably; Claims Jobs must not use `0`, blank, or synthetic owner IDs as a fallback.

A transitional `db_path` field is allowed only for a temporary compatibility step, and only if the worker validates that the path is under the expected user database root and matches the owner.

Jobs results must be small, JSON-serializable, and non-sensitive. Results may include IDs, counts, domain outcome strings such as `ok`, `skipped`, or `partial`, and reason codes. Avoid duplicating the Jobs lifecycle status inside the result payload unless the existing Jobs convention requires it.

Jobs idempotency is scoped by the existing Jobs uniqueness boundary: `(domain, queue, job_type, idempotency_key)`. Claims idempotency keys must be stable within that boundary. Changing `CLAIMS_JOBS_QUEUE` can create a separate dedupe scope, so rollout and retry documentation should call that out.

## Configuration

Stage 1 adds or formalizes:

- `CLAIMS_JOBS_ENABLED`: route Claims background work through Jobs when true.
- `CLAIMS_JOBS_WORKER_ENABLED`: start the Claims Jobs worker when true.
- `CLAIMS_JOBS_QUEUE`: queue name, default `default`.
- `CLAIMS_JOBS_MAX_RETRIES_REBUILD`
- `CLAIMS_JOBS_MAX_RETRIES_REVIEW_NOTIFICATION`
- `CLAIMS_JOBS_MAX_RETRIES_ALERT`
- `CLAIMS_JOBS_LEASE_SECONDS`
- Optional Claims-specific retry/backoff settings, falling back to core Jobs defaults.

`CLAIMS_JOBS_WORKER_ENABLED=false` means no local Claims worker processes jobs. It does not disable enqueue when `CLAIMS_JOBS_ENABLED=true`; queued jobs remain visible through Jobs/admin status.

## Routing Rules

Each background action has one routing point:

- If `CLAIMS_JOBS_ENABLED=false`, use the existing bounded local path.
- If `CLAIMS_JOBS_ENABLED=true`, enqueue a Jobs record only.

There is no local fallback after a successful enqueue. Worker-disabled or worker-unavailable states surface as queued Jobs, not hidden local execution.

If enqueue fails before a job is created, behavior must be explicit per call site:

- Explicit admin/user-triggered background actions, such as a manual Claims rebuild, should fail visibly according to the current API error pattern because the requested work was not accepted.
- Best-effort side effects, such as review notification delivery after a successful review state change, should not roll back the primary user action, but must record/log the enqueue failure and expose it through metrics or an existing operational surface.
- Worker-disabled or worker-unavailable states are not enqueue failures; they create queued Jobs that remain visible through Jobs/admin status.

## Failure And Retry Taxonomy

Claims handlers should raise a small Claims job error type that matches the attributes consumed by `WorkerSDK`:

```text
ClaimsJobError(message, retryable=True|False, failure_code="...", backoff_seconds=None)
```

`failure_code` is the value Jobs stores for handler failures. Use result `reason` values for completed-domain outcomes such as skipped work.

Retryable examples:

- Transient SQLite lock.
- Network failure.
- Egress retryable failure.
- Provider timeout where existing Claims behavior treats retry as useful.

Non-retryable examples:

- Invalid payload version.
- Missing required owner ID.
- Unsupported job type.
- Unsupported channel.
- Owner-scope violation.
- Unsafe or mismatched transitional DB path.

Skipped outcomes complete the Jobs record with a result such as `{ "outcome": "skipped", "reason": "already_delivered" }`. Skipped is a domain result, not a separate Jobs lifecycle status.

## Dashboard And Admin Behavior

Jobs remains the source of truth for queued, running, retrying, failed, cancelled, and quarantined work.

Claims DB remains the source of truth for claim rows, notification delivery timestamps, alert delivery attempts, review metrics, and analytics output.

Claims dashboards may add a `claims jobs` summary by reading Jobs with `domain="claims"`. Queue controls such as pause, resume, drain, retry, requeue, quarantine, and prune should route to existing Jobs admin endpoints and use existing Jobs RBAC.

Claims should not add duplicate queue-control APIs.

## Startup And Shutdown

`app/services/claims_jobs_worker.py` should mirror existing Jobs worker startup patterns:

- Read `CLAIMS_JOBS_WORKER_ENABLED`.
- Resolve worker ID and queue.
- Build `WorkerConfig`.
- Run `WorkerSDK` with a stop-event bridge.
- For the normal global Claims worker, do not pass an `owner_user_id` acquisition filter. The worker should acquire all jobs in the Claims domain/queue, while ownership is enforced by the Jobs row owner and the handler's payload/domain validation. Per-owner workers are a separate deployment choice and must be documented if introduced.
- Let cancellation propagate.
- Avoid broad exception tuples that swallow cancellation.

Startup wiring should live alongside existing content Jobs poller/worker startup code.

## Migration Plan

Stage 1 starts opt-in:

- Default `CLAIMS_JOBS_ENABLED=false`.
- Default `CLAIMS_JOBS_WORKER_ENABLED=false` unless the project chooses to start workers by default in development.
- Keep bounded local dispatch for disabled mode.
- Add docs/runbook notes for enabling Claims Jobs and checking Jobs admin status.

After parity tests and operations validation:

- Flip `CLAIMS_JOBS_ENABLED=true` by default.
- Keep local fallback available for one release.
- Mark in-memory rebuild health as deprecated when Jobs mode is enabled.

After Stage 3:

- Remove old local daemon dispatch and in-memory rebuild health.
- Recurring work enqueues through Scheduler/APScheduler into Jobs.

## Testing Plan

Stage 1 tests:

- Payload validation for valid payloads, missing owner, invalid IDs, unsupported channel, unsupported version, and unsupported job type.
- Enqueue contract tests proving `JobManager.create_job(...)` receives `domain="claims"`, expected `job_type`, queue, owner, idempotency key, max retries, and ID-only payload.
- Routing tests for Jobs disabled mode, Jobs enabled mode, worker disabled mode, enqueue failure defaults for explicit versus best-effort call sites, and no duplicate local execution after enqueue.
- Rebuild handler tests for success, missing media skip, owner mismatch non-retryable failure, duplicate rebuild idempotency, and strict replacement failure retry behavior.
- Review notification handler tests for already-delivered skip, disabled settings skip, partial undelivered delivery, successful delivery marking, and retryable delivery failure.
- Alert delivery handler tests for reloading event/settings from DB, persisted attempt creation, successful attempt dedupe across retry, invalid channel non-retryable failure, and transient delivery retry.
- Worker dispatch smoke tests for supported Claims job types, unsupported type handling, `failure_code` propagation, retryable failure behavior, and non-retryable failure behavior.
- Multi-user scope tests proving workers derive DB paths from `owner_user_id` and cannot process another owner data set. If the existing Jobs test harness supports PostgreSQL/RLS coverage, include at least one owner-scope test against that path.
- Dashboard summary tests proving Claims dashboards read Jobs summaries without exposing queue controls outside Jobs RBAC.

Stage 2 and Stage 3 add tests for analytics export jobs, review metrics aggregation jobs, cluster rebuild jobs, recurring enqueue decisions, and removal of local daemon paths.

## Risks And Mitigations

Duplicate work risk is mitigated with idempotency keys, no local fallback after enqueue, and handler-level delivered/attempt checks.

Sensitive data persistence risk is mitigated by ID-only payloads and non-sensitive result summaries.

Owner-scope risk is mitigated by deriving DB paths from owner ID and validating domain rows belong to that owner before processing.

Operational drift risk is mitigated by keeping Jobs as the only queue/lifecycle owner and linking Claims dashboard controls to existing Jobs admin routes.

Migration risk is mitigated with an opt-in flag, compatibility local path, parity tests, and staged default flip.

## Out Of Scope

This refactor does not rename public Claims endpoints, replace the core Jobs module, rewrite extraction strategies, change claim schemas, or move queue/admin lifecycle mechanics into Claims.

It does not move all Claims work in Stage 1. Analytics exports, review metrics aggregation, cluster rebuilds, and recurring orchestration are Stage 2 and Stage 3 work.

## Spec Review

- Placeholder scan: no placeholders remain.
- Consistency check: Jobs owns queue lifecycle throughout; Claims owns domain contracts and handlers only.
- Scope check: Stage 1 is implementation-sized; Stage 2 and Stage 3 are explicitly follow-up stages.
- Ambiguity check: toggles, payload contents, idempotency scope, retry/skipped semantics, WorkerSDK failure attributes, owner DB resolution, worker acquisition scope, enqueue failure defaults, and dashboard/admin boundaries are explicit.
