# Scheduled Tasks Phase 4B Backend API Foundation Design

Date: 2026-06-09
Status: Needs User Review
Owner: Codex brainstorming session
Backlog: TASK-2349

Related:

- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md`
- `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4a-api-first-planned-shell-implementation-plan.md`
- `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
- `Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md`
- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md`
- `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md`
- `Docs/ADR/003-jobs-vs-scheduler-default.md`

## Summary

Phase 4B turns the Phase 4A planned Recurring Question and Agent Task shell into a real API-managed foundation. It adds Scheduled Tasks-owned durable definitions, durable preview records, lifecycle mutations, definition audit, capability discovery, and reference-client WebUI behavior.

4B does not execute scheduled work. It does not dispatch RAG queries, send agent messages, create run history, create results, enqueue Jobs, create approval requests, or deliver notifications. Its purpose is to make the API contract, storage boundary, validation behavior, and client management workflow stable before 4C Recurring Question execution and 4D Agent Task execution.

The API remains the product source of truth. The WebUI is the reference and main enterprise client over this API, not the only client.

## Product Decision

Use **Scheduled Tasks-owned persisted definitions** for `recurring_question` and `agent_task`.

RAG and ACP are related capability providers. They do not own the new definitions in 4B. Existing `/acp/schedules` remains a separate expert surface and must not be treated as full Agent Task support until preview, risk, approval, result, and audit contracts are implemented.

Definitions are persisted after successful server preview. Create and update require a valid, unexpired preview record. Duplicate copies an existing definition into `paused` lifecycle without requiring a fresh preview and does not accept inline config overrides in 4B. If the user edits the duplicate, the follow-up update requires a new update preview.

## Scope

In scope:

- capability discovery for Scheduled Tasks automation families;
- durable preview records for `create` and `update` preview modes;
- persisted definitions for `recurring_question` and `agent_task`;
- lifecycle mutations: create, update, pause, resume, archive, duplicate;
- definition audit events;
- pagination and filtering for definitions, previews, and audit events;
- optional `Idempotency-Key` support for preview, create, update, duplicate, and lifecycle mutations;
- dedicated Scheduled Tasks repository interface with per-user SQLite as the first backend;
- normalized control-plane projection of new definitions into `/api/v1/scheduled-tasks`;
- WebUI reference-client wiring for capabilities, preview, create/edit, lifecycle, preview history, and definition audit;
- explicit execution-unavailable states in task detail, Results, and Home copy.

Out of scope:

- scheduled execution;
- manual run;
- Jobs enqueueing or worker implementation;
- Scheduler integration;
- RAG query execution;
- ACP/API agent dispatch;
- approval queue APIs and approve/deny mutations;
- run list/detail endpoints;
- result list/detail endpoints beyond existing Phase 3 projected behavior;
- notification delivery;
- hard delete or retention purge;
- migration, wrapping, or replacement of `/acp/schedules`;
- moving or reducing Watchlists functionality.

## Current System Constraints

Current `origin/dev` constraints that shape this design:

- `GET /api/v1/scheduled-tasks` normalizes reminder tasks and Watchlists jobs.
- Reminder tasks are native to `/scheduled-tasks`; Watchlists jobs are externally managed and deep-link to Watchlists.
- `ScheduledTaskPrimitive` currently includes `reminder_task` and `watchlist_job`.
- `ScheduledTasksControlPlaneService` reads reminders from `CollectionsDatabase` and Watchlists jobs from `WatchlistsDatabase`.
- Phase 3 Results/Home surfacing supports projected signals and future normalized results modes.
- Phase 4A WebUI shows Recurring Question and Agent Task as honest planned templates.
- `Docs/ADR/003-jobs-vs-scheduler-default.md` says new user-visible execution should default to Jobs unless an explicit Scheduler exception is accepted.

4B must be additive and compatibility-preserving.

## Architecture

4B adds a backend module with three separable responsibilities.

| Unit | Responsibility |
| --- | --- |
| API schemas | Stable request and response models for capabilities, previews, definitions, schedules, visibility, lifecycle, idempotency, and audit. |
| Scheduled Tasks store | Repository interface plus first per-user SQLite implementation for definitions, previews, idempotency records, and audit events. |
| Control-plane service | Capability evaluation, preview validation, definition lifecycle, audit recording, and normalized task projection. |

Storage should be interface-first. The first backend should be per-user SQLite:

`Databases/user_databases/<user_id>/ScheduledTasks.db`

The API contract must not expose this storage topology. A future shared tenant database can implement the same repository interface.

The existing `ScheduledTasksControlPlaneService` can either be extended or composed with a new collaborator. It should continue to project reminders and Watchlists jobs exactly as today, then add `automation_definition` rows for the new Scheduled Tasks-owned definitions.

## API Resource Shape

Use resource-oriented endpoints under the existing namespace.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/scheduled-tasks/capabilities` | Discover family support, related dependency readiness, supported actions, and disabled reasons. |
| `POST /api/v1/scheduled-tasks/previews` | Validate and normalize a proposed definition; persist a durable preview record. |
| `GET /api/v1/scheduled-tasks/previews` | Paginated and filterable preview history. |
| `GET /api/v1/scheduled-tasks/previews/{preview_id}` | Inspect preview output, validation errors, normalized config, and expiry. |
| `POST /api/v1/scheduled-tasks/definitions` | Create a definition from a valid preview record. |
| `GET /api/v1/scheduled-tasks/definitions` | Paginated and filterable persisted definitions. |
| `GET /api/v1/scheduled-tasks/definitions/{definition_id}` | Inspect a full definition. |
| `PATCH /api/v1/scheduled-tasks/definitions/{definition_id}` | Update a definition through preview-backed validation. |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/pause` | Pause a definition. |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/resume` | Resume a definition. |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/archive` | Archive a definition. |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/duplicate` | Copy config without history into `paused` lifecycle. |
| `GET /api/v1/scheduled-tasks/definitions/{definition_id}/audit` | Paginated and filterable definition audit events. |

Static child routes must be registered before the existing `GET /api/v1/scheduled-tasks/{task_id}` route. `GET /capabilities`, `GET /previews`, and `GET /definitions` must never be interpreted as task IDs. Add route regression tests that assert these static routes resolve to their intended handlers.

Existing endpoints stay intact:

- `GET /api/v1/scheduled-tasks`
- `GET /api/v1/scheduled-tasks/{task_id}`
- reminder mutation endpoints

## Capability Discovery

`GET /api/v1/scheduled-tasks/capabilities` is the first-class discovery endpoint for Scheduled Tasks automation families.

The response should separate **Scheduled Tasks family availability** from **related domain readiness**.

Example fields:

- `family`
- `family_availability`
- `actions`
- `missing_dependencies`
- `related_capabilities`
- `reason`
- `schema_version`

Availability values:

- `available`
- `planned`
- `unavailable`
- `degraded`

Actions should be explicit per-action status objects instead of a flat supported list:

- `preview`
- `create_definition`
- `update_definition`
- `pause`
- `resume`
- `archive`
- `duplicate`
- `execute`

Each action reports `status`, `reason?`, and `required_permissions?`.

Action status values:

- `available`
- `unavailable`
- `planned`
- `disabled`

In 4B, `preview`, definition management, and lifecycle actions can be available. `execute` must be returned as unavailable for both families with a disabled reason such as `execution_not_implemented`.

Related capabilities can report RAG or ACP readiness, but related readiness must never make Scheduled Tasks execution look available by itself.

## Core Data Model

4B separates definition lifecycle, execution readiness, and future run state.

| Model | Key fields |
| --- | --- |
| Capability | `family`, `family_availability`, `actions`, `missing_dependencies`, `related_capabilities`, `reason`, `schema_version` |
| Preview | `id`, `owner_id`, `mode`, `family`, `definition_id?`, `definition_version?`, `status`, `payload_hash`, `normalized_config`, `validation_errors`, `warnings`, `risk_class`, `visibility_policy`, `schedule_preview`, `redaction_policy`, `expires_at`, `created_by`, `created_at`, `consumed_at?`, `created_definition_id?` |
| Definition | `id`, `owner_id`, `version`, `family`, `name`, `description`, `lifecycle`, `health`, `schedule`, `input`, `visibility_policy`, `notification_policy`, `approval_policy`, `preview_id`, `created_by`, `updated_by`, timestamps |
| Audit event | `id`, `definition_id`, `event_type`, `actor`, `summary`, `before`, `after`, `created_at`, `request_id`, `idempotency_key` |
| Idempotency record | `owner_id`, `key`, `route`, `payload_hash`, `response_ref`, `created_at`, `expires_at` |

`owner_id` must be part of the logical repository contract even when the first implementation stores data in per-user SQLite files. Per-user files can make owner scoping implicit physically, but API behavior, tests, and any future shared tenant database must treat previews, definitions, audit reads, and idempotency records as owner-scoped resources.

Families:

- `recurring_question`
- `agent_task`

Preview modes:

- `create`
- `update`

Lifecycle values:

- `configured`
- `paused`
- `archived`
- `disabled`

Health values:

- `ready`
- `execution_unavailable`
- `capability_unavailable`
- `needs_attention`
- `permission_required`

In 4B, definitions can be created as `configured` or `paused`, but their health should normally be `execution_unavailable`. That means the definition is valid and manageable, but no run engine will execute it until 4C or 4D.

When several health conditions apply, use a stable precedence so clients do not guess:

1. `permission_required`
2. `capability_unavailable`
3. `needs_attention`
4. `execution_unavailable`
5. `ready`

In 4B, `execution_unavailable` is expected even when related RAG or ACP readiness is healthy. If related readiness is missing and would block future execution, return `capability_unavailable` with explicit related capability details, not a fake execution status.

Deletion is intentionally absent. `archive` is the user-facing removal workflow. Hard purge is a future admin or privacy retention feature.

### Lifecycle Transitions

Lifecycle transitions should be explicit and testable.

| Mutation | Allowed from | Result | Notes |
| --- | --- | --- | --- |
| create | valid create preview | `configured` or `paused` | Default is `configured` unless request asks for `paused`. Health is normally `execution_unavailable`. |
| update | `configured`, `paused`, `disabled` | same lifecycle | Requires valid update preview for the current definition version. |
| pause | `configured` | `paused` | Records audit event. |
| pause | `paused` | `paused` | Idempotent no-op; return current definition without a new audit event. |
| resume | `paused` | `configured` | Records audit event. |
| resume | `configured` | `configured` | Idempotent no-op; return current definition without a new audit event. |
| resume | `disabled` | error | `disabled` is system/admin controlled and not user-resumable. |
| archive | `configured`, `paused`, `disabled` | `archived` | Records audit event. |
| archive | `archived` | `archived` | Idempotent no-op; return current definition. |
| duplicate | `configured`, `paused`, `disabled` | new `paused` definition | Copies config only; no run/result/history copy; records deterministic audit on source and copy. Disabled sources may be duplicated only when the disabled reason is not an admin/security lock. |
| duplicate | `archived` | error | Archived definitions are not duplicated in 4B. |
| update/pause/resume archived | `archived` | error | Return `scheduled_task_definition_archived`. |

Every update preview must capture the base `definition_version`. Update mutations reject stale previews with `scheduled_task_definition_version_conflict` when the current definition version no longer matches the preview.

Lifecycle response rules:

- successful create returns `201`;
- successful update/lifecycle/duplicate returns `200`;
- idempotent no-op lifecycle mutations return `200` with the current definition;
- invalid lifecycle transitions return `409` with a typed error;
- archived-definition mutations return `409` with `scheduled_task_definition_archived`, except archive of an already archived definition, which returns `200`.

## Durable Preview Contract

Preview is required for create and update. Preview is a durable server-side validation artifact, not a transient client hint.

Preview records should include:

- normalized config;
- validation errors;
- warnings;
- schedule interpretation;
- future result routing explanation;
- missing dependencies;
- related capability status;
- expiry;
- redaction policy;
- risk class for Agent Task when possible.

Create and update must consume a valid, unexpired preview record. Expired, mismatched, or cross-user previews fail with typed errors.

Agent Task previews must not become accidental secret storage. They should store redacted message previews by default. Raw sensitive payload retention should be avoided or explicitly controlled by configuration.

Preview statuses:

- `valid`: semantic validation passed and the preview can be used for create/update until expiry or consumption.
- `invalid`: request shape was syntactically accepted, but semantic validation failed; the record is inspectable but cannot be used for create/update.
- `expired`: preview is past `expires_at`; expiry can be materialized lazily during read or mutation.
- `consumed`: preview was used by a successful create/update and cannot be reused.

Malformed requests still use normal request validation errors and do not need a preview record. Semantic validation failures should persist an `invalid` preview record and return a successful preview response with validation details so clients can inspect history.

Preview consumption checklist:

- owner matches the current user;
- status is `valid`;
- preview is not expired;
- preview is not consumed;
- preview family is supported by the target mutation;
- preview mode matches the mutation, `create` for create and `update` for update;
- create previews have no target `definition_id`;
- update previews have `definition_id` equal to the route `{definition_id}`;
- update preview `definition_version` equals the current stored definition version;
- preview payload hash is the hash of the preview artifact's canonical config, not the idempotency request hash.

Only previews that pass the full checklist can be consumed by create/update.

Preview response rules:

- syntactically valid preview requests return `201` with a persisted preview record, even when `status` is `invalid`;
- malformed preview requests return normal request validation errors and do not persist a preview;
- create/update with a missing preview returns `400` with `scheduled_task_preview_required`;
- create/update with expired, consumed, mismatched, wrong-user, or wrong-version preview returns `409` with the matching typed error.

Cross-user preview reads and mutations must not reveal preview existence. A preview owned by another user should behave like a missing or mismatched preview: detail reads return 404, and create/update use returns `scheduled_task_preview_mismatch` without ownership detail.

## Request Shapes And Preview Consumption

Preview requests carry the full proposed config. Create/update mutations consume the preview record rather than accepting a second copy of the full config.

Minimal preview request:

```json
{
  "mode": "create",
  "family": "recurring_question",
  "definition_id": null,
  "definition_version": null,
  "config": {
    "name": "Track unanswered licensing question",
    "description": null,
    "input": {},
    "schedule": {},
    "visibility_policy": "findings_only",
    "notification_policy": {}
  }
}
```

Minimal create request:

```json
{
  "preview_id": "preview_123",
  "initial_lifecycle": "configured"
}
```

Minimal update preview request:

```json
{
  "mode": "update",
  "family": "agent_task",
  "definition_id": "definition_123",
  "definition_version": 4,
  "config": {
    "name": "Weekly agent check",
    "description": null,
    "input": {},
    "schedule": {},
    "visibility_policy": "failures_and_approvals",
    "notification_policy": {}
  }
}
```

Minimal update request:

```json
{
  "preview_id": "preview_456"
}
```

Minimal duplicate request:

```json
{
  "name": "Copy of Weekly agent check"
}
```

Duplicate does not accept config overrides in 4B. It copies the source definition's normalized config, applies a new name when provided, and creates the copy in `paused` lifecycle. Any further config edits require an update preview.

Duplicate should re-evaluate capability and permission health for the copied definition at creation time. It must not let a user bypass an admin/security `disabled` state by copying the definition into a runnable future state. In 4B the copy is always `paused`; in later execution slices, resuming the copy still requires normal health and permission checks.

Canonical payload hash:

- For previews, hash the canonical JSON form of `mode`, `family`, `definition_id`, `definition_version`, and `config`.
- For create, the payload hash used for idempotency covers `preview_id` and `initial_lifecycle`.
- For update, the payload hash used for idempotency covers `preview_id`.
- For duplicate, the payload hash used for idempotency covers the optional new name and source definition ID.
- Canonicalization should sort object keys and exclude transport-only metadata such as request IDs.

## Recurring Question Contract

Recurring Question stores a user's recurring research intent and searchable scope, but it does not run in 4B.

Input fields:

- `question`
- `success_criteria?`
- `freshness_expectation?`
- `answer_policy`
- `rag_profile_ref?`
- `scope`

Scope should support:

- all searchable library;
- collection IDs;
- tags;
- saved search IDs;
- source types;
- date windows;
- future custom filters as a versioned scope object.

Preview validates:

- question is present and safe to store;
- scope object is syntactically valid;
- RAG capability and readiness are reported separately from Scheduled Tasks execution support;
- schedule is valid;
- visibility policy is valid;
- no execution will occur in 4B.

Default visibility:

- `findings_only`

## Agent Task Contract

Agent Task stores a scheduled message and selected agent contract, but it does not dispatch in 4B.

Input fields:

- `agent_ref`
- `message_payload`
- `context_ref?`
- `allowed_tool_classes`
- `denied_tool_classes`
- `approval_policy`
- `redaction_policy`

Preview validates:

- agent reference shape;
- message is present;
- schedule is valid;
- allowed and denied tool classes are syntactically valid;
- risk class is estimated where possible;
- preview output is redacted by default;
- no execution will occur in 4B.

Default visibility:

- `failures_and_approvals`

Risk classes:

- `low`
- `medium`
- `high`
- `unknown`

Risk classes are API terms in 4B. Enforcement and approval queues wait for 4D.

### Agent Task Sensitive Payload Policy

Agent Task definitions must not store raw prompt/message text inline in `Definition.input`.

Use a `message_payload` object:

- `redacted_preview`: safe preview text returned by list/detail/preview/audit responses.
- `storage_mode`: `redacted_only` or `encrypted_payload_ref`.
- `payload_ref?`: opaque reference to encrypted payload storage when configured.
- `raw_retention_allowed`: boolean explaining whether the server accepted raw retention.

Default 4B behavior is `redacted_only`. If secure encrypted payload storage is not configured, raw message retention is not allowed. A future 4D execution implementation may require `encrypted_payload_ref` or a fresh update preview before an Agent Task can execute.

List, detail, preview, and audit responses must never return raw Agent Task message text by default. Tests must prove that previews, definitions, and audit events do not leak raw prompts.

## Schedule Policy

4B should define safe schedule policy values because later execution will inherit these definitions.

Supported schedule kinds:

- `one_time`
- `interval`
- `daily`
- `weekly`
- `cron`

All schedules include:

- `timezone`
- `start_at?`
- `end_at?`
- `dst_policy`
- `missed_run_policy`
- `overlap_policy`
- `retry_policy`

`next_run_preview` is response output only. It is not persisted as the canonical schedule source.

Supported `dst_policy` values:

- `preserve_wall_time`
- `preserve_elapsed_interval`

Supported `missed_run_policy` values in 4B:

- `skip`
- `run_once`

`run_all` is deferred as a future/admin-only option.

Supported `overlap_policy` values in 4B:

- `skip_new`
- `cancel_existing`

`allow_parallel` is deferred unless a future capability explicitly enables it.

Retry policy fields:

- `strategy`: `none`, `fixed`, `exponential`
- `max_attempts`
- `initial_delay_seconds`
- `max_delay_seconds`
- `backoff_multiplier?`

Cron is allowed as an advanced schedule kind, but it requires timezone, DST policy, and clear validation errors.

## Visibility And Notification Policies

Visibility policy is authoritative. It determines which future run outputs may appear outside task history.

Defaults:

- Recurring Question: `findings_only`
- Agent Task: `failures_and_approvals`

Supported policy values:

- `every_run`
- `findings_only`
- `failures_only`
- `failures_and_approvals`
- `task_history_only`

4B stores the policy and uses it in preview copy. It does not create Home/results items from these definitions.

Notification policy is subordinate to visibility policy. It can further restrict delivery, but it cannot cause Home, Results, or notifications to receive content that visibility policy excludes.

Fields:

- `home_enabled`
- `results_enabled`
- `notifications_enabled`
- `dedupe_key_strategy`
- `failure_severity_threshold?`
- `finding_confidence_threshold?`

Preview should explain:

- run history is complete once execution exists;
- Home/results receive only what visibility policy allows;
- notification settings can narrow that surfacing;
- no run, result, or notification will be created in 4B.

## Audit Contract

4B should persist audit events for real definition lifecycle activity. Preview history is its own durable record and is not definition audit unless the preview is consumed into a definition.

Audit event types:

- `definition_created`
- `definition_updated`
- `definition_paused`
- `definition_resumed`
- `definition_archived`
- `definition_duplicated`
- `definition_duplicate_created`

Create previews, invalid previews, and abandoned previews may not have a `definition_id`; they are inspectable through preview history, not through `definitions/{definition_id}/audit`.

When a create preview is consumed successfully, the preview record should set `created_definition_id`, and the new definition audit should include `definition_created` with `preview_id`.

When an update preview is consumed successfully, the target definition audit should include `definition_updated` with `preview_id` and the consumed preview's base version.

When a duplicate succeeds, audit should be deterministic:

- source definition receives `definition_duplicated` with `new_definition_id`;
- copied definition receives `definition_duplicate_created` with `source_definition_id`;
- both events use redacted metadata and the same `request_id`/`idempotency_key` when present.

Audit `before` and `after` metadata must be concise and redacted. For Agent Task, message/input fields must be redacted or summarized. Audit must not copy raw prompts into metadata by default.

4B exposes audit through a nested per-definition endpoint. A cross-definition audit endpoint is useful later, but out of scope for 4B.

`scheduled_task_idempotency_conflict` is not a per-definition audit event in 4B because conflicts can occur before any definition exists. It should be returned as a typed error and may be recorded in service logs. A future global/security audit surface can model cross-resource idempotency conflicts.

## Idempotency

4B should support optional `Idempotency-Key` for preview, create, update, duplicate, and lifecycle mutations.

Behavior:

- same route + same key + same payload hash returns the original response;
- same route + same key + different payload hash returns `scheduled_task_idempotency_conflict`;
- records expire on a documented server policy;
- records should store payload hashes and response references, not raw request bodies.

For mutating requests, idempotency lookup must happen before preview validation or preview consumption. This lets a retried successful create/update return the original response even though the preview was already consumed by the first request. Without an idempotency key, reusing a consumed preview fails.

This gives enterprise clients a safe path without making idempotency mandatory for every early client.

## Error Handling

Errors should use a consistent envelope:

```json
{
  "code": "scheduled_task_preview_expired",
  "message": "Preview expired. Run preview again before saving.",
  "details": {},
  "field_errors": [],
  "retryable": true,
  "correlation_id": "..."
}
```

Core 4B error codes:

- `scheduled_task_family_unavailable`
- `scheduled_task_preview_required`
- `scheduled_task_preview_expired`
- `scheduled_task_preview_mismatch`
- `scheduled_task_definition_not_found`
- `scheduled_task_definition_archived`
- `scheduled_task_schedule_invalid`
- `scheduled_task_scope_invalid`
- `scheduled_task_agent_ref_invalid`
- `scheduled_task_permission_policy_invalid`
- `scheduled_task_idempotency_conflict`
- `scheduled_task_execution_unavailable`
- `scheduled_task_definition_version_conflict`
- `scheduled_task_lifecycle_transition_invalid`

Later execution-only error code:

- `scheduled_task_agent_unavailable`

The 4B Agent Task API should use `scheduled_task_agent_ref_invalid` when only reference shape can be validated. It should not imply live agent health checks unless the implementation actually performs them.

## Permissions

Use current permissions for 4B implementation:

| Permission | Endpoints |
| --- | --- |
| `TASKS_READ` | capabilities, definition list/detail, preview list/detail, audit read |
| `TASKS_CONTROL` | preview create, definition create/update, pause, resume, archive, duplicate |

Future granular permissions should be defined as terms but not required for 4B:

- definition read;
- definition write;
- preview create/read;
- lifecycle control;
- audit read;
- approval read/write;
- export;
- admin purge.

Unauthenticated capability discovery is out of scope.

## WebUI Reference Client Behavior

The WebUI should prove the API contract works without pretending execution is available.

Create tab:

- Fetch `GET /scheduled-tasks/capabilities`.
- Replace hardcoded planned state for Recurring Question and Agent Task when API capability data exists.
- Show preview/create when capability says definition management is supported.
- Keep "not executable yet" messaging visible after create.

Create/edit forms:

- Recurring Question form: question, success criteria, RAG scope, schedule, visibility.
- Agent Task form: agent ref, message, allowed/denied tool classes, approval policy, schedule, visibility.
- Both forms use preview before save.
- Create/update consumes an unexpired preview record.

Task list:

- Include 4B definitions as normalized scheduled task rows.
- Show family, lifecycle, health, schedule summary, last run as `Not run yet`, next run as `Execution unavailable` or equivalent.
- Do not show fake run or result counts.

Detail drawer:

- Show definition summary, lifecycle controls, health explanation, schedule, visibility/notification policy.
- Show preview history and definition audit events.
- Show "Execution is not available yet" in the run/result area.
- Show no fake run rows.

Results tab and Home:

- Continue showing existing projected/real items only.
- Do not create placeholder result cards for 4B definitions.
- Empty/copy states can mention future outputs route by visibility policy.

Power-user actions:

- pause/resume/archive/duplicate from table or detail;
- filters for family, lifecycle, health, visibility, text search;
- audit/preview history reachable from detail.

## Control Plane Projection

The unified `/api/v1/scheduled-tasks` list should remain backward compatible.

Add a new primitive for persisted 4B definitions without redefining existing primitives:

- `automation_definition`

This is an explicit 4B compatibility decision. `ScheduledTaskPrimitive` should be extended to include `automation_definition`, and WebUI/service clients must be updated to handle it. Existing primitive values retain their meaning.

Projected rows should include:

- stable task ID prefix, such as `automation_definition:{definition_id}`;
- `primitive: "automation_definition"`;
- `enabled`: true only for lifecycle `configured`; false for `paused`, `archived`, and `disabled`;
- `status`: a stable product-status token derived from lifecycle and health, not raw scheduler state;
- `source_ref.family` with `recurring_question` or `agent_task`;
- `source_ref.definition_id`;
- `source_ref.lifecycle`;
- `source_ref.health`;
- `source_ref.execution_available: false` in 4B;
- schedule summary;
- timezone;
- edit mode `native`;
- no last run;
- no real next run when health is `execution_unavailable`;
- source references for definition ID and family.

Reminder and Watchlists behavior must remain unchanged.

Projection status mapping should avoid both fake scheduler language and current WebUI fallback traps:

| Definition state | Projected `enabled` | Projected `status` | Product meaning |
| --- | --- | --- | --- |
| `configured` + `execution_unavailable` | true | `configured_execution_unavailable` | Valid definition, execution not implemented. |
| `configured` + `capability_unavailable` | true | `blocked_capability_unavailable` | Definition is valid, but a related capability blocks future execution. |
| `configured` + `permission_required` | true | `blocked_permission_required` | Definition needs permission/config recovery. |
| `paused` | false | `paused` | User-paused definition. |
| `disabled` | false | `disabled` | System/admin-disabled definition. |
| `archived` | false | `archived` | Removed from normal active management views unless filters include archived. |

The WebUI should add `automation_definition`-aware status and type handling instead of relying on generic substring matching. For automation definitions, lifecycle/health must be interpreted before the legacy `enabled === false` fallback so `paused` renders as "Paused" and `archived` renders as "Archived" instead of both collapsing into "Disabled." In particular, `configured_execution_unavailable` should render as "Configured, execution unavailable" or equivalent, not "Waiting for next run" and not a generic dependency failure.

## Repository And Storage Requirements

The repository interface should support:

- create/list/get/update/archive definitions;
- pause/resume definitions;
- duplicate definitions;
- create/list/get previews;
- create/list audit events;
- create/read idempotency records;
- pagination and filtering;
- per-user isolation.

First implementation:

- per-user SQLite under `Databases/user_databases/<user_id>/ScheduledTasks.db`;
- schema initialization through the store/service layer;
- no raw SQL outside DB management abstractions;
- repository tests that can later be reused against a shared tenant DB implementation.

## Pagination And Filtering

4B list endpoints should not be unbounded.

Definitions should filter by:

- family;
- lifecycle;
- health;
- visibility policy;
- text query;
- created/updated date window.

Previews should filter by:

- family;
- mode;
- status;
- definition ID;
- created date window;
- expired/not expired.

Audit should filter by:

- event type;
- actor;
- created date window;
- idempotency key;
- request/correlation ID.

Use the repository's existing pagination conventions where practical. If multiple pagination styles exist, the implementation plan should choose one and apply it consistently to new 4B endpoints.

## Trust Copy

Recommended copy for API clients and WebUI:

- `Definition saved. Execution is not available on this server yet.`
- `Preview expired. Run preview again before saving.`
- `This Agent Task preview is redacted. Raw prompt retention is disabled.`
- `RAG is available, but scheduled Recurring Question execution is not enabled.`
- `This task is configured but will not run until scheduled execution is enabled.`

Avoid:

- `Scheduled` for definitions that cannot execute.
- `Agent unavailable` when 4B only validated agent reference shape.
- `No result` without explaining execution has not run.

## Testing Requirements

Backend tests:

- capability discovery returns Recurring Question and Agent Task family states separately from RAG/ACP related readiness;
- static child routes such as `/capabilities`, `/previews`, and `/definitions` are not shadowed by `/{task_id}`;
- preview creation persists durable preview records with expiry and redacted Agent Task message preview;
- create/update require valid preview records;
- invalid previews are persisted as inspectable records but cannot be consumed;
- create previews without definitions remain visible in preview history but do not appear in definition audit;
- consumed create previews link to the created definition;
- expired and mismatched previews fail with typed errors;
- cross-user preview reads and create/update attempts do not reveal preview existence;
- stale update previews fail when the definition version has changed;
- definitions can be listed, filtered, read, updated, paused, resumed, archived, and duplicated;
- duplicate creates a paused copy without run/result history;
- duplicate does not accept inline config overrides in 4B;
- audit events are recorded for create, update, pause, resume, archive, and duplicate;
- optional idempotency replays same-payload responses and rejects same-key/different-payload conflicts;
- idempotency keys are scoped by owner and route, so one user's key cannot replay or conflict with another user's mutation;
- idempotent create/update replay succeeds after the original request consumed the preview;
- no-key reuse of a consumed preview fails;
- duplicate records deterministic audit events on both source and copied definitions;
- duplicate cannot bypass an admin/security disabled source into a future runnable state;
- automation definitions project into `/scheduled-tasks` with explicit `enabled`, `status`, `source_ref.lifecycle`, `source_ref.health`, and `source_ref.execution_available`;
- existing reminder and Watchlists list/detail behavior remains compatible.

Storage tests:

- repository interface behavior;
- per-user SQLite isolation;
- pagination/filtering for definitions, previews, and audit;
- no raw Agent Task prompt leakage in preview, definition list/detail, or audit records by default.

Frontend tests:

- capabilities enable 4B create/manage affordances only when advertised;
- preview-before-save flow for both families;
- definition detail shows preview history, audit events, and execution-unavailable state;
- `automation_definition` rows render family, lifecycle, health, and execution-unavailable copy without falling through to generic reminder/watchlist labels;
- configured non-executable definitions do not render as "Waiting for next run" and do not create fake result buttons;
- paused and archived automation definitions do not collapse into the generic "Disabled" state when `enabled` is false;
- lifecycle actions update UI without fake run/result rows;
- existing Reminder, Watch/Ingest, Results, Home, and Watchlists behavior remains unchanged.

Security tests:

- Agent Task preview and audit redaction;
- permission mapping for `TASKS_READ` and `TASKS_CONTROL`;
- idempotency conflict behavior;
- owner-scoped idempotency replay behavior;
- cross-user preview/definition access denial.

## Acceptance Criteria

4B is complete when:

- API clients can discover, preview, create, inspect, update, pause, resume, archive, and duplicate Recurring Question and Agent Task definitions.
- Persisted definitions are Scheduled Tasks-owned.
- Durable preview records are required for create/update and are redacted for Agent Task by default.
- Agent Task definitions do not expose raw prompt/message text in list/detail/preview/audit responses by default.
- Definition audit events are persisted and visible per definition.
- Definitions are projected into the unified scheduled tasks list without breaking existing reminder or Watchlists behavior.
- No execution, manual run, approval queue, run result, notification delivery, RAG dispatch, or agent dispatch is implemented.
- WebUI demonstrates the 4B API without pretending execution exists.
- Watchlists remains independent and unchanged.
- OpenAPI docs expose the new schemas and endpoint contracts.
- Bandit and focused backend/frontend tests pass for touched scope.

## Risks And Mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Persisted definitions look executable | Users and API clients may assume work will run. | Use lifecycle `configured` and health `execution_unavailable`; avoid `scheduled` until execution exists. |
| Durable previews leak secrets | Agent prompts may contain sensitive data. | Store redacted message previews by default; audit before/after is redacted/summarized. |
| Durable definitions leak secrets | Raw Agent Task messages could persist without hard purge in 4B. | Do not store raw message inline; use redacted-only default or encrypted payload references when configured. |
| Capability discovery conflates related readiness with execution support | RAG/ACP being available could incorrectly enable create/run. | Separate family availability from related capabilities; `execute` remains unavailable in 4B. |
| Normalized rows are mislabeled by legacy WebUI status mapping | `execution_unavailable` can look like a generic dependency failure or "waiting for next run." | Project explicit lifecycle/health fields and add `automation_definition`-aware WebUI status handling. |
| Idempotency keys leak across users in a future shared store | A retry key could expose or conflict with another user's mutation. | Include `owner_id` in the logical idempotency key scope and test cross-user isolation. |
| Per-user SQLite limits future enterprise reporting | Admin audit may need tenant-wide views. | Define repository interface; do not expose storage topology in API. |
| Optional idempotency behaves inconsistently | Clients cannot safely retry. | Define same-key replay and conflict semantics. |
| Archive without delete frustrates privacy needs | Users may expect removal of sensitive prompts. | Minimize/redact stored sensitive data now; define hard purge as future admin/privacy feature. |
| WebUI overpromises by showing Results/Home items | Trust loss and fake data. | Do not create fake runs/results; show execution-unavailable states. |
| Existing `/acp/schedules` mistaken for Agent Task support | Unsafe unattended work could be implied. | Leave `/acp/schedules` separate and link only as related context. |

## Future Slices

4C Recurring Question execution:

- Jobs-backed or explicitly approved Scheduler-backed execution;
- schedule activation;
- RAG query execution;
- every-run summaries;
- no-useful-match summaries;
- result routing by visibility policy.

4D Agent Task execution:

- agent availability checks;
- dispatch;
- risk enforcement;
- approval queue;
- run transcript/output references;
- audit events for tool/permission boundaries.

Future cross-cutting work:

- approval APIs;
- run/result APIs;
- global audit endpoint;
- export APIs;
- admin purge and retention controls;
- migration or normalization strategy for `/acp/schedules`.

## Open Questions For Implementation Planning

- Which pagination convention should be used for new endpoints if existing modules differ?
- What expiry window should durable preview records use?
- What retention period should preview and audit records use by default?
- How much live RAG/ACP readiness should capability discovery check in 4B versus returning static related capability status?
- Should secure encrypted payload storage be required before Agent Task definitions can move from 4B management to 4D execution?

## Spec Review

Reviewed with `spec-document-reviewer`-style subagents on 2026-06-09.

Review pass 1 found blocking issues in control-plane primitive compatibility, static route shadowing, Agent Task raw prompt retention, preview status semantics, duplicate preview behavior, update-version checks, capability action modeling, and lifecycle transitions. The spec was revised to:

- explicitly add `automation_definition` as the new primitive;
- require static routes before `/{task_id}` plus route regression tests;
- define Agent Task redacted/default payload storage;
- enumerate preview statuses and lifecycle transitions;
- remove duplicate preview mode;
- add definition version checks;
- model capability actions with per-action statuses.

Review pass 2 found issues in preview audit ownership, idempotency replay ordering, request body shape, and cross-user preview enumeration. The spec was revised to:

- separate preview history from per-definition audit;
- define idempotency lookup before preview validation/consumption for replay;
- add minimal preview/create/update/duplicate request shapes and canonical hash rules;
- require non-enumerating cross-user preview behavior.

Review pass 3 found two remaining issues: preview consumption still used generic "matches mutation payload" wording, and `idempotency_conflict` still appeared as a per-definition audit event. The spec was revised locally to:

- replace generic preview matching with an explicit preview consumption checklist;
- remove `idempotency_conflict` from per-definition audit and scope it to typed errors/service logs until a future global audit surface exists.

No fourth subagent review was dispatched because the brainstorming workflow caps review iterations at three. These final review findings are addressed in this draft and should be checked during user review before implementation planning.

Additional human-requested self-review before implementation planning found implementation-risk gaps in owner scoping, health precedence, duplicate audit determinism, and normalized WebUI projection. The spec was revised to:

- make `owner_id` part of Preview, Definition, and Idempotency record contracts, with explicit cross-user isolation expectations;
- define health precedence so related capability readiness does not blur into execution support;
- make duplicate audit deterministic on both source and copied definitions;
- require duplicate to re-evaluate health and avoid bypassing admin/security disabled states;
- define `automation_definition` projection fields and status tokens for the existing `/scheduled-tasks` list model;
- require WebUI status handling to interpret automation definition lifecycle/health before legacy `enabled` fallbacks.
