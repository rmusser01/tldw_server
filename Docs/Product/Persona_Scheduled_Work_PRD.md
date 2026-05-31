# Persona Scheduled Work PRD

Status: Draft

Owner: Persona module / background jobs integration

Tracking: #1913, split from #1902

Backlog: TASK-469

## Summary

Define Persona scheduled work as user-owned recurring assistant activity that produces reviewable outputs before delivery or mutation. This PRD covers daily briefs, scheduled research summaries, recurring follow-ups, and action suggestions that run outside an active live Persona session.

V1 should reuse the existing recurring-work architecture: APScheduler holds wall-clock triggers, then enqueues durable user-visible work into Jobs. Persona scheduled work should not introduce a parallel scheduler, and it should not execute privileged Persona tools directly from an in-memory timer.

## Problem

The completed Persona module intentionally focuses on Persona Garden and live-session foundations. Scheduled autonomous work was moved out because it needs a different product contract: a user is not actively chatting when the Persona acts, so schedules, policy, memory access, generated outputs, review gates, retries, cancellation, and audit trails need first-class semantics.

Without this PRD, future implementation risks conflating three systems:

- Live Persona sessions, where the user is present and can approve actions in context.
- Ordinary Persona-backed chat startup, where the user starts a normal chat with a Persona.
- Background Persona work, where a schedule creates a draft output or proposal for later review.

## Goals

- Let a user create recurring Persona work for a specific Persona profile.
- Support daily briefs and scheduled research/follow-up drafts as the first product shape.
- Require a human review step before external delivery or durable mutations in V1.
- Reuse Jobs for durable execution, retry, status, quota, owner, and admin visibility.
- Use APScheduler only for recurring triggers that enqueue Jobs.
- Apply Persona policy and scope evaluation before any tool-backed action.
- Persist schedule, run, output, review, approval, and delivery/audit metadata.
- Keep the design backend-first and compatible with later WebUI surfaces.

## Non-goals

- No Buddy animation or Buddy runtime work.
- No design-system backlog work.
- No implementation in this PRD slice.
- No multi-agent collaboration.
- No broad global personalization memory layer.
- No marketplace-style tool installation or administration.
- No external delivery without an explicit review/approval gate in V1.
- No replacement for live Persona sessions or ordinary Persona-backed chat startup.

## Current Contract Evidence

- `Docs/Product/Persona_Agent_Design.md` moves scheduled Persona jobs and daily briefs out of the current Persona module completion scope.
- `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md` assigns scheduled jobs, daily briefs, review/approval gates, schedule ownership, and cancellation to a future PRD.
- `tldw_Server_API/app/core/Jobs/README.md` describes Jobs as durable background work with leasing, retries, queue controls, owner quotas, metrics, admin controls, and audit hooks.
- `tldw_Server_API/app/core/Scheduler/README.md` describes the core Scheduler as internal task orchestration with idempotency, dependencies, leases, and handler registration.
- `tldw_Server_API/app/services/reading_digest_scheduler.py` shows the preferred recurring user-visible pattern: APScheduler cron trigger, schedule claim, optional online gating, then `JobManager.create_job(...)` with an idempotency key.
- `tldw_Server_API/app/services/reminders_scheduler.py` follows the same APScheduler-to-Jobs pattern for date and cron reminders.
- `tldw_Server_API/app/core/Persona/policy_evaluator.py` centralizes Persona policy semantics, scope checks, explicit deny, and confirmation requirements for Persona actions.

## Product Shape

Persona Scheduled Work V1 should model scheduled work as four related records:

1. Schedule: user-owned configuration for when and why work should run.
2. Run: one claimed execution slot for a schedule.
3. Draft artifact: generated brief, summary, recommendation, or action proposal.
4. Review decision: user approval, rejection, edit, delivery, or cancellation.

Example V1 schedule contract:

```json
{
  "persona_id": "persona-id",
  "name": "Morning research brief",
  "kind": "daily_brief",
  "cron": "0 8 * * *",
  "timezone": "America/Los_Angeles",
  "enabled": true,
  "require_online": false,
  "memory_mode": "read_only",
  "input_scope": {
    "workspace_id": "workspace-id",
    "collections": ["saved_articles"],
    "query": "AI policy updates from the last day"
  },
  "review_policy": {
    "review_required": true,
    "allow_auto_delivery": false
  },
  "delivery": {
    "channels": ["in_app"],
    "external_targets": []
  }
}
```

V1 output should be a draft, not an automatic action:

```json
{
  "run_id": "run-id",
  "schedule_id": "schedule-id",
  "persona_id": "persona-id",
  "status": "awaiting_review",
  "artifact_type": "brief",
  "artifact_id": "output-artifact-id",
  "proposed_actions": [],
  "policy_decisions": [],
  "requires_review": true
}
```

## Execution Architecture

V1 should use Jobs for execution because scheduled Persona work is user-facing and needs durable status, retries, owner scoping, quotas, queue controls, and admin visibility. The core Scheduler remains available for later internal orchestration if a scheduled Persona run needs a multi-step dependency graph, but it should not be the V1 public execution backend.

Recommended flow:

1. Persona schedule is stored in a per-user Persona or ChaCha-backed schedule table.
2. A Persona scheduled-work service loads enabled schedules into APScheduler.
3. On fire, the service claims the expected run slot with compare-and-set semantics.
4. The service enqueues a Jobs row with domain `persona`, a dedicated queue such as `persona-scheduled-work`, job type by schedule kind, owner user id, and an idempotency key based on schedule id plus normalized fire time.
5. A Jobs worker resolves the Persona profile, verifies ownership, evaluates current policy/scopes, builds read-only context, and generates a draft artifact.
6. The worker marks the run `awaiting_review` or `blocked` with structured reasons.
7. A review endpoint records approve/reject/edit/deliver decisions.

The idempotency key should use a stable slot, for example:

```text
persona_scheduled_work:{schedule_id}:{fire_time_utc_iso}
```

## Review And Approval

V1 must default to review-required behavior:

- Generated briefs are saved as in-app drafts.
- Proposed external messages, exports, write actions, deletes, and tool-backed mutations are proposals only.
- Approval must record the reviewer, timestamp, approved artifact version, delivery channel, and policy state used at approval time.
- If the Persona policy says an action requires confirmation, scheduled work cannot bypass that requirement.
- If the Persona policy changes between generation and approval, the approval path must re-check policy before delivery or mutation.

Rejected or expired drafts should remain auditable for a retention window but should not block the next scheduled run unless the schedule explicitly requires resolving prior drafts first.

## Data Model Direction

Preferred V1 tables or records:

- `persona_schedules`
  - `id`, `owner_user_id`, `persona_id`, `name`, `kind`, `cron`, `timezone`, `enabled`, `require_online`, `memory_mode`, `input_scope_json`, `review_policy_json`, `delivery_json`, `created_at`, `updated_at`
- `persona_schedule_runs`
  - `id`, `schedule_id`, `owner_user_id`, `fire_time`, `job_id`, `status`, `artifact_id`, `error_code`, `error_message`, `policy_decisions_json`, `created_at`, `updated_at`
- `persona_schedule_reviews`
  - `id`, `run_id`, `reviewer_user_id`, `decision`, `artifact_version`, `delivery_channels_json`, `policy_snapshot_json`, `created_at`

Reference-backed fields should remain references where possible:

- `persona_id` references the source Persona profile.
- Workspace, collection, template, and output artifact values should be ids, not copied content snapshots.
- Any audit snapshot should be limited to the policy decision needed to explain a past approval or block, not copied Persona profile content.

## Policy And Safety

Scheduled Persona work runs without the user currently in the loop, so V1 should be stricter than live sessions:

- Default memory mode is `read_only`.
- `read_write` memory mode is allowed only when explicitly configured and visible in schedule settings.
- Destructive actions are never auto-executed in V1.
- Export/delivery actions require review and policy re-check.
- Tool access uses existing Persona policy rules and session-like scopes; scheduled runs should not grant wildcard write scopes implicitly.
- Missing Persona, deleted Workspace, invalid collection, or unavailable tool references block the run with recoverable status rather than silently falling back.

## UX Requirements

This PRD is backend-first, but future UI should support:

- Create, pause/resume, delete, and run-now controls for schedules.
- Clear labels for inherited Persona identity, memory mode, schedule timezone, next run, last run, and review status.
- A review inbox for generated drafts and proposed actions.
- Visible blocked states for missing Persona references, invalid scopes, policy denials, or failed generation.
- Audit history for schedule edits, run attempts, approvals, delivery, and cancellations.

## Staged Delivery

### Stage 1: Contract And Storage Design

Goal: define the backend contract without running jobs.

Deliverables:

- Schedule, run, and review schemas.
- DB migration design for user-owned Persona schedules.
- API contract for create/list/get/update/delete/run-now.
- Validation rules for Persona ownership, cron/timezone, memory mode, and input scope.

### Stage 2: APScheduler To Jobs Enqueue

Goal: enqueue durable Jobs from enabled schedules.

Deliverables:

- Persona scheduled-work service modeled after reading digest and reminders schedulers.
- Run-slot claim and idempotency key handling.
- Jobs domain, queue, job type, payload, owner, and status mapping.
- Tests for duplicate fires, disabled schedules, invalid cron, and run-now behavior.

### Stage 3: Draft Generation Worker

Goal: turn a claimed run into a reviewable draft artifact.

Deliverables:

- Jobs worker for daily brief V1.
- Persona/profile resolution and policy preflight.
- Read-only context builder for Workspace/collection/query inputs.
- Draft artifact creation with structured run status and failure reasons.

### Stage 4: Review And Delivery Gate

Goal: make generated work actionable only after review.

Deliverables:

- Review decision API for approve/reject/edit/expire.
- Policy re-check before delivery or mutation.
- In-app delivery as the first channel.
- Audit records for approval and delivery.

### Stage 5: Operational Hardening

Goal: make scheduled Persona work observable and supportable.

Deliverables:

- Admin/job summaries for Persona scheduled work.
- Retention policy for runs and drafts.
- Quota and rate-limit integration.
- Recovery behavior for stuck pending/running runs.

## Risks

- Background Persona work can feel autonomous in a way that surprises users unless review-required behavior is the default.
- A schedule can drift from its Persona if the Persona profile changes; reference-backed runtime resolution is correct, but runs must record enough policy/audit information to explain historical outcomes.
- Cron/timezone handling can create duplicate or missed runs unless run-slot claiming is explicit.
- Reusing Jobs without a dedicated Persona run table may make review state hard to query.
- Allowing `read_write` memory too early could create unexpected persistent memory updates.
- External delivery channels can become a notification system before the in-app review loop is mature.

## Open Questions For Implementation Planning

- Should the schedule table live in Persona state storage, ChaChaNotes, or a dedicated Persona scheduling DB module?
- Should V1 support only `daily_brief`, or also a generic `scheduled_prompt` kind?
- Should `require_online` be available for Persona scheduled work V1, or should all V1 work run offline and land in review?
- Which existing output/artifact service should own generated briefs?
- What retention default should apply to rejected or unreviewed drafts?

## Acceptance Criteria

- Persona scheduled work is documented as background Persona activity with explicit schedule, run, draft, review, and audit concepts.
- V1 uses APScheduler to enqueue Jobs rather than creating a parallel scheduler or direct in-memory execution path.
- Human review is required before external delivery or durable mutation.
- Persona policy/scopes are evaluated during generation and re-checked before approval-driven delivery/mutation.
- Schedules are user-owned and Persona-reference-backed.
- Duplicate scheduled fires are prevented with run-slot claims and Jobs idempotency keys.
- Buddy animation, design-system backlog work, broad personalization memory, and multi-agent collaboration remain out of scope.

## Verification Plan

- Schema and migration tests for schedules, runs, and reviews.
- API tests for create/list/get/update/delete/run-now and reference validation.
- Scheduler tests for APScheduler registration, disabled schedules, invalid cron/timezone, and duplicate fire protection.
- Jobs tests for enqueue payload, owner id, queue, job type, idempotency key, retry behavior, and run status updates.
- Worker tests for missing Persona, deleted input scope, policy denial, generation failure, and successful draft creation.
- Review tests for approve/reject/edit/expire, policy re-check, and delivery audit.
- Manual backend smoke test for creating a schedule, forcing run-now, observing a Job, reviewing the draft, and confirming audit state.

## References

- `Docs/Product/Persona_Agent_Design.md`
- `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md`
- `Docs/Product/Persona_Backed_Chat_Startup_PRD.md`
- `Docs/Product/Workspace_Persona_Defaults_PRD.md`
- `tldw_Server_API/app/core/Jobs/README.md`
- `tldw_Server_API/app/core/Scheduler/README.md`
- `tldw_Server_API/app/services/reading_digest_scheduler.py`
- `tldw_Server_API/app/services/reminders_scheduler.py`
- `tldw_Server_API/app/core/Collections/reading_digest_jobs.py`
- `tldw_Server_API/app/core/Persona/policy_evaluator.py`
