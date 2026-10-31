# Scheduled Tasks Phase 4C Recurring Question Execution Design

Date: 2026-06-30
Status: Draft For User Review
Owner: Codex brainstorming session
Backlog: TASK-12072

Related:

- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md`
- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md`
- `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md`
- `Docs/ADR/003-jobs-vs-scheduler-default.md`
- `backlog/tasks/task-12072 - Design-Scheduled-Tasks-Phase-4C-Recurring-Question-execution.md`

## Summary

Phase 4C makes `recurring_question` the first executable Scheduled Tasks-owned automation family.

The user promise is:

> Run this question on a schedule against my current searchable knowledge, tell me when useful evidence appears, and keep a complete run history.

This phase adds scheduled and manual execution for Recurring Questions, durable run records, normalized Recurring Question results, result review state, and Home surfacing through the existing visibility policy model. It does not implement Agent Task execution, Watchlists migration, source-specific monitor UI, or corpus-change-triggered automation.

The product remains API-first. The WebUI is the reference and main enterprise client over the API, not the product boundary.

## Current Evidence

Current `origin/dev` constraints that shape this phase:

- Phase 4B persists Scheduled Tasks-owned `recurring_question` and `agent_task` definitions, previews, lifecycle mutations, duplicate/archive behavior, and audit.
- Phase 4B reports execution as unavailable through `GET /api/v1/scheduled-tasks/capabilities`.
- Definitions are projected into `GET /api/v1/scheduled-tasks` as `automation_definition` rows.
- Current automation definition health defaults to `execution_unavailable`.
- Current frontend creation supports Recurring Question definitions but exposes raw `Scope JSON` and generic private/shared visibility, which is too low-level for executable Recurring Questions.
- Phase 3 Results/Home uses a frontend projection adapter from task status and source refs, with a planned normalized results API.
- `Docs/ADR/003-jobs-vs-scheduler-default.md` says new user-visible work should default to Jobs, with APScheduler enqueueing into the chosen backend for recurring schedules.
- Existing RAG endpoints expose `POST /api/v1/rag/search`, `GET /api/v1/rag/features`, `GET /api/v1/rag/source-health`, and request-scoped filters such as sources, collection IDs, media IDs, note IDs, workspace ID, search mode, top K, min score, and generation controls.
- Watchlists remains a separate source-monitoring, source-curation, scraping, ingest, report, and output workspace for a different user persona/job.

## Product Decisions

1. Use a Jobs-backed executable slice for Recurring Questions.
2. Use APScheduler only for cadence and due-run enqueueing.
3. Keep Scheduled Tasks as the definition, run, result, review, visibility, resolution, and audit control plane.
4. Keep Jobs as the execution lifecycle owner for queueing, leasing, retry, cancellation, worker progress, and ops visibility.
5. Add normalized Recurring Question runs and results before claiming durable Results/Home behavior.
6. Use a capability-driven RAG scope contract, not source-specific hardcoding.
7. Let the WebUI render common scope controls first and expose raw scope JSON only as an advanced escape hatch.
8. Every execution creates a durable run record and run summary, including no-match runs.
9. Home and `/scheduled-tasks/results` receive only surfaced findings and attention-worthy failures according to visibility policy.
10. Include `Run now` and `Mark solved` in the first executable slice.
11. Support evidence-only runs when generation is unavailable; use answer synthesis only when configured and available.

## Goals

- Let API clients create, inspect, execute, and monitor Recurring Question definitions.
- Let first-time users understand what will be searched, when runs happen, how costs/quotas apply, and where results appear.
- Let power users run now, inspect many runs, debug failures, duplicate/edit definitions, and close a question as solved.
- Preserve a complete audit trail without flooding Home with routine no-match runs.
- Make source provenance, finding rationale, and confidence visible enough for research trust.
- Keep implementation scope narrow enough for one follow-up plan.

## Non-Goals

- No Agent Task execution.
- No Watchlists redesign, replacement, or migration.
- No source-specific monitor UI for GitHub, YouTube, or any other provider.
- No corpus-change-triggered automation in 4C.
- No bulk result management beyond filters and first-slice review actions.
- No external notification delivery expansion beyond existing/internal notification surfacing.
- No generalized cross-family automation runtime beyond the pieces needed for Recurring Question.
- No raw document text duplication in run/result storage.

## Terms

| Term | Meaning |
| --- | --- |
| Recurring Question | A scheduled research intent that repeatedly runs a question or prompt over searchable knowledge. |
| Definition | Durable configuration created in Phase 4B and extended in 4C with executable readiness and resolution state. |
| Run | One execution attempt, manually triggered or scheduled. |
| Result | A user-facing finding or attention item surfaced from a run. |
| Finding | A run outcome with sufficient evidence to notify or surface to Results/Home. |
| No match | A completed run where retrieval did not meet the finding policy. |
| Finding policy | Stored thresholds and behavior for deciding whether a run produces a surfaced finding. |
| Scope snapshot | Versioned resolved search scope stored with each run for audit and reproducibility. |
| Resolution state | Whether the user's question is still open or has been marked solved. |

## Architecture

Phase 4C extends the Phase 4B foundation with a narrow Recurring Question execution layer.

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Definition/previews/audit | Scheduled Tasks | Existing durable config, validation, lifecycle, and audit from Phase 4B. |
| Resolution state | Scheduled Tasks | `open` or `solved`, separate from lifecycle. |
| Cadence | APScheduler service | Register configured, open Recurring Questions and enqueue due run slots. |
| Execution lifecycle | Jobs | Durable queue, lease, retry, cancellation, worker progress, admin visibility. |
| Run/result store | Scheduled Tasks | User-facing run history, result artifacts, review state, retention, and Home routing. |
| RAG adapter | Scheduled Tasks service boundary | Convert definition input, scope, and finding policy into a safe `UnifiedRAGRequest`. |
| WebUI | API client | Reference client for create/edit, preview, run now, history, results, solved state, and recovery. |

Jobs and Scheduled Tasks must not become competing status authorities. Jobs owns execution state. Scheduled Tasks owns the user-facing run/result projection linked to Jobs by `job_id`.

Every job payload includes `run_id`, and every run stores `job_id`. A reconciliation path repairs stale or divergent records.

## API Contract

All endpoints are under `/api/v1/scheduled-tasks`.

### Capabilities

Extend `GET /api/v1/scheduled-tasks/capabilities` with action-level readiness for Recurring Question execution.

Actions:

- `create_run_manual`
- `execute_scheduled`
- `read_runs`
- `read_results`
- `mutate_results`
- `mark_solved`
- `reopen`
- existing Phase 4B actions: `preview`, `create_definition`, `update_definition`, `pause`, `resume`, `archive`, `duplicate`, and the legacy umbrella `execute`

Keep the Phase 4B action status vocabulary for client compatibility: `available`, `unavailable`, `planned`, or `disabled`. Do not add `degraded` to the action status enum in 4C.

Represent degraded execution readiness with:

- family-level `family_availability="degraded"`;
- action-level `status="available"` or `status="unavailable"` as appropriate;
- action-level `reason`, `warnings`, or `related_capabilities` details that explain the degraded dependency.

For example, manual execution can be `available` while scheduled execution is `unavailable` with reason `scheduler_unavailable`, and the family can be `degraded`.

The Phase 4B `execute` action may remain as a legacy umbrella action for compatibility, but 4C clients should prefer `create_run_manual` and `execute_scheduled` because manual execution and scheduled execution can have different readiness.

`execute_scheduled` must never report action status `degraded`; it reports one of the Phase 4B-compatible action statuses and carries degraded readiness through family availability and related capability details.

Capability readiness should include:

- Jobs backend available
- Recurring Question worker available
- scheduler registration available
- Scheduled Tasks run/result store available
- RAG search available
- generation available or unavailable
- RAG quota/cost state
- supported scope controls
- supported result review states

Manual execution can be available while scheduled execution is degraded or unavailable.

### Runs

Use a resource-style run creation endpoint. The WebUI button label is `Run now`.

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/runs` | Create a manual Recurring Question run. |
| `GET /api/v1/scheduled-tasks/definitions/{definition_id}/runs` | List complete run history for one definition. |
| `GET /api/v1/scheduled-tasks/runs/{run_id}` | Inspect run status, outcome, evidence summary, RAG settings, job link, and failure details. |

Manual runs are allowed for `configured` and `paused` Recurring Questions. Solved definitions do not run by schedule and require reopen before normal execution.

Scheduled due runs are created by APScheduler using the same internal run creation path and a deterministic idempotency key.

Run creation requires:

- active owner context and current readable access to the resolved scope;
- `TASKS_CONTROL` for manual runs;
- definition family is `recurring_question`;
- definition lifecycle is not archived or disabled;
- `resolution_state` is `open`;
- RAG search capability is available;
- scope resolves to at least one searchable source;
- quota/cost admission passes;
- overlap policy allows a new run.

Empty-scope dry runs are out of scope for 4C. Preview and run admission should fail with `scope_empty` when no searchable sources are available in the resolved scope.

### Results

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/scheduled-tasks/results` | List normalized Recurring Question results and failures, plus projected legacy signals for other families until they adopt normalized storage. |
| `GET /api/v1/scheduled-tasks/results/{result_id}` | Inspect surfaced result details, source provenance, finding rationale, review state, and linked run. |
| `POST /api/v1/scheduled-tasks/results/{result_id}/review` | Mutate first-slice review state. |

First-slice review state:

- `unread`
- `read`
- `dismissed`

Do not add cross-product `saved` behavior unless it is implemented as Scheduled Tasks-local state and does not imply Notes, collections, bookmarks, or exports.

### Resolution

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/mark-solved` | Set `resolution_state=solved`, stop future scheduled runs, preserve run/result history. |
| `POST /api/v1/scheduled-tasks/definitions/{definition_id}/reopen` | Set `resolution_state=open`; caller chooses `paused` or `configured` lifecycle only when the current lifecycle permits it. |

`resolution_state` is separate from Phase 4B lifecycle. This avoids overloading `configured`, `paused`, `archived`, and `disabled`.

Solved definitions:

- are not registered with APScheduler;
- do not create skipped runs on future schedule slots;
- stay visible in filters and history;
- can be reopened explicitly.

Updating a solved definition does not implicitly reopen it. Update previews and updates may change editable configuration on a solved definition only when the lifecycle permits editing, but `resolution_state` remains `solved` and scheduler registration remains off. To run or schedule the definition again, clients must call `POST /api/v1/scheduled-tasks/definitions/{definition_id}/reopen`.

Reopen records a dedicated audit event and requires the caller to choose whether the reopened definition should be `paused` or `configured`. Reopening to `configured` must re-run health, permission, capability, scope, and schedule checks before scheduler registration.

Resolution transition matrix:

| Current lifecycle | Current resolution | `mark-solved` behavior | `reopen` behavior |
| --- | --- | --- | --- |
| `configured` | `open` | Set `solved`, unregister future scheduled runs, preserve lifecycle as `configured`. | No-op conflict or validation error because already open. |
| `paused` | `open` | Set `solved`, preserve lifecycle as `paused`. | No-op conflict or validation error because already open. |
| `configured` | `solved` | Idempotent no-op. | Set `open`; target lifecycle can be `paused` or `configured` after checks. |
| `paused` | `solved` | Idempotent no-op. | Set `open`; target lifecycle can be `paused` or `configured` after checks. |
| `archived` | `open` or `solved` | Reject with archived-definition error. | Reject with archived-definition error. |
| `disabled` | `open` or `solved` | Reject unless the disabled lock kind explicitly allows user resolution metadata changes. | Reject. Disabled/admin/security/system locks cannot be bypassed by reopen. |

`reopen` must never clear `disabled_lock_kind`, `disabled_reason`, archive state, admin locks, security locks, or system locks.

## Data Model

### Definition Extensions

Add or expose these fields for `recurring_question` definitions:

| Field | Purpose |
| --- | --- |
| `resolution_state` | `open` or `solved`. |
| `resolved_at` | Timestamp when marked solved. |
| `resolved_by` | Actor who marked solved. |
| `resolved_result_id` | Optional result that answered the question. |
| `finding_policy` | Thresholds and mode used to decide whether a run creates a surfaced finding. |
| `retention_policy` | Run/result retention behavior. |

Default `resolution_state`: `open`.

Default visibility policy: `findings_only`.

### Scope Contract

Store scope as a versioned object.

Example:

```json
{
  "schema_version": "2026-06-30",
  "mode": "capability_defined",
  "sources": ["media_db", "notes"],
  "all_searchable_library": true,
  "collection_ids": [],
  "tag_ids": [],
  "saved_search_ids": [],
  "source_types": [],
  "date_window": null,
  "workspace_id": null,
  "advanced_filters": {}
}
```

The API must tolerate unsupported fields by rejecting them with typed validation errors or ignoring future fields only when version rules allow it. The WebUI renders supported controls from capability discovery:

- all searchable library;
- collections;
- tags;
- saved searches;
- source types;
- date window;
- workspace scope where supported;
- advanced JSON escape hatch.

GitHub, YouTube, and similar providers are examples of possible source content after ingest, not primary scheduled-task scope types.

`all_searchable_library` means all capability-reported RAG sources that are enabled, searchable, and readable by the current owner at preview and run time. It must not expand to every configured database, connector, deployment-wide source, or admin-only corpus.

### Finding Policy

Store finding policy per definition and snapshot it per run.

Recommended presets:

| Preset | Behavior |
| --- | --- |
| `balanced_findings` | Default. Surface findings with enough evidence and moderate confidence. |
| `high_confidence_only` | Require stronger retrieval score, evidence count, or synthesis confidence. |
| `every_run` | Surface each run summary when the user explicitly chooses high visibility. |

Advanced fields can include:

- `min_relevance_score`;
- `min_evidence_count`;
- `top_k`;
- `generation_mode`: `auto`, `disabled`, or `required`;
- `synthesis_confidence_threshold`;
- `duplicate_evidence_policy`;
- `no_match_visibility`.

If generation is unavailable and policy is not `required`, the run proceeds in evidence-only mode.

4C does not add new provider or model selection UX. `generation_mode` controls whether answer synthesis is requested, but provider/model/profile resolution should use existing RAG defaults unless the definition stores an approved RAG profile reference or safe request overrides validated during preview. Preview should show the resolved synthesis mode and cost/quota class without exposing secrets.

### Run

Separate execution state from research outcome.

| Field | Purpose |
| --- | --- |
| `id` | Stable run ID. |
| `definition_id` | Owning definition. |
| `definition_version` | Definition version at run creation. |
| `question_version` | Question/prompt version. |
| `job_id` | Jobs record ID. |
| `trigger_reason` | `scheduled`, `manual`, `retry`, or `system_repair`. |
| `schedule_slot` | Due slot for scheduled runs; null for manual runs unless caller provides a slot label. |
| `status` | `queued`, `running`, `completed`, `failed`, `skipped`, `cancelled`. |
| `outcome` | `finding`, `no_match`, `partial`, `degraded`, `none`. |
| `scope_snapshot` | Resolved scope used for this run. |
| `finding_policy_snapshot` | Policy used for this run. |
| `rag_request_snapshot` | Safe RAG request summary, with secrets and raw source text excluded. |
| `run_summary` | User-facing summary of what happened. |
| `evidence_summary` | Counts, top source refs, scores, and citation metadata. |
| `failure_reason` | Typed failure or skip reason. |
| `started_at`, `ended_at` | Timing. |

Typed failure and no-match reasons should include:

- `rag_unavailable`
- `scope_empty`
- `no_relevant_evidence`
- `generation_unavailable`
- `generation_failed`
- `quota_exceeded`
- `permission_denied`
- `source_unavailable`
- `job_cancelled`
- `worker_failure`
- `overlap_skipped`

### Result

Create a result only when visibility policy and finding/failure rules route the run outside task history.

| Field | Purpose |
| --- | --- |
| `id` | Stable result ID. |
| `definition_id` | Owning definition. |
| `run_id` | Source run. |
| `kind` | `finding` or `failure`. |
| `title` | Scannable result title. |
| `summary` | User-facing result summary. |
| `answer` | Optional synthesized answer. |
| `answer_mode` | `synthesized`, `evidence_only`, or `none`. |
| `confidence` | Optional normalized confidence or rationale. |
| `source_refs` | Source IDs, titles, citation refs, short redacted snippets, and scores. |
| `review_state` | `unread`, `read`, or `dismissed`. |
| `dedupe_key` | Deterministic result identity for Home/notifications. |
| `visibility_destination` | Where the result may appear. |

Do not duplicate raw document text. Store source IDs, titles, citation references, short redacted snippets, scores, and retrieval metadata sufficient for audit and triage.

## Execution Flow

1. Preview validates question/prompt, scope, schedule, finding policy, visibility, RAG readiness, generation mode, quota/cost class, retention, and result destination.
2. Create persists a definition with `resolution_state=open`, default `findings_only`, and a stored finding policy. Update preserves the current `resolution_state`; solved definitions stay solved until the explicit reopen endpoint is called.
3. APScheduler registers configured, open Recurring Questions when `execute_scheduled` capability is available.
4. A due slot or manual run creates a Scheduled Tasks run with a deterministic idempotency key.
5. The service enqueues a Jobs job with `run_id`, `definition_id`, `definition_version`, owner, trigger reason, and schedule slot.
6. Worker acquires the job, marks the run running, resolves the definition and scope, and executes the RAG adapter.
7. RAG returns evidence, optional generated answer, citations, metadata, and errors.
8. Worker writes a complete run summary and status/outcome.
9. If the finding policy is met, worker creates a result artifact and routes it according to visibility policy.
10. Home consumes surfaced result/failure items only. No-match runs remain in run history.
11. Mark solved removes future scheduler registration and preserves history.

### Idempotency And Overlap

Scheduled run idempotency key:

```text
recurring_question:{definition_id}:{definition_version}:{trigger_reason}:{schedule_slot}
```

Manual run idempotency should support an optional `Idempotency-Key` header. Without a key, repeated manual clicks can create separate manual runs, subject to overlap and quota policy.

Overlap policy comes from the definition schedule. Default:

- scheduled runs: `skip_new`;
- manual runs: block if another run is active unless caller explicitly requests allowed behavior and capability permits it.

## Visibility, Home, And Results

Every run is visible in definition run history. Not every run becomes a result.

Default Recurring Question visibility: `findings_only`.

| Visibility policy | Behavior |
| --- | --- |
| `findings_only` | Surface findings and attention-worthy failures; keep no-match runs in history. |
| `every_run` | Surface each run summary. |
| `failures_only` | Surface only failures that need attention. |
| `task_history_only` | Do not surface to Home/Results; keep complete history. |

Failure surfacing should dedupe and escalate:

- first failure;
- still failing after N runs;
- recovered after failure;
- quota or permission blocked.

Home `Automation Inbox` should consume normalized surfaced results and failures, not inferred task status. Existing projected legacy signals remain for families without normalized results and must be labeled as such in `/scheduled-tasks/results`.

## WebUI Reference Client

### First-Time Creation Flow

Recurring Question creation should be guided, capability-aware, and source-agnostic.

1. `Question or prompt`
   - Helper: "This will be sent to RAG search and, when enabled, answer synthesis."
2. `Scope`
   - Render only supported controls from capability discovery.
   - Common controls: all searchable library, collections, tags, saved searches, source types, date window, workspace scope.
   - Advanced JSON is available but not primary.
3. `Finding behavior`
   - Presets first: `Balanced findings`, `High confidence only`, `Every run`.
   - Numeric thresholds live under advanced settings.
4. `Schedule`
   - Readable cadence, timezone, missed-run policy, overlap policy.
   - Cron remains advanced.
5. `Preview`
   - Shows what will be searched, RAG readiness, generation readiness, evidence-only fallback, cost/quota class, retention, and result destinations.
   - Shows plain-language recovery notes, not raw diagnostics.
6. `Create`
   - Save as configured or paused.
   - Offer `Run now` after successful creation when manual run capability is available.

### Power User Management

Definition detail should show:

- lifecycle and resolution state;
- health;
- next run and last run;
- latest outcome;
- `Run now`;
- `Pause` and `Resume`;
- `Mark solved` and `Reopen`;
- `Duplicate`;
- `Archive`;
- run history table;
- definition-scoped results;
- audit events;
- advanced diagnostics where available.

Run history table fields:

- status;
- outcome;
- trigger;
- schedule slot;
- duration;
- evidence count;
- answer mode;
- failure/no-match reason;
- linked job ID;
- created and completed timestamps.

Default detail views should show `Why this happened` summaries. Raw diagnostics and IDs are available in advanced/admin affordances.

### Results UX

`/scheduled-tasks/results` should show:

- normalized Recurring Question findings;
- normalized Recurring Question failures;
- projected legacy signals for other families, labeled as projected until normalized APIs exist for them.

Result detail should prioritize:

1. answer or evidence summary;
2. source provenance;
3. finding rationale;
4. run metadata;
5. RAG settings snapshot;
6. review controls;
7. advanced diagnostics.

No-match runs are not shown as Results items by default.

### Solved Questions

Solved questions:

- remain visible through filters;
- show copy: "Solved questions are not scheduled.";
- preserve run/result history;
- can be reopened.

Filters should include:

- `Open`;
- `Solved`;
- `Archived`;
- family;
- lifecycle;
- health;
- result state;
- run outcome.

### Extension Behavior

The extension and WebUI share the UI package.

In the first 4C slice:

- compact list, result, and detail views should remain readable in extension width;
- complex create/edit flows may deep-link to the full WebUI if controls exceed sidepanel width;
- deep links should preserve target definition, run, or result IDs.

## Empty, Loading, Error, And Success States

| State | Copy / behavior |
| --- | --- |
| No recurring questions | "No recurring questions yet." Primary action: create one. |
| No results | "No useful matches yet. Runs are still recorded in task history." |
| No runs | "This question has not run yet." Primary action: `Run now` if available. |
| Running | Show progress status, queued/running state, and a link to run detail. Use live status text. |
| Finding found | Show answer/evidence summary, source count, confidence/rationale, and review controls. |
| Completed no match | Show no-match rationale in run history, not Home by default. |
| RAG unavailable | "Search is unavailable. Runs will be skipped until this is fixed." |
| Generation unavailable | "Evidence search can still run. Answer synthesis is disabled." |
| Scope empty | "This question has no searchable sources in scope." |
| Quota exceeded | "Run skipped because the RAG quota is exhausted." |
| Solved | "Solved questions are not scheduled." Primary action: reopen. |

## Accessibility And Usability

- Filters, row actions, review controls, `Run now`, `Mark solved`, and `Reopen` must be keyboard-operable.
- Status must not rely on color alone. Pair badges with visible text.
- Running state should update through accessible status/live-region text where practical.
- Tables need accessible names, sensible column headers, and row action labels that include the task/result context.
- Source links need accessible labels that distinguish source, citation, run, job, and result targets.
- Compact extension views must preserve focus order and not hide critical status.
- Preview errors should identify the field and recovery action.
- Advanced JSON should not be required for first-time creation.

## Backend Dependencies

4C requires backend/API work before the WebUI can claim execution:

- Scheduled Tasks storage for runs, results, review state, resolution state, finding policy snapshots, retention metadata, and job/run links.
- Jobs domain, queue, and worker for Recurring Question execution.
- APScheduler service for configured, open Recurring Questions.
- Capability probes for Jobs, scheduler registration, RAG search, optional generation, quota, and run/result store.
- RAG adapter that maps definition scope and finding policy to `UnifiedRAGRequest`.
- Retention and reconciliation services for old runs, surfaced results, orphaned jobs, and stale running runs.
- Tests proving Watchlists behavior remains unchanged.

## Retention And Privacy

Recurring Questions can produce many runs. Phase 4C needs explicit retention.

Recommended defaults:

- keep recent run summaries by count and age;
- keep surfaced results longer than no-match runs;
- preserve the final solved finding unless dismissed or removed by policy;
- preserve audit events according to existing audit retention policy;
- do not store raw document text in run/result records;
- store short redacted snippets and source references only;
- exclude secrets, provider keys, raw Agent Task payloads, and raw RAG debug dumps from user-facing records.

Preview should explain retention and redaction behavior before create/update.

## Reconciliation

The system needs repair behavior for divergent Jobs and Scheduled Tasks state.

Rules:

- every job payload includes `run_id`;
- every run stores `job_id`;
- Jobs result includes `run_id`;
- stale `queued` or `running` runs become `failed` or `skipped` with a typed repair reason when Jobs cannot complete them;
- orphaned completed Jobs without a finalized run create a `needs_attention` repair event;
- reconciliation events are visible in run detail and audit/debug surfaces.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Duplicate scheduled runs | Slot idempotency and overlap policy. |
| Home/result flooding | `findings_only`, dedupe, failure escalation. |
| Cost surprise | Preview cost/quota class, generation optional. |
| Privacy leakage | Redacted snippets, source references, retention policy. |
| RAG unavailable | Capability health and skip/recovery states. |
| Scope drift | Scope and finding-policy snapshots per run. |
| Job/run divergence | Bidirectional `job_id`/`run_id` reconciliation. |
| Power-user friction | Run filters, direct links, run IDs, audit tab. |
| First-time confusion | Guided create flow, preview explanations, plain-language states. |
| Watchlists confusion | Explicit copy that Recurring Question searches existing knowledge while Watchlists monitors and ingests sources. |

## Verification Requirements

Backend tests:

- scope normalization and validation;
- finding policy presets and advanced thresholds;
- lifecycle and resolution state rules;
- manual run creation;
- scheduled slot idempotency;
- overlap behavior;
- run/result storage;
- result review state;
- retention selection;
- reconciliation of stale or divergent Jobs/runs;
- RBAC and owner isolation;
- route ordering for static child routes;
- Watchlists projection compatibility.

Worker tests with mocked RAG:

- finding with synthesized answer;
- finding in evidence-only mode;
- completed no match;
- generation unavailable;
- RAG unavailable;
- scope empty;
- quota exceeded;
- source unavailable;
- retryable worker failure;
- terminal worker failure.

Scheduler tests:

- due slot claim;
- missed-run policy;
- overlap policy;
- paused definitions do not schedule unless manually run;
- solved definitions do not schedule;
- archived and disabled definitions do not schedule.

Frontend tests:

- capability-driven create controls;
- advanced JSON escape hatch hidden by default;
- preview readiness/cost/retention/result-destination copy;
- `Run now` action availability;
- run history table;
- result detail;
- Home dedupe and failure escalation display;
- solved/open/archive filters;
- extension-width list/result/detail behavior;
- keyboard access and color-independent states.

## Implementation Slice Recommendation

Plan the implementation in stages:

1. Backend contracts and storage for runs, results, review state, resolution state, and capabilities.
2. Manual `Run now` path with mocked or adapter-backed RAG execution and normalized run/result records.
3. Jobs worker integration and RAG adapter hardening.
4. APScheduler registration for configured open Recurring Questions.
5. WebUI guided create/edit upgrades, run history, result detail, Home normalized surfacing, and solved filters.
6. Retention, reconciliation, accessibility, extension-width hardening, and final verification.

Manual run should land before scheduled execution so the team can validate definition, scope, RAG, result, and Home behavior without waiting on scheduler registration.

## Acceptance Criteria

- Recurring Question definitions can be executed manually and on schedule when capabilities are available.
- Every execution creates a durable run summary, including no-match runs.
- Findings and attention-worthy failures create normalized results routed by visibility policy.
- Home shows only surfaced findings/failures, not routine no-match runs.
- Users can inspect run history, result detail, source provenance, and failure reasons from `/scheduled-tasks`.
- Users can mark a question solved and later reopen it.
- Jobs and Scheduled Tasks states are linked and reconcilable.
- WebUI creation is capability-aware and does not require raw JSON for common scopes.
- Watchlists remains unchanged and clearly separate.
- Tests cover API, service, worker, scheduler, frontend, accessibility, retention, and reconciliation behavior.
