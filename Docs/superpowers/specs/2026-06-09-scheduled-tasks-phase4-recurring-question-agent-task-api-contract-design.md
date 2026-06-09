# Scheduled Tasks Phase 4 Recurring Question And Agent Task API Contract Design

Date: 2026-06-09
Status: Ready for review
Owner: Codex brainstorming session
Backlog: TASK-2342

Related:

- `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
- `Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md`
- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md`
- `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md`
- `Docs/ADR/003-jobs-vs-scheduler-default.md`
- `backlog/tasks/task-2342 - Design-Scheduled-Tasks-Phase-4-API-first-recurring-question-and-agent-task-contract.md`

## Summary

Phase 4 defines an API-first product contract for two equal scheduled automation families:

- **Recurring Question**: repeatedly run a question against evolving searchable knowledge and record every run summary.
- **Agent Task**: send a scheduled prompt or message to a selected API/ACP agent with explicit preview, permission, approval, run, result, and audit behavior.

The API contract is the product source of truth. The WebUI is the reference and main enterprise client, not the boundary of the product. Other clients, including the browser extension, direct API users, and future enterprise integrations, must be able to discover capabilities, preview configs, create tasks, monitor runs, inspect results, handle approvals, and recover from failures without depending on WebUI-only state.

The first implementation slice should remain an honest reference-client shell. Recurring Question and Agent Task already exist as planned templates in the Create tab, but their current panels are thin. Slice 4A should improve those planned templates with capability-aware API-first copy, requirement lists, result destination explanations, and links to RAG or ACP surfaces. It must not create fake tasks, persist drafts, create server drafts, fake run history, or claim executable support before API capability discovery says it exists.

This spec preserves existing Watchlists behavior. Watchlists remains a separate workspace for source monitoring, source curation, scraping, ingest, reports, outputs, and its own user persona/job. Watch/Ingest are not demoted or absorbed by Recurring Question.

## Product Decision

Use an **API-first Scheduled Task contract** for Recurring Question and Agent Task, with `/scheduled-tasks` acting as a reference enterprise client over shared API resources.

The contract should define:

1. Automation capability discovery.
2. Durable automation definitions.
3. Previews before create.
4. Runs and run history.
5. User-facing results separate from raw run logs.
6. Approval requests and approval decisions for agent execution.
7. Audit/events for enterprise debugging and governance.
8. Notification and visibility policy for Home, Results, task history, and future external delivery.

The first WebUI slice should not wait for the whole executable backend. It can improve discoverability and requirements, but it must say "planned" or "API unavailable" when executable support is absent.

## Current Evidence

Source review of `origin/dev` found these constraints:

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts` already includes `recurring_question` and `agent_task` template IDs with `planned` state.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx` currently renders planned templates with a minimal panel.
- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts` is oriented around Watch/Ingest gates such as source preview, duplicate detection, Watchlists preservation, and result destinations.
- `tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py` currently supports only `reminder_task` and `watchlist_job` as normalized scheduled-task primitives.
- `tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py` exposes list/detail plus reminder mutations, not Recurring Question or Agent Task mutations.
- `tldw_Server_API/app/api/v1/endpoints/acp_schedules.py` already exposes `/acp/schedules`, but it stores ACP schedules in the Workflows Scheduler path and is not yet normalized into `/scheduled-tasks`.
- `Docs/ADR/003-jobs-vs-scheduler-default.md` says new user-visible work should default to Jobs, while Scheduler remains the default for internal orchestration. Executable Phase 4 work must resolve this ownership choice explicitly.
- RAG already exposes discovery through `/api/v1/rag/capabilities` and `/api/v1/rag/features`. Recurring Question should compose those capabilities rather than duplicate them.
- Phase 3 Results currently supports `projected_signals` and future `normalized_results_read` / `normalized_results_mutation` modes. Phase 4 must not claim durable every-run summaries or review actions until normalized run/result APIs exist.

## Goals

1. Make Recurring Question and Agent Task equal first-class planned automation families.
2. Define the API resources clients need before executable scheduling ships.
3. Keep the WebUI honest: explain the future workflow and requirements without fake creation or fake data.
4. Make every run, result, failure, approval, and notification traceable.
5. Keep Home and `/scheduled-tasks/results` aligned through a visibility policy rather than hardcoded UI promises.
6. Preserve Watchlists, RAG, ACP, Agent Tasks, and domain workspaces as deep owners for their specialist workflows.
7. Give implementation planning clear backend dependency decisions without prematurely designing database tables.

## Non-Goals

- Do not implement executable Recurring Question scheduling in Slice 4A.
- Do not implement executable Agent Task scheduling in Slice 4A.
- Do not persist local UI drafts or server-side drafts in Slice 4A.
- Do not retrofit Watch/Ingest capability gates onto Recurring Question or Agent Task.
- Do not replace `/acp/schedules` or RAG APIs in this design spec.
- Do not move Watchlists source-monitoring behavior into Recurring Question.
- Do not promise Home surfacing for every run unless a task visibility policy says so.
- Do not define storage migrations, worker implementation, or queue topology in this product spec.

## Terms

| Term | Meaning |
| --- | --- |
| Automation family | Product-level type such as Reminder, Watch/Ingest, Recurring Question, or Agent Task. |
| Automation definition | Durable scheduled task configuration with owner, family, schedule, lifecycle, input payload, and policies. |
| Capability discovery | API response that tells clients whether a family is supported, unavailable, degraded, or planned. |
| Preview | Non-destructive validation and explanation of what would run before creation. |
| Run | One execution attempt, including queued, running, awaiting approval, completed, failed, skipped, or cancelled states. |
| Result | User-facing output or finding generated from a run, separate from logs and raw execution events. |
| Visibility policy | Per-task policy that controls where run summaries, findings, failures, and approvals appear. |
| Approval policy | Agent Task policy that controls whether future runs execute automatically or require user/admin approval. |
| Risk class | Machine-readable classification of the agent action or tool class used to decide approval requirements. |

## API Resource Contract

Phase 4 should define these API resources as the source of truth. Exact endpoint names can follow project conventions during implementation, but the resource boundaries should remain stable.

| Resource | Required role | Key fields / behavior |
| --- | --- | --- |
| Capabilities | Tells clients which automation families are available and why. | Family availability, required backend modules, missing dependencies, degraded reasons, supported preview/create/run/result features. |
| Automation definitions | Durable scheduled task configs. | Stable ID, family, owner, name, description, lifecycle state, health state, schedule, input payload, result visibility policy, notification policy, approval policy, created/updated metadata. |
| Previews | Validates setup before creation. | Normalized config, validation errors, estimated behavior, risk class for Agent Task, eligible result destinations, missing capabilities. |
| Runs | Execution attempts. | Stable run ID, automation ID, status, started/ended timestamps, trigger reason, schedule slot, retry attempt, skipped/failed reason, summary, linked result IDs. |
| Results | User-facing outputs and findings, separate from raw run logs. | Stable result ID, automation ID, run ID, result type, title, summary, confidence or rationale when applicable, source links, visibility destination, read/saved/dismissed state. |
| Approvals | Human decisions for risky Agent Task execution. | Approval ID, automation/run ID, requested action, risk class, required decision, expires at, approved/denied by, audit trail. |
| Events / audit | Machine-readable history for enterprise clients and debugging. | Created, edited, paused, resumed, previewed, run started, approval requested, approved, denied, failed, result created, notification sent. |
| Notifications | Delivery and surfacing contract, not just UI toasts. | Destination, dedupe key, severity, related automation/run/result IDs, delivery state, link target. |

## Existing API Integration Constraints

The Phase 4 contract should be additive.

| Existing surface | Constraint |
| --- | --- |
| `/api/v1/scheduled-tasks` | Currently normalizes reminders and Watchlist jobs. Do not redefine `primitive` fields in place without a compatibility plan. Add family-aware resources or versioned fields that can coexist with the current read model. |
| `/acp/schedules` | Existing ACP schedule CRUD proves there is ACP scheduling functionality, but it does not satisfy the unified Scheduled Tasks contract for preview, risk, approvals, results, audit, or normalized visibility. |
| RAG capabilities | Recurring Question should compose `/api/v1/rag/capabilities` and `/api/v1/rag/features` rather than inventing a parallel RAG readiness model. |
| Phase 3 Results | Durable run summaries and review states require normalized results/read APIs. In projected mode, the UI can explain planned behavior but must not claim durable result history. |
| Jobs vs Scheduler ADR | Executable user-visible Phase 4 work needs an explicit backend ownership decision. Default to Jobs-backed user-visible runs unless maintainers accept a Scheduler exception for a specific slice. |

## State Model

Do not use one overloaded status field. Split state across layers.

| Field | Examples | Owner |
| --- | --- | --- |
| Automation lifecycle | `scheduled`, `paused`, `archived`, `completed`, `disabled` | User, admin, or system policy |
| Automation health | `ready`, `degraded`, `needs_attention`, `capability_unavailable`, `permission_required` | System evaluation |
| Run status | `queued`, `running`, `awaiting_approval`, `completed`, `completed_no_result`, `failed`, `skipped`, `cancelled` | Execution engine |
| Result state | `unread`, `read`, `saved`, `dismissed` | User or client interaction |

The WebUI should render these as separate concepts: lifecycle near task controls, health near requirements/recovery, run status in run history, and result state in the Results/Home surfaces.

## Schedule Semantics

Every automation definition should explicitly capture:

- Timezone.
- DST behavior.
- Missed-run policy.
- Overlap/concurrency policy.
- Retry and backoff policy.
- Manual-run behavior.
- Idempotency key behavior for create, update, and run requests.

Schedules should be described in user-facing language in clients, but API clients need stable machine-readable policy fields. Raw cron may exist as an advanced representation, but it should not be the only schedule contract for first-time users.

## Recurring Question Contract

Recurring Question answers: "Keep asking this question as new data arrives or as the searchable corpus changes."

API-supported setup should include:

| Concern | Contract requirement |
| --- | --- |
| Question | Question text, optional success criteria, optional answer freshness expectation, question versioning. |
| Scope | All library, collection, tags, saved search, source types, date window, or future capability-defined scopes. |
| Retrieval behavior | RAG/search profile reference, limits, confidence or match thresholds where supported. |
| Schedule | Shared schedule semantics with timezone, DST, missed-run, overlap, retry, and manual-run behavior. |
| Visibility | Whether every run, findings only, failures only, or task-history-only output appears on Home/results. |
| Run summary | Every run records what was searched, what changed, answer/finding summary, no-useful-match rationale, and source links/IDs when applicable. |
| Result artifact | Findings or configured run summaries create user-facing results based on visibility policy. |
| Provenance | Source IDs, citations, retrieval summary, corpus/scope version or resolver, question version, and match rationale. |

Every Recurring Question run should produce a run summary. That does not mean every run must create a Home item. Run history is complete; Home/results are visibility-policy routed.

### Recurring Question Edge Cases

| Edge case | Required behavior |
| --- | --- |
| Question edited | Preserve question version per run. |
| Scope changed | Run history records scope snapshot or scope resolver version. |
| No useful match | Run completes with a summary and no finding result unless visibility policy surfaces every run. |
| RAG unavailable | Automation health becomes `capability_unavailable`; run behavior follows skip/fail policy. |
| Duplicate evidence | Result can reference prior result and explain what changed. |
| Source deleted | Run records source unavailable or scope changed instead of silently changing history. |

## Agent Task Contract

Agent Task answers: "Send this message to this selected agent at this time or cadence."

Every Agent Task requires preview before creation.

API-supported setup should include:

| Concern | Contract requirement |
| --- | --- |
| Agent identity | Selected API/ACP agent, agent availability, workspace/context, optional persona/model/token settings where supported. |
| Message payload | Prompt/message that will be sent, with preview-safe redaction rules. |
| Schedule | Shared schedule semantics with timezone, DST, missed-run, overlap, retry, and manual-run behavior. |
| Tool/permission policy | Allowed tool classes, denied tool classes, environment/context limits, sandbox/governance metadata. |
| Risk class | Machine-readable risk class returned from preview and stored on definition/run where applicable. |
| Approval policy | Automatic for allowed low-risk actions; approvals required for configured risky actions or protected tool classes. |
| Result recording | Agent input summary, output/transcript/artifact references, tool boundary, failure reason, and audit events. |

The safety model is:

1. Always preview before scheduling.
2. Future runs execute automatically when the declared agent/tool class is allowed.
3. Risky tool classes require approval according to policy.
4. Run records show whether the run executed automatically, awaited approval, was approved, was denied, or expired.
5. The output shows what agent received, what it returned, and which tool/action boundary was crossed.

### Preview Response Requirements

Agent Task preview must return:

- Selected agent identity and availability.
- Message/payload that will be sent.
- Schedule interpretation.
- Tool/permission classes requested.
- Risk class.
- Whether future runs execute automatically or require approval.
- Approval expiration/default behavior.
- What will be recorded in run history and results.
- Validation errors and missing capabilities.

### Agent Task Edge Cases

| Edge case | Required behavior |
| --- | --- |
| Agent unavailable at run time | Run fails or skips with typed reason. |
| Approval expires | Run becomes `skipped` or `cancelled` based on policy. |
| Tool permission changed after creation | Health becomes `permission_required` or `needs_attention`. |
| Message references missing context | Preview or run records validation failure. |
| Agent returns partial output | Run can complete with degraded/partial result metadata. |
| Existing `/acp/schedules` schedule exists | Unified task projection may link to it, but creation/editing should not claim full Phase 4 support until preview, risk, approval, result, and audit contracts are satisfied. |

## Visibility Policy

Visibility policy controls where outputs appear.

| Policy | Behavior |
| --- | --- |
| Every run | Home/results show each run summary. Useful for high-priority questions or audit-heavy agent tasks. |
| Findings only | Home/results show confident answers, promising evidence, or meaningful agent outputs; no-match runs stay in task history. |
| Failures only | Home/results interrupt only when automation breaks or needs approval. |
| Task history only | No Home/result inbox surfacing; still fully auditable from task detail. |

The API should separate run history from result surfacing. A run can exist without creating a surfaced result. A result should always link back to its run.

## WebUI Reference Client Behavior

The WebUI should demonstrate the API contract clearly without pretending unsupported capabilities exist.

| Surface | Slice 4A shell behavior | Future behavior when API support exists |
| --- | --- | --- |
| Create tab | Show Recurring Question and Agent Task as equal planned templates with API-first requirements, result destinations, and deep links. | Enable preview/create when capability discovery says the family is executable. |
| Task list / monitoring | Do not show fake Recurring Question or Agent Task rows. If real API definitions appear later, render shared lifecycle/health/run states. | Filter by family, state, health, owner, last run, next run. Support pause/resume/edit/duplicate/archive when APIs exist. |
| Results tab | Add explanatory copy that future Recurring Question and Agent Task outputs will appear according to task visibility policy. Do not inject sample result cards. | Show run-linked result artifacts with source links, agent outputs, approvals, and result state. |
| Home Automation Inbox | Existing surfaced results remain. Add no fake items. Optional empty-state copy can mention future scheduled question findings and agent outputs when enabled by visibility policy. | Show only routed results/notifications: findings, configured run summaries, failures, approvals, saved/pinned items. |
| Detail drawer/page | Existing families only in Slice 4A. Planned templates can link to conceptual destinations. | Show definition, schedule, health, latest run, run history, results, approvals, audit events, and debugging context. |

### Create Tab Planned Template Requirements

Each planned template card or panel should include:

- Family label: `Recurring Question` or `Agent Task`.
- Status badge: `Planned` or `API unavailable`.
- One-sentence job statement.
- Requirements list driven by capability discovery when available.
- Result destinations line: task history, Results, Home if configured.
- Safety line for Agent Task: preview required; approvals may be required by permission class.
- Availability reason when disabled.
- Deep links:
  - Recurring Question: RAG/search workspace or related docs.
  - Agent Task: ACP/agent workspace or related docs.

### Slice 4A Capability Fallback

Slice 4A must not add a fake family-discovery API just to make the shell look dynamic. Use this fallback order:

1. If a Scheduled Tasks automation-family capability endpoint exists and advertises these families, render the advertised state and reason.
2. If only related domain capabilities exist, such as RAG capabilities or ACP route availability, show those as requirements or related readiness signals, not as executable Scheduled Tasks support.
3. If no family capability endpoint exists, keep Recurring Question and Agent Task in `planned` state and label the panel as `Planned automation type`.
4. Keep create/preview actions disabled or absent unless the API advertises both preview and create support for that family.

This prevents the WebUI from becoming the source of truth while still letting it explain why the future template matters.

### What Slice 4A Must Not Do

- Do not persist local drafts.
- Do not create server drafts.
- Do not use fake tasks, fake runs, or fake results.
- Do not imply GitHub or YouTube are primary source types.
- Do not merge or diminish Watchlists.
- Do not hardcode "available" without API evidence.
- Do not reuse Watch/Ingest capability gates for Recurring Question or Agent Task.

## Future Creation Flows

These flows are not part of Slice 4A, but the spec defines them so shell copy, API dependencies, and later implementation align.

| Step | Recurring Question | Agent Task |
| --- | --- | --- |
| Start | User/client chooses `Recurring Question`. | User/client chooses `Agent Task`. |
| Configure intent | Enter question, optional success criteria, answer freshness, and run/result summary preference. | Select agent, enter scheduled message/prompt, choose one-time or recurring schedule. |
| Configure scope | Select searchable scope: all library, collection, tags, saved search, source types, date window, or supported scope provider. | Select allowed tool/permission classes, environment/context, and agent-specific parameters. |
| Configure schedule | Timezone-aware cadence, missed-run policy, overlap behavior, retry policy. | Same schedule contract plus approval expiration/default behavior when required. |
| Configure visibility | Choose where run summaries, findings, and failures appear. | Choose where outputs, failures, approval requests, and completion notices appear. |
| Preview | API validates scope, schedule, expected search behavior, result routing, and missing capabilities. | API validates selected agent, message, tools, schedule, risk class, approval mode, and output recording. |
| Create | API creates automation definition only after valid preview or equivalent idempotent validation. | API creates automation definition only after valid preview and permission policy confirmation. |
| Monitor | User/client sees next run, last run, every run summary, result artifacts, no-match rationale, and failures. | User/client sees next run, approval queue, run transcript/output, tool boundary, failures, and audit events. |
| Manage | Edit, pause, resume, duplicate, archive, manual run. | Edit, pause, resume, duplicate, archive, manual run, approve/deny pending run. |

## Power-User Requirements

| Need | Requirement |
| --- | --- |
| Duplicate tasks | Duplicate a definition as a starting point without copying run/result history. |
| Bulk operations | Bulk pause/resume/archive by family/state when safe APIs exist. |
| Dense filters | Filter by owner, family, health, next run, last failure, result visibility, approval required. |
| Search | Search by task name, question text, agent, source scope, result text, run ID. |
| Export | Export definitions, run summaries, and results through API with permission checks. |
| Debugging | Inspect typed failure, failed step, retry policy, approval state, provider/agent/RAG health, and relevant audit events. |

## Empty, Loading, Running, Error, And Success States

| Surface / state | Required UX |
| --- | --- |
| Create empty | "Choose an automation type. Availability is based on server capabilities." Planned templates say what is missing and where to configure related systems. |
| Results empty | "No scheduled task results yet. When automations create results, they appear here according to each task visibility policy." |
| Run history empty | "No runs yet." Add next run time when scheduled, or explain why not scheduled. |
| Approvals empty | "No approvals waiting." Do not hide failed or expired approvals from audit history. |
| Loading capabilities | "Checking automation capabilities." |
| Previewing Recurring Question | "Previewing recurring question." |
| Previewing Agent Task | "Previewing agent task." |
| Running | Show current step, elapsed time, last update when known, and no unsupported mutation actions. |
| Awaiting approval | Show requested action, risk class, expiration, and approve/deny controls when authorized. |
| Completed no useful match | "Run completed: no useful match." Link to run summary. |
| Finding created | "Finding created." Link to result and sources. |
| Approval recorded | "Approval recorded." Link to run/audit event. |

Errors should include typed cause, affected resource, recovery action, and whether retry is safe.

| Error type | Example UX message |
| --- | --- |
| Capability unavailable | "Recurring Question scheduling is not available on this server. RAG search is available, but scheduled RAG runs are not enabled." |
| Permission required | "This agent task needs approval before it can use external tools." |
| Provider failure | "The selected provider did not respond. The run can be retried." |
| Source unavailable | "The saved search scope could not be resolved. Update the task scope before the next run." |
| Rate limited | "The provider rate limit was reached. The next retry is scheduled for 2:30 PM." |
| Validation error | "The schedule is missing a timezone." |

## UX Copy Recommendations

| Avoid | Use |
| --- | --- |
| `AI automation` | `Scheduled task` or the specific family name. |
| `Magic search` | `Run this question on a schedule across selected searchable content.` |
| `Agent will handle it` | `Send this message to the selected agent at the scheduled time.` |
| `Connected apps` when not specific | `Allowed tool classes` or `Permission scope`. |
| `Failed` alone | `Failed: provider timeout`, `Failed: permission denied`, or another typed reason. |
| `No result` alone | `Run completed: no useful match`. |

Recommended planned-shell strings:

- `Planned automation type`
- `Requires scheduled RAG query support`
- `Requires schedulable ACP/API agents`
- `Results will appear in task history and any destinations selected by the task visibility policy.`
- `Preview is required before scheduling an agent task. Some permission classes may require approval before each run.`
- `No scheduled task has been created yet.`

## Accessibility And Usability Requirements

- Status badges must include text, not color alone.
- Disabled template actions must explain why and point to requirements.
- Planned cards and future forms must be keyboard navigable.
- Icon-only controls need accessible names and tooltips.
- Run histories and result lists need semantic headings and stable focus after filtering.
- Approval actions must be reachable without hover and confirm irreversible or external-side-effect decisions.
- Timestamps must include timezone or expose it nearby.
- Result and source links must have meaningful labels, not bare IDs only.
- Dense power-user tables must support small screens with horizontal overflow or responsive detail rows.
- Capability loading and preview states should use polite live regions.
- Focus should move to the first invalid field after preview/create validation errors.

## Browser Extension Expectations

The browser extension can use the same API-first contract and shared WebUI route, but should not become a separate source of truth.

- The extension may open `/scheduled-tasks?tab=create&template=recurring_question` or `template=agent_task`.
- It should show planned/API-unavailable states exactly like the WebUI if executable APIs are absent.
- It must not infer private page data into a recurring question or agent message without visible user review.
- It should not silently attach page context, credentials, tokens, fragments, or hidden metadata to Agent Task messages.
- Extension-sized panels must keep requirements, disabled reasons, and deep links readable.

## Backend Dependencies

These are dependencies for later executable slices, not Slice 4A requirements:

- Automation family capability discovery for `recurring_question` and `agent_task`.
- Preview endpoints or equivalent validation actions for both families.
- Family-aware automation definition schema.
- Normalized run history with typed status and failure reasons.
- Normalized result model with visibility policy and result state.
- Approval queue/API for Agent Task risk classes.
- Audit/events API for enterprise clients.
- Notification/result routing with dedupe keys and exact links.
- RBAC/ownership model for definitions, runs, results, approvals, and exports.
- Pagination/filtering for tasks, runs, results, approvals, and audit events.
- Idempotency keys for create/update/run actions.
- Backend ownership decision for executable scheduling: Jobs-backed user-visible runs by default, or an explicit approved Scheduler exception.

## Risks And Mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| WebUI becomes source of truth | API and enterprise clients drift. | Make capability discovery and API contracts explicit dependencies; use WebUI only as reference client. |
| Existing ACP schedule endpoint is mistaken for full Agent Task readiness | Users may schedule unsafe or uninspectable agent work. | Treat `/acp/schedules` as an implementation dependency until preview, approval, result, and audit contracts exist. |
| Jobs/Scheduler ownership is ambiguous | Executable work may conflict with ADR-003 and ops expectations. | Require explicit backend ownership decision before 4B/4D implementation. |
| Home/results become noisy | Every-run summaries can overwhelm users. | Every run appears in run history; Home/results are routed by visibility policy. |
| Agent tasks create unsafe unattended automation | Users may not trust scheduled agent execution. | Require preview, risk class, approval policy, and audit events. |
| RAG provenance is weak | Recurring answers without sources are low-trust. | Require scope, query/question version, source links/IDs, retrieval summary, and no-match rationale. |
| Watchlists boundary blurs | Existing Watchlists persona/job is harmed. | Keep Watch/Ingest separate and preserve Watchlists as deep source-monitoring workspace. |
| Hardcoded capability copy ages badly | UI claims support that the server does not provide. | Drive availability from API capability discovery when available; otherwise show planned/API-unavailable states. |

## Recommended Delivery Slices

| Slice | Outcome | Backend dependency posture |
| --- | --- | --- |
| 4A API-first product contract + WebUI shell | This spec plus improved planned Recurring Question and Agent Task template panels. | No executable backend dependency. Can use current planned template IDs. |
| 4B Backend/API foundations | Capability discovery, preview shape, family-aware definitions, run/result/approval/audit schema alignment. | Backend design/implementation. No full scheduler execution required unless scoped. |
| 4C Recurring Question execution | Create/edit recurring question tasks; schedule RAG/search runs; record every run summary; route results by policy. | Requires RAG capability composition and normalized run/result APIs. |
| 4D Agent Task execution | Create/edit agent task schedules; preview risk; dispatch messages; approval queue; run transcript/output/audit trail. | Requires ACP/API agent integration, approval policy, and backend ownership decision. |

## Slice 4A Acceptance Criteria

Before Slice 4A is complete:

- Recurring Question and Agent Task are shown as equal planned automation families.
- Planned panels explain the API-first nature of the contract.
- Planned panels list required capabilities and result destinations.
- Planned panels follow the capability fallback order and do not treat related RAG/ACP readiness as executable Scheduled Tasks support.
- Agent Task copy states preview and approval expectations.
- Recurring Question copy states every-run history and configurable surfacing expectations.
- Results tab and Home copy, if touched, explain future result locations without adding fake items.
- No local drafts, server drafts, fake tasks, fake runs, or fake results are introduced.
- Existing Reminder, Watch/Ingest, Results Inbox, Home surfacing, and Watchlists behavior are unchanged.
- The implementation does not reuse Watch/Ingest availability gates for these two templates.
- Disabled/planned states are understandable without color and include deep links to RAG/ACP-related surfaces.

## Before Executable Phase 4 Work Starts

- Confirm the backend ownership model for user-visible Recurring Question and Agent Task runs under ADR-003.
- Decide how `/acp/schedules` should be normalized, wrapped, or migrated into the Scheduled Tasks contract.
- Decide whether Recurring Question definitions live under Scheduled Tasks, RAG, Jobs, or another domain with a normalized projection.
- Confirm result/run retention and audit retention requirements.
- Confirm approval policy taxonomy and risk classes for Agent Task.
- Confirm whether result visibility policy is global-default plus per-task override, or per-task only.
- Confirm RBAC and enterprise tenant ownership rules for definitions, runs, results, approvals, and exports.

## Open Product Questions

| Question | Why it matters |
| --- | --- |
| Should Recurring Question include a "mark solved" action in the first executable slice or later? | It affects lifecycle semantics and user expectations for recurring research tasks. |
| Which RAG scopes are safe for first launch: all library, collections, tags, saved searches, or only a simple scope? | It determines the minimum preview and provenance contract. |
| Which Agent Task risk classes can execute automatically in the first executable slice? | It determines whether Agent Task is useful without creating unsafe automation. |
| Should `/acp/schedules` remain a separate expert API after unified Scheduled Tasks integration? | It affects backwards compatibility and enterprise API expectations. |
| What is the default visibility policy for new Recurring Question tasks? | It controls Home/results noise and first-time user trust. |

## Verification Plan For Future Implementation

This document is a design spec. Future implementation plans should include:

- Unit tests for template state/capability mapping and copy helpers.
- Component tests for planned panels, disabled states, deep links, and accessibility labels.
- Service contract tests for capability discovery and preview response parsing.
- API tests for definitions, previews, runs, results, approvals, idempotency, pagination, filtering, and RBAC when backend slices start.
- RAG tests for source/scope provenance and no-useful-match summaries.
- Agent tests for risk classes, approval expiration, permission changes, and partial output.
- Browser/extension checks for `/scheduled-tasks?tab=create&template=recurring_question`, `/scheduled-tasks?tab=create&template=agent_task`, `/scheduled-tasks/results`, and Home Automation Inbox copy.
- Bandit on touched backend/Python paths when executable backend slices start.

## Spec Review

Reviewed locally against the `spec-document-reviewer` rubric on 2026-06-09.

Status: Approved.

Blocking issues: None.

Issue found and addressed:

- Slice 4A capability detection needed a fallback order so implementers would not hardcode WebUI-only availability or treat related RAG/ACP readiness as executable Scheduled Tasks support. Added `Slice 4A Capability Fallback`.

Subagent note:

- The brainstorming skill normally dispatches a spec-document-reviewer subagent. In this environment, the available subagent tool is restricted to cases where the user explicitly requests delegation, so the same review rubric was applied locally instead.
