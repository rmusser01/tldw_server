# Scheduled Tasks Phase 4D Agent Task Execution Design

Date: 2026-08-24
Status: Approved
Owner: Codex brainstorming session
Backlog: TASK-13126

Related:

- `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md`
- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md`
- `Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md`
- `Docs/ADR/003-jobs-vs-scheduler-default.md`
- `backlog/tasks/task-13126 - Design-Scheduled-Tasks-Phase-4D-Agent-Task-execution.md`
- `backlog/tasks/task-13127 - Fix-Agent-Task-Jobs-consumer-missing-definition-crash.md`

## Summary

Phase 4D makes `agent_task` an executable Scheduled Tasks automation family.

The user promise is:

> Send this message to this agent at the requested time, under authority I can inspect, and show me what happened without hiding uncertainty or repeating unsafe effects.

Scheduled Tasks owns the definition, immutable revisions, schedule, authorization, runs, attention state, result summaries, delivery, and audit. A provider-neutral execution adapter owns the detailed agent execution record. ACP is the first adapter. Full transcripts and artifacts remain in the adapter's permissioned storage and are linked from Scheduled Tasks.

The product remains API-first. The WebUI is the reference and main enterprise client, and the browser extension is a context-capture and compact-updates client. Neither client defines the product boundary.

Phase 4D does not replace the standalone Agent Tasks project workflow or Watchlists. Watchlists remains the source-monitoring and change-detection workspace for its existing persona and job. Standalone Agent Tasks remains the project-oriented planning, dependency, review, and artifact workflow.

## Current Evidence And Constraints

Repository evidence that shapes this design:

- Phase 4B persists `recurring_question` and `agent_task` definitions, previews, lifecycle mutations, duplicate/archive behavior, audit, capability reports, schedule policy, and redacted Agent Task messages.
- Phase 4B create and update mutations consume persisted, expiring previews rather than accepting a second configuration payload.
- Phase 4B explicitly prohibits raw Agent Task messages in definition input and ordinary list, detail, preview, and audit responses.
- Phase 4C adds normalized Recurring Question runs and results, Results/Home surfacing, and Jobs-backed execution.
- Current `origin/dev` includes APScheduler-to-Jobs automation dispatch, durable run and notification handling, Run Now, and a production Recurring Question executor.
- Agent Task production execution remains unwired because current previews redact the raw message and no executable prompt survives persistence.
- Current live ACP permission requests are memory-resident and WebSocket-dependent. They are not a durable unattended approval and resume model.
- Existing ACP agent discovery exposes configured agents, readiness, compatibility, and support data.
- Existing standalone Agent Tasks owns projects, dependencies, review, run diagnostics, artifacts, audit, and ACP session drill-through.
- Legacy `/api/v1/acp/schedules` stores raw ACP configuration, including prompt text, and has a cron-oriented contract without the normalized Scheduled Tasks preview, authorization, result, and audit model.
- The current reference Agent Task editor accepts an agent reference, message, comma-separated tool classes, and approval mode. It does not yet provide capability-driven target selection or bounded authority review.
- The existing Scheduled Tasks reference client has Overview, Results, Tasks, and Create surfaces plus a stable `/scheduled-tasks/results` route.
- Phase 3 Home surfacing uses a dedicated Automation Inbox module and preserves Watchlists ownership.
- `Docs/ADR/003-jobs-vs-scheduler-default.md` makes Jobs the default for new user-visible work that needs retries, pause/resume, quotas, RLS, status, and operations visibility.
- Focused baseline verification passed 71 of 72 tests. `test_missing_definition_skips` fails deterministically because run creation enforces definition existence before the consumer's missing-definition branch. TASK-13127 owns that prerequisite fix.

This design does not assume that secure payload storage, durable Agent Task approvals, pre-action tool mediation, cancellation confirmation, or production Agent Task execution currently works. Each is an explicit dependency and capability gate.

## Product Decisions

1. Scheduled Tasks owns Agent Task definitions and runs. It dispatches directly to an execution adapter and does not create standalone Agent Tasks project records.
2. ACP is the first execution adapter, but definitions use a provider-neutral stable `agent_target_ref`.
3. Raw messages are stored only in tenant-scoped encrypted payload storage and referenced by opaque IDs.
4. Normal definition, preview, audit, result, notification, and list responses never return raw messages.
5. Authorized owners may explicitly reveal a message through a separately permissioned, audited, no-store operation.
6. Authority is bounded at save time by target, identity, tools, workspace, paths, network, credentials, runtime, cost, policy version, and definition revision.
7. Actions inside the authorized envelope may run unattended. An unexpected or broader action stops before execution and creates a durable approval item.
8. The original agent session is not kept waiting for an unattended approval. Approval recovery creates a linked attempt, optionally from an adapter-provided idempotent checkpoint.
9. Material target, prompt, credential, workspace, policy, or authority drift fails closed and requires a new preview and authorization.
10. Automatic retries are allowed before adapter execution starts. After session start, retry requires proof of no effects or an idempotent checkpoint.
11. Cancellation is confirmed-state only. Unconfirmed termination becomes an unresolved cancellation with unknown effect handling.
12. Scheduled Tasks retains redacted summaries, typed effect evidence, and permission-checked adapter links. Full output remains in adapter-owned secure storage.
13. `record_policy=noteworthy_only` creates Agent Results for meaningful outputs and confirmed external actions. `every_run` uses the same behavior and creates one typed run-summary Result only when a terminal run produced no output/action Result, ensuring each terminal run has at least one Result without duplicating it. `history_only` keeps ordinary output in run history without Agent Results. Safety and recovery conditions always create approval or attention resources; Home projection remains governed by surfacing policy.
14. Home shows unread outputs and unresolved attention according to policy, not every execution.
15. Legacy ACP schedules migrate automatically only after the deployment class passes execution certification and cutover gates. Safe schedules then continue under an immutable read-only migration grant; ambiguous or side-effect-capable schedules migrate paused with `review_required` in `attention_states[]` and present as `Needs review`. Uncertified deployments remain in visible inventory/dry-run state and never silently transfer execution ownership.
16. Phase 4D is delivered in stages, but write-capable execution and durable approval recovery are required before general availability.
17. Watchlists and standalone Agent Tasks retain their current ownership and workflows.
18. Every scheduled agent execution runs inside an attested, deny-by-default isolation profile. An authority envelope is not considered enforceable when the agent can bypass mediation through host filesystem, network, subprocess, MCP, or ambient credentials.
19. Scheduled execution uses a transcript mode that does not copy the raw prompt into ordinary ACP transcript fields or expose it through normal ACP detail, fork, export, or bootstrap APIs.
20. Each execution attempt is dispatched through a transactional outbox. Generic Jobs retry or lease recovery never redispatches an effect-capable attempt after adapter start.

## Goals

- Let API clients create, inspect, authorize, schedule, run, pause, resume, duplicate, archive, and debug Agent Tasks.
- Let a first-time user understand which agent will run, what it may do, when it will run, and where outputs appear.
- Let a power user inspect many definitions, runs, approvals, effects, failures, and delivery outcomes efficiently.
- Prevent stale authority, prompt leakage, duplicate side effects, silent policy drift, and false cancellation claims.
- Make execution identity, authority, revision, effect evidence, and adapter provenance inspectable.
- Automatically move legacy ACP schedules into one canonical control plane without dual dispatch.
- Preserve additive API evolution where feasible and advertise all capability and schema changes.

## Non-Goals And Deferrals

- No generic non-ACP execution adapter in the first implementation. The contract must support one later without definition migration.
- No replacement or reduction of Watchlists.
- No automatic materialization into standalone Agent Tasks projects.
- No live WebSocket-held approval for unattended runs.
- No unrestricted autonomous tool execution.
- No parallel Agent Task execution by default.
- No editable raw JSON mode in the first reference-client release. A read-only API payload view is sufficient.
- No bulk authorization or bulk authority expansion.
- No claim that deleting Scheduled Tasks data undoes external effects or erases adapter/backups outside the declared deletion scope.
- No model-based decision about whether output is meaningful. Typed adapter evidence determines output presence.
- No unsandboxed or agent-side ACP process is treated as read-only merely because tool permissions are empty.
- No visible use of `Agent Tasks` for this automation family. The API family remains `agent_task`; the reference-client label is `Agent automation` or `Scheduled agent run` so the existing Agent Tasks workspace retains its product name.

## Terms

| Term | Meaning |
| --- | --- |
| Agent Task | API family name, `agent_task`, for a Scheduled Tasks definition that sends an encrypted message to an agent target under bounded authority. Reference clients call it an Agent automation or Scheduled agent run. |
| Agent target | Provider-neutral, permission-filtered reference to an executable agent and adapter. |
| Definition | Stable Scheduled Tasks identity and current lifecycle. |
| Definition revision | Immutable configuration snapshot used by a run and authorization grant. |
| Secure payload | Tenant-scoped encrypted message referenced by opaque ID. |
| Authorization grant | Authority bound to one revision, execution identity, target, credentials, and policy fingerprint. |
| Migrated read-only grant | Non-user grant created during legacy migration that permits only adapter-enforced no-side-effect execution. |
| Run | One durable execution attempt. Linked attempts share a root run and schedule slot. |
| Approval item | Durable record of a proposed action that exceeded current authority before it executed. |
| Adapter record | Permissioned transcript, artifact, diagnostic, and execution record owned by the adapter. |
| Effect evidence | Adapter or mediator evidence that execution did not start, produced no effects, produced effects, or remains unknown. |
| Inbox projection | Read-only normalized projection of canonical Results and attention resources for the Home Automation Inbox. |
| Attention record | Non-approvable canonical incident such as unknown effects, unresolved cancellation, policy drift, grant expiry, or delivery failure. |
| Dispatch intent | Transactional Scheduled Tasks outbox record used to materialize one idempotent Jobs attempt. |

## Product And System Boundaries

| System | Responsibility |
| --- | --- |
| Scheduled Tasks | Definition, revisions, schedule, secure payload reference, authorization, run projection, attention, result summary, delivery, audit. |
| Jobs | Durable queue, lease, worker execution lifecycle, retry bookkeeping, cancellation request, reconciliation, and operations visibility. |
| APScheduler service | Cadence calculation and idempotent enqueueing of due schedule slots. |
| Execution adapter | Target discovery, isolated session creation, pre-action mediation, detailed secure execution record, checkpoint and cancellation evidence. |
| ACP | First execution adapter and owner of scheduled-mode secure transcripts, artifacts, and diagnostics. Ordinary ACP transcript behavior is not sufficient for Phase 4D. |
| Standalone Agent Tasks | Project planning, task dependencies, project review, and project-owned executions. |
| Watchlists | Continuous source monitoring, source tuning, change detection, ingest, reports, and watchlist-specific operations. |
| WebUI | Full reference and main enterprise client over the public API. |
| Browser extension | Context-aware draft creation and result/attention access through the same API. |
| Home | Bounded Automation Inbox projection, not task administration or exhaustive history. |

Jobs and Scheduled Tasks must not become competing execution authorities. Jobs owns worker lifecycle. Scheduled Tasks owns the user-facing definition, run, result, and audit projection linked by stable IDs and reconciliation.

## Core Domain Model

### Definition And Revision

An Agent Task definition contains or references:

- stable ID, owner, tenant, family, name, and description;
- lifecycle and separate authorization state;
- current immutable revision number;
- `agent_target_ref` and adapter type;
- target identity and capability fingerprints;
- `prompt_ref`, tenant-keyed HMAC fingerprint, prompt version, encryption key version, and redacted preview;
- execution context references, such as workspace or collection references;
- schedule, timezone, DST, missed-run, overlap, and retry policy;
- requested and denied tool classes;
- workspace, canonical path, network, data, runtime, and cost boundaries;
- credential references and execution identity;
- record, Home, and notification policies;
- current authorization state and grant expiry;
- retention policy.

Raw message text is never stored inline in the definition or revision.

Every run records the exact revision, target fingerprint, authorization grant, execution principal, and adapter capability snapshot used.

### Secure Payload

Secure payload storage must support:

- tenant and owner isolation;
- authenticated encryption at rest;
- separate encryption and fingerprint keys with key versioning;
- tenant-keyed HMAC fingerprints rather than plain hashes;
- provisional, active, superseded, and deleted states;
- quotas and maximum message size;
- idempotent provisional creation;
- key rotation without plaintext exposure to ordinary APIs;
- cryptographic deletion where the storage model supports it;
- deterministic expiry cleanup for abandoned previews;
- no prompt content in logs, metric dimensions, audit payloads, errors, notifications, or analytics.

Archive retains encrypted payloads under retention policy. `Forget message` revokes grants and deletes eligible Scheduled Tasks payload versions, then reports what may remain in adapter records, transaction logs, backups, or external systems.

If secure payload storage and Scheduled Tasks metadata are not transactionally co-located, preview uses a crash-safe provisional saga:

1. Create an idempotent provisional ciphertext with a correlation ID.
2. Persist the preview and payload reference as pending.
3. Mark both usable only after each durable write is confirmed.
4. Return no valid preview until ciphertext durability is established.
5. Reconcile or garbage-collect orphan ciphertext and pending previews by correlation ID.

`Forget message` first commits an immediate logical security transition: pause future execution, revoke grants and one-run exceptions, supersede queued attempts and approvals, and request cancellation of active sessions. Physical deletion may finish asynchronously as `deletion_pending`. The response reports each payload and adapter location as deleted, pending, retained by policy, or outside managed scope.

### Agent Target

Target discovery returns only targets the caller may discover. Each target includes:

- stable provider-neutral reference;
- display name and adapter type;
- adapter instance and target identity fingerprint;
- configured, degraded, unavailable, or blocked readiness;
- read-only and tool capabilities;
- pre-action mediation, checkpoint, idempotency, and cancellation capabilities;
- credential readiness without returning credential material;
- attested isolation profile and its fingerprint;
- capability evidence source, `observed_at`, and `expires_at`;
- user-safe reason and recovery action.

Unavailable targets remain discoverable under a secondary disclosure when the user may see them. A configured target remains visible on task detail even when it later becomes unavailable.

Every scheduled execution, including generation-only execution, requires an attested deny-by-default isolation profile. The profile must enforce:

- no ambient credentials or inherited secrets;
- no host filesystem access except explicit minimal mounts;
- no subprocess or host runner escape;
- deny-by-default network egress;
- no direct MCP or tool access outside the mediator;
- brokered, per-action credentials rather than process environment credentials;
- bounded CPU, memory, time, output, and cost where supported.

Tool-enabled execution additionally requires pre-action mediation for every effect. Host-runner, `sandbox=none`, or agent-side targets that cannot prove these properties are ineligible for scheduled execution. A descriptive `read_only` label is not evidence.

Isolation attestation is accepted only when it is server-verified against a configured trust root for an approved isolation controller. The signed evidence binds runtime and image digest, mount policy, egress policy, credential-broker policy, tenant, workspace, target, issue and expiry times, signer and key ID, and the isolation profile fingerprint used by the grant and dispatch token. Admission rejects missing, self-asserted, forged, expired, not-yet-valid, revoked-signer, wrong-tenant, wrong-workspace, or fingerprint-mismatched evidence. Trust-root and signer revocation are live deny inputs and are rechecked before each mediated action and credential issuance.

### Authorization Grant

A grant records:

- authorizing and execution principals;
- act-as or delegation grant ID, version, delegated scope, and expiry when principals differ;
- tenant and credential references and credential owner/version;
- credential-use grant ID, version, scope, and expiry when the execution subject does not own the credential;
- target, adapter, capability, and policy fingerprints;
- permitted tool classes and deny rules, with deny precedence;
- canonical workspace and path boundaries plus symlink policy;
- normalized network destination boundaries;
- data, runtime, cost, and time limits;
- definition revision and prompt fingerprint;
- authorization reason, expiry, and audit correlation IDs;
- `active`, `revoked_immediately`, `expired`, `drifted`, or `superseded_for_new_dispatch` state.

Grant APIs serialize this as `grant_state` plus a typed `grant_state_reason` and derived `active_dispatch_valid`. `active` is valid for matching dispatch; `superseded_for_new_dispatch` is invalid for new dispatch but may remain valid for an already started matching token; `revoked_immediately`, `expired`, and `drifted` are invalid for both new dispatch and later mediated actions. State never returns to `active`; reauthorization creates a new grant. Compare-and-swap transition tests cover active to each terminal state and superseded to immediate revocation or expiry.

Workers execute only as the recorded execution principal. They never inherit worker-service authority.

Scheduled authority is the intersection of all current controls. An action is permitted only when an explicit exact authorization grant or exact unconsumed one-run exception allows it, the attested isolation profile and mediator allow it, every required delegation and credential-use grant remains active and in scope, and no live deny policy, principal suspension, credential revocation, signer revocation, or administrator kill switch blocks it. Missing, stale, ambiguous, or unavailable evidence denies the action or creates a durable pre-action approval when the action is eligible for approval. ACP remembered allows, session or batch approvals, model-selected approval tiers, and adapter-local defaults may narrow authority but can never expand Scheduled Tasks authority or bypass its exact action and argument checks.

The following independent permissions are normative endpoint requirements:

- `TASKS_READ`
- `TASKS_CONTROL`
- `TASKS_AUTHORIZE`
- `TASKS_APPROVE`
- `TASKS_PROMPT_REVEAL`
- `TASKS_PROMPT_COPY`
- `TASKS_PROMPT_DELETE`
- `TASKS_SECURE_PAYLOAD_CLONE`
- `TASKS_SECURE_OUTPUT_READ`

Deployments may assign several permissions to one role, but they must not alias or omit the permission checks. `TASKS_CONTROL` alone cannot authorize, approve, reveal, copy plaintext, clone or delete secure payloads, or read secure transcripts.

Permission matrix:

| Action | Required permission |
| --- | --- |
| List capabilities, targets, and caller-visible collections | `TASKS_READ`; collection responses remain permission-filtered. |
| Detail definitions, revisions, redacted runs, redacted Results, Inbox items, approvals, attention, and audit | `TASKS_READ` plus source resource access. |
| Preview, draft create/update, pause, archive, restore, cancel, ordinary safe retry, Run Now with an existing grant, review-state mutation, bulk preview, and supported bulk action | `TASKS_CONTROL` plus source resource access. |
| Create-and-authorize, update-and-reauthorize, renew, expand authority | `TASKS_AUTHORIZE` plus `TASKS_CONTROL`. |
| Authorize-and-run or acknowledge possible duplicates and start a linked attempt | `TASKS_AUTHORIZE` plus `TASKS_CONTROL`, exact source access, and configured step-up. |
| Resolve an approval | `TASKS_APPROVE` plus source resource access. |
| Reveal the stored message for view or edit | `TASKS_PROMPT_REVEAL` plus step-up when configured. |
| Copy revealed message text | `TASKS_PROMPT_REVEAL` plus `TASKS_PROMPT_COPY` and configured step-up. |
| Forget stored message content | `TASKS_CONTROL` plus `TASKS_PROMPT_DELETE`, exact source access, and mandatory step-up. |
| Duplicate task structure | `TASKS_CONTROL` plus source resource access. |
| Clone encrypted message content server-side | `TASKS_SECURE_PAYLOAD_CLONE` in addition to duplicate-structure permission; plaintext is never returned. |
| Read or export full scheduled-mode transcript/output | `TASKS_SECURE_OUTPUT_READ`; also require `TASKS_PROMPT_REVEAL` whenever prompt echo is present or cannot be conservatively excluded. |

### Actor And Resource Relationships

| Actor/resource | Normative rule |
| --- | --- |
| Definition owner | Tenant-bound owner of definition, revisions, Results, and ordinary audit. |
| Execution subject | Defaults to the definition owner. A different subject requires explicit act-as/delegation authority and active principal state. |
| Authorizer | Must control the definition, hold `TASKS_AUTHORIZE`, and be allowed to delegate to the execution subject. |
| Credential owner | Defaults to the execution subject. Foreign credentials require an explicit, revocable credential-use grant. |
| Approver | Must hold `TASKS_APPROVE` and access the definition. Deployments may additionally require separation from the authorizer. |
| Adapter record | Stores definition owner and execution subject separately; access is the intersection of Scheduled Tasks and adapter authorization. |

Cross-tenant access is always denied. Same-tenant cross-user resources require explicit delegation. Reads use concealed `404` behavior where revealing existence would leak another user's target, payload, grant, run, approval, Result, or adapter record.

Delegation and credential-use authority are bound into the immutable authorization snapshot, dispatch intent, and dispatch token by ID, version, normalized scope, and expiry. Admission and every mediated action or credential issue revalidate their live state. Revocation or scope reduction prevents later actions immediately and follows the active-run revocation rules below.

### Scheduled-Mode Adapter Records

The current ordinary ACP transcript contract is not acceptable for scheduled prompts because it persists raw prompt content in plaintext and returns full transcript detail through ordinary session reads. The ACP adapter dependency must add scheduled mode with these invariants:

- transcript events reference `prompt_ref` rather than copying the user message into ordinary transcript fields;
- transcript and output content are encrypted or stored in an equivalently protected tenant-scoped store;
- ordinary ACP detail, fork, export, bootstrap, audit, and search paths return redacted prompt sentinels;
- secure transcript reads require `TASKS_SECURE_OUTPUT_READ`, and any original-prompt disclosure additionally requires `TASKS_PROMPT_REVEAL` and configured step-up;
- full transcript or output reads require both `TASKS_SECURE_OUTPUT_READ` and `TASKS_PROMPT_REVEAL` whenever the representation can contain prompt echoes, including quoted, encoded, transformed, or tool-argument copies;
- an assistant-only output view may omit `TASKS_PROMPT_REVEAL` only after a conservative prompt-echo redaction gate succeeds; uncertainty blocks the view and requires the combined secure-output and prompt-reveal path;
- `Forget message` tombstones or deletes the adapter's prompt reference and reports possible prompt echoes in generated output separately;
- scheduled adapter logs, errors, and backups follow the same deletion-scope reporting rules.

### Run

A run stores:

- definition and immutable revision;
- root run, parent run, attempt number, and schedule slot;
- `coalesced_occurrence_count`, `first_coalesced_at`, and `last_coalesced_at` when later schedule slots are folded into this invocation or approval;
- trigger reason and Jobs ID;
- execution, result, and delivery states;
- adapter execution reference and capability fingerprint;
- authorization grant and execution principal;
- effect status, evidence source, and evidence timestamp;
- retry mode and action required;
- redacted execution and output summary;
- linked approval, Result, delivery, artifact, transcript, and diagnostic references;
- typed failure and correlation information;
- timestamps for queue, admission, dispatch, start, cancellation, and end.

Run identity is enforceable:

- one root invocation key is unique for definition and occurrence slot, independent of mutable revision;
- manual invocations use the caller's idempotency key as their occurrence identity;
- attempt number is unique within the root invocation;
- each attempt has one distinct Jobs idempotency key and dispatch token;
- each attempt has one monotonically increasing Scheduled Tasks `execution_fence` for canonical ownership;
- approval, Result, attention, and delivery identities are unique by canonical source and version;
- a revision change cannot create a second root invocation for the same scheduled slot;
- every state mutation uses record version and `execution_fence` compare-and-swap.

Effect evidence is append-only and monotonic. Stale callbacks cannot downgrade `effects_confirmed` or replace `unknown` with `none_confirmed` unless the adapter supplies stronger evidence that is valid for the exact dispatch token. `execution_uncertain` means ordinary dispatch, lease, session, or terminal-state reconciliation cannot prove how execution ended. `cancellation_uncertain` is the narrower case where a cancellation was requested but termination was not confirmed. Either may later reconcile to succeeded, failed, timed out, approval required, or cancelled from exact-token evidence; effect-only evidence may strengthen `effect_status` while leaving execution resolution uncertain.

### Durable Dispatch Protocol

Cross-store run, Jobs, and adapter handoff uses transactional outboxes rather than nested best-effort writes:

1. In one Scheduled Tasks transaction, create the root invocation or linked attempt, immutable authority snapshot, audit event, and `dispatch_intent`.
2. An outbox publisher idempotently creates one Job and records its Jobs ID through the durable Jobs idempotency ledger defined below.
3. The Job payload carries `run_id`, root and attempt IDs, revision, grant, execution subject, target and isolation fingerprints, opaque payload reference, schedule slot, scheduler generation, dispatch token, and expected execution fence.
4. On worker claim, Scheduled Tasks binds the current execution fence to the exact Jobs UUID and lease ID. Before calling the adapter, persist the stable dispatch token, bound lease, execution fence, and final admission result.
5. Adapter session creation must accept the dispatch token idempotently or support lookup by it after a crash.
6. Reconciliation detects orphan intents, Jobs, runs, and adapter sessions without redispatching an uncertain attempt.
7. Finalize execution state, effect evidence, Result or attention identity, audit, and delivery outbox in one Scheduled Tasks transaction before marking the Job complete.

The stable dispatch token identifies one adapter attempt and never changes. Separately, Scheduled Tasks issues a monotonically increasing per-attempt `execution_fence` that identifies the current canonical state owner. A reconciliation takeover or other ownership replacement increments the fence transactionally before work begins. Every worker or reconciler state write requires the run record version, current fence, Jobs UUID, and bound lease identity. A stale owner may submit adapter evidence correlated by the stable dispatch token, but it cannot mutate canonical state; the current reconciler validates and applies that evidence under the current fence.

Jobs enqueue idempotency survives completion, archival, and supported purge. A durable ledger or tombstone, unique by attempt identity, stores the Jobs idempotency key, attempt ID, dispatch token, payload digest, original Jobs ID, and terminal/archive reference. Enqueue replay returns or verifies the same logical Job across active, archived, and purged storage; a digest mismatch fails closed. The ledger is retained for at least the maximum run/outbox reconciliation horizon and cannot be purged while any related dispatch intent, run, adapter session, or reconciliation remains unresolved.

An effect-capable Agent Task Job represents exactly one attempt and uses `max_retries=0`. Only the Scheduled Tasks retry controller can persist a new linked attempt after evaluating durable effect or checkpoint evidence. Workers renew leases through the Jobs worker contract. Lease loss after adapter dispatch enters `execution_uncertain` reconciliation and unknown-effect handling; Jobs must not automatically reacquire and rerun that attempt.

Jobs cancellation is a request signal only. It cannot directly set the canonical run to `cancelled` after adapter dispatch.

### Approval Item

An approval item is valid only when pre-action mediation proves the proposed action did not execute. It stores a redacted action summary, requested authority delta, source run and revision, safe checkpoint information when available, expiry, review state, action state, resolution audit, and canonical `coalesced_occurrence_count`, `first_coalesced_at`, and `last_coalesced_at` fields.

If an action may already have occurred, the system creates an unknown-effect attention record rather than an approval request.

## API Contract

All canonical endpoints are under `/api/v1/scheduled-tasks`.

### Existing Endpoints Retained

- `GET /scheduled-tasks`
- `GET /scheduled-tasks/capabilities`
- preview list, create, and detail
- definition list, create, detail, and update
- pause, resume, archive, duplicate, Run Now, revisions through versioning, and audit
- definition runs and run detail
- result list, detail, and review

Existing query-based WebUI deep links and `/scheduled-tasks/results` remain valid during client route migration.

### Additive Agent Task Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /scheduled-tasks/agent-targets` | Discover permission-filtered targets and capability freshness. |
| `GET /scheduled-tasks/definitions/{definition_id}/revisions` | Inspect redacted immutable revision history. |
| `POST /scheduled-tasks/definitions/{definition_id}/authorizations` | Consume a valid preview for update-and-authorize or authorize-and-run, creating the matching grant and lifecycle or one-run dispatch atomically. |
| `POST /scheduled-tasks/definitions/{definition_id}/prompt-reveal` | Step-up-aware, audited, no-store prompt reveal. |
| `POST /scheduled-tasks/definitions/{definition_id}/forget-message` | Revoke grants and delete eligible Scheduled Tasks payloads with a deletion-scope report. |
| `POST /scheduled-tasks/definitions/{definition_id}/restore` | Restore an archived definition into paused state without restoring authority. |
| `POST /scheduled-tasks/runs/{run_id}/cancel` | Request cancellation without claiming termination. |
| `POST /scheduled-tasks/runs/{run_id}/retry` | Create a linked attempt only under the reported retry mode. |
| `GET /scheduled-tasks/approvals` | List durable pre-action approvals only. |
| `GET /scheduled-tasks/approvals/{approval_id}` | Inspect one approval and its revision/checkpoint context. |
| `POST /scheduled-tasks/approvals/{approval_id}/resolve` | Deny, approve a safe retry, or begin a future-policy update. |
| `POST /scheduled-tasks/approvals/{approval_id}/review` | Mark an approval read or snoozed without resolving it. |
| `GET /scheduled-tasks/attention` | List non-approvable policy, effect, cancellation, expiry, and delivery attention records. |
| `GET /scheduled-tasks/attention/{attention_id}` | Inspect one permission-checked attention record and its evidence and recovery links. |
| `POST /scheduled-tasks/attention/{attention_id}/review` | Mark an attention record read or snoozed with version preconditions. |
| `GET /scheduled-tasks/agent-results` | List versioned Agent Task output, action, and run-summary Results without changing Phase 4C Result enums. |
| `GET /scheduled-tasks/agent-results/{agent_result_id}` | Inspect one canonical Agent Task Result. |
| `POST /scheduled-tasks/agent-results/{agent_result_id}/review` | Mutate Agent Task Result review state. |
| `GET /scheduled-tasks/inbox` | Read the additive normalized Home Automation Inbox projection without breaking canonical Result or attention schemas. |
| `GET /scheduled-tasks/inbox/{inbox_item_id}` | Read one projected item and its canonical action links. |
| `POST /scheduled-tasks/bulk-previews` | Preview eligible, blocked, and consequential definition mutations. |
| `POST /scheduled-tasks/bulk-actions` | Consume a valid bulk preview for supported pause/archive operations. |

The additive Agent Results and Inbox routes avoid expanding closed existing Result enums in ways that can break generated clients. Existing `/results` resources remain canonical Recurring Question records. `/agent-results/{agent_result_id}` never returns an Inbox item, and `/inbox/{inbox_item_id}` never masquerades as a canonical Result. Resource-specific path parameter names are normative because they become generated-client parameter names. The Inbox union projects legacy Results, Agent Results, approvals, policy blocks, unknown effects, expiry warnings, and delivery failures for Home. The reference-client Results route reads canonical legacy and Agent Results; it does not list attention-only Inbox items.

### Preview And Save

Preview remains mandatory before definition create or update.

1. `POST /scheduled-tasks/previews` accepts the proposed message and configuration.
2. The server validates syntax, caller access, message size, target discoverability, and secure-payload readiness before storing content.
3. The message is encrypted into a provisional payload through the co-located transaction or provisional saga defined above.
4. The preview stores only opaque reference, HMAC fingerprint, redacted representation, and normalized config.
5. The preview returns target resolution, capability freshness, authority requested, risk, material diff, next runs, warnings, and revision fingerprint.
6. Create or update consumes the preview exactly once.
7. Expired, abandoned, and invalid provisional payloads are cleaned deterministically.

Preview supports `validation_level=save|enable`:

- `save` permits syntactically valid but execution-incomplete drafts;
- `enable` requires all target, prompt, schedule, identity, credential, authority, and delivery checks.

For updates, clients use an explicit prompt operation:

- `keep`: retain the current payload without revealing or resending it;
- `replace`: encrypt a new provisional payload;
- `remove`: allowed only when the result is a non-executable draft.

### Create, Update, And Authorize

Agent Task mutations are five mutually exclusive typed transactions:

| Transaction | Endpoint and operation | Atomic outcome |
| --- | --- | --- |
| Draft create | `POST /definitions`, `operation=create_draft` | Consume a save-valid preview; create paused definition and revision with no grant. |
| Create and authorize | `POST /definitions`, `operation=create_and_authorize` | Consume an enable-valid preview; create definition, revision, authorization assertion, grant, and configured lifecycle. |
| Update only | `PATCH /definitions/{definition_id}`, `operation=update` | Consume one preview; apply non-material changes, or require `apply_mode=save_and_pause` for material changes and revoke/supersede authority. |
| Update and reauthorize | `POST /definitions/{definition_id}/authorizations`, `operation=update_and_authorize` | Consume one enable-valid update preview; commit revision, assertion, grant, and configured lifecycle. |
| Authorize and run once | `POST /definitions/{definition_id}/authorizations`, `operation=authorize_and_run` | Consume one enable-valid preview; atomically apply its revision, create a one-use grant, root run, and unique dispatch intent while leaving the definition paused. |

Each request includes preview ID, expected preview fingerprint, expected definition version where applicable, operation, and idempotency key. Authorization operations additionally include the exact authorization assertion. Responses identify consumed preview, resulting revision, grant, lifecycle, and next eligible run.

Old clients may continue to create paused Agent Task drafts through the legacy `initial_lifecycle=paused` shape. Any old or new request that attempts `initial_lifecycle=configured` without a matching authorization transaction fails with `scheduled_task_authorization_required`. A preview is never consumed by update and then consumed again by authorization.

Resume never bypasses missing, expired, revoked, or drifted authorization. Agent Task Run Now with an existing grant requires `TASKS_CONTROL`, mandatory `Idempotency-Key`, expected definition revision, and active grant ID and version. A caller without a matching reusable grant uses `authorize_and_run`; it requires `TASKS_CONTROL` and `TASKS_AUTHORIZE`, the same concurrency preconditions, and the exact authorization assertion. It leaves the recurring definition paused. Both paths atomically create one root run and one unique dispatch intent; idempotent replay returns that same invocation.

`authorize_and_run` either consumes a no-diff preview bound to the current revision or atomically commits the preview as the new immutable and current revision while retaining `lifecycle=paused`. The one-use grant, authority snapshot, root run, dispatch intent, and dispatch token all reference that same revision and its exact prompt, target, policy, isolation, delegation, credential, and authority fingerprints. A consumed preview is never left unapplied.

Authorization confirmation includes exact revision, authority, target, prompt, policy, and risk-summary fingerprints. A generic `confirmed=true` is insufficient.

An unknown-effect manual override is a typed retry request, not an ordinary Run Now. `POST /runs/{run_id}/retry` uses `mode=acknowledge_possible_duplicates` and requires mandatory `Idempotency-Key`, expected evidence version, duplicate-effect evidence digest, expected definition revision, expected grant ID and version, a bounded reason code, configured step-up, `TASKS_CONTROL`, and `TASKS_AUTHORIZE`. In one transaction it records the acknowledgement and creates at most one linked attempt and unique dispatch intent. Any stale evidence, revision, grant, or acknowledgement digest refuses the request.

### Prompt Reveal And Copy

Prompt reveal:

- requires `TASKS_PROMPT_REVEAL` and any configured step-up authentication;
- accepts a bounded `reason_code` plus `purpose=view|edit|copy` and creates an audit event;
- returns `Cache-Control: no-store` and no redirectable URL;
- never populates ordinary detail, browser URL, history, telemetry, or notifications;
- is displayed in a temporary client panel that remasks on close, navigation, or session expiry;
- does not copy automatically.

The reference client sends `purpose=copy` before placing revealed text on the clipboard and records success or client-reported failure in a follow-up audit event. Copy purpose additionally requires `TASKS_PROMPT_COPY`. The product does not claim it can observe later operating-system or application copies. Audit stores only the bounded reason code and purpose; no caller-supplied free text is stored in ordinary or protected audit. If a deployment needs support notes, it stores them separately under encrypted prompt-reveal access and never joins them into ordinary audit, logs, metrics, or notifications.

Server-side duplication of encrypted content requires `TASKS_SECURE_PAYLOAD_CLONE`; it creates a new opaque payload without revealing plaintext to the caller and does not grant clipboard-copy authority.

Duplicate copies permitted structure and starts paused without a reusable grant. Without `TASKS_SECURE_PAYLOAD_CLONE`, the new task requires a replacement message. A cloned secure payload receives a new opaque reference and cannot preserve an authority grant.

`Forget message` requires `TASKS_PROMPT_DELETE`, `TASKS_CONTROL`, exact source access, and mandatory step-up even when other destructive task controls do not require step-up. Deletion authority never implies reveal, plaintext copy, or encrypted clone authority.

## Preview, Risk, And Material Change

Preview returns a field-level diff and server decision:

- `reauthorization_required`;
- `reauthorization_reasons`;
- current and proposed authority fingerprints;
- target and capability drift;
- next-run changes;
- visibility and delivery changes;
- typed warnings and blockers;
- preview expiry.

Material changes include:

- target or adapter identity;
- prompt or prompt version;
- execution principal or credential ownership/version;
- tool classes, deny rules, workspace, paths, network, or data access;
- runtime or cost limits;
- policy fingerprint;
- schedule changes that materially increase frequency, time window, cost, or exposure.

Name, description, and safe presentation-only changes are non-material. Delivery changes can be material when they expand disclosure to a destination.

Risk review uses plain language and identifies exactly what can run, where, under whose identity, for how long, and with which delivery destinations. Write-capable authority requires configured step-up and explicit risk acknowledgement.

## Lifecycle And State Model

Definition lifecycle, schedule state, activity, and attention are independent canonical fields:

- `lifecycle`: `configured|paused|archived|disabled`;
- `schedule_state`: `eligible|blocked|waiting_for_approval|exhausted_completed|exhausted_missed|exhausted_cancelled|exhausted_attention`;
- `activity`: `idle|queued|checking_authorization|starting|running|cancelling`;
- `attention_states[]`: zero or more of `configuration_incomplete|authorization_required|authorization_expiry|review_required|policy_review|approval_required|execution_failed|execution_uncertain|cancellation_uncertain|delivery_failed`;
- `primary_attention_state`: a deterministic summary of the active attention set, or `none`.

Clients derive friendly labels from these fields but never persist `Draft`, `Scheduled`, `Completed`, `Missed`, `Needs attention`, or `Needs review` as lifecycle values. `Needs review` is always the presentation of active `review_required` attention. A configured task blocked by review remains configured and reports `next_run_eligibility=not_eligible_until_reviewed`; a migrated task without a valid grant is paused with the same attention state.

Definition responses expose active attention counts by class and canonical source links. `primary_attention_state` is ordered by execution/cancellation uncertainty, approval required, security or policy review, known execution failure, authorization required or expiry, delivery failure, then configuration incomplete. Resolving the primary item reveals the next; it never clears unrelated attention. Task rows, compact mobile rows, filters, and accessible names expose every active attention class and count rather than announcing only the primary state. A deprecated singular `attention_state` compatibility field, if retained, aliases `primary_attention_state` and is never the only source of truth.

### Definition Presentation

| Condition | Canonical lifecycle | Schedule state | Activity | Presentation and attention |
| --- | --- | --- | --- | --- |
| Paused and no grant | `paused` | `blocked` | `idle` | `Draft`; configuration incomplete or not authorized. |
| Configured, ready, future run | `configured` | `eligible` | `idle` | `Scheduled`; no attention. |
| Configured with active run | `configured` | `eligible` | `running` | `Scheduled` plus `Running`; attention remains independent. |
| Paused with active run | `paused` | `blocked` | `running` | `Paused`; explain that the active run continues. |
| Configured but grant or policy invalid | `configured` | `blocked` | `idle` | `Needs review`; next run is `Not eligible until reviewed`. |
| Material edit saved without authorization | `paused` | `blocked` | `idle` | `Needs authorization`. |
| Recurring definition waiting on approval | `configured` | `waiting_for_approval` | `idle` | `Approval required`; future occurrences coalesce. |
| One-time definition waiting on approval | `configured` | `waiting_for_approval` | `idle` | `Approval required`; approval starts one linked attempt, denial or expiry exhausts with attention. |
| Successful exhausted one-time schedule | `configured` | `exhausted_completed` | `idle` | `Completed`; no attention. |
| Exhausted one-time schedule skipped by misfire policy | `configured` | `exhausted_missed` | `idle` | `Missed`; explain that the time passed without execution. |
| Exhausted one-time schedule cancelled with confirmed termination | `configured` | `exhausted_cancelled` | `idle` | `Cancelled`; offer `Run again` as a new manual invocation. |
| Exhausted one-time schedule failed | `configured` | `exhausted_attention` | `idle` | `Needs attention` with `execution_failed`; show `Retry as new run` only when eligible. |
| Exhausted one-time schedule with unresolved cancellation | `configured` | `exhausted_attention` | `idle` | `Cancellation unresolved`; expose `cancellation_uncertain` and evidence recovery. |
| Exhausted one-time schedule otherwise blocked or uncertain | `configured` | `exhausted_attention` | `idle` | `Needs attention`; show the exact recovery action. |
| Admin/security lock | `disabled` | `blocked` | `idle` | `Disabled`; admin action may be required. |
| Archived | `archived` | `blocked` | `idle` | `Archived`; no new execution. |

The UI never lets activity such as `Running` replace lifecycle, schedule state, or attention.

### Canonical Run State

New Agent Task contracts expose open, additive state fields while retaining deterministic mappings to existing coarse `status` fields during compatibility migration.

Execution states and resolutions include:

- `queued`
- `admission_check`
- `dispatching`
- `running`
- `cancelling`
- `succeeded`
- `failed`
- `timed_out`
- `approval_required`
- `blocked_policy_drift`
- `skipped`
- `superseded`
- `cancelled`
- `execution_uncertain`
- `cancellation_uncertain`

Run responses separate:

- `execution_state` and `execution_resolution`;
- `result_state`;
- `delivery_state`;
- `attention_states[]` and `primary_attention_state`;
- `effect_status`;
- `retry_mode`.

Effect status values are:

- `not_started`: adapter execution did not begin;
- `none_confirmed`: the mediator or adapter confirms no effects;
- `effects_confirmed`: one or more effects are evidenced;
- `unknown`: available evidence cannot establish whether effects occurred.

Each effect status includes evidence source and timestamp.

Compatibility mapping to the existing coarse `status` field is deterministic:

| Canonical execution state/resolution | Existing coarse `status` |
| --- | --- |
| `queued`, `admission_check`, `dispatching` | `queued` |
| `running`, `cancelling` | `running` |
| `succeeded` | `completed` |
| `failed`, `timed_out`, `approval_required`, `blocked_policy_drift`, `execution_uncertain`, `cancellation_uncertain` | `failed` |
| `skipped`, `superseded` | `skipped` |
| `cancelled` | `cancelled` |

The canonical transition graph is:

| From | Legal next states |
| --- | --- |
| `queued` | `admission_check`, `skipped`, `superseded`, `cancelled`. |
| `admission_check` | `dispatching`, `blocked_policy_drift`, `skipped`, `superseded`, `cancelled`. |
| `dispatching` | `running`, `failed`, `timed_out`, `execution_uncertain`, `cancelling`. |
| `running` | `succeeded`, `failed`, `timed_out`, `approval_required`, `execution_uncertain`, `cancelling`. |
| `cancelling` | `cancelled`, `cancellation_uncertain`, `succeeded`, `failed`, `timed_out`, or `approval_required`; timeout or approval requires exact evidence that it preceded the cancellation boundary. |
| `execution_uncertain` | `succeeded`, `failed`, `timed_out`, `approval_required`, `cancelled`, or a terminal failed resolution that retains `effect_status=unknown`, only from exact dispatch-token reconciliation. |
| `cancellation_uncertain` | `succeeded`, `failed`, `timed_out`, `approval_required`, `cancelled`, or a terminal failed resolution that retains `effect_status=unknown`, only from exact dispatch-token reconciliation. |
| `failed` with `execution_resolution=unknown_after_reconciliation_deadline` | `succeeded`, `failed`, `timed_out`, `approval_required`, or `cancelled` only from late exact dispatch-token evidence; preserve the correction audit. |
| Other execution-ended states | No new execution transition; append-only evidence may strengthen effect evidence without rewriting the terminal execution outcome. |

Uncertain runs expose `reconciliation_state`, `reconciliation_due_at`, last evidence time, and safe evidence source. Until the deadline they remain `execution_uncertain` or `cancellation_uncertain`. At the deadline, an unresolved run becomes `failed` with `execution_resolution=unknown_after_reconciliation_deadline` and retains `effect_status=unknown` plus unresolved attention. Late exact-token evidence may make the narrowly defined correction above; effect-only evidence can strengthen effect status without claiming a terminal execution outcome. Reconciliation to `approval_required` atomically materializes or verifies the exact pre-action approval and its no-effect evidence before exposing approval controls.

The cancellation boundary is an adapter-durable, idempotent event identified by dispatch token and cancellation-request ID in the same per-attempt monotonic event sequence as terminal and pre-action approval events. Cross-store wall-clock timestamps never establish race ordering. A terminal success, failure, timeout, or pre-action approval earlier in that sequence wins and records the cancellation as too late; confirmed termination for the requested dispatch after the boundary resolves `cancelled`; absent monotonic ordering evidence remains uncertain. State changes require record version and `execution_fence` compare-and-swap. Execution-ended records are immutable except append-only reconciliation evidence and the narrowly defined uncertainty corrections. Result and delivery state can continue after execution ends without changing the execution outcome.

Reference-client run actions are explicit:

| Run condition | Primary action | Other behavior |
| --- | --- | --- |
| Queued or admission, no adapter session | `Cancel run` | Server may confirm cancellation with `effect_status=not_started`. |
| Dispatching or running | `Cancel active run` | A new attempt is unavailable while execution may still be active. |
| Cancelling | None | Show request time, adapter, and latest confirmation source. |
| Failed with `not_started` or `none_confirmed` | `Retry as new run` | Creates a linked attempt. |
| Approval required | `Review approval` | Only the approval recovery actions are offered. |
| Unknown execution or unresolved cancellation | `Review evidence` | Starting another run remains unavailable until required risk acknowledgement and policy permit it. |
| Blocked policy drift | `Review changes` | `Run now` and `Retry as new run` remain disabled. |
| Confirmed cancelled, exhausted one-time definition | `Run again` | Creates a new manual root invocation; the original slot remains cancelled. |
| Cancelled with no effects confirmed | `Retry as new run` | Creates a linked attempt. |
| Succeeded | `Run again` | This is a new manual invocation, not a retry. |
| Skipped or superseded | `Run now`, when eligible | Original slot remains unchanged. |

### Dispatch Fencing

Immediately before adapter session creation, the worker verifies:

- lifecycle permits execution;
- queued revision is still current and eligible;
- grant is active and matches the revision;
- credentials and execution principal remain valid;
- target identity and required capabilities remain valid;
- no cancellation or admin kill switch is active.

Pause, archive, grant revocation, credential revocation, or a material edit can supersede queued work before dispatch. Active runs retain their original immutable revision, but retaining provenance does not guarantee continuing authority.

Material change effects are explicit:

- prompt replacement, target change, authority expansion, and other new-revision edits mark the old grant `superseded_for_new_dispatch`; queued work is superseded, while an already active run may continue under its original dispatch token unless separately cancelled;
- authority reduction, credential or principal revocation, administrator security deny, `Forget message`, tenant kill switch, and security-policy revocation mark affected authority `revoked_immediately`, prevent later mediated actions and credentials, and request cancellation;
- pause and archive prevent new dispatch but do not revoke an active dispatch token by default; API responses and the reference client state that an active run continues and offer a separate cancellation action.

Immutable revision provenance does not freeze live authority. Before every mediated action and every credential issuance, the mediator rechecks grant revocation, credential status, execution-subject status, deny policy, and administrator kill switch. Revocation prevents all future actions immediately, revokes outstanding action tokens, and requests session cancellation. Effects already produced remain recorded; loss of confirmation enters unknown-effect reconciliation.

Owner-matched run-slot dedupe precedes missing-definition handling. When no matching run exists, a stale Job for a never-created, deleted, or concealed definition completes deterministically as `status=skipped`, `reason=definition_missing`, and `run_id=null`; it creates no definition run, Result, or notification. A redelivered terminal slot returns its recorded run with `deduped=true`, even if the definition later becomes unavailable. An owner-mismatched run from an incorrectly injected or mixed repository is concealed as `definition_missing`. The 4D.0 prerequisite records a bounded warning and Jobs result; the canonical Phase 4D path also emits one low-cardinality metric and an owner-safe Jobs or global audit event. Neither path reveals cross-owner definition existence.

## Scheduling, Overlap, Retry, And Cancellation

Supported schedule kinds remain `one_time`, `interval`, `daily`, `weekly`, and `cron` with timezone, DST, missed-run, overlap, retry, start, and end policy.

Preview always shows concrete upcoming timestamps.

Agent Task overlap defaults to `queue_one`:

- at most one pending overlapping invocation is retained;
- later overlapping slots coalesce into that pending invocation;
- the pending run remains bound to its revision;
- a material edit supersedes the unstarted pending run;
- coalesced slot counts remain visible in history.

`skip_new` remains available. `cancel_existing` is available only for adapter-enforced read-only execution or an adapter-declared idempotent boundary. Confirmed cancellation alone is insufficient because earlier effects may already exist. Parallel execution remains unavailable unless an administrator enables it and the adapter advertises suitable isolation or idempotency.

While an approval is unresolved, later occurrences for the blocked definition revision do not start new agent sessions. They are attributed to the approval's block key, which includes the original normalized action fingerprint; the system does not claim an unexecuted future occurrence would necessarily propose the same action. They increment `coalesced_occurrence_count`, set `first_coalesced_at` once, advance `last_coalesced_at`, and project those fields into run history, Attention, and Inbox. Approval starts at most one linked attempt representing all currently coalesced occurrences and then atomically resets the open coalescing window to zero. Denial or expiry closes the window, preserves its final count on the resolved approval, and adds `review_required` to `attention_states[]`; future slots remain blocked rather than accumulating silently. A revision change closes the old window as stale and starts no new window until a newly eligible occurrence requires approval. Refresh and replay use the canonical counters and never infer counts from notification deliveries.

A one-time definition that reaches approval remains `schedule_state=waiting_for_approval` even though it has no future slot. Approval starts one linked attempt for that occurrence; denial, expiry, or stale revision changes it to `exhausted_attention`. Confirmed cancellation changes an exhausted one-time schedule to `exhausted_cancelled`; unresolved cancellation uses `exhausted_attention` with `cancellation_uncertain`. A known terminal execution failure uses `exhausted_attention` with `execution_failed`. `Run again` from an exhausted one-time definition creates a new manual root invocation and does not rewrite the original schedule slot.

Automatic retry rules:

- pre-dispatch failures may use configured fixed or exponential retry;
- after session creation, automatic retry is forbidden unless no effects are confirmed or an idempotent checkpoint exists;
- delivery retry never reruns the agent;
- attempt limits and elapsed budgets apply across linked attempts;
- exhausted retries create one typed attention item.

Cancellation rules:

- request transitions execution to `cancelling`;
- only adapter evidence produces `cancelled`;
- lost contact or timeout produces `cancellation_uncertain` and unknown-effect handling;
- ordinary session or lease loss without a cancellation request produces `execution_uncertain`;
- asynchronous reconciliation can append evidence and revise the resolution with a complete audit trail, including a late exact-token correction after the reconciliation deadline;
- side-effect-capable tasks with unresolved cancellation block future dispatch by default;
- a manual override requires explicit duplicate-effect acknowledgement and audit;
- cancellation arriving after confirmed completion preserves completion and records that the request was too late.

Pause prevents new runs but does not cancel an active run. Archive also prevents new runs but does not silently claim cancellation.

## Approval And Recovery

An out-of-envelope action must stop before execution, close the active session, and create a durable approval item. Side-effect-capable targets that cannot mediate before action are not eligible for tool-enabled Agent Tasks.

Approval resolution supports:

- `deny`;
- `approve_once_and_retry_from_checkpoint`, only with an adapter-provided safe checkpoint;
- `approve_once_and_start_new_run`, with duplicate-effect acknowledgement when required;
- `update_future_policy`, which requires a new preview and authorization.

Approval never silently widens later runs. `Approve once` creates a run-scoped exception.

Every one-run exception is single-use and bound to definition revision, target, isolation and policy fingerprints, normalized tool/action identifier, canonical normalized-argument digest, checkpoint, and expiry. A newly generated action cannot consume the exception unless it exactly matches. Without a checkpoint, `Approve once and retry as new run` authorizes only the exact proposed action digest; any changed action creates a new approval.

Approval resolution and retry intent persistence are atomic in the Scheduled Tasks transaction. A transactional outbox with a unique retry-attempt identity materializes the Job. If later Jobs publication fails, the client reports `Approval recorded. The retry could not be queued.` and retries outbox publication without creating another approval or attempt.

Approval items:

- expire independently;
- resolve idempotently;
- cannot be approved after their definition revision becomes stale;
- track review separately from action resolution;
- remain linked to the original terminal run and any retry attempt.

Recovery mapping:

| Condition | Primary action |
| --- | --- |
| Failure before adapter start | `Retry as new run`. |
| Material policy drift | `Review changes`. |
| Approval with safe checkpoint | `Approve once and retry from safe point`. |
| Approval without checkpoint and no prior effects | `Approve once and retry as new run`. |
| Approval or failed run with possible duplicate effects | `Acknowledge risk and start a new run`. |
| Effects unknown | `Review evidence`; then use the risk-labelled action only when policy permits. |
| Cancellation unresolved | `Reconcile status`. |
| Grant nearing expiry | `Review and renew authorization`. |
| Result delivery failed | `Retry delivery only`. |

## Results, Inbox, Home, And Delivery

Run history is exhaustive. Results and Home are selective.

### Recording And Surfacing Policies

Policies are separated:

```text
surfacing_policy_v2:
  record_policy: noteworthy_only | every_run | history_only
  event_selection:
    output | external_action | run_summary | execution_failure | approval |
    safety_attention | policy_review | authorization_expiry | delivery_failure
  destinations:
    results_enabled
    home_enabled
    notifications_enabled
  home_policy: outputs_and_attention | attention_only | off
  notification_policy: per-channel destinations, thresholds, dedupe,
    aggregation, and quiet hours
```

Agent Task defaults are:

- `record_policy=noteworthy_only`;
- `home_policy=outputs_and_attention`;
- notifications disabled until configured.

The new policy is stored as versioned `surfacing_policy_v2`. It preserves existing `home_enabled`, `results_enabled`, `notifications_enabled`, `dedupe_key_strategy`, `failure_severity_threshold`, and `finding_confidence_threshold` values, plus destination, aggregation, and quiet-hour settings, as independently addressable subordinate fields. `event_selection` chooses eligible event classes; destination toggles then narrow where those selected events may appear. No compatibility mapping may enable a destination, broaden an event class, weaken a threshold, or remove dedupe that the user previously disabled or narrowed.

`record_policy` controls creation of ordinary Agent Results. `noteworthy_only` creates output and confirmed-action Results. `every_run` uses the same rules and creates one `result_type=run_summary` only for a terminal run that created no output or confirmed-action Result; it never adds a summary merely to duplicate a run already represented in Results. `history_only` creates no ordinary Agent Result, even when output remains permission-visible from run history. `results_enabled` controls projection into the user-facing Results surface, while `home_enabled` and `notifications_enabled` control those destinations independently. A canonical Agent Result excluded from Results remains permission-visible from its task and run; it is not projected into a disabled destination. Approval and safety attention follow the non-optional persistence rule below.

The Phase 4B/4C `visibility_policy` field remains as a deprecated compatibility projection during the migration window. When both are present, `surfacing_policy_v2` is authoritative, and an old-client update cannot erase fields it cannot represent. Old clients update only fields they own; omitted v2 fields retain their prior values.

Deterministic compatibility mapping:

| Legacy visibility value | `record_policy` | `home_policy` | Initial `event_selection` |
| --- | --- | --- | --- |
| `every_run` | `every_run` | `outputs_and_attention` | All event classes, including `run_summary`, allowed by existing thresholds and destination toggles. |
| `findings_only` | `noteworthy_only` | `outputs_and_attention` | Output or external action plus execution failure, approval, and safety attention. |
| `failures_only` | `noteworthy_only` | `attention_only` | Execution failure, safety attention, and delivery failure. |
| `failures_and_approvals` | `noteworthy_only` | `attention_only` | Execution failure, approval, safety attention, and delivery failure. |
| `task_history_only` | `history_only` | `off` | None for Results, Home, or notifications; canonical safety records still persist in Attention. |

Legacy `home_enabled`, `results_enabled`, and `notifications_enabled` map to the identically named destination toggles without default substitution. When a legacy visibility value and subordinate field conflict, migration takes their intersection: the visibility value defines the maximum eligible events and each false destination remains false. Existing thresholds and `dedupe_key_strategy` carry forward unchanged.

Unknown legacy values preserve disabled destinations, add `review_required` to `attention_states[]`, and present as `Needs review`; they are not guessed.

A noteworthy canonical Result or attention source is created when:

- non-empty user-facing output is durably available;
- artifacts were created;
- an external action was confirmed;
- approval, policy review, or authorization renewal is required;
- execution, result finalization, or delivery failed;
- effects or cancellation remain uncertain.

`result_type=run_summary` contains terminal state, timing, redacted summary, and provenance but never fabricates output, an artifact, or an external effect. `history_only` leaves ordinary output data in run history without Agent Results or Home projection.

Approval and safety resources are control-plane records, not optional Results. Approvals, execution or cancellation uncertainty, policy blocks, revocation, and other required recovery attention are always persisted and visible on task detail and in Attention to authorized users, regardless of `record_policy`, `results_enabled`, or event selection. Home and external notifications may be narrowed or disabled, but policy cannot erase canonical attention or its task-level indicator.

Meaningful output is based on typed adapter output, artifact, and effect fields. It is not classified by another model.

Canonical Agent Task output, confirmed-action, and user-selected run-summary Results use the versioned `/agent-results` resource with an open `result_type`, common provenance, redacted summary, artifact references, effect summary, and review state. Existing Phase 4C `/results` and its closed `finding|failure` schema remain unchanged. Approvals and unknown-effect incidents are not Agent Results; they remain canonical approval or attention resources and are only projected into Inbox.

### Normalized Home Inbox Projection

`GET /scheduled-tasks/inbox` returns a read-only union with:

- `inbox_item_id` derived from canonical source identity and version;
- `source_type`, `source_id`, and `source_version`;
- task, run, and owner references;
- output and attention signals;
- review and action state;
- safe summary and redaction state;
- occurrence time;
- canonical `coalesced_occurrence_count`, `first_coalesced_at`, and `last_coalesced_at` when the source aggregates blocked schedule occurrences;
- permission-checked canonical action links.

The projection does not duplicate approval, Result, or run state. Mutations target canonical source resources.

When one run has output plus approval, unknown effect, or delivery failure, the Home Inbox groups the sources under one presentation item while preserving all source IDs and actions. It renders separate state rows in deterministic order: uncertain effect or approval, execution failure, delivery recovery, then output review. Reading or snoozing output never clears an unresolved-risk row.

The Results surface lists canonical Recurring Question findings/failures and Agent output, confirmed-action, and user-requested run-summary Results. Run summaries appear only for definitions configured with `record_policy=every_run` and are filterable separately from outputs and actions. Linked attention rows may appear only in the context of a Result from the same run and link to Attention detail for evidence and resolution. An approval, unknown effect, policy block, expiry warning, or delivery issue without a Result appears in Attention and Home when enabled, never as a standalone Results item.

Required-action items can be marked read or snoozed. Snooze suppresses only Home placement and configured notifications until `snooze_until`; it never removes the task-level attention indicator, Attention count, canonical Attention entry, or run-detail warning. Required-action items cannot be dismissed until resolved or expired. Read state never implies action resolution.

Inbox itself is read-only. Review mutations go to the canonical source routes:

- legacy Result review endpoint for Recurring Question Results;
- `/agent-results/{agent_result_id}/review` for Agent Results;
- `/approvals/{approval_id}/review` and `/resolve` for approvals;
- `/attention/{attention_id}/review` for non-approvable attention.

Review requests include source version, idempotency key, and optional `snooze_until`. A grouped Home Inbox item has no bulk review mutation; the client updates each eligible source and reports partial outcomes. Read and snooze operations can never resolve an approval, clear unknown effects, or unblock execution.

### Safe Summaries

Adapter output and summaries are untrusted for first-party or external surfacing. Before Home, notification, webhook, or ordinary Result summary display, the server applies redaction and configured data-classification policy.

If safe summarization cannot be established, the surface shows:

> Output is available, but it cannot be safely summarized here.

The item links to secure adapter details when the requester has permission.

### Home Automation Inbox

Home retains the dedicated Automation Inbox module. Ordering is:

1. unresolved approvals and unknown effects;
2. failures and policy blocks;
3. unread outputs;
4. authorization-expiry warnings.

Each Home item shows task, type, plain-language state, redacted summary, time, one primary action, and an exact canonical deep link. Routine running state stays off Home unless a stalled or unusually long run creates an attention item.

Home may aggregate repeated outputs from one task for display, but durable Results and run history remain separate. Output items link to Results; attention-only items link to Attention. The module also links to both complete destinations when each has content.

Grant-expiry warning identity is stable per definition, grant, and expiry. Renewal resolves it instead of creating another warning.

### Delivery

- Home, WebUI notifications, extension notifications, email, and webhooks use canonical source IDs.
- Delivery idempotency keys include source ID, source version, destination, and event type.
- Delivery retries update `delivery_state` and never regenerate execution.
- Quiet hours delay notification delivery, not task execution.
- Webhooks contain redacted summaries and stable API resource IDs, never raw prompts, transcripts, credentials, or temporary signed links.
- Webhook destinations require ownership verification, redaction preview, test delivery, allowlisting when configured, and secret rotation.
- A useful output with failed delivery remains one item with both facts visible.

## Retention, Deletion, And Evidence

Exact durations are deployment-configurable, but safety and product semantics are normative. Every retained resource exposes `retention_class`, `retained_until` when finite, `deletion_state`, and legal-hold or policy-block status without leaking protected content. `deletion_state` distinguishes `active`, `deletion_pending`, `deleted`, `tombstoned`, `retained_by_policy`, and `outside_managed_scope`.

| Resource | Minimum retention and deletion rule |
| --- | --- |
| Provisional previews and payloads | Expire deterministically when unused. A consumed payload follows its definition and revision policy. |
| Active and superseded secure prompts | Archive does not delete them. `Forget message` performs immediate logical revocation, then physical deletion where eligible; legal hold and external/backup scope are reported explicitly. |
| Definitions and immutable revisions | Retain while any run, grant, approval, Result, attention, audit, migration, or adapter reference depends on them. Later purge leaves a non-sensitive identity/version tombstone. |
| Runs, attempts, and effect evidence | Retain through the maximum retry, cancellation, reconciliation, dispute, and delivery-recovery horizon. Unknown effects, unresolved cancellation, and active reconciliation cannot be purged. |
| Agent Results and review state | Retain according to policy after their source run reaches a stable terminal outcome. Removing a Result projection never removes source run/effect evidence or causes re-surfacing. |
| Approvals and attention | Unresolved items cannot be purged. Resolved or expired items retain resolution evidence and a dedupe tombstone through the audit and recovery floor. |
| Grants, delegations, and credential-use references | Retain normalized scope, version, actor, and revocation evidence for every dependent run; never retain credential material. |
| Audit, migration journal, dispatch intents, execution fences, and Jobs idempotency ledger | Retain through the longest dependent recovery or compliance floor. Purge preserves the minimum immutable identity, digest, and terminal-state tombstone required to prevent replay or dual dispatch. |
| Delivery receipts and dedupe records | Retain through destination retry and dedupe horizons; deleting them never reruns the agent. |
| Adapter transcripts, artifacts, and backups | Follow adapter-owned policy and legal hold. Scheduled Tasks reports retention/deletion scope and never claims deletion it cannot verify. |

Retention configuration may lengthen these floors but cannot shorten them while a safety, recovery, idempotency, migration, audit, or legal-hold dependency remains. Shortening retention requires a preview of affected resource classes and schedules asynchronous deletion rather than claiming immediate removal. Policy changes and deletion transitions are audited without storing prompt or output content.

Archive, review/dismiss, `Forget message`, and automatic retention are distinct actions. Archive stops future scheduling but retains evidence and secure content. Review changes presentation state only. `Forget message` targets prompt content and authority but does not erase external effects or unrelated evidence. Automatic retention removes eligible content while preserving required non-sensitive tombstones.

## Reference Client Information Architecture

Recommended canonical routes:

| Route | Purpose |
| --- | --- |
| `/scheduled-tasks` | Operational overview and task inventory. |
| `/scheduled-tasks/runs` | Cross-task execution log. |
| `/scheduled-tasks/results` | Output, external-action, and user-requested run-summary Results; attention appears only when related to a listed Result. |
| `/scheduled-tasks/attention` | Attention center with separate Approval requests and Uncertain effects sections. |
| `/scheduled-tasks/new` | Full-page creation. |
| `/scheduled-tasks/definitions/{definition_id}` | Task detail and management. |
| `/scheduled-tasks/runs/{run_id}` | Run and attempt detail. |
| `/scheduled-tasks/results/{result_item_id}` | Client Result detail for a canonical legacy or Agent Result; linked attention resolves in Attention. |

Existing query links redirect to canonical routes.

Visible primary navigation is Overview, Tasks, Runs, and Results. Attention appears when supported or non-empty and always retains a stable route. Attention shows distinct counts and controls for Approval requests, Uncertain effects, Policy review, and Delivery issues. Unknown effects never display approval controls. `New task` is a persistent page command rather than a tab.

The overview uses a compact state band, not a grid of metric cards. It highlights attention, current activity, upcoming runs, recent output, and partial backend health above a dense task table.

The task table supports search, sorting, pagination, capability-driven filters, column visibility, and safe bulk actions. Pause and archive bulk actions require a server bulk preview showing eligible, blocked, active-run, and consequence counts. Bulk resume, authorization, authority expansion, and delivery expansion remain deferred.

Recommended table fields are task/type, lifecycle, activity, attention, schedule, last outcome, next run, owner when relevant, and actions.

Watchlist rows retain `Manage in Watchlists` and do not expose reduced Scheduled Tasks editing controls.

## Reference Client Creation And Editing

`/scheduled-tasks/new` begins with source-agnostic jobs:

- Notify me later.
- Monitor information.
- Keep information current.
- Recheck a question.
- Ask an agent to act.

`Ask an agent to act` creates an `Agent automation`. It explains that an agent receives a scheduled message, may use explicitly granted tools, and produces inspectable runs and results. Visible copy reserves `Agent Tasks` for the existing project workspace.

Agent automation creation is a full-page, progressively disclosed form:

1. Task: name, capability-driven target, message, and optional context.
2. Authority: enforced read-only default or explicit tools, workspace, paths, network, credentials, runtime, cost, and expiry.
3. Schedule and delivery: one-time or recurring schedule, timezone, overlap, next runs, record, Home, and notification policy.
4. Review: target identity, authority, risks, prompt and policy fingerprints, preview expiry, `Save draft`, and `Authorize & enable`.

The first release uses structured controls and a read-only `View API payload`. It does not expose an editable JSON mode.

The target picker is searchable and keyboard operable. It shows readiness, adapter, read-only/tool capability, credential health, and freshness. Unavailable discoverable targets are placed under a secondary disclosure with reasons.

The authority form defaults to server-attested no-side-effect execution and reveals additional controls only when target capability or requested authority requires them.

Normal edit views keep the message masked and show prompt version, fingerprint, date, actor, `Reveal message`, and `Replace message`. Non-material edits can preserve the existing encrypted payload without revealing it.

Task detail uses Summary, Authority, Runs, Results, Configuration, and Audit tabs. Authority is not optional: it shows execution subject, authorizer, target and isolation profile, allowed and denied actions, workspace/path/network/data boundaries, credential references and owners, runtime/cost limits, expiry, definition/grant fingerprints, drift, and last authorization event. Run detail shows timeline, revision, grant, attempts, current freshness, tool/effect evidence, cancellation request time and confirmation source, output summary, delivery, adapter links, and one emphasized recovery action with secondary alternatives.

Configuration includes a Data and retention section showing retention class, retained-until dates, legal-hold or policy blocks, adapter deletion scope, and pending deletion state. `Forget message` is a separate destructive flow, never an overflow-menu shortcut. Its confirmation states that future runs pause, authority is revoked, active cancellation is requested, prompt recovery may be impossible, generated output and external effects are not erased, and some storage may remain under policy. It requires step-up immediately before commit and leaves a persistent deletion-scope report.

Edit and management outcomes are explicit:

- `Save changes` applies only non-material edits and retains the grant.
- Material edits offer `Authorize and apply` or `Save and pause`; they never leave an old grant attached to new authority.
- An active run keeps its original revision. Prompt, target, or authority-expansion edits do not cancel it automatically; authority reduction or live security revocation blocks later mediated actions and requests cancellation as specified by the API contract.
- Duplicate opens a paused review state, copies secure content only with permission, and has no grant.
- Archive requires consequence confirmation, leaves active-run behavior explicit, and supports `Restore` into paused state without restoring authority.
- Bulk previews identify tasks blocked by activity, permission, lifecycle, or policy before commit.

Power-user support includes multi-filter URLs, copyable IDs and exact timestamps, grouped retry attempts, pagination, stable deep links, and safe batch previews where later supported.

## Browser Extension Role

The extension may create an Agent automation draft from selected text, the current page, or an active research context. It must:

- show exactly what text or references will be attached;
- default to explicit selection or durable references, not implicit whole-page capture;
- allow removal before submission;
- use the same preview and encrypted-payload API;
- avoid authorizing execution unless it can present the complete current authority review and required step-up;
- confirm that a draft will not run until authorized;
- deep-link to `Review and schedule` in the canonical WebUI.

The handoff URL contains only an opaque server-side draft ID. It never contains selected text, prompt content, page references, target details, or credentials. The WebUI rechecks ownership and capabilities before opening the draft. If WebUI launch fails, the encrypted server draft remains available and the extension states that it is neither scheduled nor authorized. The failure surface shows the draft's exact expiry or retention time and offers `Retry opening review` and `Discard draft`; discard uses the secure-payload deletion report and does not imply deletion beyond its reported scope.

The extension also provides a compact, read-focused Automation updates surface. It may mutate review presentation state but never execution, authority, approval, or safety resolution:

- separate counts for unread output and unresolved attention, derived from canonical Inbox sources;
- a bounded list showing task name, plain-language state, redacted safe summary, and time;
- output links to canonical Results and attention links to canonical Attention detail in the WebUI;
- review-state actions such as mark read or snooze only when the same API capability is available;
- no approval, authority expansion, unknown-effect acknowledgement, prompt reveal, secure-output read, or destructive action in the first release;
- authentication and resource access rechecked when a notification or deep link opens;
- stale, offline, partial-load, and signed-out states that never render a false caught-up state;
- no persistent local cache of prompts, secure outputs, credentials, or unredacted summaries.

Extension notifications use redacted canonical Inbox data. Snooze follows destination policy, while unresolved attention remains visible in the extension's Attention count. High-risk recovery always deep-links to the full WebUI evidence and step-up flow.

## UX Status And Copy

Lifecycle, activity, attention, result, and delivery copy remain separate.

| State | Label | Supporting copy |
| --- | --- | --- |
| Draft | Draft | Saved, but not authorized to run. |
| Scheduled | Scheduled | Show the exact next run and timezone. |
| Paused | Paused | Future runs are paused. |
| Admission | Checking authorization | Verifying target, credentials, and policy. |
| Dispatch | Starting agent | Creating the agent execution. |
| Active | Running | Agent execution is active. |
| Cancel request | Cancelling | Waiting for the adapter to confirm termination. |
| Execution success | Run completed | Keep output and delivery state separate. |
| Output | Output ready | A safely surfaced output is available. |
| Secure-only output | Output available in secure details | Ordinary summary was blocked by redaction policy. |
| Approval | Approval required | A proposed action exceeded this task's authority. |
| Pre-dispatch failure | Could not start | The adapter session was not created. |
| Execution uncertainty | Execution unresolved | The run ended, but its final execution state could not be confirmed. |
| Unknown effects | Effects unconfirmed | The available evidence cannot establish whether effects occurred. |
| Cancellation uncertainty | Cancellation unresolved | The stop request could not be confirmed. |
| Superseded | Superseded | A newer task revision replaced this queued run. |
| Delivery failure | Output ready; delivery failed | Execution completed, but a destination did not receive it. |

Recommended command labels:

- `New task`
- `Preview task`
- `Save draft`
- `Authorize & enable`
- `Review changes`
- `Renew authorization`
- `Run now`
- `Pause future runs`
- `Cancel active run`
- `Approve once and retry from safe point`
- `Approve once and retry as new run`
- `Acknowledge risk and start a new run`
- `Reveal message`
- `Replace message`
- `Open secure transcript`
- `View API payload`
- `Manage in Watchlists`

Avoid generic `Submit`, `Confirm`, `Execute`, `Success`, or `Failed` labels without an object and consequence.

Persistent state changes appear in the page, not only in temporary toasts. Clients show pending language such as `Pausing...` or `Requesting cancellation...` and never optimistically claim authoritative completion.

Typed recovery includes preview expiry, revision conflict, target drift, grant expiry, stale approval, redaction block, partial API failure, permission removal, and adapter unavailability. Errors expose a safe code, request ID, and copyable diagnostics without raw traces or secret fields.

## Empty, Loading, Error, Running, And Success States

### Empty

- No tasks: `No scheduled tasks yet.` with `New task`.
- No agent targets: API-provided readiness reason and recovery action.
- No runs: `This task has not run yet.` with capability-gated `Run now`.
- No Results: distinguish no output, active filters, no tasks, and unavailable data.
- No approvals: show only after a successful load.
- Home caught up: `You are caught up. New automation outputs and issues appear here.`

Watchlists appears as a separate contextual destination for continuous source monitoring, not an Agent automation template.

### Loading And Partial Data

- Preserve table, form, and detail dimensions with skeletons.
- Never show an empty state before the first request resolves.
- Show `Last updated` and stale/reconnecting state.
- Disable high-risk commands when required capability or security state is stale and explain why.
- Retain successful sections when another source fails.
- Preserve entered data, focus, and scroll after preview or save errors.

### Running

- Report queued, admission, dispatch, running, and cancelling separately.
- Show visual elapsed time without announcing every update.
- Announce phase changes only for user-initiated or currently inspected runs.
- A stalled-run attention item is separate from ordinary activity.

### Success And Mutation Feedback

- Draft: `Draft saved. It will not run until authorized.`
- Enable: `Agent automation enabled.` plus exact next run.
- Pause: `Future runs paused.` plus active-run warning when applicable.
- Cancel request: `Cancellation requested.` until confirmed.
- Approval and retry report each durable outcome. A failed queue after approval says `Approval recorded. The retry could not be queued.`
- Forget message: `Message access revoked. Deletion is being verified.` followed by persistent per-location deleted, retained, pending, or outside-scope status.
- Result review and delivery changes update persistent state and use idempotent server responses.

## Accessibility And Usability Requirements

- Target WCAG 2.2 AA for WebUI and extension surfaces.
- Expose lifecycle, activity, attention, effect, result, and delivery in text, not color alone.
- Use semantic routes, headings, tabs, tables, forms, alerts, and progress status.
- Give icon-only controls accessible names and visible focus indicators.
- Keep target, tool, and path controls keyboard operable.
- Announce user-requested mutations, the selected run's phase changes, and critical attention through polite live regions. Do not announce global background churn or elapsed-time ticks.
- Preserve focus after background refresh and return it after inline confirmation.
- Associate warnings and error summaries with affected fields.
- Do not rely on hover for disabled or unavailable reasons.
- Support 200 percent text resize and reflow at 320 CSS px without loss of information or two-dimensional scrolling except for genuinely tabular content.
- Keep pointer targets at least 24 by 24 CSS px or satisfy the WCAG spacing exception; prefer 44 by 44 CSS px for primary touch controls.
- Ensure focused controls are not fully obscured by sticky review bars, drawers, confirmations, banners, or mobile browser chrome.
- Step-up authentication must support paste and password managers where applicable and support accessible WebAuthn flows; it must not rely on cognitive-function tests.
- Support reduced motion, high contrast, light, and dark themes.
- Display localized times with timezone and expose exact ISO timestamps.
- Truncate long values visually only while preserving full accessible names.
- Preserve equivalent state hierarchy and actions in mobile structured rows.
- Prefer 44px primary touch targets in the extension and narrow layouts while meeting the normative 24 CSS px minimum or spacing exception.
- Masked prompts expose `Message hidden`; revealed content is never placed in an `aria-live` region.
- Bulk selection reports selected, eligible, and blocked counts.

## Automatic Legacy ACP Schedule Migration

Migration is automatic for users, but operational cutover is versioned, journaled, and delivered in two compatible releases. Stopping APScheduler alone is insufficient because legacy `acp_run` work may already be queued, retrying, or running.

### Canonical Schema Backfill

Before any 4D revision-dependent execution or legacy ACP row movement, Phase 4D uses an expand, converge, backfill, activate sequence:

1. Expand canonical Scheduled Tasks, Jobs, and migration-control schemas with nullable revision, grant, dispatch, generation, and writer-epoch fields that old code can tolerate.
2. Deploy compatible readers and writers that preserve unknown v2 policy fields and report a new `schema_writer_epoch` in health and worker registration.
3. Wait until every scheduler, API, publisher, and consumer instance reports the required epoch; drain or quarantine old Jobs and block stale-epoch writers at the server boundary.
4. Briefly fence relevant mutations, backfill canonical rows, verify invariants, and then enable 4D.1A revision-dependent data paths through an activation flag. Execution remains disabled until the deployment passes 4D.0F and the 4D.1B vertical slice.

Mixed old/new writers are never allowed after activation. A stale writer or consumer receives a typed epoch error and cannot create, mutate, publish, or consume revision-dependent work. Backfill is restartable and does not infer security authority from absent legacy fields.

The canonical backfill then:

- snapshot every existing definition into an immutable revision;
- preserve current definition IDs and available version/audit provenance;
- map existing Recurring Question runs and Results to their legacy-compatible execution/result fields and mark incomplete revision provenance explicitly;
- classify every pre-4D `agent_task` with `redacted_only` or missing encrypted payload as `lifecycle=paused`, `schedule_state=blocked`, with `review_required` in `attention_states[]` because its prompt cannot be recovered;
- reject or deterministically skip old queued Jobs that lack run, revision, grant, generation, or dispatch-token fields;
- test upgrades from every supported historical Scheduled Tasks schema.

### Shared Migration Control

An authoritative migration-control store accessible to every instance provides:

- leased migration and scheduler-generation fencing tokens;
- a per-row journal keyed by source system, tenant, owner, and legacy schedule ID;
- source fingerprint and compare-and-swap version;
- target definition, revision, payload, and migration-grant IDs;
- deterministic target IDs allocated and journaled in `prepared` state before any target-store create;
- prompt HMAC, current step, batch cursor, error, and cutover generation;
- stable keyset pagination and restartable batches;
- explicit `prepared`, `verified`, `fenced`, `activated`, `rollback_requested`, `legacy_restoring`, `rolled_back`, `failed`, and `remediated` states.

Legacy and Scheduled Tasks storage may be different databases, so migration is a saga. A source row becomes permanently non-runnable only after target ciphertext, definition, schedule, identity, and ownership verification succeeds.

The canonical target store enforces unique `(source_system, tenant, owner, legacy_schedule_id)` provenance. Replay performs insert-or-verify using the journaled deterministic IDs and source fingerprint. If the target commit succeeds but journal acknowledgement fails, the next attempt finds and verifies that same target; it never allocates a second payload, definition, revision, grant, or runnable schedule.

Migrated provenance is an execution fence, not informational metadata. Every Run Now, retry, approval recovery, authorization, dispatch-intent creation, outbox publisher, scheduler, and worker admission path checks that the row journal and canonical definition both report `migration_state=activated` and are bound to the currently committed canonical generation. `prepared`, `verified`, `fenced`, `rollback_requested`, `legacy_restoring`, `rolled_back`, `failed`, or mismatched-generation targets are user-mutation- and dispatch-blocked with a typed migration-state error. Migration-internal compare-and-swap writes are the only allowed changes before activation.

### Release 1: Compatibility And Fence Preparation

Before automatic cutover, all supported instances deploy legacy submission and `acp_run` admission changes that carry and enforce:

- `legacy_schedule_id`;
- normalized occurrence slot;
- scheduler generation;
- idempotency token;
- source fingerprint.

Unversioned and stale-generation legacy tasks become quarantinable. This release does not transfer execution ownership.

### Release 2: Automatic Fenced Cutover

1. Freeze legacy writes and reject new mutation races.
2. Inventory through the durable journal and compare-and-swap each source row against its fingerprint/version.
3. Encrypt prompts, create canonical definitions/revisions, and classify authority.
4. Revoke the shared legacy scheduler generation and remove registrations.
5. Reject unversioned or stale-generation queued tasks at legacy handler admission.
6. Drain or quarantine queued and retrying legacy tasks and reconcile every active legacy adapter session.
7. Record `legacy_drained=true` and its evidence watermark only after zero unaccounted legacy work remains.
8. Verify exact source/target counts, prompt fingerprints, ownership, schedule slots, and next runs.
9. In one compare-and-swap transaction in the authoritative migration-control store, allocate the canonical scheduler generation and commit `canonical_activated=true` with its activation watermark. Canonical schedulers require both that exact generation and the committed flag before enqueueing.
10. Compare-and-swap each verified row and its target provenance to `activated` under that committed generation, then mark the source migrated and permanently non-runnable. Until this per-row step completes, every canonical dispatch path remains blocked for that row.
11. Remove plaintext from active legacy storage after verification quarantine and report residual scope.

`legacy_drained` and `canonical_activated` are separate durable markers. A process crash before the atomic activation transaction leaves canonical scheduling disabled. A crash after it leaves canonical ownership active and cannot be interpreted as rollback-safe.

Rollback is legal only when `canonical_activated` was never committed. It transitions each affected journal through `rollback_requested` and `legacy_restoring`, allocates a new legacy generation, re-registers and verifies eligible source rows, marks canonical provenance `rolled_back`, then unfreezes legacy writes. A rolled-back target remains stored for audit but cannot be mutated or dispatched. A later source edit updates the journal source version and fingerprint; a later migration reuses the stable canonical definition ID and creates a new immutable revision, payload, and grant while retaining old materials as inactive evidence. Each transition is restartable and compare-and-swap protected. After canonical activation, remediation is forward-only. An administrator kill switch can block all new Agent automation dispatch while preserving definitions, evidence, and history, but it never restarts legacy dispatch.

Legacy tasks missed during cutover default to `skip`. Past one-time tasks and accumulated cron slots never execute automatically. The first future eligible slot uses the migrated schedule.

### Field Mapping And Safety Classification

| Legacy field or condition | Canonical decision |
| --- | --- |
| String prompt | Encrypt exactly as message content; store only HMAC and redacted preview outside secure storage. |
| Message list | Normalize only when roles/content map losslessly; otherwise pause with `review_required` attention. |
| Missing or implicit agent type | Pause with `review_required` attention; never guess a target. |
| `cwd="."` or relative path | Resolve only against a recorded stable legacy root; otherwise pause with `review_required` attention. |
| Absolute cwd, mounts, filesystem intent | Pause with `review_required` attention unless a certified isolation mapping proves bounded read-only access. |
| Ambient credentials, external egress, MCP/tool access, subprocess, or `sandbox=none` | Pause with `review_required` attention; create no migration grant. |
| Model, persona, workspace, or credential reference | Preserve only after owner and target validation; otherwise pause with `review_required` attention. |
| Token/runtime budget | Map to equal or stricter limits; ambiguous or unbounded cost pauses with `review_required` attention. |
| Cron and timezone | Preserve verified current legacy semantics; invalid or unknown timezone fails migration rather than defaulting silently. |
| Enabled state | Continue only with a certified deny-all-effects isolation profile; otherwise migrate paused. |
| Concurrency, coalescing, or misfire semantics | Map only when behavior is equivalent or stricter; otherwise migrate paused with a visible difference. |
| Past one-time or partial execution | Classify as completed, missed, needs attention, or failed without dispatch. |

Every legacy schedule defaults to `lifecycle=paused`, `schedule_state=blocked`, with `review_required` in `attention_states[]` unless a versioned safety classifier proves no filesystem writes, external egress, tool/MCP access, subprocess, ambient credentials, uncontrolled cost, or other bypass. Only then may the system create an immutable `migrated_read_only` grant. Classifier version drift blocks future dispatch and adds review-required attention without changing lifecycle to a non-canonical value.

### Legacy API After Cutover

Legacy reads are a deliberate redacted compatibility break:

- retain stable list/detail pagination and legacy schedule IDs;
- return canonical definition ID, migration state, safe schedule fields, and `acp_config.prompt="[redacted]"`;
- omit raw and unknown ACP configuration fields that could contain prompt or credential data;
- include deprecation metadata plus canonical `Link` and item `Location` headers;
- return failed/ambiguous rows with `migration_state` and no fabricated canonical ID;
- advertise the changed schema in OpenAPI and contract documentation.

Legacy writes consistently return `410 Gone` with a typed code and canonical migration guidance. Contract tests prove no legacy read variant, including old redaction query values, returns raw prompt text.

Migration reports separately cover active storage, transaction/WAL logs, backups, replicas, adapter records, and external systems. The product does not claim deletion beyond proven scope.

### Cutover Release Gates

Activation preconditions are:

- the deployment class is `certified` by the 4D.0F execution feasibility gate;
- 4D.1B end-to-end tests prove migrated read-only fixtures produce inspectable runs, an Agent Result when output exists, a run-history record when output does not exist, a typed run-summary Result for that no-output case only under `record_policy=every_run`, Attention when applicable, and stable API deep links without legacy ownership transfer;
- the supported API operator workflow can display migration state, pause a definition, inspect run history, open Results and Attention, and explain blocked recovery before legacy writes become `410 Gone`; bundled enterprise deployments additionally prove the same 4D.1C WebUI workflow, while headless deployments provide equivalent supported API or CLI runbook evidence without requiring WebUI installation;
- every running instance supports and enforces the legacy fence;
- every running instance and worker reports the required schema writer epoch;
- the legacy scheduler generation is revoked;
- zero queued, retrying, or processing legacy tasks outside quarantine;
- every active legacy adapter session reconciled;
- exact source/target count and fingerprint agreement;
- zero runnable ambiguous rows;
- `legacy_drained=true` and its verification evidence durably visible;
- every prepared canonical target verified and dispatch-blocked pending activation;
- plaintext-residue report generated;
- dual-dispatch detector clear.

Activation postconditions are:

- `canonical_activated=true`, the allocated canonical generation, and activation watermark durably visible from every instance;
- each runnable migrated row explicitly moved to `migration_state=activated` under that generation before any dispatch path accepts it;
- legacy admission still rejects the revoked generation;
- dual-dispatch detection remains clear.

## Capability Contract

`GET /api/v1/scheduled-tasks/capabilities` reports independent readiness for:

- draft storage;
- encrypted prompt persistence;
- prompt reveal, plaintext prompt copy, encrypted payload clone, and prompt deletion as separate capabilities;
- target discovery;
- certified attested scheduled-execution isolation and certification evidence ID;
- scheduled-mode secure transcripts;
- read-only execution;
- tool-enabled execution;
- pre-action mediation;
- durable approvals;
- checkpoint retry;
- confirmed cancellation and reconciliation;
- normalized run, Result, Inbox, and Home delivery;
- transactional dispatch and delivery outboxes;
- legacy migration.

Capability entries include action status, required permissions, evidence source, reason, recovery action, `observed_at`, and `expires_at`.

Clients hide families only when unsupported or undiscoverable to the caller. Advertised but blocked families remain visible with reason and recovery. Route existence never implies readiness. High-risk clients refresh expired capability evidence before enabling controls.

## Security And Privacy Invariants

- No raw prompt in ordinary API responses, definitions, revisions, previews, audit, Results, Inbox, Home, notifications, webhooks, logs, metrics, or errors.
- No unkeyed prompt fingerprint.
- No prompt deletion, plaintext copy, or encrypted clone through `TASKS_CONTROL` or reveal permission alone.
- No execution under worker-service authority.
- No scheduled execution outside an attested deny-by-default isolation profile.
- No self-asserted or stale isolation evidence; attestation must verify against a live configured trust root and match tenant, workspace, runtime, policy, and dispatch fingerprints.
- No ambient credential, host filesystem, uncontrolled egress, subprocess, or direct MCP/tool bypass around mediation.
- No side-effect-capable run without pre-action mediation and a matching grant.
- No adapter-local, remembered, wildcard, session, batch, or automatic approval may expand Scheduled Tasks authority or override a live deny.
- No delegated identity or foreign credential use without an active version-bound scoped grant revalidated before each mediated use.
- No ordinary ACP transcript, detail, fork, export, or bootstrap path that reveals the scheduled prompt.
- No assistant-only secure output read when conservative prompt-echo redaction is uncertain.
- No automatic post-session retry without effect or checkpoint evidence.
- No generic Jobs retry or lease reacquisition of an effect-capable attempt after adapter dispatch.
- No canonical state mutation from a stale per-attempt execution fence after worker or reconciler ownership changes.
- No second Job for one attempt after active Job archival or supported purge.
- No `cancelled` state without adapter confirmation.
- No approval for an action that may already have occurred.
- No future authority expansion through one-run approval.
- No plaintext prompt copy, encrypted payload clone, or prompt deletion without its separate permission and any operation-specific step-up.
- No purge of unresolved safety/recovery evidence or deletion of the minimum tombstones required to prevent replay, duplicate effects, or dual dispatch.
- No cross-user target, preview, grant, payload, run, approval, Result, or adapter-link existence leak.
- No Home or webhook surfacing before redaction and data-classification policy succeeds.
- No client-side storage, URL, history, or analytics persistence of revealed prompts.
- No dual legacy and canonical scheduler ownership.
- No user mutation or dispatch of migrated canonical provenance before per-row activation under the committed canonical generation, or after rollback.
- No revision-dependent write or execution from a stale schema writer or consumer epoch.

## Backend Dependencies

The implementation plan must treat these as dependencies, not assume they exist:

- secure payload store with key lifecycle, quotas, deletion, and provisional cleanup;
- cross-resource retention policy, legal-hold evaluation, asynchronous deletion, deletion-scope reporting, and non-sensitive replay/dedupe tombstones;
- scheduled-mode secure adapter transcript storage and permission gates;
- 4D.0F-certified isolation backend, egress enforcement, minimal mounts, and brokered credentials;
- server-side attestation verification, trust-root rotation, signer revocation, and evidence freshness;
- provider-neutral target registry and capability freshness;
- ACP adapter with durable execution references;
- delegated execution identity and credential ownership enforcement;
- versioned act-as and credential-use grants with live per-action revocation checks;
- pre-action tool mediation and deny precedence;
- durable grants, approvals, checkpoints, and effect evidence;
- Jobs dispatch fencing, archive/purge-safe idempotency ledger, cancellation, and reconciliation;
- Scheduled Tasks per-attempt execution-fence ownership independent of the stable adapter dispatch token;
- Scheduled Tasks transactional dispatch/result/delivery outboxes and orphan reconciliation;
- run/result/delivery state separation;
- additive normalized Inbox projection;
- existing Scheduled Tasks schema/revision backfill;
- shared migration-control journal, all-path migrated-provenance admission fencing, restartable rollback, two-release legacy fencing, and canonical scheduler generation cutover;
- event stream or bounded polling contract for live state;
- local health and observability without external telemetry;
- TASK-13127 missing-definition consumer fix.

## Operational Visibility

Local metrics and logs cover:

- runs by low-cardinality state, family, and adapter type;
- queue, admission, dispatch, execution, finalization, and delivery latency;
- policy-block and admission reason classes;
- approval age and resolution;
- cancellation uncertainty duration;
- retry safety mode;
- dispatch and delivery outbox lag;
- orphan intent, Job, run, and adapter-session counts;
- worker lease-renew failure and post-dispatch lease-loss counts;
- unknown-effect age and reconciliation outcomes;
- redaction, payload encryption, rotation, quota, and deletion health;
- migration counts, batches, failures, legacy generation, `legacy_drained` evidence watermark, `canonical_activated`, canonical generation, activation watermark, per-row activation lag, rollback state, and dual-dispatch detection;
- Results, Inbox, Home, and destination delivery failure classes.

Prompts, output text, target-specific secrets, credentials, and user-controlled names are excluded from logs and metric dimensions. Correlation IDs join definition, revision, preview, grant, run, approval, Result, delivery, Job, and adapter records. This design introduces no external telemetry.

## Error Taxonomy

Typed errors must include safe recovery metadata. Initial Agent Task additions should cover:

- secure payload unavailable, quota exceeded, or key unavailable;
- target missing, unavailable, stale, changed, or unauthorized;
- preview invalid, expired, consumed, or mismatched;
- definition revision conflict;
- authorization missing, expired, revoked, drifted, or fingerprint mismatch;
- execution identity or credential invalid;
- isolation unavailable, stale, or fingerprint mismatch;
- secure transcript unavailable or insufficient permission;
- pre-action mediation unavailable;
- dispatch intent, outbox publication, orphan reconciliation, or lease loss after dispatch;
- stale execution fence or Jobs idempotency-ledger digest conflict;
- schedule, DST, overlap, or dispatch fence refusal;
- adapter start, execution, timeout, checkpoint, or cancellation failure;
- effects unknown;
- approval stale, expired, already resolved, or unsafe to retry;
- Result finalization, redaction, Inbox projection, or delivery failure;
- legacy migration ambiguity, conflict, partial failure, or cutover refusal;
- migrated provenance not activated, generation mismatch, rollback in progress, or rolled back;
- admin kill switch active.

Errors never include raw message, output, tool arguments, credentials, filesystem secrets, or cross-tenant resource detail.

## Phase 4D.0F Execution Feasibility Gate

Executable Phase 4D work does not begin from an assumed isolation backend. Before revision-dependent execution implementation, an ADR and reproducible proof must certify each supported deployment class that may advertise Agent Task execution.

The proof covers:

- server-verified isolation attestation with tenant, workspace, runtime, image, mount, egress, credential, signer, and expiry binding;
- hostile-agent attempts to access host files, uncontrolled network, subprocesses, direct MCP/tools, inherited secrets, and ambient credentials;
- scheduled-mode transcript storage proving prompt sentinels do not appear in ordinary ACP detail, fork, export, bootstrap, search, logs, errors, or audit;
- idempotent adapter session creation and exact dispatch-token lookup after process loss;
- durable terminal, timeout, pre-action approval, effect, and cancellation evidence with monotonic cancellation ordering;
- brokered identity and credentials plus a credible pre-action mediation path for the later write-capable slice;
- operational installation, upgrade, health, and fail-closed behavior on every deployment class claimed as supported.

Outcomes are explicit:

- `certified`: execution implementation may proceed for that deployment class under the proved isolation profile;
- `draft_only`: definitions, encrypted prompts, previews, authorization review, and migration dry-run may ship, but execution and canonical migration activation remain unavailable;
- `unsupported`: Agent automation creation remains undiscoverable or visibly unavailable with a reason and recovery guidance.

A failed or partial proof never relaxes isolation requirements. It narrows the advertised capability and rollout. M2 cutover cannot activate a migrated definition on a deployment class that lacks certification.

## Rollout

| Stage | Product capability |
| --- | --- |
| 4D.0 | Fix TASK-13127 with dedupe-preserving missing-definition handling and establish a green focused baseline. |
| 4D.0F | Certify or reject execution feasibility per deployment class; publish the isolation/adapter ADR and capability outcome. |
| 4D.0E | Expand canonical schemas, deploy epoch-compatible readers/writers, converge every process on the required `schema_writer_epoch`, drain stale work, and backfill before revision-dependent execution is enabled. |
| 4D.M1 | Deploy legacy generation/idempotency fields and handler fencing to every instance; run migration inventory and dry-run without ownership transfer. |
| 4D.1A | Revisions, normative RBAC, secure payload/transcript storage, certified isolation, target discovery, typed mutations, and transactional outboxes. Draft-only deployments stop here with execution disabled. |
| 4D.1B | Adapter-enforced no-side-effect ACP execution plus durable runs, Agent Results including run summaries, Attention, Inbox/Home delivery, retention state, and end-to-end recovery APIs. |
| 4D.1C | Migration-readiness acceptance: minimum WebUI inventory, migration state, task detail, pause, run history, Results, Attention, secure deep links, and recovery guidance for bundled enterprise deployments; equivalent supported API or CLI operator evidence for headless deployments. |
| 4D.M2 | After 4D.1B and applicable 4D.1C distribution gates pass, verify the completed 4D.0E backfill, execute journaled legacy migration, drain/reconcile legacy work, and activate the canonical scheduler generation. |
| 4D.2 | Revision-bound grants, bounded tools, pre-action mediation, durable approvals, effect evidence, cancellation, and safe retry. |
| 4D.3 | Complete creation/Authority/power-user WebUI, extension draft capture and Automation updates, advanced delivery, accessibility validation, and migration operations. |
| GA | All security, migration, contract, observability, recovery, and usability gates satisfied. |

Read-only pilot execution requires enforced denial of every side-effect-capable adapter and built-in action. Write-capable scheduling remains disabled until 4D.2 gates pass. API-first does not mean safety depends on the reference client; direct API clients receive the same previews, typed refusals, grants, approval contract, and audit.

## Validation Strategy

### Contract And Security

- OpenAPI snapshot and generated-client compatibility tests, including stable resource-specific path parameter names.
- Additive enum and legacy status mapping tests.
- Route-by-route positive and negative RBAC tests for every listed read, control, authorization, approval, reveal, plaintext copy, encrypted clone, prompt deletion, secure-output, retry, cancellation, duplicate, audit, and bulk endpoint. Prove `TASKS_CONTROL` alone cannot authorize, approve, reveal, copy, clone, delete prompt content, or read secure output.
- Tenant isolation and actor-matrix tests for same-tenant cross-user delegation, inactive principals, revoked act-as, foreign credentials, and concealed `404` behavior.
- Scheduled-authority precedence tests proving ACP remembered allows, session or batch approvals, wildcard adapter permissions, and model-selected automatic tiers cannot expand an exact grant or override a live deny.
- Attestation verification tests for forged signatures, stale evidence, not-yet-valid evidence, wrong tenant or workspace binding, runtime or policy digest mismatch, and revoked signer or trust root.
- Delegation and credential-use binding tests for ID, version, scope, expiry, and mid-run revocation before every mediated action and credential issue.
- Typed draft-create, create-and-authorize, update-only, update-and-reauthorize, and authorize-and-run idempotency and old-client tests.
- Agent Task Run Now tests for mandatory idempotency key, expected revision, and active grant version, plus unknown-effect override tests for stale evidence, duplicate-effect digest, step-up, and unique dispatch intent.
- Prompt leakage checks across every ordinary API and operational surface.
- Sentinel checks across scheduled ACP detail, database, fork, export, bootstrap, logs, errors, backups, and legacy redaction query variants.
- Prompt-echo checks for exact, quoted, encoded, transformed, and tool-argument copies, including fail-closed assistant-only output reads.
- Prompt-reveal tests for bounded reason codes, `view|edit|copy` purpose, clipboard audit flow, no caller free text in audit, and sentinel exclusion from the audit record itself, ordinary logs, metrics, and notifications.
- Adversarial output redaction and data-classification checks.
- Credential, key rotation, key outage, quota, and cryptographic deletion checks.
- Hostile-agent isolation checks proving direct file write, socket, subprocess, ambient-secret, MCP, and tool bypass attempts are blocked.
- 4D.0F certification tests and ADR evidence for every deployment class that advertises execution, plus `draft_only` and `unsupported` capability behavior.
- Cross-resource retention tests for unresolved-evidence floors, legal hold, policy shortening, asynchronous deletion, adapter-scope reporting, and replay/dedupe tombstones.

### Scheduling And Execution

- Schedule, timezone, DST, missed-run, overlap, coalescing, and idempotency property tests.
- Pause, edit, archive, revoke, credential change, kill switch, and cancel dispatch-race tests.
- Material-edit tests distinguishing `superseded_for_new_dispatch` from `revoked_immediately`, including active-run continuation after pause/archive and immediate action blocking after authority reduction or security revocation.
- Mid-run grant, credential, execution-subject, and kill-switch revocation before every mediated action and credential issue.
- Failure injection before dispatch, after session start, during secure output persistence, during projection, and during delivery.
- Process termination after each outbox, Job publication, dispatch-token, adapter-session, effect, finalization, and delivery boundary.
- Orphan intent, Job, run, and adapter-session reconciliation without duplicate dispatch.
- Late worker finalization after lease loss and reconciler takeover, proving a stale execution fence cannot mutate canonical state while exact dispatch-token evidence remains usable.
- Jobs idempotency-ledger replay after creation, completion, archival, and supported purge before outbox acknowledgement, including digest-mismatch refusal.
- Jobs `max_retries=0`, lease renewal, post-dispatch lease loss, and no automatic reacquisition tests.
- Approval expiry, stale revision, concurrent resolution, checkpoint, and duplicate-effect acknowledgement tests.
- One-use approval tests for altered action arguments, changed fingerprints, concurrent consumption, and outbox replay.
- Ordinary lease/session `execution_uncertain` and `cancellation_uncertain` reconciliation tests, including late exact-token transition to succeeded, failed, timed out, approval required, or cancelled; atomic approval materialization; cancellation-race precedence; and effect-only evidence that does not falsely resolve execution.
- Process loss after the adapter records timeout or pre-action approval but before canonical finalization.
- Cancellation races with timeout and pre-action approval, including process loss immediately before and after the adapter records the monotonic cancellation boundary.
- Delivery retry tests proving no agent redispatch.
- Review/snooze race tests proving Inbox mutations never resolve approval or unknown-effect state.
- Phase 4B/4C surfacing-policy mapping, including `failures_only`, disabled destination toggles, thresholds and dedupe, and old-client round-trip tests that never broaden prior choices.
- `noteworthy_only`, `every_run`, and `history_only` tests proving output/action Results, typed no-output run-summary Results, Results filters, and run-history-only behavior remain distinct.
- Attention persistence tests proving record, Results, Home, notification, read, and snooze policy cannot remove canonical approval or safety state from task detail and Attention.

### Migration

- Active, paused, disabled, one-time-past, ambiguous, invalid, duplicate, and partially migrated fixtures.
- Crash and restart after every durable migration step.
- Target-commit-before-journal-ack crash tests proving deterministic insert-or-verify replay and one target row per legacy provenance key.
- Database-backed ownership fence and dual-dispatch prevention.
- Direct Run Now, retry, approval recovery, authorization, outbox publication, and worker-admission refusal before per-row activation and after rollback.
- Restartable rollback-edit-remigrate tests proving a new source fingerprint creates a new inactive revision/payload/grant under the stable canonical definition ID before later activation.
- Mixed-version writer and consumer tests across schema expansion, old-epoch rejection, pre-fence, fence-aware, and canonical generations.
- Process termination immediately before and after the atomic canonical activation transaction, proving rollback is possible only in the former case.
- Large-inventory bounded batching and progress recovery.
- SQLite and PostgreSQL legacy/control-store combinations supported by deployment.
- Clock-skew, cron, timezone, DST, coalescing, and misfire equivalence tests.
- Legacy read projection and `410 Gone` write compatibility.
- Plaintext cleanup and residual storage-scope reporting.
- Existing Phase 4B/4C schema backfill plus stale old-Job rejection.
- Deleted, never-created, and cross-owner no-run Jobs with `run_id=null`, plus terminal redeliveries that retain their recorded `run_id` and deduped outcome.
- M2 refusal until certified execution, the 4D.1B run/Result/Attention vertical slice, the API operator workflow, and the applicable bundled-WebUI or headless-operator 4D.1C gate pass.

### Reference Clients

- Capability combinations across WebUI and extension.
- First-time draft and read-only enablement.
- Power-user create, edit, pause, inspect, duplicate, filter, and debug journeys.
- Approval, stale approval, unknown effect, cancellation, and delivery recovery.
- Results and Home canonical identity and dedupe.
- Extension Automation updates counts, output/attention distinction, redacted notifications, read/snooze behavior, stale/offline/signed-out states, and high-risk WebUI handoff.
- Keyboard, screen reader, 200 percent zoom, responsive, high-contrast, dark/light, and reduced-motion validation.
- Explicit Watchlists and standalone Agent Tasks non-regression coverage.

### Quality Gates

- Focused backend and frontend tests.
- Broader Scheduled Tasks, Notifications, ACP, Jobs, Watchlists, and Agent Tasks regression suites appropriate to touched scope.
- Formatting, lint, type, compile, OpenAPI, and diff checks.
- Bandit on touched Python scope with no new findings.
- Browser screenshots and interaction checks for desktop, narrow WebUI, and extension surfaces.

## UX And Product Acceptance

- At least 5 of 6 representative first-time users can save and enable an enforced read-only Agent automation without assistance.
- Those users can identify the agent, next run, granted authority, and result location.
- At least 5 of 6 experienced users can create, pause, inspect, duplicate, and locate a failed run without assistance.
- At least 5 of 6 experienced users can reauthorize a material edit, resolve a coalesced approval, recover an extension handoff, inspect an uncertain one-time cancellation, and navigate correctly between Results and Attention without unintended authorization, rerun, dismissal, or disclosure.
- At least 5 of 6 representative users can distinguish an Agent automation from a Watchlist and a standalone Agent Tasks project and choose the correct destination for a described job.
- At least 5 of 6 extension users can distinguish unread output from unresolved attention, open the correct canonical WebUI destination, and understand that high-risk recovery cannot be completed in the extension.
- No participant mistakes unresolved cancellation, approval required, or delivery failure for successful completion.
- Keyboard-only users can complete creation, authorization review, run inspection, and approval recovery.
- Testing with 200 percent text resize and at 320 CSS px reflow leaves no hidden state, obscured focus, or unreachable action.
- Pointer targets meet the 24 CSS px WCAG 2.2 AA minimum or a valid spacing exception, with 44 CSS px preferred for primary touch controls.
- Step-up authentication works with paste, password managers, and accessible WebAuthn without cognitive-function barriers.

## Technical Acceptance Criteria

- No tested surface leaks raw prompt text.
- No execution capability is advertised without a current 4D.0F certification for that deployment class.
- No scheduled adapter path can bypass isolation through host filesystem, uncontrolled egress, subprocess, direct MCP/tool access, or ambient credentials.
- No ordinary ACP transcript/detail/fork/export/bootstrap path exposes the scheduled prompt.
- No migration or delivery retry produces duplicate agent execution.
- No stale worker, archived Job replay, or pre-activation migrated target produces duplicate or unauthorized execution.
- Crash recovery produces no orphaned or duplicate adapter execution across run, outbox, Job, and adapter-session boundaries.
- No write-capable run starts without pre-action mediation and a valid matching grant.
- Every active run has an inspectable revision, identity, grant, and adapter reference.
- Every unknown effect blocks unsafe automatic retry.
- Every migrated schedule has deterministic lifecycle, schedule-state, activity, and attention fields, including paused plus review-required for ambiguous rows and exhausted schedule states for completed, missed, or failed one-time rows.
- No M2 activation occurs before durable run/Result/Attention APIs and a supported migration-ready operator workflow are available; WebUI parity is additionally required for bundled enterprise deployments, not as a dependency of the API product itself.
- Retention never purges unresolved safety/recovery state or required replay/dedupe evidence, and deletion status never overstates adapter or backup scope.
- Results, approvals, runs, and adapter records have stable permission-checked deep links.
- Results/Home review and action state derive from canonical resources.
- Watchlists and standalone Agent Tasks preserve existing functionality and ownership.

## Alternatives Considered

### Wrap Legacy ACP Schedules

Rejected as the canonical design. It would retain raw prompt storage, cron-only assumptions, fragmented history, and weak authority boundaries, then require another migration for non-ACP targets.

### Materialize Standalone Agent Tasks

Rejected. It would mix recurring automation with project planning, create unnecessary project/task objects, and obscure run, retry, result, and ownership semantics.

### Never Retry Automatically

Safer but unnecessarily reduces reliability for failures proven to occur before execution. The selected design automatically retries only before dispatch or from adapter-proven safe checkpoints.

### Keep Legacy ACP Schedules Separate

Rejected by product decision. Automatic migration creates one control plane and removes split management and dispatch ownership.

### Store Full Output In Scheduled Tasks

Rejected. It duplicates sensitive transcripts and artifacts, complicates retention, and ties the control plane to ACP. Scheduled Tasks stores redacted summaries and adapter links.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Prompt or output leakage | Encrypted payload refs, tenant-keyed HMACs, no-store reveal, redaction gate, leakage tests. |
| Agent bypasses declared tool policy | Mandatory attested isolation, deny-by-default egress/mounts, no ambient credentials, brokered actions, hostile-agent tests. |
| ACP transcript bypasses prompt reveal | Scheduled-mode secure transcript, redacted ordinary ACP paths, separate secure-output and prompt-reveal permissions. |
| Duplicate side effects | Dispatch fencing, phase-aware retry, checkpoint evidence, unknown-effect block, delivery separation. |
| Crash creates orphan Job or adapter session | Transactional outboxes, dispatch tokens, idempotent adapter lookup, reconciliation, process-kill tests. |
| False cancellation confidence | Confirmation-only cancellation and asynchronous reconciliation. |
| Authority drift | Immutable revisions, fingerprints, admission revalidation, fail-closed review. |
| Target portability becomes ACP-specific | Provider-neutral target and adapter record contract. |
| Approval floods or stale requests | Durable queue, expiry, revision binding, aggregation, and typed resolution. |
| Home becomes noisy | Separate record/Home policies, bounded aggregation, attention ordering. |
| Generated clients break | Additive Inbox route, compatibility mappings, schema capability versions, contract tests. |
| Legacy and canonical schedulers both execute | Database-backed generation fence and forward-only cutover. |
| Plaintext deletion is overstated | Storage-scope report and explicit backup/log caveats. |
| Reference UI becomes the hidden product boundary | Capability-driven public API contract and direct-client acceptance tests. |

## Known Deferrals

- Generic API agent adapters beyond ACP.
- Parallel Agent Task execution.
- Editable schema-aware JSON configuration in the reference client.
- Bulk resume, authorization, authority expansion, and destination changes.
- Admin-only `run_all` missed-run replay.
- Cross-adapter transcript normalization beyond stable secure links and common metadata.

These are deferrals, not unresolved product decisions for the first Phase 4D plan.

## Open Implementation Questions

No blocking contract decision remains open. Execution feasibility remains an explicit 4D.0F go/no-go gate rather than an assumed implementation detail. The implementation plan must resolve these deployment choices without weakening the contract:

- whether secure payload/transcript storage is co-located with Scheduled Tasks or uses the provisional saga and outbox path;
- which existing or new shared store owns migration leases, scheduler generation, and the cutover journal in each deployment mode;
- whether live run updates use an event stream, bounded polling, or both behind one capability contract;
- deployment-specific retention and authorization-expiry durations within the normative safety and evidence floors;
- exact OpenAPI version names and deprecation dates for legacy ACP schedule projections.

## Review Record

Three independent review tracks completed with no unresolved findings:

- API and security review covered route-level RBAC, prompt/output isolation, authority precedence, attestation trust, delegation and credentials, typed authorization, recovery, and compatibility.
- Product UX, HCI, and accessibility review covered state comprehension, one-time and recurring recovery, coalesced approvals, Results versus Attention ownership, Home surfacing, extension handoff, naming, and WCAG 2.2 AA.
- Migration and reliability review covered schema epochs, target replay, all-path activation fencing, rollback, execution ownership fences, Jobs idempotency across archival, uncertainty reconciliation, and cancellation race ordering.

Blocking and high-severity findings from each track were incorporated before approval. This is a design review result, not evidence that the described dependencies are implemented.

A subsequent cross-section review identified and addressed seven additional issues: migration preceding result/recovery surfaces, contradictory `every_run` Result semantics, incomplete retention rules, conflated destructive/export permissions, missing execution feasibility proof, incomplete extension result/attention behavior, and non-canonical OpenAPI path parameter names.

The user approved the revised design on 2026-08-24 after the cross-section remediation and final API-first consistency review.

## Recommended Implementation Plan Shape

The follow-up implementation plan should split reviewable work into:

1. TASK-13127 dedupe-preserving missing-definition fix and contract characterization.
2. 4D.0F isolation, scheduled-transcript, adapter-idempotency, cancellation-evidence, and deployment feasibility proof plus ADR.
3. 4D.0E immutable-revision/schema backfill, normative granular RBAC, actor/delegation rules, retention fields, and compatibility mappings.
4. 4D.M1 legacy generation/idempotency fields, handler fencing, inventory, and dry run without ownership transfer.
5. Secure payload and scheduled-mode transcript stores, target registry, certified isolation integration, and preview changes.
6. Typed definition/authorization transactions, grants, material diffs, reveal/copy/clone/delete, and live revocation.
7. Transactional dispatch/result/delivery outboxes, run identity, Jobs attempt contract, dispatch and execution fences, and reconciliation.
8. ACP isolated no-side-effect adapter plus durable runs, versioned Agent Results/run summaries, canonical attention, additive Inbox/Home, retention/deletion state, notifications, and review mutations.
9. End-to-end result/recovery proof plus the migration-ready API operator workflow, minimum WebUI parity for bundled enterprise deployments or equivalent headless runbook evidence, then M2 drain/reconcile and canonical scheduler cutover.
10. Bounded tools, pre-action mediation, durable approvals, checkpoints, effects, safe retry, and cancellation.
11. Complete creation/Authority/power-user WebUI, extension draft capture and Automation updates, accessibility, advanced migration operations, observability, broad regression, and release gates.

Each implementation slice must use capability gates and test-driven development. Write-capable execution must remain disabled until its complete safety slice is present and verified.
