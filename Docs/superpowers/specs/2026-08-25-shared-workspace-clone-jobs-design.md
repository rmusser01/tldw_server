# Durable Shared Workspace Clone Jobs Design

**Status:** Revised after pre-implementation review; awaiting confirmation

**Backlog:** TASK-12020.41

**Issue:** https://github.com/rmusser01/tldw_server/issues/2734

## Problem

The recipient clone endpoint currently invents an in-memory job identifier, runs the clone through a FastAPI background task, and immediately returns `pending`. No durable job record or status API exists, application restart loses the operation, failures are reduced to server logs, and the Shared With Me UI announces success before any copy has completed.

The replacement must use the existing Jobs subsystem, keep authorization server-authoritative, prevent duplicate targets across active and archived Jobs, survive reloads and restarts, report retrieval readiness truthfully, and avoid exposing the generic Jobs administrative contract to recipients.

## Goals

- Enqueue an owner-scoped, idempotent `sharing/workspace_clone` Job.
- Preserve idempotent replay after a Job is archived.
- Expose one bounded Workspace operation envelope from enqueue and status APIs.
- Execute cloning through one reusable handler used by application-managed and standalone WorkerSDK lifecycles.
- Revalidate current share access and `allow_clone` immediately before copying owner content.
- Distinguish copy completion from text, citation, and vector retrieval readiness.
- Resume recipient polling after WebUI reload without adding a page-level status banner.
- Prevent repeated requests with the same idempotency key from creating duplicate target workspaces.

## Non-Goals

- Exposing generic Jobs rows, payloads, stack traces, or administrative controls.
- Making the cross-database copy transactionally resumable in this slice.
- Automatically retrying a clone after a worker or process failure.
- Adding a hidden or fire-and-forget embeddings dependency.
- Implementing durable workspace reindex orchestration in this task. Follow-up `TASK-12020.45` owns that capability.
- Adding route aliases or redirects.
- Combining Research Workspace and Shared With Me into one route or product surface.

## Design Principles

1. **Jobs owns execution state.** Active and archived Jobs remain the only operation-status records.
2. **A receipt owns durable idempotency correlation.** A small generic Jobs receipt records which immutable Job identity accepted an owner-scoped request; it is not a second status projection.
3. **The public API speaks Workspace language.** Recipients receive `operation_id`, `workspace_id`, `command`, and canonical Workspace operation states, never Job internals.
4. **Copy success is not indexing success.** The result reports retrieval readiness independently from copied-item counts.
5. **Authorization is evaluated at use time.** Enqueue and worker execution both resolve canonical server state rather than trusting payload claims.
6. **Failure is bounded and recoverable.** The server never reports uncertain work as success and never exposes source titles, paths, content, or raw exceptions in diagnostics.

## Architecture

### Jobs Contract

Clone work uses:

- domain: `sharing`
- queue: configurable, default `workspace-clone`
- job type: `workspace_clone`
- owner: recipient user ID
- batch group: a bounded share correlation value
- idempotency key: a digest of recipient ID plus the validated client `Idempotency-Key`; the digest is stored in the active/archived Job and receipt, while the raw key is never persisted
- maximum automatic retries: `0`

The Job payload contains only bounded identifiers and normalized request data: schema version, share ID, recipient user ID, requested name, and canonical request fingerprint. It does not contain copied content, credentials, authorization claims, source titles, or owner database paths.

`JobManager.DOMAIN_ALLOWED_QUEUES` adds `"sharing": ("workspace-clone",)`. Configuration and operator documentation list the queue and worker flag explicitly.

### Durable Idempotency Receipt

The current active-Job idempotency lookup is insufficient because archival removes the active row. This design adds a generic Jobs primitive, not a clone-specific projection:

`job_idempotency_receipts`

- immutable receipt ID
- domain, queue, job type, and owner user ID
- bounded idempotency-key digest
- canonical request fingerprint
- immutable Job UUID and internal Job ID
- created timestamp
- unique constraint over domain, queue, job type, owner user ID, and key digest

Job creation and receipt insertion occur in one Jobs-database transaction. The admission primitive takes a stable owner/share scope lock using the existing database-specific Jobs locking pattern, checks the receipt, checks for another queued/running clone for that owner/share, and then inserts. SQLite uses its write transaction; PostgreSQL uses a transaction-scoped advisory lock. This prevents two browser tabs with different keys from creating simultaneous copies of the same share. If the active clone has the same fingerprint, admission adds a receipt alias from the second key to the existing Job and returns that operation. A different fingerprint returns `409 clone_already_in_progress` with the existing bounded operation reference.

On an idempotency uniqueness race, admission reads the winning receipt, compares the request fingerprint using constant-time comparison, and returns the referenced operation when it matches. A mismatch returns `409 idempotency_key_reused`.

The receipt survives Job archival. Jobs gains one generic active-plus-archive lookup by Job UUID, performed as one consistent database read so an archival race cannot create a false not-found gap. Duplicate, missing, or malformed receipt/Job correlations fail closed with `503 clone_operation_unavailable`; they never admit another Job under the same key.

Receipts have a documented 30-day minimum replay window. They are never removed while the referenced Job is nonterminal. Jobs pruning must archive every receipt-backed Job even when global archive-before-delete is disabled, so status-by-operation remains available after the active row is pruned. Expired terminal receipts may be pruned; after that documented window, replaying the old client key is a new request. Any future permanent archive purge must remove its receipt in the same database transaction. The receipt carries correlation and fingerprint data only; progress, status, result, and error remain exclusively on the Job record.

### Admission And Replay Algorithm

`POST /api/v1/sharing/shared-with-me/{share_id}/clone` requires:

- `Idempotency-Key`: 16-200 ASCII characters from `[A-Za-z0-9._~-]`; no trimming or case normalization
- optional JSON `name`: normalized once using the Workspace name rules and bounded before fingerprinting

The API computes a fingerprint from schema version, share ID, recipient ID, and normalized name.

1. Look up an owner-scoped receipt for this key.
2. If one exists, compare its fingerprint and share correlation. A match returns the existing active or archived operation, including after later share revocation. A mismatch returns `409`.
3. If no receipt exists, resolve the canonical active recipient share and require `allow_clone=true`.
4. Under the owner/share admission lock, converge a same-fingerprint queued/running clone through a receipt alias or reject a different fingerprint.
5. Create the Job and receipt atomically.
6. Return the operation projection from the persisted Job.

This ordering lets a recipient recover an already accepted request whose response was lost even if the share is revoked before replay. It does not let a revoked recipient create new work.

A newly queued or active replay returns `202`. A terminal replay returns `200`. Jobs unavailability or an uncertain admission transaction returns typed `503`; the client keeps the same idempotency key and reconciles rather than generating a second request.

### Target Identity And Retry Policy

The worker derives the target workspace UUID deterministically from the durable Job UUID. Therefore one accepted operation cannot create multiple target workspace identities.

The Workspace DB adds explicit `reserve_clone_target`, `publish_clone_target`, and `discard_clone_target` methods rather than calling the currently nonexistent `CharactersRAGDB.create_workspace()` path. Reservation uses the deterministic ID, existing Workspace validation, `archived=true`, and a bounded system metadata marker containing the clone operation ID. It is idempotent only when an existing row carries the same marker and normalized request identity. A collision with any other row fails closed.

Targets carrying the system clone marker are excluded from normal and archived recipient lists. Publication verifies the marker and sets `archived=false` only after copy/readiness validation, but retains a `publication_pending` marker until fenced Job completion succeeds. The worker then clears the marker best-effort, making the copy discoverable. Controlled failure soft-deletes the staged target. A bounded clone-worker reconciliation pass cleans staged or publication-pending targets correlated with terminal failed/cancelled/quarantined Jobs, and clears stale markers from valid completed targets after a hard process exit. It uses the same operation ownership markers for media cleanup. This keeps partial targets out of normal Workspace navigation without inventing a second operation-status store.

The initial clone implementation is not transactionally resumable across ChaChaNotes and Media databases. Automatic retries are disabled. A terminal fatal failure is not re-enqueued with the same key. `Try again` generates a new idempotency key and therefore a new Job and deterministic target ID.

### Authorization And Capability Policy

Admission resolves the canonical active recipient share and checks `allow_clone`. The worker repeats that resolution before opening owner content databases, before each top-level source/note/artifact copy, and before publication. Revoked access or removed clone permission triggers controlled cancellation and cleanup. Authorization reads use a thread-local repository handle and never trust permission claims from the Job payload.

Status lookup authenticates the Job owner first, then verifies domain, queue, type, and share correlation from bounded identifiers. Missing, foreign, malformed, or mismatched operations return the same neutral `404`. Status remains readable by its recipient owner after later share revocation.

The shared-workspace bootstrap capability `clone_workspace` becomes authoritative:

- active share with `allow_clone=true`: `allowed=true`
- active share with `allow_clone=false`: `allowed=false`, reason `owner_disabled`
- revoked or missing share: no recipient bootstrap

The old `clone_deferred` reason is removed from current UI, API metadata, and tests.

## Public Operation Contract

### Routes

- `POST /api/v1/sharing/shared-with-me/{share_id}/clone`
- `GET /api/v1/sharing/shared-with-me/{share_id}/clone/{operation_id}`

There are no aliases or redirects. Both routes return the same `SharedWorkspaceCloneOperationResponse`.

POST uses the existing Sharing write-rate policy and GET uses its read-rate policy. Rate-limit responses include `Retry-After`; the client treats them as polling backoff, not clone failure.

### Canonical Envelope

The clone response extends the existing Workspace operation shape without changing its serialized fields:

- `schema_version`: `1`
- `operation_id`: durable Job UUID; the API never exposes `job_id`
- `workspace_id`: deterministic target workspace UUID, available while queued
- `command`: `shared_workspace_clone`
- `status`: `queued`, `running`, `succeeded`, or `failed`
- `started_at`: admission timestamp, preserving the existing Workspace operation contract
- `updated_at`: last durable Job update
- `retryable`: the canonical Workspace operation retry flag; the UI offers a new-key `Try again` action only for a terminal failed operation whose flag is true
- `diagnostics`: bounded safe codes/counts only; no arbitrary Job payload
- `poll_href`: canonical clone status route
- `share_id`: bounded share correlation
- `progress`: typed progress object or `null`
- `result`: typed terminal result only for `succeeded`
- `error`: typed safe error only for `failed`

The backend introduces a shared Workspace operation base schema so the existing `WorkspaceOperationResponse` and clone response use the same field names and semantics without changing existing endpoint JSON. The frontend mirrors this with a shared base type and a strict clone parser.

The failed error shape is limited to stable `code`, translated-message key/fallback, and `cleanup_state` of `complete`, `pending`, or `unknown`. The worker reconciliation pass owns cleanup and records its bounded outcome through a scoped terminal-metadata update on the Job, which also advances `updated_at`. Recipient GETs remain side-effect free and project only persisted Job data.

### Status Mapping

Jobs statuses map as follows:

- `queued` -> `queued`
- `processing` -> `running`
- `completed` -> `succeeded` only when the typed result validates and records `publication_confirmed=true`
- `failed`, `cancelled`, or `quarantined` -> `failed`

Unknown states, ambiguous active/archive matches, or malformed terminal results return `503 clone_operation_unavailable`; they are never coerced to success. Archived terminal rows use the same mapping.

An operation with item-level copy failures may be `succeeded` with `result.outcome=partial`. `succeeded` means the clone worker reached a controlled terminal result, not that every retrieval mode is ready.

### Progress

Progress is bounded and content-free:

- phase: `queued`, `authorizing`, `preparing`, `sources`, `notes`, `artifacts`, or `finalizing`
- integer percent from 0 through 100
- stable message code translated by the client

The persisted Jobs fields remain `progress_percent` and `progress_message`; `progress_message` stores only the stable phase code. Item totals are reported in the terminal result rather than packed into an ad hoc string or a second status table.

No source title, note title, artifact name, URL, path, or content appears in progress.

### Result And Retrieval Readiness

The result contains:

- `outcome`: `complete` or `partial`
- target workspace ID and normalized name
- `publication_confirmed`: true only after the deterministic target was published before fenced Job completion
- attempted, copied, and failed counts by item class, plus operation-owned media created count
- retrieval readiness:
  - `text_search`: `ready` or `unavailable`
  - `citations`: `ready` or `unavailable`
  - `vector_search`: `ready`, `needs_indexing`, or `not_configured`
- at most eight warning entries, each a stable code plus bounded count

Readiness is computed from the target Workspace/source state after copy; it is not inferred only from the absence of exceptions. Copied chunks can make text search and citations ready. Vector search is `ready` only when the target's canonical source status confirms vectors. If vector retrieval is configured but cloned vectors were intentionally skipped, the operation returns `outcome=partial`, `vector_search=needs_indexing`, and warning `vector_index_not_generated`. If vector retrieval is disabled for the deployment, it returns `not_configured` without pretending indexing occurred.

Publication confirmation is historical. If the recipient later archives or deletes the copied Workspace, the clone operation remains succeeded; the status API does not rewrite history by treating later user action as clone failure.

This task does not enqueue a hidden embeddings process. The UI uses exact terminal copy such as: `Copy created. Text search and citations are ready; vector search still needs indexing.` `TASK-12020.45` owns durable clone reindex orchestration and promotion to fully ready vector retrieval.

## Worker Design

### Handler Boundary

`shared_workspace_clone_jobs_worker.py` owns payload validation, canonical access revalidation, database acquisition and release, progress translation, clone execution, result normalization, safe failure classification, cleanup, and terminal audit events.

`CloneService` remains synchronous and does not depend on JobManager. The async WorkerSDK handler calls the blocking clone boundary with `asyncio.to_thread`, and all clone-related content database handles are opened and closed inside that worker thread. No database handle crosses the thread boundary. Lease renewal and WorkerSDK control remain on the event loop.

`CloneService.clone_workspace()` accepts:

- caller-provided deterministic target workspace ID
- progress callback
- cooperative `should_cancel` callback

The service checks cancellation before target reservation, between each source, note, and artifact, and before finalization. A shutdown or WorkerSDK cancellation request becomes a controlled `clone_interrupted` failure. The worker discards the staged target and does not automatically retry it. If a hard process exit prevents cleanup, lease-expiry status maps to failed with `cleanup_state=unknown`; the recipient is told cleanup is pending rather than receiving a false clean-failure claim. The bounded reconciliation pass eventually records `cleanup_state=complete` in scoped terminal Job metadata without changing the terminal clone outcome.

### Snapshot Isolation And Cleanup

CloneService loads and validates source collections before target creation where practical. It does not use `add_media_with_keywords(overwrite=False)` for clone snapshots because that path may return and touch an existing recipient row. Linking a clone to that row could silently substitute different content or mutate it while copying transcripts.

The Media repository adds an explicit operation-owned clone-snapshot insert. It creates a new media row with an operation/source-scoped deterministic storage URL, preserves the original source URL as bounded provenance and on the Workspace source, and returns typed `created` ownership metadata. Re-entry for the same operation/source may return only the same operation-owned row after its content hash validates. It never deduplicates against, updates, touches, or appends transcripts to unrelated recipient media. A hash or ownership mismatch fails that source closed.

On fatal failure:

1. soft-delete the system-staged deterministic target Workspace;
2. delete only media carrying the same operation ownership marker and expected content identity;
3. never delete or mutate unrelated recipient media;
4. record a bounded `cleanup_incomplete` code when cleanup cannot be proven complete.

Raw media IDs, titles, URLs, and exception messages do not enter the recipient envelope. A user retry always uses a new key and target identity.

### Lifecycle Ownership

The worker is registered exactly once in the declarative `provide_primary_jobs_worker_specs()` inventory used by `startup_worker_groups`. The implementation does not extend legacy startup handles that are retained only for compatibility tests.

Enablement uses a dedicated predicate:

- Sharing route must be enabled.
- `SHARED_WORKSPACE_CLONE_JOBS_WORKER_ENABLED` defaults to `true`.
- explicit false disables the worker.
- application-owned execution is disabled in sidecar mode so a sidecar and application cannot consume the same queue unintentionally.

The spec uses shutdown phase `JOB_POLLER_QUIESCE`. New acquisition stops before service teardown. Cooperative cancellation is required, but an individual synchronous database call cannot be force-killed; lifecycle tests therefore verify bounded orchestration and cancellation checks rather than claiming every in-flight operation exits within five seconds.

`run_shared_workspace_clone_jobs_worker()` is the single runner used by both the application lifecycle and standalone module entry point.

### Source Snapshot Semantics

The copy represents a point-in-time snapshot taken when worker execution begins, not an unlabelled mixture of owner edits made during the run. After permission validation, the source ChaChaNotes and Media repositories open read-only repeatable snapshots using their backend-native transaction support. Target writes use separate recipient connections. Source Workspace metadata, membership rows, media content, chunks, and transcripts are read from those snapshots.

If either backend cannot establish the required read snapshot, a referenced source cannot be read consistently, or the snapshot is lost, the worker fails with `source_snapshot_unavailable` before publication. It does not silently continue from fresh reads. Permission is still rechecked outside the content snapshot between top-level items and immediately before publication, so a revocation cancels the operation even though the source snapshot remains readable.

## WebUI Design

### Canonical Client Ownership

Clone API calls live in `apps/packages/ui/src/services/tldw/domains/shared-workspaces.ts`, using bounded strict parsing and explicit `Idempotency-Key` headers. The clone flow does not use the loose generic `useSharing.jsonPost` response path.

The active route imports `apps/packages/ui/src/components/Option/SharedWithMe.tsx`. The dead duplicate `SharedWithMe/index.tsx` is removed, and a route/component contract test protects canonical ownership.

### Recovery Record

The client stores a versioned, bounded operation-attempt map before sending POST. The storage contract uses `tldw:sharing:clone-operations:v1`, a seven-day TTL, at most 32 entries, and a 32 KiB serialized cap. Each record contains only:

- schema version and expiry timestamp
- normalized server origin and authenticated principal scope
- share ID, normalized requested name, and idempotency key
- operation ID after one is known

The record is written before network submission so an ambiguous response can replay the same request and key. It is scoped using the existing server-origin/principal pattern, has a fixed TTL, is bounded to a small number of recent entries, and is cleared on logout, server change, or principal change. Corrupt or oversized data is discarded. Storage events synchronize active rows across tabs.

On reload:

- a record with an operation ID resumes GET polling;
- a pre-response record replays POST with the same key to recover the accepted operation;
- a definitive validation/authorization rejection clears the attempt;
- an ambiguous network or invalid post-commit response keeps the attempt for reconciliation.

When browser storage is unavailable, the server remains idempotent and correct, but reload continuity after a completely lost response cannot be guaranteed. The UI must not imply otherwise.

Terminal records retain the bounded share/operation pointer through the seven-day TTL so a reload can still show the result and `Open copy`. Once an operation ID is durable, the client removes the no-longer-needed request name and idempotency key from a terminal record. Explicit retry creates a fresh record and key.

### Row Interaction

Shared With Me renders operation state inline with the relevant row:

- queued/running: translated progress, `role=status`, `aria-live=polite`, and a disabled Clone command
- succeeded/complete: copied counts and `Open copy`
- succeeded/partial: readiness/warning summary and `Open copy`
- failed/retryable: safe error copy and `Try again`
- failed/non-retryable: safe error copy without a retry command

The progress indicator exposes `aria-valuemin`, `aria-valuemax`, and `aria-valuenow`. Polling starts at two seconds, backs off to at most five seconds after transient failures or `Retry-After`, refreshes on focus, pauses when the document is hidden, and stops at a terminal state. A shared scheduler limits the page to four in-flight status reads. Multiple rows may have independent operations.

If a share disappears from Shared With Me while an owner-readable operation pointer remains, the page keeps a compact operation-only row labeled `Shared workspace no longer available` until the operation reaches terminal state or its recovery record expires. This preserves failure/completion feedback after revocation without adding a page banner or restoring access to shared content.

`Open copy` routes directly to `/research-workspace?workspace=${encodeURIComponent(workspace_id)}`. Owner display name/username is shown instead of a raw numeric account ID. No page-level banner, trust bar, redirect, or new navigation surface is added.

Success is never announced at enqueue time. Toasts may supplement a terminal transition, but inline row state is authoritative.

## Audit Semantics

Audit events distinguish request admission from completion:

- `share.clone_requested`: emitted after durable Job/receipt admission
- `share.cloned`: emitted only after controlled successful completion, with complete/partial outcome and safe counts
- `share.clone_failed`: emitted after terminal failure with stable error code

Audit writes are best-effort and contain IDs, outcome codes, and counts only. Historical analytics labels may remain in historical data, but current code must not emit `share.cloned` at enqueue time.

## Failure Semantics

- Jobs unavailable during enqueue: typed `503`; no success state.
- Lost or malformed Job/receipt correlation: typed `503`; no second admission.
- Idempotency fingerprint mismatch: typed `409`; existing operation remains unchanged.
- Revoked or unauthorized share for a new request or at worker start: fail closed.
- Foreign operation status request: neutral `404`.
- Worker interruption or fatal error: bounded failed operation; partial target cleanup is attempted.
- Item-level copy failures: succeeded result with `outcome=partial`, exact counts, and bounded warning codes.
- Vector indexing absent: explicit `needs_indexing`, never unqualified full readiness.
- Browser storage unavailable or corrupt: discard the local record; server correctness remains intact, but lost-response reload continuity may be unavailable.

## Delivery Decomposition

`TASK-12020.41` remains the umbrella outcome. Implementation is split into reviewable tracks:

1. `TASK-12020.46`: generic Jobs idempotency receipts, active/archive reads, retention, RLS, and admission locking.
2. `TASK-12020.47`: source snapshot isolation, deterministic staged Workspace lifecycle, operation-owned Media snapshots, and cleanup. This foundation can run in parallel with `TASK-12020.46`.
3. `TASK-12020.48`: clone API, canonical operation projection, worker, capability policy, lifecycle registration, and audit integration after both backend foundations land.
4. `TASK-12020.49`: strict client, recovery persistence, and Shared With Me row UX. Contract-first frontend work can run in parallel with the backend foundations, then integrate against `TASK-12020.48`.
5. `TASK-12020.50`: live backend plus WebUI CDP acceptance after API/worker and frontend integration.

`TASK-12020.45` remains the separate post-clone vector reindex workstream. It does not block truthful partial-readiness delivery in `TASK-12020.41`.

## Verification

### Backend

- Jobs DB tests for atomic receipt admission, SQLite/PostgreSQL uniqueness races, active/archive replay, archive-read consistency, mismatch conflict, corrupt correlation fail-closed behavior, and retention rules.
- Unit tests for idempotency-key validation, request fingerprints, canonical status mapping, operation bounds, payload validation, and neutral owner isolation.
- Workspace DB, CloneService, and Media repository tests for deterministic reservation/publication, staged-list exclusion, collision fail-closed behavior, cooperative cancellation, operation-owned snapshot isolation, no mutation of URL/content collisions, transcript isolation, partial counts, cleanup/reconciliation, and retrieval readiness.
- API tests for new admission, active and archived replay, response-loss replay after revocation, same-share different-key concurrency conflict, mismatch conflict, owner isolation, rate limits, and exact POST/GET envelope parity.
- Worker tests for permission revalidation, thread offload, lease-renewal compatibility, progress, completion, partial readiness, interruption, cleanup, audit timing, and standalone/application handler parity.
- Lifecycle tests for queue allow-listing, route/flag/sidecar behavior, single declarative registration, inventory visibility, quiesce ordering, and bounded shutdown orchestration.
- Capability tests for `clone_workspace` allow/deny reason codes.

### Frontend

- Strict client tests for custom header propagation, response bounds, malformed-success rejection, and safe error parsing.
- Recovery tests for pre-submit persistence, same-key replay, terminal cleanup, TTL, corrupt/oversized storage, unavailable storage, principal/server scoping, and multi-tab synchronization.
- Component tests for independent row operations, polling/backoff/focus behavior, accessibility, complete/partial/failed copy, new-key retry, owner identity, and exact open-copy routing.
- Contract test proving the active route uses the canonical Shared With Me component after duplicate removal.

### Acceptance And Quality Gates

- Live backend + WebUI CDP acceptance covering completion, partial vector readiness, fatal failure, authorization, idempotent replay, archived replay, response-loss reload, and opening the target copy.
- Confirm target Workspace source status and grounded text/citation behavior against the live backend; do not infer readiness from UI state alone.
- Ruff/ESLint, focused type checks, OpenAPI drift, Jobs shard coverage, Bandit on touched Python, and `git diff --check`.

## Security And Privacy

Jobs and receipts are recipient-owned and contain no copied content or secrets. Worker authorization is resolved from current server state rather than trusted payload claims. Recipient APIs return bounded domain projections and neutral authorization failures. Logs and audit records use IDs, stable codes, counts, and exception classes only; they do not include source content, credentials, paths, URLs, titles, or raw exception messages.

## Follow-Up Boundary

Durable clone reindex orchestration is intentionally separated because the current source-ingest status worker verifies readiness but does not generate vectors, while existing embeddings Jobs rely on deployment-specific infrastructure. The follow-up must select a universally supported indexing primitive, expose it through the same Workspace source/readiness model, and let the clone operation or target Workspace transition from `needs_indexing` to `ready` without a hidden fire-and-forget task.
