# Notes Second-Brain Graph Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Graph a first-class shared Notes workspace and add an explicit, source-grounded, review-before-mutation workflow for related-note and tag suggestions.

**Architecture:** Keep the authoritative graph, canonical Notes links, and keyword relationships unchanged. Add owner-bound provisional state to ChaChaNotes schema v64, execute one retry-disabled provider call through a dedicated Jobs queue, publish only after verifying the exact terminal Job receipt, and accept suggestions through guarded extensions to the existing Sync-aware coordinators. The shared UI composes authoritative graph responses with provisional overlays and exposes the same workflow in the WebUI and browser extension.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL, ChaChaNotes stores, Sync v2 materializers, Jobs/WorkerSDK, existing LLM adapters and structured generation helpers, React 18, TanStack Query, Cytoscape/Dagre, Ant Design, lucide-react, Vitest/Testing Library/axe, Playwright, pytest, Hypothesis, Ruff, Bandit.

---

## Scope And Execution Rules

- Approved design: `Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md`.
- Backlog authority: `TASK-13138`. Keep its status, notes, touched files, verification, and PR link current through Backlog.md MCP/CLI; do not manually edit the task file.
- Deferred work remains in `TASK-13134` through `TASK-13137`: embeddings/semantic edges, background organization, recurring themes, and saved layouts.
- Use @superpowers:test-driven-development for every implementation task, @superpowers:systematic-debugging for unexpected failures, @superpowers:requesting-code-review after the integrated implementation, and @superpowers:verification-before-completion before commits/PR handoff.
- Activate `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv` before Python, pytest, Ruff, or Bandit commands.
- Keep SQL in `app/core/DB_Management/`; core services consume typed store methods.
- Never place note text, excerpts, rationales, candidate IDs, raw prompts/responses, credentials, or endpoint URLs in Jobs payloads/results/logs.
- Each task ends in a focused commit after its listed tests pass. Rebase on current `origin/dev` before implementation and recompute the next ChaChaNotes/AuthNZ migration numbers if either baseline advanced.

## File Map

### Persistence And Domain Logic

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: schema v64 migration/bootstrap, store construction, and backend parity.
- `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_models.py`: immutable persistence records and closed state/operation enums.
- `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py`: all owner-scoped SQL for analysis reads, runs, receipts, staged publication, suggestions, evidence offsets, rejection sets, leases, invalidation, and cleanup.
- `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`: invoke suggestion invalidation in the same transaction as note edit/trash/hard-delete transitions.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_content.py`: canonical title/body text, SHA-256 fingerprints, evidence offsets, overlap checks, and token-budget helpers.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py`: deterministic term selection and typed orchestration over backend-specific store queries.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_capabilities.py`: provider/model resolution, policy/readiness, data-boundary disclosure, limits, endpoint-origin digest, and ETag revision.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_generation.py`: prompt contract, strict schema parsing, allowlist validation, duplicate filtering, and one retry-disabled provider dispatch.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_jobs.py`: closed Jobs constants and content-free payload/result builders.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_observability.py`: allowlisted structured lifecycle events and local-only metric helpers with content-free dimensions.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_service.py`: admission, status/list envelopes, cancellation, receipt continuation, staging, publication, and supersession orchestration.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_decisions.py`: reject/reset and fenced accept/reconciliation semantics.
- `tldw_Server_API/app/core/Notes_Graph/suggestion_maintenance.py`: bounded provider-independent reconciliation and retention pass.

### Sync, Jobs, API, And Lifecycle

- `tldw_Server_API/app/core/Sync/v2/materializers/guarded_product_mutation.py`: internal callback contract for a guard and finalizer that execute inside one ChaCha product transaction.
- `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`, `service.py`, and `materializers/base.py`: explicitly thread an optional guarded product mutation only for trusted internal coordinator calls.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_link.py`, `materializers/notes_organization.py`, `notes_link_coordinator.py`, and `notes_organization_coordinator.py`: guarded link/keyword mutation entry points with unchanged public defaults.
- `tldw_Server_API/app/core/DB_Management/chacha/note_link_store.py` and `organization_sync_store.py`: execute the supplied guard before canonical mutation and finalizer after the exact postcondition, inside the same transaction.
- `tldw_Server_API/app/core/Jobs/worker_sdk.py`: opt-in completion-token binding for this worker; default behavior remains compatible.
- `tldw_Server_API/app/core/Jobs/manager.py`: enforce the `notes`/`graph-suggestions` terminal-receipt retention floor and forced archive behavior without lengthening unrelated Jobs retention.
- `tldw_Server_API/app/services/notes_graph_suggestions_worker.py`: one handler shared by app-managed and standalone execution, plus post-completion publication callbacks.
- `tldw_Server_API/app/services/notes_graph_suggestions_maintenance.py`: independent startup/periodic reconciliation lifecycle.
- `tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py` and lifecycle registration tests: register the Jobs worker without duplicate app/sidecar ownership.
- `tldw_Server_API/app/api/v1/schemas/notes_graph_suggestions.py`: bounded request/response contracts.
- `tldw_Server_API/app/api/v1/endpoints/notes_graph_suggestions.py`: nested capabilities, runs, cancellation, listing, reset, accept, and reject routes.
- `tldw_Server_API/app/api/v1/router_groups/content.py`: mount the suggestion router under `/api/v1/notes`.
- `tldw_Server_API/app/core/AuthNZ/permissions.py`, `settings.py`, and `migrations.py`: define and seed `notes.graph.suggest`, `notes.link_keyword`, and `keywords.create`; grant them only to approved roles.

### Shared UI

- `apps/packages/ui/src/services/note-graph-suggestions.ts`: typed nested API client, ETag handling, idempotency headers, and stable error parsing.
- `apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx`: authoritative graph loading, expansion, loaded-node search, filters, focus, layouts, and offline-last-good state.
- `apps/packages/ui/src/components/Notes/hooks/useNotesGraphSuggestions.tsx`: capability preflight, run polling, decisions, cancellation, cache invalidation, and provisional overlays.
- `apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx`: responsive three-region workspace and canvas/relationships mode switch.
- `apps/packages/ui/src/components/Notes/NotesGraphCanvas.tsx`: Cytoscape lifecycle and authoritative/provisional visual composition.
- `apps/packages/ui/src/components/Notes/NotesGraphToolbar.tsx`: stable-size search/focus/filter/layout/fit/view controls.
- `apps/packages/ui/src/components/Notes/NotesGraphInspector.tsx`: Details/Suggestions tabs, disclosure, status, evidence, decisions, and reset confirmation.
- `apps/packages/ui/src/components/Notes/NotesGraphRelationshipsView.tsx`: canonical grouped sort and client pages of at most 100 rows.
- `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`, `NotesSidebar.tsx`, `NotesEditorPane.tsx`, `hooks/useNotesListManagement.tsx`, `hooks/useNotesEditorState.tsx`, `notes-manager-utils.ts`, and `NotesManagerOverlays.tsx`: add Graph mode, route editor graph action into it, and retire modal state.
- Delete `apps/packages/ui/src/components/Notes/NotesGraphModal.tsx` only after parity tests pass.

## Task 1: Add ChaChaNotes Schema v64 And Typed Store Skeleton

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_models.py`
- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py`
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v64.py`

- [ ] **Step 1: Write failing SQLite migration tests**

Assert v63-to-v64 and fresh-v64 creation of the four design tables, evidence table, constraints, partial active-run uniqueness, lease/retention indexes, foreign-key cascades, and schema version. Include rollback-on-failure by injecting a statement error and asserting the database remains at v63.

- [ ] **Step 2: Write failing PostgreSQL parity tests**

Use the existing backend stubs/live fixture pattern to assert equivalent columns/checks/indexes, `ENABLE ROW LEVEL SECURITY`, `FORCE ROW LEVEL SECURITY`, and owner policy predicates for every new table.

- [ ] **Step 3: Run the red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v64.py -q
```
Expected: FAIL because schema v64 and its tables do not exist.

- [ ] **Step 4: Implement the migration and records**

Define closed literals/enums for run, suggestion, operation, and receipt states. Add v64 SQLite/PostgreSQL migration methods and fresh-schema ensures to `CharactersRAGDB`; instantiate `NoteGraphSuggestionStore(self)`. Store only the fields approved in the spec, use portable booleans/timestamps, and enforce owner/dataset/resource uniqueness in the database.

- [ ] **Step 5: Run migration tests and existing v61-v63 regression tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v61.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v62.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v62.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v64.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_migration_v64.py
git commit -m "feat: add Notes graph suggestion persistence (TASK-13138)"
```

## Task 2: Implement Canonical Content, Byte Guards, FTS Retrieval, And Evidence

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_content.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py`

- [ ] **Step 1: Write failing canonicalization/property tests**

Cover CRLF/LF normalization, NFC, astral Unicode, title/body separator offsets, SHA-256 parity fixtures, bounded excerpt reconstruction, invalid offsets, and exact UTF-8 byte counts. Use Hypothesis to prove reconstructed windows never cross canonical field boundaries or exceed configured limits.

- [ ] **Step 2: Write failing retrieval tests**

Seed active, trashed, directly linked, shared-tag, and shared-source notes. Assert selected/trash/direct-link exclusion only, at most 24 deterministic retrieval terms, 60-row backend overfetch, stable rank tie-breaking, 30-candidate pruning, 100-tag catalog cap, projection freshness reporting, and oversized-candidate aggregate counts.

- [ ] **Step 3: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py -q
```
Expected: FAIL with missing suggestion modules/store methods.

- [ ] **Step 4: Implement backend-portable reads**

In the DB store, use SQLite `length(CAST(title AS BLOB)) + length(CAST(content AS BLOB))` and PostgreSQL `octet_length(title) + octet_length(content)` before transferring text. Enforce 1,000,000 selected-note bytes and 250,000 bytes per candidate before content transfer. Keep FTS5 and `notes_fts_tsv` rank SQL in the store. In core, derive deterministic terms, canonical fingerprints, at most four selected-note/two per-candidate windows of at most 480 Unicode code points, and token estimates without truncating an oversized selected note.

- [ ] **Step 5: Run unit and backend integration tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_content.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_retrieval.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_retrieval_backends.py -q
```
Expected: PASS (PostgreSQL cases may skip only through the established unavailable fixture).

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/suggestion_content.py tldw_Server_API/app/core/Notes_Graph/suggestion_retrieval.py tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: add bounded Notes suggestion retrieval (TASK-13138)"
```

## Task 3: Implement Durable Runs, Receipts, Publication, And Lifecycle Invalidations

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_store.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_lifecycle.py`

- [ ] **Step 1: Write failing store state-machine tests**

Test active-run uniqueness, canonical request fingerprints, terminal versus in-progress receipt replay, request mismatch, staged invisibility, atomic activation, supersession, rejection-set revision CAS, five-minute acceptance leases with monotonic fences, opaque cursor pagination, and exact retention horizons: 30 days for obsolete/stale/failed/cancelled detail, 90 days for terminal receipts and accepted/success audit metadata, and no age-only expiry for current-fingerprint pending/rejection state.

- [ ] **Step 2: Write failing lifecycle transaction tests**

Prove source and target edits/trash/hard-delete produce the exact state transitions in the spec in the same note transaction. Explicitly assert tag membership changes do not alter the content fingerprint or stale sibling suggestions.

- [ ] **Step 3: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_store.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_lifecycle.py -q
```
Expected: FAIL because transition methods and hooks are absent.

- [ ] **Step 4: Implement CAS-based store operations**

Every transition must include owner, dataset, current state, and revision/fence in its predicate. Reject/reset finalize their receipt in the same transaction. Evidence rows store only note/field/fingerprint/half-open offsets. Activation reloads all note fingerprints and tag identities before exposing any row.

- [ ] **Step 5: Integrate note lifecycle hooks**

Call `note_graph_suggestion_store.invalidate_for_note_change(...)` from `NoteStore` while its existing transaction is open. Persist `cancelling` plus the stable cancellation operation identity in that transaction; the independent maintenance service sends the idempotent Jobs cancellation command after commit. Never call Jobs from inside a ChaChaNotes transaction.

- [ ] **Step 6: Run tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_store.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_lifecycle.py tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py -q
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py tldw_Server_API/app/core/DB_Management/chacha/note_store.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: enforce Notes suggestion lifecycle invariants (TASK-13138)"
```

## Task 4: Add Capability Disclosure And Strict One-Call Generation

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_capabilities.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_generation.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_capabilities.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_generation.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py`

- [ ] **Step 1: Write failing capability tests**

Assert the revision changes for adapter/model, canonical endpoint origin digest, policy, data boundary, outbound categories, effective limits, or prompt version; does not change for credential values/health heartbeats; treats unknown boundary as external; and reports safe `generation_available=false` reasons.

- [ ] **Step 2: Write failing provider/validator tests**

Cover prompt injection text, unknown IDs/evidence, malformed top-level schema, duplicate pair/tag items, overlong rationales above 240 Unicode code points, normalized contiguous evidence overlap above 12 words, existing/new tag normalization, five-related/five-tag output caps, two-new-tag cap, 100-tag catalog, 24,000 estimated input tokens, 2,000 output tokens, 120-second timeout, one response candidate, no tools/stream/stop, `max_transport_attempts=1`, privacy-safe errors, and cross-origin redirect rejection. Assert the server, not the model, computes the exact Strong/Possible match rules from FTS rank, evidence, term overlap, and tag phrase occurrence.

- [ ] **Step 3: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_capabilities.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_generation.py tldw_Server_API/tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py -q
```
Expected: FAIL with missing capability/generation modules.

- [ ] **Step 4: Implement the closed provider contract**

Reuse `ProviderCallPolicy` and `perform_chat_api_call_async`. Resolve structured response mode where supported, but always parse and validate the returned JSON locally. The prompt contains only allowlisted IDs/text windows/tag labels and explicitly delimits note text as untrusted data. Unsupported adapters fail preflight rather than silently falling back.

- [ ] **Step 5: Run tests including existing no-retry coverage**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_capabilities.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_generation.py tldw_Server_API/tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py tldw_Server_API/tests/LLM_Adapters/unit/test_provider_unsafe_post_no_retry.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/suggestion_capabilities.py tldw_Server_API/app/core/Notes_Graph/suggestion_generation.py tldw_Server_API/tests/Notes_Graph tldw_Server_API/tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py
git commit -m "feat: add Notes suggestion provider contract (TASK-13138)"
```

## Task 5: Add Jobs Admission, Worker, Receipt-Bound Publication, And Maintenance

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_jobs.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_observability.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_service.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_maintenance.py`
- Create: `tldw_Server_API/app/services/notes_graph_suggestions_worker.py`
- Create: `tldw_Server_API/app/services/notes_graph_suggestions_maintenance.py`
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py`
- Test: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_jobs.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_observability.py`
- Test: `tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py`

- [ ] **Step 1: Write failing payload/admission tests**

Assert domain `notes`, queue `graph-suggestions`, type `note_graph_suggestions`, stable run UUID idempotency, `max_retries=0`, authoritative owner only on the Job row, the exact payload allowlist, one active generation run per owner, 20 admissions per owner per hour, equivalent-active-run conflict, and no provider invocation during admission replay/continuation.

- [ ] **Step 2: Write failing crash-window and receipt tests**

Cover interruption before enqueue, after staging, after Job completion, after activation, active/archive lookup, mismatched token/digest/run/owner, missing receipt before and after 30 days, cancellation before/after call, and freshness changes at worker and activation time. In both Jobs backends, backdate matching terminal Jobs to 29, 30, and 31 days and prove global/domain prune requests shorter than 30 days cannot delete them, while eligible rows are forced into `jobs_archive` even when `JOBS_ARCHIVE_BEFORE_DELETE` is disabled; unrelated queues retain their configured behavior.

- [ ] **Step 3: Write failing worker lifecycle tests**

Assert app/sidecar ownership prevents duplicate consumers, worker revalidates capability immediately before its one call, the Jobs completion token is always persisted for this worker, and maintenance starts independently of provider readiness and claims at most 100 rows once per minute. Add non-vacuous observability tests for the exact allowlisted event names `run_admitted`, `shortlist_completed`, `provider_started`, `provider_completed`, `validation_rejected`, `staged`, `published`, `cancelled`, `failed`, `accepted`, `rejected`, `stale`, and `reconciled`; assert local metric helpers record queue latency, run duration, candidate/evidence counts, bounded provider usage, validated/dropped counts, run error codes, decision outcomes, and acceptance-reconciliation outcomes without note-derived labels.

- [ ] **Step 4: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_observability.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_jobs.py tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py -q
```
Expected: FAIL with missing Jobs service/worker modules.

- [ ] **Step 5: Implement admission and worker flow**

Add an opt-in `bind_completion_token=True` field to `WorkerConfig` and use the acquired lease ID for success/failure terminalization when set. The handler transitions queued to running, retrieves, invokes once, validates, and stages. `on_completed` reads `get_job_or_archived_by_uuid(uuid, domain='notes', owner_user_id=...)` outside the ChaCha transaction and activates only an exact immutable receipt. Route every lifecycle transition through `suggestion_observability.py`; its event and metric APIs accept closed enums plus safe IDs/counts/timing/error codes only and use the existing local metrics manager without enabling telemetry export.

- [ ] **Step 6: Implement independent maintenance**

Reconcile `admitting`, `queued`, `running`, `cancelling`, `publishing`, expired accepting leases, and in-progress cancellation receipts with row-revision/lease CAS. Cleanup remains bounded and preserves current-fingerprint rejections and 90-day receipts. In `JobManager.prune_jobs`, apply a 30-day minimum cutoff to terminal `notes`/`graph-suggestions`/`note_graph_suggestions` rows even when a broader prune uses a shorter retention, and force exact matching rows into `jobs_archive` before deletion; do not change retention for any other domain/queue/type.

- [ ] **Step 7: Run tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_worker_sdk.py tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_observability.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_jobs.py tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py -q
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph tldw_Server_API/app/core/Jobs/worker_sdk.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/services/notes_graph_suggestions_worker.py tldw_Server_API/app/services/notes_graph_suggestions_maintenance.py tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py tldw_Server_API/tests/Jobs/test_worker_sdk.py tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py tldw_Server_API/tests/Notes_Graph tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py
git commit -m "feat: run Notes suggestions through durable Jobs (TASK-13138)"
```

## Task 6: Add The Guarded Sync Product-Mutation Primitive

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/guarded_product_mutation.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/base.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/notes_link.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/notes_organization.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_link_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/notes_link_coordinator.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py`
- Create test: `tldw_Server_API/tests/Sync/test_sync_v2_guarded_product_mutation.py`
- Modify test: `tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py`
- Modify test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_capture.py`
- Modify test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_materializer.py`
- Modify test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py`

- [ ] **Step 1: Write failing atomicity/fencing tests**

Use transaction barriers to prove: a valid guard permits the canonical write and finalization together; an expired/replaced fence writes neither; a finalizer failure rolls back the canonical relationship; exact replay observes the postcondition; and ordinary callers without a guard are unchanged.

- [ ] **Step 2: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_guarded_product_mutation.py -q
```
Expected: FAIL because no guarded materialization contract exists.

- [ ] **Step 3: Implement an explicit, non-persisted guard object**

Define a closed `GuardedProductMutation` carrying the expected domain/object identity plus `before(conn)` and `after(conn, resource_identity)` callbacks. Thread it only through the synchronous server-origin materialization call. Reject domain/object mismatches; never serialize callbacks into envelopes or Jobs. Keep the modified `materializers/base.py` import block Ruff-clean when adding the new contract.

- [ ] **Step 4: Execute callbacks in product transactions**

Extend link upsert and organization resource/relationship apply with optional callbacks. Invoke `before` after locking/validating product rows but before mutation, and `after` only after the exact canonical postcondition exists. For new tags, guard keyword creation without finalizing; finalize only in the guarded note-keyword relationship transaction.

- [ ] **Step 5: Run focused and regression tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_guarded_product_mutation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2 tldw_Server_API/app/core/DB_Management/chacha/note_link_store.py tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py tldw_Server_API/tests/Sync tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py
git commit -m "feat: add fenced Notes Sync mutations (TASK-13138)"
```

## Task 7: Implement Accept, Reject, Reset, And Acceptance Reconciliation

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/suggestion_decisions.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/suggestion_service.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/suggestion_maintenance.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_decisions.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_acceptance.py`

- [ ] **Step 1: Write failing decision race tests**

Cover accept versus reject, edit, regeneration, external manual mutation, duplicate pending suggestions, existing-link/tag success, new-tag two-step crash, lease expiry/takeover, old-fence late worker, stale client fingerprints, and exact terminal receipt replay. Cover existing-tag rename resolving the current display value, merge following the surviving portable identity and requiring the selected note's final membership, deletion making the suggestion stale, and concurrent new-tag name collision converging on the existing normalized keyword.

- [ ] **Step 2: Write failing rejection/reset tests**

Assert pair/tag suppression is independent of provider/model/prompt/rationale, compact rejection removes evidence/rationale, 2,000-key cap fails closed, reset uses source fingerprint and rejection-set revision, and replay of an old reset cannot remove a later rejection.

- [ ] **Step 3: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_decisions.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_acceptance.py -q
```
Expected: FAIL with missing decision service.

- [ ] **Step 4: Implement decisions through coordinators**

Relationship acceptance creates an undirected manual link with weight `1.0`, null label, and empty properties. Tag acceptance resolves portable keyword identity/name collisions and uses `NotesOrganizationCoordinator`. Stable suggestion-derived idempotency keys are scoped per canonical mutation step; decision finalization occurs only from the guarded product transaction. Emit the closed `accepted`, `rejected`, and `stale` events and corresponding content-free decision metrics only after durable state is known.

- [ ] **Step 5: Implement bounded reconciliation**

For expired accepting leases, claim with a higher fence and only finalize an existing exact postcondition, mark stale, or return to pending. Reconciliation never creates canonical state. Emit an allowlisted acceptance-reconciliation outcome plus the `reconciled` event for each durable resolution.

- [ ] **Step 6: Run tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_suggestion_decisions.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_acceptance.py tldw_Server_API/tests/Sync/test_sync_v2_guarded_product_mutation.py -q
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/suggestion_decisions.py tldw_Server_API/app/core/Notes_Graph/suggestion_service.py tldw_Server_API/app/core/Notes_Graph/suggestion_maintenance.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: add review-safe Notes suggestion decisions (TASK-13138)"
```

## Task 8: Expose Nested API, RBAC, Rate Limits, And Stable Errors

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/notes_graph_suggestions.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/notes_graph_suggestions.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/graph_service.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_suggestion_permissions.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_endpoints.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_route_order.py`

- [ ] **Step 1: Write failing schema and route tests**

Test all nested paths from the spec, bounded fields, ETag/If-Match, Idempotency-Key, opaque cursors, 1..100 limits, default pending/accepting filter, non-enumerating 404s, stable 409/412/422/429/503 codes, and durable status after 202. Extend the ordinary graph-read response with owner-scoped `active_note_count`, effective `all_notes_note_cap` (configured default 100, clamped to effective `max_nodes`), and `all_notes_eligible`; prove these fields are available with `notes.graph.read` even when `notes.graph.suggest` is absent and cannot leak another owner's count.

- [ ] **Step 2: Write the static-route regression first**

Mount the router and assert `POST /{note_id}/graph/suggestions/rejections/reset` invokes reset and is never parsed as `{suggestion_id}`. Keep the static route declaration before dynamic accept/reject routes.

- [ ] **Step 3: Write failing RBAC tests**

Add `NOTES_GRAPH_SUGGEST = 'notes.graph.suggest'`, `NOTES_LINK_KEYWORD = 'notes.link_keyword'`, and `KEYWORDS_CREATE = 'keywords.create'`. Seed the catalog through the next AuthNZ migration and update single-user defaults. Generation/read/reject/reset require graph read + suggest; relationship accept additionally requires graph write; tag accept checks `NOTES_LINK_KEYWORD`, and new-tag accept also checks `KEYWORDS_CREATE`. Grant suggest/tag decision permissions to admin and standard Notes-writing roles, never read-only roles. This task does not retroactively tighten the legacy keyword endpoints; it gives the new decision route explicit, assignable RBAC gates without a compatibility break.

- [ ] **Step 4: Run red tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_suggestion_permissions.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_endpoints.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_route_order.py -q
```
Expected: FAIL because schemas/routes/permission seed are absent.

- [ ] **Step 5: Implement thin endpoints**

Endpoints normalize IDs and headers, resolve the owner-bound DB/service, apply permission/token-scope/rate-limit dependencies, and map typed domain errors. Keep state machines and SQL out of endpoint code. Capabilities returns 200 for expected readiness failures and sets the opaque revision as `ETag`. Have `NoteGraphService` compute the active count through the existing owner-bound note store and return the effective All-notes cap/eligibility in every authoritative graph response, including centered requests; do not put this graph-view metadata behind suggestion capabilities. Remove the already-unused `is_single_user_mode` import while modifying `permissions.py`, and use `X | None` for any new setting annotation rather than extending that file's legacy `UP045` baseline.

- [ ] **Step 6: Run API/OpenAPI tests**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_suggestion_permissions.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_endpoints.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_route_order.py tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py -q
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/notes_graph.py tldw_Server_API/app/api/v1/schemas/notes_graph_suggestions.py tldw_Server_API/app/api/v1/endpoints/notes_graph.py tldw_Server_API/app/api/v1/endpoints/notes_graph_suggestions.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/core/Notes_Graph/graph_service.py tldw_Server_API/app/core/AuthNZ tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_suggestion_permissions.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: expose nested Notes graph suggestions API (TASK-13138)"
```

## Task 9: Add The Typed Shared Client And Query Hooks

**Files:**
- Create: `apps/packages/ui/src/services/note-graph-suggestions.ts`
- Create: `apps/packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts`
- Create: `apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx`
- Create: `apps/packages/ui/src/components/Notes/hooks/useNotesGraphSuggestions.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/useNotesGraphSuggestions.test.tsx`

- [ ] **Step 1: Write failing client contract tests**

Assert exact nested URLs, ETag extraction, If-Match, Idempotency-Key, query encoding, bounded response normalization, 412 refresh behavior, and sanitized stable-error parsing. Verify no endpoint/credential field is accepted by public client request types.

- [ ] **Step 2: Write failing hook tests**

Use fake timers to test focused-neighborhood loading, explicit cursor expansion, All-notes eligibility from the authoritative graph response's `active_note_count`, effective `all_notes_note_cap`, and `limits.max_nodes`, loaded-node-only search, session-local filters/layout, last-good offline state, run polling, terminal stop, cancellation, decision invalidation, and no mutations offline. Unmount/remount the suggestion hook with no in-memory run ID, return one active run from the run-list endpoint, and prove the hook selects the newest matching nonterminal run, resumes polling its detail endpoint, and stops at terminal state without creating another run.

- [ ] **Step 3: Run red tests**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx packages/ui/src/components/Notes/__tests__/useNotesGraphSuggestions.test.tsx
```
Expected: FAIL with missing modules.

- [ ] **Step 4: Implement typed clients and hooks**

Generate UUID idempotency keys once per user command and retain them across network retries. On initial load and note-focus changes, list current runs, deterministically adopt the newest matching nonterminal run, and poll its detail endpoint; never require an in-memory admission response to recover after reload. Poll only nonterminal runs, preserve the last authoritative graph on refresh failure, derive All-notes control state from server response metadata rather than a client constant, and expose provisional edges/nodes as a separate derived collection keyed by suggestion ID.

- [ ] **Step 5: Run tests**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx packages/ui/src/components/Notes/__tests__/useNotesGraphSuggestions.test.tsx
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/services/note-graph-suggestions.ts apps/packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx apps/packages/ui/src/components/Notes/hooks/useNotesGraphSuggestions.tsx apps/packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx apps/packages/ui/src/components/Notes/__tests__/useNotesGraphSuggestions.test.tsx
git commit -m "feat: add shared Notes graph suggestion client (TASK-13138)"
```

## Task 10: Replace The Modal With A First-Class Graph Workspace

**Files:**
- Create: `apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx`
- Create: `apps/packages/ui/src/components/Notes/NotesGraphCanvas.tsx`
- Create: `apps/packages/ui/src/components/Notes/NotesGraphToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesSidebar.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`
- Modify: `apps/packages/ui/src/components/Notes/notes-manager-utils.ts`
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerOverlays.tsx`
- Delete: `apps/packages/ui/src/components/Notes/NotesGraphModal.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.view-mode.test.tsx`
- Move test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphModal.stage2.graph-view.test.tsx` -> `apps/packages/ui/src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx`
- Modify test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx`

- [ ] **Step 1: Write failing view-mode tests**

Assert Graph appears with List/Timeline/Inbox/Moodboard, remains on the Notes route, keeps sidebar note selection, focuses selected or most-recent active note, shows the normal empty state without notes, enables All notes only when the server-reported `all_notes_eligible` is true and the reported active count is within both the effective `all_notes_note_cap` and `limits.max_nodes`, renders the server's concise disabled reason when false, and converts the editor's Open graph action into a mode/focus transition rather than a dialog. Include a non-default cap fixture to prevent regression to a hardcoded 100.

- [ ] **Step 2: Write failing canvas/control tests**

Assert fixed control dimensions, icon labels/tooltips, search only over loaded nodes, focus-current, edge toggles, session layouts, fit, explicit expansion, labels only for focused/selected/hovered nodes, directed arrowheads only, provisional dashed edges/ephemeral nodes, and truncation/degraded/offline states.

- [ ] **Step 3: Run red tests**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.view-mode.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx
```
Expected: FAIL because Graph is still a modal.

- [ ] **Step 4: Build the first-class workspace**

Extend `NotesListViewMode` with `graph`, but keep the Notes list query/sidebar available. In `NotesManagerPage`, render the graph workspace in the editor region when active. Extract and reuse the current Cytoscape/Dagre setup, destroy instances on mode exit, and preserve existing color tokens while adding non-color provisional styling.

- [ ] **Step 5: Remove modal state after parity passes**

Delete modal imports/rendering/state/refocus code only after the workspace tests cover the prior radius/limits/zoom/fit/navigation behavior. Rename the modal test instead of losing its assertions.

- [ ] **Step 6: Run focused Notes tests**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.view-mode.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage5.graph-panels.test.tsx packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage6.manual-links.test.tsx packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Notes apps/packages/ui/src/components/Notes/hooks
git commit -m "feat: make Notes graph a first-class view (TASK-13138)"
```

## Task 11: Add The Grounded Inspector And Accessible Relationships View

**Files:**
- Create: `apps/packages/ui/src/components/Notes/NotesGraphInspector.tsx`
- Create: `apps/packages/ui/src/components/Notes/NotesGraphRelationshipsView.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphCanvas.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/option.json`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json` through the repository locale-sync script
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphInspector.suggestions.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx`

- [ ] **Step 1: Write failing inspector tests**

Cover Details/Suggestions tabs; provider/model/boundary/outbound disclosure before Generate; all run states; Strong/Possible text labels; bounded source/target evidence; existing/new tag distinction; per-item accept/reject; cancellation; reset confirmation; escaped provider text; and authoritative refresh after acceptance. Prove a graph reader without `notes.graph.suggest` sees the first-class Graph workspace and Details tab but no Suggestions tab, generation affordance, provisional overlay, or suggestion API request.

- [ ] **Step 2: Write failing relationships/a11y tests**

Assert canonical grouped sorting, incoming/outgoing direction, counterpart focus, pages of at most 100, retained selection/group/page focus, equivalent evidence/decisions, keyboard operation, screen-reader status, visible focus, reduced motion, and no color-only states.

- [ ] **Step 3: Write failing responsive tests**

At desktop, assert sidebar/canvas/fixed inspector without nested cards. At narrow widths, assert canvas primary plus bounded in-page bottom inspector, no routine modal, stable canvas height, long title/tag wrapping, 200% zoom, and no control overlap.

- [ ] **Step 4: Run red tests**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/components/Notes/__tests__/NotesGraphInspector.suggestions.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx
```
Expected: FAIL with missing inspector/relationships components.

- [ ] **Step 5: Implement review UI and focus management**

Use ordinary text rendering for all model-derived fields. Keep decisions enabled per capability/permission and online state. Announce status changes through one polite live region, return focus to the originating row after decisions, and never resize canvas controls for loading/status labels. Add canonical English strings, run `bun --cwd apps/extension run locales:sync`, and run the existing Notes terminology locale contracts rather than relying only on inline defaults.

- [ ] **Step 6: Run tests**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/components/Notes/__tests__/NotesGraphInspector.suggestions.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx
```
Expected: PASS with no serious axe violations.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Notes/NotesGraphInspector.tsx apps/packages/ui/src/components/Notes/NotesGraphRelationshipsView.tsx apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx apps/packages/ui/src/components/Notes/NotesGraphCanvas.tsx apps/packages/ui/src/components/Notes/__tests__ apps/packages/ui/src/assets/locale/en/option.json apps/packages/ui/src/public/_locales
git commit -m "feat: add grounded Notes graph suggestion review (TASK-13138)"
```

## Task 12: Prove WebUI/Extension Parity, Quality, Privacy, And Documentation

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/option-notes-route-identity.test.tsx`
- Modify: `apps/extension/tests/e2e/notes-ux.spec.ts`
- Create: `apps/tldw-frontend/e2e/workflows/notes-graph-suggestions.spec.ts`
- Create: `tldw_Server_API/tests/Notes_Graph/evaluation/fixtures/suggestion_grounding_cases.json`
- Create: `tldw_Server_API/tests/Notes_Graph/evaluation/test_suggestion_quality_corpus.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/README.md`
- Modify: `Docs/Product/Graphing-Notes-PRD.md`
- Modify: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Modify: `Docs/Published/Code_Documentation/Data_Flow_Atlas.md`
- Create: `Docs/API/Notes_Graph_Suggestions.md`
- Modify: `backlog/tasks/task-13138 - Implement-first-class-Notes-graph-workspace-and-reviewable-AI-suggestions.md` through Backlog.md MCP/CLI only

- [ ] **Step 1: Add privacy/observability contract tests**

Drive representative admission, shortlist, provider success/failure, validation rejection, staging/publication, cancellation, acceptance, rejection, staleness, and maintenance-reconciliation paths. Assert every required event from Task 5 is emitted at least once with only safe run/Job/suggestion IDs, counts, timing, bounded usage, and stable error codes. Assert every required local metric family records a sample with only closed outcome/error dimensions; then scan captured Jobs payload/result, structured logs, metric names/labels, run rows, and receipt envelopes for forbidden content. This test must fail when the event/metric calls are removed and must confirm the feature does not enable a telemetry exporter.

- [ ] **Step 2: Add the offline quality corpus**

Include medical, technical, research, and general direct/weak matches; distractors; existing/new tags; injection attempts; Unicode; shared-tag/source candidates; unknown IDs; unsupported evidence; duplicate outputs; and tag normalization cases. With deterministic recorded provider responses and no external credentials, enforce the exact release gate: at least 90 percent expected-target recall in deterministic FTS top 30; 100 percent evidence-reference validity; zero cross-owner, unknown-candidate, already-linked, or rejected-pair output; 100 percent tag normalization and duplicate suppression; and bounded prompt/output behavior on the largest fixtures.

- [ ] **Step 3: Add shared-route and E2E tests**

Prove both route shells import the same shared `NotesManagerPage`. Exercise desktop and mobile Graph mode, capability disclosure, generation polling, page reload followed by active-run discovery/resumed polling without duplicate admission, provisional overlays, accept/reject/reset, a read-only graph user with no Suggestions tab, non-default All-notes cap gating, Relationships mode, offline disabled decisions, and long-content layouts in WebUI and extension harnesses.

- [ ] **Step 4: Run backend feature suite**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph tldw_Server_API/tests/Services/test_notes_graph_projection_worker.py tldw_Server_API/tests/Services/test_notes_graph_suggestions_workers.py tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_permissions_claims.py tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_suggestion_permissions.py -q
```
Expected: PASS, with only established environment skips.

- [ ] **Step 5: Run frontend unit, type, lint, and route parity checks**

Run:
```bash
cd apps
bunx vitest run packages/ui/src/components/Notes/__tests__ packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts packages/ui/src/routes/__tests__/option-notes-route-identity.test.tsx --maxWorkers=1 --no-file-parallelism
bun --cwd tldw-frontend run typecheck
bunx eslint packages/ui/src/components/Notes packages/ui/src/services/note-graph-suggestions.ts tldw-frontend/e2e/workflows/notes-graph-suggestions.spec.ts extension/tests/e2e/notes-ux.spec.ts
```
Expected: PASS with no type/lint errors.

- [ ] **Step 6: Run Playwright visual/interaction checks**

In separate terminals, start the real API and WebUI on reserved test ports:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 18001
```
```bash
cd apps/tldw-frontend
bun run dev:webpack -- -p 18082
```
Then run:
```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=http://127.0.0.1:18001 TLDW_WEB_URL=http://127.0.0.1:18082 TLDW_WEB_AUTOSTART=false bunx playwright test e2e/workflows/notes-graph-suggestions.spec.ts --project=chromium --reporter=line
```
Expected: PASS. Inspect desktop/mobile screenshots and canvas pixel checks for nonblank rendering, stable framing, no overlap, long-content wrapping, and visible provisional versus authoritative states. Then run:
```bash
cd apps/extension
TLDW_E2E_SERVER_URL=http://127.0.0.1:18001 bunx playwright test tests/e2e/notes-ux.spec.ts --reporter=line
```
Expected: PASS; record any established environment-only skip.

- [ ] **Step 7: Update documentation and Backlog.md**

Document authoritative versus provisional graph state, nested API examples, privacy/data-boundary disclosure, limits/retention, worker/maintenance operation, permission setup, and deferred task IDs. Correct stale implementation-status language in `Graphing-Notes-PRD.md`. Update TASK-13138 with plan/spec links, commits, touched areas, verification, and known skips.

- [ ] **Step 8: Run final security and repository checks**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/Notes_Graph \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_link_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_models.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py \
  tldw_Server_API/app/core/Sync/v2/materializers/guarded_product_mutation.py \
  tldw_Server_API/app/core/Sync/v2/materializers/base.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_link.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_organization.py \
  tldw_Server_API/app/core/Sync/v2/server_origin_batch.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/notes_link_coordinator.py \
  tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/AuthNZ/permissions.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/api/v1/endpoints/notes_graph.py \
  tldw_Server_API/app/api/v1/endpoints/notes_graph_suggestions.py \
  tldw_Server_API/app/api/v1/schemas/notes_graph.py \
  tldw_Server_API/app/api/v1/schemas/notes_graph_suggestions.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/services/notes_graph_suggestions_worker.py \
  tldw_Server_API/app/services/notes_graph_suggestions_maintenance.py \
  tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py \
  tldw_Server_API/tests/Notes_Graph
python -m ruff check --ignore UP045 tldw_Server_API/app/core/AuthNZ/settings.py
if git diff --unified=0 origin/dev...HEAD -- tldw_Server_API/app/core/AuthNZ/settings.py | rg '^\+[^+].*Optional\['; then
  echo 'New Optional[...] annotation extends the AuthNZ/settings.py UP045 baseline' >&2
  exit 1
fi
python -m bandit -r \
  tldw_Server_API/app/core/Notes_Graph \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_link_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_models.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_graph_suggestion_store.py \
  tldw_Server_API/app/core/Sync/v2/materializers/guarded_product_mutation.py \
  tldw_Server_API/app/core/Sync/v2/materializers/base.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_link.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_organization.py \
  tldw_Server_API/app/core/Sync/v2/server_origin_batch.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/notes_link_coordinator.py \
  tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/app/core/AuthNZ/permissions.py \
  tldw_Server_API/app/core/AuthNZ/settings.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/api/v1/endpoints/notes_graph.py \
  tldw_Server_API/app/api/v1/endpoints/notes_graph_suggestions.py \
  tldw_Server_API/app/api/v1/schemas/notes_graph.py \
  tldw_Server_API/app/api/v1/schemas/notes_graph_suggestions.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/services/notes_graph_suggestions_worker.py \
  tldw_Server_API/app/services/notes_graph_suggestions_maintenance.py \
  tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py \
  -f json -o /tmp/bandit_TASK-13138.json
git diff --check
```
Expected: Ruff reports no findings in fully gated files, `AuthNZ/settings.py` has only its pre-existing `UP045` class and no newly added `Optional[...]` annotation, Bandit exits 0 across the complete touched source scope with no new findings, and `git diff --check` emits no output.

- [ ] **Step 9: Request code review and commit final integration**

Use @superpowers:requesting-code-review against the approved spec and this plan. Address valid findings with @superpowers:receiving-code-review, rerun affected tests, then:
```bash
git add Docs apps tldw_Server_API backlog
git commit -m "docs: complete Notes graph suggestions rollout (TASK-13138)"
```

## Stage 10: Pre-Merge Review Remediation

**Goal**: Remove synchronous Notes graph work from the application event loop, repair the AuthNZ migration regression expectation, and route remaining toolbar copy through i18n.

**Success Criteria**: Suggestion routes, worker preparation/publication, and maintenance yield while deliberately slow synchronous collaborators run; the AuthNZ seed regression suite reaches migration 95; toolbar-visible and assistive copy uses translation keys.

**Tests**: Focused async responsiveness regressions, `test_rbac_seed_helper.py`, Notes Graph worker/endpoint/service suites, and toolbar i18n coverage.

**Status**: Complete

**Verification**: 134 affected backend tests and 121 Notes graph UI tests passed. Ruff, ESLint, extension-pinned Prettier, locale duplicate/coverage checks, Bandit with zero findings, and `git diff --check` passed. The shared-package TypeScript baseline remains unchanged with zero diagnostics in touched files. Independent follow-up review found no actionable issues.

## Final Verification Checklist

- [ ] All 23 TASK-13138 acceptance criteria map to passing tests or recorded visual verification.
- [ ] Same-key terminal replay never repeats a mutation or provider call during the 90-day receipt horizon.
- [ ] No automatic retry path can issue a second provider request.
- [ ] Hidden staged rows cannot appear without an exact owner-scoped terminal Job receipt.
- [ ] Old acceptance fences cannot create a canonical link/tag relationship.
- [ ] Graph responses remain authoritative and contain no suggestion edge type.
- [ ] WebUI and extension use the same shared implementation.
- [ ] Human requester provides the required PR `Change summary` explaining what changed and why before merge.
