# UAT225 / TASK13260.163 — diagnosis and bounded options

## Confirmed isolated cause

Both actual read routes fail on a fresh inactive-Sync database before any provider or worker readiness decision. This reproduces on official PostgreSQL and SQLite, so it is not a PostgreSQL SQL dialect or initialization failure.

1. `endpoints/notes_graph_suggestions.py:103-121` resolves a canonical Notes dataset when Sync is active; otherwise `_dataset_key` returns `legacy:<owner>`.
2. `suggestion_api.py:622-678` constructs the real store/facade. `get_capabilities:213` and `list_suggestions:383` first call `_source`.
3. `note_graph_suggestion_store.py:4734-4764` loads the owned source via `_with_dataset_scope`. Its `_set_dataset_scope:249` and `_require_dataset_scope:268` require an exact row in `note_task_scope_authority` even for the legacy key. No such row is created on this path.
4. The resulting `NotesGraphDatasetScopeError(notes_graph_dataset_scope_invalid)` is translated by `suggestion_api.py:163-188` into the generic503 seen in the native receipt. Provider resolution, FTS availability and feature/worker flags are not reached.

The actual native responses at09:16:33.010/.012 are already bound in the221/224 native audit. No native DB, Sync storage, credentials or private runtime configuration was inspected during this task. Therefore the exact fresh inactive-Sync branch on the preserved native profile is a strong source/response match, not independently confirmed native state. Parent-owned credential-free Sync/authority metadata would close that last attribution limit. The retained browser event ledger contains no Sync status response.

## Causal tests

New `tests/Notes_Graph/integration/test_suggestion_fresh_reads.py` uses actual router, `_dataset_key`, factory, facade and store. Only actor/rate/token dependencies, inactive-Sync discovery and provider defaults are controlled. Jobs/worker are deliberately unavailable; no inference or network provider call occurs. All DB setup uses the official PG fixture or temporary SQLite file.

Final formatted test:4 expected failures /4 positive controls /0 skips /8.85s,5 existing warnings. Both fresh routes on both backends return503 with the exact scope exception captured before sanitized translation. Registering the expected owner/legacy key using the existing storage-fixture pattern makes both routes200: capabilities return generation_available=false and notes_graph_suggestions_worker_unavailable; list returns an empty page. This fixture-only row is causal evidence, not a proposed runtime repair. First identical causal run4/4 is retained separately.

Ruff0; Bandit0 findings/0 errors with B101 excluded for assertions; Python compile passes. Production remains unchanged. Tests are intentionally RED pending a reviewed repair.

## Existing intended and actual contracts

- `Docs/API/Notes_Graph_Suggestions.md:28` promises expected provider/FTS/worker limits as200 unavailable capabilities. `_translate`503 here is not that intended preflight state.
- `Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md:378-397` explicitly describes inactive-Sync legacy mutation support. Thus the current failure cannot simply be called an intentional documented configuration requirement.
- Actual `suggestion_service.py:39-68` returns no decision coordinator when Sync is inactive; factory substitutes `_UnavailableDecisionService` (`suggestion_api.py:617`) whose calls fail notes_graph_sync_not_ready. Merely opening the store's legacy write scope would not deliver the complete specified lifecycle.
- `task_store.py:122-170` already distinguishes a local unbound scope from immutable canonical authority. `moodboard_sync_store.py:282-365` shows that the authority table is shared, single-owner, flag-bearing and immutable; arbitrary new legacy binding would interfere with later canonical binding.
- Suggestion maintenance (`note_graph_suggestion_store.py:652`) and note-change invalidation (`:4359`) enumerate registered authority rows. Globally allowing unregistered writes could create review state they never visit.

## Recommended bounded option A: truthful reads, strict writes

Repair only the two failing native GET contracts without pretending to implement an inactive-Sync suggestion lifecycle:

1. Add a narrow internal **read-only** legacy allowance to the existing scope helper, defaulting off. Permit only the exact server-derived `legacy:<selected owner>` and only when no authority row exists for that owner. A different existing binding or arbitrary dataset still fails closed. Keep owner predicates, source byte limits, deleted filters and PostgreSQL transaction-local scope unchanged. Do not insert/update authority, migrate data, create Sync storage or alter policies.
2. Opt in only the five pure reads reached by these endpoints: `load_source_note`, `ensure_fts_ready`, `list_suggestions`, `list_suggestion_evidence`, `get_rejection_set`. Mutation, admission, lifecycle/maintenance and cancellation calls retain strict existing scope validation. Reads use actual stored rows, not fabricated empty responses.
3. Capabilities must remain unavailable while the real factory lacks a decision coordinator, using the existing safe notes_graph_sync_not_ready concept. Preserve disabled-feature/worker/provider reasoning and never advertise usable Generate on a path whose writes remain unavailable. The exact priority should be tested with worker/provider-ready and unavailable controls. This requires the frontend capability-reason allowlist and localized disclosure to recognize the reason; currently it is only an error code, not an allowed capability reason.
4. Keep canonical registered scopes unchanged. Explicit foreign/missing/deleted sources still404; other-owner or conflicting dataset bindings stay rejected; genuine SQL/FTS faults remain truthful errors. No automatic Sync enrollment and no native model call.

Anticipated production scope:existing store helpers plus the five read opt-ins; API capability readiness; frontend capability reason allowlist and one English disclosure key as required. No schema/worker/generation/Sync-coordinator port. Concrete implementation remains held for parent review.

Required controls before GREEN:both existing fresh routes; registered canonical paths; exact owner/legacy eligibility and other-owner/dataset conflicts; foreign/missing/deleted/oversized source handling; unchanged authority table and strict mutation/admission denial; worker/provider-ready preflight cannot claim generation available without decision authority; genuine DB and FTS errors; actual frontend parsing/display of the new allowed unavailable reason. Existing scoped store/API/route/capability tests and independent review follow.

## Option B: full inactive-Sync suggestions

Implement the complete legacy decision coordinator, durable local scope, maintenance/invalidation enumeration and eventual canonical binding behavior together. This better fulfills the broader written feature specification, but exceeds the two-read failure and needs its own design/causal tests. Do not partially enable it by removing `_require_dataset_scope` or inserting an immutable legacy authority row. Option A is the bounded recommendation for225; the broader contract gap remains explicitly disclosed rather than claimed fixed.

## Reproduction

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat225-frozen-diagnosis-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_fresh_reads.py -q --tb=short
```

No production fix, native acceptance or task closure is claimed.
