# Canonical Knowledge provenance implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement each task in this session, sequentially with scoped reviews. The requester already approved the independent sidecar design and instructed continued implementation.

**Goal:** Preserve sourced Notes evidence independently of editable Markdown across WebUI, extension and Sync.

**Architecture:** Reuse Notes DB transactions, strict Sync domain adapters, object-state conflict checks and ADR034 durable batches. One `notes.provenance` record is identified by its parent note UUID and versioned independently. Keep portable markers as compatibility/export data; canonical server data wins on disagreement.

**Tech Stack:** Existing FastAPI/Pydantic, ChaChaNotes SQLite/PostgreSQL, Sync v2, shared TypeScript UI, pytest and Vitest.

**Tracking:** TASK-13514; PR3205. Approved spec: `Docs/Design/2026-10-06-knowledge-followup-source-context.md`. Governing decisions: ADR031, ADR034 and ADR065.

## Stage 1: Strict contract and owner-scoped persistence

### Task 1: Strict contract and owner-scoped persistence
**Goal:** Add the bounded canonical payload and a migration-owned one-to-one store.
**Success Criteria:** Exact allowed fields/limits, independently checked versions, active owned parent, repeatable migration, no credential-bearing arbitrary metadata.
**Tests:** Validator boundaries; SQLite migration/reopen; PostgreSQL migration/RLS via existing live fixture; stale and foreign-owner writes.
**Status:** Complete

Files: create `tldw_Server_API/app/core/Sync/v2/notes_provenance_contract.py` and `tldw_Server_API/app/core/DB_Management/chacha/note_provenance_store.py`; modify `ChaChaNotes_DB.py`, `chacha/note_store.py` and `backends/pg_rls_policies.py`; tests `tests/Sync/test_sync_v2_notes_provenance.py` and `tests/ChaChaNotesDB/test_notes_provenance.py`.

- [x] Write failing strict payload, parent ownership, independent version and persistence tests.
- [x] Run with the project venv: `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_notes_provenance.py -q`; confirm the missing capability failure.
- [x] Implement strict Pydantic contract (extra fields forbidden), same client bounds and safe integer IDs; serialize JSON with sorted keys and no NaN.
- [x] Install SQLite75/PostgreSQL79 migration-owned table: `(owner_user_id, note_id)` primary key, note FK, JSON text, positive version, canonical hash and tombstone flag. Use authenticated DB owner and parent ownership predicates for every access. Install forced PostgreSQL owner RLS.
- [x] Exercise old schema upgrade and reopen, transaction rollback, deleted parent, owner mismatch and exact-version replacement. Integrate every shared NoteStore soft-delete/tombstone/delete path, including inactive Sync; child tombstones retain payload and require an owned parent (which may be deleted). Parent restore never reactivates a child. Hard delete cascades the child; commit the verified unit.

## Stage 2: Canonical Sync authority and atomic lifecycle

### Task 2: Canonical Sync authority and atomic lifecycle
**Goal:** Declare, enroll, capture and replay independently versioned provenance using existing machinery.
**Success Criteria:** Core note v1 unchanged; stale sidecar cannot change either head; combined save/delete projects both records transactionally; interrupted durable groups resume; delayed writes cannot revive a deleted parent.
**Tests:** Real Sync store + Notes DB for old/new clients, exact independent bases, atomic failure/replay, lost acknowledgment, parent tombstone/restore and encryption-policy rejection.
**Status:** Complete

Files: create `Sync/v2/domain_adapters/notes_provenance.py` and `Sync/v2/notes_provenance.py`; modify `Sync/v2/models.py`, `factory.py`, `profile.py`, `service.py`, `materializers/notes.py`, `server_origin_batch.py`, `store.py`, `replay.py` and `DB_Management/Sync_DB.py` only where existing integration requires it. Reuse `core/Notes/organization_capture.py` compound plan when keywords/folders are included.

- [x] Write failing real-store tests for canonical create, ordinary core edit, independently stale sidecar, deleted-parent write, combined delete and durable retry.
- [x] Run `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_notes_provenance.py -q`; verify behavioral red failures.
- [x] Register `notes.provenance` v1/upsert/tombstone with exact-base restore and server-trusted materialization; do not widen `notes.note` payload fields.
- [x] Validate both accepted bases before append using existing batch preflight/append guards. Recheck the active owned parent inside the dataset append transaction, using the in-group overlay for newly created/restored parents. Pause sidecar creation after preflight, delete the parent, then resume: no sidecar append may be accepted. Keep core/provenance pair adjacent. Add one shared pair projector used by initial capture, retry, client deletion and replay/repair, even when repair filters one domain or starts at the second envelope. Project both records in one Notes transaction, then checkpoint both Sync states. Inject failure in the second product write and after product commit/before Sync commit; prove rollback or idempotent convergence without a permanently applied half-pair.
- [x] Add parent deletion expansion and enforce parent head at append/materialization, including client Sync pushes. Core restore leaves sidecar tombstoned until explicit retained-head restore.
- [x] Upgrade an existing default profile lacking the domain on its first provenance-aware Notes write, and on explicit profile enrollment. Publish initializing/ready/failed metadata; canonical provenance writes fail closed until bounded source-verified backfill completes. Capture existing valid sidecar/marker only under the exact owned parent version; retain product revisions during bootstrap rather than resetting them. Missing core Sync heads require an owner/version-verified parent bootstrap, not an invented base. Preserve existing independent tombstones on interruption/retry; no marker backfill may override them. Verify an old-profile upgrade and an old device deleting a sourced note without advertising the new adapter. Preserve current heads on replay/conflict; never infer source trust from prose.
- [x] Run nearby core-note, durable batch and capability discovery regression tests; commit the verified unit.

## Stage 3: Notes API and portable export compatibility

### Task 3: Notes API and portable export compatibility
**Goal:** Save/reopen structured provenance through canonical Notes routes.
**Success Criteria:** Create/update returns canonical history and its independent head; omission preserves; stale replacement rejects both mutations; exports remain self-contained without reviving tombstones.
**Tests:** Real REST create/update/PATCH/read/delete/restore/export/import, old/new requests, keyword/folder compounds, lost acknowledgment, unavailable encryption and pointer authorization.
**Status:** Complete

Files: modify `api/v1/schemas/notes_schemas.py`, `api/v1/endpoints/notes.py`, `core/Notes/organization_capture.py` and existing Notes export/import shared helpers as required. Tests: `tests/Notes/test_notes_provenance_api.py` and nearby Notes compatibility suites.

- [x] Write failing API tests before endpoint/schema changes; run `python -m pytest tldw_Server_API/tests/Notes/test_notes_provenance_api.py -q` and retain RED evidence.
- [x] Add optional validated `knowledge_provenance` and exact `expected_provenance_version` request fields. Return `knowledge_provenance_state` (unsupported/absent/active/deleted), `knowledge_provenance_version` (0 only for absent) and `knowledge_provenance_hash` with payload only for active. Canonical deleted heads retain version/hash. Omission must never clear the child.
- [x] Call Stage2 readiness/capture integration for active Sync and Task1 store transactions otherwise. Reuse organization compound plans, put the note/provenance pair adjacent and return fully acknowledged heads. Require exact note and independent child bases; replay a lost acknowledgment by request identity before checking stale mutable versions.
- [x] Expose explicit retained-head child restoration only after an active core restore; never implicitly revive it from a marker. Canonical tombstones forbid marker fallback/backfill. Distinguish a missing/unsupported capability from deleted history in every response.
- [x] Serialize canonical active history into portable markers for JSON/CSV exports; suppress stale valid markers for tombstoned notes. Backfill valid historical markers only through an exact owner/version mutation. Surface marker disagreement; canonical evidence wins.
- [x] Prove retained IDs/excerpts do not authorize source reads or imply inaccessible/deleted sources are live. Run REST/organization and core capability regressions, scoped Bandit and normal checks; commit the verified unit.

## Stage 4: Shared WebUI/extension compatibility

### Task 4: Shared WebUI/extension compatibility
**Goal:** Preserve and display canonical history while supporting old servers and portable files.
**Success Criteria:** Structured data wins over edited markers; deleted heads forbid fallback; direct Knowledge save, Notes Library and Research save/reopen share the contract; owned writes keep existing cancellation/draft guards.
**Tests:** Canonical preference, divergent marker status, old-server fallback, deleted marker suppression, backfill exact version, lost acknowledgment, source import and account/workspace cancellation.
**Status:** Complete

Files: modify `apps/packages/ui/src/utils/knowledge-note-provenance.ts`, both `services/tldw/domains/collections.ts` and `services/tldw/TldwApiClient.ts`, `components/Notes/hooks/useNotesEditorState.tsx`, `components/Option/KnowledgeQA/ExportDialog.tsx`, `components/Option/ResearchWorkspace/StudioPane/QuickNotesSection.tsx`, `utils/use-research-workspace-prefill.ts`, `components/Option/ResearchWorkspace/workspace-server-restore.ts`, workspace state/types and relevant existing tests. Reuse existing export helpers where a saved record must remain self-contained.

- [x] Add failing client utility and mounted workflow tests before save/reopen logic changes. Run affected Vitest files with one worker and retain behavioral RED evidence.
- [x] Match Task1 strict payload types/limits, including direct question/scope/reasons/source references and encoded portable marker ceiling. Use one shared canonical-resolution helper and optional independent head fields rather than scattering fallback decisions.
- [x] Prefer canonical active payload, surface divergent-marker reconciliation, preserve on ordinary edits and portable exports. A canonical deleted head removes/suppresses fallback markers and may only be restored explicitly with its retained version/hash; unsupported old servers retain current marker behavior.
- [x] Save new history from Knowledge or Research using top-level structured data plus the portable marker. Ordinary edits may omit unchanged structured history; valid marker backfill requires absent canonical state and exact current parent/child versions. Retry identity must survive a lost response without overwriting later user edits.
- [x] Update Notes Library/offline drafts, Quick Notes, importer acknowledgment and Workspace canonical restore to use the shared resolution and carry independent head fields. Preserve account/workspace cancellation and stale-save guards. Source pointers grant no fetch authority.
- [x] Run focused Vitest, both client type checks/builds and shared UI lint, then commit the verified unit.

## Stage 5: Release verification and PR update

### Task 5: Release verification and PR update
**Goal:** Publish evidence and final scope without overstating unrun device/participant work.
**Success Criteria:** All affected checks pass; touched Python Bandit clean; independent review completed; task/ADR/report/PR current.
**Tests:** Existing PostgreSQL fixture, browser save → remove marker → reopen → evidence retained, ordinary old-client update, delete/retry, portable export.
**Status:** In Progress

- [ ] Run focused API/Sync/DB suites and affected shared client tests; use existing Postgres fixture and report availability faithfully.
- [ ] Run Python Bandit in the project venv on every touched production Python file; fix new findings.
- [ ] Recreate only required dependency symlinks, run client types/builds and controlled fresh browser workflows.
- [ ] Request an independent correctness review and address verified findings. Update `Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md` and TASK-13514 verification notes.
- [ ] Run normal pre-commit checks, inspect diff, commit and update draft PR3205 against latest dev. Human-authored Change summary and required CI remain merge prerequisites; approval of this design alone does not supply the what/why summary.
