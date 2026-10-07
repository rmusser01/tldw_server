# Knowledge merge CI implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, following the requester’s existing authorization to rebase, fix PR issues and merge. Reviewers remain read-only.

**Goal:** Make PR3205’s latest-dev integration pass its required gates and affected regression suites before the authorized merge.
**Architecture:** Preserve the approved ADR065 capability. Reuse the existing caller-preserving Notes transaction option for pure provenance reads, repair historical and transport fixtures without weakening their contracts, and regenerate OpenAPI artifacts through the existing exporter. This is compatibility and release repair, with no new feature or dependency.
**Tech Stack:** Existing Python/pytest, SQLite/PostgreSQL fixtures, Backlog CLI, OpenAPI exporter, Bun and GitHub checks.
**Tracking:** TASK-13514; approved spec Docs/Design/2026-10-06-knowledge-followup-source-context.md; ADR065.

## Stage 1: Latest-dev integration
**Goal:** Establish a clean merge candidate with the human summary retained.
**Success Criteria:** Verbatim summary posted, tracking reopened, rebase onto verified dev complete without conflicts.
**Tests:** Git status, merge-tree preview and rebase outcome.
**Status:** Complete

- [x] Publish the requester’s paragraph verbatim in Change summary; preserve all other PR sections.
- [x] Record the actual CI failures and reopen TASK-13514 before source edits.
- [x] Verify MERGE_QUEUE is unset and rebase this merge candidate onto dev17c47c49d857a2a1fd1e58602b704213760a4a85. New candidate4aa4ba6a93bf0afd11f377eb74647b517dfadffb is clean.

## Stage 2: Preserve read ownership and fixture contracts
**Goal:** Repair demonstrated integration failures with the smallest shared fix.
**Success Criteria:** Borrowed PostgreSQL transactions retain pending writes and caller commit/rollback control; migration, Notes pagination, Sync capabilities and agentic citation fixtures exercise their intended contracts.
**Tests:** Existing failing node IDs from run37542867080; focused provenance read regression; bounded comparison against unchanged dev when attribution is unclear.
**Status:** Complete

**Files:**
- Modify tldw_Server_API/app/core/DB_Management/chacha/note_provenance_store.py: only the pure read scopes in get, read_receipt and list_parent_notes.
- Test tldw_Server_API/tests/ChaChaNotesDB/test_notes_provenance.py: parameterize those read operations under an externally opened PostgreSQL transaction; verify pending data stays private until caller commit and disappears on rollback.
- Verify existing tldw_Server_API/tests/DB_Management/test_chacha_postgres_notes_bootstrap_lifecycle.py and test_chacha_postgres_shell_read_lifecycle.py caller-owned read cases.
- Modify tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py: remove the new dependent sidecar only when constructing the historical v59 fixture; preserve RLS drift and migration rollback assertions.
- Modify tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_migration.py: pin the positive upgrade/reopen case to its intended SQLite73→74/PostgreSQL77→78 cutpoint instead of assuming the current schema ends there.
- Inspect/repair tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py and test_sync_v2_notes_task_activation.py: fixtures must register negotiated production adapters; preserve selected-dataset readiness checks.
- Modify tldw_Server_API/app/api/v1/schemas/sync_v2_models.py and test_sync_v2_models.py: reuse core supported-domain/operation constants. The API duplicate omitted notes.provenance, so its validator discarded the entire selected-domain list, including ready task domains. Preserve the API's existing private-domain boundary and legacy fallback behavior.
- Modify tldw_Server_API/tests/Notes_NEW/unit/test_notes_keyword_link_endpoint_pagination.py: give its database double the owner required by the production dependency contract; preserve pagination assertions.
- Modify tldw_Server_API/tests/RAG_NEW/unit/test_agentic_golden_citations.py and test_agentic_tracing_and_numeric.py: patch the generator actually consumed by the pipeline and use exact evidence for positive hard-span cases; retain explicit unsupported-answer coverage checks.

- [x] Run existing failing cases before changes; record RED rather than relying only on CI logs.
- [x] Trace every pure-read caller before changing the shared store. Reuse the existing option:

```python
with nullcontext(conn) if conn is not None else self._db.transaction(preserve_existing=True) as connection:
    # Existing owner-bound query; the caller still owns commit/rollback.
    ...
```

- [x] Add the small real PostgreSQL read-ownership regression, run it RED, then implement only the read-scope change and run GREEN.
- [x] Repair each fixture at its source; keep strict citation offsets, owner policies, independent versions, RLS and rollback checks intact.
- [x] Confirm whether task-capability failure reproduces on unchanged dev before choosing a fix. Do not widen advertised capabilities without verified backing adapters/materializers.
- [x] Verify the merged Email fix through the existing offline tripwire; do not duplicate PR3016.
- [x] Run affected backend suites, touched lint and Bandit through the project venv. Record existing informational mypy debt separately; do not suppress tests or strictness.

## Stage 3: Generated contract, review and merge verification
**Goal:** Publish a fully verified merge candidate and complete the authorized integration.
**Success Criteria:** OpenAPI fingerprint matches canonical schema, regenerated client types/builds and affected tests pass, independent scoped review is addressed, current-head required checks pass on current dev, PR is merged and owned artifacts cleaned.
**Tests:** Existing OpenAPI drift checker; Notes/Knowledge/Research and background authority suites; both client type checks/builds; all seven required current-head GitHub statuses.
**Status:** In Progress

**Files:** apps/tldw-frontend/lib/api/openapi.fingerprint.json (tracked); apps/tldw-frontend/lib/api/generated/openapi.json and schema.d.ts (owned ignored build outputs); Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md; TASK-13514.

- [x] Read and use apps/tldw-frontend/scripts/generate-api-types.mjs with the activated project Python; inspect the fingerprint diff and generated Notes/Sync contracts.
- [x] Run python Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json; do not replace the expected hash by hand.
- [x] Reuse primary dependency targets only through owned worktree symlinks; run appropriate affected client tests, type checks and production builds on the rebased tree.
- [x] Request one independent scoped correctness review of the new repair diff and address demonstrated findings with tests.
- [x] Update the release report and task with exact proof, preserve the requester’s Change summary, run normal hooks and git diff --check, and commit the verified repair.
- [x] Confirm remote head still matches the captured lease, then push the rebased candidate with an exact force-with-lease; mark ready and inspect new review comments without bot pings.
- [ ] Wait for seven required statuses on the exact current head and verify dev ancestry/merge mode before merge; never bypass branch protection or failing checks.
- [ ] Record successful merge and remove only owned processes, temporary profiles, links and generated artifacts. Retire only this completed plan; preserve primary user work and independent qualification follow-ups.

## Limits

Native capture, VoiceOver/mobile, real participants and external-web refresh remain the separately documented qualification/design work. Initial tests used the local Python3.11 environment below declared floors. The existing project Python3.12 environment is now reused for canonical contract generation and final affected tests; current Python3.12 CI remains required. Stop and reassess after three failed attempts at a single issue.

Publication: verified repair ad11d7e515fface58a4f97f3b995ac3c8d234018 pushed with the captured exact lease; PR3205 is ready, human summary retained verbatim. Seven current-head gates and authorized merge/owned cleanup remain.

Latest-dev refresh: documentation-only PR3206 advanced dev to1047ce10ecc191f78b0bee7e1b8ad19780c68015. Clean rebase987d1ee4705d2c15326bdbe32313436d69cab93c has exactly the inherited metadata patch and identical tested source. TASK-13514.1 and report now record the independently reproduced pre-existing SQLite startup race. Publish once with the captured ad11 lease, then await seven new-head statuses before merge.

Refreshed publication:8daeb5ca3c85d741296d7a1076a74b8d2b6a2331 pushed with exact ad11 lease onto dev1047ce10ecc191f78b0bee7e1b8ad19780c68015. Normal metadata hooks pass. Tested production/test/config trees remain identical to ad11. Current trusted license/security statuses pass; remaining current-head gates are queued/running. Optional packaging tool installation failed to resolve Requests before building project code; one failed-job retry passed without a source change.


### Second latest-dev refresh

Rebased onto dev `7ba48f251ec47a1e0bb680f49f9b7d86ec2b988d`, publishing candidate `1212f641705c8360fcc0910e8f938826b8182d87`. The inherited changes only retire the completed VN command-recovery plan and update TASK-13385. Exact binary patch comparison confirms that production code, tests, and configuration are unchanged from the validated candidate; diff check passed. Fresh required checks remain mandatory before merging.


### Third latest-dev refresh: Notes and Chat UX integration

Dev advanced to `3ca1ff055be3c75a7fa844a02b5d8ba925baa2be` through PR3203 while six of seven required gates had passed on candidate `1212f641705c8360fcc0910e8f938826b8182d87`. This update contains production changes and creates merge conflicts. Stage3 remains In Progress: preserve both incoming UX corrections and Knowledge provenance, regenerate the combined API contract, run affected Notes/Sync/RAG/client checks and a scoped review, then publish and wait for seven new-head statuses before merging. Cross-chat merge coordination is pending explicit user authorization; integration work continues independently.

### Third refresh validation and publication

The combined migration sequence and history deletion thresholds now agree with upstream's reserved cutpoints. Pure-read ownership and all REST projections remain fenced. Integration repairs reuse the existing local queue and state machine: durable create/restore replay identity, current-edit recovery without an immediate autosave loop, teardown versus explicit invalidation, and acknowledged queued titles for rename offers. Every new failure was reproduced before its fix. Final validation: backend 794 passed/one expected PG NUL skip; Docs 212 passed; focused Notes 189 passed; Knowledge/Research/native authority 312 passed; contrast check passes in its owning cwd; both types and production builds pass; canonical fingerprint2667c3566ff2fc647e9d7cbda083227a1d85445186106940b3765474fc93b1cb; production Bandit zero findings. Whole Notes 32 remaining named failures reproduce on unchanged dev; no broad-green claim. Backend independent review approves; final frontend review is pending. Stage3 remains In Progress until exact-head required checks, authorized merge and owned cleanup complete.


### Final Notes review and fourth dev refresh

Both additional reviewer findings have RED/GREEN behavior regressions: clean queued-create canonical readback and synchronous acknowledged history publication before a chained leave save. Final focused Notes: 191 passed across 11 files in35.39s; saved-monitoring module59 passed. Both type checks pass and scoped Notes ESLint remains zero errors/85 inherited production warnings/no test warnings. Final independent frontend rereview is pending. Dev advanced to005802bdb070fd68e087c4db3f831c33bef07c39 via PR3091; its Persona Buddy/docs/VN test delta has no overlap with the Knowledge source. Commit the reviewed repairs normally, rebase onto that head, run final combined client builds and affected handoffs, then publish with the captured1212 lease and require seven fresh statuses. Stage3 remains In Progress.
