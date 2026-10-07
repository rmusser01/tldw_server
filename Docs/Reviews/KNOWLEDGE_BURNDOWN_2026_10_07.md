# Knowledge workstream burndown — 2026-10-07

Baseline: `dev` at `26ae4fd679ddd43f9f904cd3cca1b1cf8af51e48`, isolated branch `codex/knowledge-burndown-20261007`. Rebased without conflicts onto current dev `5ec8c7939f46d6baecf8767caf070cb4a2eb335c`; intervening changes only touch Admin readiness and its task. Parent tracking: TASK-13530. Primary checkout, installed dependencies and existing local model service remain separate from this work.

## Current disposition

| Work item | Evidence and status | Owner |
| --- | --- | --- |
| Independent provenance and full canonical Notes Research handoff | **Merged and Done.** PR3205 merged at `e3c345b76f2d93527488b2015b9c3a2649c1f981` on 2026-10-07T06:26:02Z. GitHub and current-dev ancestry verified. TASK-13514 now supersedes its stale merge-pending summary while retaining historical notes. | TASK-13514 |
| Concurrent SQLite Sync initialization | **Implemented and independently reviewed.** Shared schema execution preserves the caller transaction. Deterministic concurrent startup and strict catalog rejection pass. Publication remains separate from local verification. | TASK-13514.1 |
| Notes frontend qualification | **Scoped repairs verified.** All 32 reproduced unchanged-dev failures repaired in test fixtures. Full Notes + NotesDock is green; broader repository lint and unrelated failure ownership remain qualified below. | TASK-13530.2 |
| Extension capture and saved-source actions | **Scoped CDP workflow passed on the final production artifact.** Actual content-script extraction, Clipper save, owner-scoped readback and settled Notes reopen produce canonical Note revision 1. Notes-only Ask returns the requested inspection date with one mapped citation; full source preview retains URL, capture date and text. Native context-menu selection is not attested. The React route crash is fixed; direct-panel chat analysis still ended with a stream error and is qualified below. | TASK-13512 |
| VoiceOver, Safari, actual iOS/mobile keyboard | **Open.** Foreground/VoiceOver permission was granted, but no spoken or actual-device session completed. Computer Use refused the Chrome app-access request; the requester then explicitly directed CDP. CDP browser checks do not constitute spoken VoiceOver or actual iOS evidence. No VoiceOver setting was changed by this run. | TASK-13512 |
| First-time and experienced human usability sessions | **Open.** Existing first-time/power-user protocol remains ready in the prior release report. No novice/power-user participant availability or actual observations have been supplied. Agent-driven browser checks are not participant sessions. | TASK-13512 |
| Explicit public-web article capture/refresh | **Design prepared; approval pending.** Proposed design and ADR-066 reuse extraction, WebClipper, canonical Media versions and Workspace APIs. No feature implementation is claimed. | TASK-13530.1 |

## SQLite fix and verification

`sqlite3.Connection.executescript` commits an existing transaction before executing schema DDL. Sync holds `BEGIN IMMEDIATE` while creating its authority catalog, so that implicit commit allowed a second initializer to see only part of the catalog. `SQLiteBackend.create_tables` now executes complete SQLite statements individually while a caller transaction is active, using `sqlite3.complete_statement` to preserve triggers, comments and quoted semicolons. Standalone schema scripts retain their existing behavior. No lock, schema cache, relaxed validator or new dependency was added.

The trace-callback regression pauses immediately before cleanup-candidates DDL. A separate handle cannot see a partial authority catalog; both concurrent initializers finish with one migration marker. Additional tests preserve caller rollback ownership and reject missing tables, missing indexes and unexpected columns.

- Original focused run: four failures before the fix; ten passing after it.
- Relevant backend run: 285 passed, seven PostgreSQL skips reported by existing fixtures when PostgreSQL was unreachable with Docker disabled. No homemade database setup or skip suppression.
- Independent reviewer: ten focused tests plus seven additional script edge probes passed; one fixture incompatibility reproduced in both sensitive-logging parametrizations.
- Corrected the logging stub with the actual SQLite `in_transaction=False` property; schema + complete sensitive-logging suites pass **62 tests**.
- Fresh parent schema/concurrency/drift check: **28 passed**, 184 deselected.
- Ruff lint passes all four touched Python files. New regression file is formatted. Three existing full files retain inherited whole-file formatter differences; no mass reformat is included.
- Production Bandit: **zero findings/errors** in the changed backend. Pytest assertions are intentional test code and are not a production security finding.
- Existing full-suite shard globs already include the new database regression module; no workflow exclusion or matrix change required.

ADR required: **no**. This restores the transaction contract governed by ADR-020 rather than changing database architecture.

## Notes fixture repair and qualification

Updated fixtures to the current public contracts: storage watch/unwatch, effective assistant state, canonical authenticated owner/version scope, explicit Connections disclosure, asynchronous note selection, permanent-error focus behavior, conflict-panel draft preservation and current Tags copy. Production Notes guards/state machines were not modified. Removed all 13 inherited lint warnings from the 11 touched fixtures without rule changes or suppressions.

| Final check | Result |
| --- | --- |
| All Notes + NotesDock | 122 files; **866 passed**; zero failures/skips/runner errors |
| Changed fixture set | 11 files; **64 passed** |
| Named historical cases | All **32** original red assertions now pass |
| Touched fixture ESLint | Zero errors and warnings |
| TypeScript syntax transpilation | Zero diagnostics |
| WebUI typecheck | Passed |
| Extension compile/typecheck | Passed |
| Final Chrome production bundle | Built in 44.5 seconds; exact-output token check passed; all five bundle receipts use one compatible React/ReactDOM pair |
| Independent review | No actionable findings |

The installed UI virtualizer and renderer resolved React 18.3.1 from different physical `.pnpm`/`.bun` trees. A temporary test config aliases/dedupes the existing React copy and inlines the original TanStack modules; it changes neither assertions nor production behavior. No installed packages were changed. Owning cwd is `apps/packages/ui`; final command was:

```sh
bun run test -- --config .vitest.burndown.config.mts --cache=false \
  src/components/Notes/__tests__ src/components/Common/NotesDock/__tests__
```

The temporary Vitest config qualifies the test environment. The separate production extension repair aligns its existing React pins and native Vite resolver as described below. The [32-case receipt](artifacts/knowledge-burndown-20261007/notes-fixture-case-receipt.json) and [exact source hashes](artifacts/knowledge-burndown-20261007/notes-final-source-receipt.json) are retained in this report's artifact directory.

Broader qualification remains explicit:

- TASK-12116 owns 85 unchanged production Notes lint warnings: 76 `any`, eight unused declarations and one callback dependency. No global lint-green claim.
- TASK-13406 owns the current empty smoke-exception allowlist; an old `/404` source guard contradicts that policy. The strict allowlist remains unchanged.
- TASK-13261.1 owns chat history edit/fork behavior; three old edit-handler tests expect unsupported edit/send and obsolete helper mocks.
- TASK-13397 owns the dynamic-UI source guard failure.
- Three storage failures in the Web shim harness do not reproduce in the owning UI harness, which passes four of four storage cases.

ADR required: **no**. This restores tests to existing contracts and qualifies environment/debt ownership.

## Extension runtime and initial source selection

The actual Ask chat route crashed because the loaded TanStack virtualizer bundled a second React dispatcher. The tracked extension also pinned React/ReactDOM 18.2.0 while the shared UI used 18.3.1 and the existing OpenUI peer required at least 18.3.1. The repair adds native Vite `resolve.dedupe`, aligns the two existing pins to 18.3.1 and regenerates only the matching lock records. No dependency, resolver alias or workaround component was added. The other 2,095 lock package records are unchanged.

A real linked-peer runtime regression reproduces the null `useReducer` error without deduplication and renders successfully with the production settings. Final verification: five runtime/asset/lazy-entry tests passed, extension compile passed, touched config/test ESLint has zero errors/warnings, Prettier passed, and the final Chrome build plus exact-directory token check passed. All five instrumented bundle receipts resolve one React/ReactDOM 18.3.1 pair. WXT owns Vite 5.4.21. Verification reused existing packages through an owned worktree link forest; it did not perform a clean dependency installation or modify the primary installation.

Selecting a specific source before asking also incorrectly showed “No results found.” The shared `SET_ERROR(null)` reducer marked the session searched while clearing a possible error. One line now preserves the initial state when clearing a null error and retains completed/failed search state. Regression red: three failures. Final provider/layout/error/cancellation checks: 18 files, **224 passed**; independent scope review: **24 passed**. The test is lint-clean. The production provider retains 11 unchanged lint warnings; no lint rule was relaxed and no global lint-green claim is made.

ADR required: **no**. Both fixes restore existing runtime and UI-state contracts.

## Final CDP walkthrough and limits

The final extension artifact was copied unchanged into the isolated installed extension and reloaded through Chrome's extension controls over CDP. Browser operations used CDP following the requester's instruction. No simulated native context-menu callback was used.

1. The extension's real content script extracted the synthetic Vega article. The existing runtime-message contract handed that immutable draft to Clipper; actual Save clip persisted it.
2. Owner-scoped GET returned HTTP200 and canonical Note `5d0d9205-008f-4307-8775-f0362d82c2a7`, revision1. The settled Notes editor shows **Saved**, version1 and the captured text; the list contains one Note.
3. Actual source controls selected that exact Note and Notes as the sole category. The initial “Ask Your Library” guidance stayed visible, with no premature empty-result message. [Source-selection screenshot](artifacts/knowledge-burndown-20261007/source-selection-before-ask.png).
4. Actual Ask sent `sources=["notes"]`, that single `include_note_ids` UUID, and web fallback false. The server-default fixture returned an unrelated uncited answer, which the UI correctly marked **Uncited answer / needs review**. This fixture output is not a successful answer-quality check.
5. Selecting the existing local Custom OpenAI provider/model and asking again produced HTTP200, one Note source and **“24 November 2026 [1]”** with one mapped citation and zero page errors. [Sanitized request/result receipt](artifacts/knowledge-burndown-20261007/cdp-scoped-ask.json).
6. Actual View source opens the full captured text with the exact Note identity, original URL and capture date. [Source preview](artifacts/knowledge-burndown-20261007/source-preview.png), [evidence card](artifacts/knowledge-burndown-20261007/scoped-evidence.png).

This uses controlled fixture embeddings with real local answer generation; it is not a general retrieval-quality evaluation. Programmatic capture entry does not prove native menu selection, spoken VoiceOver, Safari, an actual iOS keyboard or human usability sessions.

The final direct-panel Ask chat button opens the chat route without the React crash. After selecting the local model, its analysis attempt ended **Stream completion failed**. No successful chat generation is claimed from this direct-panel harness. Native-panel chat streaming remains a follow-up to reproduce with the actual launch context. The Notes editor's manual edit-state footer says “Origin: Typed manually” for the captured Note; its list tag and source preview retain the capture identity. Clarify that footer using authoritative capture provenance rather than inferring origin from an editable tag. Both observations stay under TASK-13512.

[Verification counts and qualifications](artifacts/knowledge-burndown-20261007/verification.json), [extension runtime verification](artifacts/knowledge-burndown-20261007/extension-runtime-verification.json), [build identity](artifacts/knowledge-burndown-20261007/extension-build-runtime.json) and [security receipt](artifacts/knowledge-burndown-20261007/python-security.json) are retained with this report. Bandit cannot parse TSX; TSX is validated by its own lint/type/runtime checks.

## External capture proposal and remaining validation

The proposed design is `Docs/Design/2026-10-07-knowledge-web-capture-refresh.md`; proposed ADR-066 records the durable security/persistence decision. Capture would fetch readable public text only after an explicit action, show the extraction for acceptance, save through WebClipper, and give each accepted refresh a new identity while retaining old evidence. It discloses the extra capture Note and distinguishes an exact historical Media preview from existing current-version RAG. It adds no crawler, new store, dependency or automatic refresh. The proposal is not accepted or implemented until requester approval.

Remaining completion criteria are actual native context-menu selection and native-panel chat qualification, capture-origin wording, spoken assistive checks, available real Safari/iOS/device observations, actual novice/power-user sessions, approved external capture implementation, broader owner-managed quality debt, and release publication/required checks. TASK-13530 stays In Progress; TASK-13514's merged implementation is accurately Done. Neither automated green suites nor elapsed time substitute for those missing checks.

## Cleanup and publication

The isolated Chrome profile was closed through CDP. Its browser, article fixture and API/WebUI/mock services are stopped; ports19131,19132,19133,19135 and19136 are closed. The pre-existing local model service9099 still responds HTTP200. No VoiceOver setting changed. Four owned dependency links, the owned extension link forest/Jiti cache, generated client outputs and private browser/runtime profiles were removed. Primary installed dependencies were not modified. The managed branch/worktree and sanitized report artifacts remain for review. [Cleanup receipt](artifacts/knowledge-burndown-20261007/cleanup.json).

Verified changes are published as [draft PR3211 against dev](https://github.com/rmusser01/tldw_server/pull/3211), based on current dev `5ec8c7939f46d6baecf8767caf070cb4a2eb335c`. Strict MkDocs passes in22.81s; normal hooks and diff checks pass. Initial remote checks are pending/running, and the repository requires a new human-owned Change summary before merge. TASK13514 remains Done for merged PR3205; the root burndown and qualified follow-ups remain open. New PR publication does not imply implementation of the unapproved capture proposal or completion of external sessions.
