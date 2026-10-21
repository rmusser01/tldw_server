# PR3084 CI repair and merge plan

Task: TASK-13260.278.18.83.45. The requester authorized fixing CI and merging PR3084 into dev. Existing checkpoint and live UAT are separate. Implement serially in the clean checkpoint worktree.

ADR required: no. Fixture, documentation publication and baseline-location corrections preserve existing behavior and architecture. ADR059 governs task edits. Never increase ratchet offense counts.

## Stage 1: Repair failing UI and documentation fixtures
**Goal**: Align Notes Dock request ownership and Playground history capability fixtures with current contracts; publish the missing proposed ADRs.
**Success Criteria**: Original cache invalidation, pending-history, handoff and docs assertions pass without weaker guards.
**Tests**: Current failing Notes Dock and Playground search cases, published-doc refresh parity and strict documentation builds.
**Status**: Complete

## Stage 2: Clear backend ratchets
**Goal**: Fix new test-quality offenses with actual body/state/output assertions and appropriate real boundary fixtures; refresh only moved conditional-predicate locations after verifying unchanged semantics/counts.
**Success Criteria**: No new or worse quality offenses; no added tenant predicate baseline allowances.
**Tests**: Scope and test-quality ratchets, affected test modules and touched-scope lint/Bandit.
**Status**: In Progress

## Stage 3: Publish and finish integration
**Goal**: Publish reviewed fixes through normal hooks and get the final head merged into dev.
**Success Criteria**: Current-head required CI green, valid review feedback settled, verified human-summary disposition, current dev incorporated and normal merge confirmed.
**Tests**: Final diff/source review, current-head checks/reviews, GitHub MERGED state and merge ancestry on dev.
**Status**: In Progress

## First verified publication batch

- Notes/history:33unique UI cases qualified across32passing full-run cases and one corrected focused refusal case. No single33-case green run is claimed. Three originally failing documentation checks pass under supported pytest9.0.3.
- SQLite Jobs now binds JSON-array type membership in static SQL, retaining both candidate and future-scheduled membership guards.72Jobs/SQL guard cases pass;44PostgreSQL cases were deselected. Existing json_each use means no new architecture decision.
- All13uncovered test files are assigned in the existing five CI matrices; baseline/ignore unchanged, coverage guard new_uncovered=0. All28conditional-predicate guard ASTs match dev exactly; only baseline line locations changed.
-66authentication/prompt/MCP contract cases pass. Canonical-cookie expectations were corrected after connected independent review; explicit bad-header, owner, scope and RBAC denials remain intact.26focused registration/backup/grader/Flashcards cases pass;73scope/required-workflow cases complete exit0.
-16Python files compile; Ruff adds no diagnostics; Bandit87baseline/86current, zero new AST-aligned findings/errors. Notes/history ESLint counts3/3and19/19, no new diagnostics. Independent two-round source review is clear for this batch, not whole-PR merge approval.

## Second repair batch

- Monitoring now hides stale cached health after an error and supersedes pending generations on explicit refresh/retry. Its new imperative shared-query waiter test confirms replacement preserves waiters and ignores late old results. All21 Monitoring and4 shared-cache cases passed.
- Playground no longer resets twice after the shared reset helper. Shared restoration accounts for the explicit image ownership increment. Both chat surfaces honor the existing B03 no-admission flag before restoring a failed structured result; skipped and separately tested raw dispatch-rejection recovery remain. Existing positive assertions now use an explicit preflight refusal, and both surfaces have unflagged-failure negatives. Final connected recovery/draft/offline cohort98/98 passes. Across these cohorts121 unique frontend cases are qualified; no single121-case run is claimed.
- The six originally remaining backend quality files are repaired through owned real HTTP fixtures, real deterministic token accounting, exact configured-model prompt transport, scoped override restoration, real cache lifecycle and one consolidated cold offline accounting scenario. Character1 passes; initial affected cohort87passed1failed/88 plus corrected B02 focused case exits0; offline42 passes. No single88-case green run is claimed.
- Six Python files compile and Ruff reports0 diagnostics. Bandit119baseline/119current,0new AST-aligned findings/errors. Eight frontend files add no ESLint diagnostics; Sidepanel32 and Playground17 inherited diagnostics remain. Independent connected reviews are clear for this batch only.

## Third repair batch

The11further quality-file violations are cleared without baseline changes. Added assertions check real response/state/error propagation; shared backup workspace and opt-in audio cache restart fixtures own actual environment/cache lifetimes. The transcription route fixture retains real principal admission. Global test-quality scanner5199files exits0. The strengthened outcome cohort112/112 passes. The ledger/guard/PG cohort has16passes and3failures caused by unavailable local pg_dump/pg_restore executables; those three real backup/restore scenarios remain unqualified. Linux CI provisions the clients. Thirteen Python files compile; Ruff adds0diagnostics; Bandit95baseline95current with0newAST-alignedfindings/errors. Independent source review is clear for this batch only.

## Remaining work

Current d58859b182 frontend historical-scope run has531passes42failures/573 across background auth2, setup1, recipe persistence10, Sidepanel4, account privacy2, Model Playground3 and OpenUI20. No new frontend edits from this cohort yet; diagnose before bounded focused reruns. Extension bootstrap is also pending. Current-head CI must qualify the final source including the three locally unavailable PostgreSQL-tool scenarios. The test-quality baseline remains unchanged. Review exact current-head job logs as they finish.

Auto-merge is disabled. Latest published head d58859b182 is OPEN and behind current dev; dev was fetched for inspection only. MERGE_QUEUE was verified absent; check again before final integration. The10-minute watch remains active until verified merge into dev. Main UAT worktree/runtime and the older paused heartbeat remain untouched. Broader runtime/UAT/release acceptance remains outside these CI results.
