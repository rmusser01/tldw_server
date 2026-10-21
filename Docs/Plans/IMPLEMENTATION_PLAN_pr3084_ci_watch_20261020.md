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
**Status**: Complete

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

## Fourth repair batch

The shared composer now counts an explicit image setter mutation once: it updates the existing previous-image ref before the layout fallback. Sequential drops retain the second image while reset/raw-setValues/restore invalidation and synchronous owner/ABA/latest-read fences remain. Model Playground's local-input fixture mocks its current catalog boundary; Setup asserts both the exact safe progress payload and captured request scope. All146affected image/account/recovery/draft/Model cases pass; focused Setup1passes19deselected. All3changedUI files have0ESLint diagnostics; TypeScript-only Bandit is not applicable. Independent connected source review supports this scope.

## Fifth repair batch

Recipe Pull now acquires the existing exact reconciliation lease before copying content and validates captured owner/link/schema at each actual local mutation. Denied reconciliation preserves the exact original row; acquired write/release failures retain scoped evidence. Eleven new concurrent owner/link/schema replacement controls reproduced red before repair. Direct/extension settlement observers execute the real mutator once and hold its acknowledgement. The final recipe/uncertainty/structured/auto-sync cohort passes1182/1182. The preceding auth/structured/auto-sync cohort passes1206/1206; those cohorts overlap and are not one combined run. Auth tests preserve raw credentials/session state, require the documented opaque boolean invalidation marker and withheld effective tokens, and cover collection versionsPOST positively with itemPOST denied. Four UI ESLint scopes add0diagnostics (background10 inherited); no Python scope for Bandit. Independent connected review is clear for this batch. No new ADR: this restores the existing exact-authority ownership contract without changing APIs, persistence schema or architecture rules.

## Sixth repair batch

Opening a peer WebUI tab no longer deletes the shared cookie configuration before verification. Only the existing seed's locked exact-revision cleanup can remove a current failed marker; document-local invalidation and initial/recovery joining remain. Exact recovery fixtures include the existing UUID verificationRevision and target only the intended removal failure. Held-profile tests use their captured graph's real nonblocking canonical reader for local-null assertions and await an already-started client getter after its actual probe settles. All64runtime-bootstrap cases pass, then all8cross-tab cases pass;72unique cases across those runs, not one72greenrun. Three UI scopes add0ESLint diagnostics (one inherited source warning). TypeScript-only Bandit is inapplicable. ADR044 and the existing cross-tab plan govern; this restores the accepted contract and needs no new ADR.

## Remaining work

Sidepanel4 known failures remain. OpenUI SDK WIP is unqualified (30passed23failed/53 before the separate loader prerequisite, then focused4failed); original/native scope reassessment and real metadata diagnostics are recorded in the task; extension bootstrap is qualified in the sixth batch. Diagnose focused affected cases instead of replaying the573-case cohort. Current-head CI must qualify final source including the three locally unavailable PostgreSQL-tool scenarios. Quality baselines remain unchanged and the global ratchet passed before the frontend-only batches. Review exact current-head job logs as they finish.

Auto-merge is disabled. Latest verified published headc1754dae67 is OPEN and behind currentdev; dev was fetched for inspection only. MERGE_QUEUE was verified absent; check again before final integration. The10-minute watch remains active until verified merge into dev. Main UAT worktree/runtime and older paused heartbeat remain untouched. Broader runtime/UAT/release acceptance remains outside these CI results.
