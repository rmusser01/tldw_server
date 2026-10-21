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

## Remaining work

Test-quality ratchet has six pre-existing reported file offenses remaining: native frozen-worldbook HTTP fixture injection; Character prompt transport mocks; B02 accounting fixtures; PostgreSQL sequence HTTP fixture restoration; offline-email accounting mocks; Prompt Studio lease setup. No quality baseline was increased. Original Character test is unchanged; two local runs failed, including the observed one-versus-two newline payload mismatch. Diagnose the exact current contract before a third bounded attempt.

Two additional frontend WIP files are excluded from this first commit: MonitoringDashboardPage.test.tsx and usePlaygroundSubmit.draft-ownership.test.tsx. Reviewed fixture prerequisites are repaired, but the first complete cohort is48passed6failed/54:2Monitoring stale-health/generation cases and4draft recovery/import/stale-cleanup cases. Preserve original assertions and investigate causes before staging them. Other failed frontend files are recorded in the task. No full suite, real PostgreSQL, runtime/UAT or merged-state acceptance is added here.

Auto-merge is disabled. MERGE_QUEUE was verified absent; check again before integration. The10-minute watch remains active until verified merge into dev. Main UAT worktree/runtime and the older paused heartbeat remain untouched.
