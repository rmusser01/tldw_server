# Saved-history restoration repair after 0.1.44 publication

Tracking: TASK-13264.12; integration PR3033. These development changes are excluded from the immutable published v0.1.44 source and binaries.

## Stage 1: Reproduce and isolate
**Goal**: Complete coordinator execution at the existing 4096 MB heap limit and trace saved-target rejection.
**Success Criteria**: A bounded loader regression fails before the callback fix; saved settings restoration fails with temporary mode enabled.
**Tests**: Effect-owned local load restart; delayed saved settings profile/messages.
**Status**: Complete

The local loader depended on inline setters and a controller object that changes when capture publishes. Effect-owned restoration restarted after its own state updates. Snapshot current inputs when a deliberate invocation begins and retain a stable callback; keep every generation, mount, principal, restore and selection fence. The bounded reproduction observed two loads instead of one. Full execution then completed, exposing missing native capture records and a separate temporary-to-saved transition defect.

## Stage 2: Preserve saved and temporary ownership
**Goal**: Accept saved server targets without inheriting temporary draft mode; model authority correctly in tests.
**Success Criteria**: The store atomically clears temporary mode with a nonempty server ID, retains it on null, and the controller still refuses temporary history. Both fixture endpoints resolve exact conversation records through the real selection resolver and native response validator.
**Tests**: Store transitions, delayed settings/sidebar restoration, fresh native capture on remount, existing temporary controller rejection and account/cancellation suites.
**Status**: Complete

The existing temporary toggle detaches saved history. Its inverse, accepting a saved server ID, must retire temporary mode in the same store update. Do not change loadConversation to accept temporary owners or replace native capture with ambient list data. Fixture records and bookmarks reset between cases and remain available across intentional remounts within a case; ambient-list responses retain their own network delays and native capture retains principal/signal checks. Character revisits use the real session hook, whose explicit draft reset clears the controller before clearing the session. This production behavior was absent from the previous inert fixture.

## Stage 3: Verify and integrate
**Goal**: Finish PR3033 with complete execution, independent review and all seven required checks.
**Success Criteria**: No OOM, unfinished assertion or bypass; immutable tag/source/grants remain unchanged; newer dev ancestry is preserved before normal merge.
**Tests**: Complete coordinator and loader suites, ownership regressions, current-development release contracts, diff/syntax/security checks and required CI.
**Status**: In Progress

Local verification:86/86 coordinator/loader/privacy tests and 90/90 controller/persistence/server-loader/selection tests pass without skips, worker errors or heap increase. All 95 development release/licensing/docs contracts pass. Five-file ESLint has zero errors/no added warnings (16 existing warnings); transpile diagnostics have zero syntax errors. Independent review found no actionable blocker. Existing automatic invalidation guards remain unchanged. Bandit does not analyze this TypeScript-only delta; required security/CodeQL and all seven integration gates remain pending. The immutable tag and approved protected trees match; the new five-file development delta is excluded from published source/binaries.

### Preserve newer Persona development

Dev advanced to `df1fcc7a52306f400b843c8b0ea0bc90d0396056` via PR2817 during synchronization CI. The refreshed merge retains the incoming RLS file exactly, including all native and Persona owner policies, and both branches’ tests. Workflow contracts retain strict failure/job-shape checks plus upstream exact-head/base-status cases. The production SQLite guard and frontend restoration repairs are unchanged. Independent review found no actionable findings. Bandit reports zero production findings, zero errors and no new finding signatures relative to either parent (209 existing test findings). Immutable release grants/tag/source remain unchanged. Concurrent PR3035 separately synchronizes published 0.1.45; this work does not republish or claim verification of that release. Fresh integration tests and required CI are being recorded below.

The five development-only frontend repair paths also differ from released 0.1.45 main `96bf0996eb69fe33f59c6a3b6efd2d1944d52536`; neither published release contains these fixes.

Conflict-focused verification has clean exit results: 64 workflow/RLS contracts and 13 AUTOINCREMENT guard cases passed with isolated pytest temp directories. Both broader 290-test attempts executed every test without a reported failure, but stalled in pytest garbage collection at roughly 19 GB RSS and required interruption (exit130/143). Neither is counted as a clean suite pass. No tests or plugins were disabled.

### September 29 final development integration

Current dev is `0da68530e80c713ed3a323a741998e1fed37e3e9`. PR3035 already synchronized 0.1.45; preserve its release notes and all subsequent development. Incoming product Playground, history controller, PostgreSQL policies, strict SQLite guard and workflow contracts are retained exactly. The remaining PR3033 delta is the five saved-history repair paths, test metadata and publication/closure records. The coordinator fixture also retains upstream history-context reset when clearing the persisted session.

TASK13390 independently implemented and verified explicit URL handoff admission in dev. That newer behavior is preserved; the earlier declined Qodo handoff recommendation is a historical disposition superseded by this upstream fix. All four PR3033 review threads are resolved.

Fresh verification: 128 current-development release/licensing/workflow/docs contracts exited0; 120 loader/ownership/privacy tests and all 86 coordinator/search tests passed. The complete coordinator/search run uses the existing required-CI 15-second test deadline and unchanged4096MB heap, with no skips, unfinished assertions or worker crash. Earlier local five-second runs hit timeouts; the first affected cases passed unchanged in isolation. A stale temporary OCR dependency link was repaired using the existing installed package, without repository changes.

Independent merge review found no actionable findings. Five-file ESLint has zero errors and16 unchanged warnings; syntax diagnostics are clean, without a whole-project semantic-typecheck claim. Python test metadata passes Ruff/compilation; Bandit retains two existing assertion findings with zero errors and no production Python delta. Required checks all passed on prior headb2773de89a; the refreshed head must pass all seven contexts before normal merge. Frozen tag/source/grants/artifact verification remains unchanged; fresh full UAT is not claimed.
