## Stage 1: Slot outcome ordering
**Goal**: Preserve failed-slot provenance and prevent older jobs from replacing newer slot state.
**Success Criteria**: Mixed variant outcomes retain Retry; old jobs cannot overwrite a newer batch's slot status.
**Tests**: Worker and repository regression tests for sibling completion order and cross-batch completion order.
**Status**: Complete

## Stage 2: Retry availability and fanout completion
**Goal**: Make historical legacy Retry availability accurate and finish fully processed fanout replays.
**Success Criteria**: The monitor disables Retry for an older recipe-less source; resumed fanout becomes completed when all children finished.
**Tests**: Service, frontend component, and fanout regression tests; OpenAPI drift check.
**Status**: Complete

## Stage 3: Review hygiene
**Goal**: Resolve valid performance, documentation, formatting, and test-classification comments without expanding production surface for test-only concerns.
**Success Criteria**: Blocking synchronous generation routes run off the event loop; new helpers are documented and formatted; tests are categorized; review threads have technical responses.
**Tests**: Focused endpoint tests, formatter, linter, and scoped suites.
**Status**: Complete

## Stage 4: Final verification
**Goal**: Recheck local and remote gates, update TASK-13358, and merge only when review and CI permit.
**Success Criteria**: No unresolved actionable review finding, green relevant checks, clean branch, and PR policy satisfied.
**Tests**: VN backend suite, frontend VN tests and typecheck, OpenAPI drift, Ruff, Bandit, and PR checks.
**Status**: In Progress

### Incremental Review Follow-Up (2026-09-25)

- Rebased on the latest `origin/dev` without conflicts. The rebased OpenAPI fingerprint passes the drift check.
- Record rejected parent enqueue outcomes atomically on the batch and its still-owned slots. Ignore late enqueue failures after the batch has advanced.
- Recover slot state when a persisted parent job resumes after a lost response, including children that already completed before fanout replay.
- Defer final slot/batch failure while a variant job has retries remaining. Record an exhausted failure atomically so fanout recovery cannot clear a real variant failure.
- Added seven regression tests; observed the failing enqueue, retry, and lost-response scenarios before applying fixes.
- Verification: the 312-test VN run had 308 passes, zero assertion failures, and four setup errors from disk exhaustion. All four passed on rerun in the approved temporary root. Final scoped Ruff passed and Bandit returned zero findings; prior frontend VN tests/typecheck remain applicable because this follow-up is backend-only.
- Tracking collision resolved: rebasing introduced an unrelated ADR task with the same TASK-13356 ID as the VN recipe task. With explicit requester approval, only the VN record was manually renumbered to TASK-13358; the ADR record and its references remain unchanged. Subsequent task updates use the Backlog CLI again.
- ADR check: ADR required: no for these correctness fixes. `Docs/ADR/003-jobs-vs-scheduler-default.md` continues to govern Jobs ownership; no new worker, persistence, or public API rule is introduced by this review follow-up.

### Latest Dev Rebase Verification (2026-09-26 UTC)

- Rebased onto `origin/dev` at `59bd584503` without conflicts; all five PR patches are unchanged according to `git range-diff`.
- Fresh verification: 312 VN backend tests passed without setup errors; 37 frontend VN tests, frontend typecheck, OpenAPI drift, scoped Ruff, and diff check passed. Bandit returned zero findings/errors.
- An initial typecheck environment failure was resolved by linking the existing monorepo dependency installation into the isolated worktree; no tracked dependency or configuration change was needed.
- Qodo and CodeRabbit had no unresolved findings before this push. Current-head re-review and all required remote gates remain prerequisites to merge.
- Merge method is `merge`, as required by the `dev` ruleset; no admin bypass or alternative merge method is permitted.

### Current-Head CI Diagnosis (2026-09-26 UTC)

- `backend-required` passed compilation, changed-module type checks, unit smoke, timestamp-sensitive tests, and startup smoke, then failed OpenAPI drift on head `440803c2a5`.
- The shared local environment used Pydantic 2.11.7, outside the declared `>=2.13.5,<2.14.0` range. A temporary dependency overlay using CI's Pydantic 2.13.5 reproduced its exact schema hash `f9cc19147feb9dc2bc438d19f4b7d5c8a2c9946596f715a47bff7fa69b4ddfe1`.
- Schema comparison found unchanged paths and VN schemas. The difference is Pydantic merging the equivalent OSCE patient-context input/output schemas and updating three references. Refresh the generated fingerprint and frontend types using the declared dependencies; no application-code or dependency-policy change is needed.
- Rebased onto `origin/dev` at `a2826f103f` without conflicts. `git range-diff` confirms all six PR patches unchanged. Only this task's new diagnosis notes were temporarily stashed and restored; unrelated stashes and work remain untouched.
- Refreshed the fingerprint and regenerated ignored frontend types. Fresh rebased-tree verification: 312 VN backend tests passed using CI-aligned schema libraries; 37 frontend VN tests, frontend typecheck, OpenAPI drift, scoped Ruff with documented exclusions, and diff check passed. Bandit returned zero findings/errors. The temporary shared-UI dependency link was removed after verification.
- Current-head Qodo/CodeRabbit review and all remote gates remain required after the generated-artifact update. Stage 4 stays In Progress until all ruleset gates pass and the merge is verified.

### Latest Dev Advance (2026-09-26 21:38 UTC)

- Confirmed the clean owned worktree and remote PR head `9bfeb497d849184e3c774ea83f3763ab260d0307` before rebasing onto `origin/dev` at `f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07`. The new base contains unrelated Sync blob-upload expiry work; no VN ownership overlap was found.
- Rebase completed without conflicts. `git range-diff` confirms all seven prior PR patches unchanged.
- Fresh verification: 312 VN backend tests passed using the CI-aligned temporary overlay; 37 frontend VN tests, frontend typecheck, OpenAPI drift, scoped Ruff with the documented BLE001/UP035 exclusions, and diff check passed. Bandit returned zero findings/errors. The temporary shared-UI dependency link was removed after verification.
- No runtime code, shared dependency installation, or unrelated work changed. The requester-owned Change summary remains verbatim. Current-head Qodo/CodeRabbit review and all required remote gates remain prerequisites to merge; Stage 4 stays In Progress.

### Malformed Snapshot Review Follow-Up (2026-09-26 UTC)

- Qodo's review of head `689bbe20eb` identified malformed stored Retry snapshots bypassing the documented conflict codes. All 18 API regressions failed before the fix, reproducing raw errors, incorrectly accepted retries, and inconsistent conflict codes.
- Validate consumed authored/execution slot fields with strict Pydantic models, preserving recorded values and unknown metadata. Share execution snapshot parsing between Retry and the worker. Invalid snapshots are rejected before creating any batch or job; absent legacy recipes retain their unavailable code.
- The first broader run exposed legitimate zero-variant lazy-depth slots. Matched the existing slot schema's `ge=0` bound and retained exact seed-count validation. A unit regression demonstrated the initial rejection and now verifies valid lazy-depth replay.
- Final verification: 336 VN backend tests passed, including 18 new API regressions and six recipe unit cases. The focused 81-test run also passed. Scoped Ruff with documented exclusions, compilation, OpenAPI drift, and diff checks passed; Bandit returned zero findings/errors.
- The 37 frontend VN tests and typecheck passed earlier in this same rebase. This follow-up changes no frontend or public schema. Stage 4 remains In Progress until the new head passes review and all remote gates, followed by a verified normal merge commit.
