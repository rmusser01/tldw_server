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
