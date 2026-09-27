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
**Goal**: Recheck local and remote gates, update TASK-13378, and merge only when review and CI permit.
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

### Exact Recipe Version Review Follow-Up (2026-09-26 UTC)

- Qodo cleared head `8c2811e37b`, but CodeRabbit inline comment `4112954271` identified JSON boolean versions accepted as version 1. Both authored and execution loaders used equality-only checks; floating-point 1.0 had the same issue.
- All eight new regressions failed before the fix: four real Retry API cases returned 202 instead of the documented 409, and four loader cases accepted non-integer versions. Require an exact integer version in both loaders before comparing to the supported version, preserving valid snapshots and existing error codes.
- Fresh verification: the focused 26-test run and all 344 VN backend tests passed (181.62s full run). Scoped Ruff with documented exclusions, compilation, OpenAPI drift, and diff checks passed; Bandit returned zero findings/errors. The 37 frontend VN tests and typecheck from this same dev rebase remain applicable because no frontend or public schema changed.
- Confirmed owned remote head `8c2811e37b3ade6663f114f7cf462d4268f32dbf` and unchanged dev `f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07` before publishing. Reply in the inline review thread and obtain current-head re-review; Stage 4 remains In Progress until all required checks and the normal merge complete.

### MCP-Base Rebase Verification (2026-09-27 UTC)

- Confirmed a clean owned worktree and remote head `674d13d1458554e07e74b80ea615d74705c1cac1` before rebasing onto dev `f94375c26e457be1f7752f20c9f11102f2503e42`. The base advance contains unrelated MCP filesystem/test helpers and task records; no VN runtime ownership overlap was found.
- Rebase completed without conflicts. `git range-diff` confirms all ten prior PR patches unchanged.
- Fresh verification: all 344 VN backend tests passed (304.95s) using the CI-aligned temporary overlay; 37 frontend VN tests, frontend typecheck, compilation, OpenAPI drift, scoped Ruff with documented BLE001/UP035 exclusions, and diff checks passed. Bandit returned zero findings/errors. The temporary shared-UI dependency link was removed.
- The new base introduced an unrelated MCP task also using TASK-13358. Both task files were preserved unchanged while requester approval was requested for a scoped manual renumber because Backlog's CLI has no supported renumber command. The approved resolution is recorded below.
- Prior-head Qodo and CodeRabbit reviews were clear, but the rebased head requires fresh review and all current dev gates. Publish only with an explicit lease protecting the verified owned remote head. Stage 4 remains In Progress.

### Approved Tracking Collision Resolution (2026-09-27 UTC)

- The requester explicitly approved manually renumbering only the VN task and its references. Checked task filenames across all 151 registered worktrees, including active, draft, archived, and completed records; the highest existing ID was TASK-13377. Renumbered the VN record to the unused TASK-13378 and updated its current spec, plan, and PR references.
- Preserve the task's complete history, including earlier IDs TASK-13356 and TASK-13358. The unrelated MCP TASK-13358, its parent TASK-13291, and their references remain unchanged. Subsequent task updates use the official Backlog workflow again.
- This tracking-only change does not alter runtime code, tests, dependencies, or generated schemas. The fresh 344 backend tests, 37 frontend tests, typecheck, OpenAPI, Ruff, compilation, and Bandit verification from this same dev base remain applicable. Verify task lookup, task-ID uniqueness, preserved task history, unchanged MCP records, and diff checks before publishing.
- Current-head reviews and every required dev gate remain mandatory. TASK-13378 and Stage 4 stay In Progress until the normal merge is verified.

### Zero-Variant Retry Review Follow-Up (2026-09-27 UTC)

- CodeRabbit discussion `4113825904` identified a valid zero-variant lazy-depth recipe becoming a failed Retry target after parent enqueue rejection. Retry admitted a zero-work batch that could not become terminal and could block later lazy depth generation.
- All five new regressions failed before the fix: four real Retry API combinations (explicit/implicit source and existing/absent slot failure provenance) returned 202 rather than 409; the enqueue-rejection test changed an untouched depth slot from planned to failed.
- Reject zero-variant Retry sources before batch creation using the existing documented `vn_asset_retry_source_unavailable` conflict. Mark enqueue failure only on slots with planned variants. Keep zero-variant snapshots valid for normal worker replay and verify that later background approval can still schedule one lazy depth variant.
- Fresh verification: the focused 94-test run and all 349 VN backend tests passed (303.35s full run), including five regressions and successful lazy depth scheduling after rejected Retry. Compilation, scoped Ruff with documented BLE001/UP035 exclusions, OpenAPI drift and diff checks passed; Bandit returned zero findings/errors. The 37 frontend VN tests and typecheck from this same dev base remain applicable because this follow-up changes no frontend or public schema.
- TASK-13378 records this scoped follow-up. Confirm remote ownership before publishing, reply in CodeRabbit's inline thread, and obtain review for the new head. Current-head review and all remote gates remain prerequisites to merge. Stage 4 stays In Progress.

### Active Fanout Retry Review Follow-Up (2026-09-27 UTC)

- CodeRabbit discussion `4113887376` identified manual slot Retry overlapping an automatically resumable failed fanout. Real Jobs tests reproduce queued/processing parent jobs, lost parent enqueue responses, and surviving queued/processing children after the parent stops. All eight API combinations returned 202 rather than 409 with the admission guard absent.
- Reject replacement Retry while this owner-scoped source batch has queued or processing work. Read its parent and idempotent variant jobs across both queues in one Jobs snapshot, bounded by the accepted recipe's variant count plus one parent. Preserve automatic recovery and sibling work instead of cancelling the source batch. Return documented `409 vn_asset_retry_source_active` with a wait/refresh recovery message before creating a new batch or job.
- Tests verify that the original fanout still resumes, and that once source work is terminal the rejected idempotency key can admit a faithful Retry. Existing recipe/source validation remains before this guard. Fresh verification passed: 39 focused tests and all 357 VN backend tests (296.45s); compilation, scoped Ruff with documented BLE001/UP035 exclusions, OpenAPI drift and diff checks passed; Bandit returned zero findings/errors. The 37 frontend VN tests and typecheck from the same dev base remain applicable because no frontend or public schema changed.
- Publish only from the verified owned head, reply in the inline thread with the admission-guard rationale, and obtain new-head review. Stage 4 and TASK-13378 remain In Progress until exact-head review, required CI and a verified normal merge.
