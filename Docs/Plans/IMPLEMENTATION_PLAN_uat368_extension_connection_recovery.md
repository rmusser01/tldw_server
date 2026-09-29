# UAT368: fresh extension connection recovery

Task: TASK-13260.277.18. Scope: shared UnifiedSetupWizard, MultiUserExitPanel, and their causal UI tests. Root owns native acceptance, frozen candidates, global tracker, and git integration.

## Diagnosis and approved design

The packaged extension uses the shared options HashRouter. With no server URL, first-run metadata/state cannot load and the wizard hides normal navigation. Selecting Multi-user only changes the local step; the existing Sign in callback is supplied only when server metadata confirms multi-user mode. Skip still attempts a server mutation despite having no loaded state. The known-route workaround reaches the existing native connection settings.

Add local navigation to `/settings/tldw` when metadata is unavailable or loading failed, including the Multi-user guide panel. Keep confirmed-multi Sign in and its auth-mode update. Offer Skip only with a loaded server state; continue adopting the actual server Skip response. Do not guess server mode, invent completion, modify transport, or revive the legacy onboarding entrypoint.

## Stage 1: Causal reproduction
**Goal**: Reproduce missing-server and Multi-user dead ends through the real wizard and routing.
**Success Criteria**: New tests fail for the missing settings action and dead-end Skip; existing confirmed-multi and server Skip controls remain meaningful.
**Tests**: Fresh null metadata/state with load error; local Multi-user path; actual HashRouter destination; metadata failure with valid server state.
**Status**: Complete

## Stage 2: Minimal recovery
**Goal**: Expose native connection settings and guard unavailable Skip.
**Success Criteria**: All causal tests pass; unknown metadata navigation does not write auth configuration or setup progress; confirmed-multi Sign in is retained.
**Tests**: Wizard suite and relevant onboarding/setup route controls.
**Status**: Complete

## Stage 3: Verification and native handoff
**Goal**: Verify source scope and provide a frozen candidate for root-coordinated native acceptance.
**Success Criteria**: Focused/surrounding tests pass; matched type/lint comparison introduces no diagnostics; security tooling limitations recorded; native packaged extension confirms URL/login settings and valid progress behavior.
**Tests**: Installed Vitest/type/lint entrypoints; scoped diff check; Bandit where applicable; root native extension acceptance.
**Status**: In Progress

## Evidence

Native initial failure: `.tmp/uat-frontend-repair1-20260920/sqlite-multi-ext-007` through `011` (onboarding, Multi-user guide-only exit, failed Skip, direct native settings workaround). UAT004 is related history, not assumed to have the same cause.

Causal RED: `/tmp/uat368-causal-red.log`, 5 failed / 31 passed. Failures cover no local settings action from fresh setup paths and Multi-user guide using the real HashRouter, dead-end Skip before state loads, metadata-only recovery with a valid server Skip, and the existing local Multi-user guide path.

Causal GREEN: `/tmp/uat368-causal-green.log`, 36/36 passed. Surrounding GREEN: `/tmp/uat368-surrounding.log`, 13 suites / 207 tests passed, including wizard authority/model handoff, provider/first-chat/readiness panels, setup hooks and service, and options setup route resolution. Commands use the installed `node node_modules/vitest/vitest.mjs run ... --maxWorkers=1 --no-file-parallelism` from `apps/packages/ui`.

Matched scoped ESLint comparison against HEAD: 0 errors / 0 warnings before and after, `/tmp/uat368-lint-comparison.json`. The shared frontend configuration emits its existing missing-pages-directory advisory when invoked at the repository root; no source diagnostics. Recursive Bandit ran with the project virtualenv and found 0 Python LOC in this TypeScript-only scope, `/tmp/bandit_uat368.json`; it does not provide TypeScript security coverage. Manual source review confirms the new callbacks only navigate to the existing local settings route and do not mutate credentials or setup progress.

Matched frontend TypeScript comparison: 93 existing diagnostics before and after, no added/removed or touched-file diagnostics, `/tmp/uat368-type-comparison.json`. The compiler host substituted only the two production files from HEAD `2173084838a726f96ab4fad0e8632880b90d6fe2` for baseline, using the same installed compiler/config and current surrounding source. The default Node heap exhausted at 4 GB (`/tmp/uat368-type.log`); the same script passed comparison with `node --max-old-space-size=12288 /tmp/uat368-typecheck.cjs` (`/tmp/uat368-type-retry.log`). Scoped `git diff --check` passes. Final source manifest: `/tmp/uat368-changed-files.txt`.

Native acceptance remains root-coordinated and pending. Source tests prove route navigation with the real HashRouter, not packaged browser/network/database execution. Keep the task In Progress and retain the plan until acceptance.

Independent read-only review by `/root/review_uat363` is clear: no blocker in local settings routing, unknown-state Skip guarding, confirmed-multi auth handoff, or adopting the actual Skip response. Four suites / 65 tests passed (`/private/tmp/uat368-independent-review.log`, `/private/tmp/uat368-independent-controls.log`); reviewed SHA manifest `/private/tmp/uat368-independent-manifest.txt`. Native packaged acceptance is the remaining validation.
