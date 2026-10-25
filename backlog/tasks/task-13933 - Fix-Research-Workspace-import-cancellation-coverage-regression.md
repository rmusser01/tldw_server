---
id: TASK-13933
title: Fix Research Workspace import cancellation coverage regression
status: Done
labels:
- frontend
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the real advisory coverage failure in ResearchWorkspace.stage2.responsive.test.tsx from run40139725463/job121014004169. Reproduce the workspace-switch scenario deterministically, preserve late-result cancellation coverage, and correct the test only if production behavior is already correct. Keep sidebar work paused and preserve existing worktrees and evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original CI failure is reproduced deterministically and its cause recorded.
- [x] #2 The test models a real active-workspace switch and rejects stale import side effects and stale handoff UI.
- [x] #3 Relevant unit and coverage checks, type/lint applicability and independent review pass without weakening workflows or tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Source failure: final merged header head fad63f80edfae36ae5422eaef8054560fc11bf31, shared-ui-coverage5/16,1615pass/1fail; log /private/tmp/pr3311-coverage5-fad63f.log. Unchanged isolated test passes locally on official Node20.20.2, confirming timing dependence rather than a local deterministic red. Source trace shows activeDeepResearchReturnContext is null for a different workspace, hiding its banner; the mocked store selector does not subscribe, so changing testState.workspaceId only affects UI when another async update happens to rerender. canContinueImport reads current workspace via getState before adding artifacts. Plan IMPLEMENTATION_PLAN_research_workspace_cancellation_20261024.md. ADR required: no; bounded test-contract repair changes no durable architecture or ownership rule. Sidebar stays paused.
Deterministic red confirmed on Node20.20.2: adding only rerender after mocked workspaceId change reproduced the original missing-handoff lookup at the unchanged final assertion (/private/tmp/research-workspace-cancellation-red.log and -red.patch). Component derives active return context from workspace identity; hiding the prior handoff is correct. Existing neighboring workspace-switch tests already call rerender after changing this non-subscribing mock. Corrected case requires in-flight import, explicit workspace-switch render, absence of old handoff before/after late result, and retains strict no-addArtifact side-effect assertion. No production change is needed or made.
Corrected full responsive+bundle+route suites pass45/45 under CI=true official Node20.20.2. Verified two temporary mutations fail for the intended reasons: removing the live current-workspace import guard adds an artifact and fails the retained no-side-effect assertion; removing return-context workspace identity renders the stale handoff and fails its absence assertion. Both production files restored byte-for-byte and git diff confirms no production delta. Logs /private/tmp/research-workspace-cancellation-green.log and research-workspace-mutation-{current-workspace-guard,handoff-workspace-identity}.log. Original full coverage5/16, frontend TypeScript, scoped lint comparison and independent review now running.
Independent research_cancellation_review found no issues and judged this the smallest complete test-only repair, confirming real Zustand subscriptions vs plain-object mock and both cancellation protections. Normal frontend TypeScript noEmit passed with8GB heap. Installed scoped ESLint actually checked the test:0errors/18inheritedwarnings, same rule/severity/messages as exact dev baseline; initial wrong UI CLI path was corrected to the existing frontend-owned ESLint, no install/config change. Bandit scanned zero Python lines and gives no TypeScript security clearance. Initial full coverage5/16 passed the repaired28-case ResearchWorkspace suite but failed59cases in5other files because pnpm-owned react-virtual loaded a second React18.3.1 instance while RTL/ReactDOM used Bun-owned React18.3.1. Stack/resolve evidence identified exact runtime split; a temporary local-only config aliases React and inlines react-virtual. All63cases across those5files then passed. Original config/dependencies unchanged; full shard with temporary runtime normalization now running, temporary file will be removed before commit. Evidence /private/tmp/research-workspace-{coverage5-fixed,single-react-check,coverage5-single-react}.log and eslint-comparison.json.
Final verification: exact shared-UI coverage shard5/16 passed1616/1616 tests across166 files under CI=true, official Node20.20.2 and unchanged V8 coverage settings, with local-only React runtime alignment for the confirmed pnpm/Bun split. Log /private/tmp/research-workspace-coverage5-single-react.log, report /private/tmp/research-workspace-coverage5-single-react-report. Initial original-config run passed repaired responsive28/28 but had59 mixed-runtime failures in5other suites; after normalized-runtime focused63/63 pass, the whole shard passed. Temporary config preserved privately /private/tmp/research-workspace-local-vitest.config.ts and removed from checkout before commit. Normal shipped Vitest config, dependencies and all production files remain byte-identical.45 focused responsive/bundle/route tests, two intended-failure mutation checks, TypeScript, zero lint errors/no new warnings, whitespace/canonical format and independent review all completed. Bandit zero Python scope. No timeouts, skips, workflows or assertions weakened. Draft PR publication follows; hosted CI for new branch is not yet verified and no merge is authorized by this task. Sidebar stays paused.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired the timing-dependent ResearchWorkspace import-cancellation test. The plain mocked store did not notify React after workspaceId changed, and the final assertion incorrectly required the stale handoff banner to remain. Explicit rerender now models the real Zustand update; the test proves import is pending, the old banner disappears before and after late completion, and no artifact is added. Deterministic red reproduced the exact hosted failure; two controlled mutations verified workspace write and UI identity guards remain protected.1616/1616 original-shard tests pass with temporary local React alignment,45 focused tests pass, TypeScript/lint/canonical checks pass and independent review has no findings. Temporary wiring removed; no production, dependency or workflow changes. Hosted CI remains pending for the new branch; sidebar stays paused.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
