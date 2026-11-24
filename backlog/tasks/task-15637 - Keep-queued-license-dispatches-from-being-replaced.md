---
id: TASK-15637
title: Keep queued license dispatches from being replaced
status: In Progress
labels:
- ci
- merge-queue
dependencies:
- TASK-13462
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merge queue repeatedly dispatches the license workflow while the previous request is still queued and has posted no status. Its concurrency group cancels the older request before a runner is assigned. Bind dispatch metadata to the intended PR/head and reuse matching live requests so the current policy check can retain its place, preserving existing trusted evaluation and CI admission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A matching queued or in-progress trusted license dispatch suppresses a replacement even when no license commit status exists.
- [x] #2 The dispatch identity includes the intended PR and head; other PRs/heads, wrong workflow/ref/event, and completed or cancelled runs do not suppress a necessary new dispatch. Legacy manual pr-only dispatch remains supported.
- [ ] #3 Regression tests, affected CI/workflow contracts, actionlint, scoped baseline/current static and Bandit checks, and independent actual/final-staged review are recorded without changing policy authority, admission, concurrency or retry limits.
- [ ] #4 The fix lands through normal required CI and its actual merge/head/dev ancestry is independently verified.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User approval (2026-11-24): create and land one separate queued-license deduplication fix PR, with regression tests and normal required CI; its human Change summary is explicitly waived. Continue monitoring PR3584. This waiver does not authorize merging the later UAT516 closeout PR.

Approved design: give trusted license dispatches an intended PR/head identity in workflow run metadata; recognize a matching queued/in-progress dispatch before a commit status exists; preserve policy evaluation, authority, concurrency, admission, gate names and retry limits. Keep manual pr-only dispatches supported.

Observed PR3584 evidence: on unchanged PR head e68820f8f0f6d4e4f2858daeaa5f048f55cf3b71 / base 518bda3da3db0403a239bfc5350d7e6ddd781eed, ten dispatched license jobs were cancelled before assignment (runner_id=0, empty steps). Complete annotations explicitly name a higher-priority waiting frontend-license-gate-3584 request. Runs: 45448700839, 45449269107, 45449268891, 45459115336, 45459139102, 45469798067, 45470371837, 45473358327, 45477675151, 45480306297. The queue's _appeared status branch reads posted statuses only, so an unassigned dispatch is invisible and another start replaces it. No policy-test failure or log evidence is claimed; no logs fetched.

Search before creation: no existing queued-license deduplication task or open license/queue fix PR found. TASK-13462 is the shipped original queue; TASK-13639 covers held gate approval, not this queued-status gap. TASK-13264.12's license wait is an older frontend coordinator dependency, not this repair.

ADR check: ADR required: no. Docs/ADR/063-in-repo-merge-queue.md describes the implemented queue design (its file remains Proposed and is not promoted here); this repairs its existing current-head/idempotent execution behavior without changing a workflow gate or authority rule. ADR-059 and ADR-084 govern task editing and unique IDs. Accepted rationale remains unchanged.

Plan: Docs/Plans/IMPLEMENTATION_PLAN_queued_license_dispatch_dedup_20261124.md. Stages: 1 investigation Complete; 2 regression RED/GREEN In Progress; 3 scoped verification and independent review Not Started; 4 normal publication/required CI/verified landing Not Started. Use this task and plan as the durable progress record; no new proof bundle, worktree or export.

Ownership: branch codex/ci-queued-license-dedup-20261124 in the existing worktree, base 518bda3da3db0403a239bfc5350d7e6ddd781eed. PR3584's local/remote branch is not edited while armed. UAT516 closeout remains in owned stash 7d34260a0bca99efa7a57a77fca6110e72070e22, exact 7350-byte patch SHA256 c0065d932b21448f6a58641efc689caa1bd36b0beb816810b5612bf2fce89549. No installs, app/server/browser/native/model starts, model-cache reads, operator SQL, hook bypass, settings change or protected WIP/cache cleanup.
Regression baseline: Python3.12.13 original actions/license suites 244PASS/1existingSKIP in9.93s. Before production changes, the 23behavior cases produced13RED/10controlPASS: ten live-state/start-or-retry cases, later-page identity, and API-error/page-cap guards exposed replacement dispatches or missing fail-closed behavior; three workflow schema/title/env contracts also RED. After the bounded helper and optional metadata input, actions/license suites268PASS/1existingSKIP in8.89s; four related queue rules/workflow/gate/frontend suites178PASS in5.56s, disjoint from the268. The skip is the existing local Bash<4 shared owner-evaluation case; it remains unverified locally and unchanged for hostedLinux. No server/browser/model starts or CI retries.
Static/workflow validation: Python3.12syntax3filesPASS. Ruff baseline=current: production0, actions2inherited, dispatch0; zero new findings. Full baseline/current Bandit production0/0; test files336/345 raw findings, net9new B101 diagnostics for pytest assertions only (no executable security finding). Updated assertion AST fingerprints are not distinct vulnerabilities. No assertion removal, suppression, baseline or tool-policy edits. Existing cached local actionlint source build (reports devel, Go1.26.2) passes the touched workflow and ALL workflows; this does not substitute for hosted pinnedactionlint1.7.12. Canonical backlog normalization and git diff --check PASS. Independent actual review requested; final-staged review and normal current-head hosted landing pending.
Independent actual review found a queued-to-in_progress transition gap in alphabetical status queries. Added a real queue/API transition test: it RED-dispatched a replacement before the correction. Read pre-run states before in_progress; final six affected queue suites448PASS/1inheritedBash3SKIP in11.13s, including a mixed-context control that still approves the unrelatedheldE2E gate. Reviewer re-review CLEAR. Final full Bandit counts: production0/0; actions baselineB404=1+B101=274 versus currentB404=1+B101=282, dispatchB101=61versus64. Exactly11netaddedB101 assertions, no othernewdiagnostic; asserts are intentional pytest verification, not optimized runtime security controls. No suppression or baseline writes. Three-file syntaxPASS/Ruff0new; final-staged independent review still pending.
Expanded license/admission scope: five additional suites initially277PASS/4inheritedBash3SKIP/1expectedmetadata-contractFAIL because the exact workflow top-level surface omitted the newly authorized run-name. Updated that contract to allow ONLY run-name and pin its exact PR/head expression; all permissions, concurrency and privileged job/step assertions remain exact. No production change. Verify the corrected suite and include it in final-staged review. Existing read-only cache preflight15healthy; configured pre-commit stage passed without any environment initialization/install (manual local-ci-fast and scoped wizard Ruff/Black not executed for these paths). The normal git pre-commit hook file is absent; do not claim the later git commit executed it.
Corrected privileged-surface suite14PASS in1.62s (13overlap earlier passing cases; not14extra). Final distinct affected verification:726PASS/5existingBash<4SKIP across11suites. Four touched Python files syntaxPASS; Ruff0new (actions2inherited). Additional contract Bandit baselineB101=74+B105=1/currentB101=75+B105=1. Across all touched tests, twelve net addedB101 pytest assertions only; inheritedB404/B105 unchanged. Existing policy audit/dispatch env+steps, concurrency, admission, required context names and retry rules unchanged. Final-staged review, publication and hosted CI/actual merge remain open; task staysInProgress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
