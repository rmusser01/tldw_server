---
id: TASK-15552
title: Bump Playwright for Ubuntu 26.04 and unpin runner images
status: Done
labels:
- ci
- playwright
dependencies:
- TASK-15483
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15483 pinned 34 jobs (18 playwright + 16 CI backend) to ubuntu-24.04 because the pinned Playwright cannot install chromium on ubuntu26.04-x64 and 26.04 drops ffmpeg from PATH. Bump Playwright (and any pinned browser versions) to releases supporting 26.04, verify browser install on a 26.04-equivalent image, then revert the runs-on pins to ubuntu-latest. Watch the workflow-contract ratchet (tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py) — it asserts exact job shapes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Playwright/browsers install and run on ubuntu-26.04 (CI proof)
- [x] #2 All 34 runs-on pins reverted to ubuntu-latest; contract ratchet green
- [x] #3 At least one full required-gate wave green on the unpinned image
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-11-16: @playwright/test+playwright bumped ^1.51/^1.57 → ^1.61.0 in tldw-frontend, extension, and the workspace ROOT (root is what CI's 'bunx playwright install' resolves — the critical pin; resolves 1.65.1, Chromium 159 v1255 ≥ the 149 ubuntu-26.04 threshold). All 33 runs-on: ubuntu-24.04 pins reverted to ubuntu-latest across 9 workflows. Contract ratchet: 100/100 green unpinned. Playwright 1.65 parses the suite: ux-regression 33 tests/9 files + workflows 731/103, exit 0 (skills-certification/onboarding-uat dirs env-gated by design, CI supplies vars). Ubuntu-26.04 browser-install proof rides this PR's required-gate wave on ubuntu-latest (now 26.04). Bandit: no Python touched — non-code skip.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged via merge queue: PR #3487 -> dev (fcb91ca4fc, 2026-11-17). @playwright/test+playwright bumped to ^1.61.0 across tldw-frontend (incl. the direct playwright dep), extension, and the workspace root (the pin CI's bunx resolves); lock resolves 1.65.1/playwright-core 1.65.1 only (was dual 1.58/1.65). All 33 ubuntu-24.04 runs-on pins reverted to ubuntu-latest; ubuntu-26.04 browser install+run proven by the PR's own required-gate wave (ubuntu-latest = 26.04). Queue eviction en route caught two real 1.61+ incompatibilities CI-proven fixed: context.route() now returns Promise<Disposable> (call-site adapted) and the dual-core type split (typecheck 0 repo-wide). Contract ratchet 100/100.
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
