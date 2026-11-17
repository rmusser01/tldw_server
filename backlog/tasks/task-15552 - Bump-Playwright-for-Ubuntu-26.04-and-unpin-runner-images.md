---
id: TASK-15552
title: Bump Playwright for Ubuntu 26.04 and unpin runner images
status: In Progress
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
- [ ] #1 Playwright/browsers install and run on ubuntu-26.04 (CI proof)
- [ ] #2 All 34 runs-on pins reverted to ubuntu-latest; contract ratchet green
- [ ] #3 At least one full required-gate wave green on the unpinned image
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-11-16: @playwright/test+playwright bumped ^1.51/^1.57 → ^1.61.0 in tldw-frontend, extension, and the workspace ROOT (root is what CI's 'bunx playwright install' resolves — the critical pin; resolves 1.65.1, Chromium 159 v1255 ≥ the 149 ubuntu-26.04 threshold). All 33 runs-on: ubuntu-24.04 pins reverted to ubuntu-latest across 9 workflows. Contract ratchet: 100/100 green unpinned. Playwright 1.65 parses the suite: ux-regression 33 tests/9 files + workflows 731/103, exit 0 (skills-certification/onboarding-uat dirs env-gated by design, CI supplies vars). Ubuntu-26.04 browser-install proof rides this PR's required-gate wave on ubuntu-latest (now 26.04). Bandit: no Python touched — non-code skip.
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
