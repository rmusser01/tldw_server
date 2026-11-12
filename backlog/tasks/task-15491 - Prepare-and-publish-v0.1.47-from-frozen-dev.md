---
id: TASK-15491
title: Prepare and publish v0.1.47 from frozen dev
status: In Progress
labels:
- release
- documentation
references:
- Docs/superpowers/plans/2026-11-11-release-0.1.47-plan.md
- https://github.com/rmusser01/tldw_server/pull/3454
- Docs/Evidence/PR3454-codeql-frontend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Promote reviewed dev through PR #3443 to main, preserving main-only updates and publishing coherent server release metadata and changelog after review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release candidate preserves the frozen main and dev histories and passes release metadata checks.
- [x] #2 Changelog summarizes notable shipped changes and upgrade requirements.
- [ ] #3 PR #3454 preserves the requester’s Change summary and approved release and Countdown dates, and CI passes before merge.
- [ ] #4 The reviewed main merge is tagged and published, server artifacts are verified, and main is synchronized back to dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This promotion follows the existing release and licensing process without changing policy or architecture. Candidate cutoff: dev 04905e02eafc34f3b8d82d30e013a81c660028d0 and main 25dbf5e7b26b973ae78ac302350589e720eb7980.
PR #3454 retains the requester’s original Change summary. The requester approved the 2026-11-12 release date and the 2028-11-12T12:00:00Z Countdown start, approved the CodeQL and pre-commit fixes, and waived additional human rationale for this release.
The reviewed starting candidate passed 333 focused tests and built the server wheel and sdist with GPL-3.0-only metadata and no protected frontend files. The CI repair separates the backend classifier checkouts by event and corrects whitespace in six files. The repair passed 336 focused tests and scoped pre-commit. Bandit found only the test assertion in new Python lines. The protected source manifest now includes the final newline of the extension document. Final GitHub CI, publication and dev reconciliation are pending. Full release UAT has not been run.
GitHub pre-commit passed on b8ceb43acf. CodeQL still flagged the conditional checkout, so the follow-up removes the inactive workflow-run path from backend-required. Pull requests still wait for the trusted license check, and manual diagnostics keep using their selected commit. All 357 focused workflow and ratchet tests passed with no new Bandit findings. GitHub CI must pass again on the final commit.
The requester approved the focused frontend CodeQL follow-up. Reviewed all 37 blocking frontend annotations against the exact scan. The candidate validates Audio Studio ticket destinations, constrains the workspace diagnostic link to the runtime config route, and replaces a dynamic test regex with literal matching. 588 focused Node 20 tests passed. The evidence records the separate WebUI credential persistence path and the reviewed test and cache traces. No alerts were dismissed. This is a bounded repair to existing contracts, with no new ADR required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
