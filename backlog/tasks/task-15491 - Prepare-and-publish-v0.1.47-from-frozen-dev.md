---
id: TASK-15491
title: Prepare and publish v0.1.47 from frozen dev
status: In Progress
labels:
- release
- documentation
references:
- Docs/superpowers/plans/2026-11-11-release-0.1.47-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Promote reviewed dev through PR #3443 to main, preserving main-only updates and publishing coherent server release metadata and changelog after review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release candidate preserves the frozen main and dev histories and passes release metadata checks.
- [x] #2 Changelog summarizes notable shipped changes and upgrade requirements.
- [ ] #3 The release PR passes required checks and receives human Change summary and license date approval before merge.
- [ ] #4 The reviewed main merge is tagged and published, server artifacts are verified, and main is synchronized back to dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This promotion follows the existing release and licensing process without changing policy or architecture. Candidate cutoff: dev 04905e02eafc34f3b8d82d30e013a81c660028d0; main 25dbf5e7b26b973ae78ac302350589e720eb7980.
Draft promotion PR: https://github.com/rmusser01/tldw_server/pull/3454. The final local candidate passed 285 focused tests including strict MkDocs and protected-source equality (8,300 files); the main-base task format ratchet and backlog ID check passed separately. The FastAPI entry point compiles on Python 3.12 and scoped Bandit reports zero findings. Proposed release date 2026-11-12 and Countdown start 2028-11-12T12:00:00Z remain pending human approval, as does the human Change summary. Full runtime/UAT and final-head CI are not claimed.
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
