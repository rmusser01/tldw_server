---
id: TASK-13264
title: Prepare 0.1.44 release with all changes since v0.1.43 and land prior closure
  records
status: In Progress
created_date: 2026-09-27 15:16
priority: high
updated_date: 2026-09-27 15:31
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User authorized both remaining workstream items and preparation of a new release. Freeze dev at 9668e1454b0b28b7a4de13e1a35496fa0b368c42; carry local closure commit 62a2b70e; include PR2978 and all merged changes since v0.1.43. Prepare a reviewable release candidate without publishing until final release approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prior release closure records included in the candidate
- [x] #2 Exhaustive change inventory from v0.1.43 to frozen dev
- [x] #3 0.1.44 metadata, notes and protected-source licensing records consistent
- [x] #4 Required focused checks and packaging verification pass
- [ ] #5 Release PR and remaining publication approval gates documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-09-27-release-0.1.44-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Frozen range verified: 258 commits and 34 first-parent merged PRs since v0.1.43, including synchronization PR2971. Closure commit62a2b70e carried as6414a4a830. Candidate proposes release date2026-09-27 and Countdown2028-09-27T12:00:00Z; final legal/publication approval pending.

Protected source: `9668e1454b0b28b7a4de13e1a35496fa0b368c42`.
Protected manifest SHA-256: `fcc17101e2303e612b8ab9d7824a47bc937eea2a1f216e6e2dcb1b0f53ba9b30`.
Candidate contracts: 89 passed (four warnings), including strict MkDocs and protected-source checkout equality. CI-matching OpenAPI fingerprint passes. Local 0.1.44 wheel/sdist pass Twine and backend-only contents checks; final README metadata refresh is being rebuilt. Ruff, compilation and whitespace checks pass. Bandit main.py has zero findings/errors; licensing tests retain 82 baseline B101 assertion findings, unchanged severity/type counts. Main publication, new PR-specific summary/waiver and proposed legal-date approval remain pending. Full UAT remains separately tracked.
Final corrected README/docs/licensing matrix: 33 passed, four warnings; strict MkDocs and protected source equality pass. Final rebuilt 0.1.44 wheel and sdist pass Twine and backend-only contents verification. Independent release preparation review requested. Candidate remains unpublished.
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
