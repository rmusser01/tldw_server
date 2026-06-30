---
id: TASK-2233
title: Update changelog for post-0.1.31 PRs
status: Done
labels:
- changelog
- release-notes
references:
- https://github.com/rmusser01/tldw_server/pull/1982
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a new changelog entry covering merged PR scope since the last documented version push (0.1.31).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify the last documented version boundary.
- [x] #2 Collect merged PRs after that boundary through the current dev head.
- [x] #3 Add a concise changelog entry matching existing style.
- [x] #4 Run lightweight validation for markdown/diff hygiene.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-03: Last documented version boundary is `CHANGELOG.md` section `0.1.31` and commit `42d9c31b0b` (`docs(changelog): add 0.1.31 entry`, 2026-04-19).
- 2026-06-03: Recomputed PR-numbered merge/squash coverage on current `dev` after rebasing onto PR #2255. Found 921 PR-numbered commits after `42d9c31b0b`, from older backfills #861/#961/#971/#1072 through PR #2255.
- 2026-06-03: Added an Unreleased rollup in `CHANGELOG.md` grouped by sandbox/MCP/CodeGraph, research/study workflows, onboarding/provider/admin surfaces, backend/API architecture, design-system UX migration, CI/type-check reliability, security hardening, and product workflow stabilization.
- 2026-06-03: Validation: `git diff --check` passed; exact conflict-marker scan passed; custom Node check confirmed `Unreleased` still has Added/Changed/Fixed/Removed sections. `prettier --check CHANGELOG.md` reports the existing changelog file as not Prettier-formatted, so no whole-file formatting rewrite was applied.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an Unreleased changelog rollup covering all 921 PR-numbered merge/squash commits present on current `dev` after the `0.1.31` changelog push (`42d9c31b0b`, 2026-04-19) through PR #2255. The entry groups the branch work into platform, research/study, onboarding/provider/admin, backend architecture, design-system UX, CI/reliability, security, and workflow-stabilization areas.
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
