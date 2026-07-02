---
id: TASK-12091
title: Prep 0.1.34 release from current dev
status: Done
assignee: []
created_date: '2026-07-02 02:52'
updated_date: '2026-07-02 02:58'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2557'
  - 'https://github.com/rmusser01/tldw_server/pull/2568'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare release metadata, changelog, README, and documentation version references for a new release based on the current origin/dev tip after PR #2557 and PR #2568 merged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package and docs version metadata are advanced from 0.1.33 to the new release version
- [x] #2 CHANGELOG has a new release entry covering work since 0.1.33
- [x] #3 README release line and rollup notes describe the new release state
- [x] #4 Relevant release/docs contract and packaging checks pass locally
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Prepared 0.1.34 from current origin/dev tip 30495536d3 after PR #2557 and PR #2568 merged.
- Updated pyproject package version, Docs/mkdocs version metadata, README current release line/rollup, and CHANGELOG 0.1.34 entry.
- Added release-helper regression coverage so the helper updates the repository README wording now used by the docs contract. Red/green: the new test first failed with Missing README anchor for beyond-release reference, then passed after Helper_Scripts/release.py was patched.
- Verification passed: release/docs/helper test slice (54 passed), git diff --check, Bandit on Helper_Scripts/release.py and the release docs contract test with B101 skipped for pytest asserts, pre-commit on touched files, package build, twine check, and wheel metadata check showing tldw-server 0.1.34.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared the current dev tip for release 0.1.34. Updated root package metadata, MkDocs version metadata, README release status/rollup copy, and CHANGELOG with a 0.1.34 entry covering PR #2568 follow-ups after 0.1.33. Hardened the release helper so it can update the repository README wording used by the docs contract, with a red/green regression test. Validation passed for release/docs helper tests, package build, twine artifact checks, Bandit, pre-commit, and wheel metadata. Known skip/blocker note: formal make release was not run because the release helper is intentionally main-only; this branch prepares dev metadata for the release path.
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
