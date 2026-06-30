---
id: TASK-12059
title: Update README for 0.1.32 release prep
status: Done
labels:
- docs
- release
modified_files:
- README.md
- backlog/tasks/task-12059 - Update-README-for-0.1.32-release-prep.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh README release status, quickstart, and current feature summary so it matches the 0.1.32 release/changelog state for PR #1982.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated README.md for the 0.1.32 release-prep state. Clarified that 0.1.32 is the current beta release-prep line, named the canonical PyPI package as `tldw-server` while keeping publishing marked in progress, aligned the dev/main merge wording for PR #1982, and replaced stale "What's New" bullets with a concise 0.1.32 rollup matching CHANGELOG.md. Verification: `git diff --check -- README.md` passed. Bandit skipped because the change is documentation-only.
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
