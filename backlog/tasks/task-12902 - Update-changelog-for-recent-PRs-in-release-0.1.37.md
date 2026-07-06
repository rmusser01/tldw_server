---
id: TASK-12902
title: Update changelog for recent PRs in release 0.1.37
status: Done
labels:
- docs
- release
- changelog
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2672
modified_files:
- CHANGELOG.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update CHANGELOG.md for the recent PRs included in the 0.1.37 release PR follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated CHANGELOG.md 0.1.37 to cover the recent dev PRs queued for the release: PRs #2657, #2659, #2661-#2671, #2673-#2675, PR #2656 sync context, and PR #2672 review hardening. Added grouped Keep-a-Changelog bullets for Research Workspace agent-task/discovery work, Web Scraping runtime boundary work, release/backlog hygiene, MCP discovery follow-ups, provider/UI fixes, and review hardening. Verification: inspected the rendered 0.1.37 section and ran git diff --check successfully. Bandit skipped because this is documentation and Backlog metadata only.
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
