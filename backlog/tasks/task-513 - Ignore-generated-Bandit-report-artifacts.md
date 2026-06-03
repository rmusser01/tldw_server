---
id: TASK-513
title: Ignore generated Bandit report artifacts
status: In Progress
labels:
- docs
- process
- security
- review-followup
modified_files:
- .gitignore
- Docs/ADR/006-bandit-report-path-portability.md
- backlog/tasks/task-513 - Ignore-generated-Bandit-report-artifacts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the post-merge PR #2233 review feedback by explicitly ignoring generated bandit_*.json reports now that AGENTS.md recommends a repository-relative Bandit report output path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 .gitignore explicitly ignores bandit_*.json report artifacts.
- [ ] #2 ADR-006 documents that generated bandit_*.json reports are ignored and should not be committed unless explicitly requested.
- [ ] #3 git check-ignore confirms a sample bandit_<task>.json filename is ignored.
- [ ] #4 Verification commands pass, including git diff --check.
- [ ] #5 A follow-up PR is opened and linked from this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add an explicit bandit_*.json ignore rule near generated report artifacts, update ADR-006 to state generated Bandit reports are ignored, verify with git check-ignore and git diff --check, then open a follow-up PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
