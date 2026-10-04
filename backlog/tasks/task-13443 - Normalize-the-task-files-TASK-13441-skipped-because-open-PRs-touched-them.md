---
id: TASK-13443
title: Normalize the task files TASK-13441 skipped because open PRs touched them
status: To Do
labels:
- backlog
- chore
dependencies:
- TASK-13441
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13441 normalized 2,269 task files but skipped 39 non-canonical ones that open PRs were editing, to avoid merge conflicts. backend-required's format check runs only when backend files change, so a backlog-only PR can still merge one of them unnormalized (ADR-059 accepted that gap). Once those PRs merge or close, run 'backlog-py task normalize --check' over backlog/tasks and normalize whatever is still listed, again skipping files that open PRs touch. Skipped at TASK-13441 time (one file name repeats for TASK-13145): TASK-2281, TASK-12135, TASK-12993.1, TASK-13013.7, TASK-13144, TASK-13145, TASK-13146, TASK-13147, TASK-13148, TASK-13243, TASK-13244, TASK-13245, TASK-13245.1, TASK-13245.2, TASK-13245.3, TASK-13260.270, TASK-13260.277, TASK-13260.277.1, TASK-13260.277.2, TASK-13260.277.3, TASK-13260.277.4, TASK-13260.277.5, TASK-13260.277.6, TASK-13260.277.7, TASK-13260.277.8, TASK-13260.277.9, TASK-13260.277.10, TASK-13260.277.11, TASK-13260.277.12, TASK-13260.277.13, TASK-13260.277.14, TASK-13260.277.15, TASK-13260.277.16, TASK-13260.277.26, TASK-13260.277.37, TASK-13262, TASK-13385, TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 backlog-py task normalize --check over backlog/tasks lists no file, or only files that open PRs touch at the time of the PR
- [ ] #2 No task text is lost: frontmatter parses equal and content lines are preserved in every changed file
<!-- AC:END -->

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
