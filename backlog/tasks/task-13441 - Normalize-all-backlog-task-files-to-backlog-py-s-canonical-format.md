---
id: TASK-13441
title: Normalize all backlog task files to backlog-py's canonical format
status: Done
dependencies:
- TASK-13440
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-13440: run 'backlog-py task normalize' over backlog/tasks in a backlog-only PR, then verify that a second 'normalize --check' run reports nothing. The ratchet is per-PR and has no baseline (ADR-059), so nothing else needs changing. Coordinate with peer sessions first and merge in a quiet window, because it touches about 2,300 task files that open PRs may also edit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Peer sessions were pinged before the PR opened, and no open PR's task-file edits conflict at merge
- [x] #2 Every task file not touched by an open PR is canonical (2,269 normalized) and a second normalize run changes nothing; the 39 skipped files are tracked in TASK-13443
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ran backlog-py task normalize over the 2,308 non-canonical files in backlog/tasks, minus 39 that open PRs touch (diff from merge base for all 67 open PRs) and TASK-13434..13437 (the peer's unpushed spec 2 PR C branch). 2,269 files changed: 4,576 insertions, 6,001 deletions, markers and headings only.
Verification:
- For every changed file, the YAML frontmatter parses equal before and after, and the multiset of content lines (excluding section markers, headings and blank lines) is identical: 0 lines dropped, 0 added.
- A second normalize --check lists only the 39 skipped files. backlog-py task list parses all 3,693 tasks.
- test_backlog_task_format_ratchet.py (diffing the whole PR against origin/dev), test_licensing_policy.py and tools/backlog-py/tests: 164 passed.
- The peer session was pinged before the PR opened. Before merge, re-check open PRs for overlap and restore any newly overlapping file to dev's version.
- Bandit: not applicable (task files only).
Review (#3162, Qodo): the skipped files are not guaranteed to be normalized by their PRs, because backend-required's format check runs only on backend changes and run-pre-commit is not required. AC2 no longer claims that; the 39 files are tracked as open work in TASK-13443.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized 2,269 backlog task files to backlog-py's canonical format (one IMPLEMENTATION_NOTES block, one BEGIN/END pair per section) with no text lost. 39 non-canonical files that open PRs were editing were skipped to avoid conflicts; TASK-13443 normalizes them once those PRs land. Skip: Bandit (task files only).
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
