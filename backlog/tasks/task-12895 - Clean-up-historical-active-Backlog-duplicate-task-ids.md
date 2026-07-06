---
id: TASK-12895
title: Clean up historical active Backlog duplicate task ids
status: Done
labels:
- backlog
- cleanup
modified_files:
- backlog/tasks/
- Docs/superpowers/plans/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the broad active Backlog task-id collisions left after the MCP setup stream cleanup. Scope this to active Backlog task files under backlog/tasks, preserving one canonical record per existing id and renumbering the other active records so id-based Backlog MCP/CLI lookup is deterministic again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active Backlog task ids under backlog/tasks are unique after the cleanup.
- [x] #2 Renumbered task filenames and frontmatter ids stay aligned.
- [x] #3 The cleanup is metadata-only: Backlog task files and direct Backlog/Docs references only, with no runtime source changes.
- [x] #4 A focused duplicate-id check and git diff --check are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial measurement found 2,256 active task files with ids, 1,531 unique ids, 432 duplicate-id groups, and 725 duplicate records to renumber. A single-pass reference scan found 349 duplicated ids with 2,378 non-id reference occurrences across Backlog/Docs; those references were already ambiguous, so this cleanup deliberately did not attempt semantic reference rewriting. Applied deterministic renumbering: for each duplicate group, kept one canonical record, preferring non-Done records over Done records and then path order; renumbered the remaining 725 records to TASK-12169 through TASK-12893. Verification after the rewrite found 2,256 active task files with ids, 2,256 unique ids, 0 duplicate-id groups, and 0 filename/frontmatter id mismatches. `git diff --check` passed. PR: https://github.com/rmusser01/tldw_server/pull/2661. Review follow-up updated two concrete plan references that Qodo flagged as now resolving to unrelated task ids. After rebasing on latest dev, renumbered three additional completed records to TASK-12894 through TASK-12896 to preserve the unique-id invariant and updated the direct WP3 plan reference to TASK-12894. Bandit not applicable because this is Backlog/Docs metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the broad active Backlog duplicate-id set under backlog/tasks by renumbering 725 duplicate records to TASK-12169 through TASK-12893, then renumbering three post-rebase duplicate records to TASK-12894 through TASK-12896. Active task ids are now unique and task filenames match frontmatter ids. This was intentionally metadata-only and did not attempt semantic repair of already-ambiguous historical TASK-* references across Backlog/Docs; that remains a separate, context-heavy cleanup if needed. PR: https://github.com/rmusser01/tldw_server/pull/2661.
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
