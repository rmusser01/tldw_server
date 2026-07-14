---
id: TASK-12966
title: Reconcile Workspace decision and UAT runner trackers
status: In Progress
assignee: []
created_date: '2026-07-13'
updated_date: '2026-07-14 05:21'
labels:
  - workspace
  - research-workspace
  - uat
  - tracker-cleanup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1522'
  - 'https://github.com/rmusser01/tldw_server/issues/1526'
  - 'https://github.com/rmusser01/tldw_server/issues/2605'
  - 'https://github.com/rmusser01/tldw_server/issues/2606'
  - 'https://github.com/rmusser01/tldw_server/issues/2607'
  - 'https://github.com/rmusser01/tldw_server/issues/2608'
  - 'https://github.com/rmusser01/tldw_server/pull/2609'
  - 'https://github.com/rmusser01/tldw_server/pull/2633'
documentation:
  - Docs/Design/Canonical_Workspace_Server_Record_Decision_2026_07.md
  - Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the tracker drift left after the canonical Workspace decision and Research Workspace final-runner work merged. Make the #2605 Backlog history unambiguous, close stale completed task records, close GitHub issues #1526 and #2605 with evidence, update parent #1522, and preserve #2606-#2608 as the remaining certification work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The #2605 Backlog history has one unambiguous canonical runner record and no task-ID collisions in the touched tracker set.
- [x] #2 TASK-12020.35 and the PR #2633 review/rebase record accurately reflect their completed state.
- [ ] #3 GitHub issues #1526 and #2605 are closed with evidence-linked comments, and parent #1522 marks #1526 complete.
- [ ] #4 GitHub issues #2606, #2607, and #2608 remain open and unchanged as the remaining certification work.
- [ ] #5 Backlog parsing, task-ID uniqueness, issue-state verification, and git diff hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit the duplicate and stale Workspace/UAT task records against merged PRs and current evidence. 2. Reconcile the Backlog records without losing canonical history. 3. Verify repository state, commit, and push a focused tracker-only PR. 4. Post evidence-linked GitHub closeout comments, close #1526/#2605, update #1522, and verify #2606-#2608 remain open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit confirmed #2605 runner work is complete in canonical TASK-12877, TASK-12020.28, and TASK-12020.36. Removed the obsolete duplicate #2605 TASK-12130 file while preserving the unrelated Chat Workspace TASK-12130. Marked stale TASK-12020.35 Done. PR #2633 merged at 8601d41f807be65cfb7f8a3878c2606dbb1cb1ca with 20/20 review threads resolved; its colliding Workspace TASK-12949 history is preserved under unique TASK-12967. Two unrelated pre-existing TASK-12949 records remain outside this scoped Workspace reconciliation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pending repository PR and GitHub issue closeout.
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
