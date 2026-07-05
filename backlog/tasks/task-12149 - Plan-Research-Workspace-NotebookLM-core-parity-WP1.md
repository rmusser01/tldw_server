---
id: TASK-12149
title: Plan Research Workspace NotebookLM-core parity WP1
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 00:10'
labels:
  - research-workspace
  - planning
  - notebooklm
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a narrow implementation plan for the approved Research Workspace NotebookLM Pro/Ultra review spec. Scope is WP1 only: visible source/import expectations, beginner chat style/length presets with explicit save-to-note, Studio output grouping/copy, and only a conditional extension handoff if existing routing supports it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans with exact file paths, tasks, tests, and verification commands.
- [x] #2 Plan scope is limited to WP1 plus conditional existing-route extension handoff; WP2/WP4 media and agent work remain deferred.
- [x] #3 Plan follows existing Research Workspace patterns and avoids new dependencies or speculative abstractions.
- [x] #4 Plan review pass is completed before execution handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan file: Docs/superpowers/plans/2026-07-04-research-workspace-notebooklm-core-parity-wp1-plan.md
Spec source: Docs/superpowers/specs/2026-07-04-research-workspace-notebooklm-pro-ultra-review-design.md
Plan review: third/final review approved. Prior issues around Add Source copy/test mismatch, ChatPane selectedSystemPrompt misuse, and StudioPane primary-output test expectations were resolved.
Verification: git diff --check passed for plan/task files; non-ASCII scan found no matches. Bandit skipped because this task changed only planning documentation and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a narrow WP1 implementation plan for Research Workspace NotebookLM-core parity. The plan covers Add Source expectation copy, ChatPane style/length presets via per-turn message instructions, Studio output grouping, and existing extension handoff routing. Scope explicitly defers Drive sync, video/infographic generation, Ultra agent workflows, new ingestion backends, new dependencies, and new sidepanel routes.
<!-- SECTION:FINAL_SUMMARY:END -->

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
