---
id: TASK-13114
title: Implement Chat Macros v1.1 authoring and output profiles
status: In Progress
assignee: []
created_date: '2026-08-24 04:15'
updated_date: '2026-08-24 04:23'
labels:
  - chat-macros
  - frontend
  - backend
dependencies:
  - TASK-12126
documentation:
  - Docs/superpowers/specs/2026-07-03-chat-macros-design.md
  - Docs/superpowers/plans/IMPLEMENTATION_PLAN_chat_macros_v1_1_authoring.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the first Chat Macros expansion slice: a complete user-facing authoring workflow for custom macros and a richer configurable output-profile editor, building on the merged v1 execution and persistence architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can create, edit, delete, import, and export custom macro definitions through the settings UI without manually editing server files.
- [ ] #2 The authoring workflow validates definitions before persistence and presents actionable schema, command-collision, permission, and execution-cap errors.
- [ ] #3 Users can configure single-response and structured-section output profiles, including ordering, headings, branch-result inclusion, and synthesis behavior supported by the runtime.
- [ ] #4 Existing enable, disable, clone, run, cancel, retry, and built-in /wrapup behavior remains compatible.
- [ ] #5 Backend and frontend tests cover successful authoring, validation failures, ownership boundaries, output-profile round trips, and destructive-action confirmation.
- [ ] #6 Documentation describes the authoring workflow, stored definition format, compatibility constraints, and operational/security considerations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed TDD plan: Docs/superpowers/plans/IMPLEMENTATION_PLAN_chat_macros_v1_1_authoring.md. Five stages cover backend identity/heading contracts, typed YAML helpers, macro editor UI, output-profile editor UI, and manager integration with security and visual verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-23 baseline: backend Chat_Macros suite passed 134 tests with 2 warnings. ChatMacrosSettings frontend component suite passed 4 tests. The frontend service suite could not collect in this isolated worktree because wxt/browser was unresolved across the monorepo dependency roots; stop-after-three-attempts rule applied and the plan requires a complete workspace dependency layout before Task 2.

Current tracked files: Docs/superpowers/plans/IMPLEMENTATION_PLAN_chat_macros_v1_1_authoring.md; backlog task metadata. Isolated branch: codex/chat-macros-v1-1 from merge commit 5c268daa7a.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
