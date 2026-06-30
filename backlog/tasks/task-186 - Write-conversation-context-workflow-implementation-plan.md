---
id: TASK-186
title: Write conversation context workflow implementation plan
status: Done
assignee:
  - Codex
created_date: '2026-05-09 19:51'
updated_date: '2026-05-09 19:55'
labels:
  - docs
  - ux
  - planning
  - character-chat
  - worldbooks
  - chat-dictionaries
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md
  - Docs/Reviews/CHARACTER_CARD_WORLDBOOK_DICTIONARY_UX_AUDIT_2026_05_09.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for the approved Conversation Context workflow design, including backend effective-context contracts, frontend composer popover placement, tests, validation, and follow-up boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is written under Docs/superpowers/plans with a dated conversation-context workflow filename
- [x] #2 Plan decomposes backend effective-context contract, prompt-preview parity, frontend API types, composer popover replacement or evolution of the character picker, and validation into reviewable tasks
- [x] #3 Plan explicitly keeps worldbooks and dictionaries conversation-scoped and reusable outside character chat
- [x] #4 Plan includes exact files, tests, commands, expected results, Bandit note, and browser or E2E validation
- [x] #5 Plan references the approved design spec and May 9 UX audit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review approved design, audit, and current backend/frontend context assembly files. 2. Draft implementation plan with file map, task breakdown, tests, commands, and follow-up boundaries. 3. Verify markdown/diff and commit the plan with the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: reviewed the generated plan, confirmed ASCII-only content, confirmed plan references the approved spec and May 9 UX audit, and checked the worktree status.

Bandit: not run because this task only adds documentation/planning artifacts and a Backlog task; no Python application code was touched.

Subagent plan review: skipped because current tool policy only permits spawning subagents when the user explicitly asks for delegation; local self-review was performed instead.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md. The plan decomposes the approved Conversation Context workflow into backend context-preview contract work, prompt-preview/send parity, frontend API and hook work, a composer popover replacing/evolving the existing character picker, minimal worldbook/dictionary attachment, browser validation, and documentation closeout.
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
