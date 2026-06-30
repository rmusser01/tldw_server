---
id: TASK-187
title: Revise conversation context plan for client-managed composition
status: Done
assignee:
  - Codex
created_date: '2026-05-09 19:59'
updated_date: '2026-05-09 20:02'
labels:
  - docs
  - ux
  - planning
  - architecture
  - character-chat
  - worldbooks
  - chat-dictionaries
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-09-conversation-context-workflow-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-conversation-context-workflow-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the approved Conversation Context design and implementation plan to reflect the intended architecture: the client owns rebuilding/effective context management, while the server provides composable primitives for settings, worldbook processing, dictionary processing, and prompt pieces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec no longer describes a monolithic backend effective-context contract as the source of truth
- [x] #2 Implementation plan assigns effective context composition and preview assembly to the client
- [x] #3 Plan keeps server responsibility limited to composable primitives and domain-specific processing/validation
- [x] #4 Composer popover direction remains replacing or evolving the existing character picker
- [x] #5 Verification and Backlog notes are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Search the spec and plan for backend-owned effective-context wording. 2. Revise the architecture, data flow, contracts, and task breakdown to client-managed composition with server primitives. 3. Verify diff consistency and commit the docs plus Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: searched the revised spec and plan for backend-owned context-preview/source-of-truth wording, confirmed the remaining context-preview references are explicit negative guardrails, confirmed ASCII-only content, and ran git diff --check successfully.

Bandit: not run because this task only revises design/planning markdown plus a Backlog task; no Python application code was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Revised the Conversation Context design spec and implementation plan to use the intended client-managed composition architecture. The client now owns effective context selection, preview assembly, ordering, and send-payload assembly; the server provides composable primitives for settings, prompt pieces, dictionary processing, worldbook matching, provider readiness, and validation. The composer popover direction remains replacing/evolving the existing character picker.
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
