---
id: TASK-169
title: Implement character-chat terminology alignment
status: Done
assignee: []
created_date: '2026-05-09 16:54'
updated_date: '2026-05-09 16:58'
labels:
  - character-chat
  - frontend
  - ux-audit
  - copy
dependencies:
  - TASK-159
  - TASK-161
  - TASK-166
  - TASK-167
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-character-chat-terminology-alignment-plan.md
  - Docs/superpowers/specs/2026-05-09-character-chat-ux-work-packages-design.md
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the terminology alignment work package for character-chat UX. Align only the user-facing labels and helper text that affect the character-chat workflow, while preserving domain/API terms and avoiding broad rebranding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user-facing taxonomy distinguishes Character, Character chat, Scene, Persona, Assistant, and Companion Home for this workflow.
- [x] #2 Critical character-chat entry points use consistent language without using Persona as a synonym for Character.
- [x] #3 Scene/Actor controls are presented as optional context after character selection where touched.
- [x] #4 Local helper text is added only at decision points where it reduces confusion.
- [x] #5 Existing behavior remains unchanged apart from labels/helper text.
- [x] #6 Focused string/component tests and UI typecheck are run and recorded.
- [x] #7 Bandit is skipped only if touched scope remains frontend-only TypeScript/tests/docs/backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Character-chat terminology alignment completed in Docs/superpowers/plans/2026-05-09-character-chat-terminology-alignment-plan.md.

Implemented:
- Added Docs/Product/WebUI/Character_Chat_Terminology_Taxonomy_2026_05_09.md as the user-facing taxonomy source of truth.
- Updated AssistantSelect from generic assistant language to character/persona language at the trigger, search, and tablist decision points.
- Updated the optional scene entry point label from Scene Director (Actor) to Optional scene context in the mixed picker and matching English locale entries.
- Avoided broader rebranding or API/domain term changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED verification: bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --testTimeout=20000 failed on the old Select assistant, Search assistants, and Scene Director (Actor) labels.

GREEN verification: bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --testTimeout=20000 passed, 7 tests.

Full UI typecheck: ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false passed.

git diff --check passed.

String search found no old Select assistant, Search assistants, Assistant types, or Scene Director (Actor) labels in the touched picker/locale/test/doc scope.

Bandit skipped because touched scope is frontend TypeScript/tests plus docs/backlog.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the terminology alignment slice for the character-chat identity decision point. The picker now says Select character or persona, searches characters and personas, labels the tab group as Character or persona, and presents scene controls as Optional scene context instead of Scene Director (Actor). Added a taxonomy document that defines Character, Character chat, Scene, Persona, Assistant, and Companion Home so future copy changes have a stable reference. Verification passed for focused component tests, full UI typecheck, string-search checks, and git diff hygiene. Bandit was skipped because the touched implementation scope is frontend TypeScript/tests plus docs/backlog.
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
