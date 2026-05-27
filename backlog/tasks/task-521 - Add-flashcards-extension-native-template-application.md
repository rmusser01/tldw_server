---
id: TASK-521
title: Add flashcards extension native template application
status: Done
assignee:
- Codex
labels:
- flashcards
- extension
- ux
priority: medium
references:
- Flashcards-UX-Fix-List.md
- https://github.com/rmusser01/tldw_server/pull/2081
modified_files:
- apps/packages/ui/src/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx
- apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateValueModal.tsx
- apps/packages/ui/src/components/Flashcards/utils/flashcard-template-resolution.ts
- apps/packages/ui/src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts
- Flashcards-UX-Fix-List.md
- apps/extension/docs/features/flashcards.md
- Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
- Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow F12 follow-up: let extension sidepanel users apply existing Flashcards templates to captured or generated drafts before saving, while keeping in-extension review deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel draft cards expose a native Apply template action.
- [x] #2 Applying a template uses existing Flashcards template placeholder/materialization behavior.
- [x] #3 Template application updates only the selected draft and preserves page source provenance.
- [x] #4 Saving templated drafts preserves model type, cloze/reverse flags, tags, notes, extra fields, selected deck, and source URL provenance.
- [x] #5 Captured drafts and generated draft batches both support template application.
- [x] #6 In-extension review remains deferred and documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-flashcards-extension-template-application-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review-fix pass after rebasing on latest dev: preserved hidden generated draft notes/extra when templates omit optional fields, passed selected draft defaults into FlashcardTemplateValueModal, kept front/back fallbacks null-safe, changed the sidepanel Apply template action to icon-only with its existing aria-label, and updated the sidepanel hint copy so it no longer sends users to full Flashcards for templates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed native sidepanel template application for the F12 Flashcards extension flow and addressed PR review feedback after rebasing on latest dev. Captured and generated draft cards expose Apply template, reuse the existing template value modal/materialization behavior, update only the selected draft, preserve generated tags, hidden notes/extra, and page source provenance, and save templated model/notes/extra fields through the existing sidepanel save payload. The Apply template action is now icon-only for narrow sidepanel density, and the sidepanel hint copy now describes native template application while leaving in-extension review deferred.

Draft PR: https://github.com/rmusser01/tldw_server/pull/2081

Verification:
- bunx vitest run src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx: 40 tests passed.
- bunx vitest run src/components/Flashcards/components/__tests__/FlashcardTemplateValueModal.test.tsx src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx: 41 tests passed.
- bunx vitest run src/services/__tests__/flashcards-generate-handoff.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts: 45 tests passed.
- git diff --check: passed.
- NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false: still fails only on the known unrelated CharacterListContent.design-system.test.tsx GalleryCardDensity baseline.
- Bandit not applicable: no Python files touched.
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
