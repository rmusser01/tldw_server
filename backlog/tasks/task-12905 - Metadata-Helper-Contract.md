---
id: TASK-12905
title: Metadata Helper Contract
status: Done
modified_files:
- apps/packages/ui/src/utils/character-mood.ts
- apps/packages/ui/src/utils/__tests__/character-mood.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 1 for character expression editor onboarding: make character mood image metadata helpers read/write canonical tldw.mood_images with arbitrary safe emote states while preserving legacy mood detection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add regression tests first, run Vitest red/green, update only metadata image-map helpers if needed, verify Vitest and Bandit, then commit requested files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 1 metadata helper contract: canonical tldw.mood_images supports arbitrary safe emote state keys, canonical maps win over legacy aliases, merge removes legacy mood_images/moodImages aliases while preserving unrelated tldw metadata, empty maps remove mood image metadata, and resolver falls back from custom emote lookup to legacy mood aliases.

Verification: initial Vitest red check failed for nested legacy alias cleanup and legacy resolver fallback; final `bunx vitest run src/utils/__tests__/character-mood.test.ts` from apps/packages/ui passed 11 tests. `git diff --check` passed. Bandit was run with the main repo venv against the touched TypeScript files and returned no findings, with expected TypeScript parse errors because Bandit is Python-only.
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
