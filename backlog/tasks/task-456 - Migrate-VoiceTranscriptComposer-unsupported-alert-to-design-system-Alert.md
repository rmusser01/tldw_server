---
id: TASK-456
title: Migrate VoiceTranscriptComposer unsupported alert to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-21 00:34
labels:
- design-system
- product-state
- ui
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/1897
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Flashcards VoiceTranscriptComposer unsupported-browser voice notice from AntD Alert to the canonical design-system Alert while preserving the visible message and keeping the product-state baseline clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VoiceTranscriptComposer renders the unsupported-browser notice through the canonical design-system Alert primitive.
- [x] #2 The visible unsupported voice transcript message remains unchanged.
- [x] #3 The VoiceTranscriptComposer Alert baseline exception is removed without introducing new blocked product-state findings.
- [x] #4 Focused tests and design-system product-state verification pass, with known skips documented if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented test-first: added VoiceTranscriptComposer product-state coverage for the unsupported browser notice and confirmed RED failure because the visible message was not inside a canonical data-ds-component="Alert" wrapper. Migrated the unsupported notice from AntD Alert to the design-system Alert while leaving the surrounding AntD form controls unchanged. Removed the matching VoiceTranscriptComposer Alert baseline exception.

Verification:
- RED: bunx vitest run src/components/Flashcards/components/__tests__/VoiceTranscriptComposer.product-state.test.tsx --reporter=dot failed on missing data-ds-component="Alert" wrapper.
- GREEN: bunx vitest run src/components/Flashcards/components/__tests__/VoiceTranscriptComposer.product-state.test.tsx --reporter=dot passed: 1 test.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 325.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited repo-wide TypeScript debt; /tmp/tsc_design_system_next_slice_18.log has 252 lines and touched-file filter for VoiceTranscriptComposer/baseline/task-456 matched 0 lines.
- Bandit skipped because this slice changes TypeScript UI/test, JSON baseline, and task metadata only; no Python code touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated VoiceTranscriptComposer's unsupported-browser voice transcript notice from AntD Alert to the canonical design-system Alert, added focused product-state coverage, removed the VoiceTranscriptComposer baseline exception, and addressed PR review feedback by mapping the unavailable state to the design-system Alert error variant. Product-state baseline exceptions remain 325.
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
