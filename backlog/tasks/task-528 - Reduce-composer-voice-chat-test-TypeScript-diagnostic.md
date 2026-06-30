---
id: TASK-528
title: Reduce composer voice chat test TypeScript diagnostic
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:04'
labels: []
dependencies: []
references:
  - TASK-527
  - >-
    apps/packages/ui/src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx
  - apps/packages/ui/src/components/Chat/composer/hooks/useComposerVoiceChat.ts
  - apps/packages/ui/src/hooks/useDictationStrategy.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostic in `src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx`. Current package `tsc` output reports assigning `"start_browser"` to a mock strategy result field inferred as the narrower `"start_server"` literal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current composer voice chat compiler diagnostic is captured.
- [x] #2 Root cause is documented and tied to test mock literal narrowing rather than behavior changes.
- [x] #3 The `useComposerVoiceChat.test.tsx` diagnostic is removed from package `tsc` output.
- [x] #4 Focused composer voice chat test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task527-tsc-final.txt`: package `tsc` reported one diagnostic in `src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx` where the test assigned `"start_browser"` to a mock field inferred as literal `"start_server"`.
- Root cause was test mock literal narrowing only. The production hook uses the `DictationToggleIntent` union, and the test intentionally exercises both `start_server` and `start_browser` branches.
- Imported the `DictationToggleIntent` type and widened the initial mocked `toggleIntent` value to that union.
- Focused verification: `bunx vitest run src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx` passed: 9 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task528-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 41 in `/tmp/task527-tsc-final.txt` to 40 in `/tmp/task528-tsc-final.txt`; searching for `useComposerVoiceChat.test.tsx` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `useComposerVoiceChat.test.tsx` TypeScript diagnostic by widening the mocked strategy `toggleIntent` field to the real `DictationToggleIntent` union. Focused Vitest passed with 9 tests, and package `tsc` baseline dropped from 41 to 40 with no remaining composer voice chat diagnostic.
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
