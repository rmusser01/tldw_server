---
id: TASK-513
title: Reduce WritingActionBar ref TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:08'
labels: []
dependencies: []
references:
  - TASK-512
  - >-
    apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/WritingPlayground/WritingActionBar.tsx`. Current package `tsc` output reports two ref type errors where structural refs are passed to Ant Design `Input.TextArea` and `Input` components instead of their exported ref types.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current WritingActionBar compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to Ant Design ref typing rather than behavior changes.
- [x] #3 The `WritingActionBar.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused WritingActionBar test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task512-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Option/WritingPlayground/WritingActionBar.tsx` where structural ref objects were not assignable to Ant Design `TextAreaRef` and `InputRef` legacy refs.
- Root cause was Ant Design ref typing only. The component was already reading the same DOM nodes from `resizableTextArea.textArea` and `input`; the ref declarations were just structural approximations rather than the component ref types.
- Imported `InputRef` from `antd` and `TextAreaRef` from `antd/es/input/TextArea`, then typed `customInputRef` and `toneInputRef` as those refs while preserving existing value fallback behavior.
- Focused verification attempted with `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx`; it failed before assertions with `SecurityError: localStorage is not available for opaque origins`, and all 8 tests were skipped. This matches the pre-existing opaque-origin JSDOM setup issue seen in nearby WritingPlayground focused tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task513-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 73 in `/tmp/task512-tsc-final.txt` to 71 in `/tmp/task513-tsc-final.txt`; `rg -n 'WritingActionBar\.tsx' /tmp/task513-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `WritingActionBar.tsx` TypeScript cluster by replacing structural Ant Design input ref shapes with `InputRef` and `TextAreaRef`. Package `tsc` baseline dropped from 73 to 71 with no remaining WritingActionBar diagnostics. The focused WritingActionBar suite remains blocked before assertions by the opaque-origin `localStorage` JSDOM setup issue.
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
