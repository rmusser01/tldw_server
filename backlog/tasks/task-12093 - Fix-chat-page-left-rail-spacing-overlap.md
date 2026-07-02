---
id: TASK-12093
title: Fix chat page left rail spacing overlap
status: Done
labels:
- bug
- webui
- extension
- layout
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the WebUI/extension chat page layout so the context rail and chat rail on the left side do not touch or overlap. Preserve the existing dense product UI structure while making the rail spacing predictable across desktop and narrow viewports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Context rail and chat rail have a visible, stable gap and do not overlap on the chat page
- [x] #2 Fix applies to the shared WebUI/extension chat surface without breaking rail collapse behavior
- [x] #3 Regression coverage or browser-level layout verification records the non-overlap behavior
- [x] #4 Rendered QA includes desktop and narrow viewport checks for the chat page
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the rail collision in the running WebUI and capture layout measurements.
2. Trace the rail components and CSS that control left-side positioning/spacing.
3. Add the smallest regression check possible for rail non-overlap.
4. Apply the layout fix and verify in browser at desktop and narrow viewport sizes.
5. Record verification results and update the PR branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause:
- The collapsed chat sidebar trigger and cockpit context restore tab were both positioned on the same left edge. The chat trigger used `left-0` with a 40px width, while the context restore wrapper also used `left-0`. Their vertical ranges overlapped on desktop, so the two controls touched/overlapped visually.

Implemented fix:
- Updated the shared cockpit context restore wrapper class from `left-0` to `left-12`, reserving the outer 40px edge for the chat rail and leaving an 8px gap before the context restore tab.
- Updated the rail positioning contract tests and cockpit restore component test to assert the offset.

Verification:
- Red test confirmed current code still used `left-0` for the context restore wrapper.
- `bunx vitest run ../packages/ui/src/components/Layouts/__tests__/chat-rail-positioning-contract.guard.test.ts ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx`: 5 passed.
- Browser desktop measurement at 1280x720: chat rail `left=0 right=40`, context restore `left=48 right=84`, horizontal gap `8`, overlap `false`.
- Browser interaction check: clicking the context restore tab reopened the left context rail while the chat rail remained visible.
- Browser narrow viewport at 390x720: chat edge tab not rendered, context restore mounted but hidden by `lg:inline-flex`, no mobile overlap.
- `git diff --check`: clean.
- Bandit not applicable; touched files are TypeScript/TSX only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the chat page left-side rail collision by offsetting the collapsed cockpit context restore tab away from the collapsed chat sidebar trigger. The two left controls now have an 8px desktop gap, retain their restore/collapse behavior, and remain hidden/non-overlapping on narrow viewports.
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
