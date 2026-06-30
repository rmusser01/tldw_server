---
id: TASK-596
title: Clear voice assistant SDK TypeScript baseline
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:01'
labels:
  - typescript
  - voice-assistant-sdk
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clear the voice-assistant-sdk typecheck diagnostics after the UI package baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 voice-assistant-sdk `bun run typecheck` exits cleanly.
- [x] #2 Stale TypeScript suppression is removed or replaced with typed code.
- [x] #3 React hook type diagnostics are resolved without weakening exported hook types.
- [x] #4 Verification is recorded in the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added @types/react as a voice-assistant-sdk devDependency so the package can typecheck its exported React hook in isolation. Removed the stale createScriptProcessor @ts-expect-error because the DOM lib now types the deprecated fallback API. Bandit is not applicable for this JS/TS-only touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleared the voice-assistant-sdk TypeScript baseline. Verification: bun run typecheck in apps/packages/voice-assistant-sdk exits 0; apps/extension bun run compile was also checked and exits 0.
<!-- SECTION:FINAL_SUMMARY:END -->

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
