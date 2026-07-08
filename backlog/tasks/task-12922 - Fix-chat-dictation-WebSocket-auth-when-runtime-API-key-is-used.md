---
id: TASK-12922
title: Fix chat dictation WebSocket auth when runtime API key is used
status: Done
labels:
- frontend
- audio
- bug
modified_files:
- apps/packages/ui/src/hooks/useServerDictation.tsx
- apps/packages/ui/src/hooks/__tests__/useServerDictation.source.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chat page dictation mic can open /api/v1/audio/stream/transcribe without an auth token when the WebUI is authenticated via runtime/env single-user API key but tldwConfig does not persist the key. Backend rejects with: Authentication required. Send {"type":"auth","token":"YOUR_API_KEY"}. Root cause appears to be useServerDictation reading only tldwClient config credentials while HTTP request auth uses the runtime single-user override.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Focused root-cause fix only: reuse the existing runtime-auth override helper in useServerDictation instead of persisting secrets or adding another auth path. Verification: watched new test fail before the fix; after fix, bunx vitest run ../packages/ui/src/hooks/__tests__/useServerDictation.source.test.tsx passed; bunx vitest run ../packages/ui/src/hooks/__tests__/useDictationStrategy.test.tsx ../packages/ui/src/hooks/__tests__/useServerDictation.source.test.tsx passed; git diff --check passed. Bandit skipped because only TypeScript frontend files and Backlog task metadata were touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Root cause: chat dictation WebSocket auth used only tldwConfig credentials, while the WebUI runtime/env single-user key is intentionally stored only in the runtime override and not persisted in tldwConfig. HTTP requests already used that override; dictation did not, so the server rejected the socket before config/audio frames. Fixed useServerDictation to prefer getRuntimeSingleUserApiKeyOverride() for single-user WebSocket auth, added a regression test, and verified the running frontend recompiled.
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
