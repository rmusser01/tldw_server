---
id: TASK-12137
title: Remediate audio WebSocket auth contract drift audit finding
status: Done
created_date: 2026-07-04 00:06
labels:
- audit
- remediation
- audio
- websocket
- auth
- security
- wave-2
priority: high
references:
- AUDIT-2026-06-27-AUDIO-WS-001
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/audio-streaming.md
modified_files:
- Docs/superpowers/plans/2026-07-02-audio-websocket-auth-frame-remediation.md
- backlog/tasks/task-12137 - Remediate-audio-WebSocket-auth-contract-drift-audit-finding.md
- apps/packages/ui/src/services/tldw/audio-websocket-auth.ts
- apps/packages/ui/src/services/__tests__/audio-websocket-auth.test.ts
- apps/packages/ui/src/services/tldw/voice-conversation.ts
- apps/packages/ui/src/hooks/useVoiceChatStream.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/entries/background.ts
- tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py
updated_date: 2026-07-04 00:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for the audio WebSocket contract drift finding: browser/UI audio WebSocket clients should not pass bearer tokens in the URL query string and must authenticate through the server-supported auth frame contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is present before production code changes on this current-base branch.
- [x] #2 Shared audio WebSocket client URL construction omits query-string tokens and sends auth through the first WebSocket frame.
- [x] #3 Voice chat, speech playground, and background audio clients use the shared token-free auth helper.
- [x] #4 Focused frontend and backend tests cover auth-frame WebSocket behavior and token-free URL construction.
- [x] #5 Touched-scope Bandit and frontend/backend focused verification are recorded.
- [x] #6 Residual compatibility and broader type-check tradeoffs are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-03 current-base port: origin/dev already contained a partial TASK-12106 auth-frame remediation for voice chat and extension STT, but speech TTS still embedded ?token= and the shared helper from the worker branch was absent. Ported the worker intent onto origin/dev f2d9be9864 by adding a shared audio WebSocket helper, switching speech TTS, background STT, voice chat URL construction, and voice chat auth-frame sending to the helper, and expanding backend auth-contract coverage for transcribe, chat stream, and TTS routes.

Verification recorded: npm test -- src/services/__tests__/audio-websocket-auth.test.ts src/services/__tests__/voice-conversation.test.ts src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx -> 3 test files passed, 34 tests passed. The fresh worktree initially lacked apps/node_modules/.bun, so a temporary untracked apps/node_modules symlink to the main checkout dependency directory was used for Vitest and removed before staging. Backend focused suite: PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py -k audio_ws -q --tb=short --disable-warnings -> 7 passed, 5 deselected, 21 warnings. Bandit touched test file: PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py -f json -o /tmp/bandit_audio_ws_auth_12137.json -> only LOW B101 pytest assert findings in this test file; no hardcoded-token findings remain and no production Python was touched. git diff --check -> clean.

Residual tradeoffs: existing backend compatibility toggle AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH remains for explicit opt-in tests/legacy behavior. Broad UI type-check was not run in this current-base port; focused Vitest and backend tests covered the modified contracts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Remediated the current remaining audio WebSocket auth drift by centralizing frontend audio WebSocket URL/auth-frame behavior, removing the speech TTS query-token URL, and aligning speech TTS, background STT, and voice chat with the backend-supported initial auth frame. Added helper-level Vitest coverage and expanded backend route coverage for default query-token rejection and auth-frame acceptance across TTS, STT, and voice chat routes. Focused frontend/backend verification passed; Bandit on the touched Python test file has only expected pytest assert findings.
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
