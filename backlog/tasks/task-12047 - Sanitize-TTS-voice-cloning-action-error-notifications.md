---
id: TASK-12047
title: Sanitize TTS voice cloning action error notifications
status: Done
created_date: 2026-06-26 07:26
labels:
- webui
- audio
- tts
- raw-error
priority: High
references:
- TASK-12046
- TASK-12039
- TASK-431
- TASK-420
documentation:
- Docs/superpowers/plans/2026-06-26-webui-stage18-tts-voice-cloning-error-sanitization-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage18-tts-voice-cloning-error-sanitization-plan.md
- apps/packages/ui/src/components/Option/TTS/VoiceCloningManager.tsx
- apps/packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx
updated_date: 2026-06-26 07:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TTS voice cloning actions currently surface raw exception text in user-facing notifications. Upload, encode, delete, and preview failures should use the shared backend-error sanitizer so endpoint paths, local paths, URLs, and token-like details are redacted before display.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Upload voice failures show a sanitized notification description without raw API endpoints, local file paths, or token-like secrets.
- [x] #2 Encode/delete/preview voice action failures use the shared server-error sanitizer instead of raw error.message text.
- [x] #3 Focused regression tests cover representative sanitized action notifications.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Add focused VoiceCloningManager tests that fail against current raw notification descriptions.', 'Wrap voice-cloning action notification descriptions with sanitizeServerErrorMessage.', 'Run targeted tests, lint touched files, whitespace checks, and record the known design-state verifier status.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started focused Stage 18 slice for TTS voice-cloning notification sanitization. Scope is limited to action failure notifications in VoiceCloningManager and focused regression coverage.
Verification: RED `bun run test:run ../packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` failed on raw `/api/v1`, local path, and token-like notification descriptions. GREEN `bun run test:run ../packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx ../packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx` passed 5 tests. Direct ESLint on `VoiceCloningManager.tsx` and `VoiceCloningManager.test.tsx` exited 0 with only the known Next pages-directory notice. `git diff --check` passed. `bun run verify:design-system-state` from `apps/packages/ui` remains blocked by the known missing `typescript` package import in the design-state verifier. Bandit is not applicable because this slice touched TSX/Markdown/Backlog files only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reused the shared server-error sanitizer for TTS voice-cloning upload, encode, delete, and preview failure notifications so user-facing descriptions redact backend endpoints, local filesystem paths, URLs, and token-like secrets. Added focused component regressions that drive upload and existing-voice actions through the real VoiceCloningManager controls and assert sanitized notification descriptions.
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
