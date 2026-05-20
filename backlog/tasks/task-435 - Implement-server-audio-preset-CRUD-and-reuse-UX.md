---
id: TASK-435
title: Implement server audio preset CRUD and reuse UX
status: Done
labels:
- audio
- tts
- stt
- presets
- webui
- extension
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Stage 7 from the TTS/STT WebUI and extension workflow plan. Implement per-user server-side TTS/STT presets using the Stage 6 ownership decision, with backend CRUD/validation, AuthNZ isolation, shared frontend API/hook support, WebUI and extension preset controls, deletion semantics, verification, and plan updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend exposes authenticated per-user audio preset CRUD and validate endpoints under /api/v1/audio/presets
- [x] #2 Preset storage uses the per-user Media DB boundary from Docs/Design/Audio_Presets_Ownership_2026_05.md and does not reuse TTS history or transcript rows
- [x] #3 Backend tests cover CRUD, AuthNZ/owner isolation, validation behavior, default/favorite behavior, and deletion semantics
- [x] #4 Shared UI API client and hook expose list/create/update/delete/validate behavior for WebUI and extension surfaces
- [x] #5 TTS and STT WebUI/extension pages can save, apply, duplicate/rename, favorite/default, and delete presets without auto-running generation/transcription
- [x] #6 Verification, Bandit for touched backend code, and known TypeScript/test skips are recorded in the task and plan
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `audio_presets` as per-user Media DB v2 storage with SQLite and PostgreSQL bootstrap, runtime CRUD helpers, and Audio API route mounting.
- Added backend create/list/update/delete/validate endpoints under `/api/v1/audio/presets`, preserving user scoping and soft-delete semantics.
- Browser TTS presets are accepted as a no-setup escape hatch but normalized with `browser_local` and `requires_browser_revalidation`, and validate with a warning.
- Added shared UI preset types, client methods, `useAudioPresets`, and `AudioPresetControls`; TTS/STT shared pages use the controls so WebUI and extension routes inherit the same behavior.
- Extension-specific verification uses the existing route parity guard; preset apply behavior is covered in shared page tests because extension `#/stt` and `#/tts` mount those shared pages.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 7 server audio presets. Users now have authenticated per-user TTS/STT preset CRUD and validation, backed by Media DB v2 rather than speech history rows. The shared WebUI/extension surfaces can save, apply, duplicate, rename, favorite/default, and delete presets without auto-running generation or transcription. Verification passed for focused backend tests, focused frontend preset/page tests, extension route parity, `git diff --check`, and Bandit on the touched backend preset scope. Full package TypeScript remains blocked by existing unrelated baseline errors outside the touched preset implementation files.
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
