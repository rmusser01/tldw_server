---
id: TASK-12920
title: Implement TTS settings voice preview
status: Done
labels:
- webui
- tts
- settings
- implementation
documentation:
- Docs/superpowers/specs/2026-07-07-tts-settings-voice-preview-design.md
- Docs/superpowers/plans/2026-07-08-tts-settings-voice-preview-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx
- Docs/superpowers/plans/2026-07-08-tts-settings-voice-preview-implementation-plan.md
- backlog/tasks/task-12920 - Implement-TTS-settings-voice-preview.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved inline TTS settings voice preview for the shared WebUI and browser-extension settings page.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared TTSModeSettings exposes a preview control for the active TTS provider.
- [ ] #2 Preview uses unsaved form values and does not persist settings or validation state.
- [ ] #3 Browser preview uses speechSynthesis with the unsaved voice and playback speed.
- [ ] #4 Server-backed preview uses non-streaming /api/v1/audio/speech, aborts in-flight work on stop/unmount, and does not use websocket audio endpoints.
- [ ] #5 Focused frontend tests cover provider behavior, cleanup, missing fields, and persistence guard.
- [ ] #6 Verification and Bandit/touched-scope notes are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan saved to Docs/superpowers/plans/2026-07-08-tts-settings-voice-preview-implementation-plan.md. Execute inline task-by-task: extend TTSModeSettings tests, implement minimal preview behavior in TTSModeSettings, verify targeted tests and diff checks, then update task summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added shared TTS settings preview behavior and tests. The preview button now works for Browser, tldw, OpenAI-compatible, and ElevenLabs providers using current form values without saving settings. Cleanup aborts in-flight server preview, cancels browser speech, pauses preview audio, and revokes object URLs. Tests cover provider behavior, missing-field blocking, no websocket use when tldw streaming is enabled, and persistence guard. Verification: bunx vitest run src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx --maxWorkers=1 passed with 10 tests; git diff --check passed. TypeScript package check OOMed at default heap; rerun with NODE_OPTIONS=--max-old-space-size=8192 completed but reported existing repo-wide errors outside the touched TTS settings files. Bandit skipped because no Python files were touched.
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
