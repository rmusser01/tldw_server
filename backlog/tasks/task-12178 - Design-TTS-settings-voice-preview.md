---
id: TASK-12178
title: Design TTS settings voice preview
status: In Progress
labels:
- webui
- tts
- settings
- design
documentation:
- Docs/superpowers/specs/2026-07-07-tts-settings-voice-preview-design.md
modified_files:
- Docs/superpowers/specs/2026-07-07-tts-settings-voice-preview-design.md
- backlog/tasks/task-12178 - Design-TTS-settings-voice-preview.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design inline TTS voice/backend preview for the shared WebUI and browser-extension speech settings page before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approved design is captured in Docs/superpowers/specs.
- [ ] #2 Spec covers unsaved form values, cleanup, provider-specific behavior, error handling, and tests.
- [ ] #3 Backlog task references the written spec.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only task. Approved design is captured in Docs/superpowers/specs/2026-07-07-tts-settings-voice-preview-design.md. Next step after user review is writing an implementation plan via the superpowers writing-plans workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written for inline TTS settings voice/backend preview. Scope keeps implementation in shared TTSModeSettings, reuses existing synthesis paths, previews unsaved form values, avoids new backend endpoints, and documents provider-specific caveats for Browser, tldw, ElevenLabs, and OpenAI-compatible preview.
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
