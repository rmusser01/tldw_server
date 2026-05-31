---
id: TASK-434
title: Decide server ownership for TTS/STT presets
status: Done
labels:
- audio
- tts
- stt
- design
- presets
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Stage 6 from the TTS/STT WebUI and extension workflow plan by writing the audio preset ownership decision document before any CRUD implementation. The decision must use the user-selected per-user server state direction and define backend owner, DB boundary, AuthNZ principal behavior, schema, Browser TTS persistence rules, import/export stance, migration stance, deletion semantics, frontend responsibilities, and extension parity responsibilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision document exists at Docs/Design/Audio_Presets_Ownership_2026_05.md
- [x] #2 Document defines owner module and endpoint namespace for per-user audio presets
- [x] #3 Document defines DB/schema/AuthNZ principal behavior for single-user and multi-user modes
- [x] #4 Document explicitly separates presets from TTS history, STT transcript rows, generated artifacts, and comparison history
- [x] #5 Document defines Browser TTS persistence/revalidation rules and WebUI/extension sharing behavior
- [x] #6 Plan/task records include verification and next-stage gate status
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/Design/Audio_Presets_Ownership_2026_05.md`.
- Decision: reusable TTS/STT presets are per-user server state owned by the Audio API, persisted in a new Media DB v2 `audio_presets` table, and exposed at `/api/v1/audio/presets`.
- Decision: preset CRUD should live under the existing audio endpoint subpackage at `tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py`.
- Decision: presets are configuration objects only; they are not TTS history, STT transcript rows, generated artifacts, provider credentials, or comparison history.
- Decision: Browser TTS remains a no-setup escape hatch and must be local-only or marked non-portable with browser revalidation if persisted.
- Updated the implementation plan with Stage 6 completion notes and the Stage 7 backend owner path.
- Verification is documentation-focused: required-topic scan passed and `git diff --check` passed; Bandit is skipped because this stage changes docs/task records only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6 is complete. The ownership decision now gates Stage 7 preset CRUD with a concrete server-owned, per-user Media DB model, explicit AuthNZ behavior, Browser TTS limitations, deletion semantics, import/export stance, and WebUI/extension parity requirements.
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
