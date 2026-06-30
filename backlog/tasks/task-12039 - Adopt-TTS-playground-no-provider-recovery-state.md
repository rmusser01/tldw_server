---
id: TASK-12039
title: Adopt TTS playground no-provider recovery state
status: Done
created_date: 2026-06-26 02:53
labels:
- webui
- tts
- ux
- accessibility
priority: medium
references:
- TASK-420
- TASK-418.8
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage10-tts-no-provider-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx
updated_date: 2026-06-26 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the TTS playground no-provider state. Replace the duplicate local alert-only no server TTS provider messages with one shared user-language setup state while preserving the existing TTS page heading, provider panel, voice defaults, and browser/provider workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TTS playground renders a single shared StatePanel/RecoveryCallout setup state when the tldw provider is selected and the connected server has no TTS audio provider available.
- [x] #2 The no-provider state keeps user-language title/message and the existing recovery guidance to open Speech Settings or switch to Browser TTS.
- [x] #3 The duplicate no-provider alert is removed and the no-provider state no longer depends on local AntD Alert styling.
- [x] #4 Existing TTS defaults, provider panel rendering, and normal has-audio behavior remain covered by focused tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Stage 10 plan document for TTS playground no-provider capability recovery.
2. Add failing focused tests that expect a single shared recovery/state primitive for no-provider and preserve normal/default behavior.
3. Replace duplicate local no-provider alerts with one shared StatePanel/RecoveryCallout in TtsPlaygroundPage.
4. Run focused TTS tests, touched-file ESLint, whitespace checks, and record Bandit applicability.
5. Update Backlog and commit the Stage 10 slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage 10 TTS no-provider capability recovery. The TTS playground now renders a single shared StatePanel when the selected tldw server provider has no audio/TTS provider available, preserving the existing title/body guidance to open Speech Settings or switch to Browser TTS. Removed the duplicate local no-provider AntD Alert while leaving ffmpeg and ElevenLabs notices unchanged. The focused test mock now controls provider capability state and mocks server capabilities to avoid unrelated setup probes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TTS playground no-provider state now uses a single shared StatePanel instead of duplicate local AntD alerts. Focused tests cover the normal Kitten defaults, route heading, and no-provider shared setup state. Verification: focused TTS Vitest passed; touched-file ESLint passed with only the known repo-level Next pages-directory notice; git diff --check passed. Bandit not applicable because changes are TS/TSX/docs/task metadata only.
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
