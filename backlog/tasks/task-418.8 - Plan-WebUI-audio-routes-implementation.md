---
id: TASK-418.8
title: Plan WebUI audio routes implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- audio
priority: High
parent_task_id: TASK-418
documentation:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 11A. Scope maps F2 support, F9 support, F15 support, F18 support, and F19 support into a reviewable plan for audio route canonicalization, STT/TTS readiness, speech route framing, and audiobook studio alignment without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Created the documentation-only WP11A audio route implementation plan at `Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md`.
- [x] Covered `/audio`, `/speech`, `/stt`, `/tts`, and `/audiobook-studio`.
- [x] Mapped `F2 support`, `F9 support`, `F15 support`, `F18 support`, and `F19 support` into concrete implementation tasks.
- [x] Included route inventory, route ownership, frontend-only versus backend-gated scope, non-goals, file structure, implementation tasks, acceptance criteria, and verification commands.
- [x] Kept this task limited to Markdown planning artifacts with no product frontend or backend code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created the audio routes plan as a child slice of `TASK-418`.
- Cross-checked current route ownership before writing the plan:
  - `/audio` is a Next-page alias to `/speech`.
  - `/speech` uses `SpeechPlaygroundPage`.
  - `/stt` uses `SttPlaygroundPage` in shared WebUI routes while the extension wrapper currently maps to `SpeechPlaygroundPage initialMode="speak"`.
  - `/tts` uses `SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher` while `TtsPlaygroundPage` remains present as a non-route-owner component.
  - `/audiobook-studio` uses `AudiobookStudioPage`.
- Added explicit implementation guidance to protect route ownership, extension parity decisions, hosted and capability states, beta route treatment, and responsive verification.
- Follow-up consistency review corrected the plan's `F19 support` wording so it matches the source spec: deprecated Ant Design cleanup is a blocker/noise trigger for touched audio UX work, not a responsive-landmark finding.
- Bandit was not run because this task touched only Markdown planning and Backlog task files.
- Verification performed for the plan artifact:
  - `rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.\\.\\.|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md`
  - `rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md`
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md`
  - `node -e` required-route, finding, file, and test coverage check
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the WP11A audio route implementation plan. The plan preserves current product intent, avoids backend changes, and turns the audit findings into reviewable implementation tasks for route identity, provider readiness, STT/TTS recovery, extension parity, audiobook beta status, and browser verification.
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
