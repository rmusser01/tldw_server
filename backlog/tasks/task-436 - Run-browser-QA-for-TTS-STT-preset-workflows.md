---
id: TASK-436
title: Run browser QA for TTS STT preset workflows
status: Done
labels:
- audio
- tts
- stt
- qa
- webui
- extension
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Stage 8 from the TTS/STT WebUI and extension workflow plan. Use browser-observed evidence where available to validate WebUI /tts, WebUI /stt, extension #/tts, and extension #/stt after the server preset CRUD slice. Record visible workflow evidence, accessibility/layout issues, blockers, and any focused fixes needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WebUI /tts browser QA covers first visit, Browser preview/no-setup path, provider readiness, preset controls, and result controls where possible.
- [x] #2 WebUI /stt browser QA covers upload/record entry points, model readiness, preset controls, comparison settings, and result/history states where possible.
- [x] #3 Extension #/tts and #/stt route parity or browser evidence confirms the shared surfaces remain usable at extension-like widths.
- [x] #4 Keyboard/accessibility and responsive-layout issues in the touched TTS/STT surfaces are recorded or fixed.
- [x] #5 Verification commands, browser access gaps, blockers, and final QA summary are recorded in the task and plan.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 8 browser QA completed. WebUI /tts and /stt were exercised in the in-app browser against a running local backend and frontend. Found and fixed four visible blockers: missing audio preset privilege catalog scopes prevented backend startup; WebUI settings saved tldwConfig without syncing the legacy tldw-api-host bootstrap key; absolute OpenAPI discovery to the configured server was blocked by the request guard; and audio preset WebUI mutations needed to use the active client config instead of storage-only background proxy state. TTS readiness showed Browser preview and tldw ready; TTS preset save succeeded with visible Preset saved state. STT readiness showed model health summary and STT preset save succeeded. Extension live browser surface was not loaded, but route parity was verified with option route identity and audio route parity tests. Actual audio synthesis/transcription result generation was not run because Stage 8 focused on preset/readiness workflow QA and no source audio was provided.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Stage 8 browser QA and fixed the workflow blockers found during the live pass. Backend startup now recognizes audio preset privilege scopes. WebUI connection settings now keep the canonical client config and legacy bootstrap host in sync, advanced-mode bootstrap ignores stale page-origin API hosts, configured-server absolute OpenAPI discovery is allowed, and audio preset requests on WebUI use the active client config for direct requests.

Browser evidence:
- `/settings/tldw`: saved `http://127.0.0.1:8000`; page showed `Server:http://127.0.0.1:8000`, `Core: reachable`, and `RAG: healthy`.
- `/tts`: showed `Browser preview: Ready` and `tldw: Ready`; TTS preset save succeeded and selected `QA TTS balanced direct` with visible `Preset saved`.
- `/stt`: showed `STT models: Ready. 37 listed, 33 on demand, 4 unavailable Source: model health.`; STT preset save succeeded and selected `QA STT default (default)` with visible `Preset saved`.
- Final `/tts` and `/stt` refresh smoke had no fresh console warnings and no backend unreachable dialog.
- Extension live browser surface was not loaded; route parity was verified by tests.

Validation gaps:
- Actual TTS audio synthesis and STT transcription result generation were not run in this QA pass because no source audio was provided and the focused Stage 8 blockers were connection/readiness/preset workflow blockers.
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
