---
id: TASK-13212
title: Address Persona voice PR 2927 review findings
status: In Progress
created_date: 2026-09-06 17:11
references:
- https://github.com/rmusser01/tldw_server/pull/2927
- Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
modified_files:
- tldw_Server_API/app/core/Persona/parakeet_transcriber.py
- tldw_Server_API/app/core/Persona/turn_transcriber.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/tests/Persona/test_persona_parakeet_turn.py
- tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py
updated_date: 2026-09-06 17:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Parakeet decoder failures out of voice transcripts and make expected recognition failures diagnosable without exposing speech or provider details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Known ONNX error statuses reach the STT error path and are never offered as transcript text; ordinary speech remains unchanged.
- [x] #2 Expected stopped and unavailable states use typed Persona recognition failures.
- [x] #3 Recognition failure logs retain sanitized stack locations and correlation context without speech, raw audio or exception payloads.
- [x] #4 Public recognition lifecycle contracts and test helper return types are documented; focused regressions and required checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. Existing ADR046 requires failed transcription to return an error rather than synthetic speech and retains bounded work ownership.
1. Add failing regressions for known decoder errors, typed lifecycle failures and private correlated diagnostics.
2. Apply a minimal Persona-boundary error mapping and safe diagnostic context; document public contracts.
3. Verify focused tests, lint/format/security, inspect Qodo follow-up and publish review resolutions before merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Addressed all five initial Qodo findings: exact six legacy ONNX statuses rejected before vocabulary/history publication; safe PersonaVoiceRecognitionError with unavailable/stopped/failed codes; expanded public lifecycle docstrings; bounded stack-location-only logs inheriting session/client context, with socket debug payload redacted; explicit helper return type. Red evidence: nine focused cases and one socket regression failed before implementation. Green: 14 focused cases, 202 targeted Python regressions; final stricter typed adapter assertions rerun, 13 passed (overlapping subset). Ruff/Black pass changed standalone scope; Bandit zero across seven production files. Independent read-only review found no actionable issues. Existing ADR046 extended and mirrored; no new architectural boundary. PR #2927 still awaits final CI/review and human-authored Change summary before merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
