---
id: TASK-2358
title: Add Audio Studio large-artifact WebUI transport
status: In Progress
documentation:
- Docs/superpowers/plans/2026-06-24-audio-studio-artifact-playback-implementation-plan.md
- Docs/superpowers/specs/2026-06-24-audio-studio-large-artifact-media-tickets-design.md
- Docs/superpowers/plans/2026-06-24-audio-studio-media-tickets-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement authenticated large-artifact playback/download transport for Audio Studio WebUI without query-string secrets. Compare short-lived signed URLs, service-worker/header-injection, and streamed authenticated frontend fetch after the MVP artifact playback endpoint is stable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Future acceptance criteria:
- Choose and document one authenticated large-artifact WebUI transport strategy that does not place secrets in URLs.
- Support playback and download for large Audio Studio artifacts without loading the full artifact into memory when avoidable.
- Preserve strict artifact allowlisting, project/user authorization, and no raw filesystem path exposure.
- Add focused backend/frontend tests for the chosen transport and rejected unauthorized access.

Candidate approaches to compare: short-lived signed URLs, service-worker/header-injection route, and streamed authenticated frontend fetch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented DB-backed scoped media tickets in the AuthNZ/global database.
- Added playback tickets for native audio `Range` streaming and single-use download tickets for large and non-audio artifacts.
- Kept small known-size audio Blob preview/download behavior unchanged.
- Added access-log redaction for media ticket token paths.
- Updated the WebUI to use playback tickets for oversized or unknown-size audio, click-only download tickets for ticket-backed audio and non-audio artifacts, and stale async guards for preview/download state.
- Documented proxy log-redaction responsibility in `Docs/Audio_Studio.md`.
- Backend verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_media_ticket_store.py tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_media_tickets_api.py tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_artifact_media_api.py tldw_Server_API/tests/Logging/test_access_log_redaction.py -v` -> 90 passed, 12 warnings.
- Frontend verification: `cd apps/packages/ui && bunx vitest run src/services/__tests__/audio-studio.test.ts src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx --maxWorkers=1` -> 47 passed.
- Bandit verification: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Audio_Studio/media_tickets.py tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py tldw_Server_API/app/core/Logging/access_log_middleware.py -f json -o /tmp/bandit_audio_studio_media_tickets.json` -> passed, JSON report written.
- Whitespace verification: `git diff --check` -> passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audio Studio large-artifact transport now uses short-lived scoped media tickets for native playback and downloads while preserving strict artifact ownership, safe-root validation, and token redaction.
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
