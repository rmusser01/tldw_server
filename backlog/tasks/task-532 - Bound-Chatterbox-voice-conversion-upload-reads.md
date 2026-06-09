---
id: TASK-532
title: Bound Chatterbox voice-conversion upload reads
status: Done
references:
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the Chatterbox voice-conversion endpoint so source and target multipart uploads are read with a maximum byte limit instead of unbounded `UploadFile.read()`. Add a focused integration regression test for oversized uploads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatterbox voice-conversion uploads are now materialized in chunks with a 50 MiB per-upload/payload limit. Oversized source or target uploads return HTTP 413, conversion is not invoked, and partial temp files are cleaned up. Verification: `python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py -k oversized_source_upload -v` failed before implementation and passed after; `python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py -k "chatterbox_voice_conversion" -v` passed; `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py -f json -o /tmp/bandit_voice_conversion_task532.json` passed with zero findings; `git diff --check` passed.
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
