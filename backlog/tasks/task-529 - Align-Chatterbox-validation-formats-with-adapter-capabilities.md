---
id: TASK-529
title: Align Chatterbox validation formats with adapter capabilities
status: Done
references:
- https://github.com/devnen/Chatterbox-TTS-Server
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the Chatterbox TTS validation metadata so central request validation and provider limits allow the same output formats advertised by the adapter, including FLAC and PCM. Add focused tests and run targeted verification plus Bandit on touched backend paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatterbox format validation now accepts FLAC and PCM, and provider limits report wav/mp3/opus/flac/pcm consistently with the adapter. Verification: `python -m pytest tldw_Server_API/tests/TTS/test_tts_validation.py -k "chatterbox_accepts_adapter_advertised_formats or get_provider_limits" -v` failed before implementation and passed after; `python -m pytest tldw_Server_API/tests/TTS/test_tts_validation.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v` passed; `python -m bandit -r tldw_Server_API/app/core/TTS/tts_validation.py -f json -o /tmp/bandit_tts_validation_task529.json` passed with zero findings; `git diff --check` passed.
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
