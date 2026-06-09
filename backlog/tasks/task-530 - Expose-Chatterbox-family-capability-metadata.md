---
id: TASK-530
title: Expose Chatterbox family capability metadata
status: Done
references:
- https://github.com/devnen/Chatterbox-TTS-Server
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add adapter capability metadata that makes Chatterbox model families discoverable to API clients, including Original, Multilingual language codes, Turbo, and voice-conversion endpoint details. Use focused adapter tests and update the ongoing Chatterbox parity plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatterbox adapter capabilities now include JSON-serializable metadata for model families and voice conversion: standard/emotion model IDs, multilingual language codes, Turbo paralinguistic tags, and the `/api/v1/audio/voice-conversion` endpoint. Verification: `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k family_metadata -v` failed before implementation and passed after; `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v` passed; `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task530.json` passed with zero findings; `git diff --check` passed.
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
