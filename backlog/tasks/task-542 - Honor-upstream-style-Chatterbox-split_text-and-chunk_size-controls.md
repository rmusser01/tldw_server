---
id: TASK-542
title: Honor upstream-style Chatterbox split_text and chunk_size controls
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-08 23:57'
labels:
  - tts
  - chatterbox
  - compatibility
  - chunking
dependencies: []
references:
  - 'https://github.com/devnen/Chatterbox-TTS-Server'
  - Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Map upstream Chatterbox request controls split_text and chunk_size into the existing TTS service chunking path so non-streaming long-text Chatterbox requests can use familiar upstream parameters without bypassing current chunk assembly and audio-quality checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 extra_params.split_text=true enables service-level chunking when explicit chunking_service/chunking flags are absent.
- [x] #2 extra_params.chunk_size sets the service chunk target and max character settings.
- [x] #3 extra_params.split_text=false disables the upstream-style chunking path even when chunk_size is present.
- [x] #4 The Chatterbox setup runbook and parity plan document the compatibility behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add failing service tests proving extra_params.split_text=true enables service chunking and extra_params.chunk_size sets chunk target/max chars. Implement aliases inside _resolve_chunking_params while preserving existing chunking_service/chunking precedence, update docs/plan, then verify with focused TTS service tests, relevant Chatterbox adapter tests, Bandit on tts_service_v2.py, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Mapped upstream-style split_text into _resolve_chunking_params as a service chunking enable alias, and mapped chunk_size to both chunk target and max character settings. Existing chunking_service and chunking flags retain precedence. Documentation clarifies that these aliases apply to non-streaming Chatterbox requests so the service can assemble PCM segments into one encoded response.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed upstream-style Chatterbox long-text alias support by mapping split_text and chunk_size into the existing non-streaming service chunking path. Verification: python -m pytest tldw_Server_API/tests/TTS/test_tts_service_v2.py -v -> 32 passed; python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v -> 39 passed; python -m bandit -r tldw_Server_API/app/core/TTS/tts_service_v2.py -f json -o /tmp/bandit_chatterbox_chunk_alias_task542.json -> clean; git diff --check -> clean.
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
