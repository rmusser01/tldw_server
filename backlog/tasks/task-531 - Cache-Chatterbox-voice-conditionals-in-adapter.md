---
id: TASK-531
title: Cache Chatterbox voice conditionals in adapter
status: Done
references:
- https://github.com/devnen/Chatterbox-TTS-Server
- https://yocxy2-chatterboxyocxy.mintlify.app/api/chatterbox-tts
- https://yocxy2-chatterboxyocxy.mintlify.app/api/conditionals
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an in-process Chatterbox voice conditioning cache so repeated requests using the same reference audio, model family, and exaggeration reuse prepared conditionals instead of re-encoding the reference. Keep behavior fallback-safe when the upstream model lacks prepare_conditionals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatterbox speech generation now has an in-process voice conditionals cache keyed by model family, reference-audio hash, and exaggeration. When the upstream runtime exposes `prepare_conditionals()`, repeated requests reuse cached conditionals and omit `audio_prompt_path`; when preparation is unavailable or fails, the adapter falls back to the previous `audio_prompt_path` behavior. The cache is cleared on adapter close/cleanup. Verification: `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k cached_voice_conditionals -v` failed before implementation and passed after; `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v` passed; `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task531.json` passed with zero findings; `git diff --check` passed.
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
