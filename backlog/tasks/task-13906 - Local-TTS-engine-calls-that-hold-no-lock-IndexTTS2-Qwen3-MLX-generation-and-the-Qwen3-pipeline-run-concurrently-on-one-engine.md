---
id: TASK-13906
title: 'Local TTS engine calls that hold no lock: IndexTTS2, Qwen3 MLX generation
  and the Qwen3 pipeline run concurrently on one engine'
status: To Do
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Thread-offloaded engine calls with no lock, on dev e06eeec68c: index_tts_adapter.py:290 (generate), :427 and :437 (stream); qwen3_runtime_mlx.py:195 (generation); qwen3_tts_adapter.py:740 (_invoke_backend) and :1020 (the stream worker, which keeps running after its consumer is cancelled). Two requests run them concurrently on one engine, and a cancelled request's call keeps running beside the next one. Recorded as out of scope in the TASK-13856 plan (Docs/superpowers/plans/2026-10-18-tts-cancellation-registry.md, Review Focus 4) and never filed; TASK-13893 changed only model loads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed call is held through hold_tts_engine on a per-engine lock, or the engine is documented safe for concurrent calls with the reason
- [ ] #2 Each locked site has a test for a plain task cancel and an anyio cancel scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
