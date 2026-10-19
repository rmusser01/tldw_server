---
id: TASK-13857
title: TTS capability listing cancels slow local model loads and marks the provider
  failed
status: To Do
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TTSAdapterRegistry.get_all_capabilities (core/TTS/adapter_registry.py:1023-1034) gives each enabled provider's initialization 5 s with asyncio.wait_for and, on timeout, calls _schedule_retry, which marks the provider failed for adapter_failure_retry_seconds (300 in tts_providers_config.yaml:629); get_adapter then returns None to real synthesis requests for that window. An adapter that loads its model on a worker thread (luxtts, index_tts and pocket_tts today) is cancelled mid-load: the thread keeps loading a model nobody uses, the per-provider init lock is released so the next caller starts a second load, and later polls repeat it. The listing backs GET /api/v1/audio/health, which the WebUI polls every 60 s, and GET /api/v1/audio/providers. Found in the re-review of PR #3264, which therefore keeps NeuTTS's model load on the event loop: it blocks the server once per process, including the in-process TTS jobs worker's lease renewal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A capability-listing timeout neither cancels an in-flight adapter initialization nor marks the provider failed, and later callers join the running initialization
- [ ] #2 A registry test with an adapter whose initialization outlasts the listing timeout shows one load, and the adapter available to the next request
- [ ] #3 NeuTTS imports its vendor module and loads its model on a worker thread, with a test that neither runs on the loop thread
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
