---
id: TASK-13857
title: TTS capability listing cancels slow local model loads and marks the provider
  failed
status: Done
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
- [x] #1 A capability-listing timeout neither cancels an in-flight adapter initialization nor marks the provider failed, and later callers join the running initialization
- [x] #2 A registry test with an adapter whose initialization outlasts the listing timeout shows one load, and the adapter available to the next request
- [x] #3 NeuTTS imports its vendor module and loads its model on a worker thread, with a test that neither runs on the loop thread
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: TTSAdapterRegistry.get_adapter awaits asyncio.shield(self._base.get_adapter_async(...)), so a caller that stops waiting (the capability listing's 5 s timeout, now adapter_registry._CAPABILITY_LISTING_TIMEOUT_SECONDS, or a client disconnect) no longer cancels the initialization; the shielded task keeps the base registry's per-provider lock, so later callers wait on it and then share the adapter rather than starting a second load. get_all_capabilities logs a timeout at info and no longer calls _schedule_retry for it; list_capabilities, find_adapter_for_requirements and TTSServiceV2._get_fallback_adapter wait at most that timeout for a provider and skip one still loading, without marking it failed (a request that names the loading provider still joins its initialization); the shielded task is kept referenced in _PENDING_INITIALIZATIONS; real initialization errors still mark the provider failed for adapter_failure_retry_seconds (ADR-011). NeuTTS now imports its vendor module and loads its model in _load_engine on a worker thread (run_tts_blocking_call); #3264's ponytail comment is gone. test_registry_closes_adapter_and_reraises_initialization_cancellation now cancels _initialize_adapter directly: a caller's cancellation no longer reaches the initialization, and loop shutdown still closes the half-built adapter. Plan: Docs/superpowers/plans/2026-10-18-tts-cancellation-registry.md (PR B of two; merges before TASK-13856's PR).
Tests: test_registry_capability_timeout_neither_restarts_nor_fails_a_slow_load (an adapter that loads on a worker thread for longer than the listing timeout: one load, provider still enabled, adapter available to the next request) RED on dev (provider marked failed); test_registry_cancelled_caller_leaves_the_initialization_to_the_next_caller (task cancel and anyio cancel scope) RED on dev (assert 2 == 1, a second adapter built); test_registry_list_capabilities_does_not_wait_out_a_slow_load RED on dev (availability failed) and, with the shield alone, TimeoutError from the test's own wait_for; test_registry_best_adapter_skips_a_provider_that_is_still_loading and test_service_fallback_skips_a_provider_that_is_still_loading RED on dev and with the shield alone (TimeoutError: the callers waited out the load); all GREEN with the bounds. test_neutts_imports_and_loads_its_model_off_the_event_loop RED on dev (import on the loop thread). Mutations: dropping the shield fails 5 registry cases (both cancelled-caller cases, the capability test, the list and best-adapter tests); keeping _schedule_retry on timeout fails the capability and list_capabilities tests; removing each wait_for fails exactly its own test. Registry and NeuTTS files: 49 passed in three runs. Sweep (tests/TTS, tests/TTS_NEW, tests/Docs, tests/Infrastructure and every other test file that references the registry, list_capabilities, find_adapter_for_requirements, get_best_adapter, _get_fallback_adapter or neutts, 19 files): 5 failed, 2802 passed, 27 skipped, 1 xfailed, 1 xpassed; the 5 are the plan's baseline (No module named 'av'; torch not installed) and fail on unchanged dev too. Neighbours of the bounded paths (test_tts_adapters, test_tts_service_v2, test_audio_health_helpers, test_tts_runtime_hardening_boundaries): 142 passed, 3 skipped. test_test_quality_ratchet.py passes: the plan's _SlowLoadAdapter asserted its release guard inside a fake class (tautology_suspect), so the guard raises RuntimeError instead. Ruff: only adapter_registry.py C420, present on dev. Bandit -ll (uvx) on adapter_registry.py, tts_service_v2.py, neutts_adapter.py: clean. ADR assessment: none needed; consistent with ADR-011 (a still-running initialization is not a failure). No docs describe the listing timeout.
Clean-session review (#3272), Minors 1-3 applied. M1/M2: TTSAdapterRegistry keeps one in-flight get_adapter_async task per provider in _initializations (keyed by provider and registration generation, with its loop start time), replacing the module set _PENDING_INITIALIZATIONS. Every caller joins it, and its done-callback removes the entry. The registry's three listings (get_all_capabilities, list_capabilities, find_adapter_for_requirements) go through _get_adapter_if_ready, which waits for a provider's load only until that load has run _CAPABILITY_LISTING_TIMEOUT_SECONDS, measured from the load's start, then skips it at once (asyncio.TimeoutError) without a new task or a failure mark. So a slow load costs the first poll at most 5 s and later polls nothing, and 20 polls leave 1 pending task instead of 20. TTSServiceV2._get_fallback_adapter still waits up to 5 s per loading provider, because its registry may be a test double, but it now joins the one task. The generation in the key makes a request after an unload or re-registration load afresh rather than join the superseded load. M3: the done-callback calls task.exception(), because asyncio.shield stops forwarding a late inner error once its caller is cancelled (CPython 3.12). The plan's sentence claiming shield retrieves it is corrected. Tests: 6 new in test_tts_adapter_registry_wrapper_migration.py. 4 are RED on the branch before the fix: 20 == 1 pending tasks; 3 == 2; a second listing waits out the bound; 'Task exception was never retrieved' logged. 2 guard the table: a new adapter after close_all; no joining a superseded load after unload. Mutations, one per mechanism: dropping task.exception(), the reuse, the age skip, the pop on done, or the generation in the key each fails its own test. Registry, tts_adapters, tts_service_v2, NeuTTS and the quality ratchet: 135 passed, 3 skipped. Ruff: only C420 at :1451, also on dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A slow local model load now survives the 5 s capability listing behind GET /audio/health and /audio/providers, and a client disconnect: it is neither cancelled, restarted nor marked failed, and the next request gets the adapter once it loads. The listings and failover skip a provider that is still loading instead of waiting on it. With that, NeuTTS imports and loads its model on a worker thread instead of blocking the event loop once per process.
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
