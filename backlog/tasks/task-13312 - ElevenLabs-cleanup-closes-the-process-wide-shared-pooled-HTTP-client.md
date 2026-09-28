---
id: TASK-13312
title: ElevenLabs cleanup closes the process-wide shared pooled HTTP client
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-23 23:13'
labels:
  - bug
  - tts
  - reliability
dependencies: []
references:
  - 'tldw_Server_API/app/core/TTS/adapters/elevenlabs_adapter.py:676'
  - 'tldw_Server_API/app/core/TTS/adapters/openai_adapter.py:529'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/TTS/adapters/elevenlabs_adapter.py:_cleanup_resources` calls `aclose()` on a client it borrowed from the shared pool:

```python
if self.client:
    await self.client.aclose()     # <- closes the POOLED, process-wide client
    self.client = None
```

`tts_resource_manager.ConnectionPool.get_client` caches by provider name and returns the **same object** to every later borrower. The only sanctioned teardown is `close_pool`, which closes **and evicts** from `_pools`. Closing without evicting leaves a dead client cached.

The correct sibling documents exactly why this is wrong — `openai_adapter.py:529-541`:

```python
# Note: HTTP clients are now managed by the resource manager
# No need to manually close them as they use connection pooling
self.client = None
```

**Effect:** the documented admin operation `POST /api/v1/audio/tts/providers/elevenlabs/unload` — whose docstring says "so it can be reloaded on demand" — closes the pooled client and never calls `close_pool`. Every subsequent `/audio/speech` request re-inits the adapter, gets the cached **closed** client, and raises `RuntimeError: Cannot send a request, as the client has been closed.` **ElevenLabs is dead until process restart.** The same trigger exists on the abandoned-init path.

This defeats ADR-011, which explicitly states adapter init failures "can be retried after a configured cooldown" and lists "permanent until process restart" as the *rejected* alternative.

`qwen3_runtime_remote.py` correctly inherits the base no-op. Note the one test that touches this **pins the defect** rather than catching it.

Found by the comprehensive core-module review; independently verified by the orchestrator against the correct sibling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test unloads the provider then issues a request and asserts it succeeds rather than raising client-has-been-closed
- [x] #2 If a provider genuinely needs pool teardown, it calls close_pool (which evicts) rather than aclose
- [x] #3 The existing test that pins the current behaviour is updated rather than left asserting the defect
- [x] #4 ADR-011's retry-after-cooldown property holds for ElevenLabs after the fix
- [x] #5 The ElevenLabs _cleanup_resources override closes only a client the adapter created itself (never a pooled one), matching openai_adapter for the pooled case
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Duplicate of TASK-13314 (same finding, filed twice), fixed in b5b6dc66a9. AC2 reworded 2026-09-23: deleting the override outright would leak the client the convenience API creates without initialize(); b5b6dc66a9 instead tracks _owns_client and closes only an owned client, which is the openai_adapter behaviour for the pooled case. AC1/AC5: unload_provider (POST .../unload) and the init-failure finally both go through adapter.close() -> _cleanup_resources; test_pool_still_serves_a_live_client_after_adapter_close runs that path against the real HTTPConnectionPool and asserts the next get_client returns the same live, unclosed client, so the retry-after-cooldown request gets a working client. AC3: no provider needs pool teardown; none calls aclose on a pooled client. AC4: test_cleanup_failure_log_sanitizes_exception_text was retargeted to the owned path in b5b6dc66a9. Follow-up found while verifying, fixed in 9569e97ecd: clone_voice, get_usage and generate_stream still created their own client inline without setting _owns_client, so close()/unload dropped it unclosed (leak). They now use _ensure_client. Regression test test_other_convenience_calls_own_and_close_their_client[clone_voice|get_usage]: red on ea1cbc6941, green after. tests/TTS: 8 failed/602 passed before vs 6/604 after; the only difference is the two new tests (6 pre-existing, unrelated). Bandit clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Already fixed as TASK-13314 (b5b6dc66a9): the adapter closes only clients it owns, so unload no longer kills the pooled ElevenLabs client. Closed the remaining leak where three convenience calls created unowned clients (9569e97ecd).
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
