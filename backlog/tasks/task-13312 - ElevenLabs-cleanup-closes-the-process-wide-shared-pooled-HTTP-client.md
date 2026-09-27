---
id: TASK-13312
title: ElevenLabs cleanup closes the process-wide shared pooled HTTP client
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-23 00:12'
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
- [x] #2 The ElevenLabs _cleanup_resources override is deleted so it inherits the base no-op, matching openai_adapter
- [x] #3 If a provider genuinely needs pool teardown, it calls close_pool (which evicts) rather than aclose
- [x] #4 The existing test that pins the current behaviour is updated rather than left asserting the defect
- [x] #5 ADR-011's retry-after-cooldown property holds for ElevenLabs after the fix
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed test-first and merged to dev in PR #2980 (merge commit 8045fa2956). A failing test reproduced the defect before any code changed, with controls pinning the behaviour that had to stay unchanged. Qodo review then found follow-on defects in three of this batch's fixes; those were corrected in the same PR before merge.
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
