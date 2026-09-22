---
id: TASK-13314
title: ElevenLabs cleanup closes the process-wide shared pooled HTTP client
status: To Do
assignee: []
created_date: '2026-09-22 04:54'
labels:
  - bug
  - tts
dependencies: []
references:
  - 'tldw_Server_API/app/core/TTS/adapters/elevenlabs_adapter.py:676'
  - 'tldw_Server_API/app/core/TTS/adapters/openai_adapter.py:529'
  - 'tldw_Server_API/app/core/TTS/tts_resource_manager.py:282'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_cleanup_resources calls await self.client.aclose() on a client fetched from the SHARED ConnectionPool cache. aclose() does not evict from _pools - only close_pool/close_client do - so the dead client stays cached.

Init fetches the pooled client then calls _fetch_user_voices() which does a live GET. One transient blip raises, _initialize_adapter finally calls adapter.close(), and every subsequent request including every ADR-011 cooldown retry gets the dead client and raises "Cannot send a request, as the client has been closed". One transient failure becomes permanent until restart. Also reachable per-request in any BYOK deployment via _close_request_adapter.

openai_adapter.py:529-541 is the correct copy; qwen3_runtime_remote has no override and is also correct. The only test touching _cleanup_resources PINS THE DEFECT (mocks the client and asserts the aclose path).

Source: synthesis F14
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Adapter cleanup no longer closes a pooled client it does not own
- [ ] #2 Existing test retargeted; regression test asserts the pool survives adapter close
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
