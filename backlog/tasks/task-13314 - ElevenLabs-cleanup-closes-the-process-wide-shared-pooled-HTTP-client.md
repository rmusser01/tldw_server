---
id: TASK-13314
title: ElevenLabs cleanup closes the process-wide shared pooled HTTP client
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-22 22:34'
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
- [x] #1 Adapter cleanup no longer closes a pooled client it does not own
- [x] #2 Existing test retargeted; regression test asserts the pool survives adapter close
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in b5b6dc66a9.

Reproduced first, against the real HTTPConnectionPool: after aclose() on a pooled client, "elevenlabs" is still in _pools, get_client hands back the same object, and is_closed is True. Confirms the finding -- aclose() does not evict, only close_pool/close_client do.

AC1: the naive fix (copy openai_adapter, which just drops the reference) would have leaked. ElevenLabs has two client provenances: the shared pool in initialize(), and its own create_async_client() in the convenience API (fetch_voices/get_voice_info), which is reachable without initialize(). So the adapter now tracks ownership -- _owns_client, False on borrow and True on self-create -- and closes only what it owns. The two duplicated lazy-create blocks collapsed into _ensure_client, which lives on ElevenLabsTTSAdapter alongside its only callers.

AC2: test_cleanup_failure_log_sanitizes_exception_text pinned the defect (mocked the client, asserted the aclose path). Retargeted to the owned path -- which is where aclose can legitimately raise, so the test keeps its real purpose of checking the failure log is sanitized -- with the reason in a comment rather than changed silently. Four new tests in TestElevenLabsPooledClientOwnership: pooled client not closed, owned client closed, the pool still serves a live client after adapter close (the end-to-end property), and the convenience-API client is owned and closed. All four are red against the pre-fix adapter.

Checked the sibling adapters for the same pattern: omnivoice_sidecar_supervisor has its own get_http_client but owns the client, clears the reference, and re-creates it if closed -- no defect. qwen3_runtime_remote has no override. openai_adapter is correct as the finding said. ElevenLabs was the only one.

Verification: tests/TTS 54 failed / 553 passed with the change vs 58 / 549 without (stash-isolated); the difference is exactly these four tests, so no regression. The 54 are pre-existing and untouched by this change. Bandit clean on the adapter (run via uvx; bandit is CI-only, not a declared local dependency).

Environment note: the real client factory cannot be exercised locally -- create_async_client raises PackageNotFoundError for "tldw-server" because `pip install -e .` has not been run in this checkout, which HTTPConnectionPool.get_client converts into a generic TTSNetworkError. Not a code defect; the reproduction used a stub client to isolate the pool semantics under test.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapter now tracks whether it owns its HTTP client and closes only its own, so a transient init failure no longer poisons the process-wide pool for every later ElevenLabs request. Copying the openai_adapter version would have leaked the convenience API's self-created client.
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
