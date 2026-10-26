---
id: TASK-13923
title: One FVA request can hold most of the shared sync-adapter pool
status: To Do
labels:
- claims
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
POST /claims/verify/fva runs up to max_concurrent_falsifications claims at once (FVAVerifyRequest allows 20, claims_schemas.py near line 763; FVAPipeline.process_batch's semaphore near fva_pipeline.py:403). Each concurrent claim holds one SYNC_ADAPTER_CALL_POOL slot (bounded_daemon.py near line 367, default 32 workers) for each of its up to 21 sequential LLM calls, for up to 60 s per call (claims_engine._run_bounded_claims_analyze, drain_after_timeout). Two concurrent requests at 20 need 40 slots, so other users' non-streaming chat calls on providers without achat (chat_service.py near lines 2706-2710), and other claims and RAG evaluation calls, fail with 'Provider adapter capacity is exhausted'. The endpoint has no rate-limit dependency. Reachable since TASK-13881 (PR #3307) made the endpoint run; on dev before it, every request failed before any LLM call. Clamping through suggest_claims_concurrency, as verify_claims_only does (claims_engine.py near line 1647), does not fix it: that function returns the requested value unless CLAIMS_ADAPTIVE_THROTTLE_ENABLED is set, and that is false by default (config.txt, config.py). Options: a process-wide cap on FVA falsification concurrency (one shared semaphore), a lower schema maximum (an OpenAPI change), or a per-user rate limit on the endpoint. Found by the clean review of PR #3307 (M4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In the default configuration, concurrent FVA requests together cannot hold more than a bounded share of SYNC_ADAPTER_CALL_POOL, with a test
- [ ] #2 While FVA requests run at their maximum concurrency, another user's non-streaming chat call still gets an adapter slot, with a test
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
