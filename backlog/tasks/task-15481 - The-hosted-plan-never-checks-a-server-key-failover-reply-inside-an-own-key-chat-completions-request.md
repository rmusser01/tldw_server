---
id: TASK-15481
title: The hosted plan never checks a server-key failover reply inside an own-key
  /chat/completions request
status: Done
labels:
- quotas
- billing
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-13955 (ADR-083), from its final review (declined to judge) and its plan's Task 3 ruling, an owner-approved trade-off. /chat/completions asks users_own_key_serves once, before the hosted plan's pre-check (_own_key_before_the_hosted_plan, endpoints/chat.py:419-426 on cb70f0c5d1); when the provider runs on the user's own key no LimitEnforcer is entered (:4415-4429), so the request is neither checked against the org's plan (llm_tokens_month) nor added to its hot cache. With provider fallback on (ENABLE_PROVIDER_FALLBACK, :517, default off; allow_provider_fallback_for_request, :3600-3603), a failover runs on the fallback provider's key, usually the server's, by four paths: the pre-dispatch health-check switch (chat.py:5246-5276, which swaps selected_provider for a healthy one after the own-key decision at :4415), the stream retry loop (:5861-5900), execute_streaming_call (Chat/chat_service.py:4667) and _execute_non_stream_call_impl (:6107-6125). That call is checked at the call only against the per-user limits.llm_* (Usage/llm_quota_gate.py:50-69; not at all when the user is within them, because the request is then marked prechecked), never against the hosted plan, so an org over its plan gets a server-key reply. Its usage row (server_default) joins the plan's sum at the next billing cache refresh (BILLING_CACHE_TTL_SECONDS, 60 s, Billing/enforcement.py:26; _OWN_KEY_EXCLUDED, :54, leaves out user rows only); the hot cache gets nothing, because the enforcer is None (_build_streaming_commit_cb, chat.py:6555-6560). No hosted deployment exists, so this is low priority. Options: check the plan at the call for a server-key call inside an own-key request; refuse provider fallback for an own-key request when the hosted plan applies (set allow_provider_fallback_for_request False once own_key is True, chat.py:4415; recommended by TASK-15427's plan); or accept and document. Trade-off of the refusal: own_key is True only when the hosted plan applies (:419-426), so it is scoped and fail-closed (every later failover reads allow_provider_fallback_for_request: :5246, :5661, :5865, :5965), but it removes failover for every own-key request under a hosted plan, including orgs within their plan. Keeping failover for orgs within their plan needs a plan check at the boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner chooses, and the notes and ADR-083 record it: check the hosted plan at the call, refuse provider fallback for an own-key request when the plan applies, or accept and document the gap
- [x] #2 If a check or a refusal is chosen: an own-key /chat/completions request from an org over its plan gets no server-key failover reply, stream and non-stream, by any of the four failover paths, with tests; if the refusal is chosen, the notes and docs say that own-key requests under a hosted plan never fail over (if the check at the call is chosen, an org within its plan keeps failover)
- [x] #3 Docs/Operations/Usage_Quotas.md and ADR-083 (and its Docs/Published mirror) state the behaviour
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-16-hosted-plan-own-key-failover.md. Owner decision (AC #1), 2026-11-16: option B, the plan is asked at the boundary and fallback is refused for an own-key request only when the plan would refuse a server-key call. Fail-closed on a non-HTTP billing error (owner, 2026-11-16). AC #2 read as: no failover reply on another key for an org over its plan, by any of the four paths, and an org within its plan keeps failover (AC text unchanged). The within-plan failover reply stays out of the hot cache, as documented in the ADR-083 amendment.
Verification (2026-11-16): RED on dev with the test half: 6 failed (5 over-plan + broken-plan), 5 passed (within-plan), 8 deselected (plan: 5 failed/5 passed + broken-plan 1 failed). GREEN: quota file 19 passed, netlines 0. Ratchet: create_chat_completion 2902/2908, execute_streaming_call 1542/1545, _execute_non_stream_call_impl 1054/1054; function-size, endpoint-monkeypatch and test-quality ratchets 62 passed; test_quality_triage --enforce exit 0. Wider set: Billing+Usage 21 passed; tests/Docs 214 passed; Chat boundary+Audio hotwords 49 passed; test_chat_endpoint_simplified 223 passed 1 skipped 2 failed under load (test_concurrent_empty_and_valid_streams_keep_usage_accounting_isolated, test_stream_prime_bounds_endless_metadata_and_closes[elapsed]), both pass on rerun (4 passed); tests/lint+CI 1374 passed, 24 failed: 22 workflow-file assertions (ubuntu-latest not in supported images; no workflow file touched) and the 2 known coverage-module env failures; test_runtime_package_discovery.py ignored (no setuptools). All runs netlines 0. shard coverage OK, check_backlog_task_ids exit 0.
Fix pass (2026-11-17), final review Important 1 and 2. Controller ruling 1: a billing data-source failure follows BILLING_ENFORCEMENT_FAILURE_MODE (open: failover runs; closed: withheld), as the server-key pre-check does; the enforcer absorbs those failures itself, so the helper cannot override them; any other error from the check withholds the failover and the own-key request still runs. Code: _the_hosted_plan_refuses catches Exception and logs only the type name; ADR-083 amendment and Usage_Quotas.md reworded. RED: _PlanBroken raising the real NativeQuotaUnavailableError made the broken-plan test fail (error escaped); GREEN: it passes; quota file 19 passed. Follow-up: TASK-15558 (the server-key pre-check returns 500 on the same error). Controller ruling 2 (option a, a streaming-executor fallback test) was NOT implementable: the endpoint wraps the provider call in _sanitize_provider_stream_call, which turns every eager provider error into a lazy iterator error, so execute_streaming_call's own except (and its direct fallback) never fires from the endpoint, even with QUEUED_EXECUTION on and no queue (a temporary probe at that except logged nothing). The stream_in_call case is the retry loop; comment fixed, no chat.py patch added, no ratchet drain done. Minor 1 and 4 done. Verify: quota file 19 passed; function-size, endpoint-monkeypatch and test-quality ratchets 62 passed; test_quality_triage --enforce rc 0; Docs refresh and onboarding parity 37 passed; trust-boundary file 17 passed; all netlines 0.
Controller ruling (2026-11-17, final review Important 2): option (b). The streaming executor's own fallback (chat_service.execute_streaming_call, direct and queued) cannot be reached from /chat/completions: chat.py _sanitize_provider_stream_call turns every eager provider error except LLMPolicyRefusal into a lazy iterator error, so execute_streaming_call's except never sees it and the endpoint's retry loop fails over instead (fix-pass probe: no executor 'Trying fallback provider' line with QUEUED_EXECUTION True). It is also gated by the request flag at enable_provider_fallback=(allow_provider_fallback_for_request and QUEUED_EXECUTION). AC #2 reads as: the three reachable failover paths (pre-dispatch health switch, stream retry loop entered lazily and eagerly, non-stream executor fallback) are tested over and within plan; the fourth is covered by the flag by construction. Ruling (Important 1): a billing data-source failure follows BILLING_ENFORCEMENT_FAILURE_MODE as the server-key pre-check does; any other error withholds the failover; the own-key request still runs. Follow-up: TASK-15558.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
An own-key /chat/completions request with fallback on now asks the hosted plan at the boundary whether it would refuse a server-key call; if so provider fallback is off for the request (covers all four failover paths). Within-plan orgs keep failover; a billing lookup error withholds failover (fail-closed). Documented in the ADR-083 amendment (2026-11-16) and Usage_Quotas.md; 11 new tests.
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
