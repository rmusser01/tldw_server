---
id: TASK-15481
title: The hosted plan never checks a server-key failover reply inside an own-key
  /chat/completions request
status: To Do
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
- [ ] #1 The owner chooses, and the notes and ADR-083 record it: check the hosted plan at the call, refuse provider fallback for an own-key request when the plan applies, or accept and document the gap
- [ ] #2 If a check or a refusal is chosen: an own-key /chat/completions request from an org over its plan gets no server-key failover reply, stream and non-stream, by any of the four failover paths, with tests; if the refusal is chosen, the notes and docs say that own-key requests under a hosted plan never fail over (if the check at the call is chosen, an org within its plan keeps failover)
- [ ] #3 Docs/Operations/Usage_Quotas.md and ADR-083 (and its Docs/Published mirror) state the behaviour
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
