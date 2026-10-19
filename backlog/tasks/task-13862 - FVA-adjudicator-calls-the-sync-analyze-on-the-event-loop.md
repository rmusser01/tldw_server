---
id: TASK-13862
title: FVA adjudicator calls the sync analyze on the event loop
status: To Do
labels:
- quotas
- claims
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ClaimAdjudicator._call_llm (Claims_Extraction/adjudicator.py, near lines 291-325) calls the sync analyze function (_fva_claims_analyze_call) directly from a coroutine, so the LLM quota check at the call is skipped (LLM usage accounting design section 5) and the server loop blocks for the provider call. Its except Exception in _assess_documents (near line 231) would also swallow a refusal once the call moves off the loop. Found while planning TASK-13638.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The adjudicator's LLM call runs off the event loop (asyncio.to_thread or the bounded adapter pool), with a test
- [ ] #2 An LLM quota refusal there is let through, not counted as a failed assessment
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
