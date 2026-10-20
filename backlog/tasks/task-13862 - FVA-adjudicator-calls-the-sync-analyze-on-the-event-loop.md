---
id: TASK-13862
title: FVA adjudicator calls the sync analyze on the event loop
status: Done
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
- [x] #1 The adjudicator's LLM call runs off the event loop (asyncio.to_thread or the bounded adapter pool), with a test
- [x] #2 An LLM quota refusal there is let through, not counted as a failed assessment
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-fva-adjudicator-off-loop.md. The adjudicator's two analyze calls go through claims_engine._run_bounded_claims_analyze (the pool the verifier uses); a refusal is re-raised before the broad clauses in _llm_assess, _assess_documents and FVAPipeline.process_claim. Measured: POST /claims/verify/fva, the only app caller, answers 500 to every request (three constructor calls that cannot succeed: ClaimsJobContext(job_id=...), MultiDatabaseRetriever(media_db=db) without db_paths, RetrievalConfig(top_k=...)); owner Decision 1.
RED on origin/dev: 3 failed (the off-loop test assert [(False, False)] == [(True, True)]: the call ran on the loop's thread and was not checked; the adjudicate and process_claim refusal tests DID NOT RAISE, the refusal became NEUTRAL 0.5). GREEN: 3 passed. The off-loop test also asserts that the quota decision is made for the request's user (decided_for == [7]); a move off the loop without the LLM call state fails it. tests/Claims + tests/Claims_Extraction: 1222 passed, 1 failed, 1 skipped in the worktree, twice; the failure is the Hypothesis test test_review_metrics_payload_inclusive_window_property, FailedHealthCheck too_slow (4 inputs in 1.4-4.5 s). It passes alone (also with the failing seed) and tests/Claims alone passes (1081 passed, 1 skipped). The same branch files in a scratch copy of the tree: 1223 passed, 1 skipped; unchanged dev files there: 1220 passed, 1 skipped, twice. So it is the run's location and load, not the change (the plan's reviewer also saw it fail on unchanged dev). Test-quality ratchet exit 0, no offenses beyond the baseline; CI ratchet + tests/Docs: 213 passed; ruff: the same 4 findings as dev; bandit -q: clean, exit 0.
Decision 1, recommended option: fix the adjudicator now; the endpoint repair is TASK-13881. POST /claims/verify/fva answers 500 to every request (three constructor calls that cannot succeed), so the change is latent until then; a prototype with the endpoint repaired answered 402 for a refusal at the adjudication call (200 with an unchecked call on dev).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ClaimAdjudicator calls its analyze function through claims_engine._run_bounded_claims_analyze, so the call runs on the bounded adapter pool, is checked at the call and no longer blocks the loop. A quota refusal is re-raised before the broad clauses in _llm_assess, _assess_documents and FVAPipeline.process_claim, so it fails the batch instead of becoming a neutral stance. The only caller, POST /claims/verify/fva, answers 500 to every request; its repair is TASK-13881, and the operator page says so.
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
