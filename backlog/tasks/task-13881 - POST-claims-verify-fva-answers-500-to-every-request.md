---
id: TASK-13881
title: POST /claims/verify/fva answers 500 to every request
status: To Do
labels:
- claims
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
verify_claims_with_fva (Claims_Extraction/claims_service.py) has three constructor calls that cannot succeed, so the endpoint answers 500 to every request before any LLM call: (1) ClaimsJobContext(job_id=..., user_id=...) near line 4759, made when fva_config.max_budget_usd is set; ClaimsJobContext has no job_id field. (2) MultiDatabaseRetriever(media_db=db) near line 4770, without the required positional db_paths. (3) RetrievalConfig(top_k=..., search_mode='hybrid') near line 4782; RetrievalConfig has max_results, use_fts and use_vector. All date from the endpoint's first commit (05245a3d59, 2026-02-05); no test calls it. The user_id query parameter needs a decision before call (1) is fixed. Today it reaches the usage row only through ClaimsJobContext; with call (1) fixed and user_id=user_id kept, the claims recorder (claims_engine.py near line 677) writes each claims_verify usage row for whatever id any authenticated caller passes, and for the caller's username when none is passed, so the caller's own verify usage never counts toward its limits. Quota decisions stay with the authenticated user. Recommended: admin-gate the parameter as rebuild_claims does (claims_service.py near line 4581: the parameter only for _legacy_user_has_platform_admin_claims, else current_user.id), and record the authenticated user's id, not the username. Alternatives: drop the parameter (an OpenAPI and apps/packages/api-types change), or accept and ignore it. Also: retrieval should be scoped to the authenticated user; FVAPipeline reads the verifier's lazily loaded _nli before it is loaded, so the adjudicator never uses NLI. Since the endpoint has never worked, removing it is the other option. Found while planning TASK-13862: its prototype with calls (2) and (3) fixed answered 402 for a refusal at the verify or adjudication call, with every decision made for the request's user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 POST /claims/verify/fva runs the FVA pipeline on the caller's media DB, with an endpoint test that checks a result
- [ ] #2 A quota refusal at its verify or adjudication call answers 402, with a test
- [ ] #3 Claims usage rows are written for the authenticated user's id unless an admin passes user_id, with a test
- [ ] #4 Usage_Quotas.md lists POST /claims/verify/fva among the endpoints that answer 402
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision, 2026-10-20 (approving the TASK-13862 plan): repair the endpoint in this task, and admin-gate the user_id parameter as the sibling claims endpoints do (rebuild_claims); the two alternatives are not taken. Until this lands, Docs/Operations/Usage_Quotas.md says the endpoint answers 500.
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
