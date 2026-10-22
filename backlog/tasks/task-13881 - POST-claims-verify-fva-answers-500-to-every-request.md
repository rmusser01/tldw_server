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
- [ ] #1 POST /claims/verify/fva runs the FVA pipeline on the media DB and vector store of the user that AC #3 resolves, with an endpoint test that checks a result
- [ ] #2 A quota refusal at its verify or adjudication call answers 402, with a test
- [ ] #3 The user_id query parameter counts only for a platform admin, as rebuild_claims does (_legacy_user_has_platform_admin_claims). With an admin's user_id, the media DB, the vector store, the quota decision and the usage rows all belong to that user; otherwise all of them belong to the authenticated user, by id, not username. With a test for each case: a non-admin's user_id is ignored for all four, and an admin's is honoured for all four
- [ ] #4 Usage_Quotas.md lists POST /claims/verify/fva among the endpoints that answer 402
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision, 2026-10-20 (approving the TASK-13862 plan): repair the endpoint in this task, and admin-gate the user_id parameter as the sibling claims endpoints do (rebuild_claims); the two alternatives are not taken. Until this lands, Docs/Operations/Usage_Quotas.md says the endpoint answers 500.
Review of PR #3287 (I1), 2026-10-20: AC #1 and AC #3 reworded. As first written, AC #3 admin-gated user_id only for the usage row, so a repair could meet every AC and still let a non-admin's ?user_id=2 read user 2's embeddings: the user_id passed to MultiDatabaseRetriever picks the vector store (MediaDBRetriever: create_from_settings_for_user near database_retrievers.py:997, collection user_{id}_media_embeddings near :2203), and the snippets come back as evidence. AC #1's 'the caller's media DB' also contradicted the admin case. AC #3 now gives user_id one meaning for the media DB, vector store, quota decision and usage rows, with a test for the non-admin and the admin case; AC #1 follows it.
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
