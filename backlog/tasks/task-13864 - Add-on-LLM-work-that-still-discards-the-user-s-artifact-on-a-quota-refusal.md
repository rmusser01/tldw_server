---
id: TASK-13864
title: Add-on LLM work that still discards the user's artifact on a quota refusal
status: Done
labels:
- quotas
- media
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13638 Decision 3 keeps the artifact the user asked for and records the refusal when only an add-on LLM step is refused (transcript reuse; plan_audio_analysis.md does the same for audio jobs). These paths still discard the artifact: media ingest jobs, single and batch (the item is marked Error, Processing error: LLMQuotaExceededError; a batch marks every item); /media/ingest-web-content with analysis and /media/process-web-scraping with summarization, which discard the request's results; POST /watchlists/outputs with summarize, which discards them too (Usage_Quotas.md, the HTTP endpoint bullet under Everything else is checked at the call).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Usage_Quotas.md says what each path keeps
- [x] #2 Each listed path keeps the transcript or page it produced and records llm_quota_exceeded where the user can see it, or refuses with the 402 before it fetches or produces anything (LLM usage accounting design section 4a), with a test per path
- [x] #3 A Prompt Studio evaluation refused part-way keeps test_run_ids for the cases that ran, with a test
- [x] #4 A scraping queue job runs in the context of the request that submitted it, and a quota refusal fails only that job, with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the #3277 clean-session review (M2): a Prompt Studio evaluation refused part-way drops the links to the cases that already ran. EvaluationManager.run_evaluation_with_existing_record (prompt_studio/evaluation_manager.py near lines 258-268) writes status, completed_at and error_message on the failed update, but not the test_run_ids collected so far; those test runs stay stored, but the evaluation record no longer points to them. Fix: add "test_run_ids": test_run_ids to that update, with a test that refuses the second of two cases and reads the record's test_run_ids back.
Plan: Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md. Per its Decision 2, AC1 is replaced: the HTTP endpoints refuse with the 402 before any work, so nothing is produced and discarded. The reworded criterion is now #2; #1 is 'Usage_Quotas.md says what each path keeps'; #3 is the Prompt Studio fix from these notes; #4 is the scraping queue defect found by the plan review.
RED on origin/dev (implemented on bfeec716be; touched files unchanged since the plan's 2e6ceab9fb): Jobs 3 failed, 1 passed (document batch: Processing error: LLMQuotaExceededError, media_id None; audio: Failed to call processor: LLMQuotaExceededError; admitted job checked at the call, [1] != [None]); with the job-start check but without its llm_call_context() scope, the admission-scope test failed (assert True is False); reused transcript failed with KeyError 'video_lite' after Task 1; web endpoints and watchlist outputs 3 failed, 3 passed (200 where 402 is expected; 3 controls passed); scraping queue 2 failed (job ran as user 7 prechecked; TimeoutError after the worker died on a refusal); Prompt Studio 1 failed, 2 passed (KeyError: 'test_run_ids'). GREEN: Task 1 4 passed; Task 2 1 passed, 13 passed with the dedupe, reused-transcript and Task 1 files; Task 3 7 passed with test_output_summaries_return_a_quota_refusal_as_402; Task 4 2 passed; Task 5 3 passed. Neighbourhood suites (-n 4): A 1321 passed, 14 skipped, 1 xfailed after Task 2 (after Task 1 alone, 1 failed, 1319 passed: test_transcription_failure_handling, the known dev xdist order dependence, 1 passed alone); B 4064 passed, 37 skipped, 1 xpassed; tests/prompt_studio 1181 passed, 17 skipped; Services/test_openapi_contracts.py alone 74 passed (alone because of the OpenAPI operationId order dependence, TASK-13890); tests/Docs 212 passed. Test-quality triage: exit 0; ratchet: 1 passed; shard coverage OK; scope predicates baseline exact; ruff 23 findings before and after, nothing new; OpenAPI gate: fingerprint and path union match.
Decisions taken (owner, 2026-10-20, both as recommended): Decision 1, a media ingest job checks the quota once when it starts; refused, it stores the item without analysis and records the refusal in its result, and a reused transcript keeps its item-level summary_error; admitted, it runs to completion. Decision 2, the three HTTP endpoints answer 402 before any work (AC reworded). Batch: each media ingest job checks once when it starts; no batch-wide flag. Follow-ups: TASK-13889 (LLM extraction during web content ingest without a named provider), TASK-13890 (connectors webhook duplicate operationId).
Known gaps (plan Review Focus): a refused scraping queue job outside the app's routes becomes a failed page, not a 402; an admitted job can pass the limit by its own analysis, as an admitted request can; enrichment run as a request background task inherits the request's admission; a lazily started core Scheduler inherits the starting request's admission (harmless while every LLM-calling handler runs under llm_call_root). Bandit (uvx bandit -q -ll) on the 8 touched app files: no findings.
#3292 clean review (I1, Important): at the first push AC #2 was not fully met. The url_level and recursive_scraping methods of /media/ingest-web-content summarize with api_name or "openai", so a request with the default perform_analysis and no api_name skipped the boundary check, crawled, and was then refused (402 after fetching). Fixed in this PR: the endpoint now checks whenever perform_analysis is on and either a provider is named or the method is url_level or recursive_scraping. Two new WEB_SUMMARY_REQUESTS cases (url_level and recursive_scraping, no api_name) failed before the fix (200 with the pages fetched and summarized, since the test fixture's adapter skips the check at the call) and pass after (402, no fetch, no LLM call). Usage_Quotas.md and the design's §4a table row say so, and TASK-13889's description is corrected: it now covers only the individual and sitemap methods. With this, AC #2 is met for every listed path.
#3292 clean review (M1, Minor): the Known gaps note above said a lazily started core Scheduler is harmless because every Scheduler handler that makes an LLM call runs under llm_call_root. At the first push that was not true: watchlist_run had no root and can make an LLM extraction call (run_watchlist_job, fetch_site_article, scrape_article_blocking with allow_llm_extraction defaulting to True). It was inert only because watchlist_run is submitted by the workflows recurring scheduler, which starts the global Scheduler at startup in a clean context. Fixed in this PR: watchlist_run now runs its pipeline under llm_call_root(user), as watchlists_enrich_output does, with a test that the run sees its payload's user and no admission even inside another user's admitted context. The statement now holds.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A media ingest job that asks for analysis checks the owner's LLM quota once when it starts: refused, it stores the item without analysis and its result says why (status Warning, analysis_error.code llm_quota_exceeded), and a reused transcript still records video_lite.summary_error; admitted, it runs to completion like an admitted request, and the admission ends with the job. /media/ingest-web-content with analysis, /media/process-web-scraping with summaries and POST /watchlists/outputs with summarize refuse an over-quota user with the 402 before any page is fetched or output rendered. Scraping queue jobs run in their submitter's context; a refusal fails only that job. A Prompt Studio evaluation refused part-way keeps test_run_ids. Usage_Quotas.md, the design doc and the media ingest jobs API doc say what each path keeps.
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
