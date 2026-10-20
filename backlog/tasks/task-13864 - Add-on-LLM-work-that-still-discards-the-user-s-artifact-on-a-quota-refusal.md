---
id: TASK-13864
title: Add-on LLM work that still discards the user's artifact on a quota refusal
status: In Progress
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
- [ ] #1 Usage_Quotas.md says what each path keeps
- [ ] #2 Each listed path keeps the transcript or page it produced and records llm_quota_exceeded where the user can see it, or refuses with the 402 before it fetches or produces anything (LLM usage accounting design section 4a), with a test per path
- [ ] #3 A Prompt Studio evaluation refused part-way keeps test_run_ids for the cases that ran, with a test
- [ ] #4 A scraping queue job runs in the context of the request that submitted it, and a quota refusal fails only that job, with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the #3277 clean-session review (M2): a Prompt Studio evaluation refused part-way drops the links to the cases that already ran. EvaluationManager.run_evaluation_with_existing_record (prompt_studio/evaluation_manager.py near lines 258-268) writes status, completed_at and error_message on the failed update, but not the test_run_ids collected so far; those test runs stay stored, but the evaluation record no longer points to them. Fix: add "test_run_ids": test_run_ids to that update, with a test that refuses the second of two cases and reads the record's test_run_ids back.
Plan: Docs/superpowers/plans/2026-10-20-keep-artifact-on-llm-quota-refusal.md. Per its Decision 2, AC1 is replaced: the HTTP endpoints refuse with the 402 before any work, so nothing is produced and discarded. The reworded criterion is now #2; #1 is 'Usage_Quotas.md says what each path keeps'; #3 is the Prompt Studio fix from these notes; #4 is the scraping queue defect found by the plan review.
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
