---
id: TASK-13864
title: Add-on LLM work that still discards the user's artifact on a quota refusal
status: To Do
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
- [ ] #1 Each listed path keeps the transcript, page or output it produced and records llm_quota_exceeded where the user can see it, with a test per path
- [ ] #2 Usage_Quotas.md says what each path keeps
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
