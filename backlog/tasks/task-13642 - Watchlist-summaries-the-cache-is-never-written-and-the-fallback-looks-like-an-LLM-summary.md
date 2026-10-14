---
id: TASK-13642
title: 'Watchlist summaries: the cache is never written and the fallback looks like
  an LLM summary'
status: To Do
labels:
- outputs
- watchlists
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
summarize_items_for_output (app/services/outputs_service.py) persists cached_summary only through update_item_metadata, which no DB class defines, so nothing is cached and every output re-summarizes. When a summary fails, the first 300 characters of the article are stored in item.llm_summary and has_llm_summaries is set, so the fallback renders as if the model wrote it. A model reply that itself starts with 'Error:' is also treated as a failure (analyze's error convention). Found while fixing TASK-13636.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Summaries are cached by content hash and reused, or the dead cache code is removed
- [ ] #2 A fallback excerpt is not presented as an LLM summary
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
