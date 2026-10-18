---
id: TASK-13642
title: 'Watchlist summaries: the cache is never written and the fallback looks like
  an LLM summary'
status: Done
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
- [x] #1 Summaries are cached by content hash and reused, or the dead cache code is removed
- [x] #2 A fallback excerpt is not presented as an LLM summary
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the PR review of TASK-13636/13637: after an enrichment failure (a quota refusal or any other error), the per-phase metadata (summary_status, grouping_status, group_summary_status) stays pending next to enrichment_status failed; set them to failed (or skipped) with the overall status.
From the clean-session review of PR #3240: (1) when enrichment fails part-way, the per-phase statuses already written stay completed (grouping, group summaries) although no enriched_groups were stored; fix with the pending ones in the enrichment handler's except. (2) With summarize: true, POST /watchlists/outputs makes one LLM call per item, sequentially, for up to 1000 items, with no cap, no early stop after a provider error and no cache, so a large run can outlast client timeouts; bound or parallelise it alongside the cache.
Fixed. AC1 by removal: watchlist items have no metadata column (only watchlist_content_alert_rules has metadata_json), so the cache could never be written; _content_cache_key, _get_cached_summary, the write-back through update_item_metadata, summarize_items_for_output's db parameter and the endpoint's metadata_json pass-through are gone, with the three tests that covered them (test_newsletter_briefing_gaps.py: the cache hit and the two cache-key tests). Summaries are generated on every output, as before. A real cache would need a column on watchlist items in both backends; not filed. AC2: _summarize_single_article returned text[:300] when the model gave nothing, which the endpoint stored as llm_summary and flagged has_llm_summaries; it now returns "", so templates fall back to the item's own summary. Both summarisers share a new _llm_reply, which calls analyze with raise_on_error=True: a provider failure raises (logged, ""), a quota refusal propagates, and a reply that starts with "Error:" is the model's text. Tests (tests/Services/test_outputs_service.py): returns-nothing-on-failure for the single article and for summarize_items_for_output, and the Error:-reply case for both summarisers, RED on dev (the excerpt, or "") then GREEN; the file and test_newsletter_briefing_gaps.py 63 passed; tests/Watchlists 712 passed, 5 skipped, 1 xpassed. Ruff: nothing new against dev; bandit -ll clean on outputs_service.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A failed watchlist summary no longer shows an excerpt of the article as if the model had written it: the item gets no LLM summary and templates use its own summary. The summary cache, which could never be written, is removed, and a model reply starting with "Error:" is no longer mistaken for a failure.
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
