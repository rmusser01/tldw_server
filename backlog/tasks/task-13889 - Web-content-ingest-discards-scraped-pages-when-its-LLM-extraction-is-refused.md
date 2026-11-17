---
id: TASK-13889
title: Web content ingest discards scraped pages when its LLM extraction is refused
status: Done
labels:
- quotas
- media
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
/media/ingest-web-content allows the LLM extraction strategy whenever perform_analysis is true, which is the default (services/web_scraping_service.py, ingest_web_content_orchestrate: allow_llm_extraction=bool(request.perform_analysis) in the individual and sitemap methods). The strategy is in DEFAULT_EXTRACTION_STRATEGY_ORDER (Web_Scraping/extraction/pipeline.py) and runs when jsonld, schema and regex fail and a provider resolves from llm_settings or RAG_DEFAULT_LLM_PROVIDER. TASK-13864 checks the quota at the request boundary for these two methods only when a summary provider is named (media_analysis_requested), because without one they make no summary call. With no api_name the extraction call is checked at the call, and its refusal is re-raised (extraction/strategies/llm.py, call_llm_provider). The individual method then answers 402, discarding the pages already scraped in that request; what the sitemap method (scrape_from_sitemap, run in a thread) does with the refusal is not measured. The url_level and recursive_scraping methods are not affected: they summarize with a default provider (api_name or "openai"), so after the #3292 clean review (I1) TASK-13864 checks them at the boundary whenever perform_analysis is on, and an admitted crawl's queue jobs run prechecked. Found while planning TASK-13864; description corrected after that review, which found the earlier text wrong about the crawl methods.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An over-quota user's /media/ingest-web-content request with perform_analysis and no api_name gets the 402 before any page is fetched, or keeps its pages without LLM extraction, with a test
- [x] #2 Usage_Quotas.md says which
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: 2 failed, 80 passed (test_web_summary_service_prompt.py, individual and sitemap no-provider params). GREEN: 106 passed with the sanitization file. Decision 1 = A: 402 before any fetch whenever perform_analysis is on. Usage_Quotas.md and the usage accounting design row updated. Bandit -ll clean. No ADR needed. Not covered: a user going over quota mid-request.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Boundary quota check runs whenever perform_analysis is on.
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
