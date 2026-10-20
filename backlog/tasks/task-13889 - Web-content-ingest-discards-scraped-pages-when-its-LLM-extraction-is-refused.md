---
id: TASK-13889
title: Web content ingest discards scraped pages when its LLM extraction is refused
status: To Do
labels:
- quotas
- media
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
/media/ingest-web-content allows the LLM extraction strategy whenever perform_analysis is true, which is the default (services/web_scraping_service.py, ingest_web_content_orchestrate: allow_llm_extraction=bool(request.perform_analysis)). The strategy is in DEFAULT_EXTRACTION_STRATEGY_ORDER (Web_Scraping/extraction/pipeline.py) and runs when jsonld, schema and regex fail and a provider resolves from llm_settings or RAG_DEFAULT_LLM_PROVIDER. TASK-13864 checks the quota at the request boundary only when a summary provider is named (media_analysis_requested), so with no api_name the extraction call is checked at the call; its refusal is re-raised (extraction/strategies/llm.py, call_llm_provider). The individual method then answers 402, discarding the pages already scraped in that request; the url_level and recursive_scraping methods run their pages as scraping queue jobs, where a refused page becomes a failed page carrying the 402 text (TASK-13864 Task 4). Found while planning TASK-13864.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An over-quota user's /media/ingest-web-content request with perform_analysis and no api_name gets the 402 before any page is fetched, or keeps its pages without LLM extraction, with a test
- [ ] #2 Usage_Quotas.md says which
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
