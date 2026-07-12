---
id: TASK-12111
title: Prevent metadata-only web ingestion records
status: In Progress
labels:
- bug
- web-scraping
- ingestion
priority: High
modified_files:
- tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py
- tldw_Server_API/app/services/enhanced_web_scraping_service.py
- tldw_Server_API/app/services/web_scraping_service.py
- tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py
- tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py
- tldw_Server_API/tests/Web_Scraping/test_auto_chunking_web_ingest.py
documentation:
- docs/superpowers/specs/2026-07-12-metadata-only-web-ingestion-guard-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix web ingestion so description-only JSON-LD does not short-circuit full-page extraction, preserve structured summaries while later extractors obtain body content, and skip empty-body persistence in enhanced and legacy paths with regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Description-only JSON-LD does not count as successful body extraction.
- [ ] #2 A structured-data summary is preserved when a later extractor supplies the page body.
- [ ] #3 Enhanced and legacy persistence skip whitespace-only article bodies and report per-URL errors without aborting the batch.
- [ ] #4 Focused tests and Bandit pass for touched Python scope.
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
