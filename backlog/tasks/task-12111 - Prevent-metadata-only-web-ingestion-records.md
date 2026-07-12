---
id: TASK-12111
title: Prevent metadata-only web ingestion records
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-12 20:35
labels:
- bug
- web-scraping
- ingestion
dependencies: []
documentation:
- docs/superpowers/specs/2026-07-12-metadata-only-web-ingestion-guard-design.md
- docs/superpowers/plans/2026-07-12-metadata-only-web-ingestion-guard-implementation-plan.md
priority: high
modified_files:
- tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py
- tldw_Server_API/app/services/enhanced_web_scraping_service.py
- tldw_Server_API/app/services/web_scraping_service.py
- tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py
- tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py
- tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix web ingestion so description-only JSON-LD does not short-circuit full-page extraction, preserve structured summaries while later extractors obtain body content, and skip empty-body persistence in enhanced and legacy paths with regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Description-only JSON-LD does not count as successful body extraction.
- [ ] #2 A structured-data summary is preserved when a later extractor supplies the page body and is not erased when optional legacy summarization is disabled.
- [ ] #3 Enhanced and legacy persistence reject missing, non-string, whitespace-only, and recognized metadata-envelope-only bodies; they report per-URL errors without aborting valid siblings.
- [ ] #4 Persistence response status remains backward-compatible while media IDs, stored counts, and errors expose skipped items.
- [ ] #5 Focused tests and Bandit pass for touched Python scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec approved; implementation planning pending.
<!-- SECTION:NOTES:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec approved. TDD implementation plan written, independently reviewed, and approved. Ready for execution handoff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
