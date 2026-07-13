---
id: TASK-12951
title: Prevent metadata-only web ingestion records
status: Done
labels:
- bug
- web-scraping
- ingestion
priority: high
documentation:
- docs/superpowers/specs/2026-07-12-metadata-only-web-ingestion-guard-design.md
modified_files:
- tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py
- tldw_Server_API/app/services/enhanced_web_scraping_service.py
- tldw_Server_API/app/services/web_scraping_service.py
- tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py
- tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py
- tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py
references:
- https://github.com/rmusser01/tldw_server/pull/2718
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix web ingestion so description-only JSON-LD does not short-circuit full-page extraction, preserve structured summaries while later extractors obtain body content, and skip empty-body persistence in enhanced and legacy paths with regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Description-only JSON-LD does not count as successful body extraction.
- [x] #2 A structured-data summary is preserved when a later extractor supplies the page body and is not erased when optional legacy summarization is disabled.
- [x] #3 Enhanced and legacy persistence reject missing, non-string, whitespace-only, and recognized metadata-envelope-only bodies; they report per-URL errors without aborting valid siblings.
- [x] #4 Persistence response status remains backward-compatible while media IDs, stored counts, and errors expose skipped items.
- [x] #5 Focused tests and Bandit pass for touched Python scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation completed using TDD and subagent-driven review. Final post-rebase verification on current origin/dev: 44 focused tests passed; Black passed for task-owned test files; Ruff passed for all touched files except the unchanged pre-existing I001/F841 findings in Article_Extractor_Lib.py; current origin/dev already has unrelated Black debt in both service files and Black proposed no changes to TASK-12951-added service lines; compileall passed; Bandit reported 0 findings and 0 errors; git diff checks passed; final whole-range review found no issues. The post-rebase scrape_article test double was aligned with the upstream allow_llm_extraction keyword. Known skip: the full repository suite was not run because verification was scoped to affected web-ingestion and adjacent persistence suites.
Draft PR created: https://github.com/rmusser01/tldw_server/pull/2718. The PR remains draft pending the repository-required human-authored Change summary.
PR #2718 review remediation started: rebasing/auditing against latest origin/dev, adding test-first regressions for normalized wrapped-body persistence and safe URL logging, and addressing all actionable inline feedback.
PR review remediation implemented test-first. Added non-string envelope handling, normalized-body persistence for enhanced and legacy paths, redacted warning-log URLs while preserving raw per-URL response errors, unit markers, helper docstring, and explicit B108 suppression rationale. Kept the existing nullable `errors` response key to preserve compatibility with the enhanced service and approved response contract. Verification: 40 focused/adjacent tests passed; Black passed on changed tests; Ruff passed on touched files with only the two documented pre-existing Article_Extractor_Lib I001/F841 findings excluded; compileall and git diff checks passed; Bandit production scan reported 0 findings and 0 errors.
Follow-up Qodo re-review required per-test classification decorators rather than a module-level marker. Replaced the module marker with explicit `@pytest.mark.unit` on all 10 tests; the file's 10 tests, Black, Ruff, and diff checks passed.
All PR review threads were replied to and resolved. GitHub Actions remained queued with no observed failures; the requester explicitly instructed that GitHub CI checks be ignored, so they were not awaited as a completion gate. PR #2718 is ready for human review, subject to the repository's human-authored Change summary policy.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corrected JSON-LD extraction so summaries no longer count as page bodies, preserved structured summaries through successful fallback extraction and legacy no-LLM processing, and added pre-persistence body guards to enhanced and legacy ingestion. Canonical metadata envelopes are parsed with robust leading JSON boundaries and safely handle malformed or deeply nested input. Invalid articles are skipped with URL-scoped errors while valid siblings persist and the existing persist-ok response contract remains intact.
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
