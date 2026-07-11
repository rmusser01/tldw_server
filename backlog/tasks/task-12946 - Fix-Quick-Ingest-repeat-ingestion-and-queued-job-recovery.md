---
id: TASK-12946
title: Fix Quick Ingest repeat ingestion and queued job recovery
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-11 07:53'
labels: []
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-10-quick-ingest-pr-2709-review-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate repeated user reports of WebUI/browser-extension Quick Ingest failures: `Maximum update depth exceeded` after repeated YouTube/web ingestion and backend ingest jobs appearing queued at 0%. Require real end-to-end user acceptance walkthroughs before relying on automated E2E coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeat Quick Ingest submissions complete without Maximum update depth errors and classify existing media as skipped.
- [ ] #2 Restored direct ingest jobs survive transient status-read failures and terminate correctly for permanent missing jobs.
- [ ] #3 Webpack dev watch ignores preserve existing patterns and match backend runtime directories using absolute normalized paths.
- [ ] #4 Stale yt-dlp installations produce actionable diagnostics and current installations continue normally.
- [ ] #5 Touched persistence logs redact user-controlled URL secrets.
- [ ] #6 Focused automated verification, Bandit, and full PDF, local-link, and YouTube Shorts UAT pass outside the sandbox.
- [ ] #7 All actionable PR review threads are resolved and the branch is current with dev.
- [ ] #8 Existing perform_analysis and summarize_checkbox declarations govern request-level LLM extraction without a new public API field, while non-API scraper consumers retain the established default order.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-10-quick-ingest-pr-2709-review-remediation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-10 follow-up UAT isolated the repeat YouTube Shorts failure to stale yt-dlp in the project venv. Before updating, yt-dlp 2025.8.11 quarantined `https://www.youtube.com/shorts/6-rf_YXDpPg` at 20% with "content is not available on this app." After updating the venv to yt-dlp 2026.7.4, the same Quick Ingest browser flow completed the YouTube job at 100% and added media id 5. Raised `pyproject.toml` yt-dlp floor to `>=2026.7.4` so fresh installs pick up the extractor fix.

Draft PR: https://github.com/rmusser01/tldw_server/pull/2709
2026-07-10: Rebasing review follow-up completed cleanly onto current origin/dev. Approved remediation design committed at Docs/superpowers/specs/2026-07-10-quick-ingest-pr-2709-review-remediation-design.md; implementation remains pending spec/plan gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-10: Corrected the remediation design after tracing both public web ingestion contracts. No new strategy request field will be added; existing perform_analysis/summarize_checkbox intent will be propagated to the internal extraction pipeline.

Task 1 complete at 7be983885e: Webpack watch ignores now preserve valid existing semantics in schema-valid shapes and append absolute normalized backend runtime roots. Outside-sandbox focused Vitest passed 7/7; spec and quality reviews approved.

Task 2 complete at 853f6a5a77: restored direct-job reads retry status-less/network/status-0, 408, 429, and 5xx failures up to three attempts while permanent/malformed responses interrupt immediately. Outside-sandbox focused Vitest passed 17/17 and adjacent session/store suites passed 45/45; spec and quality reviews approved.

Task 3 complete at 02da73990a: persistence uses structured extractor signals and exact repository messages, repeated real repository writes classify as skipped duplicates, null-ID/storage failures are accounted, touched URL/exception logs are sanitized, and frontend terminal status uses one typed helper with duplicate precedence. Outside-sandbox backend pytest passed 15/15, frontend Vitest 39/39, Bandit 0 findings; reviews approved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Investigated with real WebUI walkthroughs before relying on E2E. Root causes fixed: repeat/duplicate web scrape responses were returned as HTTP 200 with zero stored articles and not classified as skipped/failure consistently; Quick Ingest direct scrape handling treated some error payloads as success; default web scraping extraction implicitly invoked LLM analysis without an explicit provider; Next dev mode watched backend runtime DB/log writes and could remount the UI; AntD Modal portal styling in QuickIngestWizardModal still triggered the reported Maximum update depth crash on repeat duplicate URL submissions; restored queued-job polling did not force the direct backend path and could leave refresh/reopen flows stuck at 0%; stale yt-dlp 2025.8.11 caused current YouTube/Shorts extractor failures that disappeared after updating to yt-dlp 2026.7.4. Added regression coverage for duplicate persistence, extraction method defaults, batch result classification, restored job polling, modal session stability, dev watcher config, and Quick Ingest E2E helper timing/isolation. Verification was run outside the sandbox: manual UAT covered duplicate URLs, mixed file+URL ingest, backend job status progressing to completed/100%, and PDF + local link + YouTube Shorts ingestion after the yt-dlp update; focused Vitest passed 75 tests across 6 files; focused backend pytest passed 11 tests; Bandit on touched backend code reported 0 findings; targeted Playwright Quick Ingest/media ingest set passed 14 tests.
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
