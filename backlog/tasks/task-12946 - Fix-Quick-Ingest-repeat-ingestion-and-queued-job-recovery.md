---
id: TASK-12946
title: Fix Quick Ingest repeat ingestion and queued job recovery
status: In Progress
modified_files:
- apps/extension/tests/e2e/live-ux-workflows.spec.ts
- apps/extension/tests/e2e/quick-ingest-cancel.spec.ts
- apps/extension/tests/e2e/quick-ingest-ux-audit.spec.ts
- apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
- apps/packages/ui/src/services/tldw/ingest-job-results.ts
- apps/packages/ui/src/services/tldw/quick-ingest-batch.ts
- apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts
- apps/packages/ui/src/services/__tests__/ingest-job-results.test.ts
- apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
- apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts
- apps/tldw-frontend/package.json
- apps/tldw-frontend/next.config.mjs
- apps/tldw-frontend/__tests__/frontend-dev-config.test.ts
- apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts
- apps/tldw-frontend/e2e/onboarding-uat/helpers.ts
- apps/tldw-frontend/e2e/utils/journey-helpers.ts
- apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts
- Docs/Getting_Started/TROUBLESHOOTING.md
- Docs/Published/Getting_Started/TROUBLESHOOTING.md
- pyproject.toml
- tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py
- tldw_Server_API/app/services/enhanced_web_scraping_service.py
- tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py
- tldw_Server_API/tests/WebScraping/test_extraction_pipeline_router.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate repeated user reports of WebUI/browser-extension Quick Ingest failures: `Maximum update depth exceeded` after repeated YouTube/web ingestion and backend ingest jobs appearing queued at 0%. Require real end-to-end user acceptance walkthroughs before relying on automated E2E coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-10 follow-up UAT isolated the repeat YouTube Shorts failure to stale yt-dlp in the project venv. Before updating, yt-dlp 2025.8.11 quarantined `https://www.youtube.com/shorts/6-rf_YXDpPg` at 20% with "content is not available on this app." After updating the venv to yt-dlp 2026.7.4, the same Quick Ingest browser flow completed the YouTube job at 100% and added media id 5. Raised `pyproject.toml` yt-dlp floor to `>=2026.7.4` so fresh installs pick up the extractor fix.

Draft PR: https://github.com/rmusser01/tldw_server/pull/2709
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
