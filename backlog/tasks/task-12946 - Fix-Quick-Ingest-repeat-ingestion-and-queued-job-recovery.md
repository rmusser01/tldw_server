---
id: TASK-12946
title: Fix Quick Ingest repeat ingestion and queued job recovery
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-11 16:23'
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
- [x] #1 Repeat Quick Ingest submissions complete without Maximum update depth errors and classify existing media as skipped.
- [x] #2 Restored direct ingest jobs survive transient status-read failures and terminate correctly for permanent missing jobs.
- [x] #3 Webpack dev watch ignores preserve existing patterns and match backend runtime directories using absolute normalized paths.
- [x] #4 Stale yt-dlp installations produce actionable diagnostics and current installations continue normally.
- [x] #5 Touched persistence logs redact user-controlled URL secrets.
- [x] #6 Focused automated verification, Bandit, and full PDF, local-link, and YouTube Shorts UAT pass outside the sandbox.
- [ ] #7 All actionable PR review threads are resolved and the branch is current with dev.
- [x] #8 Existing perform_analysis and summarize_checkbox declarations govern request-level LLM extraction without a new public API field, while non-API scraper consumers retain the established default order.
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

Task 4 complete at a72449d733: restored llm in the global extraction default and explicitly propagated existing perform_analysis/summarize_checkbox intent through enhanced, queued, crawl, legacy, and friendly-ingest paths via internal default-true arguments only. Async enhanced extraction is offloaded from the event loop. No public schema/request field changed. Outside-sandbox 115 tests passed and Bandit reported 0 findings; reviews approved.

Task 5 complete at f4e6e81c33: added a thread-safe, exactly-once, nonblocking yt-dlp floor diagnostic at all seven validated Video_DL boundaries plus existing-environment update docs. Lazy version-helper imports cannot break ingestion. Outside-sandbox focused pytest passed 24/24, adjacent video tests 33/33, Bandit 0 findings; reviews approved.

2026-07-11 renewed host-side UAT: a clean PDF run completed, but every same-mounted-session second submission reproduced Maximum update depth in @rc-component/portal before the link result rendered. Three isolated attempts disproved getContainer and stale-tracking-only fixes. Proven findings retained: returning a session to draft must clear old direct job tracking; the media worker's generic exponential idle backoff reached a 30-second ceiling (observed queued for 28 seconds before claim), so its user-facing default ceiling is reduced to 2 seconds while preserving the env override. Remaining proposed architecture change requires approval: Ingest More should call the existing replaceWithNewDraft() so the provider key/session/run refs remount instead of reusing the completed provider. Full PDF/link/YouTube repeat UAT is not yet passing; prior final summary cleared as stale.

2026-07-11 approved architecture fix and fresh host-side UAT: Ingest More now calls the existing replaceWithNewDraft path, creating a new session/provider key instead of reusing completed reducer and run refs. Draft upserts also clear stale direct-job tracking. One mounted WebUI walkthrough passed PDF, RFC 9110 link, repeated RFC link, exact YouTube Short https://www.youtube.com/shorts/6-rf_YXDpPg, and repeated YouTube Short. First submissions succeeded; repeats were classified as skipped existing. pageErrors and consoleErrors were empty. PDF and YouTube jobs began within one second of creation, confirming the media worker 2-second default idle-backoff ceiling removes the observed 28-second queued-at-0-percent delay. Jobs DB: three completed 100-percent jobs with Success, Success, Skipped. Media DB: exactly three rows for PDF, RFC link, and YouTube Short. Evidence: /tmp/task12946_quick_ingest_uat_10_evidence.json and /tmp/task12946_quick_ingest_uat_10_final.png. Focused UI Vitest passed 46/46, media worker pytest passed 19/19, extension TypeScript compile passed, and Bandit reported zero findings. The focused extension Playwright test could not be completed locally after three launch attempts: two headless attempts exposed no MV3 targets and the required headful attempt timed out; repository CI uses xvfb with TLDW_E2E_EXTENSION_HEADLESS=0, so this test remains for CI verification after push.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
