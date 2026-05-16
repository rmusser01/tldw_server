---
id: TASK-349.2.5
title: Stage 5E Watchlist report presets docs and real-server QA
status: Done
assignee: []
created_date: '2026-05-15 21:40'
updated_date: '2026-05-16 04:21'
labels:
  - watchlists
  - stage5
  - docs
  - qa
dependencies:
  - TASK-349.2.4
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
  - Docs/API-related/Watchlists_API.md
  - Docs/Published/API-related/Watchlists_API.md
parent_task_id: TASK-349.2
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close out Stage 5 by adding or updating report presets, documenting the API and user-facing evidence/readiness contract, running focused backend/frontend verification, running Bandit on touched backend code, and completing a real-server CDP smoke through /watchlists without server mocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CTI/OSINT and news report presets render evidence/readiness/source context from the Stage 5 output context while preserving Markdown/HTML/Chatbook/audio compatibility.
- [x] #2 API docs and published docs describe Stage 5 output fields, evidence/readiness endpoints, metadata, warning codes, legacy behavior, and CTI/news examples.
- [x] #3 Focused backend and frontend tests for Stage 5 plus existing output/triage regressions pass or any failures are documented with blockers.
- [x] #4 Bandit is run on touched backend scope and no new findings are introduced.
- [x] #5 Real FastAPI plus real WebUI CDP smoke covers CTI and news report creation, evidence inspection, preview/download, and constrained viewport management with screenshots and notes recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 5E after Stage 5D commit 46bb8919b. Scope: report presets/templates, API docs parity, focused backend/frontend verification, Bandit for backend touched scope, and real-server CDP smoke through /watchlists without server mocks.

Implemented built-in `cti_osint_report_markdown` and `news_briefing_markdown` presets in `tldw_Server_API/app/core/Watchlists/template_store.py`. Added focused preset coverage that verifies readiness, source diversity, alert evidence, excluded trails, and follow-up links render from Stage 5 report context.

Updated `Docs/API-related/Watchlists_API.md` and `Docs/Published/API-related/Watchlists_API.md` with Stage 5 output creation fields, report presets, `GET /outputs/{output_id}/evidence`, `GET /outputs/{output_id}/readiness`, metadata fields, readiness warning codes, legacy live-only output behavior, and CTI/news examples.

Fixed a real-server UI issue observed during CDP QA: the report builder showed selected included items but rendered readiness counts as `0` because the i18n mock/default formatting path mishandled option-object defaults. `ReportBuilderDrawer.tsx` now uses deterministic local count labels for queued, source, and excluded counts; focused Vitest coverage asserts `1 source` and `1 update not queued`.

Verification recorded:

- Backend focused pytest: `22 passed, 5 warnings` for built-in templates, report evidence, report API, output provenance, and triage API coverage.
- Frontend focused Vitest: `7` files and `30` tests passed for report service, metadata, builder drawer, evidence panel, defensible reports, outputs smoke, and preview focus management.
- `git diff --check`: passed.
- API docs mirror check: `cmp -s Docs/API-related/Watchlists_API.md Docs/Published/API-related/Watchlists_API.md` passed.
- Bandit touched backend scope: exit `0`, no findings in touched scope; metrics reported zero high/medium/low findings.

Real-server CDP QA recorded:

- Real FastAPI: `http://127.0.0.1:18102`.
- Real Next WebUI: `http://127.0.0.1:18180`.
- Verified route: `/watchlists?view=all&tab=outputs`.
- Result artifact: `/private/tmp/tldw-watchlists-stage5-18102/cdp-results.json` with `ok: true`, no page errors, and no request failures.
- Screenshots:
  - `/private/tmp/tldw-watchlists-stage5-18102/watchlists-desktop-loaded.png`
  - `/private/tmp/tldw-watchlists-stage5-18102/watchlists-cti-evidence-preview.png`
  - `/private/tmp/tldw-watchlists-stage5-18102/watchlists-constrained-reports.png`
  - `/private/tmp/tldw-watchlists-stage5-18102/watchlists-constrained-report-builder.png`
  - `/private/tmp/tldw-watchlists-stage5-18102/watchlists-news-evidence-preview-constrained.png`
- Download artifact from real API response after UI download click: `/private/tmp/tldw-watchlists-stage5-18102/cti-report-20260516025818.md`.

Known QA notes:

- No mocked server was used. Report creation, evidence preview, readiness, and download verification ran through the real WebUI and real FastAPI.
- Seed data was created through real API objects where public APIs exist. Deterministic run/items/alerts were inserted directly into the server-owned Watchlists SQLite DB because there is no public create-item API for QA setup.
- The live server used the worktree user database path for Watchlists despite the temporary `USER_DB_BASE_DIR` override. This is a QA isolation/config-precedence note, not a mock.
- Rapid repeated CDP attempts hit the real in-memory rate limit during debugging; the final smoke used a restarted real API instance.
- Non-escalated API bind attempts were sandbox-blocked on `127.0.0.1:18102`, so the final API server was started with approved escalation.
- The CDP harness verified the real `/api/v1/watchlists/outputs/{id}/download` response because the UI download path uses a Blob/object URL.
- The CDP harness attempted the Ant Select UI first, then used the page-exposed Zustand store only after the Ant dropdown portal became unstable after drawer/download interactions; it still verified the selected Watchlist heading after the switch.
- Observed console warnings were existing Ant Design warnings: `Modal destroyOnClose`, `Alert message`, and static `message` context. No page errors or request failures were recorded.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
