---
id: TASK-349.3.3
title: Stage 6C Activity Reports and Templates constrained management
status: Done
dependencies:
- TASK-349.3.2
labels:
- watchlists
- stage6
- frontend
- reports
- activity
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace wide table-only Activity, Reports, Templates, run-detail item, and report-evidence surfaces with constrained list/detail patterns while preserving preview, evidence, download, regenerate, export, template edit/delete, and relationship-jump flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Activity/Runs constrained viewport renders run cards/list details instead of the wide table and preserves filters, export, cancel where available, detail open, and relationship jumps.
- [x] #2 Run detail drawer presents run items without horizontal table scrolling at 420x760.
- [x] #3 Reports constrained viewport renders report cards/list details instead of the wide table and preserves create, preview, evidence, download, regenerate, filters, delivery issue actions, and relationship jumps.
- [x] #4 Report evidence panel renders included evidence without horizontal table scrolling at 420x760.
- [x] #5 Templates constrained viewport renders template cards/list details and preserves create, edit/preview, delete safety, refresh, and format/version context.
- [x] #6 Focused Vitest coverage proves constrained Activity, Reports, Evidence, and Templates behavior plus existing Stage 5 report regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 6C implemented constrained-width management for Activity/Runs, Run detail scraped items, Reports/Outputs, report evidence, and Templates. Activity now renders run cards at constrained width with filters/export/refresh, detail open, report relationship jumps, cancel actions, status, timing, and run metrics while preserving desktop tables. Run detail shows scraped items as cards at constrained width and uses a full-width drawer wrapper there. Reports now render report cards at constrained width with create/refresh/filter controls, preview/download/regenerate, readiness, delivery status, evidence snapshot context, and monitor/run jumps. Report evidence renders included evidence as cards at constrained width. Templates now render constrained cards with create/refresh, edit/delete, format, version, history, and updated context while preserving desktop table behavior.

Verification:
- cd apps/packages/ui && bunx vitest run src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.extension-management.test.tsx src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.advanced-filters.test.tsx src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.source-column.test.tsx src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.extension-management.test.tsx src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.extension-management.test.tsx src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.extension-management.test.tsx src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.delete-safety.test.tsx --maxWorkers=1 --no-file-parallelism: 8 files, 29 tests passed. Existing run-detail error-path tests intentionally log failed fetches while asserting mapped error UI.
- cd apps/packages/ui && bun run test:watchlists:typecheck: 1 file, 3 tests passed.
- cd apps/packages/ui && node -e JSON locale parse for src/assets/locale/en/watchlists.json and src/public/_locales/en/watchlists.json: both ok.
- git diff --check: passed.
- Bandit: not applicable; touched files are frontend TypeScript/tests, locale JSON, docs, and Backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6C replaced table-only constrained Activity, Run detail items, Reports, report evidence, and Templates surfaces with card/list management views while preserving desktop tables and existing relationship/action flows. Focused Vitest coverage, the Watchlists static guard, locale JSON parsing, and whitespace checks passed. Bandit was not applicable for this frontend-only slice.
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
