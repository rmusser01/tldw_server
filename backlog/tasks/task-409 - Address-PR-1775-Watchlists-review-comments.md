---
id: TASK-409
title: Address PR 1775 Watchlists review comments
status: Done
assignee:
- Codex
labels:
- watchlists
- pr-review
- frontend
priority: High
modified_files:
- apps/extension/tests/e2e/watchlists.spec.ts
- apps/packages/ui/src/assets/locale/en/watchlists.json
- apps/packages/ui/src/components/Option/Watchlists/AlertsTab/AlertsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/ItemsTab/ItemsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobPreviewModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/ScopeSelector.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobPreviewModal.focus.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SettingsTab/SettingsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesBulkImport.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesBulkImport.preflight-commit.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplateEditor.tsx
- apps/packages/ui/src/public/_locales/en/watchlists.json
- apps/packages/ui/src/services/__tests__/watchlists-content-alerts.test.ts
- apps/packages/ui/src/services/watchlists.ts
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/DB_Management/Watchlists_DB.py
- tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sweep live review threads, CodeRabbit outside-diff comments, and failing PR checks for Watchlists PR #1775. Fix still-valid issues narrowly and verify before resolving threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All currently unresolved actionable PR #1775 review threads are fixed or explicitly determined non-actionable with evidence.
- [ ] #2 CodeRabbit outside-diff actionable findings are addressed where valid.
- [ ] #3 The failing Watchlists Extension E2E check root cause is fixed and locally verified where feasible.
- [ ] #4 Focused frontend tests and static checks pass for touched Watchlists files.
- [ ] #5 PR branch is pushed and PR review surface is refreshed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1775 review comments across Watchlists alerts, constrained pagination, async stale guards, report builder/evidence handling, settings loading feedback, source bulk import state invalidation, quick setup auto-open scoping, copy/locale fixes, and extension E2E drift. Added server-side alert search support and regression coverage. Verified focused UI tests, backend content-alert tests, Bandit, extension build, git diff whitespace, and strict Watchlists extension E2E with 13 passed / 0 skipped.
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
