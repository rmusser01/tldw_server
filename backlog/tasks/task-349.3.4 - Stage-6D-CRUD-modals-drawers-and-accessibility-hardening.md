---
id: TASK-349.3.4
title: Stage 6D CRUD modals drawers and accessibility hardening
status: Done
dependencies:
- TASK-349.3.3
labels:
- watchlists
- stage6
- frontend
- accessibility
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden constrained CRUD modals, drawers, action focus, and keyboard behavior for Watchlists source forms, OPML import, monitor forms/previews, template editor, settings drawer, and cross-tab navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Source form, OPML import, monitor form, monitor preview, template editor, and settings drawer primary actions remain visible and usable at constrained width.
- [x] #2 Modal/drawer footers do not clip at 420x760, and dense editors use full-width or stacked layouts where needed.
- [x] #3 Keyboard navigation reaches constrained navigation, create/edit/delete actions, and primary drawer/modal actions with accessible names.
- [x] #4 Escape/cancel closes constrained drawers/modals without leaving focus trapped in removed nodes.
- [x] #5 Focused Vitest coverage records modal/drawer width, accessibility, and keyboard regression behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 6D added a shared Watchlists constrained modal chrome helper and applied it to Source form, OPML import, Monitor form, Monitor preview, and Template editor modals. Nested OPML preflight/failure content, Monitor preview candidates, and Settings claim-cluster subscriptions now render as constrained cards/lists instead of table-only content at extension width. Existing focus restoration paths remain in place for Source form, Monitor form, and Monitor preview, while the broader Watchlists navigation, Items accessibility, and keyboard shortcut baselines continue to pass.

Verification:
- cd apps/packages/ui && bunx vitest run src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.accessibility-baseline.test.tsx src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.keyboard-shortcuts.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesBulkImport.preflight-commit.test.tsx src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx src/components/Option/Watchlists/JobsTab/__tests__/JobPreviewModal.focus.test.tsx src/components/Option/Watchlists/TemplatesTab/__tests__/TemplateEditor.mode-contract.test.tsx src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx --maxWorkers=1 --no-file-parallelism: 9 files, 67 tests passed. Existing error-path tests intentionally log failed feed/monitor errors while asserting mapped remediation UI.
- cd apps/packages/ui && bun run test:watchlists:typecheck: 1 file, 3 tests passed.
- git diff --check: passed.
- Bandit: not applicable; touched files are frontend TypeScript/tests, documentation, and Backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6D hardened constrained CRUD and supporting management surfaces without adding a parallel mobile route. Source, OPML import, Monitor, Monitor preview, and Template editor modals now use full-width constrained Ant Design modal chrome, nested preflight/preview/settings tables have constrained list alternatives, and focused accessibility/keyboard regression tests pass.
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
