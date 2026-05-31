---
id: TASK-45.44.13.4
title: Migrate LocaleJsonDiagnostics alert to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.13
references:
- apps/packages/ui/src/components/Common/LocaleJsonDiagnostics.tsx
- apps/packages/ui/src/components/Common/__tests__/LocaleJsonDiagnostics.design-system.test.tsx
- apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/PlaylistPreflightPanel.test.tsx
- apps/packages/ui/src/components/Option/Sources/SourceForm.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1823
- apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx
- apps/packages/ui/src/assets/locale/en/sources.json
documentation:
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
modified_files:
- apps/packages/ui/src/components/Common/LocaleJsonDiagnostics.tsx
- apps/packages/ui/src/components/Common/__tests__/LocaleJsonDiagnostics.design-system.test.tsx
- apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/PlaylistPreflightPanel.test.tsx
- apps/packages/ui/src/components/Option/Sources/SourceForm.tsx
- apps/packages/ui/src/components/Option/Sources/__tests__/SourceForm.test.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
- apps/packages/ui/src/assets/locale/en/sources.json
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace LocaleJsonDiagnostics' remaining AntD Alert product-state usage with the shared design-system Alert primitive while preserving dev-only locale JSON error diagnostics and line/column details. Current dev also had product-state verifier drift in PlaylistPreflightPanel, SourceForm, and WorkspacePlayground ChatPane; resolve that drift in this slice so the verifier can pass without adding new baseline debt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LocaleJsonDiagnostics renders locale JSON parse failures through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused coverage proves the dev diagnostics banner uses data-ds-component="Alert" and retains error details.
- [x] #3 The design-system product-state baseline no longer contains the LocaleJsonDiagnostics AntD Alert exception or migrated SourceForm stale Alert exceptions.
- [x] #4 Focused tests, design-system verifier, git diff check, and TypeScript/Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented via TDD for LocaleJsonDiagnostics and PlaylistPreflightPanel.

Added a focused LocaleJsonDiagnostics panel regression, migrated the dev-only locale parse error banner from AntD Alert to the shared design-system Alert, and removed its baseline exception.

During verifier proof, current dev had product-state drift outside LocaleJsonDiagnostics: PlaylistPreflightPanel AntD Alert/Tag findings, SourceForm stale/new Alert findings, and WorkspacePlayground ChatPane Ready/Degraded canonical labels. Resolved that drift by migrating PlaylistPreflightPanel state indicators to Alert/Badge, migrating SourceForm product-state alerts to the design-system Alert, routing ChatPane Ready/Degraded fallbacks through the state registry exports, and removing the resolved SourceForm stale baseline entries instead of adding new debt.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated LocaleJsonDiagnostics to the shared design-system Alert and resolved current-dev product-state verifier drift in PlaylistPreflightPanel, SourceForm, and WorkspacePlayground ChatPane. PR opened: https://github.com/rmusser01/tldw_server/pull/1823. Follow-up review comments addressed duplicate LocaleJsonDiagnostics keys, PlaylistPreflightPanel Alert title usage, and SourceForm i18n for the changed locked/source-sync alert strings. Verification: focused red tests failed for all three review findings before the fixes; focused LocaleJsonDiagnostics + PlaylistPreflightPanel + SourceForm + product-state guard + sources locale Vitest passed (73 tests); bun run verify:design-system-state passed with 399 remaining allowed baseline exceptions; touched JSON parse passed; git diff --check passed; bunx tsc --noEmit still exits 2 on existing package-wide type debt, with no touched-file matches in /tmp/ds-locale-tsc.log. Bandit is not applicable because this slice touched frontend TypeScript/JSON/backlog files only.
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
