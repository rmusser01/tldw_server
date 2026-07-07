---
id: TASK-12167
title: Implement settings IA split route scaffold
status: Done
labels:
- frontend
- settings
- ux
- implementation
documentation:
- Docs/superpowers/specs/2026-07-06-settings-ia-recovery-preferences-ui-design.md
- Docs/superpowers/plans/2026-07-07-settings-ia-recovery-preferences-ui-implementation.md
modified_files:
- apps/packages/ui/src/assets/locale/ar/settings.json
- apps/packages/ui/src/assets/locale/da/settings.json
- apps/packages/ui/src/assets/locale/de/settings.json
- apps/packages/ui/src/assets/locale/en/settings.json
- apps/packages/ui/src/assets/locale/es/settings.json
- apps/packages/ui/src/assets/locale/fa/settings.json
- apps/packages/ui/src/assets/locale/fr/settings.json
- apps/packages/ui/src/assets/locale/it/settings.json
- apps/packages/ui/src/assets/locale/ja-JP/settings.json
- apps/packages/ui/src/assets/locale/ko/settings.json
- apps/packages/ui/src/assets/locale/ml/settings.json
- apps/packages/ui/src/assets/locale/no/settings.json
- apps/packages/ui/src/assets/locale/pt-BR/settings.json
- apps/packages/ui/src/assets/locale/ru/settings.json
- apps/packages/ui/src/assets/locale/sv/settings.json
- apps/packages/ui/src/assets/locale/uk/settings.json
- apps/packages/ui/src/assets/locale/zh-TW/settings.json
- apps/packages/ui/src/assets/locale/zh/settings.json
- apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx
- apps/packages/ui/src/components/Layouts/__tests__/settings-layout-active-route.test.ts
- apps/packages/ui/src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx
- apps/packages/ui/src/components/Layouts/__tests__/settings-nav.guardian.test.ts
- apps/packages/ui/src/components/Layouts/settings-active-route.ts
- apps/packages/ui/src/components/Layouts/settings-nav-config.ts
- apps/packages/ui/src/components/Layouts/settings-nav.ts
- apps/packages/ui/src/components/Option/Settings/QuickIngestSettings.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/GeneralSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/PreferencesSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/QuickIngestSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/SearchModeSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/UiCustomizationSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/general-settings.tsx
- apps/packages/ui/src/components/Option/Settings/preferences-settings.tsx
- apps/packages/ui/src/components/Option/Settings/search-mode.tsx
- apps/packages/ui/src/components/Option/Settings/setup-recovery-settings.tsx
- apps/packages/ui/src/components/Option/Settings/ui-customization.tsx
- apps/packages/ui/src/components/Sidepanel/Settings/__tests__/body.test.tsx
- apps/packages/ui/src/components/Sidepanel/Settings/body.tsx
- apps/packages/ui/src/routes/__tests__/option-settings-route-split.test.tsx
- apps/packages/ui/src/routes/option-settings-health.tsx
- apps/packages/ui/src/routes/option-settings-processed.tsx
- apps/packages/ui/src/routes/option-settings-route-registry.tsx
- apps/tldw-frontend/e2e/page-mapping.ts
- apps/tldw-frontend/e2e/smoke/page-inventory.ts
- apps/tldw-frontend/pages/settings/data.tsx
- apps/tldw-frontend/pages/settings/index.tsx
- apps/tldw-frontend/pages/settings/preferences.tsx
- backlog/tasks/task-12167 - Implement-settings-IA-split-route-scaffold.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first reviewable slice of the approved settings IA plan: make /settings a Setup & Recovery page, add /settings/preferences, keep the legacy GeneralSettings export compatible, and add focused TDD coverage for the route split.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-07-settings-ia-recovery-preferences-ui-implementation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: focused Vitest suite passed (13 files, 43 tests); locale settings JSON parsed successfully; git diff --check passed; Playwright route sweep passed for /settings, /settings/preferences, /settings/ui, and /settings/data at 1440x1000 and 390x844 with expected headings, active nav state, no horizontal overflow, and updated screenshots under /private/tmp/settings-ia-qa. Typecheck was run and still fails only in pre-existing untouched baseline files: AudioStudio/TimelineEditor.tsx, ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx, Skills/Manager.tsx, scheduled-tasks-control-plane.ts, tldw/mcp-hub.ts, tldw/voice-cloning.ts, e2e/fixtures/knowledge-qa-live.ts, and e2e/workflows/tier-2-features/flashcards.spec.ts. Bandit not applicable: frontend TypeScript/JSON/Backlog-only changes.
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
