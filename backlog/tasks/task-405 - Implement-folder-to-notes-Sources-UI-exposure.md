---
id: TASK-405
title: Implement folder-to-notes Sources UI exposure
status: Done
labels:
- webui
- extension
- sources
- notes-sync
- implementation
documentation:
- Docs/superpowers/specs/2026-05-17-folder-notes-sources-ui-exposure-design.md
- Docs/superpowers/plans/2026-05-17-folder-notes-sources-ui-exposure-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-folder-notes-sources-ui-exposure-implementation-plan.md
- tldw_Server_API/Config_Files/config.txt
- tldw_Server_API/tests/Config/test_config_txt_inline_comment_regression.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved plan for exposing local-directory-to-notes Sources sync from Notes and shortcut surfaces, including backend local-directory entitlement, authenticated capability plumbing, SourceForm preset and schedule switch, Notes Sync folder entry point, real Alt+2 Sources navigation, shortcut/help/header discoverability, tests, Bandit, and browser QA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-folder-notes-sources-ui-exposure-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 backend entitlement/enforcement completed and reviewed. Commits: cebe876df initial backend gate, 87b4fac5 request-state org scope/no-op patch/capability test fixes, 92003c4f authorization-before-path-validation/sink_type/response model fixes, 15f3f7a3 normalized-equivalent config patch fix. Spec review approved and code-quality review approved with only non-blocking minor notes. Focused tests reported passing: 30 access-policy unit/integration tests, 16 existing ingestion source API tests, Bandit touched backend source with 0 findings, git diff --check clean.
Task 2 frontend capability contract implemented locally after stopping a timed-out partial worker. Added canCreateLocalDirectoryIngestionSource to ServerCapabilities, fetched authenticated /api/v1/ingestion-sources/capabilities only when generic ingestion sources are available, kept endpoint failure non-fatal, scoped the capabilities cache with buildChatSurfaceScopeKeyFromConfig, and bumped persisted cache storage to V4. Verification: from apps/packages/ui, bunx vitest run src/services/__tests__/server-capabilities.test.ts -> 33 passed; git diff --check passed.
Task 2 quality review follow-up: gated the authenticated source entitlement fetch to authoritative OpenAPI discovery so bundled fallback Sources routes do not probe older or misconfigured servers. Added fallback regression coverage asserting local-directory creation remains disabled and `/api/v1/ingestion-sources/capabilities` is not called on fallback discovery. Verification: from `apps/packages/ui`, `bunx vitest run src/services/__tests__/server-capabilities.test.ts` passed with 33 tests; `git diff --check` passed.
Task 2 spec re-review follow-up: tightened the entitlement fetch gate to require the authoritative OpenAPI spec to advertise `/api/v1/ingestion-sources/capabilities`, not just generic Sources routes. Added regression coverage for older authoritative Sources servers that lack the entitlement route. Verification: from `apps/packages/ui`, `bunx vitest run src/services/__tests__/server-capabilities.test.ts` passed with 34 tests; `git diff --check` passed.
Task 2 frontend capability contract passed both review gates after follow-ups. Spec review approved after the older-authoritative-server regression; code-quality review approved with no blocking or important issues. Verification remains `bunx vitest run src/services/__tests__/server-capabilities.test.ts` from `apps/packages/ui` with 34 tests passing and `git diff --check` clean.
Task 3 Sources form preset implemented. Added `buildSourcesNewPath({ preset: "notes-folder-sync" })`, `/sources/new` query-param preset handling, explicit SourceForm preset defaults for local-directory-to-Notes canonical sync, visible `Scheduled rescans` switch, and local-directory create submit gating through `useServerCapabilities`. Archive snapshot and git repository creation remain available when local-directory creation is disabled. Verification: from `apps/packages/ui`, `bunx vitest run src/routes/__tests__/route-paths.sources.test.ts src/components/Option/Sources/__tests__/SourceForm.test.tsx` passed with 15 tests. Bandit skipped for this frontend-only task.
Task 3 Sources preset and schedule switch completed in commit c17c846d78aef58f4a0c8ffe95473fbc06c0d327 and passed both review gates. Verification: from `apps/packages/ui`, `bunx vitest run src/routes/__tests__/route-paths.sources.test.ts src/components/Option/Sources/__tests__/SourceForm.test.tsx` passed with 15 tests; `git diff --check` passed. A route-guard spot-check outside this task still has an existing Router-context failure and is tracked as outside Task 3 scope.
Task 4 Notes entry point implemented. Added the small `Sync folder` Notes list action beside Import, gated it by online state, active Notes view, capability loading, Notes support, Sources support, and local-directory entitlement, with disabled details kept in tooltip/title copy. Threaded `onSyncFolder` through `NotesSidebar` and routed `NotesManagerPage` clicks to `/sources/new?preset=notes-folder-sync` via `buildSourcesNewPath`. Verification: from `apps/packages/ui`, `bunx vitest run src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx` passed with 6 tests. Bandit skipped for this frontend-only task.
Task 4 Notes Sync folder entry point completed in commit 4620162e880fe4ce401aca38d2ccade8817e0623 and passed both review gates. Verification: from `apps/packages/ui`, `bunx vitest run src/components/Notes/__tests__/NotesListPanel.sources-sync.test.tsx src/components/Notes/__tests__/NotesManagerPage.sources-sync-entry.test.tsx` passed with 6 tests; `git diff --check` passed. Bandit skipped for frontend-only Task 4 slice.
Task 5 real Sources navigation shortcut implemented. Added `modeSources` as `Alt+2`, merged legacy shortcut configs over defaults, preserved unknown persisted shortcut keys during individual update/reset, added shared mode navigation shortcuts with route-transition start calls, and initialized the hook in extension/options and WebUI layouts. Verification: from `apps/packages/ui`, `bunx vitest run src/hooks/keyboard/__tests__/useShortcutConfig.test.ts src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx` passed with 6 tests; `git diff --check` passed. Bandit skipped for frontend-only Task 5 slice.
Task 5 quality review follow-up: loosened the persisted shortcut config storage type so `defaultShortcuts` remains assignable, and added `modeSources` to the existing typed CommandPalette shortcut fixture. Verification: from `apps/packages/ui`, `bunx vitest run src/hooks/keyboard/__tests__/useShortcutConfig.test.ts src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx` passed with 10 tests; `git diff --check` passed.
Task 5 real Sources navigation shortcut completed across commits e1f445442664da3c255b626c47fbf7c318b531e9 and 271ee95a2, and passed both review gates after the typed-storage and CommandPalette fixture follow-up. Verification: from `apps/packages/ui`, `bunx vitest run src/hooks/keyboard/__tests__/useShortcutConfig.test.ts src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx` passed with 10 tests; `git diff --check` passed. Bandit skipped for frontend-only Task 5 slice.
Task 6 shortcut modal and header launcher discoverability implemented. Added Sources to Keyboard Shortcuts and Page Help navigation rows, exposed Sources in default header shortcuts with legacy full-default migration only, and added the Sources library launcher item. Verification: from `apps/packages/ui`, `bunx vitest run src/components/Common/__tests__/KeyboardShortcutsModal.focus.test.tsx src/components/Common/__tests__/PageHelpModal.shortcuts.test.tsx src/services/__tests__/ui-settings.header-shortcuts.test.ts src/components/Layouts/__tests__/HeaderShortcuts.test.tsx src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts` passed with 41 tests. Bandit skipped for frontend-only Task 6 slice.
Task 7 final verification/browser QA completed. Browser QA found the default config did not force-enable the experimental ingestion-sources router, causing the fresh Sources UI to fall back to "This server does not advertise ingestion source support." Added a config regression and enabled `ingestion-sources` in `tldw_Server_API/Config_Files/config.txt`. Verification recorded: config regression red/green; backend-focused suite 33 passed; frontend-focused Vitest suite 100 passed; Bandit touched backend source returned 0 findings; git diff --check clean; browser QA verified the source preset form, Notes Sync folder navigation, Alt+5 Notes navigation, shortcut modal Sources row, header launcher Sources row, and mobile-width Source/Notes drawer rendering.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WebUI/extension exposure for folder-to-notes Sources sync. The backend now gates local-directory ingestion sources by single-user mode or the admin feature flag `ingestion_sources.local_directory`, exposes an authenticated current-user capability endpoint, and enforces entitlement before local-directory create/retarget operations. The shared UI now consumes that capability, adds the notes-folder-sync Source preset with visible scheduled-rescan control, exposes a Notes `Sync folder` action that routes to the preset, wires real mode-navigation shortcuts including `Alt+2` for Sources, and surfaces Sources in shortcut/help/header launcher surfaces. Final follow-up enabled the ingestion-sources router in the default config so the exposed UI is reachable in the default local WebUI setup, with regression coverage.
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
