---
id: TASK-405
title: Implement folder-to-notes Sources UI exposure
status: In Progress
labels:
- webui
- extension
- sources
- notes-sync
- implementation
documentation:
- Docs/superpowers/specs/2026-05-17-folder-notes-sources-ui-exposure-design.md
- Docs/superpowers/plans/2026-05-17-folder-notes-sources-ui-exposure-implementation-plan.md
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
