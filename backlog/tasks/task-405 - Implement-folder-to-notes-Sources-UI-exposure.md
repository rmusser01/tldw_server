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
