---
id: TASK-478.25
title: Gate G validate Research Workspace migration import-export recovery walkthrough
status: Done
labels:
- research-workspace
- migration
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 25
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-025 Partial gap by validating the Research Workspace migration import/export recovery walkthrough against a live backend and WebUI. Scope this slice to proving the user can understand and recover from eligible true-move, blocked inventory, ineligible server verification, and resumable recovery states without re-persisting deleted legacy content. Update RW-UAT-025 only as far as current live backend + WebUI + CDP/Playwright evidence supports. Do not add /workspace-playground aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migration runner failed states preserve enough recovery identity to resume or diagnose (`migrationId`, `manifestHash`, local inventory, no local deletion, no client-delete ack).
- [x] #2 Research Workspace exposes migration recovery details through the existing status surface without adding another banner, including retained/deleted surfaces, server eligibility, manifest hash, and retry guidance.
- [x] #3 Workspace import/export recovery accepts current Research Workspace bundles and intentionally supported legacy workspace export bundles as recovery inputs while current exports keep the `tldw.research-workspace.bundle` format.
- [x] #4 Live backend + WebUI + Playwright/CDP walkthrough validates the covered migration recovery/import/export paths and confirms `/workspace-playground` has no alias or redirect.
- [x] #5 RW-UAT-025 is updated only with verified evidence; remaining gaps stay marked Partial with explicit follow-up.
- [x] #6 Focused frontend tests and applicable backend/Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Plan: `Docs/superpowers/plans/2026-05-27-research-workspace-migration-recovery-walkthrough-plan.md`
- Added migration runner recovery metadata for failed API paths and a local-storage tombstone preflight so local content is not deleted unless the client can record the tombstone first.
- Added an existing-status-bar `Details` action that opens migration recovery details without adding a new banner.
- Allowed legacy `tldw.workspace-playground.bundle` imports as recovery inputs while keeping current exports on `tldw.research-workspace.bundle`.
- Updated `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` RW-UAT-025 with TASK-478.25 live CDP evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.25 completed the guided migration/import/export recovery slice. The migration runner now preserves diagnostic identity on failed API paths, tombstone writes are preflighted before local deletion, retained/blocked/deleted states expose recovery details from the existing status bar, and current plus supported legacy workspace export bundles import through the Research Workspace flow.

Verification:
- `bunx vitest run src/store/__tests__/workspace-migration.test.ts src/store/__tests__/workspace.test.ts src/components/Option/ResearchWorkspace/__tests__/WorkspaceStatusBar.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx` passed: 4 files, 97 tests.
- Live backend + WebUI + Playwright/CDP validated eligible true-move deletion, blocked unknown-inventory retention, current ZIP export/import, legacy JSON recovery import, and `/workspace-playground` 404 with no redirect. Screenshots: `/private/tmp/research-workspace-migration-eligible.png`, `/private/tmp/research-workspace-migration-blocked.png`, `/private/tmp/research-workspace-import-export.png`.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` is still blocked by the unrelated baseline error `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3): Type '"comfortable"' is not assignable to type 'GalleryCardDensity'.`
- Bandit was not run because this slice touched TypeScript/TSX tests, docs, and Backlog records only; no Python/backend files changed.
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
