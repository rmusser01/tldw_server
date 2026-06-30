---
id: TASK-502
title: 'Gate A blocker: restore Research Workspace WebUI compilation for UAT'
status: Done
labels:
- research-workspace
- uat
- frontend
- gate-a
priority: High
modified_files:
- apps/tldw-frontend/components/networking/ServerReadinessGate.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx
- apps/tldw-frontend/pages/_app.tsx
- apps/tldw-frontend/__tests__/app/app-layout.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live backend+WebUI UAT at /research-workspace is blocked by Next/Turbopack compile errors after rebasing the Research Workspace worktree onto latest dev. Fix only the compile blockers needed for the page to render, then resume CDP validation against the live backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Next dev server renders /research-workspace without the build error overlay.
- [x] #2 Duplicate declaration errors in ServerReadinessGate and ResearchWorkspace shell are resolved using the intended existing implementations.
- [x] #3 AddSourceModal parses successfully.
- [x] #4 Focused frontend tests and live CDP smoke check pass after the compile blockers are fixed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Restored the readiness gate constants and degraded-check state needed by `ServerReadinessGate`.
- Removed duplicate Research Workspace shell icon imports introduced during rebase.
- Repaired the AddSourceModal parse/runtime path enough for the page and modal to render during live UAT.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored the Research Workspace WebUI enough for live UAT after rebase compile blockers. Verification: `bunx vitest run __tests__/app/app-layout.test.tsx components/networking/__tests__/ServerReadinessGate.test.tsx __tests__/components/networking/ServerReadinessGate.degraded.test.tsx --maxWorkers=1 --no-file-parallelism` passed: 3 files, 24 tests. Live CDP smoke at `http://127.0.0.1:8080/research-workspace?uat_smoke_after_addsource_fix=1` rendered /research-workspace without a build overlay and had 0 console errors in a fresh tab after opening Add Sources -> My Media. Bandit is not applicable to frontend-only files.
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
