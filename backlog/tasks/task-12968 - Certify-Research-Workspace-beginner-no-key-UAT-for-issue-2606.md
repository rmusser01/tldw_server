---
id: TASK-12968
title: Certify Research Workspace beginner no-key UAT for issue 2606
status: Done
labels:
- research-workspace
- uat
- beginner
- no-key
- cdp
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/2606
- TASK-12020.13
- TASK-12020.28
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
documentation:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
modified_files:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage9.error.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage1.onboarding.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/store/__tests__/workspace.split-storage.test.ts
- apps/packages/ui/src/store/workspace-slices/workspace-list-slice.ts
- apps/packages/ui/src/store/workspace.ts
- apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
- apps/tldw-frontend/components/layout/WebLayout.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete GitHub issue #2606 by certifying the beginner/no-key `/research-workspace` persona against a real FastAPI backend and Next.js WebUI through a clean CDP-controlled browser session. Verify direct entry, readiness and empty states, tour/replay behavior, workspace search, Add Sources authentication recovery, mobile layout, and absence of unexpected global backend/runtime overlays. Update the live UAT matrix with evidence and split any remaining reproducible defect into a focused follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run RW-UAT-027 from clean browser storage against a real backend and WebUI using CDP, recording backend, auth, viewport, and browser-state assumptions.
- [x] #2 Verify direct entry, readiness gate, empty states, first-run tour and replay, workspace search, Add Sources no-key recovery, mobile layout, and no unexpected global backend/runtime modal.
- [x] #3 Capture reproducible screenshots plus console/network evidence and classify any non-pass as product defect, prerequisite, or environment limitation.
- [x] #4 Update the live Research Workspace UAT matrix and create a focused follow-up task for any remaining product gap with exact reproduction steps.
- [x] #5 Run existing focused beginner/no-key tests and repository hygiene checks; document Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit the current RW-UAT-027 contract, existing runner coverage, auth bootstrap, and browser prerequisites. 2. Start isolated real backend and WebUI instances and execute the clean beginner/no-key desktop and mobile CDP walkthrough with evidence capture. 3. Diagnose and fix only reproducible product failures using focused tests, or split blocked residuals into explicit follow-ups. 4. Update the UAT matrix and task record, run focused verification, and prepare a reviewable change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Completed the beginner/no-key certification against an isolated live stack: FastAPI single-user backend on 127.0.0.1:18160 with a server-only key and disposable SQLite/user databases; advanced-mode Next.js WebUI on 127.0.0.1:18161 with NEXT_PUBLIC_X_API_KEY and NEXT_PUBLIC_API_BEARER explicitly empty; fresh Chrome 145 profile on CDP 127.0.0.1:18162. The task-owned CDP runner passed 17/17 desktop/mobile checkpoints. Its final manifest durably records exactly one active/saved workspace, no visible migration status, and zero migration API calls at desktop entry/final and mobile entry. Diagnostics record zero page errors, request failures, unexpected HTTP errors, credential-bearing requests, migration requests, global backend dialogs, or runtime overlays. Expected no-key warnings/guards and one Next development HMR warning remained scoped.

Product fixes: always mount the shared TutorialRunner when global chrome is hidden; suppress the initial empty workspace persistence write in both split and monolithic modes; defer first-workspace initialization safely under StrictMode; scope fresh-initialization migration suppression to the exact workspace ID; replace the persistent tour-start notice with transient message feedback; compact the mobile header into bounded context/action rows; and update the stale AddSource partial-success regression to the current persistent row contract. Independent review found and the task fixed the monolithic empty-write gap, global marker overreach, and insufficient durable UAT assertions.

Verification: focused shared UI suite passed 9 files / 164 tests; WebLayout suite passed 14/14; maintained real-backend UAT entry evidence passed 1/1 in 28.4s; targeted ESLint exited 0 with no errors (remaining warnings are existing and outside changed lines); git diff --check passed; no /workspace-playground alias or redirect was added. The shared-UI TypeScript gate was attempted three times: the default 4 GB run exhausted memory, while the 8 GB run completed with 206 existing diagnostics across 26 unrelated files and zero diagnostics in task-owned files; full log is /private/tmp/task12968-research-workspace-uat/typescript-shared-ui.log. Whole-file Prettier remains blocked by existing formatting drift in touched legacy files; no broad unrelated reformat was applied. Bandit is not applicable because no Python files changed. No unresolved task-scope product gap required a follow-up. Task-owned backend, WebUI, and Chrome processes were stopped and ports 18160/18161/18162 verified free.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Certified GitHub issue #2606's beginner/no-key Research Workspace journey and promoted RW-UAT-027 to Pass. A fresh CDP-only live run passed all 17 checkpoints and now provides machine-readable proof of exactly one initialized workspace with no false migration status or migration traffic. The implementation fixes the underlying hydration/migration race, restores route tours without global chrome, removes persistent tour-banner clutter, and makes the 390x844 header usable without dropping controls. Focused regressions, the maintained real-backend check, lint, and diff hygiene pass; only unrelated repository-wide TypeScript/formatting baselines remain documented.
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
