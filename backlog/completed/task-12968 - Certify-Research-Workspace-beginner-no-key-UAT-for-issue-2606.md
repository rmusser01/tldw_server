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
- https://github.com/rmusser01/tldw_server/pull/2731
documentation:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/README.md
modified_files:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/README.md
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/run-beginner-uat.mjs
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/uat-diagnostic-gate.mjs
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/uat-diagnostic-gate.test.mjs
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/checkpoints.json
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/diagnostics.json
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/desktop-settled-workspace.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/desktop-first-run-tour.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/desktop-visible-search.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/desktop-add-url-auth-recovery.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/mobile-direct-entry.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/mobile-sources-tab.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/mobile-chat-tab.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/mobile-studio-tab.png
- Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/mobile-search.png
- apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage9.error.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage1.onboarding.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/store/__tests__/workspace.split-storage.test.ts
- apps/packages/ui/src/store/__tests__/workspace.test.ts
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
Completed the beginner/no-key certification against an isolated live stack: FastAPI single-user backend on 127.0.0.1:18160 with a server-only key and disposable SQLite/user databases; advanced-mode Next.js WebUI on 127.0.0.1:18161 with NEXT_PUBLIC_X_API_KEY and NEXT_PUBLIC_API_BEARER explicitly empty; fresh Chrome 145.0.7632.6 profile on CDP 127.0.0.1:18162 with browser and component extensions disabled. The task-owned CDP runner passed 17/17 desktop/mobile checkpoints. Its manifest records exactly one active/saved workspace, no visible migration status, and zero migration API calls at desktop entry/final and mobile entry. Diagnostics record zero page errors, request failures, unexpected HTTP errors, credential-bearing requests, migration requests, global backend dialogs, or runtime overlays. Expected no-key warnings/guards and one Next development HMR warning remained scoped.

Product fixes: always mount the shared TutorialRunner when global chrome is hidden; suppress the initial empty workspace persistence write in both split and monolithic modes; defer first-workspace initialization safely under StrictMode; scope fresh-initialization migration suppression to the exact workspace ID; replace the persistent tour-start notice with transient message feedback; compact the mobile header into bounded context/action rows; and update the stale AddSource partial-success regression to the current persistent row contract.

Evidence hardening commits the exact runner, force-tracked JSON manifests, and representative screenshots under Docs/Reviews/assets/2026-07-14-research-workspace-beginner-no-key-uat/. Browser network events are captured at BrowserContext scope for direct backend /api/v1, same-origin /api/v1, and hosted /api/proxy request shapes. A browser-level CDP target observer fails on transient service workers or extension background pages. Dedicated disposable browser contexts emit unique start/end health sentinels that bracket the backend log without contaminating the desktop/mobile persona contexts. The final correlated segment contained 11 API access-log lines and zero workspace migration lines.

Independent pre-merge review found and the task fixed three additional issues: a suppressed empty hydration write no longer emits a false cross-tab BroadcastChannel update; mobile server workspace identity/context/recovery copy remains in the accessibility tree; and non-empty pageErrors, requestFailures, or httpErrors now fail the UAT run. Focused regressions cover split and monolithic broadcast suppression plus all three diagnostic buckets.

Final verification: focused shared UI suite passed 9 files / 220 tests; WebLayout suite passed 14/14; diagnostic gate passed 4/4; the live CDP rerun passed 17/17 with Chrome 145.0.7632.6, 11 correlated API access-log lines, zero migration traffic, and clean diagnostic buckets. Targeted repository-pinned ESLint exited 0 with zero errors (16 pre-existing warnings outside changed lines); artifact byte-for-byte checks, node --check, git diff --check, and the removed-route probe passed. /workspace-playground returned HTTP 404 with no Location header. Bandit is not applicable because no Python files changed. Task-owned processes were stopped, ports 18160/18161/18162 were free, and temporary untracked setup artifacts were removed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed TASK-12968 and addressed every verified pre-merge review finding. Fresh beginner/no-key Research Workspace entry now creates exactly one workspace without false migration persistence or cross-tab broadcasts; the tour remains available on the shell-less route; mobile server workspace context remains accessible; and the committed CDP runner fails on runtime/network diagnostics. The authoritative rerun used Chrome 145.0.7632.6 and passed 17/17 checkpoints with zero page errors, request failures, HTTP errors, credential-bearing requests, background targets, or workspace migration calls. Final focused verification is 220 shared-UI tests, 14 WebLayout tests, and 4 diagnostic-gate tests. The earlier Chrome 143 and 164-test figures were intermediate runs and are superseded by this final record. Independent re-review found no Critical or Important issues. The requester-owned human Change summary remains the only PR policy gate.
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
