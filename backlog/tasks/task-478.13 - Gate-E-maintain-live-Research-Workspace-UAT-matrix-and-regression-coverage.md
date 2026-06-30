---
id: TASK-478.13
title: 'Gate E: maintain live Research Workspace UAT matrix and regression coverage'
status: Done
labels:
- research-workspace
- uat
- gate-e
- tests
- playwright
- regression
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
documentation:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/superpowers/plans/2026-05-25-task-478-13-research-workspace-uat-matrix.md
modified_files:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/superpowers/plans/2026-05-25-task-478-13-research-workspace-uat-matrix.md
- apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Process need: the current UAT found hidden breakages only after running a real backend and WebUI. Future tasks must prove behavior with a live app, not just static code review.

User goal: know which exposed workspace functionality works, which is broken, and which fixes have been verified against the running product.

Scope:
- Convert the live UAT checklist into a maintained matrix covering first-time flow, power-user flow, source ingestion, status, selection, RAG, Studio, My Media, folders, annotations, settings/share, responsive layout, old-route 404/no redirects, and extension handoff.
- Record reproduction steps, expected result, actual result, backend/API evidence, screenshots where useful, and pass/fail status per task.
- Add automated regression tests for high-risk paths as each child task is fixed instead of waiting until the end.
- Keep validation anchored to live backend + WebUI + CDP/Playwright, with configured providers or local llama.cpp when needed.

Acceptance criteria:
- A current UAT matrix exists and is updated as each child task completes.
- Each fixed child task records tests and live verification evidence in its backlog final summary.
- Old `/workspace-playground` remains 404/no redirect and current UI/route metadata use `/research-workspace` names.
- Final matrix shows no critical or high hidden-broken functionality remaining, or explicitly documents unresolved blockers.

Depends on: all functional child tasks for final completion; matrix scaffolding can start immediately.
Parallelization: each task owner updates its own row(s); final consolidation happens last.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current UAT matrix exists and is updated with child task evidence.
- [x] #2 Fixed child task evidence is represented or linked in the matrix.
- [x] #3 Old `/workspace-playground` remains 404/no redirect with regression coverage.
- [x] #4 Critical/high hidden-broken functionality is either passing or explicitly blocked/gap-tracked.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` as the maintained acceptance ledger for Research Workspace UAT.
- Added a focused real-backend Playwright regression for canonical `/research-workspace` boot and removed `/workspace-playground` no-redirect 404 behavior.
- Ran a live backend/WebUI Playwright/CDP probe against backend `http://127.0.0.1:18002` and WebUI `http://localhost:3000`. The probe passed route, copy, tour, and diagnostic assertions and saved `/private/tmp/task47813-live-matrix-research-workspace.png`.
- Bandit was skipped because this task touched docs and frontend E2E TypeScript only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the current Research Workspace live UAT matrix and added focused route replacement regression coverage. The matrix covers first-time and power-user flows, source acquisition, ingestion/status, selection, grounded RAG, Studio, source preview/annotations, responsive layout, onboarding/tour, old-route removal, and the Shared Workspaces/MCP/ACP/Sandbox model. Live backend/WebUI CDP validation passed for canonical route boot, old /workspace-playground 404/no redirect, contextual local/self-hosted and missing-model copy, rejected workspace-trust/left-panel copy absence, visible tour overlay, and no critical console/page errors. Focused Playwright regression passed: research-workspace.real-backend.spec.ts --grep "keeps research-workspace canonical" (1 passed). Known blockers remain explicit in the matrix: browser extension handoff waits for TASK-478.12/current extension build; MCP/ACP/Sandbox live user flows remain Partial/Gap where only the contract exists. Bandit skipped because this task touched docs and frontend E2E TypeScript only.
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
