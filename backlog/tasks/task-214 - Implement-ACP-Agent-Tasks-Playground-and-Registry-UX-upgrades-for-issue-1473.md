---
id: TASK-214
title: Implement ACP Agent Tasks Playground and Registry UX upgrades for issue 1473
status: Done
assignee: []
created_date: '2026-05-10 02:44'
updated_date: '2026-05-10 03:15'
labels:
  - ACP
  - frontend
  - UX
  - AgentTasks
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1473'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Plans/IMPLEMENTATION_PLAN_acp_frontend_ux_1473.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the #1473 ACP frontend UX workstream in the isolated ACP productionization worktree. Improve Agent Tasks, ACP Playground, and Agent Registry flows so first-time users get actionable setup/unsupported-state guidance, regular users can create/run/review/diagnose without manual ID copying, shared ACP connection/auth handling is consistent, and the main setup/run/diagnose path is covered by frontend tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-time users can understand what is missing when ACP cannot run
- [x] #2 Regular users can create run review tasks and inspect failures without manually copying IDs
- [x] #3 ACP Playground and Agent Tasks share consistent connection/auth handling
- [x] #4 E2E or focused frontend coverage exercises the main ACP setup run diagnose path
- [x] #5 GitHub issue #1473 is updated with implementation status and verification evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Current UX Contract
**Goal**: Ground the #1473 UI work in the existing Agent Tasks, ACP Playground, Agent Registry, and ACP service contracts.
**Success Criteria**: Identify the frontend seams for shared connection/auth handling, setup readiness, and task run drill-through without inventing parallel backend APIs.
**Tests**: Existing focused connection tests for Agent Tasks, ACP Playground, and Agent Registry.
**Status**: Complete

## Stage 2: Shared Readiness And Setup State
**Goal**: Give first-time users actionable ACP setup state in the task execution surface.
**Success Criteria**: Agent Tasks can explain missing orchestration routes, missing ACP health, runner/agent/API-key setup gaps, and route users to Registry or Playground diagnostics.
**Tests**: Add failing Agent Tasks frontend tests for setup/readiness states and shared auth transport.
**Status**: Complete

## Stage 3: Run/Review Drill-Through
**Goal**: Let regular users inspect task runs, session diagnostics, artifacts, and failures from Agent Tasks without copying IDs.
**Success Criteria**: Task cards expose a detail action that fetches task detail, shows run/review history, failure context, session IDs, and ACP diagnostic/artifact/audit links.
**Tests**: Add failing Agent Tasks frontend tests for task detail fetch and diagnostics rendering.
**Status**: Complete

## Stage 4: Cross-Surface Navigation And Coverage
**Goal**: Connect Agent Registry, ACP Playground, and Agent Tasks into a coherent setup/run/diagnose path.
**Success Criteria**: Registry launch links, Agent Tasks setup links, and Playground health handling use the same connection assumptions; focused frontend and E2E coverage document the main path.
**Tests**: Focused Vitest suite plus targeted Playwright coverage where feasible.
**Status**: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current UX contract findings: Agent Tasks already detects missing orchestration routes but only shows a generic unsupported message and duplicates auth/URL handling. ACP Playground and Agent Registry already use the shared ACP connection helpers. Backend #1475 exposes GET /api/v1/agent-orchestration/tasks/{task_id} with enriched runs, session links, history, failure_context, diagnostics, artifacts, and review_decision; Agent Tasks currently does not consume that drill-through contract.

Implemented shared ACP readiness normalization in apps/packages/ui/src/services/acp/readiness.ts, wired Agent Tasks to /api/v1/acp/health with setup-gap guidance, reused shared ACP auth/transport helpers across Agent Tasks and Agent Registry, added task detail drill-through diagnostics from GET /api/v1/agent-orchestration/tasks/{task_id}, and connected setup links to Agent Registry and ACP Playground. Added focused Vitest coverage and targeted Playwright coverage for the setup/run/diagnose path. Bandit is not applicable for this #1473 slice because only frontend TypeScript, E2E, docs, and Backlog files were touched.

GitHub issue #1473 updated with implementation summary and verification evidence: https://github.com/rmusser01/tldw_server/issues/1473#issuecomment-4414319018. Known skip: Bandit not run because this workstream changed frontend TypeScript, E2E, docs, and Backlog files only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
#1473 completed in the ACP productionization worktree. Added shared ACP readiness handling, actionable Agent Tasks setup gaps, consistent ACP auth/transport reuse, task diagnostics drill-through without manual ID copying, focused Vitest coverage, targeted Playwright E2E coverage, and ACP docs/readiness updates. Verification: focused UI Vitest 3 files / 9 tests passed, targeted Agent Tasks Playwright E2E passed on localhost:18080, and git diff --check was clean. GitHub issue updated: https://github.com/rmusser01/tldw_server/issues/1473#issuecomment-4414319018.
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
