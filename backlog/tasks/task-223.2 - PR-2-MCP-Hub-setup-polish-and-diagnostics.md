---
id: TASK-223.2
title: 'PR 2: MCP Hub setup polish and diagnostics'
status: Done
assignee:
  - Codex
created_date: '2026-05-10 06:13'
updated_date: '2026-05-28 20:17'
labels:
  - mcp
  - webui
  - ux
  - diagnostics
dependencies:
  - TASK-223.1
documentation:
  - Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md
  - Docs/superpowers/plans/2026-05-28-mcp-hub-setup-polish-diagnostics-plan.md
parent_task_id: TASK-223
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second PR-sized remediation slice from the MCP Hub walkthrough. This phase should make setup states easier to understand after the live-discovery and chat blocker fixes land.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No-auth local stdio servers render a neutral or healthy no-credentials-required state instead of missing-auth warnings.
- [x] #2 Legacy Secret Fallback appears only when the selected managed server actually uses the transitional server-level secret flow.
- [x] #3 Tool Catalog empty and stale states offer clear Add server and Refresh discovery actions with setup, runtime, and permissions distinctions.
- [x] #4 MCP Hub or shared diagnostics expose effective deployment mode, API origin, and health endpoint enough to diagnose quickstart versus advanced split-brain configuration.
- [x] #5 Setup isolation expectations for local walkthrough or E2E runs are documented or verified where practical.
- [x] #6 Focused UI tests and a toy MCP E2E smoke cover the polished setup path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-05-28-mcp-hub-setup-polish-diagnostics-plan.md. Executed Tasks 1-2 with TDD for readiness diagnostics and deployment-panel display, then reran the existing no-auth/categorical setup polish tests.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed MCP Hub setup diagnostics closeout for PR 2. ServerReadinessGate now publishes a bounded readiness snapshot containing state, degraded checks, health URL, last HTTP status code, health status, error message, and checked timestamp, and stores it on window for panels mounted after readiness. DeploymentDiagnosticsPanel now reads that snapshot and updates from tldw:server-readiness-state events, exposing Last health status, Last status code, Last checked, and Readiness health URL alongside deployment mode, request mode, API origin, computed Health URL, and MCP health. Existing no-auth stdio, legacy secret fallback gating, Tool Catalog empty/stale guidance, setup isolation docs, and toy MCP E2E coverage were verified present. Verification: ServerReadinessGate Vitest 9 passed; DeploymentDiagnosticsPanel Vitest 3 passed; MCP Hub setup polish Vitest 34 passed; git diff --check passed. Live Playwright toy smoke was not run because no local backend was listening on 127.0.0.1:8000 in this turn.

Bandit was not run because this slice touched TypeScript/TSX, docs, and Backlog metadata only; no Python source changed.

Additional typecheck attempts: `bunx tsc -p tsconfig.json --noEmit --pretty false` in apps/tldw-frontend failed on existing unrelated E2E type debt outside this slice; the same command in apps/packages/ui terminated with Node heap OOM before reporting diagnostics.
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
