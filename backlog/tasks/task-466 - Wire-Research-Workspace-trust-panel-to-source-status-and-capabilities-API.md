---
id: TASK-466
title: Wire Research Workspace trust panel to source status and capabilities API
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 21:31'
labels:
  - frontend
  - research-workspace
  - workspaces
  - trust
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-23-research-workspace-trust-panel-api-wiring-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase D first slice for Research Workspace trust and transparency. Add frontend API client methods for workspace source status and capabilities, render a compact trust/capability panel in /research-workspace, and reconcile local source status from the authoritative backend projection without reintroducing workspace-playground routes or aliases. Validate with focused Vitest and CDP/browser when feasible against a running backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-research-workspace-trust-panel-api-wiring-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented review follow-ups after code review: neutral pre-load trust panel state; backend-schema-aligned source-status types including nullable media_id and job uuid/status/job_type fields; fail-closed unknown lifecycle handling; stale trust state clearing on missing or switched workspace; workspace_id mismatch protection; Promise.allSettled partial-success fetch handling; and in-flight polling suppression.

Verification recorded:
- cd apps/packages/ui && bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx -> 3 files passed, 26 tests passed.
- git diff --check on touched frontend and plan paths -> clean.
- Real backend started on http://127.0.0.1:18002 and WebUI on http://127.0.0.1:18013. CDP/Playwright validation against /research-workspace passed: no /workspace-playground redirect, trust panel rendered, GET /api/v1/workspaces/{id}/sources/status returned 200, GET /api/v1/workspaces/{id}/capabilities returned 200. Screenshot: /tmp/research-workspace-trust-panel-cdp.png.

Bandit: skipped for TASK-466 because this slice only changed frontend TypeScript/tests plus the plan file; no backend Python code was touched in this task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired Research Workspace to the authoritative workspace source-status and capabilities projections, hardened the trust panel against loading/stale/partial-failure states, and validated the route with a real backend plus CDP browser automation. No legacy /workspace-playground aliases or redirects were added.
<!-- SECTION:FINAL_SUMMARY:END -->

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
