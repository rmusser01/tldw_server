---
id: TASK-2384
title: Verify single-user Workspace Phase 2 container loop end-to-end
status: Done
labels:
- workspace
- phase2
- verification
- backend
- frontend
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/1995
- https://github.com/rmusser01/tldw_server/issues/1984
- https://github.com/rmusser01/tldw_server/pull/2387
- https://github.com/rmusser01/tldw_server/issues/1995#issuecomment-4738013294
- https://github.com/rmusser01/tldw_server/issues/1984#issuecomment-4738013358
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track #1995 release evidence for the single-user Workspace Phase 2 container loop. Build the scenario matrix, verify backend/API/client/manual/browser evidence across the agreed resource set, document limitations/deferrals, and post final evidence back to #1984.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scenario matrix covers #1995 single-user Workspace Phase 2 loop requirements and identifies explicit deferrals.
- [x] #2 Backend Workspace suite passes after reconciling stale ACP session eligibility expectations.
- [x] #3 Focused frontend Workspace/ACP/route contract tests pass.
- [x] #4 Live browser smoke verifies the canonical Workspaces manager and Research Workspace handoff in single-user mode.
- [x] #5 Final evidence is posted to GitHub issues #1995 and #1984.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Plan: `Docs/superpowers/plans/2026-06-18-workspace-phase2-e2e-verification-plan.md`.
- Evidence matrix: `Docs/Validation/workspace-phase2-single-user-container-evidence.md`.
- Baseline found one stale backend expectation: `acp_session` is currently supported by the membership/runtime binding adapters, so unsupported-type coverage now uses reserved future type `acp_run`.
- Frontend focused run found one stale ACP history test ordering issue: the test clicked a modal-closing footer action before checking diagnostics. It now verifies diagnostics first, then Agent Tasks handoff.
- Production backend code was not changed. Bandit is not applicable for this docs/test-only branch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Backend verification: `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_eligibility.py -q` passed with `16 passed, 6 warnings`; `python -m pytest tldw_Server_API/tests/Workspaces -q` passed with `461 passed, 8 warnings`.
- Frontend verification: focused Vitest Workspace/Research Workspace/MCPHub/ACP/route suite passed with `10 passed (10 files), 79 passed (79 tests)`.
- Browser verification: `workspaces-manager.spec.ts` passed against a local single-user API server with `2 passed`.
- GitHub evidence comments posted to #1995 and #1984, and PR #2387 opened for review.
- Remaining gaps are explicitly documented: Research Workspace source/chat/studio browser E2E and parity specs were not rerun in this slice; workflow/watchlist/sandbox frontend evidence remains backend/API-only or partial.
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
