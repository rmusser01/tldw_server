---
id: TASK-490
title: Address prototype signoff PR review feedback for stale collaborator sessions
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 04:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR #1980 review feedback: stored collaborator session state must not force collaborator view for a different workspace owner URL. Track implementation, verification, and PR follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Owner workspace URLs render the owner view when only stale stored collaborator data exists for another workspace.
- [x] #2 Stored collaborator session context is scoped to the workspace created by the collaborator branch session after token cleanup.
- [x] #3 Focused tests and verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1980 review feedback by scoping stored collaborator session state to a collaboratorWorkspaceId. PrototypeWorkspacePage now only treats stored collaborator data as collaborator entry context when it matches the workspace in the URL, preventing stale collaborator sessions from forcing owner workspaces into collaborator view. Added regressions for stale-session owner fallback and collaborator branch session workspace scoping. Verification: route shim vitest 1 passed; focused prototype UI vitest 32 passed; prototype docs/readiness pytest 5 passed; git diff --check passed; Bandit backend security scope returned no errors/results. UI package tsc was attempted with increased heap and failed on existing repo-wide baseline errors outside the touched prototype workspace files.
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
