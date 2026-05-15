---
id: TASK-324
title: Risk Gate 1 prototype workspace threat model and auth invariants
status: Done
assignee: []
created_date: '2026-05-14 01:10'
updated_date: '2026-05-14 01:19'
labels:
  - prototype-workspaces
  - risk-gate-1
  - security
  - docs
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1440'
  - 'https://github.com/rmusser01/tldw_server/issues/1453'
  - 'https://github.com/rmusser01/tldw_server/pull/1466'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-09-prototype-workspace-productionization-issue-tree-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-prototype-workspace-productionization-tracker-implementation-plan.md
  - Docs/API-related/Prototype_Workspaces_Contract_Matrix.md
  - Docs/API-related/Prototype_Workspaces_API.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1453 under tracker #1440. Produce the Risk Gate 1 security-contract package for prototype workspace collaboration: actor definitions, authorization invariants, non-enumerating error behavior, token/session security requirements with explicit dispositions, audit/quota deferrals, draft frontend contract matrix updates, frontend fixture/mock-state prep notes, and focused backend tests for existing MVP ownership, revocation, expiration, and cross-workspace isolation behavior. Work must be isolated from the dirty main checkout and should preserve the risk-gated productionization plan created in PR #1466.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actor identities and authorization invariants for owner, internal collaborator, designated promoter, and external shared actor are documented.
- [x] #2 Token and session security requirements are documented with explicit dispositions: enforce now, document existing behavior, or defer to a later Risk Gate.
- [x] #3 Revocation, expiration, ownership, non-enumeration, and cross-workspace isolation expected behavior is documented.
- [x] #4 The prototype workspace contract matrix is updated or extended with a frontend-consumable Risk Gate 1 draft.
- [x] #5 Frontend/Product fixture, mock-state, route-state audit checklist, and Risk Gate 4 feedback expectations are documented.
- [x] #6 Focused backend tests cover core ownership, revocation, expiration, and cross-workspace isolation behavior already present in the MVP.
- [x] #7 Verification results are recorded, including focused pytest results and Bandit if backend code changes occur.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-13-prototype-risk-gate-1-auth-invariants.md

Worktree: `.worktrees/prototype-risk-gate-1-auth-invariants` on branch `codex/prototype-risk-gate-1-auth-invariants` from `origin/dev`.

Baseline verification before edits: `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_service_authorization.py -q` -> 3 passed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in isolated worktree `.worktrees/prototype-risk-gate-1-auth-invariants` on branch `codex/prototype-risk-gate-1-auth-invariants`. Added RED tests for revoked and expired external shared actors submitting promotion requests with still-signed session tokens; initial run failed with a server error for revoked actors and 201 for expired actors. Added endpoint guard so promotion submission revalidates active session/shared-actor state before creating promotion requests.

Verification: focused RED command failed as expected before implementation; focused GREEN command passed after implementation. Full Risk Gate 1 backend set passed: `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_service_authorization.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q` -> 41 passed, 5 warnings. Bandit passed with zero findings for `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`; JSON written to `/tmp/bandit_prototype_risk_gate_1.json`. `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Risk Gate 1 for prototype workspace collaboration. Added a dedicated threat model documenting actor identities, trust boundaries, authorization invariants, non-enumerating error behavior, token/session security dispositions, audit taxonomy, quota policy, frontend prep requirements, and deferred follow-ups. Expanded the contract matrix with concrete frontend state buckets and token/session dispositions, and linked the security contract from the Prototype Workspaces API doc.

Backend hardening closes a promotion-submission gap: collaborator promotion requests now revalidate the active shared actor and session state instead of trusting only the signed session token. Revoked or expired external shared actors now receive the stable inactive-session 403 response, and race-time repo ValueErrors for inactive shared actors are mapped away from 500s. Added regression tests for revoked and expired shared actors submitting promotion requests.

Verification: focused prototype authorization baseline passed before edits; RED tests failed as expected; GREEN focused tests passed; full Risk Gate 1 backend set passed with 41 tests; Bandit on the touched endpoint reported zero findings; git diff --check passed.
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
