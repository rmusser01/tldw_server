---
id: TASK-506
title: Stage 5 prototype promotion review service boundary
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 02:29'
labels:
  - api-boundary
  - prototype-workspaces
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move prototype promotion review decisions behind a public PrototypeWorkspaceService method so endpoint code no longer performs promoter authorization or direct rejection state transitions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-api-boundary-remediation-implementation-plan.md#stage-5-prototype-promotion-review-ownership
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a public PrototypeWorkspaceService.review_promotion_request(...) method that owns promotion request lookup, workspace lookup, promoter authorization, rejection state transitions, and approval delegation through promote_candidate(...). The prototype promotion review endpoint now delegates the whole decision to the service and only maps PermissionError/ValueError to HTTP responses.

Verification so far:
- RED: pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q failed on missing review_promotion_request.
- RED: pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py::TestPrototypeWorkspaceEndpoints::test_review_endpoint_delegates_decision_to_service -q failed because endpoint called repo.get_promotion_request.
- GREEN: pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py -q => 13 passed.
- GREEN: focused endpoint review tests => 7 passed.
- GREEN: pytest tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q => 31 passed.
- Smoke: rg -n "_is_promoter|repo\.update_promotion_request" tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py => no matches.
- Bandit: python -m bandit -r service.py prototype_workspaces.py -f json -o /tmp/bandit_api_boundary_stage5.json => results [].
- git diff --check => clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Public promotion review handling now lives in PrototypeWorkspaceService.review_promotion_request(...). The endpoint delegates review decisions to the service and keeps HTTP-only error mapping. Added service and endpoint regressions for owner/designated promoter rejection, forbidden reviewers, missing requests, approval delegation, and endpoint delegation.

Verification recorded in Implementation Notes. No known skips or blockers.
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
