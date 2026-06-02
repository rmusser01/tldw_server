---
id: TASK-502
title: Implement API boundary remediation Stage 1 router metadata derivation
status: Done
labels:
- api-boundary
- router-groups
- stage-1
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md
modified_files:
- tldw_Server_API/app/api/v1/router_groups/selection.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/tests/Services/test_router_groups_contract.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 1 of the accepted API boundary remediation plan: derive minimal-test router specs from canonical production RouterSpec metadata instead of duplicating route policy metadata in minimal.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared RouterSpec selection helper exists and preserves canonical route_key/default_stable/tags metadata by default.
- [x] #2 Minimal always-included router specs derive duplicated production routers from core/content/admin canonical specs instead of hand-copying metadata.
- [x] #3 Explicit RouterSpec overrides are supported only when a minimal-test route intentionally differs.
- [x] #4 Router group contract tests cover metadata preservation, override behavior, and minimal route-policy participation.
- [x] #5 Focused pytest and Bandit verification results are recorded in the task final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-api-boundary-remediation-implementation-plan.md#stage-1-router-metadata-derivation
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green path was used for the new selector and minimal metadata policy tests. A code-quality review found the canonical content group's eager VN play import could be triggered while deriving minimal specs; this was fixed by converting VN play to the lazy ImportedRouterSpec pattern and tightening the minimal import-deferral test to record swallowed eager import attempts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 1 router metadata derivation. Added a shared RouterSpec selection helper with explicit override support, changed minimal required router specs to derive from canonical core/content/admin metadata while preserving required skip_exceptions=() behavior, and converted the canonical VN play router to lazy ImportedRouterSpec so minimal derivation does not trigger unselected endpoint imports. Updated router group contract coverage for metadata preservation, override behavior, route-key fallback, missing names, minimal route-policy participation, workspace_migrations parity, VN play laziness, and minimal endpoint import deferral.

Verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q => 175 passed, 31 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/router_groups/selection.py tldw_Server_API/app/api/v1/router_groups/minimal.py tldw_Server_API/app/api/v1/router_groups/content.py -f json -o /tmp/bandit_api_boundary_stage1.json => exit 0, zero findings.

Reviews:
- Spec compliance re-review: PASS.
- Final code-quality review: PASS.

Known skips/blockers: none.
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
