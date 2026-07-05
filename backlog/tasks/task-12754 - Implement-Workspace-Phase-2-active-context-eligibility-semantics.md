---
id: TASK-12754
title: Implement Workspace Phase 2 active-context eligibility semantics
status: Done
labels:
- Workspace
- Phase 2
- Backend
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/1992
- https://github.com/rmusser01/tldw_server/issues/1984
- https://github.com/rmusser01/tldw_server/issues/1990
- https://github.com/rmusser01/tldw_server/issues/2378
modified_files:
- tldw_Server_API/app/core/Workspaces/eligibility.py
- tldw_Server_API/app/api/v1/endpoints/workspace_eligibility.py
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/app/core/Workspaces/README.md
- tldw_Server_API/tests/Workspaces/test_workspace_eligibility.py
- tldw_Server_API/tests/Workspaces/test_workspace_eligibility_api.py
- Docs/superpowers/plans/2026-06-17-workspace-active-context-eligibility.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the shared active-context eligibility contract for issue #1992, using the landed workspace membership service as the canonical membership source while preserving global browse/open visibility and fail-closed active operation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared eligibility service/helper accepts active_workspace_id, resource_type, resource_id, operation, and runtime/membership context.
- [x] #2 Workspace-sensitive operation matrix covers staging, RAG grounding, prompt use, tool use, agent manipulation, ACP run, sandbox operation, workflow launch, and watchlist run.
- [x] #3 Allowed browse/search/open/edit visibility remains separate from active-context eligibility decisions.
- [x] #4 Denied active-context operations return stable reason codes and user-actionable recovery copy for no active workspace, unlinked/cross-workspace resource, archived workspace, missing runtime, unsupported resource type, and permission failure.
- [x] #5 Contract is documented for frontend adoption.
- [x] #6 Focused backend tests and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-17-workspace-active-context-eligibility.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No known skips or blockers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Workspace Phase 2 active-context eligibility semantics for #1992 and addressed PR review feedback after rebasing on latest dev. Added a shared WorkspaceEligibilityService with explicit visibility vs active-context operation sets, runtime-required gates, stable reason codes, recovery actions, and compact membership references. Added POST /api/v1/workspace-eligibility/check, schemas, router registration, tests, and Workspace README documentation. Review fixes: sync FastAPI handler for sync DB work, required explicit runtime_state/permission_state in the API request, runtime-bound operations fail closed unless runtime_state is ready, normalized unsupported resource type denials, safer label reads, active workspace filtered from cross-workspace reverse membership results, and workspace_not_found coverage. Verification: python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_eligibility.py tldw_Server_API/tests/Workspaces/test_workspace_eligibility_api.py tldw_Server_API/tests/Workspaces/test_workspace_membership_adapters.py tldw_Server_API/tests/Workspaces/test_workspace_memberships_api.py tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py::TestWorkspaceLifecycle tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py -q passed with 163 tests. Bandit on touched backend modules wrote /tmp/bandit_workspace_eligibility_phase2_rebase.json and reported results=0.
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
