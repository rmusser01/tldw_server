---
id: TASK-2234
title: Implement Workspace Core contract slice
status: In Progress
priority: high
documentation:
- Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
- Docs/superpowers/plans/2026-06-03-workspace-core-contract-implementation-plan.md
references:
- 'Task 1 commit: d2f1527e5f feat: add workspace core model contracts'
modified_files:
- tldw_Server_API/app/core/Workspaces/models.py
- tldw_Server_API/tests/Workspaces/test_workspace_core_models.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py
- tldw_Server_API/tests/Workspaces/test_workspaces_api.py
- tldw_Server_API/app/core/Workspaces/context.py
- tldw_Server_API/app/core/Workspaces/status_projection.py
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/Workspaces/test_workspace_core_context.py
- tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the first backend Workspace Core contract implementation plan: add model contracts, persisted workspace_profile, Workspace-owned primary root persistence, read-only context resolver, additive API schemas/endpoints, and focused verification. Keep UI, root attach UX, Sandbox volume creation, file scanner Jobs, and indexing workers out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace Core model contracts exist and are covered by focused tests.
- [x] #2 Workspace rows persist workspace_profile with research defaults and project upgrade semantics.
- [x] #3 Workspace-owned primary root records support one primary root with host_local and sandbox_volume backend types.
- [x] #4 Workspace context/capability projection exposes persisted profile, compatibility workspace_kind, project_root state, resolver status, and fail-closed project actions.
- [ ] #5 Existing workspace API responses are extended additively, including a read-only roots endpoint, without route aliases or redirects.
- [ ] #6 Focused backend tests, compile smoke, diff hygiene, and Bandit touched-scope verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete: added Workspace Core model contracts and focused tests in commit d2f1527e5f. Spec-compliance review approved. Code-quality review initially found an unused test import; implementer amended the commit and re-review approved. Local verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py -q` -> 4 passed, 6 warnings. `git diff --check` passed.
Task 2 complete: persisted workspace_profile defaults/validation and Workspace-owned primary root table/methods in commit 1d99282dc5. Spec-compliance review approved. Code-quality review found and verified fixes for parent workspace version bumps, same-root idempotency, replayed attach payload operational-state preservation, and operational-only state updates. Local verification: project root DB tests -> 12 passed, 6 warnings; workspace API/sub-resource regressions -> 70 passed, 6 warnings; git diff --check passed.
Task 3 complete with additive schema bridge: added read-only Workspace Core context resolver fail-closed gates, projected workspace_profile/workspace_kind/resolution/project_root through capability/context responses, preserved legacy PUT profile semantics, and loaded attached primary roots for `/capabilities` and `/context`. Code review initially found fail-open partial states, root omission, and profile-downgrade risks; re-review approved after fixes. Local verification: focused review regressions -> 4 passed, 6 warnings; Workspace subset -> 86 passed, 6 warnings; compile smoke for touched backend modules passed; `git diff --check` passed; Bandit touched backend modules -> 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
