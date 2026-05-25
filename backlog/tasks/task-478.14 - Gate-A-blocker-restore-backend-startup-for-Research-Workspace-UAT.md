---
id: TASK-478.14
title: 'Gate A blocker: restore backend startup for Research Workspace UAT'
status: Done
labels:
- research-workspace
- uat
- gate-a
- backend
- startup
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
modified_files:
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live backend validation for Research Workspace is blocked after rebasing onto latest dev because Uvicorn fails during import: `NameError: name 'ConfigDict' is not defined` in `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py` while registering workspace migration routes. Fix the startup blocker with focused coverage so backend+WebUI UAT can continue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend starts successfully in single-user local mode for Research Workspace UAT.
- [x] #2 The workspace schema module imports cleanly and any Pydantic v2 config usage has required imports.
- [x] #3 Focused backend verification covers the import/startup failure or equivalent schema import path.
- [x] #4 Live WebUI validation can proceed against a running backend.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added the missing Pydantic v2 `ConfigDict` import used by workspace migration schemas.
- Restored workspace rate-limit policy constants from the canonical `workspaces_rate_limit_policy` module.
- Removed the duplicate `map_db_error_to_http` import while touching the workspace endpoint import block.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored backend import/startup for Research Workspace UAT by adding the missing Pydantic `ConfigDict` import and using the correct workspace rate-limit policy constants import. Verification: `AUTH_MODE=single_user SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -c "import tldw_Server_API.app.api.v1.endpoints.workspaces"` passed; `AUTH_MODE=single_user SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_rate_limit_contract.py -q` passed: 3 tests. Live backend+WebUI UAT continued against the running backend with workspace status/capabilities calls returning 200. Bandit on touched backend files reported 0 findings in `/tmp/bandit_research_workspace_uat.json`.
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
