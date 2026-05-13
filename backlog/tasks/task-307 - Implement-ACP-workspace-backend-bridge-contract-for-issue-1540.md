---
id: TASK-307
title: Implement ACP workspace backend bridge contract for issue 1540
status: Done
assignee:
  - codex
created_date: '2026-05-13 00:21'
updated_date: '2026-05-13 00:44'
labels:
  - ACP
  - workspace
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1540'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
documentation:
  - Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md
  - Docs/Development/Agent_Client_Protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 1 from Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md: explicit backend bridge between canonical workspaces and ACP execution workspaces, with ownership, allowlist, trusted-root, and duplicate-bridge tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP execution workspace metadata carries canonical_workspace_id and source when linked
- [x] #2 Backend helper/API path can find or create a linked ACP execution workspace for a canonical workspace
- [x] #3 Project/task detail responses expose canonical workspace metadata through the ACP execution workspace
- [x] #4 Ownership, missing workspace, missing allowlist, trusted-root inheritance, cwd containment, and duplicate bridge behavior are covered by focused tests
- [x] #5 Issue #1540 can be updated with implementation evidence and remaining frontend slices
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add OrchestrationDB helper coverage first: verify ACP workspaces can be found by metadata.canonical_workspace_id and that linking an existing workspace merges canonical metadata without duplicating rows.
2. Add agent orchestration API tests for the canonical bridge path: canonical workspace must exist for the current user, root_path must pass the existing ACP allowlist validation, an existing linked workspace is reused idempotently, root conflicts fail closed when linked to another canonical workspace, and an unlinked root workspace can be linked.
3. Add response contract tests proving project detail and task detail expose a canonical_workspace bridge object inherited from the bound ACP execution workspace.
4. Implement minimal backend support in tldw_Server_API/app/core/DB_Management/Orchestration_DB.py and tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py, reusing existing root validation, workspace CRUD, and dispatch cwd containment helpers.
5. Run focused pytest for the changed Agent_Orchestration tests, run Bandit on touched backend implementation files, update TASK-307 acceptance criteria/notes, and prepare the PR with #1540 remaining frontend slices still open.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation evidence:
- Added ACP workspace metadata bridge helpers in OrchestrationDB: get_workspace_by_canonical_workspace_id and link_workspace_to_canonical.
- Added POST /api/v1/agent-orchestration/workspaces/canonical-bridge to validate canonical workspace existence, enforce ACP root allowlisting, reuse existing links, link existing unlinked roots, and reject conflicting root/canonical links.
- Project and task detail responses now expose canonical_workspace inherited through the bound ACP execution workspace.
- Added focused tests for ownership/missing canonical workspace behavior, missing allowlist, duplicate bridge prevention, trusted-root inheritance, cwd escape rejection, and response metadata.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_workspace_db.py tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q => 80 passed, 5 warnings.
- git diff --check => clean.
- source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py tldw_Server_API/app/core/DB_Management/Orchestration_DB.py -f json -o /tmp/bandit_task307.json => 0 findings.
- Ruff focused check after import cleanup still reports pre-existing baseline issues in touched files: B904/F841/UP037/SIM118/F841 unrelated to this bridge slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the ACP workspace backend bridge contract for #1540 and opened PR #1615. The slice adds the canonical bridge endpoint, metadata helpers, response exposure for project/task details, documentation, and focused tests for ownership/missing workspace, allowlist failures, duplicate bridge behavior, trusted-root inheritance, cwd containment, and canonical response metadata. Verification: focused Agent Orchestration pytest 80 passed; git diff --check clean; Bandit on touched backend files 0 findings. Known non-blocker: focused Ruff still reports unrelated pre-existing baseline issues in touched files after import cleanup.
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
