---
id: TASK-12148
title: Implement first-run MCP tool packs setup
status: In Progress
assignee: []
created_date: '2026-07-04 23:41'
labels:
  - implementation
  - mcp
  - setup
  - first-run
dependencies:
  - TASK-12132
references:
  - >-
    Docs/superpowers/plans/2026-07-04-first-run-mcp-tool-packs-implementation-plan.md
  - Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the reviewed first-run MCP tool packs implementation plan using subagent-driven development. Scope includes backend catalog/apply/validate APIs, MCP Hub profile integration, frontend onboarding step, MCP Hub follow-up status, tests, verification, and commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend setup catalog, apply, validate, and admin recovery endpoints are implemented per plan.
- [ ] #2 Frontend onboarding MCP tools step and MCP Hub follow-up/recovery UI are implemented per plan.
- [ ] #3 Focused backend/frontend tests and touched-scope Bandit verification are recorded.
- [ ] #4 Implementation commits are reviewed with spec and code-quality subagent gates.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 backend catalog/policy slice: created tldw_Server_API/app/core/Setup/first_run_mcp_tools.py and tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py. Red check confirmed ModuleNotFoundError before implementation. Verification: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v -> 14 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py -f json -o /tmp/bandit_first_run_mcp_tools.json -> 0 findings; git diff --check -> clean.
Task 1 follow-up code-quality fix: kept MCP discovery list tools available with realistic unclassified registry metadata and moved broad grants from allowed_tools into top-level capabilities. Red check before fix: catalog tests failed on missing mcp.catalogs.list/mcp.modules.list/mcp.tools.list and missing capabilities field. Verification: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v -> 15 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py -f json -o /tmp/bandit_first_run_mcp_tools_followup.json -> 0 findings; git diff --check -> clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
