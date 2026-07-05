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
Follow-up hash fix: included normalized capabilities in the first-run MCP generated policy hash. Added a public-generator regression where selections and allowed tools stay identical while filesystem.delete capability changes, proving generated_policy_hash changes. Verification: red check failed before helper patch with TypeError on missing capabilities parameter; after fix `python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v` -> 16 passed; Bandit on helper -> 0 findings; `git diff --check` -> clean.
Task 2 backend MCP Hub apply service: added service-local apply request/result dataclasses, apply flow for first-run generated permission profiles, global default assignment creation/update, provenance-based profile lookup, manual-edit conflict handling, keep_existing/replace_existing paths, and safe mcp_tools step payload. Red check: service test collection failed before implementation on missing compute_first_run_policy_hash. Verification: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 7 passed; python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 23 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py -f json -o /tmp/bandit_first_run_mcp_tools_service.json -> 0 findings; git diff --check -> clean.
Task 2 self-review follow-up: MCP Hub repo list methods treat owner_scope_id=None and target_id=None as broad filters, so the apply service now filters returned rows to global/null profiles and default/null assignments before mutating. Added a regression that failed before the fix on assignment_id 9 vs expected 10. Verification after fix: python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py::test_apply_filters_broad_hub_lists_to_global_null_scope_and_default_null_target -q -> 1 passed; python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 24 passed; Bandit touched-scope report -> 0 findings; git diff --check -> clean.
Task 2 code-quality follow-up: preserved unrelated policy_document fields when refreshing generated policy keys, stopped clearing default assignment path/workspace/inline/approval controls when repointing profile_id, required matching profile_id for keep_existing/replace_existing conflict-resolution requests, filtered confirmed_addon_ids to selected add-ons in the safe step payload, and removed the unused service result serializer. Red check: targeted service tests failed before implementation on dropped denied_tools, cleared assignment fields, missing ValueError, and unrelated confirmed add-on persistence. Verification: targeted red/green subset -> 5 passed after fix; python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v -> 28 passed; python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py -f json -o /tmp/bandit_first_run_mcp_tools_service_followup.json -> 0 findings; git diff --check -> clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
