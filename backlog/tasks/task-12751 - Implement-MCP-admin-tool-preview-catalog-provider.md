---
id: TASK-12751
title: Implement MCP admin tool preview catalog provider
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-17 05:35'
labels:
  - mcp
  - policy
  - implementation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the MCP effective permission explain implementation plan: add an unfiltered admin tool catalog provider and wire policy preview to include denied installed tools without changing model-facing tool discovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Admin catalog preview tests cover denied installed tools from the admin catalog.
- [x] #2 mcp_unified/gateway/tool_discovery.py exposes a public admin catalog entry/helper that does not hide denied tools.
- [x] #3 GatewayPolicyExplainService.preview_profile_tools uses admin_tool_catalog_provider or installed_tool_catalog fallback and preserves degraded behavior when no catalog is available.
- [x] #4 Focused policy explain service pytest passes or blockers are documented.
- [x] #5 Changes are committed separately for Task 2.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 2 implementation under the approved subagent-driven workflow.

Implemented Task 2 preview catalog wiring. Added AdminToolCatalogEntry and list_admin_tool_catalog without model-facing visibility filtering, wired GatewayPolicyExplainService to prefer admin_tool_catalog_provider and fall back to installed_tool_catalog/list_admin_tool_catalog, and added denied-installed admin catalog preview coverage. Verification so far: focused pytest passed (35 passed), Bandit passed with no issues, git diff --check passed.

Code-quality review follow-up: added installed_tool_catalog fallback coverage, model-facing discovery exclusion assertions, preview filter coverage for include_denied/include_recommendations/category, and pagination-after-filtering coverage. The new pagination test exposed a row-ordering bug in preview catalog normalization; fixed by preserving catalog/provider order while de-duplicating.

Verification caveat: requested command for tldw_Server_API/tests/MCP_unified/test_gateway_tool_discovery.py returned pytest exit 4 because that file does not exist in this worktree; no new test file was created because the review follow-up kept the owned-file scope unchanged.

Final review results: latest spec compliance review passed; latest code-quality review found no Critical or Important issues. The only reviewer note was final task bookkeeping, addressed here.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 added an unfiltered admin tool catalog provider and wired policy preview to include denied installed tools for admin-only previews while preserving model-facing filtering. It added AdminToolCatalogEntry/list_admin_tool_catalog, installed-catalog fallback wiring, install-status preservation, filter handling, and regression coverage for denied installed tools, model-facing denied-tool hiding, include_denied/include_recommendations/category filters, and cursor pagination after filtering.

Verification from current HEAD using the root checkout virtualenv:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v -> 38 passed, 7 warnings
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/policy_explain.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -s B101 -f json -o /tmp/bandit_task2371.json -> 0 findings
- git diff --check HEAD~2..HEAD -> exit 0

Known caveat: tldw_Server_API/tests/MCP_unified/test_gateway_tool_discovery.py does not exist in this worktree, so that reviewer-suggested command is not a valid local gate.
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
