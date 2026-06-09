---
id: TASK-2320
title: Address PR 2311 review comments for MCP file-policy action taxonomy
status: Done
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2311
modified_files:
- mcp_unified/interfaces/file_policy_actions.py
- tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify current PR #2311 review findings after rebasing on latest dev, fix only still-valid issues, and validate the focused touched scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR branch is rebased on latest origin/dev and pushed safely.
- [ ] #2 Still-valid review comments are addressed with minimal code/test changes.
- [ ] #3 Invalid or superseded comments are skipped with a documented reason.
- [ ] #4 Focused tests, lint/compile, Bandit, and diff checks are run for touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review current unresolved threads, patch valid issues in file-policy metadata, path enforcement preview semantics, and fs.patch rollback handling, then validate and push.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2311 on origin/dev and pushed codex/mcp-file-policy-action-taxonomy at 92892b7121. Addressed still-valid review findings: metadata lookup now fails with ValueError on action/metadata drift; effective permission preview now returns ask for true force-approval scope blocks while explicit path-grant deny/not_granted remains hard deny; missing nested path_decisions are normalized safely; fs.patch rollback catches/logs unexpected restore failures without masking the original partial-write error. Validation passed: focused review regression pytest set (9 passed), broader touched-scope pytest set (35 passed), ruff touched files, py_compile touched production files, Bandit touched production files, and git diff --check.
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
