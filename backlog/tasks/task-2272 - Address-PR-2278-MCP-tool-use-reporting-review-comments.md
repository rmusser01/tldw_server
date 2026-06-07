---
id: TASK-2272
title: Address PR 2278 MCP tool-use reporting review comments
status: Done
labels:
- mcp
- review-fix
- tool-use-reporting
- gateway
modified_files:
- mcp_unified/gateway/cli.py
- mcp_unified/gateway/profile_runtime.py
- mcp_unified/gateway/tool_use_reporting.py
- mcp_unified/tool_use_reporting/reporting.py
- mcp_unified/tool_use_reporting/store.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_use_reporting.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address still-valid PR #2278 review comments for MCP tool-use reporting after rebasing onto latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review-fix pass for PR #2278: verify each unresolved Gemini thread against rebased code; add failing regression tests for still-valid findings; patch only the reporting isolation, bridge metadata, cutoff math, percentile rounding, and CLI guard paths; run focused tests, Bandit, and diff checks; push the review-fix commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2278 onto latest origin/dev and addressed all unresolved Gemini review threads that were still valid. Added regression tests for tool-use event construction failures, bridge metadata mutation, direct cutoff epoch math, and half-up percentile selection. Verification: affected red tests failed before fixes, focused suite passed with 165 passed and 5 warnings, Bandit passed with 0 findings in /tmp/bandit_mcp_tool_use_reporting_pr2278_review.json, and git diff --check passed. Known skips: no code changes were needed beyond a defensive CLI guard for the empty --since comment because the existing helper already rejects blank CLI text before indexing.
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
