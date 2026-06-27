---
id: TASK-12050
title: Rebase PR 2534 MCP Unified UX remediation on dev
status: Done
labels:
- mcp
- pr-2534
- rebase
- tests
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2534
modified_files:
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py
- tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2534 onto current dev, resolve the MCP server import conflict, validate external review import feedback, and update tests for current dev MCP protocol/RBAC contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2534 onto current origin/dev. Resolved the only content conflict in MCP server imports by preserving dev JSON-RPC/AuthNZ imports and the PR module-surface import. Verified review-reported imports are present (`re` in server.py and `urlsplit` in wizard cli.py). Updated tests for current dev dependency/RBAC contracts. Verification: 49-test MCP docs/catalog/http/wizard group passed; 54-test MCP runtime/security/websocket group passed; Bandit broad MCP scope reported 0 high, 17 existing medium baseline findings, and 0 medium/high findings in files changed during this pass; git diff --check passed.
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
