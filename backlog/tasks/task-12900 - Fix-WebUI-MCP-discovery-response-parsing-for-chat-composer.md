---
id: TASK-12900
title: Fix WebUI MCP discovery response parsing for chat composer
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-06 02:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the verified first-run MCP issue where setup saves and validates MCP packs, but the chat composer MCP configuration modal shows 0 tools because the WebUI parser does not unwrap tools returned under result.content[0].json from /api/v1/mcp/tools/execute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat MCP discovery normalizes tools returned under result.content[0].json.tools.
- [x] #2 Catalog/module discovery continues to normalize existing supported shapes.
- [x] #3 A focused frontend test fails before the parser fix and passes after it.
- [x] #4 Relevant frontend test command passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: red test failed with [] before fix; focused vitest passed after fix; useMcpTools gating test passed; real browser post-fix check against live backend/frontend showed 146 tools and no "No MCP tools discovered" message. Bandit not applicable: touched TypeScript/frontend and Backlog task only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chat composer MCP discovery now handles the MCP execute response envelope returned by the backend, so the tool settings modal renders discovered tools after first-run MCP setup instead of showing 0 tools.
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
