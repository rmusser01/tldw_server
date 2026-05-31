---
id: TASK-77
title: Document native CodeGraph MCP operator workflow
status: Done
assignee: []
created_date: '2026-05-05 16:36'
updated_date: '2026-05-05 16:46'
labels:
  - codegraph
  - mcp
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1259'
documentation:
  - Docs/MCP/Unified/README.md
  - tldw_Server_API/Config_Files/mcp_modules.yaml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add operator-facing documentation for the native CodeGraph MCP module now that the staged implementation has landed. The work should make enabling the optional parser dependencies, module configuration, supported languages, indexing modes, tool usage, storage behavior, and troubleshooting clear enough for future users and maintainers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP docs link to a dedicated native CodeGraph guide from the Unified docs index.
- [x] #2 The guide documents install extras, YAML settings, supported languages, foreground and Jobs-backed indexing modes, storage location, read/write tool permissions, and dependency-health troubleshooting.
- [x] #3 The GitHub epic can be refreshed with links to the completed backlog tickets and the new docs PR.
- [x] #4 Focused docs verification and CodeGraph regression tests are run or any blocker is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Docs/MCP/Unified/CodeGraph.md, linked it from Unified README/User Guide, and exposed max_index_seconds in mcp_modules.yaml. Verification: git diff --check passed; CodeGraph focused pytest passed (160 passed, 5 warnings). Bandit skipped because this slice touched docs/YAML/backlog only, no Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a dedicated native CodeGraph MCP guide covering optional parser dependency installation, module enablement, settings, supported languages, foreground and Jobs-backed indexing modes, tool usage, storage behavior, permissions, and troubleshooting. Linked the guide from the Unified MCP docs index and user guide, made max_index_seconds explicit in the default CodeGraph module YAML, opened PR #1312, and refreshed GitHub issue #1259 with the related Backlog.md task trail. Verification: git diff --check passed; focused CodeGraph/MCP pytest passed with 160 passed and 5 warnings. Bandit skipped because no Python code was touched.
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
