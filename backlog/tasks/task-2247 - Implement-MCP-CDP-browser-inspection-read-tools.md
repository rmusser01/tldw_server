---
id: TASK-2247
title: Implement MCP CDP browser inspection read tools
status: In Progress
labels:
- mcp-unified
- browser
- cdp
- profiles
priority: medium
references:
- Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md
- Docs/superpowers/plans/2026-06-03-mcp-default-profile-tooling-presets-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-06-04-mcp-cdp-browser-inspection-read-tools-design.md
- Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-04-mcp-cdp-browser-inspection-read-tools-design.md
- Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md
- backlog/tasks/task-2247 - Implement-MCP-CDP-browser-inspection-read-tools.md
- tldw_Server_API/app/core/MCP_unified/browser_cdp/__init__.py
- tldw_Server_API/app/core/MCP_unified/browser_cdp/client.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py
- tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first native CDP-backed read-only browser inspection MCP tool slice: status/discovery, DOM snapshot/page-state, screenshot, console, and network read surfaces. Keep browser mutation/interactions out of scope, gate availability on explicit CDP configuration, expose the tools through profile-scoped discovery for browser-capable profiles, and validate package/runtime boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-04-mcp-cdp-browser-inspection-read-tools-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Planning/design review completed before implementation. Tightened loopback endpoint validation requirements to avoid DNS-based trust, clarified explicit-disable precedence over MCP_BROWSER_CDP_URL auto-registration, and chose a dedicated browser CDP server-registration test file instead of modifying filesystem tests.

Task 1 completed: added the browser_cdp client seam with endpoint validation, target discovery, CDP command dispatch, and bounded event observation. Verification: pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py -q (12 passed); ruff check touched browser CDP files (passed); git diff --check (passed).

Task 2 completed: added BrowserCDPModule descriptors and strict read-only argument validation. Verification: pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py -q (2 passed); ruff check touched browser CDP files (passed); git diff --check (passed).

Task 3 completed: implemented read-only tool execution for status, pages, snapshots, page state, screenshots, console events, and network events using fake-client coverage for target resolution, truncation, and screenshot payload limits. Verification: pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py -q (21 passed); ruff check touched browser CDP files (passed); git diff --check (passed).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
