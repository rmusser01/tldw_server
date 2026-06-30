---
id: TASK-2247
title: Implement MCP CDP browser inspection read tools
status: Done
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
- mcp_unified/USER_GUIDE.md
- mcp_unified/profiles/presets.py
- tldw_Server_API/app/core/MCP_unified/browser_cdp/__init__.py
- tldw_Server_API/app/core/MCP_unified/browser_cdp/client.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/browser_cdp_module.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py
- tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
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

Task 4 completed: added optional BrowserCDPModule default registration when MCP_ENABLE_BROWSER_CDP_MODULE is true or MCP_BROWSER_CDP_URL is configured, with explicit false taking precedence; added browser read tools/capabilities to Frontend Engineer, QA Engineer, and SDET preset tooling metadata. Verification: pytest tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_client.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_module.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q (46 passed); ruff check touched files (passed); git diff --check (passed).

Task 5 completed: documented CDP configuration, read-only tool list, loopback defaults, non-goals, and budget settings in mcp_unified/USER_GUIDE.md. Final verification: pytest focused Task 5 suite (83 passed); Bandit touched Python scope passed with report /tmp/bandit_mcp_cdp_browser_tools.json; ruff check touched files passed; git diff --check passed. Optional live CDP smoke skipped because no local listener was running on TCP port 9222.

PR review follow-up completed after rebasing on latest dev: addressed Qodo items 1-4 by adding an overall send_command deadline, rejecting oversized screenshot base64 from decoded-size estimate before full decode, clamping default observation windows against max_observation_window_ms, and clearing MCP_BROWSER_CDP_URL in the unconfigured status test. Gemini's base64-size review overlapped Qodo item 2 and was addressed by the same change.

CodeRabbit follow-up completed: aligned browser-capable preset policy capabilities with advertised native CDP read tooling, validated target WebSocket URLs with the same loopback policy as debugger URLs, and made observe_events surface CDP enable-command errors instead of silently continuing. Verification: pytest focused MCP/CDP suite (89 passed); ruff check touched scope passed; Bandit touched Python scope passed with report /tmp/bandit_mcp_cdp_browser_tools_coderabbit.json; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added optional native CDP-backed read-only MCP browser inspection tools, with loopback-safe client validation, strict read-only tool schemas, bounded execution behavior, optional server registration, browser-capable preset discovery metadata, tests, and package-local user guide documentation. PR review follow-up added overall command deadlines, pre-decode screenshot payload rejection, observation-window clamping, env-isolated status tests, browser policy capability alignment, target WebSocket URL validation, and enable-command error propagation. Live browser smoke was skipped because no local CDP endpoint was already running.
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
