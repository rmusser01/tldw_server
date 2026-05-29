---
id: TASK-549
title: Implement MCP Unified Stage 3I module circuit-breaker seam
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 19:36'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage3
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2128'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining direct host circuit-breaker dependency from MCP module base behavior while preserving tldw_server compatibility through the injected runtime circuit-breaker factory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP module base no longer imports tldw_Server_API circuit breaker internals directly.
- [x] #2 tldw_server module registration injects the host circuit-breaker factory through runtime dependencies for compatibility.
- [x] #3 Regression tests cover the boundary rule and factory injection behavior.
- [x] #4 Focused pytest, Ruff, Bandit, and diff whitespace verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Rebased cleanly onto origin/dev at 02a017e655 before final verification.
- Added RED coverage for the modules/base.py circuit-breaker host import and for MCPServer default-module ModuleConfig breaker factory injection.
- Replaced the modules/base.py fallback host circuit-breaker import with a small host-neutral async breaker supporting can_attempt(), record_failure(), record_success(), and call_async().
- Preserved tldw_server compatibility by injecting self.dependencies.circuit_breaker_factory when MCPServer._register_default_modules() builds ModuleConfig instances.
- RED focused tests failed as expected before implementation: direct host import and missing injection.
- Focused pytest after rebase: .venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::TestBaseModule tldw_Server_API/app/core/MCP_unified/tests/test_concurrency_and_breaker.py -q -> 38 passed, 3 warnings.
- Ruff touched scope after rebase: All checks passed.
- Bandit touched implementation scope after rebase: 0 findings in /tmp/bandit_mcp_stage3i_module_breaker_after_rebase.json.
- Whitespace after rebase: git diff --check -> passed.
- Known skips/blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the remaining direct tldw_server circuit-breaker dependency from MCP module base behavior and routed host module registrations through the injected runtime breaker factory. Direct module construction now has a host-neutral fallback breaker for standalone-safe behavior while the tldw server path continues to use the existing host breaker adapter.
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
