---
id: TASK-480
title: Design MCP Unified standalone library and gateway extraction
status: In Progress
labels:
- design
- mcp
- mcp-unified
documentation:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-26-mcp-unified-stage1-adapter-seams-implementation-plan.md
- Docs/MCP/mcp_unified_module_ownership_inventory.md
modified_files:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-26-mcp-unified-stage1-adapter-seams-implementation-plan.md
- Docs/MCP/mcp_unified_module_ownership_inventory.md
- tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py
- tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py
- tldw_Server_API/app/core/MCP_unified/interfaces/policy.py
- tldw_Server_API/app/core/MCP_unified/interfaces/storage.py
- tldw_Server_API/app/core/MCP_unified/adapters/__init__.py
- tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py
- tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/modules/base.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py
- tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
- backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-26-mcp-unified-stage1-adapter-seams-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 1 adapter-seam implementation added runtime/policy/storage interface contracts, tldw host adapters, protocol/server dependency injection seams, BaseModule circuit-breaker factory injection, extraction boundary tests, and module ownership inventory. Verification recorded on 2026-05-27 against the dev-based PR branch: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py -v` passed 38 tests; `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py -v` passed 47 tests; Bandit on touched MCP scope wrote `/tmp/bandit_mcp_unified_stage1_adapter_seams_dev_pr.json` with 0 findings. Known note: the source worktree commit operation reported an existing Git GC warning about unreachable loose objects; no feature tests failed.
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
