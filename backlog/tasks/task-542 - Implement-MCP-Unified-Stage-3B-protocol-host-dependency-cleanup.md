---
id: TASK-542
title: Implement MCP Unified Stage 3B protocol host-dependency cleanup
status: Done
labels:
- mcp
- mcp-unified
- standalone
- stage3
priority: medium
documentation:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue MCP Unified standalone extraction after Stage 3A merged by removing small remaining protocol-level host dependencies for telemetry, Redis factory defaults, and truthiness helpers. Keep tldw_server compatibility behavior intact and do not start standalone gateway work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCPProtocol uses injected runtime dependencies for telemetry and Redis idempotency defaults without importing tldw_server telemetry or redis factory directly.
- [x] #2 Protocol truthiness checks use a runtime-neutral local helper instead of importing tldw_server testing helpers.
- [x] #3 Default tldw_server runtime dependencies preserve dynamic current telemetry manager behavior through a host adapter/proxy.
- [x] #4 Focused tests cover the protocol import cleanup and compatibility behavior.
- [x] #5 Focused pytest, Ruff, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-29-mcp-unified-stage3b-protocol-host-deps-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added an AST contract test that fails if `protocol.py` imports the tldw Redis factory, telemetry manager, or testing truthiness helper directly.
- Added a dynamic telemetry regression test that monkeypatches the tldw runtime adapter and verifies trace calls dispatch to the current telemetry manager through the default dependency bundle.
- Removed direct protocol imports for telemetry, Redis factory fallback, and testing truthiness parsing; added local neutral helpers for truthy parsing and no-Redis fallback behavior.
- Added `TldwTelemetryProvider` to the host adapter bundle so default in-repo runtime dependencies preserve current telemetry manager lookup without protocol-level host imports.
- Review fix pass: rebased onto current `origin/dev`, moved the tldw runtime dependency builder import out of protocol module import time, annotated `MCPProtocol.telemetry`, added import-boundary coverage for relative imports, and logged the Redis factory-None fallback.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 3B protocol host-dependency cleanup. `MCPProtocol` now depends on injected runtime dependencies for telemetry and Redis idempotency defaults, local protocol truthiness parsing no longer imports `tldw_Server_API.app.core.testing`, and the tldw adapter supplies a dynamic telemetry proxy for default server compatibility.

Verification:
- RED: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py -q` failed as expected on forbidden protocol imports and telemetry compatibility.
- GREEN: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py tldw_Server_API/app/core/MCP_unified/tests/test_stage2_context_session.py tldw_Server_API/app/core/MCP_unified/tests/test_scope_and_fallbacks.py -q` -> 56 passed, 5 warnings.
- Ruff: `.venv/bin/python -m ruff check tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces` -> all checks passed.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces -f json -o /tmp/bandit_mcp_stage3b_protocol_host_deps.json` -> `"results": []`.

Known skips or blockers: none.

Review fix verification after rebasing on current `origin/dev`:
- RED: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py::test_idempotency_warns_when_redis_factory_returns_none -q` failed as expected on the top-level adapter import and missing Redis factory-None warning.
- GREEN: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_server_batch_and_formatting.py tldw_Server_API/app/core/MCP_unified/tests/test_stage2_context_session.py tldw_Server_API/app/core/MCP_unified/tests/test_scope_and_fallbacks.py -q` -> 68 passed, 5 warnings.
- Ruff: `.venv/bin/python -m ruff check tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces` -> all checks passed.
- Bandit runtime scope: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces -f json -o /tmp/bandit_mcp_stage3b_review_fixes_runtime.json` -> `"results": []`.
- Note: a test-inclusive Bandit run was also attempted and only surfaced the repo's existing pytest assert/temp-path baseline noise in MCP test modules.
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
