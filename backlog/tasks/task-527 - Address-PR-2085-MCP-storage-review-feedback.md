---
id: TASK-527
title: Address PR 2085 MCP storage review feedback
status: Done
labels:
- mcp
- review-fix
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2085
modified_files:
- mcp_unified/storage/models.py
- mcp_unified/interfaces/storage.py
- tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix live Gemini and Qodo review comments on PR #2085 for MCP Unified Stage 2D storage contracts: add storage model cross-field validators, align external-server transport literals with runtime support, preserve external registry list compatibility, and verify focused tests/security checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Profile assignments reject records that do not target a principal, workspace, or default binding.
- [x] External server definitions only accept runtime-supported transports.
- [x] Enabled external server definitions enforce transport-specific command/url requirements.
- [x] External registry protocol preserves the existing no-argument runtime list shape while exposing typed filtered listing separately.
- [x] Focused tests, lint/type checks, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Removed `http` from `ExternalServerDefinition.transport` because the runtime adapter builder only supports `stdio` and `websocket`.
- Added cross-field Pydantic validators for profile assignment targets and enabled external-server transport fields.
- Kept disabled external-server definitions able to represent incomplete draft rows, while enabled rows must be runnable.
- Changed `ExternalRegistryStore.list_servers()` back to the no-argument shape used by `ExternalServerManager.list_servers()` and added `list_server_definitions(*, enabled=None)` for future typed persistence stores.
- Added regression tests for all live Gemini and Qodo inline comments on PR #2085.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2085 review feedback for MCP storage contracts. Verification: storage contracts + external server manager tests passed; Ruff and Mypy passed on touched files; Bandit reported 0 findings for mcp_unified/storage and mcp_unified/interfaces.
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
