---
id: TASK-540
title: Implement MCP Unified Stage 3 host adapter shim slice
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 19:56'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage3
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first narrow Stage 3 host-adapter/shim slice after Stage 2F merged. Scope this to explicit tldw_server host adapter contracts for server auth, lifecycle, permission seeding, module config, and legacy database-path behavior while preserving existing MCP route and JSON-RPC behavior. Do not start standalone gateway work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime dependency contracts expose host-adapter seams needed by MCPServer without making the standalone package import tldw_Server_API.
- [x] #2 Default tldw_server runtime dependencies bind those seams back to existing AuthNZ, MCP Hub policy, DB path, lifecycle, shutdown transport, module config, and permission-seeding behavior.
- [x] #3 MCPServer uses injected dependencies for the Stage 3 seam paths covered by this slice while preserving legacy defaults.
- [x] #4 Focused compatibility tests cover injected fake dependencies and default tldw adapter construction.
- [x] #5 Focused pytest, ruff, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-28-mcp-unified-stage3-host-adapter-shim-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added Stage 3 runtime protocols for server auth, lifecycle, permission seeding, module defaults, and policy-context feature flags.
- Added default tldw_server host adapters and wired MCPServer through injected dependencies for the slice paths.
- Re-exported new contracts from standalone and compatibility interface packages.
- Added extraction-contract tests for injected server dependencies, public re-exports, lifecycle registration, permission seeding, media DB default resolution, and auth/policy helpers.
- Delegated wildcard permission seeding to the existing admin permission service instead of embedding SQL in the MCP adapter.
- Rebased PR #2096 onto latest origin/dev and reopened the task for PR review fixes.
- Addressed Gemini review feedback by making AuthNZ websocket scope projection fail closed when websocket headers/client data are malformed.
- Re-ran focused verification after the rebase/review fix: 52 focused MCP tests passed, Ruff passed on touched scope, and Bandit reported zero findings for touched code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Stage 3 host-adapter shim slice for MCP Unified. MCPServer now accepts injected host services for auth, lifecycle, permission seeding, module config defaults, and policy-context flags while default tldw_server adapters preserve existing behavior. After rebasing PR #2096 onto the latest origin/dev, addressed review feedback so malformed AuthNZ websocket header/client scope projection fails closed before token verification. Verification: focused MCP pytest suite passed with 52 tests, Ruff passed on touched scope, and Bandit reported zero findings for touched code.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation/plan updated when relevant
- [x] #4 Bandit run for touched code when applicable
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
