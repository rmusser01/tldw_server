---
id: TASK-12027
title: Add RPG MCP module
status: Done
created_date: 2026-06-25 04:23
labels:
- rpg
- ttrpg
- backend
- mcp
- implementation
priority: high
references:
- TASK-12018
- TASK-12024
- TASK-12026
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/tests/RPG/test_rpg_mcp_module.py
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
updated_date: 2026-06-25 04:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement an optional RPG MCP module with read-only adapter/rules/context tools, authenticated session tools, protocol-level authorization tests, and optional server registration behind an environment flag.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RPG MCP module exposes expected read/write tool metadata
- [x] #2 Read-only adapter listing works without DB context
- [x] #3 Database-backed RPG tools fail closed without authenticated user context
- [x] #4 MCP protocol authorization denies missing permissions and allows exact/wildcard RPG permissions as appropriate
- [x] #5 Optional server registration is gated by MCP_ENABLE_RPG_MODULE
- [x] #6 Focused MCP/RPG tests, compileall, Bandit, and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect existing MCP module/tool authorization patterns', 'Write failing RPG MCP module and protocol authorization tests', 'Implement RPGModule with tool definitions and safe read-only adapter listing', 'Wire database-backed tool execution using authenticated context', 'Add optional server registration behind MCP_ENABLE_RPG_MODULE', 'Run focused tests/security checks, update plan/task, then commit']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented optional RPG MCP module with read-only adapter/session/rules/context tools, write-classified event/proposal tools, authenticated ChaCha DB context binding, idempotency enforcement, and optional MCP_ENABLE_RPG_MODULE server registration. Review pass tightened direct argument validation so invalid write/read inputs fail before DB binding, including idempotency key length, event sequence, positive IDs, query length, review note length, and context max_chars bounds. Verification passed: RPG MCP target tests 13 passed; focused RPG/MCP suite 75 passed; compileall passed; Bandit report /tmp/bandit_rpg_mcp.json had 0 results/errors/skips; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the RPG MCP module and optional registration path. The module exposes bounded read tools and approval-required write tools, uses authenticated per-user ChaCha DB context, enforces idempotency for mutating calls, and is covered by module, protocol authorization, registration, and validation regression tests.
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
