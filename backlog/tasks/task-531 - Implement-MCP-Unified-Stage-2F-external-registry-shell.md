---
id: TASK-531
title: Implement MCP Unified Stage 2F external registry shell
status: Done
labels:
- mcp-unified
- standalone
- stage2
- federation
references:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
modified_files:
- Docs/superpowers/plans/2026-05-28-mcp-unified-stage2f-external-registry-shell-plan.md
- mcp_unified/federation/__init__.py
- mcp_unified/federation/manager.py
- mcp_unified/federation/models.py
- mcp_unified/federation/transports.py
- tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and implement the next reviewable MCP Unified standalone Stage 2F slice: package-local external-server registry models and a non-spawning fake transport lifecycle shell with health state, virtual-tool metadata, and policy-gated execution tests. Defer real stdio process spawning and host MCP Hub wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mcp_unified exposes package-local external registry/federation primitives with no tldw_Server_API imports.
- [x] #2 Fake/non-spawning transport lifecycle can register, start/stop, report health, and expose namespaced virtual-tool metadata without launching processes.
- [x] #3 Policy-gated execution denies or allows fake upstream tool calls based on injected profile/effective policy decisions and records audit events where available.
- [x] #4 Existing tldw_server MCP behavior and Stage 2A-2E tests remain compatible.
- [x] #5 Focused pytest and Bandit checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-28-mcp-unified-stage2f-external-registry-shell-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Spec: Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md Stage 2F.
- PR #2094 review pass addressed lifecycle rollback, best-effort stop cleanup, refresh diagnostics, locked read snapshots, and helper docstrings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2F adds a standalone, non-spawning external federation shell under mcp_unified. It uses injected registry/transport/audit dependencies, exposes ext.<server>.<tool> virtual metadata, reports fake transport health, and gates execution on effective policy external-server, allowed-tool, denied-tool, and credential-slot grants. Real stdio/websocket process management remains deferred.
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
