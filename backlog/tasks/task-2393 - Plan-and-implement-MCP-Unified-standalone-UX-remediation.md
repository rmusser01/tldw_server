---
id: TASK-2393
title: Plan and implement MCP Unified standalone UX remediation
status: In Progress
labels:
- mcp
- ux
- docs
- standalone
priority: high
references:
- Docs/MCP/Unified/README.md
- Docs/MCP/Unified/User_Guide.md
- tldw_Server_API/app/core/MCP_unified/README.md
- tldw_Server_API/app/core/MCP_unified/docker/Dockerfile
- tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/cli/wizard/cli.py
documentation:
- Docs/superpowers/plans/2026-06-26-mcp-unified-standalone-ux-remediation.md
modified_files:
- Docs/MCP/Unified/README.md
- Docs/MCP/Unified/User_Guide.md
- Docs/MCP/Unified/Client_Snippets.md
- Docs/Operations/Env_Vars.md
- Docs/Product/MCP-Unified-Extraction.md
- Docs/superpowers/plans/2026-06-26-mcp-unified-standalone-ux-remediation.md
- backlog/tasks/task-2393 - Plan-and-implement-MCP-Unified-standalone-UX-remediation.md
- tldw_Server_API/app/core/MCP_unified/README.md
- tldw_Server_API/app/core/MCP_unified/docker/README.md
- tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
- tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the UX/product review findings for Unified MCP standalone/embedded experience: launch trust, standalone status clarity, first-run workflow, auth terminology, effective tool surface, catalog behavior, diagnostics, client installer readiness, and power-user documentation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current embedded vs planned standalone state is clear in primary MCP docs.
- [x] #2 Documented launch paths are either verified working or explicitly marked unsupported/experimental.
- [x] #3 Quickstart reaches a successful authenticated tools/list and read-only tool call.
- [x] #4 Auth methods are described in one canonical matrix and primary examples avoid disabled-by-default query auth.
- [ ] #5 Effective enabled MCP surface is visible by module/risk tier.
- [ ] #6 Unresolved catalog filters do not silently broaden discovery.
- [ ] #7 Status or diagnostics surface sanitized module/config problems with next actions.
- [ ] #8 Client installer can verify readiness or clearly reports missing credentials.
- [ ] #9 Power-user MCP workflows have a compact command reference.
- [ ] #10 Focused tests and Bandit touched-scope scan pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-26-mcp-unified-standalone-ux-remediation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-26: Task 1 complete in worktree `codex/mcp-unified-ux-remediation`. Added docs contract test for embedded-vs-standalone clarity, observed the expected red failure, added current-state banners to primary MCP docs, added PRD status note, and verified the focused test now passes.
- 2026-06-26: Task 2 complete. Replaced brittle MCP-specific Dockerfile launch assertions with an explicit experimental-status contract, added `tldw_Server_API/app/core/MCP_unified/docker/README.md`, and added a warning to the core MCP README Docker section. Verified Docker contract tests pass.
- 2026-06-26: Task 3 complete. Added User Guide Golden Path quickstart with supported auth header, initialize, `tools/list`, and read-only `tools/call`; added canonical auth matrix; aligned client snippets with header/subprotocol auth and strict catalog examples; expanded MCP env var docs; linked core README quickstart to the User Guide. Verified docs contract tests pass.
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
