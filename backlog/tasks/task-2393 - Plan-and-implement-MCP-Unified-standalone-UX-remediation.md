---
id: TASK-2393
title: Plan and implement MCP Unified standalone UX remediation
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-27 04:11'
labels:
  - mcp
  - ux
  - docs
  - standalone
dependencies: []
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
priority: high
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
- [x] #5 Effective enabled MCP surface is visible by module/risk tier.
- [x] #6 Unresolved catalog filters do not silently broaden discovery.
- [x] #7 Status or diagnostics surface sanitized module/config problems with next actions.
- [x] #8 Client installer can verify readiness or clearly reports missing credentials.
- [ ] #9 Power-user MCP workflows have a compact command reference.
- [ ] #10 Focused tests and Bandit touched-scope scan pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-26-mcp-unified-standalone-ux-remediation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-06-26: Task 1 complete in worktree `codex/mcp-unified-ux-remediation`. Added docs contract test for embedded-vs-standalone clarity, observed the expected red failure, added current-state banners to primary MCP docs, added PRD status note, and verified the focused test now passes.
- 2026-06-26: Task 2 complete. Replaced brittle MCP-specific Dockerfile launch assertions with an explicit experimental-status contract, added `tldw_Server_API/app/core/MCP_unified/docker/README.md`, and added a warning to the core MCP README Docker section. Verified Docker contract tests pass.
- 2026-06-26: Task 3 complete. Added User Guide Golden Path quickstart with supported auth header, initialize, `tools/list`, and read-only `tools/call`; added canonical auth matrix; aligned client snippets with header/subprotocol auth and strict catalog examples; expanded MCP env var docs; linked core README quickstart to the User Guide. Verified docs contract tests pass.
- 2026-06-26: Task 4 complete. Added `module_surface.py` to group enabled MCP modules into user-facing risk tiers, exposed the additive `surface` status field, documented risk tiers/default examples, and added focused tests for helper/status behavior.
- 2026-06-26: Task 5 complete. Changed catalog-scoped tools/resources discovery to fail closed by default on unresolved catalogs, added `_meta.catalog.status` (`resolved`, `unresolved`, or `fail_open`), exposed explicit `catalog_fail_open` through HTTP and `mcp.tools.list`, updated docs/snippets, and verified protocol/discovery/docs tests pass.

- 2026-06-26: Task 6 complete. Added structured 400 responses for invalid safe config, sanitized config warnings for bad optional config inputs, `problem_modules`/`config_warnings` on MCP status, troubleshooting rows, and focused docs/runtime tests. Verified HTTP mapping, config defaults, basic MCP functionality, and docs contracts pass.

- 2026-06-26: Task 7 complete. Added --api-key, --api-key-env, and --verify to the MCP client installer; no-credential installs are reported as configured but not ready; verification surfaces usable/auth/server-unreachable states; updated wizard docs and tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pending until all remediation tasks are complete.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
