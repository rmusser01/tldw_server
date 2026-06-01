---
id: TASK-586
title: Harden MCP external stdio process policy
status: In Progress
labels:
- mcp
- security
- external-runtime
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-mcp-stdio-process-policy-design.md
- Docs/superpowers/plans/2026-06-01-mcp-stdio-process-policy-implementation-plan.md
modified_files:
- mcp_unified/federation/process_policy.py
- mcp_unified/federation/stdio_transport.py
- mcp_unified/federation/__init__.py
- mcp_unified/federation/transports.py
- mcp_unified/gateway/config.py
- mcp_unified/gateway/cli.py
- tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit process-execution policy for standalone MCP external stdio transports: executable allowlisting, bounded cwd validation, environment allowlist checks, deterministic denial/status payloads, and focused tests before real installer execution work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-06-01-mcp-stdio-process-policy-implementation-plan.md. Stages: add process_policy helper tests/module, wire stdio transport enforcement, add gateway config/CLI wiring, add runtime-manager redaction coverage, then run targeted pytest and Bandit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added and verified MCP external stdio process-policy hardening. The package now has explicit, tested controls for command execution, cwd boundaries, PATH lookup, env inheritance, and safe error/status reporting before real upstream stdio processes are launched. Verification recorded: targeted pytest for stdio transport, gateway FastAPI package, gateway CLI package, and runtime package boundary passed (245 passed); Bandit on mcp_unified/federation and mcp_unified/gateway completed with no findings at /tmp/bandit_mcp_stdio_process_policy.json; git diff --check passed.
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
