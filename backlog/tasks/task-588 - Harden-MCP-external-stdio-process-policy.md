---
id: TASK-588
title: Harden MCP external stdio process policy
status: Done
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
PR review pass after rebasing onto latest `origin/dev`: reproduced the active
Gemini/Qodo findings with failing tests, then fixed the basename allowlist
bypass, POSIX case-sensitive path comparisons, explicit package-factory
process-policy wrapping, and private helper docstrings. The rebase introduced a
Backlog ID collision with the `dev` branch's Personas documentation task, so
this branch task moved from `TASK-586` to `TASK-588`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added and verified MCP external stdio process-policy hardening. The package now has explicit, tested controls for command execution, cwd boundaries, PATH lookup, env inheritance, and safe error/status reporting before real upstream stdio processes are launched. PR review fixes also deny basename allowlist bypasses, preserve POSIX case-sensitive path semantics, and apply configured policy when the package stdio factory is explicitly injected. Verification recorded: targeted pytest for stdio transport, gateway FastAPI package, gateway CLI package, and runtime package boundary passed (249 passed); Bandit on mcp_unified/federation and mcp_unified/gateway completed with no findings at /tmp/bandit_mcp_stdio_process_policy.json; git diff --check passed.
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
