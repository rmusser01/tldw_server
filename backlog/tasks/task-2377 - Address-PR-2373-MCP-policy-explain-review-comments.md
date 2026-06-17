---
id: TASK-2377
title: Address PR 2373 MCP policy explain review comments
status: Done
labels:
- mcp
- policy
- review
modified_files:
- mcp_unified/README.md
- mcp_unified/USER_GUIDE.md
- mcp_unified/gateway/cli.py
- mcp_unified/gateway/policy_explain.py
- mcp_unified/gateway/fastapi.py
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address still-valid PR review comments on the MCP policy explain/profile preview surface after rebasing on latest dev: test markers/docstrings, preview catalog exception diagnostics, admin route rate limiting, preview mode semantics, CLI session-id propagation, event-loop offloading, fallback hardening, and command-shaped redaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New policy explain test modules include module docstrings and a single accepted test marker without relying on unsupported asyncio markers.
- [x] #2 Preview catalog fallback records diagnostics instead of silently swallowing exceptions.
- [x] #3 Admin policy explain routes have explicit finite rate limiting or an existing rate limit integration is correctly applied.
- [x] #4 Profile tool preview mode semantics are corrected so response metadata does not misrepresent runtime-effective behavior.
- [x] #5 CLI preview propagates session_id, grant-store simulation is offloaded, missing policy fallback is guarded, and command-shaped tool redaction handles spacing/case variants.
- [x] #6 Focused policy explain service/API/CLI tests pass.
- [x] #7 Bandit and diff checks pass for touched scope.
- [x] #8 Review feedback outcome is recorded and committed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified all nine current inline comments against the rebased branch. Still-valid findings addressed: missing module docstrings and accepted markers in new test modules; broad preview catalog fallback lacked diagnostics; admin policy explain routes lacked visible route-level rate limiting; preview mode exposed runtime/static metadata without differing computation semantics; preview CLI dropped session_id; runtime-effective policy simulation ran inline in async service methods; missing policy documents broke fallback tool-name extraction; command-shaped tool identifiers with spacing/case variants were not redacted.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall -q mcp_unified/gateway/policy_explain.py mcp_unified/gateway/fastapi.py mcp_unified/gateway/cli.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py -v` (68 passed)
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/policy_explain.py mcp_unified/gateway/fastapi.py mcp_unified/gateway/cli.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py -s B101 -f json -o /tmp/bandit_task2377.json` (0 findings)
- `git diff --check`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR 2373 review feedback with scoped fixes for policy explain tests, diagnostics, rate limiting, runtime preview semantics, CLI session propagation, offloaded simulation, fallback guards, and redaction hardening. Added focused regression coverage and updated policy explain docs for session-scoped preview grants.
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
