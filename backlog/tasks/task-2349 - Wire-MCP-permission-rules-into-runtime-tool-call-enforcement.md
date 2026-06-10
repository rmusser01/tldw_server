---
id: TASK-2349
title: Wire MCP permission rules into runtime tool-call enforcement
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-10 06:07'
labels:
  - mcp
  - profiles
  - permissions
  - runtime
  - agentic-execution
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first runtime enforcement slice for compiled MCP profile permission_rules so profile-scoped tool calls can evaluate relevant tool/path/domain/command/MCP subjects before execution. Keep scope minimal, test-first, and preserve existing legacy allowed_tools/denied_tools behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime tool-call authorization evaluates compiled permission_rules for the applicable subject family without bypassing existing exact tool allow/deny behavior.
- [x] #2 Denied permission-rule decisions block execution with an explainable, redacted error payload; ask decisions remain non-callable until approval support exists.
- [x] #3 Focused tests cover allowed legacy tools plus permission-rule denial for at least one runtime tool-call path.
- [x] #4 Docs or task notes record remaining deferred surfaces such as approval prompts, hooks, and shell alias parsing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the first standalone gateway runtime enforcement slice in mcp_unified.gateway.profile_runtime. The gateway still runs existing legacy allowed_tools/denied_tools and capability checks first. Once a backend tool is allowed, it compiles profile permission_rules and evaluates extracted tool, path, domain, command, and mcp subjects from the call arguments. Matched deny rules raise GatewayPolicyDenied with status=denied. Matched ask rules raise GatewayPolicyDenied with status=approval_required until approval prompts are wired. Unmatched default-deny decisions from permission-rule evaluation are ignored so path/domain/command rules do not grant tool execution. Denial provenance is redacted to profile_id, tool_name, subject_type, and matched rule metadata; raw path, URL, command, and argument values are not included.

TDD evidence:
- Baseline before edits: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q` passed with 243 tests.
- Red tests: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q -k "permission_rule"` failed because calls succeeded instead of returning policy denials.
- Green tests: same command passed with 3 tests.

Verification:
- Focused regression: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py -q` passed with 246 tests.
- Ruff: `python -m ruff check mcp_unified/gateway/profile_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` passed.
- Compile smoke: `python -m compileall -q mcp_unified/gateway/profile_runtime.py` passed.
- Bandit: `python -m bandit -r mcp_unified/gateway/profile_runtime.py -f json -o /tmp/bandit_mcp_permission_runtime_enforcement.json` passed.
- Whitespace: `git diff --check` passed.

Deferred: approval prompt/lease flow for ask decisions, hook integration, richer shell alias parsing, deeper tool-specific argument extraction, and wiring similar enforcement into the older in-process tldw_Server_API MCP protocol path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired compiled MCP profile permission_rules into standalone gateway runtime calls as an additional deny/ask enforcement layer after existing legacy tool policy allows execution. Added coverage for path, domain ask, and MCP wildcard runtime denials with redacted error payloads.
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
