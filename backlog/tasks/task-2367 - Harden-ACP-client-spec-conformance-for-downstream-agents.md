---
id: TASK-2367
title: Harden ACP client spec conformance for downstream agents
status: Done
labels:
- ACP
- protocol
- backend
priority: High
modified_files:
- Docs/superpowers/plans/2026-06-16-acp-client-spec-conformance.md
- tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py
- tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py
- tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py
- tldw_Server_API/app/core/Agent_Client_Protocol/stdio_client.py
- tldw_Server_API/app/core/Agent_Client_Protocol/stream_client.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_request_schema.py
- tools/tldw-agent/internal/acp/runner.go
- tools/tldw-agent/internal/acp/runner_test.go
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first ACP protocol-hardening slice from the spec audit: standard session close support, stricter session setup validation, and MCP server transport/schema handling for downstream agents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP runner and Python clients use standard session/close where supported, with compatibility fallback for older runner private close method.
- [x] #2 Runner forwards or rejects MCP server transports according to ACP mcpCapabilities instead of silently passing unsupported HTTP/SSE transports downstream.
- [x] #3 API session setup rejects invalid cwd and malformed MCP server configs with clear 4xx validation errors.
- [x] #4 Focused Python and Go tests cover the new close and validation behavior.
- [x] #5 Touched Python and Go scope passes focused tests (>38 tests) and Bandit security analysis with zero findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-16-acp-client-spec-conformance.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reopened to address PR #2372 review comments after rebasing the branch onto latest dev.
- Fixed schema path/url normalization and env/header null handling review findings.
- Fixed runner MCP type validation and Backlog readability review findings.
- Stabilized an ACP audit endpoint test whose fixed timestamp had aged outside the default retention window.
- Verification completed: `python -m compileall -q ...` on touched Python modules passed; `python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_request_schema.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py` passed with 73 tests; `go test ./internal/acp -count=1` passed from `tools/tldw-agent`; Bandit touched-scope run wrote `/tmp/bandit_acp_client_spec_conformance_review.json` with zero results/errors; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ACP client spec-conformance hardening slice. Python standard and sandbox clients now call standard session/close first and fall back to private _tldw/session/close only for method-not-found compatibility. The Go ACP runner accepts standard session/close, stores downstream capabilities per session, forwards close when supported, and rejects HTTP/SSE MCP server transports unless the downstream agent explicitly advertises matching mcpCapabilities. Public ACP session setup schemas now validate absolute cwd, support stdio/http/sse/websocket MCP transport shapes, require absolute stdio commands and URL-based transport URLs, and normalize env/header dicts to ACP name/value arrays. Added focused Python and Go tests plus endpoint validation tests.
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
