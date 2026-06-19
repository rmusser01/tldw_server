---
id: TASK-2387
title: Address MCP smoke harness final review findings
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-19 17:02'
labels:
  - mcp
  - testing
  - smoke-client
  - review-followup
dependencies: []
references:
  - backlog/completed/task-2387 - Design-MCP-smoke-client-transport-harness.md
documentation:
  - >-
    Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md
  - >-
    Docs/superpowers/plans/2026-06-19-mcp-smoke-client-transport-harness-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix still-valid final review issues in the MCP smoke client transport harness: stdio fixture baseline pass, notification no-response assertions, response size caps, and JSON report schema alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stdio fixture baseline scenario passes through the CLI over subprocess stdio.
- [x] #2 notifications/initialized response violations are detected by the smoke scenario where transports can observe them.
- [x] #3 Live transport response size caps produce bounded structured diagnostics before full parsing.
- [x] #4 JSON report output includes the documented top-level and per-step compatibility fields while preserving existing redacted/bounded fields.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the final MCP smoke harness review findings. The stdio fixture now passes the baseline scenario over subprocess stdio, unknown tool calls return method-not-found, advertised resources/prompts are implemented, initialized notifications fail if a transport observes a response, HTTP/WebSocket/stdio responses are size-capped before JSON parsing with response_too_large diagnostics, and reports include documented compatibility fields while preserving existing redacted/bounded fields. Verification: python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q passed with 68 tests; python -m py_compile mcp_unified/smoke/*.py tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py passed; python -m ruff check mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py passed; python -m bandit -r mcp_unified/smoke -f json -o /tmp/bandit_mcp_smoke_client.json reported 0 findings; git diff --check passed.
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
