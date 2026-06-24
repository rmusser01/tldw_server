---
id: TASK-2424
title: Verify and remediate MCP Unified review findings
status: In Progress
assignee: []
created_date: '2026-06-23 18:27'
updated_date: '2026-06-24 04:50'
labels:
  - review
  - mcp
  - security
  - refactor
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify current MCP Unified module review findings, address validated issues with focused tests and security verification, and capture the protocol.py refactor brainstorming/spec workflow separately before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Validated review findings are recorded as accepted or rejected with evidence
- [ ] #2 Accepted bug/security/reliability findings have regression tests written before production changes
- [ ] #3 Focused MCP Unified tests, diff check, and Bandit touched-scope verification are recorded
- [ ] #4 protocol.py refactor brainstorming produces an approved design/spec before implementation planning
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification pass completed before implementation. Validated findings: MCP legacy refresh/revocation state is process-local; fs.write_text bypasses structured fs.write preimage/receipt protections and virtual CLI write maps to it; metadata category web/utility falls back to read rate bucket; fallback RBAC USER role wildcard-executes arbitrary tools; stale WebSocket cleanup removes connections without decrementing per-IP counts; MCP configure_logging removes existing global Loguru sinks; invalid tool names return INTERNAL_ERROR despite being invalid params. Focused existing suite: selected MCP tests passed 9/9 in 229.16s, with live WebSocket test slow but completed.

Implemented remediation for validated MCP Unified findings: gated legacy refresh behind demo auth, routed fs.write_text through structured preimage-checked writer, moved virtual CLI write to fs.write create mode, preserved network/utility metadata categories for rate limiting, narrowed fallback RBAC user/moderator tool execution, decremented WS per-IP counts during stale cleanup, preserved non-MCP Loguru sinks, and mapped invalid tool names to INVALID_PARAMS. Verification: focused MCP regression slice passed (21 passed); HTTP refresh gate tests passed (2 passed); touched modules py_compile passed; direct logging preservation check passed; Bandit on touched implementation files passed with 0 findings.

Protocol.py refactor brainstorming completed. Approved direction: security-pipeline extraction for tools/call, keeping MCPProtocol as JSON-RPC facade. Design spec written at Docs/superpowers/specs/2026-06-23-mcp-protocol-tool-execution-refactor-design.md and self-reviewed for placeholders, contradictions, scope drift, and ambiguity.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
