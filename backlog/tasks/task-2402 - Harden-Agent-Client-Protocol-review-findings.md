---
id: TASK-2402
title: Harden Agent Client Protocol review findings
status: Done
assignee: []
created_date: '2026-06-23 18:09'
updated_date: '2026-06-23 18:28'
labels:
  - acp
  - security
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current-code review findings in tldw_Server_API/app/core/Agent_Client_Protocol: session launch input hardening, permission tier classification, RPC timeouts, bounded update queues, MCP HTTP/SSE SSRF controls, stderr redaction, sandbox SSH key handling, and ACP sandbox egress config alignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Non-sandbox ACP session creation rejects unsafe cwd/env/MCP server launch surfaces unless explicitly allowed by policy.
- [x] #2 Permission tier resolution cannot auto-approve destructive or execution-like tool names through read/get/list substrings.
- [x] #3 ACP stdio and stream RPC calls time out and clean pending futures.
- [x] #4 Runner update queues and stream buffers are bounded.
- [x] #5 MCP HTTP/SSE transports reject unsafe endpoints and same-origin SSE post URL violations.
- [x] #6 Downstream stderr logging is redacted and truncated.
- [x] #7 Sandbox SSH private-key handling is encrypted or bounded by config and documented in code.
- [x] #8 ACP sandbox allowed egress configuration is either enforced through sandbox policy or removed from the active contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ACP hardening plan docs/superpowers/plans/2026-06-23-acp-agent-client-protocol-hardening.md. Added shared hardening helpers for URL validation, redaction, bounded queues, and launch validation; wired host runner, stream/stdio clients, MCP transports, and sandbox metadata handling.
Verification: focused ACP pytest slice with local confcutdir passed: 86 passed, 4 warnings in 10.78s. Bandit on touched ACP code wrote /tmp/bandit_acp_hardening.json with 0 results and 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Agent_Client_Protocol current-code review findings: host runner launch surfaces now require explicit policy opt-ins, permission tiers classify destructive tokens conservatively, RPC calls time out and clean pending futures, update queues and stream buffers are bounded, MCP HTTP/SSE transports reject unsafe endpoints by default, stderr logging is redacted, sandbox SSH private keys are not persisted by default, and unsupported ACP sandbox egress allowlists now fail fast.
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
