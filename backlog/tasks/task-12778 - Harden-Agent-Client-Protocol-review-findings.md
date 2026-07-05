---
id: TASK-12778
title: Harden Agent Client Protocol review findings
status: Done
assignee: []
created_date: 2026-06-23 18:09
updated_date: 2026-06-24 01:24
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
Hardened Agent_Client_Protocol current-code review findings and PR review follow-ups: host runner launch surfaces now require explicit policy opt-ins and fail closed when cwd roots are unset, permission tiers classify destructive/write/update tokens conservatively, RPC calls time out and clean pending futures, update queues and stream buffers are bounded, MCP HTTP/SSE transports reject unsafe endpoints by default with DNS-resolving host checks, stderr logging truncates before redaction, sandbox SSH private keys are not persisted by default, unsupported ACP sandbox egress allowlists fail fast, and review-requested docstrings/type hints/SSE cleanup are in place.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR review follow-up on 2026-06-24: rebased branch onto origin/dev and addressing review comments from Qodo/Gemini covering DNS-resolving SSRF validation, fail-closed cwd roots, stricter permission tokens, redact-before-regex, SSE URL cleanup, direct config field access, docstrings, and type hints.
PR review follow-up completed on 2026-06-24: rebased onto origin/dev at 2ecb83fa4 and addressed Qodo/Gemini comments. Added DNS-resolving MCP HTTP host blocking with a bounded cache, fail-closed behavior when host runner cwd roots are unset, stricter destructive permission tokens, redact-before-regex output truncation, SSE post URL resolution cleanup, direct ACPRunnerConfig field access, module/function docstrings, and missing test type hints.
Verification: focused ACP pytest slice including test_acp_config_cwd.py passed: 109 passed, 4 warnings in 0.57s. Bandit on the touched ACP code files wrote /tmp/bandit_acp_pr_review_touched.json with 0 results and 0 errors. Full ACP-directory Bandit scan still reports one pre-existing low finding in tldw_Server_API/app/core/Agent_Client_Protocol/events.py (B105 token_usage), which is outside the touched PR scope.
Final rebase correction on 2026-06-24: origin/dev advanced again during review-fix work, so the branch was rebased onto 4fb8eafd5 before push. Re-ran the focused ACP pytest slice after that rebase: 109 passed, 4 warnings in 0.74s. Re-ran Bandit on touched ACP code files after that rebase: /tmp/bandit_acp_pr_review_rebased_touched.json completed with exit 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
