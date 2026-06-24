---
id: TASK-12008
title: Harden Security module review findings
status: Done
assignee: []
created_date: '2026-06-23 21:17'
updated_date: '2026-06-24 19:07'
labels:
  - security
  - hardening
  - review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current-code Security module review findings: DNS rebinding/check-then-use hardening, trusted-proxy setup client-IP parsing, strict AES key validation, explicit auth/webhook secret configuration, setup CSP nonce/dead-code cleanup, and DNS timeout race cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Egress policy detects private/reserved DNS drift when pinned resolution is reused.
- [x] #2 Setup access guard resolves proxied client IPs without trusting spoofed leftmost X-Forwarded-For.
- [x] #3 Crypto helpers fail closed on invalid configured AES keys instead of deriving keys silently.
- [x] #4 Secret manager has explicit strong configs for JWT and webhook helper secrets.
- [x] #5 Setup CSP removes ineffective nonce-injection behavior or wires it correctly, with tests updated.
- [x] #6 Focused Security tests and Bandit touched-scope scan pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Add failing regression tests for each reviewed behavior.
- Implement minimal Security module changes to satisfy the tests.
- Update central HTTP egress validation to use the safer pinned-resolution policy.
- Run focused tests and Bandit on touched production files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Manual Backlog task creation was approved because Backlog MCP resources were unavailable and `backlog search`, `backlog task list`, and `backlog task create` hung in this workspace.
- Added pinned DNS drift detection to egress policy and updated the central HTTP client to reuse cached resolutions as pins for later policy checks.
- Replaced process-global DNS timeout mutation with a bounded daemon-thread resolver wrapper.
- Updated trusted-proxy setup guard parsing to walk `X-Forwarded-For` from the proxy side and return the first untrusted client IP.
- Made AES-GCM JSON helper key loading strict for configured and explicit keys.
- Added explicit strong secret manager configs for JWT and webhook master secrets.
- Removed ineffective setup CSP nonce generation/injection state and updated tests/docs around the intentionally relaxed setup CSP.
- Moved the finished work to a clean worktree from `origin/dev` on branch `codex/security-module-review-fixes-12008`.
- Rebasing follow-up: addressed Qodo review comments by adding test classification markers/import cleanup, moving async egress validation off the event loop, replacing per-call DNS resolver threads with a bounded executor, logging invalid AES key configuration without leaking key material, aligning docs/tests to strict 32-byte AES-256 keys, skipping malformed trusted-proxy XFF entries, and reading JWT secrets from `[AuthNZ]`.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `env SINGLE_USER_API_KEY=0123456789abcdef0123456789abcdef python -m pytest --confcutdir=tldw_Server_API/tests/Security tldw_Server_API/tests/Security -q` -> 66 passed.
- `python -m pytest --confcutdir=tldw_Server_API/tests/http_client tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_truthiness_flags.py -q` -> 24 passed.
- `python -m bandit -r ...touched production files... -f json -o /tmp/bandit_security_12008_worktree.json` -> 0 errors, 0 findings.
- `git diff --check` -> passed.
- Follow-up after rebase on latest `origin/dev`:
  - `env SINGLE_USER_API_KEY=0123456789abcdef0123456789abcdef /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/Security tldw_Server_API/tests/Security -q` -> 69 passed, 7 warnings.
  - `env SINGLE_USER_API_KEY=0123456789abcdef0123456789abcdef /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/http_client tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_truthiness_flags.py -q` -> 25 passed, 2 warnings.
  - `env SINGLE_USER_API_KEY=0123456789abcdef0123456789abcdef /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/External_Sources/test_reference_manager_storage.py::test_oauth_state_metadata_is_encrypted_at_rest_when_crypto_enabled -q` -> 1 passed, 9 warnings.
  - `env SINGLE_USER_API_KEY=0123456789abcdef0123456789abcdef /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/Security/crypto.py tldw_Server_API/app/core/Security/secret_manager.py tldw_Server_API/app/core/Security/setup_access_guard.py tldw_Server_API/app/core/http_client.py -f json -o /tmp/bandit_security_12008_followup.json` -> 0 errors, 0 findings.
  - `git diff --check` -> passed.
  - `rg -n "WORKFLOWS_ARTIFACT_ENC_KEY.*16/24/32|base64 16/24/32|reference-manager-test-key|config_section=\"Auth\"" Docs tldw_Server_API/tests tldw_Server_API/app/core/Security -g '*.md' -g '*.py'` -> no matches.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the current Security module review findings across egress, setup proxy access control, crypto key validation, secret configuration, and setup CSP cleanup. Regression coverage was added for the reviewed behaviors, focused Security/HTTP client tests pass, and touched production files have a clean Bandit scan.

Follow-up review comments were addressed after rebasing on the latest `dev`: async HTTP egress validation now leaves the event loop, egress DNS resolution uses a bounded executor, invalid AES env keys emit explicit non-secret error logs, AES docs/tests require strict base64 32-byte keys, malformed trusted-proxy XFF entries are skipped instead of falling back to the proxy peer, JWT secret config lookup uses `[AuthNZ]`, and new/modified tests carry unit markers with corrected imports.
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
