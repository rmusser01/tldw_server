## Stage 1: Regression Tests
**Goal**: Capture the ACP review findings as focused tests for launch validation, permission tiers, timeouts, queue bounds, URL safety, redaction, SSH key persistence, and unsupported egress allowlists.
**Success Criteria**: New tests fail against the current implementation for the risky behaviors.
**Tests**: Targeted pytest runs under `tldw_Server_API/tests/Agent_Client_Protocol/`.
**Status**: Complete

## Stage 2: Host Runner And Permission Hardening
**Goal**: Prevent unsafe non-sandbox launch inputs and make permission tier matching prefer destructive operations over broad substring auto-approval.
**Success Criteria**: Host session creation rejects unsafe cwd/env/stdio MCP inputs by default; permission tiers classify ambiguous/destructive tools conservatively.
**Tests**: ACP host-launch and permission tier tests.
**Status**: Complete

## Stage 3: Transport And Runtime Safety
**Goal**: Add bounded RPC waits, bounded update queues, MCP HTTP/SSE URL controls, and stderr/output redaction.
**Success Criteria**: Pending calls time out and clean up; update queues are capped; MCP transports reject local/cross-origin URLs; logs redact obvious secrets.
**Tests**: ACP client, stream transport, MCP transport, and queue tests.
**Status**: Complete

## Stage 4: Sandbox Controls
**Goal**: Stop persisting SSH private keys by default and fail fast when unsupported per-session egress allowlists are configured.
**Success Criteria**: Durable control metadata omits private keys unless explicitly enabled; unsupported egress allowlist config is rejected instead of silently ignored.
**Tests**: Sandbox runner configuration and metadata tests.
**Status**: Complete

## Stage 5: Verification
**Goal**: Run the focused ACP tests and security scan for touched paths.
**Success Criteria**: Targeted pytest suite passes and Bandit reports no new actionable findings in the touched scope.
**Tests**: Focused pytest plus Bandit JSON output.
**Status**: Complete

## Stage 6: PR Review Follow-Up
**Goal**: Rebase on current `dev` and address PR review comments from Qodo and Gemini.
**Success Criteria**: DNS-resolving SSRF checks, fail-closed cwd roots, stricter destructive permission tokens, redact-before-regex behavior, SSE URL cleanup, direct config field access, docstrings, and type hints are covered by focused tests.
**Tests**: Focused ACP pytest slice including config cwd tests plus Bandit on touched ACP code.
**Status**: Complete
