# Implementation Plan: bgstream_webui_auth

## Stage 1: Reproduce Stream Auth Failure
**Goal**: Capture the WebUI streaming fallback bug with a focused automated test.
**Success Criteria**: A targeted test fails because direct stream fallback uses incorrect WebUI transport/auth behavior.
**Tests**: `bunx vitest run apps/packages/ui/src/services/__tests__/background-proxy.test.ts -t "uses hosted WebUI stream transport without browser auth headers"`
**Status**: Complete

## Stage 2: Align Stream Fallback With Shared Request Transport
**Goal**: Update direct stream fallback to follow the same WebUI transport and auth rules as normal requests.
**Success Criteria**: Hosted/quickstart/advanced WebUI stream requests resolve the correct URL and auth behavior.
**Tests**: Existing targeted Vitest coverage for `background-proxy` plus the new failing test.
**Status**: Complete

## Stage 3: Verify Behavior And Security
**Goal**: Confirm the fix with targeted tests and Bandit on touched scope.
**Success Criteria**: Relevant Vitest tests pass and Bandit reports no new findings in touched files.
**Tests**: `bunx vitest run apps/packages/ui/src/services/__tests__/background-proxy.test.ts` and `source .venv/bin/activate && python -m bandit -r apps/packages/ui/src/services -f json -o /tmp/bandit_bgstream_webui_auth.json`
**Status**: Complete
