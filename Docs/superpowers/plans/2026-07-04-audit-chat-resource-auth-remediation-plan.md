## Stage 1: Route Scope Regression Coverage
**Goal**: Capture the audit finding as a focused route contract test for resource-spending Chat, RAG, document-generation, and embedding routes.
**Success Criteria**: The new test fails on current route decorators because affected routes do not declare `TokenScopeGuard` metadata.
**Tests**: `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_scoped_token_route_auth_chain.py -k resource_spending`
**Status**: Complete

## Stage 2: Shared Guard Wiring
**Goal**: Add route-level `TokenScopeGuard` dependencies without changing existing ownership, RBAC, rate-limit, billing, or provider checks.
**Success Criteria**: Each affected route has a stable logical endpoint ID and `count_as="call"` so scoped JWT/API-key endpoint constraints and max-call counters are evaluated.
**Tests**: Focused AuthNZ scoped-token tests plus affected contract coverage.
**Status**: Complete

## Stage 3: Verification And Tracking
**Goal**: Verify the branch and record the audit remediation in Backlog.md.
**Success Criteria**: Focused tests pass, whitespace diff check passes, Bandit reports no new production findings on touched backend files, and TASK-12140 records touched files and verification.
**Tests**: Focused pytest commands, `git diff --check`, Bandit touched-scope JSON report.
**Status**: Complete
