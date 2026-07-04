# Audit AuthNZ Impersonation Remediation Plan

## Stage 1: Regression Coverage
**Goal**: Capture the three audit findings as failing focused tests.
**Success Criteria**: Tests fail because impersonation tokens use the normal TTL, raw connection acquisition is used for lookup SQL, audit emission is missing, and impersonation metadata is not carried into `AuthContext`.
**Tests**: `python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py -q`
**Status**: Complete

## Stage 2: Token And Endpoint Hardening
**Goal**: Make the impersonation endpoint issue an explicitly short-lived access token, use backend-neutral DB helper calls, and emit a durable audit event for issuance.
**Success Criteria**: Endpoint tests prove the JWT call includes the impersonation TTL, the endpoint never calls `pool.acquire()` for user/role lookups, and the audit helper receives actor and subject metadata.
**Tests**: `python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py -q`
**Status**: Complete

## Stage 3: Auth Context Propagation
**Goal**: Preserve impersonation actor metadata when decoding access tokens and constructing request auth context.
**Success Criteria**: `AuthPrincipal` and `AuthContext` expose impersonation state for downstream audit hooks without changing normal user/API-key principals.
**Tests**: `python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py -q`
**Status**: Complete

## Stage 4: Verification And Task Finalization
**Goal**: Run focused tests, Bandit on touched production files, diff checks, and update Backlog task `TASK-12139`.
**Success Criteria**: Focused tests pass, Bandit has no new production findings, `git diff --check` is clean, and the Backlog task records verification and summary.
**Tests**: focused pytest, `python -m bandit -r <touched production files> -f json -o /tmp/bandit_audit_auth_impersonation.json`, `git diff --check`
**Status**: Complete
