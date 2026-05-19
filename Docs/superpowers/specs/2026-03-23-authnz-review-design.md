# AuthNZ Review Design

Date: 2026-03-23
Topic: Sequential AuthNZ review plan
Status: Approved design

## Objective

Review the AuthNZ surface in `tldw_server` in a controlled sequence so findings stay attributable to one subsystem at a time. The review should identify bugs, security issues, brittle assumptions, backend parity gaps, and practical hardening opportunities without mixing unrelated concerns across passes.

## Scope

The review covers five slices of the AuthNZ system:

1. Dependency and auth boundary wiring
2. Login, session, JWT, and MFA flows
3. API keys, virtual keys, budgets, and quotas
4. RBAC, org/team authorization, and admin protection
5. Whole-surface synthesis and remediation ordering

The review is read-first and stage-gated. It does not include code changes unless the user later asks to move from review into remediation.

## Approaches Considered

### Recommended: D -> A -> B -> C -> E

Start with the dependency layer and boundary wiring before reviewing individual auth features. This catches precedence bugs, bypass paths, and principal-shaping mistakes early, which makes later findings more trustworthy.

### Alternative: A -> B -> C -> D -> E

This follows feature groupings more literally, but it risks revisiting earlier conclusions if the auth boundary is flawed.

### Alternative: E overview -> D -> A -> B -> C

This gives a quick top-level risk map, but it duplicates effort and usually produces a weaker first pass.

## Chosen Sequence

### Stage 1: Boundary and Dependency Review

Goal: validate the auth boundary before trusting downstream behavior.

Primary areas:
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- auth principal resolver and adjacent auth dependency utilities
- tests covering precedence, test mode, principal construction, and dependency hardening

Review focus:
- header and token precedence
- single-user vs multi-user mode handling
- test-mode bypasses
- request state caching
- principal construction and sanitization
- auth-related fallback logic and exception handling

Deliverable:
- findings on bypass risk, unsafe precedence, dependency bugs, and cleanup candidates

### Stage 2: Login, Session, JWT, and MFA Review

Goal: validate the main authentication state machine.

Primary areas:
- `tldw_Server_API/app/api/v1/endpoints/auth.py`
- `jwt_service.py`
- `session_manager.py`
- `password_service.py`
- `mfa_service.py`
- related unit and integration tests

Review focus:
- login correctness
- refresh rotation
- logout and revocation
- forgot/reset flows
- email verification
- MFA setup, verify, login, and disable paths
- CSRF and session coupling where applicable

Deliverable:
- findings on broken transitions, token lifecycle issues, consistency bugs, and missing flow coverage

### Stage 3: API Key, Virtual Key, Budget, and Quota Review

Goal: verify issuance and enforcement paths for programmatic access.

Primary areas:
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py`
- `tldw_Server_API/app/core/AuthNZ/quotas.py`
- related tests for keys, budgets, and usage accounting

Review focus:
- key creation, rotation, and revocation
- scoping and allowlists
- virtual key limits
- budget enforcement timing
- quota and usage accounting integrity
- degraded behavior when usage logging or budget state fails

Deliverable:
- findings on overbroad key power, enforcement gaps, accounting drift, and hardening opportunities

### Stage 4: RBAC, Org, Team, and Admin Authorization Review

Goal: validate permission resolution and protected admin paths.

Primary areas:
- `rbac.py`
- `permissions.py`
- `org_rbac.py`
- `orgs_teams.py`
- admin endpoints and claim-first dependencies
- tests covering scoped permissions and admin enforcement

Review focus:
- claim-first protection
- scoped permission checks
- org and team inheritance
- overrides and deny behavior
- admin endpoint consistency
- risks from legacy shim removal or partial migration

Deliverable:
- findings on privilege escalation, inconsistent permission semantics, missing guards, and test matrix gaps

### Stage 5: Whole-Surface Synthesis

Goal: consolidate the earlier passes into one remediation-oriented picture.

Primary areas:
- outputs from stages 1 through 4
- cross-cutting settings, migration, and backend parity concerns

Review focus:
- repeated failure patterns
- hidden coupling across subsystems
- migration debt
- test coverage blind spots
- prioritized fix ordering by severity and leverage

Deliverable:
- final ranked issue list and a recommended remediation sequence

## Per-Stage Review Method

Each stage follows the same method:

1. Define the exact slice under review, including code, dependencies, and relevant tests.
2. Trace the request or enforcement path end to end.
3. Review against a fixed checklist:
   - security
   - correctness
   - privilege boundaries
   - state transitions
   - error handling
   - configuration sensitivity
   - backend parity
   - test coverage gaps
4. Verify with focused evidence from code, tests, and targeted validation commands when useful.
5. Produce a pass-specific report with findings first, ordered by severity, with file references and concrete remediation directions.

## Deliverables Per Stage

Each stage should produce:

- Findings: bugs, bypass risks, privilege escalation risks, consistency bugs, or brittle assumptions
- Coverage gaps: missing or weak tests
- Improvements: lower-risk hardening and cleanup opportunities
- Exit note: whether the next stage can proceed cleanly or should account for unresolved foundational risk

## Operating Rules

- Complete each stage before starting the next.
- Keep the review read-first and non-invasive unless the user later requests fixes.
- If a foundational issue found in one stage affects later stages, call it out explicitly and carry that context forward.
- The final synthesis stage should consolidate prior work, not repeat every earlier deep dive.

## Success Criteria

A stage is considered complete when:

- scope boundaries are explicit
- primary code paths and supporting tests were inspected
- any targeted validation commands used were recorded
- findings were ranked by severity with concrete file references
- test gaps were separated from confirmed bugs
- the stage ends with a clear recommendation for how the next stage should proceed

## Expected Outcome

This design yields a sequential, individual review of each major AuthNZ subsystem while preserving comparability across stages and producing a final prioritized remediation view instead of isolated notes.
