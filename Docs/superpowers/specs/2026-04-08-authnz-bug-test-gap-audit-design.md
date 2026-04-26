# AuthNZ Bug And Test-Gap Audit Design

Date: 2026-04-08
Topic: Bug-and-test-gap audit of AuthNZ runtime behavior
Status: Revised after spec self-review

## Overview

This review will audit the AuthNZ subsystem in `tldw_server` for bugs, unsafe edge cases, and missing or weak tests. The goal is not a broad style review. The goal is a prioritized findings report that identifies concrete correctness and security risks, then maps those risks to the test coverage that should exist.

## Goals

- Produce a prioritized set of findings focused on real bugs, regressions, and unsafe behavior.
- Include missing-test and weak-test findings when they materially increase regression risk.
- Cover the runtime path from request authentication through authorization and stateful session/key handling.
- Keep findings grounded in exact code references and current project behavior.

## Non-Goals

- No broad refactor proposal unless it directly addresses a bug pattern or persistent test gap.
- No style-only review.
- No review of unrelated modules outside AuthNZ and its directly related auth/admin entry points.
- No remediation or code changes during the audit unless explicitly requested in a later step.

## Audit Scope

The audit covers:

- `tldw_Server_API/app/core/AuthNZ/`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Related auth and admin API endpoints that depend on AuthNZ enforcement paths
- AuthNZ integration, unit, and property tests that exercise those paths

Initial hotspot files should include at least:

- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/rbac.py`
- `tldw_Server_API/app/core/AuthNZ/settings.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/auth.py`

Representative endpoint and test entry points should include at least:

- auth endpoints under `tldw_Server_API/app/api/v1/endpoints/auth.py`
- representative admin/authz endpoints that materially depend on claim-first enforcement
- `tldw_Server_API/tests/AuthNZ/integration/`
- `tldw_Server_API/tests/AuthNZ/unit/`
- `tldw_Server_API/tests/AuthNZ/property/`

Representative admin/authz endpoint selection should favor:

- endpoints that issue, refresh, revoke, or invalidate credentials
- endpoints that mutate RBAC, roles, permissions, or admin-only state
- endpoints whose protection depends on shared AuthNZ dependencies rather than local ad hoc checks
- an adjacent non-auth endpoint only when it is needed to confirm that AuthNZ enforcement is actually applied at a real caller boundary

Expansion beyond the initial hotspot set should be justified by one or more of:

- high fan-out into other AuthNZ flows
- recent churn or repeated regressions
- backend divergence between SQLite and PostgreSQL
- privileged or externally reachable behavior
- unusually large or multi-responsibility implementations

The audit will treat the following areas as highest-risk surfaces:

- Single-user vs multi-user auth branching
- JWT parsing, validation, refresh, and revocation
- Session lifecycle and blacklist behavior
- API key validation, rotation, and scope enforcement
- Admin and privileged endpoint protection
- RBAC and permission resolution
- Lockout, rate limit, quota, and governor interactions
- Test mode and environment guardrails

## Review Method

The review will use a risk-driven method with a test-gap overlay.

### Pass 1: Trust Boundary Review

Inspect the request entry points where principals are resolved and privileged routes are gated. Focus on authentication precedence, token/key parsing, mode-specific behavior, and any branches that could bypass intended checks.

### Pass 2: Stateful Flow Review

Trace flows that depend on mutable state:

- login
- refresh rotation
- session validation and revocation
- token blacklist usage
- API key lifecycle
- lockout and rate-limit state transitions

The purpose of this pass is to catch correctness bugs that emerge only when state changes across requests.

### Pass 3: Authorization Review

Inspect how permissions are derived, propagated, and enforced across auth/admin endpoints. Focus on privilege escalation risks, incorrect default behavior, inconsistent permission checks, and org/team/admin edge cases.

### Pass 4: Test-Gap Review

Compare the high-risk behaviors from the first three passes against current AuthNZ tests. Flag:

- behavior with no direct test coverage
- tests that assert happy paths but not failure paths
- tests that mock away the exact logic that should be verified
- coverage that exists at unit level but not at integration boundaries

### Pass 5: Evidence Cross-Check

Use recent git history, the most relevant docs, and targeted verification only where they materially change confidence in a suspected issue. This pass exists to prevent two failure modes:

- reporting stale-doc mismatches as defects when the runtime contract already changed intentionally
- reporting probable risks as confirmed bugs when the critical branch or backend divergence has not been validated

## Prioritization Model

Findings will be ranked by operational impact:

- High: auth bypass, privilege escalation, revocation failure, broken admin enforcement, unsafe test-mode or environment behavior
- Medium: correctness bugs that deny valid access, break refresh or lockout flows, or create inconsistent authorization outcomes
- Low: missing tests, brittle abstractions, duplicate enforcement paths, or maintainability issues likely to cause future regressions

## Finding Format

Each finding should include:

- title
- severity
- confidence
- classification
- affected file and exact line reference
- concrete failure mode
- reason the behavior is risky or incorrect
- current test coverage status
- specific test that should be added or strengthened

Classification should be one of:

- Confirmed finding
- Probable risk
- Improvement

## Evidence Standards

- Prefer direct code-path evidence over inference.
- Use tests to confirm intended behavior when the implementation alone is ambiguous.
- Treat inconsistencies between endpoint wiring, dependency logic, and test assumptions as findings when they could hide real regressions.
- Perform targeted runtime verification when a high-risk claim depends on stateful behavior, backend divergence, or ambiguous guard execution and the answer is feasible to verify locally.
- Be explicit when a conclusion is limited by unavailable runtime verification or fixture constraints.
- Keep the final output focused on findings first. Any summary should be secondary.

Targeted runtime verification is especially appropriate for:

- refresh rotation and revocation behavior
- lockout and rate-limit state transitions
- SQLite versus PostgreSQL divergence on the same reviewed path
- test-mode or environment-guard branches
- endpoint-level enforcement where the dependency chain is unclear from static inspection alone

## Deliverable

The resulting review should be a concise, prioritized audit report with:

1. Confirmed findings ordered by severity
2. Probable risks or open questions that materially affect confidence
3. Improvements that would reduce future AuthNZ regression risk
4. Brief residual-risk notes where coverage or runtime verification remains incomplete
5. A coverage note listing:
   - hotspot files reviewed directly
   - files or tests only spot-checked
   - notable scoped surfaces not reviewed deeply and why

## Success Criteria

This audit is successful if it:

- identifies real, code-backed AuthNZ risks instead of generic advice
- distinguishes bugs from lower-signal cleanup suggestions
- maps important gaps in tests to the behaviors they fail to protect
- gives the maintainer a clear order of operations for follow-up fixes
