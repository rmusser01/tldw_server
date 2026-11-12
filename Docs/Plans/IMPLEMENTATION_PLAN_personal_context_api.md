# TASK-13145 — Authenticated Personal Context API

## Stage 1: Existing-contract analysis and RED evidence

**Goal**: Pin the server's authentication, dependency, router, and error
conventions in failing Personal Context service and endpoint tests.

**Success Criteria**: Focused tests fail only because the new service,
dependency, schemas, and routes do not exist.

**Tests**: `test_personal_context_service.py`,
`test_personal_context_endpoints.py`, and
`test_personal_context_auth_boundary.py`.

**Status**: Complete

## Stage 2: Canonical service boundary

**Goal**: Implement lifecycle, scope, record, proposal-review, runtime-policy,
export, and purge operations over the encrypted repository from TASK-13144.

**Success Criteria**: Service tests prove optimistic updates, semantic-key
uniqueness, bounded reads, profile isolation, and approved lifecycle behavior.

**Tests**: New service tests plus the TASK-13144 repository suite.

**Status**: Complete

## Stage 3: Authenticated API and stable errors

**Goal**: Resolve the authenticated user before repository construction and
expose strict, bounded request/response contracts through the canonical route
groups.

**Success Criteria**: Endpoint and auth-boundary tests cover every route,
cross-user not-found behavior, typed conflicts, locked state, payload/search
limits, and unknown-field rejection.

**Tests**: New endpoint/auth tests plus existing Personalization endpoints.

**Status**: Complete

## Stage 4: Security, regression, and review

**Goal**: Complete targeted regressions, static/security gates, documentation,
and independent review.

**Success Criteria**: Focused tests, Ruff/format, compilation, Bandit, diff
hygiene, and independent review pass; TASK-13145 is complete.

**Tests**: All Personal Context service/API tests and touched Personalization
regressions.

**Status**: Complete

## ADR check

ADR required: yes (existing)

ADR path: `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

Reason: The existing ADR already governs authenticated service ownership,
server authority, encryption, and cross-application contract semantics; this
task introduces no new architectural decision.
