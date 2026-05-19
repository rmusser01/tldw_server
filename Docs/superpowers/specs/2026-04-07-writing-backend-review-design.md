# Writing Backend Review Design

Date: 2026-04-07
Topic: Backend-only review of the Writing module in `tldw_server`

## Goal

Produce an evidence-based backend review of the Writing module that identifies:

- correctness bugs and edge-case failures
- data integrity and optimistic-locking/versioning risks
- security and input-validation issues
- contract mismatches between endpoints, schemas, and persistence helpers
- maintainability and performance concerns that can plausibly create production defects
- missing or misleading automated tests

The review is intended to prioritize concrete findings over stylistic commentary.

## Scope

This review covers the backend Writing surface area, centered on:

- `tldw_Server_API/app/api/v1/endpoints/writing.py`
- `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- `tldw_Server_API/app/api/v1/schemas/writing_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
- related persistence helpers only where the listed Writing endpoints depend on them directly, especially `ManuscriptDBHelper` and `CharactersRAGDB` call paths used by writing and manuscript routes
- related backend tests under `tldw_Server_API/tests/Writing/`

The review includes API behavior, schema handling, auth and rate-limit boundaries, error mapping, create/update/delete and reorder flows, snapshot import/export behavior, analysis endpoints, tokenizer and capability metadata behavior, wordcloud job/cache behavior, and any backend coupling that can affect correctness or data integrity.

## Non-Goals

This review does not cover:

- frontend Writing Playground code in `apps/`
- visual UX issues except where backend contracts create downstream bugs
- implementing fixes during the review phase
- broad unrelated refactoring outside the Writing backend surface
- deep provider-specific internals unless a Writing endpoint depends on them incorrectly

## Approaches Considered

### 1. Risk-first audit

Inspect the highest-risk runtime paths first, then use tests as supporting evidence.

Strengths:

- fastest way to surface real bugs and integrity issues
- strong fit for a code-review style output
- aligns with the user goal of finding issues, bugs, and potential improvements

Weaknesses:

- can miss lower-priority blind spots unless test coverage is checked explicitly

### 2. Coverage-first audit

Map the Writing test suite first, then inspect weakly tested or untested branches.

Strengths:

- good at exposing false confidence and coverage gaps
- efficient for identifying where additional tests would pay off

Weaknesses:

- weaker at finding design-level defects when current tests encode flawed behavior

### 3. Boundary-first audit

Focus first on endpoint contracts, schema validation, auth/rate limits, and DB boundaries.

Strengths:

- strong for integration inconsistencies and caller-visible defects
- useful for security and reliability issues

Weaknesses:

- can underweight internal maintainability and branching problems inside the large endpoint modules

## Recommended Approach

Use a hybrid review led by the risk-first audit.

Execution order:

1. inspect endpoint contracts and dependency boundaries
2. trace state-changing flows and persistence assumptions
3. inspect analysis, tokenizer/capability, and wordcloud helper behavior that can fail silently or misreport results
4. compare findings against current test coverage and identify gaps

This balances bug-finding speed with enough coverage analysis to make the output actionable.

## Review Method

### Pass 1: Entry points and contracts

Inspect:

- route definitions and parameter handling
- request and response schemas, aliases, defaults, and normalization
- auth, dependency, and rate-limit boundaries
- error handling and HTTP mapping

Primary questions:

- do endpoint contracts match actual behavior?
- are invalid states rejected early and consistently?
- do error responses preserve useful semantics for callers?

### Pass 2: State and persistence

Inspect:

- create, update, delete, restore, and fetch flows
- optimistic locking, reorder semantics, and version handling
- soft-delete and not-found behavior
- snapshot import/export merge and replace behavior
- create-then-read assumptions and helper return-value handling

Primary questions:

- can data be corrupted, lost, duplicated, or silently reshaped?
- are concurrent edits handled consistently?
- do state transitions preserve invariants?

### Pass 3: Analysis, capability, and async helper behavior

Inspect:

- manuscript analysis request validation and result shaping
- tokenizer and detokenizer resolution behavior
- provider capability and compatibility lookups used by Writing endpoints
- wordcloud job lifecycle, caching, and response shaping
- fallback paths that may return misleading metadata or silently degrade behavior

Primary questions:

- can helper failures be misreported or masked?
- are capability claims accurate for clients?
- are failure modes bounded and explicit?

### Pass 4: Test adequacy

Inspect:

- unit and integration tests under `tldw_Server_API/tests/Writing/`
- mismatches between critical branches and test coverage
- tests that assert incidental behavior but miss invariants

Primary questions:

- which risky paths are untested or under-tested?
- which existing tests might give false confidence?

## Review Criteria

Each potential issue is evaluated against these categories:

- correctness and edge-case safety
- data integrity and version/concurrency behavior
- security and input-validation rigor
- maintainability and code organization
- performance and unnecessary work
- test coverage and test quality

## Evidence Standard

The review should avoid speculative claims. A finding should be backed by at least one of:

- a concrete code path that produces incorrect or risky behavior
- an invariant mismatch between layers
- an unguarded failure mode
- a missing test around a critical branch or state transition

Ambiguous items should be labeled as open questions or assumptions rather than overstated as defects.

## Deliverable Format

The final review output should be organized as:

1. findings first, ordered by severity
2. open questions or assumptions
3. short coverage summary
4. lower-priority improvement opportunities

Each finding should include:

- severity (`High`, `Medium`, or `Low`)
- type (`correctness`, `security`, `performance`, `maintainability`, or `test gap`)
- impact
- concrete reasoning
- file reference(s)

## Severity Model

- `High`: likely bug, data loss/corruption risk, security issue, or major behavioral regression risk
- `Medium`: correctness edge case, contract inconsistency, or meaningful maintainability/performance problem
- `Low`: smaller cleanup, localized code smell, or minor missing-test issue

## Constraints and Assumptions

- This phase is analysis-only; no code changes are part of the review deliverable.
- The review will prioritize local code and tests over stale docs.
- Large endpoint files may contain duplicated logic; duplication alone is not a finding unless it creates correctness, drift, or maintenance risk.
- If a behavior is unclear from local code and tests, that uncertainty should be called out explicitly.

## Success Criteria

This design is successful if it produces a backend Writing review that:

- is scoped tightly enough to stay on the backend surface
- is rigorous enough to produce ranked, defensible findings
- highlights not just bugs but also integrity, contract, and coverage risks
- gives the user a practical map of what is solid versus what needs attention
