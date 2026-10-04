# Domain Cache Ownership Implementation

Task: TASK-13425
Spec: Docs/superpowers/specs/2026-10-04-domain-cache-ownership-design.md

ADR required: no. Reuse native authority comparison and account-boundary events;
no new authentication or persistence rule.

## Stage 1: Reproduce
**Goal**: Demonstrate current-dev shared-cache ownership failures.
**Success Criteria**: Actual domain/client tests fail for stale cached values,
in-flight joins, or completion effects across native boundaries.
**Tests**: JWT/API-key/server/org/cookie boundaries and delayed transport.
**Status**: Complete

Evidence: initial current-dev suite had 84 ownership failures and 12 passing
same-owner compatibility cases. Subsequent self-review race tests demonstrated
12 post-await failures and 6 stale configuration-failure cleanup failures before
their respective fixes.

## Stage 2: Fence
**Goal**: Bind shared domain caches to current connection authority and epoch.
**Success Criteria**: All new regressions pass; owned/fresh bypass is unchanged;
old completion cannot remove a newer in-flight request.
**Tests**: Regression file, saved-profile scope tests, related client suites.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Review and publish a generic upstream-only fix against dev.
**Success Criteria**: Focused tests, typecheck results, security/ADR assessment,
task notes, commit, push, and PR are recorded; no private artifacts published.
**Tests**: Focused Vitest, TypeScript, diff/privacy self-review.
**Status**: In Progress

Verification basis: upstream dev `502da5bf0ccd1bc3aa4323e0d0fc430f36821a78`.
All 417 tests in ten focused suites passed, including 114 ownership regressions.
Focused TypeScript comparison reports the same seven baseline errors and zero
introduced diagnostics. Full UI checking exceeded Node's default heap; the
focused check completed with an 8 GiB limit. ESLint reports zero errors and the
same 830 warnings as baseline; the new regression file has no warnings.
Bandit is unavailable in the project venv and is not applicable to the
TypeScript-only touched scope. No new auth protocol or credential persistence is
introduced. Diff whitespace validation passed. Independent upstream-diff review
completed and identified a lower-level GET coalescing gap. Six real-transport
regressions failed with the new API-key owner receiving the previous owner's
response. Fenced reads now pass the existing native `configSnapshot`, preventing
transport-only joins while retaining domain single-flight. The updated ten-suite
matrix passes 423 tests; the ownership/scope subset passes 168 tests. Focused
TypeScript still reports the same seven baseline diagnostics and zero introduced
errors. Draft PR: https://github.com/rmusser01/tldw_server/pull/3170. Follow-up
publication and conversion to ready for review remain to be recorded.
