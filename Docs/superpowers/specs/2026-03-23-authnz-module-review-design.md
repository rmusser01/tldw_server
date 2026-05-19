# AuthNZ Module Review Design

Date: 2026-03-23
Topic: AuthNZ module review
Status: Approved design

## Objective

Review the AuthNZ module in `tldw_server` for concrete issues, bugs, risks, and improvement opportunities without drifting into endpoint wiring or downstream consumer behavior.

The review should prioritize actionable findings over exhaustive narration, preserve the difference between confirmed defects and softer risks, and produce a final report that is easy to use for remediation planning.

## Scope

This review is limited to:

- `tldw_Server_API/app/core/AuthNZ`
- `tldw_Server_API/tests/AuthNZ`

This review excludes:

- API endpoints outside the AuthNZ core module
- app wiring outside local AuthNZ dependencies
- fixes or refactors during the review itself unless the user explicitly requests remediation afterward

## Review Sequence

The review will proceed straight through four passes in this order:

1. Security
2. Correctness
3. Maintainability
4. Test Gaps

The output will be one consolidated report with four matching sections in the same order.

## Approaches Considered

### Recommended: Risk-first deep review with full coverage

Start with the highest-risk AuthNZ primitives such as token handling, sessions, password and MFA logic, secret storage, API keys, quotas, and authorization resolution. Then expand to full module coverage so low-level security assumptions are established before judging correctness and maintainability in less sensitive areas.

Why this is preferred:

- surfaces serious auth and secret-handling issues early
- gives later correctness judgments a more reliable foundation
- keeps the final report prioritized instead of merely exhaustive

### Alternative: Linear file-by-file sweep

Read every AuthNZ file in repository order and classify findings into the four final categories.

Trade-offs:

- simpler to execute mechanically
- more exhaustive in appearance
- weaker prioritization and more likely to mix critical issues with minor cleanup

### Alternative: Test-first review

Start from `tldw_Server_API/tests/AuthNZ`, infer intended behavior, and then inspect the implementation for drift.

Trade-offs:

- useful for contract mismatches
- good at identifying missing invariants
- weaker for latent security issues where tests already encode incomplete assumptions

## Chosen Method

Use a hybrid of the recommended risk-first deep review and full module coverage:

1. Inspect the highest-risk security-critical primitives first.
2. Use the corresponding AuthNZ tests as corroborating evidence rather than as the sole source of truth.
3. Continue through the rest of the module so the final review covers the full scoped surface.
4. Separate confirmed findings from probable risks and lower-priority improvements.

## Evidence Model

The review will rely on:

- direct source inspection in `tldw_Server_API/app/core/AuthNZ`
- direct test inspection in `tldw_Server_API/tests/AuthNZ`
- targeted static inspection commands where useful
- focused validation commands only when they materially strengthen or falsify a claim

The review is not planned as a full integration or end-to-end runtime exercise. Runtime verification may be used selectively, but only if it helps confirm a specific claim within the approved scope.

## Finding Categories

Each observation will be placed into one of these categories:

### Confirmed finding

A concrete bug, vulnerability, inconsistency, or design flaw supported directly by code, tests, or clear control/data-flow evidence.

### Probable risk

Something that appears unsafe, brittle, or likely wrong, but depends on assumptions, integration context, or behavior not fully proven inside the scoped review.

### Improvement

A maintainability, hardening, or refactoring opportunity that is worth doing but is not necessarily evidence of a current defect.

## Severity And Confidence

Severity and confidence will be reported separately.

- Severity communicates impact and urgency.
- Confidence communicates how directly the claim is supported by the reviewed evidence.

This prevents lower-confidence concerns from being overstated as confirmed bugs and prevents lower-severity cleanup items from crowding out higher-risk AuthNZ findings.

## Review Focus Areas

The review order inside the module should bias toward:

- JWT issuance, verification, and claims handling
- session lifecycle and revocation behavior
- password hashing, password policy, and reset/verification flows
- MFA setup, validation, and recovery behavior
- API key and virtual key creation, validation, and rotation
- secret storage and key material handling
- RBAC, permissions, org/team authorization logic inside the module
- quotas, rate limits, lockout, and budget enforcement logic
- configuration defaults, fail-open/fail-closed behavior, and migration-sensitive paths

## Testing Analysis Method

The `Test Gaps` pass should distinguish between:

- important invariants that are not tested at all
- tests that exist but are weak, overly coupled to implementation, or misleading
- implicit contracts relied on by the module that the test suite never makes explicit

The final testing analysis should emphasize which missing tests would most reduce security or regression risk first.

## Execution Boundaries

- The review remains inside the AuthNZ module and AuthNZ test suite.
- Cross-module behavior may be noted only when a local AuthNZ file clearly depends on or assumes it.
- The review remains read-first and non-invasive.
- Severe findings should include remediation guidance, but not code changes.

## Final Deliverable

The final report will contain four ordered sections:

1. `Security`
2. `Correctness`
3. `Maintainability`
4. `Test Gaps`

Within each section:

- confirmed high-severity findings come first
- lower-confidence risks come after confirmed issues
- improvement opportunities follow and are clearly labeled
- file and line references are included for actionable items

If a section has no confirmed issues, the report should say so explicitly and then note any residual risks or blind spots without padding the section.

## Success Criteria

The review is successful when:

- the scoped AuthNZ module and test suite are fully covered
- findings are separated cleanly by category and confidence
- the final report is organized by the four approved passes
- serious defects are easy to prioritize
- softer risks and improvements remain visible but do not obscure confirmed bugs
- test gaps identify the highest-leverage missing coverage first

## Constraints

- Do not broaden the review to downstream API endpoints in this run.
- Do not silently convert the review into implementation work.
- Do not present speculation as a confirmed bug.
- Do not let minor maintainability observations dominate the report.

## Expected Outcome

This design yields a focused review of the AuthNZ module that is broad enough to cover the full scoped surface, prioritized enough to surface the most important security and correctness issues first, and structured enough to support follow-up remediation planning without mixing hard defects and softer improvements.
