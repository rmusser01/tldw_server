# AuthNZ Module Review Design

Date: 2026-04-07
Topic: AuthNZ module review
Status: Approved design

## Objective

Review the AuthNZ module in `tldw_server` for concrete issues, bugs, risks, and improvement opportunities without drifting into endpoint wiring or downstream consumer behavior.

The review should prioritize actionable findings over exhaustive narration, preserve the difference between confirmed defects and softer risks, and produce a remediation-oriented report that is immediately usable for follow-up work.

## Scope

This review is limited to:

- `tldw_Server_API/app/core/AuthNZ`
- primary test evidence in `tldw_Server_API/tests/AuthNZ`

This review excludes:

- API endpoints outside the AuthNZ core module
- app wiring outside local AuthNZ dependencies
- fixes or refactors during the review itself unless the user explicitly requests remediation afterward

Code scope remains limited to `tldw_Server_API/app/core/AuthNZ`. Test evidence may also be drawn selectively from other AuthNZ-specific test directories when those tests directly exercise the scoped core module, especially backend-specific behavior:

- `tldw_Server_API/tests/AuthNZ_SQLite`
- `tldw_Server_API/tests/AuthNZ_Postgres`
- `tldw_Server_API/tests/AuthNZ_Unit`
- `tldw_Server_API/tests/AuthNZ_Federation`

## Deliverable

The final output is a remediation-oriented review report.

Each finding will be labeled with:

- classification: `confirmed finding`, `probable risk`, or `improvement`
- severity
- confidence
- remediation size: `quick fix`, `medium change`, or `larger refactor`

## Review Sequence

The review will proceed through four passes in this order:

1. Security
2. Correctness
3. Maintainability
4. Test Gaps

The final report will keep these four sections in the same order.

## Approaches Considered

### Recommended: Risk-first review with targeted verification

Start with the highest-risk AuthNZ primitives such as JWT handling, sessions, revocation, password and MFA logic, API keys, secret handling, authorization resolution, and configuration fail-open or fail-closed paths. Use the AuthNZ tests as corroborating evidence and run only targeted verification commands that can confirm or falsify a specific concern.

Why this is preferred:

- surfaces serious auth and secret-handling issues early
- keeps the report prioritized instead of merely exhaustive
- adds stronger evidence when static inspection alone leaves uncertainty

### Alternative: Linear file-by-file sweep

Read every scoped AuthNZ file in repository order and classify findings afterward.

Trade-offs:

- simple to execute mechanically
- exhaustive in appearance
- weaker prioritization and more likely to bury critical findings among lower-value notes

### Alternative: Test-led contract audit

Start from `tldw_Server_API/tests/AuthNZ`, infer intended behavior, and inspect the implementation for drift.

Trade-offs:

- useful for mismatch and gap detection
- effective for finding untested assumptions
- weaker for latent security problems when tests already encode blind spots

## Chosen Method

Use a hybrid of the recommended risk-first review and full scoped coverage:

1. Inspect the highest-risk security-critical primitives first.
2. Use the corresponding AuthNZ tests as corroborating evidence rather than as the source of truth.
3. Continue through the rest of the scoped AuthNZ module so the final review covers the full approved surface.
4. Run targeted verification only when it materially strengthens or weakens a specific claim.
5. Separate confirmed findings from probable risks and lower-priority improvements.

## Review Focus Areas

The review order inside the module should bias toward:

- JWT issuance, verification, key rotation, and claims handling
- session lifecycle, refresh rotation, revocation, and blacklist behavior
- password hashing, password policy, reset, and verification flows
- MFA setup, validation, recovery, and lockout interactions
- API key and virtual key creation, validation, rotation, and scoping
- secret storage and key material handling
- RBAC, permissions, org or team authorization logic inside the scoped module
- quotas, rate limits, lockout, and budget enforcement logic
- migrations, schema evolution, backend divergence, and repository persistence paths
- configuration defaults, migration-sensitive paths, and fail-open or fail-closed behavior

Initial hotspot files include:

- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- `tldw_Server_API/app/core/AuthNZ/password_service.py`
- `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/settings.py`
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- selected files under `tldw_Server_API/app/core/AuthNZ/repos/`

## Evidence Model

The review will rely on:

- direct source inspection in `tldw_Server_API/app/core/AuthNZ`
- direct test inspection in `tldw_Server_API/tests/AuthNZ`
- targeted test inspection in `tldw_Server_API/tests/AuthNZ_SQLite`, `tldw_Server_API/tests/AuthNZ_Postgres`, `tldw_Server_API/tests/AuthNZ_Unit`, and `tldw_Server_API/tests/AuthNZ_Federation` when those tests directly validate scoped AuthNZ core behavior
- targeted static inspection commands where useful
- focused verification runs only when they answer a specific question

The review is not planned as a full integration or broad runtime test sweep. Verification is allowed only when it helps resolve a concrete claim such as:

- whether a suspicious branch is currently exercised
- whether an intended invariant already fails under current fixtures
- whether code and tests disagree on observable behavior

If a concern depends on behavior outside the approved scope, it should be downgraded to a probable risk unless the local AuthNZ evidence is enough to prove it directly.

The older plan in `Docs/superpowers/plans/2026-03-23-authnz-sequential-review.md` should not be reused as-is for this run because it intentionally broadens into API deps and endpoint files outside the approved boundary.

## Finding Categories

Each observation will be placed into one of these categories:

### Confirmed Finding

A concrete bug, vulnerability, inconsistency, or design flaw supported directly by code, tests, or clear control-flow or data-flow evidence.

### Probable Risk

Something that appears unsafe, brittle, or likely wrong, but depends on assumptions, integration context, or behavior not fully proven inside the scoped review.

### Improvement

A maintainability, hardening, or refactoring opportunity that is worth doing but is not necessarily evidence of a current defect.

## Severity, Confidence, And Remediation Size

Severity and confidence will be reported separately.

- severity communicates impact and urgency
- confidence communicates how directly the claim is supported by the reviewed evidence

Each item will also carry a remediation size tag:

- `quick fix` for localized changes with low coordination cost
- `medium change` for multi-file or behavior-sensitive remediation
- `larger refactor` for structural issues that need broader redesign or staged work

This keeps urgent but uncertain concerns from being overstated and prevents lower-value cleanup items from crowding out higher-risk AuthNZ findings.

## Testing Analysis Method

The `Test Gaps` pass should distinguish between:

- important invariants that are not tested at all
- tests that exist but are weak, overly implementation-coupled, or misleading
- behaviors that appear security-sensitive but are never directly asserted

The testing analysis should emphasize which missing tests would most reduce security or regression risk first.

## Execution Boundaries

- The review remains inside the approved AuthNZ module and AuthNZ test suite.
- Cross-module behavior may be noted only when a local AuthNZ file clearly depends on or assumes it.
- The review remains read-first and non-invasive.
- Severe findings should include remediation guidance, but not code changes.
- The review must not silently expand into endpoint or platform-wide behavior analysis.
- Targeted verification should prefer the smallest test selection that can answer the question and should avoid broad backend sweeps unless a backend-specific claim requires them.

## Final Deliverable Contract

The final report will contain four ordered sections:

1. `Security`
2. `Correctness`
3. `Maintainability`
4. `Test Gaps`

It will also end with a short `Blind Spots / Not Reviewed` section that makes any meaningful excluded surface or unresolved evidence limits explicit.

Within each section:

- findings are ordered by severity
- confirmed findings come before probable risks
- improvements remain clearly labeled
- file and line references are included where possible

Each reported item should include:

- a short title
- classification
- severity
- confidence
- remediation size
- concise reasoning
- actionable file references

If a section has no confirmed issues, the report should say so explicitly and then note any residual risks or blind spots without padding.

## Success Criteria

The review is successful when:

- the scoped AuthNZ module and test suite are fully covered
- findings are separated cleanly by category and confidence
- the final report is organized by the four approved passes
- serious defects are easy to prioritize
- remediation effort is easy to estimate from the report
- targeted verification is used only where it improves confidence materially
- test gaps identify the highest-leverage missing coverage first
- report blind spots are explicit so scope limits are not mistaken for clearance

## Constraints

- Do not broaden the review to downstream API endpoints in this run.
- Do not silently convert the review into implementation work.
- Do not present speculation as a confirmed bug.
- Do not let minor maintainability observations dominate the report.
- Do not replace source inspection with broad test execution.

## Expected Outcome

This design yields a focused, evidence-driven review of the AuthNZ module that is broad enough to cover the full scoped surface, prioritized enough to surface the most important security and correctness issues first, and structured enough to support remediation planning without mixing hard defects and softer improvements.
