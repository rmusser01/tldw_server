# DB_Management Review Design

Date: 2026-04-07
Topic: DB_Management module review
Status: Approved design

## Objective

Review `DB_Management` in `tldw_server` for concrete issues, bugs, risks, and improvement opportunities with a broad but risk-weighted audit.

The review should prioritize actionable findings over exhaustive commentary, keep confirmed issues separate from softer risks, and stay focused on the database-management subsystem rather than drifting into unrelated endpoint or product behavior.

## Scope

This review is centered on:

- `tldw_Server_API/app/core/DB_Management`
- tests and callers that materially define or validate `DB_Management` behavior

This includes:

- shared backend abstractions and connection/transaction helpers
- path, scope, tenancy, and policy helpers
- database factories and entrypoints
- migration and schema bootstrap paths
- the `media_db` surface and other high-fanout or high-risk `*_DB.py` modules

This review excludes:

- unrelated endpoint logic except where a caller is needed to confirm intended `DB_Management` behavior
- implementation work or refactors during the review itself unless the user explicitly requests remediation afterward
- equal-depth treatment of every low-risk domain database helper

## Approaches Considered

### Recommended: Risk-led audit

Split `DB_Management` into high-blast-radius slices, inspect the shared foundations first, and follow suspicious paths into representative callers and tests before writing findings.

Why this is preferred:

- surfaces real correctness and isolation risks earlier than a flat sweep
- gives better signal for a broad module than purely linear reading
- preserves enough depth to avoid reporting style debt as if it were a bug

### Alternative: Surface risk scan

Read only the main abstractions, recent migration changes, and a representative sample of tests and domain DB helpers.

Trade-offs:

- fastest option
- useful for a quick health check
- lower confidence on deeper edge cases and medium-severity issues

### Alternative: Exhaustive file-by-file audit

Read the entire subtree in repository order and evaluate every file with roughly equal effort.

Trade-offs:

- highest apparent coverage
- weaker prioritization for a subsystem this wide
- more time spent on low-risk helpers and less on blast-radius analysis

## Chosen Method

Use the recommended risk-led audit with four ordered passes:

1. `Shared foundations`
   Review backend abstractions, connection lifecycle, transaction helpers, async wrappers, path resolution, scope propagation, and policy/RLS support.
2. `High-risk execution paths`
   Trace factories and entrypoints into `media_db` and representative high-usage `*_DB.py` modules, focusing on transaction consistency, unsafe SQL construction, path leakage, migration safety, and failure handling.
3. `Evidence cross-check`
   Compare suspicious paths against tests, recent commits, and representative callers to distinguish confirmed defects from intentional constraints or already-covered behavior.
4. `Findings synthesis`
   Report only substantive findings with severity, impact, and concrete file references.

## Evidence Model

The review will rely on:

- direct source inspection in `tldw_Server_API/app/core/DB_Management`
- direct inspection of relevant tests across `tldw_Server_API/tests`
- targeted caller inspection where `DB_Management` behavior depends on external assumptions
- recent git history when it helps explain likely weak points or regression-prone areas

The default git-history baseline should start with the last 20 commits touching `tldw_Server_API/app/core/DB_Management`. The review should expand beyond that window only when a hotspot needs older context, such as a migration chain, a repeated regression pattern, or a long-running backend abstraction change.

The review is primarily static and read-first. Runtime verification may be used selectively only if it materially strengthens or falsifies a specific claim.

When a finding depends on backend semantics rather than only source reading, the review should perform targeted verification whenever feasible. This applies especially to:

- transaction boundaries and implicit commit or rollback behavior
- migration idempotency and partial-failure recovery
- SQLite versus PostgreSQL behavioral divergence
- locking, busy-timeout, WAL, or connection-lifecycle claims
- FTS or dialect-specific query behavior

If targeted verification cannot be run, the limitation should be stated explicitly and the observation should be downgraded to a probable risk unless the source evidence is still conclusive on its own.

## Findings Model

The final report should include only substantive findings.

Each finding should contain:

- severity
- confidence
- concise statement of the issue
- why it matters
- concrete file and line references

Observations should be classified as:

### Confirmed finding

A concrete bug, security or isolation flaw, correctness issue, migration hazard, or architectural problem supported directly by source, tests, or control/data-flow evidence.

### Probable risk

Something that appears unsafe or brittle but depends on assumptions that are not fully proven within the scoped review.

### Improvement

A worthwhile hardening, maintainability, or architectural simplification opportunity that is not strong evidence of a current defect but would reduce future failure or regression risk.

Minor style commentary and low-signal cleanup should be omitted.

## Severity Model

Severity should be balanced across these three axes equally:

- correctness and data-loss risk
- security and multi-tenant isolation risk
- maintainability and architectural debt

When several findings are comparable, rank higher the issue with the larger blast radius, higher likelihood of silent corruption or tenant leakage, or greater probability of causing repeated regressions.

## Review Focus Areas

The review order inside the module should bias toward:

- backend abstraction mismatches between SQLite and PostgreSQL
- connection lifecycle, transaction boundaries, and implicit commit or rollback behavior
- path resolution, per-user database selection, and filesystem safety
- scope propagation and row-level isolation assumptions
- migration idempotency, schema-version correctness, and partial-failure recovery
- raw SQL construction, parameterization discipline, and dialect-specific hazards
- `media_db` runtime composition, legacy wrappers, and high-fanout entrypoints
- representative per-feature databases with broad usage or unusual schema logic
- gaps where tests fail to cover high-risk invariants

## Representative Selection Rule

When the review samples representative callers or `*_DB.py` modules, the selection should be deliberate rather than ad hoc. The sampled set should preferentially include modules that meet one or more of these criteria:

- high fan-out or factory ownership
- recent churn, migrations, or bug-fix history
- custom transaction, migration, or schema-bootstrap logic
- path, scope, tenancy, or policy-sensitive behavior
- unusually large or SQL-heavy implementations

At minimum, the audit should cover enough representatives to include shared foundations, `media_db`, and multiple non-`media_db` database helpers from different risk profiles so the final findings are not driven by a single subsystem.

The initial audit seed set should include these files before expanding further:

- `tldw_Server_API/app/core/DB_Management/content_backend.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
- `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- `tldw_Server_API/app/core/DB_Management/media_db/api.py`

From there, the review should branch into the highest-risk adjacent modules identified by churn, fan-out, migration complexity, or tenancy sensitivity.

## Execution Boundaries

- The review remains inside the `DB_Management` subsystem and closely related tests and callers.
- Cross-module behavior may be noted only when a local database-management file clearly depends on it.
- The review remains non-invasive and should not silently turn into remediation work.
- The final output is a findings list, not a fix plan.

## Final Deliverable

The final response to the user will be a code-review style findings list ordered by severity.

Each finding will include:

- severity and confidence
- a short explanation of the failure mode or risk
- why the issue matters across correctness, isolation/security, or maintainability
- file and line references

The report may include confirmed findings, probable risks, and improvements. If an area appears healthy, the review may say so briefly rather than inventing debt. If uncertainty remains, it should be labeled as an assumption or open question rather than overstated as a confirmed bug.

A probable risk may be upgraded to a confirmed finding when at least one of these is true:

- the source code alone establishes the defect unambiguously
- targeted verification reproduces or directly confirms the behavior
- independent evidence from source, tests, and callers converges strongly enough that no material ambiguity remains

## Success Criteria

The review is successful when:

- the high-risk `DB_Management` surfaces are covered without pretending every helper received equal depth
- findings are evidence-based and separated from softer risks
- representative sampling is explained by explicit selection criteria
- severity reflects a balanced view of correctness, isolation/security, and maintainability
- backend-sensitive claims are verified or clearly labeled with the verification limitation
- the final report is easy to use as a triage artifact
- low-value cleanup does not crowd out real defects or regression risks

## Constraints

- Do not broaden this run into a full application review.
- Do not silently convert the review into implementation work.
- Do not present speculation as a confirmed defect.
- Do not over-index on style or naming concerns when more substantive risks exist.

## Expected Outcome

This design yields a broad, high-signal review of `DB_Management` that prioritizes shared foundations and high-blast-radius paths first, validates suspicious behavior against tests and callers, and produces a concise findings list that the user can use directly for follow-up remediation planning.
