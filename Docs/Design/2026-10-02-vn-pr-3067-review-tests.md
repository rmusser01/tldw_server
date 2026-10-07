# VN Stacked Review Test Follow-up

Task: TASK-13518 (formerly TASK-13369). Human approval: "address all the ci failures and review findings then!"
This answers the previously presented four-findings bounded test/docs design.
Parent #3016 and child #3067 production behavior remain unchanged by these fixes.

## Approved Changes

- Move the existing isolated corruption fixture SQL into private test support
  under `app/core/DB_Management`, without exporting a runtime repository or HTTP
  corruption API. Preserve statements, parameters and native errors.
- Replace caller-frame/module assertions with real generation/receipt outcomes
  and rollback/failure assertions for start, recovery and claim operations.
  Exercise real SQLite transactions; do not mock their behavior away.
- Remove the committed optional raw runtime artifact-copy hook. Preserve safe
  result assertions and every already-frozen artifact; do not copy operator data.
- Preserve exact current plan and task bytes in durable ignored archives before
  condensing repetitive monitoring records. Keep current stages, approvals,
  blockers and links readable. Task changes use official Backlog tooling.

## Verification

Use focused tests for affected fixture consumers, native completed deletion,
receipt outcomes and rollback. A copied omission control must demonstrate that
rollback assertions detect loss of atomicity. Check helper imports and compile
changed Python. Run touched-scope Ruff and Bandit against the same baseline;
inherited findings are not a blanket clean result. Independent SPEC/QUALITY
review follows implementation. No broad suite or unrelated scanner reruns.

## CI Scope And Gates

The parent nested pytest plugin fix already exists in the child. Parent old-head
Jobs failure does not become green because of that fix. Diagnose child macOS and
Ubuntu bootstrap failures before proposing changes; their underlying database
error is currently sanitized. A new CI fix needs its concrete plan approved
under gh-fix-ci. No PR body edit, parent dev reconciliation, merge or cleanup is
implied. Child human-written Change summary and parent guarded Verification-only
body approval remain separate. Preserve Jobs lease authority, original outcomes,
approved bytes, counters and deliberate-deletion semantics.

ADR required: no. This applies existing DB ownership and test/privacy rules;
it changes no durable architecture policy. ADR-002, ADR-004 and ADR-006 govern
tracking, human ownership and touched-scope security verification.

## Approved Additional Timeout Test

Task67 diagnosis found shared sqlite3 instrumentation observes other backend
handles before timeout configuration, while intended VN read-only handles already
use timeout10. Human explicitly approved the bounded test-only fix. Observe only
the intended read-only VN URI; keep native execution, event-loop responsiveness,
off-thread, handle closure and timeout10000 assertions. A copied reader without
its explicit timeout must fail the timeout assertion. No production settings
change. This approval does not answer Task66's separate guard-fix proposal.

## Approved Bootstrap Compatibility Fix

Requester subsequently replied "APPROVED" after the concrete Task66 fix,
publication and parent integration actions were presented. Accept only the exact
SQLite AutoIncrementColumnConstraint AST type with no arguments. SQLGlot30.20.0
requires column context to render AUTOINCREMENT, unlike29.0.1; standalone text
is not a valid canonicality check for this node. Preserve every other column,
schema, constraint and one-shot capability check. Do not pin dependencies,
disable protection, or change DDL/workflows.

Add a real private SQLite setup/bootstrap/UsersDB initialization regression,
including repeated initialization and preservation of the single-user admin.
Exercise it with both SQLGlot versions; first demonstrate failure on30.20.0.
Reject nonempty AST arguments, non-SQLite nodes and derived node types. Run the
existing guard rejection/one-shot controls. Record scoped Ruff/Bandit baselines
and independent SPEC/QUALITY review. Local matching-dependency verification
does not replace native exact-head E2E/CI or claim PostgreSQL execution.
