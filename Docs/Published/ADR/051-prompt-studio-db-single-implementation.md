# ADR-051: One Prompt Studio database implementation, moved aggregate by aggregate

**Status:** Accepted (2026-09-23)
**Date:** 2026-09-23
**Backfilled from:** not backfilled
**Decision owner:** repository owner (pending)
**Related task:** TASK-13318
**Related spec/plan:** `Docs/Design/2026-09-23-prompt-studio-db-consolidation-design.md`; review finding F20

## Decision (proposed)

Replace the two parallel Prompt Studio implementations with one backend-neutral implementation over
`DatabaseBackend`, organised as a `prompt_studio_db/` package with one repository per aggregate,
following the shipped `media_db/` split. Move one aggregate at a time, each gated by a behavioural
parity harness run against both existing implementations, smallest aggregates first. Retire the
facade's `*args/**kwargs` forwarding last, so mypy sees real signatures.

## Context

`PromptStudioDatabase.py` (7,426 lines) holds `_BackendPromptStudioDatabase` and
`_SQLitePromptStudioDatabase`, sharing 60 methods over parallel SQL, behind a forwarding facade that
hides every signature from type checking. Drift is measured and ongoing: one missing method
(TASK-13290), seven signature mismatches (four latent on public methods), an arity difference in
`_format_test_case`, and a busy-retry loop on the SQLite side only.

## Alternatives considered

- **Promote the PostgreSQL class to sole implementation first.** It already branches on backend type
  and `DatabaseBackend` has a SQLite implementation, so this is the fastest route. Rejected as the
  first step because it switches the default SQLite deployment to different code in one move, after
  porting migrations, `transaction()`, FTS setup and the retry loop.
- **Keep both; fix drift only.** Cheapest, but leaves the business logic duplicated and catches only
  signature drift, not behavioural drift.

## Consequences

- Seven stages, each independently shippable and reversible; both old classes stay alive until the
  last aggregate moves.
- A behavioural parity harness becomes a prerequisite, and needs PostgreSQL in CI to be meaningful.
- The signature-parity ratchet (`tests/DB_Management/test_prompt_studio_backend_parity.py`) shrinks
  as mismatches are resolved during the move.

## Open

Busy-retry policy for the unified implementation; whether to align the four latent public mismatches
ahead of the refactor.
