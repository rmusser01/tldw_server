# UAT324: Character activity timestamps

Backlog: TASK13260.262. The native SQLite response serializes the default seed's UTC timestamp without an offset, and the browser interprets it as local time. Correct the response boundary, preserving stored values and explicit offsets. Follow the existing World Book SQLite normalization pattern; avoid assuming the timezone of PostgreSQL values.

## Stage 1: Reproduce the response contract
**Goal**: Exercise fresh seeded SQLite and official PostgreSQL Character responses.
**Success Criteria**: SQLite seed/list/query/detail exposes a causal missing-offset failure; explicit offsets and PostgreSQL remain stable.
**Tests**: Existing database fixtures, real endpoint response conversion, stored-row preservation and non-UTC controls.
**Status**: Complete

## Stage 2: Repair and verify
**Goal**: Attach UTC only to SQLite naive Character response timestamps at the common converter.
**Success Criteria**: List/query/detail/create/update paths use the backend-aware conversion; values already carrying offsets remain unchanged.
**Tests**: Focused and adjacent Character regressions, UI activity display contract, Bandit, lint and independent review.
**Status**: Complete

## Stage 3: Native acceptance
**Goal**: Confirm a fresh default Character displays its real age.
**Success Criteria**: Committed SQLite and PostgreSQL API responses retain truthful timestamps; normal UI display, source audit and owned cleanup recorded.
**Tests**: Targeted native acceptance before closure; preserve full-matrix pending status.
**Status**: Not Started

Causal3fail/3pass precede the repair. Final116passes/0skips include actual PostgreSQL. Bandit0production/0tests; two unchanged baseline Ruff import warnings. Reviewer clear. Native acceptance remains pending alongside325.
