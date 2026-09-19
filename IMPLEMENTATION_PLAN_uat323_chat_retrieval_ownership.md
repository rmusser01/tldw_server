# UAT323: Chat retrieval ownership

Backlog: TASK13260.261. Restore private Chat evidence boundaries using the existing PostgreSQL conversation client owner; retain per-user SQLite semantics. Cover both substring retrieval and the database full-text fallback, plus metadata resolution. Do not change stored messages or sharing rules.

## Stage 1: Reproduce
**Goal**: Attribute the native PostgreSQL leak and reproduce both retrieval paths against real databases.
**Success Criteria**: Foreign and deleted records fail explicit isolation assertions, while owned controls remain available.
**Tests**: SQLite and official PostgreSQL fixtures; native Bob identity, direct foreign Chat denial, cache-disabled retrieval.
**Status**: Complete

## Stage 2: Repair
**Goal**: Apply conversation ownership and active-record constraints before limiting results.
**Success Criteria**: Owned results and pagination survive; foreign metadata and evidence are absent.
**Tests**: Focused real-database regressions and adjacent Chat/Knowledge retrieval tests.
**Status**: Complete

## Stage 3: Verify and accept
**Goal**: Review, static checks, and fresh native owner/Bob controls on the committed repair.
**Success Criteria**: Actual PostgreSQL is tested without skips; tracker and task contain evidence and remaining limitations.
**Tests**: Bandit, scoped lint, independent review, native controls, source parity and owned runtime cleanup.
**Status**: In Progress
