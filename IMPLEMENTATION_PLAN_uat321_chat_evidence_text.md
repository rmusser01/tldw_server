# UAT321: Visible Chat evidence

Backlog: TASK13260.259. Reuse the existing public Chat text projection before formatting retrieved messages, preserving raw storage and ownership filters. Cover both Chat and Character retrievers, their full-text fallbacks, and legacy per-user SQLite reads. No changes to model prompting or message storage.

## Stage 1: Reproduce
**Goal**: Capture reasoning in real retrieved evidence with unchanged raw messages.
**Success Criteria**: Tests expose reasoning leakage and reasoning-only evidence.
**Tests**: Real SQLite/PostgreSQL adapter searches and legacy SQLite controls.
**Status**: Complete

## Stage 2: Repair and review
**Goal**: Return visible text only, omitting empty evidence.
**Success Criteria**: Owned visible answers and user turns remain available; raw messages are unchanged.
**Tests**: Focused retrieval and adjacent QA suites, Bandit, lint, review.
**Status**: Complete

## Stage 3: Native acceptance
**Goal**: Verify fresh UI evidence and owner isolation alongside UAT322/323.
**Success Criteria**: Native source cards contain visible answers with no reasoning, and Bob sees no foreign sources.
**Tests**: Fresh SQLite/PostgreSQL native QA, source audits and owned cleanup.
**Status**: In Progress
