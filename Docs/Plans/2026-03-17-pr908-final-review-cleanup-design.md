## Context

PR 908 still has unresolved review threads after the boundary-redesign rebase. The remaining feedback is concentrated in the new metering repositories, the new jobs repository/session layer, and the `JobManager` fair-share admission flow.

## Goals

- Close the remaining correctness gaps without widening the PR beyond the persistence-boundary redesign.
- Keep `JobManager` as the orchestration entry point while making admission control fail closed on repository errors.
- Improve the new repositories enough to satisfy the active review threads, including optional pool support, integrity checks, and low-risk hygiene fixes.

## Design

### AuthNZ metering repository

- Add an explicit `DuplicateActiveSubscriptionError` so duplicate active subscription rows fail fast instead of silently selecting one row.
- Resolve active subscriptions by fetching up to two matches in each query branch and raising on duplicates.
- Add a secondary `day` index for `metering_sync_log` in both PostgreSQL and SQLite schema creation paths.

### Jobs repository

- Add optional pool injection so `JobsRepository` can reuse caller-managed connections while preserving existing direct-connect behavior.
- Move session acquisition into a single context-managed helper that works for both pooled and non-pooled connections.
- Replace silent SQLite policy suppression with diagnostic logging and use SQLite `RETURNING *` on insert to align the SQLite and PostgreSQL insert paths.
- Clean up the remaining style issues called out by review bots (`collections.abc.Iterator`, unquoted forward references).

### Job manager

- Validate injected repository compatibility at construction time by checking the required interface and backend alignment.
- Make fair-share admission fail closed: repository/admission errors should raise a retryable request error instead of returning permissive defaults.
- Run the SQLite fair-share check inside the same transaction as the insert so count and insert share one connection/transaction window.

### Tests

- Extend repository tests to cover pooled sessions and SQLite `RETURNING` behavior.
- Extend metering repository tests to cover duplicate subscription detection and schema indexing.
- Extend fair-share tests to cover fail-closed admission and repository validation.
