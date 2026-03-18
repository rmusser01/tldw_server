# PR 898 Jobs and Metering Boundary Redesign

## Context

PR 898 fixed the production correctness issues, but three larger architecture comments remain:

- `JobManager` still owns a fair-share read against the jobs table.
- `JobManager.create_job()` cannot reuse that read inside its existing create-time DB transaction boundary.
- `StripeMeteringService` still owns direct persistence for `usage_daily`, subscription lookup, and `metering_sync_log`.

This follow-up intentionally takes the broader redesign path rather than another local extraction pass.

## Goals

- Move Jobs persistence concerns out of `tldw_Server_API/app/core/Jobs/manager.py`.
- Move Stripe metering persistence concerns out of `tldw_Server_API/app/services/stripe_metering_service.py`.
- Introduce explicit repository/session boundaries that can be injected and unit-tested independently.
- Preserve current user-visible behavior while allowing internal interface changes.
- Make the new boundaries usable from both SQLite and PostgreSQL without leaking backend-specific SQL into orchestration code.

## Non-Goals

- Replacing `JobManager` as the public entry point across the codebase in this pass.
- Changing queue semantics, fair-share policy semantics, or Stripe API behavior.
- Introducing a full migration framework beyond the schema/bootstrap needs already owned by these modules.

## Architecture

### 1. Jobs Boundary

Add a new DB layer under `tldw_Server_API/app/core/DB_Management` that owns Jobs persistence.

Planned pieces:

- `JobsRepository`
  - Public persistence API for job reads/writes needed by `JobManager`.
  - Owns SQL for both SQLite and PostgreSQL.
- `JobsSession`
  - Backend-agnostic wrapper around a live jobs DB connection/transaction.
  - Lets create-time reads and writes share the same connection.
- `JobsRepositoryFactory` or constructor helpers
  - Builds the repository from the existing `JobManager` config (`backend`, `db_path`, `db_url`).

`JobManager` remains the public façade used across the codebase, but its role narrows to:

- queue policy validation
- fair-share policy application
- payload hygiene / secret rejection
- orchestration of repository calls

The key structural change is that `create_job()` will ask the repository for a transactional session, perform the active-job count through that session, compute fair-share priority, and persist the job through the same session. That removes the extra connection and removes the raw SQL from `manager.py`.

### 2. Metering Boundary

Split Stripe metering into orchestration and repository concerns.

Planned pieces:

- `AuthnzUsageDailyRepository`
  - Reads `usage_daily` and handles legacy-schema fallback (`bytes_in_total` missing).
- `AuthnzBillingSubscriptionRepository`
  - Resolves active Stripe subscriptions through membership and org-owner paths.
- `AuthnzMeteringSyncLogRepository`
  - Owns `metering_sync_log` schema/bootstrap, duplicate checks, sync writes, and sync-total reads.
- `StripeMeteringOrchestrator` or a refactored `StripeMeteringService`
  - Keeps Stripe API calls, per-user sync flow, and reconciliation assembly.

The service layer will no longer contain SQL or table DDL. It will depend on injected repository interfaces plus a DB pool provider and Stripe client adapter.

### 3. Schema Ownership

`metering_sync_log` remains logically owned by metering, but schema bootstrap moves into the repository layer instead of the service layer. The repository exposes an `ensure_schema()` method and encapsulates backend-specific DDL.

This is intentionally less ambitious than introducing a global migration subsystem, but it is enough to move schema ownership out of the service/orchestrator.

## Data Flow

### Jobs Create Flow

1. Caller invokes `JobManager.create_job(...)`.
2. `JobManager` validates queue/domain and redacts or rejects payload secrets.
3. `JobManager` opens a repository session.
4. Repository counts active jobs for `owner_user_id` using the same session that will persist the new row.
5. `JobManager` applies fair-share policy and computes final priority.
6. Repository persists the job and returns the created row.

### Metering Sync Flow

1. Caller invokes `StripeMeteringService.sync_daily_usage(...)`.
2. Service acquires/injects AuthNZ DB pool once.
3. Sync-log repository ensures schema.
4. Usage repository reads daily usage rows.
5. Subscription repository resolves Stripe subscription per user.
6. Sync-log repository checks idempotency state.
7. Service reports usage to Stripe.
8. Sync-log repository records successful sync.

### Reconciliation Flow

1. Usage repository reads local totals.
2. Sync-log repository reads synced totals.
3. Service compares the two datasets and builds drift results.

## Error Handling

- Repositories raise backend-specific query failures upward as typed project exceptions or plain exceptions with preserved context.
- `JobManager` keeps the current behavior of warning and continuing when fair-share evaluation cannot be completed.
- Stripe metering orchestration keeps its current skip/error semantics, but the repository boundary makes failures attributable to usage reads, subscription resolution, or sync-log access separately.
- Legacy-schema fallback remains in the usage repository so callers receive normalized row dictionaries.

## Testing Strategy

### Jobs

- Add repository-focused tests for:
  - active-job counting in SQLite
  - active-job counting in PostgreSQL fixture paths where available
  - create-time session reuse / single-transaction behavior
- Keep the current fair-share integration tests, updated to assert the repository-backed path.

### Metering

- Add repository-focused tests for:
  - usage query normalization with and without `bytes_in_total`
  - membership and owner subscription lookup
  - sync-log ensure/check/write/read behavior
- Keep the current service/orchestrator tests, but swap their mocks to repository interfaces instead of direct SQL helpers.

### Regression Suite

- Re-run the PR 898 targeted suite:
  - consent endpoints
  - audit chain
  - overage enforcement
  - fair-share integration
  - Stripe metering
- Add any new repository unit tests to that verification command.
- Run Bandit on touched Jobs/DB/metering files before publishing.

## Risks

- `JobManager` is used pervasively, so constructor changes must stay backward-compatible or defaultable.
- Repository/session abstractions can accidentally duplicate existing `JobManager` logic if the split is not kept strict.
- Stripe metering tests currently patch service internals; they will need careful rewrite to avoid coupling to deleted helpers.

## Recommended PR Strategy

- Keep PR 898 limited to the correctness fixes already pushed.
- Open this follow-up as a stacked draft PR against `feat/production-readiness-gaps`.
- Land the repository/session scaffolding first, then the `JobManager` refactor, then the Stripe metering split, with verification at each step.
