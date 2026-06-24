# UserProfiles Contract-First Refactor Design

## Context

The UserProfiles module provides the unified profile read and update API for
self-service users and admins. Current behavior is spread across large modules:

- `tldw_Server_API/app/core/UserProfiles/service.py`
- `tldw_Server_API/app/core/UserProfiles/update_service.py`
- `tldw_Server_API/app/core/UserProfiles/overrides_repo.py`
- `tldw_Server_API/app/api/v1/endpoints/users.py`
- `tldw_Server_API/app/services/admin_profiles_service.py`

These files mix API orchestration, profile assembly, validation, authorization,
transaction handling, storage translation, response mapping, audit behavior, and
bulk semantics. The refactor should cleanly separate those responsibilities
while allowing profile API contract cleanup through an explicit migration path.

## Goals

- Define a clean profile API contract for reads, single updates, dry-runs, and
  bulk updates.
- Keep existing routes compatible through adapters while the clean contract is
  introduced separately.
- Make dry-run and apply behavior share the same validation and authorization
  path.
- Make version locking, transaction boundaries, and after-commit effects
  explicit.
- Make authorization and error mapping reusable across self, admin, org, team,
  platform, and bulk flows.
- Stage the refactor so each step is testable, mergeable, and reversible.

## Non-Goals

- No storage schema changes in the first refactor stage.
- No removal of existing profile endpoints during the first stage.
- No frontend rewrite.
- No unrelated AuthNZ, quota, audit, or membership subsystem redesign.

If durable effect queues, contract-version tables, or new audit state become
necessary, they require a separate design and implementation plan.

## Proposed Architecture

### API Layer

`users.py` and admin route handlers should become thin adapters. They should:

- parse and validate request schemas,
- resolve the current principal,
- choose the compatibility or clean contract mapper,
- call an application service,
- return the mapped response.

They should not own profile update rules, membership checks, optimistic locking,
bulk semantics, or detailed error classification.

### Application Services

Introduce application-level orchestration services:

- `ProfileQueryService`: self, admin, and batch profile reads.
- `ProfileCommandService`: self/admin single update dry-run and apply.
- `ProfileBulkCommandService`: bulk dry-run and apply.

These services own calling order, transaction boundaries, required audit policy,
and compatibility with endpoint-level behavior.

### Domain Policy And Planning

Introduce narrow domain components:

- catalog lookup and key metadata access,
- value validation,
- actor/target/scope authorization,
- membership mutation validation,
- profile version guard,
- update error taxonomy,
- response contract mapping.

Single updates and bulk updates should use an `UpdatePlanner` that returns a
typed `UpdatePlan`. The plan is a data artifact, not a hidden service object. It
contains normalized input, authorization decisions, validated mutations, version
requirements, and after-commit effect descriptors.

### Executors

Executors apply validated plans. Execution logic should not live in
`UpdatePlan`.

Executor responsibilities:

- run profile/user override mutations,
- run identity and quota mutations,
- run membership mutations,
- apply backend-specific transaction behavior,
- return mutation results for response mapping.

### Persistence And Gateways

Keep database details behind repositories and gateways:

- user, org, and team profile override repositories,
- AuthNZ user and membership gateway,
- quota and security source gateways,
- effective-config source gateways.

SQLite and Postgres differences should remain contained in repository/gateway
helpers with focused backend-selection tests.

### Contract And Compatibility Layer

Existing routes keep legacy-compatible response mapping. The clean contract
should use an explicit surface, such as versioned routes or documented API
version/media-type negotiation. Avoid ad hoc query parameters such as
`?contract=v2`.

## Read Flow

1. API adapter resolves principal, target user, requested sections, source
   visibility, and masking mode.
2. `ProfileQueryService` builds a typed `ProfileReadRequest`.
3. Authorization verifies target access and masking permissions.
4. Section builders assemble identity, memberships, security, quotas,
   preferences, raw overrides, and effective config.
5. A response mapper returns either legacy-compatible output or clean contract
   output.

Hard failures:

- authentication failure,
- target user not found,
- forbidden target access,
- invalid section names,
- masking policy violations.

Optional section failures may be returned as `section_errors` only for
non-critical sections such as security summaries, BYOK status, usage snapshots,
or policy summaries. Identity and authorization failures are always hard
failures.

## Single Update Flow

1. API adapter maps the request into a `ProfileUpdateCommand`.
2. `ProfileCommandService` calls the planner.
3. The planner performs catalog lookup, value validation, authorization,
   membership existence checks, and version requirement capture.
4. For `dry_run=true`, the service returns a plan summary without mutation.
5. For apply mode, the service starts the write transaction.
6. Apply mode rechecks `profile_version` inside the write transaction.
7. Executors apply the validated mutation plan.
8. The transaction commits.
9. After-commit effects run according to their required or best-effort policy.
10. The response mapper returns the selected contract shape.

Dry-run and apply must share all validation and authorization paths. Apply mode
adds only the transactional version recheck and mutation execution.

## Bulk Flow

`ProfileBulkCommandService` uses the same planner per target user.

Bulk behavior:

- resolve candidate users,
- enforce confirmation thresholds,
- run each target user as an isolated unit,
- keep per-user atomicity,
- allow one user failure without rolling back other users,
- aggregate per-user errors and applied keys,
- compute diffs only where requested and within configured cost limits.

Bulk diffs should be optional or capped for large target sets. The service should
define whether diffs come from preloaded before-snapshots, post-apply reads, or
both for each endpoint contract.

## Contract Rules

### Clean Single Update Contract

Clean v2 single updates are atomic all-or-reject:

- if the request is valid and authorized, all planned mutations apply;
- if any entry is invalid or forbidden, no mutation occurs;
- the clean single-update response does not include `skipped`.

`skipped` remains only in legacy adapters and bulk per-user results.

### Bulk Contract

Bulk owns partial reporting. Each target user result includes:

- `applied`,
- `errors`,
- optional `diffs`,
- resulting `profile_version` when available.

### Error Mapping

Use stable internal error codes and deterministic HTTP mapping:

- `400`: unknown key, invalid section name, malformed patch semantics, invalid
  update action.
- `422`: known key with invalid type, value, range, or invalid operation against
  current user state.
- `403`: forbidden key, forbidden scope, forbidden role escalation.
- `404`: target user, team, or org does not exist when existence can be safely
  disclosed.
- `409`: profile version mismatch.
- `500`: unexpected storage or executor failure, sanitized.

For scoped admin operations, `403` wins over `404` when revealing existence
would leak cross-scope data.

Membership-specific mapping:

- missing team/org maps to `404` only when actor scope allows existence
  disclosure;
- "user is not a member for this requested role update" maps to `422` unless
  the operation is cross-scope, in which case it maps to `403`.

## Effects And Audit

Separate transactional database mutations from after-commit effects.

Effect descriptors should identify:

- effect type,
- target user/org/team,
- required vs best-effort policy,
- sanitized failure behavior,
- logging/metric/audit metadata.

Required admin audit follows the project's strict audit policy. When durable
audit is required, audit failure blocks success rather than returning a false
success. Best-effort effects such as cache invalidation and limiter refresh are
logged and tracked in metrics, but do not make a committed database mutation
appear failed.

Effect internals are not exposed in normal v2 responses. If the API needs to
surface non-critical issues, use a generic warnings mechanism rather than
cache/limiter-specific details.

## Versioning And Transactions

Single update apply mode must re-read and compare the expected profile version
inside the write transaction.

Backend behavior:

- Postgres should use row locks where applicable for user-profile version state.
- SQLite should rely on transaction serialization behavior, such as immediate
  write transactions, where row-level locks are unavailable.
- Override writes must use the active transaction connection when part of a
  profile update.

Tests should cover backend-specific behavior only where behavior differs.

## Migration Strategy

Use readiness gates instead of a big-bang rewrite:

1. Add typed internal request/result/plan models and tests while preserving
   current behavior.
2. Introduce planner and executors behind existing routes.
3. Route existing endpoints through compatibility mappers.
4. Add the clean v2 surface only after existing routes are stable on the new
   internals.
5. Publish migration docs covering changed response fields, removal of
   single-update `skipped`, new error codes, and versioning behavior.
6. Deprecate legacy quirks through documentation and response headers over a
   release window.

Legacy characterization tests should protect only adapter behavior. They should
not block clean contract implementation.

## Testing Strategy

### Characterization Tests

Capture current behavior at legacy adapter boundaries:

- existing route response shapes,
- deprecation behavior,
- legacy `skipped` behavior,
- audit events,
- profile version conflicts,
- bulk partial reporting.

### Clean Contract Tests

Define target behavior:

- atomic all-or-reject single updates,
- no `skipped` field in clean single-update responses,
- deterministic error mapping,
- dry-run/apply parity through the same planner,
- version recheck inside write transactions,
- read `section_errors` only for allowed optional sections,
- per-user bulk isolation and error aggregation.

### Unit Tests

Cover the new seams:

- update planner,
- authorization policy,
- value validation,
- mutation executors,
- response mappers,
- effect dispatcher,
- repository/backend helpers.

### Backend Tests

- SQLite and backend-selection tests are required at every stage.
- Postgres integration is required only where behavior differs: row locks,
  transaction semantics, membership writes, and bulk update isolation.

### Security Tests

Verify:

- required admin audit failure blocks writes,
- best-effort cache/limiter failures do not falsely report DB failure,
- secret masking,
- forbidden scope precedence,
- role escalation prevention,
- cross-scope existence hiding.

## Risks And Mitigations

- Risk: file shuffling without simpler behavior.
  Mitigation: each extracted component must have a concrete interface and test
  responsibility.

- Risk: compatibility adapters preserve too many quirks.
  Mitigation: legacy tests stay at adapter boundaries; clean contract tests
  define the target behavior.

- Risk: planner becomes a new monolith.
  Mitigation: `UpdatePlan` remains typed data; validation, authorization, and
  execution stay in narrow collaborators.

- Risk: v2 contract ships before internals are stable.
  Mitigation: v2 surface waits for existing routes to pass on planner/executor
  internals.

- Risk: backend differences create hidden race conditions.
  Mitigation: isolate backend behavior in gateways and test Postgres only where
  semantics differ from SQLite.

## Approval State

This design direction was approved interactively on 2026-06-24. The next step is
for the user to review this written spec. After approval, proceed to an
implementation plan using the writing-plans workflow.
