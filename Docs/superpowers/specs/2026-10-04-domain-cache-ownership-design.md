# Domain Cache Ownership

Task: TASK-13425

## Evidence and Scope

Current upstream dev stores character profiles by character ID and chat messages
by chat ID plus query, without connection ownership. The prototype uses domain
mixins, so fixing only the retained base methods would leave the live bug intact.
The existing saved-profile test explicitly retains Alice's unscoped cache after
switching to Bob. This work fixes that generic native client behavior only.

## Contract

- Resolve effective native connection configuration before shared cache reads
  and in-flight joins. Compare it with `connectionAuthoritiesMatch`.
- Observe `watchChatAccountChanges` once for an account-boundary revision. Even
  an identical cookie-session configuration after logout/login is a new epoch.
- Discard shared profile/message caches and in-flight indexes on a boundary.
- Recheck ownership after asynchronous work, including path resolution and
  response completion. Reject superseded unscoped reads with the existing scope
  changed error; never repopulate a newer epoch or delete its in-flight entry.
- Keep same-owner JWT refresh, current-owner TTL caching and coalescing, and
  explicit `requestScope`, `fresh`, and `forceRefresh` behavior compatible.
- No new cookies, authentication protocols, server APIs, or dependencies.

## ADR Check

ADR required: no. This enforces existing native connection/account/request-scope
ownership contracts rather than changing a durable architecture decision.
Task editing follows [ADR-059](../../ADR/059-backlog-py-task-editor-cutover.md).

## Verification

TDD against real domain mixins and the public client with synthetic transport
responses: JWT principal changes, API-key changes, server/org/auth-source changes,
same-principal JWT refresh, account events with identical cookie-session configs,
late completion, concurrent reads, and explicit-scope bypass. Run related client
tests, TypeScript checking where available, and touched-scope security assessment.
