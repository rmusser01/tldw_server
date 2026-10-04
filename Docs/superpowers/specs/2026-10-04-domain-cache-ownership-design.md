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
- Pin fenced reads to the captured native `configSnapshot`. Domain single-flight
  remains owner-aware; lower-level GET coalescing must not join another owner's
  request merely because a transport scope key is identical.
- Overlapping initialization may complete out of order, but an older completion
  cannot overwrite a newer authority or newer same-owner credentials.
- For extension IPC, send only a SHA-256 authority comparison identifier and an
  opaque worker account/lifetime epoch. A bounded authority check obtains the
  epoch; dispatch and completion reject mismatches without direct fallback.
  The identifier alone does not distinguish identical cookie-session logins.
  Observe native config/cookie storage boundaries to rotate the worker epoch.
  Do not let runtime credential overrides replace a checked worker credential;
  reject a request combining snapshot and Service Prompt scopes.
- Recheck ownership after asynchronous work, including path resolution and
  response completion. Reject superseded unscoped reads with the existing scope
  changed error; never repopulate a newer epoch or delete its in-flight entry.
- Keep same-owner JWT refresh, current-owner TTL caching and coalescing, and
  explicit `requestScope`, `fresh`, and `forceRefresh` behavior compatible.
- Keep one fresh effective-config resolution per cache hit and at most two per
  fetch. Retain fresh server/API-key/removed-credential checks even without an
  account event; eliminate duplicate token/session checks after hydration and
  use the synchronous revision guard at the character path-resolution boundary.
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
Real background-proxy transport tests reproduce API-key owner joins and verify
separate dispatch, current-owner cache publication, and stale-response rejection.

Review follow-up also exercises out-of-order direct initialization, real
extension messaging/worker mismatches, identical cookie-config epoch roundtrips,
response-time epoch changes, runtime override suppression, mixed-scope rejection,
handshake cancellation/timeouts/no-fallback, and bounded storage read counts.
Race tests wait for storage, dispatch, or in-flight lookup milestones; publication
tests use payload consumption rather than revision-helper call ordinals.
ADR reassessment: no new ADR. This is a correction to existing native ownership
and IPC dispatch contracts, not a new authentication or persistence protocol.
