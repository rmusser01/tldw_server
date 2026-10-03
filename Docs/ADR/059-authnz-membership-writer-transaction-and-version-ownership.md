# ADR-059: AuthNZ Membership Writer Transaction and Version Ownership

**Status:** Accepted
**Date:** 2026-10-02
**Backfilled from:** `Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md`, shared membership writer section
**Decision owner:** Human requester, approved Stage 2 design and Work Package 2 implementation in the UserProfiles workstream
**Related task:** TASK-13001.2
**Related spec/plan:** `Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md`, `Docs/superpowers/plans/2026-07-25-userprofiles-single-update-pipeline-stage2-implementation-plan.md`

## Decision

AuthNZ owns one transaction-aware organization/team membership writer that rechecks stored authorization under ordered locks and makes profile-version ownership explicit for every runtime membership mutation.

## Context

Profile versions include inherited organization/team state. Independent membership writes, ownership transfers, provisioning, and cascading scope deletion can otherwise race profile updates, leave partial provisioning behind, or invalidate a client's profile snapshot without advancing its durable version.

The approved Stage 2 design establishes this shared storage boundary before the typed UserProfiles command pipeline is introduced. Work Package 2 implements it for existing callers; this record backfills that approved decision rather than expanding its scope.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep independent endpoint and repository membership SQL | Authorization, lock ordering, and version updates can diverge between callers. |
| Acquire missing locks as mutations execute | Expanding the lock set out of order can deadlock with another writer or miss a concurrent scope change. |
| Always let both caller and writer advance profile versions | Produces duplicate version updates and obscures transaction ownership. |
| Defer all membership changes until the typed pipeline is complete | Leaves existing runtime writers outside the concurrency and version protocol needed by that pipeline. |

## Consequences

- Runtime membership DML delegates to the shared AuthNZ writer on the caller's managed transaction connection. Structural tests enforce the writer and parent-delete inventories. Offline migrations remain explicitly classified exceptions.
- PostgreSQL locks affected users, organizations, teams, and memberships in a total order. SQLite uses its existing immediate write transaction. Scope deletion validates its discovered lock set after locking and restarts within a bounded policy if the set changed; it does not add lower-order locks in place.
- Actor contexts re-read persisted authorization on that connection. Trusted-system contexts require a closed, audited reason. This complements [ADR-017](017-scoped-org-team-rbac-core-semantics.md), without making scoped roles platform grants; platform administration follows the canonical AuthNZ sets covered by [ADR-052](052-mcp-admin-claims.md).
- Direct APIs and provisioning use `WRITER_OWNS_ANCHOR`: one final profile-version update per affected user. Profile orchestration uses `CALLER_OWNS_ANCHOR`: the writer reports affected users and version-floor inputs for the caller's final update.
- Existing caller response contracts are preserved. Transaction failures and cancellation propagate rather than being reported as best-effort successes; retryable contention uses the shared bounded, sanitized transaction policy.
- Centralizing these rules increases shared-writer test and migration responsibility. New membership entry points must join this protocol rather than introducing independent SQL or a second transaction.

## Follow-up

- Work Package 3 will build the typed command pipeline on the supplied-connection and caller-owned-anchor contracts. It is not part of Work Package 2.
- Implementation and review evidence: [PR #2821](https://github.com/rmusser01/tldw_server/pull/2821) and TASK-13001.2.
