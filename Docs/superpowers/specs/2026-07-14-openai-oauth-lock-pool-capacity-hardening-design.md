# OpenAI Credential Lock Pool Capacity Hardening Design

**Task:** TASK-12963

## Problem

The PostgreSQL OpenAI OAuth refresh lock currently acquires a connection from
the main AuthNZ pool and holds that session-level advisory lock connection
while the external token refresh runs. Same-user contenders release losing
connections between attempts, but distinct users can each win a different
advisory lock. A pool-sized distinct-user burst can therefore occupy every
main AuthNZ connection and starve unrelated authentication and authorization
work.

SQLite uses a native file lock and Redis uses an ownership-token lease. Those
backends do not consume the main PostgreSQL pool, but their selection and
failure behavior need explicit regression coverage and deployment guidance.

## Chosen Design

`DatabasePool` will own a second asyncpg pool dedicated to session-level OpenAI
credential-mutation advisory locks. It will use the same PostgreSQL DSN and connection lifetime
settings as AuthNZ, with `min_size=0` and a fixed `max_size=4`. Zero idle
connections avoids a standing database cost; the four-connection ceiling
bounds a refresh burst independently of the main AuthNZ pool.

`DatabasePool` will expose
`acquire_openai_credential_lock_connection(timeout=...)`. The BYOK runtime will
use only that method for PostgreSQL advisory locks and will expose one shared
`openai_credential_mutation_lock(...)` context manager for refresh and all
whole-row OpenAI credential mutations. The existing refresh-lock helper will
delegate to it for compatibility. The runtime will keep the existing
connection-bound secret repository, reload, and compare-and-swap behavior. It
will fail closed if the dedicated pool is unavailable rather than falling back
to the main pool.

The connection-bound repository adapter exposes the exact PostgreSQL methods
used by locked mutations: `fetchone` for reload/upsert/CAS and `execute` for
revoke/touch paths. Both delegate to the advisory-lock-owning connection, so a
locked delete cannot silently borrow the main pool or run on another session.

The public lock boundary canonicalizes provider identity before deriving its
key. `openai`, case variants, and registered aliases such as `oai` therefore
contend on the same row lock; non-OpenAI identities are rejected rather than
creating a misleading independent lock namespace.

The runtime will also expose
`openai_oauth_credential_generation(payload)` as the public comparison seam
used by mutation callers. Its digest remains based on the OAuth access token:
an access-token change proves that another contender completed a refresh,
while refresh-token rotation or unrelated metadata alone does not. The
existing private helper remains as a compatibility wrapper.

The dedicated pool will be created with the main PostgreSQL pool and closed by
the existing `DatabasePool.close()` lifecycle before the main pool is closed.
SQLite will not create a dedicated pool and will continue using `FileLock`.

## Backend Behavior

- Missing or invalid `OPENAI_OAUTH_REFRESH_LOCK_BACKEND` resolves to `db`.
- The existing environment name is retained for compatibility, but the chosen
  backend serializes every OpenAI credential mutation, not only token refresh.
- `db` uses the dedicated PostgreSQL pool or the process-shared SQLite file
  lock, according to the AuthNZ database backend.
- Explicit `redis` requires a nonempty valid `REDIS_URL`; missing configuration
  fails with the bounded `credential_store_unavailable` error and never falls
  back to memory or DB locking.
- `memory` remains an explicit single-process compatibility choice.

## Failure and Cancellation Semantics

Pool acquisition, advisory-lock polling, and file-lock polling remain bounded
by the existing OAuth lock timeout. Advisory unlock, file-lock release, and
connection return must complete before cancellation propagates. A failed or
missing dedicated PostgreSQL pool is a credential-store failure, not a reason
to borrow a main AuthNZ connection.

PostgreSQL release succeeds only when `pg_advisory_unlock` positively confirms
ownership. A false result or transport failure after a successful protected
body fails closed with `credential_store_unavailable`. Cleanup failure is
logged without secret material when an exception is already propagating from
the protected body, and that original exception is preserved.

PostgreSQL session advisory locks require a direct PostgreSQL connection or a
session-pooled PgBouncer connection. PgBouncer transaction pooling can switch
server sessions between lock and unlock statements and is therefore unsafe;
those deployments must select the Redis lock backend.

## Verification

Regression coverage will prove:

1. A dedicated-pool-sized burst of distinct-user advisory-lock owners does not
   prevent unrelated work from acquiring the main AuthNZ pool.
2. The dedicated pool and shared mutation-lock API are bounded, lazily
   connected (`min_size=0`), and closed with the main pool.
3. Real SQLite file locks serialize callers running in independent event loops,
   time out predictably, release on owner cancellation, and do not let a
   cancelled waiter release another owner's lock.
4. Backend settings default and invalid values normalize to `db`.
5. Explicit Redis without `REDIS_URL` fails closed.
6. OpenAI aliases and case variants contend on one canonical lock, and
   non-OpenAI identities are rejected.
7. PostgreSQL unlock must be positively confirmed, completes before
   cancellation returns, and does not mask an existing protected-body error.
8. Bound PostgreSQL revoke and CAS operations execute on the lock-owning
   connection; the CAS query contains one active-row/blob predicate.
9. The public OAuth generation seam changes only when access-token generation
   changes and does not expose token material.
10. Canonical environment and horizontal-scaling documentation identifies the
   DB default, the four-session per-process ceiling, the PgBouncer transaction
   pooling restriction, and Redis as the required scale path for multi-process
   or high credential-mutation concurrency.

## Rejected Alternatives

- A global BYOK-owned asyncpg pool would duplicate startup, event-loop, reset,
  and shutdown state already owned by `DatabasePool`.
- Opening a new PostgreSQL connection per refresh would add connection churn
  and still require separate semaphore lifecycle state.
- A database lease table would require schema, clock, renewal, and abandoned
  lease handling for a problem already solved by PostgreSQL advisory locks.
