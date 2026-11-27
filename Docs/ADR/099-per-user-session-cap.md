# ADR-099: Cap live sessions per user

**Status:** Accepted
**Date:** 2026-11-24
**Backfilled from:** not backfilled
**Decision owner:** repository owner (design approval 2026-11-23, eviction order revised 2026-11-24)
**Related task:** TASK-15628, TASK-15632
**Related spec/plan:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md` (section 1), `Docs/superpowers/plans/2026-11-23-session-cap.md`

## Decision

Every AuthNZ session row is created through one per-user-serialized, capped insert that keeps at most `MAX_SESSIONS_PER_USER` (default 4, minimum 1) live sessions: it revokes (does not delete) the least recently active live session of the same client install first (sign-ins without an install id are one group), then the user's least recently active live session, and sign-in is never refused.

## Context

The 2026-11-23 storm created a new 30-day session row about every 1.7 seconds. Nothing in AuthNZ bounded the number of live sessions per user. In single-user mode every browser looked identical, so there was nothing to group by.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Refuse sign-in at the cap | It can lock the owner out of their own server. |
| Short-lived mint, extended on first use (TASK-15628 option b) | It does not bound live rows from a fast client. |
| Group by IP or user agent | Both are spoofable, and IP merges separate machines behind one NAT. |
| Evict the oldest created session | It signs out a long-lived, actively used browser before an abandoned newer one. |
| Delete instead of revoke | The evicted client could not be told why it was signed out. |
| A single-key advisory lock | The key space is shared with other code. The cap uses a namespaced two-key lock. |
| Per-row `revoke_session` calls | Each opens its own transaction, so the cap would not be atomic. |

## Consequences

- The live predicate `COALESCE(refresh_expires_at, expires_at) > now` is shared by the cap, the session listing, `require_online` and cleanup.
- An idle session now stays refreshable for the full `REFRESH_TOKEN_EXPIRE_DAYS`. Before this change, cleanup deleted the row about 1 day after the access token expired, which ended the session early.
- `require_online` changes in two ways. It treats a multi-user session as online for as long as it is refreshable, up to `REFRESH_TOKEN_EXPIRE_DAYS`. A single-user account that only called `/login` no longer counts as online.
- Evicted JWT clients get 401 `session_replaced`, read from the `token_blacklist` reason, which outlives the purged session row. Only the request path (`verify_jwt_and_fetch_user`) and `POST /auth/refresh` answer it. Other revoked-token rejection sites (`jwt_service.verify_token_async` and `refresh_access_token`, `require_token_scope`, service tokens) keep the generic 401. Single-user cookie sessions get a plain 401 and sign in again on the next page load (the page does not re-mint mid-session).
- Cap rows are purged after `max(1 h, access-token lifetime)`.
- `token_blacklist` is never purged today (`TokenBlacklist.cleanup_expired` has no caller). TASK-15642 schedules that purge.
- Each evicted JWT session costs up to two `token_blacklist` writes after the cap transaction commits. Only the `SESSION_CAP_INLINE_BLACKLIST_LIMIT` (100) most recently active evicted sessions are blacklisted; a very large pre-upgrade backlog pays that once, on the first capped sign-in. The rest stay rejected through their revoked rows, and get a generic 401 (not `session_replaced`) once those rows are purged.
- The session listings' `expires_at` is now the session end, `COALESCE(refresh_expires_at, expires_at)`, not the access-token expiry.
- The admin and monitoring "active sessions" counters (`admin_system_service.py`, `monitoring_repo.py`) keep the old access-token definition and do not use the shared live predicate, so they can disagree with the listings.
- The install id is a grouping hint only. Scripts should use API keys. Password-login clients without an id (the CLI wizard, the admin-ui login route) share slots with each other. So do sign-ins through the SSO/OIDC callback, which is a browser redirect and carries no install id. The admin E2E bootstrap (`bootstrap_admin_e2e_jwt_session`) also creates sessions without an install id and is now capped per principal.
- The MFA challenge is consumed after the code verifies, and a session row exists only once MFA completes. The single-user `/login` creates no row. API keys are not sessions.
- On PostgreSQL a session logged out at the same instant as an eviction can be stamped `session_cap`. The cap's upper bound always holds.
- Test suites that sign one user in from more than four parallel workers must raise `MAX_SESSIONS_PER_USER` in that suite's backend env.
- The account-lock rule from spec section 3 gets its own ADR with PR 3.

## Follow-up

TASK-15642 schedules the `token_blacklist` purge in the AuthNZ scheduler.
