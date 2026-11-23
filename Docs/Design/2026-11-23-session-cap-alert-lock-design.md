# Session cap, new-sign-in alert and account lock

**Status:** Draft for owner approval
**Date:** 2026-11-23
**Task:** TASK-15632 (design); supersedes the options in TASK-15628
**ADR assessment:** an ADR is required. This creates durable security and persistence rules (a per-user session cap, a locked-account state enforced on every authentication path). It is written with PR 1 as the next free ADR number.

## Intent

Give a user a bounded number of signed-in sessions, tell them when a new one appears, and give them one decisive response to a suspected takeover: lock the account so that nothing works until they come back in through their email.

Why now: on 2026-11-23 a browser that would not keep the sign-in cookie created a new 30-day single-user session about every 1.7 seconds (TASK-15627 fixed the client; TASK-15628 recorded that the server has no bound). Nothing in AuthNZ limits how many sessions a user holds, nothing tells a user that someone else signed in, and the only response to a suspected takeover today is an administrator setting the account inactive.

## Decisions made by the owner

1. A user has at most **4** signed-in sessions at once.
2. When a new session is created, the user's other signed-in sessions are alerted.
3. The alert offers **lock**: all sessions are signed out and the user must come back in through an emailed link.
4. Where no email link can be sent (single-user mode, or no real mail provider configured), the cap and the alert still apply, and the alert offers **sign out all other sessions** instead of lock. Lock exists only in multi-user mode with a working mail provider.
5. While an account is locked its **API keys stop working** until it is unlocked. They are suspended, not deleted.

## Goals

- The sessions table cannot grow without bound for any user, in either auth mode, on SQLite and PostgreSQL.
- A user learns of a new sign-in from any browser they are already signed in on, and can see and end their sessions.
- A user can lock their own account from any signed-in session, and recover without an administrator when email works.
- One place decides whether an account is locked, and every way of authenticating goes through it.

## Non-goals

- Device fingerprinting, trusted devices, or "remember this browser".
- Geolocation or risk scoring of sign-ins.
- Emailing the user on every new sign-in. The alert is in-app only.
- Changing the failed-login lockout (`lockout_tracker`), MFA, or password rules.
- Closing streams and WebSockets that are already open at the moment of a lock (see Known limits).
- Making the cap configurable per user or per organisation.

## Anti-goals

- A lock that the owner cannot undo by themselves when email works.
- A lock that leaves any credential working (a session, a refresh token, an API key).
- A cap that can keep the real owner out: a full set of sessions must never block a new sign-in.
- An alert that the new session can silence for everyone before the owner sees it.

## What exists today (origin/dev 2aa1f81c)

- `sessions` is the only live session table (`AuthNZ/session_manager.py`, `AuthNZ/repos/sessions_repo.py`). Rows are created by password login, magic-link verify, the OIDC callback, the single-user cookie mint (`single_user_session.mint_single_user_session`) and an e2e support service. `create_session` is a plain insert with no count check.
- Revoking a session takes effect on the next request for JWT and cookie sessions (`verify_jwt_and_fetch_user` checks the token blacklist and revoked rows; cookie sessions are validated against the table). `revoke_all_user_sessions` exists.
- `GET /api/v1/auth/sessions`, `DELETE /api/v1/auth/sessions/{id}` and `POST /api/v1/auth/sessions/revoke-all` exist. `SessionResponse` has no "this is the current session" field. The WebUI has no sessions page; it uses the GET only as a connectivity probe.
- There is no locked state. `users.is_active=false` is enforced everywhere but also blocks magic-link sign-in. `users.locked_until` exists as a column that nothing reads. `users.is_locked` is written by an admin endpoint but has no column.
- Email: `AuthNZ/email_service.py`, provider `mock` by default (links are redacted and unusable) or `smtp`. Purpose-scoped single-use email tokens exist (`jwt_service._create_short_lived_email_token`; magic link consumes its `jti` through `token_blacklist`). Magic link and admin re-auth are multi-user only. Single-user mode has no deliverable address.
- Notifications are per user in the Collections DB (`create_user_notification`), streamed over SSE, with a closed list of kinds (`NotificationKind`). A notification can carry a link but no action. Any session can mark one read or dismiss it for the whole user.

## Approaches

### Where the alert and the lock control live

| Approach | Trade-off |
| --- | --- |
| **A. Notifications inbox + a new Active sessions page (chosen)** | Reuses the stream, toast and inbox. The notification links to the page; the page holds the buttons. Needs one new notification kind and the source session id on the notification. |
| B. Notifications with action buttons that POST | A larger change to a shared system, and a destructive button inside a toast is easy to hit by accident. |
| C. A separate per-session alert channel | Duplicates the stream the app already has, for one event type. |

### Where "locked" is stored

| Approach | Trade-off |
| --- | --- |
| **A. A dedicated lock timestamp on the user, checked in one shared function (chosen)** | One new column and one function that every authentication path calls. |
| B. Reuse `is_active=false` | Magic-link and the unlock link would be refused too, so the owner could not recover. Also indistinguishable from an administrator's deactivation. |
| C. A flag inside `users.metadata` JSON | Not indexable, easy to miss on a path, and not visible in admin listings. |

### How the cap is enforced

One sensible way: inside session creation, under a per-user lock, so that two simultaneous sign-ins cannot both pass the count. The oldest live session is revoked, not deleted, so its browser sees an ordinary sign-out and its tokens stop on the next request.

## Design

### 1. Session cap (PR 1)

- **Limit:** 4 live sessions per user. A setting `MAX_SESSIONS_PER_USER` (default 4, minimum 1) exists for tests and operators; there is no per-user override.
- **What counts as live:** a row with `is_active` true, not revoked, whose longest-lived credential has not expired: `COALESCE(refresh_expires_at, expires_at) > now`. Today the listing query ignores expiry and cleanup lags it by a day; the cap, the listing and the alert all use this one predicate.
- **What counts as a session:** rows created by password login, magic-link verify, the OIDC callback, MFA completion, single-user `/login`, and the single-user cookie mint. An MFA-pending placeholder row does not count until MFA completes. API keys are not sessions.
- **Pending rows:** MFA completion does not call `create_session`; it makes the placeholder row live through `update_session_tokens`, and nothing on the row marks it as pending today. PR 1 adds a `sessions.is_pending` column (SQLite, PostgreSQL and both DDL copies in `tests/AuthNZ/conftest.py`). A pending row neither counts nor evicts; the cap runs when a row becomes live, at `create_session` for ordinary sign-ins and at MFA completion otherwise. Someone holding only the password can therefore not sign the owner out.
- **At the cap:** making a session live revokes live sessions until the new one fits, with `revoke_reason="session_cap"`. Sign-in is never refused because of the cap.
- **Which session is evicted:** first the oldest live session from the *same browser* as the new one (same `device_id` where set, otherwise same IP address and user agent); only if there is none, the oldest live session of the user by `created_at`. Without this, one browser that keeps dropping its cookie would evict every other browser within seconds.
- **Atomicity:** count, evict and insert (or promote) run in one transaction per user. PostgreSQL takes a transaction-scoped advisory lock keyed on the user id; SQLite uses its `BEGIN IMMEDIATE`. Eviction is done with statements inside that transaction, not by calling `revoke_session` (which opens its own transaction and would wait on the outer one under SQLite); token-blacklist entries and cache invalidation for the evicted sessions are written after commit.
- **Bounded storage:** rows revoked with `revoke_reason="session_cap"` are purged by cleanup one hour after revocation instead of seven days, so a misbehaving client cannot accumulate revoked rows for a week.
- **Single-user mode:** the same cap applies to the fixed principal's cookie sessions. This is the fix for TASK-15628: a browser that drops its cookie can hold at most 4 rows, and a second or third real browser profile keeps its session.
- **Revocation cost:** Redis invalidation scans the whole `session:*` keyspace today. The cache keys are `session:{token_hash}`, so the eviction statement returns the evicted rows' token hashes and those keys are deleted directly. The cache is a hint only (validation always re-reads the database), so this is about cost, not correctness.
- **Cleanup interaction:** cleanup keeps deleting expired and long-revoked rows as now. PR 1 does not change JWT session expiry semantics.

### 2. New-sign-in alert and Active sessions page (PR 2)

- **When:** after a session becomes live (after MFA where MFA applies), if the user has at least one other live session.
- **What:** one notification for the user, kind `new_sign_in` (added to `NotificationKind`), severity warning, title "New sign-in to your account", message naming the browser/OS family parsed from the user agent, the IP address and the time. It links to the Active sessions page. The existing `source_domain="auth_session"` and `source_job_id=<new session id>` columns record which session caused it; no notifications schema change.
- **Not alerting itself:** the WebUI learns its own session id from the sessions listing (`is_current`) and does not toast or badge a `new_sign_in` whose source is itself. The inbox entry stays visible to it.
- **Not silenceable by a newer session:** a `new_sign_in` notification can be marked read, dismissed or snoozed only by a session that was created before the notification's source session. The source session, any later session, and credentials that are not sessions (API keys) are refused. The preference toggles do not offer an off switch for this kind.
- **Storm guard:** the cap bounds how many sessions exist, not how often they are created. `dedupe_key` is user + new session's IP + user agent + a 10-minute time bucket (the key is permanently unique and a clash returns the existing row unchanged, so the bucket is what makes it expire). Repeated sign-ins from the same browser inside a bucket produce one notification; it is not re-raised.
- **Single-user cookie sessions** alert the same way (other browser profiles of the same install see it).
- **Sessions API:** `SessionResponse` gains `is_current` and `device_label` (derived from the user agent). List and revoke-one are used as they are. Revoke-all today signs out the caller too and existing clients may rely on that, so it gains an opt-in `keep_current` flag that the new page sets; its default behaviour is unchanged. JWT principals carry no session id, so `is_current` and `keep_current` read the session id from the bearer token the way logout already does.
- **Active sessions page** (`/settings/sessions`, linked from the account menu and from the notification): lists live sessions with device label, IP, signed-in time, last activity, and "This device". Per-row "Sign out"; "Sign out all other sessions"; and, where lock is available, "Lock account" with a confirmation that states what will happen and where the link will be sent (masked address). Where lock is not available the page says why in one line.
- **Is lock available?** A capability flag from the server: multi-user mode, a mail provider other than `mock`, and the user has a verified email address.

### 3. Account lock and email-link unlock (PR 3)

- **State:** `users.security_locked_at` (timestamp, null when unlocked) and `users.security_locked_by_session` (the session id that requested it, for the audit trail), on SQLite and PostgreSQL. The unused `locked_until` column and the existing admin `is_locked` field (which today means the failed-login lockout in the admin profile API) are left exactly as they are.
- **Locking** (`POST /api/v1/auth/account/lock`, any authenticated session of the user, CSRF-protected, no password prompt). The steps cannot share one transaction (session revocation and the token blacklist open their own, Redis cannot roll back, and an SMTP send must not run inside SQLite's write lock), so the order is: (1) hand the unlock email to the mail provider; if that fails, stop with an error and change nothing; (2) set the lock; (3) revoke every session and token of the user, including the caller's. From step 2 the enforcement function refuses everything, so a failure in step 3 cannot leave a working credential; step 3 is retried by the request and is idempotent. A link sent for a lock that then failed to commit is harmless: redeeming it on an unlocked account does nothing.
- **One enforcement function**, `assert_account_not_locked(user)`, called from: password login (after the password has been verified, so that a wrong password cannot reveal the lock), MFA completion, token refresh, the per-request JWT check, single-user cookie validation (never true in single-user mode, but the call is there), magic-link verify, the OIDC callback and password reset. On these paths a locked account gets HTTP 423 with a stable error code `account_locked`. Password reset does not unlock.
- **Anonymous request endpoints keep their generic answers.** Magic-link request and forgot-password already answer the same whether or not the account exists; they do not return 423. For a locked account they send the unlock email instead of their usual one, so the owner still receives something usable.
- **API keys:** `APIKeyManager.validate_api_key` is called directly from about eight places (the core resolver, MCP over HTTP and WebSocket, the LLM budget middleware, audio streaming, CSRF), so the lock is enforced inside it: a key of a locked user validates as invalid. The core resolver recognises the case and answers 423 `account_locked`; the other callers answer with their existing refusal (401 or a WebSocket close). Nothing is written to the key rows, so unlocking restores them exactly. The MCP paths do not check `is_active` today; that existing gap is fixed by the same change.
- **Unlock link:** a purpose-scoped (`account_unlock`), single-use token with a 30-minute life, built with the existing short-lived email token helper and consumed through the existing single-use mechanism. The link opens a WebUI page that redeems it at `POST /api/v1/auth/account/unlock`. Redeeming clears the lock and signs the user in with a new session. For an account with MFA enabled the lock is cleared only when MFA completes: the link yields a pending session, as password login does, and API keys stay suspended until then. Magic-link verify has no MFA step today, so the unlock endpoint builds that flow itself.
- **Requesting another link:** the sign-in page shows "Account locked" with "Send me a new link" when it receives `account_locked`. That request is anonymous, answers the same whether or not the account exists or is locked, and is rate-limited per address and per IP like the magic-link request.
- **Email template:** a new `account_unlock` template stating that the account was locked from a signed-in session, when, and from which browser and IP, with the link and its lifetime.
- **Admin:** an administrator can clear a user's security lock through a new explicit action (`POST /admin/users/{id}/security-unlock`, with the same admin re-authentication that changing `is_active` requires); admin user listings show the security lock as its own field. This is the recovery path if a user's mailbox is lost. Administrators do not get a "security lock" action: to block a user they deactivate the account as today, which the user cannot undo.
- **Audit:** lock, unlock, unlock-link-sent and cap evictions are written to the existing audit log with the user id, acting session id, IP and user agent.
- **After unlock:** the user lands signed in with one session. The page suggests changing the password and reviewing API keys; nothing is forced.

## Constraints

- SQLite and PostgreSQL both, with migrations for each; `tests/AuthNZ/conftest.py` carries its own users/sessions DDL and must be updated in step.
- Multi-user JWT revocation already works on the next request; nothing in this design depends on shortening token lifetimes.
- Single-user mode has no deliverable email and its API key can always mint a new cookie session, so lock is not offered there by decision 4.
- The notifications kind list is closed: an unknown kind makes the inbox endpoint fail, so the kind is added in the same PR that writes it.
- MFA is available only on PostgreSQL in multi-user mode; the MFA parts of unlock apply only there.
- No new dependencies. User-agent labelling uses a small in-repo parser (browser and OS family only); an unrecognised agent is shown as "Unknown browser".

## Known limits

- **Scheduled and background work (owner decision pending):** automations, reminders, workflows and agent tasks run as the user without authenticating (the workflows scheduler mints its own tokens). An intruder could set one up before the owner locks the account, and this design as written does not stop it.
- A lock does not close SSE streams or WebSockets that are already open; they end when they next authenticate or reconnect. Recorded as a follow-up rather than solved here.
- The alert is in-app. A user with no other signed-in browser learns nothing from it; that case is what the email on lock, and existing email notices, are for.
- Any signed-in session can lock, including an intruder's. The cost to the owner is one email round trip. Requiring the password first was rejected: the password may be what leaked.
- IP addresses in alerts are whatever the server sees; behind a proxy that is the proxy's address unless forwarding is configured.

## Delivery

Three PRs against `dev`, each with its own Backlog task, independently reviewed:

1. **Cap** - closes TASK-15628. Backend only. ADR written here.
2. **Alert + Active sessions page** - backend notification and API fields, WebUI page and toast handling.
3. **Lock + unlock link** - schema, enforcement function on every path, endpoints, email template, WebUI lock and locked-sign-in flows, admin lock/unlock.

Each PR is usable without the next: after PR 1 the table is bounded; after PR 2 users can see and end sessions; PR 3 adds lock.

## Left to the builder

- Exact SQL, index choice for the live-session predicate, and migration numbering.
- Wording of UI copy and the email, within the facts stated above.
- The user-agent label rules beyond "browser family and OS family".
- Test layout, following the existing AuthNZ unit / SQLite / PostgreSQL split; every enforcement point in section 3 gets a test that a locked account is refused there.
