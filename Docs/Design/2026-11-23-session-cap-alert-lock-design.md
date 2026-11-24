# Session cap, new-sign-in alert and account lock

**Status:** Approved by the owner 2026-11-23; revised the same day after an independent review against the code (see "Changes from review")
**Date:** 2026-11-23
**Task:** TASK-15632 (design); supersedes the options in TASK-15628
**ADR assessment:** an ADR is required. This creates durable security and persistence rules (a per-user session cap, a locked-account state enforced on every authentication path and on background work). It is written with PR 1 as the next free ADR number.

## Intent

Give a user a bounded number of signed-in sessions, tell them when a new one appears, and give them one decisive response to a suspected takeover: lock the account so that nothing works until they come back in through their email.

Why now: on 2026-11-23 a browser that would not keep the sign-in cookie created a new 30-day single-user session about every 1.7 seconds (TASK-15627 fixed the client; TASK-15628 recorded that the server has no bound). Nothing in AuthNZ limits how many sessions a user holds, nothing tells a user that someone else signed in, and the only response to a suspected takeover today is an administrator setting the account inactive.

## Decisions made by the owner

1. A user has at most **4** signed-in sessions at once.
2. When a new session is created, the user's other signed-in sessions are alerted.
3. The alert offers **lock**: all sessions are signed out and the user must come back in through an emailed link.
4. Where no email link can be sent (single-user mode, or no SMTP provider configured), the cap and the alert still apply, and the alert offers **sign out all other sessions** instead of lock. Lock exists only in multi-user mode with SMTP.
5. While an account is locked its **API keys stop working** until it is unlocked. They are suspended, not deleted.
6. While an account is locked its **scheduled and background work is paused** until it is unlocked.
7. After unlock, **API keys created in the 7 days before the lock stay suspended** until the user restores each one.

## Goals

- The sessions table cannot grow without bound for any user, in either auth mode, on SQLite and PostgreSQL.
- A user learns of a new sign-in from any browser they are already signed in on, and can see and end their sessions.
- A user can lock their own account from any signed-in browser session, and recover without an administrator when email works.
- One place decides whether an account is locked, and every way of authenticating or running work for the user goes through it.

## Non-goals

- Device fingerprinting, trusted devices, or "remember this browser". (The per-install id in section 1 only groups a browser's own sessions for eviction; it grants nothing.)
- Geolocation or risk scoring of sign-ins.
- Emailing the user on every new sign-in. The alert is in-app only.
- Changing the failed-login lockout (`lockout_tracker`), MFA, or password rules.
- Closing streams and WebSockets that are already open at the moment of a lock, and stopping jobs that are already executing (see Known limits).
- Making the cap configurable per user or per organisation.

## Anti-goals

- A lock that the owner cannot undo by themselves when email works.
- A lock whose email an intruder can redirect to themselves.
- A lock that leaves any credential working (a session, a refresh token, an API key, a token minted before the lock).
- A cap that can keep the real owner out: a full set of sessions must never block a new sign-in.
- An alert that the new session can silence for everyone before the owner sees it.

## What exists today (origin/dev 2aa1f81c, verified by the review)

- `sessions` is the only live session table (`AuthNZ/session_manager.py`, `AuthNZ/repos/sessions_repo.py`). Rows are created by password login, magic-link verify, the OIDC callback, the single-user cookie mint (`single_user_session.mint_single_user_session`) and an e2e support service. `create_session` is a plain insert with no count check.
- `sessions.device_id` is set only by the single-user cookie mint, always to the constant `single-user-cookie:v1`, and cookie validation requires that exact value. Multi-user logins never set it. The bundled Next.js proxy forwards the browser's user agent but not its IP.
- Single-user `POST /login` also inserts a session row that stores the API key as both tokens; nothing ever validates that row.
- Revoking a session takes effect on the next request (`verify_jwt_and_fetch_user` checks the token blacklist and revoked rows; cookie sessions are validated against the table). `revoke_all_user_sessions` exists but blacklists only tokens that belong to session rows: self-service virtual-key JWTs (up to 24 h), impersonation tokens and workflow-minted tokens are not touched.
- `GET /api/v1/auth/sessions`, `DELETE /api/v1/auth/sessions/{id}` and `POST /api/v1/auth/sessions/revoke-all` exist and accept any credential, including an API key. `SessionResponse` has no "current session" field. The WebUI has no sessions page.
- There is no locked state. `users.is_active=false` is enforced on request authentication but also blocks magic-link sign-in, and nothing in Jobs, the Scheduler or the workflow engine checks it. `users.locked_until` is a column nothing reads. The admin `is_locked` field writes a column that does not exist.
- **Email can be changed by any signed-in session with no password and no confirmation** (`PUT /users/me`, `PATCH /users/me/profile` → `update_user_email`), and `is_verified` stays true.
- Email: `AuthNZ/email_service.py`, provider `mock` by default (links are redacted and unusable) or `smtp`; any other value fails. The SMTP connection has no timeout and `send_email` returns False instead of raising. Magic link is single-use (its `jti` goes into `token_blacklist`), multi-user only, and has no MFA step.
- `APIKeyManager.validate_api_key` has about nine direct callers, never loads the user row, records usage before returning, and turns every exception except `DatabaseError` into None. `authenticate_api_key_user` loads the user and checks `is_active`. The MCP paths do not check `is_active`.
- Notifications are per user in the Collections DB (`create_user_notification`), streamed over SSE, with a closed list of kinds. Any session can mark one read or dismiss it for the whole user. The WebUI only increments an unread count for a new notification (no toast); the extension raises an OS notification for every event.
- Telegram, Slack and Discord integrations queue jobs owned by the linked user with no credential of that user.

## Approaches

### Where the alert and the lock control live

| Approach | Trade-off |
| --- | --- |
| **A. Notifications inbox + a new Active sessions page (chosen)** | Reuses the stream and inbox. The notification links to the page; the page holds the buttons. Needs one new notification kind and a small banner/toast in the WebUI. |
| B. Notifications with action buttons that POST | A larger change to a shared system, and a destructive button inside a toast is easy to hit by accident. |
| C. A separate per-session alert channel | Duplicates the stream the app already has, for one event type. |

### Where "locked" is stored

| Approach | Trade-off |
| --- | --- |
| **A. A dedicated lock timestamp on the user, checked in one shared function (chosen)** | One new column and one function that every authentication path calls. |
| B. Reuse `is_active=false` | Magic-link would be refused too, so the owner could not recover. Also indistinguishable from an administrator's deactivation. |
| C. A flag inside `users.metadata` JSON | Not indexable, easy to miss on a path, and not visible in admin listings. |

### How the owner comes back

| Approach | Trade-off |
| --- | --- |
| **A. The existing magic-link sign-in clears the lock (chosen)** | No new token type, endpoint or page: the lock email carries a magic link, and "send me a new link" is the existing magic-link request. |
| B. A separate `account_unlock` token, endpoint and page | Three new surfaces that do what magic link already does. |

### How the cap is enforced

One sensible way: inside session creation, under a per-user lock, so that two simultaneous sign-ins cannot both pass the count. The evicted session is revoked, not deleted, so its tokens stop on the next request and its browser can be told why.

## Design

### 0. Email-change hardening (PR 0, prerequisite for lock)

Without this, an intruder with a session changes the email and then locks: every recovery link goes to the intruder. It is a takeover hole today, independent of lock.

- Both self-service email-change paths require the **current password** in the same request. An account with no password (SSO only) cannot change its email self-service; the response says to ask an administrator.
- On a change, the old address is kept in `users.previous_email` with `users.email_changed_at`, and a notice is sent to the old address.
- For **7 days** after a change, lock and unlock mail is sent to **both** the previous and the current address. (This covers the case where the password itself leaked.)
- Admin email changes are unchanged.

### 1. Session cap (PR 1)

- **Limit:** 4 live sessions per user. A setting `MAX_SESSIONS_PER_USER` (default 4, minimum 1) exists for tests and operators; there is no per-user override.
- **What counts as live:** a row with `is_active` true, not revoked, whose longest-lived credential has not expired: `COALESCE(refresh_expires_at, expires_at) > now`. The cap, the listing and the alert all use this one predicate. Changing it also changes the other callers of `get_active_sessions` (`require_online` in the workflows scheduler, the admin session listing, the `/users/sessions` aliases); each gets a test. Cleanup today deletes by access-token expiry, which removes idle sessions whose refresh token is still valid; cleanup moves to the same predicate.
- **What counts as a session:** rows created by password login (after MFA where MFA applies), magic-link verify, the OIDC callback, and the single-user cookie mint. Single-user `POST /login` stops inserting its row (nothing validates it, and it would let a script evict real browsers). API keys are not sessions.
- **MFA:** the session row is created when MFA completes, not before; the MFA payload already carries the user id. There is no pending row and no new column, and someone holding only the password cannot sign the owner out. (Builder: confirm nothing else reads the placeholder row before removing it.)
- **Per-install id:** the WebUI and the extension generate a random id once, keep it in localStorage, and send it on login and on the cookie mint. It is stored in `sessions.device_id` (single-user: `single-user-cookie:v1:<id>`, with validation changed from exact match to that prefix). It is a grouping hint only; a client that lies about it can only evict its own claimed group or fall through to the default rule.
- **At the cap:** creating a session revokes live sessions until the new one fits, with `revoke_reason="session_cap"`. Sign-in is never refused because of the cap.
- **Which session is evicted:** first the least recently active live session in the **same group** as the new one; if there is none, the least recently active live session of the user (`COALESCE(last_activity, created_at)`, then `created_at`). A group is one `device_id`; sign-ins that send no id (scripts, the CLI wizard, the admin UI's login route, OIDC redirects) form one group of their own, so a script that logs in repeatedly replaces its own sessions before any browser's. Least-recently-active is used rather than oldest-created because refresh keeps a daily-used session alive indefinitely with its original creation time; evicting by creation time would sign out the most-used device first. IP address and user agent are not used: they are spoofable and merge machines behind one NAT. Without the group rule, one browser that keeps dropping its cookie would evict every other browser within seconds; with it, that browser only replaces its own row.
- **Atomicity:** count, evict and insert run in one transaction per user. PostgreSQL takes a transaction-scoped advisory lock using the two-key namespaced form (as `user_provider_secrets_repo` does; the single-key space is shared with other code). SQLite uses its `BEGIN IMMEDIATE`. Eviction is done with statements inside that transaction, not by calling `revoke_session` (which opens its own). Token-blacklist entries for the evicted sessions are written after commit. The account-lock state is re-read inside this transaction (see section 3).
- **Telling the evicted browser:** a multi-user request or refresh on a session revoked with `session_cap` gets 401 with code `session_replaced` (the reason is read from the token blacklist, so it survives the purge below), and the WebUI shows "You were signed out because your account was signed in on another device" on the sign-in page.
- **Bounded storage:** rows revoked with `session_cap` are purged by cleanup after `max(1 hour, access-token lifetime)` instead of seven days.
- **Single-user mode:** the same cap applies to cookie sessions. This is the fix for TASK-15628: a browser that drops its cookie holds one row (its own id), and other browser profiles keep theirs.

### 2. New-sign-in alert and Active sessions page (PR 2)

- **When:** after a session is created, if the user has at least one other live session. The notification write goes to the per-user content database and is best-effort, after the response.
- **What:** one notification, kind `new_sign_in` (added to `NotificationKind` in the same PR), severity warning, title "New sign-in to your account", message naming the browser/OS family, the IP address and the time. It links to the Active sessions page. `source_domain="auth_session"` and `source_job_id=<new session id>` record which session caused it; no notifications schema change.
- **Seeing it:** the WebUI shows a banner for unread `new_sign_in` notifications on load and when one arrives on the stream (today it only bumps a count). An owner who was signed out by the cap therefore sees it on their next sign-in.
- **Not alerting itself:** a client suppresses the banner/OS notification when the notification's IP and user-agent label match its own current session (from `is_current` in the sessions listing). The inbox entry stays visible.
- **Not silenceable by a newer session:** a `new_sign_in` notification can be marked read, dismissed or snoozed only by a browser session created before the notification's source session. The source session, later sessions and non-session credentials are refused. Once the notification is 24 hours old any browser session may dismiss it, so alerts cannot become undismissable. The preference toggles do not offer an off switch for this kind.
- **Storm guard:** `dedupe_key` is user + new session's `device_id` (or IP + user agent when absent) + a 10-minute bucket. Repeated sign-ins from one browser inside a bucket produce one notification.
- **Client IP:** the bundled Next.js proxy forwards the browser's IP, and the backend uses it only where its existing trusted-proxy handling allows; otherwise the alert shows what the server saw.
- **Sessions API:** `SessionResponse` gains `is_current` and `device_label`. Revoke-all gains an opt-in `keep_current` flag (default behaviour unchanged). The session id is read from the bearer token the way logout already does.
- **Interactive credential required:** revoke-one, revoke-all, dismissing a `new_sign_in`, and lock (section 3) require a browser session principal (an access token carrying a session id, or a single-user cookie session). API keys, virtual keys and service tokens are refused. A stolen API key can therefore not sign the owner out or lock the account.
- **Active sessions page** (`/settings/sessions`, linked from the account menu and from the notification): lists live sessions with device label, IP, signed-in time, last activity, and "This device". Per-row "Sign out"; "Sign out all other sessions"; and, where lock is available, "Lock account" with a confirmation that states what will happen, where the link will be sent (masked address or addresses), and that team API keys owned by this account stop too. Where lock is not available the page says why in one line.
- **Is lock available?** A capability flag from the server: multi-user mode, mail provider is `smtp`, and the user has a verified email address.

### 3. Account lock and email-link unlock (PR 3)

- **State:** `users.security_locked_at` (null when unlocked) and `users.credentials_valid_after` (set at unlock), on SQLite and PostgreSQL. The acting session is recorded in the audit log, not on the row. The unused `locked_until` column is left alone; the admin `is_locked` field that writes a missing column is removed.
- **Locking** (`POST /api/v1/auth/account/lock`; browser session only, CSRF-protected, no password prompt). Order: (1) set the lock; (2) revoke every session and token of the user, including the caller's; (3) send the lock email. From step 1 the enforcement function refuses everything, so a failure in step 2 cannot leave a working credential; step 2 is idempotent. If the email fails, the response says the account is locked and to use "Send me a new link" on the sign-in page. The SMTP connection gets a timeout and the send result is checked.
- **One enforcement function**, `assert_account_not_locked(user, token_iat=None)`, called from: password login (after the password is verified, so a wrong password cannot reveal the lock), token refresh, the per-request JWT check (`verify_jwt_and_fetch_user`, which WebSockets and MCP also use), the OIDC callback, password reset, the virtual-key mint, admin impersonation, and inside the cap transaction (so a sign-in racing the lock cannot leave a session behind). A locked account gets HTTP 423 with code `account_locked`. It also refuses any JWT issued before `credentials_valid_after` and any session row created before it; that is what stops tokens that have no session row (virtual-key JWTs, workflow-minted tokens) from working again after unlock. Password reset does not unlock.
- **API keys:** the lock is checked inside `validate_api_key`, before usage is recorded, with a direct (uncached) lookup; a key of a locked user validates as invalid and callers answer with their existing refusal (401 or a WebSocket close). Nothing is written to the key rows. The MCP paths' missing `is_active` check is fixed by the same change.
- **API keys after unlock:** keys created more than 7 days before the lock work again immediately. Keys created in the 7 days before the lock stay suspended (`api_keys.suspended_at`) until the user restores each one from the API-keys page, which lists them with created time and last-used IP. (Owner decision 7.)
- **Unlock:** the lock email (template `account_locked`: locked from a signed-in session, when, from which browser and IP, link and lifetime) carries an ordinary magic-link token. `magic-link/verify` on a locked account clears the lock, sets `credentials_valid_after = now`, and signs the user in with one new session. For an account with MFA the verify step returns the existing MFA challenge with an `unlock` marker in its payload, and `/mfa/login` clears the lock when it completes; the lock stays until then.
- **Requesting another link:** the sign-in page shows "Account locked" with "Send me a new link" when it receives `account_locked`. That is the existing anonymous magic-link request (same generic answer, same rate limits); for a locked account it sends the `account_locked` template. Forgot-password for a locked account sends the same email instead of a reset link.
- **Chat integrations:** Telegram, Slack and Discord refuse a linked user who is locked where the request arrives.
- **Admin:** `POST /admin/users/{id}/security-unlock` (with the admin re-authentication that changing `is_active` requires) clears the lock; admin listings show it as its own field; the `reset_admin_password` CLI also clears it, so a sole admin with broken SMTP is not stranded. Admin impersonation of a locked user is refused. Administrators do not get a "security lock" action: to block a user they deactivate the account as today.
- **Audit:** lock, unlock, lock-email-sent and cap evictions are written to the audit log with user id, acting session id, IP and user agent.
- **After unlock:** the user lands signed in with one session on a page that suggests changing the password and shows the suspended recent API keys.

### 4. Pausing background work (PR 4)

One helper in AuthNZ, `blocked_owner_ids()` / `owner_blocked(user_id)`: true when `security_locked_at` is set **or `is_active` is false** (administrator deactivation does not stop background work today; the same check fixes that). Backed by the AuthNZ database with a cache of about 30 seconds, because the Jobs database is separate and sidecar workers cannot join against it.

- **Jobs claim:** every worker claims through `JobManager.acquire_next_job`. The candidate SQL in both backends (`Jobs/operations/sqlite/lifecycle.py`, `.../postgres/lifecycle.py`) excludes jobs whose `owner_user_id` is blocked. Queued jobs wait and consume no retries. The worker SDK's claim-then-release hook is not used (a released job is picked first again and would block its queue).
- **Scheduled fires are skipped at the producer:** `_run_schedule` and `_run_acp_schedule` in `services/workflows_scheduler.py` and `_fire` in `services/scheduled_task_automation_scheduler.py` skip a blocked owner and compute the next run, the way `require_online` already does. Scheduler handlers (`core/Scheduler/handlers/workflows.py`, `watchlists.py`, `acp.py`) complete an already-queued task of a blocked owner as skipped rather than failing it.
- **Running workflows:** `_wait_if_paused` in `core/Workflows/engine.py` also waits while the owner is blocked, so runs stop at the next step boundary and resume on unlock.
- **Webhook retries:** `services/workflows_webhook_dlq_service.py` skips blocked owners (it re-sends run data to outside URLs an intruder could have configured).
- **Due while locked:** skipped, not caught up. Catching up would run an intruder's automations the moment the owner unlocks. Producers that hold one slot per schedule (reminders, reading digest) may run one stale slot on unlock.

## Constraints

- SQLite and PostgreSQL both, with migrations for each; `tests/AuthNZ/conftest.py` carries its own users/sessions DDL and must be updated in step.
- Single-user mode has no deliverable email and its API key can always mint a new cookie session, so lock is not offered there, and no lock check is added to single-user cookie validation.
- The notifications kind list is closed: an unknown kind makes the inbox endpoint fail, so the kind is added in the PR that writes it.
- MFA is available only on PostgreSQL in multi-user mode; the MFA part of unlock applies only there.
- No new dependencies. User-agent labelling uses a small in-repo parser (browser and OS family only); an unrecognised agent is shown as "Unknown browser".

## Known limits

- A single-user cookie session evicted by the cap gets a plain 401 and its browser silently signs in again; only multi-user sessions are told why.
- Scripts that sign in with a password share the no-id group with OIDC sign-ins and the admin UI; scripts should use API keys.
- A job that is already executing when the lock is set finishes; Jobs have no cooperative pause.
- Not paused: the embeddings workers on Redis Streams, detached in-process tasks (evaluation runs, research, sandbox, persona, WebSub push ingestion), and direct per-user loops (claims-alert emails, WebSub renewal, metrics). Recorded as a follow-up task.
- A lock does not close SSE streams or WebSockets that are already open; they end when they next authenticate or reconnect.
- The background pause takes up to the cache interval (about 30 seconds) to apply.
- The alert is in-app. A user with no other signed-in browser sees it on their next sign-in, not before.
- Any signed-in browser session can lock, including an intruder's. The cost to the owner is one email round trip. Requiring the password first was rejected: the password may be what leaked.
- Only one previous email address is kept. If the owner changes their email and an intruder holding the password changes it again within 7 days, lock mail goes to the oldest and the newest address; the one in between gets only the change notice.
- **Decision owed when PR 3 is planned:** if an intruder holding the password changes both the email and the password, the lock mail reaches the owner's previous address for 7 days, but the unlock link then also reaches the intruder and the owner cannot change the email back without the new password. PR 3 must decide whether a link sent to the previous address within 7 days may restore that address, or whether forgot-password within 7 days also mails the previous address. Either choice briefly gives the holder of the old mailbox power over the account.
- The notice to the old address can be abused to mail a third party (set an account's email to their address, then change it away). Bounded by the follow-up rate-limit task (TASK-15641).
- Administrators can still change their own email through the admin endpoints without a password; admin sessions are outside this threat model.
- Org and team API keys owned by a locked user stop working for the whole team until unlock.
- Magic link has no MFA step for ordinary sign-in today (only the unlock path gets one here). Existing gap, separate task.
- The hosted-mode `/api/auth/*` proxy is outside this repo; `is_current` and client IPs there are unverified.

## Delivery

Five PRs against `dev`, each with its own Backlog task, independently reviewed:

0. **Email-change hardening** - password on change, previous address kept, notice to the old address.
1. **Cap** - closes TASK-15628. Backend plus the per-install id in the WebUI and extension. ADR written here.
2. **Alert + Active sessions page** - notification, API fields, interactive-credential rule, WebUI page and banner.
3. **Lock + unlock link** - schema, enforcement function on every path, API-key suspension, magic-link unlock, email template, WebUI flows, admin unlock.
4. **Background pause** - Jobs claim filter, scheduler skips, engine wait, webhook retries.

Each is usable without the next: after PR 1 the table is bounded; after PR 2 users can see and end sessions; PR 3 adds lock; PR 4 extends it to background work. PR 3 must not ship before PR 0.

## Changes from review (2026-11-23)

An independent reviewer checked the first draft against the code. What changed:

- Added PR 0: email could be changed with no password, so the lock email could be redirected.
- Same-browser eviction now uses a client-sent per-install id; the server had nothing to tell browsers apart (and in single-user mode every session looked identical).
- Evicted browsers are told why; the alert is shown as a banner on next load.
- Session management and lock require a browser session, not an API key.
- Lock is set before the email is sent, not after.
- Tokens issued before an unlock are refused (`credentials_valid_after`); recent API keys stay suspended.
- Unlock reuses magic link instead of a new token, endpoint and page; the pending-session column, the locked-by column and Redis key deletion were dropped.
- Single-user `/login` no longer inserts a session row.
- Second review round (2026-11-24): eviction is by least recent activity, not creation time; sign-ins without an install id form their own group; the eviction reason survives the row purge.
- Background pause designed (PR 4), covering administrator deactivation as well.

## Left to the builder

- Exact SQL, index choice for the live-session predicate, and migration numbering.
- Wording of UI copy and the emails, within the facts stated above.
- The user-agent label rules beyond "browser family and OS family".
- Test layout, following the existing AuthNZ unit / SQLite / PostgreSQL split; every enforcement point in sections 3 and 4 gets a test that a locked account is refused or skipped there.
