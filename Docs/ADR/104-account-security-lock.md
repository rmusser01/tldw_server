# ADR-104: Account security lock

**Status:** Accepted
**Date:** 2026-11-26
**Backfilled from:** not backfilled
**Decision owner:** repository owner (design approval 2026-11-23; open questions decided 2026-11-26)
**Related task:** TASK-15671, TASK-15672
**Related spec/plan:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md` (section 3), `Docs/superpowers/plans/2026-11-26-account-lock-backend.md`

## Decision

A signed-in browser session can lock its own account. While the account is locked, every session and token of that account is refused with 423 `account_locked`, API keys stop working, and recently created keys stay suspended after the unlock. Only the link mailed to the account's address (through the existing `POST /api/v1/auth/magic-link/verify`) or an administrator unlocks it. For 7 days after an email change, the link sent to the previous address takes the account back: it restores that address, voids the password, ends every credential and wins over a link sent to the newer address.

This record covers the server half (PR 3a). The WebUI and admin UI half is PR 3b.

## Context

The design (section 3) asks for a lock the owner can set when they suspect an intruder, with the mail as the only way back. The plan found that nothing in the code could enforce "locked" in one place: credentials reach the server as JWTs, sessions, API keys, virtual keys, MCP tokens and chat integration actors. The design also left one decision open: an intruder who holds the password can change both the email and the password, so the unlock link reaches the intruder too. The owner decided that on 2026-11-26.

## The state

Three nullable columns, added by SQLite migration 109 and by the PostgreSQL startup ensure step:

- `users.security_locked_at`: set while the account is locked.
- `users.credentials_valid_after`: a token with `iat` before this instant is refused (401).
- `api_keys.suspended_at`: a suspended key is refused until its owner restores it.

The lock columns are in the users write guard's raw-safe set. The email restore of a take-back goes through the versioned user writer, because `email` is profile-visible.

## The rule and where it is called

One function, `assert_account_not_locked` in `core/AuthNZ/account_lock.py`, is the rule. It is called from the JWT request path (`verify_jwt_and_fetch_user`), sign-in, refresh, MFA sign-in, magic-link verify, password reset, the capped session insert, API-key authentication (SQLite and PostgreSQL lookups carry the owner's state), MCP HTTP, admin impersonation, ACP trigger webhooks, Telegram approvals, and the Slack and Discord job-owner check (`locked_actor_error`). The exceptions and constants live in a leaf module, `core/AuthNZ/account_lock_errors.py`, so the session store and session manager can raise them without a cycle. The interactive-credential rule of PR 2 applies: lock, lock-availability and unsuspend need a browser session, not an API key.

## 423 against 401

A locked account answers 423 with `detail={"code": "account_locked", "message": ...}`. A token issued before `credentials_valid_after` answers the ordinary 401: it is a dead token, not a locked account. Sessions the lock revokes carry `revoke_reason="account_locked"`, and the 423 is given for such a token only while its user is locked at that moment; after the unlock the same token gets 401. Sessions revoked by an unlock or a take-back carry `account_restored`, for the record. `AccountSecurityLockedError` subclasses `HTTPException`, so the sign-in paths that re-raise `HTTPException` pass it through; refresh gets its own clause. The failed-login lockout (`AccountLockedException`) is a different thing and is untouched.

## Credentials that outlive a lock

Virtual-key and impersonation JWTs have no session row, and reset tokens live an hour, so revoking sessions is not enough. Unlock sets `credentials_valid_after` to a whole second strictly after the lock second and waits until that second has begun, so every token minted up to the lock is older and the redeemer's new tokens are not. The rule covers access, refresh and password-reset tokens and MFA sign-in challenges (their payload gains `issued_at`). API keys have no issue time, so keys created in the week before the lock (and, on a take-back, before the email change) are suspended at unlock, in two sweeps with the session revoke between them. A suspended key is restored one at a time through `POST /api/v1/users/api-keys/{key_id}/unsuspend`; the restore and its mandatory audit event commit together. Rotating a suspended key is refused, because rotation would otherwise turn it into a working one.

## Unlock

An unlock link is a magic-link token with `purpose="account_unlock"`. On a locked account `magic-link/verify` accepts only such tokens; an ordinary magic link minted before the lock is refused with 423. Every magic link is also refused (400) when its `email` claim is neither the user's current address nor, for a take-back link, the previous address within 7 days. The check runs before the link is spent and again after the session is created; a mismatch on the second check revokes the new session. `clear_lock` unlocks in a fixed order: suspend recent keys, revoke sessions, take the address back if asked, suspend keys again, clear the lock by compare-and-set with the new `credentials_valid_after`, then wait for that second. Only then is the session created.

## The take-back (owner decision 1A)

A link sent to the previous address, within 7 days of an email change, takes the account back even when the newer address clicked first. It restores the previous address, replaces the password with one nobody knows, ends every session and token, suspends keys, removes SSO links added since the email change, and signs the redeemer in. A link to the newer address is dead afterwards. A take-back writes a fresh lock first, so nothing can be minted while it runs; if its compare-and-set loses it locks again and repeats, at most 3 attempts, then leaves the account locked. It is refused, with the account locked, when another account holds the previous address. To keep that from being engineered, a self-service email change to an address another account released in the last 7 days is refused with `email_recently_released`.

## Mail and availability

Locking mails one link per recipient: the current address and, for 7 days after an email change, the previous address. Locking is offered only when the link can be opened: multi-user mode, `EMAIL_PROVIDER=smtp`, `PUBLIC_WEB_BASE_URL` set, and a verified email. `GET /api/v1/auth/account/lock-availability` returns the first missing condition as `reason` (`single_user_mode`, `mail_not_configured`, `public_url_not_configured`, `email_not_verified`) and the masked addresses; `POST /api/v1/auth/account/lock` answers 409 `lock_unavailable` with the same reason. The mail and the availability answer read one value. The lock is set first and the mail second, so a failed mail never undoes the lock. SMTP gets a fixed 15-second connection timeout; there are no new settings. Anonymous routes (`magic-link/request`, `forgot-password`) keep their generic answers for a locked account and send the "account locked" mail instead.

## No MFA step on unlock (owner decision, Q6)

The design asked for an MFA step on unlock. It is not built. Enrolling MFA today needs no password and accepts any credential, even an API key, so an intruder could enrol their own authenticator before the lock. Every unlock link would then ask for the intruder's code and the owner could not undo the lock, which is the first anti-goal. The step would also protect little, because ordinary magic-link sign-in has no MFA step either. The follow-up, TASK-15672, makes MFA enrolment require the password and adds an MFA step to magic-link sign-in and unlock.

## Other behaviour

- API keys: refused when the owner is locked, inactive or the key is suspended; usage is not recorded for a refused key. An inactive owner's key now answers 401 "Invalid API key" instead of 400 "Inactive user", which closes the gap where inactive users' keys worked on MCP. MCP HTTP answers a refused key with 403 `permission_denied`, the same body any invalid key gets.
- Admin: `POST /api/v1/admin/users/{user_id}/security-unlock` needs the step-up check (`verify_privileged_action`) and writes the audit events. `security_locked_at` is added to the admin listings. `is_locked` is removed from `UserUpdateRequest`: it wrote a column that does not exist and returned 500. Impersonating a locked user is refused. An administrator changing their own email through `PUT /admin/users/{id}` needs the step-up check. `reset_admin_password` (the console) also clears a lock.
- Password reset refuses a locked account (423), a token issued before `credentials_valid_after` (400) and a token whose email is no longer the account's (400). It now revokes the user's sessions after its transaction commits; before, on SQLite, the revoke waited on the request's own write lock and the reset either failed with 500 or left every session valid.
- Audit events (lock, unlock, lock mail, API-key restore) are written from core through the existing lazy audit helper, so no new import pair enters the layering baseline. Audit failures never fail a lock, an unlock or a sign-in, except the API-key restore, whose audit is mandatory. Session-cap evictions are now audited too.
- Lock-time work is not touched here: queued and scheduled work that runs for a locked account is PR 4.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Take-back, first click wins (A2) | An intruder watching their inbox beats the owner. |
| Previous-address link restores the address but keeps the password (B) | The intruder signs straight back in with the password. |
| Previous-address link only signs in (C) | The owner cannot change the email back without the intruder's new password. |
| Forgot-password also mails the previous address for 7 days (D) | The old mailbox could reset the password at any time, with no lock involved. |
| MFA step on unlock | See above: an intruder-enrolled authenticator would make the lock irreversible. Deferred to TASK-15672. |
| Offer lock without a usable link | A lock the owner cannot undo defeats its purpose. |
| Require the password to lock | The password may be what leaked. |
| Restore suspended keys on a new API-keys page | No self-service API-keys page exists; the restore route and the `suspended_at` list field are provided and PR 3b uses the Active sessions page. |

## Consequences

- Locking costs the owner one email round trip; the lock itself takes effect before any mail is sent.
- For 7 days after an email change, whoever holds the previous mailbox can take the account whenever it is locked. An owner who moved away from a compromised mailbox and then locks could lose the account to that mailbox. Locking needs a signed-in session, so the holder of the old mailbox cannot start this alone.
- The take-back also removes SSO links the owner added in that window, and an administrator re-links them where the provider has no email linking.
- Until PR 3b, the only way to redeem a link in the WebUI is the token paste field in Settings; the mail template keeps its manual token box for that reason.
- The two Backlog follow-ups are TASK-15672 (MFA) and a task for the password-reset table defect below.

## Known limits

- Streams and WebSockets already open when the lock is set keep running until they next authenticate. Jobs already running finish; queued and scheduled work is PR 4.
- Public share links, chat share tokens and media tickets created before the lock keep serving the content they point at.
- Clock skew between workers can refuse fresh tokens for a few seconds after an unlock, or let a token minted on a fast-clock worker just before the lock survive it by the amount of the skew.
- Any signed-in browser session can lock, including an intruder's. The cost to the owner is one email round trip.
- If the intruder unlocks first, the owner has only their link's lifetime (15 minutes by default) to take the account back. An administrator-route email change records no previous address, so it opens no take-back window and gives no such protection.
- A take-back is refused, and an administrator must act, when another account already holds the previous address. This includes an account that is already unlocked under the newer address.
- An intruder can enrol MFA before the lock. The owner signs in by magic link (no MFA step there), sets a new password through forgot-password, then disables MFA with it (`/mfa/disable` needs the password). Both are the follow-up TASK-15672.
- After a take-back the owner is meant to set a new password through forgot-password. On a fresh install that cannot work yet: the packaged SQLite schema creates `password_reset_tokens` without the `token_hash` column the code uses, and no PostgreSQL DDL creates the table. This is an older defect with its own follow-up task. Until it is fixed the owner signs in by emailed link and an administrator sets the password.
- An API key or SSO link whose lock check passed while the intruder's newer address had unlocked, and that commits after the winning take-back's last sweep, survives it (keys carry no issue time to compare).
- An impersonation token minted by an administrator who is then locked works until it expires (15 minutes); its subject is the impersonated user.
- If session creation fails right after an unlock, the link is spent and the account is unlocked with every old credential dead. The owner asks for another link or signs in with the password.
- Service tokens are not tied to a user and are not checked. MCP demo tokens (`MCP_ENABLE_DEMO_AUTH`) name a user id without reading the user row and are not checked.
- Slack and Discord have no account link: the check applies only when the resolved owner id is all digits and names a locked user, so an actor id made only of digits is read as a tldw user id and other actor ids are not checked.
- Password reset: the session revoke runs a moment after the 200 is sent.
- Lock mail is sent while the request waits; with a hanging mail server the lock request can take up to the SMTP timeout per step.
- Lock is unavailable without `PUBLIC_WEB_BASE_URL`, and nothing in the WebUI redeems the link by a click until PR 3b.
