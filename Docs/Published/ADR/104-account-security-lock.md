# ADR-104: Account security lock

**Status:** Accepted
**Date:** 2026-11-26
**Backfilled from:** not backfilled
**Decision owner:** repository owner (design approval 2026-11-23; open questions decided 2026-11-26)
**Related task:** TASK-15679, TASK-15680
**Related spec/plan:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md` (section 3), `Docs/superpowers/plans/2026-11-26-account-lock-backend.md`

## Decision

A signed-in browser session can lock its own account. While the account is locked, every session and token of that account is refused with 423 `account_locked`, API keys stop working, and recently created keys stay suspended after the unlock. Only the link mailed to the account's address (through the existing `POST /api/v1/auth/magic-link/verify`) or an administrator unlocks it. For 7 days after an email change, the link sent to the previous address takes the account back: it restores that address, voids the password, ends every session and every token minted before it, suspends the API keys and removes the SSO links created since a week before the email change or the lock (whichever is earlier), and wins over a link sent to the newer address. It does not remove Telegram links, agent triggers or scheduled workflows the intruder created (see Known limits).

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

The rule is `assert_account_not_locked` in `core/AuthNZ/account_lock.py`: a locked account raises `AccountSecurityLockedError` (423) and a token issued before `credentials_valid_after` raises `StaleCredentialError`. It is enforced by mechanism, not by one call everywhere:

- **On the user row**, `assert_account_not_locked`: the JWT request path (`verify_jwt_and_fetch_user`, which the MCP WebSocket also uses), the SSO callback, password sign-in, refresh, MFA sign-in and admin impersonation.
- **The identity-write guard**, `assert_identity_write_allowed`: reset-password (which maps a stale token to 400), change-password and the self-service email change (`authorize_own_email_change`, which `PUT /users/me` and `PATCH /users/me/profile` v1 and v2 run). These requests check a credential, spend 100 to 300 ms hashing, then write; a take-back that ran in between could be undone by the write. The guard reads the lock columns on the request's own transaction before the credential that allows the write is read, and on PostgreSQL holds the user's row (`FOR NO KEY UPDATE`) until the request commits; on SQLite the request transaction already holds the write lock. A lock or take-back that starts meanwhile waits for the request. A take-back then overrides what the request wrote; a plain lock does not undo it, it only stops the account. One that committed first is seen, as the lock (423) or as the take-back's new password hash and `credentials_valid_after`, which the password or token check then refuses. Not `FOR UPDATE`: a password reset on PostgreSQL revokes sessions on another connection while it holds the row, and the blacklist rows' foreign key takes `KEY SHARE`, which `FOR UPDATE` would block.
- **API keys**: the check inside `validate_api_key` (its SQLite and PostgreSQL lookups carry the owner's lock and active state); MCP HTTP and the WebSockets get its refusal.
- **The capped session insert** re-reads `security_locked_at` in its own transaction (`FOR SHARE` on PostgreSQL), so a sign-in racing the lock leaves no session.
- **A fresh read of the row**, `is_user_security_locked`: the integrations (Telegram asks and approvals, the Slack and Discord job-owner check `locked_actor_error`, agent trigger webhooks), and the session manager, to tell a token the lock revoked (423) from a dead one (401).
- **Magic links**: `magic-link/verify` uses `is_security_locked` plus the purpose check below; `magic-link/request` and `forgot-password` use it to mail the unlock link instead. `forgot-password` finds the account only by its current address (`fetch_user_by_email_for_password_reset`); only `magic-link/request` also finds a locked account by its previous address within the 7 days (`find_locked_user_by_previous_address`). `forgot-password` answers the same status and body for a locked account, an unlocked one and an unknown address, also when the reset token cannot be stored (the database error is logged with the user id). Its response time can still differ, because a locked account's mail is sent while the request waits. `magic-link/request` has one exception, when the mail service raises (see Known limits).

The exceptions and constants live in a leaf module, `core/AuthNZ/account_lock_errors.py`, so the session store and session manager can raise them without a cycle. The interactive-credential rule of PR 2 applies: lock, lock-availability and unsuspend need a browser session, not an API key.

## 423 against 401

A locked account answers 423 with `detail={"code": "account_locked", "message": ...}`. A token issued before `credentials_valid_after` answers the ordinary 401: it is a dead token, not a locked account. Sessions the lock revokes carry `revoke_reason="account_locked"`, and the 423 is given for such a token only while its user is locked at that moment; after the unlock the same token gets 401. Sessions revoked by an unlock or a take-back carry `account_restored`, for the record. `AccountSecurityLockedError` subclasses `HTTPException`, so the sign-in paths that re-raise `HTTPException` pass it through; refresh gets its own clause. The failed-login lockout (`AccountLockedException`) is a different thing and is untouched.

## Credentials that outlive a lock

Virtual-key and impersonation JWTs have no session row, and reset tokens live an hour, so revoking sessions is not enough. Unlock sets `credentials_valid_after` to a whole second strictly after the lock second and waits until that second has begun, so every token minted up to the lock is older and the redeemer's new tokens are not. The rule covers access, refresh and password-reset tokens and MFA sign-in challenges (their payload gains `issued_at`). API keys have no issue time, so keys created in the week before the lock (and, on a take-back, before the email change) are suspended at unlock, in two sweeps with the session revoke between them. A suspended key is restored one at a time through `POST /api/v1/users/api-keys/{key_id}/unsuspend`; the restore and its mandatory audit event commit together. Rotating a suspended key is refused, because rotation would otherwise turn it into a working one.

## Unlock

An unlock link is a magic-link token with `purpose="account_unlock"`. On a locked account `magic-link/verify` accepts only such tokens; an ordinary magic link minted before the lock is refused with 423. Every magic link is also refused (400) when its `email` claim is neither the user's current address nor, for a take-back link, the previous address within 7 days. The check runs before the link is spent and again after the session is created; a mismatch on the second check revokes the new session. `clear_lock` unlocks in a fixed order: suspend recent keys, revoke sessions, take the address back if asked, suspend keys again, clear the lock by compare-and-set with the new `credentials_valid_after`, then wait for that second. Only then is the session created.

## The take-back (owner decision 1A)

A link sent to the previous address, within 7 days of an email change, takes the account back even when the newer address clicked first. It restores the previous address, replaces the password with one nobody knows, ends every session and every token minted before it, suspends the API keys and removes the SSO links created since a week before the email change or the lock (whichever is earlier), and signs the redeemer in. It does not remove Telegram links, agent triggers or scheduled workflows the intruder created (see Known limits). A link to the newer address is dead afterwards. The link takes the account back whenever its address is the stored `previous_email` and the email change is inside the 7 days: also when the newer address unlocked first and the intruder changed the email back to that address (the self-service change keeps `previous_email` for 7 days, so it is then the email too), and whether or not the account is locked at that moment. A take-back writes a fresh lock first, so nothing can be minted while it runs; if its compare-and-set loses it locks again and repeats, at most 3 attempts, then leaves the account locked. It removes the SSO links before it restores the address, and an error removing them fails the take-back with the account still locked (nothing else removes an intruder's SSO link). The address is not restored yet, so `previous_email` stays recorded and a new take-back link runs the whole take-back again (`magic-link/request` with the previous address mails one while it is not the email). The link that failed is spent and its verify answers with a server error. A take-back that fails after it wrote its fresh lock, or gives up, leaves the account locked even when another link cleared that lock meanwhile. One that is refused before it writes anything (another account already holds the address when the link is checked) changes nothing, so an account the newer address had unlocked stays unlocked. On PostgreSQL, where the SSO tables are created on first use, a missing link table means there are no links to remove. A second link to the previous address, redeemed after a completed take-back (which clears `previous_email`) once the account is locked again, finds its address is the account's email and unlocks it plainly. A take-back is refused, with the account locked, when another account holds the previous address. To keep that from being engineered, a self-service email change to an address another account released in the last 7 days is refused with `email_recently_released`; open registration, or an administrator creating an account, can still put another account on the old address, and the take-back is then refused.

After a take-back the password is void. Until the password-reset defect in Known limits is fixed, forgot-password cannot store a reset token, change-password needs the voided password and `/mfa/disable` needs a password, so only an administrator can give a taken-back account a password. The owner signs in by emailed link meanwhile.

## Mail and availability

Locking mails one link per recipient: the current address and, for 7 days after an email change, the previous address. Locking is offered only when the link can be opened: multi-user mode, `EMAIL_PROVIDER=smtp`, `PUBLIC_WEB_BASE_URL` set, and a verified email. `GET /api/v1/auth/account/lock-availability` returns the first missing condition as `reason` (`single_user_mode`, `mail_not_configured`, `public_url_not_configured`, `email_not_verified`) and the masked addresses; `POST /api/v1/auth/account/lock` answers 409 `lock_unavailable` with the same reason. The mail and the availability answer read one value. The lock is set first and the mail second, so a failed mail never undoes the lock. SMTP gets a fixed 15-second connection timeout; there are no new settings. Anonymous routes (`magic-link/request`, `forgot-password`) keep their generic answers for a locked account and send the "account locked" mail instead.

## No MFA step on unlock (owner decision, Q6)

The design asked for an MFA step on unlock. It is not built. Enrolling MFA today needs no password and accepts any credential, even an API key, so an intruder could enrol their own authenticator before the lock. Every unlock link would then ask for the intruder's code and the owner could not undo the lock, which is the first anti-goal. The step would also protect little, because ordinary magic-link sign-in has no MFA step either. The follow-up, TASK-15680, makes MFA enrolment require the password and adds an MFA step to magic-link sign-in and unlock.

## Other behaviour

- API keys: refused when the owner is locked, inactive or the key is suspended; usage is not recorded for a refused key. An inactive owner's key now answers 401 "Invalid API key" instead of 400 "Inactive user", which closes the gap where inactive users' keys worked on MCP. MCP HTTP answers a refused key with 403 `permission_denied`, the same body any invalid key gets.
- Admin: `POST /api/v1/admin/users/{user_id}/security-unlock` needs the step-up check (`verify_privileged_action`) and writes the audit events. `security_locked_at` is added to the admin listings. `is_locked` is removed from `UserUpdateRequest`: it wrote a column that does not exist and returned 500. Impersonating a locked user is refused. An administrator changing their own email through `PUT /admin/users/{id}` needs the step-up check. `reset_admin_password` (the console) also clears a lock, writes the unlock audit event (`via: "cli"`) and says so; if clearing the lock fails after the password changed, it says that and exits non-zero.
- Password reset refuses a locked account (423), a token issued before `credentials_valid_after` (400) and a token whose email is no longer the account's (400), through the identity-write guard. On SQLite it now revokes the user's sessions after its transaction commits; before, the revoke waited on the request's own write lock and the reset either failed with 500 or left every session valid. PostgreSQL revokes inside the request, as before.
- Audit events (lock, unlock, lock mail, API-key restore) are written from core through the existing lazy audit helper, so no new import pair enters the layering baseline. Audit failures never fail a lock, an unlock or a sign-in, except the API-key restore, whose audit is mandatory. Session-cap evictions are now audited too.
- Lock-time work is not touched here: queued and scheduled work that runs for a locked account is PR 4.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Take-back, first click wins (A2) | An intruder watching their inbox beats the owner. |
| Previous-address link restores the address but keeps the password (B) | The intruder signs straight back in with the password. |
| Previous-address link only signs in (C) | The owner cannot change the email back without the intruder's new password. |
| Forgot-password also mails the previous address for 7 days (D) | The old mailbox could reset the password at any time, with no lock involved. |
| MFA step on unlock | See above: an intruder-enrolled authenticator would make the lock irreversible. Deferred to TASK-15680. |
| Offer lock without a usable link | A lock the owner cannot undo defeats its purpose. |
| Require the password to lock | The password may be what leaked. |
| Restore suspended keys on a new API-keys page | No self-service API-keys page exists; the restore route and the `suspended_at` list field are provided and PR 3b uses the Active sessions page. |

## Consequences

- Locking costs the owner one email round trip; the lock itself takes effect before any mail is sent.
- For 7 days after an email change, whoever holds the previous mailbox can take the account whenever it is locked. An owner who moved away from a compromised mailbox and then locks could lose the account to that mailbox. Locking needs a signed-in session, so the holder of the old mailbox cannot start this alone.
- The take-back also removes SSO links the owner added in that window, and an administrator re-links them where the provider has no email linking.
- Until PR 3b, the only way to redeem a link in the WebUI is the token paste field in Settings; the mail template keeps its manual token box for that reason.
- The Backlog follow-ups are TASK-15680 (MFA), TASK-15678 (the password-reset table defect below), TASK-15681 (the take-back races and the admin bulk route below), TASK-15682 (SQLite routes that write through a second connection inside a request transaction) and TASK-15683 (sweeping what an intruder set up to act later).

## Known limits

- Streams and WebSockets already open when the lock is set keep running until they next authenticate. Jobs already running finish; queued and scheduled work is PR 4.
- Public share links, chat share tokens and media tickets created before the lock keep serving the content they point at.
- Clock skew between workers can refuse fresh tokens for a few seconds after an unlock, or let a token minted on a fast-clock worker just before the lock survive it by the amount of the skew.
- Any signed-in browser session can lock, including an intruder's. The cost to the owner is one email round trip.
- If the intruder unlocks first, the owner has only their link's lifetime (15 minutes by default) to take the account back. An administrator-route email change records no previous address, so it opens no take-back window and gives no such protection.
- Once the intruder has changed the email back to the previous address, the take-back survives only in the link already mailed. A lock set afterwards mails one plain link (the previous address is the email, so it gets no take-back link), and so does `magic-link/request` after a take-back of that state failed. A plain unlock keeps the password and the SSO links; the owner recovers the password through forgot-password (but see the reset-table defect below).
- A take-back ends every token minted before it only to the second: a token without a session row (a virtual-key, impersonation or password-reset token) minted in the same second as the take-back's last step, by a request that passed its lock check before the take-back's fresh lock, has `iat` equal to `credentials_valid_after` and survives. TASK-15681.
- The administrator routes (`services/admin_users_service.py`, and the bulk profile route `POST /admin/users/profile/bulk` in `services/admin_profiles_service.py`) change an email or a password without the identity-write guard, so such a write already in flight when a take-back starts can land after it.
- The bulk profile route also lets an administrator change their own email without the password or the step-up check that `PUT /admin/users/{id}` asks for. The gap is older than the lock (the previous address is still recorded, so the take-back window opens) and is TASK-15681.
- Password sign-in checks the lock after it verified the password, then rewrites an out-of-date password hash with an unconditional update. If a take-back completes inside that one request, the rewrite puts the old password back. It needs a hash that is due for rehashing, so it can happen once per account. TASK-15681.
- A password sign-in that verified just before a take-back can create its session just after the take-back's last sweep, on the account the take-back left unlocked. That session lives until the owner locks again. TASK-15681.
- A take-back, like an unlock, leaves what the intruder set up to act later: Telegram actor links, agent (ACP) triggers and scheduled workflows keep acting for the account. Sweeping them belongs with the background pause (PR 4, TASK-15683).
- A take-back is refused, and an administrator must act, when another account already holds the previous address. A locked account stays locked; an account the newer address already unlocked stays unlocked under the newer address. If the other account takes the address while the take-back is already running, the take-back stops with the account locked, and the SSO links created inside its window are already removed.
- An intruder can enrol MFA before the lock. The owner signs in by magic link (no MFA step there), but `/mfa/disable` needs the password, so the owner needs a password first (after a take-back only an administrator can set one, see above). The fixes are the follow-up TASK-15680.
- Password reset on a fresh install: forgot-password cannot store a token, because the packaged SQLite schema creates `password_reset_tokens` without the `token_hash` column the code uses and no PostgreSQL DDL creates the table. This is an older defect, TASK-15678. Until it is fixed only an administrator can give a taken-back account a password; the owner signs in by emailed link meanwhile. It also stops an owner whose password the intruder changed from replacing it after a plain unlock: change-password needs the intruder's password and forgot-password cannot store a token, so again only an administrator can set it.
- An API key or SSO link whose lock check passed while the intruder's newer address had unlocked, and that commits after the winning take-back's last sweep, survives it (keys carry no issue time to compare).
- When the mail service raises (rather than reporting a failed send), `POST /auth/magic-link/request` answers 500 for an unlocked account and 200 for a locked one, which tells them apart in that failure.
- An impersonation token minted by an administrator who is then locked works until it expires (15 minutes); its subject is the impersonated user.
- If session creation fails right after an unlock, the link is spent and the account is unlocked with every old credential dead. The owner asks for another link or signs in with the password.
- Service tokens are not tied to a user and are not checked. MCP demo tokens (`MCP_ENABLE_DEMO_AUTH`) name a user id without reading the user row and are not checked.
- Slack and Discord have no account link: the check applies only when the resolved owner id is all digits, lies between 1 and 2,147,483,647 and names a locked user. A short all-digit actor id is therefore read as a tldw user id; a raw Discord user id (17 to 19 digits) and every other actor id are not checked, so in practice the lock reaches Slack and Discord through the policy's `user_mappings` (or its `service_user_id` when the event names no user).
- Password reset on SQLite: the session revoke runs just after the response is sent, and if it fails the sessions stay valid until they expire. PostgreSQL revokes inside the request, as before.
- Lock mail is sent while the request waits; with a hanging mail server the lock request can take up to the SMTP timeout per step.
- Lock is unavailable without `PUBLIC_WEB_BASE_URL`, and nothing in the WebUI redeems the link by a click until PR 3b.
