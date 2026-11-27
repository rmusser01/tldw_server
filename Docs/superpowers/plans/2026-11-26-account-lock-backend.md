# Account Lock and Email-Link Unlock: Backend (PR 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A signed-in browser session can lock its own account. From that moment no credential of the account works (session tokens, refresh tokens, API keys, virtual-key and impersonation JWTs, password sign-in, SSO). The owner comes back through a link mailed to the account's address. After unlock, every token issued before the unlock stays dead and API keys created in the 7 days before the lock stay suspended until restored one by one.

**This plan is the server half.** The WebUI half (PR 3b) is listed under "PR 3b scope" and gets its own plan after this one merges. Until 3b ships nothing in the UI offers lock, so 3a changes nothing a user can see except the fixes listed under "Behaviour changes".

**Architecture:**
- State: three nullable timestamp columns. `users.security_locked_at`, `users.credentials_valid_after`, `api_keys.suspended_at`. SQLite migration 109; PostgreSQL ensure-on-startup plus the serving catalog.
- One core module, `core/AuthNZ/account_lock.py`, owns the rule: the exception, the check, lock, unlock, lock availability and who gets the mail. Endpoints call it; nothing else decides "locked".
- JWT paths call `assert_account_not_locked(user_row, token_iat)` at the places listed in Task 3. API keys are refused inside `APIKeyManager.validate_api_key`, which learns the owner's state from the key lookup query itself (a sub-select on `users`), so there is no cache and no second round trip.
- Unlock reuses magic link. The lock mail carries a magic-link token with `purpose="account_unlock"`; `POST /auth/magic-link/verify` clears the lock when it redeems one.
- Sign-in racing a lock: the session-cap transaction re-reads the lock on the same connection before it inserts.

**Tech Stack:** FastAPI, SQLite and PostgreSQL through `DatabasePool`, python-jose JWTs, smtplib, loguru, pytest.

**Spec:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`, section "3. Account lock and email-link unlock (PR 3)", with "Decisions made by the owner" 3-5 and 7, "Anti-goals", "Known limits", and the owner decisions of 2026-11-25 (below). Section 4 (background pause) is PR 4 and out of scope.

## Questions for the owner (decided 2026-11-26)

The owner chose 1A, confirmed the split (Q2) and no MFA step on unlock (Q6), and accepted the defaults of Q3-Q5 as written. Nothing here blocks execution.

1. **The decision the design left open: an intruder who has the password changes the email and the password.** You decided the unlock link also goes to the previous address for 7 days. That alone leaves a stalemate: both parties get a link, and the intruder still holds the password and the email. Options:
   - **A (recommended, in the plan): the link sent to the previous address takes the account back, and it wins even if the newer address clicked first.** Redeeming it restores that address as the account's email, replaces the password with an unusable one (the owner sets a new one through forgot-password), ends every session and token, suspends every API key created since a week before the email change, removes SSO links added since the email change, and signs the redeemer in. A link sent to the newer address stops working once this has happened.
     - What it costs: for 7 days after an email change, whoever holds the **old** mailbox can take the account whenever it is locked. An owner who moved away from a compromised mailbox and then locks would lose the account to that mailbox. Locking needs a signed-in session, so the old-mailbox holder cannot start this alone.
     - Its limit: if the intruder clicks first, the owner has only their own link's lifetime (15 minutes by default) to take the account back, because "send me a new link" works only while the account is locked.
     - A second limit: the take-back also removes SSO links the owner added in that window, and it is refused (the account stays locked, an administrator must act) if another account has since been given the old address. The plan stops an account from taking an address that was someone's address in the last 7 days.
   - A2: the same take-back, but **first click wins**: once either link has unlocked the account, the other is dead. Simpler and it removes the compromised-old-mailbox loss, but an intruder watching their inbox beats the owner.
   - B: the link to the previous address restores the address but keeps the password. The intruder signs straight back in with it.
   - C: the link to the previous address only signs in. The owner cannot change the email back without the intruder's new password.
   - D: forgot-password also mails the previous address for 7 days. Worse than A: the old mailbox can reset the password at any time, with no lock involved.
2. **Split PR 3 in two (recommended, assumed here).** The WebUI has no page that consumes an emailed link (`/auth/magic-link` is a placeholder), no MFA code step anywhere, and no self-service API-keys page. Building those with the server work makes one PR too large to review well. 3a = server, 3b = WebUI and admin UI.
3. **Lock is offered only when the mailed link can be opened.** Today the link points at `PUBLIC_WEB_BASE_URL/auth/magic-link`, or at the API server (`BASE_URL/magic-link`, a 404) when that setting is empty. The plan makes lock available only when all hold: multi-user mode, `EMAIL_PROVIDER=smtp`, `PUBLIC_WEB_BASE_URL` set, and the user's email verified. Otherwise the availability answer says which one is missing.
4. **Suspended keys are restored from the Active sessions page (3b), not a new API-keys page.** No self-service API-keys page exists. 3a provides the list field and the restore endpoint either way.
5. **An inactive owner's API key now answers 401 "Invalid API key" instead of 400 "Inactive user".** This is how the MCP gap (inactive users' keys work on MCP) is closed in one place. Scripts that match on the 400 text would notice.
6. **No MFA step on unlock (recommended, in the plan; the design asked for one).** Enrolling MFA today needs no password and accepts any credential, even an API key (`EP:3257-3260`, `:3337-3342`). An intruder could enrol their own authenticator; every unlock link would then ask for the intruder's code and the owner could not undo the lock, which is the first anti-goal. The step would also protect little, because ordinary magic-link sign-in has no MFA step today. The plan files a follow-up task: MFA enrolment must require the password, and magic-link sign-in needs an MFA step; unlock gets its MFA step with that work.

## Decisions (settled, not open)

1. Owner 2026-11-25: an admin changing **their own** email through `PUT /admin/users/{id}` needs the step-up check (`verify_privileged_action`). Task 7.
2. Owner 2026-11-25: within 7 days of an email change the lock mail also goes to `users.previous_email`.
3. HTTP answer for a locked account: **423** with `detail={"code": "account_locked", "message": ...}`. A token issued before `credentials_valid_after` gets the ordinary 401 (it is a dead token, not a locked account).
4. `AccountSecurityLockedError` subclasses `HTTPException`. `login`, `verify_magic_link`, `mfa_login` and `reset_password` already re-raise `HTTPException` first. `refresh_token` and `forgot_password` do **not** (`HTTPException` is inside `_AUTH_NONCRITICAL_EXCEPTIONS`, `auth.py:185-201`): refresh gets an explicit `except` clause, and forgot-password never raises (it mails instead).
5. The existing `AccountLockedException` (`core/AuthNZ/exceptions.py:32`, failed-login lockout) is a different thing and is not reused.
6. Sessions revoked by a lock carry `revoke_reason="account_locked"`; `SessionManager.revoked_token_exception` maps that reason to the 423, as it maps `session_cap` to `session_replaced`. The 423 is given only while the token's user **is locked at that moment** (`is_user_security_locked`, a lookup that runs only for already-revoked tokens); after an unlock the same token gets the ordinary 401. The blacklist keeps the first reason for a token (`ON CONFLICT DO NOTHING`, `token_blacklist_repo.py:132-151`), so the stored reason alone cannot tell the two apart. Sessions revoked by an unlock or a take-back carry `"account_restored"`, for the record only.
7. On a locked account `magic-link/verify` accepts only tokens with `purpose="account_unlock"`. An ordinary magic link minted before the lock is refused with 423.
8. A magic-link token is refused (400) when its `email` claim is neither the user's current email nor, for a take-back token, the user's `previous_email` inside the 7-day window. This is what kills the intruder's link after a take-back. The rule is checked before the link is consumed and **again after the session is created** (a take-back can commit in between); a mismatch on the second check revokes the new session.
9. Unlock order is the numbered list in Task 2 (`clear_lock`), then the session is created. A take-back of an account that someone already unlocked **locks it again first**, so nothing can be minted while it runs. If session creation fails the account is unlocked with every old credential dead and the link spent; the owner asks for another link or signs in with the password.
10. A token is refused when `iat < credentials_valid_after`. JWT `iat` is whole seconds, so `credentials_valid_after` is set to a whole second strictly after the lock second (`max(floor(now), floor(security_locked_at) + 1)`), and `clear_lock` waits until that second has begun (at most one second) before it returns. Every token minted up to the lock is therefore older, and the redeemer's new tokens are not. The rule applies to access, refresh and password-reset tokens and to MFA sign-in challenges (their payload gains `issued_at`). Known limit: clock skew between workers.
11. The lock check is not added to single-user paths or to service tokens (no `users` row). MCP's demo JWT family (`MCP_unified/auth/jwt_manager.py`, minted only with `MCP_ENABLE_DEMO_AUTH`) uses its `sub` as a user id without reading the row; it is left alone and listed as a limit.
12. Slack and Discord have no account link: the platform id, or a free-text mapping, becomes the job owner. The check there applies only when the resolved owner id is all digits and names a locked user.
13. SMTP gets a fixed 15-second connection timeout, as a module constant. No new setting.
14. No new settings or environment variables.
15. Audit events are written from core through the existing lazy helper `core/AuthNZ/api_key_audit._get_or_create_audit_service(user_id)`, which adds no new import pair to `layering_baseline.txt`. Audit failures are logged and never fail a lock, an unlock or a sign-in. The admin unlock also writes the admin audit event from `services/`.
16. Cap evictions are written to the audit log here (the design's Audit bullet lists them and nothing writes them today).
17. Rotating a suspended API key is refused; otherwise rotation, which any credential may call, would turn a suspended key into a working one.
18. A self-service email change to an address that is another account's `previous_email` inside the 7-day window is refused (code `email_recently_released`). Without this, an intruder with a second account could park it on the owner's old address and make the take-back impossible. For the cases this cannot cover (registration, admin-created accounts), a take-back to an address another account holds is refused before the link is consumed and the account stays locked.
19. The exception classes and constants live in a leaf module, `core/AuthNZ/account_lock_errors.py` (imports only fastapi and `core/AuthNZ/exceptions`), because `sessions_repo` and `session_manager` raise them and `account_lock` calls the session manager. `account_lock` re-exports them; `session_manager` imports `write_account_audit` and `is_user_security_locked` inside the functions that use them; `account_lock` imports `get_session_manager` inside its functions.

## Behaviour changes

- New: `POST /api/v1/auth/account/lock`, `GET /api/v1/auth/account/lock-availability`, `POST /api/v1/users/api-keys/{key_id}/unsuspend`, `POST /api/v1/admin/users/{user_id}/security-unlock`.
- `magic-link/verify`: the email-claim check (Decision 8).
- `magic-link/request` and `forgot-password` for a locked account send the "account locked" mail. Both keep their generic answer.
- Self-service email change refuses an address another account released in the last 7 days (`email_recently_released`).
- `reset-password` refuses a locked account (423), a token issued before `credentials_valid_after` (400) and a token whose email is no longer the account's (400).
- API keys: refused when the owner is locked, the owner is inactive (Q5), or the key is suspended. Usage is not recorded for a refused key. A suspended key cannot be rotated.
- Admin: `is_locked` is removed from `UserUpdateRequest` (it wrote a column that does not exist and returned 500). Listings gain `security_locked_at`. Impersonating a locked user is refused. Own-email change needs step-up.
- `reset_admin_password` also clears a security lock.
- ACP trigger webhooks and Telegram approvals of a locked owner are refused.

## PR 3b scope (separate plan, not built here)

Lock button, confirmation and "why not available" line on `/settings/sessions`; treating 423 as signed-out in `TldwAuth.ts`, `background-proxy.ts` and `lib/api.ts`; "Account locked" notice and "Send me a new link" on the sign-in page; a real `/auth/magic-link` page that redeems `?token=`; landing on the sessions page with "change your password" and the suspended keys with Restore; admin UI: lock shown, Unlock with the step-up dialog, own-email step-up; E2E `hosted-placeholder-routes.spec.ts` update.

Until 3b, the only way to redeem a link in the WebUI is the token paste field in Settings (`apps/packages/ui/src/components/Option/Settings/tldw.tsx`, `TldwAuth.verifyMagicLink`). The mail template keeps the manual token box for that reason; the PR body says so.

## Global Constraints

- **Worktree:** `git -C /Users/macbook-dev/Documents/GitHub/tldw_server fetch origin dev && git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add .claude/worktrees/account-lock -b feat/account-lock-backend origin/dev`. Work only there. One PR, base `dev`. Never `git stash`, never `--no-verify`, never arm auto-merge.
- **Re-verify every cited line at branch time.** Anchors were read at `0d4cac9bc8` (2026-11-26). Cite by name when a line no longer matches. `EP` means `tldw_Server_API/app/api/v1/endpoints/auth.py`; other paths are under `tldw_Server_API/app/` unless they start with `tldw_Server_API/tests`, `Docs/` or `apps/`.
- **Backlog:** file the task with `tools/backlog-py` and an explicit id (Task 1). Highest id seen on 2026-11-26 was 15670 in open PRs; take the next free ones after `git ls-tree --name-only origin/dev backlog/tasks/` and `gh pr list --state open --json number,files`. Written `TASK-<ID>` below. Edit only with `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task edit TASK-<ID> ...`; stage with `git add -A backlog/tasks`.
- **ADR:** required (the design assigns the lock ADR to PR 3). Next free number was 102 on 2026-11-26; re-check `git ls-tree --name-only origin/dev Docs/ADR/` and open PRs. Written `ADR-<N>` below.
- `run_py <id> <files...>` means:
  `zsh ~/.cache/claude-tools/disk_guard.zsh "$TMPDIR/al-<id>" -- /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider --basetemp "$TMPDIR/al-<id>" -q <files...>`. The guard refuses below 100 GiB free; do not override it.
- PostgreSQL tests use the `isolated_test_environment` fixture of `tldw_Server_API/tests/AuthNZ/conftest.py`; never build a database by hand. A PostgreSQL step passes only if the output shows the tests **passed**; if skipped, say so in the PR.
- Test emails use `@example.com`.
- **Ratchets that will bite** (under `tldw_Server_API/tests/lint` unless noted):
  - `function_size_baseline.json`: `User_DB_Handling:verify_jwt_and_fetch_user` is 436 of 436 and must not grow. `auth:login` is 435 of 454. `migrations:get_authnz_migrations` is 441 of 443, so the one registration line fits with no baseline change. New functions stay under 300 lines; `refresh_token` is 267, so its additions stay under 33 lines.
  - `test_no_raw_sql_in_endpoints.py`: no SQL in `api/v1/endpoints`. SQL goes in `core/AuthNZ/repos` or `core/DB_Management`.
  - `test_scope_predicate_ratchet.py`: a repo method that filters by `user_id` applies it unconditionally.
  - `test_service_singleton_ratchet.py`: it counts zero-argument `get_\w+()` calls in endpoint bodies, and `users.py` is at its ceiling of 10, `auth.py` at 34. No new route body may call `get_api_key_manager()`, `get_session_manager()` or similar: routes call `account_lock.<function>(...)`, which fetches its own collaborators.
  - `test_noncritical_exception_tuples.py`: tuples are named `*_NONCRITICAL_EXCEPTIONS`, never contain `BaseException` or `CancelledError`; no bare `except Exception`.
  - `test_env_read_ratchet.py`: `core/AuthNZ/email_service.py` is at 18 reads. No new `os.getenv`; the public URL is read through the existing `_read_setting_text` (Task 5).
  - `test_route_auth_ratchet.py`: every new route declares an auth dependency.
  - `layering_baseline.txt` and `test_core_to_api_import_boundary.py`: `core.AuthNZ -> api.v1.API_Deps` is frozen at 3 and `core.AuthNZ -> services` at 1. `account_lock.py` imports neither (Decision 15).
  - Test-quality ratchet (`Helper_Scripts/ci/test_quality_triage.py --enforce`): every HTTP test asserts a body or a database effect, never only `.status_code`; no test-defined stub classes injected into the object under test; patch `core` module attributes with `monkeypatch.setattr` and plain functions.
  - **Users write guard, both backends** (`core/AuthNZ/database.py:509-515`, `:757-765`; `profile_user_write_guard.py:762-772`): a raw `UPDATE users` is refused unless every column is in `RAW_SAFE_USER_UPDATE_FIELDS` and none is profile-visible. The two lock columns are added to that set (Task 1). `email` is profile-visible, so the take-back write goes through `VersionedUserWriteGateway` (Task 2), as the email change of commit `91fc351d5f` does.
  - `Docs/` edits are mirrored: run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff in the same commit.
  - OpenAPI: regenerate with `Helper_Scripts/export_openapi_schema.py --fingerprint apps/packages/api-types/openapi.fingerprint.json` and the `paths.d.ts` it checks (Task 9).
- Use loguru. **Never log tokens, email addresses, install ids or full user agents.** Log the user id only.
- Every commit ends with the `Co-Authored-By` trailer from the executing session's attribution reminder, as a second `-m`.
- Change summary: waived by the owner on 2026-11-23 for the five session-feature PRs. If the owner confirms the split (Q2), confirm the waiver covers both halves before merging.
- Before merge: clean-subagent `/superpowers:requesting-code-review` with the outcome posted on the PR; diff the non-required Frontend E2E runs against a recent `dev` run.

## Review Focus

1. **Nothing works while locked.** One parametrized test drives every credential kind against a locked account and requires a refusal for each. Task 3 `test_no_credential_works_while_locked[...]`, Task 4 `test_api_key_of_a_locked_owner_is_invalid_and_usage_is_not_recorded`, `test_mcp_http_refuses_api_key_of_locked_owner`, `test_sqlite_lookup_carries_the_owners_state`.
2. **Unlock does not revive old credentials.** Virtual-key and impersonation JWTs have no session row, and reset tokens live an hour. Task 3 `test_token_minted_before_unlock_stays_dead_after_unlock`, `test_reset_token_minted_before_unlock_is_refused_after_unlock`, Task 6 `test_recent_api_key_stays_suspended_after_unlock`.
3. **A sign-in racing the lock leaves no session.** Task 3 `test_capped_insert_refuses_a_locked_user` (SQLite) and `test_sign_in_waits_for_an_uncommitted_lock_and_is_refused_pg`.
4. **The wrong party cannot use the link or survive a take-back.** Task 6 `test_plain_magic_link_does_not_unlock`, `test_link_to_newer_address_is_dead_after_take_back`, `test_take_back_after_the_newer_address_unlocked_first_leaves_the_intruder_nothing`, `test_take_back_committed_during_the_other_links_verify_revokes_its_session`, Task 5 `test_api_key_cannot_lock`.
5. **A lock that half-fails is still a lock, and no anonymous answer reveals it.** Task 5 `test_lock_holds_when_mail_fails`, `test_lock_holds_when_session_revoke_fails`, Task 6 `test_anonymous_answers_do_not_reveal_the_lock`.

## Files

Create:
- `core/AuthNZ/account_lock.py`, `core/AuthNZ/account_lock_errors.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/_account_lock_fixtures.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_schema_sqlite.py`, `test_account_lock_enforcement_sqlite.py`, `test_account_lock_api_keys_sqlite.py`, `test_account_lock_endpoint_sqlite.py`, `test_account_unlock_sqlite.py`, `test_account_lock_admin_sqlite.py`, `test_account_lock_integrations_sqlite.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py`
- `Docs/ADR/<N>-account-security-lock.md`

Modify: `core/AuthNZ/migrations.py`, `pg_migrations_extra.py`, `core/DB_Management/postgres_auth_catalog.py`, `core/DB_Management/user_profile_writes.py`, `core/AuthNZ/profile_user_fields.py`, `repos/users_repo.py`, `repos/api_keys_repo.py`, `repos/sessions_repo.py`, `repos/federated_identity_repo.py`, `core/AuthNZ/email_change.py`, `core/exceptions.py`, `services/admin_api_keys_service.py`, `session_manager.py`, `User_DB_Handling.py`, `api_key_manager.py`, `email_service.py`, `reset_admin_password.py`, `services/auth_service.py`, `services/admin_users_service.py`, `EP`, `api/v1/endpoints/users.py`, `api/v1/endpoints/admin/admin_user.py`, `admin_impersonation.py`, `api/v1/schemas/auth_schemas.py`, `admin_schemas.py`, `api_key_schemas.py`, `api/v1/utils/chatops/telegram_support.py`, `api/v1/endpoints/slack.py`, `discord.py`, `core/Agent_Client_Protocol/triggers.py`, `core/MCP_unified/adapters/tldw_runtime.py`, `core/Audit/unified_audit_service.py`, `apps/packages/ui/src/services/tldw/TldwApiClient.ts` (type only), the tests named in Task 1, docs in Task 9.

---

### Task 1: Branch, Backlog tasks, and the three columns

**Files:** `core/AuthNZ/migrations.py`, `core/AuthNZ/pg_migrations_extra.py`, `core/DB_Management/postgres_auth_catalog.py`, `core/AuthNZ/profile_user_fields.py`, `tldw_Server_API/tests/AuthNZ/conftest.py`, `tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py`, `test_pg_migrations_authnz_core.py`, `test_postgres_runtime_checks.py`, `tldw_Server_API/tests/AuthNZ/integration/test_postgres_runtime_schema.py`. Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_schema_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py`.

**Interfaces:**
- Produces: columns `users.security_locked_at`, `users.credentials_valid_after`, `api_keys.suspended_at`. SQLite type `TIMESTAMP`, PostgreSQL `TIMESTAMPTZ`, all nullable with no default.

The checklist is commit `464194e7dd` (it added `previous_email`): run `git show --stat 464194e7dd` and touch the same files for the same reasons.

- [ ] **Step 1:** Create the worktree. File two Backlog tasks: this one ("Account lock and email-link unlock: backend", acceptance criteria = the Behaviour changes list) and the follow-up of Q6 ("MFA enrolment needs the password; magic-link sign-in and unlock need an MFA step"). Commit.
- [ ] **Step 2: Write the failing tests.** Build the SQLite database the way `tests/AuthNZ_SQLite/test_email_change_sqlite.py:30-47` does (a raw `sqlite3` file run through the real migrations); there is no shared fixture for it.

```python
# test_account_lock_schema_sqlite.py
def test_migration_109_adds_the_lock_columns(tmp_path):
    users = {r[1] for r in db.execute("PRAGMA table_info(users)")}
    keys = {r[1] for r in db.execute("PRAGMA table_info(api_keys)")}
    assert {"security_locked_at", "credentials_valid_after"} <= users
    assert "suspended_at" in keys

def test_migration_109_is_idempotent(tmp_path):
    migration_109_add_account_lock_columns(db)   # second run
    assert [r[1] for r in db.execute("PRAGMA table_info(users)")].count("security_locked_at") == 1

def test_migration_109_skips_missing_tables(tmp_path):
    # an empty database: no exception, and still no users table (model: test_migration_108_skips_without_users_table)

# test_account_lock_postgres.py
async def test_lock_columns_exist_and_are_timestamptz_pg(isolated_test_environment):
    # information_schema.columns for the three columns -> data_type == "timestamp with time zone"
async def test_serving_catalog_accepts_the_lock_columns_pg(isolated_test_environment):
    # the non-owner serving-pool admission check passes (model on test_postgres_runtime_schema.py:502-511)
async def test_raw_update_of_lock_columns_passes_the_users_write_guard_pg(isolated_test_environment):
    # UPDATE users SET security_locked_at = now() WHERE id = $1 through the pool succeeds and reads back non-null
```

Add the SQLite twin of the last test (`test_raw_update_of_lock_columns_passes_the_users_write_guard`) through the SQLite pool: the guard runs there too.

- [ ] **Step 3:** Run them; expected FAIL (no such column).
- [ ] **Step 4: Implement.**
  - SQLite: `migration_109_add_account_lock_columns(conn)` in the shape of `migration_108_add_users_email_change_columns` (`migrations.py:6589-6599`), guarding each table with `_sqlite_table_exists` and each `ALTER TABLE` with `PRAGMA table_info`. Register `Migration(109, "Add account security lock columns", ...)`.
  - PostgreSQL: `_ACCOUNT_LOCK_COLUMNS` and `_ensure_account_lock_columns(conn)` in the shape of `_ensure_users_email_change_columns` (`pg_migrations_extra.py:3674-3690`), probing `information_schema.columns` first, called right after it in `ensure_authnz_core_tables_pg`. `api_keys.suspended_at` goes with the other `api_keys` ALTERs (`:1487-1510`).
  - Catalog: add the two user columns to `_USERS_COLUMNS` (`postgres_auth_catalog.py:19`), `suspended_at` to the `api_keys` column set, and all three names to the timestamp-name set in `_column_type_allowed` (`~:590-625`).
  - Write guard: add `security_locked_at` and `credentials_valid_after` to `RAW_SAFE_USER_UPDATE_FIELDS` (`profile_user_fields.py:21-33`).
  - Test DDL: add the columns to **both** copies of the `users` DDL (`tests/AuthNZ/conftest.py:724`, `:1516`) and both copies of `api_keys` (`:854`, `:1591`).
  - Pinned counts: `test_mcp_discovery_migration.py:41` and `:97` (108 becomes 109, and the `[-8:]` window), `test_postgres_runtime_checks.py:~58`, `test_pg_migrations_authnz_core.py`, `test_postgres_runtime_schema.py`.
- [ ] **Step 5:** `run_py t1 tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_schema_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py tldw_Server_API/tests/AuthNZ/unit/test_postgres_runtime_checks.py tldw_Server_API/tests/AuthNZ/integration/test_postgres_runtime_schema.py tldw_Server_API/tests/lint/test_function_size_ratchet.py`. Expected: all pass, PostgreSQL ones not skipped.
- [ ] **Step 6:** Commit `feat(authnz): account lock columns (TASK-<ID>)`.

### Task 2: The lock module: state, check, lock, unlock, take-back

**Files:** Create `core/AuthNZ/account_lock.py`, `core/AuthNZ/account_lock_errors.py` (Decision 19). Modify `repos/users_repo.py`, `repos/api_keys_repo.py`, `repos/federated_identity_repo.py`, `core/DB_Management/user_profile_writes.py`, `core/AuthNZ/email_change.py`, `core/Audit/unified_audit_service.py`. Test: `test_account_lock_schema_sqlite.py`, `test_account_lock_postgres.py`.

**Interfaces:**
- Consumes: the Task 1 columns; `AuthnzUsersRepo.from_pool()`; `SessionManager.revoke_all_user_sessions(user_id, reason=..., revoked_by=...)`; `core/Utils/iso_datetime.parse_iso_utc` (aware UTC from a datetime or a string with `T` or a space; reuse it, do not write another parser); `VersionedUserWriteGateway.execute_update` as `update_user_email` uses it (`user_profile_writes.py:28-55`).
- Produces (importable from `core/AuthNZ/account_lock.py`; the two exception classes and the three string constants are defined in `account_lock_errors.py`):

```python
ACCOUNT_LOCKED_CODE = "account_locked"
ACCOUNT_LOCK_REVOKE_REASON = "account_locked"
ACCOUNT_RESTORED_REVOKE_REASON = "account_restored"
RECENT_API_KEY_DAYS = 7
PREVIOUS_ADDRESS_MAIL_DAYS = 7          # same window as user_profile_writes.py:14-25

class AccountSecurityLockedError(HTTPException):
    """423, detail {"code": "account_locked", "message": "This account is locked. Use the link sent to its email address to unlock it."}"""

class StaleCredentialError(InvalidTokenError):
    """A token issued before credentials_valid_after."""

class TakeBackUnavailableError(ValueError):     # also defined in account_lock_errors.py
    """The previous address now belongs to another account. Nothing was changed; the account is still locked."""

def is_security_locked(user: Mapping[str, Any]) -> bool
    # True whenever user.get("security_locked_at") is not None, even if the value does not parse (fail closed)
def assert_account_not_locked(user: Mapping[str, Any], token_iat: int | float | None = None) -> None
    # locked -> AccountSecurityLockedError
    # token_iat given and credentials_valid_after parses and token_iat < int(cva.timestamp()) -> StaleCredentialError
async def is_user_security_locked(user_id: int) -> bool          # direct read, no cache; False for an unknown id
def security_mail_recipients(user: Mapping[str, Any], *, now: datetime | None = None) -> list[tuple[str, bool]]
    # [(address, is_previous_address)]; the current email first; previous_email only when email_changed_at is
    # within PREVIOUS_ADDRESS_MAIL_DAYS and it differs from the current one
def take_back_address(user: Mapping[str, Any], token_payload: Mapping[str, Any], *, now: datetime | None = None) -> str | None
    # the token's email when token_payload["restore_email"] is True and that email is the user's previous address
    # by security_mail_recipients; otherwise None

async def set_lock(user_id: int) -> datetime | None
    # the lock time if this call locked it, None if it was already locked
async def clear_lock(user_id: int, *, take_back: str | None = None) -> datetime | None
    # the credentials_valid_after it set; None when nothing was done (not locked and no take_back, or another
    # request cleared the same lock first). Raises TakeBackUnavailableError before any write.
async def write_account_audit(event_type: AuditEventType, *, user_id: int, session_id: int | None = None,
                              ip_address: str | None = None, user_agent: str | None = None, metadata: dict | None = None) -> None
    # through api_key_audit._get_or_create_audit_service (Decision 15), then `await svc.flush()` (these events are
    # rare and the tests read them back); swallows and logs failures
```

- Produces (repos and DB-owned statements):

```python
# AuthnzUsersRepo
async def set_security_lock(self, user_id: int, *, now: datetime) -> bool
    # UPDATE users SET security_locked_at = :now WHERE id = :id AND security_locked_at IS NULL; True when one row changed
async def clear_security_lock(self, user_id: int, *, locked_at: Any, valid_after: datetime) -> bool
    # UPDATE users SET security_locked_at = NULL, credentials_valid_after = :valid_after
    #   WHERE id = :id AND security_locked_at = :locked_at     (the stored value exactly as read; compare-and-set)
    # True when one row changed
async def force_security_lock(self, user_id: int, *, now: datetime) -> None
    # UPDATE users SET security_locked_at = :now WHERE id = :id      (unconditional; take-back only)
async def set_credentials_valid_after(self, user_id: int, *, valid_after: datetime) -> None
async def find_locked_user_by_previous_email(self, email: str, *, since: datetime) -> dict | None
    # the user with security_locked_at set, lower(previous_email) = lower(:email) and email_changed_at > :since.
    # None when more than one row matches. SQLite compares with datetime(email_changed_at) > datetime(?).
async def email_in_use_by_another_user(self, email: str, *, user_id: int) -> bool

# core/DB_Management/user_profile_writes.py
async def restore_previous_email(connection, *, backend: str, user_id: int, previous_email: str, password_hash: str,
                                 gateway=None, ownership=UserVersionOwnership.GATEWAY_OWNS_ANCHOR) -> UserWriteResult
    # UPDATE users SET email = ?, previous_email = NULL, email_changed_at = CURRENT_TIMESTAMP, password_hash = ?,
    #   updated_at = CURRENT_TIMESTAMP WHERE id = ? AND LOWER(previous_email) = LOWER(?)
    # through VersionedUserWriteGateway.execute_update(profile_visible_fields=("email",)), one statement per backend

# AuthnzApiKeysRepo
async def suspend_keys_created_since(self, *, user_id: int, since: datetime, now: datetime) -> int
    # UPDATE api_keys SET suspended_at = :now WHERE user_id = :user_id AND status = 'active' AND suspended_at IS NULL AND created_at > :since
async def unsuspend_key_for_user(self, *, key_id: int, user_id: int) -> bool

# FederatedIdentityRepo (repos/federated_identity_repo.py:21)
async def delete_identity_links_created_since(self, *, user_id: int, since: datetime) -> int
    # created_at exists on both backends; SQLite compares datetime(created_at) > datetime(?)
```

`clear_lock(user_id, take_back=None)`, in this order:
1. Read the user row once; every value below comes from this read. Not locked and `take_back is None`: return None. With `take_back`: if `email_in_use_by_another_user(take_back)`, raise `TakeBackUnavailableError`. Compute `since`: `locked_at - RECENT_API_KEY_DAYS` when locked; `email_changed_at - RECENT_API_KEY_DAYS` for a take-back of an unlocked account; the earlier of the two for a take-back of a locked one.
2. **Every** take-back, locked or not, first writes a fresh lock with `force_security_lock(now)` and then re-reads `security_locked_at`; that re-read stored value is "the lock value" from here on. From this point nothing can mint a credential until step 7, and a plain unlock racing this take-back loses its compare-and-set because the value changed.
3. `suspend_keys_created_since(since)`.
4. `revoke_all_user_sessions(user_id, reason=ACCOUNT_RESTORED_REVOKE_REASON)`. Failure is logged and the steps continue (the `iat` rule covers the tokens).
5. With `take_back`: `restore_previous_email(...)` with `password_hash = PasswordService().hash_password(secrets.token_urlsafe(48))` (the column is NOT NULL on SQLite), then `FederatedIdentityRepo.delete_identity_links_created_since(since=since)` after `ensure_tables()`; if the federation tables are missing, log and continue.
6. `suspend_keys_created_since(since)` again (idempotent; catches a key created between steps 1 and 2).
7. `clear_security_lock(locked_at=<the lock value, bound exactly as it was read>, valid_after=max(floor(now), floor(locked_at) + 1))`. When it returns False another request cleared or replaced this lock first: a plain unlock returns None here. A take-back (which can only lose to another take-back) runs step 4 again, calls `set_credentials_valid_after(floor(now) + 1)`, and continues.
8. Sleep until `valid_after` has begun (at most one second), then return it.

`email_change.authorize_own_email_change` (`core/AuthNZ/email_change.py:34`): after the existing checks, refuse with code `email_recently_released` when any address in `new_emails` that differs from the stored one is another user's `previous_email` with `email_changed_at` inside the last 7 days. The gate must read **only on the `db_conn` it is given** (a pool read would break `tests/AuthNZ_Postgres/test_profile_update_pool_capacity_pg.py`), so the query is a connection-level helper beside `password_service.fetch_email_and_password_hash`: `async def email_recently_released_by_another_user(db_conn, emails: Sequence[str], *, user_id: int, is_postgres: bool) -> bool`, with the 7-day comparison in SQL per backend. The callers and the frontend need no change (Decision 18).

`created_at` on `api_keys` is written by the database default: UTC text `YYYY-MM-DD HH:MM:SS` on SQLite and a naive `TIMESTAMP` on PostgreSQL. Compare inside SQL, per backend: SQLite `datetime(created_at) > datetime(?)`; PostgreSQL `created_at > $2::timestamptz` (the cast uses the session time zone, which is what the default wrote in).

Add to `AuditEventType` (`unified_audit_service.py:383`): `AUTH_ACCOUNT_LOCKED = "auth.account.locked"`, `AUTH_ACCOUNT_UNLOCKED = "auth.account.unlocked"`, `AUTH_ACCOUNT_LOCK_EMAIL_SENT = "auth.account.lock_email_sent"`, `AUTH_SESSION_EVICTED = "auth.session.evicted"`.

- [ ] **Step 1: Write the failing tests** (pure functions unit-style in `test_account_lock_schema_sqlite.py`; state tests against a migrated SQLite pool with one user, built as `two_user_client` builds its pool; the same state tests in `test_account_lock_postgres.py`).

```python
def test_locked_user_raises_423_with_the_code():
    with pytest.raises(AccountSecurityLockedError) as exc:
        assert_account_not_locked({"security_locked_at": "2026-11-26 10:00:00"})
    assert exc.value.status_code == 423 and exc.value.detail["code"] == "account_locked"

def test_an_unreadable_lock_value_still_counts_as_locked():
    assert is_security_locked({"security_locked_at": "not a date"}) is True

@pytest.mark.parametrize("stored", ["2026-11-26 10:00:00", "2026-11-26T10:00:00+00:00", datetime(2026, 11, 26, 10, tzinfo=timezone.utc)])
def test_token_issued_before_unlock_is_stale(stored):
    user = {"security_locked_at": None, "credentials_valid_after": stored}
    at = int(datetime(2026, 11, 26, 10, tzinfo=timezone.utc).timestamp())
    with pytest.raises(StaleCredentialError):
        assert_account_not_locked(user, token_iat=at - 1)
    assert_account_not_locked(user, token_iat=at)        # the same second passes (Decision 10)
    assert_account_not_locked(user)                      # no token, nothing to compare

def test_mail_goes_to_the_previous_address_for_seven_days_only():
    now = datetime(2026, 11, 26, tzinfo=timezone.utc)
    recent = {"email": "new@example.com", "previous_email": "old@example.com", "email_changed_at": now - timedelta(days=6)}
    stale = {**recent, "email_changed_at": now - timedelta(days=8)}
    assert security_mail_recipients(recent, now=now) == [("new@example.com", False), ("old@example.com", True)]
    assert security_mail_recipients(stale, now=now) == [("new@example.com", False)]
    assert take_back_address(recent, {"email": "old@example.com", "restore_email": True}, now=now) == "old@example.com"
    assert take_back_address(recent, {"email": "new@example.com", "restore_email": True}, now=now) is None
    assert take_back_address(stale, {"email": "old@example.com", "restore_email": True}, now=now) is None

async def test_set_lock_is_first_writer_wins(pool, user_id):
    first, second = await set_lock(user_id), await set_lock(user_id)
    assert first is not None and second is None

async def test_clear_lock_suspends_only_keys_created_in_the_last_seven_days(pool, user_id):
    # three active keys with created_at backdated 10 days, 6 days and 1 hour before the lock; lock; clear_lock
    assert suspended_ids == {six_day_key, one_hour_key}
    assert row["security_locked_at"] is None and parse_iso_utc(row["credentials_valid_after"]) is not None

async def test_clear_lock_on_an_unlocked_account_does_nothing(pool, user_id):
    assert await clear_lock(user_id) is None and row["credentials_valid_after"] is None

async def test_valid_after_is_strictly_later_than_the_lock_second_and_already_begun(pool, user_id):
    locked_at = await set_lock(user_id); valid_after = await clear_lock(user_id)
    assert int(valid_after.timestamp()) > int(locked_at.timestamp()) and valid_after <= datetime.now(timezone.utc)

async def test_two_unlocks_of_one_lock_only_one_wins(pool, user_id):
    await set_lock(user_id); results = await asyncio.gather(clear_lock(user_id), clear_lock(user_id))
    assert sum(r is not None for r in results) == 1

async def test_take_back_restores_the_address_and_kills_the_password(pool, user_id):
    # set up with update_user_email (user_profile_writes.py:28), as test_email_change_sqlite.py does: a raw UPDATE of email is
    # refused by the write guard. email=new@, previous_email=old@, known password; lock; clear_lock(user_id, take_back="old@example.com")
    assert row["email"] == "old@example.com" and row["previous_email"] is None
    assert PasswordService().verify_password(known_password, row["password_hash"])[0] is False

async def test_take_back_on_an_already_unlocked_account_still_suspends_keys_and_drops_new_sso_links(pool, user_id):
    # as above, then clear_lock() (the intruder's unlock), unsuspend one key, create a new key, add an identity link, then take back
    assert all_three_recent_keys_suspended and identity_links == [] and row["email"] == "old@example.com"

async def test_take_back_is_refused_and_the_lock_kept_when_another_user_holds_the_address(pool, user_id, other_user_holding_old_address):
    with pytest.raises(TakeBackUnavailableError):
        await clear_lock(user_id, take_back="old@example.com")
    assert row["email"] == "new@example.com" and row["security_locked_at"] is not None

async def test_take_back_of_an_unlocked_account_leaves_nothing_made_while_it_ran(pool, user_id, monkeypatch):
    # the intruder unlocked first. Wrap AuthnzApiKeysRepo.suspend_keys_created_since with a plain function that, on its
    # first call, tries what an active intruder would: insert an API key row directly and call
    # session_manager.create_session. Then run the take-back.
    assert create_session_error_type is AccountSecurityLockedError          # step 2 locked the account first
    assert unsuspended_active_keys(user_id) == []                            # step 6 caught the key
    assert row["security_locked_at"] is None and row["email"] == "old@example.com"   # and it ends unlocked and restored

async def test_plain_unlock_racing_a_take_back_of_a_locked_account_does_not_win(pool, user_id, monkeypatch):
    # locked, with a recent previous address. Wrap AuthnzApiKeysRepo.suspend_keys_created_since with a plain function
    # that on its first call runs clear_lock(user_id) (the other link) and records the result, then run the take-back.
    assert plain_unlock_result is None
    assert row["email"] == "old@example.com" and row["security_locked_at"] is None and live_session_count(user_id) == 0

async def test_take_back_removes_sso_links_made_since_the_week_before_the_email_change_even_if_the_email_moved_again(pool, user_id):
    # email changed 2 days ago, an identity link created 1 day ago, the email changed again 1 hour ago (previous_email is kept)
    assert identity_links == []

async def test_email_change_to_an_address_another_account_just_released_is_refused(pool, user_id, other_user_id):
    # other user's previous_email = old@example.com, changed 1 day ago; user_id asks for old@example.com with the right password
    assert refused.code == "email_recently_released"


async def test_previous_address_lookup_refuses_an_ambiguous_match(pool):
    # two locked users with the same recent previous_email
    assert await repo.find_locked_user_by_previous_email("old@example.com", since=week_ago) is None
```

- [ ] **Step 2:** Run; expected FAIL (module missing).
- [ ] **Step 3:** Implement the module, the repo methods and `restore_previous_email` as specified.
- [ ] **Step 4:** `run_py t2 tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_schema_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py tldw_Server_API/tests/lint`. Expected: pass.
- [ ] **Step 5:** Commit `feat(authnz): account lock state and rule`.

### Task 3: Enforcement on every JWT and sign-in path

**Files:** `core/AuthNZ/User_DB_Handling.py`, `session_manager.py`, `repos/sessions_repo.py`, `EP`, `api/v1/endpoints/admin/admin_impersonation.py`, `core/MCP_unified/adapters/tldw_runtime.py`. Test: `tldw_Server_API/tests/AuthNZ_SQLite/_account_lock_fixtures.py`, `test_account_lock_enforcement_sqlite.py`, `test_account_lock_postgres.py`.

**Interfaces:**
- Consumes: Task 2 `assert_account_not_locked`, `AccountSecurityLockedError`, `StaleCredentialError`, `ACCOUNT_LOCK_REVOKE_REASON`, `set_lock`, `clear_lock`, `write_account_audit`.
- Produces: `_account_lock_fixtures.py` with `async def lock_user(user_id: int) -> None` (calls `set_lock` then `revoke_all_user_sessions(reason=ACCOUNT_LOCK_REVOKE_REASON)`), `async def unlock_user(user_id: int) -> None` (`clear_lock`), and `def assert_locked(response) -> None` (`status_code == 423` and `json()["detail"]["code"] == "account_locked"`). Later tasks use all three. The HTTP tests use `two_user_client`, `login`, `bearer`, `api_key_for` from `tests/AuthNZ_SQLite/conftest.py` and `_sessions_http_fixtures.py`.

Where the check goes:

| Path | Anchor | Change |
| --- | --- | --- |
| Per-request JWT | `User_DB_Handling.py:686-693` | Replace the inline `is_active` block with one call to a new module-level `_enforce_account_state(user, user_data, payload, credentials_exception)` that keeps the `InactiveUserError` behaviour and then calls `assert_account_not_locked(user_data, token_iat=payload.get("iat"))`, turning `StaleCredentialError` into `credentials_exception`. The function must not grow: confirm with the size ratchet. |
| Revoked token | `session_manager.py:1799` `revoked_token_exception` | Reason `ACCOUNT_LOCK_REVOKE_REASON` **and** the token's subject locked right now (`is_user_security_locked(int(sub))`, the subject read from the already-verified token) returns `AccountSecurityLockedError()`; otherwise the default (Decision 6). |
| Refresh | `EP:2383-2393`, `:2422-2432`, `:2538-2565` | The blacklist branch (`:2391`) maps the lock reason to `AccountSecurityLockedError` under the same "locked right now" condition; after the user fetch, `assert_account_not_locked(user, token_iat=token_payload.get("iat"))`. Add `except AccountSecurityLockedError: raise` **before** `except _AUTH_NONCRITICAL_EXCEPTIONS` (which would turn it into a 500); `StaleCredentialError` is an `InvalidTokenError` and already becomes the 401. |
| Password sign-in | `EP:1822-1827` | `assert_account_not_locked(user)` right after the `is_active` check: after the password is verified, before rehash and the MFA branch. |
| SSO callback | `EP:678-682` | The same call right after the `is_active` check, before `upsert_identity_link`. |
| `/mfa/login` | `EP:1883`, `EP:3572-3580` | The challenge payload gains `"issued_at": int(time.time())`. After the user fetch: `assert_account_not_locked(user, token_iat=payload.get("issued_at"))`; `StaleCredentialError` becomes the existing 400 "MFA session expired or invalid" through its own `except` clause (it is an `InvalidTokenError`, which the function's catch-all would turn into a 500). A payload without `issued_at` (cached before the deploy) is checked for the lock only. |
| `reset-password` | `EP:2690-2756` | After the token-record lookup, load the user with `_auth_service.fetch_active_user_by_id(db, user_id)`. When a row comes back: `assert_account_not_locked(user, token_iat=payload.get("iat"))`, and refuse when the token's `email` claim differs (case-insensitive) from the user's email. Both the stale token and the email mismatch answer 400 "Invalid or expired reset token": catch `StaleCredentialError` in its own clause, because `InvalidTokenError` is inside `_AUTH_NONCRITICAL_EXCEPTIONS` and would otherwise become the 500 at `:2767`. When no row comes back (an inactive user), behave as today. |
| Impersonation | `admin_impersonation.py:95-100` | `assert_account_not_locked(target_user)` after the inactive check. |
| Cap transaction | `repos/sessions_repo.py:146-160` | On the same `conn`, after the PostgreSQL advisory lock (`:153`) and at the top of the SQLite transaction: `SELECT security_locked_at FROM users WHERE id = ...` (PostgreSQL adds `FOR SHARE`, so it waits for an uncommitted lock); a non-null value raises `AccountSecurityLockedError`. `HTTPException` is not in `_SESSION_MANAGER_NONCRITICAL_EXCEPTIONS`, so it reaches the endpoint unchanged. |
| Cap eviction audit | `session_manager.py:1160-1180` | After the evicted sessions are blacklisted, one `write_account_audit(AuditEventType.AUTH_SESSION_EVICTED, user_id=..., session_id=<new id>, metadata={"evicted_session_ids": [...]})` in its own `try` like the alert scheduling beside it. |
| MCP WebSocket adapter | `tldw_runtime.py:272-283` | `authenticate_authnz_websocket_token` also catches `HTTPException` and `InactiveUserError` and returns the "not authenticated" result it returns for an invalid token, so the socket closes 1008. |

Not changed, and why: the virtual-key mints (`EP:1492`, `workflows.py:1610`) take an already-authenticated principal, which passed one of the rows above; `evaluations_auth.py:~207` flattens every `HTTPException` to 401, still a refusal; MCP HTTP (`mcp_unified_endpoint.py:431`) swallows it to "unauthenticated"; email-verification tokens only mark an address verified.

- [ ] **Step 1: Write the failing tests.**

```python
# test_account_lock_enforcement_sqlite.py
CREDENTIALS = ["access_token", "refresh_token", "virtual_jwt", "impersonation_jwt", "password"]

@pytest.mark.parametrize("kind", CREDENTIALS)
async def test_no_credential_works_while_locked(two_user_client, kind):
    # arrange per kind BEFORE locking: sign alice in; mint a virtual-key JWT with JWTService.create_virtual_access_token(scope="workflows");
    # mint an impersonation JWT the way admin_impersonation.py:103 does. Then lock_user(alice).
    # access_token: GET /api/v1/auth/sessions      refresh_token: POST /api/v1/auth/refresh
    # virtual_jwt: a route guarded by require_token_scope("workflows")   impersonation_jwt: GET /api/v1/users/me/profile
    # password: POST /api/v1/auth/login with the right password
    assert_locked(response)

async def test_wrong_password_on_a_locked_account_answers_like_any_wrong_password(two_user_client):
    assert response.status_code == 401 and response.json()["detail"] == "Incorrect username or password"

async def test_token_minted_before_unlock_stays_dead_after_unlock(two_user_client):
    # virtual-key JWT and impersonation JWT minted; lock_user; unlock_user (clear_lock itself waits out the second)
    assert response.status_code == 401 and response.json()["detail"] == "Could not validate credentials"   # both tokens
    # the pre-lock session token also answers 401 now, not 423: its blacklist reason is still "account_locked" but the user is not locked
    # and a fresh password sign-in returns an access_token that lists exactly one session

async def test_refresh_of_a_locked_account_is_423_not_500(two_user_client): assert_locked(response)

async def test_reset_password_token_minted_before_the_lock_is_refused_while_locked(two_user_client):
    # a reset token made as forgot_password makes it (jwt create_password_reset_token + the stored record); lock; POST /auth/reset-password
    assert_locked(response); assert password_hash_unchanged

async def test_reset_token_minted_before_unlock_is_refused_after_unlock(two_user_client):
    assert response.status_code == 400 and response.json()["detail"] == "Invalid or expired reset token"; assert password_hash_unchanged

async def test_reset_token_for_an_address_the_account_no_longer_has_is_refused(two_user_client):
    # token minted for alice@example.com, then the stored email changed to other@example.com
    assert response.status_code == 400 and password_hash_unchanged

async def test_admin_cannot_impersonate_a_locked_user(two_user_client): assert_locked(response)

async def test_capped_insert_refuses_a_locked_user(two_user_client):
    await set_lock(alice)      # the lock only, sessions untouched
    with pytest.raises(AccountSecurityLockedError):
        await session_manager.create_session(user_id=alice, access_token="a", refresh_token="r")
    assert live_session_count(alice) == count_before

async def test_cap_eviction_is_audited(two_user_client, real_audit_service):
    # five sign-ins at cap 4: one auth.session.evicted event for alice naming the evicted session id

async def test_other_user_is_untouched_by_alices_lock(two_user_client):
    assert bob_sessions_response.status_code == 200 and len(bob_sessions_response.json()) == 1

# test_account_lock_postgres.py
async def test_mfa_challenge_started_before_an_unlock_is_refused_pg(isolated_test_environment):
    # build the MFA user as the existing /mfa/login tests do (git grep -l "mfa/login" tldw_Server_API/tests);
    # password sign-in -> 202 challenge; lock_user; unlock_user; POST /mfa/login with a valid code
    assert response.status_code == 400 and response.json()["detail"] == "MFA session expired or invalid"

async def test_sign_in_waits_for_an_uncommitted_lock_and_is_refused_pg(isolated_test_environment):
    # connection B: BEGIN; UPDATE users SET security_locked_at = now() WHERE id = $1   (not committed)
    # start create_session as a task; assert it has not finished after 0.3 s; COMMIT B
    # the task raises AccountSecurityLockedError and no sessions row was inserted. No revoke step is run.
    # Without FOR SHARE the READ COMMITTED read sees NULL and the session is inserted: the test fails before the change.
```

- [ ] **Step 2:** Run; expected FAIL (locked accounts still answer 200).
- [ ] **Step 3:** Implement the table above.
- [ ] **Step 4:** `run_py t3 tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_enforcement_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py tldw_Server_API/tests/AuthNZ_SQLite/test_sessions_api_sqlite.py tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py tldw_Server_API/tests/AuthNZ_SQLite/test_magic_link_redemption.py tldw_Server_API/tests/lint/test_function_size_ratchet.py`. Expected: pass. Then every existing test of the touched functions: `git grep -l "verify_jwt_and_fetch_user\|auth/refresh\|reset-password\|impersonat" tldw_Server_API/tests | xargs` through `run_py t3b`.
- [ ] **Step 5:** Commit `feat(authnz): refuse every token and sign-in of a locked account`.

### Task 4: API keys: refused while locked, suspended after

**Files:** `repos/api_keys_repo.py`, `core/AuthNZ/api_key_manager.py`, `core/AuthNZ/account_lock.py`, `api/v1/schemas/api_key_schemas.py`, `api/v1/endpoints/users.py`. Test: `test_account_lock_api_keys_sqlite.py`, `test_account_lock_postgres.py`.

**Interfaces:**
- Consumes: Task 2 `unsuspend_key_for_user`, `clear_lock`; `require_interactive_session` (`api/v1/API_Deps/interactive_session_deps.py:25`); `emit_mandatory_api_key_management_audit` (`core/AuthNZ/api_key_audit.py:37`), as rotate and revoke use it.
- Produces: the key lookup row gains `suspended_at`, `owner_security_locked_at`, `owner_is_active`; `APIKeyMetadata.suspended_at: datetime | None`; `async def unsuspend_api_key(*, user_id: int, key_id: int) -> dict | None` in `account_lock.py` (the key's metadata row, or None when the key is not that user's or not suspended; writes the mandatory API-key audit with action `"unsuspend"`); `POST /api/v1/users/api-keys/{key_id}/unsuspend` returning `APIKeyMetadata` (404 `{"code": "api_key_not_suspended"}` on None).

The refusal rule in `validate_api_key`, exactly:
`row.get("owner_security_locked_at") is not None or row.get("owner_is_active") in (False, 0) or row.get("suspended_at") is not None`.
A row without the `owner_*` keys (the unit tests feed such rows: `tests/AuthNZ/unit/test_api_key_manager_validation.py:7-22`) is not refused.

- [ ] **Step 1: Write the failing tests.**

```python
async def test_sqlite_lookup_carries_the_owners_state(two_user_client):
    # fetch_active_by_key_id and fetch_active_by_hash_candidates for alice's key, before and after set_lock(alice)
    assert before["owner_security_locked_at"] is None and before["owner_is_active"] in (1, True)
    assert after["owner_security_locked_at"] is not None

async def test_api_key_of_a_locked_owner_is_invalid_and_usage_is_not_recorded(two_user_client):
    key = api_key_for(client, alice_tokens); usage_before = usage_count_of(key)
    await lock_user(alice)
    response = client.get("/api/v1/users/me/profile", headers={"X-API-KEY": key})
    assert response.status_code == 401 and response.json()["detail"] == "Invalid API key"
    assert usage_count_of(key) == usage_before and last_used_at_of(key) == last_used_before

async def test_mcp_http_refuses_api_key_of_locked_owner(two_user_client):
    # an MCP HTTP route that answers 200 for the key before the lock answers 401 after it; assert both bodies

async def test_api_key_of_an_inactive_owner_is_invalid_on_mcp_http(two_user_client):   # the existing gap, Q5

async def test_recent_api_key_stays_suspended_after_unlock(two_user_client):
    # old_key backdated 10 days, new_key created now; lock_user; unlock_user
    assert client.get(PROFILE, headers={"X-API-KEY": old_key}).json()["id"] == alice
    assert client.get(PROFILE, headers={"X-API-KEY": new_key}).status_code == 401
    listed = {k["id"]: k for k in client.get("/api/v1/users/api-keys", headers=bearer(new_tokens)).json()}
    assert listed[new_id]["suspended_at"] is not None and listed[old_id]["suspended_at"] is None

async def test_unsuspend_restores_one_key_and_needs_a_browser_session(two_user_client):
    assert client.post(UNSUSPEND(new_id), headers={"X-API-KEY": old_key}).json()["detail"]["code"] == "interactive_session_required"
    assert client.post(UNSUSPEND(new_id), headers=bearer(new_tokens)).json()["suspended_at"] is None
    assert client.get(PROFILE, headers={"X-API-KEY": new_key}).json()["id"] == alice

async def test_cannot_unsuspend_another_users_key(two_user_client):
    assert response.status_code == 404 and response.json()["detail"]["code"] == "api_key_not_suspended"

async def test_a_suspended_key_cannot_be_rotated(two_user_client):
    # POST /users/api-keys/{new_id}/rotate with the old key: refused, and no new active key row exists for alice
    assert response.status_code == 409 and response.json()["detail"]["code"] == "api_key_suspended"

async def test_bobs_key_keeps_working_while_alice_is_locked(two_user_client): ...
```

`test_account_lock_postgres.py` adds `test_key_lookup_carries_owner_state_pg` (both lookup functions) and `test_suspended_key_is_invalid_pg`.

- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3: Implement.**
  - Lookup SQL, both functions (`fetch_active_by_key_id` `api_keys_repo.py:244`, `fetch_active_by_hash_candidates` `:183`) and both backends: add `suspended_at` and two sub-selects, `(SELECT security_locked_at FROM users WHERE users.id = api_keys.user_id) AS owner_security_locked_at` and the same for `is_active AS owner_is_active`.
    - SQLite goes through `_SQLITE_API_KEY_LOOKUP_COLUMNS` (`:32-55`). The builder (`:127`) uses the fallback whenever the **alias** is not an `api_keys` column, so the two owner entries must have fallback `None` (the sub-select is then always emitted); `suspended_at` is a real column and keeps fallback `NULL`.
    - PostgreSQL is the literal column list (`:208-218`, `:256-263`).
  - `validate_api_key` (`api_key_manager.py:785`): right after the key row is verified (`:816-818`) and before the expiry check, return `None` by the rule above. Pop the two `owner_*` fields before returning the row. This is before `record_key_usage` (`:933`).
  - Rotation: `fetch_key_for_user` is `SELECT *`, so `suspended_at` is available where the manager loads the key (`api_key_manager.py:~981`). Raise `APIKeySuspendedError`, a new subclass of `APIKeyRotationRejected` in `core/exceptions.py`. Catch it **before** the existing 404 clause in `api/v1/endpoints/users.py:945` and in `services/admin_api_keys_service.py:130` (admin rotation uses the same manager) and answer 409 `{"code": "api_key_suspended"}`.
  - `APIKeyMetadata` gains `suspended_at`; the list endpoint already returns `SELECT *` rows.
  - The unsuspend route in `users.py` beside the other api-key routes, with `interactive: InteractiveSession = Depends(require_interactive_session)` and `_require_principal_active_verified`; its body calls only `account_lock.unsuspend_api_key(...)` (no getter call, see the ratchet note).
- [ ] **Step 4:** `run_py t4 tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_api_keys_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py tldw_Server_API/tests/AuthNZ_Unit/test_api_key_usage_at_route.py tldw_Server_API/tests/lint/test_service_singleton_ratchet.py`. Then `git grep -ln "Inactive user" tldw_Server_API/tests` and run every hit: a test that expects 400 for an inactive owner **using an API key** now gets 401; update only those, and list each in the PR.
- [ ] **Step 5:** Commit `feat(authnz): API keys stop with a locked owner; recent keys stay suspended`.

### Task 5: The lock endpoint, its availability answer and the mail

**Files:** `core/AuthNZ/account_lock.py`, `core/AuthNZ/email_service.py`, `EP`, `api/v1/schemas/auth_schemas.py`. Test: `test_account_lock_endpoint_sqlite.py`; extend `_account_lock_fixtures.py`.

**Interfaces:**
- Consumes: Tasks 2-3; `JWTService.create_magic_link_token(email=, user_id=, expires_in_minutes=, additional_claims=)` (`jwt_service.py:353`); the device label function of `core/AuthNZ/device_label.py`; `_auth_request_client_ip`; `EmailService._read_setting_text` (`email_service.py:495`) and `_mask_email` (`:443`).
- Produces:

```python
# email_service.py
SMTP_TIMEOUT_SECONDS = 15
def public_web_base_url(self) -> str | None
    # the value _resolve_public_web_base_url uses for PUBLIC_WEB_BASE_URL (environment first, then settings), or None.
    # _resolve_public_web_base_url calls it, so availability and the link can never disagree.
async def send_account_locked_email(self, *, to_email, username, magic_token, expires_in_minutes, locked_at, device,
                                    ip_address, restores_address=False, base_url=None) -> bool

# account_lock.py
UNLOCK_PURPOSE = "account_unlock"
@dataclass(frozen=True)
class LockAvailability: available: bool; reason: str | None; sent_to: list[str]     # masked addresses
def lock_availability(user: Mapping[str, Any]) -> LockAvailability
    # reason, the first that applies: "single_user_mode", "mail_not_configured" (get_email_service().provider != "smtp"),
    # "public_url_not_configured" (get_email_service().public_web_base_url() is None), "email_not_verified"
@dataclass(frozen=True)
class LockResult: locked_at: datetime; email_sent: bool; sent_to: list[str]         # masked addresses that were mailed
async def send_lock_mail(user: Mapping[str, Any], *, locked_at: datetime | None, device: str, ip_address: str | None,
                         only_address: str | None = None) -> list[str]
    # one token per recipient of security_mail_recipients (or just only_address when it is one of them):
    # email=<that address>, additional_claims={"purpose": UNLOCK_PURPOSE, "restore_email": <is_previous_address>}.
    # The sends run concurrently (asyncio.gather). Returns the masked addresses whose send returned True.
async def lock_account(*, user_id: int, acting_session_id: int, ip_address: str | None, user_agent: str | None) -> LockResult

# auth_schemas.py
class AccountLockResponse(BaseModel): locked: bool; email_sent: bool; sent_to: list[str]
class LockAvailabilityResponse(BaseModel): available: bool; reason: str | None; sent_to: list[str]
```

- Routes, both with `Depends(require_interactive_session)` (an API key must not learn the masked addresses): `POST /api/v1/auth/account/lock` (409 `{"code": "lock_unavailable", "reason": <reason>}` when not available) and `GET /api/v1/auth/account/lock-availability`.
- Test fixture: `lockable(monkeypatch)` in `_account_lock_fixtures.py` sets `EMAIL_PROVIDER=smtp` and `PUBLIC_WEB_BASE_URL=https://app.example.com`, resets the email-service singleton (the module global behind `get_email_service`, `email_service.py:1180`), and replaces `EmailService._deliver_smtp_message` with a plain function appending `(to_address, body_text)` to the list it returns. `unlock_token_from(outbox, address) -> str` reads the `token=` value out of the captured body. Mail in these tests goes through the real template and link builder. Read `_deliver_smtp_message`'s real signature first and take the recipient and body from its arguments.

`lock_account` order: (1) availability; not available raises the 409. (2) `set_lock`; `None` means already locked: raise `AccountSecurityLockedError`. (3) `revoke_all_user_sessions(user_id, reason=ACCOUNT_LOCK_REVOKE_REASON, revoked_by=user_id)`, failure logged and swallowed (the lock is already refusing everything). (4) `send_lock_mail`, failure logged. (5) `write_account_audit` for `AUTH_ACCOUNT_LOCKED` (acting session id, IP, user agent) and `AUTH_ACCOUNT_LOCK_EMAIL_SENT` (recipient count).

The template `account_locked` (new key in `EMAIL_TEMPLATES`, `email_service.py:33`) states: the account was locked from a signed-in session, when, from which device and IP, the unlock link and its lifetime, the manual token box the magic-link template has, and, when `restores_address`, one line saying that using this link makes this address the account's email again and removes the current password. `locked_at` None (a re-sent link) omits the "when/where" lines. Sent with `redact_mock_tokens=True`. The link is built like the magic link's (`_resolve_public_web_base_url`, hosted path `/auth/magic-link`).

`smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=SMTP_TIMEOUT_SECONDS)` in `_deliver_smtp_message` (`:892-901`).

- [ ] **Step 1: Write the failing tests.**

```python
async def test_lock_signs_everything_out_and_mails_the_owner(two_user_client, lockable):
    first, second = login(client, "alice", pw), login(client, "alice", pw)
    body = client.post(LOCK, headers=bearer(second)).json()
    assert body == {"locked": True, "email_sent": True, "sent_to": ["a***@example.com"]}
    assert_locked(client.get(SESSIONS, headers=bearer(first))); assert_locked(client.get(SESSIONS, headers=bearer(second)))
    assert [to for to, _ in lockable] == ["alice@example.com"]
    assert "https://app.example.com/auth/magic-link?token=" in lockable[0][1]

async def test_lock_mail_also_goes_to_a_recent_previous_address(two_user_client, lockable):
    # previous_email=old@example.com, email_changed_at=now-1d
    assert sorted(to for to, _ in lockable) == ["alice@example.com", "old@example.com"]
    assert decode(unlock_token_from(lockable, "old@example.com"))["restore_email"] is True
    assert decode(unlock_token_from(lockable, "alice@example.com"))["restore_email"] is False

async def test_api_key_cannot_lock(two_user_client, lockable):
    assert response.json()["detail"]["code"] == "interactive_session_required"
    assert stored_row(alice)["security_locked_at"] is None

@pytest.mark.parametrize("setup, reason", [("mock_mail", "mail_not_configured"), ("no_public_url", "public_url_not_configured"), ("unverified", "email_not_verified")])
async def test_lock_is_refused_when_the_link_could_not_be_delivered(two_user_client, monkeypatch, setup, reason):
    assert response.status_code == 409 and response.json()["detail"] == {"code": "lock_unavailable", "reason": reason}
    assert client.get(SESSIONS, headers=bearer(tokens)).status_code == 200 and stored_row(alice)["security_locked_at"] is None

async def test_lock_holds_when_mail_fails(two_user_client, lockable, monkeypatch):
    # _deliver_smtp_message raises smtplib.SMTPException
    assert response.json() == {"locked": True, "email_sent": False, "sent_to": []}
    assert_locked(client.post(LOGIN, data=alice_credentials))

async def test_lock_holds_when_session_revoke_fails(two_user_client, lockable, monkeypatch):
    # SessionManager.revoke_all_user_sessions raises SessionError
    assert response.json()["locked"] is True
    assert_locked(client.get(SESSIONS, headers=bearer(tokens)))

async def test_availability_answer(two_user_client, lockable):
    assert client.get(AVAILABILITY, headers=bearer(tokens)).json() == {"available": True, "reason": None, "sent_to": ["a***@example.com"]}
    assert client.get(AVAILABILITY, headers={"X-API-KEY": key}).json()["detail"]["code"] == "interactive_session_required"

async def test_lock_writes_audit_events(two_user_client, lockable, real_audit_service):
    # events auth.account.locked and auth.account.lock_email_sent exist for alice; the first carries the acting session id

def test_smtp_connection_has_a_timeout(monkeypatch):
    # replace smtplib.SMTP with a plain function that records kwargs and returns
    # contextlib.nullcontext(SimpleNamespace(starttls=lambda: None, login=lambda *a: None, send_message=lambda *a, **k: None))
    assert recorded["timeout"] == 15
```

- [ ] **Step 2:** Run; expected FAIL (404 on the routes).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `run_py t5 tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_endpoint_sqlite.py tldw_Server_API/tests/lint`. Also the existing email-service tests: `git grep -l "email_service" tldw_Server_API/tests | xargs` through `run_py t5b`.
- [ ] **Step 5:** Commit `feat(authnz): lock an account from a signed-in session`.

### Task 6: Unlock through the mailed link

**Files:** `EP` (`verify_magic_link`, `request_magic_link`, `forgot_password`), `services/auth_service.py`, `core/AuthNZ/account_lock.py`. Test: `test_account_unlock_sqlite.py`, `test_account_lock_postgres.py`.

**Interfaces:**
- Consumes: Task 2 `clear_lock`, `security_mail_recipients`, `take_back_address`, `find_locked_user_by_previous_email`, `write_account_audit`; Task 5 `send_lock_mail`, `UNLOCK_PURPOSE`, `lockable`, `unlock_token_from`.
- Produces in `account_lock.py`:

```python
async def magic_link_email_allowed(user: Mapping[str, Any], token_payload: Mapping[str, Any]) -> bool
    # the token's email equals the user's email (case-insensitive); or take_back_address(user, token_payload) is not None
    # and that address is not in use by another user
async def redeem_unlock(user: Mapping[str, Any], token_payload: Mapping[str, Any], *, via: str,
                        ip_address: str | None, user_agent: str | None) -> bool
    # clear_lock(user_id, take_back=take_back_address(user, token_payload)); when it returned a time, writes
    # AUTH_ACCOUNT_UNLOCKED with metadata {"via": via, "took_back_address": bool}. via: "magic_link", "admin", "cli".
```

`verify_magic_link` (`EP:3055-3252`), in order. No MFA step (Q6); the decorator and response model do not change.
1. After the user is loaded (`:3118-3128`), and only when a user exists: `magic_link_email_allowed(user, payload)` must hold, else 400 "Invalid magic link token" (Decision 8), before the link is consumed.
2. Locked account and `purpose != "account_unlock"`: `AccountSecurityLockedError`, before the link is consumed (Decision 7).
3. Consume the link as today (`:3144`).
4. If the token has `purpose == "account_unlock"`: `await redeem_unlock(...)`. It unlocks a locked account; on an account that is no longer locked it acts only when the token is a take-back. `TakeBackUnavailableError` (the address was taken between rule 1 and here) answers the same 400; the account stays locked.
5. Create the session as today.
6. Inside the existing `revoke_on_failure` block, before the tokens are returned: re-read the user and require that the token's `email` claim equals the user's **current** email (after a take-back by this same request that is the restored address). If not, raise the same 400; `revoke_on_failure` revokes the session just made. This closes the race where a take-back commits between rule 1 and rule 5.

`request_magic_link` (`EP:2897-2974`): look the user up as today; when no user is found, also try `find_locked_user_by_previous_email(email, since=now - 7 days)`. For a locked user call `send_lock_mail(user, locked_at=None, ..., only_address=<the typed address>)` instead of the ordinary magic link. The answer stays the generic one; rate limits unchanged.

`forgot_password` (`EP:2586-2654`): `_fetch_user_by_email_for_password_reset` (`services/auth_service.py:~246`) also selects `security_locked_at`, `previous_email`, `email_changed_at`. For a locked user, send the lock mail to the typed address instead of a reset link. Nothing is raised (Decision 4).

- [ ] **Step 1: Write the failing tests.**

```python
async def test_unlock_link_signs_in_with_one_session_and_clears_the_lock(two_user_client, lockable):
    tokens = login(...); client.post(LOCK, headers=bearer(tokens))
    body = client.post(VERIFY, json={"token": unlock_token_from(lockable, "alice@example.com")}).json()
    sessions = client.get(SESSIONS, headers=bearer(body)).json()
    assert len(sessions) == 1 and sessions[0]["is_current"] is True
    old = client.get(SESSIONS, headers=bearer(tokens))
    assert old.status_code == 401 and old.json()["detail"] == "Could not validate credentials"   # dead, and not "locked"

async def test_unlock_link_is_single_use(two_user_client, lockable):
    assert second.status_code == 400 and second.json()["detail"] == "Invalid magic link token"

async def test_plain_magic_link_does_not_unlock(two_user_client, lockable):
    plain = JWTService().create_magic_link_token(email="alice@example.com", user_id=alice, expires_in_minutes=5)   # minted before the lock
    client.post(LOCK, ...); assert_locked(client.post(VERIFY, json={"token": plain}))
    assert stored_row(alice)["security_locked_at"] is not None

async def test_link_to_previous_address_takes_the_account_back(two_user_client, lockable):
    # email=new@example.com, previous_email=alice@example.com changed 1 day ago; lock; redeem the link mailed to alice@example.com
    assert row["email"] == "alice@example.com" and row["previous_email"] is None
    assert client.post(LOGIN, data={"username": "alice", "password": pw}).json()["detail"] == "Incorrect username or password"

async def test_link_to_newer_address_is_dead_after_take_back(two_user_client, lockable):
    # redeem the previous-address link first, then the one mailed to new@example.com
    assert response.status_code == 400 and response.json()["detail"] == "Invalid magic link token"

async def test_take_back_after_the_newer_address_unlocked_first_leaves_the_intruder_nothing(two_user_client, lockable):
    # redeem new@ first (the intruder is signed in); with that session create an API key and unsuspend a suspended one;
    # then redeem the alice@ link
    assert row["email"] == "alice@example.com"
    assert client.get(SESSIONS, headers=bearer(intruder)).status_code == 401
    assert client.get(PROFILE, headers={"X-API-KEY": created_key}).status_code == 401
    assert client.get(PROFILE, headers={"X-API-KEY": unsuspended_key}).status_code == 401

async def test_take_back_committed_during_the_other_links_verify_revokes_its_session(two_user_client, lockable, monkeypatch):
    # wrap SessionManager.create_session with a plain function that first runs clear_lock(alice, take_back="alice@example.com"),
    # then calls the real one; redeem the new@ link
    assert response.status_code == 400 and live_session_count(alice) == 0

async def test_take_back_link_is_refused_before_it_is_spent_when_another_account_holds_the_address(two_user_client, lockable):
    # bob's email set to alice's old address by direct setup (the route now refuses it); redeem alice's previous-address link
    assert response.status_code == 400 and stored_row(alice)["security_locked_at"] is not None
    assert token_blacklist_has(jti) is False                                   # the link was not consumed

async def test_send_me_a_new_link_mails_the_lock_template_to_the_typed_address_only(two_user_client, lockable):
    # locked with a recent previous address; POST /auth/magic-link/request {"email": "old@example.com"}
    assert response.json() == {"message": "If the account exists, a sign-in link has been sent"}
    assert [to for to, _ in new_mail] == ["old@example.com"] and decode(token)["purpose"] == "account_unlock"

async def test_forgot_password_on_a_locked_account_mails_the_unlock_link_not_a_reset_link(two_user_client, lockable):
    assert decode(unlock_token_from(new_mail, "alice@example.com"))["type"] == "magic_link"

async def test_anonymous_answers_do_not_reveal_the_lock(two_user_client, lockable):
    # for forgot-password, magic-link/request and a wrong-password login: the status code and body for locked alice equal those for unlocked bob

async def test_unlock_writes_an_audit_event(two_user_client, lockable, real_audit_service): ...   # auth.account.unlocked, via == "magic_link"
```

`test_account_lock_postgres.py` repeats the first, the take-back and the "newer address unlocked first" tests at the `account_lock` function level against `isolated_test_environment` (the versioned writer and the timestamp comparisons differ per backend).

- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement the six numbered rules and the two request endpoints.
- [ ] **Step 4:** `run_py t6 tldw_Server_API/tests/AuthNZ_SQLite/test_account_unlock_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_account_lock_postgres.py tldw_Server_API/tests/AuthNZ_SQLite/test_magic_link_redemption.py`. Then every existing magic-link and forgot-password test: `git grep -l "magic-link\|magic_link\|forgot-password" tldw_Server_API/tests | xargs` through `run_py t6b`.
- [ ] **Step 5:** Commit `feat(authnz): unlock an account through the mailed link`.

### Task 7: Admin: unlock, listings, own-email step-up, CLI

**Files:** `services/admin_users_service.py`, `api/v1/endpoints/admin/admin_user.py`, `api/v1/schemas/admin_schemas.py`, `repos/users_repo.py` (`list_users` projection `~:304-345`), `core/AuthNZ/reset_admin_password.py`, `apps/packages/ui/src/services/tldw/TldwApiClient.ts:1559`. Test: `test_account_lock_admin_sqlite.py`.

**Interfaces:**
- Consumes: Task 2 `clear_lock`, `set_lock`; `verify_privileged_action` (`services/admin_guardrails_service.py:94`); `emit_admin_account_audit_event` (`services/admin_audit_service.py:21`).
- Produces: `POST /api/v1/admin/users/{user_id}/security-unlock` with body `SecurityUnlockRequest{reason: str, admin_password: SecretStr | None, admin_reauth_token: SecretStr | None}` returning `{"unlocked": bool}` (false when the account was not locked); `UserSummary.security_locked_at` and `UserDetailResponse.security_locked_at` (`datetime | None`).

- [ ] **Step 1: Write the failing tests.** The file defines its own `admin_client` fixture: `two_user_client`'s setup plus one user with the `admin` role and a known password. Model the admin sign-in and the `reason` + `admin_password` body on the existing tests of `PUT /admin/users/{id}` (`git grep -l "admin_password" tldw_Server_API/tests/Admin`). The CLI test builds its own pool and admin user the same way.

```python
async def test_admin_unlock_needs_step_up_and_clears_the_lock(admin_client):
    refused = client.post(UNLOCK(alice), json={"reason": "owner called support"}, headers=admin)
    assert refused.status_code == 400 and stored_row(alice)["security_locked_at"] is not None
    assert client.post(UNLOCK(alice), json={"reason": "owner called support", "admin_password": admin_pw}, headers=admin).json() == {"unlocked": True}
    assert login(client, "alice", pw)["access_token"]

async def test_admin_unlock_also_suspends_recent_keys_and_kills_old_tokens(admin_client): ...
async def test_admin_listing_and_detail_show_the_lock(admin_client):
    assert listed[alice]["security_locked_at"] is not None and listed[bob]["security_locked_at"] is None

async def test_is_locked_is_no_longer_an_accepted_field(admin_client):
    response = client.put(USER(bob), json={"is_locked": True}, headers=admin)
    assert response.status_code == 400 and response.json()["detail"] == "No fields to update"     # was a 500

async def test_admin_changing_their_own_email_needs_step_up(admin_client):
    assert client.put(USER(admin_id), json={"email": "new-admin@example.com"}, headers=admin).status_code == 400
    assert stored_row(admin_id)["email"] == "admin@example.com"
    assert client.put(USER(admin_id), json={"email": "new-admin@example.com", "reason": "moved mailbox", "admin_password": admin_pw}, headers=admin).json()["email"] == "new-admin@example.com"

async def test_admin_changing_another_users_email_is_unchanged(admin_client):
    assert client.put(USER(bob), json={"email": "bob2@example.com"}, headers=admin).json()["email"] == "bob2@example.com"

async def test_reset_admin_password_cli_clears_a_lock(tmp_path, monkeypatch, fresh_services):
    await set_lock(admin_id); await reset_user_password("admin", new_password)
    assert row["security_locked_at"] is None and parse_iso_utc(row["credentials_valid_after"]) is not None
```

Read the exact "No fields to update" text from `admin_users_service.py:436-442` before writing that assert.

- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3: Implement.**
  - `security_unlock` service function: `enforce_admin_user_scope(principal, user_id, require_hierarchy=True)`, `verify_privileged_action(...)`, `clear_lock(user_id)`, then when it returned a time `write_account_audit(AUTH_ACCOUNT_UNLOCKED, metadata={"via": "admin"})` and `emit_admin_account_audit_event(event_type=AuditEventType.AUTH_ACCOUNT_UNLOCKED, action="admin.user.security_unlock", ...)`.
  - `update_user` (`admin_users_service.py:380`): the step-up condition becomes `role is not None or is_active is not None or (email is not None and int(principal.user_id) == int(user_id))`. Delete the `is_locked` branch (`:425-432`) and the field from `UserUpdateRequest` (`admin_schemas.py:73`) and from `AdminUserUpdateRequest` (`TldwApiClient.ts:1559`). Leave the response fields named `is_locked` alone.
  - `list_users` selects `security_locked_at` on both backends; both row builders and both schemas carry it.
  - `reset_user_password` (`reset_admin_password.py:26`) calls `clear_lock(user_id)` after the password update.
- [ ] **Step 4:** `run_py t7 tldw_Server_API/tests/AuthNZ_SQLite/test_account_lock_admin_sqlite.py` plus every test under `tldw_Server_API/tests/Admin` that mentions `admin/users`.
- [ ] **Step 5:** Commit `feat(admin): security unlock, lock in listings, step-up for own email`.

### Task 8: Integrations that act for a user without that user's credential

**Files:** `api/v1/utils/chatops/telegram_support.py` (`~:955-970` ask, `:669-717` approval callback), `api/v1/endpoints/slack.py` (`:175-182`, `:263-265`), `discord.py` (`:172-174`), `core/Agent_Client_Protocol/triggers.py` (`:396-460` `handle_webhook`). Test: `test_account_lock_integrations_sqlite.py`.

**Interfaces:** Consumes `is_user_security_locked(user_id)` (Task 2).

- Telegram ask: right after `linked_actor` is fetched and before the job is queued, a locked `auth_user_id` returns `_telegram_webhook_error(403, "account_locked")`.
- Telegram approval callback: the same check on the linked user before the decision is recorded; refuse the way that function refuses an unlinked actor.
- Slack and Discord: after the actor id is resolved, when it is all digits and names a locked user, answer with the file's own policy-error helper (`_slack_policy_error_response`, `_discord_policy_error_response`) carrying code `account_locked`.
- ACP trigger webhook: after the trigger row is loaded and its secret verified, a locked trigger owner is refused the way `handle_webhook` refuses a disabled trigger. Nothing is submitted.

- [ ] **Step 1: Write the failing tests** (model each on the existing test of that entry point: `git grep -l "telegram/webhook\|slack/commands\|discord/interactions\|triggers/webhook" tldw_Server_API/tests`).

```python
async def test_telegram_ask_from_a_locked_linked_user_queues_nothing(...): assert jobs_for(owner) == [] and <the refusal body>
async def test_telegram_approval_from_a_locked_linked_user_records_nothing(...): ...
async def test_slack_command_mapped_to_a_locked_user_queues_nothing(...): ...
async def test_discord_interaction_mapped_to_a_locked_user_queues_nothing(...): ...
async def test_slack_actor_that_is_not_a_tldw_user_id_is_untouched(...):   # "U123" still queues its job
async def test_acp_trigger_webhook_of_a_locked_owner_submits_nothing(...): ...
```

Read the refusal body each helper produces from the neighbouring assertions in the existing tests before writing each assert; every test asserts that body and the absence of the job or decision.

- [ ] **Step 2:** Run; expected FAIL. **Step 3:** Implement. **Step 4:** `run_py t8` on the new file and the existing chatops and ACP trigger tests. **Step 5:** Commit `feat(integrations): refuse a locked owner`.

### Task 9: ADR, docs, OpenAPI, full verification and PR

- [ ] **Step 1: ADR** `Docs/ADR/<N>-account-security-lock.md` and its row in `Docs/ADR/README.md`: the state (three columns), the one rule and where it is called, 423 against 401, API keys through the lookup, unlock by magic link with the purpose claim, the take-back (Q1), no MFA on unlock and why (Q6), the availability rule (Q3), and the limits below. It restates the interactive-credential rule from PR 2.
- [ ] **Step 2: Docs.** `Docs/API-related/AuthNZ-API-Guide.md`: an "Account lock" section under Session Management (the four routes, the 423 code in "Error Codes" `:669` and "HTTP Status Codes" `:653`, what stops and what comes back, the suspended keys). Design doc: a "Changes from the PR 3 plan (2026-11-26)" section listing the split, Decisions 3-17 and the answers to Q1-Q6, and remove the "Decision owed" paragraph. `CHANGELOG.md`. Run `Helper_Scripts/refresh_docs_published.sh`.
- [ ] **Step 3: OpenAPI.** Regenerate the fingerprint and `paths.d.ts`; `Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` passes.
- [ ] **Step 4: Full verification** (long runs in the background):
  - `run_py full-a tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Postgres`
  - `run_py full-b tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_Unit`
  - `run_py full-c tldw_Server_API/tests/Admin tldw_Server_API/tests/Notifications tldw_Server_API/tests/UserProfile`
  - `run_py full-d tldw_Server_API/tests/lint tldw_Server_API/tests/Docs --ignore tldw_Server_API/tests/lint/test_runtime_package_discovery.py`
  - The MCP tests: `git grep -l "validate_api_key\|X-API-KEY" tldw_Server_API/tests/MCP_unified | xargs` through `run_py full-e`.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce`
  - Frontend, for the one type edit: `cd apps/tldw-frontend && bun run typecheck --incremental false`.
  - Any failure is compared with the same test on `origin/dev` before it is called unrelated.
- [ ] **Step 5:** Close the Backlog task, push, open the PR against `dev` with the Behaviour changes, Known limits, the tests that changed expectation (Task 4), what was not run locally, how to redeem a link before 3b, and the Change summary waiver line. Then the clean-subagent review, its outcome posted on the PR.

## Known limits (go into the ADR and the PR)

- Streams and WebSockets already open when the lock is set keep running until they next authenticate. Jobs already running finish; queued and scheduled work is PR 4.
- Public share links, chat share tokens and media tickets created before the lock keep serving the content they point at.
- Clock skew between workers can refuse fresh tokens for a few seconds after an unlock, or let a token minted on a fast-clock worker just before the lock survive it by the amount of the skew (Decision 10).
- Any signed-in browser session can lock, including an intruder's. The cost to the owner is one email round trip.
- For 7 days after an email change, the holder of the previous mailbox can take the account back while it is locked (Q1 A). If the intruder unlocks first, the owner has only their link's lifetime to take it back. A take-back also removes SSO links the owner added in that window (where the provider has no email linking, an administrator re-links), and it is refused when another account already holds the old address (an administrator must act). An admin-route email change does not record a previous address, so it opens no such window and gives no such protection.
- An intruder can enrol MFA before the lock. The owner signs in by magic link (no MFA step there), sets a new password through forgot-password, then disables MFA with it (`/mfa/disable` needs the password, `EP:3448-3475`). Both are the follow-up task of Q6.
- An impersonation token minted by an admin who is then locked works until it expires (15 minutes); its subject is the impersonated user.
- If session creation fails right after an unlock, the link is spent and the account is unlocked with every old credential dead.
- Service tokens are not tied to a user. MCP demo tokens (`MCP_ENABLE_DEMO_AUTH`) name a user id without reading the user row and are not checked.
- Slack and Discord actors that are not numeric user ids are not checked.
- Lock mail is sent while the request waits; with a hanging mail server the lock request can take up to the SMTP timeout per step. The lock itself is set first.
- Lock is unavailable without `PUBLIC_WEB_BASE_URL`, and nothing in the WebUI redeems the link by a click until PR 3b.

## Review history

- 2026-11-26, first independent review (against `0d4cac9bc8`): not ready, five Critical. Fixed here: the take-back write goes through the versioned writer; the SQLite key lookup now really carries the owner's state; reset tokens are bound to `credentials_valid_after` and the account's email; a take-back always suspends keys and removes new SSO links; MFA is dropped from unlock (Q6); audit writes stay inside the layering baseline; the two-link race is closed by a second email check; the PostgreSQL race test now fails without `FOR SHARE`; cap evictions are audited; ACP trigger webhooks and Telegram approvals are covered; availability and the mail link read one value; the unsuspend route stays under the getter ceiling.
- 2026-11-26, second review of the rewrite: ready after fixes, one new Critical. Fixed here: a take-back to an address another account holds is refused with the lock kept, and an account can no longer take an address released in the last 7 days; the 423 is given only while the user is locked; `clear_lock` locks first on a take-back, clears by compare-and-set, suspends keys twice and sets `credentials_valid_after` strictly after the lock second; the exceptions live in a leaf module; SSO links are removed from the key-suspension start; MFA challenges carry `issued_at`.
- 2026-11-26, confirmation pass: no Critical. Fixed here: every take-back writes a fresh lock first, so a plain unlock racing it loses; the compare-and-set binds the re-read stored value; the released-address check reads on the gate's own connection; `mfa_login` gets its own clause for the stale challenge.
