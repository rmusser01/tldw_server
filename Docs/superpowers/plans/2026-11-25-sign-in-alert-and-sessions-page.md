# New-Sign-In Alert and Active Sessions Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a session is created while the user already has another live session, the user's other signed-in browsers learn of it (in-app banner plus an inbox notification). A new Active sessions page lists the live sessions with "This device" marked and lets the user sign out one session or all others. Ending sessions, and silencing the alert, require an interactive browser session, never an API key, virtual key or service token. In single-user mode the protection is the alert plus the session cap, not isolation from an API-key holder: `POST /auth/single-user/session` (`auth.py:1260-1317`) mints a cookie session for anyone holding the API key, and that session is "interactive". The remedy there is to rotate the API key.

**Architecture:**
- No schema change. `sessions` already holds `ip_address`, `user_agent`, `device_id`, `last_activity`; `user_notifications` already holds `source_job_id`, `source_domain`, `dedupe_key`.
- One core module per concern: `core/AuthNZ/device_label.py` (user agent to "Chrome on macOS"), `core/AuthNZ/interactive_session.py` (which session, if any, a request is), `core/AuthNZ/sign_in_alert.py` (build, write, dedupe, and who may silence the notification).
- `SessionManager.create_session` gets an opt-in `alert_new_sign_in` flag; the four real sign-in call sites pass it. The alert write is a detached, best-effort task.
- The sessions endpoints and the notification endpoints share one rule: the caller's session id comes only from a verified credential (a signed access token's `session_id` claim, or the validated single-user cookie) and must belong to the caller's own live sessions.
- The banner and the extension's OS notification stay quiet for the user's own browsers by a server-computed `same_install` flag on each session (the row's install id equals the current session's install id), never by IP or device label.
- WebUI: a shared service and page in `packages/ui` (so the extension gets the page too) and a banner mounted beside the existing toast bridge in `tldw-frontend`.

**Tech Stack:** FastAPI, SQLite and PostgreSQL through `DatabasePool` / `CollectionsDatabase`, loguru, pytest; TypeScript, Next.js pages router, antd plus the design-system primitives, vitest + Testing Library.

**Spec:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`, section "2. New-sign-in alert and Active sessions page (PR 2)", with "Decisions made by the owner" 2-4, "Anti-goals" and "Known limits". Sections 0 and 1 are built; 3 and 4 are later PRs and out of scope.

## Questions for the owner (decided 2026-11-25)

The owner approved all three defaults on 2026-11-25, adjusted per the second review (see Review history). Nothing here blocks execution.

1. **IP shown, behind the bundled WebUI proxy. Decided: show the IP the server saw, change nothing in Next, document the two proxy settings.** The design says the Next.js proxy forwards the browser's IP. It cannot: Next's rewrite proxy sets only `x-forwarded-host` (`apps/node_modules/.bun/next@16.1.4+363cf534159dda1d/node_modules/next/dist/server/lib/router-utils/proxy-request.js:30-36`), and a Next process with no reverse proxy in front never learns the peer address. A deployment with a reverse proxy in front already passes its `X-Forwarded-For` through, and the backend honours it only with `AUTH_TRUST_X_FORWARDED_FOR` plus `AUTH_TRUSTED_PROXY_IPS` (`core/AuthNZ/ip_allowlist.py:77`). A quickstart install therefore shows the proxy's address. The IP plays no part in banner suppression (Decision 15): that is by install id, so the proxy case needs no special handling.
2. **`POST /auth/logout` and API keys. Decided: `all_devices: true` requires an interactive session, and the revoke-all fallback is deleted.** `logout` signs every session out in two ways: `all_devices: true`, and the single-session branch's fallback (`auth.py:2102-2108`) that runs `revoke_all_user_sessions` whenever the bearer carries no `session_id`, which an API key sent as `Authorization: Bearer` always does. The design does not list `logout`, yet both defeat "a stolen API key cannot sign the owner out". A token without a `session_id` gets 403 `interactive_session_required` and revokes nothing (Task 4). The cost is that a script holding an API key can no longer sign sessions out through `logout`. The reviewer read that `refresh` copies the `session_id` claim (`auth.py:2395`, `:2439`) and that all four sign-in paths mint it, so a browser's refreshed token still logs out; Task 4 keeps a test for that and the implementer re-reads both lines.
3. **`MAX_SESSIONS_PER_USER=1` and the alert. Decided: also write the alert when the capped insert evicted a session, but only when the evicted session's install id differs from the new one's (or either is null).** An owner signed out by the cap still sees the new sign-in on their next sign-in. The same rule applies to a sign-in from an install that already has a live session (Decision 6, Task 5).

## Global Constraints

- **Precondition:** branch from `origin/dev` only after PR #3615 (`feat/session-cap-15628`, the session cap) has merged. Until then `create_session_record_with_cap`, the live predicate, `request_device_id` and `SESSION_CAP_REVOKE_REASON` do not exist on `dev` and every task below fails. Do not stack on the cap branch.
  `git -C /Users/macbook-dev/Documents/GitHub/tldw_server fetch origin dev && git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add .claude/worktrees/sign-in-alert -b feat/sign-in-alert-sessions-page origin/dev`. Work only in that worktree. One PR, base `dev`. Never `git stash`, never `--no-verify`, never arm auto-merge.
- **Re-verify every cited line at branch time.** Anchors below were read on the cap branch (first at `dbaa3761e7`, re-checked at `bbb5ec9ee5` on 2026-11-25); they shift if #3615 changed afterwards. Cite by name when a line no longer matches.
- **Backlog:** the executor files the task with `tools/backlog-py` and an explicit id (Task 1 step 1). The highest ids seen on `origin/dev` and in open PRs on 2026-11-25 are around TASK-15642; take the next free id after `git ls-tree --name-only origin/dev backlog/tasks/` and `gh pr list --state open --json number,files`, written `TASK-<ID>` below. Edit only with `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task edit TASK-<ID> ...`; stage with `git add -A backlog/tasks`.
- **ADR:** none in this PR. ADR-099 (the cap) arrives with #3615; open PRs claim 096-098, so the next free number is 100 if the owner wants one (re-check with `git ls-tree --name-only origin/dev Docs/ADR/` and `gh pr list --state open --json number,files`). The design assigns the lock ADR to PR 3; the interactive-credential rule and the non-silenceable alert are recorded in `Docs/API-related/AuthNZ-API-Guide.md` here (Task 10) and restated by PR 3's ADR.
- **Migration:** none. Neither the SQLite numbered migrations (`AuthNZ/migrations.py`; 108 is claimed by open PR #3614) nor the PostgreSQL owner migration (`pg_migrations_extra.py`, `postgres_auth_catalog.py`, and its canonical-column write guard) is touched. If a column turns out to be needed, stop: it is a design change.
- `run_py <id> <files...>` means:
  `zsh ~/.cache/claude-tools/disk_guard.zsh "$TMPDIR/sia-<id>" -- /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider --basetemp "$TMPDIR/sia-<id>" -q <files...>`
- PostgreSQL tests use the `isolated_test_environment` fixture of `tldw_Server_API/tests/AuthNZ/conftest.py`; never build a database by hand. A PostgreSQL step passes only if the output shows the tests **passed**; if skipped, say so in the PR and rely on the CI Postgres job.
- Frontend: `cd apps/packages/ui && bunx vitest run <files> --maxWorkers=2` or `cd apps/tldw-frontend && bunx vitest run <files> --maxWorkers=2`, one vitest process at a time.
- Test emails use `@example.com` (pydantic `EmailStr` rejects `.test`).
- **Ratchets that will bite** (under `tldw_Server_API/tests/lint` unless noted):
  - `function_size_baseline.json`: `User_DB_Handling:verify_jwt_and_fetch_user` is at 436 of 436 and must not grow: the session id is read in `interactive_session.py`. `auth:login` is 433 of 454 lines, so Task 5's one added keyword argument fits. New functions stay under 300 lines.
  - `test_service_singleton_ratchet.py` (getter count): it counts zero-argument `get_\w+()` calls inside endpoint function bodies, and `endpoint_getter_calls_baseline.json` freezes `api/v1/endpoints/notifications.py` at 2 calls, `auth.py` at 34 and `users.py` at 10. No endpoint body this PR edits may gain a getter call. `resolve_current_session_id` therefore calls `get_session_manager()` itself when it is given no manager (core is outside the ratchet's scope); the notification endpoints call it without a manager, and the `auth.py` and `users.py` handlers pass the manager they already receive through `Depends(get_session_manager_dep)`. Calls through `interactive_session.resolve_current_session_id(...)` are not getters. Re-run the ratchet after Tasks 3, 4 and 6.
  - `layering_baseline.txt`: new imports go downward only (`core.AuthNZ` to `core.DB_Management`, `core.AuthNZ` to `core.AuthNZ`). Dependency wrappers live in `api/v1/API_Deps`; nothing under `core` imports `api`.
  - Endpoint-monkeypatch ratchet: tests patch `core.*` modules, never new attributes on `api/*/endpoints` modules.
  - Raw SQL: none in app code (repo and `CollectionsDatabase` methods suffice). Tests may use raw SQL to backdate rows.
  - `test_noncritical_exception_tuples.py`: tuples are named `*_NONCRITICAL_EXCEPTIONS`, never contain `BaseException` or `CancelledError`, and no bare `except Exception`.
  - Global-statements baseline: no `global` statement (a module-level `set` mutated with `.add` is fine).
  - Test-quality ratchet (`Helper_Scripts/ci/test_quality_triage.py --enforce`): the tests below assert on real objects and HTTP bodies. Do not add asserts inside test-defined fake classes, do not inject test-defined stub classes into the object under test, write no test whose asserts touch only `.status_code`, and no `status_code in (200, 5xx)`; every HTTP test asserts a body or a database effect. `test_quality_baseline.txt` must not grow. Patch `core` module attributes with `monkeypatch.setattr` and plain functions.
  - Route-auth ratchet: no new route. Fields and query parameters (`keep_current`, `kind`) change the OpenAPI schema, so regenerate with `make openapi-fingerprint` (Task 10).
  - `Docs/` edits are mirrored: run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff in the same commit.
  - Design-system ratchet (`apps/packages/ui/scripts/design-system-product-state-rules.mjs`): new UI imports `Alert`, `Badge`, `EmptyState`, `LoadingState` from `@/components/ui/...`, never antd's `Alert`, `Badge`, `Empty`, `Result`, `Spin`, `Tag`.
- Use loguru. **Never log tokens, cookie values, install ids (`device_id`), full user agents or email addresses.** Log the user id only, and honour `PII_REDACT_LOGS` like the neighbouring lines in `session_manager.py`. Counters go through `log_counter` from `core.Metrics.metrics_logger` (already imported in `session_manager.py:67`).
- **Test users need notification permissions.** `two_user_client` users are bare `users` rows (`create_authnz_test_user` inserts only the row); `get_effective_permissions` reads `user_roles`, which migration backfilled only at migrate time. Existing notification HTTP tests avoid this by overriding the principal (`Notifications/test_notifications_api.py:49-72`). The fixture built in Task 2 assigns the `user` role to both users, and its first test is a smoke test (Task 2).
- No new dependencies, settings or environment variables (the env-read ratchet stays untouched).
- Every commit ends with the `Co-Authored-By` trailer from the executing session's attribution reminder (the earlier plans used `Claude Opus 5.5 <noreply@anthropic.com>`); pass it as a second `-m`.
- Change summary: owner-written or owner waiver recorded before merge. Before merge: clean-subagent `/superpowers:requesting-code-review`, and diff the non-required Frontend E2E and admin-ui real-backend runs against a recent `dev` run.

## Review Focus

1. **Never a client-supplied "current".** `is_current` and every interactive check derive from a signed token claim or the validated cookie, and the id must be one of the caller's own live sessions. Task 2 `test_claim_naming_another_users_session_is_not_current`, Task 3 `test_client_cannot_claim_current_session`.
2. **Every route that ends a session is guarded, including the `/users/sessions` aliases.** The aliases call the `/auth` handlers as plain functions (`users.py:947`, `:981`, `:1009`), so a `Depends` default left unpassed would arrive as a truthy `Depends` object. Task 3 `test_session_token_ends_another_session_which_then_stops_working[/api/v1/users]` and the explicit keyword arguments in the alias bodies.
3. **A new sign-in can never silence its own alert.** The rule compares session ids (monotonic per table on both backends), not timestamps. Task 6 `test_source_session_cannot_dismiss_its_own_alert`, `test_later_session_cannot_mark_read_or_snooze`, `test_after_24_hours_any_browser_session_may_dismiss`.
4. **Sign-in never depends on the alert.** The write runs in a detached task and swallows named exceptions, and the scheduling call itself sits in its own `try` so an error there cannot turn a stored session into a `SessionError`. Task 5 `test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids`, `test_sessions_read_failure_does_not_fail_sign_in`, `test_scheduling_failure_does_not_fail_sign_in`.
5. **Test pollution.** Every production sign-in passes the alert flag, so any existing test that signs one user in twice would write through `CollectionsDatabase.for_user` into the real default `Databases/user_databases/`. `git status` cannot see it (`.gitignore` ignores `*.db` and `/Databases/`). Task 5 therefore adds an autouse fixture that makes alerts silent by default; only tests carrying the `sign_in_alerts` marker run the real scheduler, and those set a tmp `USER_DB_BASE_DIR`. After the full run, compare `find Databases/user_databases -newer <branch-start marker file>` (or the directory's file count) before and after.
6. **Own-browser suppression is by install id, never by IP or label.** The device label is a client-chosen User-Agent and a public IP is shared (CGNAT, VPN, office), so an attacker on the owner's IP who copies the User-Agent must not silence the banner. The server computes `same_install` per session row (row `device_id` not null, not the bare single-user cookie tag, and equal to the current session's `device_id`); the id itself is never returned. Task 3 `test_same_install_marks_only_sessions_of_the_callers_install`, Task 7 `isOwnSignIn` tests, Task 9 banner tests.
7. **A refusal code must survive the real HTTP client.** `apiClient` drops a 4xx object `detail` unless its code is allowlisted (`normalizeApiErrorBody`, `lib/api.ts:78-108`), so mocked tests alone would pass while the real banner showed a generic error. Task 9 adds the allowlist and a test that drives the real `apiClient` against a mocked `fetch`.
8. **A refused dismiss must not stop the notification runtime.** `reportMutationError` maps a non-401 4xx to `unavailable` and stops the stream (`notification-lifecycle.ts:83-106`, `NotificationLifecycleProvider.tsx:610-622`), and the banner renders only while `state === "active"`. Task 9 never calls `reportMutationError` or `recordFailedMutation` for `new_sign_in_protected`.

## Files

| File | Change |
| --- | --- |
| `tldw_Server_API/app/core/AuthNZ/device_label.py` (new) | `device_label(user_agent)` |
| `tldw_Server_API/app/core/AuthNZ/interactive_session.py` (new) | `InteractiveSession`, `resolve_current_session_id`, `install_id_or_none`, refusal exceptions |
| `tldw_Server_API/app/api/v1/API_Deps/interactive_session_deps.py` (new) | `get_current_session_id`, `require_interactive_session` |
| `tldw_Server_API/app/core/AuthNZ/sign_in_alert.py` (new) | alert build, write, dedupe, schedule; the silence rule |
| `tldw_Server_API/app/core/AuthNZ/session_manager.py` | `create_session(alert_new_sign_in=False)` hook |
| `tldw_Server_API/app/core/AuthNZ/single_user_session.py`, `app/api/v1/endpoints/auth.py` | the four call sites pass `alert_new_sign_in=True`; sessions routes and `logout` |
| `tldw_Server_API/app/api/v1/endpoints/users.py` | the three `/users/sessions` aliases |
| `tldw_Server_API/app/api/v1/schemas/auth_schemas.py` | `SessionResponse.is_current`, `.device_label`, `.same_install` |
| `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`, `app/api/v1/endpoints/notifications.py`, `app/core/DB_Management/Collections_DB.py` | `new_sign_in` kind; gating of mark-read, dismiss, snooze; `list_user_notifications_of_kind`; a `kind` filter on `GET /notifications` |
| `tldw_Server_API/app/services/notifications_prune_service.py` | `new_sign_in` retention, 30 days |
| `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py` | `revoke_all_sessions_for_user` and `mark_sessions_revoked_for_user_with_audit` touch live rows only; the capped insert returns `device_id` for evicted rows |
| `tldw_Server_API/tests/conftest.py`, `pyproject.toml` | autouse silent-alert fixture, `sign_in_alerts` marker |
| `tldw_Server_API/tests/AuthNZ_SQLite/conftest.py`, `_sessions_http_fixtures.py` (new) | the `two_user_client` fixture; shared helpers |
| `apps/tldw-frontend/lib/api.ts` | `normalizeApiErrorBody` passes `new_sign_in_protected` and `interactive_session_required` through as `errorCode` |
| `apps/packages/ui/src/services/notifications.ts`, `apps/tldw-frontend/lib/api/notifications.ts` | `markNotificationsRead` return type; `kind` parameter on `listNotifications` |
| `Docs/Design/2026-11-23-session-cap-alert-lock-design.md` | "Changes from the PR 2 plan" section |
| `apps/packages/ui/src/services/auth-sessions.ts`, `sign-in-alert.ts` (new) | session API and own-sign-in matching |
| `apps/packages/ui/src/components/Option/Settings/ActiveSessionsSettings.tsx` (new) | the page |
| `apps/packages/ui/src/routes/option-settings-route-registry.tsx`, `components/Layouts/settings-nav-config.ts`, `components/Layouts/AccountMenu.tsx`, `assets/locale/en/settings.json`, `assets/locale/en/option.json` | route, nav, account menu, copy |
| `apps/tldw-frontend/extension/routes/route-registry.tsx`, `apps/tldw-frontend/pages/settings/sessions.tsx` (new) | the extension route and the Next page |
| `apps/tldw-frontend/components/notifications/NewSignInBanner.tsx` (new), `NotificationToastBridge.tsx`, `NotificationsRoute.tsx`, `components/layout/WebLayout.tsx` | banner, toast skip, inbox, mount |
| `apps/packages/ui/src/entries/shared/notification-subscription.ts` | extension OS-notification suppression |
| `Docs/API-related/AuthNZ-API-Guide.md`, `Docs/Published/...`, `apps/packages/api-types/*` | docs mirror, OpenAPI fingerprint and paths |

---

### Task 1: Branch, Backlog task and the user-agent label

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/device_label.py`
- Test (create): `tldw_Server_API/tests/AuthNZ/unit/test_device_label.py`

**Interfaces:**
- Produces `device_label(user_agent: str | None) -> str`: `"<Browser> on <OS>"`; `"<Browser>"` when only the OS is unknown; exactly `"Unknown browser"` when the browser is unrecognised (spec: "an unrecognised agent is shown as 'Unknown browser'"). Only the first 512 characters are read.
- Rules, first match wins. Browser: `Edg/`, `EdgA/`, `EdgiOS/` Edge; `OPR/` Opera; `FxiOS/` or `Firefox/` Firefox; `CriOS/` or `Chrome/` Chrome; `Version/` together with `Safari/` Safari. OS: `iPhone` or `iPad` iOS; `Android` Android; `Windows NT` Windows; `CrOS` ChromeOS; `Mac OS X` macOS; `Linux` Linux.

- [ ] **Step 1: Create the worktree and file the Backlog task**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server fetch origin dev
git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add .claude/worktrees/sign-in-alert -b feat/sign-in-alert-sessions-page origin/dev
cd /Users/macbook-dev/Documents/GitHub/tldw_server/.claude/worktrees/sign-in-alert
git log --oneline -1 -- Docs/ADR/099-per-user-session-cap.md   # must print a commit: #3615 is merged
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task create \
  "New-sign-in alert and Active sessions page" -s "In Progress" -l authnz,security,webui \
  -d "PR 2 of the session cap, alert and lock design (Docs/Design/2026-11-23-session-cap-alert-lock-design.md section 2): new_sign_in notification, is_current and device_label on /auth/sessions, keep_current on revoke-all, interactive-session requirement for ending sessions and silencing the alert, Active sessions settings page and banner." \
  --ac "A session created while the user has another live session writes one new_sign_in notification (warning, title New sign-in to your account) that stays until read or dismissed" \
  --ac "GET /auth/sessions reports is_current and device_label; POST /auth/sessions/revoke-all accepts keep_current" \
  --ac "Revoke-one, revoke-all and logout (all_devices, or a token with no session id) refuse API keys, virtual keys and service tokens with 403 interactive_session_required" \
  --ac "A new_sign_in notification can be marked read, dismissed or snoozed only by a browser session older than its source session, or by any browser session once 24 hours old" \
  --ac "The WebUI shows a banner for an unread new_sign_in alert from another session and an Active sessions page at /settings/sessions" \
  --notes "ADR assessment: no ADR in this PR; the rules are recorded in AuthNZ-API-Guide.md and restated by the lock ADR in PR 3. No schema change, so no migration." \
  --id TASK-<ID>
git add -A backlog/tasks
```
Expected: `backlog/tasks/task-<ID> - New-sign-in-alert-and-Active-sessions-page.md` exists.

- [ ] **Step 2: Write the failing tests** in `test_device_label.py`

```python
import pytest

from tldw_Server_API.app.core.AuthNZ.device_label import device_label

CHROME_MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
SAFARI_IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"


@pytest.mark.parametrize(
    "agent,expected",
    [
        (CHROME_MAC, "Chrome on macOS"),
        (SAFARI_IPHONE, "Safari on iOS"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0", "Firefox on Windows"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0", "Edge on Windows"),
        ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 OPR/115.0.0.0", "Opera on Linux"),
        ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36", "Chrome on Android"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15", "Safari on macOS"),
        ("Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36", "Chrome on ChromeOS"),
        ("Mozilla/5.0 (Unknown OS) Chrome/130.0.0.0", "Chrome"),
        ("python-requests/2.32.3", "Unknown browser"),
        ("curl/8.7.1", "Unknown browser"),
        ("Unknown", "Unknown browser"),
        ("", "Unknown browser"),
        (None, "Unknown browser"),
    ],
)
def test_device_label(agent, expected):
    assert device_label(agent) == expected


def test_label_never_echoes_versions_or_the_raw_agent():
    label = device_label(CHROME_MAC)
    assert "130" not in label and "Mozilla" not in label and "10_15" not in label


def test_only_the_first_512_characters_are_read():
    assert device_label("x" * 600 + " Firefox/132.0 (Windows NT 10.0)") == "Unknown browser"
```

- [ ] **Step 3: Run, expect failure** (`ModuleNotFoundError`): `run_py t1-red tldw_Server_API/tests/AuthNZ/unit/test_device_label.py`

- [ ] **Step 4: Implement** `device_label.py` as two ordered tuples of `(needle, label)` and a `for` loop each; no regex. The module docstring states the "browser and OS family only" rule.

- [ ] **Step 5: Run, expect pass**: the Step 3 command; then `ruff check` on both files.

- [ ] **Step 6: Commit** `feat(authnz): label a session's browser and OS from its user agent (TASK-<ID>)` with the trailer.

---

### Task 2: Which session is this request? Interactive-session resolution

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/interactive_session.py`
- Create: `tldw_Server_API/app/api/v1/API_Deps/interactive_session_deps.py`
- Create: `tldw_Server_API/tests/AuthNZ_SQLite/_sessions_http_fixtures.py` (shared by Tasks 2-6; not a test module): the `SessionsCtx` dataclass and plain helper functions, no fixtures
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/conftest.py` (16 lines today: it only re-exports three shared fixtures inside a `try`). Add the `two_user_client` fixture here, importing `SessionsCtx` from `_sessions_http_fixtures`. Test modules then receive the fixture by name without importing it (importing a fixture by name triggers ruff F811 on every test that takes it as a parameter).
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_interactive_session_sqlite.py`

**What the code shows (cap branch):**
- A multi-user browser holds an access JWT whose `session_id` claim is added at every sign-in (`auth.py:431` in `_issue_multi_user_tokens`, `:1915` in `login`, `:3202` in `verify_magic_link`) and kept on refresh (`:2439`). `verify_jwt_and_fetch_user` (`core/AuthNZ/User_DB_Handling.py:470`) checks the blacklist and builds a principal with `kind="user"`, `token_type="access"` (`:875`) but exposes no session id. `logout` reads it the same way (`auth.py:2087`, `payload.get("session_id")`).
- A single-user cookie session resolves to `token_type="single_user_session"` and sets `request.state.single_user_session_id` (`core/AuthNZ/auth_principal_resolver.py:364`).
- API keys resolve to `token_type="api_key"`; service tokens to `kind="service"` (service branch of `get_auth_principal`, `auth_principal_resolver.py:382`); scoped (virtual-key) JWTs are refused with 403 by `verify_jwt_and_fetch_user` on routes that do not enforce scope (the scoped-claim block, `User_DB_Handling.py:~530-575`) and carry no `session_id`; impersonation sets `principal.impersonation`.

**Interfaces:**
- Produces, in `interactive_session.py`:
  - `INTERACTIVE_REQUIRED_CODE = "interactive_session_required"`
  - `@dataclass(frozen=True) class InteractiveSession: user_id: int; session_id: int`
  - `def interactive_required_http_exception() -> HTTPException`: 403, `detail={"code": "interactive_session_required", "message": "This action needs a browser sign-in. API keys, virtual keys and service tokens cannot end sessions."}`
  - `SESSION_STORE_UNAVAILABLE_CODE = "session_store_unavailable"`; `def session_store_unavailable_http_exception() -> HTTPException`: 503, `detail={"code": "session_store_unavailable", "message": "Sessions could not be read. Try again."}`.
  - `def install_id_or_none(device_id: str | None) -> str | None`: None when `device_id` is None, empty, or the bare single-user cookie tag `SESSION_DEVICE_ID` (`"single-user-cookie:v1"`, `single_user_session.py:20`, stored by `mint_single_user_session` when the request has no install header, `:71`); otherwise `device_id` unchanged. A bare tag is shared by every header-less cookie session, so it must never count as "the same install" (Task 3 `same_install`, Task 5 alert fatigue). Import `SESSION_DEVICE_ID` from `core.AuthNZ.single_user_session` (core to core).
  - `async def resolve_current_session_id(request: Request, principal: AuthPrincipal, session_manager: SessionManager | None = None) -> int | None`. When `session_manager` is None it calls `get_session_manager()` (`core.AuthNZ.session_manager`, `:2170`) itself; this keeps the getter out of endpoint bodies (the getter ratchet in Global Constraints). None unless all hold: `principal.kind == "user"`, `principal.user_id` is a positive int, `principal.impersonation` is false, and the id found is one of `await session_manager.get_user_sessions(principal.user_id, strict=True)` (`strict` exists, `session_manager.py:1759-1788`; non-strict returns `[]` on a database error, which would read as "read-only API key"). A `SessionError` from the strict read is turned into `session_store_unavailable_http_exception()`. The id comes from `request.state.single_user_session_id` when `principal.token_type == "single_user_session"`, or from `get_jwt_service().decode_access_token(bearer_credential(request.headers.get("Authorization")))["session_id"]` when `principal.token_type == "access"`. A decode error or a non-int claim gives None.
- Produces, in `interactive_session_deps.py`:
  - Import `get_auth_principal` and `get_session_manager_dep` from `tldw_Server_API.app.api.v1.API_Deps.auth_deps`, as `auth.py` and `users.py` do, not from the core resolver.
  - `async def get_current_session_id(request: Request, principal: AuthPrincipal = Depends(get_auth_principal), session_manager: SessionManager = Depends(get_session_manager_dep)) -> int | None`
  - `async def require_interactive_session(request, principal, session_manager) -> InteractiveSession` (same dependencies): raises `interactive_required_http_exception()` when the id is None.
- Produces, in `AuthNZ_SQLite/conftest.py`: fixture `two_user_client` yielding the frozen dataclass `SessionsCtx(client, alice_id, bob_id, password)` (the dataclass lives in `_sessions_http_fixtures.py`). Build it from `multi_user_client` in `test_session_cap_http_sqlite.py:20-52` with two verified users `alice` and `bob` (`alice@example.com`, `bob@example.com`), `MAX_SESSIONS_PER_USER=4`, and `USER_DB_BASE_DIR` set to `tmp_path / "user_dbs"` as `Notifications/test_notifications_api.py::notifications_app` does (lines 53-60; that app also sets `settings.USER_DB_BASE_DIR`, so set both and restore). **Permissions:** after creating each user, `await AuthnzUsersRepo(pool).assign_role_if_missing(user_id=<id>, role_name="user")` (`core/AuthNZ/repos/users_repo.py:468`), because `create_authnz_test_user` inserts only the `users` row and `RequirePermission(NOTIFICATIONS_READ / NOTIFICATIONS_CONTROL)` reads `user_roles`. `assign_role_if_missing` is a no-op when the `user` role row does not exist; if the smoke test below still gets 403, seed the roles first with `async with pool.transaction() as conn: await ensure_baseline_rbac_seed(conn)` (`core/AuthNZ/rbac_seed.py:148`, which grants `notifications.read` and `notifications.control` to `user`). Helpers in `_sessions_http_fixtures.py`: `login(client, username, password, *, device=None, user_agent=None) -> dict`, `bearer(tokens) -> dict`, `api_key_for(client, tokens) -> str` (POST `/api/v1/users/api-keys`, `users.py:769`, body `{"name": "t", "scope": "write"}`, returns the `key` field of the 201 body), `session_id_of(tokens) -> int` (decode the access token's `session_id` claim with `get_jwt_service().decode_access_token`). Also `async seed_session_revoked_with_reason(user_id, reason) -> int` (a dead `sessions` row inserted through `AuthnzSessionsRepo.create_session_record` and revoked with `revoke_session_record`) and `async revoke_reason_of(session_id) -> str | None` (a raw SELECT, allowed in tests), used by Task 3.

- [ ] **Step 1: Write the failing tests** in `test_interactive_session_sqlite.py` (`pytestmark = pytest.mark.asyncio`; import the helpers from `_sessions_http_fixtures`; the `two_user_client` fixture comes from `conftest.py` and is not imported). `_request(headers)` builds `starlette.requests.Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})`; `manager = await get_session_manager()`. Every test body starts by binding `ctx = two_user_client; client = ctx.client; pw = ctx.password` (the snippets abbreviate with `ctx`/`client`/`pw`; `A`, `B`, `C` are `login(...)` results named in the test), and calls `get_jwt_service()` from `tldw_Server_API.app.core.AuthNZ.jwt_service`. The new tests also import `HTTPException`, `AuthnzSessionsRepo`, `DatabaseError` from `tldw_Server_API.app.core.exceptions` and `install_id_or_none`.

```python
async def test_two_user_client_users_may_read_notifications(two_user_client):
    # smoke test for the fixture's role grant: a bare users row would get 403 here. It runs first.
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    r = ctx.client.get("/api/v1/notifications", headers=bearer(tokens))
    assert r.status_code == 200 and r.json()["items"] == []
    assert ctx.client.get("/api/v1/notifications/unread-count", headers=bearer(tokens)).json()["unread_count"] == 0


async def test_access_token_session_is_current(two_user_client):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    got = await resolve_current_session_id(_request({"Authorization": f"Bearer {tokens['access_token']}"}), principal, manager)
    assert got == session_id_of(tokens)


@pytest.mark.parametrize("principal_kwargs", [
    {"kind": "service", "subject": "service:workflow-engine", "token_type": "service"},
    {"kind": "user", "token_type": "api_key"},
    {"kind": "user", "token_type": "access", "impersonation": True, "impersonated_by": 99},
])
async def test_service_api_key_and_impersonation_principals_are_not_interactive(two_user_client, principal_kwargs):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    principal = AuthPrincipal(user_id=ctx.alice_id, **principal_kwargs)
    request = _request({"Authorization": f"Bearer {tokens['access_token']}"})  # a valid session token is on the request
    assert await resolve_current_session_id(request, principal, manager) is None


async def test_claim_naming_another_users_session_is_not_current(two_user_client):
    ctx = two_user_client
    bob_tokens = login(ctx.client, "bob", ctx.password)
    forged = get_jwt_service().create_access_token(
        user_id=ctx.alice_id, username="alice", role="user",
        additional_claims={"session_id": session_id_of(bob_tokens)})   # validly signed, wrong owner
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request({"Authorization": f"Bearer {forged}"}), principal, manager) is None


async def test_claim_naming_a_revoked_session_is_not_current(two_user_client):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    await manager.revoke_session(session_id_of(tokens), revoked_by=ctx.alice_id, reason="test")
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request({"Authorization": f"Bearer {tokens['access_token']}"}), principal, manager) is None


async def test_token_without_a_session_claim_is_not_interactive(two_user_client):
    ctx = two_user_client
    bare = get_jwt_service().create_access_token(user_id=ctx.alice_id, username="alice", role="user")
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request({"Authorization": f"Bearer {bare}"}), principal, manager) is None


async def test_refreshed_token_is_still_interactive(two_user_client):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    new = ctx.client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).json()["access_token"]
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request({"Authorization": f"Bearer {new}"}), principal, manager) == session_id_of(tokens)


async def test_magic_link_token_is_interactive(two_user_client):
    # modelled on AuthNZ_SQLite/test_magic_link_redemption.py (token minted with create_magic_link_token, redeemed at /auth/magic-link/verify)
    ctx = two_user_client
    link = get_jwt_service().create_magic_link_token(email="alice@example.com", user_id=ctx.alice_id, expires_in_minutes=5)
    r = ctx.client.post("/api/v1/auth/magic-link/verify", json={"token": link}, headers={"X-TLDW-Device-Id": "dev-magic-0001"})
    assert r.status_code == 200, r.text
    got = await resolve_current_session_id(_request({"Authorization": f"Bearer {r.json()['access_token']}"}), AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access"), manager)
    magic_rows = [row["id"] for row in await manager.get_user_sessions(ctx.alice_id) if row["device_id"] == "dev-magic-0001"]
    assert magic_rows and got == magic_rows[0]


async def test_resolver_without_a_manager_uses_the_app_manager(two_user_client):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request({"Authorization": f"Bearer {tokens['access_token']}"}), principal) == session_id_of(tokens)


async def test_session_read_failure_is_a_503_not_a_read_only_key(two_user_client, monkeypatch):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    async def down(self, user_id):
        raise DatabaseError("sessions unavailable")      # core.exceptions.DatabaseError, the class the repo raises
    monkeypatch.setattr(AuthnzSessionsRepo, "get_active_sessions_for_user", down)
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    with pytest.raises(HTTPException) as caught:
        await resolve_current_session_id(_request({"Authorization": f"Bearer {tokens['access_token']}"}), principal, manager)
    assert caught.value.status_code == 503 and caught.value.detail["code"] == "session_store_unavailable"


@pytest.mark.parametrize("stored,expected", [
    (None, None), ("", None), ("single-user-cookie:v1", None),
    ("single-user-cookie:v1:abcd1234", "single-user-cookie:v1:abcd1234"), ("dev-aaaaaaaa", "dev-aaaaaaaa"),
])
def test_install_id_or_none_ignores_the_bare_cookie_tag(stored, expected):
    assert install_id_or_none(stored) == expected
```
The single-user cookie branch is covered by Task 3's cookie tests. `test_session_read_failure_is_a_503_not_a_read_only_key` patches a method on the repo class with a plain async function (a `core` module attribute, not an endpoint module).

- [ ] **Step 2: Run, expect failure** (import error): `run_py t2-red tldw_Server_API/tests/AuthNZ_SQLite/test_interactive_session_sqlite.py`

- [ ] **Step 3: Implement** the two modules as specified.

- [ ] **Step 4: Run, expect pass**; `ruff check`; `git status --short` shows nothing new under `Databases/`.

- [ ] **Step 5: Commit** `feat(authnz): resolve the interactive browser session behind a request (TASK-<ID>)`.

---

### Task 3: Sessions API: `is_current`, `device_label`, `keep_current`, interactive guard

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/auth_schemas.py:334-341` (`SessionResponse`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py`: `list_user_sessions` (`:2147`), `revoke_session` (`:2184`), `revoke_all_sessions` (`:2246`). `logout` is Task 4.
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py:928-1015` (the three aliases)
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`, two methods, both backends each. Add the live predicate (`LIVE_SESSION_PREDICATE_PG` / `LIVE_SESSION_PREDICATE_SQLITE`, `sessions_repo.py:20-24`) to every WHERE clause (four statements in each method, `... WHERE user_id = ? AND <predicate>` and `... AND id != ? AND <predicate>`):
  - `revoke_all_sessions_for_user` (`:435-524`): updates every row of the user with no live filter, so `sessions_revoked` counts dead rows and the update overwrites the `revoke_reason` of rows already revoked (for example `session_cap`).
  - `mark_sessions_revoked_for_user_with_audit` (`:728-815`): `SessionManager.revoke_all_user_sessions` (`session_manager.py:1432`) runs the first method and then `blacklist.revoke_all_user_tokens` (`session_manager.py:1458`, `token_blacklist.py:463`), which runs this one. Its UPDATE has no live filter and sets `revoke_reason = COALESCE(<new reason>, revoke_reason)`, so without this change it overwrites the `session_cap` reason that the first method just preserved and `test_revoke_all_counts_only_live_sessions_and_keeps_old_reasons` cannot pass.
  - **Callers checked** (the executor re-runs `git grep -n "mark_sessions_revoked_for_user_with_audit\|revoke_all_user_tokens\|revoke_all_sessions_for_user" -- tldw_Server_API`): `mark_sessions_revoked_for_user_with_audit` has one app caller, `TokenBlacklist.revoke_all_user_tokens`; that method is called by `SessionManager.revoke_all_user_sessions` (logout, revoke-all, admin revoke-all, `admin_sessions_mfa_service.py:163`) and by password reset (`auth.py:2732`). None relies on re-marking a dead row: a dead row (already revoked, or its refresh token expired) has nothing left to revoke. Three tests call the repo method directly with live sessions and still pass (`test_authnz_sessions_repo_sqlite.py:332`, `test_authnz_sessions_repo_postgres.py:178`, `test_authnz_sessions_repo_backend_selection.py:221`; the last asserts the substring `where user_id = $1 and id != $2`, which the predicate follows). If a caller turns up that does rely on re-marking, flip this method to `COALESCE(revoke_reason, <new>)` instead and say why in the PR.
  - `fetch_session_token_metadata_for_user` (`:526-603`) also loads dead rows. It is left alone: it only feeds the blacklist, and blacklisting an already-dead session's JTIs again is harmless.
- Modify: the existing tests that call `revoke_all_sessions`, `revoke_session` or `list_user_sessions` with a session-less credential or as plain functions. Find them with `git grep -n -E "sessions/revoke-all|users/sessions/[0-9]|auth\.(revoke_all_sessions|revoke_session|list_user_sessions)\(" -- tldw_Server_API/tests`. Known: `tests/AuthNZ/unit/test_auth_endpoints_extended.py:~1827` calls `auth.revoke_all_sessions(...)` directly and `:~1851` `auth.list_user_sessions(...)`; the new parameters would receive `Query`/`Depends` default objects, so pass `keep_current=False, interactive=InteractiveSession(user_id=..., session_id=...)` (and `current_session_id=None`) explicitly. Also `tests/AuthNZ/unit/test_user_endpoints.py:~227-330`, `tests/AuthNZ/integration/test_user_endpoints_integration.py`, `test_auth_comprehensive.py`, `test_single_user_login_api_key.py`, `tests/UserProfile/test_user_management_account_state_guards.py`: give each a session-bearing credential.
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_sessions_api_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_cookie.py`, `tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_postgres.py`

**Interfaces:**
- `SessionResponse` gains `is_current: bool = Field(False, description="True only for the session making this request, derived from the verified credential.")`, `device_label: str = Field("Unknown browser", description="Browser and OS family from the user agent.")` and `same_install: bool = Field(False, description="True when this session was signed in from the same browser install as the session making this request. The install id itself is never returned.")`.
- `list_user_sessions(current_user=..., session_manager=..., current_session_id: Optional[int] = Depends(get_current_session_id))`: each row sets `is_current=(row["id"] == current_session_id)`, `device_label=device_label(row.get("user_agent"))` and `same_install`, computed from the rows `get_user_sessions` returned (they carry `device_id`): `current_install = install_id_or_none(<device_id of the row whose id is current_session_id>)` (None when `current_session_id` is None), and `same_install = current_install is not None and install_id_or_none(row.get("device_id")) == current_install`. It still accepts any credential (`store/connection.tsx:813` and `Settings/tldw.tsx:641` probe it with whatever credential they hold).
- `revoke_session(session_id, current_user, session_manager, interactive: InteractiveSession = Depends(require_interactive_session))`: ownership still comes from `get_user_sessions(user_id)`; the revoke passes `expected_user_id=user_id` (`SessionManager.revoke_session`, `session_manager.py:1370`). A session that is not the caller's stays the existing idempotent 200 no-op.
- `revoke_all_sessions(current_user, session_manager, keep_current: bool = Query(False, description="Keep the caller's own session."), interactive: InteractiveSession = Depends(require_interactive_session))`: `except_session_id = interactive.session_id if keep_current else None`, passed to the existing `revoke_all_user_sessions(except_session_id=...)` (`session_manager.py:1416`, which already forwards it to the repo and the blacklist). Without the flag every live session is revoked (the repo now counts only live rows, so `sessions_revoked` is the number of live sessions ended), as today, apart from the new requirement.
- Aliases in `users.py`: the list alias adds `current_session_id: Optional[int] = Depends(get_current_session_id)`; the revoke aliases add `interactive: InteractiveSession = Depends(require_interactive_session)` and revoke-all also `keep_current: bool = Query(False)`. **Every new handler parameter is passed by keyword in the alias body** (Review Focus 2).

- [ ] **Step 1: Write the failing SQLite HTTP tests** in `test_sessions_api_sqlite.py` (fixture `two_user_client` from `conftest.py`; `A = login(alice, device="dev-aaaaaaaa", user_agent=CHROME_MAC)` and `B = login(alice, device="dev-bbbbbbbb", user_agent=SAFARI_IPHONE)`, constants as in Task 1):

```python
def test_list_marks_exactly_the_callers_session_current(two_user_client):
    rows = client.get("/api/v1/auth/sessions", headers=bearer(A)).json()
    assert [r["id"] for r in rows if r["is_current"]] == [session_id_of(A)]
    assert {r["id"]: r["device_label"] for r in rows} == {session_id_of(A): "Chrome on macOS", session_id_of(B): "Safari on iOS"}


def test_client_cannot_claim_current_session(two_user_client):
    other = session_id_of(B)
    r = client.get(f"/api/v1/auth/sessions?is_current=true&current={other}&session_id={other}",
                   headers={**bearer(A), "X-TLDW-Device-Id": "dev-bbbbbbbb"})
    assert [x["id"] for x in r.json() if x["is_current"]] == [session_id_of(A)]


def test_same_install_marks_only_sessions_of_the_callers_install(two_user_client):
    C = login(client, "alice", pw, device="dev-aaaaaaaa", user_agent=CHROME_MAC)      # a second sign-in from A's install
    rows = {r["id"]: r for r in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()}
    assert {i: r["same_install"] for i, r in rows.items()} == {session_id_of(A): True, session_id_of(C): True, session_id_of(B): False}
    raw = json.dumps(rows)
    assert "dev-aaaaaaaa" not in raw and "dev-bbbbbbbb" not in raw     # the install id is never returned


def test_api_key_lists_sessions_but_none_is_current(two_user_client):
    rows = client.get("/api/v1/auth/sessions", headers={"X-API-KEY": api_key_for(client, A)}).json()
    assert len(rows) == 2 and not any(r["is_current"] or r["same_install"] for r in rows)


@pytest.mark.parametrize("prefix", ["/api/v1/auth", "/api/v1/users"])
def test_api_key_cannot_end_sessions(two_user_client, prefix):
    key = {"X-API-KEY": api_key_for(client, A)}
    one = client.delete(f"{prefix}/sessions/{session_id_of(B)}", headers=key)
    all_ = client.post(f"{prefix}/sessions/revoke-all", headers=key)
    assert one.status_code == all_.status_code == 403
    assert one.json()["detail"]["code"] == all_.json()["detail"]["code"] == "interactive_session_required"
    assert len(client.get("/api/v1/auth/sessions", headers=bearer(A)).json()) == 2   # nothing was revoked



def test_scoped_virtual_key_cannot_end_sessions(two_user_client):
    minted = client.post("/api/v1/auth/virtual-key", json={"scope": "workflows", "ttl_minutes": 5}, headers=bearer(A)).json()["token"]
    r = client.delete(f"/api/v1/auth/sessions/{session_id_of(B)}", headers={"Authorization": f"Bearer {minted}"})
    assert r.status_code == 403 and "Scoped token" in r.json()["detail"]
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()} >= {session_id_of(A), session_id_of(B)}   # B survived


@pytest.mark.parametrize("prefix", ["/api/v1/auth", "/api/v1/users"])
def test_session_token_ends_another_session_which_then_stops_working(two_user_client, prefix):
    assert client.delete(f"{prefix}/sessions/{session_id_of(B)}", headers=bearer(A)).json()["message"] == "Session revoked successfully"
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 200


@pytest.mark.parametrize("prefix", ["/api/v1/auth", "/api/v1/users"])
def test_revoke_all_keep_current_keeps_only_the_callers_session(two_user_client, prefix):
    r = client.post(f"{prefix}/sessions/revoke-all?keep_current=true", headers=bearer(A))
    assert r.json()["details"]["sessions_revoked"] == 1
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 200
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401


def test_revoke_all_without_the_flag_still_signs_out_everything(two_user_client):
    assert client.post("/api/v1/auth/sessions/revoke-all", headers=bearer(A)).json()["details"]["sessions_revoked"] == 2
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 401


async def test_revoke_all_counts_only_live_sessions_and_keeps_old_reasons(two_user_client):
    ctx = two_user_client
    dead = await seed_session_revoked_with_reason(ctx.alice_id, "session_cap")   # raw INSERT/UPDATE helper in _sessions_http_fixtures.py
    r = ctx.client.post("/api/v1/auth/sessions/revoke-all", headers=bearer(A))
    assert r.json()["details"]["sessions_revoked"] == 2
    assert await revoke_reason_of(dead) == "session_cap"


def test_users_sessions_alias_reports_is_current(two_user_client):
    rows = client.get("/api/v1/users/sessions", headers=bearer(A)).json()
    assert [r["id"] for r in rows if r["is_current"]] == [session_id_of(A)]


def test_one_user_cannot_see_or_end_another_users_sessions(two_user_client):
    bob = login(client, "bob", password, device="dev-cccccccc")
    alice_ids = {session_id_of(A), session_id_of(B)}
    assert alice_ids.isdisjoint({r["id"] for r in client.get("/api/v1/auth/sessions", headers=bearer(bob)).json()})
    client.delete(f"/api/v1/auth/sessions/{session_id_of(A)}", headers=bearer(bob))          # idempotent no-op
    client.post("/api/v1/auth/sessions/revoke-all?keep_current=true", headers=bearer(bob))
    assert {r["id"] for r in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()} == alice_ids
```

- [ ] **Step 2: Write the failing cookie tests** in `test_sessions_api_cookie.py` (import `single_user_cookie_client` and `_mint` from `test_single_user_cookie_session.py`; CSRF is on there, so mutating cookie calls send `X-CSRF-Token: client.cookies["csrf_token"]`):
  - `test_cookie_session_lists_itself_as_current_and_api_key_does_not`: mint two installs (clear cookies between); with cookie 1 exactly one row is `is_current` and it is cookie 1's; with `X-API-KEY` none is.
  - `test_cookie_session_ends_another_cookie_session`: DELETE with cookie 1 plus the CSRF header returns the revoke message, and cookie 2 then gets 401.
  - `test_api_key_cannot_end_single_user_sessions`: `X-API-KEY` DELETE and revoke-all give 403 `interactive_session_required`; both cookies still work.
  - `test_cookie_revoke_all_keep_current`: cookie 1 survives, cookie 2 is revoked.
  - `test_cookie_sessions_without_an_install_header_are_never_same_install`: two mints with no `X-TLDW-Device-Id` (stored device tag is the bare `single-user-cookie:v1`); with cookie 1 every row has `same_install == False`, the current row included, and exactly one row has `is_current == True`. A second pair of mints that both send `X-TLDW-Device-Id: dev-cookie-0001` gives `same_install == True` on both rows.

- [ ] **Step 3: Write the PostgreSQL test** in `test_sessions_api_postgres.py` (`isolated_test_environment`, as `test_session_cap_postgres.py:336-363`: register `pgalice` and `pgbob` at `@example.com`, `AUTH_MODE=multi_user`, `JWT_SECRET_KEY="X"*40`): one test `test_sessions_api_postgres` running the `is_current` and `same_install` checks, `keep_current`, the API-key refusal and the cross-user isolation sequence above, each with body assertions, and the live-only revoke count with a `session_cap` row left untouched (this covers the PostgreSQL branch of both repo methods).

- [ ] **Step 4: Run, expect failure** (missing fields, 200 instead of 403, 422 for `keep_current`): `run_py t3-red <the three new files>`

- [ ] **Step 5: Implement** the schema, handlers and aliases; then update the old tests listed under Files.

- [ ] **Step 6: Run, expect pass**: `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py` (no new getter call in `auth.py` or `users.py`), the repo tests named under Files, the Step 4 files, `tldw_Server_API/tests/AuthNZ_SQLite/test_interactive_session_sqlite.py`, every test file the Files-list grep found, and `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`. PostgreSQL tests must run, not skip.

- [ ] **Step 7: Commit** `feat(authnz): mark the current session and require a browser session to end sessions (TASK-<ID>)`.

---

### Task 4: `logout` cannot be used with an API key to end sessions

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py:2005-2140` (`logout`; it is not in the size baseline)
- Modify: the existing logout tests (list below)
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_logout_guard_sqlite.py`

**What the code does today:** the single-session branch (`auth.py:2052-2108`) decodes the bearer with `jwt_service.verify_token`; a non-JWT bearer (an API key sent as `Authorization: Bearer`) fails to decode, so `payload = {}`, `session_id` is None, and the code falls into "Fallback to full session revoke" (`:2102-2108`), which calls `revoke_all_user_sessions`. So `POST /auth/logout` with an API key and no body already signs every session out, in addition to `all_devices: true` (`:2033-2039`).

**Interfaces:**
- `all_devices: true`: before revoking, `if await resolve_current_session_id(request, current_user, session_manager) is None: raise interactive_required_http_exception()` (the handler's existing `session_manager` dependency is passed, so no getter call enters the endpoint body; a database error becomes the 503 from Task 2, not a misleading 403).
- Premise check (Question 2): `refresh_token` (`auth.py:2288`) reads `session_id` from the old token (`:2395`) and copies it into the new access token (`:2439`), and `_issue_multi_user_tokens` (`:431`), `login` (`:1915`) and `verify_magic_link` (`:3198`) each mint it. The executor re-reads those four lines before implementing; if any no longer sets the claim, stop and raise it, because a browser holding such a token would get 403 on logout.
- Single-session branch: delete the revoke-all fallback. When the decoded `session_id` is None, raise `interactive_required_http_exception()` before blacklisting or revoking anything. A token that has a `session_id` is handled as today (blacklist the jti, revoke that one session).

- [ ] **Step 1: Write the failing tests** (fixture `two_user_client`; A and B are two alice logins):

```python
def test_api_key_as_bearer_cannot_log_everyone_out(two_user_client):
    key = api_key_for(client, A)
    r = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "interactive_session_required"
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()} == {session_id_of(A), session_id_of(B)}


def test_api_key_header_cannot_log_everyone_out_with_all_devices(two_user_client):
    r = client.post("/api/v1/auth/logout", json={"all_devices": True}, headers={"X-API-KEY": api_key_for(client, A)})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "interactive_session_required"
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()} == {session_id_of(A), session_id_of(B)}


def test_session_token_logout_ends_only_its_own_session(two_user_client):
    assert client.post("/api/v1/auth/logout", headers=bearer(A)).json()["message"] == "Successfully logged out"
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 401
    assert [x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()] == [session_id_of(B)]


def test_refreshed_token_can_still_log_out(two_user_client):
    new = client.post("/api/v1/auth/refresh", json={"refresh_token": A["refresh_token"]}).json()["access_token"]
    assert client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {new}"}).json()["message"] == "Successfully logged out"
    assert [x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()] == [session_id_of(B)]


def test_session_token_logout_all_devices_still_works(two_user_client):
    assert client.post("/api/v1/auth/logout", json={"all_devices": True}, headers=bearer(A)).json()["message"].startswith("Logged out from")
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401
```
The first assertion of each of the last three tests carries the body, so none is status-only.

- [ ] **Step 2: Update the existing logout tests.** `git grep -n -E "auth\.logout\(|/auth/logout|all_devices" -- tldw_Server_API/tests` finds them. Known:
  - `tests/AuthNZ/unit/test_auth_endpoints_extended.py:~915, ~1599, ~1653, ~1708, ~1754, ~1794, ~1902` call `auth.logout(...)` as a plain function with `SimpleNamespace(id=12)` and a stub session manager. `resolve_current_session_id` reads `principal.kind` and `principal.user_id`, so pass a real `AuthPrincipal(kind="user", user_id=12, token_type="access")` and a request whose bearer carries a `session_id` claim for the single-session cases (the tests that expect the old revoke-all fallback for a token without a session id now expect the 403), and for `all_devices=True` cases give the stub manager's `get_user_sessions` a row for that id.
  - `tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py:~410`, `test_auth_endpoints_integration.py`, `test_auth_comprehensive.py`, `tests/Sync/test_sync_v2_retention.py`, `tests/e2e/fixtures.py`: log out with the session token the login returned; a call that used an API key must change its credential or expect 403.

- [ ] **Step 3: Run, expect failure** (the first two tests get 200 and every session dies): `run_py t4-red tldw_Server_API/tests/AuthNZ_SQLite/test_logout_guard_sqlite.py`
- [ ] **Step 4: Implement.** **Step 5: Run, expect pass**, plus every file from Step 2.
- [ ] **Step 6: Commit** `fix(authnz): logout needs a browser session to end other sessions (TASK-<ID>)`.

---

### Task 5: The new-sign-in alert

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/sign_in_alert.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py:100-114` (`NotificationKind`: add `"new_sign_in"`)
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py:1083-1093` (`create_session`) and its success path after the cap block (`:1143-1190`)
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`, `create_session_record_with_cap` (`:118-234`): the eviction `RETURNING` clause of both backends (PostgreSQL `:189-190`, SQLite `:207-208`) gains `device_id`, and the docstring's key list (`:147-149`) too. Today the evicted dicts carry only `id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at, last_activity, created_at`. The extra key is harmless to `_blacklist_session_tokens`, which reads named keys. `test_session_cap_sqlite.py` and `test_session_cap_postgres.py` gain one assertion: the evicted dicts returned by the repo carry the evicted row's `device_id`.
- Modify (one keyword argument each): `tldw_Server_API/app/api/v1/endpoints/auth.py:416` (`_issue_multi_user_tokens`), `:1897` (`login`), `:3190` (`verify_magic_link`); `tldw_Server_API/app/core/AuthNZ/single_user_session.py:65` (`mint_single_user_session`)
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_alert_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_new_sign_in_alert_postgres.py`; extend `test_sessions_api_cookie.py`
- Modify: `tldw_Server_API/tests/conftest.py` (an autouse fixture `_sign_in_alerts_silent_by_default`) and `pyproject.toml:582` (register the marker `sign_in_alerts`; `--strict-markers` is on, `pyproject.toml:668`)

**Interfaces, in `sign_in_alert.py`:**
- Constants, verbatim from the spec: `NEW_SIGN_IN_KIND = "new_sign_in"`, `NEW_SIGN_IN_TITLE = "New sign-in to your account"`, `NEW_SIGN_IN_SEVERITY = "warning"`, `NEW_SIGN_IN_SOURCE_DOMAIN = "auth_session"`, `NEW_SIGN_IN_LINK_URL = "/settings/sessions"`, `NEW_SIGN_IN_DEDUPE_BUCKET_SECONDS = 600`, `NEW_SIGN_IN_OPEN_AFTER = timedelta(hours=24)`.
- `def new_sign_in_message(*, user_agent: str | None, ip_address: str | None, now: datetime) -> str`: `"{label} (as the browser reported it) signed in from {where} at {YYYY-MM-DD HH:MM} UTC. If this was not you, sign out your other sessions."`, with `label = device_label(user_agent)` and `where = ip_address[:64]`, or `"an unknown address"` when it is empty or `"unknown"` (the value `_auth_request_client_ip` returns, `auth.py:1003`). The IP is cut to 64 characters because it can come from a forwarded header. The label comes from a client-chosen User-Agent, so the wording does not present it as authoritative.
- `def new_sign_in_dedupe_key(*, user_id: int, device_id: str | None, ip_address: str | None, user_agent: str | None, now: datetime) -> str`: `"new_sign_in:" + sha256(f"{user_id}|{group}|{int(now.timestamp()) // 600}").hexdigest()[:32]`, where `group = install_id_or_none(device_id)` when that is not None, else `f"{ip_address}|{user_agent}"` (a bare single-user cookie tag is not an install id). It is a hash, never the install id: `dedupe_key` is returned by the notifications API and stream.
- `def write_new_sign_in_alert(db: CollectionsDatabase, *, session_id: int, ip_address: str | None, user_agent: str | None, device_id: str | None, user_id: int, now: datetime | None = None) -> UserNotificationRow`: one `db.create_user_notification(kind=..., title=..., message=..., severity=..., source_job_id=str(session_id), source_domain="auth_session", link_type="settings_route", link_url="/settings/sessions", dedupe_key=...)`. A dedupe hit returns the existing row (the behaviour of `create_user_notification`, `Collections_DB.py:7546-7620`).
- `SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS = (AuthnzDatabaseError, BackendDatabaseError, StoragePathValidationError, sqlite3.Error, OSError, KeyError, ValueError, TypeError, RuntimeError, TimeoutError, AttributeError)`. `AuthnzDatabaseError` is `core.exceptions.DatabaseError` (what `AuthnzSessionsRepo.get_active_sessions_for_user` raises, `sessions_repo.py:884-891`, re-exported through `core.AuthNZ.exceptions`); `BackendDatabaseError` is `core.DB_Management.backends.base.DatabaseError` (what `CollectionsDatabase` raises); `StoragePathValidationError` (`core/exceptions.py:1680`) is the base of `StorageUnavailableError` and `InvalidStorageUserIdError`, which `CollectionsDatabase.for_user` can raise. These are three unrelated classes, so naming only one lets the others escape. Never `BaseException` or `CancelledError`.
- `def schedule_new_sign_in_alert(*, user_id: int, session_id: int, ip_address, user_agent, device_id, db_pool: DatabasePool, evicted_device_ids: tuple[str | None, ...] = ()) -> None`: a no-op without a running loop; otherwise it creates a task, adds it to a module-level `set` of pending tasks and discards it on completion (a done-callback). `evicted_device_ids` holds one entry per session this sign-in evicted (empty when none; `None` entries are evicted sessions without an install id). The task reads `others = [r for r in await AuthnzSessionsRepo(db_pool).get_active_sessions_for_user(user_id) if r["id"] != session_id]` (live predicate shared with the cap; rows carry `device_id`) and then decides:
  - nothing to alert about when `not others and not evicted_device_ids` (a first sign-in);
  - **alert fatigue:** nothing to write when `new = install_id_or_none(device_id)` is not None and `new` equals `install_id_or_none(...)` of any row in `others` or of any entry of `evicted_device_ids` (the same install signing in again);
  - otherwise it runs `asyncio.to_thread` over `with CollectionsDatabase.for_user(user_id=user_id) as db: write_new_sign_in_alert(db, ...)`.

  The task body catches `SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS`, calls `log_counter("auth_sign_in_alert_failed")` and logs `logger.bind(error_type=type(exc).__name__).warning("New sign-in alert not written")` with no ids.
- `async def drain_pending_alerts(timeout: float = 5.0) -> None` (tests only): the tasks are created on the `TestClient` portal's event loop and mutated from that thread, while tests run on the pytest-asyncio loop, so `asyncio.gather` over them would raise "attached to a different loop". Poll instead: `deadline = time.monotonic() + timeout; while list(_PENDING) and time.monotonic() < deadline: await asyncio.sleep(0.02)`; raise `AssertionError` on timeout.
- `create_session(..., alert_new_sign_in: bool = False)`: after the cap block and before the Redis cache, the call sits in **its own `try`**, because `create_session`'s outer `except _SESSION_MANAGER_NONCRITICAL_EXCEPTIONS` (`session_manager.py:1209-1216`) would otherwise turn any error here into `SessionError` after the session row is already inserted:

  ```python
  if alert_new_sign_in:
      try:
          schedule_new_sign_in_alert(user_id=user_id, session_id=session_id, ip_address=ip_address, user_agent=user_agent,
                                     device_id=device_id, db_pool=db_pool,
                                     evicted_device_ids=tuple(e.get("device_id") for e in evicted))
      except SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS as exc:
          log_counter("auth_sign_in_alert_failed")
          logger.bind(error_type=type(exc).__name__).warning("New sign-in alert not scheduled")
  ```

  (`evicted` is the list `create_session_record_with_cap` returned, `session_manager.py:1144`; the call is in `session_manager.py` and imports `schedule_new_sign_in_alert`, `SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS` from `sign_in_alert`; `log_counter` is already imported there.) So an owner signed out by the cap still gets an alert on the next sign-in, including with `MAX_SESSIONS_PER_USER=1`, unless the evicted session had the same install id as the new one. The default False keeps `services/admin_e2e_support_service.py:695` silent.
- Test isolation, `tldw_Server_API/tests/conftest.py`: an autouse fixture `_sign_in_alerts_silent_by_default(request, monkeypatch)` that, unless `request.node.get_closest_marker("sign_in_alerts")` is set, does `monkeypatch.setattr(session_manager_module, "schedule_new_sign_in_alert", lambda **_: None)`. Every production sign-in passes the flag, so without it any test that signs one user in twice writes into the real `Databases/user_databases/` (ignored by `.gitignore`, so `git status` cannot show it). The alert test modules set `pytestmark = [pytest.mark.asyncio, pytest.mark.sign_in_alerts]`. Tests added to `test_sessions_api_cookie.py` (an existing module that must not be marked as a whole) carry `@pytest.mark.sign_in_alerts` on each test; the cookie and PostgreSQL HTTP tests below also `monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))`.

- [ ] **Step 1: Write the failing tests** in `test_new_sign_in_alert_sqlite.py` (fixture `two_user_client` from `conftest.py`; imports include `DatabaseError as AuthnzDatabaseError` from `tldw_Server_API.app.core.exceptions`, `DatabaseError as BackendDatabaseError` from `core.DB_Management.backends.base`, `StorageUnavailableError` from `core.exceptions`, `AuthnzSessionsRepo`; `notifications(client, tokens)` awaits `drain_pending_alerts()` then returns `GET /api/v1/notifications` `items`):

```python
async def test_first_sign_in_writes_nothing(two_user_client):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    assert await notifications(client, A) == []


async def test_second_sign_in_writes_one_warning_notification_with_the_spec_fields(two_user_client):
    A = login(client, "alice", pw, device="dev-aaaaaaaa", user_agent=SAFARI_IPHONE)
    B = login(client, "alice", pw, device="dev-bbbbbbbb", user_agent=CHROME_MAC)
    (n,) = await notifications(client, A)
    ip = next(r["ip_address"] for r in client.get("/api/v1/auth/sessions", headers=bearer(B)).json() if r["is_current"])
    assert (n["kind"], n["severity"], n["title"]) == ("new_sign_in", "warning", "New sign-in to your account")
    assert (n["source_domain"], n["source_job_id"], n["link_url"]) == ("auth_session", str(session_id_of(B)), "/settings/sessions")
    where = ip if ip not in (None, "", "unknown") else "an unknown address"
    assert n["message"].startswith(f"Chrome on macOS (as the browser reported it) signed in from {where} at ") and " UTC." in n["message"]
    assert n["read_at"] is None and n["dismissed_at"] is None


async def test_same_browser_inside_the_bucket_collapses_to_one_notification(two_user_client):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    for _ in range(3):
        login(client, "alice", pw, device="dev-bbbbbbbb")
    assert len(await notifications(client, A)) == 1
    login(client, "alice", pw, device="dev-cccccccc")
    assert len(await notifications(client, A)) == 2


async def test_alert_carries_no_credentials_or_install_id(two_user_client):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")
    raw = json.dumps(await notifications(client, A))
    for secret in (B["access_token"], B["refresh_token"], "dev-bbbbbbbb", pw):
        assert secret not in raw


async def test_alert_is_written_with_every_notification_preference_off(two_user_client):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    client.patch("/api/v1/notifications/preferences", json={"reminder_enabled": False, "job_completed_enabled": False, "job_failed_enabled": False}, headers=bearer(A))
    login(client, "alice", pw, device="dev-bbbbbbbb")
    assert len(await notifications(client, A)) == 1


@pytest.mark.parametrize("error", [BackendDatabaseError("db down"), StorageUnavailableError("no base dir")])
async def test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids(two_user_client, monkeypatch, error):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    def boom(**_):
        raise error
    monkeypatch.setattr(sign_in_alert.CollectionsDatabase, "for_user", staticmethod(boom))
    lines = []   # only this module's records, so an unrelated DB-layer debug line cannot match
    sink = logger.add(lines.append, level="DEBUG", filter=lambda r: r["name"] == sign_in_alert.__name__)
    try:
        B = login(client, "alice", pw, device="dev-bbbbbbbb")
        await drain_pending_alerts()
    finally:
        logger.remove(sink)
    text = "".join(map(str, lines))
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 200
    assert "New sign-in alert not written" in text
    for secret in (B["access_token"], B["refresh_token"], "dev-bbbbbbbb"):
        assert secret not in text


async def test_sessions_read_failure_does_not_fail_sign_in(two_user_client, monkeypatch):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    async def down(self, user_id):
        raise AuthnzDatabaseError("sessions unavailable")      # the class AuthnzSessionsRepo raises, not the backend one
    monkeypatch.setattr(AuthnzSessionsRepo, "get_active_sessions_for_user", down)
    counters = []
    monkeypatch.setattr(sign_in_alert, "log_counter", lambda name, *a, **k: counters.append(name))
    B = login(client, "alice", pw, device="dev-bbbbbbbb")       # login() asserts 200: the alert task is the only reader of that method here
    await drain_pending_alerts()
    assert "auth_sign_in_alert_failed" in counters
    assert await notifications(client, A) == []                 # nothing was written


async def test_scheduling_failure_does_not_fail_sign_in(two_user_client, monkeypatch):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    def boom(**_):
        raise RuntimeError("scheduler down")
    monkeypatch.setattr(session_manager_module, "schedule_new_sign_in_alert", boom)
    counters = []
    monkeypatch.setattr(session_manager_module, "log_counter", lambda name, *a, **k: counters.append(name))
    B = login(client, "alice", pw, device="dev-bbbbbbbb")       # login() asserts 200: no SessionError after the row was inserted
    assert session_id_of(B) in {r["id"] for r in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()}
    assert "auth_sign_in_alert_failed" in counters


async def test_alert_is_written_when_the_cap_evicted_the_only_other_session(two_user_client, monkeypatch):
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1"); reset_settings()
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")      # evicts A (another install); no other live session remains
    (n,) = await notifications(client, B)
    assert n["source_job_id"] == str(session_id_of(B))


async def test_evicting_the_same_install_writes_nothing(two_user_client, monkeypatch):
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1"); reset_settings()
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-aaaaaaaa")      # evicts A, same install signing in again
    assert await notifications(client, B) == []


async def test_signing_in_again_from_an_install_with_a_live_session_writes_nothing_new(two_user_client):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")      # one alert, for B
    C = login(client, "alice", pw, device="dev-aaaaaaaa")      # A's install again: another live session has that install id
    items = await notifications(client, A)
    assert [n["source_job_id"] for n in items] == [str(session_id_of(B))]


async def test_sign_ins_without_an_install_id_still_alert(two_user_client):
    A = login(client, "alice", pw)                              # no X-TLDW-Device-Id header: device_id is None
    B = login(client, "alice", pw)
    assert [n["source_job_id"] for n in await notifications(client, A)] == [str(session_id_of(B))]


def test_message_cuts_a_forwarded_ip_to_64_characters():
    text = new_sign_in_message(user_agent=CHROME_MAC, ip_address="1" * 100, now=datetime(2026, 11, 25, 14, 3, tzinfo=timezone.utc))
    assert ("1" * 64) in text and ("1" * 65) not in text and text.endswith("sign out your other sessions.")



async def test_alert_goes_only_to_the_signing_in_users_inbox(two_user_client):
    login(client, "alice", pw, device="dev-aaaaaaaa")
    login(client, "alice", pw, device="dev-bbbbbbbb")
    bob = login(client, "bob", pw)
    assert await notifications(client, bob) == []


def test_every_real_sign_in_call_site_requests_the_alert():
    """AST pin: all four production create_session calls pass alert_new_sign_in=True."""
    for rel, expected in (("api/v1/endpoints/auth.py", 3), ("core/AuthNZ/single_user_session.py", 1)):
        calls = [n for n in ast.walk(ast.parse((APP_ROOT / rel).read_text())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "create_session"]
        assert len(calls) == expected
        assert all(any(k.arg == "alert_new_sign_in" and getattr(k.value, "value", None) is True for k in c.keywords) for c in calls)
```
(`APP_ROOT` comes from `tldw_Server_API.tests.lint._ast_cache`.) Add two tests to `test_sessions_api_cookie.py`, each decorated `@pytest.mark.sign_in_alerts`: `test_single_user_cookie_mint_alerts_the_other_install` (the second mint, sending a different `X-TLDW-Device-Id`, writes one alert into the Collections DB of user `SINGLE_USER_FIXED_ID`; the first writes none) and `test_cookie_mints_without_an_install_header_still_alert` (two header-less mints store the bare cookie tag as `device_id`, which `install_id_or_none` treats as no install id, so the second mint writes one alert).

- [ ] **Step 2: PostgreSQL**, `test_new_sign_in_alert_postgres.py` (`isolated_test_environment`; build the Collections DB on PostgreSQL the way `Collections/test_collections_postgres_integration.py:29-35` does with `_pg_backend(db_name)` and `CollectionsDatabase.from_backend(user_id=..., backend=...)`):
  - `test_alert_write_dedupes_and_isolates_users_postgres`: `write_new_sign_in_alert` twice for user "1" with one device inside a bucket gives one row; once for user "2" gives a separate row; `db.list_user_notifications()` of each shows only its own.
  - `test_second_sign_in_writes_an_alert_over_http_postgres`: register `pgalice` (`@example.com`), log in twice with different install ids, `await drain_pending_alerts()`, with `USER_DB_BASE_DIR` set to a tmp dir (the Collections DB of the HTTP path is the per-user SQLite file), read `GET /api/v1/notifications`; exactly one `new_sign_in` item.

- [ ] **Step 3: Run, expect failure**: `run_py t4-red tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_alert_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_new_sign_in_alert_postgres.py tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_cookie.py`

- [ ] **Step 4: Implement**, then add the `alert_new_sign_in=True` keyword to each of the four calls; then add the autouse fixture and the marker; then the `device_id` in the eviction `RETURNING` clauses and the two repo-test assertions.

- [ ] **Step 5: Run, expect pass**; also `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`, `test_session_cap_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py`, `tldw_Server_API/tests/Notifications`, `tldw_Server_API/tests/lint/test_function_size_ratchet.py`. `git status --short` shows nothing new under `Databases/`.

- [ ] **Step 6: Commit** `feat(authnz): alert a user's other sessions to a new sign-in (TASK-<ID>)`.

---

### Task 6: Only an older browser session can silence a new-sign-in alert

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/sign_in_alert.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`: `NotificationsMarkReadResponse` gains `refused_ids: list[int] = Field(default_factory=list)`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notifications.py`: `list_notifications` (`:157`, new `kind` filter), `mark_read` (`:213`), `dismiss_notification` (`:229`), `snooze_notification` (`:244`)
- Modify: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`: next to `get_user_notification` (`:7813`) add `def list_user_notifications_of_kind(self, kind: str, ids: list[int]) -> list[UserNotificationRow]`, one `SELECT ... WHERE user_id = ? AND kind = ? AND id IN (...)` (no per-id loop; bulk mark-read can send hundreds of ids). No method of that name exists yet (`git grep list_user_notifications_of_kind` is empty on the cap branch); do not confuse it with the paged filter below. Also `list_user_notifications` (`:7642`) and `count_user_notifications` (`:7658`) gain `kind: str | None = None` (`AND kind = ?` when set, nothing otherwise).
- Modify: `tldw_Server_API/app/services/notifications_prune_service.py:30` (`DEFAULT_RETENTION_DAYS` gains `"new_sign_in": 30`). The env-override dict at `:146-151` is not touched (no new setting); `NotificationsPruneService.__init__` starts from `DEFAULT_RETENTION_DAYS`, so the default applies. Without an entry the alert would never expire by age.
- The frontend files for the mark-read response and the inbox are edited in Task 9, which consumes `refused_ids` and the `kind` filter.
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_silence_sqlite.py`; `tldw_Server_API/tests/AuthNZ/integration/test_new_sign_in_silence_postgres.py`; extend `test_sessions_api_cookie.py`; both modules carry `pytest.mark.sign_in_alerts`

**Interfaces, in `sign_in_alert.py`:**
- `NEW_SIGN_IN_PROTECTED_CODE = "new_sign_in_protected"`; `def new_sign_in_protected_http_exception() -> HTTPException`: 403, `detail={"code": "new_sign_in_protected", "message": "A new sign-in alert can only be dismissed from a browser that was signed in before it, or after 24 hours."}`.
- `def caller_may_silence(row: UserNotificationRow, caller_session_id: int | None, now: datetime | None = None) -> bool`: True for any kind other than `new_sign_in`; False when `caller_session_id is None` (API key, virtual key, service token: refused even after 24 hours); True when the row is at least `NEW_SIGN_IN_OPEN_AFTER` old; otherwise `caller_session_id < int(row.source_job_id)`. `row.created_at` is a `str` (`UserNotificationRow.created_at: str`), so the age is `now - created` with `created = CollectionsDatabase._parse_iso_utc(row.created_at)` (`Collections_DB.py:7874`, a staticmethod returning an aware UTC datetime or None); a missing or unparsable time gives `created is None`, which skips the age rule (the row is treated as protected, not as old). A non-numeric `source_job_id` is False. **Ordering assumption (Decision 8):** `sessions.id` is `INTEGER PRIMARY KEY AUTOINCREMENT` on SQLite (`core/DB_Management/authnz_session_schema.py:19-20`) and `SERIAL` on PostgreSQL (`pg_migrations_extra.py:1080`), so ids are monotonic and never reused and a lower id is an earlier session, and an id survives the purge of the session row. PostgreSQL sequences are not commit-ordered, so two sign-ins racing within a few milliseconds can invert by one id; that is accepted. The task also checks that the test DDL matches production: `tests/AuthNZ/conftest.py:755` and `:1545` also declare `id SERIAL PRIMARY KEY`; if a fixture DDL used a non-incrementing id, the PostgreSQL silence test would pass for the wrong reason.
- `def split_silenceable(rows: list[UserNotificationRow], caller_session_id: int | None, now: datetime | None = None) -> tuple[list[int], list[int]]`: pure; the allowed and refused ids among `rows` (the `new_sign_in` rows the endpoint fetched with `list_user_notifications_of_kind`).
- `GET /notifications` gains `kind: NotificationKind | None = Query(None)` (the literal type in `reminders_schemas.py:100`, so an unknown kind is a 422), passed to `db.list_user_notifications(..., kind=kind)` and `db.count_user_notifications(..., kind=kind)`; `total` and `pagination` then describe the filtered set. It is ignored when `only_snoozed` is true. The banner (Task 9) asks for `kind=new_sign_in` only, so a flood of other notifications cannot push the alert out of its page.
- Endpoints: each takes `request: Request` and `principal: AuthPrincipal = Depends(RequirePermission(...))` (replacing `_principal`). They fetch the target rows first with `db.list_user_notifications_of_kind("new_sign_in", ids)`; **only when that list is non-empty** do they resolve the caller with `caller = await interactive_session.resolve_current_session_id(request, principal)` with **no manager argument** (call it through the module so a test can patch the core module): the resolver gets the manager itself, which keeps a `get_session_manager()` call out of the endpoint body (the getter ratchet freezes `notifications.py` at 2 calls). A notifications app with no AuthNZ database, like `Notifications/test_notifications_api.py::notifications_app`, never touches it for other kinds. **Contract (one, used by Tasks 6 and 9):** `mark_read` marks the allowed ids and returns 200 `{"updated": n, "refused_ids": [...]}` when at least one id was allowed or no `new_sign_in` row was involved; when every requested id was refused it raises 403 `new_sign_in_protected`. `dismiss_notification` and `snooze_notification` raise 403 `new_sign_in_protected` for a refused row before doing anything else.
- `NotificationPreferencesResponse` is unchanged: no off switch exists for this kind.

- [ ] **Step 1: Write the failing tests** (`two_user_client` from `conftest.py`; A, B and C are three alice logins in that order with devices `dev-aaaaaaaa`, `dev-bbbbbbbb`, `dev-cccccccc`; `nB` and `nC` are the alert ids whose `source_job_id` is B's and C's session; `item(client, tokens, id)` reads one notification from `GET /api/v1/notifications`; `backdate(ctx, id, hours)` opens `CollectionsDatabase.for_user(user_id=ctx.alice_id)` and runs `db.backend.execute("UPDATE user_notifications SET created_at = ? WHERE id = ?", (iso, id))`):

```python
async def test_older_session_can_mark_read_and_dismiss(two_user_client):
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(A)).json() == {"dismissed": True}
    assert client.post("/api/v1/notifications/mark-read", json={"ids": [nC]}, headers=bearer(A)).json()["updated"] == 1


async def test_source_session_cannot_dismiss_its_own_alert(two_user_client):
    r = client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(B))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "new_sign_in_protected"
    assert item(client, A, nB)["dismissed_at"] is None


async def test_later_session_cannot_mark_read_or_snooze(two_user_client):
    r = client.post("/api/v1/notifications/mark-read", json={"ids": [nB]}, headers=bearer(C))
    assert r.status_code == 403 and r.json()["detail"]["code"] == "new_sign_in_protected"
    assert item(client, A, nB)["read_at"] is None
    s = client.post(f"/api/v1/notifications/{nB}/snooze", json={"minutes": 30}, headers=bearer(C))
    assert s.status_code == 403 and s.json()["detail"]["code"] == "new_sign_in_protected"


async def test_api_key_cannot_silence_even_an_old_alert(two_user_client):
    backdate(ctx, nB, hours=25)
    r = client.post(f"/api/v1/notifications/{nB}/dismiss", headers={"X-API-KEY": api_key_for(client, A)})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "new_sign_in_protected"
    assert item(client, A, nB)["dismissed_at"] is None


async def test_after_24_hours_any_browser_session_may_dismiss(two_user_client):
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(C)).status_code == 403   # new
    backdate(ctx, nB, hours=25)
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(C)).json() == {"dismissed": True}


async def test_other_kinds_are_unaffected(two_user_client):
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        other = db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")
    r = client.post(f"/api/v1/notifications/{other.id}/dismiss", headers={"X-API-KEY": api_key_for(client, A)})
    assert r.json() == {"dismissed": True}


async def test_user_cannot_dismiss_another_users_alert(two_user_client):
    bob = login(client, "bob", pw)
    assert client.post(f"/api/v1/notifications/{nB}/dismiss", headers=bearer(bob)).json() == {"dismissed": False}
    assert item(client, A, nB)["dismissed_at"] is None


async def test_preferences_expose_no_switch_for_the_alert(two_user_client):
    keys = set(client.get("/api/v1/notifications/preferences", headers=bearer(A)).json())
    assert keys == {"user_id", "reminder_enabled", "job_completed_enabled", "job_failed_enabled", "updated_at"}


async def test_mixed_mark_read_updates_the_allowed_ids_and_reports_the_refused(two_user_client):
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        other = db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")
    r = client.post("/api/v1/notifications/mark-read", json={"ids": [nB, other.id]}, headers=bearer(C))
    assert r.json() == {"updated": 1, "refused_ids": [nB]}
    assert item(client, A, nB)["read_at"] is None and item(client, A, other.id)["read_at"] is not None


async def test_list_filters_by_kind_and_reports_the_filtered_total(two_user_client):
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        for _ in range(3):
            db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")   # newer than the alerts
    body = client.get("/api/v1/notifications?kind=new_sign_in&limit=1", headers=bearer(A)).json()
    assert [i["kind"] for i in body["items"]] == ["new_sign_in"] and body["total"] == 2   # the alerts for B and C
    assert client.get("/api/v1/notifications?kind=not_a_kind", headers=bearer(A)).status_code == 422


def test_new_sign_in_alerts_expire_after_30_days():
    assert DEFAULT_RETENTION_DAYS["new_sign_in"] == 30


@pytest.mark.parametrize("created_at,caller,expected", [
    ("garbage", 9, False),                      # unparsable time: protected, not "old"
    ("", 9, False),
    ("2026-11-23T00:00:00+00:00", 9, True),     # more than 24 hours before NOW
    ("2026-11-25T11:00:00+00:00", 9, False),    # one hour old, caller newer than the source (id 7)
    ("2026-11-25T11:00:00+00:00", 3, True),     # one hour old, caller older than the source
    ("2026-11-25T11:00:00+00:00", None, False),
])
def test_caller_may_silence_rule(created_at, caller, expected):
    now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    row = UserNotificationRow(id=1, user_id="1", kind="new_sign_in", title="t", message="m", severity="warning",
                              source_task_id=None, source_task_run_id=None, source_job_id="7", source_domain="auth_session",
                              source_job_type=None, link_type=None, link_id=None, link_url=None, dedupe_key=None,
                              retention_until=None, archived_at=None, created_at=created_at, read_at=None, dismissed_at=None)
    assert caller_may_silence(row, caller, now) is expected


async def test_other_kinds_never_resolve_the_session_manager(two_user_client, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("caller resolved for a kind that needs no check")
    monkeypatch.setattr(interactive_session, "resolve_current_session_id", boom)   # core module: the endpoint calls it through the module, so a call would raise
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        other = db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")
    assert client.post(f"/api/v1/notifications/{other.id}/dismiss", headers=bearer(A)).json() == {"dismissed": True}
```
Add two cookie tests to `test_sessions_api_cookie.py`, each decorated `@pytest.mark.sign_in_alerts` (not a module marker): an older cookie session dismisses the alert (with the CSRF header) and gets `{"dismissed": true}`; `X-API-KEY` gets 403 `new_sign_in_protected`. `test_older_session_can_mark_read_and_dismiss` is the first to exercise `NOTIFICATIONS_CONTROL` through a real login, so a missing role grant in the Task 2 fixture shows up there as a 403.

- [ ] **Step 1b: PostgreSQL silence test** `test_new_sign_in_silence_postgres.py` (`isolated_test_environment`, `USER_DB_BASE_DIR` tmp, `AUTH_MODE=multi_user`, register `pgalice`, three logins A, B, C with distinct install ids): the source session B gets 403 `new_sign_in_protected` dismissing its own alert and the alert stays undismissed; the older session A dismisses it and gets `{"dismissed": true}`; an API key gets 403 `new_sign_in_protected`.

- [ ] **Step 2: Run, expect failure** (200 for every case today): `run_py t5-red tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_silence_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_cookie.py`

- [ ] **Step 3: Implement.** Leave `list_user_notifications` and the stream untouched: a banner needs the alert even from a session that cannot silence it.

- [ ] **Step 4: Run, expect pass**, plus `tldw_Server_API/tests/Notifications` (the existing tests touch only other kinds, which never resolve a session, so they pass unchanged; this is pinned by `test_other_kinds_never_resolve_the_session_manager`), `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py` (`notifications.py` must still count 2 getter calls) and the prune-service tests under `tldw_Server_API/tests` (`git grep -l notifications_prune -- tldw_Server_API/tests`).

- [ ] **Step 5: Commit** `feat(notifications): only an older browser session can silence a new-sign-in alert (TASK-<ID>)`.

---

### Task 7: Frontend services: session API, refusal codes and own-sign-in matching

**Files:**
- Modify: `apps/tldw-frontend/lib/api.ts:44-103` (a `PUBLIC_AUTH_REFUSAL_MESSAGES` table, `asPublicAuthRefusalCode`, and `normalizeApiErrorBody`)
- Create: `apps/packages/ui/src/services/auth-sessions.ts`
- Create: `apps/packages/ui/src/services/sign-in-alert.ts`
- Test (extend): `apps/tldw-frontend/lib/__tests__/api-client.fetch.test.ts`
- Test (create): `apps/packages/ui/src/services/__tests__/auth-sessions.test.ts`, `sign-in-alert.test.ts`

**Why `api.ts` is in this task:** the banner and the inbox call `dismissNotification` and `markNotificationsRead` from `@web/lib/api/notifications`, which go through `apiClient`. `normalizeApiErrorBody` (`lib/api.ts:78-108`) keeps an object `detail` only for provider `error_code` values and the VN generation rejection codes; every other 4xx object becomes `{}`, and `apiClient` then throws `ApiError(statusText, { status, detail: undefined, errorCode: undefined })` (`:665-681`). Without a change the server's 403 `{"detail":{"code":"new_sign_in_protected"}}` never reaches `isProtectedAlertError`, and a test that hand-builds an error object passes while the real banner shows a generic failure.

**Interfaces:**
- `lib/api.ts`: add a table `PUBLIC_AUTH_REFUSAL_MESSAGES = { new_sign_in_protected: "A new sign-in alert can only be dismissed from a browser that was signed in before it, or after 24 hours.", interactive_session_required: "This action needs a browser sign-in. API keys, virtual keys and service tokens cannot end sessions." } as const` (fixed client-owned text, like `PUBLIC_PROVIDER_ERROR_MESSAGES`), `asPublicAuthRefusalCode(value: unknown)` (own-property check, as `asPublicProviderErrorCode`), and widen `PublicErrorCode` to include the two keys. In `normalizeApiErrorBody`, inside the object-`detail` branch and only when `statusCode === 403`: `const refusal = asPublicAuthRefusalCode(errorBody.detail.code); if (refusal) return { detail: PUBLIC_AUTH_REFUSAL_MESSAGES[refusal], errorCode: refusal }`. The resulting `ApiError` then has `status: 403`, `errorCode` and the fixed message. Neither message contains "csrf", so the CSRF branch at `:652-664` does not catch them. `buildSafeErrorHistoryBody` records `{ detail: { error_code, message } }`, which is already safe.
- `auth-sessions.ts` (uses `bgRequest`, `appendPathQuery`, `toAllowedPath` the way `services/guardian.ts:302-316` does):
  - `type AuthSession = { id: number; ip_address: string | null; user_agent: string | null; device_label: string; created_at: string; last_activity: string; expires_at: string; is_current: boolean; same_install: boolean }`
  - `listAuthSessions(): Promise<AuthSession[]>` (`GET /api/v1/auth/sessions`)
  - `revokeAuthSession(id: number): Promise<void>` (`DELETE /api/v1/auth/sessions/{id}`)
  - `revokeOtherAuthSessions(): Promise<{ sessions_revoked: number }>` (`POST /api/v1/auth/sessions/revoke-all?keep_current=true`, reading `details.sessions_revoked`)
  - `INTERACTIVE_REQUIRED_CODE = "interactive_session_required"`; `errorCode(error: unknown): string | undefined` reads, in order, `error.errorCode` when it is a string (an `ApiError` from `apiClient`, after the `api.ts` change), `error.details.detail.code` (a `bgRequest` failure, see the shape below), then `error.detail.code` (as `noteSessionReplaced` does in `services/tldw/session-identity.ts:41-55`); `isInteractiveRequiredError(error): boolean`.
  - **`bgRequest` failure shape (verified in the cap tree, to be re-read):** `buildRequestError` (`services/background-proxy.ts:1049-1073`) returns a plain `Error` whose `status` is the HTTP status and whose `details` is `sanitizeResponseData(<parsed response body>)` (`:242-267`); `handleFailedResponse` (`:1100-`) feeds it `resp.data`. A 403 `{"detail": {"code": ..., "message": ...}}` therefore arrives as `error.status === 403` and `error.details.detail.code`. `sanitizeResponseData` redacts keys that contain words such as `session` or `private` (`isSensitiveKey`, `:236`); the keys `detail`, `code` and `message` survive, but the executor confirms that by reading `isSensitiveKey` before writing the test.
- `sign-in-alert.ts`:
  - `NEW_SIGN_IN_KIND = "new_sign_in"`, `NEW_SIGN_IN_PROTECTED_CODE = "new_sign_in_protected"`, `isProtectedAlertError(error): boolean` (true when `errorCode(error) === NEW_SIGN_IN_PROTECTED_CODE`, so it covers both the `apiClient` and the `bgRequest` shape).
  - `isOwnSignIn(sourceJobId: string | number | null | undefined, sessions: AuthSession[]): boolean`. False when no session in the list has `is_current` (API-key connection). True when the current session's id equals `sourceJobId` (compared as strings), or when the session whose id equals `sourceJobId` is in the list and has `same_install === true`. False when the source session is gone from the list. There is no IP and no device-label comparison anywhere: the server computes `same_install` from the install id (Task 3).
  - `visibleSignInAlerts(items: NotificationItem[], sessions: AuthSession[]): NotificationItem[]`: kind `new_sign_in`, `read_at` and `dismissed_at` empty, not `isOwnSignIn`, newest first.

- [ ] **Step 1: Write the failing tests.**
  - `sign-in-alert.test.ts`, `isOwnSignIn`: id equality true; a different id whose session row has `same_install: true` true; a different id whose row has `same_install: false` false, even when the two rows carry the same `ip_address` and `device_label`; source session absent false; no `is_current` row false (API-key connection) even if `same_install` is true on a row; string `"7"` against numeric id 7 true. `visibleSignInAlerts`: drops read, dismissed, other-kind and own alerts, and sorts newest first. `isProtectedAlertError`: true for an `ApiError`-shaped `{ status: 403, errorCode: "new_sign_in_protected" }`, true for a `bgRequest`-shaped `{ status: 403, details: { detail: { code: "new_sign_in_protected" } } }`, false for `interactive_session_required`, false for a plain `Error`.
  - `auth-sessions.test.ts`: with `bgRequest` mocked through `vi.mock("@/services/background-proxy")`, assert the exact `path` and `method` of the three calls (including `?keep_current=true`); `errorCode` for the three error shapes (`errorCode` property, `details.detail.code`, `detail.code`). One further test drives the **real** `bgRequest` (direct path, no `browser.runtime`) against a stubbed `fetch` that answers 403 with `{"detail":{"code":"interactive_session_required","message":"x"}}` and asserts `isInteractiveRequiredError(caught) === true`. If driving the real function proves impractical after one attempt in this test environment, copy the shape from an existing test that asserts `details` on a failed `bgRequest` (`git grep -n "details" -- 'apps/packages/ui/src/services/__tests__/background-proxy*'`), and say so in the PR.
  - `api-client.fetch.test.ts` (the pattern of the `credential_store_unavailable` test, `:360-390`; `vi.stubGlobal("fetch", ...)` and `loadApiModule()`): the real `apiClient.post("/notifications/1/dismiss")` against a mocked `fetch` answering `jsonResponse({ detail: { code: "new_sign_in_protected", message: "raw server text" } }, { status: 403, statusText: "Forbidden" })` rejects with `toMatchObject({ name: "ApiError", status: 403, errorCode: "new_sign_in_protected", detail: <the fixed message>, message: <the fixed message> })`, and `JSON.stringify(storedRequestHistory()[0])` does not contain `raw server text`. A second case does the same for `interactive_session_required`. A third case: a 403 with `{ detail: { code: "something_else", message: "m" } }` still rejects with `errorCode` undefined and the status text (the allowlist is closed). A fourth: a 500 with `code: "new_sign_in_protected"` is not exposed.

- [ ] **Step 2: Run, expect failure**: `cd apps/packages/ui && bunx vitest run src/services/__tests__/auth-sessions.test.ts src/services/__tests__/sign-in-alert.test.ts --maxWorkers=2`, then `cd apps/tldw-frontend && bunx vitest run lib/__tests__/api-client.fetch.test.ts --maxWorkers=2`.

- [ ] **Step 3: Implement.** **Step 4: Run, expect pass.** **Step 5: Commit** `feat(ui): session API client, refusal codes and own-sign-in matching (TASK-<ID>)`.

---

### Task 8: The Active sessions page, route, nav entry and account-menu link

**Files:**
- Create: `apps/packages/ui/src/components/Option/Settings/ActiveSessionsSettings.tsx` (exports `ActiveSessionsSettings`)
- Create: `apps/tldw-frontend/pages/settings/sessions.tsx` (copy `pages/settings/provider-keys.tsx`: `dynamic` + `SettingsRoute` + `withPageTitle("Active sessions", ...)`)
- Modify: `apps/packages/ui/src/routes/option-settings-route-registry.tsx` (a `createSettingsRoute(() => import("~/components/Option/Settings/ActiveSessionsSettings"), "ActiveSessionsSettings")` constant next to `OptionProviderKeysSettings`, and `{ kind: "options", path: "/settings/sessions", element: <OptionActiveSessionsSettings /> }`)
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts` (an entry after `/settings/provider-keys`: `group: "setupRecovery"`, `labelToken: "settings:activeSessions.navTitle"`, `icon: MonitorSmartphone`, `order: 3.5`, chosen because `/settings/model` already holds `order: 4` at `settings-nav-config.ts:~87`; re-read the `setupRecovery` group and take any free order)
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx` (an entry like the provider-keys one at `:263`, with inline `nav: { group: "server", labelToken: "settings:activeSessions.navTitle", icon: ..., order: 1.6 }`)
- Modify: `apps/packages/ui/src/components/Layouts/AccountMenu.tsx` (a `Link to="/settings/sessions"` row between "Account settings" and "Settings", label `t("header.activeSessions", "Active sessions")`)
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json` (an `activeSessions` object) and `assets/locale/en/option.json` (`header.activeSessions`)
- Test (create): `apps/packages/ui/src/components/Option/Settings/__tests__/ActiveSessionsSettings.test.tsx`, `apps/packages/ui/src/routes/__tests__/option-settings-sessions-route.test.ts`; extend `apps/packages/ui/src/components/Layouts/__tests__/AccountMenu.test.tsx`
- Test (create): `apps/tldw-frontend/__tests__/pages/settings-sessions-route.test.tsx`, following `__tests__/pages/settings-provider-keys-route.test.tsx` (the page renders `SettingsRoute` with the component, and the title is "Active sessions"); also add the page to `__tests__/pages/core-route-titles.test.tsx` if that file lists settings pages
- Before editing: `git grep -n "provider-keys" -- apps` and replicate every registration hit the new page needs (the settings registry, the nav config, the extension registry and the Next page are the known ones).

**Page sketch** (reused: antd `Table` and `Modal.confirm` as in `ProviderKeysSettings.tsx:120`, `Alert` and `Badge` from `@/components/ui/primitives`, `EmptyState` and `LoadingState` from `@/components/ui/feedback`):

```text
Active sessions                                         <- h1, tabIndex=-1, receives focus after an action
These are the browsers and devices signed in to your account. Sign out any you do not recognise.
                                              [ Sign out all other sessions ]
+------------------------------+--------------+--------------------+--------------------+-----------+
| Device                       | IP address   | Signed in          | Last active        |           |
+------------------------------+--------------+--------------------+--------------------+-----------+
| Chrome on macOS [This device]| 203.0.113.7  | 25 Nov 2026, 14:03 | 25 Nov 2026, 14:41 |           |
| Safari on iOS                | 198.51.100.9 | 24 Nov 2026, 09:12 | 24 Nov 2026, 18:30 | [Sign out]|
+------------------------------+--------------+--------------------+--------------------+-----------+
(status line, role="status": "Signed out Safari on iOS. 1 other session remains.")
```

**Exact copy** (`settings:activeSessions.*`, English; defaults in code as `ProviderKeysSettings` does):
- `navTitle` "Active sessions"; `title` "Active sessions"; `intro` "These are the browsers and devices signed in to your account. Sign out any you do not recognise."
- Columns `colDevice` "Device", `colIp` "IP address", `colSignedIn` "Signed in", `colLastActive` "Last active"; badge `thisDevice` "This device"; row action `signOut` "Sign out" with `aria-label` "Sign out {device}, signed in {time}".
- `signOutOthers` "Sign out all other sessions"; confirm title "Sign out all other sessions?"; content "{count, plural, one {# other session} other {# other sessions}} will be signed out. This device stays signed in."; ok "Sign out other sessions"; cancel "Cancel" (`common:cancel`).
- Status: `signedOut` "Signed out {device}."; `signedOutOthers` "{count, plural, one {Signed out # other session.} other {Signed out # other sessions.}}"; `remaining` "{count, plural, one {# other session remains.} other {# other sessions remain.}}".
- Loading: `LoadingState` label "Loading your sessions...". Empty: `EmptyState` title "No sessions to show", description "Sessions appear here for browsers signed in to this account." Error: `Alert` variant `error`, title "Could not load your sessions", action "Try again".
- Single-user mode (the connection config's `authMode` is `"single-user"`; reuse the way neighbouring settings components read it: `git grep -n "authMode" -- apps/packages/ui/src/components/Option/Settings`): a `Text` line under the intro, `singleUserNote` "If you did not sign in, rotate your API key." In single-user mode anyone holding the API key can mint a browser session (`POST /auth/single-user/session`), so ending sessions does not shut that holder out; rotating the key does. The line is absent in multi-user mode.
- API-key connection (the list loads and no row is `is_current`): `Alert` variant `info`, title "Sessions are read-only on this connection", body "This connection uses an API key. Sign in with the browser to sign sessions out." Every sign-out control is disabled.
- Server refusal `interactive_session_required`: `Alert` variant `warning`, "Sign in from a browser to sign sessions out."; any other failure: `Alert` variant `error`, "Could not sign out that session. Try again."
- The page has no lock control and no lock explanation: lock is PR 3, which adds its control beside "Sign out all other sessions" and its availability line below the intro.

**Behaviour and accessibility:** the current row has no Sign out button (the account menu signs this browser out). "Sign out all other sessions" is disabled when there are no other sessions. After a successful action the list reloads, the status line announces the result, and focus moves to the `h1` so a keyboard user is not dropped onto `body` when the activated row disappears. The `Modal.confirm` traps focus and returns it to the trigger; Escape cancels. Times use `Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" })`, with the full ISO time in the cell `title`. On a narrow viewport the table scrolls inside its container, not the page.

- [ ] **Step 1: Write the failing tests.** `ActiveSessionsSettings.test.tsx` renders with real English resources and ICU (the pattern of `ProviderKeysSettings.i18n.test.tsx:1-30`) and mocks `@/services/auth-sessions`:
  - `renders the loading state, then one row per session with This device on the current row only`
  - `the current row has no Sign out button; other rows have one labelled with device and time`
  - `Sign out calls revokeAuthSession with that id, reloads, announces "Signed out Safari on iOS." in the status region and moves focus to the heading`
  - `Sign out all other sessions asks for confirmation with the count, then calls revokeOtherAuthSessions; Cancel calls nothing`
  - `Sign out all other sessions is disabled when only this device is listed`
  - `shows the read-only notice and disables sign-out when no row is current`
  - `shows the browser-sign-in warning when the server answers interactive_session_required`
  - `shows the error alert with Try again, and Try again reloads`
  - `shows the empty state for an empty list`
  - `shows "If you did not sign in, rotate your API key." in single-user mode and not in multi-user mode`

  `option-settings-sessions-route.test.ts` is the source-reading pattern of `option-settings-provider-keys-route.test.ts`: the settings registry, `settings-nav-config.ts` and the extension registry each contain `/settings/sessions` and `settings:activeSessions.navTitle`. The `AccountMenu` test asserts a link named "Active sessions" with `href="/settings/sessions"`.

- [ ] **Step 2: Run, expect failure.** **Step 3: Implement.**

- [ ] **Step 4: Run, expect pass**: `cd apps/packages/ui && bunx vitest run src/components/Option/Settings/__tests__/ActiveSessionsSettings.test.tsx src/routes/__tests__/option-settings-sessions-route.test.ts src/components/Layouts/__tests__/AccountMenu.test.tsx src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/routes/__tests__/option-settings-route-split.test.tsx --maxWorkers=2`. If a locale-parity test fails for the new keys (`git grep -l "_locales" -- 'apps/packages/ui/src/**/*.test.ts*'`), mirror them into `public/_locales/en/settings.json` in its message format; other locales fall back to English.

- [ ] **Step 5: Type, lint, design system**: `cd apps/tldw-frontend && bunx tsc --noEmit -p tsconfig.strict.json 2>&1 | tail` compared with `dev`; `bun run lint:ui` shows no new findings; `node apps/packages/ui/scripts/verify-design-system-product-state.mjs` passes.

- [ ] **Step 6: Commit** `feat(ui): Active sessions settings page (TASK-<ID>)`.

---

### Task 9: The banner, the toast skip, the inbox and the extension's OS notification

**Files:**
- Create: `apps/tldw-frontend/components/notifications/NewSignInBanner.tsx`
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx:917-924` (`NotificationRuntimeOwner`: render `<NewSignInBanner />` after `<NotificationToastBridge />`, before `{children}`; the banner itself is `position: fixed`, see Placement)
- Modify: `apps/tldw-frontend/components/notifications/NotificationToastBridge.tsx` (the `notification` branch of the events effect, `:~92-106`): skip kind `new_sign_in`
- Modify: `apps/tldw-frontend/components/notifications/NotificationsRoute.tsx`: `handleMarkRead` (`:386-412`), `handleDismiss` (`:414-439`), the "View" button (`:~712-735`) and the Snooze button (`:~745-754`), see Behaviour
- Modify: `apps/packages/ui/src/services/notifications.ts` (`markNotificationsRead`, `:316`: returns `{ updated: number; refused_ids?: number[] }`; `listNotifications`, `:294`: the params type gains `kind?: string`, which the generic `buildNotificationsQuery` already serialises) and `apps/tldw-frontend/lib/api/notifications.ts` (`markNotificationsRead`, `:151`: same return type; `listNotifications`, `:122`: params gain `kind?: string` and pass `kind: params?.kind` into `buildNotificationsQueryShared`)
- Modify: `apps/packages/ui/src/entries/shared/notification-subscription.ts` (`handleStreamEvent`, the `notification` branch that calls `notify(...)`)
- Test (create): `apps/tldw-frontend/__tests__/components/new-sign-in-banner.test.tsx`; extend `apps/tldw-frontend/__tests__/components/notification-toast-bridge.test.tsx`, the inbox test for `NotificationsRoute` (`git grep -l NotificationsRoute -- apps/tldw-frontend/__tests__`; `__tests__/pages/notifications.test.tsx` mocks `reportMutationError`) and `apps/packages/ui/src/entries/shared/__tests__/notification-subscription.test.ts`

**Check before writing code (read, do not assume):**
- `apps/packages/ui/src/components/ui/primitives/Alert.tsx`: the props `variant`, `title`, `action` (`{ label, onClick, ... }`), `secondaryAction`, `role` and `dismissible`; confirm `variant="warning"` renders `role="alert"` by default and that `secondaryAction` renders a clickable control with its label. Adjust the banner's props to what the component really offers.
- `NotificationsRoute.tsx` View button (`:~712-735`): its `onClick` awaits `handleMarkRead` inside `try { ... } catch {}` and then navigates. Confirm that after the change below `handleMarkRead` returns normally on a refusal, so navigation still runs, and that the button is `disabled` only by `canRequest || isRetryingMutation`.
- `WebLayout.tsx`: `NotificationRuntimeOwner` renders inside the pages router. The banner reads the route with `useRouter` from `next/router` (as `NotificationsRoute.tsx:2` does) and tests mock `next/router` the way the inbox test does; confirm `router.pathname` is `"/settings/sessions"` on the sessions page and that `router.push` is what navigation uses.
- `useOptionalNotificationLifecycle()` (`NotificationLifecycleProvider.tsx:134`) returns `connectionVerified`, `state`, `events`, `lifecycleEpoch`; the toast bridge already reads `events` and `lifecycleEpoch`.

**Banner sketch** (the design's chosen approach A: the banner links to the page; the destructive buttons live on the page):

```text
+--------------------------------------------------------------------------------------+
| (!) New sign-in to your account                                                      |
|     Chrome on macOS signed in from 203.0.113.7 at 2026-11-25 14:03 UTC. If this was |
|     not you, sign out your other sessions.                                          |
|     1 more new sign-in.                       (only when several are unread)        |
|     [ Review sessions ]   Dismiss                                                    |
+--------------------------------------------------------------------------------------+
```
Built from the `Alert` primitive with `variant="warning"` (it renders `role="alert"`), `action={{ label: "Review sessions", ... }}` and `secondaryAction={{ label: "Dismiss", ... }}`. Plural line: "1 more new sign-in." and "{n} more new sign-ins." After a refused dismiss the body is replaced by the explanation and the secondary action becomes "Hide on this browser".

**Placement and gating:** `NotificationRuntimeOwner` (`WebLayout.tsx:~911-925`) wraps the whole app shell, so a block-level `Alert` ahead of `{children}` would sit above a full-viewport shell and break the layout. The banner is therefore `fixed inset-x-0 top-0 z-50 mx-auto max-w-3xl px-4 pt-2` (no layout shift; it overlays the header edge). It renders and fetches only when `useOptionalNotificationLifecycle()` returns a value with `connectionVerified === true` and `state === "active"`. The provider's `enabled` prop is false when the header is hidden (for example `/login`) or in demo mode, and then the lifecycle never leaves `connecting`, so no unauthenticated call is made there.

**Behaviour:**
- **Fetch:** `listNotifications({ kind: "new_sign_in", limit: 100, offset: 0, include_archived: false })` from `@web/lib/api/notifications` (the server filters on `kind`, Task 6, so other notifications cannot push the alert out of the page) and `listAuthSessions()`, then `visibleSignInAlerts(items, sessions)`. It shows the newest and adds the "more" line for the rest. When the session list cannot be loaded (API-key connection, error) the alert stays visible: a failure never hides a security alert.
- **Reload triggers:** on mount; on every stream event whose `event.event === "notification"` and payload `kind` is `new_sign_in`; on `notifications_coalesced` (the stream sends one such event instead of the individual notifications when more than 50 rows arrive in one poll, `notifications.py:~420-433`); on `reset_required`; and when `lifecycleEpoch` changes. A flooder therefore cannot bury the alert.
- It is hidden while the route is `/settings/sessions`.
- "Review sessions" navigates to `/settings/sessions` and does not mark the alert read.
- **"Dismiss"** calls `dismissNotification(id)`. On success the banner removes the alert. On `isProtectedAlertError(error)` it keeps the banner, replaces the body with "This alert can be dismissed from a browser that was signed in before it, or after 24 hours." and offers **"Hide on this browser"** (a legitimate user who cleared cookies and signed in again holds a higher session id than their other devices, so the server refuses their dismiss for 24 hours). "Hide on this browser" stores `localStorage["tldw:new-sign-in-hidden:" + notificationId] = "1"` inside `try/catch` (storage can be absent or throw), makes no server call, and the banner skips every alert whose key is set (the read is also in `try/catch`; when storage is unavailable the hide holds for the page's lifetime in component state only). Any other failure shows "Could not dismiss the alert. Try again."
- **The banner never calls `reportMutationError` or `recordFailedMutation`** (the banner does not have them, and the inbox handlers skip them for a refusal, below): `classifyNotificationError` maps a non-401 4xx to `unavailable` (`packages/ui/src/services/notification-lifecycle.ts:83-106`), `applyFailure` then stops the stream (`NotificationLifecycleProvider.tsx:234-261`), and the banner renders only while `state === "active"`, so one refused dismiss would otherwise shut down the notification runtime and hide the very alert it was about.
- **Inbox (`NotificationsRoute.tsx`):** `handleMarkRead` marks the item read locally and decrements the unread count only for ids not in the response's `refused_ids`. When the call throws and `isProtectedAlertError(error)` (the 403 where every id was refused), it shows the explanatory text as a toast (`variant: 'warning'`), changes nothing locally, and does **not** call `reportMutationError` or `recordFailedMutation`; it returns normally, so the View button still navigates to `/settings/sessions`. `handleDismiss` does the same on a 403. Other errors keep today's path. The Snooze button is not rendered for a `new_sign_in` item.
- An owner signed out by the cap sees the alert on the next sign-in: it is stored unread and the first load of the new session fetches it.
- **Extension:** `handleStreamEvent` skips `notify(...)` for a `new_sign_in` payload when `isOwnSignIn(payload.source_job_id, await listAuthSessions())` is true (a same-install session); a failed lookup raises the OS notification. The extension has no in-page banner: it gets the page (shared registry) and the suppression.

- [ ] **Step 1: Write the failing tests.** Banner test (mock `@web/lib/api/notifications`, `@/services/auth-sessions` and `next/router`; wrap in `NotificationLifecycleProvider` as `notification-toast-bridge.test.tsx:1-60` does; `listNotifications` is a `vi.fn` whose calls are asserted):
  - `shows a role=alert banner for an unread new_sign_in from another session`
  - `asks the server for kind=new_sign_in only` (the mock was called with `{ kind: "new_sign_in", ... }`)
  - `does not show it when the source session is the current session`
  - `does not show it when the source session has same_install true` (the mocked inbox list still contains it)
  - `shows it when same_install is false even though the source and current sessions share IP and device label`
  - `does not show read, dismissed or other-kind notifications`
  - `a streamed new_sign_in notification triggers a reload and shows the banner`
  - `a notifications_coalesced event reloads the list and shows an alert that was not in the stream`
  - `a reset_required event reloads the list`
  - `a lifecycle epoch change reloads the list`
  - `still shows the alert when the session list cannot be loaded`
  - `Review sessions navigates to /settings/sessions and does not mark read`
  - `Dismiss calls dismissNotification and removes the banner`
  - `Dismiss answered with new_sign_in_protected keeps the banner, shows the explanation and offers Hide on this browser`
  - `Hide on this browser removes the banner, writes the localStorage key and makes no server call; the banner stays hidden after a remount`
  - `Hide on this browser still hides for the page's lifetime when localStorage throws`
  - `lifecycle state stays active after a refused dismiss` (wrap in the real `NotificationLifecycleProvider`, mock `dismissNotification` to reject with `{ status: 403, errorCode: "new_sign_in_protected" }`, and assert through `useOptionalNotificationLifecycle()` that `state === "active"` and `mutationError === null`, and that the banner is still rendered)
  - `is not rendered on /settings/sessions`
  - `is not rendered, and fetches nothing, when the lifecycle is not active (disabled provider, connecting, auth-required)`
  - inbox, in the `NotificationsRoute` test: `a refused_ids response leaves the alert unread and the unread count unchanged`; `a 403 new_sign_in_protected shows the explanation toast, leaves the item unread, does not call reportMutationError, and View still navigates`; `Dismiss answered 403 new_sign_in_protected does not call reportMutationError`; `an ordinary mark-read failure still calls reportMutationError`; `no Snooze button on a new_sign_in item`
  - `two unread alerts show the newest and "1 more new sign-in."`

  Toast bridge: a `notification` stream event with `kind: "new_sign_in"` queues no toast; a `job_completed` one still does. Subscription: a `new_sign_in` event whose source session has `same_install: true` or is the current session raises no `notify`; for another session it does; a failing `listAuthSessions` raises it.

- [ ] **Step 2: Run, expect failure**: `cd apps/tldw-frontend && bunx vitest run __tests__/components/new-sign-in-banner.test.tsx __tests__/components/notification-toast-bridge.test.tsx __tests__/pages/notifications.test.tsx --maxWorkers=2`, then `cd apps/packages/ui && bunx vitest run src/entries/shared/__tests__/notification-subscription.test.ts --maxWorkers=2`.

- [ ] **Step 3: Implement.** **Step 4: Run, expect pass**, plus the existing `notification-rotation.integration.test.tsx`, `Layout.command-palette.integration.test.tsx` and `OptionIndex.loader-focus.integration.test.tsx` (the last two live under `apps/tldw-frontend/__tests__/components/layout/`), which mount `WebLayout`.

- [ ] **Step 5: Commit** `feat(ui): new-sign-in banner, quieter toast and extension notification (TASK-<ID>)`.

---

### Task 10: Docs, OpenAPI, full verification and PR

**Files:**
- Modify: `Docs/API-related/AuthNZ-API-Guide.md` (sessions section, `:227-280`) and its mirror `Docs/Published/API-related/AuthNZ-API-Guide.md`
- Modify: `apps/packages/api-types/openapi.fingerprint.json` (regenerated; `paths.d.ts` does not change because no route is added, but pass both flags to the check)
- Modify: `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`: append a "Changes from the PR 2 plan (2026-11-25)" section recording the proxy-IP finding, banner suppression by install id (`same_install`) instead of IP and device label, the `kind` filter on `GET /notifications`, the deferred lock flag, the `logout` guard, the alert on eviction, the live-only revoke-all count and the existing WebUI toast (and mirror it with `refresh_docs_published.sh` if the design is mirrored)

- [ ] **Step 1: Docs.** In the sessions section document: `is_current`, `device_label` and `same_install` on the listing (`same_install` is true when a session was signed in from the same browser install as the caller's; the install id is never returned); `?keep_current=true`; the 403 `interactive_session_required` rule for revoke-one, revoke-all and `logout` (with `all_devices: true`, or with a token that has no `session_id`) (a browser session means an access token carrying `session_id`, or a single-user cookie; API keys, virtual keys, service tokens and impersonation tokens are refused); the `new_sign_in` notification (title, severity, link, dedupe bucket) and who may mark it read, dismiss or snooze it (`new_sign_in_protected`, the older-session rule, the 24-hour rule, `refused_ids` on mark-read); the `kind` query filter on `GET /notifications` and the 30-day retention of `new_sign_in`; the IP caveat of Question 1 with `AUTH_TRUST_X_FORWARDED_FOR` and `AUTH_TRUSTED_PROXY_IPS`; and the single-user caveat: there the protection is the alert plus the session cap, not isolation from a holder of the API key (`POST /auth/single-user/session` mints a cookie session for anyone with the key), and the remedy is to rotate the API key. Then `Helper_Scripts/refresh_docs_published.sh` and stage `Docs/Published`.

- [ ] **Step 2: OpenAPI.** `make openapi-fingerprint` (Makefile `:536`); then `python Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` exits 0.

- [ ] **Step 3: Targeted backend run.** `run_py t9-a` over every test file this PR created, `tldw_Server_API/tests/AuthNZ_SQLite`, `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`, `.../test_session_cap_postgres.py`, `.../test_authnz_sessions_repo_postgres.py`, `tldw_Server_API/tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py`, `.../test_sessions_api_postgres.py`, `.../test_new_sign_in_alert_postgres.py`, `tldw_Server_API/tests/Notifications`, `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`, and every file from `git grep -l -E "sessions/revoke-all|auth/sessions|users/sessions|create_session\(|NotificationKind|mark-read" -- tldw_Server_API/tests`. PostgreSQL tests must run, not skip. `git status --short` shows nothing new under `Databases/`.

- [ ] **Step 4: Repository gates.** `run_py t9-ci tldw_Server_API/tests/CI` and `run_py t9-lint tldw_Server_API/tests/lint`; then `python Helper_Scripts/ci/test_quality_triage.py --enforce` and `ruff check` on new files (for modified files compare with `git show origin/dev:<f> | ruff check --stdin-filename <f> -`). A failure is compared against the same test at `origin/dev` (a throwaway worktree, removed afterwards) before it is called pre-existing.

- [ ] **Step 5: Frontend suites**, one process at a time: `cd apps/packages/ui && bunx vitest run src/services src/components/Option/Settings src/components/Layouts src/routes src/entries --maxWorkers=2`; `cd apps/tldw-frontend && bunx vitest run __tests__ lib/__tests__ --maxWorkers=2`; `bunx vitest run -c vitest.extension.config.ts __tests__/extension --maxWorkers=2`. Failures are compared with `dev` the same way.

- [ ] **Step 6: Close out the Backlog task.** `task edit TASK-<ID> -s Done --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-dod 1 ... --check-dod 6 --append-notes "<tests run, Postgres skips, deviations, known limits>" --final-summary "<what changed>"`, then `git add -A backlog/tasks` and commit with the trailer.

- [ ] **Step 7: Open the PR.** `git push -u origin feat/sign-in-alert-sessions-page`, then `gh pr create --base dev --title "feat(authnz): new-sign-in alert and Active sessions page (TASK-<ID>)" --body-file <body.md>`. The body lists the summary, the `logout all_devices` behaviour change, the test evidence (Postgres run or skip), the Deviations and Known limits below, and ends with the Claude Code line from the attribution reminder. Then run the clean-subagent review, wait for green CI, diff the non-required Frontend E2E and admin-ui real-backend results against `dev` (the banner can overlay journeys that hold two cookie sessions, for example `e2e/single-user-cookie-lifecycle.spec.ts`), and merge directly when CLEAN.

---

## Self-review

- **Spec coverage:**
  - alert when another live session exists, best-effort, off the request path: Task 5
  - kind, severity, title, message, link, `source_domain`, `source_job_id`, no schema change: Task 5
  - banner on load and on stream arrival: Task 9
  - not alerting itself, inbox entry stays: Tasks 7 and 9
  - not silenceable by a newer session, 24-hour rule, no preference switch: Task 6
  - storm guard with the 10-minute bucket: Task 5
  - client IP: Question 1 and Deviation 2
  - own-browser banner suppression: Tasks 3, 7 and 9 (by `same_install`, Deviation 7)
  - sessions API (`is_current`, `device_label`, `keep_current`, id read from the bearer token): Tasks 2 and 3 (live-only revoke count included)
  - interactive credential required (revoke-one, revoke-all, dismissing the alert; lock is PR 3): Tasks 2, 3, 4 (`logout`) and 6
  - Active sessions page, account-menu and notification links: Tasks 8 and 9
  - "Is lock available?" flag: deferred to PR 3 (Deviation 1)
- **Security requirement to test:**
  - revoke and dismiss need an interactive session: `test_api_key_cannot_end_sessions`, `test_scoped_virtual_key_cannot_end_sessions`, `test_api_key_as_bearer_cannot_log_everyone_out`, `test_api_key_header_cannot_log_everyone_out_with_all_devices`, `test_service_api_key_and_impersonation_principals_are_not_interactive`, `test_api_key_cannot_silence_even_an_old_alert`, `test_new_sign_in_silence_postgres`
  - a client cannot claim "current": `test_client_cannot_claim_current_session`, `test_claim_naming_another_users_session_is_not_current`
  - cross-user isolation on both backends: `test_one_user_cannot_see_or_end_another_users_sessions` and `test_sessions_api_postgres`, `test_user_cannot_dismiss_another_users_alert`, `test_alert_goes_only_to_the_signing_in_users_inbox`, `test_alert_write_dedupes_and_isolates_users_postgres`
  - no credentials in the alert: `test_alert_carries_no_credentials_or_install_id`; no install id in the sessions list: `test_same_install_marks_only_sessions_of_the_callers_install`
  - the refusal codes survive the real HTTP client: the `api-client.fetch.test.ts` cases of Task 7
  - a refused dismiss does not stop the notification runtime: `lifecycle state stays active after a refused dismiss` (Task 9)
  - the alert write cannot fail or leak out of sign-in: `test_sessions_read_failure_does_not_fail_sign_in`, `test_scheduling_failure_does_not_fail_sign_in`
  - nothing new logs tokens, cookies or install ids: `test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids`
- **Names used identically across tasks:** `device_label`, `resolve_current_session_id`, `InteractiveSession`, `require_interactive_session`, `get_current_session_id`, `INTERACTIVE_REQUIRED_CODE`, `write_new_sign_in_alert`, `schedule_new_sign_in_alert`, `drain_pending_alerts`, `caller_may_silence`, `split_silenceable`, `NEW_SIGN_IN_PROTECTED_CODE`, `alert_new_sign_in`, `SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS`, `evicted_device_ids`, `install_id_or_none`, `same_install`, `session_store_unavailable_http_exception`, `list_user_notifications_of_kind`, `PUBLIC_AUTH_REFUSAL_MESSAGES`, `listAuthSessions`, `revokeAuthSession`, `revokeOtherAuthSessions`, `errorCode`, `isProtectedAlertError`, `isInteractiveRequiredError`, `isOwnSignIn`, `visibleSignInAlerts`, `refused_ids`, `kind` (the list filter).

## Decisions the spec left open

1. **No schema change.** Session ids order the silence rule, `source_job_id` carries the session id, and `device_label` is computed on read. Reason: nothing needs a column, and a migration would add a PostgreSQL owner-migration deploy step.
2. **Session id from a verified credential, ownership-checked.** It comes from a signed token claim or the validated cookie and must be in the caller's live list. Reason: `is_current` stays unspoofable even by a validly signed token that names someone else's session.
3. **Dependency wrappers in `API_Deps`, logic in `core/AuthNZ`.** Reason: the layering ratchet forbids `core` importing `api`.
4. **`alert_new_sign_in` is an opt-in flag on `create_session`.** Reason: the admin E2E bootstrap and many tests create sessions and must not write notifications.
5. **Detached task, not FastAPI `BackgroundTasks`.** Reason: three call sites are helpers or the cookie mint with no response object; the task is best-effort and swallows named exceptions.
6. **The alert is also written when the capped insert evicted a session, but not when the same install signs in again.** Reason: with `MAX_SESSIONS_PER_USER=1` (or any full set) the owner whose session was just evicted has no other live session, yet must still see the sign-in on their next sign-in (Question 3). The cap evicts the same install first, so a browser signing in again would otherwise alert on every sign-in. No alert is written when the new session's install id (`install_id_or_none`, never the bare single-user cookie tag) equals that of another live session or of a session this sign-in evicted.
7. **The dedupe key is a hash.** Reason: `dedupe_key` is returned by the notifications API and stream, and the install id must never leave the server.
8. **Session ids, not timestamps, order sessions for the silence rule.** Reason: a source session whose row was purged still leaves its id in `source_job_id`; `sessions.id` is `INTEGER PRIMARY KEY AUTOINCREMENT` on SQLite (`core/DB_Management/authnz_session_schema.py:19-20`) and `SERIAL` on PostgreSQL (`pg_migrations_extra.py:1080`), so ids are monotonic and never reused. PostgreSQL ids are not commit-ordered, so two sign-ins racing within milliseconds can invert by one id; that is accepted.
9. **After 24 hours any interactive browser session may dismiss; API keys never.** Reason: the spec says "any browser session", and non-session credentials stay refused.
10. **`mark-read` returns 200 with `refused_ids` when some ids were allowed, and 403 `new_sign_in_protected` when every id was refused; dismiss and snooze return 403.** Reason: the inbox's bulk mark-read keeps working for its other items, and the inbox cannot show a refused alert as read.
11. **`logout` needs an interactive session for `all_devices: true`, and its revoke-all fallback for a token without a `session_id` is removed.** Reason: an API key sent as a Bearer token otherwise signs every session out (Question 2). This protects multi-user sessions. In single-user mode it does not isolate the owner from a holder of the API key, who can mint a cookie session that counts as interactive; there the protection is the alert plus the cap, and the remedy is to rotate the API key (stated in the guide and on the page).
12. **Revoke-one of a session the caller does not own stays an idempotent 200.** Reason: unchanged behaviour, and it reveals nothing about other users' ids.
13. **The current row has no Sign out button, and per-row sign-out needs no confirmation.** Reason: the account menu already signs this browser out, and a signed-out browser can sign in again; only the bulk action confirms.
14. **Nav group `setupRecovery`, order 4.** Reason: next to Provider Keys and the server connection, where account access is already configured.
15. **The banner is hidden on `/settings/sessions`.** Reason: that page is the review. Own-browser suppression is the current session's id, or the server's `same_install` flag on the source session, never IP or device label.
16. **`GET /notifications` gains an optional `kind` filter, and the banner asks for `new_sign_in` only.** Reason: reading the newest 100 items of every kind let a flooder bury the alert, and the stream sends one `notifications_coalesced` event for a burst of more than 50 rows. The banner also reloads on `notifications_coalesced`, `reset_required` and a lifecycle-epoch change, and `new_sign_in` has a 30-day retention in `DEFAULT_RETENTION_DAYS` so it expires.
17. **Extension: the page and the OS-notification suppression, no in-page banner.** Reason: the extension has no layout that hosts one.
18. **No ADR.** Reason: see Global Constraints; the owner may overrule and take number 100 after re-checking.
19. **`revoke_all_sessions_for_user` updates live rows only.** Reason: `sessions_revoked` and the "Signed out N other sessions" line were counting dead rows, and the update overwrote the `revoke_reason` of rows already revoked (for example `session_cap`).
20. **Suppression is by install id, through a server-computed `same_install` flag.** Reason: the label is a client-chosen User-Agent and a public IP is shared (CGNAT, VPN, office), so an attacker on the owner's IP who copies the User-Agent would silence the banner. `same_install` is true when the row's `device_id` is not null (and not the bare single-user cookie tag) and equals the current session's; the id is never returned. Same-install sessions coexist below the cap (eviction happens only over it), so the flag is meaningful.
21. **Alert tests are opt-in.** An autouse fixture silences the scheduler unless a test carries the `sign_in_alerts` marker. Reason: every production sign-in passes the flag and would otherwise write into the real `Databases/` directory from any test that signs a user in twice.
22. **`drain_pending_alerts` polls.** Reason: the alert tasks live on the `TestClient` portal loop, not the pytest-asyncio loop.
23. **The alert message says the label is "as the browser reported it".** Reason: the User-Agent is client-chosen.
24. **The refusal codes are allowlisted in `normalizeApiErrorBody`.** Reason: `apiClient` discards every 4xx object `detail` not on its allowlist, so `new_sign_in_protected` and `interactive_session_required` would never reach the UI. The two codes get fixed client-owned messages, like the provider error codes.
25. **A refused dismiss never reaches `reportMutationError`; the user may hide the alert on one browser.** Reason: reporting it would shut down the notification runtime. "Hide on this browser" (a `localStorage` key per notification id) covers the legitimate user whose session id is higher than their other devices' after a cookie clear.
26. **The interactive-session resolver reads the session list strictly and maps a database error to 503.** Reason: the non-strict read returns `[]` on error, which would read as "read-only API key".
27. **`resolve_current_session_id` obtains its own session manager when given none.** Reason: the getter ratchet freezes the notification endpoint module at 2 getter calls.

## Deviations from spec

1. **The "Is lock available?" capability flag is not built.** The spec lists it under section 2, but lock does not exist until PR 3; a flag with no consumer, and a page line explaining why a nonexistent control is missing, would be dead and misleading. PR 3 adds the flag beside its endpoint and puts its button next to "Sign out all other sessions" (the page leaves that action row as the one place to extend).
2. **The Next.js proxy does not forward the browser's IP.** Unbuildable as written (Question 1): Next's rewrite proxy cannot see the peer address and sets only `x-forwarded-host`. The alert shows the IP the server saw, subject to the existing trusted-proxy settings.
3. **The WebUI already toasts notifications.** The spec says it "only bumps a count"; `NotificationToastBridge` (`apps/tldw-frontend/components/notifications/NotificationToastBridge.tsx`, mounted at `WebLayout.tsx:922`) toasts every stream notification. The plan skips that toast for `new_sign_in` so the banner is the one surface.
4. **`logout` is guarded** (both `all_devices: true` and the no-`session_id` fallback) although the spec's list of guarded actions does not name it (Decision 11).
5. **The alert sets `link_url`, not a `link_type` the inbox recognises.** `resolveRouteForLinkType` in `NotificationsRoute.tsx` knows only reading, note, document, watchlist and job; the inbox's `link_url` branch already navigates to same-origin URLs.
6. **The mark-read response gains `refused_ids`, and a request whose ids are all refused gets 403.** The spec only says such requests are "refused".
7. **Banner suppression is by install id, not by IP and device label** (Decision 20); the spec says IP and label match. Reason: the label is client-chosen and a public IP is shared, so IP and label can be matched by an attacker on the owner's network.
8. **The alert is also written on eviction** (Decision 6), except when the same install signs in again; the spec says "at least one other live session".
9. **`GET /notifications` gains a `kind` filter and `new_sign_in` gets a 30-day retention** (Decision 16); the spec adds neither.
10. **Single-user mode is described as alert plus cap, not isolation** (Decision 11); the design implies a stolen API key cannot sign the owner out, which holds only in multi-user mode.

## Known limits

- The alert is written when another live session exists or the sign-in evicted one. A user with exactly one session and no eviction (their first sign-in) gets nothing.
- The install id is a client-chosen header (`X-TLDW-Device-Id`). `same_install` only suppresses a banner for sessions that share the current session's id, and the id is never returned, so a client cannot read another browser's id to silence it; two sign-ins that really share an install id (the same browser profile) are treated as one install.
- PostgreSQL session ids are not commit-ordered, so two sign-ins racing within milliseconds can invert by one id in the silence rule (Decision 8).
- Behind the bundled proxy with no reverse proxy in front, the alert and the sessions list show the proxy's address instead of the browser's (Question 1).
- The hosted-mode `/api/auth/*` proxy is outside this repo (`TldwAuth.authPath` rewrites `/api/v1/auth/` to `/api/auth/` in hosted mode), so `is_current`, `device_label` and `keep_current` are unverified there.
- A single-user API key can still mint a new cookie session, so "sign out all other sessions" in single-user mode ends the other browsers but not a script that holds the key; the cap bounds what that script can create, and the remedy is to rotate the API key (the page says so in single-user mode).
- Extension single-user connections use an API key, so the page there is read-only and every OS notification is raised (no own-session match is possible).
- The alert write is detached: a process stop between the sign-in response and the write loses that one alert.

## Review history

- 2026-11-25: second independent review (`.superpowers/alert-plan-review-2.md`, kept out of the repo). Four blockers and seven important findings applied, plus the minors and the owner-default adjustments: the web `apiClient` allowlist for the refusal codes (B1), live-only `mark_sessions_revoked_for_user_with_audit` (B2), the notification-permission grant and smoke test in the shared fixture (B3), the getter ratchet and the manager-less resolver (B4), alert-write containment (I5), a refused dismiss no longer stopping the runtime (I6), `same_install` replacing IP and label suppression (I7), the `kind` filter and retention (I8), the single-user wording (I9), "Hide on this browser" (I10), the id-ordering assumption (I11). Two points the review did not name were found while checking the code: the bare single-user cookie tag is shared by every header-less cookie session and is excluded from every install comparison (`install_id_or_none`), and the sessions repo raises `core.exceptions.DatabaseError` while `CollectionsDatabase` raises the backend's `DatabaseError`.
