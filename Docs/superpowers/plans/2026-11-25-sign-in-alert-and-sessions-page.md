# New-Sign-In Alert and Active Sessions Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a session is created while the user already has another live session, the user's other signed-in browsers learn of it (in-app banner plus an inbox notification). A new Active sessions page lists the live sessions with "This device" marked and lets the user sign out one session or all others. Ending sessions, and silencing the alert, require an interactive browser session, never an API key, virtual key or service token.

**Architecture:**
- No schema change. `sessions` already holds `ip_address`, `user_agent`, `device_id`, `last_activity`; `user_notifications` already holds `source_job_id`, `source_domain`, `dedupe_key`.
- One core module per concern: `core/AuthNZ/device_label.py` (user agent to "Chrome on macOS"), `core/AuthNZ/interactive_session.py` (which session, if any, a request is), `core/AuthNZ/sign_in_alert.py` (build, write, dedupe, and who may silence the notification).
- `SessionManager.create_session` gets an opt-in `alert_new_sign_in` flag; the four real sign-in call sites pass it. The alert write is a detached, best-effort task.
- The sessions endpoints and the notification endpoints share one rule: the caller's session id comes only from a verified credential (a signed access token's `session_id` claim, or the validated single-user cookie) and must belong to the caller's own live sessions.
- WebUI: a shared service and page in `packages/ui` (so the extension gets the page too) and a banner mounted beside the existing toast bridge in `tldw-frontend`.

**Tech Stack:** FastAPI, SQLite and PostgreSQL through `DatabasePool` / `CollectionsDatabase`, loguru, pytest; TypeScript, Next.js pages router, antd plus the design-system primitives, vitest + Testing Library.

**Spec:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`, section "2. New-sign-in alert and Active sessions page (PR 2)", with "Decisions made by the owner" 2-4, "Anti-goals" and "Known limits". Sections 0 and 1 are built; 3 and 4 are later PRs and out of scope.

## Questions for the owner

Neither blocks execution; the default applies unless the owner answers otherwise.

1. **IP shown, and banner suppression, behind the bundled WebUI proxy.** The design says the Next.js proxy forwards the browser's IP. It cannot: Next's rewrite proxy sets only `x-forwarded-host` (`apps/node_modules/.bun/next@16.1.4+363cf534159dda1d/node_modules/next/dist/server/lib/router-utils/proxy-request.js:30-36`), and a Next process with no reverse proxy in front never learns the peer address. A deployment with a reverse proxy in front already passes its `X-Forwarded-For` through, and the backend honours it only with `AUTH_TRUST_X_FORWARDED_FOR` plus `AUTH_TRUSTED_PROXY_IPS` (`core/AuthNZ/ip_allowlist.py:77`). **Recommendation (the default, a controller ruling):** show the IP the server saw, change nothing in Next, and document the two settings. A quickstart install then shows the proxy's address, so every client looks like it comes from one IP and an "IP and device label match" suppression would degrade to a label-only match on a client-chosen User-Agent. Therefore an IP match counts for banner suppression **only when the IP is public-routable**; for loopback, private, link-local, CGNAT and unparsable addresses the banner is suppressed on session-id equality only (Task 7).
2. **`POST /auth/logout` and API keys.** `logout` signs every session out in two ways: `all_devices: true`, and the single-session branch's fallback (`auth.py:2102-2108`) that runs `revoke_all_user_sessions` whenever the bearer carries no `session_id`, which an API key sent as `Authorization: Bearer` always does. The design does not list `logout`, yet both defeat "a stolen API key cannot sign the owner out". **Recommendation (the default):** `all_devices: true` requires an interactive session, and the revoke-all fallback is deleted: a token without a `session_id` gets 403 `interactive_session_required` and revokes nothing (Task 4). The cost is that a script holding an API key can no longer sign sessions out through `logout`.
3. **`MAX_SESSIONS_PER_USER=1` and the alert.** **Recommendation (the default, a controller ruling):** also schedule the alert when the capped insert evicted a session, so an owner signed out by the cap still sees the new sign-in on their next sign-in (Task 5).

## Global Constraints

- **Precondition:** branch from `origin/dev` only after PR #3615 (`feat/session-cap-15628`, the session cap) has merged. Until then `create_session_record_with_cap`, the live predicate, `request_device_id` and `SESSION_CAP_REVOKE_REASON` do not exist on `dev` and every task below fails. Do not stack on the cap branch.
  `git -C /Users/macbook-dev/Documents/GitHub/tldw_server fetch origin dev && git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add .claude/worktrees/sign-in-alert -b feat/sign-in-alert-sessions-page origin/dev`. Work only in that worktree. One PR, base `dev`. Never `git stash`, never `--no-verify`, never arm auto-merge.
- **Re-verify every cited line at branch time.** Anchors below were read on the cap branch (`dbaa3761e7`); they shift if #3615 changed afterwards. Cite by name when a line no longer matches.
- **Backlog:** the executor files the task with `tools/backlog-py` and an explicit id (Task 1 step 1). The highest ids seen on `origin/dev` and in open PRs on 2026-11-25 are around TASK-15642; take the next free id after `git ls-tree --name-only origin/dev backlog/tasks/` and `gh pr list --state open --json number,files`, written `TASK-<ID>` below. Edit only with `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task edit TASK-<ID> ...`; stage with `git add -A backlog/tasks`.
- **ADR:** none in this PR. ADR-099 (the cap) arrives with #3615; open PRs claim 096-098, so the next free number is 100 if the owner wants one (re-check with `git ls-tree --name-only origin/dev Docs/ADR/` and `gh pr list --state open --json number,files`). The design assigns the lock ADR to PR 3; the interactive-credential rule and the non-silenceable alert are recorded in `Docs/API-related/AuthNZ-API-Guide.md` here (Task 10) and restated by PR 3's ADR.
- **Migration:** none. Neither the SQLite numbered migrations (`AuthNZ/migrations.py`; 108 is claimed by open PR #3614) nor the PostgreSQL owner migration (`pg_migrations_extra.py`, `postgres_auth_catalog.py`, and its canonical-column write guard) is touched. If a column turns out to be needed, stop: it is a design change.
- `run_py <id> <files...>` means:
  `zsh ~/.cache/claude-tools/disk_guard.zsh "$TMPDIR/sia-<id>" -- /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider --basetemp "$TMPDIR/sia-<id>" -q <files...>`
- PostgreSQL tests use the `isolated_test_environment` fixture of `tldw_Server_API/tests/AuthNZ/conftest.py`; never build a database by hand. A PostgreSQL step passes only if the output shows the tests **passed**; if skipped, say so in the PR and rely on the CI Postgres job.
- Frontend: `cd apps/packages/ui && bunx vitest run <files> --maxWorkers=2` or `cd apps/tldw-frontend && bunx vitest run <files> --maxWorkers=2`, one vitest process at a time.
- Test emails use `@example.com` (pydantic `EmailStr` rejects `.test`).
- **Ratchets that will bite** (under `tldw_Server_API/tests/lint` unless noted):
  - `function_size_baseline.json`: `auth:login` is 433-434 lines against a ceiling of 454 and 10 parameters. Task 5 adds one keyword argument to an existing call inside it and nothing else. `User_DB_Handling:verify_jwt_and_fetch_user` (436) is not touched: the session id is read in `interactive_session.py`. New functions stay under 300 lines.
  - `layering_baseline.txt`: new imports go downward only (`core.AuthNZ` to `core.DB_Management`, `core.AuthNZ` to `core.AuthNZ`). Dependency wrappers live in `api/v1/API_Deps`; nothing under `core` imports `api`.
  - Endpoint-monkeypatch ratchet: tests patch `core.*` modules, never new attributes on `api/*/endpoints` modules.
  - Raw SQL: none in app code (repo and `CollectionsDatabase` methods suffice). Tests may use raw SQL to backdate rows.
  - `test_noncritical_exception_tuples.py`: tuples are named `*_NONCRITICAL_EXCEPTIONS`, never contain `BaseException` or `CancelledError`, and no bare `except Exception`.
  - Global-statements baseline: no `global` statement (a module-level `set` mutated with `.add` is fine).
  - Test-quality ratchet (`Helper_Scripts/ci/test_quality_triage.py --enforce`): the tests below assert on real objects and HTTP bodies. Do not add asserts inside test-defined fake classes, do not inject test-defined stub classes into the object under test, write no test whose asserts touch only `.status_code`, and no `status_code in (200, 5xx)`; every HTTP test asserts a body or a database effect. `test_quality_baseline.txt` must not grow. Patch `core` module attributes with `monkeypatch.setattr` and plain functions.
  - Route-auth ratchet: no new route. Fields and a query parameter change the OpenAPI schema, so regenerate with `make openapi-fingerprint` (Task 10).
  - `Docs/` edits are mirrored: run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff in the same commit.
  - Design-system ratchet (`apps/packages/ui/scripts/design-system-product-state-rules.mjs`): new UI imports `Alert`, `Badge`, `EmptyState`, `LoadingState` from `@/components/ui/...`, never antd's `Alert`, `Badge`, `Empty`, `Result`, `Spin`, `Tag`.
- Use loguru. **Never log tokens, cookie values, install ids (`device_id`), full user agents or email addresses.** Log the user id only, and honour `PII_REDACT_LOGS` like the neighbouring lines in `session_manager.py`.
- No new dependencies, settings or environment variables (the env-read ratchet stays untouched).
- Every commit ends with the `Co-Authored-By` trailer from the executing session's attribution reminder (the earlier plans used `Claude Opus 5.5 <noreply@anthropic.com>`); pass it as a second `-m`.
- Change summary: owner-written or owner waiver recorded before merge. Before merge: clean-subagent `/superpowers:requesting-code-review`, and diff the non-required Frontend E2E and admin-ui real-backend runs against a recent `dev` run.

## Review Focus

1. **Never a client-supplied "current".** `is_current` and every interactive check derive from a signed token claim or the validated cookie, and the id must be one of the caller's own live sessions. Task 2 `test_claim_naming_another_users_session_is_not_current`, Task 3 `test_client_cannot_claim_current_session`.
2. **Every route that ends a session is guarded, including the `/users/sessions` aliases.** The aliases call the `/auth` handlers as plain functions (`users.py:947`, `:981`, `:1009`), so a `Depends` default left unpassed would arrive as a truthy `Depends` object. Task 3 `test_session_token_ends_another_session_which_then_stops_working[/api/v1/users]` and the explicit keyword arguments in the alias bodies.
3. **A new sign-in can never silence its own alert.** The rule compares session ids (monotonic per table on both backends), not timestamps. Task 6 `test_source_session_cannot_dismiss_its_own_alert`, `test_later_session_cannot_mark_read_or_snooze`, `test_after_24_hours_any_browser_session_may_dismiss`.
4. **Sign-in never depends on the alert.** The write runs in a detached task and swallows named exceptions. Task 5 `test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids`.
5. **Test pollution.** Every production sign-in passes the alert flag, so any existing test that signs one user in twice would write through `CollectionsDatabase.for_user` into the real default `Databases/user_databases/`. `git status` cannot see it (`.gitignore` ignores `*.db` and `/Databases/`). Task 5 therefore adds an autouse fixture that makes alerts silent by default; only tests carrying the `sign_in_alerts` marker run the real scheduler, and those set a tmp `USER_DB_BASE_DIR`. After the full run, compare `find Databases/user_databases -newer <branch-start marker file>` (or the directory's file count) before and after.
6. **Banner suppression behind one NAT.** The design suppresses the banner when IP and device label match the current session, so two identical machines behind one NAT silence each other's banner (the inbox entry stays). Narrowed by a controller ruling: an IP match counts only for a public-routable IP (Task 7), which also covers the quickstart proxy case where every client shares the proxy's address. Recorded under Known limits.

## Files

| File | Change |
| --- | --- |
| `tldw_Server_API/app/core/AuthNZ/device_label.py` (new) | `device_label(user_agent)` |
| `tldw_Server_API/app/core/AuthNZ/interactive_session.py` (new) | `InteractiveSession`, `resolve_current_session_id`, refusal exception |
| `tldw_Server_API/app/api/v1/API_Deps/interactive_session_deps.py` (new) | `get_current_session_id`, `require_interactive_session` |
| `tldw_Server_API/app/core/AuthNZ/sign_in_alert.py` (new) | alert build, write, dedupe, schedule; the silence rule |
| `tldw_Server_API/app/core/AuthNZ/session_manager.py` | `create_session(alert_new_sign_in=False)` hook |
| `tldw_Server_API/app/core/AuthNZ/single_user_session.py`, `app/api/v1/endpoints/auth.py` | the four call sites pass `alert_new_sign_in=True`; sessions routes and `logout` |
| `tldw_Server_API/app/api/v1/endpoints/users.py` | the three `/users/sessions` aliases |
| `tldw_Server_API/app/api/v1/schemas/auth_schemas.py` | `SessionResponse.is_current`, `.device_label` |
| `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`, `app/api/v1/endpoints/notifications.py`, `app/core/DB_Management/Collections_DB.py` | `new_sign_in` kind; gating of mark-read, dismiss, snooze; `list_user_notifications_of_kind` |
| `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py` | `revoke_all_sessions_for_user` touches live rows only |
| `tldw_Server_API/tests/conftest.py`, `pyproject.toml` | autouse silent-alert fixture, `sign_in_alerts` marker |
| `apps/packages/ui/src/services/notifications.ts`, `apps/tldw-frontend/lib/api/notifications.ts` | `markNotificationsRead` return type |
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
- Create: `tldw_Server_API/tests/AuthNZ_SQLite/_sessions_http_fixtures.py` (shared by Tasks 2-6; not a test module)
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_interactive_session_sqlite.py`

**What the code shows (cap branch):**
- A multi-user browser holds an access JWT whose `session_id` claim is added at every sign-in (`auth.py:431` in `_issue_multi_user_tokens`, `:1915` in `login`, `:3202` in `verify_magic_link`) and kept on refresh (`:2439`). `verify_jwt_and_fetch_user` (`core/AuthNZ/User_DB_Handling.py:470`) checks the blacklist and builds a principal with `kind="user"`, `token_type="access"` (`:875`) but exposes no session id. `logout` reads it the same way (`auth.py:2087`, `payload.get("session_id")`).
- A single-user cookie session resolves to `token_type="single_user_session"` and sets `request.state.single_user_session_id` (`core/AuthNZ/auth_principal_resolver.py:364`).
- API keys resolve to `token_type="api_key"`; service tokens to `kind="service"` (service branch of `get_auth_principal`, `auth_principal_resolver.py:310`); scoped (virtual-key) JWTs are refused with 403 by `verify_jwt_and_fetch_user` on routes that do not enforce scope (the scoped-claim block, `User_DB_Handling.py:~530-575`) and carry no `session_id`; impersonation sets `principal.impersonation`.

**Interfaces:**
- Produces, in `interactive_session.py`:
  - `INTERACTIVE_REQUIRED_CODE = "interactive_session_required"`
  - `@dataclass(frozen=True) class InteractiveSession: user_id: int; session_id: int`
  - `def interactive_required_http_exception() -> HTTPException`: 403, `detail={"code": "interactive_session_required", "message": "This action needs a browser sign-in. API keys, virtual keys and service tokens cannot end sessions."}`
  - `async def resolve_current_session_id(request: Request, principal: AuthPrincipal, session_manager: SessionManager) -> int | None`. None unless all hold: `principal.kind == "user"`, `principal.user_id` is a positive int, `principal.impersonation` is false, and the id found is one of `await session_manager.get_user_sessions(principal.user_id)`. The id comes from `request.state.single_user_session_id` when `principal.token_type == "single_user_session"`, or from `get_jwt_service().decode_access_token(bearer_credential(request.headers.get("Authorization")))["session_id"]` when `principal.token_type == "access"`. A decode error or a non-int claim gives None.
- Produces, in `interactive_session_deps.py`:
- Import `get_auth_principal` and `get_session_manager_dep` from `tldw_Server_API.app.api.v1.API_Deps.auth_deps`, as `auth.py` and `users.py` do, not from the core resolver.
  - `async def get_current_session_id(request: Request, principal: AuthPrincipal = Depends(get_auth_principal), session_manager: SessionManager = Depends(get_session_manager_dep)) -> int | None`
  - `async def require_interactive_session(request, principal, session_manager) -> InteractiveSession` (same dependencies): raises `interactive_required_http_exception()` when the id is None.
- Produces, in `_sessions_http_fixtures.py`: fixture `two_user_client` yielding the frozen dataclass `SessionsCtx(client, alice_id, bob_id, password)`. Build it from `multi_user_client` in `test_session_cap_http_sqlite.py:20-52` with two verified users `alice` and `bob` (`@example.com`), `MAX_SESSIONS_PER_USER=4`, and `USER_DB_BASE_DIR` set to `tmp_path / "user_dbs"` as `Notifications/test_notifications_api.py::notifications_app` does (lines 53-60). Helpers: `login(client, username, password, *, device=None, user_agent=None) -> dict`, `bearer(tokens) -> dict`, `api_key_for(client, tokens) -> str` (POST `/api/v1/users/api-keys`, `users.py:769`, body `{"name": "t", "scope": "write"}`, returns the `key` field of the 201 body), `session_id_of(tokens) -> int` (decode the access token's `session_id` claim with `get_jwt_service().decode_access_token`). Also `async seed_session_revoked_with_reason(user_id, reason) -> int` (a dead `sessions` row inserted through `AuthnzSessionsRepo.create_session_record` and revoked with `revoke_session_record`) and `async revoke_reason_of(session_id) -> str | None` (a raw SELECT, allowed in tests), used by Task 3.

- [ ] **Step 1: Write the failing tests** in `test_interactive_session_sqlite.py` (`pytestmark = pytest.mark.asyncio`; import the fixture and helpers from `_sessions_http_fixtures`). `_request(headers)` builds `starlette.requests.Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})`; `manager = await get_session_manager()`. Every test body starts by binding `ctx = two_user_client; client = ctx.client; pw = ctx.password` (the snippets abbreviate with `ctx`/`client`/`pw`; `A`, `B`, `C` are `login(...)` results named in the test), and calls `get_jwt_service()` from `tldw_Server_API.app.core.AuthNZ.jwt_service`.

```python
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
    # sign in through POST /api/v1/auth/magic-link/verify with a token minted the way the existing magic-link tests do
    # (git grep -n 'magic-link/verify' -- tldw_Server_API/tests); the returned access token must resolve to its own session id
    ctx = two_user_client  # `verified` is the JSON body of that verify call
    assert await resolve_current_session_id(_request({"Authorization": f"Bearer {verified['access_token']}"}), AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access"), manager) is not None
```
The single-user cookie branch is covered by Task 3's cookie tests.

- [ ] **Step 2: Run, expect failure** (import error): `run_py t2-red tldw_Server_API/tests/AuthNZ_SQLite/test_interactive_session_sqlite.py`

- [ ] **Step 3: Implement** the two modules as specified.

- [ ] **Step 4: Run, expect pass**; `ruff check`; `git status --short` shows nothing new under `Databases/`.

- [ ] **Step 5: Commit** `feat(authnz): resolve the interactive browser session behind a request (TASK-<ID>)`.

---

### Task 3: Sessions API: `is_current`, `device_label`, `keep_current`, interactive guard

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/auth_schemas.py:334-341` (`SessionResponse`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py`: `list_user_sessions` (`:2146`), `revoke_session` (`:2183`), `revoke_all_sessions` (`:2245`). `logout` is Task 4.
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py:928-1015` (the three aliases)
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py:436-520` (`revoke_all_sessions_for_user`): both the PostgreSQL and the SQLite branches update every row of the user with no live filter, so `sessions_revoked` counts dead rows and the update overwrites the `revoke_reason` of rows already revoked (for example `session_cap`). Add the live predicate (`LIVE_SESSION_PREDICATE_PG` / `LIVE_SESSION_PREDICATE_SQLITE`, `sessions_repo.py:20-24`) to each WHERE clause. The other callers (`logout`, password reset, admin revoke-all) only lose updates to rows that are already dead.
- Modify: the existing tests that call `revoke_all_sessions`, `revoke_session` or `list_user_sessions` with a session-less credential or as plain functions. Find them with `git grep -n -E "sessions/revoke-all|users/sessions/[0-9]|auth\.(revoke_all_sessions|revoke_session|list_user_sessions)\(" -- tldw_Server_API/tests`. Known: `tests/AuthNZ/unit/test_auth_endpoints_extended.py:~1827` calls `auth.revoke_all_sessions(...)` directly and `:~1851` `auth.list_user_sessions(...)`; the new parameters would receive `Query`/`Depends` default objects, so pass `keep_current=False, interactive=InteractiveSession(user_id=..., session_id=...)` (and `current_session_id=None`) explicitly. Also `tests/AuthNZ/unit/test_user_endpoints.py:~227-330`, `tests/AuthNZ/integration/test_user_endpoints_integration.py`, `test_auth_comprehensive.py`, `test_single_user_login_api_key.py`, `tests/UserProfile/test_user_management_account_state_guards.py`: give each a session-bearing credential.
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_sessions_api_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_cookie.py`, `tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_postgres.py`

**Interfaces:**
- `SessionResponse` gains `is_current: bool = Field(False, description="True only for the session making this request, derived from the verified credential.")` and `device_label: str = Field("Unknown browser", description="Browser and OS family from the user agent.")`.
- `list_user_sessions(current_user=..., session_manager=..., current_session_id: Optional[int] = Depends(get_current_session_id))`: each row sets `is_current=(row["id"] == current_session_id)` and `device_label=device_label(row.get("user_agent"))`. It still accepts any credential (`store/connection.tsx:813` and `Settings/tldw.tsx:641` probe it with whatever credential they hold).
- `revoke_session(session_id, current_user, session_manager, interactive: InteractiveSession = Depends(require_interactive_session))`: ownership still comes from `get_user_sessions(user_id)`; the revoke passes `expected_user_id=user_id` (`SessionManager.revoke_session`, `session_manager.py:1370`). A session that is not the caller's stays the existing idempotent 200 no-op.
- `revoke_all_sessions(current_user, session_manager, keep_current: bool = Query(False, description="Keep the caller's own session."), interactive: InteractiveSession = Depends(require_interactive_session))`: `except_session_id = interactive.session_id if keep_current else None`, passed to the existing `revoke_all_user_sessions(except_session_id=...)` (`session_manager.py:1416`, which already forwards it to the repo and the blacklist). Without the flag every live session is revoked (the repo now counts only live rows, so `sessions_revoked` is the number of live sessions ended), as today, apart from the new requirement.
- Aliases in `users.py`: the list alias adds `current_session_id: Optional[int] = Depends(get_current_session_id)`; the revoke aliases add `interactive: InteractiveSession = Depends(require_interactive_session)` and revoke-all also `keep_current: bool = Query(False)`. **Every new handler parameter is passed by keyword in the alias body** (Review Focus 2).

- [ ] **Step 1: Write the failing SQLite HTTP tests** in `test_sessions_api_sqlite.py` (fixture `two_user_client`; `A = login(alice, device="dev-aaaaaaaa", user_agent=CHROME_MAC)` and `B = login(alice, device="dev-bbbbbbbb", user_agent=SAFARI_IPHONE)`, constants as in Task 1):

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


def test_api_key_lists_sessions_but_none_is_current(two_user_client):
    rows = client.get("/api/v1/auth/sessions", headers={"X-API-KEY": api_key_for(client, A)}).json()
    assert len(rows) == 2 and not any(r["is_current"] for r in rows)


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

- [ ] **Step 3: Write the PostgreSQL test** in `test_sessions_api_postgres.py` (`isolated_test_environment`, as `test_session_cap_postgres.py:336-363`: register `pgalice` and `pgbob` at `@example.com`, `AUTH_MODE=multi_user`, `JWT_SECRET_KEY="X"*40`): one test `test_sessions_api_postgres` running the `is_current` check, `keep_current`, the API-key refusal and the cross-user isolation sequence above, each with body assertions, and the live-only revoke count with a `session_cap` row left untouched.

- [ ] **Step 4: Run, expect failure** (missing fields, 200 instead of 403, 422 for `keep_current`): `run_py t3-red <the three new files>`

- [ ] **Step 5: Implement** the schema, handlers and aliases; then update the old tests listed under Files.

- [ ] **Step 6: Run, expect pass**: the Step 4 files, `tldw_Server_API/tests/AuthNZ_SQLite/test_interactive_session_sqlite.py`, every test file the Files-list grep found, and `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`. PostgreSQL tests must run, not skip.

- [ ] **Step 7: Commit** `feat(authnz): mark the current session and require a browser session to end sessions (TASK-<ID>)`.

---

### Task 4: `logout` cannot be used with an API key to end sessions

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py:2005-2140` (`logout`; it is not in the size baseline)
- Modify: the existing logout tests (list below)
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_logout_guard_sqlite.py`

**What the code does today:** the single-session branch (`auth.py:2052-2108`) decodes the bearer with `jwt_service.verify_token`; a non-JWT bearer (an API key sent as `Authorization: Bearer`) fails to decode, so `payload = {}`, `session_id` is None, and the code falls into "Fallback to full session revoke" (`:2102-2108`), which calls `revoke_all_user_sessions`. So `POST /auth/logout` with an API key and no body already signs every session out, in addition to `all_devices: true` (`:2033-2039`).

**Interfaces:**
- `all_devices: true`: before revoking, `if await resolve_current_session_id(request, current_user, session_manager) is None: raise interactive_required_http_exception()`.
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


def test_session_token_logout_all_devices_still_works(two_user_client):
    assert client.post("/api/v1/auth/logout", json={"all_devices": True}, headers=bearer(A)).json()["message"].startswith("Logged out from")
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401
```
The first assertion of each of the last two tests carries the body, so none is status-only.

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
- Modify (one keyword argument each): `tldw_Server_API/app/api/v1/endpoints/auth.py:416` (`_issue_multi_user_tokens`), `:1897` (`login`), `:3190` (`verify_magic_link`); `tldw_Server_API/app/core/AuthNZ/single_user_session.py:65` (`mint_single_user_session`)
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_alert_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_new_sign_in_alert_postgres.py`; extend `test_sessions_api_cookie.py`
- Modify: `tldw_Server_API/tests/conftest.py` (an autouse fixture `_sign_in_alerts_silent_by_default`) and `pyproject.toml:582` (register the marker `sign_in_alerts`; `--strict-markers` is on, `pyproject.toml:668`)

**Interfaces, in `sign_in_alert.py`:**
- Constants, verbatim from the spec: `NEW_SIGN_IN_KIND = "new_sign_in"`, `NEW_SIGN_IN_TITLE = "New sign-in to your account"`, `NEW_SIGN_IN_SEVERITY = "warning"`, `NEW_SIGN_IN_SOURCE_DOMAIN = "auth_session"`, `NEW_SIGN_IN_LINK_URL = "/settings/sessions"`, `NEW_SIGN_IN_DEDUPE_BUCKET_SECONDS = 600`, `NEW_SIGN_IN_OPEN_AFTER = timedelta(hours=24)`.
- `def new_sign_in_message(*, user_agent: str | None, ip_address: str | None, now: datetime) -> str`: `"{label} (as the browser reported it) signed in from {where} at {YYYY-MM-DD HH:MM} UTC. If this was not you, sign out your other sessions."`, with `label = device_label(user_agent)` and `where = ip_address`, or `"an unknown address"` when it is empty or `"unknown"` (the value `_auth_request_client_ip` returns, `auth.py:1003`). The label comes from a client-chosen User-Agent, so the wording does not present it as authoritative.
- `def new_sign_in_dedupe_key(*, user_id: int, device_id: str | None, ip_address: str | None, user_agent: str | None, now: datetime) -> str`: `"new_sign_in:" + sha256(f"{user_id}|{group}|{int(now.timestamp()) // 600}").hexdigest()[:32]`, where `group = device_id` when present, else `f"{ip_address}|{user_agent}"`. It is a hash, never the install id: `dedupe_key` is returned by the notifications API and stream.
- `def write_new_sign_in_alert(db: CollectionsDatabase, *, session_id: int, ip_address: str | None, user_agent: str | None, device_id: str | None, user_id: int, now: datetime | None = None) -> UserNotificationRow`: one `db.create_user_notification(kind=..., title=..., message=..., severity=..., source_job_id=str(session_id), source_domain="auth_session", link_type="settings_route", link_url="/settings/sessions", dedupe_key=...)`. A dedupe hit returns the existing row (the behaviour of `create_user_notification`, `Collections_DB.py:7546-7620`).
- `def schedule_new_sign_in_alert(*, user_id: int, session_id: int, ip_address, user_agent, device_id, db_pool: DatabasePool, evicted: bool = False) -> None`: a no-op without a running loop; otherwise it creates a task, adds it to a module-level `set` of pending tasks and discards it on completion (a done-callback). The task counts `AuthnzSessionsRepo(db_pool).get_active_sessions_for_user(user_id)` (the live predicate shared with the cap) and returns only when no row other than `session_id` exists **and** `evicted` is false; otherwise it runs `asyncio.to_thread` over `with CollectionsDatabase.for_user(user_id=user_id) as db: write_new_sign_in_alert(db, ...)`. It catches `SIGN_IN_ALERT_NONCRITICAL_EXCEPTIONS = (DatabaseError, sqlite3.Error, OSError, KeyError, ValueError, TypeError, RuntimeError, TimeoutError)` (`DatabaseError` from `core.DB_Management.backends.base`; never `BaseException` or `CancelledError`) and logs `logger.bind(error_type=type(exc).__name__).warning("New sign-in alert not written")` with no ids.
- `async def drain_pending_alerts(timeout: float = 5.0) -> None` (tests only): the tasks are created on the `TestClient` portal's event loop and mutated from that thread, while tests run on the pytest-asyncio loop, so `asyncio.gather` over them would raise "attached to a different loop". Poll instead: `deadline = time.monotonic() + timeout; while list(_PENDING) and time.monotonic() < deadline: await asyncio.sleep(0.02)`; raise `AssertionError` on timeout.
- `create_session(..., alert_new_sign_in: bool = False)`: after the cap block and before the Redis cache, `if alert_new_sign_in: schedule_new_sign_in_alert(user_id=user_id, session_id=session_id, ip_address=ip_address, user_agent=user_agent, device_id=device_id, db_pool=db_pool, evicted=bool(evicted))` (`evicted` is the list `create_session_record_with_cap` returned, `session_manager.py:1144`), so an owner signed out by the cap still gets an alert on the next sign-in, including with `MAX_SESSIONS_PER_USER=1`. The default False keeps `services/admin_e2e_support_service.py:695` silent.
- Test isolation, `tldw_Server_API/tests/conftest.py`: an autouse fixture `_sign_in_alerts_silent_by_default(request, monkeypatch)` that, unless `request.node.get_closest_marker("sign_in_alerts")` is set, does `monkeypatch.setattr(session_manager_module, "schedule_new_sign_in_alert", lambda **_: None)`. Every production sign-in passes the flag, so without it any test that signs one user in twice writes into the real `Databases/user_databases/` (ignored by `.gitignore`, so `git status` cannot show it). The alert test modules set `pytestmark = [pytest.mark.asyncio, pytest.mark.sign_in_alerts]`; the cookie and PostgreSQL HTTP tests below also `monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))`.

- [ ] **Step 1: Write the failing tests** in `test_new_sign_in_alert_sqlite.py` (fixture `two_user_client`; `notifications(client, tokens)` awaits `drain_pending_alerts()` then returns `GET /api/v1/notifications` `items`):

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


async def test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids(two_user_client, monkeypatch):
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    def boom(**_):
        raise DatabaseError("db down")
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


async def test_alert_is_written_when_the_cap_evicted_the_only_other_session(two_user_client, monkeypatch):
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1"); reset_settings()
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")      # evicts A; no other live session remains
    (n,) = await notifications(client, B)
    assert n["source_job_id"] == str(session_id_of(B))



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
(`APP_ROOT` comes from `tldw_Server_API.tests.lint._ast_cache`.) Add `test_single_user_cookie_mint_alerts_the_other_install` to `test_sessions_api_cookie.py`: the second mint from another install writes one alert into the Collections DB of user `SINGLE_USER_FIXED_ID`; the first writes none.

- [ ] **Step 2: PostgreSQL**, `test_new_sign_in_alert_postgres.py` (`isolated_test_environment`; build the Collections DB on PostgreSQL the way `Collections/test_collections_postgres_integration.py:29-35` does with `_pg_backend(db_name)` and `CollectionsDatabase.from_backend(user_id=..., backend=...)`):
  - `test_alert_write_dedupes_and_isolates_users_postgres`: `write_new_sign_in_alert` twice for user "1" with one device inside a bucket gives one row; once for user "2" gives a separate row; `db.list_user_notifications()` of each shows only its own.
  - `test_second_sign_in_writes_an_alert_over_http_postgres`: register `pgalice` (`@example.com`), log in twice with different install ids, `await drain_pending_alerts()`, with `USER_DB_BASE_DIR` set to a tmp dir (the Collections DB of the HTTP path is the per-user SQLite file), read `GET /api/v1/notifications`; exactly one `new_sign_in` item.

- [ ] **Step 3: Run, expect failure**: `run_py t4-red tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_alert_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_new_sign_in_alert_postgres.py tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_cookie.py`

- [ ] **Step 4: Implement**, then add the `alert_new_sign_in=True` keyword to each of the four calls (`login` has a ceiling of 454 lines at 433-434 today: put the keyword on an existing line break, not a new statement); then add the autouse fixture and the marker.

- [ ] **Step 5: Run, expect pass**; also `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`, `tldw_Server_API/tests/Notifications`, `tldw_Server_API/tests/lint/test_function_size_ratchet.py`. `git status --short` shows nothing new under `Databases/`.

- [ ] **Step 6: Commit** `feat(authnz): alert a user's other sessions to a new sign-in (TASK-<ID>)`.

---

### Task 6: Only an older browser session can silence a new-sign-in alert

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/sign_in_alert.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/reminders_schemas.py`: `NotificationsMarkReadResponse` gains `refused_ids: list[int] = Field(default_factory=list)`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notifications.py`: `mark_read` (`:213`), `dismiss_notification` (`:229`), `snooze_notification` (`:244`)
- Modify: `tldw_Server_API/app/core/DB_Management/Collections_DB.py` (next to `get_user_notification`, `:7813`): `def list_user_notifications_of_kind(self, kind: str, ids: list[int]) -> list[UserNotificationRow]`, one `SELECT ... WHERE user_id = ? AND kind = ? AND id IN (...)` (no per-id loop; bulk mark-read can send hundreds of ids)
- Modify: `apps/packages/ui/src/services/notifications.ts:316` (`markNotificationsRead` returns `{ updated: number; refused_ids?: number[] }`) and `apps/tldw-frontend/lib/api/notifications.ts:151` (same type)
- Modify: `apps/tldw-frontend/components/notifications/NotificationsRoute.tsx` (`handleMarkRead`, `:~386-411`; the "View" button, `:~712-728`): see Task 9
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_silence_sqlite.py`; `tldw_Server_API/tests/AuthNZ/integration/test_new_sign_in_silence_postgres.py`; extend `test_sessions_api_cookie.py`; both modules carry `pytest.mark.sign_in_alerts`

**Interfaces, in `sign_in_alert.py`:**
- `NEW_SIGN_IN_PROTECTED_CODE = "new_sign_in_protected"`; `def new_sign_in_protected_http_exception() -> HTTPException`: 403, `detail={"code": "new_sign_in_protected", "message": "A new sign-in alert can only be dismissed from a browser that was signed in before it, or after 24 hours."}`.
- `def caller_may_silence(row: UserNotificationRow, caller_session_id: int | None, now: datetime | None = None) -> bool`: True for any kind other than `new_sign_in`; False when `caller_session_id is None` (API key, virtual key, service token: refused even after 24 hours); True when `now - created_at >= NEW_SIGN_IN_OPEN_AFTER`; otherwise `caller_session_id < int(row.source_job_id)`. Session ids are monotonic per table on both backends, so a lower id is an earlier session, and an id survives the purge of the session row. A non-numeric `source_job_id` is False.
- `def split_silenceable(rows: list[UserNotificationRow], caller_session_id: int | None, now: datetime | None = None) -> tuple[list[int], list[int]]`: pure; the allowed and refused ids among `rows` (the `new_sign_in` rows the endpoint fetched with `list_user_notifications_of_kind`).
- Endpoints: each takes `request: Request` and `principal: AuthPrincipal = Depends(RequirePermission(...))` (replacing `_principal`). They fetch the target rows first with `db.list_user_notifications_of_kind("new_sign_in", ids)`; **only when that list is non-empty** do they resolve the caller with `caller = await interactive_session.resolve_current_session_id(request, principal, await get_session_manager())` (call it through the module so a test can patch the core module) (core `get_session_manager`, not a `Depends`, so a notifications app with no AuthNZ database, like `Notifications/test_notifications_api.py::notifications_app`, never touches it for other kinds). **Contract (one, used by Tasks 6 and 9):** `mark_read` marks the allowed ids and returns 200 `{"updated": n, "refused_ids": [...]}` when at least one id was allowed or no `new_sign_in` row was involved; when every requested id was refused it raises 403 `new_sign_in_protected`. `dismiss_notification` and `snooze_notification` raise 403 `new_sign_in_protected` for a refused row before doing anything else.
- `NotificationPreferencesResponse` is unchanged: no off switch exists for this kind.

- [ ] **Step 1: Write the failing tests** (`two_user_client`; A, B and C are three alice logins in that order with devices `dev-aaaaaaaa`, `dev-bbbbbbbb`, `dev-cccccccc`; `nB` and `nC` are the alert ids whose `source_job_id` is B's and C's session; `item(client, tokens, id)` reads one notification from `GET /api/v1/notifications`; `backdate(ctx, id, hours)` opens `CollectionsDatabase.for_user(user_id=ctx.alice_id)` and runs `db.backend.execute("UPDATE user_notifications SET created_at = ? WHERE id = ?", (iso, id))`):

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


async def test_other_kinds_never_resolve_the_session_manager(two_user_client, monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("caller resolved for a kind that needs no check")
    monkeypatch.setattr(interactive_session, "resolve_current_session_id", boom)   # core module: the endpoint calls it through the module, so a call would raise
    with CollectionsDatabase.for_user(user_id=ctx.alice_id) as db:
        other = db.create_user_notification(kind="job_completed", title="Done", message="m", severity="info")
    assert client.post(f"/api/v1/notifications/{other.id}/dismiss", headers=bearer(A)).json() == {"dismissed": True}
```
Add two cookie tests to `test_sessions_api_cookie.py`: an older cookie session dismisses the alert (with the CSRF header) and gets `{"dismissed": true}`; `X-API-KEY` gets 403 `new_sign_in_protected`.

- [ ] **Step 1b: PostgreSQL silence test** `test_new_sign_in_silence_postgres.py` (`isolated_test_environment`, `USER_DB_BASE_DIR` tmp, `AUTH_MODE=multi_user`, register `pgalice`, three logins A, B, C with distinct install ids): the source session B gets 403 `new_sign_in_protected` dismissing its own alert and the alert stays undismissed; the older session A dismisses it and gets `{"dismissed": true}`; an API key gets 403 `new_sign_in_protected`.

- [ ] **Step 2: Run, expect failure** (200 for every case today): `run_py t5-red tldw_Server_API/tests/AuthNZ_SQLite/test_new_sign_in_silence_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_sessions_api_cookie.py`

- [ ] **Step 3: Implement.** Leave `list_user_notifications` and the stream untouched: a banner needs the alert even from a session that cannot silence it.

- [ ] **Step 4: Run, expect pass**, plus `tldw_Server_API/tests/Notifications` (the existing tests touch only other kinds, which never resolve a session, so they pass unchanged; this is pinned by `test_other_kinds_never_resolve_the_session_manager`).

- [ ] **Step 5: Commit** `feat(notifications): only an older browser session can silence a new-sign-in alert (TASK-<ID>)`.

---

### Task 7: Frontend services: session API and own-sign-in matching

**Files:**
- Create: `apps/packages/ui/src/services/auth-sessions.ts`
- Create: `apps/packages/ui/src/services/sign-in-alert.ts`
- Test (create): `apps/packages/ui/src/services/__tests__/auth-sessions.test.ts`, `sign-in-alert.test.ts`

**Interfaces:**
- `auth-sessions.ts` (uses `bgRequest`, `appendPathQuery`, `toAllowedPath` the way `services/guardian.ts:302-316` does):
  - `type AuthSession = { id: number; ip_address: string | null; user_agent: string | null; device_label: string; created_at: string; last_activity: string; expires_at: string; is_current: boolean }`
  - `listAuthSessions(): Promise<AuthSession[]>` (`GET /api/v1/auth/sessions`)
  - `revokeAuthSession(id: number): Promise<void>` (`DELETE /api/v1/auth/sessions/{id}`)
  - `revokeOtherAuthSessions(): Promise<{ sessions_revoked: number }>` (`POST /api/v1/auth/sessions/revoke-all?keep_current=true`, reading `details.sessions_revoked`)
  - `INTERACTIVE_REQUIRED_CODE = "interactive_session_required"`; `errorCode(error: unknown): string | undefined` (reads `status`, `details.detail.code` and `detail.code` exactly as `noteSessionReplaced` does in `services/tldw/session-identity.ts`); `isInteractiveRequiredError(error): boolean`
- `sign-in-alert.ts`:
  - `NEW_SIGN_IN_KIND = "new_sign_in"`, `NEW_SIGN_IN_PROTECTED_CODE = "new_sign_in_protected"`, `isProtectedAlertError(error): boolean`
  - `isPublicIp(ip: string | null | undefined): boolean`: true only for a parsable public-routable address; false for loopback, private (10/8, 172.16/12, 192.168/16), link-local (169.254/16, fe80::/10), CGNAT (100.64/10), unique-local (fc00::/7), 0.0.0.0/8, `::1`, and anything unparsable.
  - `isOwnSignIn(sourceJobId: string | number | null | undefined, sessions: AuthSession[]): boolean`. True when the current session's id equals `sourceJobId`; or when the source session is in the list with the same `ip_address` **and `isPublicIp(ip_address)`** and the same `device_label` as the current session and that label is not `"Unknown browser"`. For non-public addresses (every client behind the bundled proxy appears from one) only session-id equality counts. False when there is no current session (API-key connection) or the source session is gone.
  - `visibleSignInAlerts(items: NotificationItem[], sessions: AuthSession[]): NotificationItem[]`: kind `new_sign_in`, `read_at` and `dismissed_at` empty, not `isOwnSignIn`, newest first.

- [ ] **Step 1: Write the failing tests.** `isOwnSignIn`: id equality true; same public IP and label with a different id true; the same two sessions with `127.0.0.1`, `10.0.0.5`, `192.168.1.9`, `172.20.0.2`, `169.254.1.1`, `100.64.0.7`, `::1` and `fd00::1` false (id equality still true); `isPublicIp` for each of those and for `203.0.113.7` / `2001:4860:4860::8888` (true); same IP, different label false; same label, different IP false; label `"Unknown browser"` on both false; source session absent false; no `is_current` row false; string `"7"` against numeric id 7 true. `visibleSignInAlerts`: drops read, dismissed, other-kind and own alerts, and sorts newest first. `auth-sessions`: with `bgRequest` mocked through `vi.mock("@/services/background-proxy")`, assert the exact `path` and `method` of the three calls (including `?keep_current=true`) and `errorCode` for the three error shapes.

- [ ] **Step 2: Run, expect failure**: `cd apps/packages/ui && bunx vitest run src/services/__tests__/auth-sessions.test.ts src/services/__tests__/sign-in-alert.test.ts --maxWorkers=2`

- [ ] **Step 3: Implement.** **Step 4: Run, expect pass.** **Step 5: Commit** `feat(ui): session API client and own-sign-in matching (TASK-<ID>)`.

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
- Modify: `apps/tldw-frontend/components/notifications/NotificationsRoute.tsx` (the Snooze button, `:~735-745`): not rendered for a `new_sign_in` item; `handleMarkRead` (`:~386-411`) marks the item read locally and decrements the unread count only for ids not in the response's `refused_ids`, and treats a 403 `new_sign_in_protected` (every id refused) as a refusal: it shows the explanatory text as a toast, changes nothing locally. The "View" button (`:~712-728`) calls `handleMarkRead` first and then navigates regardless, so it still reaches `/settings/sessions`. Dismiss shows the same text on 403
- Modify: `apps/packages/ui/src/entries/shared/notification-subscription.ts` (`handleStreamEvent`, the `notification` branch that calls `notify(...)`)
- Test (create): `apps/tldw-frontend/__tests__/components/new-sign-in-banner.test.tsx`; extend `apps/tldw-frontend/__tests__/components/notification-toast-bridge.test.tsx`, the inbox test for `NotificationsRoute` (`git grep -l NotificationsRoute -- apps/tldw-frontend/__tests__`) and `apps/packages/ui/src/entries/shared/__tests__/notification-subscription.test.ts`

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
Built from the `Alert` primitive with `variant="warning"` (it renders `role="alert"`), `action={{ label: "Review sessions", ... }}` and `secondaryAction={{ label: "Dismiss", ... }}`. Plural line: "1 more new sign-in." and "{n} more new sign-ins."

**Placement and gating:** `NotificationRuntimeOwner` (`WebLayout.tsx:~911-925`) wraps the whole app shell, so a block-level `Alert` ahead of `{children}` would sit above a full-viewport shell and break the layout. The banner is therefore `fixed inset-x-0 top-0 z-50 mx-auto max-w-3xl px-4 pt-2` (no layout shift; it overlays the header edge). It renders and fetches only when `useOptionalNotificationLifecycle()` returns a value with `connectionVerified === true` and `state === "active"`. The provider's `enabled` prop is false when the header is hidden (for example `/login`) or in demo mode, and then the lifecycle never leaves `connecting`, so no unauthenticated call is made there.

**Behaviour:**
- On mount, and again on every stream event whose payload `kind` is `new_sign_in` (`event.event === "notification"`; `useOptionalNotificationLifecycle()` supplies `events` and `lifecycleEpoch`, as the toast bridge uses them), the banner calls `listNotifications({ limit: 100, offset: 0, include_archived: false })` from `@web/lib/api/notifications` and `listAuthSessions()`, then `visibleSignInAlerts(items, sessions)`. It shows the newest and adds the "more" line for the rest. When the session list cannot be loaded (API-key connection, error) the alert stays visible: a failure never hides a security alert.
- It is hidden while the route is `/settings/sessions`.
- "Review sessions" navigates to `/settings/sessions` and does not mark the alert read. "Dismiss" calls `dismissNotification(id)`; on `isProtectedAlertError` it keeps the banner and replaces the body with "This alert can be dismissed from a browser that was signed in before it, or after 24 hours."; any other failure shows "Could not dismiss the alert. Try again."
- An owner signed out by the cap sees the alert on the next sign-in: it is stored unread and the first load of the new session fetches it.
- Extension: `handleStreamEvent` skips `notify(...)` for a `new_sign_in` payload when `isOwnSignIn(payload.source_job_id, await listAuthSessions())` is true; a failed lookup raises the OS notification. The extension has no in-page banner: it gets the page (shared registry) and the suppression.

- [ ] **Step 1: Write the failing tests.** Banner test (mock `@web/lib/api/notifications`, `@/services/auth-sessions` and the router; wrap in `NotificationLifecycleProvider` as `notification-toast-bridge.test.tsx:1-60` does):
  - `shows a role=alert banner for an unread new_sign_in from another session`
  - `does not show it when the source session is the current session`
  - `does not show it when IP and device label match the current session` (the mocked inbox list still contains it)
  - `does not show read, dismissed or other-kind notifications`
  - `a streamed new_sign_in notification triggers a reload and shows the banner`
  - `still shows the alert when the session list cannot be loaded`
  - `Review sessions navigates to /settings/sessions and does not mark read`
  - `Dismiss calls dismissNotification and removes the banner`
  - `Dismiss answered with new_sign_in_protected keeps the banner and shows the explanation`
  - `is not rendered on /settings/sessions`
  - `is not rendered, and fetches nothing, when the lifecycle is not active (disabled provider, connecting, auth-required)`
  - inbox: `a refused_ids response leaves the alert unread and the unread count unchanged; a 403 new_sign_in_protected shows the explanation and View still navigates`
  - `two unread alerts show the newest and "1 more new sign-in."`

  Toast bridge: a `notification` stream event with `kind: "new_sign_in"` queues no toast; a `job_completed` one still does. Subscription: a `new_sign_in` event for the current session raises no `notify`; for another session it does; a failing `listAuthSessions` raises it.

- [ ] **Step 2: Run, expect failure**: `cd apps/tldw-frontend && bunx vitest run __tests__/components/new-sign-in-banner.test.tsx __tests__/components/notification-toast-bridge.test.tsx --maxWorkers=2`, then `cd apps/packages/ui && bunx vitest run src/entries/shared/__tests__/notification-subscription.test.ts --maxWorkers=2`.

- [ ] **Step 3: Implement.** **Step 4: Run, expect pass**, plus the existing `notification-rotation.integration.test.tsx`, `Layout.command-palette.integration.test.tsx` and `OptionIndex.loader-focus.integration.test.tsx` (the last two live under `apps/tldw-frontend/__tests__/components/layout/`), which mount `WebLayout`.

- [ ] **Step 5: Commit** `feat(ui): new-sign-in banner, quieter toast and extension notification (TASK-<ID>)`.

---

### Task 10: Docs, OpenAPI, full verification and PR

**Files:**
- Modify: `Docs/API-related/AuthNZ-API-Guide.md` (sessions section, `:227-280`) and its mirror `Docs/Published/API-related/AuthNZ-API-Guide.md`
- Modify: `apps/packages/api-types/openapi.fingerprint.json` (regenerated; `paths.d.ts` does not change because no route is added, but pass both flags to the check)
- Modify: `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`: append a "Changes from the PR 2 plan (2026-11-25)" section recording the proxy-IP finding, the public-IP-only banner suppression, the deferred lock flag, the `logout` guard, the alert on eviction, the live-only revoke-all count and the existing WebUI toast (and mirror it with `refresh_docs_published.sh` if the design is mirrored)

- [ ] **Step 1: Docs.** In the sessions section document: `is_current` and `device_label` on the listing; `?keep_current=true`; the 403 `interactive_session_required` rule for revoke-one, revoke-all and `logout` (with `all_devices: true`, or with a token that has no `session_id`) (a browser session means an access token carrying `session_id`, or a single-user cookie; API keys, virtual keys, service tokens and impersonation tokens are refused); the `new_sign_in` notification (title, severity, link, dedupe bucket) and who may mark it read, dismiss or snooze it (`new_sign_in_protected`, the older-session rule, the 24-hour rule, `refused_ids` on mark-read); and the IP caveat of Question 1 with `AUTH_TRUST_X_FORWARDED_FOR` and `AUTH_TRUSTED_PROXY_IPS`. Then `Helper_Scripts/refresh_docs_published.sh` and stage `Docs/Published`.

- [ ] **Step 2: OpenAPI.** `make openapi-fingerprint` (Makefile `:536`); then `python Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` exits 0.

- [ ] **Step 3: Targeted backend run.** `run_py t9-a` over every test file this PR created, `tldw_Server_API/tests/AuthNZ_SQLite`, `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`, `.../test_session_cap_postgres.py`, `.../test_sessions_api_postgres.py`, `.../test_new_sign_in_alert_postgres.py`, `tldw_Server_API/tests/Notifications`, `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`, and every file from `git grep -l -E "sessions/revoke-all|auth/sessions|users/sessions|create_session\(|NotificationKind|mark-read" -- tldw_Server_API/tests`. PostgreSQL tests must run, not skip. `git status --short` shows nothing new under `Databases/`.

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
  - sessions API (`is_current`, `device_label`, `keep_current`, id read from the bearer token): Tasks 2 and 3 (live-only revoke count included)
  - interactive credential required (revoke-one, revoke-all, dismissing the alert; lock is PR 3): Tasks 2, 3, 4 (`logout`) and 6
  - Active sessions page, account-menu and notification links: Tasks 8 and 9
  - "Is lock available?" flag: deferred to PR 3 (Deviation 1)
- **Security requirement to test:**
  - revoke and dismiss need an interactive session: `test_api_key_cannot_end_sessions`, `test_scoped_virtual_key_cannot_end_sessions`, `test_api_key_as_bearer_cannot_log_everyone_out`, `test_api_key_header_cannot_log_everyone_out_with_all_devices`, `test_service_api_key_and_impersonation_principals_are_not_interactive`, `test_api_key_cannot_silence_even_an_old_alert`, `test_new_sign_in_silence_postgres`
  - a client cannot claim "current": `test_client_cannot_claim_current_session`, `test_claim_naming_another_users_session_is_not_current`
  - cross-user isolation on both backends: `test_one_user_cannot_see_or_end_another_users_sessions` and `test_sessions_api_postgres`, `test_user_cannot_dismiss_another_users_alert`, `test_alert_goes_only_to_the_signing_in_users_inbox`, `test_alert_write_dedupes_and_isolates_users_postgres`
  - no credentials in the alert: `test_alert_carries_no_credentials_or_install_id`
  - nothing new logs tokens, cookies or install ids: `test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids`
- **Names used identically across tasks:** `device_label`, `resolve_current_session_id`, `InteractiveSession`, `require_interactive_session`, `get_current_session_id`, `INTERACTIVE_REQUIRED_CODE`, `write_new_sign_in_alert`, `schedule_new_sign_in_alert`, `drain_pending_alerts`, `caller_may_silence`, `split_silenceable`, `NEW_SIGN_IN_PROTECTED_CODE`, `alert_new_sign_in`, `listAuthSessions`, `revokeAuthSession`, `revokeOtherAuthSessions`, `isOwnSignIn`, `visibleSignInAlerts`, `refused_ids`.

## Decisions the spec left open

1. **No schema change.** Session ids order the silence rule, `source_job_id` carries the session id, and `device_label` is computed on read. Reason: nothing needs a column, and a migration would add a PostgreSQL owner-migration deploy step.
2. **Session id from a verified credential, ownership-checked.** It comes from a signed token claim or the validated cookie and must be in the caller's live list. Reason: `is_current` stays unspoofable even by a validly signed token that names someone else's session.
3. **Dependency wrappers in `API_Deps`, logic in `core/AuthNZ`.** Reason: the layering ratchet forbids `core` importing `api`.
4. **`alert_new_sign_in` is an opt-in flag on `create_session`.** Reason: the admin E2E bootstrap and many tests create sessions and must not write notifications.
5. **Detached task, not FastAPI `BackgroundTasks`.** Reason: three call sites are helpers or the cookie mint with no response object; the task is best-effort and swallows named exceptions.
6. **The alert is also written when the capped insert evicted a session.** Reason: with `MAX_SESSIONS_PER_USER=1` (or any full set) the owner whose session was just evicted has no other live session, yet must still see the sign-in on their next sign-in (controller ruling, Question 3).
7. **The dedupe key is a hash.** Reason: `dedupe_key` is returned by the notifications API and stream, and the install id must never leave the server.
8. **Session ids, not timestamps, order sessions for the silence rule.** Reason: a source session whose row was purged still leaves its id in `source_job_id`; `sessions.id` is `AUTOINCREMENT` on SQLite and `SERIAL` on PostgreSQL, so ids are never reused and a lower id is always an earlier session.
9. **After 24 hours any interactive browser session may dismiss; API keys never.** Reason: the spec says "any browser session", and non-session credentials stay refused.
10. **`mark-read` returns 200 with `refused_ids` when some ids were allowed, and 403 `new_sign_in_protected` when every id was refused; dismiss and snooze return 403.** Reason: the inbox's bulk mark-read keeps working for its other items, and the inbox cannot show a refused alert as read.
11. **`logout` needs an interactive session for `all_devices: true`, and its revoke-all fallback for a token without a `session_id` is removed.** Reason: an API key sent as a Bearer token otherwise signs every session out (Question 2).
12. **Revoke-one of a session the caller does not own stays an idempotent 200.** Reason: unchanged behaviour, and it reveals nothing about other users' ids.
13. **The current row has no Sign out button, and per-row sign-out needs no confirmation.** Reason: the account menu already signs this browser out, and a signed-out browser can sign in again; only the bulk action confirms.
14. **Nav group `setupRecovery`, order 4.** Reason: next to Provider Keys and the server connection, where account access is already configured.
15. **The banner is hidden on `/settings/sessions`.** Reason: that page is the review. Suppression is `is_current` equality plus the design's IP and label match.
16. **The banner reads the first 100 inbox items and filters on the client.** Reason: the list endpoint has no kind filter, and adding one is an API change this PR does not need.
17. **Extension: the page and the OS-notification suppression, no in-page banner.** Reason: the extension has no layout that hosts one.
18. **No ADR.** Reason: see Global Constraints; the owner may overrule and take number 100 after re-checking.
19. **`revoke_all_sessions_for_user` updates live rows only.** Reason: `sessions_revoked` and the "Signed out N other sessions" line were counting dead rows, and the update overwrote the `revoke_reason` of rows already revoked (for example `session_cap`).
20. **IP matches suppress the banner only for public-routable IPs.** Reason: behind the bundled proxy every client shares one address, and the label comes from a client-chosen User-Agent (Question 1).
21. **Alert tests are opt-in.** An autouse fixture silences the scheduler unless a test carries the `sign_in_alerts` marker. Reason: every production sign-in passes the flag and would otherwise write into the real `Databases/` directory from any test that signs a user in twice.
22. **`drain_pending_alerts` polls.** Reason: the alert tasks live on the `TestClient` portal loop, not the pytest-asyncio loop.
23. **The alert message says the label is "as the browser reported it".** Reason: the User-Agent is client-chosen.

## Deviations from spec

1. **The "Is lock available?" capability flag is not built.** The spec lists it under section 2, but lock does not exist until PR 3; a flag with no consumer, and a page line explaining why a nonexistent control is missing, would be dead and misleading. PR 3 adds the flag beside its endpoint and puts its button next to "Sign out all other sessions" (the page leaves that action row as the one place to extend).
2. **The Next.js proxy does not forward the browser's IP.** Unbuildable as written (Question 1): Next's rewrite proxy cannot see the peer address and sets only `x-forwarded-host`. The alert shows the IP the server saw, subject to the existing trusted-proxy settings.
3. **The WebUI already toasts notifications.** The spec says it "only bumps a count"; `NotificationToastBridge` (`apps/tldw-frontend/components/notifications/NotificationToastBridge.tsx`, mounted at `WebLayout.tsx:922`) toasts every stream notification. The plan skips that toast for `new_sign_in` so the banner is the one surface.
4. **`logout` is guarded** (both `all_devices: true` and the no-`session_id` fallback) although the spec's list of guarded actions does not name it (Decision 11).
5. **The alert sets `link_url`, not a `link_type` the inbox recognises.** `resolveRouteForLinkType` in `NotificationsRoute.tsx` knows only reading, note, document, watchlist and job; the inbox's `link_url` branch already navigates to same-origin URLs.
6. **The mark-read response gains `refused_ids`, and a request whose ids are all refused gets 403.** The spec only says such requests are "refused".
7. **Banner suppression by IP is narrowed to public IPs** (Decision 20); the spec says IP and label match.
8. **The alert is also written on eviction** (Decision 6); the spec says "at least one other live session".

## Known limits

- The alert is written when another live session exists or the sign-in evicted one. A user with exactly one session and no eviction (their first sign-in) gets nothing.
- The banner reads the newest 100 inbox items; an unread alert buried under more than 100 newer notifications is not shown (the inbox still lists it).
- Two machines behind one NAT with the same public IP, browser and OS family silence each other's banner (Review Focus 6); the inbox entry and the Active sessions page still show the sign-in. Behind a private or proxy address only session-id equality suppresses.
- Behind the bundled proxy with no reverse proxy in front, the alert and the sessions list show the proxy's address instead of the browser's (Question 1).
- The hosted-mode `/api/auth/*` proxy is outside this repo (`TldwAuth.authPath` rewrites `/api/v1/auth/` to `/api/auth/` in hosted mode), so `is_current`, `device_label` and `keep_current` are unverified there.
- A single-user API key can still mint a new cookie session, so "sign out all other sessions" in single-user mode ends the other browsers but not a script that holds the key; the cap bounds what that script can create.
- Extension single-user connections use an API key, so the page there is read-only and every OS notification is raised (no own-session match is possible).
- The alert write is detached: a process stop between the sign-in response and the write loses that one alert.
