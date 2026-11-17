# TASK-15498 Plan: Telegram tests 403, and no CI shard runs tests/Telegram

**Goal:** Fix the 17 failing Telegram tests (test setup, not product), run `tests/Telegram` in a CI shard, and empty its nine files from `shard_coverage_baseline.txt`.

**Architecture:** The shared-BYOK write path authorizes the actor against persisted `users`/`organizations`/`teams` rows. Four Telegram test files fake the principal with ids that have no rows. One shared fixture in a new `tests/Telegram/conftest.py` backs the principal with real rows. A new `telegram` shard is added to the five `ci.yml` shard matrices.

**Tech Stack:** pytest, FastAPI TestClient, AuthNZ `DatabasePool` (SQLite), `.github/workflows/ci.yml`.

**Spec:** `backlog/tasks/task-15498 - 17-Telegram-tests-fail-with-403-on-dev-and-no-CI-shard-runs-tests-Telegram.md`

Base: `origin/dev` at 3883a73712. Prototype: `proto_15498.patch` (test side only, see "Not prototyped").

## Root cause (measured, with evidence)

Verdict: **test-setup gap. Not a product regression. No owner decision.**

- Reproduced on `origin/dev`: `tests/Telegram` gives 17 failed, 44 passed (`-n 2`, 324s). All 17 are `_seed_telegram_bot` -> `PUT /api/v1/telegram/admin/bot` returning `{"detail":"Not authorized to manage shared BYOK keys"}`.
- Raise site: `AuthnzOrgProviderSecretsRepo._lock_authorized_active_parent_scope` (`app/core/Provider_Credentials/repos/org_provider_secrets_repo.py` L202-262). With `required_authority=PLATFORM_ADMIN` it SELECTs the actor from `main.users`. No row, or inactive, raises `MembershipAuthorizationError`, which `_shared_key_control_http_exception` maps to the 403. It then needs the org/team parent row (else 404 `MembershipScopeNotFound`).
- The Telegram endpoint builds that context from `principal.user_id` only (`telegram_support.telegram_admin_put_bot_impl` -> `_platform_admin_membership_context`). Roles and permissions on the principal are deliberately ignored.
- The four failing files (`test_telegram_commands`, `_jobs_and_delivery`, `_linking_and_policy`, `_webhook`) fake `AuthPrincipal(user_id=202/303/404, roles=["admin"])` with hardcoded scope ids (22, 55, 66, org 1, 11) and no DB rows.
- `test_telegram_admin_api.py` already uses the correct pattern (`ensure_test_user` + `AuthnzOrgsTeamsRepo`) and passes. `tests/AuthNZ_SQLite/test_authnz_org_provider_secrets_repo_sqlite.py::test_claimed_platform_admin_cannot_mutate_shared_secret_or_alias` asserts that a claimed admin without persisted authority is refused. The policy is intentional and tested.
- The DB-authoritative actor-row check predates `f7af8c9181` (2026-08-25): it is already in `a7fa919b80` (2026-08-09, "refactor(authnz): complete membership writer migration"), possibly earlier. `f7af8c9181` and `e57fce5ff4` only tightened it. The Telegram tests went stale because no shard runs them. Use this wording, not "introduced by f7af8c9181", in the task notes.

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`). Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. A new follow-up task takes its id under the session's id lock (the controller does that; write `TASK-<NEW>`).
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server. Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR. (This plan touches no mirrored doc.)
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required. Regenerate it with the script's `--write-baseline`. (Run the ratchet; no listed file is expected to change.)
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. No source function is touched.
- A plan with embedded diffs fails trailing-whitespace when it is committed: keep the plan free of trailing spaces.
- Run pytest with `PYTHONPATH=<plans dir>:<plans dir>/pyext` (the shared venv lacks pytest-mock). Do not use an empty stub.
- Task-specific: the fix is test-only. Do not loosen `_lock_authorized_active_parent_scope`. Do not add a bypass for fake principals.

## Review Focus

- The fixture seeds real rows; it must not weaken or monkeypatch the authorization check.
- `principal.model_copy(update={"user_id": ...})` changes the user id the tests see. Confirm no test asserts the old 202/303/404 ids.
- The `telegram` shard is added to all five matrices (linux 3.12, linux 3.13, macos 3.12, windows 3.12, os-313-release), as every other shard is.
- Shard job duration is unmeasured on CI. Serial `tests/Telegram` took 738 s here at load ~88; the limit is 55 minutes.

## Task 0: Setup

**Files:** none.

- [ ] Mark In Progress: `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15498 -s "In Progress"` then `git add -A backlog/tasks`.
- [ ] `git fetch origin dev && git checkout -b fix/task-15498-telegram-tests origin/dev`; open a draft PR to `dev`.
- [ ] Baseline: `PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 tldw_Server_API/tests/Telegram -q -n 2`. Expected: `17 failed, 44 passed`.

## Task 1: Back the fake principal with real rows (AC#1)

**Files:**
- Create `tldw_Server_API/tests/Telegram/conftest.py`
- Modify the `principal_override` fixture in `test_telegram_commands.py`, `test_telegram_jobs_and_delivery.py`, `test_telegram_linking_and_policy.py`, `test_telegram_webhook.py`

**Interfaces:** fixture `back_principal(principal) -> principal`. It builds a tmp-path `DatabasePool`, monkeypatches `telegram_support.get_db_pool` to it, `ensure_test_user(role="admin")`, `INSERT OR IGNORE` the organization (id `active_org_id` or `org_ids[0]` or 1) and the team (id `active_team_id`) with the ids the tests already use, and returns the principal with the real `user_id`. Full code (verbatim from `proto_15498.patch`, plus one comment line):

```python
"""Shared fixture: back a fake admin principal with real AuthNZ rows.

The shared-BYOK write path (PUT /telegram/admin/bot) authorizes the actor against
the users table and the org/team rows, never against the principal's claims. A
bare ``AuthPrincipal(user_id=202, roles=["admin"])`` therefore gets a 403.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.utils.chatops import telegram_support
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.fixture()
def back_principal(client_user_only, monkeypatch, tmp_path):
    """Return ``back(principal) -> principal`` that seeds the principal's user and scope rows."""
    pool = DatabasePool(Settings(
        AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'telegram_authnz.db'}",
        JWT_SECRET_KEY="x" * 32,
    ))

    async def _get_pool():
        return pool

    # Patches get_db_pool for telegram_support only; the runtime repo uses the global pool, so two pools coexist in one test.
    monkeypatch.setattr(telegram_support, "get_db_pool", _get_pool)

    async def _seed(principal: AuthPrincipal) -> AuthPrincipal:
        await pool.initialize()
        user_id = await ensure_test_user(pool, f"telegram-actor-{principal.user_id}", role="admin")
        org_id = principal.active_org_id or (principal.org_ids or [1])[0]
        await pool.execute(
            "INSERT OR IGNORE INTO organizations (id, name) VALUES (?, ?)",
            org_id, f"telegram-org-{org_id}",
        )
        if principal.active_team_id:
            await pool.execute(
                "INSERT OR IGNORE INTO teams (id, org_id, name) VALUES (?, ?, ?)",
                principal.active_team_id, org_id, f"telegram-team-{principal.active_team_id}",
            )
        return principal.model_copy(update={"user_id": user_id})

    def back(principal: AuthPrincipal) -> AuthPrincipal:
        return client_user_only.portal.call(_seed, principal)

    yield back
    client_user_only.portal.call(pool.close)
```

- [ ] RED (already measured): the baseline above, 17 failed with the 403 detail.
- [ ] Create `conftest.py` and change each `principal_override` to `def principal_override(client, back_principal):` with `_override_principal(client, back_principal(principal))`. No call-site changes: `_seed_telegram_bot` still gets the same arguments.
- [ ] GREEN: run the four files. Expected: 0 failed. Measured: `test_telegram_webhook.py` 9 passed (118s); `test_telegram_linking_and_policy.py` 8 passed serial (395s); full directory with `-n 2`: `60 passed, 1 failed`, see below.
- [ ] Known flake under `-n 2`: `test_linked_user_is_allowed_for_privileged_action` returned 403 `invalid_pairing_code` once, then passed alone (273s) and in the serial file run. Likely cause (not proven): `get_telegram_runtime_repo` uses the global AuthNZ pool, which xdist workers share, so another worker's `clear_all_for_tests` clears the pairing code. CI runs shards serially (no xdist), so it does not apply there. Re-measure serially: `python -m pytest tldw_Server_API/tests/Telegram -q` (no `-n`). Expected: `61 passed`. If it fails serially, stop and debug the runtime-repo pool.

## Task 2: Add the shard, shrink the baseline (AC#2)

**Files:**
- Modify `.github/workflows/ci.yml` (five matrices)
- Modify `Helper_Scripts/ci/shard_coverage_baseline.txt` (remove nine `tests/Telegram/...` lines)

- [ ] RED: with the baseline lines removed and no shard, `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` should list nine unshared files and exit non-zero.
- [ ] After each `- name: auth-unit-m-z` shard block (5 occurrences, lines ~925, 2680, 4243, 5710, 7168), insert:

```yaml
          - name: telegram
            paths: >-
              tldw_Server_API/tests/Telegram
```

- [ ] Remove the nine Telegram lines from the baseline.
- [ ] GREEN: the coverage script exits 0; `python -m pytest tldw_Server_API/tests/CI -q`.
- [ ] Check no other list needs the shard name (summary jobs, timeouts): `grep -n "auth-unit-m-z" .github/workflows/ci.yml` should show only the five matrix entries.

## Task 3: Verify, close out, review, PR

- [ ] Wider set: `git grep -l "telegram_support\|Telegram" tldw_Server_API/tests` and run those serially, including `tests/AuthNZ_Unit/test_telegram_admin_write_authority.py`.
- [ ] `tests/Docs`, `tests/lint`, `tests/CI`; `Helper_Scripts/ci/test_quality_triage.py --enforce`; `check_shard_coverage.py`; `check_backlog_task_ids.py` (exit 0); `git diff --check`. Bandit: no source touched, document the skip.
- [ ] Known environment-only failures: `test_runtime_package_discovery.py` (no setuptools); `test_jobs_postgres_exports_recorded_coverage` x2 (no coverage); docling missing.
- [ ] Backlog: tick AC#1 and AC#2 and the DoD items, `--append-notes` with the root cause above, final summary, status Done (`git add -A backlog/tasks`).
- [ ] Clean-session final review (`/superpowers:requesting-code-review`), then mark the PR ready. Merges go through the controller's queue.
- [ ] Gating step before merge: watch the first PR run of the `telegram` shard. It must be green on all 5 matrices and its duration recorded; the macOS and Windows runs have not been measured.

## Decisions

None required. The 403 is the intended, tested policy (DB-authoritative platform-admin check for shared BYOK writes); only the tests were stale.
1. Org/team admins managing their own Telegram bot: recommend no change now, consistent with shared BYOK (platform admin only); open a follow-up only if the owner wants it.
2. Parallelism: keep the `telegram` shard serial. Note only: each xdist worker would need its own `DATABASE_URL` before the shard is parallelized (the runtime repo uses the shared global pool).
3. Optional, recommend no: a lighter path to avoid per-test DB seeding (override the repo in tests). Not recommended, since it would stop testing the real authorization path.
ADR-004 Change summary waiver: the controller will ask.

## Not prototyped / not measured

- The `ci.yml` edit was not prototyped: the environment denied that edit. Task 2 is specified exactly but unmeasured, including the coverage-script RED and GREEN and the `tests/CI` run.
- The Telegram shard's duration on a CI runner, and its behaviour on macOS and Windows.
- Postgres: the Telegram tests run on SQLite via `client_user_only`; no PostgreSQL file names a symbol touched.
- My first reproduction (17 failed) used an empty `pytest_mock` stub; no Telegram test uses `mocker` (grep found none). The serial reruns used the real pytest-mock.
- The netprobe plugin was not used in my runs; Task 0 and 3 must run with it.
