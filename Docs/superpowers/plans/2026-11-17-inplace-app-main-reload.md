# TASK-15497 In-place app.main reloads: guard, lint, env restore - Implementation Plan

> For agentic workers: use superpowers:executing-plans or superpowers:subagent-driven-development.

**Goal:** A test that reloads `app.main` in place (`importlib.reload(app_main)`) can no longer leave a rebuilt app to
later tests, new ones are kept out, and the two session-scoped env fixtures stop leaking.

**Architecture:** One root fix. `app_main_isolated` (the autouse guard in `tests/conftest.py`) compares module identity
only. An in-place reload keeps the module object and rebinds `app` and every other global, so the guard never fires.
The guard now also snapshots the module namespace and, on exit, puts it back when `app` changed. That covers all 37
reloading files at once, with no edit to any of them. Two files reload at import time (collection), which no per-test
guard can see: they move the reload into a module-scoped fixture that restores. A lint test with a baseline file keeps
new in-place reloads out. The Jobs env fixture goes from session to module scope; the e2e one stays session scoped and
restores through `MonkeyPatch`.

**Tech Stack:** pytest, importlib, FastAPI test app.

**Spec:** `backlog/tasks/task-15497 - Test-files-that-reload-app.main-or-app-modules-in-place-leak-a-rebuilt-app-into-later-tests.md`

## AC corrections (evidence)

- AC #2 says "34 in-place reloads". Counted on `origin/dev` 3883a73712: **37 files, 62 calls** reload `app.main`
  in place (`reload(app_main)`, `reload(main_module)`, `reload(importlib.sys.modules["...app.main"])`). The task's 34
  predates later files (PR D counted 33 + 1). Corrected AC #2: "the in-place reloads (37 files today) are covered by the
  guard; a lint ratchet keeps new ones out".
- AC #1 offers "guard detects and undoes, or every test uses `reload_app_main()`". Converting 37 files is 62 hand
  edits that each need `app_main = reload_app_main()` rebinding; the guard is one 19-line change. Plan takes the guard.
- Not in scope: ~27 `importlib.reload(config_mod)` and other module reloads (audit, auth, collections services...).
  Same class of leak, different modules; see Decisions.
- Import-time reloads are not covered by the guard: `Billing/test_billing_public_api_removed.py:12` and
  `Config/test_route_and_cors_guards.py:33` rebuild `app.main` during collection, for every worker, for the whole
  session. Task text does not mention them; this plan fixes them.

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin:
  `PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and
  logs every outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A
  commercial provider otherwise calls the real API. (None here.)
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and
  out of every average. Never 0.0, never a success. Owner's rule, 2026-11-10 and 2026-11-12. (None here.)
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`).
  Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files.
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server.
  Redis: never FLUSHALL or SCRIPT FLUSH. (None used here.)
- A mirrored doc needs `bash Helper_Scripts/refresh_docs_published.sh` with the `Docs/Published` diff in the same PR.
  (No mirrored doc is touched.)
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required. Check
  with `git grep -n` of each touched path in that file before the PR; regenerate with `--write-baseline` if hit.
- Size-capped functions: `tests/lint/test_function_size_ratchet.py` scans only `tldw_Server_API/app`, so these
  test-only edits cannot trip it (reviewer verified). Run it anyway in Task 2 as a smoke check.
- A plan with embedded diffs fails trailing-whitespace: keep this plan free of trailing spaces.
- Test-only: no app code changes. `app_main_isolated` restores only when `app` changed, so a test that only
  monkeypatches a main global is untouched.
- The new in-place reload test must restore the app in its own right; do not mark any test `skip`.

## Review Focus

1. `restore_app_main_namespace`: restores every global, deletes names the reload added, and only runs when
   `app` identity changed (so ordinary `monkeypatch.setattr(app_main, ...)` is left to monkeypatch's own undo).
2. Fixture ordering: the module-scoped `_reloaded_app_main` runs before the function-scoped autouse guard, so
   the guard's baseline is the reloaded state and its restore lands there, not on the pre-reload app.
3. The Jobs fixture's per-test `_jobs_minimal_env` still sets everything per test, so module scope loses nothing.
4. The e2e fixture stays session scoped on purpose (session fixtures start the server from those vars).
5. The lint baseline: 38 files (37 existing + the guard test's own deliberate polluter).

## Task 0: Setup and baseline

**Files:** none.

- [ ] Mark In Progress: `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15497 -s "In Progress"`
  (check the flag with `--help`); `git add -A backlog/tasks`.
- [ ] `git fetch origin dev && git switch -c test/inplace-app-main-reload-15497 origin/dev`.
- [ ] Commit the status change, `git push -u origin HEAD`, `gh pr create --draft --base dev`.
- [ ] Baseline (serial, `-p no:randomly`, `-n 0`): the five groups listed in Task 3, recorded in the PR body.

## Task 1: Guard undoes an in-place reload (AC #1)

**Files:**
- Modify: `tldw_Server_API/tests/helpers/app_main_state.py`
- Create: `tldw_Server_API/tests/Infrastructure/test_app_main_inplace_reload_guard.py`

**Interfaces:**
- `snapshot_app_main_namespace(module) -> dict | None`: shallow `dict(module.__dict__)`.
- `restore_app_main_namespace(module, saved) -> None`: deletes names not in `saved`, then `vars(module).update(saved)`.
- `app_main_isolated()`: takes `saved` on entry; on exit, if the module object is unchanged and
  `getattr(module, "app", None) is not saved.get("app")`, calls `restore_app_main_namespace`.

- [ ] RED. Create the test file (4 tests: fake-module restore, untouched module left alone, a real
  `importlib.reload(app_main)` polluter, then its victim asserting `app_main.app is _APP_AT_IMPORT`). The patch
  `proto_15497.patch` carries the exact file.
  Run on `origin/dev` helper:
  `... pytest -p no:randomly tldw_Server_API/tests/Infrastructure/test_app_main_inplace_reload_guard.py`
  Expected (measured): `2 failed, 2 passed`; failures `test_guard_restores_app_after_in_place_reload` and
  `test_zz_victim_sees_the_original_app` (`assert <object ...> is <object ...>`), netprobe 0 lines.
- [ ] GREEN. Apply the `app_main_state.py` hunk of the patch.
  Expected (measured): `4 passed` (73 to 150 s, app import dominates), netprobe 0 lines.
- [ ] Real polluter then victim, one process: `tldw_Server_API/tests/Collections/test_reading_highlights_reanchor.py`
  followed by `...guard.py::test_zz_victim_sees_the_original_app`. Expected: RED on the old helper (victim fails on
  `app_main.app is _APP_AT_IMPORT`), GREEN with the patch. Run with `-o timeout=1800` (the default 300 s
  pytest-timeout trips under load). Measured by the reviewer on dev 61ad5280ca: RED `1 failed, 1 passed` (82 s);
  GREEN, reanchor + the whole guard file, `5 passed` (76 s).

## Task 2: Cover import-time reloads and keep new ones out (AC #2)

**Files:**
- Modify: `tldw_Server_API/tests/Billing/test_billing_public_api_removed.py`
- Modify: `tldw_Server_API/tests/Config/test_route_and_cors_guards.py`
- Create: `tldw_Server_API/tests/lint/test_no_new_inplace_app_main_reload.py`
- Create: `tldw_Server_API/tests/lint/inplace_app_main_reload_baseline.txt` (38 lines + 2 comment lines)

- [ ] In both files replace the module-level `app_main = importlib.reload(app_main)` with the module-scoped autouse
  fixture `_reloaded_app_main` (snapshot, `importlib.reload(app_main)`, `yield`, restore). The global `app_main` stays
  valid because the module object does not change. `ALLOWED_ORIGINS` stays set at import (the module needs it to import).
- [ ] Lint: two tests (`test_no_new_inplace_app_main_reload`, `test_baseline_has_no_stale_entries`) scanning
  `tests/**/*.py` except `lint/` and `helpers/` with `\breload\(\s*(app_main|main_module|importlib\.sys\.modules\[)`.
  Baseline generated by the one-off script in the patch notes (38 files).
- [ ] RED: add a throwaway `tests/Infrastructure/test_zz_tmp.py` containing `importlib.reload(app_main)`.
  Expected (measured): `1 failed, 1 passed`. Remove it. GREEN: `2 passed` (23 s).
- [ ] Run both converted files serially, then each followed by the victim, and the function-size ratchet:
  `tests/Billing/test_billing_public_api_removed.py tests/Config/test_route_and_cors_guards.py
  tests/lint/test_function_size_ratchet.py`. Expected: pass, same counts as the Task 0 baseline.

## Task 3: Restore the session env fixtures (AC #3)

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/conftest.py` (`_bootstrap_jobs_routes_env`: session to module scope, `pytest.MonkeyPatch`, undo)
- Modify: `tldw_Server_API/tests/e2e/conftest.py` (`disable_rate_limiting`: stays session scoped, `MonkeyPatch`, undo)

- [ ] Keep the old `setdefault` semantics: set `TEST_MODE`, `MINIMAL_TEST_APP`, `ROUTES_STABLE_ONLY` and
  `PRIVILEGE_METADATA_VALIDATE_ON_STARTUP` only when absent (`if key not in os.environ: mp.setenv(key, value)`), as the
  patch does. A plain `mp.setenv` would overwrite a caller's value; the root conftest deliberately unsets `TEST_MODE`
  for the "Security suites with TEST_MODE unset" CI step (`tests/conftest.py` ~631).
- [ ] RED (throwaway probe, not committed): a test file that records `dict(os.environ)` at collection and asserts equality at
  run time, ignoring `PYTEST_CURRENT_TEST` (pytest sets it per test, so a literal compare fails even when green),
  placed after two Jobs files: `RUN_JOBS=1 pytest tests/Jobs/test_jobs_idempotency_scope_sqlite.py
  tests/Jobs/test_jobs_completion_idempotent_sqlite.py <probe>`.
  Expected (measured) on dev: probe fails with
  `{'ROUTES_STABLE_ONLY': (None, '0'), 'ROUTES_ENABLE': ('workflows,scheduler', 'workflows,scheduler,jobs'), 'PRIVILEGE_METADATA_VALIDATE_ON_STARTUP': (None, '0')}`.
  With the change: `4 passed`. Reviewer also measured the whole non-PG Jobs suite with module scope
  (`RUN_JOBS=1 -m "not pg_jobs" -k "not postgres and not pg_"`): `2478 passed, 2 skipped`.
- [ ] e2e fixture: the old teardown deleted `TESTING` and `TEST_MODE` even when the caller had them. `MonkeyPatch.undo`
  restores the previous values. Not exercised end to end (the e2e suite needs a live server); `python -c` import and
  `pytest --collect-only tldw_Server_API/tests/e2e -q` only. State this in the PR.

## Task 4: Verify, close out, review, PR

- [ ] Wider set, serial, `-p no:randomly -n 0`: every file in the baseline list that is not skipped locally. Measure the
  pass counts on `origin/dev` and with the patch and diff them; at minimum the 12 Collections files, Admin x4, Notes_NEW,
  Notes_Graph, Writing, kanban, Security/test_runtime_fixme_hotspots, Billing, Config (3). Postgres files: none name
  the touched symbols (`git grep -l "app_main_state\|app_main_isolated"` lists SQLite/unit files only); run those hits too.
- [ ] `tests/Docs`, `tests/lint`, `tests/CI`; `python Helper_Scripts/ci/test_quality_triage.py --enforce`;
  `python Helper_Scripts/ci/check_shard_coverage.py` (new files sit under `tests/Infrastructure` and `tests/lint`,
  already in shards); `check_backlog_task_ids.py` exit 0; `bandit` on the touched `.py` (test-only: expect 0 new);
  `git diff --check`; scope-predicate baseline grep.
- [ ] Backlog: tick ACs #1-#3 with the corrected #2 wording, DoD, `--append-notes`, final summary, Done.
- [ ] Clean-session review (`/superpowers:requesting-code-review`), fold findings, then ready the PR; merge through
  the controller's queue. PR body: Why, What changes, Review Focus, Known limits (below).

## Known limits (for the PR body)

- Only `app.main` is covered. Reloads of other modules (`config_mod` x27, audit, auth, collections services) can
  leak the same way and are not guarded.
- The guard acts when `app` changed. A reload that rebinds other globals but leaves `app` identical is not undone.
- A reload in a module- or session-scoped fixture that is not the Billing/Config pair is baselined, not fixed.
- The guard test's own polluter is in the lint baseline.
- The guard restores the module namespace only. A reload also re-runs main.py's module-level side effects
  (`logger.remove()` and sink re-add, `_redirect_external_loggers()`, `_install_stderr_redirect()`), and endpoint
  modules imported under the reloading test's env stay in `sys.modules`. Same as today; not undone.
- If `app.main` was never imported before the first reloading test, the snapshot is `None` and the guard does
  nothing. Collection imports it in practice.
- Pre-existing failures seen in the wider set, on dev too: `Health/...::test_security_critical_when_high_risk_meets_threshold`
  (fails in isolation) and `Tools/test_tools_permissions.py::test_tools_execute_forbidden_without_permission_multi_user`
  `[sqlite|postgres]` (CSRF 403 after Security, Billing, Config, Meetings in one process). Not caused by this PR;
  the Tools one goes to the decision-2 follow-up.

## Decisions (owner)

1. **Scope of the lint ratchet.** Options: (a) baseline the 37 files (this plan; they are covered by the guard, new
   ones fail); (b) convert all 37 to `reload_app_main()` (about 62 edits, each needs `app_main = reload_app_main()`,
   more review surface, no extra protection). Recommend (a).
2. **Other in-place module reloads** (`config_mod` x27 and the rest). Option: file a follow-up generalising the
   guard to snapshot-restore any reloaded module. Cost: a wider hook (`importlib.reload` wrapper like the PR D
   imports fixture). Recommend a follow-up task, not this PR. The follow-up also takes the Tools multi-user CSRF
   403 that reproduces on dev after Security/Billing/Config/Meetings run in one process (likely a config or AuthNZ
   reload leak). Reviewer agrees.
3. **e2e `disable_rate_limiting` stays session scoped** (restores at session end). Option: module scope would
   break the session server fixtures. Recommend keep, documented.
4. ADR-004 Change summary waiver: controller asks.
