# Replace ad hoc test-mode branches with injectable test seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the test-mode branches in `tldw_Server_API/app`, delete the private "is this a test" predicates, move the security-relevant branches out of production code into fixtures and dependency overrides, and make a required gate run the AuthNZ, http_client and Security suites with `TEST_MODE` unset.

**Architecture:** Ratchet first, then one PR per seam, security seams first.
- A new AST ratchet keys two ceilings per module: test-mode reference lines and private predicate definitions.
- A new step in the existing required `security-required` job runs the unit-level AuthNZ suites, http_client and Security with `TEST_MODE`, `TESTING` and `TLDW_TEST_MODE` unset. No ruleset change is needed.
- Each security seam then becomes production behaviour in app code. The test-only behaviour moves to the test side, as one of:
  - a fixture or a root-conftest default;
  - a FastAPI dependency override;
  - a constructor argument.
- The middleware stack stops depending on import timing.
- Last, the 40 private predicates are deleted, and their callers use `core.testing` (one new helper, `is_test_context()`).

**Tech Stack:** Python 3.12/3.13, FastAPI, pytest (`pytest_asyncio`, `pytest_timeout`), AST lint tests under `tldw_Server_API/tests/lint/` sharing `_ast_cache.py` and `_ceilings.py`, GitHub Actions (`security-required.yml`, `backend-required.yml`).

**Spec:**
- `backlog/tasks/task-13950.19 - Replace-ad-hoc-test-mode-branches-with-injectable-test-seams.md`, AC #1-#5.
- The `.19` row of `## Phase 2` in `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`: effort L; one PR slice per seam, security seams first; the `TEST_MODE`-unset job is a required check in the first slice (AuthNZ, http_client and Security tests).

**Conventions:**
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Run every command from the repo root.
- pytest runs as `PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-<name>"`. Do not pass `-p pytest_asyncio.plugin` locally ("Plugin already registered"), except under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` as in C4, which mirrors CI.
- File:line references are on `origin/dev` b9eed09f91. Re-locate them by the quoted text.

## Measured on 2026-10-30 (`origin/dev` b9eed09f91)

- **Reference lines and predicates.** The committed counter (Task 1) counts 710 test-mode reference lines in 185 modules, and 40 private predicates in 36 modules. The backlog's figures were 491 lines in 147 files and "about 33" predicates.
  - The difference is the matcher. This one also counts references to signal-named identifiers (`test_mode`, `_is_test_context`, `is_explicit_pytest_runtime`, ...), `"pytest" in sys.modules`, `TLDW_TEST_*` names and `"tldw_Server_API.tests..."` module sniffing.
  - It does not count imports, comments, docstrings, `core/testing.py` or in-app `tests/` packages.
  - Without the MCP Unified exemption (see Task 1), the predicate count is 44 in 38 modules.
- **Evidence drift** (backlog file:line → now):

  | Backlog evidence | Now on b9eed09f91 |
  |---|---|
  | `tests/conftest.py:598` (`TEST_MODE=1`) | `:631` |
  | `http_client.py:1789-1804` | `:1960-1981`, in `_validate_egress_or_raise` |
  | `auth_deps.py:437-446` (transaction adapter) | `:429-654`, in `get_db_transaction` |
  | `auth_deps.py:739-756` (stub session manager) | `:730-913`, in `get_session_manager_dep` |
  | `auth_deps.py:2157` (ingress skip) | `:2148-2150` |
  | `User_DB_Handling.py:318,330` | `_is_test_context` `:324`, `_is_strict_test_bypass_context` `:336` |
  | `User_DB_Handling.py:1034` | `:1020-1023` |
  | `auth_principal_resolver.py:172` | `:175` |
  | `embeddings_v5_production_enhanced.py:912` | same |
  | `Audit_DB_Deps.py:194` | same |
  | `unified_audit_service.py:1059` | same |
  | `main.py:2629-2649` | `:2618-2797` |
  | `app/__init__.py:46` | same |

- **Re-measured on `origin/dev` 4d0701bece** (2026-10-30, fix round): same counts (710 / 185 / 40 / 36). None of the 10 newer commits touches a file this plan edits.
- **Collection-time imports.** 239 test files import `app.main` at module level (backlog: 233). A plain `git grep` of `^from` lines gives 238, because it misses imports inside module-level `try`/`if`. Measure with this AST scan:

  ```bash
  "$PY" - <<'EOF'
  import ast, pathlib
  def is_main(n):
      return (isinstance(n, ast.ImportFrom) and (n.module == "tldw_Server_API.app.main" or (n.module == "tldw_Server_API.app" and any(a.name == "main" for a in n.names)))) or (isinstance(n, ast.Import) and any(a.name == "tldw_Server_API.app.main" for a in n.names))
  count = 0
  for p in pathlib.Path("tldw_Server_API/tests").rglob("*.py"):
      try:
          body = ast.parse(p.read_text()).body
      except SyntaxError:
          continue
      blocks = [s for s in body if isinstance(s, (ast.Try, ast.If))]
      count += any(is_main(s) for s in body) or any(is_main(n) for b in blocks for n in ast.walk(b))
  print(count)
  EOF
  ```

  Expected: `239`.
- **Security branches found beyond the backlog's list**, on the same code paths:
  - `core/Utils/egress.py:749`: loopback hosts skip the port allow-list under pytest or `TESTING`.
  - `Audio_Files.py:435`, `Video_DL_Ingestion_Lib.py:175` and `:1384`: private-IP blocking is turned off for media URLs under pytest or `TESTING`.
  - `Evaluations/webhook_manager.py:583`: webhook delivery skips the DNS/SSRF validation under pytest.
  - `services/jobs_webhooks_service.py:139`: the Jobs webhook worker skips its egress policy in `TEST_MODE`.
  - `services/workflows_webhook_dlq_service.py:112`: the DLQ worker accepts a pattern-only host match in test contexts.
  - `auth_deps.py:219-291`: in `TEST_MODE`, the single-user key or the test key mints a `["*"]` admin in either auth mode.
  - `auth_principal_resolver.py:342-543`: under pytest, a non-JWT bearer token is tried as a JWT first, and the test key is accepted.
  - `AuthNZ/initialize.py:916-950`: under `TEST_MODE`, `ensure_single_user_rbac_seed_if_needed` stores `SINGLE_USER_TEST_API_KEY` as a non-virtual `scope="admin"` key. `:1281-1287` then lets that key pass `bootstrap_single_user_profile`'s invariant check. `authenticate_api_key_user` falls through to `validate_api_key`, so the seeded key stays an admin credential.
    - Measured: forcing both off and running the 4,100 tests that reach seeding or the test key (every file naming `ensure_single_user_rbac_seed_if_needed`, `bootstrap_single_user_profile` or `SINGLE_USER_TEST_API_KEY`, plus the in-app MCP tests) changed nothing.
    - The root `conftest.py` already makes the configured key equal the test key (`SINGLE_USER_API_KEY = SINGLE_USER_TEST_API_KEY`), so the primary-key seed covers the suite.
  - `mcp_unified_endpoint.py:285` and `MCP_unified/server.py:453`: MCP accepts the test key, and the mounted-HTTP path grants `["*"]` in `TEST_MODE`.
  - `evaluations_auth.py:49,53`: under pytest, `TESTING` or `TEST_MODE` bypasses Evaluations auth.
  - `User_DB_Handling.py:1556-1570`: under pytest, `TESTING` bypasses `get_request_user`.
  - `csrf_protection.py:64`: `TEST_MODE` turns the CSRF default off.
- **What the TEST_MODE-unset run fails today.** Measured with the three flags set to `0`:
  - http_client and Security: 872 tests, 1 failure, `Security/test_runtime_fixme_hotspots.py::test_cors_policy_not_wildcard_in_production`. It sets `ENV=production` with a legacy-format key and now hits the production key-format hard-fail.
  - `AuthNZ/unit` and `AuthNZ_Unit`: 2,747 passed and 1 skipped, in about 5.6 minutes locally either way.
- **Why only unit-level AuthNZ.** The full AuthNZ suite takes about 90 shard-minutes in `ci.yml` (`auth-integration-b-z` alone is 40). The two gates' current timings:
  - `backend-required`: 18-26 minutes of its 30-minute timeout.
  - `security-required`: about 3 minutes of its 20-minute timeout.

  So the new step goes in `security-required` (Task 2), and AC3's "AuthNZ tests" means `tests/AuthNZ/unit` plus `tests/AuthNZ_Unit`.
- **The MCP Unified environment provider is an injected seam, not an ad-hoc predicate.** `MCPServer.dependencies.environment_flags_provider` has two implementations:
  - `core/MCP_unified/environment.py`, the dependency-free default for the standalone package boundary;
  - `TldwEnvironmentFlagsProvider` in `adapters/tldw_runtime.py`, the host adapter that calls `core.testing`.

  The ratchet exempts these two modules from the predicate count (their lines still count).
- **Blast radius of the seam slices.** I forced every Task 3-9 site to its production branch in a scratch tree, except the webhook sites. Then I ran these suites, about 13,500 tests in one session, against dev:
  - http_client, Security, `AuthNZ/unit`, `AuthNZ_Unit`, Audit;
  - MCP_unified (both test trees);
  - Web_Scraping, WebScraping, Media_Ingestion_Modification, Admin;
  - Evaluations (`RUN_EVALUATIONS=1`).

  Dev fails 224 test ids locally; they are pre-existing and unrelated. The forced tree added 75 failing ids and fixed 2.
  - One added id, `Admin/test_admin_ops_new_endpoints.py::TestEmailDeliveries::test_email_delivery_log_cap` (a 60 s timeout), passes alone.
  - Another, `Security/test_runtime_fixme_hotspots.py::test_cors_policy_not_wildcard_in_production`, is the test Task 2 fixes.
  - A second run forced only the three webhook sites, over the 24 test files that reach them (`RUN_JOBS=1 RUN_EVALUATIONS=1`). It added 3.
  - That leaves 77 added ids. Each task's Step 4 lists its share:
    - 32 pin the deleted branch itself; the task rewrites or deletes them.
    - 37 are Evaluations and MCP tests that get 401: 31 from Task 4's sites, 6 from Task 5's.
    - 4 are Audit replay tests, which keep passing because of Task 8's fixture.
    - 2 are resolver tests that send a non-JWT token as a JWT.
    - 1 is the Evaluations webhook test that delivers to loopback.
    - 1 is the CORS test.
  - The suites outside this set (notably AuthNZ integration, about 90 CI shard-minutes) are covered by C6.
- **One order-dependent failure on dev, and its cause.** With `TEST_MODE=true`, `AuthNZ/unit/test_csrf_binding.py::test_csrf_middleware_rotates_unbound_cookie` fails whenever `Security/test_runtime_fixme_hotspots.py` runs before it in the same session (bisected over every `tests/Security` file).
  - Cause: the hotspot tests `importlib.reload` `core/config.py`, which rebinds `config.settings`. `csrf_protection.global_settings` still holds the pre-reload object.
  - The CSRF test sets `CSRF_ENABLED` on the new object, so the middleware reads the old `False` and sets no cookie.
  - Task 2 touches the hotspot file and runs both suites, so Task 2 fixes it. The test sets `CSRF_ENABLED` on `csrf_protection.global_settings`, the object the middleware reads (verified: 15 passed after the hotspot file).
  - The required step does not hit it, because it runs `tests/Security` last.

## Scope rulings

1. **AC3's "security-relevant branches"** are the seven backlog sites, plus the same-path siblings listed under Measured. Without the siblings, the TEST_MODE-unset run would still take the test branch on those paths. The other security-marked reads stay, counted by the ratchet, and Task 13 files them as one follow-up (Inventory, "Deferred").
2. **AC3's "AuthNZ tests"** in the required run are `tests/AuthNZ/unit` and `tests/AuthNZ_Unit`. They run as a step of `security-required`, with its timeout raised from 20 to 40 minutes, so no ruleset change is needed.
3. **Decided (controller, 2026-10-30): the AuthNZ integration suites do not join the required TEST_MODE-unset run.** That would need a new job and a ruleset change. C6 already runs those suites on every seam PR, and once a seam is removed its code no longer reads TEST_MODE.

## Global Constraints

From the roadmap (verbatim):

- **Branching and PR size.** Every PR targets `dev`. One backlog task per PR. A task with more than about 400 changed lines is split into ordered PRs, each passing CI on its own.
- **Ratchet rules.** Every new ratchet:
  - lives in `tldw_Server_API/tests/lint/` (backend) or runs in `frontend-required` (frontend);
  - seeds its baseline from a counter script committed in the same PR (review numbers are matcher-dependent, so the committed script defines the number);
  - keys entries by module or qualified name, never by line number;
  - fails on growth and on a stale entry (an entry whose subject is gone or now under the threshold; size baselines are ceilings, so shrinking needs no baseline edit);
  - runs in a required gate: appended to backend-required's "Enforce CI contracts and code ratchets" step (frontend: frontend-required), with a workflow-contract assertion like `tests/CI/test_backlog_task_format_ratchet.py`; a ratchet that only runs in the optional full-suite shards enforces nothing (today `test_core_to_api_import_boundary.py` is in that state, and `.9` wires it in);
  - ships with a self-test that seeds one violation and asserts the ratchet fails.
- **Raising a baseline.** It is allowed only through a documented bump-with-reason entry, reviewed in the PR, so that security fixes inside frozen functions are not blocked.
- **AST caching.** Backend AST ratchets share one cached parse in `tests/lint/_ast_cache.py`, which `.8` creates and `.9`/`.11` reuse; a full-tree scan takes about 9 s.
- **Line-keyed legacy baseline.** `Helper_Scripts/ci/scope_predicate_baseline.txt` is still keyed by line number until `.23` re-keys it. Any PR that shifts lines in a file it lists regenerates it with `--write-baseline`.
- **Shims.** A move keeps a re-export shim at the old import path until `git grep` shows no importer and no string monkeypatch target still uses it.
- **Postgres proof.** Postgres-dependent acceptance criteria are proven in the Postgres shard with `TLDW_TEST_POSTGRES_REQUIRED=1`. A local skip is not evidence.
- **Flipping a global default** (moderation per-surface flags, production detection, middleware stack, pool backend): `git grep` the symbol across all tests and run every hit.
- **ADR edits** (ADR-003, ADR-025) need `Helper_Scripts/refresh_docs_published.sh` and the `Docs/Published` diff in the same PR.
- **Backlog edits** use `tools/backlog-py` only; stage with `git add -A backlog/tasks`. Cite duplicate ids (TASK-12108, TASK-12013) by title.
- **Review gates.** Each AI-authored PR carries a `## Change summary` (ADR-004) and gets a clean-subagent `/superpowers:requesting-code-review` before merge. Each task's own plan gets an independent review plus a dry run before execution.

The four 2026-10-25 owner decisions (moderation, keywords, scheduler, admin) do not bind this task. The owner waived Change summaries for all TASK-13950 PRs and authorized merging them after green CI and a clean review (2026-10-25); record the summary section anyway.

Controller rulings (2026-10-30, binding), each with its reason:

1. **PR 1 executes only after `.13`'s PRs have landed** (#3374 lands C1, C2, D and D2 together). It seeds its baseline at execution time with `"$PY" -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --write-baseline`.
   - Reason: #3369, #3370 and #3373 delete all 8 test-mode reference lines in `core/Chat/chat_service.py`. A baseline seeded before them would hold an entry those PRs turn stale, which fails backend-required.
   - Expected seed: 710 lines in 185 modules on 4d0701bece. On the `.13` stack head (2f27eedb8a) it is 702 lines in 184 modules, because `core/Chat/chat_service.py` drops out.
   - Predicates are 40 in 36 modules either way.
2. **`.18` owns the service container and its root-fixture reset** (its Task 4, slice C, adds the session-slot reset to `_restore_auth_env_and_singletons`). Task 6 runs after `.18` Task 4 lands. It consumes `app_services().session_manager` and changes no root fixture.
   - Reason: one owner for the container's lifecycle in tests. This replaces the backlog's whole-task dependency on `.18`; the other tasks do not wait for it.
3. **The bootstrap seeding of `SINGLE_USER_TEST_API_KEY` (`initialize.py:916-950`, with its invariant allowance `:1281-1287`) is in scope.** Task 4 deletes it, and its tests prove the test key does not authenticate when no seam injects it.
   - Reason: otherwise the test key stays a stored admin credential after Task 4.
4. **The review's and dry run's Importants are fixed here**: Task 3 runs C6; the required step sets `TLDW_TEST_NO_DOCKER=1`; the contract rejects a neutered step; a function-level zero pin covers the security functions; Task 5 Step 1 and Task 7 Step 1 are corrected.
   - Reason: each was a way for a step or a gate to pass while checking less than the plan claims.
5. **The order-dependent `test_csrf_middleware_rotates_unbound_cookie` failure is fixed in Task 2**, because its cause is CSRF state leaked by a file Task 2 edits (Measured).
6. **The 239 figure has a command** (Measured).
7. **Minor findings are folded**, unless a fix-round note says why not.

Task-specific:

- **Lower the ratchets in every slice.** Each seam or predicate PR ends with two steps:
  - `"$PY" -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --tighten`, committing the baseline diff.
  - Removing the `test_test_seam_env_reads_guarded.py` ALLOWLIST pairs that go stale. Each task names the pairs it expects to drain; the test's stale message is the authority.
- **No new env read of a test signal in app code.** A seam is a fixture, a dependency override, a constructor argument or a root-conftest default. It is never a new `os.getenv`, and never a new name in `GUARDED_TEST_SEAM_KEYS`.
- **The TEST_MODE-unset step stays green in every PR.** It is required from PR 1 on. A slice that breaks a test in it fixes the test in the same PR.
- **Gated suites.** Jobs tests run only with `RUN_JOBS=1`, and Evaluations tests only with `RUN_EVALUATIONS=1`. A slice touching either sets the variable for its local run. A local Postgres skip is not a pass; see C5.
- **No legacy baseline regeneration.** No file this task edits is listed in `scope_predicate_baseline.txt`. Check before each PR with `git diff --name-only origin/dev | sed 's#^#^#' | grep -f - Helper_Scripts/ci/scope_predicate_baseline.txt`, which must print nothing.
- **Function sizes.** Every app edit here deletes code, so a frozen function only shrinks, and no key in `function_size_baseline.json` is re-keyed. If `test_function_size_ratchet.py` warns that a ceiling can be tightened, run its `--tighten` in that PR.
- **Import aliases a deletion leaves unused.** When a deletion leaves a `core.testing` import alias unused in an app module, delete the import. Then delete the test lines that patch that alias, since they no longer change behaviour. Find them with `git grep -n 'setattr(<module alias>, "<alias>"' -- tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests`.
  - Example: on b9eed09f91, 8 test lines patch `auth_deps._is_explicit_pytest_runtime`, `_is_production_like_env` or `_is_test_mode`. Tasks 4 and 6 leave the first two (and `_env_flag_enabled`) unused once both land.
  - Keep an alias that is still used, such as `auth_deps._is_test_mode`, which the diagnostics at `:973-1151` read. The test lines that patch it keep their meaning.
- **Test-quality triage.** New tests must not add `stub_injection`, `mock_density`, `tautology_suspect` or `status_only` offenses.
  - Assert on observable state (a raised status, a resolved URL, a DB row), not on a list of calls a fake collected.
  - Deleting a test of a deleted seam lowers counts. Run `"$PY" Helper_Scripts/ci/test_quality_triage.py --write-baseline` when C2 reports a lower count.

## Review Focus

1. **The TEST_MODE-unset run passes for the wrong reason.** A conftest or fixture may write a flag back into `os.environ`, or the step may stop running a suite. Pinned in Task 2:
   - `test_unset_run_has_no_test_mode_flags` runs last in the step;
   - `test_test_mode_unset_contract_fails_when_the_step_changes` mutates the step eleven ways (dropped suite or sentinel, `-k`, `--ignore`, `|| true`, no pytest, flags set, Docker allowed, continue-on-error, narrowed `if`);
   - `test_root_conftest_drops_the_test_flags_under_the_unset_switch` pins the switch;
   - Task 1's `test_security_functions_read_no_test_signal` covers what the run cannot see: a bypass keyed on `PYTEST_CURRENT_TEST`, which pytest always sets.
2. **The ratchet misses a branch.** The shapes at risk are aliased imports (`is_test_mode as _x`), `self._is_test_mode()`, nested predicates, keys inside tuples, and `"pytest" in sys.modules`. Pinned in Task 1 Step 1 by `test_the_counter_counts_each_reference_shape` and `test_a_new_predicate_growth_and_a_stale_entry_fail`.
3. **A removed branch was the only thing keeping a whole suite working.** The suite-wide defaults at risk are the stub session manager, the transaction adapter, the ingress-limit skip, audit inline I/O and the middleware stack.
   - Each such task runs the Full Suite comparison (C6).
   - The behaviour the suite needs moves into test configuration with a pinning test: `test_suite_defaults_disable_the_auth_fallback_limiter` (Task 7) and `test_suite_audit_services_use_inline_io` (Task 8).
4. **A seam deletion also deletes a production check.** Each security task first pins the production behaviour with a test that runs under pytest (`PYTEST_CURRENT_TEST` set). Examples:
   - `test_validate_egress_never_relaxes_the_private_ip_policy_in_a_test_context` (Task 3);
   - `test_test_key_is_not_accepted_when_it_differs_from_the_configured_key` (Task 4);
   - `test_csrf_default_is_on_when_unconfigured_even_in_test_mode` (Task 5).
5. **A test-mode decision taken at import time.** `main.py` builds its middleware stack from whether `PYTEST_CURRENT_TEST` was set when it was first imported. Pinned in Task 9 by `test_middleware_stack_does_not_depend_on_import_timing`, which reloads `app.main` with and without the variable and compares the stacks.

## CI-equivalent checks (each task names the ones it needs)

| Id | What | Command | Pass means |
|---|---|---|---|
| C1 | backend-required "Enforce CI contracts and code ratchets" | Run the step's two pytest commands from `.github/workflows/backend-required.yml` (`:289-325` on b9eed09f91) verbatim, with `-p pytest_asyncio.plugin` and `-p pytest_timeout` removed and `-p no:cacheprovider --basetemp "$TMPDIR/t1319-c1"` added, under `BACKLOG_TASK_FORMAT_BASE=origin/dev PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1`. The step includes `test_license_first_workflow_contracts.py`, the size, layering and HTTP ratchets, `.6`'s two lints and three guard tests, this task's ratchet (from Task 1), `test_auth_token_parsing_ratchet.py` and `test_websocket_route_auth_matrix.py` | all pass |
| C2 | backend-required "Enforce tenant isolation ratchets" | same env, `tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` | all pass |
| C3 | mypy on changed modules (the CI step is non-blocking) | `uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports $(git diff --name-only --diff-filter=d origin/dev -- 'tldw_Server_API/app/*.py')` | no error on a line this PR changed |
| C4 | security-required "Security suites with TEST_MODE unset" (exists from Task 2) | `TLDW_UNSET_TEST_FLAGS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. DISABLE_HEAVY_STARTUP=1 TLDW_TEST_NO_DOCKER=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-c4" -p pytest_asyncio.plugin -p pytest_timeout --timeout=300 -o timeout_method=signal tldw_Server_API/tests/http_client tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Security tldw_Server_API/tests/CI/test_unset_test_flags_run.py`, with `TEST_MODE`, `TESTING` and `TLDW_TEST_MODE` absent from the shell (`env -u TEST_MODE -u TESTING -u TLDW_TEST_MODE ...`) | all pass (about 3,620 tests, 6-7 minutes locally) |
| C5 | Postgres shard | `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$TMPDIR/t1319-pg" <files>`, with `POSTGRES_TEST_*` set as in ci.yml's Postgres shard, or Docker auto-start (`tests/_plugins/postgres.py`) | pass, and `-rs` lists no skipped Postgres case |
| C6 | Full Suite comparison, a **merge gate** for every task that names it (Tasks 2-9 and 12) | After the PR's `CI` run finishes: `gh run view <pr-run> --log-failed \| grep -oE '(FAILED\|ERROR) tldw_Server_API/[^ ]+' \| sort -u > pr.txt`; the same over the `CI` runs of the three most recent other backend PRs on dev, unioned into `base.txt`; then `comm -23 pr.txt base.txt`. **Flake rule:** before calling a printed id new, re-run it on dev (`git switch --detach origin/dev` in a scratch worktree, then the test alone and in its file). An id that fails there too is not new | prints nothing before merge, and the PR description pastes the comparison |
| C8 | ci.yml shard coverage, for every PR that adds a test file | `"$PY" Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` | exit 0. A new file under `tests/Security` must be added to the `core-security` shard lists (Task 9); the other new files' directories are covered by glob |
| C7 | OpenAPI drift gate (only if a request model or route signature changes; none is planned) | `PYTHONPATH=. "$PY" Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | exit 0 |

## PR slices

| PR | Task(s) | Base | Hand-written lines (est.) | Ratchet change (lines / predicates) |
|---|---|---|---|---|
| 1 | 1, 2 | dev after `.13`'s PRs (#3369, #3370, #3373, #3374) | ~420 + generated baseline (184 entries) | seed 702 / 40 (710 / 40 if `.13` has not landed; ruling 1) |
| 2 | 3 | dev after PR 1 | ~260 | −8 / 0 (measured) |
| 3 | 4 | dev after PR 1 | ~330 | about −25 / −3 |
| 4 | 5 | dev after PR 1 | ~230 | about −12 / −3 |
| 5 | 6 | dev after PR 1 and `.18` Task 4 (slice C) | ~250 (about 190 deleted) | about −4 / 0 |
| 6 | 7 | dev after PR 1 | ~330 (about 230 deleted) | about −4 / 0 |
| 7 | 8 | dev after PR 1 | ~120 | about −8 / 0 |
| 8 | 9 | dev after PR 1 | ~230 (about 90 lines dedented) | about −11 / −1 |
| 9 | 10 | dev after PRs 3, 4 | ~250 | about −18 / −8 |
| 10 | 11 | dev after PR 9 | ~300 | about −25 / −13 |
| 11 | 12 | dev after PR 9 | ~260 | about −20 / −12 |
| (close-out) | 13 | dev after PRs 1-11 | backlog only | 0 predicates |

- PRs 2-8 each need only PR 1, so they merge in any order; PR 5 also waits for `.18` Task 4 (ruling 2). PRs 3, 4, 5 and 6 all edit `auth_deps.py` in different functions; the later one rebases.
- Every PR from 2 on edits `test_mode_branch_baseline.json`. On a rebase conflict, take dev's file, re-run `--tighten`, and re-add your own bump entries if any.
- PR 9 needs PRs 3 and 4. Both edit `User_DB_Handling.py` near `_is_test_context`'s call sites: Task 4 deletes the `:1020` call, and Task 5 deletes `_is_strict_test_bypass_context`. Task 10's call-site list assumes both have landed.
- PRs 10 and 11 need PR 9, which adds `core.testing.is_test_context()`.
- The ratchet numbers are estimates. Each PR records the `--tighten` result in its description.

## Sibling coordination

- **`.18` service container (plan `2026-10-30-task-13950-18-service-container.md`).** Only Task 6 waits for `.18` (ruling 2): it runs after `.18` Task 4 (slice C). Every other seam is a fixture, a dependency override or a constructor argument that works with or without a container. Names used verbatim from `.18`'s Design:
  - `tldw_Server_API.app.core.Utils.app_services`: `app_services() -> AppServices`, with `AppServices.session_manager: SessionManager | None` and `session_manager_lock: asyncio.Lock`;
  - `fresh_app_services()`, an async context manager, and the `fresh_services` fixture from `tests/_plugins/app_services_fixtures.py`.
  - **`.18` Task 4 vs Task 6.** After `.18` Task 4, `get_session_manager()` reads the slot, and `_restore_auth_env_and_singletons` empties it after every test (`.18` Task 4 Files). Task 6 deletes only the stub branch in `get_session_manager_dep`, which then returns `await get_session_manager()`. It changes no root fixture. A test that needs a fake sets `fresh_services.session_manager` or overrides `get_session_manager_dep`.
  - **`.18`'s db_pool slice (Task 3, slice B) vs Task 7.** `auth_deps.get_db_transaction` calls `get_db_pool()`; `.18`'s `database.get_db_transaction` is a different function. Task 7 deletes only the adapter branch, so either order rebases cleanly.
  - **`.18` AC4 (fresh-container fixtures) vs Task 8.** Task 8's audit fixture sets a class attribute, not a container slot. `.18` adds no audit slot (its Design lists `db_pool`, `session_manager`, `jwt_service`, `mcp_server` and `chacha_runtime`).
  - **`.18` Task 1 vs Task 1.** Each adds one ratchet line to `backend-required.yml` and the contract file. Take both.
  - **`.18` F2b vs Tasks 6 and 9.** F2b rewrites two pieces of `tests/AuthNZ/conftest.py`:
    - `reset_singletons` (`:465-600`). Task 9 removes one name from its strip tuple.
    - `isolated_test_environment` (`:1363-1404`). Task 6 deletes the `AUTHNZ_FORCE_REAL_SESSION_MANAGER` setenv beside it (`:1345`).

    Either order rebases.
  - **Disagreements left between the two plans:** none in substance. `.18` cites `get_session_manager_dep` at `auth_deps.py:725-730`; it is at `:730` on dev.
- **`.13` LLM gateway (#3369, #3370, #3373, #3374).**
  - All four edit `tldw_Server_API/tests/conftest.py`: an autouse `_healthy_empty_llm_provider_override_store` fixture near `:927-955`, and a `live_discovery` target at `:330`.
  - #3369 also edits `backend-required.yml` and `test_license_first_workflow_contracts.py`, in the lines next to the ones Tasks 1-2 edit.
  - Task 2 touches other hunks of `tests/conftest.py` (`:631`, `:968`). Either order is a textual rebase.
  - **Baseline interaction (ruling 1).** #3369, #3370 and #3373 each delete all 8 test-mode reference lines in `core/Chat/chat_service.py` (the `_resolve_test_module_provider_api_key` / `use_module_overrides` code at `:2123-2128` and `:2218-2221`). So PR 1 executes after `.13` lands, and seeds with `--write-baseline` at that time.
    - Expected seed, measured on the `.13` stack head 2f27eedb8a: 702 lines in 184 modules, 40 predicates in 36 modules.
    - It is 710 / 185 on 4d0701bece; the 8 `chat_service.py` lines are the whole difference.
  - **Any other in-flight PR** that adds or removes the last test-mode reference in a module after PR 1 lands runs `"$PY" -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --tighten`, or adds a bump entry with a reason. PR 1's description announces the new ratchet and the new required step to open PRs.
- **`.14` schema defaults (not executed).** It moves import-time config reads out of `api/v1/schemas`. `api/v1/schemas/chat_request_schemas.py` has 2 test-mode reference lines; if `.14` moves them, its PR runs `--tighten` or re-keys the entry with a "moved from" reason.
- **`.15` schemas to core (#3377-#3384, S10 later).** No overlap: none of their files is in this plan.
- **`.17` provider credentials (A1 #3379 queued; A2 and B not executed).**
  - A2 moves `core/AuthNZ/byok_testing.py`, and B moves `llm_budget_guard.py` and `llm_budget_middleware.py` (to `core/Usage/`), each behind a `sys.modules` alias.
  - **Whichever lands second re-keys.** If `.17` lands after PR 1, it re-keys those modules' entries in `test_mode_branch_baseline.json` with the reason "moved from core/AuthNZ/<name>.py (<PR>)". The `.17` plan already re-keys `.6`'s allowlist the same way.
  - Task 10 deletes `byok_testing._is_test_mode` in whichever file holds the code on dev at the time.
  - The alias keeps `monkeypatch.setattr(byok_testing, ...)` working, so the retargeted tests work at either path.
- **`.6` (merged).** This plan drains its `test_test_seam_env_reads_guarded.py` ALLOWLIST by the end: 19 `PYTEST_CURRENT_TEST` module pairs (Tasks 3, 4, 8, 9, 10, 11 and 12 name them) and the `_TEST_MODE` main.py entry. It adds nothing to `GUARDED_TEST_SEAM_KEYS`.

---

## Inventory (b9eed09f91, measured by Task 1's counter)

Reference lines per package:

| Package | Lines | Modules | Predicates |
|---|---|---|---|
| `api/v1/endpoints` (top-level modules) | 140 | 20 | 6 |
| `services` | 71 | 28 | 2 |
| `core/AuthNZ` | 67 | 14 | 4 |
| `core/Jobs` | 43 | 2 | 0 |
| `core/Workflows` | 35 | 20 | 0 |
| `api/v1/endpoints/admin` | 33 | 7 | 1 |
| `core/TTS` | 31 | 7 | 4 |
| `api/v1/API_Deps` | 30 | 7 | 1 |
| `core/MCP_unified` | 29 | 6 | 3 |
| `core/DB_Management` | 19 | 8 | 1 |
| app root (`__init__.py`, `main.py`) | 18 | 2 | 2 |
| `api/v1/endpoints/media` | 17 | 6 | 3 |
| `core/Evaluations` | 16 | 7 | 2 |
| `core/Ingestion_Media_Processing` | 16 | 4 | 0 |
| `api/v1/endpoints/evaluations` | 15 | 4 | 3 |
| `core/Watchlists` | 15 | 2 | 1 |
| `core/Chat` | 13 | 4 | 1 |
| `core/Sandbox` | 12 | 2 | 1 |
| core top-level (`http_client.py`, `config.py`) | 11 | 2 | 0 |
| `core/Audit` | 9 | 1 | 0 |
| `api/v1/router_groups`, `core/RAG` | 8 each | 3, 4 | 0 |
| `core/Research` | 6 | 4 | 0 |
| `api/v1/endpoints/audio`, `prompt_studio`; `core/Chunking`, `Slides` | 5 each | 2, 1, 1, 1 | 2, 1, 0, 0 |
| `core/Character_Chat`, `core/Embeddings` | 4 each | 1, 2 | 0, 2 |
| `core/Chatbooks`, `core/Flashcards` | 3 each | 1 each | 0 |
| `api/v1/schemas`; `core/Prompt_Management`, `Resource_Governance`, `Scheduler`, `Utils` | 2 each | 1, 2, 1, 1, 2 | 0 |
| `core/Collections`, `Infrastructure`, `Kanban`, `Monitoring` | 1 each | 1 each | 0 |
| **Total** | **710** | **185** | **40** |

Security-relevant reads, with what this plan does to each:

| Kind | Site | Disposition |
|---|---|---|
| SSRF / egress | `core/http_client.py:1969`; `core/Utils/egress.py:749`; `Audio_Files.py:435`; `Video_DL_Ingestion_Lib.py:175,1384` | Task 3: deleted |
| SSRF / egress | `Evaluations/webhook_manager.py:583`; `services/jobs_webhooks_service.py:139`; `services/workflows_webhook_dlq_service.py:112` | Task 3: deleted |
| Auth bypass | `auth_deps.py:219-291` (test key, `["*"]` admin); `:2826-2829`; `User_DB_Handling.py:1020-1023`; `auth_principal_resolver.py:175,342-543`; `mcp_unified_endpoint.py:285,508-530`; `MCP_unified/server.py:453,489`; the stored test key `AuthNZ/initialize.py:916-950,1281-1287` | Task 4: deleted |
| Auth bypass | `evaluations_auth.py:49,53` and their call sites `:109,159,182`; `User_DB_Handling.py:1556-1570` | Task 5: deleted |
| CSRF | `csrf_protection.py:64` | Task 5: deleted |
| Session revocation | `auth_deps.py:730-913` (stub whose blacklist check is always False) | Task 6: deleted |
| Transactions | `auth_deps.py:429-654` (lightweight adapter) | Task 7: deleted |
| Rate limit off | `auth_deps.py:2148-2150` (ingress fallback skip) | Task 7: deleted; the suite's limit moves to a conftest default |
| Audit | `unified_audit_service.py:1059-1064` (inline flush, no pool) | Task 8: constructor argument plus a conftest fixture |
| Middleware | `main.py:2632-2700` (security headers, metrics, access log and usage logging skipped by import timing) | Task 9: deleted |
| OAuth callback base | `connectors.py:188` (trusts the request base in test mode) | Task 11: rerouted through `core.testing`; behaviour kept, deferred |
| Deferred to a follow-up (Task 13 files it) | listed below the table | Kept as they are; still counted by the ratchet |

The deferred reads:

- **The spoofed-marker path.** In a runtime not labelled production-like, a spoofed `PYTEST_CURRENT_TEST`, with or without `TEST_MODE`, passes `validate_test_runtime_flags()`. That switches on every remaining `is_explicit_pytest_runtime()` / `is_test_context()` branch. Tasks 10-12 route more callers through `is_test_context()`; these reads are those callers or sit beside them:
  - `User_DB_Handling.py:385,1157,1160` (test user ids, and the legacy users-DB fallback inside API-key auth);
  - `endpoints/auth.py:1175` (MFA without Redis);
  - `Sandbox/runners/lima_enforcer.py:41-69` (an env-overridable "deny-all ready");
  - `AuthNZ/initialize.py:763-764,1501`.
- **The other security-marked reads:** rate limiters `Chat/rate_limiter.py:407`, `Character_Chat/character_rate_limiter.py:415-450`, `Evaluations/user_rate_limiter.py:1065`, `Resource_Governance/governor_redis.py:473`; MCP IP filter `MCP_unified/security/ip_filter.py:152`, `request_guards.py:73,98`; file roots `Monitoring/notification_service.py:286`; billing `API_Deps/billing_deps.py:74`, `RAG/rag_service/transport.py:316`; `sandbox.py:1890-1899`; `kanban_deps.py:533`; key fallbacks `AuthNZ/crypto_utils.py:164-199`, `AuthNZ/settings.py:949-1016` (production-refused since `.6`); `AuthNZ/session_manager.py:163,447`; `AuthNZ/database.py:413-426,997-1008`; `AuthNZ/migrations.py:2365,2575,6731`.
- **Leftover references to the removed `AUTHNZ_FORCE_REAL_SESSION_MANAGER`:** `Env_Vars.md:630` and `apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts:73` (Task 6).

### Task 1: The test-mode branch ratchet (AC1, AC5)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`
- Create (generated): `tldw_Server_API/tests/lint/test_mode_branch_baseline.json`
- Modify: `.github/workflows/backend-required.yml`. Add `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py \` after `tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py \` (:311). After the `.16` comment line (:279), add `# The test-mode branch ratchet (TASK-13950.19) does the same for its two ceilings.`
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py:1493-1502`. Append the new path to `TEST_SEAM_GATE_SUITES`, and change the comment's first line to `# TASK-13950.6 and .19: the production-detector, test-seam and test-mode branch ratchets and the startup-guard /`.

**Interfaces:**
- Consumes:
  - `tldw_Server_API.tests.lint._ceilings`: `Ratchet`, `violations`, `reasonless_bumps`, `warn_if_tightenable`, `load`, `main`;
  - `tldw_Server_API.tests.lint._ast_cache`: `APP_ROOT`, `parse_tree`.
- Produces (module `tldw_Server_API.tests.lint.test_test_mode_branch_ratchet`):
  - `SIGNAL_KEYS: frozenset[str] = frozenset({"TEST_MODE", "TESTING", "TLDW_TEST_MODE", "PYTEST_CURRENT_TEST", "pytest"})`.
  - `CORE_TESTING_SIGNALS: frozenset[str] = frozenset({"is_test_mode", "is_explicit_pytest_runtime", "is_test_context", "audio_imports_enabled_for_runtime"})`.
  - `EXEMPT: frozenset[str] = frozenset({"core/testing.py"})`. Not scanned at all.
  - `PREDICATE_EXEMPT: frozenset[str] = frozenset({"core/MCP_unified/environment.py", "core/MCP_unified/adapters/tldw_runtime.py"})`. Lines count; predicates are reported as 0. The module docstring states the reason: these two modules implement `MCPServer.dependencies.environment_flags_provider`, which is an injected seam.
  - `is_signal_name(name: str) -> bool`. Split the name on `_` after `.lower().strip("_")`. True when the words include one of `{"test", "tests", "testing", "pytest"}` and one of `{"mode", "context", "runtime", "pytest"}`.
  - `is_signal_constant(value: object) -> bool`. True for a `str` that is one of:
    - in `SIGNAL_KEYS`;
    - a full match of `TLDW_TEST_[A-Z0-9_]+`;
    - a string starting with `"tldw_Server_API.tests"`.
  - `count_module(tree: ast.Module, *, count_predicates: bool = True) -> dict[str, int]`. Returns `{"lines": n, "predicates": k}`.
    - `n` is the number of distinct line numbers holding a *reference*. A reference is any of:
      - an `ast.Constant` that passes `is_signal_constant`;
      - an `ast.Name` whose id passes `is_signal_name`, is in `CORE_TESTING_SIGNALS`, or is a local name bound by `from tldw_Server_API.app.core.testing import <one of CORE_TESTING_SIGNALS> [as <alias>]`;
      - an `ast.Attribute` whose attr passes `is_signal_name` or is in `CORE_TESTING_SIGNALS`;
      - a sync or async function definition whose name passes `is_signal_name` (its `def` line).
    - Import statements, comments and docstrings never count: alias nodes and non-matching constants are not references.
    - `k` counts the function definitions (sync or async, at any depth) that are predicates. A predicate is a definition whose:
      - name's words include `test`, `tests`, `testing` or `pytest`;
      - return annotation is exactly the name `bool`;
      - body holds at least one reference.
  - `measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]`. For every module under `root`, keyed by its root-relative POSIX path:
    - skip modules with a parent directory named `tests`, and modules in `EXEMPT`;
    - keep only modules where lines or predicates is above 0;
    - call `count_module(..., count_predicates=key not in PREDICATE_EXEMPT)`.
  - `RATCHET = _ceilings.Ratchet(...)`, with:
    - `baseline=Path(__file__).resolve().with_name("test_mode_branch_baseline.json")`;
    - `command="python -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet"`;
    - `fix="Read test mode through core.testing and drive the behaviour from a fixture, a dependency override or a constructor argument instead of a new branch; private 'is this a test' predicates are deleted, not added (TASK-13950.19)."`;
    - `stale="The module no longer reads a test-mode signal."`.
  - `SECURITY_FUNCTIONS: frozenset[str]`, keyed `"<app-relative module>:<qualified name>"`. It is the function-level zero pin (review I7). The module ratchet allows a bump with a reason; this pin has no bump path. PR 1 seeds it with the three security functions that read no test signal on dev:
    - `api/v1/endpoints/evaluations/evaluations_auth.py:get_eval_request_user`;
    - `core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py:_raise_if_url_blocked`;
    - `core/MCP_unified/server.py:MCPServer._single_user_compat_auth_via`.

    Tasks 3-8 each add their functions when the branch is gone (each task lists them). Removing an entry needs a reason in the PR description.
  - `function_reference_lines(tree: ast.Module, qualname: str) -> list[int] | None`.
    - Resolve the dotted `qualname` through module-level classes and functions.
    - Return `None` if it does not resolve to a function.
    - Otherwise return the sorted line numbers of references (as `count_module` defines them, with the module's `core.testing` aliases) inside the function, excluding its own `def` line.
    - The pin sees only the function's own body. A bypass moved into a callee is still counted by the module ratchet.
  - Later tasks run `"$PY" -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --tighten`.

- [ ] **Step 1: Write the self-tests** in the new module. Seed the tree with a `_seed(root, files)` helper, as `test_direct_llm_adapter_calls_ratchet.py` does.

```python
def test_the_counter_counts_each_reference_shape(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/env_reads.py": 'import os\nA = os.getenv("TEST_MODE")\nB = os.environ.get("PYTEST_CURRENT_TEST")\n'
                               'KEYS = ("TESTING", "TLDW_TEST_MODE")\nC = "TLDW_TEST_DB_DIR"\n',
        "core/X/aliased.py": "from tldw_Server_API.app.core.testing import is_test_mode as _tm\n\n\ndef f():\n"
                             "    return 1 if _tm() else 0\n",
        "core/X/predicate.py": 'import os\n\n\ndef _is_test_context() -> bool:\n    return bool(os.getenv("PYTEST_CURRENT_TEST"))\n',
        "core/X/method.py": "class S:\n    def _is_test_mode(self) -> bool:\n        return self.flags.is_test_mode()\n",
        "core/X/nested.py": 'import sys\n\n\ndef outer():\n    def _in_tests() -> bool:\n        return "pytest" in sys.modules\n'
                            "    return _in_tests()\n",
        "core/X/sniff.py": 'X = "tldw_Server_API.tests.helpers"\n',
        "core/X/domain.py": 'def test_connection() -> bool:\n    return True\n\n\nLATEST = "ABTEST"\n',
        "core/X/docs.py": '"""Reads TEST_MODE."""\n# os.getenv("TEST_MODE")\n',
        "core/testing.py": 'import os\nX = os.getenv("TEST_MODE")\n',
        "core/X/tests/test_x.py": 'X = "TEST_MODE"\n',
        "core/MCP_unified/environment.py": 'def is_test_mode() -> bool:\n    return env_flag_enabled("TEST_MODE")\n',
    })

    assert measure(root) == {
        "core/MCP_unified/environment.py": {"lines": 2, "predicates": 0},
        "core/X/aliased.py": {"lines": 1, "predicates": 0},
        "core/X/env_reads.py": {"lines": 4, "predicates": 0},
        "core/X/method.py": {"lines": 2, "predicates": 1},
        "core/X/nested.py": {"lines": 1, "predicates": 1},
        "core/X/predicate.py": {"lines": 2, "predicates": 1},
        "core/X/sniff.py": {"lines": 1, "predicates": 0},
    }


def test_a_new_predicate_growth_and_a_stale_entry_fail(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/a.py": 'import os\n\n\ndef _is_test_mode() -> bool:\n    return os.getenv("TEST_MODE") == "1"\n',
        "core/X/new.py": 'import os\nX = os.getenv("PYTEST_CURRENT_TEST")\n',
    })
    baseline = {
        "core/X/a.py": {"lines": 2, "predicates": 0, "reason": None},
        "core/X/gone.py": {"lines": 1, "predicates": 0, "reason": None},
    }

    problems = _ceilings.violations(RATCHET, measure(root), baseline)

    assert [p.split(":")[0] for p in problems] == ["core/X/a.py", "core/X/gone.py", "core/X/new.py"]
    assert "predicates 1 > ceiling 0" in problems[0]
    assert "stale entry" in problems[1]
    assert problems[2].startswith("core/X/new.py: lines 1, predicates 0, and no ceiling")
```

```python
def test_the_function_pin_finds_a_signal_and_a_missing_function() -> None:
    tree = ast.parse(
        'import os\n\ndef guarded(url):\n    if os.getenv("PYTEST_CURRENT_TEST"):\n        return True\n    return False\n'
        "\nclass C:\n    def clean(self):\n        return 1\n"
    )
    assert function_reference_lines(tree, "guarded") == [4]
    assert function_reference_lines(tree, "C.clean") == []
    assert function_reference_lines(tree, "C.gone") is None


def test_security_functions_read_no_test_signal() -> None:
    trees = parse_tree(APP_ROOT)
    problems = []
    for key in sorted(SECURITY_FUNCTIONS):
        module, qualname = key.split(":")
        lines = function_reference_lines(trees[APP_ROOT / module], qualname)
        if lines is None:
            problems.append(f"{key}: not found; update SECURITY_FUNCTIONS with the new name")
        elif lines:
            problems.append(f"{key}: reads a test-mode signal at lines {lines}; drive it from a fixture instead")
    assert problems == [], "\n".join(problems)
```

  Also add the two ratchet tests, `test_test_mode_branches_do_not_grow` and `test_raised_test_mode_ceilings_carry_a_new_reason`. They are copies of `test_direct_llm_adapter_calls_do_not_grow` and `test_raised_adapter_call_ceilings_carry_a_new_reason` with this module's `RATCHET` and `measure`. End the module with `if __name__ == "__main__": raise SystemExit(_ceilings.main(RATCHET, measure))`.

- [ ] **Step 2: Run them; they fail.**
  Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t1" tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py -k "counter or predicate_growth or function_pin"`
  Expected: 3 FAIL with `NameError`, on `measure`, `RATCHET` or `function_reference_lines`, whichever each test reaches first.

- [ ] **Step 3: Implement the counter** with the signatures above, in a `# --- The counter ---` section above the self-tests. Use one `ast.walk` per module. Collect the alias names first, then the reference line numbers, then the predicates.

- [ ] **Step 4: Run the self-tests; they pass.** Same command. Expected: `3 passed`.

- [ ] **Step 5: Seed the baseline from the counter.**
  Run: `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --write-baseline`
  Expected after `.13` has landed (ruling 1): `tldw_Server_API/tests/lint/test_mode_branch_baseline.json: 184 entries`.

  Then check the totals: `"$PY" -c "import json; d=json.load(open('tldw_Server_API/tests/lint/test_mode_branch_baseline.json')); print(len(d), sum(v['lines'] for v in d.values()), sum(v['predicates'] for v in d.values()), sum(1 for v in d.values() if v['predicates']))"`.
  - Expected `184 702 40 36`, as measured on the `.13` stack head 2f27eedb8a.
  - On 4d0701bece, before `.13`, it is `185 710 40 36`.
  - If dev has moved further, record the new numbers in the PR description; the committed counter defines them.

- [ ] **Step 6: Run the whole module.** Expected: `6 passed`, in about 13 s, most of it the shared app parse. The first round's four tests were measured in a scratch tree on b9eed09f91, and `function_reference_lines` on 4d0701bece: the three seeds give `[]`, and `http_client.py:_validate_egress_or_raise` gives `[1969]`.

- [ ] **Step 7: Wire the ratchet into the gate.** Make the two edits in Files. Then run C1. Expected: all pass. The parametrized `test_test_seam_gate_contract_fails_when_a_suite_leaves_the_step` now has 12 cases (6 suites × 2 mutations), and each new case passes.

- [ ] **Step 8: Commit.**

```bash
git add tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py tldw_Server_API/tests/lint/test_mode_branch_baseline.json .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): freeze test-mode reference lines and private test-mode predicates per module (TASK-13950.19)"
```

### Task 2: The TEST_MODE-unset required run (AC3's job, AC5)

**Files:**
- Modify: `tldw_Server_API/tests/conftest.py`.
  - `:630-631`, inside the existing `try:` block (4-space indent): the unset switch. Replace `    os.environ.setdefault("TEST_MODE", "1")` with:

    ```python
        if os.getenv("TLDW_UNSET_TEST_FLAGS") == "1":
            for _flag in ("TEST_MODE", "TESTING", "TLDW_TEST_MODE"):
                os.environ.pop(_flag, None)
        else:
            os.environ.setdefault("TEST_MODE", "1")
    ```

    Put a comment above it naming security-required's "Security suites with TEST_MODE unset" step and TASK-13950.19.
  - `:968` (`client_with_single_user`): guard the `os.environ.setdefault("TESTING", "true")` with `if os.getenv("TLDW_UNSET_TEST_FLAGS") != "1":`.
- Create: `tldw_Server_API/tests/CI/test_unset_test_flags_run.py`, the sentinel. It lives in `tests/CI` so the step can list it last (C8 covers `tests/CI` by glob).
- Modify: `tldw_Server_API/tests/Security/test_runtime_fixme_hotspots.py:50-68`. In `test_cors_policy_not_wildcard_in_production`:
  - set `SINGLE_USER_API_KEY` to a server-generated-format key, `format_api_key(generate_api_key_id(), generate_api_key_secret())` from `core.AuthNZ.api_key_crypto`;
  - restore it in `finally`, the way the test handles its four variables.

  The test checks CORS. With `ENV=production` and TEST_MODE unset, the suite's legacy key `test-api-key-12345` trips the production key-format and weak-key checks (`settings.py:1025-1039`) before CORS is reached. `TLDW_ALLOW_LEGACY_SINGLE_USER_KEY` alone does not clear the weak-key check.
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py:185` (ruling 5). Import `global_settings` from `tldw_Server_API.app.core.AuthNZ.csrf_protection`, not from `core.config`.
  - That is the object the middleware reads. `test_runtime_fixme_hotspots.py` reloads `core.config`, so the two stop being one object (Measured).
  - Verified: `tests/Security/test_runtime_fixme_hotspots.py` then `tests/AuthNZ/unit/test_csrf_binding.py` gives 15 passed; on dev the second file has 1 failure.
- Modify: `.github/workflows/security-required.yml`.
  - `:152`: `timeout-minutes: 20` → `40`.
  - After the "Enforce high/critical threshold with allowlist" step (ends at `:265`), add the three steps below.
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`.
  - `:1123`: `assert setup_helper_count == 25` → `26`.
  - After `test_test_seam_gate_contract_fails_when_the_step_is_continue_on_error` (`:1550-1560`), add the contract below.
- Modify: `Docs/Development/CI_REQUIRED_GATES.md`. After `### Container Build Check Details`, add `### security-required: TEST_MODE-unset suites`. In six sentences, say:
  - what the step runs, and that `TLDW_UNSET_TEST_FLAGS=1` makes `tests/conftest.py` drop the three flags;
  - that AuthNZ means the unit-level suites, because the integration suites take about 90 shard-minutes;
  - that the run is **not** production parity:
    - `PYTEST_CURRENT_TEST`-keyed branches stay live;
    - the root conftests' other test settings still apply (`WORKFLOWS_EGRESS_BLOCK_PRIVATE=false`, `WORKFLOWS_EGRESS_ALLOWED_PORTS=*`, `WORKFLOWS_WEBHOOK_ALLOWLIST=*`, `MINIMAL_TEST_APP=1`, the single-user test key);
  - that Postgres cases skip (`TLDW_TEST_NO_DOCKER=1`) and run in ci.yml's Postgres shard;
  - that a test failing only here means production code still branches on TEST_MODE, TESTING or TLDW_TEST_MODE.

  This doc has no `Docs/Published` mirror.

The three steps:

```yaml
      # TASK-13950.19: the AuthNZ (unit-level), http_client and Security suites with TEST_MODE,
      # TESTING and TLDW_TEST_MODE unset; tests/conftest.py drops them when TLDW_UNSET_TEST_FLAGS=1.
      # Not production parity: PYTEST_CURRENT_TEST and the root conftests' other test settings
      # still apply. Postgres cases skip here and run in ci.yml's Postgres shard.
      - name: Install PortAudio for the TEST_MODE-unset suites
        if: needs.changes.outputs.backend_changed == 'true'
        uses: ./.github/actions/setup-ffmpeg
        with:
          install-ffmpeg: 'false'
          install-portaudio: 'true'

      - name: Setup Python dependencies for the TEST_MODE-unset suites
        if: needs.changes.outputs.backend_changed == 'true'
        uses: ./.github/actions/setup-python-deps
        with:
          python-version: "3.12"
          use-uv: "true"
          enable-pip-cache: ${{ github.event_name != 'workflow_run' }}
          cache-dependency-path: |
            pyproject.toml
            uv.lock
          extras: dev

      - name: Security suites with TEST_MODE unset
        if: needs.changes.outputs.backend_changed == 'true'
        env:
          PYTHONPATH: .
          TLDW_UNSET_TEST_FLAGS: "1"
          TLDW_TEST_NO_DOCKER: "1"
          DISABLE_HEAVY_STARTUP: "1"
        run: |
          python -m pytest -q --disable-warnings -p pytest_asyncio.plugin \
            -p pytest_timeout --timeout=300 -o timeout_method=signal \
            tldw_Server_API/tests/http_client \
            tldw_Server_API/tests/AuthNZ/unit \
            tldw_Server_API/tests/AuthNZ_Unit \
            tldw_Server_API/tests/Security \
            tldw_Server_API/tests/CI/test_unset_test_flags_run.py
```

**Interfaces:**
- Produces:
  - the env switch `TLDW_UNSET_TEST_FLAGS=1`, honoured by `tests/conftest.py`;
  - the step name `"Security suites with TEST_MODE unset"`;
  - in the contracts module, `TEST_MODE_UNSET_SENTINEL`, `TEST_MODE_UNSET_SUITES: tuple[str, ...]` (the five paths above) and `_test_mode_unset_problems(workflow: dict[str, Any]) -> list[str]`.
- Tasks 3-12 run this step's command (C4) before each PR.

- [ ] **Step 1: Write the failing contract** in `test_license_first_workflow_contracts.py`. The module already imports `shlex`, `yaml`, `Any` and `REPO_ROOT`.

```python
TEST_MODE_UNSET_STEP = "Security suites with TEST_MODE unset"
TEST_MODE_UNSET_SENTINEL = "tldw_Server_API/tests/CI/test_unset_test_flags_run.py"
TEST_MODE_UNSET_SUITES = (
    "tldw_Server_API/tests/http_client",
    "tldw_Server_API/tests/AuthNZ/unit",
    "tldw_Server_API/tests/AuthNZ_Unit",
    "tldw_Server_API/tests/Security",
    TEST_MODE_UNSET_SENTINEL,
)
_TEST_MODE_FLAGS = ("TEST_MODE", "TESTING", "TLDW_TEST_MODE")
_NARROWING_ARGS = ("-k", "-m", "--deselect", "--ignore", "--co", "--collect-only", "--lf", "--last-failed")
_SHELL_SEPARATORS = {"||", "&&", "|", ";"}
_SWALLOWING_WORDS = {"||", "+e", "exit"}


def _test_mode_unset_problems(workflow: dict[str, Any]) -> list[str]:
    job = workflow["jobs"]["security-required"]
    steps = [step for step in job["steps"] if step.get("name") == TEST_MODE_UNSET_STEP]
    if len(steps) != 1:
        return [f"expected one {TEST_MODE_UNSET_STEP!r} step, found {len(steps)}"]
    step, problems = steps[0], []
    if step.get("continue-on-error", False):
        problems.append("step is continue-on-error")
    if step.get("if") != "needs.changes.outputs.backend_changed == 'true'":
        problems.append("step does not run on every backend change")
    env = step.get("env", {})
    for name in ("TLDW_UNSET_TEST_FLAGS", "TLDW_TEST_NO_DOCKER"):
        if str(env.get(name)) != "1":
            problems.append(f"step does not set {name}=1")
    scopes = {"workflow": workflow.get("env", {}), "job": job.get("env", {}), "step": env}
    problems += [f"{scope} env sets {flag}" for scope, values in scopes.items() for flag in _TEST_MODE_FLAGS if flag in values]
    words = shlex.split(step.get("run", "").replace("\\\n", " "), comments=True)
    if "pytest" not in words:
        return problems + ["step does not invoke pytest"]
    args = words[words.index("pytest") + 1 :]
    args = args[: next((i for i, word in enumerate(args) if word in _SHELL_SEPARATORS), len(args))]
    problems += [f"step narrows the run with {arg}" for arg in args if arg.startswith(_NARROWING_ARGS)]
    problems += [f"step can swallow a failure with {word}" for word in words if word in _SWALLOWING_WORDS]
    problems += [f"step does not run {suite}" for suite in TEST_MODE_UNSET_SUITES if suite not in args]
    if TEST_MODE_UNSET_SENTINEL in args and args[-1] != TEST_MODE_UNSET_SENTINEL:
        problems.append(f"step does not run {TEST_MODE_UNSET_SENTINEL} last")
    return problems


@pytest.mark.unit
def test_security_required_runs_the_test_mode_unset_suites():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/security-required.yml").read_text())
    assert _test_mode_unset_problems(workflow) == []  # nosec B101 - regression assertion


@pytest.mark.unit
def test_root_conftest_drops_the_test_flags_under_the_unset_switch():
    text = (REPO_ROOT / "tldw_Server_API/tests/conftest.py").read_text()
    assert 'if os.getenv("TLDW_UNSET_TEST_FLAGS") == "1":' in text  # nosec B101 - regression assertion
    assert 'for _flag in ("TEST_MODE", "TESTING", "TLDW_TEST_MODE"):' in text  # nosec B101 - regression assertion
    assert (REPO_ROOT / TEST_MODE_UNSET_SENTINEL).is_file()  # nosec B101 - regression assertion


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("drop-suite", "step does not run tldw_Server_API/tests/AuthNZ_Unit"),
        ("drop-sentinel", f"step does not run {TEST_MODE_UNSET_SENTINEL}"),
        ("narrow-k", "step narrows the run with -k"),
        ("narrow-m", "step narrows the run with -m"),
        ("deselect", "step narrows the run with --deselect=tldw_Server_API/tests/Security/test_csrf_binding.py"),
        ("ignore", "step narrows the run with --ignore=tldw_Server_API/tests/Security"),
        ("or-true", "step can swallow a failure with ||"),
        ("set-plus-e", "step can swallow a failure with +e"),
        ("no-pytest", "step does not invoke pytest"),
        ("set-test-mode", "step env sets TEST_MODE"),
        ("job-testing", "job env sets TESTING"),
        ("no-docker-dropped", "step does not set TLDW_TEST_NO_DOCKER=1"),
        ("continue-on-error", "step is continue-on-error"),
        ("narrow-if", "step does not run on every backend change"),
    ],
)
def test_test_mode_unset_contract_fails_when_the_step_changes(mutation: str, expected: str):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/security-required.yml").read_text())
    job = workflow["jobs"]["security-required"]
    step = next(s for s in job["steps"] if s.get("name") == TEST_MODE_UNSET_STEP)
    edits = {
        "drop-suite": lambda: step.update(run=step["run"].replace("tldw_Server_API/tests/AuthNZ_Unit", "")),
        "drop-sentinel": lambda: step.update(run=step["run"].replace(TEST_MODE_UNSET_SENTINEL, "")),
        "narrow-k": lambda: step.update(run=step["run"].replace("pytest -q", 'pytest -k "not csrf" -q')),
        "narrow-m": lambda: step.update(run=step["run"].replace("pytest -q", 'pytest -m "not slow" -q')),
        "deselect": lambda: step.update(run=step["run"].replace("pytest -q", "pytest --deselect=tldw_Server_API/tests/Security/test_csrf_binding.py -q")),
        "ignore": lambda: step.update(run=step["run"].replace("pytest -q", "pytest --ignore=tldw_Server_API/tests/Security -q")),
        "or-true": lambda: step.update(run=step["run"].rstrip() + " || true\n"),
        "set-plus-e": lambda: step.update(run="set +e\n" + step["run"]),
        "no-pytest": lambda: step.update(run="echo skipped\n"),
        "set-test-mode": lambda: step["env"].update(TEST_MODE="true"),
        "job-testing": lambda: job.setdefault("env", {}).update(TESTING="true"),
        "no-docker-dropped": lambda: step["env"].pop("TLDW_TEST_NO_DOCKER"),
        "continue-on-error": lambda: step.update({"continue-on-error": True}),
        "narrow-if": lambda: step.update({"if": "needs.changes.outputs.security_relevant_changed == 'true'"}),
    }
    edits[mutation]()
    assert _test_mode_unset_problems(workflow) == [expected]  # nosec B101 - regression assertion
```

  These run in backend-required's contracts step (the file is already on its first line), so dropping the security-required step fails a required gate.
  - Deleting the sentinel file fails twice: the step itself (pytest exits 4 on a missing path) and `test_root_conftest_drops_the_test_flags_under_the_unset_switch`.
  - Deleting the conftest switch fails that test, and the sentinel's run in the step.

- [ ] **Step 2: Run them; they fail.**
  Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t2" tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k "test_mode_unset or unset_switch"`
  Expected: 16 FAIL.
  - The first fails with `["expected one 'Security suites with TEST_MODE unset' step, found 0"]`.
  - The conftest test fails on its first assertion.
  - The 14 mutation cases fail with `StopIteration`.

- [ ] **Step 3: Write the sentinel test**, `tldw_Server_API/tests/CI/test_unset_test_flags_run.py`.

```python
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        os.getenv("TLDW_UNSET_TEST_FLAGS") != "1",
        reason="env gate: runs only in security-required's TEST_MODE-unset step (TASK-13950.19)",
    ),
]


def test_unset_run_has_no_test_mode_flags() -> None:
    assert [flag for flag in ("TEST_MODE", "TESTING", "TLDW_TEST_MODE") if flag in os.environ] == []
    assert testing.is_test_mode() is False
```

  `testing` is `tldw_Server_API.app.core.testing`. The step lists this file last, so it runs after every suite in the step. So it catches a fixture anywhere in them that wrote a flag into `os.environ` without monkeypatch.

- [ ] **Step 4: Add the conftest switch** (the two `tests/conftest.py` edits in Files).

- [ ] **Step 5: Run C4; it shows the one known failure.**
  Expected: `1 failed`, `Security/test_runtime_fixme_hotspots.py::test_cors_policy_not_wildcard_in_production`, with `ValueError: In a production-like runtime (ENV), SINGLE_USER_API_KEY must use the server-generated format`. About 3,620 tests pass, and the sentinel test passes.
  - Measured on b9eed09f91 with the three flags set to `0`: 1 failure; 872 + 2,747 tests.
  - The dry run reproduced it with the flags truly unset, on 4d0701bece.
  - If more tests fail with the flags truly unset, fix each in the test itself, in this PR. Listing them in a skip file is not an option.

- [ ] **Step 6: Fix the CORS test and the CSRF test** (the two Files edits).

- [ ] **Step 7: Run C4 again, then the same suites with the switch off and `tests/Security` before the AuthNZ suites.**
  The second run: `PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1 TLDW_TEST_NO_DOCKER=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t2b" tldw_Server_API/tests/http_client tldw_Server_API/tests/Security tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/CI/test_unset_test_flags_run.py`.
  Expected: both pass. In the second run, the sentinel test is skipped.
  - Measured on dev without this task, the second run's order has one failure: `test_csrf_middleware_rotates_unbound_cookie` (dry run: 1 failed, 3,618 passed, 2 skipped). The CSRF fix removes it.
  - Measured on 4d0701bece in a scratch tree with exactly this task's edits (switch, sentinel in `tests/CI`, server-format CORS key, CSRF import, workflow step and contract). Each run was alone; two concurrent runs in different worktrees made `test_user_endpoints.py` setups collide.
    - C4 (the step's command): 3,620 passed and 1 skipped (`AuthNZ_Unit/test_byok_runtime.py:2706`, an aiosqlite capability check), in 7 min 8 s.
    - Second run: 3,619 passed and 2 skipped (that one and the sentinel), in 6 min 10 s.
    - The contract file: 72 passed, and the 16 new cases fail as Step 2 says when the step and switch are absent.

- [ ] **Step 8: Add the workflow steps and the doc section** (Files). Run C1.
  Expected: all pass, including:
  - `test_admitted_jobs_restore_but_cannot_save_shared_caches` at 26;
  - the 16 new contract cases;
  - `test_ordinary_pr_workflow_inventory_and_direct_triggers_are_frozen`, which still passes because the triggers are unchanged.

  If `actionlint` is installed, run `actionlint .github/workflows/security-required.yml`. Expected: no output.

- [ ] **Step 9: Run C8, commit, then run C6 before merge.** `tests/conftest.py` is the root test conftest, so the Full Suite comparison applies even though the switch is inert unless set.

```bash
git add tldw_Server_API/tests/conftest.py tldw_Server_API/tests/CI/test_unset_test_flags_run.py tldw_Server_API/tests/Security/test_runtime_fixme_hotspots.py tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py .github/workflows/security-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py Docs/Development/CI_REQUIRED_GATES.md
git commit -m "ci(security): run the AuthNZ unit, http_client and Security suites with TEST_MODE unset in security-required (TASK-13950.19)"
```

  On the PR, confirm that security-required ran the new step: `gh run view <security-required run> --log | grep -c "Security suites with TEST_MODE unset"` is at least 1, and the step's summary line shows passed tests, not `no tests ran`. PR 1's description also announces the new ratchet and the new required step to open PRs (Sibling coordination, `.13`).

### Task 3: SSRF seams: outbound fetch and webhook delivery (AC3)

**Files:**
- Modify: `tldw_Server_API/app/core/http_client.py`.
  - `:1960-1981` in `_validate_egress_or_raise`: delete the "In test environments..." comment, `block_override`, and the `if is_explicit_pytest_runtime() or env_flag_enabled("TESTING") or is_test_mode():` block. Keep the `host = ...` lines; `cache_host` uses them.
  - `:1991`: `policy_kwargs: dict[str, Any] = {}`.
  - `:137-142`: drop `env_flag_enabled`, `is_explicit_pytest_runtime` and `is_test_mode` from the `core.testing` import. `is_truthy` stays; it has 6 other uses.
- Modify: `tldw_Server_API/app/core/Utils/egress.py:749-750`. Delete the loopback `allowed_ports = []` override.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py:433-437`. The body becomes `result = evaluate_url_policy(url)`, then the existing check.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py`.
  - `:174-177`: delete `_egress_block_private_override`.
  - `:181-184`: `policy_result = evaluate_url_policy(url)`.
  - `:1382-1386`: `policy_result = evaluate_url_policy(video_input)`.
- Modify: `tldw_Server_API/app/core/Evaluations/webhook_manager.py:578-592` in `_deliver_webhook`. Delete `skip_dns` and its comment; always run `resolve_safe_delivery_target_async(url)`.
- Modify: `tldw_Server_API/app/services/jobs_webhooks_service.py:138-160`. Delete `_is_test = is_test_mode()` / `if not _is_test:` and dedent the policy check. Keep the `:111` cursor rule, which is not security.
- Modify: `tldw_Server_API/app/services/workflows_webhook_dlq_service.py:110-128`. Delete the "In test contexts (no DNS)" fallback, keeping one `return False`, not two. Then drop `is_explicit_pytest_runtime` and `is_test_mode` from its imports; nothing else in the module uses them.
- Modify: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`. Add to `SECURITY_FUNCTIONS`:
  - `core/http_client.py:_validate_egress_or_raise`;
  - `core/Utils/egress.py:evaluate_url_policy`;
  - `core/Ingestion_Media_Processing/Audio/Audio_Files.py:_validate_outbound_url`;
  - `core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py:process_single_video`;
  - `core/Evaluations/webhook_manager.py:WebhookManager._deliver_webhook`;
  - `services/workflows_webhook_dlq_service.py:_host_allowed`.

  Measured on dev, each holds exactly the reference this task deletes (`:1969`, `:749`, `:435`, `:1384`, `:583` and `:112`). `jobs_webhooks_service.py:run_jobs_webhooks_worker` is not added: it keeps the non-security cursor read at `:111`.
- Test: `tldw_Server_API/tests/http_client/test_http_client_truthiness_flags.py`.
- Test: `tldw_Server_API/tests/Security/test_egress_no_test_context_relaxation.py` (new).
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py`.
- Test: `tldw_Server_API/tests/Workflows/test_webhook_dlq_worker.py`.

**Interfaces:**
- Consumes: Task 1's `--tighten`; C4.
- Produces: no new names. `evaluate_url_policy` and `_validate_egress_or_raise` keep their signatures.

- [ ] **Step 1: Pin the production behaviour under pytest, with failing tests.** Each test runs with `PYTEST_CURRENT_TEST` set (pytest sets it) plus `TESTING=1` and `TLDW_TEST_MODE=y`.
  - In `test_http_client_truthiness_flags.py`, replace `test_validate_egress_treats_tldw_test_mode_y_as_test_context` with `test_validate_egress_never_relaxes_the_private_ip_policy_in_a_test_context`.
    - Keep its fake policy, but change the signature to `(url, **kwargs)` and have it store `captured["kwargs"] = kwargs`.
    - Keep `PYTEST_CURRENT_TEST` set.
    - It asserts `captured["kwargs"].get("block_private_override") is None`.
    - In `test_validate_egress_keeps_ip_policy_for_literal_ip_in_test_context`, change the fake the same way, and change its assertion to `captured["kwargs"].get("block_private_override") is None`.
  - `test_egress_no_test_context_relaxation.py`, with `WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443`:
    - `test_loopback_hosts_keep_the_port_policy_under_pytest`, with `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false` (the suite default): `evaluate_url_policy("http://localhost:11434/x").reason_code == "port_not_allowed"`. On dev the URL is allowed.
    - The next two set `WORKFLOWS_EGRESS_BLOCK_PRIVATE=true`. A scratch check confirmed the dev and fixed results below.
    - `test_audio_url_validation_blocks_private_addresses_under_pytest`: `Audio_Files._validate_outbound_url("http://10.0.0.5/a.mp3")` returns a non-empty reason string.
    - `test_video_url_validation_blocks_private_addresses_under_pytest`: `pytest.raises(ValueError, match="blocked by security policy")` around `Video_DL_Ingestion_Lib._raise_if_url_blocked("http://10.0.0.5/v.mp4")`.
  - In `test_evaluations_core_hardening.py`, add `test_registered_webhook_delivery_resolves_the_target_under_pytest`. It copies `test_registered_webhook_delivery_uses_resolved_safe_target` (`:177-218`) without `monkeypatch.delenv("PYTEST_CURRENT_TEST", ...)`, and asserts `kwargs["url"] == "http://93.184.216.34/webhook"` and `kwargs["headers"]["Host"] == "attacker.test"`.
  - In `test_webhook_dlq_worker.py`, replace `test_host_allow_deny_logic_uses_test_mode_single_letter_y_fallback` (`:27-38`) with `test_host_allowed_has_no_test_context_fallback`. Same setup, except it does not patch `dlq_mod.is_explicit_pytest_runtime`. It asserts `dlq_mod._host_allowed("https://foo.ok.test/h", "default") is False`.

- [ ] **Step 2: Run them; they fail.**
  Run: `RUN_EVALUATIONS=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t3" tldw_Server_API/tests/http_client/test_http_client_truthiness_flags.py tldw_Server_API/tests/Security/test_egress_no_test_context_relaxation.py tldw_Server_API/tests/Evaluations/test_evaluations_core_hardening.py tldw_Server_API/tests/Workflows/test_webhook_dlq_worker.py -k "never_relaxes or under_pytest or no_test_context_fallback"`
  Expected: 6 FAIL. On dev:
  - the http_client override is `False`;
  - the loopback URL and both private-address URLs are allowed;
  - the webhook URL is unresolved (`http://attacker.test/webhook`);
  - the DLQ check returns `True`.

- [ ] **Step 3: Delete the branches** (the Files list).

- [ ] **Step 4: Run the Step 2 command; all pass.** Then run every test that reaches these modules:
  - `RUN_EVALUATIONS=1 RUN_JOBS=1 ...` over `tldw_Server_API/tests/http_client`, `tldw_Server_API/tests/Security`, `tldw_Server_API/tests/Web_Scraping`, `tldw_Server_API/tests/WebScraping`, `tldw_Server_API/tests/Media_Ingestion_Modification`, `tldw_Server_API/tests/Workflows`, `tldw_Server_API/tests/Evaluations`;
  - `$(git grep -l "jobs_webhooks_service\|_validate_outbound_url\|_raise_if_url_blocked\|webhook_manager" -- 'tldw_Server_API/tests/*test_*.py')`;
  - every test that turns the suite's egress relaxation back off: `$(git grep -l "WORKFLOWS_EGRESS_BLOCK_PRIVATE\|WORKFLOWS_EGRESS_ALLOWED_PORTS" -- 'tldw_Server_API/tests/*test_*.py' 'tldw_Server_API/app/core/MCP_unified/tests/*.py')`. These now resolve DNS for made-up hostnames. The list covers `Agent_Client_Protocol`, `Audio`, `Audio_Studio`, `Chat_NEW`, `Config`, `Ingestion_Sources`, `Slides`, `TTS_NEW`, `unit` and the in-app MCP tests, which the forced-tree run did not.

  Expected: no failure beyond those the same command shows on dev. Diff the `FAILED` lines against a dev run.
  - A test that sets `WORKFLOWS_EGRESS_BLOCK_PRIVATE=true` and uses a made-up hostname now resolves DNS. Fix it in the test: `monkeypatch.setattr(egress, "_resolve_host_ips", lambda host, **_k: ["93.184.216.34"])`, the convention in `tests/Admin/test_admin_webhooks_service.py:542`.
  - A test that delivers an Evaluations webhook to a made-up or loopback host stubs `webhook_validator.resolve_safe_delivery_target_async`, as the existing hardening tests stub `socket.getaddrinfo`.
  - Measured in the forced tree, beyond Step 1's tests:
    - `Jobs/test_jobs_webhooks_worker.py::test_webhooks_tldw_test_mode_y_skips_egress_policy_check` pins the deleted skip. Rename it `test_webhooks_enforce_the_egress_policy_in_test_mode`.
      - Drop the call counter.
      - Capture loguru with `logger.add(messages.append, level="WARNING")`, removing the sink in `finally`, as `tests/Security/test_security_headers_middleware.py:11-17` does.
      - Assert that a message contains `URL not allowed by egress policy (blocked-by-test)`. The worker logs that warning when it refuses to start (`jobs_webhooks_service.py:145`).
    - `Evaluations/integration/test_api_endpoints.py::TestWebhookEndpoints::test_webhook_delivery_on_evaluation` delivers to its own loopback receiver. Inside that test only, monkeypatch `webhook_validator.resolve_safe_delivery_target_async` to return `(url, {})`, with a comment saying why.

- [ ] **Step 5: Lower the ratchets.**
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_test_mode_branch_ratchet --tighten`.
  - Remove the stale ALLOWLIST pairs `PYTEST_CURRENT_TEST` → `core/Utils/egress.py` and `core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py`. `Audio_Files.py` keeps its pair because of the `:1143` and `:1221` reads.
  - Run C1, C2 and C4. Expected: pass.
  - Run C3. Expected: no error on a changed line.

- [ ] **Step 6: Commit, then run C6 before merge.** Every outbound `http_client` call in every test goes through `_validate_egress_or_raise`.

```bash
git add tldw_Server_API/app/core/http_client.py tldw_Server_API/app/core/Utils/egress.py tldw_Server_API/app/core/Ingestion_Media_Processing tldw_Server_API/app/core/Evaluations/webhook_manager.py tldw_Server_API/app/services/jobs_webhooks_service.py tldw_Server_API/app/services/workflows_webhook_dlq_service.py tldw_Server_API/tests tldw_Server_API/tests/lint/test_mode_branch_baseline.json
git commit -m "fix(security): egress and webhook SSRF checks no longer relax under pytest, TESTING or TEST_MODE (TASK-13950.19)"
```

### Task 4: Single-user test key, non-JWT bearer handling and MCP compat (AC3)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`.
  - `:219-291` in `_authenticate_api_key_from_request`: delete the whole `test_mode` block, so the function starts at the `try:` that calls `get_api_key_manager()`.
  - `:312-313` in the same function: delete the `if _is_test_mode(): logger.exception(...)` arm, and keep its `else` body (the type-only `logger.error`) unconditional. The zero pin covers this function, and the type-only log is the one that leaks nothing.
  - `:2826-2829`: delete the `if _is_test_mode(): test_key = ...` lines.
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`.
  - `:1020-1023` in `authenticate_api_key_user`: delete the test-key lines.
  - `:1004-1008`: the docstring drops "/ SINGLE_USER_TEST_API_KEY".
- Modify: `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`.
  - Delete `_is_test_context` (`:175-179`) and `test_context = ...` (`:342`).
  - `:356`: `elif not token_is_jwt:`.
  - `:484-486`: delete the `elif raw_token and not token_is_jwt and test_context:` arm.
  - `:490-492`: delete the `if ... test_context:` arm of the `except Exception` handler; keep its `else` body, dedented.
  - `:540-543`: delete the test-key lines.
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`.
  - Delete `_single_user_test_key_compat_allowed` (`:285-322`).
  - In `_resolve_token_data_compat` (`:508-530`): delete `test_mode`, the test-key lines and the `auth_via = "single_user_test_api_key"` lines, and set `perms = []`.
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`.
  - Delete `_single_user_test_key_compat_allowed` (`:453-476`).
  - In `_single_user_compat_auth_via`, delete `test_key` (`:485`) and the `elif test_key ...` arm (`:489-490`).
- Modify: `tldw_Server_API/app/core/AuthNZ/initialize.py` (ruling 3).
  - `:916-950` in `ensure_single_user_rbac_seed_if_needed`: delete the `test_api_key = os.getenv("SINGLE_USER_TEST_API_KEY")` / `if test_mode and test_api_key:` block that stores the test key as a `scope="admin"` key.
  - `:1281-1287` in `bootstrap_single_user_profile`: delete the test-key addition to `allowed_hashes`, so `allowed_hashes = {key_hash}`.
  - Keep `:763-764` (`test_mode` forces a pool and settings reset) and `:1501` (the CLI continues after a failed bootstrap in TEST_MODE). Neither stores or accepts a credential. Both go to the Inventory's Deferred row.
- Modify: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`. Add to `SECURITY_FUNCTIONS`:
  - `api/v1/API_Deps/auth_deps.py:_authenticate_api_key_from_request`;
  - `core/AuthNZ/auth_principal_resolver.py:get_auth_principal`;
  - `api/v1/endpoints/mcp_unified_endpoint.py:_resolve_token_data_compat`;
  - `core/AuthNZ/initialize.py:bootstrap_single_user_profile`.

  On dev these hold 3, 5, 4 and 1 references, all of them deleted here. `authenticate_api_key_user` is not added: its legacy users-DB fallback (`:1157`, `:1160`) stays, and goes to the Deferred row.
- Modify: `tldw_Server_API/tests/Evaluations/conftest.py`. Add the autouse fixture `_configured_key_is_the_test_key`: `monkeypatch.setenv("SINGLE_USER_API_KEY", test_config.TEST_API_KEY)`, then `reset_settings()`, then yield. `test_config` is `tldw_Server_API.tests.test_config.test_config`.
- Modify: `tldw_Server_API/tests/MCP_unified/test_tool_catalogs_api.py:162`. Send `get_settings().SINGLE_USER_API_KEY` (from `core.AuthNZ.settings`) instead of `os.getenv("SINGLE_USER_TEST_API_KEY", ...)`.
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_single_user_rbac_seed_sqlite.py` (one new test).
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py`.
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_single_user_test_key_is_not_a_credential.py` (new).
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py` and `tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py`.

**Interfaces:**
- Consumes: Task 1's `--tighten`; C4.
- Produces: no new names. Deleted: `auth_principal_resolver._is_test_context`, `mcp_unified_endpoint._single_user_test_key_compat_allowed` and `MCPServer._single_user_test_key_compat_allowed`.
- After this task, `SINGLE_USER_TEST_API_KEY` stays a guarded seam in exactly two places, both in `settings.py:988-1005`: settings' pytest-context default for `SINGLE_USER_API_KEY`, and the placeholder normalization. The app neither stores nor accepts the test key as its own credential.
- Tests get the key from fixtures: the root `conftest.py` makes `SINGLE_USER_API_KEY` equal the test key, and the Evaluations fixture aligns its own key.

- [ ] **Step 1: Write the failing tests.**
  - In `test_auth_deps_precedence.py`, replace `test_authenticate_api_key_test_mode_blocked_in_production` and `test_authenticate_api_key_test_mode_allows_non_production` (`:49-101`) with `test_api_key_auth_in_test_mode_never_mints_a_wildcard_admin`.
    - Setup: the same stub settings (`AUTH_MODE="multi_user"`, `SINGLE_USER_API_KEY="local-ci-test-key"`), `TEST_MODE=1`, `ENVIRONMENT=development`. Also patch `auth_deps.get_api_key_manager` with an async function that raises `RuntimeError("no key store")`, as `AuthNZ_Unit/test_auth_deps_hardening.py:366` does, so the production path ends in its 401 without a database.
    - It asserts that `await auth_deps._authenticate_api_key_from_request(request, "local-ci-test-key")` raises `HTTPException` with `status_code == 401`.
    - On dev, it returns `{"id": 1, "roles": ["admin"], "permissions": ["*"], ...}`.
  - `test_single_user_test_key_is_not_a_credential.py`. Each test sets `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY=configured-key-1234567890`, `SINGLE_USER_TEST_API_KEY=other-test-key-0987654321` and `TEST_MODE=1`, then calls `reset_settings()`:
    - `test_test_key_is_not_accepted_when_it_differs_from_the_configured_key`: `await User_DB_Handling.authenticate_api_key_user(_request(), "other-test-key-0987654321")` raises `HTTPException` 401. Build `_request()` as `test_auth_deps_precedence._build_request` does.
    - `test_resolver_rejects_the_test_key_under_pytest`: `await auth_principal_resolver.get_auth_principal(<request with X-API-KEY: other-test-key-0987654321>)` raises `HTTPException` 401.
  - In `app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py`, replace two tests that pin the test key's acceptance:
    - `test_mounted_http_single_user_test_api_key_attaches_test_metadata_with_guard` (`:193-222`) becomes `test_mounted_http_single_user_test_api_key_rejected_in_test_mode`;
    - `test_mounted_ws_single_user_test_api_key_attaches_test_metadata_with_guard` (`:379-410`) becomes `test_mounted_ws_single_user_test_api_key_rejected_in_test_mode`.

    Each keeps its setup (`test_mode=True, debug_mode=True`) and takes the assertions of its `..._rejected_when_test_mode_false` neighbour (`:179-189` for HTTP, and the WS one at `:354-377`).
  - In `tests/MCP_unified/test_mcp_http_auth_paths.py`, add `test_mcp_single_user_api_key_grants_no_wildcard_permissions_in_test_mode`. Copy the setup of `test_mcp_single_user_api_key_respects_ip_allowlist` (`:690-725`), add `monkeypatch.setenv("TEST_MODE", "1")`, and send the allowed request. Assert `user_allowed.permissions == []`; on dev it is `["*"]`.
  - In `tests/AuthNZ/unit/test_single_user_rbac_seed_sqlite.py`, add `test_single_user_rbac_seed_does_not_store_the_test_key_in_test_mode`.
    - Use the setup of `test_single_user_rbac_seed_sqlite_file_applies_schema_and_seeds` (`:13-41`), with three changes: `TEST_MODE=1`, `SINGLE_USER_API_KEY=configured-key-1234567890`, and `SINGLE_USER_TEST_API_KEY=other-test-key-0987654321`.
    - After `await ensure_single_user_rbac_seed_if_needed()`, assert `conn.execute("SELECT COUNT(*) FROM api_keys WHERE name = 'single-user test key'").fetchone()[0] == 0`. On dev it is 1.
    - Then assert that `await User_DB_Handling.authenticate_api_key_user(_request(), "other-test-key-0987654321")` raises `HTTPException` 401. That proves the key does not authenticate when no seam injects it.
    - As a positive control, the configured key authenticates.
  - No new test pins the resolver's non-JWT bearer change. Under pytest, it only adds a JWT attempt that falls through to the same API-key path. `tests/AuthNZ_Unit/test_auth_principal_resolver.py` covers the API-key path.

- [ ] **Step 2: Run them; they fail.**
  Run: `PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t4" tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py tldw_Server_API/tests/AuthNZ_Unit/test_single_user_test_key_is_not_a_credential.py tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py -k "wildcard or test_key or rejected_in_test_mode"`
  Then: `... tldw_Server_API/tests/AuthNZ/unit/test_single_user_rbac_seed_sqlite.py -k does_not_store_the_test_key`.

  Expected: 7 FAIL in total, each with the dev behaviour above:
  - the wildcard admin;
  - the test key accepted by `authenticate_api_key_user`, by the resolver, and by MCP over HTTP and WS;
  - `["*"]`;
  - the stored test-key row.

- [ ] **Step 3: Delete the branches** (the Files list).

- [ ] **Step 4: Run the tests and every hit.**
  - The Step 2 command passes.
  - Then run `git grep -l "SINGLE_USER_TEST_API_KEY\|_single_user_test_key_compat_allowed\|single_user_test_api_key\|_is_test_context" -- 'tldw_Server_API/tests/*test_*.py' 'tldw_Server_API/app/core/MCP_unified/tests/*.py'` (36 test files on b9eed09f91), plus `tldw_Server_API/tests/MCP_unified` and `tldw_Server_API/tests/AuthNZ/test_websocket_route_auth_matrix.py`.
  - Expected: no failure that the same files don't show on dev.
  - A test that authenticated with a test key different from `SINGLE_USER_API_KEY` sets `SINGLE_USER_API_KEY` to that key and calls `reset_settings()`. A test that asserted the `single_user_test_api_key` auth source, or `["*"]` from the compat path, asserts the production values (`single_user_api_key`, `[]`).
  - Measured in the forced tree, beyond Step 1's tests:
    - `tests/MCP_unified/test_mcp_http_auth_paths.py::test_single_user_test_api_key_allowed_in_test_mode_dev_context` pins the deleted acceptance. Delete it; Step 1's `..._rejected_in_test_mode` tests replace it.
    - `AuthNZ_Unit/test_authnz_invariants.py::test_get_current_user_test_mode_single_user_respects_ip_allowlist` drove the deleted fast path. Rename it `test_get_current_user_single_user_respects_ip_allowlist`, and send the configured key, so it checks `authenticate_api_key_user`'s allow-list.
    - `AuthNZ_Unit/test_auth_deps_hardening.py::test_test_mode_ensure_authnz_tables_fallback_log_is_sanitized` tests the deleted fast path's log. Delete it.
    - `AuthNZ_Unit/test_auth_principal_resolver.py::test_get_auth_principal_jwt_path` and `::test_get_auth_principal_jwt_path_ignores_boolean_is_admin_without_claims` send a non-JWT string as a JWT in multi-user mode. Give them a JWT-shaped token (`"header.payload.sig"`).
    - 31 tests get 401: 27 in `Evaluations/test_evaluations_unified.py`, 3 in `Evaluations/test_proposition_evaluations.py`, and 1 in `MCP_unified/test_tool_catalogs_api.py`. Forcing only this task's sites reproduces exactly these 31.
      - They send `test_config.TEST_API_KEY` (`tests/test_config.py:18`), and only the test-key branch accepted it as the configured key.
      - The fix is in the test helpers, not the app (Files): the `_configured_key_is_the_test_key` fixture for the 30 Evaluations tests, and the configured key in `test_tool_catalogs_api.py`.
      - Measured: with Task 4's sites forced and both changes made, all 31 pass, and the 7 files give 64 passed. The other 6 failures there are Task 5's.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`. Expected: predicates 40 → 37.
  - Remove the stale ALLOWLIST pairs for `core/AuthNZ/auth_principal_resolver.py` and `api/v1/endpoints/mcp_unified_endpoint.py`.
  - Run C1 (it includes `test_websocket_route_auth_matrix.py` and `test_auth_token_parsing_ratchet.py`), C2, C3 and C4.

- [ ] **Step 6: Commit, then run C6 before merge.** Every test that authenticates goes through these functions.

```bash
git add tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py tldw_Server_API/app/core/AuthNZ/initialize.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests tldw_Server_API/tests/lint
git commit -m "fix(authnz): the single-user test key, wildcard compat permissions and test-context bearer handling leave production code (TASK-13950.19)"
```

### Task 5: Test-only auth bypass flags and the CSRF default (AC3)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`.
  - Delete `_evals_testing_bypass_enabled` and `_evals_test_mode_bypass_enabled` (`:49-54`).
  - Delete their three call sites: `:109-110`, `:159-160` and `:182-185`.
  - Delete the `is_explicit_pytest_runtime` import if it is unused.
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`.
  - In `get_request_user`, delete the whole "Test-mode bypass for evaluations" `try`/`except` (`:1554-1570`), including the `import os as _os` / `_prod = ...` lines and the `except` that logs "test-mode bypass env detection failed".
  - Delete the following "Warn if test flags are present in production deployments" `try`/`except` (`:1572-1590`). It is redundant: `validate_test_runtime_flags()` refuses `TEST_MODE`, `TESTING` and `TLDW_TEST_MODE` at startup in a production-like runtime. It is also the last test-signal read in the function, which the zero pin needs.
  - Delete `_is_strict_test_bypass_context` (`:336-353`), whose only caller was the bypass.
- Modify: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`. Add to `SECURITY_FUNCTIONS`:
  - `api/v1/endpoints/evaluations/evaluations_auth.py:verify_api_key`;
  - `core/AuthNZ/User_DB_Handling.py:get_request_user`;
  - `core/AuthNZ/csrf_protection.py:resolve_effective_csrf_enabled`.

  On dev these hold 2, 4 and 1 references.
- Modify the 6 Evaluations tests measured under Step 4. Their changes are listed there.
- Modify: `tldw_Server_API/app/core/AuthNZ/csrf_protection.py:64-65`. Delete `if is_test_mode(): return False`, and the `is_test_mode` import if it is unused.
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`.
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py`.
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_no_test_auth_bypass.py` (new).

**Interfaces:**
- Consumes: C4.
- Produces: the CSRF default is `AUTH_MODE in {"single_user", "multi_user"}` whenever `CSRF_ENABLED` is not configured, in every runtime.
  - Tests that need CSRF off configure it, as `tests/AuthNZ/conftest.py:492-493` already does.
  - Measured: in the forced tree, no test turned red through CSRF. The set included `AuthNZ/unit/test_csrf_binding.py`. So the suite gets no `CSRF_ENABLED` default. If C6 shows CSRF 403s, the affected test's fixture sets `CSRF_ENABLED=false`, as the AuthNZ conftest does, never the root conftest.

- [ ] **Step 1: Write the failing tests.**
  - In `test_evaluations_auth_runtime_guards.py`:
    - rename `test_verify_api_key_testing_bypass_allows_under_explicit_pytest_runtime` (`:74-82`) to `test_verify_api_key_has_no_testing_bypass_under_pytest`, and make it expect `HTTPException` 401;
    - delete `test_get_eval_request_user_testing_bypass_allows_under_explicit_pytest_runtime` (`:128-150`). It stubs `get_request_user`, which `get_eval_request_user` only passes through (`evaluations_auth.py:261-274`), so it never tested a bypass of its own. The bypass lived in `get_request_user`, and `test_get_request_user_has_no_testing_bypass_under_pytest` below covers it. `get_eval_request_user` is already in `SECURITY_FUNCTIONS`;
    - add `test_verify_api_key_rejects_a_wrong_single_user_key_in_test_mode_under_pytest`: the `:85-102` setup with `PYTEST_CURRENT_TEST` kept, expecting 401.
  - In `test_csrf_binding.py`, add `test_csrf_default_is_on_when_unconfigured_even_in_test_mode`:
    - `monkeypatch.delenv("CSRF_ENABLED", raising=False)`;
    - `monkeypatch.setitem(csrf_protection.global_settings, "CSRF_ENABLED", None)`;
    - `TEST_MODE=1`, `AUTH_MODE=single_user`, `reset_settings()`;
    - asserts `csrf_protection.resolve_effective_csrf_enabled() is True`.
  - In `test_no_test_auth_bypass.py`, add `test_get_request_user_has_no_testing_bypass_under_pytest`:
    - `TESTING=1`, `EVALS_HEAVY_ADMIN_ONLY=false`, `AUTH_MODE=multi_user`, `reset_settings()`;
    - `await User_DB_Handling.get_request_user(<request with no credentials>)` raises `HTTPException` 401.
    - Build the call as the existing `get_request_user` tests in `tests/AuthNZ_Unit/test_authenticated_content_scope.py` do.

- [ ] **Step 2: Run them; they fail.**
  Run: `PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t5" tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py tldw_Server_API/tests/AuthNZ_Unit/test_no_test_auth_bypass.py -k "no_testing_bypass or wrong_single_user_key or csrf_default_is_on"`
  Expected: 4 FAIL. On dev, the bypasses return `"test_user"` or the single-user instance, and CSRF resolves `False`.

- [ ] **Step 3: Delete the branches** (the Files list).

- [ ] **Step 4: Run every hit.**
  Run: `RUN_EVALUATIONS=1 ...` over `$(git grep -l "EVALS_HEAVY_ADMIN_ONLY\|resolve_effective_csrf_enabled\|CSRF_ENABLED\|_evals_.*bypass" -- 'tldw_Server_API/tests/*test_*.py')`, plus `tldw_Server_API/tests/Evaluations` and `tldw_Server_API/tests/AuthNZ/integration`.
  Expected: no failure that the same files don't show on dev.
  - Measured in the forced tree, beyond Step 1's tests:
    - `AuthNZ/unit/test_auth_principal_service_and_single_user_tokens.py::test_get_request_user_bypass_allows_strict_test_context` pins the deleted bypass. Delete it; Step 1's `test_get_request_user_has_no_testing_bypass_under_pytest` replaces it.
    - 6 Evaluations tests get 401, because they reached their endpoints through the bypass. Forcing only Task 4's sites leaves all six green, and Task 4's key fixture alone does not fix them. Each gets one concrete change:
      - `integration/test_admin_cleanup.py:21`, `integration/test_rate_limits_endpoint.py:21,76` and `integration/test_batch_propositions_ocr.py:20` send no credentials. They relied on their `_setup_env` (`TESTING=true`, `EVALS_HEAVY_ADMIN_ONLY=false`). Each `TestClient(app)` becomes `TestClient(app, headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY})`, with `get_settings` from `core.AuthNZ.settings`, so they authenticate through the real dependency. That covers 5 tests.
      - `test_evaluation_integration.py::TestAuthentication::test_single_user_auth_respects_ip_allowlist` (`:452`) sets `mock_settings.SINGLE_USER_API_KEY`. But `verify_api_key` reads `os.getenv("SINGLE_USER_API_KEY")` first (`evaluations_auth.py:163-167`). Add `monkeypatch` to its arguments and `monkeypatch.setenv("SINGLE_USER_API_KEY", mock_settings.SINGLE_USER_API_KEY)` after that line.
      - Measured: with Tasks 4 and 5's sites forced and these changes made, the four files give 19 passed.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`. Expected: predicates 37 → 34.
  - Expect no stale `.6` pair: neither module reads the `PYTEST_CURRENT_TEST` literal.
  - Run C1, C2, C3 and C4.

- [ ] **Step 6: Commit, then run C6 before merge.** The CSRF default is a global default flip.

```bash
git add tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/app/core/AuthNZ/csrf_protection.py tldw_Server_API/tests
git commit -m "fix(authnz): Evaluations and get_request_user have no TESTING/TEST_MODE auth bypass; TEST_MODE no longer turns the CSRF default off (TASK-13950.19)"
```

### Task 6: The stub session manager leaves `get_session_manager_dep` (AC3)

**Precondition (ruling 2):** `.18` Task 4 (slice C) has landed.
- `get_session_manager()` reads `app_services().session_manager` (`core/Utils/app_services.py`).
- The root autouse `_restore_auth_env_and_singletons` (`tests/conftest.py:374`) empties that slot after every test.
- This task changes no root fixture.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`.
  - `:730-913` (re-locate on `.18`'s version): `get_session_manager_dep` becomes `return await get_session_manager()` with its docstring. Delete `_StubSessionManager`, the `AUTHNZ_FORCE_REAL_SESSION_MANAGER` read and the try/except around them.
  - `:124-130`: delete `_TEST_SESSION_STATE`, `_TEST_SESSION_LOCKS`, `_TEST_SESSION_LOCK_GUARD`, `_TEST_SESSION_STATE_GUARD`, `_TEST_EPHEMERAL_STATE` and `_TEST_EPHEMERAL_STATE_GUARD`.
  - `:325-345`: delete `_get_test_session_lock` and `reset_test_ephemeral_state`. Nothing outside `auth_deps.py` references them (`git grep` on b9eed09f91).
  - Drop imports that this leaves unused (`WeakKeyDictionary`, `threading`, if nothing else uses them).
- Modify: delete the now no-op `AUTHNZ_FORCE_REAL_SESSION_MANAGER` lines in tests:
  - `tests/AuthNZ/conftest.py:1345,1720`;
  - `tests/AuthNZ/integration/test_single_user_cookie_session.py:41`;
  - `tests/performance/test_authnz_multiuser_sqlite_load.py:40`;
  - the `delenv` lines in `tests/AuthNZ_Unit/test_auth_deps_hardening.py:1119,1138` and in `test_session_manager_dep_production_guard.py`. The ones at `:252` and `:274` go with their deleted tests.

  Two references stay, deliberately:
  - `Env_Vars.md:630` is the raw scan that `Helper_Scripts/generate_env_vars_md.py` regenerates, and it already says it may list deprecated flags. Regenerating it here would pull in unrelated drift. The curated `Docs/Operations/Env_Vars.md` and its `Docs/Published` mirror do not list the variable.
  - `apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts:73` passes an env the server now ignores. Editing it would run the frontend gates for no behaviour change. Task 13's follow-up lists both.
- Modify: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`. Add `api/v1/API_Deps/auth_deps.py:get_session_manager_dep` to `SECURITY_FUNCTIONS`; it holds 2 references on dev.
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_dep_production_guard.py`.
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`. Delete `test_stub_session_manager_uses_timezone_aware_timestamps` and `test_stub_session_manager_supports_admin_endpoint_contract` (`:250-305`), which test the deleted stub.

**Interfaces:**
- Produces: `async def get_session_manager_dep() -> SessionManager` returns `await get_session_manager()` in every runtime.
- Consumes `.18`'s names verbatim: `app_services() -> AppServices`, `AppServices.session_manager: SessionManager | None`, and the `fresh_services` fixture.
- A test that wants a fake does one of two things:
  - sets `fresh_services.session_manager = <fake>`;
  - overrides the dependency: `app.dependency_overrides[auth_deps.get_session_manager_dep] = lambda: <fake>`. All 24 app uses are `Depends(get_session_manager_dep)`, so the override reaches them.

- [ ] **Step 1: Write the failing test.** In `test_session_manager_dep_production_guard.py`:
  - Replace `test_session_manager_dep_uses_test_stub_in_non_production` (`:9-29`) with `test_session_manager_dep_returns_the_slot_manager_under_pytest_and_test_mode(fresh_services, monkeypatch)`.
    - Set `fresh_services.session_manager = marker`, where `marker` is a `SessionManager` created with `SessionManager.__new__(SessionManager)` and never initialized. Set `TEST_MODE=1`.
    - Drop the old `_is_explicit_pytest_runtime` and `_is_production_like_env` patches.
    - Then `assert await auth_deps.get_session_manager_dep() is marker`.
    - If `.18`'s `get_session_manager()` initializes a slot manager that is not yet marked initialized, set `marker._initialized = True` first, as `.18` Task 4's concurrency test does.
  - In `test_session_manager_dep_never_uses_test_stub_in_production_like_env`, drop the same two patches (the import-alias rule).
  - Delete `test_session_manager_dep_test_stub_create_session_accepts_expiry_overrides` (`:57-`).

- [ ] **Step 2: Run it; it fails.**
  Run: `PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t6" tldw_Server_API/tests/AuthNZ/unit/test_session_manager_dep_production_guard.py`
  Expected: 1 FAIL, `assert <..._StubSessionManager object> is <...SessionManager object>` (the stub, not the slot's marker).
  Cross-test leakage of the slot is pinned by `.18` Task 4's `test_root_autouse_empties_the_session_slot` (the root fixture empties the slot after every test, ruling 2); this task adds no fixture.

- [ ] **Step 3: Delete the stub** (the Files list).

- [ ] **Step 4: Run every hit.**
  Run: `$(git grep -l "get_session_manager_dep\|SessionManager\|/auth/login\|/auth/logout\|/auth/refresh" -- 'tldw_Server_API/tests/*test_*.py')`. Run `tldw_Server_API/tests/AuthNZ/integration` locally too if time allows (about 40 CI minutes); otherwise C6 covers it through ci.yml's `auth-integration-*` shards.
  Expected: no failure that the same files don't show on dev.
  - A test that needs an in-memory session store overrides `get_session_manager_dep` with a fake defined in that test's module, as `tests/AuthNZ/integration/test_auth_integration_flow_extended.py:170-184` does.
  - Measured: in the forced tree, only the stub's own four tests fail: the two Step 1 handles, and the two in `test_auth_deps_hardening.py` listed in Files. That run was on pre-`.18` dev, without the per-test slot reset. The measured set did not include AuthNZ integration; C6 covers it.
  - Session state across tests is `.18`'s per-test slot reset (ruling 2). If C6 shows a cross-test session failure, it goes back to `.18`'s owner, not to a root-fixture edit here.

- [ ] **Step 5: Lower the ratchets.** Run `--tighten`. Run C1, C2, C3 and C4. If C2 shows fewer test-quality offenses, run `test_quality_triage.py --write-baseline`.

- [ ] **Step 6: Commit, then run C6 before merge** (a suite-wide default flip).

```bash
git add tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/tests
git commit -m "fix(authnz): get_session_manager_dep always returns the real session manager; the revocation-blind test stub is gone (TASK-13950.19)"
```

### Task 7: The transaction adapter and the auth ingress-limit skip (AC3)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`.
  - `:429-654` in `get_db_transaction`: delete `use_adapter`, `_ConnAdapter` and the adapter branch. Dedent the `else:` body (`:655-` to the end of the function), which is the request-scoped transaction with lock retries. Update the docstring (`:421-426`).
  - `:2148-2150` in `_enforce_auth_deps_ingress_guard`: delete `if _is_test_mode(): return` and its comment.
- Modify: the repo-root `conftest.py`, after its `os.environ.setdefault("AUTH_MODE", ...)` line (`:12`). Add, with a comment naming TASK-13950.19:

  ```python
  os.environ.setdefault("AUTH_DEPS_FALLBACK_RATE_LIMIT", "1000000")
  os.environ.setdefault("AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT", "1000000")
  ```

  - The comment says this replaces the `TEST_MODE` skip, and that a test of the brute-force floor sets its own limit.
  - It goes in the repo-root conftest, like Tasks 8 and 12's defaults, so the in-app MCP tests get it too.
- Modify: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`. Add `api/v1/API_Deps/auth_deps.py:get_db_transaction` and `api/v1/API_Deps/auth_deps.py:_enforce_auth_deps_ingress_guard` to `SECURITY_FUNCTIONS`; each holds 1 reference on dev.
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_db_adapter_backend_selection.py`. Delete the 11 adapter tests (`:176-600`), which test the deleted adapter.
  - Keep the file's helpers only if a remaining test uses them; otherwise delete the file.
  - No port is needed; the production path already pins both behaviours:
    - rollback on a failed version touch: `AuthNZ_Unit/test_auth_login_sqlite_connection_scope.py:145` (`test_sqlite_versioned_user_write_rolls_back_when_anchor_touch_fails`, with the `reject_profile_version_touch` trigger);
    - cancellation precedence: `AuthNZ_Unit/test_auth_deps_hardening.py:1294` (`test_get_db_transaction_cancellation_wins_over_cleanup_lock_error`).
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`.
  - Delete the three `test_test_db_adapter_*` tests (`:181-248`, 4 cases), and `test_get_db_transaction_uses_adapter_when_tldw_test_mode_is_y`.
  - Rename `test_get_db_transaction_requires_explicit_test_mode` (`:1152`) to `test_get_db_transaction_uses_the_pool_transaction`. Drop its two `TEST_MODE`/`TLDW_TEST_MODE` setenvs, and change its `acquire()` assertion message to "get_db_transaction uses pool.transaction(), never pool.acquire()".
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_ingress_and_transaction_seams.py` (new).

**Interfaces:**
- Produces:
  - `get_db_transaction()` always yields the connection `db_pool.transaction()` yields, on either backend.
  - `_enforce_auth_deps_ingress_guard` enforces the fallback bucket in every runtime.
- Consumes: none.

- [ ] **Step 1: Write the failing tests** in the new file.
  - `test_get_db_transaction_yields_the_pool_transaction_in_test_mode`.
    - `TEST_MODE=1`. Build a real SQLite AuthNZ pool the way `AuthNZ_Unit/test_auth_refresh_sqlite_connection_scope.py:55-75` does: `Settings(AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'txn.db'}", ...)`, then `pool = database.DatabasePool(settings=settings)` and `await pool.initialize()`.
    - Point `auth_deps.get_db_pool` at it with an async lambda.
    - In `finally`, run `await pool.close()` and `reset_settings()`.
    - Record what the pool itself yields: `async with pool.transaction() as raw: expected = type(raw)`.
    - Drive the generator with `agen = auth_deps.get_db_transaction(); conn = await agen.__anext__()`.
    - Assert `type(conn) is expected`, then close the generator.
    - On dev, `conn` is the nested `_ConnAdapter`.
  - `test_auth_fallback_limit_applies_in_test_mode`.
    - `TEST_MODE=1`, `AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT=1`, and `auth_deps._rg_enabled_for_request` patched to return `False`.
    - A request with no principal from client `198.51.100.7`, path `/api/v1/auth/login`.
    - The first `await auth_deps.check_auth_rate_limit(request)` returns `None`; the second raises `HTTPException` 429.
    - Clear `auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS` before and after.
  - `test_suite_defaults_disable_the_auth_fallback_limiter`: `os.environ["AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT"] == "1000000"` and `os.environ["AUTH_DEPS_FALLBACK_RATE_LIMIT"] == "1000000"`. This pins the conftest default the suite now depends on.

- [ ] **Step 2: Run them; they fail.**
  Run: `PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t7" tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_ingress_and_transaction_seams.py`
  Expected: 3 FAIL: `_ConnAdapter`; no 429; and `KeyError: 'AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT'`.

- [ ] **Step 3: Delete the branches, and add the conftest defaults** (the Files list).

- [ ] **Step 4: Run every hit.**
  Run: `$(git grep -l "get_db_transaction\|check_rate_limit\|check_auth_rate_limit\|AUTH_DEPS_.*FALLBACK" -- 'tldw_Server_API/tests/*test_*.py')` (18 files reference `get_db_transaction`), plus `tldw_Server_API/tests/AuthNZ_SQLite`.
  - Then C5 over `tldw_Server_API/tests/AuthNZ_Postgres` and the Postgres cases kept or ported in `test_auth_deps_db_adapter_backend_selection.py`. The adapter had a Postgres branch, so the production path must be proven on Postgres.
  - Expected: no failure that the same files don't show on dev.
  - Measured: in the forced tree, only the adapter's own 16 test cases fail, and no measured test reached the auth fallback limit. The 16:
    - 11 in `test_auth_deps_db_adapter_backend_selection.py`;
    - 4 `test_test_db_adapter_*` cases in `test_auth_deps_hardening.py`;
    - `test_auth_deps_hardening.py::test_get_db_transaction_uses_adapter_when_tldw_test_mode_is_y`. Delete it with the others.

- [ ] **Step 5: Lower the ratchets.** Run `--tighten`, then C1, C2, C3 and C4.

- [ ] **Step 6: Commit, then run C6 before merge** (a root-conftest change and a suite-wide default flip).

```bash
git add tldw_Server_API/app/api/v1/API_Deps/auth_deps.py conftest.py tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/lint
git commit -m "fix(authnz): get_db_transaction always uses the pool transaction and the auth ingress fallback limit always applies; the suite raises its limit in conftest (TASK-13950.19)"
```

### Task 8: The audit service's inline I/O becomes a constructor argument (AC3)

**Files:**
- Modify: `tldw_Server_API/app/core/Audit/unified_audit_service.py`.
  - In `class UnifiedAuditService` (`:961`), add `inline_io_default: ClassVar[bool] = False`.
  - In `__init__` (`:975-987`), add a keyword `inline_io: bool | None = None`.
  - Replace `:1055-1064` with `self._inline_io = type(self).inline_io_default if inline_io is None else inline_io`.
  - Rename the other `self._test_mode` uses to `self._inline_io` (`:1145`, `:1150`, `:2385`, `:2518`, `:3129`).
  - In the module docstring (`:26`), the `TEST_MODE/TLDW_TEST_MODE` line is replaced by the `inline_io` argument.
- Modify: `tldw_Server_API/app/core/Audit/README.md:10,50,109`. Describe `inline_io`, and drop the env flags.
- Modify: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py`. Add `core/Audit/unified_audit_service.py:UnifiedAuditService.__init__` to `SECURITY_FUNCTIONS`. Its 4 reference lines on dev (`:1059-1064`) are the ones this task replaces; `_inline_io` and `inline_io_default` are not signal names.
- Modify: `conftest.py` (repo root). Add `import pytest` and the session fixture:

```python
@pytest.fixture(autouse=True, scope="session")
def _audit_inline_io():
    """Audit services flush inline and start no background loops in tests (TASK-13950.19).

    Many tests patch asyncio.sleep, which would spin the flush loop. A test of the
    background path passes UnifiedAuditService(..., inline_io=False).
    """
    from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService

    UnifiedAuditService.inline_io_default = True
    yield
    UnifiedAuditService.inline_io_default = False
```

  It goes in the repo-root conftest so `app/core/MCP_unified/tests` gets it too.
  - A session fixture runs after collection, so a service built at import time would miss the default. Measured with an AST scan of `tests/` and the in-app MCP tests: no module-level `UnifiedAuditService(` call exists.
- Test: `tldw_Server_API/tests/Audit/test_unified_audit_service.py`. Add two tests. Make `test_stop_safe_when_owner_loop_closed` (`:2139`) and the other tests that delete `PYTEST_CURRENT_TEST` to reach the background path pass `inline_io=False` instead; they keep their env edits only if something else needs them.
- Test: `tldw_Server_API/tests/Audit/test_audit_fallback_lock.py` (the same change).

**Interfaces:**
- Produces:
  - `UnifiedAuditService(..., inline_io: bool | None = None)`;
  - `UnifiedAuditService.inline_io_default: ClassVar[bool]`.
- If `.18` adds an audit slot to its container, its fixture sets the same class default.

- [ ] **Step 1: Write the tests.**

```python
@pytest.mark.asyncio
async def test_audit_background_loops_follow_inline_io_not_the_test_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")  # PYTEST_CURRENT_TEST is set by pytest as well
    service = UnifiedAuditService(db_path=str(tmp_path / "audit.db"), inline_io=False)
    await service.initialize()
    try:
        assert service._flush_task is not None
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_suite_audit_services_use_inline_io(tmp_path):
    service = UnifiedAuditService(db_path=str(tmp_path / "audit.db"))
    await service.initialize()
    try:
        assert service._flush_task is None  # the root conftest's _audit_inline_io fixture
    finally:
        await service.stop()
```

- [ ] **Step 2: Run them.**
  Run: `PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t8" tldw_Server_API/tests/Audit/test_unified_audit_service.py -k "inline_io"`
  Expected: the first fails with `TypeError: ... unexpected keyword argument 'inline_io'`; the second passes (it pins today's suite behaviour).

- [ ] **Step 3: Implement** (the Files list).

- [ ] **Step 4: Run every hit.**
  Run: `$(git grep -l "UnifiedAuditService\|get_audit_service_for_user\|unified_audit_service" -- 'tldw_Server_API/tests/*test_*.py' 'tldw_Server_API/app/core/MCP_unified/tests/*.py')` and `tldw_Server_API/tests/Audit`.
  Expected: no failure that the same files don't show on dev.
  - Measured: with production audit I/O forced and no fixture, the 4 `test_replay_fallback_queue*` tests in `tests/Audit/test_unified_audit_service.py` fail (0 rows inserted, 2 expected). With the fixture they stay on inline I/O, and they must pass unchanged.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`.
  - Remove the stale ALLOWLIST pair `PYTEST_CURRENT_TEST` → `core/Audit/unified_audit_service.py`.
  - Run C1, C2, C3 and C4.

- [ ] **Step 6: Commit, then run C6 before merge** (a root-conftest fixture).

```bash
git add tldw_Server_API/app/core/Audit conftest.py tldw_Server_API/tests/Audit tldw_Server_API/tests/lint
git commit -m "refactor(audit): inline audit I/O is a constructor argument set by a test fixture, not a TEST_MODE/pytest env check (TASK-13950.19)"
```

### Task 9: A deterministic middleware stack (AC4)

**Files:**
- Modify: `tldw_Server_API/app/main.py`.
  - Delete `_TEST_FLAGS_SET`, `_EXPLICIT_PYTEST_RUNTIME`, `_TEST_MODE`, `_runtime_test_mode_active` and the `_TEST_FLAGS_SET and not _EXPLICIT_PYTEST_RUNTIME` warning (`:2632-2650`).
  - Delete the `if _TEST_MODE:` branch (`:2652-2717`), and dedent the `else:` body (`:2718-2797`) to module level.
  - `:1298`: `test_mode=_shared_is_explicit_pytest_runtime(),`. This is equivalent: `_runtime_test_mode_active()` reduces to `is_explicit_pytest_runtime()`, because its last disjunct repeats its first conjunct.
  - Drop the `_test_env_flag_enabled`, `_is_explicit_pytest_runtime` and `_shared_is_test_mode` imports (`:2622-2630`) if nothing else uses them.
- Modify: `tldw_Server_API/tests/AuthNZ/conftest.py:565-575`. Drop `SecurityHeadersMiddleware` (`_SHM`) from the tuple `reset_singletons` strips, and its import. The header middleware does no background DB work, which is that strip's stated reason.
- Modify: `tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py`. Remove `"main.py"` from the `PYTEST_CURRENT_TEST` pair list, and remove the `"_TEST_MODE": ("main.py's module global, not an env read", ("main.py",))` entry.
- Test: `tldw_Server_API/tests/Security/test_app_middleware_stack.py` (new).
- Modify: `.github/workflows/ci.yml:654,2344,3873`. Add `tldw_Server_API/tests/Security/test_app_middleware_stack.py` after `test_security_headers_middleware.py` in the three `core-security` shard lists. `tests/Security` is listed file by file, so without this the new file runs in no full-suite shard and fails C8.

**Interfaces:**
- Consumes: `tldw_Server_API.tests.helpers.app_main_state`: `app_main_isolated() -> ContextManager[None]` and `reload_app_main() -> ModuleType`.
- Produces: `app.main.app.user_middleware` holds `SecurityHeadersMiddleware` (unless `ENABLE_SECURITY_HEADERS` is falsy), `HTTPMetricsMiddleware`, `AccessLogMiddleware` and `UsageLoggingMiddleware`, whenever `app.main` is imported.

- [ ] **Step 1: Write the failing tests.**

```python
def _stack(app) -> list[str]:
    return [m.cls.__name__ for m in app.user_middleware]


def test_the_suite_app_installs_security_headers() -> None:
    from tldw_Server_API.app import main as app_main

    assert "SecurityHeadersMiddleware" in _stack(app_main.app)


def test_middleware_stack_does_not_depend_on_import_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    with app_main_isolated():
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        collection_time = _stack(reload_app_main().app)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_y (call)")
        test_time = _stack(reload_app_main().app)
    assert collection_time == test_time
    assert {"SecurityHeadersMiddleware", "HTTPMetricsMiddleware", "UsageLoggingMiddleware"} <= set(test_time)
```

- [ ] **Step 2: Run them; the second fails.**
  Run: `PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t9" tldw_Server_API/tests/Security/test_app_middleware_stack.py`
  Expected:
  - `test_middleware_stack_does_not_depend_on_import_timing` fails: the test-time stack lacks `SecurityHeadersMiddleware`, `HTTPMetricsMiddleware`, `AccessLogMiddleware` and `UsageLoggingMiddleware`.
  - The first test passes or fails depending on when `app.main` was first imported. That import-timing dependence is the bug.

- [ ] **Step 3: Implement** (the Files list).

- [ ] **Step 4: Run the tests and every hit.**
  - The Step 2 command passes.
  - Then, per the flipping-a-default rule, run `$(git grep -l "user_middleware\|SecurityHeadersMiddleware\|HTTPMetricsMiddleware\|UsageLoggingMiddleware\|AccessLogMiddleware\|_runtime_test_mode_active\|main._TEST_MODE\|reload_app_main" -- 'tldw_Server_API/tests/*test_*.py')`, plus `tldw_Server_API/tests/Logging`, `tldw_Server_API/tests/Monitoring` and `tldw_Server_API/tests/Usage`.
  - Expected: no failure that the same files don't show on dev.
  - Measured: in the forced tree, no test changed result when the import-time branch was removed. The CORS test is Task 2's.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`. Expected: predicates down by 1 (`_runtime_test_mode_active`).
  - Run C1 (it runs `test_test_seam_env_reads_guarded.py` with the two removals), C2, C3 and C4.

- [ ] **Step 6: Run C8, commit, then run C6 before merge** (a middleware-stack flip).

```bash
git add tldw_Server_API/app/main.py tldw_Server_API/tests/AuthNZ/conftest.py tldw_Server_API/tests/Security/test_app_middleware_stack.py tldw_Server_API/tests/lint .github/workflows/ci.yml
git commit -m "fix(app): main installs one middleware stack, with security headers, whatever the import timing (TASK-13950.19)"
```

### Task 10: `core.testing.is_test_context()` and the AuthNZ, API_Deps, MCP, DB-path and services predicates (AC2)

**Files:**
- Modify: `tldw_Server_API/app/core/testing.py`. After `is_explicit_pytest_runtime` (`:100-110`), add:

```python
def is_test_context() -> bool:
    """TEST_MODE, TLDW_TEST_MODE or TESTING is on, or pytest is running a test.

    The one replacement for the modules' private "is this a test" predicates
    (TASK-13950.19). validate_test_runtime_flags() refuses the three flags outside
    pytest, so in a started server this is is_explicit_pytest_runtime().
    """
    return is_test_mode() or env_flag_enabled("TESTING") or is_explicit_pytest_runtime()
```

- Modify, deleting each predicate and calling `core.testing` at its call sites:

  | Module | Predicate | Call sites → replacement | Test patch lines to retarget |
  |---|---|---|---|
  | `core/AuthNZ/User_DB_Handling.py` | `_is_test_context` (`:324`) | `:385`, `:1157`, `:1160` → `is_test_context()` | 11 lines in 3 files |
  | `core/AuthNZ/byok_testing.py` (or its `.17` path) | `_is_test_mode` (`:139`) | `:597` → `is_test_context()` | 8-10 lines in 3-4 files |
  | `api/v1/API_Deps/Audit_DB_Deps.py` | `_is_test_context` (`:194`) | `:343`, `:842` → `is_test_context()` | 4 lines in 1 file |
  | `core/DB_Management/db_path_utils.py` | `_is_test_context` (`:32`) | `:53`, `:178`, `:458`, `:480` → `is_test_context()` | 18 lines in 2 files |
  | `core/MCP_unified/server.py` | `_is_test_mode` (`:428`), `_is_explicit_pytest_runtime` (`:435`) | `:722`, `:1029`, `:1501` → `self.environment_flags_provider.is_test_mode()` / `.is_explicit_pytest_runtime()` | 4 lines in 2-3 files |
  | `services/quiz_generator.py` | `_should_use_deterministic_test_mode` (`:1542`) | `:1900` → `is_test_mode() and _call_quiz_generation_llm is _ORIGINAL_CALL_QUIZ_GENERATION_LLM` | 0 |
  | `services/startup_chacha_warmup.py` | `_is_test_mode` (`:42`) | `:18` → `is_test_mode()` (module-level import from `core.testing`) | 0 |

  `server.py`'s wrappers caught exceptions from the provider. Both providers' methods are total (they return a bool or raise nothing), so the calls go in without a try. The WS test helper patches the deleted wrappers (`app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py:310-311`); it switches to `monkeypatch.setenv("TEST_MODE", ...)`, which the host provider reads.

  Retarget rule for the patch lines:
  - A test that forced a predicate patches the environment instead: `monkeypatch.setenv("TEST_MODE", "1")`, or `monkeypatch.delenv("PYTEST_CURRENT_TEST")` plus `monkeypatch.setenv("TEST_MODE", "0")` and `monkeypatch.setenv("TESTING", "0")` for the production branch.
  - A test that imported the predicate to test it is deleted. `test_is_test_context_*` below covers the shared helper.
  - Find the lines with `git grep -nE '"<name>"|\.<name>\b|import .*\b<name>\b' -- tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests`.
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_test_mode_runtime_guard.py` (it runs in the C1 gate).

**Interfaces:**
- Produces: `tldw_Server_API.app.core.testing.is_test_context() -> bool`, consumed by Tasks 11 and 12.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"TEST_MODE": "1"}, True),
        ({"TLDW_TEST_MODE": "y"}, True),
        ({"TESTING": "true"}, True),
        ({"PYTEST_CURRENT_TEST": "t.py::t (call)"}, True),
        ({"TEST_MODE": "0", "TESTING": "0"}, False),
    ],
)
def test_is_test_context_is_the_union_of_the_test_signals(monkeypatch, env, expected):
    for key in ("TEST_MODE", "TLDW_TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert testing.is_test_context() is expected
```

  The file imports names from `core.testing` today (`:4`); add `from tldw_Server_API.app.core import testing`. Then hand-lower the `predicates` ceiling to 0 in `test_mode_branch_baseline.json` for the seven modules in the table.

- [ ] **Step 2: Run them; they fail.**
  - `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1319-t10" tldw_Server_API/tests/AuthNZ/unit/test_test_mode_runtime_guard.py -k is_test_context` → 6 FAIL, `AttributeError: ... has no attribute 'is_test_context'`.
  - `... tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py` → FAIL, 7 lines of `predicates N > ceiling 0`.

- [ ] **Step 3: Implement** `is_test_context()`, delete the predicates, replace the calls, and retarget the patch lines.

- [ ] **Step 4: Run both, and every test file the retarget touched, plus each module's own test directory** (`tests/AuthNZ_Unit`, `tests/Audit`, `tests/DB_Management`, `tests/MCP_unified`, `app/core/MCP_unified/tests`, `tests/Quizzes`, `tests/Services`). Expected: pass, with no failure that the same files don't show on dev.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`. Expected: predicates down by 8.
  - Remove the stale ALLOWLIST pairs for `api/v1/API_Deps/Audit_DB_Deps.py`, `core/AuthNZ/byok_testing.py` (or its moved path) and `core/DB_Management/db_path_utils.py`.
  - Run C1, C2, C3 and C4.

- [ ] **Step 6: Commit.**

```bash
git add tldw_Server_API/app tldw_Server_API/tests
git commit -m "refactor(testing): one core.testing.is_test_context(); AuthNZ, API_Deps, MCP, DB-path and service predicates deleted (TASK-13950.19)"
```

### Task 11: Endpoint predicates (AC2)

**Files:**
- Modify, deleting each predicate and calling `core.testing` at its call sites:

  | Module (`api/v1/endpoints/...`) | Predicate | Replacement | Test patch lines |
  |---|---|---|---|
  | `media/item.py`, `media/listing.py`, `media/versions.py` | `_is_test_mode` (`:44`, `:89`, `:76`) | `is_test_mode()` | 5 lines in 5 files (`item`) |
  | `writing.py` | `_is_test_mode` (`:720`) | `is_test_context()` at `:1914` | 0 |
  | `connectors.py` | `_is_connector_test_mode` (`:90`) | `is_test_mode() or env_flag_enabled("TESTING")` at `:145`, `:188`. This **widens** the set (see below) | 0 |
  | `character_chat_sessions.py` | `_is_char_chat_test_mode` (`:864`) | `is_test_context()` at `:886`, `:897` | 0 |
  | `embeddings_v5_production_enhanced.py` | `_is_test_context` (`:912`) | `is_test_context()` at `:925`, `:1499` | 3 lines in 1 file |
  | `evaluations/evaluations_unified.py` | `_is_eval_test_mode` (`:176`) | `is_test_context()` | 10 lines in 4 files |
  | `prompt_studio/prompt_studio_evaluations.py` | `_is_prompt_studio_test_mode` (`:100`) | `is_test_context()` | 10 lines in 3 files |
  | `admin/admin_rbac.py` | `is_test_mode` (`:135`, a "legacy patch point") | its `_shared_is_test_mode()` | 4 lines in 2 files |
  | `auth.py` | `_is_pytest_context` (`:981`) | `is_test_context()` at `:1175` | 1 line |
  | `audio/audio_streaming.py`, `audio/audio_transcriptions.py` | nested `_is_test_override` (`:344`, `:220`) | delete it and its precedence arm. The `default_value` identity rule below it still picks the patched shim. No caller passes a name outside `defaults`, so production, where neither shim is patched, is unchanged | 0 |

  - `writing` and `character_chat_sessions` read the raw keys today, and their replacements keep their sets: `TESTING` is added only where pytest was already in the set.
  - `connectors` is the exception. Today it is exactly `TEST_MODE == "true"` or `TESTING == "true"`. The replacement accepts any truthy `TEST_MODE`, `TLDW_TEST_MODE` or `TESTING`, so the suite's default `TEST_MODE=1` now switches on "trust the request base for the OAuth callback" (`:145`, `:188`) in every connector test that does not unset it.
    - Run `tldw_Server_API/tests/External_Sources` on dev and on the branch, and diff the `FAILED` lines.
    - If any connector test changes result, keep the narrow check instead, as `os.getenv("TEST_MODE", "").strip().lower() == "true" or os.getenv("TESTING", "").strip().lower() == "true"` inlined at the two call sites, and say so in the PR.
  - Retarget the patch lines by the Task 10 rule.
- Test: `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py` (via the baseline). Each module's own tests: `tests/Media`, `tests/Writing`, `tests/External_Sources`, `tests/Character_Chat_NEW`, `tests/Embeddings`, `tests/Evaluations`, `tests/prompt_studio`, `tests/Admin`, `tests/AuthNZ`, `tests/Audio`.

**Interfaces:**
- Consumes: `core.testing.is_test_context()` (Task 10), `is_test_mode()` and `env_flag_enabled()`.

- [ ] **Step 1: Hand-lower `predicates` to 0** for the 13 modules in `test_mode_branch_baseline.json`. No new behaviour test: every replacement keeps its module's signal set. `tests/Audio`'s shim-patching tests pin the audio lookup, which loses only a tie-break between two patched shims that no test or production path uses.

- [ ] **Step 2: Run the ratchet; it fails** with 13 `predicates N > ceiling 0` lines.

- [ ] **Step 3: Implement** (the table), and retarget the patch lines.

- [ ] **Step 4: Run the ratchet and every listed directory**, with `RUN_EVALUATIONS=1`. Expected: pass, with no failure that the same files don't show on dev.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`. Expected: predicates down by 13.
  - Remove the stale ALLOWLIST pairs for `api/v1/endpoints/character_chat_sessions.py` and `api/v1/endpoints/writing.py`. `embeddings_v5_production_enhanced.py` keeps its pair (`:521`), and so does `prompt_studio_evaluations.py` (`:378`).
  - Run C1, C2, C3 and C4.

- [ ] **Step 6: Commit.**

```bash
git add tldw_Server_API/app/api tldw_Server_API/tests
git commit -m "refactor(endpoints): endpoint test-mode predicates deleted; callers use core.testing (TASK-13950.19)"
```

### Task 12: Core and package-root predicates (AC2)

**Files:**
- Modify, deleting each predicate:

  | Module | Predicate | Replacement |
  |---|---|---|
  | `core/TTS/adapters/{chatterbox,dia,higgs,vibevoice}_adapter.py` | `_is_test_runtime` (`:67`, `:44`, `:44`, `:68`) | `(is_test_context() or env_flag_enabled("MINIMAL_TEST_APP"))` at each `_get_torch(allow_import=... and not ...)` call, as `kokoro_adapter.py:209` does. frontend-required (`:712`) and frontend-ux-gates start servers outside pytest with `MINIMAL_TEST_APP=1`, and those must keep skipping the torch import |
  | `core/Embeddings/audit_adapter.py`, `core/Evaluations/audit_adapter.py` | `_in_test_mode` (`:56`, `:55`) | `is_test_context()`. Delete `test_embeddings_audit_test_mode_parsing` and `test_evaluations_audit_test_mode_parsing`, which test the deleted copies |
  | `core/Evaluations/unified_evaluation_service.py` | `_await_webhook_inline_in_test_mode` (`:97`) | `is_test_mode()` at `:789`, `:927`, `:1039`. Retarget 2 patch lines |
  | `core/Watchlists/fetchers.py` | `_in_test_mode` (`:78`) | its imported `_is_test_mode()` (`core.testing.is_test_mode`) at `:299`, `:376`, `:511` |
  | `core/Chat/rate_limiter.py` | `_legacy_test_limits_enabled` (`:186`) | inline at `:289`: `is_test_mode() and any(os.getenv(name) is not None for name in _LEGACY_TEST_LIMIT_KEYS)`, with the five `TEST_CHAT_*` names as a module tuple |
  | `core/Embeddings/Embeddings_Server/Embeddings_Create.py` | `_should_use_inprocess_test_embeddings` (`:362`) | inline at `:2023`: `provider.strip().lower() == "huggingface" and is_test_mode() and env_flag_enabled("E2E_INPROCESS")` |
  | `core/Sandbox/runners/lima_enforcer.py` | `_test_override_mode` (`:41`) | `is_test_context()` at `:59` |
  | `__init__.py` | `_under_pytest` (`:46`) and the `if _env_flag_true("TESTING") or _under_pytest():` block (`:61-72`) | delete both. The block's `os.environ.setdefault(...)` lines move to the repo-root `conftest.py`, above `pytest_plugins`. Delete `_env_flag_true` if nothing else uses it |

- The TTS replacement drops only `_is_test_runtime`'s `sys.argv` check. That matters only before pytest sets `PYTEST_CURRENT_TEST`, and the four `_get_torch(...)` calls run inside device-probe functions (`dia_adapter.py:77`), not at import.
- Inlining can raise a module's line count, because the inline `is_test_mode()` is a counted signal and the old call name was not. Candidates: `Embeddings_Create.py:2023` and `Chat/rate_limiter.py:289`. If `--tighten` cannot absorb it, add a bump entry with the reason "predicate inlined (TASK-13950.19)".
- Test: the ratchet via the baseline, and each module's own tests: `tests/TTS`, `tests/TTS_NEW`, `tests/Embeddings`, `tests/Evaluations`, `tests/Watchlists`, `tests/Chat`, `tests/sandbox`.
- Test: `tldw_Server_API/tests/unit/test_pytest_thread_env.py` (new).

**Interfaces:**
- Consumes: `core.testing.is_test_context()`, `is_test_mode()` and `env_flag_enabled()`.

- [ ] **Step 1: Write the failing pins.**
  - Hand-lower `predicates` to 0 for the 12 modules.
  - In `test_pytest_thread_env.py`, add `test_pytest_runs_with_single_threaded_numeric_libraries`, asserting `os.environ["JOBLIB_MULTIPROCESSING"] == "0"`, `os.environ["OMP_NUM_THREADS"] == "1"` and `os.environ["TOKENIZERS_PARALLELISM"] == "false"`. It passes before and after; it pins that moving the defaults from app import to the root conftest keeps them.

- [ ] **Step 2: Run them.** Expected: the ratchet FAILs with 12 `predicates N > ceiling 0` lines; the env pin passes.

- [ ] **Step 3: Implement** (the table).

- [ ] **Step 4: Run the ratchet, the new test file and the listed directories** (`RUN_EVALUATIONS=1`). Expected: pass, with no failure that the same files don't show on dev.

- [ ] **Step 5: Lower the ratchets.**
  - Run `--tighten`. Expected: predicates 0 in every entry.
  - Add `test_no_private_test_mode_predicates` to Task 1's module: `assert sum(entry["predicates"] for entry in measure().values()) == 0`, with no bump path. From here on a predicate cannot come back through a reviewed bump.
  - Remove the stale ALLOWLIST pairs for:
    - `__init__.py`;
    - `core/Embeddings/audit_adapter.py` and `core/Evaluations/audit_adapter.py`;
    - `core/Sandbox/runners/lima_enforcer.py`;
    - the four `core/TTS/adapters/*_adapter.py`.
  - Run C1, C2, C3 and C4.

- [ ] **Step 6: Commit, then run C6** (a root-conftest change).

```bash
git add tldw_Server_API/app conftest.py tldw_Server_API/tests
git commit -m "refactor(core): the last private test-mode predicates are deleted; pytest thread limits move to the root conftest (TASK-13950.19)"
```

### Task 13: Close-out (AC1-AC5, DoD)

**Files:**
- Modify (via `tools/backlog-py` only): `backlog/tasks/task-13950.19 - ....md`.
- Create (via `tools/backlog-py`): one follow-up task for the deferred security-marked reads.

- [ ] **Step 1: Check each AC on dev.**
  - **AC1.** `test_mode_branch_baseline.json` exists, and C1 runs it. Record the seed measured at PR 1's execution (702 / 184 / 40 expected on the `.13` stack; ruling 1) and the final totals.
  - **AC2.** `"$PY" -c "import json; d=json.load(open('tldw_Server_API/tests/lint/test_mode_branch_baseline.json')); print(sum(v['predicates'] for v in d.values()))"` prints `0`.
  - **AC3.** Tasks 3-9 merged. security-required's step ran on the last PR with the expected suite count.
  - **AC4.** `test_app_middleware_stack.py` passes.
  - **AC5.** These pass: `test_backend_required_runs_the_test_seam_ratchets_and_guard_tests`, `test_security_required_runs_the_test_mode_unset_suites` and their mutation tests.
- [ ] **Step 2: File the follow-up** with `tools/backlog-py`.
  - Title: "Move the remaining security-marked test-mode reads behind seams".
  - Description: the "Deferred" row of the Inventory table, with file:line re-measured on dev. That includes the spoofed-marker path and the two `AUTHNZ_FORCE_REAL_SESSION_MANAGER` leftovers.
  - Dependency: TASK-13950.19.
  - Also record with `tools/backlog-py` that TASK-13950.19's dependency on TASK-13950.18 applies only to Task 6 (controller ruling 2, 2026-10-30).
- [ ] **Step 3: Record notes and the final summary** with `tools/backlog-py`.
  - Record:
    - the measured drift (710/185/40 vs the review's 491/147/~33);
    - each PR number;
    - the Full Suite comparisons;
    - the follow-up id.
  - DoD #4: run `bandit -q -r <changed app files>` per PR, and record "no new findings".
  - Stage with `git add -A backlog/tasks`.

## Self-review

- **Spec coverage.**
  - AC1: Task 1. The seed is 702 lines in 184 modules after `.13` (710 / 185 before it), and 40 predicates in 36 modules.
  - AC2: Tasks 4, 5, 9, 10, 11 and 12 delete all 40 predicates, and every caller uses `core.testing` or the injected MCP provider. Task 12 then pins the total at 0 with no bump path.
  - AC3: Tasks 3-9 cover the seven backlog sites and their same-path siblings, including the stored test key (ruling 3). Task 2 adds the required TEST_MODE-unset run, hardened against a neutered step. Task 1's function-level zero pin covers what that run cannot see.
  - AC4: Task 9.
  - AC5: Task 1 Step 7 and Task 2 Steps 1 and 8.
  - The roadmap's "required check in the first slice" is PR 1, with no ruleset change.
- **Ratchets and gates each PR trips.**
  - Task 1's ceilings: lowered by `--tighten` in every PR. Its `SECURITY_FUNCTIONS` pin grows in Tasks 3-8.
  - `.6`'s ALLOWLIST: drained per task, with the pairs named.
  - `.6`'s production-detector lint: untouched; no task reads a production key.
  - `.8` size ceilings: only shrink.
  - `.9` layering and `.16`'s `db_management_logic_imports`: no new import edge. `core.testing` is foundation, and `is_test_context()` imports nothing.
  - `.11` HTTP usage guard (`SCOPE_ALLOWLIST` by file): no HTTP call changes in `http_client.py` or the webhook modules.
  - `.13`'s direct-adapter ratchet lists `core/AuthNZ/byok_testing.py`. Task 10 touches only its predicate there, so the count is unchanged. The LLM_Calls import-isolation guard is untouched.
  - `.5` keyword ratchet: no keyword or tag code is touched.
  - `.7` WebSocket ratchets: Task 4 changes MCP WebSocket compat auth and runs them in C1.
  - `.12` foundation pins: `core/testing.py` gains a stdlib-only function, and `http_client.py` only loses imports.
  - `scope_predicate_baseline.txt`: no listed file.
  - ci.yml shard coverage: C8, with one shard edit in Task 9.
  - Test-quality triage: covered in Global Constraints.
- **Type consistency.**
  - `is_test_context() -> bool` is defined in Task 10 and used in Tasks 11-12 under that name.
  - `inline_io` / `inline_io_default` are defined and used only in Task 8.
  - `TEST_MODE_UNSET_SENTINEL`, `TEST_MODE_UNSET_SUITES` and `_test_mode_unset_problems` are defined and used only in Task 2.
  - `count_module(tree, *, count_predicates=...)`, `function_reference_lines(tree, qualname)` and `SECURITY_FUNCTIONS` match between Task 1's Interfaces, its tests, and Tasks 3-8's additions.
  - `.18`'s `app_services()`, `AppServices.session_manager` and `fresh_services` are used verbatim in Task 6.
