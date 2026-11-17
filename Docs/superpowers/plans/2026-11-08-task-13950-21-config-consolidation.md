# Consolidate configuration sources behind typed sections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze env reads outside the config modules and undocumented env names, route every env-over-config resolution through one resolver in `core/config_sections`, and give each key that two typed sources resolve differently one home in `config_sections`, with LazySettings, AuthNZ `Settings` and MCPConfig delegating to it.

**Architecture:** Ratchets first, then one shared key per PR.
- Two new AST ratchets in `tests/lint` run in backend-required's ratchet step:
  - a per-module ceiling on env reads, `{"reads", "import_time"}`;
  - a shrink-only list of env names that the curated Env_Vars doc does not mention.
- One resolver, `config_sections/precedence.py:resolve`, replaces the private precedence copies in `config_sections`, `config.py`'s helper family and its 27 standalone getters, and AuthNZ's `maybe_set`. A golden characterisation test pins every value, every env key and config option read, and their order. It is generated on dev before each refactor.
- The `.14`/`.15` import-time handoffs move to call time, or to a typed section.
- The shared keys then move one per PR, following a template.
  - Each key gets a full resolver (`resolve_<key>`) for readers that want the effective setting, and an env-layer accessor (`<key>_env`) for readers that need the raw environment value: env-drift detectors, key rings, the generic content backend and optional-feature switches.
  - A reader's behaviour changes only through an owner-approved item of decision D1.

**Tech Stack:** Python 3.12/3.13, FastAPI, pydantic-settings, pytest, AST lint tests under `tldw_Server_API/tests/lint/` sharing `_ast_cache.py` and `_ceilings.py`, GitHub Actions (`backend-required.yml`, `security-required.yml`).

**Spec:**
- `backlog/tasks/task-13950.21 - Consolidate-configuration-sources-behind-typed-sections.md`, AC #1-#5 and its Implementation Notes (the `.15` S2 handoff).
- The `.21` row of `## Phase 2` in `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`: effort M; "ratchets first, then one shared key per PR"; after `.6`. Its metric row: "Env reads outside config modules, about 2,694", owned by `.21`.

**Conventions:**
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Run every command from the repo root with `PYTHONDONTWRITEBYTECODE=1`.
- pytest runs as `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-<name>"`.
  - Do not pass `-p pytest_asyncio.plugin` locally ("Plugin already registered").
  - Do not set `PYTEST_DISABLE_PLUGIN_AUTOLOAD` locally either, except in C4, which mirrors CI and passes the plugins explicitly.
- File:line references are on `origin/dev` 7a020da14a. Re-locate them by the quoted text.

## Measured (`origin/dev` 7a020da14a, final counter spec)

- **The typed sources.**
  - `core/config.py`: 6,063 lines. `LazySettings` at `:5843`, imported by 118 app modules.
  - AuthNZ `Settings`: `AuthNZ/settings.py:159`, imported by 99 app modules.
  - `MCPConfig`: `MCP_unified/config.py:191`, imported by 11.
  - `config_sections`: imported by 5 modules outside itself.
  - Backlog figures: 6,084 lines and 117 / 103 / 5 importers.
- **What the env-read ratchet counts** (Task 1 specifies the matcher):
  - direct keyed reads;
  - literal-key calls to env helpers, found transitively and through any parameter position;
  - resolver calls outside the config modules;
  - import time, including module-level functions that import-time code calls.
- **What it excludes, and why:**
  - Reads whose literal names are all test-mode flags (`is_signal_constant`): 74 reads in 43 modules. Ruling 3: `.19`'s baseline freezes them.
  - `core/testing.py`: `.19`'s seam module. Its helpers still count at their call sites.
  - In-app `tests` packages.
  - Writes: item assignment (45), `setdefault` (36), `pop` (4).
  - Bulk copies and scans: `dict(os.environ)`, `.copy()`, `**os.environ`, `.items()`, `.keys()` (14 copies). They pass the environment on to subprocesses; they read no setting.
  - Pydantic `BaseSettings` reads: implicit. Task 2 counts their names.
  - Inside the config modules only: resolver calls and calls to resolver-backed helpers. That is the sanctioned path.

| Scope | Reads | Modules | At import time |
|---|---|---|---|
| Outside the config modules (the roadmap metric) | 3,300: 2,613 direct, 687 helper calls | 609 | 266 |
| Config modules: `core/config.py` 359, `core/AuthNZ/settings.py` 28, `core/config_sections/*` 63 in 14 modules, `core/MCP_unified/config.py` 1 | 451 | 17 | 6 |
| Baseline entries | 626 | | |

- **#3433 shifts the seed.** With #3433's `app/` diff applied:
  - outside the config modules: 3,302 reads, 608 modules, 265 at import time; 625 entries;
  - `app/__init__.py` drops out;
  - `chatterbox_adapter.py`, `dia_adapter.py` and `vibevoice_adapter.py` grow from 4 to 5 reads each, because `env_flag_enabled("MINIMAL_TEST_APP")` counts.
  - PR 1 therefore seeds after #3433 merges (Sibling coordination).
- **Drift.** The review measured 2,694 reads in 592 files, 162 at import time, and 1,508 names. Direct reads alone are 2,613 in 588 modules; the helper calls and the import-time closure are the difference.
- **Undocumented env names (AC2).**
  - The names are the literal names of every read (test flags and config modules included) plus the 208 `BaseSettings` env names: 2,380 names in all.
  - The curated `Docs/Operations/Env_Vars.md`, mirrored to `Docs/Published/Env_Vars.md`, documents 485 of them.
  - With 9 OS/runtime names allowlisted, the seed is 1,886.
  - The review's "about 570" also counted the root `Env_Vars.md`. That file is generated from code (scope ruling 1), so a regeneration would "document" any name.
- **Restated precedence (AC3).**
  - `config_sections`: ten `_get_raw` copies plus inline `env_map.get(...)` reads in `auth`, `audio`, `providers` and `rag`.
  - `config.py`: the `:100-197` helper family and 27 standalone getters (Task 4).
  - AuthNZ: `maybe_set` (`settings.py:1371-1376`).
  - These copies differ on whether a blank env value wins, and on stripping.
  - Three frozen functions keep inline reads: `load_settings` (191), `load_comprehensive_config` (58) and `load_and_log_configs` (49). Scope ruling 2 covers them.
- **Shared keys (AC4).** A shared key is an env name that two or more typed sources resolve: 11 names in 8 moves (Task 7's table).
  - MCPConfig resolves none; all 76 of its names are `MCP_`-prefixed aliases.
  - Every shared key resolves differently in at least two sources, which is why D1 exists. For example, with `AUTH_MODE=multi_user` and `APP_MODE` unset, LazySettings reports `SINGLE_USER_MODE=True`.
- **The `.14` / `.15` handoffs.**
  - `api/v1/schemas/chat_validators.py:30-67` (`MAX_REQUEST_SIZE` at import). It is the only entry of `test_schemas_config_boundary.py:52-57`.
  - `core/RAG/rag_schemas_unified.py:21-40`. Its contextual `_DEF_*` defaults never read config, because `isinstance(LazySettings, dict)` is False. `_DEF_FTS_LVL` works, but materialises `load_settings()` at import.

## Rulings

Controller rulings (binding):

1. Ratchets come first, each with a seeded-violation self-test and a contract assertion in `tests/CI/test_license_first_workflow_contracts.py`. Then one shared key moves per PR. The metric is measured on dev, and Measured states what is counted and what is excluded.
2. `MAX_REQUEST_SIZE` and the RAG contextual defaults are covered, in Tasks 5 and 6. Task 6 applies the configured values, per D3.
3. Test-mode flag reads are excluded from the count and not re-routed.
4. `.20` moves code with its config read unchanged; `.21` then retargets the read in its new home.
5. Defaults, precedence and key names do not change silently. Each one this plan touches is pinned, and each real conflict is an owner decision.
6. (Fix round 1)
   - PR 1 seeds on the dev tip after #3433 merges, and `.21` SDD starts only after `.19` closes.
   - A sibling that grows a count after the seed raises that module's ceiling in `env_read_baseline.json`, and fills the entry's `"reason"` field with text naming its PR (the `_ceilings` bump-with-reason rule; this baseline has no `# bump:` comments).
   - Task 5 keeps the fail-soft config load.
   - The AUTH_MODE overlay must fail closed on a bad `config.txt` value.
   - `DatabaseConfig.from_env()` keeps today's source.
   - The admin-webhook key ring loses no key.

Scope rulings (this plan's, with reasons):

1. **"The Env_Vars docs" in AC2 means the curated `Docs/Operations/Env_Vars.md`**, which `Docs/Published/Env_Vars.md` mirrors. The root `Env_Vars.md` is a generated dump.
2. **AC3's "one resolver is used" is met when:**
   - every precedence helper and standalone getter in the config modules goes through `resolve`;
   - `config_sections` holds no private copy;
   - the env ratchet freezes the inline reads in the three frozen loader functions;
   - Task 8 files those inline reads as a follow-up and records AC3 as met under this ruling.
3. **AC4's MCPConfig clause is met by measurement.** Task 1's `test_mcp_config_resolves_no_shared_key` keeps it so. MCP Unified's standalone boundary is untouched.
4. **Retargeted readers keep their timing.** A reader that read `os.getenv` live calls a live `config_sections` function, never a cached settings object.
5. **"AuthNZ `Settings` delegates" means the config layer.** Pydantic keeps reading the env layer itself (`case_sensitive=False`). `_load_overrides_from_config` takes each shared key's `config.txt` value from `resolve_<key>`.
6. **A split key is not an exception.** When an env name serves two settings, each setting gets its own home and one rule in `config_sections`. `DATABASE_URL` is both AuthNZ's users DB and the generic content-backend override. `REDIS_URL` is AuthNZ's, LazySettings' `[Redis]`, and the optional-feature switch. Task 8 records the splits.

## Owner decisions (answered 2026-11-08)

- **The owner's answers (2026-11-08):**
  - **D1:** all items (a)-(h) are approved.
  - **D2:** keep `ROUTES_*` test-only, as recommended.
  - **D3:** apply the configured values now, which was not the recommendation. Task 6 does it.
- The rejection paths below were the fallbacks if an item had been rejected. They are now moot, and are kept only as the record of what each item changes.


- **D1. One rule per shared key: approved (a)-(h).** Each item was approvable on its own, with a no-op rejection path. The PRs for (c) and (e) carry a release note.
  - **(a) `SINGLE_USER_MODE` follows `AUTH_MODE`.** LazySettings' `SINGLE_USER_MODE` becomes `load_auth_config().mode == "single_user"`, replacing its `APP_MODE` read (`config.py:965-966`). LazySettings stops reading `APP_MODE`.
    - `config_sections.auth` keeps today's tolerant `APP_MODE` fallback, which is reached only when `[AuthNZ] auth_mode` is invalid. AuthNZ already refuses to start in that case.
    - Visible effects:
      - character rate limiting (`character_rate_limiter.py:472-474`) turns on by default for multi-user deployments that never set `APP_MODE=multi`;
      - it turns off by default for single-user deployments that set `APP_MODE=multi`;
      - `load_settings`' warnings (`config.py:2320-2343`) change for multi-user deployments. The false "SINGLE_USER_API_KEY is not configured" error stops. "Multi-user mode enabled (APP_MODE=multi), but USERS_DB_ENABLED..." and the CRITICAL default-JWT warning start whenever `USERS_DB_ENABLED` or `JWT_SECRET_KEY` is not set in the environment.
    - **Rejection:** LazySettings keeps `SINGLE_USER_MODE = APP_MODE != "multi"`. `APP_MODE` stays out of `MOVED_SHARED_KEYS`, and Task 8 records an AC4 exception for it. `AUTH_MODE` still moves.
  - **(b) The env-only `AUTH_MODE` readers follow the effective mode** through `load_auth_config().mode`, which reads env and then `config.txt`, and never raises. The readers:
    - `setup.py:808`, `command_router.py:195`, `Sync/v2/security.py:131`, `claims_review_metrics_scheduler.py:132,164`, `startup_heavy_init.py:405`, `Evaluations/config_validator.py:82`, `main.py:2508` (its CORS fallback), and the telemetry in `endpoints/chat.py:2788,3867`;
    - `db_path_utils.py:62`: multi-user deployments configured only in `config.txt` stop using the single-user id for DB paths, which is an isolation tightening.
    - **Rejection:** these readers call `auth.auth_mode_env()`, with today's values.
  - **(c) Connector secret encryption follows the effective mode** (`connectors_service.py:144`), through `load_auth_config().mode`. It becomes required for multi-user deployments that set `auth_mode` only in `config.txt`. Without an encryption key, their connector saves fail closed, as they already do when the mode is set in the environment.
    - **Rejection:** it reads `auth.auth_mode_env()`, as today.
  - **(d) The `SINGLE_USER_API_KEY` setting readers see `[AuthNZ] single_user_api_key`:** LazySettings (and through it `prompts.py:382,390`), and `config_validator.py:86`.
    - **Rejection:** they read `single_user_api_key_env() or api_key_env()`, as today.
  - **(e) `require_token_scope` accepts only AuthNZ's resolved key** (`auth_deps.py:2294`). It stops accepting an env `SINGLE_USER_API_KEY` or `API_KEY` that differs from that key, so a client that still sends such a legacy key gets 401.
    - **Rejection:** it keeps adding `single_user_api_key_env() or api_key_env()` to the accepted set, as today.
  - **(f) LazySettings honours `[AuthNZ] jwt_secret_key`.**
    - Notes attachment cursors are signed with it, so each cursor issued before the change gets one 400.
    - The chat share-link fallback (`chat.py:6884`) signs with it instead of raising.
    - `config_validator.py:113` sees it.
    - **Rejection:** these readers use `jwt_secret_key_env()`, and LazySettings falls back to its own default, as today.
  - **(g) `config_sections.auth` stops reading `[AuthNZ] single_user_fixed_id`** (a rule change). The shipped `config.txt` does not set it, a warning is logged once if it is set, and AuthNZ and LazySettings never read it.
    - **Rejection:** row 2 is dropped, `SINGLE_USER_FIXED_ID` stays out of `MOVED_SHARED_KEYS`, and Task 8 records an AC4 exception.
  - **(h) The RAG default LLM rule** (a rule change). A blank env value no longer wins in LazySettings or `generation_defaults.py:18,22`; it falls through to `config.txt`. `config_sections.rag` reads the shipped option names.
    - **Rejection:** row 8 is dropped, both RAG keys stay out of `MOVED_SHARED_KEYS`, and Task 8 records an AC4 exception.
  - **Unchanged by design, each through an env-layer accessor or a split (scope ruling 6):**
    - `DATABASE_URL`: the generic content backend and RLS-ensure target, preflight, AuthNZ's env-drift detection;
    - `REDIS_URL`'s optional features;
    - the admin-webhook key ring, Prototype Workspaces' signing candidates, and the setup wizard's `.env` checks;
    - `load_auth_config`'s tolerant validation.

    `ACCESS_TOKEN_EXPIRE_MINUTES`: LazySettings' unused entry is deleted.
- **D2. `ROUTES_*` env overrides** are honoured only in a test runtime, but `config.txt:1445-1448`, `Config_Files/README.md:702-706` and `Tools_API_Documentation.md:20-25` tell users to export them.
  - **Recommendation:** keep them test-only, as ef1088b3c1 chose. Pin that as the documented exception, and correct the three docs (Task 4). This changes no behaviour.
- **D3. RAG contextual defaults never reached the request models: apply them now.**
  - The recommendation was literals plus a follow-up; the owner chose to apply the values now.
  - Task 6 applies the configured values at call time when a request leaves the field unset. OpenAPI defaults stay static literals.

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

The four 2026-10-25 owner decisions do not bind this task. The owner waived Change summaries for all TASK-13950 PRs and authorized merging them after green CI and a clean review (2026-10-25); record the summary section anyway.

Task-specific:

- **No silent behaviour change.**
  - A refactor slice (Tasks 3, 4) proves equality with a golden generated on dev *before* its edits.
  - A behaviour change is a reviewed diff of expected values, committed with the code: an approved D1 item (Task 7), or D3 (Task 6).
- **What raises, and where.**
  - `resolve` and every `resolve_<key>` never validate. They return the raw text, and raise only what the parser raises (a `configparser.Error` other than a missing section or option, such as `InterpolationSyntaxError`), exactly as today's copies do.
  - `load_auth_config` keeps today's tolerant validation and never raises for a bad value.
  - The only validation that rejects a bad value is pydantic's `ValidationError` in `Settings(**overrides)`, outside `_load_overrides_from_config`'s swallow-all `try` (`_SETTINGS_NONCRITICAL_EXCEPTIONS`, `settings.py:42-50`).
  - So nothing inside the overlay raises for a bad value, and the whole `[AuthNZ]` overlay is never dropped.
- **Secrets.** A typed field holding a secret, or a URL that can carry a password, is declared `field(repr=False)` and is never logged. Pinned by `test_auth_config_repr_hides_secrets`.
- **Security pins run in a required gate.** They live in `tldw_Server_API/tests/AuthNZ/unit/test_shared_config_key_security.py`, which security-required's TEST_MODE-unset step (C4) runs by directory.
- **Lower the ratchets in every slice.**
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_env_read_ratchet --tighten` and `"$PY" -m tldw_Server_API.tests.lint.test_env_vars_docs_ratchet --tighten`, and commit the diffs.
  - A slice that moves reads to a module with no entry runs `--write-baseline`, checks that the diff only adds that entry and lowers or removes others, and gives the new entry a `reason`.
  - A refactor that makes a hidden env name visible documents it in the same PR (C9) or allowlists it with a reason.
- **Mirrored docs.** Any edit to `Docs/Operations/Env_Vars.md` or `Docs/API-related/*` runs `bash Helper_Scripts/refresh_docs_published.sh` and commits the `Docs/Published` diff (C9).
- **Function sizes.** `config:load_settings` (1,416), `config:load_and_log_configs` (1,993) and `auth_deps:require_token_scope._checker` (446) are frozen. An edit inside them keeps its net line count at zero or below.
- **`.18`'s getter ratchet.** In `api/v1/endpoints/**`, a new zero-argument `get_*()` call in a function body raises its baseline. Retargets there call `config_sections` functions (`load_*`, `resolve_*`, `*_env`), which do not match `get_*`.
- **No conftest or shared-fixture edits.** No slice deletes or renames a test file. Only Task 7's `AUTH_MODE` row needs C6, because D1(a) (approved) flips `SINGLE_USER_MODE`.
- **No legacy baseline regeneration expected.** `git diff --name-only origin/dev | sed 's#^#^#' | grep -f - Helper_Scripts/ci/scope_predicate_baseline.txt` prints nothing.
- **Test-quality triage.** `monkeypatch.setattr` → `monkeypatch.setenv` keeps `mock_density` counts. Re-run C2 after every fix round that touches tests.
- **Chat tests can hit the network.** Task 5 edits `Chat/integration/test_chat_endpoint_image_validation.py`. The oversized request must be rejected before any provider call; stub the adapter if one is reached.

## Review Focus

1. **A ratchet passes for the wrong reason.** Pinned in Task 1 by `test_the_counter_counts_each_read_shape` and `test_reads_carry_their_literal_names`. The ways in:
   - an env read behind a wrapper of a wrapper, or a key in a non-first or keyword-only parameter;
   - an `os.environ` alias made by an annotated assignment;
   - `env_flag_enabled` from `core/testing.py`;
   - a resolver call outside the config modules;
   - an import-time read inside a function called at module level.

   On the docs side, the doc-index pins in Task 2 cover a name "documented" by a generated list or by a placeholder inside a lower-case word.
2. **The resolver refactor drifts.** The risks: a blank or whitespace value, an unparseable value, a renamed key or option, a changed alias order, or a parser error that is now swallowed. Pinned by `test_loaders_and_helpers_match_the_dev_golden` (Tasks 3-4). Its spy fakes record the values *and* every env key and config option asked for, in order, over eleven scenarios, `cfg_raises` included.
3. **Auth fails open or loses data in a key move.** Pinned in `test_shared_config_key_security.py`:
   - a bad `AUTH_MODE`, in the environment *or* in `config.txt`, must still stop AuthNZ: `test_invalid_auth_mode_fails_closed_in_every_source`;
   - stored admin-webhook secrets must still decrypt: `test_webhook_secret_ring_still_decrypts`;
   - the accepted single-user keys change only under D1(e): `test_single_user_api_key_accepted_keys`;
   - secrets stay out of `repr`: `test_auth_config_repr_hides_secrets`.
4. **The RLS target or the content backend moves.** `DatabaseConfig.from_env()` must not pick up `[AuthNZ] database_url`. Pinned on Postgres by `test_rls_ensure_target_ignores_authnz_database_url` (Task 7, row 4).
5. **Route toggles honour env in production after the refactor**, or stop honouring it in tests. Pinned in Task 4 by the two `test_route_toggle_env_overrides_*` tests.

## CI-equivalent checks (each task names the ones it needs)

| Id | What | Command | Pass means |
|---|---|---|---|
| C1 | backend-required "Enforce CI contracts and code ratchets" | Run the step's two pytest commands from `.github/workflows/backend-required.yml` (`:296-338`) verbatim, with `-p pytest_asyncio.plugin` and `-p pytest_timeout` removed and `-p no:cacheprovider --basetemp "$TMPDIR/t1321-c1"` added, under `BACKLOG_TASK_FORMAT_BASE=origin/dev PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1`. Do not set `PYTEST_DISABLE_PLUGIN_AUTOLOAD`, because the plugin flags are removed | all pass |
| C2 | backend-required "Enforce tenant isolation ratchets" | same env, `tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` | all pass. A lone `libc++abi ... recursive_mutex` crash in `test_route_auth_ratchet.py` is a local flake: rerun the file |
| C3 | mypy on changed modules (the CI step is non-blocking) | `uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports $(git diff --name-only --diff-filter=d origin/dev -- 'tldw_Server_API/app/*.py')` | no error on a line this PR changed |
| C4 | security-required "Security suites with TEST_MODE unset" | `env -u TEST_MODE -u TESTING -u TLDW_TEST_MODE TLDW_UNSET_TEST_FLAGS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. DISABLE_HEAVY_STARTUP=1 TLDW_TEST_NO_DOCKER=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-c4" -p pytest_asyncio.plugin -p pytest_timeout --timeout=300 -o timeout_method=signal tldw_Server_API/tests/http_client tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Security tldw_Server_API/tests/CI/test_unset_test_flags_run.py` | all pass |
| C5 | Postgres shard | `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$TMPDIR/t1321-pg" <files>`, with `POSTGRES_TEST_*` set as in ci.yml's Postgres shard, or Docker auto-start (`tests/_plugins/postgres.py`) | pass, and `-rs` lists no skipped Postgres case |
| C6 | Full Suite comparison, a **merge gate** where named | After the PR's `CI` run: `gh run view <pr-run> --log-failed \| grep -oE '(FAILED\|ERROR) tldw_Server_API/[^ ]+' \| sort -u > pr.txt`; the same over the `CI` runs of the three most recent other backend PRs on dev, unioned into `base.txt`; then `comm -23 pr.txt base.txt`. Re-run any printed id on dev in a scratch worktree; one that fails there is not new. `test_model_completion_accounting.py` ("fixture 'mocker' not found") fails on every dev run | prints nothing; the PR description pastes it |
| C7 | OpenAPI drift gate (Task 6) | `PYTHONPATH=. "$PY" Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | exit 0 |
| C8 | Shard coverage and required-workflow contracts, for every PR that adds a test file | `"$PY" Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` and `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-c8" tldw_Server_API/tests/CI/test_required_workflow_contracts.py` | exit 0; pass |
| C9 | Docs mirror, for every PR that edits a mirrored doc | `bash Helper_Scripts/refresh_docs_published.sh`, then `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-c9" tldw_Server_API/tests/Docs/test_docs_published_refresh.py` | pass, with the `Docs/Published` diff committed |

## PR slices

| PR | Task(s) | Base | Hand-written lines (est.) | Ratchet change |
|---|---|---|---|---|
| 1 | 1 | dev after #3433 merges and `.19` closes (ruling 6) | ~430, plus a generated baseline (~625 entries) | seeds the env ratchet |
| 2 | 2 | dev after PR 1 | ~170, plus a generated baseline (~1,886 names) | seeds the docs ratchet |
| 3 | 3 | dev after PR 2 | ~300, plus a generated golden | `config_sections` 63 → about 1 (`precedence.py`); `config.py` falls |
| 4 | 4 | dev after PR 3 | ~360 | `config.py` falls by the getters' reads; docs −4 names (`ROUTES_*`) |
| 5 | 5 | dev after PR 3 | ~140 | the `chat_validators.py` entry is removed |
| 6 | 6 | dev after PR 3 | ~370 | `config.py` −4 (the contextual lambda); docs −4 names |
| 7-14 | 7, once per row, in Task 7's order | dev after PR 4 | 50-300 per row (Task 7 table) | each row's readers fall |
| (close-out) | 8 | dev after PRs 1-14 | backlog only | `MOVED_SHARED_KEYS == SHARED_KEYS` |

- PRs 4, 5 and 6 all branch from PR 3, and all lower `env_read_baseline.json`. Whichever lands second merges dev and re-runs `--tighten`, which only lowers, so it is safe.
- PRs 7-14 all edit `config_sections/auth.py`, `AuthNZ/settings.py`, `config.py:load_settings` and the `MOVED_SHARED_KEYS` line. Land them one at a time. A rebase takes dev's files, re-runs both `--tighten`s and re-adds its own key.
- PR 1 announces both ratchets to open PRs. A PR that lands after the seed and adds an env read either avoids the read, or raises the module's ceiling with a `reason` naming the PR and documents any new name.

## Sibling coordination

- **`.20` endpoint shared code** (`2026-11-08-task-13950-20-endpoint-shared-code.md`; read and reconciled; both plans state the same rule).
  - **The config-read rule:** `.20` moves code with its env and config reads unchanged; `.21` then retargets each read in its new home.
    - If PR 1 is on dev when a `.20` move lands, the `.20` PR tightens the old module's `env_read_baseline.json` entry. It adds the new module's entry with the reason `"moved from <old path> (TASK-13950.20 PR <n>)"`.
    - If `.20` lands first, PR 1 seeds after it.
    - The modules involved are `_chatops/settings.py`, `chat_import_transport.py`, `discord_support.py`, `slack_support.py`, and the `media_embeddings.py` reads that move to `core/Embeddings/media_embeddings.py`.
  - **No Task 7 reader moves.** `.20` keeps `evaluations_auth.py:147` in place.
  - **Rebases.** `.20`'s codemod waves edit lines in `chat.py`, `setup.py`, `auth.py`, `embeddings_v5_production_enhanced.py`, `prompts.py` and `notes.py`, which Task 7 also edits. Whichever lands second rebases. A `.20` wave re-runs its Appendix A on the rebased file, and `.21`'s retargets keep their lines.
  - **The codemod and the counter.** The codemod rewrites `from M import f` + `f(...)` into `_m.f(...)`. Task 1's matcher resolves `<m>.<name>` through module aliases, so `.20`'s per-wave "totals unchanged" check holds.
  - Both plans add one line to `backend-required.yml` and one parametrize entry to the contract file, so those rebases are textual.
- **`.19`.**
  - PR 1 seeds after #3433 merges and `.19` closes. #3433 grows three TTS adapters and drops `app/__init__.py` (Measured), so seeding earlier would fail it.
  - `.19`'s follow-up ("Move the remaining security-marked test-mode reads behind seams") moves test-flag reads, which this ratchet excludes. If it also moves non-flag reads, it tightens; any growth takes a reason.
- **`.14` / `.15` (done).**
  - Task 5 deletes the last `IMPORT_TIME_CONFIG_ALLOWLIST` entry. `test_schemas_config_boundary.py` already runs in the step, under `test_backend_required_enforces_layering_ratchets`.
  - Task 6 covers `.15` S2.
- **`.6` (done).** Its guards are unaffected. The seam keys (`MINIMAL_TEST_APP`, `SINGLE_USER_TEST_API_KEY`, ...) are not test-mode flags, so they count.
- **Open PRs into `dev` touching planned files.** File lists come from `gh api repos/rmusser01/tldw_server/pulls/<n>/files --paginate`, because `gh pr list --json files` truncates at 100 files.

  | PR | Planned files or areas it touches | Sequence |
  |---|---|---|
  | #3394, #3405 | `backend-required.yml`, `test_license_first_workflow_contracts.py` | Textual rebase (adjacent step lines) |
  | #2869, #3269, #3294 | `test_license_first_workflow_contracts.py` | Textual rebase |
  | #3269 | `Env_Vars.md` and its mirror, `config.txt`, `notes.py`, `prompts.py`, `auth_deps.py`, `chat.py`, `setup.py`, `evaluations_auth.py`, `initialize.py`, `main.py`, `claims_review_metrics_scheduler.py` | After PR 1 it tightens, and gives any growth a reason. Tasks 4 and 7 rebase onto it |
  | #3434 | `core/config.py`, `db_path_utils.py` (`AUTH_MODE` row), `config.txt` | Tasks 3-4 regenerate the golden on the new dev before their edits |
  | #3394 | `AuthNZ/initialize.py`, `AuthNZ/migrations.py`, `connectors_service.py`, `main.py`, `function_size_baseline.json` | Task 7 rows rebase |
  | #2855 | `Admin_Webhooks/legacy_import.py`, `services/admin_webhooks_service.py`, both callers of the webhook key ring | Rows 6-7 rerun `test_webhook_secret_ring_still_decrypts` after it lands |
  | #3436 | `Web_Scraping/browser_transport.py` | `AUTH_MODE` row rebases |
  | #3437 | `Audio/stt_policy.py`, `endpoints/chat.py` | Task 3 (stt) and the `AUTH_MODE` row rebase |
  | #3020 | `character_rate_limiter.py` | `AUTH_MODE` row rebases |
  | #3071, #3226 | `endpoints/notes.py`; #3071 also `chat.py` and `main.py` | `JWT_SECRET_KEY` row rebases |
  | #3430 | `endpoints/auth.py` | `SINGLE_USER_API_KEY` row rebases |
  | #854 | `endpoints/chat.py`, `embeddings_v5_production_enhanced.py` | `AUTH_MODE` and `REDIS_URL` rows rebase |

- **Downstream.** `.26` and `.28` depend on `.20`. Nothing here renames a public function; the typed loaders only gain optional arguments.

---

### Task 1: The env-read ratchet and the shared-key pins (AC1, AC4 tracker, AC5)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_env_read_ratchet.py`
- Create (generated): `tldw_Server_API/tests/lint/env_read_baseline.json`
- Modify: `.github/workflows/backend-required.yml`.
  - After the comment line `# The test-mode branch ratchet (TASK-13950.19) does the same for its two ceilings.` (`:285`), add `# The env-read ratchet (TASK-13950.21) does the same for its two ceilings.`
  - After `tldw_Server_API/tests/lint/test_test_mode_branch_ratchet.py \` (`:323`), add `tldw_Server_API/tests/lint/test_env_read_ratchet.py \`.
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Before `test_backend_required_enforces_isolation_ratchets` (`:1510`), add `test_backend_required_enforces_config_ratchets(ratchet)`.
  - It copies `test_backend_required_enforces_llm_gateway_guards` (`:1484-1507`): the same three assertions.
  - Its docstring: `"""TASK-13950.21: the config ratchets run in a required step, with the PR base their reason check reads."""`.
  - It is parametrized over `["tldw_Server_API/tests/lint/test_env_read_ratchet.py"]`.

**Interfaces:**
- Consumes:
  - `_ceilings`: `Ratchet`, `violations`, `reasonless_bumps`, `warn_if_tightenable`, `load`, `main`;
  - `_ast_cache`: `APP_ROOT`, `parse_tree`;
  - `tldw_Server_API.tests.lint.test_test_mode_branch_ratchet.is_signal_constant`.
- Produces (module `tldw_Server_API.tests.lint.test_env_read_ratchet`):
  - **Constants:**
    - `CONFIG_MODULES: frozenset[str] = frozenset({"core/config.py", "core/AuthNZ/settings.py", "core/MCP_unified/config.py"})`;
    - `is_config_module(key: str) -> bool`, true for those keys and for any key under `core/config_sections/`;
    - `EXEMPT: frozenset[str] = frozenset({"core/testing.py"})`;
    - `RESOLVER: tuple[str, str] = ("tldw_Server_API.app.core.config_sections.precedence", "resolve")`.
  - **The excluded flag set.** A read is skipped when its names are non-empty and every one passes `.19`'s `is_signal_constant`. That accepts exactly `TEST_MODE`, `TESTING`, `TLDW_TEST_MODE`, `PYTEST_CURRENT_TEST` and `TLDW_TEST_[A-Z0-9_]+`.
    - Calls with no key, such as `is_test_mode()`, `is_explicit_pytest_runtime()` or `is_test_context()`, are not env reads at all.
    - `env_flag_enabled("TESTING")` is skipped. `env_flag_enabled("JOBS_COUNTERS_ENABLED")` and `env_flag_enabled("MINIMAL_TEST_APP")` count.
  - `class Read(NamedTuple)`: `line: int`, `names: tuple[str, ...]` (empty for a non-literal key), `kind: Literal["direct", "helper", "resolver"]`, `import_time: bool`.
  - **Direct reads:**
    - the forms: `os.getenv(k)`, `os.getenvb(k)`, `getenv(k)` imported from `os`, `<env>.get(k)`, `<env>[k]` in Load context, and `k in <env>` / `not in`;
    - `<env>` is `os.environ` under any `import os as x` alias, `environ` from `from os import environ [as y]`, or any name the module binds, by an `ast.Assign` **or an `ast.AnnAssign`**, to `os.environ` or to an `IfExp` with an `os.environ` branch.
  - `Helper = tuple[int, str, Literal["direct", "resolver"]]`: the key's index among the parameters (`self`/`cls` dropped, keyword-only included), its name, and its backing.
  - `env_helpers(trees: Mapping[str, ast.Module]) -> dict[tuple[str, str], Helper]`, keyed `(dotted module, function name)`. It runs over **every** non-`tests` module, `core/testing.py` included; only the read scan skips `EXEMPT`.
    - A function is `"direct"`-backed when one of its parameters is the key of a direct read in its body. A parameter iterated by a `for` or comprehension counts too, through its target.
    - It is also a helper when it passes a parameter, alone or inside a tuple or list, in the key position of a call to a helper or to `RESOLVER`. It inherits the callee's backing (`"resolver"` for `RESOLVER`). `"direct"` wins over `"resolver"`.
    - The search repeats to a fixed point.
    - A call's target, for helpers and `RESOLVER` alike, is found by:
      - a bare name in the same module, or `self.<name>`;
      - a name imported by `from <module> import <name> [as x]`, absolute or relative;
      - `<m>.<name>`, with `<m>` bound to the module by `import <module> as <m>` or `from <package> import <module> [as <m>]`. `.20`'s codemod writes `_testing.env_flag_enabled(...)`, and it must count the same as before.
    - The key argument is positional at the index, or the keyword of the parameter's name.
  - `module_reads(tree: ast.Module, module: str, helpers: Mapping[tuple[str, str], Helper]) -> list[Read]`.
    - Every direct read is kind `"direct"`, with the literal names of its key.
    - A call to `RESOLVER`, or to a `"resolver"`-backed helper, whose key argument is a string literal or a tuple or list of them, is kind `"resolver"`. The same call to a `"direct"` helper is kind `"helper"`.
    - A helper call with a non-literal key is not a read: the helper's own body read is counted.
    - Import time: outside every `def`/`lambda` body (decorators and defaults included), skipping `if __name__ == "__main__":` blocks, plus the bodies of module-level functions that import-time code calls by bare name, transitively.
  - `measure_reads(root: Path = APP_ROOT) -> dict[str, list[Read]]`: keyed by root-relative POSIX path. It skips `tests` directories and `EXEMPT`, and builds `env_helpers` over all non-`tests` modules.
  - `measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]`: `{"reads": n, "import_time": k}` per module with `n > 0`. It counts `"direct"` and `"helper"` everywhere, and `"resolver"` only outside the config modules, minus excluded flags.
  - `summary(measured) -> str`: the two lines `outside config modules: <n> reads in <m> modules, <k> at import time` and `config modules: ...`.
  - `settings_env_names(tree: ast.Module) -> frozenset[str]`. For classes whose bases end in `BaseSettings`, each class-level annotated field gives its `Field(validation_alias="X")` constant, or else its name if it is upper-case.
  - `SHARED_KEYS: frozenset[str] = frozenset({"AUTH_MODE", "APP_MODE", "SINGLE_USER_API_KEY", "API_KEY", "JWT_SECRET_KEY", "DATABASE_URL", "REDIS_URL", "SINGLE_USER_FIXED_ID", "ACCESS_TOKEN_EXPIRE_MINUTES", "RAG_DEFAULT_LLM_PROVIDER", "RAG_DEFAULT_LLM_MODEL"})` and `MOVED_SHARED_KEYS: frozenset[str] = frozenset()`.
  - `moved_key_problems(reads: Mapping[str, list[Read]], moved: frozenset[str]) -> list[str]`: for each module outside `core/config_sections/` with a read naming a moved key, the message `"<module>: reads <KEY>, which moved to core/config_sections (TASK-13950.21); read it through the typed section"`, sorted.
  - `RATCHET = _ceilings.Ratchet(baseline=Path(__file__).resolve().with_name("env_read_baseline.json"), command="python -m tldw_Server_API.tests.lint.test_env_read_ratchet", fix="Read the setting through its typed section in core/config_sections (or the settings object that owns it) instead of a new env read (TASK-13950.21).", stale="The module no longer reads the environment.")`.
  - `@cache def app_reads() -> dict[str, list[Read]]` (one counter pass, about 25 s, shared with Task 2's module), and `_app_measurement()` built from it.
  - `__main__`: `--summary` prints `summary(measure())`; any other arguments go to `_ceilings.main(RATCHET, measure)`.

- [ ] **Step 1: Write the self-tests** at the bottom of the module, with a `_seed(root, files)` helper as `test_direct_llm_adapter_calls_ratchet.py:128` has. The expected values were reproduced by running this spec over the same seed.

```python
SEED = {
    "core/X/direct.py": 'import os\nA = os.getenv("A1")\nB = os.environ.get("B1")\nC = os.environ["C1"]\nD = "D1" in os.environ\n',
    "core/X/aliased.py": 'import os as _os\nfrom os import environ as E, getenv as g\n\n\ndef f(env=None):\n'
                         '    env = _os.environ if env is None else env\n    return env.get("A2"), E["B2"], g("C2")\n',
    "core/X/annotated.py": 'import os\nfrom typing import Mapping\n\n\ndef f(env=None):\n'
                           '    env_map: Mapping[str, str] = env if env is not None else os.environ\n    return env_map.get("A3")\n',
    "core/X/helpers.py": 'import os\n\n\ndef _env_int(name, default=0):\n    return int(os.getenv(name) or default)\n\n\n'
                         'def _first(keys):\n    for key in keys:\n        if os.environ.get(key):\n            return key\n\n\n'
                         'def _outer(k):\n    return _env_int(k)\n\n\ndef _raw(cp, *, env_key):\n    return os.environ.get(env_key) or cp\n\n\n'
                         'X = _env_int("H1")\n\n\ndef g():\n'
                         '    return _env_int("H2"), _first(("H3", "H4")), _outer("H6"), _raw(None, env_key="H7")\n',
    "core/X/imported.py": 'from tldw_Server_API.app.core.X.helpers import _env_int\n\n\ndef h():\n    return _env_int("H5")\n',
    "core/X/flags.py": 'from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode\n\n'
                       'A = env_flag_enabled("FEATURE_X")\nB = env_flag_enabled("TESTING")\nC = is_test_mode()\n',
    "core/X/import_time.py": 'import os\n\n\ndef _limit():\n    return os.getenv("I1")\n\n\nLIMIT = _limit()\n\n\n'
                             'def later():\n    return os.getenv("I2")\n\n\nif __name__ == "__main__":\n    print(os.getenv("I3"))\n',
    "core/X/not_reads.py": 'import os\nos.environ["W1"] = "1"\nos.environ.setdefault("W2", "1")\nENV = dict(os.environ)\n'
                           'KEYS = [k for k in os.environ.keys() if k.startswith("TLDW_")]\n',
    "core/X/test_flags.py": 'import os\nA = os.getenv("TEST_MODE")\nB = os.getenv("TLDW_TEST_DB")\n',
    "core/X/resolver_user.py": 'from tldw_Server_API.app.core.config_sections.precedence import resolve\n\n\n'
                               'def _via(k):\n    return resolve((k,), config_parser=None)\n\n\n'
                               'def f(cp):\n    return resolve(("R1",), (("S", "o"),), config_parser=cp), _via("R3")\n',
    "core/config_sections/x.py": 'from .precedence import resolve\n\n\ndef _get_raw(cp, *, env_keys):\n'
                                 '    return resolve(env_keys, config_parser=cp)\n\n\ndef f(cp):\n'
                                 '    return resolve(("R2",), config_parser=cp), _get_raw(cp, env_keys=("K1",))\n',
    "core/config_sections/y.py": 'import os\n\n\ndef _get_raw(env_key):\n    return os.environ.get(env_key)\n\n\nA = _get_raw("K2")\n',
    "core/testing.py": 'import os\nX = os.getenv("Z1")\n\n\ndef env_flag_enabled(name):\n    return bool(os.getenv(name))\n',
    "core/X/tests/test_x.py": 'import os\nX = os.getenv("Z2")\n',
}


def test_the_counter_counts_each_read_shape(tmp_path: Path) -> None:
    assert measure(_seed(tmp_path / "app", SEED)) == {
        "core/X/aliased.py": {"reads": 3, "import_time": 0},
        "core/X/annotated.py": {"reads": 1, "import_time": 0},
        "core/X/direct.py": {"reads": 4, "import_time": 4},
        "core/X/flags.py": {"reads": 1, "import_time": 1},  # FEATURE_X; TESTING excluded; is_test_mode() has no key
        "core/X/helpers.py": {"reads": 8, "import_time": 2},  # 3 helper bodies + H1, H2, (H3, H4), H6 via _outer, H7 by keyword
        "core/X/import_time.py": {"reads": 3, "import_time": 1},  # I1 via LIMIT = _limit(); not I2, not the __main__ block
        "core/X/imported.py": {"reads": 1, "import_time": 0},
        "core/X/resolver_user.py": {"reads": 2, "import_time": 0},  # R1 and R3 (resolver-backed _via) count outside config
        "core/config_sections/y.py": {"reads": 2, "import_time": 2},  # a direct-backed helper counts even in a config module
    }


def test_reads_carry_their_literal_names(tmp_path: Path) -> None:
    reads = measure_reads(_seed(tmp_path / "app", SEED))
    assert sorted(r.names for r in reads["core/X/helpers.py"]) == [(), (), (), ("H1",), ("H2",), ("H3", "H4"), ("H6",), ("H7",)]
    assert {(r.kind, r.names) for r in reads["core/config_sections/x.py"]} == {("resolver", ("R2",)), ("resolver", ("K1",))}
    assert [r.names for r in reads["core/X/test_flags.py"]] == [("TEST_MODE",), ("TLDW_TEST_DB",)]  # names kept, reads not counted
    assert "core/testing.py" not in reads and "core/X/tests/test_x.py" not in reads


def test_growth_a_new_module_and_a_stale_entry_fail(tmp_path: Path) -> None:
    measured = measure(_seed(tmp_path / "app", {"core/X/a.py": 'import os\nA = os.getenv("A")\nB = os.getenv("B")\n',
                                                "core/X/new.py": 'import os\nA = os.getenv("A")\n'}))
    baseline = {"core/X/a.py": {"reads": 1, "import_time": 2, "reason": None},
                "core/X/gone.py": {"reads": 1, "import_time": 0, "reason": None}}
    problems = _ceilings.violations(RATCHET, measured, baseline)
    assert [p.split(":")[0] for p in problems] == ["core/X/a.py", "core/X/gone.py", "core/X/new.py"]
    assert "reads 2 > ceiling 1" in problems[0]
    assert "stale entry" in problems[1]


def test_a_moved_key_read_outside_config_sections_fails(tmp_path: Path) -> None:
    reads = measure_reads(_seed(tmp_path / "app", {
        "core/X/a.py": 'import os\nA = os.getenv("AUTH_MODE")\n',
        "core/config_sections/auth.py": 'import os\nA = os.environ.get("AUTH_MODE")\n',
    }))
    assert moved_key_problems(reads, frozenset({"AUTH_MODE"})) == [
        "core/X/a.py: reads AUTH_MODE, which moved to core/config_sections (TASK-13950.21); read it through the typed section"
    ]
```

  Also add, against the real tree:
  - `test_env_reads_do_not_grow` and `test_raised_env_read_ceilings_carry_a_new_reason`, copies of `test_direct_llm_adapter_calls_ratchet.py:204-215` that use `_app_measurement()`;
  - `test_moved_shared_keys_are_read_only_in_config_sections`: `moved_key_problems(app_reads(), MOVED_SHARED_KEYS) == []` and `MOVED_SHARED_KEYS <= SHARED_KEYS`;
  - `test_mcp_config_resolves_no_shared_key`: `settings_env_names(parse_tree(APP_ROOT)[APP_ROOT / "core/MCP_unified/config.py"]) & SHARED_KEYS == frozenset()`, plus `len(...) >= 70` so the pin is not vacuous (dev has 76).

- [ ] **Step 2: Run them; they fail.**
  - Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t1" tldw_Server_API/tests/lint/test_env_read_ratchet.py`.
  - Expected: each test fails with `NameError` on the first undefined name.
- [ ] **Step 3: Implement the counter** to the spec above.
  - Build the `_Scope` per module once: its import bindings and `os.environ` aliases.
  - Precompute each function's calls and their targets once, then iterate `env_helpers` to its fixed point.
  - The module docstring lists what is counted and excluded, with reasons, from Measured.
- [ ] **Step 4: Run the self-tests.** Run the same command with `-k "shape or names or growth or moved_key"`. Expected: `4 passed`.
- [ ] **Step 5: Seed on the dev tip** (after #3433). Run `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_env_read_ratchet --write-baseline`, then `... --summary`.
  - Expected with #3433 merged: `625 entries`; `outside config modules: 3302 reads in 608 modules, 265 at import time`; `config modules: 451 reads in 17 modules, 6 at import time`.
  - Without #3433 it is 626 entries and 3,300 / 609 / 266. Record the printed numbers in the PR description.
- [ ] **Step 6: Run the whole module.** Expected: `8 passed`, in about 45 s (the parse, one counter pass, and the merge-base check).
- [ ] **Step 7: Wire the ratchet in**, with the workflow and contract edits from Files, then run C1. Expected: all pass, including `test_backend_required_enforces_config_ratchets[...test_env_read_ratchet.py]`.
- [ ] **Step 8: Commit.**

```bash
git add tldw_Server_API/tests/lint/test_env_read_ratchet.py tldw_Server_API/tests/lint/env_read_baseline.json .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): freeze env reads per module, outside and inside the config modules (TASK-13950.21)"
```

### Task 2: The undocumented-env-name ratchet (AC2, AC5)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_env_vars_docs_ratchet.py`
- Create (generated): `tldw_Server_API/tests/lint/env_vars_undocumented_baseline.txt`, one name per line, sorted, ending with a newline.
- Modify: `.github/workflows/backend-required.yml`: add `tldw_Server_API/tests/lint/test_env_vars_docs_ratchet.py \` after the Task 1 line.
- Modify: `test_license_first_workflow_contracts.py`: append this module to the parametrize list of `test_backend_required_enforces_config_ratchets`.

**Interfaces:**
- Consumes: Task 1's `app_reads`, `measure_reads`, `settings_env_names`; `_ast_cache.REPO_ROOT`, `APP_ROOT`, `parse_tree`.
- Produces (module `tldw_Server_API.tests.lint.test_env_vars_docs_ratchet`):
  - `DOC: Path = REPO_ROOT / "Docs" / "Operations" / "Env_Vars.md"`.
  - `BASELINE: Path = Path(__file__).resolve().with_name("env_vars_undocumented_baseline.txt")`.
  - `ALLOWLIST: dict[str, str]`: `APPDATA`, `LOCALAPPDATA`, `XDG_CONFIG_HOME`, `PATH`, `CUDA_VISIBLE_DEVICES`, `TORCH_HOME`, `FORCE_COLOR`, `GITHUB_ACTIONS` and `PYTEST_XDIST_WORKER`. Each reason names who sets the variable and ends with `(TASK-13950.21)`.
  - `env_names(root: Path = APP_ROOT) -> frozenset[str]`: every name of every `Read`, plus `settings_env_names` over every module. On `APP_ROOT` it uses `app_reads()`.
  - `class DocIndex(NamedTuple)`: `tokens: frozenset[str]`, `patterns: tuple[re.Pattern[str], ...]`, and `covers(name: str) -> bool`.
  - `doc_index(text: str) -> DocIndex`.
    - `tokens` are the full identifiers `re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)`.
    - `patterns` come from each match of `r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]*(?:(?:\{[^}]+\}|<[^>]+>|\*)[A-Z0-9_]*)+"`. Split the match on its `{...}`, `<...>` and `*` placeholders, escape the literal parts, and join them with `[A-Z0-9_]+`. `covers` uses `fullmatch`.
    - The lookbehind stops a placeholder inside a lower-case word, such as `custom_openai_api_N_*` at `Env_Vars.md:590`, from documenting every `N_` name.
  - `undocumented(names: Iterable[str], index: DocIndex, allowlist: Mapping[str, str]) -> list[str]`: sorted.
  - `problems(current: Sequence[str], baseline: Sequence[str]) -> list[str]`, sorted by name:
    - a name in `current` but not in `baseline` gives `"<NAME>: read in app/ but missing from Docs/Operations/Env_Vars.md. Document it there and run Helper_Scripts/refresh_docs_published.sh, or add it to ALLOWLIST in <this file> with a reason (TASK-13950.21)"`;
    - a name in `baseline` but not in `current` gives `"<NAME>: stale baseline entry (now documented, allowlisted or no longer read). Run: python -m tldw_Server_API.tests.lint.test_env_vars_docs_ratchet --tighten"`.
  - `allowlist_problems(allowlist) -> list[str]`: a reason without `TASK-<n>` or `#<n>`, or with an unfilled `<...>`, fails.
  - `__main__`:
    - `--write-baseline` writes `undocumented(env_names(), doc_index(DOC.read_text()), ALLOWLIST)`;
    - `--tighten` keeps only the baseline names still in that list.

- [ ] **Step 1: Write the self-tests.**

```python
def test_doc_index_matches_whole_names_and_placeholder_patterns() -> None:
    index = doc_index("- `AUTH_MODE`: mode.\n- `CUSTOM_OPENAI_API_{N}_IP` and `RG_TEST_*`; also AUTH_MODE_X.\n"
                      "- fields use `custom_openai_api_N_*`.\n")
    assert [n for n in ("AUTH_MODE", "AUTH_MODE_X", "CUSTOM_OPENAI_API_7_IP", "RG_TEST_FOO") if not index.covers(n)] == []
    assert [n for n in ("AUTH", "CUSTOM_OPENAI_API_IP", "RG_TEST", "N_FOO") if index.covers(n)] == []


def test_a_new_undocumented_name_and_a_stale_entry_fail(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {"core/X/a.py": 'import os\nA = os.getenv("NEW_ONE")\nB = os.getenv("OLD_ONE")\n'})
    found = problems(undocumented(env_names(root), doc_index("`OLD_ONE`"), {}), ["GONE", "OLD_ONE"])
    assert [p.split(":")[0] for p in found] == ["GONE", "NEW_ONE", "OLD_ONE"]
    assert "stale" in found[0] and "missing from Docs/Operations/Env_Vars.md" in found[1] and "stale" in found[2]


def test_settings_classes_contribute_their_env_names(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {"core/X/s.py": (
        "from pydantic import Field\nfrom pydantic_settings import BaseSettings\n\n\nclass S(BaseSettings):\n"
        '    FOO: int = 1\n    bar: str = Field(default="", validation_alias="MCP_BAR")\n    baz: int = 0\n')})
    assert env_names(root) == {"FOO", "MCP_BAR"}


def test_only_the_curated_doc_documents() -> None:
    # The root Env_Vars.md is generated from code (Helper_Scripts/generate_env_vars_md.py): counting it
    # would let a regeneration "document" every name without writing a line (TASK-13950.21 scope ruling 1).
    assert DOC == REPO_ROOT / "Docs" / "Operations" / "Env_Vars.md"


def test_allowlist_reasons_cite_a_task() -> None:
    assert allowlist_problems(ALLOWLIST) == []
    assert allowlist_problems({"X": "because"}) != []
```

  Also add `test_every_env_name_is_documented_allowlisted_or_baselined`: `problems(undocumented(env_names(), doc_index(DOC.read_text(encoding="utf-8")), ALLOWLIST), BASELINE.read_text(encoding="utf-8").split()) == []`, with the problems joined into the failure message.
- [ ] **Step 2: Run them; they fail.** Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t2" tldw_Server_API/tests/lint/test_env_vars_docs_ratchet.py`. Expected: `NameError`, or `FileNotFoundError` for the baseline.
- [ ] **Step 3: Implement** the signatures above.
- [ ] **Step 4: Seed.** Run `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_env_vars_docs_ratchet --write-baseline`, then `wc -l tldw_Server_API/tests/lint/env_vars_undocumented_baseline.txt`.
  - Expected: `1886` on dev, with or without #3433 (2,380 names, 485 documented, 9 allowlisted).
  - Record the number in the PR description.
- [ ] **Step 5: Run the module.** Expected: `6 passed`, in about 30 s, or less after Task 1's module in the same process.
- [ ] **Step 6: Wire it in, then run C1 and C2.** Expected: all pass.
- [ ] **Step 7: Commit.**

```bash
git add tldw_Server_API/tests/lint/test_env_vars_docs_ratchet.py tldw_Server_API/tests/lint/env_vars_undocumented_baseline.txt .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): freeze env names missing from the curated Env_Vars doc (TASK-13950.21)"
```

### Task 3: One resolver; the config_sections copies and config.py's helper family use it (AC3)

**Files:**
- Create: `tldw_Server_API/app/core/config_sections/precedence.py`
- Create: `tldw_Server_API/tests/Config/test_config_resolver_characterization.py`
- Create (generated before the edits): `tldw_Server_API/tests/Config/config_resolver_golden.json`
- Modify: the `_get_raw` bodies and inline env reads in `tldw_Server_API/app/core/config_sections/{auth,audio,chat,chunking,database,embeddings,jobs,logging,moderation,providers,rag,server,slides,stt}.py`.
- Modify: `tldw_Server_API/app/core/config.py:100-167`. `_first_nonempty_env`, `_env_or_config_value` and `_env_or_config_option_value` keep their signatures and call `resolve`.
- Modify: `tldw_Server_API/tests/Config/test_config_precedence_contract.py`, for the resolver's unit tests.
- Modify: `env_read_baseline.json`.

**Interfaces:**
- Produces (`tldw_Server_API.app.core.config_sections.precedence`; stdlib only):
  - `Source = Literal["env", "config", "default"]`;
  - `class Resolved(NamedTuple)`: `value: str | None`, `source: Source`, `origin: str | None` (the env key, or `"<Section>.<option>"`);
  - the resolver:

    ```python
    def resolve(
        env_keys: Sequence[str] = (),
        locations: Sequence[tuple[str, str]] = (),
        *,
        config_parser: ConfigParserLike | None = None,
        env: Mapping[str, str] | None = None,
        default: str | None = None,
        blank_env: bool = False,
        blank_config: bool = False,
    ) -> Resolved
    ```

    - Order:
      - the first env key whose value is not None, and, unless `blank_env`, not blank after `strip()`;
      - then the first location whose `config_parser.get(section, option, fallback=None)` is not None, and, unless `blank_config`, not blank;
      - then `default`.
    - `env=None` reads `os.environ` at call time. Values are returned unstripped.
    - **Parser errors propagate.** A missing section or option is absent through `fallback`, but `InterpolationSyntaxError` and every other `configparser.Error` raise, as today's copies do.
    - `resolve` never validates.

- [ ] **Step 1: Write the characterisation harness and generate its golden on unmodified dev** (`test_config_resolver_characterization.py`).
  - `class _AnyEnv(dict)`:
    - it overrides `__getitem__`, `get` and `__contains__`, so every key reads one value, or is absent when the value is None;
    - it records each key asked for, in first-access order, in `.asked`;
    - writes and `copy()` go to the plain dict and never change reads.
  - `class _AnyParser`:
    - `get(section, option, fallback=None)`, `has_section` and `has_option` return the value, or `fallback`, or `value is not None`;
    - with `raises` set, `get` raises it;
    - it records `[section, option]` pairs in first-access order in `.asked`, and ignores `has_section`.
  - `SCENARIOS: dict[str, tuple[str | None, str | None | BaseException]]` gives `(env value, parser value)`:
    - `"none"`: `(None, None)`;
    - `"env_blank"`: `("", None)`;
    - `"env_space"`: `("  7  ", None)`;
    - `"env_word"`: `("abc", None)`;
    - `"env_true"`: `("true", None)`;
    - `"cfg_blank"`: `(None, "")`;
    - `"cfg_space"`: `(None, "  7  ")`;
    - `"cfg_false"`: `(None, "false")`;
    - `"both"`: `("8", "9")`;
    - `"env_blank_cfg"`: `("", "9")`;
    - `"cfg_raises"`: `(None, configparser.InterpolationSyntaxError("o", "S", "bad %"))`.
  - `SUBJECTS: dict[str, Callable[[Mapping[str, str], Any], object]]`:
    - the 14 `load_<x>_config(parser, env)` loaders;
    - `_first_nonempty_env(("K1", "K2"), environment=env)`;
    - `_env_or_config_value(("K1",), parser, "S", "o", fallback="d", environment=env)`;
    - `_env_or_config_option_value(("K1",), parser, "S", ("o1", "o2"), fallback="d", environment=env)`.
  - `outcome(subject, env_value, parser_value) -> dict` runs inside `with pytest.MonkeyPatch.context() as mp:` and returns `{"value": ..., "env": env.asked, "config": parser.asked}`.
    - `value` is `dataclasses.asdict(result)` for a dataclass, the result otherwise, or `{"raises": type(exc).__name__}`.
    - Tuples become lists, `Path`s become strings, and the repo root path is replaced by `<repo>`.
  - `test_loaders_and_helpers_match_the_dev_golden()` compares every `subject × scenario` outcome with `GOLDEN`, collecting all mismatches into one message.
  - `test_the_golden_holds_no_harness_errors()` asserts that no golden value is `{"raises": "AttributeError"}` or `{"raises": "TypeError"}`. Those mean the fake is missing a method, not a behaviour.
  - `__main__ --write-golden` rewrites the golden.

  Run on unmodified dev: `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.Config.test_config_resolver_characterization --write-golden`, then the test file. Expected: `2 passed`.
- [ ] **Step 2: Commit the harness and golden alone** (`test(config): pin section-loader and helper precedence, keys and order over eleven scenarios (TASK-13950.21)`), so the refactor commit shows no golden change.
- [ ] **Step 3: Write the resolver's failing unit tests** in `test_config_precedence_contract.py`.

```python
from configparser import ConfigParser, InterpolationSyntaxError

from tldw_Server_API.app.core.config_sections.precedence import Resolved, resolve


def _parser(**options: str) -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("S")
    for key, value in options.items():
        parser.set("S", key, value)
    return parser


def test_resolve_takes_the_first_set_env_key_then_config_then_default() -> None:
    parser = _parser(o1="", o2="cfg")
    assert resolve(("A", "B"), (("S", "o1"), ("S", "o2")), config_parser=parser, env={"B": " b "}, default="d") == Resolved(" b ", "env", "B")
    assert resolve(("A",), (("S", "o1"), ("S", "o2")), config_parser=parser, env={"A": "  "}, default="d") == Resolved("cfg", "config", "S.o2")
    assert resolve(("A",), (("S", "missing"), ("Nope", "o")), config_parser=parser, env={}, default="d") == Resolved("d", "default", None)


def test_resolve_can_let_a_blank_env_or_config_value_win() -> None:
    assert resolve(("A",), env={"A": ""}, blank_env=True) == Resolved("", "env", "A")
    assert resolve((), (("S", "o1"),), config_parser=_parser(o1=""), env={}, blank_config=True) == Resolved("", "config", "S.o1")


def test_resolve_lets_a_parser_error_propagate() -> None:
    with pytest.raises(InterpolationSyntaxError):
        resolve((), (("S", "o"),), config_parser=_parser(o="p%40ss"), env={})


def test_resolve_reads_os_environ_at_call_time(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_RESOLVE_PROBE", "1")
    assert resolve(("TLDW_RESOLVE_PROBE",)).source == "env"
    monkeypatch.delenv("TLDW_RESOLVE_PROBE")
    assert resolve(("TLDW_RESOLVE_PROBE",)).source == "default"
```

- [ ] **Step 4: Run them; they fail** with `ModuleNotFoundError: ...config_sections.precedence`. Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t3" tldw_Server_API/tests/Config/test_config_precedence_contract.py -k resolve`.
- [ ] **Step 5: Implement `resolve`, then rewrite each copy as a `resolve(...)` call** with the flags its old code implies:
  - `env_value is not None and str(env_value).strip() != ""` means `blank_env=False`;
  - `if env_key in env_map` (slides) means `blank_env=True`;
  - `config_parser.get(section, option, fallback=default)` followed by `.strip() or default` means `blank_config=False` with that `default`.

  Keep each caller's own stripping. `_first_nonempty_env` strips its return value, so its wrapper keeps the `.strip()`. Slides keeps its own `except NoSectionError`. Keep the `_get_raw` names that loaders call, with one-line bodies.
- [ ] **Step 6: Run the golden and the loader tests.**
  - Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t3" tldw_Server_API/tests/Config tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py`.
  - Expected: all pass, and `git diff --quiet -- tldw_Server_API/tests/Config/config_resolver_golden.json` succeeds.
  - A mismatch means fixing the call's flags or order, never the golden.
- [ ] **Step 7: Tighten and check.**
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_env_read_ratchet --write-baseline`, then `git diff tldw_Server_API/tests/lint/env_read_baseline.json`. It must only:
    - remove the 14 `config_sections/*` entries;
    - add `core/config_sections/precedence.py`;
    - lower `core/config.py`, by the helper family's reads and their literal-key callers, which are now resolver-backed.
  - Give the new entry `"reason": "the one resolver (TASK-13950.21)"`.
  - The docs baseline must not change: `--tighten` gives no diff.
  - Then run C1, C2 and C3.
- [ ] **Step 8: Commit** (`refactor(config): one env-over-config resolver; section loaders and config helpers delegate (TASK-13950.21)`).

### Task 4: The standalone getters and AuthNZ use the resolver; the route-toggle exception is pinned (AC3, AC5; D2)

**Files:**
- Modify: `test_config_resolver_characterization.py` (new subjects) and `config_resolver_golden.json`, regenerated on unmodified dev in Step 1.
- Modify: `tldw_Server_API/app/core/config.py`. The 27 getters:
  - `rag_enable_structure_index`, `rag_strict_extractive`, `rag_require_hard_citations`, `rag_low_confidence_behavior`, `rag_agentic_cache_backend`, `rag_agentic_cache_ttl_sec`, `implicit_feedback_enabled`;
  - `web_outbound_policy_mode`, `web_browser_transport_mode`;
  - `rg_enabled`, `rg_policy_store`, `rg_policy_reload_enabled`, `rg_policy_reload_interval_sec`, `rg_backend`, `rg_redis_fail_mode`, `rg_policy_path`, `resolve_governance_rollout_mode`, `usage_quotas_enabled`;
  - `get_llamacpp_handler_settings`, `get_ingestion_source_allowed_roots`, `get_workspace_project_root_allowed_roots`, `_load_claims_review_metrics_settings`;
  - `_resolve_run_first_rollout_mode`, `_resolve_run_first_provider_allowlist`, `_resolve_run_first_presentation_variant`;
  - `_resolve_disable_cors_value_and_source`, `_resolve_cors_allow_credentials_value_and_source`;
  - and the four `ROUTES_*` reads inside `_route_toggle_policy`'s existing `if allow_env_overrides:` block.
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py:1371-1376`. `maybe_set` becomes `resolved = resolve((field,), (("AuthNZ", key),), config_parser=cfg, blank_env=True, blank_config=True)`; if `resolved.source == "config"`, set `overrides[field]` to the cast value. This never raises for a bad value.
- Modify: `tldw_Server_API/tests/Config/test_config_precedence_contract.py` (route pins).
- Modify (docs; D2's recommended outcome):
  - `Docs/Operations/Env_Vars.md` `## Testing & CI Controls` (`:101`): add ``- `ROUTES_STABLE_ONLY`, `ROUTES_DISABLE`, `ROUTES_ENABLE`, `ROUTES_EXPERIMENTAL`: route-toggle overrides, honoured only in a test runtime (pytest or `TEST_MODE`). In production, `[API-Routes]` in `config.txt` is authoritative.``
  - `tldw_Server_API/Config_Files/config.txt:1445-1448` and `tldw_Server_API/Config_Files/README.md:702-706`: say the overrides are test-runtime only.
  - `Docs/API-related/Tools_API_Documentation.md:20-25`: replace the "Or enable via environment" block with the same statement.
  - The `Docs/Published` mirrors (C9).
- Modify: `backend-required.yml` and the contract file. Add `tldw_Server_API/tests/Config/test_config_precedence_contract.py \` to the step after Task 2's line, and to `test_backend_required_enforces_config_ratchets`. It now holds the documented exception, so AC5 applies.
- Modify: both baselines (tighten).

**Interfaces:**
- Consumes: Task 3's `resolve`, `Resolved`, and the harness.
- Produces: `DOCUMENTED_PRECEDENCE_EXCEPTIONS: dict[str, str]` in `test_config_precedence_contract.py`, set to `{"core.config._route_toggle_policy": "ROUTES_* env overrides apply only in a test runtime (pytest or TEST_MODE); config.txt [API-Routes] is authoritative in production (ef1088b3c1, TASK-13950.21 D2)"}`.

- [ ] **Step 1: Add the subjects and regenerate the golden on unmodified dev.**
  - In each new subject, `mp.setattr(os, "environ", _AnyEnv(...))` swaps the environment inside `outcome`'s `MonkeyPatch.context()`.
  - `config.load_comprehensive_config`, and `settings.load_comprehensive_config` for AuthNZ, are patched with a loader that returns the `_AnyParser` and has a no-op `cache_clear` (`clear_config_cache()` calls it).
  - Each getter is called with no arguments, except:
    - `_load_claims_review_metrics_settings`, which takes the parser;
    - the private run-first resolvers, reached through their `resolve_chat_*` wrappers (`:3162-3214`);
    - AuthNZ, whose subject calls `_load_overrides_from_config()`.
  - `--write-golden`; `git diff` must show only added subjects. Commit (`test(config): pin the standalone getters' and AuthNZ overlay precedence (TASK-13950.21)`).
- [ ] **Step 2: Write the route pins.**

```python
REPO_ROOT = Path(__file__).resolve().parents[3]


def _routes_loader(stable_only: str):
    parser = configparser.ConfigParser()
    parser.add_section("API-Routes")
    parser.set("API-Routes", "stable_only", stable_only)

    def _load() -> configparser.ConfigParser:
        return parser

    _load.cache_clear = lambda: None  # the autouse fixture's clear_config_cache() calls it before monkeypatch undoes
    return _load


def test_route_toggle_env_overrides_apply_in_a_test_runtime(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_comprehensive_config", _routes_loader("true"))
    monkeypatch.setenv("ROUTES_DISABLE", "persona")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "false")
    policy = config._route_toggle_policy()
    assert "persona" in policy["disable"] and policy["stable_only"] is False


def test_route_toggle_env_overrides_are_ignored_outside_a_test_runtime(monkeypatch) -> None:
    monkeypatch.setattr(config, "load_comprehensive_config", _routes_loader("true"))
    monkeypatch.setattr(config, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(config, "is_test_mode", lambda: False)
    monkeypatch.setenv("ROUTES_DISABLE", "persona")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "false")
    policy = config._route_toggle_policy()
    assert "persona" not in policy["disable"] and policy["stable_only"] is True


def test_documented_precedence_exceptions_are_in_the_env_vars_doc() -> None:
    doc = (REPO_ROOT / "Docs" / "Operations" / "Env_Vars.md").read_text(encoding="utf-8")
    assert all(name in doc for name in ("ROUTES_STABLE_ONLY", "ROUTES_DISABLE", "ROUTES_ENABLE", "ROUTES_EXPERIMENTAL"))
    assert set(DOCUMENTED_PRECEDENCE_EXCEPTIONS) == {"core.config._route_toggle_policy"}
```

- [ ] **Step 3: Run them.** The first two pass on dev, which pins today. The doc test fails, because `ROUTES_STABLE_ONLY` is not in the doc.
- [ ] **Step 4: Convert the getters, the route reads and `maybe_set` to `resolve`**, with the flags each old body implies. `os.getenv(X) is None` means `blank_env=True`; a truthy test means `False`. Keep each getter's existing `try`/`except` around the converted call. Then make the doc edits.
- [ ] **Step 5: Run.**
  - Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t4" tldw_Server_API/tests/Config tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/Web_Scraping`.
  - Expected: all pass, with the golden unchanged since Step 1.
- [ ] **Step 6: Tighten both ratchets** (`ROUTES_*` leaves the docs baseline). Then run C1, C2, C3, C4, C9, and C5 over `tldw_Server_API/tests/helpers/test_pg_env_precedence.py tldw_Server_API/tests/AuthNZ/unit`.
- [ ] **Step 7: Commit** (`refactor(config): getters and the AuthNZ overlay resolve through one resolver; pin route-toggle env as test-only (TASK-13950.21)`).

### Task 5: `MAX_REQUEST_SIZE` is read at call time (`.14` handoff)

**Files:**
- Modify: `tldw_Server_API/app/core/config_sections/chat.py`. Add `DEFAULT_MAX_REQUEST_SIZE_CHARS` and `max_request_size_chars`.
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_validators.py`.
  - Delete `_get_max_request_size` and `MAX_REQUEST_SIZE` (`:30-67`).
  - `validate_request_size` (`:321`) first loads the parser with today's fail-soft rule:

    ```python
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        parser = load_comprehensive_config()
    except (AttributeError, FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError, configparser.Error) as cfg_err:
        logger.debug(f"Failed to load max_request_size from config: {cfg_err}")
        parser = None
    limit = max_request_size_chars(parser)
    ```

    This sits before the existing `try`, so a config failure falls back to the default instead of rejecting the request. `limit` is then used in the check and the message.
- Modify tests:
  - `Chat/integration/test_chat_endpoint_image_validation.py`: delete the `setattr` at `:223`. `:222` already sets `CHAT_REQUEST_MAX_SIZE`.
  - `Chat/unit/test_chat_token_estimators_agree.py:92` and `Chat/unit/test_data_uri_redaction_is_linear.py:66,92`: `monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", n)` becomes `monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", str(n))`.
  - `schemas/test_chat_validators.py`:
    - `:21` imports `DEFAULT_MAX_REQUEST_SIZE_CHARS` and `max_request_size_chars` from `config_sections.chat`;
    - `test_request_too_large` (`:245`) gains a `monkeypatch` parameter, calls `monkeypatch.delenv("CHAT_REQUEST_MAX_SIZE", raising=False)`, and uses `DEFAULT_MAX_REQUEST_SIZE_CHARS + 1`;
    - `:426` becomes `assert max_request_size_chars(None, env={}) == 1_000_000`.
- Modify: `tldw_Server_API/tests/lint/test_schemas_config_boundary.py:52-57`: `IMPORT_TIME_CONFIG_ALLOWLIST: dict[str, str] = {}`.

**Interfaces:**
- Produces (`config_sections.chat`): `DEFAULT_MAX_REQUEST_SIZE_CHARS: int = 1_000_000` and `max_request_size_chars(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> int`, which keeps `_get_max_request_size`'s rule:
  - `CHAT_REQUEST_MAX_SIZE` when set and an integer;
  - else `[Chat-Module] max_request_size_chars` when present and an integer;
  - else the default;
  - each clamped with `max(1, value)`.
  - A non-integer env value falls through to `config.txt`. A non-integer config value, or a parser error, gives the default.
  - Implement it as two `resolve` calls, each followed by `int(...)`: env-only with `blank_env=True`, then config-only with `blank_config=True` inside `except (TypeError, ValueError, configparser.Error)`.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize(("env", "cfg", "expected"), [
    ({"CHAT_REQUEST_MAX_SIZE": "200"}, None, 200), ({"CHAT_REQUEST_MAX_SIZE": "0"}, None, 1),
    ({"CHAT_REQUEST_MAX_SIZE": "-5"}, None, 1), ({"CHAT_REQUEST_MAX_SIZE": "abc"}, "300", 300),
    ({"CHAT_REQUEST_MAX_SIZE": ""}, "300", 300), ({}, "300", 300), ({}, "x", 1_000_000), ({}, "0", 1), ({}, None, 1_000_000),
])
def test_max_request_size_chars_keeps_its_precedence(env, cfg, expected) -> None:
    from tldw_Server_API.app.core.config_sections.chat import max_request_size_chars

    parser = ConfigParser()
    if cfg is not None:
        parser.add_section("Chat-Module")
        parser.set("Chat-Module", "max_request_size_chars", cfg)
    assert max_request_size_chars(parser, env=env) == expected
```

  In `tests/schemas/test_chat_validators.py`:

```python
def test_request_size_limit_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", "50")
    with pytest.raises(ValueError, match="max 50 chars"):
        validate_request_size("x" * 60)
    monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", "100")
    assert validate_request_size("x" * 60) is True


def test_request_size_limit_falls_back_when_config_cannot_load(monkeypatch):
    from tldw_Server_API.app.core import config

    def _broken():
        raise OSError("config.txt unreadable")

    _broken.cache_clear = lambda: None
    monkeypatch.setattr(config, "load_comprehensive_config", _broken)
    monkeypatch.delenv("CHAT_REQUEST_MAX_SIZE", raising=False)
    assert validate_request_size("x" * 60) is True  # the 1M default applies, as today
    monkeypatch.setenv("CHAT_REQUEST_MAX_SIZE", "50")
    with pytest.raises(ValueError, match="max 50 chars"):
        validate_request_size("x" * 60)  # env still applies when config cannot load
```

- [ ] **Step 2: Run them; they fail.**
  - Run `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t5" "tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py::test_max_request_size_chars_keeps_its_precedence" tldw_Server_API/tests/schemas/test_chat_validators.py -k "call_time or cannot_load or max_request_size_chars"`.
  - Expected:
    - the parametrized test fails with `ImportError`;
    - `call_time` fails with "DID NOT RAISE", because the import-time constant ignores the later env;
    - `cannot_load` fails on dev only at its env assertion, for the same reason. Its default assertion already holds today, which pins the fail-soft behaviour.
- [ ] **Step 3: Implement**, and retarget the patches and imports.
- [ ] **Step 4: Run.**
  - Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t5" tldw_Server_API/tests/schemas/test_chat_validators.py tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_image_validation.py tldw_Server_API/tests/lint/test_schemas_config_boundary.py tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py`.
  - Expected: all pass, and `git grep -nw MAX_REQUEST_SIZE -- tldw_Server_API` prints nothing.
- [ ] **Step 5: Tighten the env ratchet.** The `api/v1/schemas/chat_validators.py` entry drops out. Then run C1, C2 and C3.
- [ ] **Step 6: Commit** (`refactor(chat): read the request size limit at call time through config_sections (TASK-13950.21)`).

### Task 6: RAG schema defaults apply configured contextual values at call time (`.15` handoff; D3)

**Scope.** Configured values apply on every path that builds `UnifiedRAGRequest` or `UnifiedBatchRequest`, or goes through `resolve_rag_request`. These are:
- HTTP `/api/v1/rag/search`, `/rag/search/stream` and `/rag/batch` (`rag_unified.py:1417,2282,1679`);
- the MCP rag tool (`MCP_unified/modules/implementations/rag_module.py:426`);
- recurring questions (`Scheduled_Tasks/recurring_question_rag_adapter.py:79`);
- the retrieval-tuning eval recipe (`Evaluations/recipes/rag_retrieval_tuning_execution.py:107`);
- batch resume (`rag_unified.py:473-484`). It rebuilds the request from the checkpoint, so values written at the first run come back as explicit fields. That is harmless, because they are the same values.

`unified_rag_pipeline`'s own parameter defaults (`unified_pipeline.py:2056-2059`) do **not** change: it is a frozen hotspot, and D3 was framed for the request models. Direct callers of the pipeline therefore keep today's literals, and Task 8 files the follow-up.

**Precedence on a covered path:** an explicit request value, then a `rag_profile`'s defaults (`resolve_rag_request`, `request_resolution.py:202`), then the configured value, then the literal.

**Files:**
- Modify: `tldw_Server_API/app/core/config_sections/rag.py`. Add the following.
  - `default_fts_level`.
  - `class ContextualDefaults` (frozen dataclass), with fields `include_parent_document: bool = False`, `parent_max_tokens: int = 1200`, `include_sibling_chunks: bool = False` and `sibling_window: int = 1`. These are today's literals.
  - `contextual_defaults`.
- Modify: `tldw_Server_API/app/core/config.py:1959-2000`. LazySettings' `"RAG_CONTEXTUAL_DEFAULTS"` lambda becomes `dataclasses.asdict(contextual_defaults(load_comprehensive_config()))`. That leaves one rule, owned by `config_sections`, and shrinks the frozen `load_settings`.
- Modify: `tldw_Server_API/app/core/RAG/rag_schemas_unified.py`.
  - **`:21-40`.** The five `_DEF_*` contextual names become static literals (`1`, `False`, `1200`, `False`, `False`), so OpenAPI defaults stay static.
  - **`_DEF_FTS_LVL = default_fts_level(load_comprehensive_config())`**, followed by the existing `in ("media", "chunk")` check, under today's broad `except Exception:  # noqa: BLE001 - import-time default; any failure falls back to "media", as before (TASK-13950.21)`.
  - **The configured values.** Add `_apply_contextual_defaults(model: BaseModel, fields: tuple[str, ...]) -> None`. For each named field not in `model.model_fields_set`, it writes the configured value into `model.__dict__`. Writing there leaves `model_fields_set` reporting the field as unset, which `rag_service/request_resolution.py:_request_fields_set` relies on.
  - **Fail-soft.** The helper loads config with Task 5's rule:
    - `try: parser = load_comprehensive_config()`, under `except (AttributeError, FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError, configparser.Error)`;
    - then `contextual_defaults(parser)`, under the same tuple.
    - On an error it logs `"RAG contextual defaults unavailable; using built-in defaults: {}"` and leaves the literals.
    - A bad value in one key fails the whole call. For example, `RAG_PARENT_MAX_TOKENS=abc` raises `ValueError` in `_int_env_or_cfg`, so all four fields keep their literals.
    - Each distinct warning is logged once per process, through a module-level `_CONTEXTUAL_WARNED: set[str]`. The validator runs on every request, so a broken `config.txt` would otherwise log on each one.
  - **Bounds come from the field.** Each configured value is checked with a cached `TypeAdapter(Annotated[field.annotation, *field.metadata]).validate_python(value)`, where `field` is `type(model).model_fields[name]`. Checking against the field itself means the bounds cannot drift from it (today `sibling_window` 0-50, `parent_max_tokens` ≥ 1).
    - A `ValidationError` skips that field with the once-only warning `"RAG contextual default {name}={value} is out of range; using the built-in default"`, and its literal stays.
  - **How the module reaches config.** It imports `from tldw_Server_API.app.core import config as _config` and calls `_config.load_comprehensive_config()`, both at import (for FTS) and in the helper. A test's `monkeypatch.setattr(config, "load_comprehensive_config", ...)` therefore reaches it. Importing `core.config` does not materialise LazySettings; the subprocess test pins that.
  - **Where it is called.** Add a `model_validator(mode="after")` that returns `self`, guarded by `if model_validator is not None:` as the file's other validators are. On the pinned pydantic v2 that import never fails, so the guard never disables the feature. Apply the values as follows:
    - `UnifiedRAGRequest` applies `("include_sibling_chunks", "sibling_window", "include_parent_document", "parent_max_tokens")`;
    - `UnifiedBatchRequest` applies `("sibling_window", "include_parent_document", "parent_max_tokens")`.

    These are the fields that name a `_DEF_*` default today. Note: the batch model's `include_sibling_chunks` was never config-driven; it stays literal `False` and is out of scope.
- Modify: `Docs/Operations/Env_Vars.md` `## RAG Module` (`:118`). Document `RAG_INCLUDE_PARENT_DOCUMENT`, `RAG_PARENT_MAX_TOKENS`, `RAG_INCLUDE_SIBLING_CHUNKS` and `RAG_SIBLING_WINDOW`, their `[RAG]` keys, and that they set request defaults when a request leaves the field unset. Run C9. The docs baseline goes down by 4.
- Create: `tldw_Server_API/tests/RAG/test_rag_schema_defaults.py`.

**Interfaces:**
- Consumes: Task 3's `resolve`; `config._int_env_or_cfg` (`config.py:170-197`), imported inside the function as `load_config_sections` imports `config`, so no copy is added.
- Produces (`config_sections.rag`):
  - **`default_fts_level(config_parser, env=None) -> str`.** It keeps LazySettings' rule at `config.py:2117-2121`:
    - `RAG_DEFAULT_FTS_LEVEL`'s `.lower()` when set, blank included;
    - else, when `[RAG]` exists, `[RAG] default_fts_level` (fallback `"media"`) `.lower()`;
    - else `"media"`.

    It returns the value unvalidated.
  - **`contextual_defaults(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> ContextualDefaults`.** Its precedence is the owning source's rule, LazySettings' `RAG_CONTEXTUAL_DEFAULTS` (`config.py:1959-2000`), unchanged.
    - **Booleans** (`include_parent_document`, `include_sibling_chunks`):
      - env `RAG_INCLUDE_PARENT_DOCUMENT` / `RAG_INCLUDE_SIBLING_CHUNKS` when set, blank included → `value.lower() == "true"`;
      - else `[RAG] include_parent_document` / `include_sibling_chunks` → `str(value).lower() == "true"`;
      - else `False`.
    - **Integers** (`parent_max_tokens`, `sibling_window`): `_int_env_or_cfg(env RAG_PARENT_MAX_TOKENS / RAG_SIBLING_WINDOW, [RAG] parent_max_tokens / sibling_window, 1200 / 1)`.
      - A value that is empty, `none`, `null` or `nil` counts as unset.
      - A non-integer value raises `ValueError`, as today.
    - Each env value and each config value is read separately, with an env-only `resolve((KEY,), env=env, blank_env=True)` and a config-only `resolve((), (("RAG", option),), config_parser=..., blank_config=True)`. They are then combined as above, because `_int_env_or_cfg` needs both values.

- [ ] **Step 1: Write the tests.** Import `default_fts_level`, `contextual_defaults` and `ContextualDefaults` inside the tests that use them, so the rest run on dev.

```python
@pytest.fixture
def rag_env(monkeypatch):
    for key in ("RAG_INCLUDE_PARENT_DOCUMENT", "RAG_PARENT_MAX_TOKENS", "RAG_INCLUDE_SIBLING_CHUNKS", "RAG_SIBLING_WINDOW"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rag_schemas_unified, "_CONTEXTUAL_WARNED", set(), raising=False)
    return monkeypatch


def _loader(**rag: str):
    parser = ConfigParser()
    if rag:
        parser.add_section("RAG")
        for key, value in rag.items():
            parser.set("RAG", key, value)

    def _load() -> ConfigParser:
        return parser

    _load.cache_clear = lambda: None
    return _load


@pytest.mark.parametrize(("env", "cfg", "expected"), [
    ({}, {}, (False, 1200, False, 1)),
    ({"RAG_SIBLING_WINDOW": "3", "RAG_INCLUDE_PARENT_DOCUMENT": "TRUE"}, {}, (True, 1200, False, 3)),
    ({"RAG_INCLUDE_PARENT_DOCUMENT": "1"}, {"include_parent_document": "true"}, (False, 1200, False, 1)),  # env set wins; only "true" is true
    ({"RAG_PARENT_MAX_TOKENS": ""}, {"parent_max_tokens": "800", "include_sibling_chunks": "true"}, (False, 800, True, 1)),
    ({"RAG_SIBLING_WINDOW": "3"}, {"sibling_window": "5"}, (False, 1200, False, 3)),  # an env int beats a configured int
    ({"RAG_INCLUDE_PARENT_DOCUMENT": ""}, {"include_parent_document": "true"}, (False, 1200, False, 1)),  # a blank boolean env is set, and false
])
def test_contextual_defaults_keep_the_lazy_settings_rule(env, cfg, expected) -> None:
    from tldw_Server_API.app.core.config_sections.rag import contextual_defaults

    d = contextual_defaults(_loader(**cfg)(), env=env)
    assert (d.include_parent_document, d.parent_max_tokens, d.include_sibling_chunks, d.sibling_window) == expected


def test_configured_value_applies_when_the_request_leaves_the_field_unset(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4", include_parent_document="true"))
    request = UnifiedRAGRequest(query="q")
    assert (request.sibling_window, request.include_parent_document) == (4, True)
    assert "sibling_window" not in request.model_fields_set
    assert UnifiedBatchRequest(queries=["q"]).sibling_window == 4


def test_an_explicit_request_value_wins(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
    rag_env.setenv("RAG_INCLUDE_PARENT_DOCUMENT", "true")
    request = UnifiedRAGRequest(query="q", sibling_window=0, include_parent_document=False)
    assert (request.sibling_window, request.include_parent_document) == (0, False)


@pytest.mark.parametrize("model", ["single", "batch"])
def test_an_explicit_value_equal_to_the_default_stays_set(rag_env, model) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
    request = (UnifiedRAGRequest(query="q", sibling_window=1) if model == "single"
               else UnifiedBatchRequest(queries=["q"], sibling_window=1))
    assert request.sibling_window == 1 and "sibling_window" in request.model_fields_set


def test_resolve_rag_request_carries_the_configured_value(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
    resolved = resolve_rag_request(UnifiedRAGRequest(query="q"), single_user_id_resolver=lambda: 1)
    assert resolved.payload["sibling_window"] == 4


def test_a_profile_beats_the_configured_value(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4", include_parent_document="true"))
    resolved = resolve_rag_request(
        UnifiedRAGRequest(query="q", rag_profile="fast"),
        get_profile_kwargs_fn=lambda name: {"sibling_window": 2, "include_parent_document": False},
        single_user_id_resolver=lambda: 1,
    )
    assert (resolved.payload["sibling_window"], resolved.payload["include_parent_document"]) == (2, False)


def test_unset_config_gives_todays_literals(rag_env) -> None:
    rag_env.setattr(config, "load_comprehensive_config", _loader())
    request = UnifiedRAGRequest(query="q")
    assert (request.include_sibling_chunks, request.sibling_window, request.include_parent_document, request.parent_max_tokens) == (False, 1, False, 1200)


@pytest.mark.parametrize("broken", ["load_error", "out_of_range", "bad_value"])
def test_config_errors_fall_back_to_the_literals_and_log(rag_env, broken) -> None:
    if broken == "load_error":
        def _raise():
            raise OSError("config.txt unreadable")

        _raise.cache_clear = lambda: None
        rag_env.setattr(config, "load_comprehensive_config", _raise)
    elif broken == "out_of_range":
        rag_env.setattr(config, "load_comprehensive_config", _loader())
        rag_env.setenv("RAG_SIBLING_WINDOW", "100")  # above the field's le=50
    else:
        rag_env.setattr(config, "load_comprehensive_config", _loader(sibling_window="4"))
        rag_env.setenv("RAG_PARENT_MAX_TOKENS", "abc")  # one bad key: all four keep their literals
    seen: list[str] = []
    sink = logger.add(seen.append, level="WARNING")
    try:
        request = UnifiedRAGRequest(query="q")
    finally:
        logger.remove(sink)
    assert request.sibling_window == 1
    assert any("RAG contextual default" in message for message in seen)


def test_openapi_defaults_stay_static(rag_env) -> None:
    rag_env.setenv("RAG_SIBLING_WINDOW", "4")
    properties = UnifiedRAGRequest.model_json_schema()["properties"]
    assert (properties["sibling_window"]["default"], properties["include_parent_document"]["default"]) == (1, False)
```

  Also keep these, from the earlier draft of this task:
  - the `default_fts_level` parametrized test (`{"RAG_DEFAULT_FTS_LEVEL": "CHUNK"}` → `"chunk"`; `""` → `""`; `" chunk"` → `" chunk"`; `[RAG] default_fts_level = Chunk` → `"chunk"`; none → `"media"`);
  - `test_importing_rag_schemas_does_not_load_lazy_settings`, a subprocess that prints whether `config.settings._data is None` after importing the module, and asserts `"True"`.

  The imports are `config` (`tldw_Server_API.app.core.config`), `logger` (loguru), `rag_schemas_unified` and its `UnifiedRAGRequest` and `UnifiedBatchRequest`, and `resolve_rag_request` (`tldw_Server_API.app.core.RAG.rag_service.request_resolution`). `test_openapi_defaults_stay_static` is trivially true with literals; C7 is the real gate. `REPO_ROOT = Path(__file__).resolve().parents[3]`.
- [ ] **Step 2: Run them.** Run: `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1321-t6" tldw_Server_API/tests/RAG/test_rag_schema_defaults.py`. Expected on dev:
  - the `contextual_defaults` and `default_fts_level` cases fail with `ImportError`;
  - `test_configured_value_applies_when_the_request_leaves_the_field_unset`, `test_resolve_rag_request_carries_the_configured_value` and the `bad_value` case fail: today the schema ignores config;
  - the subprocess test fails, printing `False`;
  - the out-of-range case fails (no warning is logged);
  - the load-error case fails only at its log assertion;
  - `test_an_explicit_request_value_wins`, `test_an_explicit_value_equal_to_the_default_stays_set`, `test_a_profile_beats_the_configured_value`, `test_unset_config_gives_todays_literals` and `test_openapi_defaults_stay_static` pass. They pin today's behaviour.
- [ ] **Step 3: Implement** to the Files and Interfaces above. In `load_settings`, the replacement is net negative, as the frozen-size rule requires.
- [ ] **Step 4: Run** the file, `tldw_Server_API/tests/RAG`, `tldw_Server_API/tests/RAG_NEW`, `tldw_Server_API/tests/Config`, then C1, C2, C3, C7 and C9.
  - Expected: all pass, and the OpenAPI fingerprint is unchanged.
  - Then `git grep -n "RAG_CONTEXTUAL_DEFAULTS\|sibling_window\|include_parent_document" -- tldw_Server_API/tests`, and run every hit. Configured defaults are a changed global default.
- [ ] **Step 5: Tighten both ratchets.** `config.py` falls by the contextual lambda's 4 reads, and the docs baseline by the 4 newly documented names.
- [ ] **Step 6: Commit** (`feat(rag): apply configured contextual-retrieval defaults at call time; schema defaults stop loading LazySettings at import (TASK-13950.21)`). The PR body's `## Change summary` carries this release-note line: "Deployments that set `RAG_INCLUDE_PARENT_DOCUMENT`, `RAG_PARENT_MAX_TOKENS`, `RAG_INCLUDE_SIBLING_CHUNKS`, `RAG_SIBLING_WINDOW` or their `[RAG]` keys will see different RAG request defaults when this lands. This covers HTTP `/api/v1/rag/search`, `/rag/search/stream` and `/rag/batch`, the MCP rag tool, recurring questions and the retrieval-tuning eval recipe. A request's explicit values and its `rag_profile` still win. Code that calls `unified_rag_pipeline` directly keeps today's built-in defaults: the workflows RAG search adapter, the shared-workspace chat service, slides, data tables and eval_runner. Until now, these settings never reached requests (owner decision D3, 2026-11-08)."

### Task 7: Move one shared key (template, once per row; AC4; D1 approved 2026-11-08)

Run this task once per row of the table, one PR each, in the table's order. Row 1 creates both test files.

**Files (per row):**
- Modify: the row's `config_sections` home.
- Modify: `AuthNZ/settings.py` (the row's `maybe_set` line and any explicit read of the key).
- Modify: `config.py:load_settings` (the row's entry).
- Modify: each reader in the row.
- Modify: `test_env_read_ratchet.py` (`MOVED_SHARED_KEYS`), and both baselines.
- Test:
  - `tldw_Server_API/tests/Config/test_shared_config_keys.py` (characterisation);
  - `tldw_Server_API/tests/AuthNZ/unit/test_shared_config_key_security.py` (the security pins, which C4 runs as a required gate).

**Interfaces:**
- Consumes: `resolve` and `Resolved` (Task 3); `moved_key_problems` and `MOVED_SHARED_KEYS` (Task 1).
- Produces, per row:
  - `<home>.resolve_<key>(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> Resolved`. It gives the raw value under the row's rule and never validates; only parser errors propagate (Global Constraints, "What raises, and where").
  - The env-layer accessor `<home>.<key>_env(env: Mapping[str, str] | None = None) -> str | None`. It is exactly `env.get(KEY)`, read live, with `env=None` meaning `os.environ`.
  - The typed field named in the row.
  - `load_auth_config` and `load_rag_config` gain a default `config_parser=None`, which loads `load_comprehensive_config()` lazily.
  - Each secret-bearing field is `field(repr=False)`.

**Reader classes.**
- A *setting reader* wants the effective value. Under its approved D1 item it calls `resolve_<key>`, or the typed loader.
- An *env-layer reader* calls `<key>_env()`, with no behaviour change. This covers:
  - env-drift detectors;
  - key rings and signing-candidate lists;
  - the generic content backend;
  - optional-feature switches;
  - the setup wizard's `.env` checks, which become `<key>_env() or env_values.get(KEY)`.
- An inline reader with no callable of its own (the `chat.py` telemetry dicts) is not pinned; the PR says so.

**The rows:**

| # | Key (aliases) | Home and rule | Setting readers (change only under the D1 item) | Env-layer readers (no change) | Extra checks | Lines |
|---|---|---|---|---|---|---|
| 1 | `AUTH_MODE` (`APP_MODE`) | `auth.resolve_auth_mode` is the AuthNZ overlay's rule: exact-case env `AUTH_MODE` (`blank_env=True`) > `[AuthNZ] auth_mode` (`blank_config=True`) > `single_user`. A blank value counts as set and reaches pydantic, which rejects it, as `maybe_set` does today (pinned by the C1 test's `""` cases). `auth.auth_mode_env`. `load_auth_config().mode` is unchanged: tolerant, blank = unset, an invalid value falls to the next layer, the `APP_MODE` fallback comes last, and it never raises (pinned by Step 1's matrix). Under D1(a), LazySettings' `SINGLE_USER_MODE = load_auth_config(load_comprehensive_config()).mode == "single_user"` replaces `config.py:965-966` | (b) `setup.py:808`, `command_router.py:195`, `db_path_utils.py:62`, `Evaluations/config_validator.py:82`, `Sync/v2/security.py:131`, `main.py:2508`, `claims_review_metrics_scheduler.py:132,164`, `startup_heavy_init.py:405`, `chat.py:2788,3867`. (c) `connectors_service.py:144`. Each calls `load_auth_config().mode` and keeps its own normalisation of the result | `AuthNZ/initialize.py:424` (setup), `:776` (drift); `browser_transport.py:212` (its explicit check becomes `auth_mode_env(environment)`; its `load_auth_config` call is unchanged) | C4; C6 if D1(a) is approved; `git grep -n SINGLE_USER_MODE -- tldw_Server_API/tests`, run every hit | ~300 |
| 2 | `SINGLE_USER_FIXED_ID` | `auth.resolve_single_user_fixed_id`: env > 1 (AuthNZ's and LazySettings' rule). `config_sections.auth` stops reading `[AuthNZ] single_user_fixed_id` and warns once if it is set (D1g; rejection drops this row) | `Watchlists/websub.py:294` (no value change) | none | `git grep -n SINGLE_USER_FIXED_ID -- tldw_Server_API/tests`, run every hit | ~100 |
| 3 | `ACCESS_TOKEN_EXPIRE_MINUTES` | `auth.resolve_access_token_expire_minutes` for the AuthNZ overlay. LazySettings' entry (`config.py:979`, `:1508`) is deleted: no LazySettings reader exists, and every reader uses AuthNZ's field | none | none | `tests/AuthNZ/unit` | ~50 |
| 4 | `DATABASE_URL` | Split (scope ruling 6). `database.database_url_env` is the content-backend override: env only, exactly today's reads. `auth.resolve_database_url` is the AuthNZ users DB: env > `[AuthNZ] database_url` > the `db_type` synthesis moved from `settings.py:1487-1508`. A synthesised URL reports `source="config"` with `origin="AuthNZ.db_type"`, so the overlay applies it as today. With `pg_password` unset it logs today's warning and reports `source="default"`, so nothing is overridden, as today. LazySettings takes `database_url_env()` when set (blank kept, as `os.getenv(k, default)`), else its own default. No change anywhere | none | `DB_Management/backends/base.py:187` (`DatabaseConfig.from_env`: the generic backend and the RLS auto-ensure target), `core/startup_preflight.py:134`, `AuthNZ/db_config.py:172`, `AuthNZ/initialize.py:772` (drift) | C4; C5 over `tests/helpers/test_pg_env_precedence.py`, `tests/AuthNZ/unit`, and the new `tests/DB/integration/test_rls_ensure_target_pg.py`; also `tests/DB/unit/test_db_backends_normalization.py`, `tests/Services/test_startup_pg_rls_mandatory.py`, `tests/Services/test_startup_infra_services.py` | ~200 |
| 5 | `REDIS_URL` | Split. `auth.resolve_redis_url` for the AuthNZ overlay (`[AuthNZ] redis_url`). The new `config_sections/redis.py:resolve_redis_url` (non-empty env > `[Redis] redis_url`) for LazySettings, which keeps its host/port synthesis after it. `redis.redis_url_env` for features. No change anywhere | none | `embeddings_v5_production_enhanced.py:567`, `AuthNZ/migrations.py:6822`, `DB_Management/db_migration.py:688`, `Embeddings/worker_config.py:15`, `Resource_Governance/governor_factory.py:43`, `Sandbox/streams.py:618` | `git grep -n REDIS_URL -- tldw_Server_API/tests`, run every hit | ~180 |
| 6 | `JWT_SECRET_KEY` | `auth.resolve_jwt_secret_key`: env > `[AuthNZ] jwt_secret_key`. `auth.jwt_secret_key_env`. AuthNZ's `_ensure_jwt_secret` is unchanged: it skips single-user and RS/ES keys, has a pytest-only deterministic fallback, and otherwise raises. LazySettings: the resolved value, else its own public default (D1f) | (f) LazySettings' entry (notes cursors), `chat.py:6884`, `Evaluations/config_validator.py:113` | `AuthNZ/admin_webhook_secrets.py:24` (key ring), `Prototype_Workspaces/access.py:49`, `preview_broker.py:63`, `AuthNZ/initialize.py:268` (setup) | C4; `tests/Notes`; `tests/Admin_Webhooks`, `tests/Admin/test_admin_webhooks_service.py` | ~200 |
| 7 | `SINGLE_USER_API_KEY` (`API_KEY`) | `auth.resolve_single_user_api_key`: env `SINGLE_USER_API_KEY`, blank included > `[AuthNZ] single_user_api_key`. `auth.single_user_api_key_env` and `auth.api_key_env`. The explicit read at `settings.py:972` keeps `single_user_api_key_env() or api_key_env()`. Under D1(d), LazySettings takes the resolved key, else `api_key_env()` | (d) LazySettings (`prompts.py:382,390`), `config_validator.py:86`. (e) `auth_deps.py:2294`, in the frozen `require_token_scope._checker`, net lines ≤ 0 | `endpoints/auth.py:1848,2343,2439` and `evaluations_auth.py:147`: their env fallbacks come after AuthNZ's key, so they take the accessors with today's values; `admin_webhook_secrets.py:25` (the ring keeps `single_user_api_key_env() or api_key_env()`, so stored secrets stay decryptable), `Prototype_Workspaces/access.py:50`, `preview_broker.py:64`, `AuthNZ/initialize.py:254` (setup), `:780` (drift) | C4; `tests/AuthNZ` in both shards (C5); `tests/Evaluations` with `RUN_EVALUATIONS=1`; `tests/Admin_Webhooks` | ~280 |
| 8 | `RAG_DEFAULT_LLM_PROVIDER`, `RAG_DEFAULT_LLM_MODEL` | `rag.resolve_default_llm_provider` and `rag.resolve_default_llm_model`: non-empty env > `[RAG] default_llm_provider` / `default_llm_model` as stored (a blank value stays blank) > None (D1h; rejection drops this row). LazySettings (`config.py:2105-2114`), `load_and_log_configs` (`:4358-4362`) and `RAGConfig` delegate. `config_sections.rag` reads the shipped option names; `test_config_sections_typed_loaders.py:539` changes | (h) `RAG/rag_service/generation_defaults.py:18,22` | none | `tests/RAG` | ~110 |

- [ ] **Step 1: Pin today's behaviour, for every source and reader in the row.**
  - In `tests/Config/test_shared_config_keys.py`, add `test_<key>_resolution_today`, parametrized over env set, env blank, env invalid (for validated keys), config only, and neither. It covers:
    - `config_sections`, through its current function;
    - AuthNZ: `reset_settings()`, then `get_settings()`, catching the validation error;
    - LazySettings: `config.clear_config_cache()`, then `config.load_settings()[<entry>]`;
    - each reader's nearest callable.
  - Delete `AUTH_MODE` and `DATABASE_URL` with `monkeypatch.delenv` where a case needs them unset, because the root conftest sets both (`tests/conftest.py:53,56`).
  - In `tests/AuthNZ/unit/test_shared_config_key_security.py`, add the row's security pins in their *today* form:
    - **Row 1:** `test_invalid_auth_mode_fails_closed_in_every_source`, parametrized over source (env | config) × value (`"multi-user"`, `"MULTI_USER"`, `""`). The config case deletes env `AUTH_MODE` and patches `settings.load_comprehensive_config` with a parser holding `[AuthNZ] auth_mode = <value>`. It asserts that `reset_settings(); get_settings()` raises. This passes on dev.
    - **Rows 6 and 7:** `test_webhook_secret_ring_still_decrypts`.
      - Env `API_KEY=X`; no env `SINGLE_USER_API_KEY`, `JWT_SECRET_KEY`, `BYOK_*` or `SESSION_ENCRYPTION_KEY`; `[AuthNZ] single_user_api_key = Y` patched in.
      - `encrypt_admin_webhook_secret("s")` gives key id `single_user_api_key`.
      - `decrypt_admin_webhook_secret` returns `"s"`.
    - **Row 7:** `test_single_user_api_key_accepted_keys`, modelled on `test_require_token_scope_enforces_bearer_api_key` (`tests/AuthNZ/unit/test_scoped_token_enforcement.py:164`). With the same X/Y setup in single-user mode, both X and Y authenticate today.
  - Run both files on dev; they pass. Commit them alone (`test(config): pin <KEY> resolution in every source today (TASK-13950.21)`).
- [ ] **Step 2: Write the failing tests for the row.**
  - `test_<key>_resolves_through_config_sections`: `resolve_<key>` returns the row's rule for each input above.
  - Add the key and its aliases to `MOVED_SHARED_KEYS`. `test_moved_shared_keys_are_read_only_in_config_sections` now fails, listing every reader in the row.
  - **Row 1** adds `test_resolve_auth_mode_passes_a_blank_through`:
    - `resolve_auth_mode` returns `Resolved("", "env", "AUTH_MODE")` for env `AUTH_MODE=""`;
    - it returns `Resolved("", "config", "AuthNZ.auth_mode")` for a blank `[AuthNZ] auth_mode` with env unset;
    - `load_auth_config(parser, env={"AUTH_MODE": ""}).mode` keeps today's value, because blank is unset there.

    The fail-closed pin from Step 1 must keep passing after Step 3.
  - **Rows 4, 6 and 7** add `test_auth_config_repr_hides_secrets`: `"s3cret"` must not appear in `repr(load_auth_config(parser, env={<KEY>: "s3cret"}))`.
  - **Row 4** adds `test_rls_ensure_target_ignores_authnz_database_url` in `tests/DB/integration/test_rls_ensure_target_pg.py`:
    - with `pg_temp_db`, delete env `DATABASE_URL`;
    - set `TLDW_DB_BACKEND=postgresql` and `TLDW_PG_HOST`/`PORT`/`DATABASE`/`USER`/`PASSWORD` from `pg_temp_db`;
    - patch `config.load_comprehensive_config` with a parser holding `[AuthNZ] database_url = sqlite:///./Databases/users.db`;
    - `monkeypatch.setattr(infra, "_postgres_content_mode_active", lambda: True)`, where `infra` is `tldw_Server_API.app.services.startup_infra_services`;
    - `await infra._maybe_ensure_pg_rls(lambda backend: seen.append(backend.config))`;
    - assert `seen[0].backend_type == BackendType.POSTGRESQL` and `seen[0].pg_database == pg_temp_db["database"]`.

    It passes on dev too; it is a guard.
- [ ] **Step 3: Move the resolution into `config_sections`, and move the env-layer readers (no D1 needed).**
  - In `_load_overrides_from_config`, the row's `maybe_set(...)` becomes `resolved = resolve_<key>(cfg)`; `if resolved.source == "config": overrides[FIELD] = caster(resolved.value)`.
  - The raw value reaches pydantic, which rejects a bad `auth_mode` exactly as today. Nothing in this `try` validates or raises.
  - Every env-layer reader in the row calls `<key>_env()`. Every explicit `os.getenv(<KEY>)` in `settings.py` does too.
  - Run Step 1's tests. Nothing has changed.
- [ ] **Step 4: Move the setting readers and LazySettings** (all D1 items are approved).
  - `load_settings`' entry calls `resolve_<key>(load_comprehensive_config())`; the loader is cached. Its net line count stays at zero or below.
  - Each setting reader calls the live loader: `load_auth_config().mode` or the row's typed field. Neither raises for a bad value, and each reader keeps its own normalisation of the result.
  - Update Step 1's expected values for the approved items in this commit. That diff is the reviewed behaviour change. For row 7 under D1(e), `test_single_user_api_key_accepted_keys` now expects Y → 200 and X → 401.
- [ ] **Step 5: Run both test files, the row's extra checks, C1, C2 and C3**, then `--tighten` both ratchets. A row moves no name out of `app/`, so the docs baseline is unchanged.
- [ ] **Step 6: Commit** (`refactor(config): <KEY> resolves through config_sections; LazySettings and AuthNZ delegate (TASK-13950.21)`).

### Task 8: Close-out (AC1-AC5, DoD)

**Files:**
- Modify (via `tools/backlog-py` only): `backlog/tasks/task-13950.21 - Consolidate-configuration-sources-behind-typed-sections.md`.
- Create (via `tools/backlog-py`): the follow-ups in Step 2.

- [ ] **Step 1: Check each AC on dev.**
  - **AC1:** `env_read_baseline.json` exists, and C1 runs it. Record the seed, the final `--summary`, and the roadmap metric (reads outside the config modules).
  - **AC2:** the docs baseline exists, and C1 runs it. Record the seed (1,886 expected) and the final count.
  - **AC3, met under scope ruling 2:**
    - `core/config_sections/precedence.py` is the only `config_sections` entry;
    - every `_get_raw` body is one `resolve` call;
    - the route-toggle pins pass in C1.
  - **AC4:**
    - `MOVED_SHARED_KEYS == SHARED_KEYS`;
    - the splits are recorded: `DATABASE_URL` and `REDIS_URL`, plus the env-layer readers of every key;
    - the moved-key and MCP pins pass.
  - **AC5:** `test_backend_required_enforces_config_ratchets` passes for its three parameters, and security-required runs `tests/AuthNZ/unit`.
- [ ] **Step 2: File the follow-ups** with `tools/backlog-py`, each depending on TASK-13950.21:
  - "Route config.py's inline env reads through the resolver" (`load_settings` 191, `load_comprehensive_config` 58, `load_and_log_configs` 49 reads on 7a020da14a, re-measured);
  - "Authenticate notes attachment cursors with a real per-install secret" (security, low severity);
  - "Document the env names in env_vars_undocumented_baseline.txt";
  - "Count env names that reach a read only through a variable" (module-level `*_ENV_KEYS` tuples, f-string keys, a key copied into a local before the read);
  - "Decide whether optional Redis features and the generic content backend honour config.txt" (the `REDIS_URL` and `DATABASE_URL` splits);
  - "Give admin-webhook secrets a dedicated encryption key, with re-encryption" (the ring depends on auth keys today);
  - "Apply configured RAG contextual defaults to direct unified_rag_pipeline callers". Its description lists the callers that keep `unified_pipeline.py:2056-2059`'s literals: the workflows RAG search adapter (`Workflows/adapters/rag/search.py:131`); the shared-workspace chat service (literal table `Sharing/shared_workspace_chat_service.py:233-234`, pipeline binding `:491`); slides (`api/v1/endpoints/slides.py:2739`, `MCP_unified/modules/implementations/slides_module.py:1649`); data tables (`Data_Tables/jobs_worker.py:828`); and eval_runner (`Evaluations/eval_runner.py:774`). It also asks for a `git grep -n 'unified_rag_pipeline(' -- tldw_Server_API/app` re-check for others, such as `Evaluations/embeddings_abtest_service.py:683`.
- [ ] **Step 3: Record notes and the final summary** with `tools/backlog-py`.
  - Record the measured drift: 3,302 reads / 608 modules / 265 at import time / 1,886 names (or the seed PR's numbers), against the review's 2,694 / 592 / 162 / ~570, with the matcher and docs reasons.
  - Also record: D1-D3 answers, PR numbers, the C6 comparison of the `AUTH_MODE` PR, and the follow-up ids.
  - Add the metric to TASK-13950's notes.
  - DoD #4: `bandit -q -r <changed app files>` per PR, "no new findings".
  - Stage with `git add -A backlog/tasks`.

## Self-review

- **Spec coverage.**
  - AC1: Task 1.
  - AC2: Task 2.
  - AC3: Tasks 3-4, under scope ruling 2.
  - AC4: Task 7 and `MOVED_SHARED_KEYS`; MCPConfig by measurement; splits by scope ruling 6.
  - AC5: Task 1 Step 7, Task 2 Step 6 and Task 4; the security pins run under C4.
  - The handoffs: Tasks 5-6. Rulings 1-6: Rulings section.
- **Fix round 1.**
  - C1: the overlay passes raw values, and is pinned with config-file cases.
  - C2: `from_env` stays on `database_url_env()`, pinned on Postgres.
  - C3: the ring keeps both env keys, and is pinned by decryption.
  - I1-I3: spy fakes, `MonkeyPatch.context`, and parser errors propagate (`cfg_raises`).
  - I4: transitive helpers with key index (seeds re-measured).
  - I5: security pins under C4.
  - I6: `REDIS_URL` readers stay env-layer.
  - I7: D1 lists each effect.
  - I8: reader classes, plus `test_single_user_api_key_accepted_keys`.
  - Dry run 1-4: seeding after #3433, `AnnAssign`, helpers over `core/testing.py`, Task 5 fail-soft.
- **Names used across tasks:**
  - `resolve`/`Resolved` (Task 3) are used in Tasks 4-7.
  - `Read`, `app_reads`, `measure_reads`, `settings_env_names`, `moved_key_problems`, `SHARED_KEYS` and `MOVED_SHARED_KEYS` (Task 1) are used in Tasks 2 and 7.
  - `resolve_<key>`, `<key>_env` and the security test names match between Task 7's Interfaces, its table, its steps and Review Focus.
- **Guards each PR can trip.** The ratchets are named in Global Constraints. `.9`/`.12` are untouched, because `config_sections` is foundation and `precedence.py` is stdlib-only. `.11`, `.13`, `.5`, `.7` and `.16` are untouched. `.19`'s test flags are excluded.
