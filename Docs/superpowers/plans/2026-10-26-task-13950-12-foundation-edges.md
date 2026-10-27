# Cut Foundation-Layer Upward Imports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete every foundation-upward line from the `.9` layering baseline (9 pairs, 20 statements), plus `core.Metrics -> core.Ingestion_Media_Processing` (AC3) and `core.Security -> core.stt_observability_context` (cut by the same move), then pin foundation and `core.Metrics` to foundation-only imports, statically and at import time.

**Architecture:** Each edge gets the cheapest fix that keeps behaviour. Code that depends only on foundation moves down, with a re-export shim at the old path. Code that is domain logic moves up to its callers, and every caller is retargeted in the same commit (a shim left in foundation would be the same upward import). Dead code is deleted. The one optional dependency, metrics and tracing for `http_client`, is injected by `core.Metrics`. Two PRs: slice A cuts nine small edges; slice B moves the egress policy into foundation, injects metrics into `http_client`, and lands the pins.

**Tech Stack:** Python 3.12/3.13, FastAPI, pytest, the `.9` AST contract (`tldw_Server_API/tests/lint/test_layering_contract.py` over `tests/lint/_ast_cache.py`).

**Spec:**
- Backlog task: `backlog/tasks/task-13950.12 - Cut-foundation-layer-upward-imports.md` (AC #1-#5 bind).
- Roadmap row: `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`, `## Phase 1`, row `.12` (effort S, 1-2 PR slices, "Removes every foundation-upward entry from the `.9` baseline").

## Measured edges and decisions

Measured on `origin/dev` `35e711080a` merged with `origin/chore/13950-9-layering-contract` `9bff409144` (TASK-13950.9, PR #3337, which carries `.8`/#3334). Merging `origin/fix/13950-6-test-seam-guard-runtime-env` `0ce3828c4d` (#3331) on top changes none of these numbers. Re-checked on `origin/dev` `06a2ceb3cb` (2026-10-26): no file this plan touches changed, so every line number holds. The independent dry run reproduced every number: 253/669 and SCC 100/128 after slice A, and 252/663 and SCC 89/117 after Task 9. Today the baseline holds 262 pairs and 683 statements, and the largest package SCC is 102 (module-level) / 130 (with function-level imports). This is `.9`'s measure, with endpoint families split.

| # | Baseline line deleted | Statements (where) | Decision | Why this and not the alternatives | Task |
|---|---|---|---|---|---|
| 1 | `core.exceptions -> core.AuthNZ: 1` | 1 module-level: `core/exceptions.py:16`, used only as the base of `AudioQuotaStoreUnavailable` (`:836`) | Move down: `UserRegistrationException` and `DatabaseError` are defined in `core/exceptions.py`; `core/AuthNZ/exceptions.py` re-exports both | AC #1. The base class has to move too, or `except UserRegistrationException` stops catching `DatabaseError`. Both classes have empty bodies, and `AuthNZ -> core.exceptions` points down | 1 |
| 2 | `core.config -> core.LLM_Calls: 1` | 1 module-level: `core/config.py:34` (`canonical_provider_name`) | Move down: `core/LLM_Calls/provider_identity.py` becomes `core/Utils/provider_identity.py`, with a shim at the old path | AC #2. The module imports only stdlib and `core.custom_openai_providers` (foundation). The 27 other app importers and 2 test importers stay on the shim; nothing monkeypatches it | 2 |
| 3 | `core.config -> core.Local_LLM: 1` | 1 function-level: `core/config.py:2946` in `get_llamacpp_handler_config` (the TYPE_CHECKING import at `:44` is not counted) | Move the importing code up: config returns plain settings (`get_llamacpp_handler_settings`), and the one caller (`services/startup_heavy_init.py:107-113`) builds `LlamaCppConfig` | TYPE_CHECKING cannot cover a constructor call, and a shim in config would be the same import. There is one caller and one test patch target | 3 |
| 4 | `core.Logging -> core.Security: 1` | 1 module-level: `core/Logging/log_context.py:26` (`_clean_request_id`) | Move down: the stdlib-only id-cleaning helpers (`request_id_middleware.py:12-13,18-19,26-41`) move into `log_context.py`, and the middleware imports them back | `Security -> Logging` points down. The helpers have no state | 4 |
| 5 | `core.http_client -> core.stt_observability_context: 2` | 2 module-level: `core/http_client.py:129-134` | Move down: the 13-line, stdlib-only `core/stt_observability_context.py` becomes `core/Logging/stt_observability_context.py`, with a shim at the old path | It is a `ContextVar`, which is state, not a hook. Injecting it would allow two vars to exist and silently stop STT log redaction; a move keeps one object | 4 |
| 6 | `core.Security -> core.stt_observability_context: 1` | 1 module-level: `core/Security/egress.py:17-19` | Same move: egress is retargeted to `core.Logging` | A platform -> domain edge that the move removes for free | 4 |
| 7 | `core.Utils -> core.Context_Integrity: 2` | 2 module-level: `core/Utils/prompt_loader.py:11-18` | Move the importing code up: `prompt_loader.py` becomes `core/Context_Integrity/prompt_loader.py`. Retarget 20 statements in 19 app files and 16 lines in 7 test files; no shim | `Context_Integrity` is "context integrity controls for prompt-bearing assets", and `prompt_loader` is its integrity-checked loader. `Context_Integrity` imports nothing from the app, so its new importers cannot form a cycle. Injecting the resolver would let a missing installer skip the integrity check (fail open) | 5 |
| 8 | `core.Utils -> core.DB_Management: 4` | 4 function-level: `core/Utils/Utils.py:752,758` (in `get_db_config`, 2 under try/except ImportError) and `core/Utils/metadata_utils.py:125,128` (in `update_version_safe_metadata_in_transaction`) | Delete `get_db_config`: it is deprecated and has zero callers repo-wide. Move the safe-metadata writer up into `core/DB_Management/media_db/runtime/safe_metadata_write_ops.py` and retarget 2 app files and 1 test file | Dead code is deleted, not moved. The writer is raw SQL on `DocumentVersions`/`DocumentVersionIdentifiers`, which belongs in DB_Management | 6 |
| 9 | `core.Metrics -> core.Ingestion_Media_Processing: 1` | 1 module-level: `core/Metrics/metrics_manager.py:23-25` | Move down: `iter_email_metric_definitions` and `_DURATION_BUCKETS` move into `core/Metrics/email_metrics.py`, mirroring `stt_metrics.py`. No shim: `metrics_manager` is the only importer | AC #3 ("Metrics imports only foundation"). AC #4 needs it as well: the SCC stays at 91 without it | 7 |
| 10 | `core.http_client -> core.Security: 6` | 1 module-level (`http_client.py:128`) + 5 function-level (`:1603`, `:1775`, `:1872`, `:2572`, `:4697`) | Move down: `core/Security/egress.py` becomes `core/Utils/egress.py`, and `core/Security/egress.py` becomes a `sys.modules` alias of it | Egress is the SSRF boundary on every outbound request. An injected hook would fail open when nothing installed it, or refuse all outbound HTTP in any process that never imported the installer. `egress.py` already imports only stdlib, loguru, `core.testing` and the STT context (foundation after Task 4). The alias keeps about 109 patch sites working without edits: 11 string targets in 7 test files, plus 98 module-object `setattr`s in 37 of the 68 test files that import egress (AST count from the dry run). It also keeps all 49 app modules that import egress working. A plain re-export shim would not work: patches would land on the shim while `http_client` reads `Utils.egress`, and private names would not be re-exported | 9 |
| 11 | `core.http_client -> core.Metrics: 2` | 2 module-level: `http_client.py:122-127` | Invert: `http_client.install_observability()` takes the registry and tracing-manager getters. `core.Metrics` installs them on import and registers the four `http_client_*` metric families | Metrics and tracing are best-effort, so a no-op default is safe, provided it covers every method `http_client` calls, including `span`/`async_span`, which open each request. `core.Metrics` is platform-layer and too large to move | 10 |

**TYPE_CHECKING is not used for any edge.** `http_client.py:128` (`ConfiguredEndpointScope`) is annotation-only, but the five function-level egress calls still need the module, so moving the module down covers line 128 as well.

**Totals.** Slice A deletes 9 lines and 14 statements (measured by the dry run: 253 pairs, 669 statements; SCC 100/128). Slice B deletes 2 lines and 8 statements (251 pairs, 661 statements; SCC 89/117; the dry run measured 89/117 after Task 9 and simulated Task 10), which meets AC #4's "at most 89" on the module-level measure. Absolute counts move with dev; what each slice proves is its deltas and the named lines.

**Drift from the backlog task**, which was measured on `dca11342de`:
- **Edge count.** The task says 7 edges and 10 statements. Measured, there are 9 foundation pairs and 20 statements: 10 module-level plus 10 function-level. The review counted module-level imports only.
- **Edges the review missed.** `core.Utils -> core.Context_Integrity` (2, module-level), `core.Utils -> core.DB_Management` (4) and `core.config -> core.Local_LLM` (1).
- **Edge it counted that is not foundation.** It included `core.Metrics -> core.Ingestion_Media_Processing`, which `.9` classes as platform.
- **SCC.** The task says 100 -> 89, measured with endpoints collapsed. On `.9`'s measure it is 102 -> 89.
- **Unchanged.** Every file:line the task cites still matches.
- **Not applicable here.** The task's "Keep re-export shims: 86 test files import Chat.Chat_Deps" belongs to `.13`.

## Global Constraints

Copied from the roadmap's `## Global Constraints` where they bind this task:

- **Branching and PR size.** Every PR targets `dev`. One backlog task per PR. A task with more than about 400 changed lines is split into ordered PRs, each passing CI on its own.
- **Ratchet rules.** Every new ratchet:
  - lives in `tldw_Server_API/tests/lint/` (backend) or runs in `frontend-required` (frontend);
  - keys entries by module or qualified name, never by line number;
  - runs in a required gate: appended to backend-required's "Enforce CI contracts and code ratchets" step (frontend: frontend-required), with a workflow-contract assertion like `tests/CI/test_backlog_task_format_ratchet.py`; a ratchet that only runs in the optional full-suite shards enforces nothing;
  - ships with a self-test that seeds one violation and asserts the ratchet fails.
  (Applied: the two pins in Task 10 go into `test_layering_contract.py`. That file is already in the step, and `.9`'s `test_backend_required_enforces_layering_ratchets[tldw_Server_API/tests/lint/test_layering_contract.py]` in `tests/CI/test_license_first_workflow_contracts.py` already asserts it, so AC #5 needs no new wiring.)
- **Raising a baseline.** It is allowed only through a documented bump-with-reason entry, reviewed in the PR, so that security fixes inside frozen functions are not blocked. (This task only deletes lines.)
  - **Exception: foundation and `core.Metrics` admit no `# bump:`.** From Task 10 on, `test_foundation_and_metrics_import_only_foundation` fails on any pair from a foundation package or from `core.Metrics` into a package above foundation, whatever the baseline says. `test_foundation_and_metrics_load_only_foundation_at_import_time` does the same at import time.
  - This is stricter than the roadmap's bump-with-reason rule. The controller accepted it on 2026-10-26 as the Phase 1 exit condition.
  - It binds every later task, `.13`, `.14` and `.17` included. Code those tasks add under a foundation path (for example `core/Utils/template_render.py`, `core/Utils/tokenizer.py` helpers, `core/config_llm_providers.py`) must import only foundation.
  - A security fix that needs foundation to reach up moves the code down or injects it from above.
- **AST caching.** Backend AST ratchets share one cached parse in `tests/lint/_ast_cache.py`, which `.8` creates and `.9`/`.11` reuse; a full-tree scan takes about 9 s.
- **Line-keyed legacy baseline.** `Helper_Scripts/ci/scope_predicate_baseline.txt` is still keyed by line number until `.23` re-keys it. Any PR that shifts lines in a file it lists regenerates it with `--write-baseline`.
  - Measured: of this task's files it lists only `core/RAG/rag_service/unified_pipeline.py:3889`. Task 5 edits that file at `:996` in place, so no line shifts.
  - Task 5 Step 4 re-runs `tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` after its import-order fix. If that run reports a moved line, run `$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline` and commit the result.
- **Shims.** A move keeps a re-export shim at the old import path until `git grep` shows no importer and no string monkeypatch target still uses it.
- **Postgres proof.** Postgres-dependent acceptance criteria are proven in the Postgres shard with `TLDW_TEST_POSTGRES_REQUIRED=1`. A local skip is not evidence.
  - No AC depends on Postgres. The moved safe-metadata writer keeps its Postgres branch verbatim, and Task 6 pins that branch with a backend-typed fake.
  - One Postgres file imports egress: `tests/Admin_Webhooks/test_recovery_backend_matrix.py`, which builds a `URLPolicyResult`. Task 11 Step 2 runs it serially against Postgres, on the base and on slice B.
  - It must run serially. Its fixture `setup_test_database` (`tests/AuthNZ/conftest.py:1453`) drops the fixed-name `tldw_test` database on teardown, so results under xdist are meaningless.
- **Flipping a global default** (moderation per-surface flags, production detection, middleware stack, pool backend): `git grep` the symbol across all tests and run every hit. (Applied in Task 10: the moment `http_client` metrics are registered moves from `http_client` import to `core.Metrics` import. Applied in Task 9: the egress module's identity changes.)
- **ADR edits** (ADR-003, ADR-025) need `Helper_Scripts/refresh_docs_published.sh` and the `Docs/Published` diff in the same PR. (None is planned. ADR-026 cites `core/Security/egress.py`, which stays importable through the alias; do not edit it here.)
- **Review gates.** Each AI-authored PR carries a `## Change summary` (ADR-004) and gets a clean-subagent `/superpowers:requesting-code-review` before merge. Each task's own plan gets an independent review plus a dry run before execution.

Task-specific:
- **Owner waiver.** The owner waived the ADR-004 Change summary for TASK-13950 PRs (2026-10-25) and authorized merging them after green CI plus a clean review. Record it as a `## Change summary` section in each PR body.
- **Backlog.** TASK-13950 task files are not on `dev` (`git ls-tree origin/dev backlog/tasks/` lists none). These PRs edit nothing under `backlog/`. AC checks and notes are made on the docs branch by the orchestrator with `tools/backlog-py`. AC #3 is already amended there, with numbering unchanged: egress and the STT context move into foundation, and metrics and tracing are injected.
- **Ruff.** A slice leaves no ruff finding on a line it wrote, and no new unsorted import block. `$RUFF` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff`.
  - **Never run `ruff --fix`.** Several touched files already have findings at the base (ruff 0.16.5, `--select I,UP045`):
    - `core/config.py`: 59 `UP045`, none safely fixable (`target-version = "py39"`);
    - `I001` in `core/Embeddings/Embeddings_Server/Embeddings_Create.py`, `core/External_Sources/reference_manager_import.py`, `core/MCP_unified/modules/implementations/prompts_catalog.py`, `core/RAG/rag_service/generation.py`, `services/document_processing_service.py` and `services/enhanced_web_scraping_service.py`;
    - `I001` in `tests/DB_Management/test_media_db_api_imports.py` (`:1`, `:1227`) and in 20 function-level blocks of `tests/http_client/test_http_client.py`.

    `--fix` would re-sort those unrelated blocks: about 175 lines in `test_media_db_api_imports.py`, 20 blocks in `test_http_client.py` (which #3345 also edits), and 2-3 lines in each of the six app files (`Embeddings_Create.py` is also edited by #3331).
  - **Place every edited or added import in its sorted position by hand.** `$RUFF check --select I --diff <file>` may be used to see where a line belongs. In the files listed above, its diff also shows unrelated re-sorts; do not apply those.
  - **Checked in Tasks 8 and 11** by "the ruff check" (Conventions), with two parts: findings on lines the slice wrote, and `--select I` on the changed files that were clean at the base.
  - Moved code counts as written unless git sees it as a rename or copy. Task 3 writes the new return type as `dict[str, Any] | None` (`config.py` has `from __future__ import annotations`), and Task 6 writes `str | None`, so neither adds a `UP045`.
- **mypy.** No error on a line this slice wrote. Errors that moved with the code count as written; Task 3 types the llama.cpp settings as `dict[str, Any]` so the 8 `arg-type` errors at `config.py:3027` do not move into `startup_heavy_init.py`.
- **No contract edits.** Do not change `LAYERS`, the package mapping or the counting rules in `test_layering_contract.py`. Task 10 only adds tests to that file. Do not replace a cut import with `importlib.import_module`, `__import__` or a `sys.modules` lookup. The contract's docstring says that is not a cut, and the runtime pin in Task 10 catches module-level cases.
- **Baseline per commit.** Each task's commit deletes exactly the baseline lines named in that task, with `--tighten`, and no other line changes. On a baseline conflict after merging dev, take dev's side, run `--tighten`, and confirm that only this task's lines are gone.
- **Frozen functions.** `function_size_baseline.json` freezes `http_client:fetch` (428), `_afetch_httpx` (418), `_afetch_aiohttp` (352), `_fetch_httpx_response` (333), `metrics_manager:MetricsRegistry._register_standard_metrics` (1596), `config:load_and_log_configs` and `config:load_settings`. No step changes the line count of any of them. The only edit inside one is Task 9's in-place module-path change at `http_client.py:4697`, inside `fetch`. The metric and tracing calls inside the frozen `http_client` functions keep calling `get_metrics_registry()`/`get_tracing_manager()` by module-global name, so Task 10 does not touch them.
- **ADR assessment:** not required. The task enforces the existing layer direction (the `.9` contract), and the egress policy (ADR-026) is unchanged. State this in each PR body.

## #3331 (TASK-13950.6) and other open PRs

- **#3331 changes no foundation edge.** It edits two foundation modules, `core/config.py` (`:35-41` import, `:3451-3470`, `:5999`) and `core/testing.py` (`runtime_environment()`, `is_production_like_env()`). Re-measured with #3331 merged, the foundation baseline lines and the 262/683 totals are identical.
- **Slice A does not stack on #3331.** Its `config.py` hunks (`:34`, `:44`, `:2925-3030`) do not overlap #3331's. It also edits one import line in `Embeddings_Create.py` and one in `unified_pipeline.py`, which #3331 also touches; resolve those by hand if dev has moved.
- **Slice B lands after #3331 merges** (base: dev after slice A and #3331). #3331 adds two path-keyed lint allowlists that name `core/Security/egress.py`:
  - `tests/lint/test_production_detector_single_source.py:72`: `ALLOWLIST["tldw_Server_API/app/core/Security/egress.py"]`, the `ENVIRONMENT`/`APP_ENV`/`ENV` label read, "owner decision pending";
  - `tests/lint/test_test_seam_env_reads_guarded.py:95`: `"core/Security/egress.py"` in the `PYTEST_CURRENT_TEST` module tuple.

  Once egress moves, both fail: the old key goes stale and the new path is unlisted. Task 9 Step 3 renames both keys (the allowlist bullet in Task 9's Files). If #3331 is closed unmerged, skip that rename.
- **`runtime_environment()` is not used here.** Egress reads its own env labels; routing that through `runtime_environment()` is follow-up `.60` ("Egress default profile"). Task 9 moves `egress.py` byte for byte. Nothing else from #3331 applies: `http_client`'s test-mode egress relaxation already imports `core.testing`, which is foundation.
- **Other open PRs** (each three-way merged against the slices by the reviewer):
  - **#3191 and #2855 edit `core/Security/egress.py`.** Both are stale and conflicting, so slice B does not wait for them. If either lands after slice B, it gets a guaranteed conflict there, because the alias keeps a file at the old path and git does not detect a rename. Port it like this: `git diff <its merge base> <its tip> -- tldw_Server_API/app/core/Security/egress.py | sed 's#core/Security/egress.py#core/Utils/egress.py#g' | git apply -3`, then `git checkout origin/dev -- tldw_Server_API/app/core/Security/egress.py` to keep dev's alias. If the PR takes its own side of the old file, `test_old_security_egress_path_is_the_moved_module` goes red.
  - **#3187, #3191, #3344, #3345, #3347, #3348 and #3349 edit `core/http_client.py`.** None adds an upward import or a new observability call. None conflicts with slice B in `http_client.py` beyond the conflicts #3187 and #3191 already have against plain dev. #3345 conflicts trivially with Task 10 in `tests/http_client/test_http_client.py`: it appends after `:2014`, next to the rewritten `:1979-2016`; keep both.
  - **#3019, #2932, #2821 and #2738 edit `core/exceptions.py`.** #2821 adds `from .AuthNZ.exceptions import RollbackSignal, UserRegistrationException` and subclasses them: a new `core.exceptions -> core.AuthNZ` statement, which the layering check rejects after slice A. When it is revived, it imports `UserRegistrationException` from `core.exceptions`, and moves `RollbackSignal` down or subclasses something else.
  - **Recommended merge order:**
    1. #3337, then slice A;
    2. #3331;
    3. #3335, then #3344, #3345, #3347 and #3348, then #3349, preferably before slice B;
    4. slice B;
    5. #3191 and #2855, whenever they are revived, ported as above.

## Review Focus

1. **Egress must stay unconditionally enforced.** A process that imports `http_client` and nothing else must still deny `http://169.254.169.254/` (the cloud-metadata SSRF target), and must not load `core.Security`. Pinned by `test_http_client_denies_metadata_ip_without_core_security` (Task 9).
2. **There must be one STT `ContextVar`.** If the shim and the moved module held two vars, `opaque_stt_http_observability()` would set one, the egress and log-redaction readers would read the other, and STT endpoint redaction would silently stop. Pinned by `test_stt_observability_context_is_one_contextvar` (Task 4).
3. **Old-path egress patches must still steer the policy.** About 109 test patch sites and the string targets `"tldw_Server_API.app.core.Security.egress.evaluate_url_policy"` and `"...is_webhook_url_allowed_for_tenant"` must reach the code `http_client` runs, whichever path a process imports first. Pinned in Task 9 by `test_old_security_egress_path_is_the_moved_module` and `test_old_egress_path_aliases_when_imported_first` (the latter in a fresh interpreter).
4. **Metrics and tracing must degrade, not crash or vanish.**
   - A process that never imports `core.Metrics` must still make requests. Every request opens a span on the tracing manager, so the no-op needs `span`/`async_span`. The root `conftest` always imports `core.Metrics`, so only a fresh interpreter can see the gap.
   - After `core.Metrics` is imported the hooks are live, a failing registry must not crash that import, and `X-Request-Id` still flows from tracing baggage.

   Pinned in Task 10 by:
   - `test_a_process_without_core_metrics_still_fetches` (subprocess, `fetch` and `afetch`);
   - `test_http_client_observability_is_noop_until_installed`;
   - `test_core_metrics_installs_http_client_observability`;
   - `test_installing_observability_survives_a_failing_registry`;
   - `test_installed_tracing_manager_feeds_x_request_id`;
   - `test_http_client_metric_families_register_on_a_fresh_registry`.
5. **A cut must not pass for the wrong reason.** A later `# bump:` could re-add a foundation pair, or a module-level `importlib` call could relaunder one. Pinned by `test_foundation_and_metrics_import_only_foundation` and `test_foundation_and_metrics_load_only_foundation_at_import_time`, each with a seeded self-test (Task 10).

## PR slices

| Slice | Tasks | Base | Hand-written lines (est.) | Mechanical (est.) |
|---|---|---|---|---|
| A. Small edges | 1-8 | `dev` after #3337 (`.9`, carrying #3334) merges | ~250: tests ~125, shims and helpers ~50, `config`/`startup` ~25, import retargets ~45, docs ~5 (ruff's import-order fixes are tool output, not counted) | `git mv` of `provider_identity.py` (86), `stt_observability_context.py` (13), `prompt_loader.py` (425). Cut and paste of the safe-metadata writer (~105), the request-id helpers (~30) and the email definitions (~25). Deletion of `get_db_config` (~36) |
| B. `http_client` | 9-11 | `dev` after slice A and #3331 merge | ~300: hooks ~50, Metrics module ~50, tests ~170, docs and allowlists ~10, alias ~15 | `git mv` of `egress.py` (981) |

The roadmap says 1-2 slices; this plan uses 2. Slice B waits for #3331's allowlists. Slice A unblocks the SCC drop without waiting. Each slice is green on its own, because each task deletes its own baseline lines.

## Conventions

- Run from the repo root of the slice branch. `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; `RUFF=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff`. The shell is zsh: `${B}` must be braced before `:path` (`"${B}:file"`), and word-splitting pipelines use `xargs`.
- **pytest:** `PYTHONPATH=. $PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t1312-<name>" -q <paths>`.
- **The layering check:** `git fetch -q origin dev && PYTHONPATH=. $PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t1312-layering" -q tldw_Server_API/tests/lint/test_layering_contract.py`.
- **The tighten command:** `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_layering_contract --tighten`.
- **The report:** `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_layering_contract`. Its last lines read `ratcheted now: <P> pairs, <S> statements ...` and `largest package SCC: {'module_level': <m>, 'with_function_level': <f>}`.
- **"Lock the cut" step:**
  1. Run the layering check. Expected: FAIL with `<pair>: stale, the edge is gone. Delete the line ...` for each line the task names, and nothing else.
  2. Run the tighten command.
  3. Run `git diff -U0 tldw_Server_API/tests/lint/layering_baseline.txt`. Expected: exactly the named `-` lines.
  4. Run the layering check again. Expected: PASS.
- **The ruff check** (Tasks 8 and 11) has two parts.
  1. Create `$TMPDIR/t1312_ruff_written.py` with the script below, then run `RUFF=$RUFF $PY $TMPDIR/t1312_ruff_written.py`. Expected: no output. The script prints every ruff finding (repo config, all rules) on a line this branch wrote since its merge base with `origin/dev`. It runs `git diff -M -C`, so a rename, or a move whose old file stays as a shim or alias, counts only its changed lines. It was tested in a scratch tree: a git-mv'd file with an alias at the old path reported only the appended line's `F401`.

```python
"""Print ruff findings on lines this branch wrote since its merge base with origin/dev; no output means pass (TASK-13950.12)."""
import os
import re
import subprocess

base = subprocess.run(["git", "merge-base", "origin/dev", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
diff = subprocess.run(
    ["git", "diff", "-U0", "-M", "-C", base, "HEAD", "--", "*.py"], capture_output=True, text=True, check=True
).stdout
written: dict[str, set[int]] = {}
path = None
for line in diff.splitlines():
    if line.startswith("+++ "):
        path = None if line == "+++ /dev/null" else line[6:]
    elif path and (hunk := re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)):
        start, count = int(hunk[1]), int(hunk[2] or 1)
        written.setdefault(path, set()).update(range(start, start + count))
files = sorted(p for p in written if os.path.exists(p))
if files:
    report = subprocess.run(
        [os.environ["RUFF"], "check", "--output-format", "concise", "--exit-zero", *files],
        capture_output=True, text=True,
    ).stdout
    for finding in report.splitlines():
        hit = re.match(r"(.+?):(\d+):\d+: ", finding)
        if hit and int(hit[2]) in written.get(hit[1], ()):
            print(finding)
```

  2. `I001` is reported at the first line of a block, which may not be the line the slice changed. So also run `git diff --name-only --diff-filter=d $(git merge-base origin/dev HEAD) HEAD -- '*.py' | grep -vE 'Embeddings_Create\.py|reference_manager_import\.py|prompts_catalog\.py|rag_service/generation\.py|document_processing_service\.py|enhanced_web_scraping_service\.py|test_media_db_api_imports\.py|tests/http_client/test_http_client\.py|tests/Metrics/test_telemetry_import_fallback\.py' | xargs $RUFF check --select I`. This covers every changed file except those with base findings. Expected: `All checks passed!`.
- **Commits:** conventional style with `(TASK-13950.12)`, ending with the session's `Co-Authored-By` trailer. Never `git stash`. Scratch worktrees go only under `$TMPDIR`.

---

## Slice A (PR 1): small edges

**Before Task 1:**
1. Branch `refactor/13950-12a-foundation-edges` from `origin/dev` after #3337 has merged.
2. Run `grep -nE "^core\.(config|exception|Utils|Logging|testing|http_client|feature_flags|custom_openai_providers)[^ ]* -> |^core\.Metrics -> core\.Ingestion|^core\.Security -> core\.stt" tldw_Server_API/tests/lint/layering_baseline.txt`. Expected: the 11 lines of the table, with the same counts.
3. If a count differs, dev drifted. Re-list the statements with the snippet below, and replan only the pairs that changed:

```bash
PYTHONPATH=. $PY -c "from tldw_Server_API.tests.lint.test_layering_contract import scan, package_of, layer_of, rule
for r in scan():
    a = package_of(r.module)
    if (layer_of(a) == 0 or a in ('core.Metrics', 'core.Security')) and rule(a, r.target):
        print(a, '->', r.target, r.module, 'function' if r.function_level else 'module')"
```

Expected today: 23 lines:
- the 20 foundation statements;
- `core.Metrics -> core.Ingestion_Media_Processing core.Metrics.metrics_manager module`;
- `core.Security -> core.stt_observability_context core.Security.egress module`;
- the out-of-scope `core.Security -> services core.Security.drain_gate_middleware module`, which this plan does not touch.

### Task 1: `DatabaseError` lives in `core.exceptions`

**Files:**
- Modify: `tldw_Server_API/app/core/exceptions.py:16` (delete the import) and `:836` (the base becomes `DatabaseError`). Add the two classes directly above `:836`.
- Modify: `tldw_Server_API/app/core/AuthNZ/exceptions.py:12-14,206-208` (delete both class definitions and import them instead)
- Create: `tldw_Server_API/tests/unit/test_foundation_layer_moves.py` (shard-covered by `tests/unit`; runs in backend-required "Backend unit smoke")
- Modify: `tldw_Server_API/tests/lint/layering_baseline.txt` (delete `core.exceptions -> core.AuthNZ: 1`)

**Interfaces:**
- Produces: `core.exceptions.UserRegistrationException(Exception)` (docstring "Base exception for user registration system") and `core.exceptions.DatabaseError(UserRegistrationException)` (docstring "Base database exception"). `core.AuthNZ.exceptions` exposes the same two objects.

- [ ] **Step 1: Create the test file with this test.**

```python
"""The moves that cut the foundation layer's upward imports (TASK-13950.12)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_database_error_is_defined_in_core_exceptions_and_reexported_by_authnz() -> None:
    from tldw_Server_API.app.core import exceptions as core_exc
    from tldw_Server_API.app.core.AuthNZ import exceptions as authnz_exc

    assert core_exc.DatabaseError.__module__ == "tldw_Server_API.app.core.exceptions"
    assert authnz_exc.DatabaseError is core_exc.DatabaseError
    assert authnz_exc.UserRegistrationException is core_exc.UserRegistrationException
    assert issubclass(core_exc.DatabaseError, core_exc.UserRegistrationException)
    assert issubclass(authnz_exc.TransactionError, core_exc.DatabaseError)
    assert issubclass(core_exc.AudioQuotaStoreUnavailable, authnz_exc.DatabaseError)
```

- [ ] **Step 2: Run it.** pytest on `tldw_Server_API/tests/unit/test_foundation_layer_moves.py`. Expected: FAIL, `AttributeError: module 'tldw_Server_API.app.core.exceptions' has no attribute 'DatabaseError'`.
- [ ] **Step 3: Implement.**
  - In `core/exceptions.py`, add the two classes above `AudioQuotaStoreUnavailable`. Put a one-line comment above them: AuthNZ's base errors live here so foundation code can subclass them, and `core/AuthNZ/exceptions.py` re-exports them (TASK-13950.12).
  - Change `class AudioQuotaStoreUnavailable(AuthNZDatabaseError)` to `(DatabaseError)`, and delete line 16.
  - In `core/AuthNZ/exceptions.py`, delete both class definitions. After the stdlib imports, add `from tldw_Server_API.app.core.exceptions import DatabaseError, UserRegistrationException  # noqa: F401 - re-exported; defined in core.exceptions (TASK-13950.12)`. The AuthNZ subclasses keep their bases by name.
  - Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff).
- [ ] **Step 4: Run the test and its neighbours.** Run pytest on `tldw_Server_API/tests/unit/test_foundation_layer_moves.py` together with the output of `git grep -lE "UserRegistrationException|AudioQuotaStoreUnavailable|AuthNZ\.exceptions import.*DatabaseError|AuthNZ\.exceptions import \($" -- tldw_Server_API/tests` (19 files today). Expected: PASS.
- [ ] **Step 5: Lock the cut** (Conventions). Line: `core.exceptions -> core.AuthNZ: 1`.
- [ ] **Step 6: Commit.**

```bash
git add tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/core/AuthNZ/exceptions.py tldw_Server_API/tests/unit/test_foundation_layer_moves.py tldw_Server_API/tests/lint/layering_baseline.txt
git commit -m "refactor(core): define DatabaseError in core.exceptions; AuthNZ re-exports it (TASK-13950.12)"
```

### Task 2: `provider_identity` moves to `core/Utils`

**Files:**
- Move: `tldw_Server_API/app/core/LLM_Calls/provider_identity.py` -> `tldw_Server_API/app/core/Utils/provider_identity.py` (`git mv`, content unchanged)
- Create (shim at the old path): `tldw_Server_API/app/core/LLM_Calls/provider_identity.py`
- Modify: `tldw_Server_API/app/core/config.py:34` (import from `core.Utils.provider_identity`)
- Modify: `tldw_Server_API/app/core/Utils/README.md` (add a "`provider_identity.py` canonical provider names and aliases" line under the module map)
- Modify: `tldw_Server_API/tests/unit/test_foundation_layer_moves.py`
- Modify: `layering_baseline.txt` (delete `core.config -> core.LLM_Calls: 1`)

**Interfaces:**
- Produces: `core.Utils.provider_identity` with `PROVIDER_ALIASES: dict[str, tuple[str, ...]]`, `canonical_provider_name(provider: str) -> str` and `provider_lookup_names(provider: str) -> tuple[str, ...]`.
- The shim re-exports exactly these three names. They are the only names the 28 app and 2 test importers use; there are no patch targets.

- [ ] **Step 1: Add the test.**

```python
def test_provider_identity_lives_in_utils_and_the_llm_calls_path_reexports_it() -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_identity as old
    from tldw_Server_API.app.core.Utils import provider_identity as new

    assert new.__name__ == "tldw_Server_API.app.core.Utils.provider_identity"
    assert old.canonical_provider_name is new.canonical_provider_name
    assert old.provider_lookup_names is new.provider_lookup_names
    assert old.PROVIDER_ALIASES is new.PROVIDER_ALIASES
    assert new.canonical_provider_name("OpenAI-Compatible") == "custom-openai-api"
```

- [ ] **Step 2: Run it.** Expected: FAIL, `ImportError: cannot import name 'provider_identity' from 'tldw_Server_API.app.core.Utils'`.
- [ ] **Step 3: Implement.**
  - Run `git mv`.
  - Write the shim: a docstring saying it moved to `core/Utils/provider_identity.py` (TASK-13950.12) because `core.config` is foundation, then `from tldw_Server_API.app.core.Utils.provider_identity import (  # noqa: F401` with `PROVIDER_ALIASES, canonical_provider_name, provider_lookup_names`.
  - Retarget `config.py:34`.
  - Update the README line.
  - Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff).
- [ ] **Step 4: Run the tests.** Run pytest on the unit file, `tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py`, `tldw_Server_API/tests/LLM_Calls/test_adapter_registry_defaults.py` and `tldw_Server_API/tests/test_config.py`. Expected: PASS.
- [ ] **Step 5: Lock the cut.** Line: `core.config -> core.LLM_Calls: 1`.
- [ ] **Step 6: Commit** the 5 paths plus the baseline: `refactor(core): move provider_identity into core.Utils behind a shim (TASK-13950.12)`.

### Task 3: config returns llama.cpp settings; startup builds `LlamaCppConfig`

**Files:**
- Modify: `tldw_Server_API/app/core/config.py`:
  - `:44`: delete the TYPE_CHECKING `LlamaCppConfig` import; keep `:43`.
  - `:2925`: rename and retype the function.
  - `:2945-2949`: delete the import `try` block.
  - `:2974`: `kwargs: dict[str, object]` becomes `kwargs: dict[str, Any]` (`Any` is already imported at `:15`).
  - `:3026-3030`: replace with `return kwargs`.
- Modify: `tldw_Server_API/app/services/startup_heavy_init.py:107-113` (call the new helper); add the helper above `_init_local_llm_manager`
- Modify: `tldw_Server_API/tests/Services/test_startup_heavy_init.py:259`; add 2 tests
- Modify: `tldw_Server_API/tests/unit/test_foundation_layer_moves.py`
- Modify: `layering_baseline.txt` (delete `core.config -> core.Local_LLM: 1`)

**Interfaces:**
- Produces: `core.config.get_llamacpp_handler_settings() -> dict[str, Any] | None`. Write it with `|`, not `Optional[...]`, so the written line adds no `UP045`; `config.py` already has `from __future__ import annotations`. `Any`, not `object`, so `LlamaCppConfig(**settings)` type-checks; with `object`, the 8 `arg-type` errors at `config.py:3027` would move into `startup_heavy_init.py`. It returns the `LlamaCppConfig` keyword arguments it builds today, `{"enabled": True, ...}`. It returns `None` when the handler is disabled or the config cannot load. It replaces `get_llamacpp_handler_config`, whose only caller and only patch target are retargeted here.
- Produces: `services.startup_heavy_init._llamacpp_handler_config() -> LlamaCppConfig | None`. It builds the config from those settings. On invalid settings (`TypeError`/`ValueError`, the latter including pydantic `ValidationError`) it logs `llama.cpp handler config invalid; disabling handler: <exc>` at warning and returns `None`. This keeps today's behaviour: an invalid config disables only the handler, not the Local-LLM manager.

- [ ] **Step 1: Add the tests.** Add to the unit file:

```python
def test_llamacpp_handler_settings_are_plain_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core import config

    monkeypatch.setenv("LLAMACPP_ENABLED", "1")
    monkeypatch.setenv("LLAMACPP_PORT", "9001")
    monkeypatch.setenv("LLAMACPP_ALLOWED_PATHS", "/models/a, /models/b")

    settings = config.get_llamacpp_handler_settings()

    assert settings is not None
    assert settings["enabled"] is True and settings["default_port"] == 9001
    assert settings["allowed_paths"] == [Path("/models/a"), Path("/models/b")]


def test_llamacpp_handler_settings_are_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.setenv("LLAMACPP_ENABLED", "0")
    assert config.get_llamacpp_handler_settings() is None
```

  Add to `tests/Services/test_startup_heavy_init.py`. These use no fake classes, so the test-quality `stub_injection` count for that file stays at its baseline of 1:

```python
def test_llamacpp_handler_config_is_built_from_config_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    startup_heavy = _import_startup_heavy_init()
    from tldw_Server_API.app.core import config as config_module
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig

    monkeypatch.setattr(config_module, "get_llamacpp_handler_settings", lambda: {"enabled": True, "default_port": 9001})
    cfg = startup_heavy._llamacpp_handler_config()
    assert isinstance(cfg, LlamaCppConfig) and cfg.default_port == 9001


def test_invalid_llamacpp_settings_disable_only_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    startup_heavy = _import_startup_heavy_init()
    from tldw_Server_API.app.core import config as config_module

    monkeypatch.setattr(config_module, "get_llamacpp_handler_settings", lambda: {"enabled": True, "default_port": "not-a-port"})
    assert startup_heavy._llamacpp_handler_config() is None
```

  Retarget `:259`: `monkeypatch.setattr(config_module, "get_llamacpp_handler_settings", lambda: None)`.
- [ ] **Step 2: Run them.** Run pytest on the unit file and `tldw_Server_API/tests/Services/test_startup_heavy_init.py`. Expected: FAIL. All 5 fail with `AttributeError ... has no attribute 'get_llamacpp_handler_settings'`: the 2 unit tests call it, and the 2 new startup tests and the retargeted `:259` test fail at their `monkeypatch.setattr` before they reach `_llamacpp_handler_config`.
- [ ] **Step 3: Implement the config and startup edits** (Files and Interfaces above).
  - `_llamacpp_handler_config` imports `get_llamacpp_handler_settings` and `LlamaCppConfig` inside the function, the way the module already imports lazily.
  - In `_init_local_llm_manager`, line 107's config import goes, and line 110 becomes `_llama_cfg = _llamacpp_handler_config()`.
  - Update the config function's docstring first line to say it returns settings, not a config object.
  - Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff). Write the new return type as `dict[str, Any] | None` (not `Optional[...]`, which adds a `UP045`).
- [ ] **Step 4: Run the same tests.** Expected: PASS. Then run `git grep -n "get_llamacpp_handler_config" -- tldw_Server_API`. Expected: no output.
- [ ] **Step 5: Lock the cut.** Line: `core.config -> core.Local_LLM: 1`.
- [ ] **Step 6: Commit:** `refactor(core): config returns llama.cpp settings; startup builds LlamaCppConfig (TASK-13950.12)`.

### Task 4: request-id helpers and the STT observability context move into `core/Logging`

**Files:**
- Modify: `tldw_Server_API/app/core/Logging/log_context.py`:
  - `:26`: replace the import with the moved helpers.
  - Add `import uuid`.
  - Change `from collections.abc import Iterator` to `Callable, Iterator`.
- Modify: `tldw_Server_API/app/core/Security/request_id_middleware.py`:
  - Delete `:12-13` (`SAFE_REQUEST_ID_PATTERN`, `MAX_REQUEST_ID_LENGTH`), `:18-19` (`_generate_request_id`), `:26-41` (`_clean_id`, `_clean_request_id`) and `import re`.
  - Add `from tldw_Server_API.app.core.Logging.log_context import _clean_id, _clean_request_id  # noqa: F401 - re-exported; defined in Logging (TASK-13950.12)`.
- Move: `tldw_Server_API/app/core/stt_observability_context.py` -> `tldw_Server_API/app/core/Logging/stt_observability_context.py` (`git mv`, content unchanged)
- Create (shim at the old path): `tldw_Server_API/app/core/stt_observability_context.py`
- Modify: `tldw_Server_API/app/core/http_client.py:129,132` and `tldw_Server_API/app/core/Security/egress.py:17`. Change only the module path, to `tldw_Server_API.app.core.Logging.stt_observability_context`, keeping both `http_client` statements so no line shifts.
- Modify: `tldw_Server_API/app/core/Logging/README.md:36-37`:
  - `log_context.py` owns request-id sanitizing, and `Security/request_id_middleware.py` applies it to inbound `X-Request-ID`.
  - Add a `stt_observability_context.py` line: the opaque STT endpoint id for log redaction.
- Modify: unit test file; `layering_baseline.txt`

**Interfaces:**
- Produces: `core.Logging.log_context._clean_request_id(value: str | None) -> str`, `_clean_id(value: str | None, generator: Callable[[], str]) -> str`, `_generate_request_id() -> str`, `SAFE_REQUEST_ID_PATTERN` and `MAX_REQUEST_ID_LENGTH`, all verbatim.
- Produces: `core.Logging.stt_observability_context.OPAQUE_STT_ENDPOINT_ID: ContextVar[str | None]` and `get_opaque_stt_endpoint_id() -> str | None`. The old module re-exports both (the same objects).
- Leave the four `source_modules` strings in `Ingestion_Media_Processing/Audio/stt_provider_adapter.py` (`:1825-1827`, `:2156-2158`, `:2415-2417`, `:2719-2721`) unchanged. They name `...core.stt_observability_context` and `...core.Security.egress`, which stay importable. Renaming them changes STT plan descriptors, and `tests/Audio/test_stt_execution_plan_network.py:359,859-861,1649-1651` and `tests/Audio/test_stt_provider_adapter.py:2448-2456` pin those names. The rename belongs to the shim-retirement follow-up.

- [ ] **Step 1: Add the tests.**

```python
def test_request_id_helpers_live_in_logging() -> None:
    from tldw_Server_API.app.core.Logging import log_context
    from tldw_Server_API.app.core.Security import request_id_middleware

    assert log_context._clean_request_id.__module__ == "tldw_Server_API.app.core.Logging.log_context"
    assert request_id_middleware._clean_request_id is log_context._clean_request_id
    assert log_context._clean_request_id("req-1") == "req-1"
    assert log_context._clean_request_id("bad id!") != "bad id!"


def test_stt_observability_context_is_one_contextvar() -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core import stt_observability_context as old
    from tldw_Server_API.app.core.Logging import stt_observability_context as new

    assert old.OPAQUE_STT_ENDPOINT_ID is new.OPAQUE_STT_ENDPOINT_ID
    endpoint_id = "sha256:" + "a" * 64
    with http_client.opaque_stt_http_observability(endpoint_id):
        assert old.get_opaque_stt_endpoint_id() == endpoint_id
    assert new.get_opaque_stt_endpoint_id() is None
```

- [ ] **Step 2: Run them.** Expected: FAIL. The first test fails on its `__module__` assertion: today `log_context` only imports the function, and the function is defined in `...Security.request_id_middleware`. The second fails with `ImportError` for `core.Logging.stt_observability_context`.
- [ ] **Step 3: Implement the moves** (Files above). The shim docstring says it moved to `core/Logging/stt_observability_context.py` (TASK-13950.12) and is re-exported so the old path keeps the same `ContextVar`. Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff).
- [ ] **Step 4: Run the tests.** Run pytest on the unit file, `tldw_Server_API/tests/Security/test_request_id_middleware.py`, `tldw_Server_API/tests/Logging`, `tldw_Server_API/tests/http_client/test_stt_opaque_observability.py`, `tldw_Server_API/tests/Audio/test_stt_execution_plan_network.py` and `tldw_Server_API/tests/Audio/test_stt_provider_adapter.py`. Expected: PASS.
- [ ] **Step 5: Lock the cut.** Lines: `core.Logging -> core.Security: 1`, `core.http_client -> core.stt_observability_context: 2`, `core.Security -> core.stt_observability_context: 1`.
- [ ] **Step 6: Commit:** `refactor(core): request-id helpers and the STT observability context move into core.Logging (TASK-13950.12)`.

### Task 5: `prompt_loader` moves into `core/Context_Integrity`

**Files:**
- Move: `tldw_Server_API/app/core/Utils/prompt_loader.py` -> `tldw_Server_API/app/core/Context_Integrity/prompt_loader.py` (`git mv`, content unchanged; no shim)
- Modify (retarget the import; 20 statements in 19 app files): `core/Chat/document_generator.py:140`, `core/Chunking/strategies/propositions.py:31`, `core/Chunking/strategies/rolling_summarize.py:13`, `core/Claims_Extraction/claims_engine.py:83`, `core/Claims_Extraction/ingestion_claims.py:72`, `core/Claims_Extraction/prompt_validation.py:27`, `core/Embeddings/ChromaDB_Library.py:78`, `core/Embeddings/Embeddings_Server/Embeddings_Create.py:139`, `core/Ingestion_Media_Processing/Audio/Audio_Files.py:1309`, `core/LLM_Calls/Summarization_General_Lib.py:57`, `core/MCP_unified/modules/implementations/prompts_catalog.py:32` (relative `....Utils.prompt_loader`), `core/Prompt_Management/service_prompts.py:769,779`, `core/RAG/rag_service/advanced_reranking.py:33`, `core/RAG/rag_service/generation.py:27`, `core/RAG/rag_service/prompt_templates.py:16`, `core/RAG/rag_service/unified_pipeline.py:996`, `core/Slides/standalone_html_config.py:16`, `services/document_processing_service.py:25`, `services/enhanced_web_scraping_service.py:36` (all under `tldw_Server_API/app/`)
- Modify (retarget; 16 lines in 7 test files): `tests/Media/test_audio_summary_service_prompt.py:24`, `tests/Media/test_web_summary_service_prompt.py:22`, `tests/Prompt_Management/test_service_prompts_api.py:462,515`, `tests/RAG_NEW/unit/test_generation_prompt_loader.py:22`, `tests/Slides/test_standalone_html_config.py:23`, `tests/Utils/test_prompt_loader_env_overrides.py:19,28,57`, `tests/Utils/test_prompt_loader_paths.py:6,24,42,60,78,114,146` (under `tldw_Server_API/`). These tests patch the module object (`monkeypatch.setattr(prompt_loader, ...)`); there are no string targets.
- Modify: `tldw_Server_API/app/core/Utils/README.md:12-13,38` (drop `prompt_loader.py`; add a "prompt loading moved to `Context_Integrity/prompt_loader.py`" note)
- Modify: unit test file; `layering_baseline.txt`

**Interfaces:**
- Produces: `core.Context_Integrity.prompt_loader`, the same module, with `load_prompt`, `load_prompt_strict`, `PromptAssetUnavailableError` and its private helpers unchanged.

- [ ] **Step 1: Add the test.**

```python
def test_prompt_loader_lives_in_context_integrity_with_no_utils_shim() -> None:
    import importlib

    from tldw_Server_API.app.core.Context_Integrity import prompt_loader

    assert callable(prompt_loader.load_prompt)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_Server_API.app.core.Utils.prompt_loader")
```

- [ ] **Step 2: Run it.** Expected: FAIL, `ImportError: cannot import name 'prompt_loader' from 'tldw_Server_API.app.core.Context_Integrity'`.
- [ ] **Step 3: Implement.**
  1. Run `git mv`.
  2. Run the codemod. The pathspec excludes the unit file: its Step 1 test must keep naming the old path.

     ```bash
     git grep -lE "core\.Utils\.prompt_loader|core\.Utils import prompt_loader|\.\.\.\.Utils\.prompt_loader" -- tldw_Server_API ':!tldw_Server_API/tests/unit/test_foundation_layer_moves.py' | xargs sed -i '' -E 's/core\.Utils\.prompt_loader/core.Context_Integrity.prompt_loader/g; s/core\.Utils import prompt_loader/core.Context_Integrity import prompt_loader/g; s/\.\.\.\.Utils\.prompt_loader/....Context_Integrity.prompt_loader/g'
     ```

  3. Edit the README.
  4. Run `git grep -nE "Utils\.prompt_loader|Utils import prompt_loader" -- tldw_Server_API`. Expected: exactly one line, the unit file's `importlib.import_module("tldw_Server_API.app.core.Utils.prompt_loader")`.
  5. The codemod changes each path in place, and `Context_Integrity` sorts differently from `Utils`. Move each retargeted import to its sorted position by hand. `git diff --name-only -- '*.py' | xargs $RUFF check --select I --diff` shows where each one belongs; the dry run found 16 files that need a move. In the files with base `I001` findings (Global Constraints, Ruff: `Embeddings_Create.py`, `prompts_catalog.py`, `generation.py`, `document_processing_service.py` and `enhanced_web_scraping_service.py`), move only the retargeted line. Never run `--fix`.
- [ ] **Step 4: Run the tests.**
  - Run pytest on the unit file, the 7 test files above, `tldw_Server_API/tests/Context_Integrity` and `tldw_Server_API/tests/Slides`. Expected: PASS.
  - Run pytest on `tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py`. Its baseline lists `unified_pipeline.py:3889`. Expected: PASS. If it reports a moved line, run `$PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline` and add `Helper_Scripts/ci/scope_predicate_baseline.txt` to the commit.
- [ ] **Step 5: Lock the cut.** Line: `core.Utils -> core.Context_Integrity: 2`.
- [ ] **Step 6: Commit:** `refactor(core): prompt_loader moves into Context_Integrity; importers retargeted (TASK-13950.12)`.

### Task 6: `core.Utils` stops reaching into DB_Management

**Files:**
- Modify: `tldw_Server_API/app/core/Utils/Utils.py:737-772` (delete `get_db_config` and its "DB Config Loading" banner)
- Create: `tldw_Server_API/app/core/DB_Management/media_db/runtime/safe_metadata_write_ops.py`.
  - It holds `update_version_safe_metadata_in_transaction`, moved verbatim from `core/Utils/metadata_utils.py:104-208`.
  - It starts with `from __future__ import annotations`, as `safe_metadata_search_ops.py` does. `safe_metadata_json` is therefore annotated `str | None` (ruff UP045).
  - The function's local imports (`sqlite3`, `datetime`, `logger`, `BackendType`, `DatabaseError`) become module-level.
  - Docstring: "Package-owned helper for safe-metadata writes (moved from core/Utils/metadata_utils.py, TASK-13950.12)."
- Modify: `tldw_Server_API/app/core/Utils/metadata_utils.py` (delete `:104-208`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/versions.py:46-49` and `tldw_Server_API/app/core/External_Sources/reference_manager_import.py:29-32`. Keep `normalize_safe_metadata` from `metadata_utils`, and import the writer from the new module.
- Modify: `pyproject.toml:1310`. Move the grandfathered `["BLE001"]` key from `.../Utils/metadata_utils.py` to `.../DB_Management/media_db/runtime/safe_metadata_write_ops.py`, in sorted position. It covers the two moved `except Exception` lines; `metadata_utils.py` has no other.
- Modify: `tldw_Server_API/tests/MediaDB2/test_safe_metadata_utils.py:66,82,99` (import path); add 1 test
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`. Its `test_metadata_utils_imports_database_error_from_media_db_errors` (`:2160`) asserts that `inspect.getsource(metadata_utils)` names `core.DB_Management.media_db.errors`, which fails once the writer leaves.
  - Rename the test to `test_safe_metadata_write_ops_imports_database_error_from_media_db_errors`, and make it inspect `safe_metadata_write_ops`. Keep both assertions.
  - Replace the module-level `from tldw_Server_API.app.core.Utils import metadata_utils` (`:95`), which is otherwise unused, with `from tldw_Server_API.app.core.DB_Management.media_db.runtime import safe_metadata_write_ops`.
- Modify: unit test file; `layering_baseline.txt`

**Interfaces:**
- Produces: `core.DB_Management.media_db.runtime.safe_metadata_write_ops.update_version_safe_metadata_in_transaction(db: Any, dv_id: int, safe_metadata_json: str | None, merged_metadata: Mapping[str, Any], connection: Any) -> None`. There is no shim: `metadata_utils` is foundation, and a re-export there would be the same upward import.

- [ ] **Step 1: Add the tests.** Add to the unit file:

```python
def test_utils_no_longer_owns_db_code() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import safe_metadata_write_ops
    from tldw_Server_API.app.core.Utils import Utils, metadata_utils

    assert callable(safe_metadata_write_ops.update_version_safe_metadata_in_transaction)
    assert not hasattr(metadata_utils, "update_version_safe_metadata_in_transaction")
    assert not hasattr(Utils, "get_db_config")
```

  Add to `tests/MediaDB2/test_safe_metadata_utils.py`. This pins the Postgres branch through the move with no live Postgres:

```python
def test_update_version_safe_metadata_uses_on_conflict_upsert_on_postgres():
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.safe_metadata_write_ops import (
        update_version_safe_metadata_in_transaction,
    )

    db = _FakeMetadataDB()
    db.backend_type = BackendType.POSTGRESQL
    update_version_safe_metadata_in_transaction(
        db=db, dv_id=7, safe_metadata_json="{}", merged_metadata={"doi": "10.1000/x"}, connection=object()
    )

    ident_sql = next(query for query, _params, _conn in db.queries if "DocumentVersionIdentifiers" in query)
    assert "ON CONFLICT (dv_id) DO UPDATE" in ident_sql
```

- [ ] **Step 2: Run them.** Expected: FAIL. The unit test fails with `ImportError: cannot import name 'safe_metadata_write_ops' from '...media_db.runtime'`; the MediaDB2 test fails with `ModuleNotFoundError: No module named '...safe_metadata_write_ops'`.
- [ ] **Step 3: Implement** (Files above).
  - Retarget the three existing MediaDB2 tests' imports to the new module, and rewrite the `test_media_db_api_imports.py` test and import.
  - Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff). `test_media_db_api_imports.py` has base `I001` findings, so add its new import at the place `:95` held.
- [ ] **Step 4: Run the tests.** Run pytest on the unit file, `tldw_Server_API/tests/MediaDB2/test_safe_metadata_utils.py`, `tldw_Server_API/tests/MediaDB2/test_safe_metadata_properties.py`, `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`, `tldw_Server_API/tests/MediaDB2/test_safe_metadata_endpoints.py`, `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py`, `tldw_Server_API/tests/External_Sources/test_reference_manager_contract.py` and `tldw_Server_API/tests/External_Sources/test_connectors_worker_reference_sync.py`. Expected: PASS. Then run `git grep -n "get_db_config" -- tldw_Server_API/app/core/Utils`. Expected: no output.
- [ ] **Step 5: Lock the cut.** Line: `core.Utils -> core.DB_Management: 4`.
- [ ] **Step 6: Commit:** `refactor(core): delete dead Utils.get_db_config; safe-metadata writer moves into media_db (TASK-13950.12)`.

### Task 7: email metric families move into `core/Metrics`

**Files:**
- Create: `tldw_Server_API/app/core/Metrics/email_metrics.py`. It holds `_DURATION_BUCKETS` and `iter_email_metric_definitions`, verbatim from `Ingestion_Media_Processing/Email/email_ingestion_metrics.py:18,32-56`. Docstring: "Email ingestion metric families for central registry initialization (moved from Ingestion_Media_Processing/Email/email_ingestion_metrics.py, TASK-13950.12)."
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Email/email_ingestion_metrics.py`. Delete `:18` and `:32-56`, and change `from collections.abc import Callable, Iterator` to `Callable`.
- Modify: `tldw_Server_API/app/core/Metrics/metrics_manager.py:23-25`, which becomes `from .email_metrics import iter_email_metric_definitions`, like `:27`'s `stt_metrics`. `_register_standard_metrics` (frozen) is not edited.
- Modify: unit test file; `layering_baseline.txt`

**Interfaces:**
- Produces: `core.Metrics.email_metrics.iter_email_metric_definitions() -> Iterator[dict[str, Any]]`. There is no shim, because `metrics_manager` was the only importer.

- [ ] **Step 1: Add the test.**

```python
def test_email_metric_families_are_defined_in_metrics() -> None:
    from tldw_Server_API.app.core.Metrics.email_metrics import iter_email_metric_definitions
    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry

    names = {definition["name"] for definition in iter_email_metric_definitions()}
    assert names == {
        "email_ingestion_parse_total", "email_ingestion_parse_seconds",
        "email_ingestion_persist_total", "email_ingestion_persist_seconds",
        "email_native_persist_seconds", "email_ingestion_dedupe_total",
    }
    assert {"email_ingestion_parse_total", "email_ingestion_dedupe_total"} <= set(MetricsRegistry().metrics)
```

- [ ] **Step 2: Run it.** Expected: FAIL, `ModuleNotFoundError: ... core.Metrics.email_metrics`.
- [ ] **Step 3: Implement** (Files above). Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff).
- [ ] **Step 4: Run the tests.** Run pytest on the unit file and `tldw_Server_API/tests/Metrics`. Expected: PASS.
- [ ] **Step 5: Lock the cut.** Line: `core.Metrics -> core.Ingestion_Media_Processing: 1`.
- [ ] **Step 6: Commit:** `refactor(metrics): email metric families move into core.Metrics (TASK-13950.12)`.

### Task 8: Slice A close-out

- [ ] **Step 1: Run the report.** Expected, measured on the dev+`.9` base:
  - `ratcheted now: 253 pairs, 669 statements`, which is 9 pairs and 14 statements fewer than the slice base;
  - `largest package SCC: {'module_level': 100, 'with_function_level': 128}` (reproduced by the dry run; record the measured values);
  - `git diff $(git merge-base origin/dev HEAD) -- tldw_Server_API/tests/lint/layering_baseline.txt` shows only the 9 slice-A deletions.
- [ ] **Step 2: Run the CI-equivalent checks.** Each must show its expected result.
  - "Enforce tenant isolation ratchets": `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 $PY -m pytest -q --disable-warnings -p pytest_asyncio.plugin -p no:cacheprovider --basetemp "$TMPDIR/t1312-ratchets1" tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py`. Expected: PASS.
  - "Enforce CI contracts and code ratchets": the same env plus `BACKLOG_TASK_FORMAT_BASE=origin/dev`, run on every file in that step's `pytest` list in `.github/workflows/backend-required.yml` on the slice base. Today that is `tests/CI/test_license_first_workflow_contracts.py` through `tools/backlog-py/tests`, 16 entries, plus #3331's two lint tests if it has merged.
    - Expected: PASS, except for any size-ratchet failures that already fail identically on the slice base.
    - On dev + #3337 as measured, 3 do: `test_function_size_ratchet` ×2 (for example `chat_service:build_context_and_messages` 821 > 787) and `test_charactersragdb_surface_ratchet` (`delegated_methods` 306 > 302).
    - They are pre-existing drift that #3334's re-seed fixes, not caused by this slice. No frozen function changed size.
    - If they appear, run the same command on the slice base and confirm the same names and messages.
  - "Backend unit smoke": the same env, `-m "unit and not e2e and not jobs" tldw_Server_API/tests/unit`. Expected: PASS.
  - mypy on changed modules: `git diff --name-only --diff-filter=d origin/dev...HEAD -- 'tldw_Server_API/*.py' | xargs uvx mypy --python-executable $PY --follow-imports=silent --ignore-missing-imports`. Expected: no error on a line this slice wrote. Task 3's `dict[str, Any]` keeps the 8 `LlamaCppConfig(**...)` arg-type errors from moving, and its `assert settings is not None` avoids 2 `[index]` errors. Pre-existing errors elsewhere in moved files match the same command on their old paths at the base. (The CI step is continue-on-error.)
  - Ruff: run the ruff check (Conventions), both parts. Expected: part 1 prints nothing, and part 2 prints `All checks passed!`. Base findings in the files that already had them stay; for example, `config.py`'s 59 `UP045`.
    - `git diff --check origin/dev...HEAD`: no output.
  - OpenAPI drift, because `endpoints/media/versions.py` changed an import: `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` (`make openapi-drift-check` with `CI_LOCAL_PYTHON=$PY`). Expected: exit 0 with no drift.
  - `$PY Helper_Scripts/ci/minimal_env_smoke.py --timeout 150`. Expected: exit 0, `/health` 200.
  - Bandit: `git diff --name-only --diff-filter=d origin/dev...HEAD -- 'tldw_Server_API/app/*.py' | xargs uvx bandit -q`. Expected: no finding on new lines.
  - Postgres shard: not applicable (see Global Constraints).
- [ ] **Step 3: Open the PR.**
  - Push and open against `dev`, titled `refactor(core): cut nine foundation-layer upward imports (TASK-13950.12, 1/2)`.
  - In the body:
    - list the 9 deleted baseline lines and the measured report numbers;
    - add an `## ADR assessment` (not required) and a `## Change summary` (owner waiver);
    - add the shim list: `LLM_Calls/provider_identity.py` (27 app + 2 test importers) and `core/stt_observability_context.py` (2 `source_modules` strings).
  - Run the clean-subagent `/superpowers:requesting-code-review` and address findings before merge.
- [ ] **Step 4: Hand off to the orchestrator.** Ask it to append a note on the docs branch, via `tools/backlog-py`, with the measured drift (this plan's "Drift" paragraph) and the slice-A report numbers. AC #3 is already amended, so there is nothing to change there.

---

## Slice B (PR 2): `http_client`

**Before Task 9:** Branch `refactor/13950-12b-http-client-edges` from `origin/dev` after slice A and #3331 have merged. Confirm that the baseline holds `core.http_client -> core.Metrics: 2` and `core.http_client -> core.Security: 6`, and no other foundation line. Confirm that `git grep -n "Security/egress.py" -- tldw_Server_API/tests/lint` shows #3331's two allowlist entries.

### Task 9: the egress policy moves into foundation

**Files:**
- Move: `tldw_Server_API/app/core/Security/egress.py` -> `tldw_Server_API/app/core/Utils/egress.py` (`git mv`, content unchanged; its STT import already points to `core.Logging` since Task 4)
- Create (alias at the old path): `tldw_Server_API/app/core/Security/egress.py`
- Modify: `tldw_Server_API/app/core/http_client.py`. Change `:128` to `from tldw_Server_API.app.core.Utils.egress import ConfiguredEndpointScope  # noqa: E402`. Change the five function-level imports to `tldw_Server_API.app.core.Utils.egress`, keeping them function-level so `monkeypatch` on the module still reaches them:
  - `:1603` `_normalize_dns_pin_host`;
  - `:1775` `_validate_egress_or_raise`;
  - `:1872` `_is_url_allowed`;
  - `:2572` `_instantiate_client`;
  - `:4697` (`public_url_policy_active`), inside the frozen `fetch`.

  Only the module path changes, in place, so no line count changes.
- Modify (#3331's allowlists, renaming the key only and keeping its reason and keys):
  - `tldw_Server_API/tests/lint/test_production_detector_single_source.py:72`: `"tldw_Server_API/app/core/Security/egress.py"` becomes `"tldw_Server_API/app/core/Utils/egress.py"`;
  - `tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py:95`: `"core/Security/egress.py"` becomes `"core/Utils/egress.py"`, moved to its sorted place after `"core/TTS/voice_manager.py"`.
- Modify: `tldw_Server_API/app/core/Security/README.md:19,38` (`egress.py` is an alias of `core/Utils/egress.py`, moved by TASK-13950.12) and `tldw_Server_API/app/core/Utils/README.md` (add the `egress.py` line)
- Modify: unit test file; `layering_baseline.txt`

**Interfaces:**
- Produces: `core.Utils.egress`, the same module as before with every name unchanged (`evaluate_url_policy`, `public_url_policy_active`, `ConfiguredEndpointScope`, `URLPolicyResult`, `is_webhook_url_allowed_for_tenant`, `_normalize_hostname`, `_canonical_ip_literal`, ...).
- Produces: `core.Security.egress`, which is that same module object. The alias body is exactly the following. Its docstring wording matches `.17`'s aliases. The `TYPE_CHECKING` star import keeps the 49 old-path importers type-checkable: without it, mypy reports `attr-defined` on them (the reviewer verified the fix on `Security/url_validation.py`). The layering scan skips `TYPE_CHECKING` bodies.

```python
"""Moved to core/Utils/egress.py (TASK-13950.12).

An alias, not a re-export: the old name *is* the moved module, so old imports and
monkeypatch targets (string or module object) reach the code that runs. core.http_client
is foundation and enforces this policy on every outbound request, so the policy lives in
foundation too. Retire this once no importer or patch target uses the old path.
"""

import sys
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Utils import egress as _egress

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Utils.egress import *  # noqa: F403

sys.modules[__name__] = _egress
```

- [ ] **Step 1: Add the tests.**

```python
def test_old_security_egress_path_is_the_moved_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest import mock

    import tldw_Server_API.app.core.Security.egress as old
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as via_package
    from tldw_Server_API.app.core.Utils import egress as new

    # tests/conftest.py defaults this to "false"; the policy reads it per call.
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    assert old is new and via_package is new
    with mock.patch(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        return_value=SimpleNamespace(allowed=True),
    ):
        assert http_client._is_url_allowed("http://169.254.169.254/") is True
    assert http_client._is_url_allowed("http://169.254.169.254/") is False


def test_http_client_denies_metadata_ip_without_core_security() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import sys\n"
        "from tldw_Server_API.app.core import http_client\n"
        "from tldw_Server_API.app.core.exceptions import EgressPolicyError\n"
        "try:\n"
        "    http_client._validate_egress_or_raise('http://169.254.169.254/latest/meta-data/')\n"
        "except EgressPolicyError:\n"
        "    print('denied')\n"
        "print(sorted(m for m in sys.modules if m.startswith('tldw_Server_API.app.core.Security')))\n"
    )
    out = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "WORKFLOWS_EGRESS_BLOCK_PRIVATE": "true"},
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout.splitlines()
    assert out[-2:] == ["denied", "[]"]


def test_old_egress_path_aliases_when_imported_first() -> None:
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import tldw_Server_API.app.core.Security.egress as old\n"
        "from tldw_Server_API.app.core.Utils import egress as new\n"
        "from tldw_Server_API.app.core.Security import egress as via_package\n"
        "assert old is new and via_package is new\n"
    )
    result = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
```

  The in-process test runs after `conftest` has imported the new path. This subprocess imports the old path first, in a fresh interpreter, matching `.17`'s `test_old_paths_alias_when_imported_first`.

- [ ] **Step 2: Run them.** Expected: FAIL.
  - The first fails with `ImportError: cannot import name 'egress' from 'tldw_Server_API.app.core.Utils'`.
  - The second fails on `out[-1]`, which lists `tldw_Server_API.app.core.Security` and `...Security.egress` (today `http_client` imports `core.Security` at module level).
  - The third fails with a non-zero return code, its stderr ending in `ImportError: cannot import name 'egress' from 'tldw_Server_API.app.core.Utils'`.
- [ ] **Step 3: Implement** (Files above). Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff).
- [ ] **Step 4: Run every test that names the moved module.** This applies "flipping a global default": the module identity changed.

```bash
git grep -lE "Security\.egress|Security import egress|Utils\.egress|Utils import egress" -- tldw_Server_API/tests | xargs env PYTHONPATH=. $PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t1312-egress" -q tldw_Server_API/tests/unit/test_foundation_layer_moves.py tldw_Server_API/tests/http_client tldw_Server_API/tests/lint/test_production_detector_single_source.py tldw_Server_API/tests/lint/test_test_seam_env_reads_guarded.py
```

  Expected: PASS, about 70 files; the dry run got 2464 passed, 95 skipped.
  - Run this serially, as written; do not add `-n`. `tests/Admin_Webhooks/test_recovery_backend_matrix.py` is in the list, and under xdist its fixture drops the shared `tldw_test` database (Global Constraints, Postgres proof). Without Postgres it skips.
  - Task 11 Step 2 runs it against Postgres.
- [ ] **Step 5: Lock the cut.** Line: `core.http_client -> core.Security: 6`.
- [ ] **Step 6: Commit:** `refactor(core): the egress policy moves into foundation; Security.egress aliases it (TASK-13950.12)`.

### Task 10: `http_client` gets metrics and tracing by injection; pin foundation

**Files:**
- Modify: `tldw_Server_API/app/core/http_client.py`. Delete `:122-127` (the `core.Metrics` and `Metrics.traces` imports). Replace `:685-739` (`_HTTP_CLIENT_METRICS_REGISTERED`, `_register_http_client_metrics_once` and the import-time call) with the hook block in Interfaces. Call sites are unchanged.
- Create: `tldw_Server_API/app/core/Metrics/http_client_metrics.py`
- Modify: `tldw_Server_API/app/core/Metrics/__init__.py`. Append after `__version__`: a two-line comment (`core.http_client` is foundation and cannot import this package; wire it here, TASK-13950.12), `from .http_client_metrics import install_http_client_observability  # noqa: E402`, and `install_http_client_observability()`.
- Modify: `tldw_Server_API/tests/http_client/test_http_client.py:1979-2016`. Retarget `test_metric_registration_suppresses_noncritical_failure` and `test_metric_registration_propagates_unexpected_failure` to call `http_client_metrics.register_http_client_metrics(types.SimpleNamespace(register_metric=fail_registration))`. Drop their `_HTTP_CLIENT_METRICS_REGISTERED` patch and assert, because the flag is deleted. The suppression case asserts that nothing raises; the propagation case still expects `UnexpectedMetricError`.
- Modify: `tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py`.
  - In `test_telemetry_import_fallback_with_missing_opentelemetry`'s script, add `importlib.import_module("tldw_Server_API.app.core.http_client")  # fastapi requires opentelemetry-api` right after `import importlib` and before `original_import = builtins.__import__`.
  - **Why:** after this task, `core/Metrics/__init__.py` imports `http_client`, whose chain (`http_client` → `core/exceptions.py:12` → `fastapi` → `fastapi/telemetry/_api.py:9`) imports `opentelemetry`. So the test's blocked import would fail before it reaches `Metrics.telemetry`.
  - **Not a production regression.** fastapi 0.142.1 hard-requires `opentelemetry-api>=1.44.0`, so no environment with fastapi lacks opentelemetry.
  - **Verified by the reviewer:** with the line added, the test passes and still asserts `OTEL_AVAILABLE` is `False` and the dummy tracer and meter.
- Modify: `tldw_Server_API/tests/lint/test_layering_contract.py`:
  - Add the two pins, their self-tests, and `import json, sys`.
  - Add one docstring sentence: since TASK-13950.12, foundation and `core.Metrics` have no baseline entries, and these tests pin them at zero, statically and at import time, with no `# bump:` exception.
  - Siblings edit the same file. If `.16` has landed, it has hoisted `_scan`'s module-name walk into `module_names()`; build `foundation_modules()` by filtering that with `layer_of(package_of(name)) == 0` instead of walking the tree again. `.17` appends `SIBLING_BANS` entries and a self-test at the end of the file; on a textual conflict, keep both sides.
- Modify: unit test file; `layering_baseline.txt`

**Interfaces:**
- `core.http_client` (new, replacing the deleted block):
  - `class _NoObservability` has seven no-ops:
    - `increment(*args, **kwargs) -> None`, `observe(*args, **kwargs) -> None`, `set_attributes(*args, **kwargs) -> None` and `add_event(*args, **kwargs) -> None`;
    - `get_baggage(key: str) -> None`;
    - `span(*args, **kwargs)` and `async_span(*args, **kwargs)`, each returning `contextlib.nullcontext()`, which supports both `with` and `async with` on Python 3.10+. `nullcontext` is already imported at `http_client.py:38`.

    These seven are every method `http_client` calls on the registry and the tracing manager. Every request opens a span first: `_http_client_observability_span` (`:380`, `:385`) calls `span`, and `_async_http_client_observability_span` (`:408`, `:413`) calls `async_span`. Without those two, every `fetch`/`afetch` in a process that never imported `core.Metrics` raises `AttributeError` (reproduced by the reviewer). `_NO_OBSERVABILITY = _NoObservability()`.
  - `_metrics_registry_getter: Callable[[], Any] | None = None` and `_tracing_manager_getter: Callable[[], Any] | None = None`.
  - `install_observability(*, metrics_registry: Callable[[], Any], tracing_manager: Callable[[], Any]) -> None` sets both getters.
  - `get_metrics_registry() -> Any` and `get_tracing_manager() -> Any` return `getter()` or `_NO_OBSERVABILITY`. They keep today's module-global names, so the frozen fetch functions and the existing patch `monkeypatch.setattr(hc, "get_tracing_manager", ...)` (`tests/Web_Scraping/test_http_client_fetch.py:138`) are untouched.
- `core.Metrics.http_client_metrics` (new):
  - `register_http_client_metrics(registry: MetricsRegistry) -> None` registers the four definitions moved verbatim from `http_client.py:697-734`: `http_client_requests_total`, `http_client_request_duration_seconds`, `http_client_retries_total` and `http_client_egress_denials_total`. Each registration is wrapped in `suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS)`, where that tuple is `(AttributeError, OSError, RuntimeError, TypeError, ValueError)`.
  - `install_http_client_observability() -> None` works in three steps:
    1. It calls `http_client.install_observability(metrics_registry=get_metrics_registry, tracing_manager=get_tracing_manager)`, passing the `metrics_manager` and `traces` functions themselves (by their module-global names in `http_client_metrics`), so a replaced global registry is still read at call time.
    2. It runs `registry = None` and then `with suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS): registry = get_metrics_registry()`.
    3. It calls `register_http_client_metrics(registry)` only if `registry is not None`.

    Hooks go first and registry failures are suppressed, so a broken registry can never crash `import core.Metrics` or leave the hooks uninstalled. This keeps today's behaviour at `http_client.py:692-695`.
- `test_layering_contract.py` (new):
  - `FOUNDATION_ONLY_IMPORTERS = frozenset({"core.Metrics"})`;
  - `foundation_breaches(records: Sequence[Import]) -> list[str]` returns the sorted, de-duplicated `"<importer> -> <target> (<module>)"` for every record whose importer package is foundation or in `FOUNDATION_ONLY_IMPORTERS` and whose target is above foundation;
  - `foundation_modules(root: Path = APP_ROOT) -> list[str]` returns the sorted app-relative names of every foundation module, skipping in-app `tests` packages like `_scan` does;
  - `upper_layer_loads(modules: Sequence[str]) -> dict[str, list[str]]` runs one `subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=300)`.
    - The script imports each `APP + "." + module` with `importlib` inside `try`/`except Exception`, recording each failure as `"<module>: <ExcType>: <message>"`.
    - It prints `json.dumps({"failed": failed, "loaded": sorted(sys.modules)})` as its last line.
    - The function returns `{"failed": <those strings>, "upper": <sorted packages above foundation among the loaded app modules>}`, ignoring `app` and `core`, the parent packages every import runs.
    - A foundation module that cannot import in the CI image (say, a missing optional dependency) therefore shows up by name, not as an opaque `CalledProcessError`.

- [ ] **Step 1: Add the pins to `test_layering_contract.py`, after the existing self-tests.**

```python
def test_a_seeded_foundation_or_metrics_breach_is_reported_even_inside_a_function(tmp_path: Path) -> None:
    files = {
        "core/Utils/x.py": f"def f():\n    from {APP}.core.Security import egress\n",
        "core/Metrics/y.py": f"from {APP}.core.Security import egress\n",
        "core/Metrics/z.py": f"from {APP}.core.Utils import x\n",
    }
    assert foundation_breaches(scan(_seed(tmp_path / "app", files))) == [
        "core.Metrics -> core.Security (core.Metrics.y)",
        "core.Utils -> core.Security (core.Utils.x)",
    ]


def test_foundation_and_metrics_import_only_foundation() -> None:
    breaches = foundation_breaches(scan())
    assert not breaches, (
        "Foundation packages and core.Metrics import only foundation (TASK-13950.12); no `# bump:` "
        "admits one. Move the shared code down or inject it from above:\n" + "\n".join(breaches)
    )


def test_upper_layer_loads_reports_upper_packages_and_failed_imports() -> None:
    assert upper_layer_loads(["core.Context_Integrity.canonicalization", "core.no_such_module"]) == {
        "failed": [f"core.no_such_module: ModuleNotFoundError: No module named '{APP}.core.no_such_module'"],
        "upper": ["core.Context_Integrity"],
    }


def test_foundation_and_metrics_load_only_foundation_at_import_time() -> None:
    assert upper_layer_loads(foundation_modules()) == {"failed": [], "upper": []}
    assert upper_layer_loads(["core.Metrics"]) == {"failed": [], "upper": ["core.Metrics"]}
```

  Add to the unit file:

```python
def test_http_client_observability_is_noop_until_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tldw_Server_API.app.core import http_client

    monkeypatch.setattr(http_client, "_metrics_registry_getter", None)
    monkeypatch.setattr(http_client, "_tracing_manager_getter", None)
    assert http_client.get_metrics_registry().increment("http_client_requests_total", 1, labels={}) is None
    assert "X-Request-Id" not in http_client._inject_trace_headers({})
    client = http_client.create_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    try:  # the request path opens a span on the no-op tracing manager
        assert http_client.fetch(method="GET", url="http://93.184.216.34/ok", client=client).status_code == 200
    finally:
        client.close()


def test_a_process_without_core_metrics_still_fetches() -> None:
    import subprocess
    import sys
    from pathlib import Path

    # tests/conftest.py imports core.Metrics, so only a fresh interpreter has no installed hooks.
    script = (
        "import asyncio, sys\n"
        "import httpx\n"
        "from tldw_Server_API.app.core import http_client\n"
        "def handler(request):\n"
        "    return httpx.Response(200, request=request, text='ok')\n"
        "client = http_client.create_client(transport=httpx.MockTransport(handler))\n"
        "print(http_client.fetch(method='GET', url='http://93.184.216.34/ok', client=client).status_code)\n"
        "async def main():\n"
        "    aclient = http_client.create_async_client(transport=httpx.MockTransport(handler))\n"
        "    try:\n"
        "        return (await http_client.afetch(method='GET', url='http://93.184.216.34/ok', client=aclient)).status_code\n"
        "    finally:\n"
        "        await aclient.aclose()\n"
        "print(asyncio.run(main()))\n"
        "print('tldw_Server_API.app.core.Metrics' in sys.modules)\n"
    )
    result = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[-3:] == ["200", "200", "False"]


def test_core_metrics_installs_http_client_observability() -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Metrics import get_metrics_registry, get_tracing_manager

    assert http_client._metrics_registry_getter is get_metrics_registry
    assert http_client._tracing_manager_getter is get_tracing_manager


def test_installing_observability_survives_a_failing_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Metrics import http_client_metrics, traces

    def broken_registry() -> None:
        raise RuntimeError("metrics registry unavailable")

    monkeypatch.setattr(http_client_metrics, "get_metrics_registry", broken_registry)
    monkeypatch.setattr(http_client, "_metrics_registry_getter", None)
    monkeypatch.setattr(http_client, "_tracing_manager_getter", None)
    http_client_metrics.install_http_client_observability()  # must not raise
    assert http_client._metrics_registry_getter is broken_registry
    assert http_client._tracing_manager_getter is traces.get_tracing_manager


def test_installed_tracing_manager_feeds_x_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import http_client

    class _Tracing:
        def get_baggage(self, key: str) -> str | None:
            return "rid-1" if key == "request_id" else None

    monkeypatch.setattr(http_client, "_tracing_manager_getter", _Tracing)
    assert http_client._inject_trace_headers({})["X-Request-Id"] == "rid-1"


def test_http_client_metric_families_register_on_a_fresh_registry() -> None:
    from tldw_Server_API.app.core.Metrics.http_client_metrics import register_http_client_metrics
    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry

    registry = MetricsRegistry()
    register_http_client_metrics(registry)
    assert {
        "http_client_requests_total", "http_client_request_duration_seconds",
        "http_client_retries_total", "http_client_egress_denials_total",
    } <= set(registry.metrics)
```

- [ ] **Step 2: Run them.** Run the layering check and pytest on the unit file. Expected: FAIL.
  - `test_foundation_and_metrics_import_only_foundation` lists exactly `core.http_client -> core.Metrics (core.http_client)`.
  - `test_foundation_and_metrics_load_only_foundation_at_import_time` gets `{"failed": [], "upper": ["core.Metrics"]}` for the foundation import. The dry run measured exactly this after Task 9: all 51 foundation modules import, and only `core.Metrics` is above foundation.
  - In the unit file:
    - the `monkeypatch` tests fail with `AttributeError ... _metrics_registry_getter`;
    - the registry tests fail with `ModuleNotFoundError ... http_client_metrics`;
    - `test_a_process_without_core_metrics_still_fetches` fails on its last line, which prints `True` (today `http_client` imports `core.Metrics`).
  - Both self-tests already PASS: the seeded breach is reported, and so are a domain package and a missing module.
- [ ] **Step 3: Implement** (Files and Interfaces above), including the one-line pre-import in `test_telemetry_import_fallback.py`. Place each edited or added import in its sorted position by hand; never run `ruff --fix` (Global Constraints, Ruff).
- [ ] **Step 4: Run every test that names the changed symbols.** This applies "flipping a global default": registration moved from `http_client` import to `core.Metrics` import.

```bash
git grep -lE "core\.http_client|core import http_client|core\.Metrics|core import Metrics" -- 'tldw_Server_API/tests/*test_*.py' | xargs env PYTHONPATH=. $PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t1312-metrics" -q tldw_Server_API/tests/unit/test_foundation_layer_moves.py tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py
```

  - The pathspec matters. Without it, the grep also matches `tests/lint/function_size_baseline.json`, `layering_baseline.txt`, `AuthNZ/conftest.py` and `_plugins/http_client_patch_guard.py`, and pytest stops with `ERROR: not found` and runs nothing.
  - With it, 186 test modules on dev + `.9`. `tests/Metrics/test_telemetry_import_fallback.py` is named explicitly as well: without this task's one-line fix it fails, because fastapi imports opentelemetry.
  - Expected: the same failures as the base run with the same command and the same mode (serial or xdist). Run it on the base first and save the failing node ids, then run it on the slice and diff the two sets: the slice may add no failing id. On dev + `.9` the base fails `Streaming/test_streams.py` ×3 and `test_ocr_runtime_managed.py` ×2, and, run serially, 9 further `tests/Metrics` tests; these are pre-existing.
  - The written-lines checker sees only lines in the diff. It cannot see a finding that a change causes on an unchanged line (for example an import block re-sorted by a moved line), which is why the `--select I` pass on files clean at the base runs as well.
- [ ] **Step 5: Lock the cut.** Line: `core.http_client -> core.Metrics: 2`. Then run the layering check. Expected: PASS, including both pins.
- [ ] **Step 6: Commit:** `refactor(core): http_client takes metrics and tracing by injection; pin foundation imports (TASK-13950.12)`.

### Task 11: Slice B close-out

- [ ] **Step 1: Run the report.** Expected, measured on the dev+`.9` base:
  - `ratcheted now: 251 pairs, 661 statements`, which is 2 pairs and 8 statements fewer than the slice-B base;
  - `largest package SCC: {'module_level': 89, 'with_function_level': 117}`. The dry run measured this after Task 9; Task 10 only removes an edge. AC #4 needs module-level `<= 89`. If the measured value is higher, list the extra edges from `package_scc_sizes` and stop: do not cut further here.
  - `grep -cE "^core\.(config|exception|Utils|Logging|testing|http_client|feature_flags|custom_openai_providers)[^ ]* -> " tldw_Server_API/tests/lint/layering_baseline.txt` prints `0`.
- [ ] **Step 2: Run the CI-equivalent checks.**
  - Repeat Task 8 Step 2, with the same expectations, including the note on pre-existing size-ratchet failures: both ratchet steps (now including #3331's two lint tests), unit smoke, mypy, ruff and `git diff --check`, minimal-env smoke and bandit.
  - Run `$PY -m pytest ... tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k layering`. This confirms AC #5: `test_backend_required_enforces_layering_ratchets` passes for `test_layering_contract.py`.
  - OpenAPI drift is not needed, since no endpoint or schema changes.
  - **Postgres, serially, on the base and on slice B.** The one Postgres file that imports egress runs once on a scratch worktree of the slice-B base and once on the slice-B branch: `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. $PY -m pytest -p no:xdist -p no:cacheprovider --basetemp "$TMPDIR/t1312-pg-<base|b>" -q tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py`.
    - `tests/AuthNZ/conftest.py` starts a Docker Postgres unless `TLDW_TEST_NO_DOCKER=1`; or set `TEST_DATABASE_URL`.
    - The controller's ruling names `tests/DB_Management/`, but the file is in `tests/Admin_Webhooks/`.
    - Expected: the same result on both trees, with none skipped (`TLDW_TEST_POSTGRES_REQUIRED=1` turns a skip into a failure). Parallel runs are meaningless (Global Constraints, Postgres proof).
- [ ] **Step 3: Open the PR.**
  - Push and open against `dev`, titled `refactor(core): http_client stops importing Metrics and Security; foundation pinned (TASK-13950.12, 2/2)`.
  - In the body: the 2 deleted lines, the report numbers, both SCC measures, the egress-alias rationale (Review Focus 1-3), `## ADR assessment` and `## Change summary`.
  - Add one line for log consumers: egress log records are now named `tldw_Server_API.app.core.Utils.egress`, not `...Security.egress`. No in-repo filter depends on the old name, but external log queries might.
  - Note the Postgres serial-run results from Step 2.
  - Run the clean-subagent review before merge.
- [ ] **Step 4: Hand off to the orchestrator.**
  - Check ACs #1-#5 as amended.
  - Record both SCC measures and the baseline size in TASK-13950's notes, as the roadmap's Phase 1 exit check asks.
  - File the follow-up below.

## Follow-ups (for the orchestrator to file)

- **Retire the shims.**
  - `core/LLM_Calls/provider_identity.py`: 27 app files and 2 test files import it; no patch targets.
  - `core/Security/egress.py`: 49 app modules import it. There are 11 string patch targets in 7 test files, and 98 module-object `setattr`s in 37 of the 68 test files that import it.
  - `core/stt_observability_context.py`: no importers after Task 4, but the four `stt_provider_adapter.py` `source_modules` strings name it, and so does `...Security.egress`. Renaming them changes STT plan descriptors and needs the `tests/Audio` pins updated.

  Retarget the importers in conflict-light batches, after #3187/#3191 (which touch `byok_runtime.py` and `chat_service.py`) merge, then delete the shims.

## Fix round (2026-10-26): review and dry-run findings

Self-review after applying both reports and the controller's rulings.

- **Review C1** (`_NoObservability` lacked `span`/`async_span`): fixed. The class is in Task 10 Interfaces. `test_a_process_without_core_metrics_still_fetches` runs `fetch` and `afetch` in a fresh interpreter, and the in-process no-op test now issues a `fetch` too. A prototype in a scratch tree (`http_client` with the hook block, no `core.Metrics` import) printed `200`, `200`, `False`.
- **Review I1** (no `# bump:` for foundation or `core.Metrics`): now stated in Global Constraints under "Raising a baseline", binding `.13`, `.14` and `.17`.
- **Review I2** (the installer must not crash `import core.Metrics`): hooks are installed first and registry failures are suppressed. Pinned by `test_installing_observability_survives_a_failing_registry`.
- **Dry-run I1** (the codemod rewrote Task 5's own test): the unit file is excluded from the pathspec, and Step 3.4 expects exactly that one remaining line.
- **Dry-run I2** (`test_media_db_api_imports.py`): retargeted in Task 6. The test at `:2160` now inspects `safe_metadata_write_ops`, and the import at `:95` is replaced.
- **Rulings 1 and 7-8:**
  - The AC #3 amendment step is gone; the AC was amended on the docs branch with numbering unchanged.
  - Task 8 Step 2 now expects the 3 pre-existing size-ratchet failures, no ruff finding on a written line (see fix round 2), and no new mypy errors (`dict[str, Any]` plus `assert settings is not None`).
  - Task 11 Step 2 adds the serial Postgres run, on base and slice B, of `tests/Admin_Webhooks/test_recovery_backend_matrix.py`. That is the file's real path; the ruling named `tests/DB_Management/`.
- **Minors:** all folded in, none declined.
  - Review M1 (`TYPE_CHECKING` star import in the alias), M2 (old-path-first subprocess test and the `.17` docstring wording), M3 (open PRs, port recipe, merge order), M4 (step references), M5 (`upper_layer_loads` reports failed imports by name), M6 (`.16`/`.17` edits to `test_layering_contract.py`), M7 (log-name line in the PR body).
  - Dry-run m3-m10 (pre-existing ratchet failures, ruff, mypy, the `scope_predicate_baseline` claim now naming `unified_pipeline.py:3889`, expected-failure wording in Tasks 3 and 6, `Logging/README.md:36-37`, 98 `setattr`s in 37 files).
- **Re-checked:**
  - On `origin/dev` `06a2ceb3cb`, no touched file changed since `35e711080a`.
  - `nullcontext` is imported at `http_client.py:38`.
  - The dict form of `upper_layer_loads` returns exactly the self-test's expected value (prototyped).
  - Type consistency: every test name in Review Focus matches its task.

## Fix round 2 (2026-10-26): scoped re-review

The re-review confirmed all 14 earlier findings were addressed and raised three new Important ones. All three are applied; none is declined.

- **N1** (Task 10 broke `tests/Metrics/test_telemetry_import_fallback.py`): fixed. Task 10's Files add the one-line `http_client` pre-import (reviewer-verified), and Step 4 names the file explicitly.
- **N2** (Task 10 Step 4 ran no tests): fixed. The grep uses the pathspec `'tldw_Server_API/tests/*test_*.py'` (186 modules). The Task 1 and Task 9 greps match only test modules today (19 and 68), so they stay as they are.
- **N3** (false ruff premise):
  - Global Constraints, Ruff, now lists the base findings: `config.py`'s 59 `UP045`, and `I001` in six app files plus two test files.
  - All per-task `ruff --fix` steps are gone. Edited imports are placed by hand; `ruff check --select I --diff` may be used to locate them.
  - Task 3 writes `dict[str, Any] | None`.
  - Tasks 8 and 11 use the ruff check (Conventions) in two parts: a script that prints findings on written lines (tested in a scratch tree, with `-M -C` so shims and aliases count only changed lines), and `--select I` on the changed files that were clean at the base.
