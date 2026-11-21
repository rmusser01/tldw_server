# Refuse an admin-disabled provider or model on every key and path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended for this size) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A provider the admin explicitly disables (the override's `is_enabled = false`), or a model left off the override's `allowed_models`, is refused on every key (the user's own, a team or org key, the server's) and on every path. PR #3523's server-key fallback goes away. A provider the server merely has no key or credits for keeps working on a user's own key.

**Architecture:** One guard where every provider adapter call already passes: the accounted `ChatProvider` base class (ADR-067). Its four wrappers in `core/LLM_Calls/usage_accounting.py` now refuse an admin-disabled provider or model before the quota check, on any key, with a `ChatConfigurationError` that every adapter caller already handles; the app answers such a refusal with 403 (or 503 for an unreadable store) where no endpoint maps it, through a `ChatConfigurationError` handler, and RAG, evaluations and MCP keep its code. Four paths keep an explicit check, because they need their own shape or never reach an adapter:
- `_summarize_via_adapter`, for the analyze path's `configuration` failure before a stream is handed back;
- the MCP `model_completion` authoritative branch of `ProviderCredentialRuntime._resolve_entry`, whose bounded transport is not an adapter;
- Slides standalone-HTML generation, which sends its own HTTP request;
- `POST /api/v1/llamacpp/inference`, open to any authenticated user, which posts to the managed llama.cpp server itself.

All of them call one new helper, `enforce_provider_override(provider, model)`. Because the guard reads the override store on every call, every process that can make an LLM call must load the store: the app does at startup, and a new shared bootstrap (`standalone_llm_provider_overrides`, with a sync variant for CLIs) loads it in each standalone worker and evaluation CLI. #3523's fallback (`server_key_fallback`, `log_server_key_fallback`, `override_outcome`, the quota gate's `strict=` parameters, the ingest worker's `answered` marking, the judges' fallback branches) is deleted; `llm_quota_gate.py`, `eval_runner.py` and `ms_g_eval.py` return byte-for-byte to their state before #3523, and `media_ingest_jobs_worker.py` differs from it only by the bootstrap's `main()`.

**Tech Stack:** Python 3, FastAPI, click, pytest (pytest-asyncio auto mode, xdist), loguru, backlog-py.

**Spec:** the owner's rule (2026-11-19, below), the controller's planning brief, its rulings on three independent plan reviews (recorded under Rulings below), and the task files TASK-15571, TASK-15576, TASK-15577 on `origin/dev` and TASK-15581, TASK-15582, TASK-15583 on the branch (commit `3b006a1340`).

**Base:** `origin/dev` at `f8f141b82e` (#3521 after `93f5aaa684`; it touches only the RAG contextual-retrieval defaults, `config.py`, `Env_Vars.md` and two lint baselines, none of them a file this plan changes). Branch `fix/admin-disabled-provider-refused-15581`, already pushed with the three task files as draft PR #3537. Measured in a worktree detached at `f8f141b82e` (rounds 0 and 1 at `93f5aaa684`, as marked), with `origin/dev` comparisons on a `git archive` export of the same commit (fold round 3's REDs ran in place instead: the tree reset to `origin/dev`, only the new test hunks applied, then the full diff restored and checked byte-identical).

## The owner's rule (2026-11-19)

Verbatim: "if a user tries to use a model/service thats been explicitly disabled by the admin, they shouldn't be able to use it; if its just been disabled as in 'no credits/key enabled' then tehy should be able to use it via BYOK".

An explicit disable is the provider override's `is_enabled=False` (`provider_disabled`), or a model not on its `allowed_models` when that list is set (`model_not_allowed`). Either one is refused on every key and every path. A provider the server has no key or credits for is not a disable. This reverses TASK-15571's rule, which #3523 shipped.

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`). Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. TASK-15581, TASK-15582 and TASK-15583 are already filed; create no other task.
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server. Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, `Env_Vars.md` and the other files under `Docs/Published`) needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR.
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required. Regenerate it with the script's `--write-baseline`, in the same PR. (Measured: no file this plan touches is listed.)
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. Measure each touched function before and after. (Measured: no function this plan changes is in `function_size_baseline.json`, and all stay under its 300-line threshold. `process_standalone_html_generation_job` is capped at exactly its size, 425 lines, so I1's check lives in the provider module instead. See Task 5.)
- A plan with embedded diffs fails trailing-whitespace when it is committed: keep the plan free of trailing spaces.
- Task-specific:
  - `$PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Every command that runs Python uses `$PY` with `PATH="$(dirname $PY):$PATH"`; bare `python` is not on this machine's PATH. Pytest also needs `PYTHONPATH=<plans dir>:<plans dir>/pyext` (pytest-mock). The machine is shared: `-n 2` at most, targeted files.
  - `PYTEST` below abbreviates `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=<scratchpad>/<step>.net $PY -m pytest -p netprobe_15427 -q -n 2`, and `BACKLOG` abbreviates `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd .`.
  - The ADR-004 Change summary waiver is answered "yes" for this PR. The PR body records it as a full `## Change summary` section: "ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-19."
  - Tests that replace a provider method (`Adapter.chat = ...`, or a non-`ChatProvider` stand-in from `get_adapter`) bypass the base-class guard (ADR-067, "Tests that replace a provider method"). A test of the guard itself uses a `ChatProvider` subclass (`_Provider` in `test_llm_quota_byok_calls.py`).
  - The guard fails closed when the store is unloaded, and every test process gets a healthy, empty store from the root conftest (and the MCP conftest). So a test never shows a standalone process's state: Task 1's tests exercise each entry point's `main` instead.

## Review Focus

1. **One guard, every path.** `_refuse_admin_disabled` runs first in each of the four `usage_accounting` wrappers, so every adapter call (gateway or direct: analyze, judges, Prompt Studio, RAG, claims, workflows and the other callers that run on the server key with no check of their own) meets the rule, whatever the key. The independent review audited all 124 registered adapters: each is a `ChatProvider`, carries the wrappers, and its `name` canonicalizes to its override key. Exempt calls (key tests) are not refused: D2.
2. **Every process loads the store.** Before the guard lands (Task 1 precedes Task 2), each standalone entry point that can reach an adapter loads the override store at start and keeps it fresh; the list, with what is excluded and why, is under B1. A CLI loads it only for its LLM commands, never for `--help` or shell completion, and stops with a message naming `DATABASE_URL` when it cannot (fold round 2, M1).
3. **Error shape.** The guard raises `ChatConfigurationError` with `error_code` `provider_disabled`, `model_not_allowed` or `credential_store_unavailable` (three codes added to its allowlist), status 500 like every `ChatConfigurationError`. Every existing mapper honours it unchanged: chat's `_provider_credential_error_code` and `normalize_provider_stream_error` give 403/503 with the shared messages; `_summary_error_code` gives `configuration`, or the retryable `provider_failure` for an unreadable store (fold 3, I2); G-Eval's retry predicate excludes `ChatConfigurationError`; the privacy-safe rebuild keeps the code. RAG (`classify_rag_provider_error`), the evaluations endpoints (`_eval_provider_http_exception`) and MCP `model_completion` (its failure code) keep the reason, 403 or 503, where they used to downgrade it to `provider_configuration_invalid` or `invalid_provider_credentials` (fold 3, I1). Prompt improvement still maps it to `provider_not_configured`, as it maps every `ChatConfigurationError`: left as it is. Where no endpoint maps it, the app's `ChatConfigurationError` handler answers 403 or 503 (M7, fold 3 P1). One table, `ADMIN_REFUSAL_STATUS` in `provider_results.py`, gives the status everywhere. The guard checks each OpenRouter fallback model too (fold 3, I3). Chat's own 403 (`validate_provider_override`, chat.py:4469/5295) still fires first.
4. **Exact reverts.** `llm_quota_gate.py`, `eval_runner.py` and `ms_g_eval.py` equal `b1a32fbba4^1`; `media_ingest_jobs_worker.py` differs from it by the B1 `main()` only (9 lines in, 2 out). Non-strict quota-gate behaviour is therefore exactly as before #3523.
5. **MCP order.** The authoritative branch still validates the scope first (a revoked scope fails as `credential_scope_revoked`), then refuses on whichever key the scope chose, server key included, before any reservation.
6. **BYOK with no server key.** The guard reads only `is_enabled` and `allowed_models`. A provider with no override, or one that leaves it on with no key of its own, is not refused; an own-key call to it runs and is not checked against the limit (ADR-083). A call that names no model is checked for `provider_disabled` only (M1).
7. **The llama.cpp inference route.** `POST /api/v1/llamacpp/inference` is open to any authenticated user and posts to the managed server itself, so no adapter guard sees it. It calls `enforce_provider_override("llama.cpp", None)` before its `try`, because the route's `except Exception` would turn the refusal into a 500. Every spelling of the provider (`llamacpp`, `llama-cpp`, `llama_cpp`, `llama`) folds to `llama.cpp` in the store, so one call covers them all.

## Rulings (fold round 1, 2026-11-19)

The independent plan review returned "Execute after fixes": 1 Blocker, 1 Important, 9 Minor, 1 owner question. The controller's rulings, and how each is folded:

| Ruling | Folded as |
|---|---|
| **B1** (Blocker): standalone processes never load the store, so the guard refuses every LLM call there | New Task 1, before the guard: `standalone_llm_provider_overrides()` (async context manager, the Research worker's pattern) and `standalone_llm_provider_overrides_sync()` (its loop on a daemon thread, for click CLIs), called from every standalone entry point that can reach an adapter (list below). One parametrized test per kind (workers, CLIs) starts from an unloaded store over an empty AuthNZ database and asserts the store reads inside the work (workers) or after an LLM command, and not after another command (CLIs). New AC on TASK-15581. One sentence each in `Usage_Quotas.md` and the sidecar-workers guide. The guard stays fail-closed |
| **I1** (Important): Slides standalone-HTML generation calls OpenAI/Anthropic itself and never reads the override | Task 3: `generate_standalone_html` calls `enforce_provider_override(target.provider, target.model)` after its target check and before any request. A disable fails with `standalone_html_provider_disabled` / `standalone_html_model_not_allowed`, which the job worker terminalizes (not retried); an unreadable store fails with `standalone_html_provider_unavailable`, which it retries (`standalone_html_generation_jobs_worker.py:916-924`). The check lives in the provider module because `process_standalone_html_generation_job` sits exactly at its size cap. The ADR-086 sentence now names the non-adapter paths and what is out of scope |
| **M1**: the no-model case is unstated | Stated in Review Focus 6, in `Usage_Quotas.md`, and pinned by `test_a_call_that_names_no_model_is_checked_for_provider_disabled_only`. `require_model` is not set at the base class (it would refuse local providers). The known gap: Google, DeepSeek, Mistral, Qwen, HuggingFace, Cohere and Moonshot fill a default model after the guard; no caller sends them no model today |
| **M2**: bare `python` in Task 0 | Every command uses `$PY` with the `PATH=` prefix (`BACKLOG`, `PYTEST`) |
| **M3**: wrong expected backlog-py output | backlog-py prints `TASK-NNNNN - <title> (<status>)` for an edit; Task 0 expects that |
| **M4**: wrong Task 2 RED reason | Corrected: on `origin/dev`, sync `chat` and `stream` run on every key (`DID NOT RAISE`); only `achat` and `astream` on a non-own key get the 402, because the quota check cannot run on the accounting loop's own thread |
| **M5**: Task 4's llm-providers row leaned on the patch | The row's text, code spans included, is written into Task 4 Step 4 |
| **M6**: Bandit missed the reverted files | Task 5 Step 4's Bandit command lists every touched source file: 19 in fold round 1, 22 after fold round 2 |
| **M7**: an endpoint that maps nothing answers 500 | Fixed at the root: the global handler (`api/v1/utils/exception_handlers.py`) answers a `ChatConfigurationError` whose `error_code` is one of the three refusal codes with 403 (`provider_disabled`, `model_not_allowed`) or 503 (`credential_store_unavailable`). Not a subclass or an instance status: that handler returns a fixed 500 for every exception, and a 403 `status_code` would be misread as an auth failure by two existing mappers that test the status before the type (`evaluations_unified.py:250-253`, `error_utils.privacy_safe_chat_error`). One test drives an endpoint with no mapping. `quizzes.py:842` (400 "No usable chat provider and model are configured") is left: telling an admin refusal apart there is not a one-liner, and the study assistants refuse earlier, at `resolve_chat_target` |
| **M8**: ADR-067 not amended | ADR-067 gains an amendment: its wrappers also enforce the admin policy, pointing to ADR-086; both ADRs are mirrored |
| **M9**: renumbering and D3 | Done throughout: TASK-15581/15582/15583, branch `fix/admin-disabled-provider-refused-15581`, PR #3537; the old explicit-disallow listing note is dropped (it lives in TASK-15583); the final summary names TASK-15583. The deep-copy cost is recorded under Known limits |
| **Q1**: does a store outage stop every LLM call? | The owner's call: Decision Q1. The plan keeps fail-closed |

## Rulings (fold round 2, 2026-11-20)

The independent delta re-review returned "Execute after fixes": 0 Blockers, 1 Important, 5 Minor. It accepted both planner deviations (M7 through the global handler; `File_Artifacts/jobs_worker` excluded) and the CLI daemon-thread loop. The controller's rulings:

| Ruling | Folded as |
|---|---|
| **I1** (Important): `POST /api/v1/llamacpp/inference` is open to any authenticated user and skips the override; the plan and ADR-086 called it admin-only | Refused at the route (option a), Task 3 Step 9: `_refuse_admin_disabled_llamacpp()` calls `enforce_provider_override("llama.cpp", None)` before the route's `try` (its `except Exception` would answer 500) and raises `HTTPException` 403 (`provider_disabled`) or 503 (`credential_store_unavailable`). No model: the route runs the loaded one. The out-of-scope list and the ADR-086 sentence no longer exclude it. One test, two cases: RED on `origin/dev` 2 failed (500, the stubbed server ran), GREEN 7 passed |
| **M1**: the CLI bootstrap dies with a traceback, even for `--help`, when the store cannot load | Task 1 Step 5: a `click.Group` subclass, `LLMStoreGroup` (`cli/utils/llm_store.py`), loads the store in `resolve_command`, where the subcommand's arguments are visible (a group callback cannot see them). It skips the load under `ctx.resilient_parsing` (shell completion) and when a help option is among them, and turns `LLMProviderOverridesRefreshError` into a `click.ClickException` naming `DATABASE_URL` and the AuthNZ setup. The round-1 callback blocks are gone. Two tests, three CLIs each: the help/completion test passes on `origin/dev` (nothing loads there) and fails on round 1's code (3, `DID NOT RAISE`); the error test fails on both (exit 2 on `origin/dev`, an uncaught `LLMProviderOverridesRefreshError` with no message on round 1's code); both GREEN |
| **M2**: the reading-import worker can make LLM claims calls | Added to B1 (one `main`, one test case): its ingest path runs `extract_claims_if_requested` when `ENABLE_INGESTION_CLAIMS` is on and `CLAIM_EXTRACTOR_MODE` names a provider. The existing `main` becomes `run_reading_import_jobs_worker`. RED `AttributeError`, GREEN |
| **M3**: the committed plan cited private planning files | No brief, ruling, review or prototype patch is cited by file name; Task 0 Step 5 says so and checks it |
| **M4**: the Bandit count | 22 files, in the command and in the M6 row above |
| **M5**: the handler's status is the client's only | Known limits: server and access logs still record a 500; Task 2 Step 6 says so |
| CLI race mitigation | Not taken: it would change `_schedule_override_recovery`, which the app process shares. The race stays in Known limits with the re-review's bound. Reversed in fold round 3 (I4) |

## Rulings (fold round 3, 2026-11-20)

The owner chose Q1 (a), fail-closed, and asked for a design review before implementation. The independent design review returned "Proceed after fixes": 0 Blockers, 4 Important, 7 Minor, 2 Improvements. The controller folded every item. RED is `origin/dev` with the new tests; GREEN is the patched tree; 0 netprobe lines everywhere.

| Ruling | Folded as | RED / GREEN |
|---|---|---|
| **I1**: RAG, evaluations and MCP lose a refusal's identity (503 "configuration invalid"; `invalid_provider_credentials`) | Task 2 Step 7: `classify_rag_provider_error` and `_eval_provider_http_exception` answer the three codes from `ADMIN_REFUSAL_STATUS` before their allowlists; Task 3 Step 6: MCP `model_completion` publishes `policy_code or error.code`, and the MCP test's code column is the reason. Prompt improvement's `provider_not_configured` is left as it is | RED: RAG case 1 failed, evaluations test 1 failed, MCP 9 failed (4 of them new: provider-disabled on every key). GREEN: 33, 30 and 91 passed |
| **I2**: a store blip permanently drops audio analysis | Task 3 Step 5: the analyze check, and `_summary_error_code` for a refusal at the adapter, give `provider_failure` (retried) for `credential_store_unavailable`, with "Error: Provider credential storage is temporarily unavailable."; everything else stays `configuration` | RED: boundary 22 failed, 89 passed (store cases and `_summary_error_code`'s store case). GREEN: 111 passed |
| **I3**: OpenRouter fallback models bypass `allowed_models` | Task 2 Step 5: the guard checks each string in `request["extra_body"]["models"]` for `openrouter`. AC #1 and the ADR-086 sentence say so. No other adapter reads a model list from `extra_body`: the gateway validates and passes one only for OpenRouter (`gateway.py:1119-1134`), and the other adapters forward `extra_body` unread, so a custom OpenAI-compatible endpoint that routes on a key of its own (a proxy's `fallbacks`) is outside the guard (Known limits) | RED: the OpenRouter test fails (`DID NOT RAISE`: the fallback runs). GREEN: 132 passed in its file |
| **I4**: reverses the round-2 ruling. The race is frequent (`benchmark_cli` runs each batch in its own `asyncio.run`), and the two loops thrash the PostgreSQL AuthNZ pool | Task 1 Step 3: `_schedule_override_recovery` returns early while a live refresh service runs on another loop (a no-op in the app, whose service runs on the caller's loop). Checking the pool question found a second user on the command loop: LLM usage accounting, which in a CLI ran on the command's loop for every async call. The sync bootstrap now sets the store's loop as the accounting loop, as the app sets its own, and drains the writes on exit. The race paragraph is gone from Known limits | RED: both tests fail (a recovery is scheduled on the caller's loop; no sync bootstrap). GREEN: 90 passed in their file |
| **M1**: the startup line reads as benign | "App Startup: LLM provider overrides unavailable; every LLM call is refused until they load" (Task 2 Step 8) | RED: 1 failed (the old line). GREEN: 47 passed |
| **M2**: a production `reset_db_pool()` voids every admin disable | Task 1 Step 3: `reset_db_pool` calls `mark_llm_provider_overrides_stale()`, which keeps the snapshot, the TTL and the refresh service and makes the next read reload. The test fallout, measured first, is none: the 45 test files that call `reset_db_pool()` gave 873 passed, 1 failed and 6 errors in one `-n 2` run, and the 2 files those 7 are in pass alone on the branch (264 passed): the errors were `Database initialization failed` while the machine's disk was full, and the failure an IP-allowlist WebSocket case. Only 2 test files call it and touch the store, and both pass. In tests the store's TTL is off, so the stale mark changes nothing there. The ruling's example, "unhealthy, next retry now", was not taken: it would refuse every LLM call between a production pool reset and the reload, while the last snapshot is still the admin's policy | RED: the pool-reset test fails (the store comes back empty). GREEN: 90 passed in its file |
| **M3**: D2's reason does not fit the first-chat check | D2 corrected; a comment on `LLMCallContext.exempt` says it also skips the guard (Task 2 Step 5) | Doc and comment only |
| **M4**: docs accuracy | ADR-086 names STT, not speech; the BYOK guide notes TASK-15583; the analyze refusal's legacy text is "Error: The provider or model is disabled by the administrator." | RED: the analyze refusal cases fail on the old text (in the boundary count). GREEN: 111 passed |
| **M5**: nothing proves `python -m <worker>` runs `main()` | Task 1 Step 1: `runpy.run_module` over the six worker modules, with `asyncio.run` patched to raise naming the coroutine it was given | RED: 5 of 6 fail (reading import's `main` is its runner on `origin/dev`). GREEN: 90 passed in its file |
| **M6**: open PR overlap | Task 3 Step 3's expectation reworded; #3477 and #3071 named under Open PR overlap | No code |
| **M7**: no upgrade note | Task 4 Step 8: a `CHANGELOG.md` Unreleased/Changed entry; the PR body (Task 5 Step 8) has a "Before upgrading" line | Doc only |
| **P1**: a class handler in place of the catch-all branch | Task 2 Step 6: `admin_refusal_exception_handler`, registered in `main.py` for `ChatConfigurationError`. It runs in Starlette's `ExceptionMiddleware`, which does not re-raise. Any other `ChatConfigurationError`, and any on a standalone Slides route, is raised on to the catch-all handler as before. The catch-all branch is gone. The "logs show 500" Known limit is dropped on measurement: the M7 test now runs the real `AccessLogMiddleware` under a test client that re-raises any server exception, and passes: the access log records 403/503 and no exception escapes to `ServerErrorMiddleware` | RED: the app test fails (no handler); GREEN: 5 passed, and the M7 test 3 of 3 in 132 |
| **P2**: a store health signal | Task 2 Step 8: `/api/v1/health` reports `llm_provider_overrides` as `healthy` or `unavailable` and never changes its overall status or code. The top-level `/health` is the immutable liveness contract (`{"status": "ok"}`) and `/readyz` the readiness snapshot; neither changes | RED: 2 failed (no store check). GREEN: 10 passed |

## Rulings (fold-3 check, 2026-11-20)

The design reviewer checked fold round 3: every fold correct, and both planner choices accepted (M2's stale mark keeps the admin's last policy and the refresh service; I4's daemon accounting loop is the smallest fix and exits within 5 s). It found two lint ratchet failures on the patched tree. The controller folded the rest here without a prototype round:

| Finding | Folded as |
|---|---|
| **L1**: `test_service_singleton_ratchet[globals]`: `llm_provider_overrides.py` 35 > 34, from M2's `global` | Task 1 Step 3: one entry in `tests/lint/global_statements_baseline.json`, `{"globals": 35, "reason": "TASK-15581 M2: a pool reset marks the override store stale instead of emptying it (fail-open fix)"}`; the ratchet allows a raise for a security fix |
| **L2**: `test_service_singleton_ratchet[getters]`: `health.py` 3 > 2, from P2's bare `get_llm_provider_overrides_snapshot()` | Task 2 Step 8: a store helper `llm_provider_overrides_status() -> str` in `llm_provider_overrides.py`; `health.py` calls it, so the getter count does not grow |
| **P2-a**: the health block catches only `ByokResolutionError`; a `RuntimeError` from the recovery scheduler would answer 500 | The L2 helper catches `Exception` and returns `"unavailable"`; one health test case with the snapshot raising `RuntimeError` |
| **P1-a**: with the class handler, Starlette 1.6.0 turns a `ChatConfigurationError` raised mid-stream into a logged `RuntimeError` (`starlette/_exception_handler.py:55-56`); clients are unaffected | Known limits, one line |
| **M2 test gap**: the pool-reset test proves the snapshot is kept, not that a reload follows | Task 1 Step 1: the test also asserts the next read after the reset reloads (about 3 lines) |
| I4's side effect is narrower than stated: only CLI calls with a user in context are recorded, and their sync quota checks now run | Task 5 Step 8: a PR-body line, a CLI user over a limit now gets the 402 |
| PostgreSQL: 49 cases skipped in the check (the local cluster was unreachable) | Task 5 Step 2: run them if the cluster is up; never start Docker for it; otherwise CI's PostgreSQL jobs are the measure, checked before merge |

**B1 entry points.** Included (each can reach an adapter and none loaded the store):

| Entry point | Why it can make an LLM call |
|---|---|
| `services/media_ingest_jobs_worker` (`worker-media-ingest`) | Media analysis (`analyze`) |
| `services/audio_jobs_worker` (`worker-audio`) | `audio_analyze` |
| `core/Data_Tables/jobs_worker` (`worker-data-tables`) | LLM table generation; it already read the store itself and failed standalone on dev |
| `core/Evaluations/embeddings_abtest_jobs_worker` (`worker-evals-abtest`) | Hybrid search through `unified_rag_pipeline` and the LLM-scoring reranker |
| `core/Explainer/jobs_worker` | `make_configured_explainer_generator` calls an adapter |
| `core/Collections/reading_import_jobs_worker` (`worker-reading-import`) | LLM claims extraction (`extract_claims_if_requested`, `claims_utils.py:44-71`) when `ENABLE_INGESTION_CLAIMS` is on and `CLAIM_EXTRACTOR_MODE` names a provider; off and `heuristic` by default (fold round 2, M2) |
| `tldw-evals` (`cli/evals_cli.py`) | Its `eval`, `benchmark`, `recipes` and `persona-chat-judge` groups |
| `core/Evaluations/cli/evals_cli.py` (deprecated, `python -m`) | Its `run` and `test-api` commands |
| `core/Evaluations/cli/evals_cli_enhanced.py` (`python -m`) | Its `run`, `test-api` and `recipes` commands |

Excluded, with the reason:

| Entry point | Reason |
|---|---|
| Research, Prompt Studio and TTS workers | They already load the store (`Research/jobs_worker.py:280-281`, `prompt_studio/services/jobs_worker.py:2087-2089`, `TTS/tts_jobs_worker.py:981-983`); moving them onto the helper is a larger diff than leaving them |
| `core/File_Artifacts/jobs_worker` (`worker-files`) | Its `get_adapter` is the file-type registry and its image adapter is `Image_Generation`'s; no chat adapter, no store read. A deliberate exclusion from the ruling's "at least" list: one line and one test case add it if wanted |
| `services/core_jobs_worker` (`worker-chatbooks`), `Chatbooks/services/jobs_worker` | Chatbook export and import only |
| `Collections/reading_digest_jobs_worker` | Digests render stored summaries; no LLM call |
| `services/connectors_worker` | `process_pdf` without analysis (`connectors_worker.py:564-570`) |
| `services/privilege_snapshot_worker` | No LLM call |
| `Embeddings/services/jobs_worker`, `redis_worker` | No contextualization runs in the workers (`process_and_store_content` is not called there) |
| `Audio_Studio/jobs_worker` | Its `provider_registry` is the audio-generation registry |
| `audiobook_jobs_worker`, `audio_transcribe_gpu_worker`, `llamacpp_acquisition_jobs_worker`, `CodeGraph`, `Document_Workspace/pdf_worker`, `Prototype_Workspaces`, `Sharing/shared_workspace_clone_jobs_worker` | No LLM call |
| `Evaluations/wordbench_runner.py`, `qa_benchmark_helper.py`, `Web_Scraping/enhanced_web_scraping.py` `__main__` blocks | Hard-coded examples, not deployed entry points |
| `main.py`, `scripts/run_server_guarded_mcp.py`, `scripts/server_lifecycle.py` | They run the app, which loads the store at startup |
| AuthNZ, DB migration, scheduler and MCP test `__main__` blocks | No LLM call |

## Decisions for the owner

The rule itself is decided. These remain; the plan implements each recommendation.

- **Q1. Does a store outage stop every LLM call?** Today an unhealthy store stops chat (503 `credential_store_unavailable`), the 19 resolve-time runtimes and BYOK resolution; server-key work with no runtime (RAG generation, workflows, claims, research, notes graph, the no-handle analyze path, the judges' server-key calls) keeps running. After this plan it stops every adapter call. One failed refresh marks the cache unhealthy at once (`_complete_override_refresh_failure`, `llm_provider_overrides.py:318-320`), and recovery retries with a 1 s to 30 s backoff, so one AuthNZ database blip becomes an app-wide LLM outage of up to about 30 s.
  - (a) **Recommended, planned:** keep fail-closed. An admin disable that lapses during an outage defeats the rule, and chat already behaves this way. `Usage_Quotas.md` says so.
  - (b) Let a failed refresh keep the last good snapshot until it is `_OVERRIDE_MAX_STALE_SECONDS` (30 s) old, the staleness a healthy cache already allows. It changes the store for chat too.
- **D1. TASK-15577 AC #1 says "refuse ... at the boundary".** The boundary is `users_own_key_credentials`, which feeds `admit_media_analysis` (six media endpoints), the media ingest worker and `enforce_llm_quota_on_own_key` (benchmarks, evaluation runs, the study assistants).
  - (a) **Recommended, planned (the review agrees):** the boundary keeps returning None (the request keeps the server's key) and the call is refused at dispatch, the same outcome as for a user with no stored key: a media request is ingested and its analysis fails `configuration` (a warning on the item; the audio worker does not retry `configuration`); an evaluation's judge calls fail as sample errors. One side effect: an over-limit own-key user of a disabled provider gets the 402, not a policy error. Correct AC #1 to: "A disabled provider and an unreadable override store refuse an own-key analyze request instead of running it on the server key, with tests".
  - (b) Refuse the whole request at the boundary with 403 (503 for the store). It needs the final model, often resolved later, and gives two outcomes for one disable by whether a key is stored. About 30 lines of source and 60 of tests.
- **D2. Key tests** (the review agrees). Provider key validation (user, team, org and admin BYOK key tests, `byok_testing.py:648`, `:673`) and the setup first-chat check (`first_chat_verifier.py:70`) run inside `llm_call_context(exempt=True)`.
  - (a) **Recommended, planned:** the guard skips exempt calls. The admin provider test's `use_override=false` exists to try a provider before turning it on; a key test is a fixed `"ping"` with `max_tokens: 1` whose content is never returned. The setup first-chat check is different: it takes a caller prompt of up to 1000 characters and returns the reply (`setup_schemas.py:326,336`, `setup.py:1660-1673`), and it is safe only because of the first-run gate (local setup access, setup not completed: `_require_first_run_write_access`, `setup.py:716-733`). The guard keys on `LLMCallContext.exempt`, an accounting flag, so a comment on that field says it also skips the guard (fold 3, M3).
  - (b) Refuse key validation for a disabled provider too: the user and shared key endpoints need their own check, about 10 lines and tests.
- **D3. The WebUI cannot offer BYOK for a provider the server has no key for.** Filed as TASK-15583 (commit `3b006a1340`); Task 0 adds the review's AC to it. No code here.
- **D4. Jobs retry a refusal** (the review agrees it is not a storm). `WorkerSDK` retries an exception without `retryable = False`, up to the job's attempt limit, with backoff. Media analysis failures are not exceptions, the audio worker does not retry `configuration`, and Data Tables retries only `credential_store_unavailable`. Recommended: no change here.

## What dev does today, re-verified

| Path | Today (`f8f141b82e`) | This plan |
|---|---|---|
| Chat (`validate_provider_override`, chat.py:4469, 5295), the study assistants (`chat_target_resolution.py:207`), the 19 resolve-time runtimes (`_capture_override_snapshot` before resolution, every key source) | Refused on every key | Unchanged; the base-class guard backs them up at dispatch |
| `_summarize_via_adapter`, own/team/org handle | `model_not_allowed` falls back to the server key (`override_outcome`) | Refused, `configuration` |
| `_summarize_via_adapter`, no handle or a `server_default` handle | Not checked: a disabled provider or disallowed model runs | Refused, `configuration` |
| Judges `eval_runner._call_adapter_text`, `ms_g_eval._call_adapter_text` | Handle: fallback; server key: not checked | Refused at the adapter, `ChatConfigurationError` |
| MCP `model_completion` (authoritative `_resolve_entry`) | BYOK: `model_not_allowed` re-resolves onto the server key; server key: not checked | Refused on every key after the scope check |
| `users_own_key_credentials` | Catch-all returns None for every failure and logs "the server key is used" | Returns None (D1 a); the server-key call is refused at dispatch |
| Prompt Studio direct adapters (`evaluation_manager`, `prompt_executor`, `test_runner`) | A disallowed `DEFAULT_MODEL_<PROVIDER>` runs on a user or team key (`configured_provider_model_from_snapshot` ignores `DEFAULT_MODEL_*`) | Refused at the adapter: the request carries the final model |
| Slides standalone-HTML generation (`standalone_html_provider.py`, its own `httpx` request) | Not checked on any key | Refused before the request (I1) |
| Every other adapter caller with no check of its own (claims, RAG generation, notes graph, research synthesis, web scraping LLM extraction, voice assistant, workflows content adapters and the rest) | A disabled provider or disallowed model runs on the server key | Refused at the adapter |
| `POST /api/v1/llamacpp/inference` (any authenticated user; posts to the managed llama.cpp server, no adapter) | Not checked: a disabled llama.cpp runs | Refused before the model runs: 403, or 503 for the store (fold round 2, I1) |
| Standalone workers and evaluation CLIs | Never load the store: the server-key path already fails there (`resolve_provider_api_key` raises `credential_store_unavailable`; measured), explicit-key and own-key calls run | Load the store at start (B1) |

Out of scope, each with its own configuration: embeddings (they check the override at their endpoint, `embeddings_v5_production_enhanced.py:820-821`), the llama.cpp reranking routes (`/llamacpp/rerank`, `/llamacpp/reranking`: they run the `llama-embedding` binary with an embedding model, not the chat server), image generation (including the OpenRouter image backend), TTS and STT, the OCR backends, and token counting. The llama.cpp inference route is in scope (fold round 2, I1).

**The shared choke point.** Every provider adapter's own `chat`, `achat`, `stream` and `astream` is wrapped by `instrument_provider_class` (`ChatProvider.__init_subclass__`, ADR-067), whatever the caller: the gateway (`perform_chat_api_call[_async]`) or a direct `get_adapter(...).chat(...)`. That is where the guard lives. The gateway pre-call hooks were rejected because they miss the direct callers (ADR-067 rejected recording there for the same reason). The adapter registry was rejected because `get_adapter` does not know the model. `_enforce_call_policy` (the resolve-time step) was not changed: the Prompt Studio `DEFAULT_MODEL_*` gap is closed at dispatch, where the model is final, so the 19 runtimes are untouched. The paths that do not reach an adapter keep their own check: the MCP bounded transport (`_resolve_entry`), the Messages API native path (its runtime resolves with the model, messages.py:1958-1963/1997, unchanged), Slides standalone-HTML generation (I1) and the llama.cpp inference route (fold round 2, I1).

**The HyDE note (TASK-15576).** `hyde.py:132-150` marks the handle used in `on_success`, after `sgl.analyze(..., raise_on_error=True)` returns. With no fallback, a call either runs on the handle or raises `SummaryProviderError`, so the handle is marked used only after a call on it. The note goes away with the fallback; no change to `hyde.py`.

## What this plan deletes from #3523

| Item | Where | Other users (`git grep`) |
|---|---|---|
| `server_key_fallback`, `log_server_key_fallback`, `override_outcome` | `llm_provider_overrides.py:892-953` | Only `Summarization_General_Lib`, `eval_runner`, `ms_g_eval`, `provider_credential_runtime` (all changed here) and #3523's tests |
| `_denied_model_takes_the_server_key` and its call | `eval_runner.py:111-125, 156-163` | None |
| The `override_outcome` block | `ms_g_eval.py:115-133` | None |
| The `server_key` branch and the "could not be checked" message | `Summarization_General_Lib.py:315-337` | None |
| The authoritative re-resolution onto the server key | `provider_credential_runtime.py:807-833` | None |
| `strict=` on `user_to_check`, `check_user_sync`, `check_user_async`; `check_user_async`'s bool return | `llm_quota_gate.py` | Only `override_outcome` and the ingest worker |
| The worker's `answered` marking and its docstring sentence | `media_ingest_jobs_worker.py:378-405` | None |
| "; the server key is used" in the boundary's warning | `usage_quota_deps.py:126` | #3523's decision test |
| #3523's fallback tests | `test_llm_quota_byok_calls.py` (7 tests, 4 helpers), `test_provider_adapter_runtime_boundary.py` (6 tests), `test_summarization_adapter.py` (1), `test_llm_quota_decision.py` (1), MCP `test_model_completion_integration.py` (2), `test_model_completion_quota_compatibility.py` (1) | Replaced by the refusal tests below |

Kept: `ProviderCredentialRuntime.resolve(..., model=)` passing `model` to `_resolve_entry` (the new MCP check needs it); the boundary's bounded reason code in its warning (TASK-15571 AC #4); the MCP tests' store fixture, rewritten (TASK-15577 M2).

## Open PR overlap (read only, 2026-11-20)

- #3477 (media ingest; already conflicting) touches `media_ingest_jobs_worker.py`, its nearest hunk about 25 lines above Task 1's `main()`.
- #3071 edits `eval_runner._call_adapter_text` three lines below #3523's hunk. If it lands first, Task 3 Step 3's reverse patch still applies, and the file then equals `b1a32fbba4^1` plus #3071.
- `exceptions.py` is touched by #2932, #3437, #3480, #3019 and #2738, all outside `ChatConfigurationError._ERROR_CODES`. #3269 is stale.

## Known limits

- **A call that names no model** is checked for `provider_disabled` only (M1). Seven adapters fill a default model after the guard (`google_adapter.py:138-139`, `deepseek_adapter.py:135`, `mistral_adapter.py:64`, `qwen_adapter.py:81`, `huggingface_adapter.py:163`, Cohere and Moonshot via `providers/base.py:119-120`); no caller sends them no model today.
- **Cost per call:** `capture_provider_override_call_snapshot` deep-copies the whole override map (`_get_healthy_override_snapshot`, `:720`), twice for an `achat` that runs `chat` in a thread. Fine at today's size; copying only the provider's entry would remove it. Accepted (M9).
- **Routing a custom endpoint does itself.** The guard checks the model and OpenRouter's fallback list. A custom OpenAI-compatible endpoint that routes on a key of its own in `extra_body` (a proxy's `fallbacks`) is not seen; that endpoint's operator configures its routing (fold round 3, I3).
- **A refusal raised mid-stream is logged as a `RuntimeError`.** With the `ChatConfigurationError` class handler registered, Starlette 1.6.0 turns one raised after the response started into `RuntimeError("Caught handled exception, but response already started.")` before the handler runs (`starlette/_exception_handler.py:55-56`). The guard refuses before a stream starts, so clients and refusals are unaffected (fold-3 check, P1-a).
- **D2, D4 and Q1** above; **TASK-15583** for the WebUI listing.

---

### Task 0: Branch, backlog, draft PR, baseline

**Files:**
- Modify: the TASK-15581, TASK-15583, TASK-15571, TASK-15576, TASK-15577 task files (backlog-py)
- Create: `Docs/superpowers/plans/2026-11-19-refuse-admin-disabled-provider.md` (this plan, verbatim)

- [ ] **Step 1: Check out the existing branch.** It was pushed by the controller with the three task files (`3b006a1340`); do not create it.

```bash
git fetch origin dev fix/admin-disabled-provider-refused-15581
git switch fix/admin-disabled-provider-refused-15581
git log --oneline -1
```

Expected: `3b006a1340 chore(backlog): file TASK-15581, TASK-15582 and TASK-15583`. If `origin/dev` has moved past `f8f141b82e`, merge it in (`git merge origin/dev`) and re-run Step 6.

- [ ] **Step 2: Start TASK-15581, extend AC #1 and add the B1 AC.** backlog-py cannot edit an AC in place, so AC #1 is removed and added again with "and each OpenRouter fallback model" (fold 3, I3).

```bash
BACKLOG task edit TASK-15581 -s "In Progress" --remove-ac 1 \
  --ac "Every provider adapter call (chat, achat, stream, astream) refuses an admin-disabled provider or model, and each OpenRouter fallback model, on every key (own, team, org, server, no handle) before the quota check, with ChatConfigurationError carrying the reason as error_code, with tests" \
  --ac "Every standalone worker process and CLI that can make an LLM call loads and refreshes the provider-override store at start, so a provider with no override is unaffected outside the app process, with tests"
```

Expected: `TASK-15581 - Refuse an admin-disabled provider or model on every key and path (reverses TASK-15571's server-key fallback) (In Progress)`. Old ACs #2 to #6 become #1 to #5; the extended AC #1 is #6 and the B1 AC #7.

- [ ] **Step 3: Add the review's AC to TASK-15583.**

```bash
BACKLOG task edit TASK-15583 \
  --ac "With no server key there is no model inventory, so the picker gets the provider's models from the user's own key (or accepts a typed model ID), never from another user's key"
```

Expected: `TASK-15583 - The WebUI cannot pick a provider the server has no key for, even with the user's own key (To Do)`. TASK-15582 gets no edit.

- [ ] **Step 4: Notes on TASK-15571, TASK-15576, TASK-15577.**

```bash
BACKLOG task edit TASK-15571 \
  --append-notes "2026-11-19: the owner reversed this rule. A provider or model the admin explicitly disables (is_enabled=False, or a model off allowed_models) is refused on every key and path; nothing falls back to the server key. TASK-15581 implements the reversal and removes the fallback PR #3523 shipped."
BACKLOG task edit TASK-15576 -s "In Progress" \
  --append-notes "Owner's answer, 2026-11-19: keep refusing; explicit disable is refused everywhere. So no runtime adopts the fallback, and #3523's is removed (TASK-15581). Chat's 403 and the 19 resolve-time runtimes are unchanged; the provider base class now refuses at dispatch too, which closes the Prompt Studio DEFAULT_MODEL_* gap, and the HyDE mark-used note goes away with the fallback (a call either runs on the handle or raises)."
BACKLOG task edit TASK-15577 -s "In Progress" \
  --append-notes "TASK-15581 does this task (2026-11-19). AC #1 is met at dispatch, not at the boundary (Decision D1 of TASK-15581's plan): users_own_key_credentials still returns None for a policy or store failure, and the server-key call it leaves is refused by the analyze path's check and the provider base class, as for a user with no stored key; an over-limit own-key user of a disabled provider gets the 402. M1 is moot: there is no fallback any more. M2 is fixed: the MCP tests' store fixture uses set_llm_provider_overrides_cache_for_tests and restores the store."
```

Expected: three lines in the `TASK-NNNNN - <title> (<status>)` form, `(Done)`, `(In Progress)` and `(In Progress)`. If the owner picks D1 (b), Task 3 grows by the boundary refusal and this note changes; the controller re-plans that step.

- [ ] **Step 5: Copy the plan, commit, push, update the PR.** Copy this file to `Docs/superpowers/plans/2026-11-19-refuse-admin-disabled-provider.md`. It names no private planning file (the briefs, rulings, reviews and prototype patch are cited as "the controller's planning brief" and "the independent plan review"); check with `grep -n -E '_1557[0-9]\.(md|patch)' <copy>`, which must print nothing.

```bash
git add -A backlog/tasks Docs/superpowers/plans/2026-11-19-refuse-admin-disabled-provider.md
git commit -m "chore(backlog): start TASK-15581 with its plan (TASK-15581)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin fix/admin-disabled-provider-refused-15581
gh pr edit 3537 --body-file <plans dir>/body_15581.md
```

The PR already exists (draft #3537); do not create another.

- [ ] **Step 6: Baseline.** Each file in its own process:

```bash
for f in tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py \
         tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py \
         tldw_Server_API/tests/Slides/test_standalone_html_provider.py \
         tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py \
         tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py \
         tldw_Server_API/tests/Usage/test_llm_quota_decision.py \
         tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
         tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py \
         tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_quota_compatibility.py \
         tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py \
         tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py \
         tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py \
         tldw_Server_API/tests/Slides/test_standalone_html_generation.py \
         tldw_Server_API/tests/LLM_Local/test_llamacpp_inference_api.py; do
  PYTEST "$f" | tail -1
done
```

Expected (measured on `f8f141b82e`): 90, 66, **84 failed** / 64 passed, 104, 128, 36, 59, 85, 8, 8, 4, 113, 2, 5 passed; 0 netprobe lines, except `test_standalone_html_provider.py`. Its 84 failures are tests listed in `tests/live_network_baseline.txt` (they resolve `api.openai.com`, which the probe blocks); they fail the same way on every tree under the probe, and the plan compares their failing ids, which are identical before and after.

---

### Task 1: Load the provider-override store in every standalone process, and keep it loaded (B1, fold 3 I4/M2)

This lands before the guard, so no commit on the branch breaks a standalone worker.

**Files:**
- Modify: `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` (imports `:3-10`; two context managers before `_parse_json_value`, `:590`)
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py` (`__main__`, `:875-880`), `tldw_Server_API/app/services/audio_jobs_worker.py` (`:630-632`), `tldw_Server_API/app/core/Data_Tables/jobs_worker.py` (`:1335-1336`), `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py` (`:278-279`), `tldw_Server_API/app/core/Explainer/jobs_worker.py` (`:119-120`), `tldw_Server_API/app/core/Collections/reading_import_jobs_worker.py` (`main` `:24`, `__main__` `:44-45`)
- Create: `tldw_Server_API/cli/utils/llm_store.py` (`LLMStoreGroup`)
- Modify: `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` (`_schedule_override_recovery`, `:386`; `mark_llm_provider_overrides_stale` before `set_llm_provider_overrides_cache_for_tests`) and `tldw_Server_API/app/core/AuthNZ/database.py` (`reset_db_pool`, `:1973-1979`)
- Modify: `tldw_Server_API/cli/evals_cli.py` (the `@click.group` of `main`, `:115`; import after `cli.utils.config`), `tldw_Server_API/app/core/Evaluations/cli/evals_cli.py` (`@click.group` of `cli`, `:92`) and `.../evals_cli_enhanced.py` (`:370`); each core CLI imports `LLMStoreGroup` after its `adapter_utils` import
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py` (append)

**Interfaces:**
- `standalone_llm_provider_overrides() -> AbstractAsyncContextManager[None]`: `refresh_llm_provider_overrides(force=True)` (raises if the store cannot be loaded), `start_llm_provider_override_refresh_service()`, then on exit `shutdown_llm_provider_override_recovery()`.
- `standalone_llm_provider_overrides_sync() -> AbstractContextManager[None]`: the same on a daemon-thread loop, for click CLIs (`ctx.with_resource(...)`).
- `standalone_llm_provider_overrides_sync()` also makes the store's loop the LLM usage accounting loop while it is entered, and drains the writes on exit (fold 3, I4).
- `mark_llm_provider_overrides_stale() -> None`: the next read reloads; snapshot, TTL and refresh service stay (fold 3, M2).
- `LLMStoreGroup(click.Group)`, keyword `llm_commands: Iterable[str]`: loads the store with `standalone_llm_provider_overrides_sync()` in `resolve_command` before one of those commands runs.
- Each worker gains `async def main(...)`; its `__main__` runs `asyncio.run(main(...))`.

- [ ] **Step 1: Write the failing tests.** Append to `test_llm_provider_overrides.py`:

```python
# The standalone entry points that can make an LLM call (TASK-15581, B1): (module, its job loop or click group).
_STANDALONE_WORKERS = (
    ("tldw_Server_API.app.services.media_ingest_jobs_worker", "run_media_ingest_jobs_worker"),
    ("tldw_Server_API.app.services.audio_jobs_worker", "run_audio_jobs_worker"),
    ("tldw_Server_API.app.core.Data_Tables.jobs_worker", "run_data_tables_jobs_worker"),
    ("tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs_worker", "run_embeddings_abtest_jobs_worker"),
    ("tldw_Server_API.app.core.Explainer.jobs_worker", "run_explainer_jobs_worker"),
    ("tldw_Server_API.app.core.Collections.reading_import_jobs_worker", "run_reading_import_jobs_worker"),
)
# (module, click group, a command that makes LLM calls, one that makes none)
_EVALUATION_CLIS = (
    ("tldw_Server_API.cli.evals_cli", "main", "eval", "config"),
    ("tldw_Server_API.app.core.Evaluations.cli.evals_cli", "cli", "run", "list-benchmarks"),
    ("tldw_Server_API.app.core.Evaluations.cli.evals_cli_enhanced", "cli", "test-api", "list-benchmarks"),
)


def _unloaded_store_over_an_empty_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store as a fresh standalone process has it: unloaded, over an AuthNZ database that holds no override."""
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store

    async def _pool():
        return object()

    async def _no_rows(_repo, provider=None):
        return []

    monkeypatch.setattr(store, "get_db_pool", _pool)
    monkeypatch.setattr(store.AuthnzLLMProviderOverridesRepo, "list_overrides", _no_rows)
    set_llm_provider_overrides_cache_for_tests({}, healthy=False)


_WORKER_IDS = ["media-ingest", "audio", "data-tables", "evals-abtest", "explainer", "reading-import"]


@pytest.mark.parametrize(("module_name", "run_name"), _STANDALONE_WORKERS, ids=_WORKER_IDS)
async def test_each_standalone_worker_loads_the_store_before_it_takes_jobs(monkeypatch, module_name, run_name):
    """TASK-15581 (B1): outside the app nothing loads the provider-override store, and every LLM call refuses until it
    is loaded. Each worker's main loads it before taking jobs, so the store reads inside the work, and stops its
    refresh service after."""
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store

    module = importlib.import_module(module_name)
    _unloaded_store_over_an_empty_database(monkeypatch)
    worked = asyncio.Event()

    async def _work(*_args, **_kwargs):
        capture_provider_override_call_snapshot("openai")  # credential_store_unavailable while unloaded
        worked.set()

    monkeypatch.setattr(module, run_name, _work)
    await module.main()
    assert worked.is_set()
    assert store._OVERRIDE_REFRESH_SERVICE_TASK is None


_CLI_IDS = ["tldw-evals", "evals_cli", "evals_cli_enhanced"]


@pytest.mark.parametrize(("module_name", "group_name", "llm_command", "other_command"), _EVALUATION_CLIS, ids=_CLI_IDS)
def test_each_evaluation_cli_loads_the_store_for_its_llm_commands(
    monkeypatch, module_name, group_name, llm_command, other_command
):
    """TASK-15581 (B1): a CLI command that makes LLM calls loads the store, on its own loop thread, before it runs. A
    command that makes none leaves the store alone."""
    from click.testing import CliRunner

    group = getattr(importlib.import_module(module_name), group_name)
    _unloaded_store_over_an_empty_database(monkeypatch)

    other = CliRunner().invoke(group, [other_command, "--help"], obj={})
    with pytest.raises(ByokResolutionError, match="credential_store_unavailable"):
        capture_provider_override_call_snapshot("openai")
    CliRunner().invoke(group, [llm_command], obj={})  # it may then fail its own parse; the store loads first
    assert other.exit_code == 0
    assert capture_provider_override_call_snapshot("openai").provider == "openai"


@pytest.mark.parametrize(("module_name", "group_name", "llm_command", "other_command"), _EVALUATION_CLIS, ids=_CLI_IDS)
def test_an_evaluation_cli_skips_the_store_for_help_and_completion(
    monkeypatch, module_name, group_name, llm_command, other_command
):
    """TASK-15581 (fold 2, M1): help for an LLM command, and shell completion, never touch the AuthNZ database."""
    from click.testing import CliRunner

    group = getattr(importlib.import_module(module_name), group_name)
    _unloaded_store_over_an_empty_database(monkeypatch)

    help_text = CliRunner().invoke(group, [llm_command, "--help"], obj={})
    completion_ctx = group.make_context("cli", [llm_command], resilient_parsing=True)
    with completion_ctx:
        group.resolve_command(completion_ctx, [llm_command])
    assert help_text.exit_code == 0
    with pytest.raises(ByokResolutionError, match="credential_store_unavailable"):
        capture_provider_override_call_snapshot("openai")


@pytest.mark.parametrize(("module_name", "group_name", "llm_command", "other_command"), _EVALUATION_CLIS, ids=_CLI_IDS)
def test_an_evaluation_cli_that_cannot_load_the_store_says_why(
    monkeypatch, module_name, group_name, llm_command, other_command
):
    """TASK-15581 (fold 2, M1): an LLM command whose store cannot load stops with a message naming DATABASE_URL and
    the AuthNZ setup, not a traceback."""
    from click.testing import CliRunner

    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store

    async def _unreadable(*_args, **_kwargs):
        raise LLMProviderOverridesRefreshError()

    group = getattr(importlib.import_module(module_name), group_name)
    monkeypatch.setattr(store, "refresh_llm_provider_overrides", _unreadable)
    result = CliRunner().invoke(group, [llm_command], obj={})

    assert result.exit_code == 1
    assert "DATABASE_URL" in result.output and "AuthNZ" in result.output


@pytest.mark.parametrize(("module_name", "run_name"), _STANDALONE_WORKERS, ids=_WORKER_IDS)
def test_each_standalone_worker_runs_main_as_a_module(monkeypatch, module_name, run_name):
    """TASK-15581 (fold 3, M5): sidecars start a worker with ``python -m <module>``, so its ``__main__`` block must run
    ``main``, which loads the store, not the bare ``run_*`` coroutine."""
    import runpy
    import sys
    import warnings

    class _Ran(Exception):
        """Carries the name of the coroutine ``asyncio.run`` was given, and stops the module there."""

    def _run(coro, *_args, **_kwargs):
        coro.close()
        raise _Ran(coro.__qualname__)

    monkeypatch.setattr(asyncio, "run", _run)
    monkeypatch.setattr(sys, "argv", [module_name])
    with warnings.catch_warnings(), pytest.raises(_Ran, match=r"^main$"):
        warnings.simplefilter("ignore", RuntimeWarning)  # runpy warns that the module is already imported
        runpy.run_module(module_name, run_name="__main__")


async def test_a_stale_read_leaves_the_refresh_to_a_service_on_another_loop():
    """TASK-15581 (fold 3, I4): in a CLI the refresh service runs on its own daemon loop. A read on a command's own
    ``asyncio.run`` loop schedules no refresh there, which that loop's closing could cancel (marking the store
    unhealthy) and which would move the PostgreSQL AuthNZ pool between loops."""
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store

    background = _BackgroundEventLoop()
    service = background.call(lambda: background.loop.create_task(asyncio.sleep(3600))).result(timeout=5)
    try:
        with store._OVERRIDE_LOCK:
            store._OVERRIDE_REFRESH_SERVICE_TASK = service
            store._OVERRIDE_CACHE_HEALTHY = False
            store._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
        store._schedule_override_recovery()
        with store._OVERRIDE_LOCK:
            scheduled = store._OVERRIDE_RECOVERY_TASK
    finally:
        with store._OVERRIDE_LOCK:
            store._OVERRIDE_REFRESH_SERVICE_TASK = None
        background.loop.call_soon_threadsafe(service.cancel)
        background.close()
    assert scheduled is None


def test_a_cli_runs_usage_accounting_on_the_stores_loop(monkeypatch):
    """TASK-15581 (fold 3, I4): a CLI's LLM usage writes run on the store's daemon loop, as they run on the server's
    loop in the app, so no write moves the PostgreSQL AuthNZ pool to a command's own loop; the loop is put back after."""
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store
    from tldw_Server_API.app.core.Usage.llm_accounting import get_accounting_loop

    _unloaded_store_over_an_empty_database(monkeypatch)
    before = get_accounting_loop()
    with store.standalone_llm_provider_overrides_sync():
        same_loop = get_accounting_loop() is store._OVERRIDE_REFRESH_SERVICE_TASK.get_loop()
    assert (same_loop, get_accounting_loop()) == (True, before)


def test_an_authnz_pool_reset_keeps_every_admin_disable():
    """TASK-15581 (fold 3, M2): production resets the AuthNZ pool when it finds it closed or on another loop. The store
    keeps the last snapshot, so an admin disable keeps holding while it reloads from the new pool; it must not come
    back empty and healthy, with no refresh, until an admin edits an override."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", is_enabled=False)}, ttl_enabled=True
    )
    asyncio.run(reset_db_pool())

    override = get_llm_provider_override("openai")
    assert override is not None and override.is_enabled is False
```

The pool-reset test (M2) also asserts that a reload follows the reset, not only that the snapshot is kept: after `reset_db_pool()`, with the store's TTL on, the next read schedules a refresh (fold-3 check; about 3 lines).

- [ ] **Step 2: Run them on the unchanged source.**

```bash
PYTEST tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py
```

Expected (measured): 20 failed, 70 passed. The 6 worker cases fail with `AttributeError` (five modules have no `main`; the reading-import module has no `run_reading_import_jobs_worker`). The 3 CLI load cases fail on their last line, `ByokResolutionError: credential_store_unavailable: openai`. The 3 CLI error cases exit 2, not 1: nothing loads, so the command fails its own argument parse. 5 of the 6 `python -m` cases run the bare `run_*` coroutine; the reading-import case passes, because that module's `main` on `origin/dev` is its runner (the worker test above catches it). The I4 test fails (a recovery refresh is scheduled on the caller's loop), the accounting test fails (no sync bootstrap), and the M2 test fails (the reset store comes back empty). The help/completion test passes: nothing loads on `origin/dev`, so it pins that the bootstrap keeps it that way. 0 netprobe lines. (The `python -m` test was then reshaped to raise instead of filling a list, for the triage gate; its assertion is the same and was not re-run on `origin/dev`.)

- [ ] **Step 3: The bootstrap.** In `llm_provider_overrides.py`, `import contextlib` after `import asyncio`, and `from collections.abc import AsyncIterator, Iterator, Mapping`. Before `def _parse_json_value`:

```python
@contextlib.asynccontextmanager
async def standalone_llm_provider_overrides() -> AsyncIterator[None]:
    """Load the provider-override store and keep it fresh, in a process the app did not start (TASK-15581).

    The app loads the store at startup. A standalone worker or CLI must do it itself: the admin-override check
    refuses every LLM call while the store is unloaded. The Research worker's pattern: a forced refresh, the
    refresh service, and the recovery tasks shut down on exit. A refresh that fails raises, so the process
    does not start.
    """
    await refresh_llm_provider_overrides(force=True)
    try:
        start_llm_provider_override_refresh_service()
        yield
    finally:
        await shutdown_llm_provider_override_recovery()


@contextlib.contextmanager
def standalone_llm_provider_overrides_sync() -> Iterator[None]:
    """``standalone_llm_provider_overrides`` for a synchronous CLI: the store's loop runs on a daemon thread.

    LLM usage accounting runs on that loop too, as it runs on the server's loop in the app: the AuthNZ pool is bound
    to one loop, and on PostgreSQL a write from a command's own ``asyncio.run`` loop would recreate it under the
    store's refresh.
    """
    from tldw_Server_API.app.core.Usage.llm_accounting import (
        drain_accounting,
        get_accounting_loop,
        set_accounting_loop,
    )

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="llm-provider-overrides", daemon=True)
    thread.start()
    store = standalone_llm_provider_overrides()
    try:
        asyncio.run_coroutine_threadsafe(store.__aenter__(), loop).result()
        previous_accounting_loop = get_accounting_loop()
        set_accounting_loop(loop)
        try:
            yield
        finally:
            set_accounting_loop(previous_accounting_loop)
            asyncio.run_coroutine_threadsafe(drain_accounting(), loop).result()
            asyncio.run_coroutine_threadsafe(store.__aexit__(None, None, None), loop).result()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
```

Then, in `_schedule_override_recovery`, first thing under `with _OVERRIDE_LOCK:` (fold 3, I4):

```python
        service = _OVERRIDE_REFRESH_SERVICE_TASK
        if service is not None and not service.done() and service.get_loop() is not loop:
            # The refresh service keeps the cache fresh from its own loop (a CLI's daemon loop, TASK-15581). A refresh
            # on this loop could be cancelled when it closes, and would move the PostgreSQL AuthNZ pool between loops.
            return
```

and, before `set_llm_provider_overrides_cache_for_tests` (fold 3, M2):

```python
def mark_llm_provider_overrides_stale() -> None:
    """Make the next read reload the store, keeping its snapshot, TTL and refresh service (TASK-15581).

    For an AuthNZ pool reset: the stored policy has not changed, so an admin disable keeps holding from the last
    snapshot while the reload runs, and the refresh service keeps refreshing from the new pool.
    """
    global _OVERRIDE_CACHE_REFRESHED_AT
    with _OVERRIDE_LOCK:
        due = time.monotonic() - _OVERRIDE_REFRESH_INTERVAL_SECONDS
        _OVERRIDE_CACHE_REFRESHED_AT = min(_OVERRIDE_CACHE_REFRESHED_AT, due)
```

In `reset_db_pool` (`AuthNZ/database.py`), the last `try` block calls it in place of the test setter:

```python
    try:
        from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
            mark_llm_provider_overrides_stale,
        )
        # Reload from the new pool, keeping the last snapshot: emptying it would void every admin disable (TASK-15581).
        mark_llm_provider_overrides_stale()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring LLM provider overrides cache reset error: {e}")
```

The new `global` raises `llm_provider_overrides.py` to 35 globals, over the ratchet's 34 (`test_service_singleton_ratchet[globals]`). In `tldw_Server_API/tests/lint/global_statements_baseline.json`, set that file's entry to `{"globals": 35, "reason": "TASK-15581 M2: a pool reset marks the override store stale instead of emptying it (fail-open fix)"}`, and add the baseline file to Step 7's `git add` (fold-3 check, L1).

- [ ] **Step 4: The workers.** In each of the six modules, add a `main` before the `__main__` block and run it there. `reading_import_jobs_worker.py` already has a `main` that runs the worker: rename it to `run_reading_import_jobs_worker` and add the new `main` after it (its `__main__` is unchanged). For `audio_jobs_worker.py` (the others are the same with their own `run_*` coroutine; `Data_Tables`, the A/B test and the Explainer have no `contextlib.suppress`):

```python
async def main() -> None:
    """Load the provider-override store, then take jobs: every LLM call refuses until it is loaded (TASK-15581)."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import standalone_llm_provider_overrides

    async with standalone_llm_provider_overrides():
        await run_audio_jobs_worker()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
```

`media_ingest_jobs_worker.py` keeps its `--heavy` flag:

```python
async def main(heavy: bool = False) -> None:
    """Load the provider-override store, then take jobs: every LLM call refuses until it is loaded (TASK-15581)."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import standalone_llm_provider_overrides

    async with standalone_llm_provider_overrides():
        await (run_media_ingest_heavy_jobs_worker() if heavy else run_media_ingest_jobs_worker())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a media ingest Jobs worker")
    parser.add_argument("--heavy", action="store_true", help="Run the heavy ingest worker pool")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(heavy=parser.parse_args().heavy))
```

- [ ] **Step 5: The CLIs.** Create `tldw_Server_API/cli/utils/llm_store.py`:

```python
"""A click group whose LLM commands load the admin provider-override store first (TASK-15581)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import click


class LLMStoreGroup(click.Group):
    """Load the provider-override store before one of ``llm_commands`` runs, and keep it fresh until exit.

    Every LLM call refuses while the store is unloaded, and nothing else loads it in a CLI process. A request for help
    and shell completion skip the load, so they never need the AuthNZ database.
    """

    def __init__(self, *args: Any, llm_commands: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.llm_commands = frozenset(llm_commands)

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple[Any, Any, list[str]]:
        cmd_name, cmd, rest = super().resolve_command(ctx, args)
        if cmd_name in self.llm_commands and not ctx.resilient_parsing and set(ctx.help_option_names).isdisjoint(rest):
            from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
                LLMProviderOverridesRefreshError,
                standalone_llm_provider_overrides_sync,
            )

            try:
                ctx.with_resource(standalone_llm_provider_overrides_sync())
            except LLMProviderOverridesRefreshError:
                raise click.ClickException(
                    "The admin LLM provider overrides could not be loaded from the AuthNZ database, and every LLM "
                    "call refuses without them. Check DATABASE_URL, and that AuthNZ is set up "
                    "(python -m tldw_Server_API.app.core.AuthNZ.initialize)."
                ) from None
        return cmd_name, cmd, rest
```

and make each group one, with that CLI's LLM commands (`{"eval", "benchmark", "recipes", "persona-chat-judge"}` for `tldw-evals`, `{"run", "test-api"}` for the deprecated CLI, `{"run", "test-api", "recipes"}` for the enhanced one). For `tldw-evals`:

```python
@click.group(
    cls=LLMStoreGroup,
    # These make LLM calls, which need the provider-override store (TASK-15581).
    llm_commands={"eval", "benchmark", "recipes", "persona-chat-judge"},
    context_settings={'help_option_names': ['-h', '--help']},
)
```

Commands that make no LLM call (`health`, `db`, `config`, `users` and the rest), help and shell completion never need the AuthNZ database for it. The load runs inside the parent context (`ctx.with_resource`), so the store stays fresh until the command exits.

- [ ] **Step 6: Run again.** Same command as Step 2, plus the existing CLI tests, which now run the bootstrap against the test AuthNZ database:

```bash
PYTEST tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py
PYTEST tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py \
  tldw_Server_API/tests/Evaluations/unit/test_evals_cli_custom_command.py \
  tldw_Server_API/tests/Evaluations/unit/test_evals_cli_recipe_commands.py \
  tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py
```

Expected (measured): 90 passed; 28 passed (fold round 2; not re-run in round 3). 0 netprobe lines.

- [ ] **Step 7: Commit.**

```bash
git add tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/app/services/audio_jobs_worker.py \
  tldw_Server_API/app/core/Data_Tables/jobs_worker.py tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py \
  tldw_Server_API/app/core/Explainer/jobs_worker.py tldw_Server_API/app/core/Collections/reading_import_jobs_worker.py \
  tldw_Server_API/cli/utils/llm_store.py tldw_Server_API/cli/evals_cli.py \
  tldw_Server_API/app/core/Evaluations/cli/evals_cli.py tldw_Server_API/app/core/Evaluations/cli/evals_cli_enhanced.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_overrides.py \
  tldw_Server_API/tests/lint/global_statements_baseline.json
git commit -m "fix(llm): load the provider-override store in standalone workers and CLIs (TASK-15581)

Outside the app nothing loaded the store, so the next commit's admin-override
check would refuse every LLM call there. Each worker or evaluation CLI that can
make an LLM call now loads the store at start and keeps it fresh. A CLI
skips it for help and shell completion, and says what to fix when it fails.

A CLI's own event loop no longer schedules store refreshes or accounting writes
while the refresh service runs on its daemon loop, so the two loops stop racing
each other and the PostgreSQL AuthNZ pool. A production AuthNZ pool reset now
marks the store stale instead of replacing it with an empty, never-refreshed
one, which had silently voided every admin disable.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: One refusal helper, the guard in the provider base class, and how a refusal surfaces

**Files:**
- Modify: `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` (add `enforce_provider_override` after `validate_provider_override`; the #3523 helpers stay until Task 3)
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (module docstring `:1-6`; imports `:23`; new `_refuse_admin_disabled` before `_wrap_chat` `:362`; first line of each wrapper `chat` `:366`, `achat` `:390`, `stream` `:418`, `astream` `:443`)
- Modify: `tldw_Server_API/app/core/exceptions.py` (`ChatConfigurationError._ERROR_CODES`, `:996-1002`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_results.py` (`ADMIN_REFUSAL_STATUS` after `_PROVIDER_STREAM_ERROR_STATUS`, `:67`), `tldw_Server_API/app/core/Usage/llm_call_context.py` (a comment on `exempt`, `:58`)
- Modify: `tldw_Server_API/app/api/v1/utils/exception_handlers.py` (new `admin_refusal_exception_handler` before `tts_public_http_exception_handler`; imports; `__all__`), `tldw_Server_API/app/main.py` (its registration before the `TTSPublicHTTPException` handler, `:2080`; imports `:2000-2005`)
- Modify: `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py` (`classify_rag_provider_error`, `:108`), `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py` (`_eval_provider_http_exception`, `:257`; import `:70`)
- Modify: `tldw_Server_API/app/services/startup_auth.py` (`:117`), `tldw_Server_API/app/api/v1/endpoints/health.py` (`api_health`, before `body = {`)
- Test: `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py` (replace `:479-718`, #3523's fallback tests; imports `:4`, `:10`, `:18`), `tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py` (append), `tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py` (one case), `tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py` (append), `tldw_Server_API/tests/Services/test_startup_auth.py` (`:950`), `tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py` (append)

**Interfaces:**
- `enforce_provider_override(provider: str, model: str | None) -> None` in `llm_provider_overrides`: raises `ProviderOverridePolicyError` (`policy_code` `provider_disabled` / `model_not_allowed`) or `ByokResolutionError("credential_store_unavailable")`; logs one warning `"Provider override for {provider}: {reason}; the call is refused"`.
- `_refuse_admin_disabled(provider: Any, request: Any) -> None` in `usage_accounting`: no-op in an exempt call context; else raises `ChatConfigurationError(PROVIDER_STREAM_ERROR_MESSAGES[code], provider=..., error_code=code)`.

- [ ] **Step 1: Write the failing tests.** In `test_llm_quota_byok_calls.py`, delete everything from `_OVERRIDE_MODULE = ...` (`:479`) to just before `def _chat_request()` (`:720`): the helpers `_denied_model_fallback`, `_slow_decision`, `_failing_decision`, `_evaluation_judge`, `_judge_modules` and #3523's seven fallback tests. Put this in their place:

```python
# An admin override for openai that refuses REQUEST's model "m1", by its reason (TASK-15581).
_DISABLES = {"provider_disabled": {"is_enabled": False}, "model_not_allowed": {"allowed_models": ["x"]}}


def _admin_override(**override_kwargs: Any) -> None:
    """Give openai one admin provider override."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    set_llm_provider_overrides_cache_for_tests({"openai": LLMProviderOverride(provider="openai", **override_kwargs)})


async def _call(method: str, request: dict[str, Any]) -> Any:
    """One call through ``method`` of the accounted provider; a stream is read to its end."""
    if method == "chat":
        return _Provider().chat(request)
    if method == "achat":
        return await _Provider().achat(request)
    if method == "stream":
        return list(_Provider().stream(request))
    return [line async for line in _Provider().astream(request)]


_METHODS = pytest.mark.parametrize("method", ["chat", "achat", "stream", "astream"])


@_METHODS
@pytest.mark.parametrize("reason", sorted(_DISABLES))
@pytest.mark.parametrize(
    "source", ["user", "team", "org", "server_default", None], ids=["user", "team", "org", "server", "no-handle"]
)
async def test_an_admin_disabled_provider_or_model_is_refused_on_every_key(refused, method, reason, source) -> None:
    """TASK-15581, the owner's rule: whatever the key, the provider base class refuses the call before the provider
    runs or the limit is checked, so nothing is recorded; the error code is the reason."""
    _admin_override(**_DISABLES[reason])
    request = dict(REQUEST)
    if source is not None:
        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = await _handle(source)

    with llm_call_context(user_id=7), pytest.raises(ChatConfigurationError) as refusal:
        await _call(method, request)
    await llm_accounting.drain_accounting()
    assert (refusal.value.error_code, refused) == (reason, [])


@_METHODS
async def test_an_unreadable_override_store_refuses_the_call(refused, method) -> None:
    """TASK-15581: a policy that cannot be read refuses at the provider, as it does at resolution."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        set_llm_provider_overrides_cache_for_tests,
    )

    set_llm_provider_overrides_cache_for_tests({}, healthy=False)
    with pytest.raises(ChatConfigurationError) as refusal:
        await _call(method, dict(REQUEST))
    assert refusal.value.error_code == "credential_store_unavailable"


@pytest.mark.parametrize(
    "override", [None, {"is_enabled": True}, {"allowed_models": ["m1"]}], ids=["none", "on", "allowed"]
)
async def test_the_users_own_key_serves_a_provider_the_server_has_no_key_for(refused, override) -> None:
    """TASK-15581: 'no key or credits' on the server is not a disable. Resolved with the same override check every
    runtime uses, and an empty server config, the user's own key serves the provider whether it has no override or
    one that leaves it on (with no key of its own); its calls run, are not checked against the limit, and are the
    user's."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        capture_provider_override_call_snapshot,
    )

    if override is not None:
        _admin_override(**override)

    async def _users_key(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        """The user's own key; the server has none."""
        return ResolvedByokCredentials(
            provider=provider, api_key="sk-user", app_config={}, credential_fields={}, source="user", allowlisted=True
        )

    runtime = ProviderCredentialRuntime(
        user_id=7, team_ids=[], org_ids=[], trusted_base_url_override=False,
        fallback_resolver=lambda _provider: None, server_config_snapshot={}, resolver=_users_key,
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    request = {**REQUEST, PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: await runtime.resolve("openai", model="m1")}

    with llm_call_context(user_id=7):
        await _call_all_four(request)
    await llm_accounting.drain_accounting()
    assert [row["credential_source"] for row in refused] == ["user"] * 4


async def test_a_key_test_is_not_refused(refused) -> None:
    """TASK-15581 (Decision D2): provider key validation, the admin provider test and the setup check run exempt. A
    key test is not a use of the provider, and the admin test can try a provider before turning it on."""
    _admin_override(is_enabled=False)
    with llm_call_context(user_id=7, exempt=True):
        assert _Provider().chat(dict(REQUEST))["choices"][0]["message"]["content"] == "hi"


@pytest.mark.parametrize("reason", sorted(_DISABLES))
async def test_a_call_that_names_no_model_is_checked_for_provider_disabled_only(reason) -> None:
    """TASK-15581 (M1): with no model in the request there is no model to hold against the allowlist, so a disabled
    provider is refused and an allowlist is not. The gap: an adapter that fills a default model after the guard."""
    _admin_override(**_DISABLES[reason])
    request = {**REQUEST, "model": None}
    if reason == "provider_disabled":
        with pytest.raises(ChatConfigurationError):
            _Provider().chat(request)
    else:
        assert _Provider().chat(request)["choices"][0]["message"]["content"] == "hi"


@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "status", "code"),
    [
        ({"is_enabled": False}, True, 403, "provider_disabled"),
        ({"allowed_models": ["x"]}, True, 403, "model_not_allowed"),
        ({}, False, 503, "credential_store_unavailable"),
    ],
    ids=["provider-disabled", "model-not-allowed", "store-unavailable"],
)
def test_an_endpoint_that_maps_nothing_answers_with_the_refusals_own_status(override_kwargs, healthy, status, code):
    """TASK-15581 (M7, P1): an endpoint with no mapping of its own reaches the app's ChatConfigurationError handler,
    which answers an admin disable with 403 and an unreadable store with 503, not a 500. It runs inside Starlette's
    ExceptionMiddleware and re-raises nothing, so the test client sees no server exception and the access log records
    the same status."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from loguru import logger

    from tldw_Server_API.app.api.v1.utils.exception_handlers import (
        admin_refusal_exception_handler,
        global_unhandled_exception_handler,
    )
    from tldw_Server_API.app.core.Logging.access_log_middleware import AccessLogMiddleware
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    app = FastAPI()
    app.add_middleware(AccessLogMiddleware)
    app.add_exception_handler(ChatConfigurationError, admin_refusal_exception_handler)
    app.add_exception_handler(Exception, global_unhandled_exception_handler)

    @app.get("/llm")
    def _llm() -> dict[str, Any]:
        """Call the provider and map nothing."""
        return _Provider().chat(dict(REQUEST))

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", **override_kwargs)}, healthy=healthy
    )
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="INFO")
    try:
        response = TestClient(app).get("/llm")  # raises if any server exception escapes
    finally:
        logger.remove(sink)
    access_statuses = [line.split()[4] for line in logged if line.startswith("HTTP GET /llm -> ")]

    assert (response.status_code, response.json()["error"]["code"], access_statuses) == (status, code, [str(status)])


def test_openrouter_fallback_models_are_checked_against_the_allowlist() -> None:
    """TASK-15581 (I3): OpenRouter serves a model from ``extra_body["models"]`` when the primary fails, so a caller
    could name an allowed primary and reach a disallowed model. Each fallback model is checked too."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    class _OpenRouter(_Provider):
        name = "openrouter"

    set_llm_provider_overrides_cache_for_tests(
        {"openrouter": LLMProviderOverride(provider="openrouter", allowed_models=["openai/gpt-4o-mini"])}
    )
    request = {**REQUEST, "model": "openai/gpt-4o-mini", "extra_body": {"models": ["anthropic/claude-opus-4.1"]}}

    with pytest.raises(ChatConfigurationError) as refused:
        _OpenRouter().chat(request)
    assert refused.value.error_code == "model_not_allowed"


def _judge_modules() -> Any:
    """The two evaluation judges that call the adapter directly."""
    from tldw_Server_API.app.core.Evaluations import eval_runner, ms_g_eval

    return pytest.mark.parametrize("module", [eval_runner, ms_g_eval], ids=["eval_runner", "ms_g_eval"])


@_judge_modules()
@pytest.mark.parametrize("own_key", [True, False], ids=["own-key", "server-key"])
async def test_an_evaluation_judge_refuses_a_disallowed_default_model_on_any_key(monkeypatch, module, own_key) -> None:
    """TASK-15581: a DEFAULT_MODEL_<PROVIDER> the override does not allow is refused on the run's own key and on the
    server's, with ChatConfigurationError, which G-Eval does not retry."""
    monkeypatch.setattr(module, "get_adapter_or_raise", lambda _provider: _Provider())
    monkeypatch.setattr(module, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: "sk-server")
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "gpt-denied")
    _admin_override(**_DISABLES["model_not_allowed"])
    handle = await _handle("user") if own_key else None

    def _judge() -> str:
        """One call with no model, so DEFAULT_MODEL_OPENAI applies."""
        return module._call_adapter_text(
            api_endpoint="openai", messages_payload=[{"role": "user", "content": "q"}], provider_credentials=handle
        )

    with pytest.raises(ChatConfigurationError) as refusal:
        await asyncio.to_thread(_judge)
    assert refusal.value.error_code == "model_not_allowed"


async def test_prompt_studio_refuses_a_disallowed_default_model_on_the_users_key(monkeypatch) -> None:
    """TASK-15581 (from TASK-15576): Prompt Studio's direct adapter call takes DEFAULT_MODEL_<PROVIDER>, which
    resolution never sees; the provider base class refuses it when the override does not allow it."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import test_runner

    monkeypatch.setattr(test_runner, "get_adapter_or_raise", lambda _provider: _Provider())
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "gpt-denied")
    _admin_override(**_DISABLES["model_not_allowed"])
    handle = await _handle("user")

    with pytest.raises(ChatConfigurationError) as refusal:
        test_runner.TestRunner(None)._call_adapter(
            provider="openai", model=None, messages_payload=[{"role": "user", "content": "q"}], system_message=None,
            temperature=0.0, max_tokens=1, provider_credentials=handle,
        )
    assert refusal.value.error_code == "model_not_allowed"
```

Imports at the top of the file: `from collections.abc import AsyncIterator, Iterable` (drop `Callable`), drop `from loguru import logger`, and `from tldw_Server_API.app.core.exceptions import ChatConfigurationError, LLMQuotaExceededError`.

`test_client_disconnect_exception_handler.py` (P1), appended:

```python
@pytest.mark.asyncio
async def test_the_app_answers_an_admin_refusal_and_passes_any_other_configuration_error_on():
    """TASK-15581 (P1): the app registers a ChatConfigurationError handler. An admin provider-override refusal gets its
    own 403; any other configuration error is raised on to the catch-all handler, as before."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.main import app

    handler = app.exception_handlers[ChatConfigurationError]
    refusal = ChatConfigurationError("refused", provider="openai", error_code="provider_disabled")
    other = ChatConfigurationError("misconfigured", provider="openai")

    response = await handler(_build_request(request_id="req-403"), refusal)
    with pytest.raises(ChatConfigurationError) as passed_on:
        await handler(_build_request(), other)
    assert (response.status_code, passed_on.value) == (403, other)
```

`test_rag_provider_credentials.py` (I1), a case in `test_typed_provider_errors_map_to_bounded_http_responses` after the `missing_provider_credentials` one:

```python
        (
            # The provider base class's admin-override refusal keeps its code and 403 (TASK-15581).
            ChatConfigurationError(_SENTINEL_SECRET, provider="openai", error_code="provider_disabled"),
            403,
            "provider_disabled",
        ),
```

`test_specialized_provider_auth_failures.py` (I1), appended:

```python
def test_an_admin_override_refusal_keeps_its_code_and_status() -> None:
    """TASK-15581: the provider base class's refusal of a disallowed model is a 403 with its own code, not a 503
    'configuration invalid' that invites a retry."""
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

    response = eval_endpoint._eval_provider_http_exception(
        ChatConfigurationError("refused", provider="openai", error_code="model_not_allowed")
    )

    assert (response.status_code, response.detail["error_code"]) == (403, "model_not_allowed")
```

`test_startup_auth.py:950` (M1): the expected warning becomes "App Startup: LLM provider overrides unavailable; every LLM call is refused until they load". `test_readiness_health_sanitizers.py` (P2), appended:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(("healthy", "store_status"), [(True, "healthy"), (False, "unavailable")])
async def test_api_health_reports_the_provider_override_store_without_changing_its_status(
    monkeypatch, healthy, store_status
):
    """TASK-15581 (P2): every LLM call refuses while the provider-override store is unavailable, so /health says so;
    but the check is informational, so a store blip never turns a probe on this route into a restart."""
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.AuthNZ import database as auth_database
    from tldw_Server_API.app.core.Metrics import metrics_manager
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        set_llm_provider_overrides_cache_for_tests,
    )

    class _HealthyPool:
        async def health_check(self):
            return {"status": "healthy"}

    async def _healthy_pool():
        return _HealthyPool()

    monkeypatch.setattr(auth_database, "get_db_pool", _healthy_pool)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: object())
    monkeypatch.setattr(chacha_deps, "get_chacha_health_snapshot", lambda: {"status": "healthy"})
    set_llm_provider_overrides_cache_for_tests({}, healthy=healthy)

    response = await health_mod.api_health()
    body = json.loads(response.body.decode("utf-8"))

    assert (response.status_code, body["status"], body["checks"].get("llm_provider_overrides")) == (
        200,
        "ok",
        {"status": store_status},
    )
```

- [ ] **Step 2: Run them on Task 1's source.** Each file in its own process:

```bash
for f in tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py \
         tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py \
         tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py \
         tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py \
         tldw_Server_API/tests/Services/test_startup_auth.py \
         tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py; do
  PYTEST "$f" | tail -1
done
```

Expected (measured on `origin/dev` with these files; Task 1 changes none of the code they exercise): `test_llm_quota_byok_calls.py` 54 failed, 78 passed. Failing: `test_an_admin_disabled_provider_or_model_is_refused_on_every_key`, all 40 (sync `chat` and `stream` run on every key, `DID NOT RAISE`; `achat` and `astream` on a non-own key get the 402, because the quota check cannot run on the accounting loop's own thread for a sync method); `test_an_unreadable_override_store_refuses_the_call`, all 4; the judge test, all 4, and the Prompt Studio test (they run); `test_a_call_that_names_no_model_is_checked_for_provider_disabled_only[provider_disabled]`; `test_an_endpoint_that_maps_nothing_answers_with_the_refusals_own_status`, all 3 (200: the provider runs); and `test_openrouter_fallback_models_are_checked_against_the_allowlist` (`DID NOT RAISE`: the disallowed fallback runs). The pins pass: the BYOK test's 3 cases, the key test and the no-model allowlist case. `test_client_disconnect_exception_handler.py` 1 failed, 4 passed (no `ChatConfigurationError` handler); `test_rag_provider_credentials.py` 1 failed, 32 passed (503 `provider_configuration_invalid`); `test_specialized_provider_auth_failures.py` 1 failed, 29 passed (503 `provider_configuration_invalid`); `test_startup_auth.py` 1 failed, 46 passed (the old line); `test_readiness_health_sanitizers.py` 2 failed, 8 passed (no store check). 0 netprobe lines.

- [ ] **Step 3: The helper.** In `llm_provider_overrides.py`, after `validate_provider_override`:

```python
def enforce_provider_override(provider: str, model: str | None) -> None:
    """Refuse a call the admin provider override disables, whatever its key (TASK-15581).

    The owner's rule (2026-11-19): a provider the override turns off (``provider_disabled``), or a model off its
    ``allowed_models`` (``model_not_allowed``), is refused on the user's own key, a team or org key and the
    server's. A provider the server merely has no key for is not refused. Raises ``ProviderOverridePolicyError``,
    or ``ByokResolutionError("credential_store_unavailable")`` when the policy cannot be read. Each refusal writes
    one warning naming the provider and the reason, never a key.
    """
    provider_norm = canonical_provider_name(provider)
    try:
        capture_provider_override_call_snapshot(provider_norm).enforce(model)
    except ByokResolutionError as exc:
        reason = getattr(exc, "policy_code", None) or exc.code
        logger.warning("Provider override for {}: {}; the call is refused", provider_norm, reason)
        raise
```

- [ ] **Step 4: The error codes.** In `exceptions.py`, `ChatConfigurationError._ERROR_CODES` gains:

```python
            # An admin provider override refused the call at the provider base class (TASK-15581).
            "provider_disabled",
            "model_not_allowed",
            "credential_store_unavailable",
```

- [ ] **Step 5: The guard.** In `usage_accounting.py`, the module docstring's second paragraph becomes:

```python
``instrument_provider_class`` wraps a provider class's own ``chat``, ``achat``,
``stream`` and ``astream``. The admin provider-override check and the quota check
run before the provider method and may raise ``ChatConfigurationError`` or
``LLMQuotaExceededError``; nothing else here may raise into the call.
```

Add `Mapping` to the `collections.abc` import, and two imports above `from tldw_Server_API.app.core.Usage.llm_accounting import submit_accounting`:

```python
from tldw_Server_API.app.core.exceptions import ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls.provider_results import PROVIDER_STREAM_ERROR_MESSAGES
```

Add before `def _wrap_chat`:

```python
def _refuse_admin_disabled(provider: Any, request: Any) -> None:
    """Refuse a provider or model the admin provider override disables, on any key (TASK-15581).

    Every adapter call passes here, from the gateway or a direct caller, so the owner's rule holds on every path.
    A key test (provider key validation, the admin provider test, the setup check) runs exempt and is not refused.
    """
    if current_llm_call_context().exempt:
        return
    # Imported here: the override module imports the adapter registry, which loads the provider classes.
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import enforce_provider_override

    name = getattr(provider, "name", "")
    request = request if isinstance(request, dict) else {}
    extra_body = request.get("extra_body")
    # OpenRouter serves a model from extra_body["models"] when the primary fails, so each one is checked too.
    fallbacks = extra_body.get("models") if name == "openrouter" and isinstance(extra_body, Mapping) else None
    try:
        for model in [request.get("model"), *(fallbacks if isinstance(fallbacks, list) else [])]:
            enforce_provider_override(name, model)
    except ByokResolutionError as error:
        code = getattr(error, "policy_code", None) or error.code
        # A ChatAPIError, which every adapter caller handles: on analyze 'configuration' (a store outage
        # 'provider_failure', which a job retries), no retry on a judge.
        message = PROVIDER_STREAM_ERROR_MESSAGES[code]
        raise ChatConfigurationError(message, provider=error.provider, error_code=code) from None
```

and make `_refuse_admin_disabled(self, request)` the first line of each of the four inner functions (`chat`, `achat`, `stream`, `astream`), before `source = call_credential_source(request)`. In `Usage/llm_call_context.py`, above `exempt: bool = False` (M3):

```python
    # Not accounted, and also skips the admin provider-override guard (TASK-15581): key tests and the setup check only.
```

In `provider_results.py`, after `_PROVIDER_STREAM_ERROR_STATUS` (the one status table every mapper below uses):

```python
# The admin provider-override refusals the provider base class raises (TASK-15581), with their HTTP status.
ADMIN_REFUSAL_STATUS = {"provider_disabled": 403, "model_not_allowed": 403, "credential_store_unavailable": 503}
```

- [ ] **Step 6: The status where nothing maps it (M7, fold 3 P1).** In `api/v1/utils/exception_handlers.py`, import `from tldw_Server_API.app.core.exceptions import ChatConfigurationError` and `from tldw_Server_API.app.core.LLM_Calls.provider_results import ADMIN_REFUSAL_STATUS`, add `"admin_refusal_exception_handler"` to `__all__`, and before `tts_public_http_exception_handler`:

```python
async def admin_refusal_exception_handler(request: Request, exc: ChatConfigurationError) -> JSONResponse:
    """Answer an admin provider-override refusal no endpoint mapped (TASK-15581): 403, or 503 for the store.

    Registered for ``ChatConfigurationError``, it runs in Starlette's ``ExceptionMiddleware``, which does not
    re-raise, so the logs and every middleware see the 403/503 too. Any other ``ChatConfigurationError`` is raised
    on to the catch-all handler, as before.
    """
    code = getattr(exc, "error_code", None)
    if code not in ADMIN_REFUSAL_STATUS:
        raise exc
    request_id = _get_request_id(request)
    return JSONResponse(
        status_code=ADMIN_REFUSAL_STATUS[code],
        headers={"X-Request-ID": request_id},
        content={"error": {"code": code, "message": exc.message, "request_id": request_id}},
    )
```

In `main.py`, import it as `admin_refusal_exception_handler as _admin_refusal_handler` in the `exception_handlers` import, add `ChatConfigurationError` to the `core.exceptions` import, and before the `TTSPublicHTTPException` handler:

```python
@app.exception_handler(ChatConfigurationError)
async def _admin_refusal_exception_handler(request: Request, exc: ChatConfigurationError):
    # An admin provider-override refusal no endpoint mapped answers 403/503 (TASK-15581). A standalone Slides route
    # and any other ChatConfigurationError go on to the catch-all handler, as before.
    if is_standalone_sensitive_route(request.method, request.url.path):
        raise exc
    response = await _admin_refusal_handler(request, exc)
    return _apply_runtime_cors_headers(request, response)
```

`global_unhandled_exception_handler` is unchanged.

- [ ] **Step 7: RAG and evaluations keep the reason (I1).** In `classify_rag_provider_error`'s `ChatConfigurationError` branch, after `code = str(...)`, with `from tldw_Server_API.app.core.LLM_Calls.provider_results import ADMIN_REFUSAL_STATUS`:

```python
        if code in ADMIN_REFUSAL_STATUS:  # the provider base class's admin-override refusal (TASK-15581)
            return code, ADMIN_REFUSAL_STATUS[code], _RAG_PROVIDER_ERROR_MESSAGES[code]
```

In `_eval_provider_http_exception`, a branch before `elif isinstance(exc, ChatConfigurationError):`, with `from tldw_Server_API.app.core.LLM_Calls.provider_results import ADMIN_REFUSAL_STATUS, PROVIDER_STREAM_ERROR_MESSAGES`:

```python
    elif isinstance(exc, ChatConfigurationError) and exc.error_code in ADMIN_REFUSAL_STATUS:
        # The provider base class refused an admin-disabled provider or model, or an unreadable store (TASK-15581).
        code = exc.error_code
        status_code = ADMIN_REFUSAL_STATUS[code]
        message = PROVIDER_STREAM_ERROR_MESSAGES[code]
```

- [ ] **Step 8: Observability (M1, P2).** `startup_auth.py:117`'s warning becomes "App Startup: LLM provider overrides unavailable; every LLM call is refused until they load". In `api_health`, before `body = {`:

```python
    # The admin provider-override store, informational only (TASK-15581): every LLM call is refused while it is
    # unavailable. It never changes the overall status or code, so a probe on this route cannot restart a process
    # over a store blip and prolong the outage.
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import llm_provider_overrides_status

    checks["llm_provider_overrides"] = {"status": llm_provider_overrides_status()}
```

and in `llm_provider_overrides.py`, after `get_llm_provider_overrides_snapshot`:

```python
def llm_provider_overrides_status() -> str:
    """Return ``"healthy"`` when an LLM call would find a usable override snapshot, else ``"unavailable"``.

    For health reporting (TASK-15581): it never raises, so a health route that shows it cannot fail over it.
    """
    try:
        get_llm_provider_overrides_snapshot()
    except Exception:  # noqa: BLE001 - any failure to read the snapshot means calls are refused
        return "unavailable"
    return "healthy"
```

A helper, not a bare `get_*` call in `health.py`: the endpoint getter ratchet (`test_service_singleton_ratchet[getters]`, `health.py` ceiling 2) counts bare getter calls in endpoint modules and only falls (fold-3 check, L2). It catches every exception, not only `ByokResolutionError`: the snapshot read can also raise `RuntimeError` through `_schedule_override_recovery`, and the route must not answer 500 over it (P2-a). Add one case to the health test: with `get_llm_provider_overrides_snapshot` patched to raise `RuntimeError`, `/api/v1/health` keeps its status code and reports `unavailable`.

It reads the snapshot through the public accessor, so it reports what a call would meet, staleness included. `/api/v1/health` maps its overall status to 200/206/503, and probes and scripts call it (`production_preflight.py`, the deployment guides), so the check stays out of `overall`.

- [ ] **Step 9: Run again.** Same loop as Step 2.

Expected (measured): 132, 5, 33, 30, 47 and 10 passed. 0 netprobe lines.

- [ ] **Step 10: The files the guard reaches.** The other #3523 test files still pass: their adapters are not `ChatProvider` subclasses, so they test the paths Task 3 changes.

```bash
for f in tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py \
         tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py \
         tldw_Server_API/tests/Usage/test_llm_quota_decision.py \
         tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py; do
  PYTEST "$f" | tail -1
done
```

Expected (measured on `origin/dev`): 104, 128, 36 and 85 passed: unchanged. Also unchanged: `test_provider_credential_runtime.py` 59 and MCP `test_model_completion_quota_compatibility.py` 8.

- [ ] **Step 11: Commit.**

```bash
git add tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py \
  tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/exceptions.py \
  tldw_Server_API/app/core/LLM_Calls/provider_results.py tldw_Server_API/app/core/Usage/llm_call_context.py \
  tldw_Server_API/app/api/v1/utils/exception_handlers.py tldw_Server_API/app/main.py \
  tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py \
  tldw_Server_API/app/services/startup_auth.py tldw_Server_API/app/api/v1/endpoints/health.py \
  tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py \
  tldw_Server_API/tests/Logging/test_client_disconnect_exception_handler.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py \
  tldw_Server_API/tests/Evaluations/unit/test_specialized_provider_auth_failures.py \
  tldw_Server_API/tests/Services/test_startup_auth.py tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py
git commit -m "fix(llm): refuse an admin-disabled provider or model at the provider base class (TASK-15581)

The owner's rule: a provider or model the admin explicitly disables is
refused on every key and path. Every adapter call passes the accounted
base class, so the check lives there, before the quota check, for the model
and each OpenRouter fallback model. RAG, evaluations and an endpoint that
maps nothing answer the refusal with 403, or 503 for an unreadable store;
/api/v1/health reports the store, without changing its status.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Remove #3523's fallback; analyze, MCP, the boundary, Slides standalone HTML and llama.cpp inference refuse

**Files:**
- Modify: `tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py` (delete `server_key_fallback`, `log_server_key_fallback`, `override_outcome`, `:892-953`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` (`_summarize_via_adapter`, `:315-337`)
- Modify: `tldw_Server_API/app/core/Provider_Credentials/provider_credential_runtime.py` (`_resolve_entry`, `:807-833`)
- Modify (revert of #3523's hunks): `tldw_Server_API/app/core/Evaluations/eval_runner.py`, `tldw_Server_API/app/core/Evaluations/ms_g_eval.py`, `tldw_Server_API/app/core/Usage/llm_quota_gate.py`, `tldw_Server_API/app/services/media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py` (`users_own_key_credentials`' warning, `:120-128`)
- Modify: `tldw_Server_API/app/core/Slides/standalone_html_provider.py` (new `_verify_admin_override` before `_load_current_config`, `:334`; its call in `generate_standalone_html`, after `_verify_current_target`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py` (new `_refuse_admin_disabled_llamacpp` after the `LlamaCppInferenceRequest` import, `:1285`; its call as the first statement of `run_llamacpp_inference_endpoint`, before `try:`)
- Test: `tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py` (`:433-660`), `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (`:834-860`), `tldw_Server_API/tests/Usage/test_llm_quota_decision.py` (`:380-404`), `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py` (`:377-424`), `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_quota_compatibility.py` (`:92-110`, import `:29`), `tldw_Server_API/app/core/MCP_unified/tests/conftest.py` (`:43-55`), `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py` (`:1895-1896`), `tldw_Server_API/tests/Slides/test_standalone_html_provider.py` (new test after `test_unrelated_aggregate_unavailability_does_not_disable_stored_target`), `tldw_Server_API/tests/LLM_Local/test_llamacpp_inference_api.py` (append)

- [ ] **Step 1: Write the failing tests.**

`test_provider_adapter_runtime_boundary.py`: `_POLICY_DENIED` becomes `"Error: The provider or model is disabled by the administrator."` and `_STORE_DOWN = "Error: Provider credential storage is temporarily unavailable."` follows it (fold 3, I2 and M4); delete `_FALLBACK_LINE` and its comment; `_refused_line`'s docstring becomes "The one warning a refused call writes (TASK-15571, TASK-15581)."; replace the four tests at `:540-612` (`test_a_denied_model_on_the_users_own_key_runs_on_the_server_key`, `test_a_disabled_provider_or_an_unreadable_policy_refuses_the_call`, `test_a_denied_model_on_a_shared_key_runs_on_the_server_key`, `test_a_denied_model_never_takes_the_callers_key_to_the_server_key_path`) with the block below, and delete `test_a_denied_model_on_the_server_key_path_runs_unchecked` and `test_a_denied_model_on_a_server_default_handle_runs_unchecked` (`:632-655`). `test_an_allowed_model_stays_on_the_users_own_key` and `test_own_key_summary_with_an_allowlist_and_no_default_model_stays_missing_model` stay.

```python
# Each admin disable, and a policy that cannot be read (TASK-15581).
_REFUSALS = pytest.mark.parametrize(
    ("override_kwargs", "healthy", "reason", "refused"),
    [
        ({"allowed_models": ["x"]}, True, "model_not_allowed", _POLICY_DENIED),
        ({"is_enabled": False}, True, "provider_disabled", _POLICY_DENIED),
        ({}, False, "credential_store_unavailable", _STORE_DOWN),
    ],
    ids=["model-not-allowed", "provider-disabled", "override-cache-unhealthy"],
)


async def _refused_summary(
    monkeypatch: pytest.MonkeyPatch, override_kwargs: dict[str, Any], healthy: bool, **summary_kwargs: Any
) -> tuple[str | None, list[tuple[str, str | None, bool]], list[str]]:
    """``_summary`` under one admin override for openai."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import LLMProviderOverride

    overrides = {"openai": LLMProviderOverride(provider="openai", **override_kwargs)}
    return await _summary(monkeypatch, overrides=overrides, healthy=healthy, **summary_kwargs)


@pytest.mark.asyncio
@_MODEL_SOURCES
@_REFUSALS
async def test_an_admin_disabled_provider_or_model_is_refused_on_the_users_own_key(
    monkeypatch: pytest.MonkeyPatch,
    override_kwargs: dict[str, Any],
    healthy: bool,
    reason: str,
    refused: str,
    config_model: str | None,
    model_override: str | None,
    expected: str,
) -> None:
    """TASK-15581, the owner's rule: whatever the model's source, a provider or model the admin disabled is refused
    with 'configuration' (as TASK-15496 did), and the reason is logged. A policy that cannot be read refuses too, as
    the retryable 'provider_failure', so a job tries again once it loads."""
    result, calls, warnings = await _refused_summary(
        monkeypatch, override_kwargs, healthy, config_model=config_model, model_override=model_override
    )

    assert (result, calls, warnings) == (refused, [], [_refused_line(reason)])


@pytest.mark.asyncio
@_REFUSALS
@pytest.mark.parametrize(
    "source", ["team", "org", "server_default", None], ids=["team", "org", "server-default-handle", "no-handle"]
)
async def test_an_admin_disabled_provider_or_model_is_refused_on_every_other_key(
    monkeypatch: pytest.MonkeyPatch,
    override_kwargs: dict[str, Any],
    healthy: bool,
    reason: str,
    refused: str,
    source: str | None,
) -> None:
    """TASK-15581: the same on a team's or an org's key and on the server's, with a handle or none."""
    result, calls, warnings = await _refused_summary(
        monkeypatch, override_kwargs, healthy, source=source, config_model="cfg-m"
    )

    assert (result, calls, warnings) == (refused, [], [_refused_line(reason)])


@pytest.mark.parametrize(
    ("error_code", "summary_code"),
    [("provider_disabled", "configuration"), ("credential_store_unavailable", "provider_failure")],
)
def test_a_refusal_at_the_provider_base_class_is_retryable_only_for_an_unreadable_store(error_code, summary_code):
    """TASK-15581 (I2): the audio worker keeps a transcript without its analysis for good on 'configuration', so
    a store blip must come back as the retryable 'provider_failure'."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError

    error = ChatConfigurationError("refused", provider="openai", error_code=error_code)

    assert sgl._summary_error_code(error) == summary_code
```

`test_summarization_adapter.py`, replacing `test_explicit_missing_summary_model_uses_default_model_environment_and_a_denied_one_runs_on_the_server_key` (TASK-15496's assertion, through the real override cache; `_model_allowed_by_override` no longer exists):

```python
@pytest.mark.unit
def test_explicit_missing_summary_model_uses_default_model_environment_and_a_denied_one_is_refused(monkeypatch):
    # TASK-15496 gave a handle call the DEFAULT_MODEL_<PROVIDER> fallback. A fallback model the provider override
    # does not allow is refused with 'configuration' (TASK-15581 restores TASK-15496's refusal).
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    adapter = _Adapter(response={"choices": [{"message": {"content": "Cedar summary"}}]})
    monkeypatch.setenv("DEFAULT_MODEL_MOONSHOT", "server-env-model")
    monkeypatch.setattr(sgl, "get_registry", lambda: _Registry(adapter))
    set_llm_provider_overrides_cache_for_tests(
        {"moonshot": LLMProviderOverride(provider="moonshot", allowed_models=["server-env-model"])}
    )

    assert _analyze_with_empty_moonshot_handle() == "Cedar summary"
    monkeypatch.setenv("DEFAULT_MODEL_MOONSHOT", "blocked-model")
    with pytest.raises(sgl.SummaryProviderError) as exc_info:
        _analyze_with_empty_moonshot_handle()

    assert exc_info.value.code == "configuration"
    assert [request["model"] for request, _timeout in adapter.requests] == ["server-env-model"]
```

`test_llm_quota_decision.py`, replacing `test_a_configured_model_the_override_denies_keeps_the_servers_key_and_its_limit`:

```python
@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "reason"),
    [
        ({"allowed_models": ["x"]}, True, "model_not_allowed"),
        ({"is_enabled": False}, True, "provider_disabled"),
        ({}, False, "credential_store_unavailable"),
    ],
    ids=["model-not-allowed", "provider-disabled", "store-unavailable"],
)
async def test_an_admin_disabled_own_key_keeps_no_handle_and_its_limit(
    monkeypatch, override_kwargs, healthy, reason
) -> None:
    """TASK-15577 / TASK-15581: refused at resolution, so no handle. The request keeps the server's key, where the call
    is refused too (test_provider_adapter_runtime_boundary), and an over-limit user is refused at the boundary. One
    warning names the provider and the reason, and no key."""
    from loguru import logger

    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as runtime_module
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )
    from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_work, users_own_key

    users_own_key(monkeypatch, "openai")
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", lambda: {"openai_api": {"model": "gpt-denied"}})
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", **override_kwargs)}, healthy=healthy
    )
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        assert await usage_quota_deps.users_own_key_credentials(1, "openai") is None
    finally:
        logger.remove(sink)
    assert warnings == [f"Own-key credential resolution for openai failed: {reason}"]
    refuse_llm_work(monkeypatch)
    with pytest.raises(LLMQuotaExceededError):
        await usage_quota_deps.enforce_llm_quota_on_own_key(1, "openai")
```

MCP `test_model_completion_integration.py`, replacing both `test_a_model_the_provider_override_denies_runs_on_the_server_key` and `test_a_disabled_provider_or_an_unreadable_store_refuses_an_own_key_completion` (`:377-424`):

```python
@pytest.mark.parametrize("selected", ["user", "team", "org", "server"])
@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "reason", "domain"),
    [
        ({"allowed_models": ["x"]}, True, "model_not_allowed", ModelFailureDomain.CREDENTIAL_SCOPE),
        ({"is_enabled": False}, True, "provider_disabled", ModelFailureDomain.CREDENTIAL_SCOPE),
        ({}, False, "credential_store_unavailable", ModelFailureDomain.SHARED_INFRASTRUCTURE),
    ],
    ids=["model-not-allowed", "provider-disabled", "store-unavailable"],
)
async def test_an_admin_disabled_provider_or_model_is_refused_on_every_key(
    composed, selected, override_kwargs, healthy, reason, domain
):
    """TASK-15581: after the scope check, on whichever key it chose, a provider or model the admin disabled, or a store
    that cannot be read, fails closed before any reservation or provider request, with the reason as its code (not an
    auth code), and the reason is logged."""
    env = composed
    sources = ("user", "team", "org")
    await seed_keys(env, sources[sources.index(selected) :] if selected != "server" else ())
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": llm_provider_overrides.LLMProviderOverride(provider="openai", **override_kwargs)}, healthy=healthy
    )
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    with pytest.raises(ModelCompletionFailure) as caught:
        await env.build().complete(request(), invocation)
    assert_failure(caught.value, reason, domain)
    assert env.provider.requests == []
    assert await assert_safe_surfaces(env, invocation) is None
    refusals = [line for line, _extra, _exception in env.logs if "Provider override for openai" in line]
    assert len(refusals) == 1
    assert f"Provider override for openai: {reason}; the call is refused" in refusals[0]
```

MCP `test_model_completion_quota_compatibility.py`: delete `test_an_own_key_completion_whose_model_the_override_denies_obeys_the_users_limit` and the now-unused `from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides`.

MCP `conftest.py` (TASK-15577 M2), replacing the fixture that patched three private globals:

```python
@pytest.fixture(autouse=True)
def _healthy_empty_llm_provider_override_store(request):
    """Every test starts with the healthy, empty provider-override store app startup gives production.

    The tldw_Server_API/tests conftest does the same for its tests; this directory is outside it. The MCP
    authoritative path reads the store on every resolution (TASK-15581), and fails closed when it cannot. The
    store's own setter also resets its recovery state, and the store is restored after (TASK-15577 M2). Mark a test
    ``ambient_llm_provider_override_store`` to see the import-time (unloaded, unhealthy) store instead.
    """
    if request.node.get_closest_marker("ambient_llm_provider_override_store") is not None:
        yield
        return
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store

    with store._OVERRIDE_LOCK:
        overrides = dict(store._OVERRIDE_CACHE)
        healthy = store._OVERRIDE_CACHE_HEALTHY
        ttl_disabled = store._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    set_overrides = store.set_llm_provider_overrides_cache_for_tests  # bound now: a test may patch the setter
    set_overrides({})
    try:
        yield
    finally:
        set_overrides(overrides, healthy=healthy, ttl_enabled=not ttl_disabled)
```

`test_provider_credential_runtime.py`: rename `test_authoritative_runtime_checks_the_scope_before_a_denied_model_falls_back` to `test_authoritative_runtime_checks_the_scope_before_the_admin_override`, docstring `"""TASK-15571, TASK-15581: a revoked scope fails as before; the admin override is checked only after it."""`; the body is unchanged (it still passes: the scope fails first).

`test_standalone_html_provider.py` (I1), after `test_unrelated_aggregate_unavailability_does_not_disable_stored_target`:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "code"),
    [
        ({"is_enabled": False}, True, "standalone_html_provider_disabled"),
        ({"allowed_models": ["another-model"]}, True, "standalone_html_model_not_allowed"),
        ({}, False, "standalone_html_provider_unavailable"),
    ],
    ids=["provider-disabled", "model-not-allowed", "store-unavailable"],
)
async def test_an_admin_disabled_provider_or_model_fails_before_network(
    provider_module: ModuleType,
    override_kwargs: dict[str, Any],
    healthy: bool,
    code: str,
) -> None:
    """TASK-15581 (I1): this call is not an adapter call, so it checks the admin provider override itself. A disable
    is a terminal job failure; a store that cannot be read gets the retryable provider-unavailable code."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    target = _target("openai_official_chat_v1")
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", **override_kwargs)}, healthy=healthy
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as exc_info:
        await _call(provider_module, _config(target), httpx.MockTransport(handler))

    assert (exc_info.value.code, calls) == (code, 0)
```

`test_llamacpp_inference_api.py` (fold round 2, I1), appended; the stubbed manager's `run_inference` fails the test if it is reached:

```python
@pytest.mark.parametrize(
    ("override_kwargs", "healthy", "status", "code"),
    [({"is_enabled": False}, True, 403, "provider_disabled"), ({}, False, 503, "credential_store_unavailable")],
    ids=["provider-disabled", "store-unavailable"],
)
def test_llamacpp_inference_refuses_an_admin_disabled_provider(override_kwargs, healthy, status, code) -> None:
    """TASK-15581: this route posts to the managed server itself, not through an adapter, so it checks the admin
    provider override: a disabled llama.cpp is refused for every caller before anything runs, and a store that
    cannot be read is a 503."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    class _Mgr(_DefaultMgr):
        async def run_inference(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("a refused call must not run")

    set_llm_provider_overrides_cache_for_tests(
        {"llama.cpp": LLMProviderOverride(provider="llama.cpp", **override_kwargs)}, healthy=healthy
    )
    with TestClient(_make_app_with_manager(_Mgr())) as client:
        response = client.post("/api/v1/llamacpp/inference", json={"messages": [{"role": "user", "content": "hello"}]})

    assert (response.status_code, response.json()["detail"]["error_code"]) == (status, code)
```

- [ ] **Step 2: Run them on Task 2's source.**

```bash
for f in tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py \
         tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py \
         tldw_Server_API/tests/Usage/test_llm_quota_decision.py \
         tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py \
         tldw_Server_API/tests/Slides/test_standalone_html_provider.py \
         tldw_Server_API/tests/LLM_Local/test_llamacpp_inference_api.py; do
  PYTEST "$f" | tail -1
done
```

Expected (measured):
- boundary 22 failed, 89 passed (own key: all 9 cases, since model-not-allowed falls back to the server key and provider-disabled and store-unavailable return the old "credential context" text; other keys: all 12 team, org, server-default-handle and no-handle cases, which fall back or run unchecked; and `_summary_error_code`'s store case, which is `configuration`);
- summarization 1 failed, 127 passed (the denied fallback model runs on the server key);
- decision 3 failed, 35 passed (the warning still ends "; the server key is used");
- MCP integration 9 failed, 82 passed (model-not-allowed on all 4 keys re-resolves onto the server key; provider-disabled fails on all 4, with `invalid_provider_credentials` on the 3 stored keys and running on the server key; store-unavailable on the server key runs);
- Slides provider 86 failed, 65 passed, 3 errors: the 84 probe-blocked baseline failures, plus the new test's provider-disabled and model-not-allowed cases, which reach the request; all 3 new cases also error at teardown on the repo's own live-network blocker (the request resolves `api.openai.com`). The store-unavailable case's assertion passes there only because the blocked DNS lookup maps to the same retryable code;
- llama.cpp inference 2 failed, 5 passed (both new cases: the route ran the stubbed server, whose `run_inference` raised, and answered 500 with the string detail "An unexpected error occurred.").

The same counts on unchanged `origin/dev` source: the guard does not reach these tests' stand-in adapters, and Slides never reaches an adapter. 0 netprobe lines for every new or changed test.

- [ ] **Step 3: The reverts.**

```bash
git diff b1a32fbba4 b1a32fbba4^1 -- tldw_Server_API/app/core/Usage/llm_quota_gate.py \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/app/core/Evaluations/eval_runner.py \
  tldw_Server_API/app/core/Evaluations/ms_g_eval.py | git apply
git diff b1a32fbba4^1 --stat -- tldw_Server_API/app/core/Usage/llm_quota_gate.py \
  tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/app/core/Evaluations/eval_runner.py \
  tldw_Server_API/app/core/Evaluations/ms_g_eval.py
```

Expected (measured): the reverse patch applies (Task 1's `main()` sits in another hunk). Each file then equals `b1a32fbba4^1` plus the commits merged after #3523 (none today; #3071 would add its `eval_runner` hunk), and `media_ingest_jobs_worker.py` adds Task 1's `main()`: today the stat shows only `media_ingest_jobs_worker.py | 11 +++++++++--` (fold 3, M6).

- [ ] **Step 4: Delete the fallback helpers** (`server_key_fallback`, `log_server_key_fallback`, `override_outcome`) from `llm_provider_overrides.py`, leaving `enforce_provider_override`.

- [ ] **Step 5: The analyze path.** In `_summarize_via_adapter`, replace the block from `outcome = None` through the `if outcome == "server_key":` branch with:

```python
    # Imported here: a module-level import would add AuthNZ modules to the LLM_Calls load ratchet.
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import enforce_provider_override

    try:
        # Refused on every key (TASK-15581). The provider base class refuses it too; here it keeps this path's
        # failure codes, and comes before a stream is handed back.
        enforce_provider_override(provider, model)
    except ByokResolutionError as exc:
        store_down = exc.code == "credential_store_unavailable"  # retryable: a job tries again once it loads
        return _summary_failure(
            code="provider_failure" if store_down else "configuration",
            provider=provider,
            raise_on_error=raise_on_error,
            legacy_message=(
                "Error: Provider credential storage is temporarily unavailable."
                if store_down
                else "Error: The provider or model is disabled by the administrator."
            ),
        )
```

and in `_summary_error_code`, the `ChatConfigurationError` branch (a refusal at the provider base class) becomes (fold 3, I2):

```python
    if isinstance(exc, ChatConfigurationError):
        # An unreadable provider-override store passes once it loads (TASK-15581); the rest needs a config change.
        return "provider_failure" if exc.error_code == "credential_store_unavailable" else "configuration"
```

The audio worker retries only `rate_limit` and `provider_failure` (`audio_jobs_worker.py:28`, `:553-566`) and keeps a transcript without analysis for good on anything else, so a store blip must not be `configuration`.

- [ ] **Step 6: MCP.** In `_resolve_entry`, replace the `if (self._authoritative_scope is not None and ... resolution.uses_byok):` block (through the server-key re-resolution) with:

```python
            if self._authoritative_scope is not None:
                # Imported here: the override module imports the adapter registry, which imports this module.
                from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
                    enforce_provider_override,
                )

                # After the scope check, on whichever key it chose: an admin-disabled provider or model is
                # refused (TASK-15581). This scope captures no override snapshot, so nothing else checks it.
                enforce_provider_override(provider, model)
```

In `MCP_unified/adapters/model_completion/adapter.py` (`_run`'s `except ByokResolutionError`, `:376-383`), the failure code is the policy reason (fold 3, I1):

```python
            except ByokResolutionError as error:
                failure = _failure(
                    # An admin override's reason ('provider_disabled', 'model_not_allowed'), not an auth code.
                    getattr(error, "policy_code", None) or error.code,
                    (
                        ModelFailureDomain.SHARED_INFRASTRUCTURE
                        if error.code == "credential_store_unavailable"
                        else ModelFailureDomain.CREDENTIAL_SCOPE
                    ),
                )
```

- [ ] **Step 7: The boundary's warning.** In `users_own_key_credentials`' `except`, add the comment line `# The request keeps the server's key, where an admin-disabled provider or model is refused too (TASK-15581).` and change the message to `"Own-key credential resolution for {} failed: {}"` (drop `"; the server key is used"`).

- [ ] **Step 8: Slides standalone HTML (I1).** In `standalone_html_provider.py`, before `_load_current_config`:

```python
def _verify_admin_override(stored_target: ResolvedExecutionTarget) -> None:
    """Refuse a provider or model the admin provider override disables, on any key (TASK-15581).

    This call is not an adapter call, so the provider base class never sees it. A refusal is terminal; an
    override store that cannot be read is retried, as a provider outage is.
    """
    # Imported here: the override module imports the adapter registry, which this module's importers load.
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import enforce_provider_override

    try:
        enforce_provider_override(stored_target.provider, stored_target.model)
    except ByokResolutionError as exc:
        policy_code = getattr(exc, "policy_code", None)
        _fail(f"standalone_html_{policy_code}" if policy_code else "standalone_html_provider_unavailable")
```

and in `generate_standalone_html`, right after `current = _verify_current_target(stored_target, current)`, add `_verify_admin_override(stored_target)`. The job worker already terminalizes every provider code except `standalone_html_provider_timeout`, `standalone_html_provider_unavailable` and `standalone_html_provider_http_error`, which it retries (`standalone_html_generation_jobs_worker.py:916-924`): so a disable is a terminal failure and a store outage is retried, with no change to `process_standalone_html_generation_job` (capped at 425 lines, its size).

- [ ] **Step 9: llama.cpp inference (fold round 2, I1).** In `endpoints/llamacpp.py`, after `from tldw_Server_API.app.api.v1.schemas.llamacpp_schemas import LlamaCppInferenceRequest`:

```python
def _refuse_admin_disabled_llamacpp() -> None:
    """Refuse inference when the admin provider override disables llama.cpp (TASK-15581).

    This route posts to the managed server itself, not through an adapter, so the provider base class never sees it.
    It names no model (it runs the loaded one), so only ``provider_disabled`` applies. 403, or 503 for a store that
    cannot be read.
    """
    from tldw_Server_API.app.core.LLM_Calls.provider_results import (
        ADMIN_REFUSAL_STATUS,
        PROVIDER_STREAM_ERROR_MESSAGES,
    )
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import enforce_provider_override

    try:
        enforce_provider_override("llama.cpp", None)  # every stored spelling folds to this canonical name
    except ByokResolutionError as error:
        code = getattr(error, "policy_code", None) or error.code
        detail = {"error_code": code, "message": PROVIDER_STREAM_ERROR_MESSAGES[code]}
        raise HTTPException(status_code=ADMIN_REFUSAL_STATUS[code], detail=detail) from None
```

and make `_refuse_admin_disabled_llamacpp()` the first statement of `run_llamacpp_inference_endpoint`, after its docstring and before `try:`. Inside the `try`, its `except Exception` would answer 500. The provider name is the canonical one: `canonical_provider_name` maps `llamacpp`, `llama-cpp`, `llama_cpp` and `llama` to `llama.cpp`, and `fold_provider_credential_rows` stores every override row under it.

- [ ] **Step 10: Run again.** Same loop as Step 2, plus `test_model_completion_quota_compatibility.py`, `test_provider_credential_runtime.py`, `test_llm_quota_byok_calls.py`, `test_standalone_html_generation_jobs.py`, `test_standalone_html_generation.py`, `MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py` and `tests/LLM_Local`.

Expected (measured): boundary 111 passed; summarization 128; decision 38; MCP integration 91; Slides provider 84 failed, 67 passed, with the same 84 failing ids as on `origin/dev`; MCP quota compatibility 7; credential runtime 59; `test_llm_quota_byok_calls.py` 132; Slides generation jobs 113; Slides generation 2; media ingest refusal 8; llama.cpp inference 7; `tests/LLM_Local` 464 passed, 1 skipped. 0 netprobe lines for every new or changed test.

- [ ] **Step 11: Commit.**

```bash
git add tldw_Server_API/app tldw_Server_API/tests
git commit -m "fix(llm): remove the denied-model server-key fallback (TASK-15581)

The owner reversed TASK-15571's rule: an admin-disabled provider or model is
refused on every key. The analyze path, the evaluation judges and the MCP
completion tool refuse instead of re-running on the server key (a store outage
is retryable on analyze, and MCP reports the reason), Slides
standalone-HTML generation checks the override before its own request, the
llama.cpp inference route refuses a disabled llama.cpp for every caller, and
the quota gate and media ingest worker return to their state before #3523.
Closes TASK-15577 (M2: the MCP tests' store fixture uses the store's setter).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Docs

**Files:**
- Modify: `Docs/ADR/086-authoritative-provider-credential-scope.md` (append an amendment after `:33`)
- Modify: `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md` (an amendment at the end of Consequences, after "**Still unrecorded:**")
- Modify: `Docs/Operations/Usage_Quotas.md` (`:120`, the sentences #3523 added)
- Modify: `Docs/User_Guides/Server/BYOK_User_Guide.md` (`:97`, the sentences #3523 added)
- Modify: `Docs/Architecture/llm-providers.md` (`:33`, and the failure table `:44-55`)
- Modify: `Docs/Deployment/Sidecar_Workers.template.md` (after the `TLDW_WORKERS_SIDECAR_MODE` line); generated: `Docs/Deployment/Sidecar_Workers.md`, `Docs/Published/Deployment/Sidecar_Workers.md`
- Modify: `Docs/superpowers/plans/2026-11-19-denied-model-server-key-fallback.md` (#3523's plan: a superseded note under its title)
- Modify (generated): `Docs/Published/ADR/067-...md`, `Docs/Published/ADR/086-...md`, `Docs/Published/User_Guides/Server/BYOK_User_Guide.md`, `Docs/Published/Deployment/Sidecar_Workers.template.md`

- [ ] **Step 1: ADR-086.** Append after the TASK-15571 amendment:

```markdown

Amendment (TASK-15581, 2026-11-19): the owner reversed the TASK-15571 rule, so the amendment above is withdrawn. The owner's rule: "if a user tries to use a model/service thats been explicitly disabled by the admin, they shouldn't be able to use it; if its just been disabled as in 'no credits/key enabled' then tehy should be able to use it via BYOK". An explicit disable is the admin provider override's `is_enabled = false` (`provider_disabled`), or a model left off its `allowed_models` when that list is set (`model_not_allowed`). After every supplied scope is validated, this runtime refuses such a call with its policy error on whichever credential the scope chose: the user's, a team's, an organization's or the server's. Nothing falls back to the server credential. An override store that cannot be read still fails closed with `credential_store_unavailable`. A provider the server has no key or credits for is not a disable, and a user's own credential still serves it. Every provider adapter call meets the same check in the adapter base class (ADR-067), for its model and each OpenRouter fallback model, before the quota check. The chat calls that do not use an adapter check it themselves: this runtime for the MCP completion tool, the Messages API's native path at resolution, Slides standalone-HTML generation before its request, and the llama.cpp inference endpoint before it runs the loaded model (a provider disable only, since that endpoint names no model). So no LLM chat path runs a disabled provider or model. Embeddings check the override at their endpoint. Image generation, speech-to-text and the OCR backends are outside this rule. Outside the app process, each standalone worker or CLI that can make an LLM call loads the override store when it starts.
```

- [ ] **Step 2: ADR-067 (M8).** After the "**Still unrecorded:**" bullet, at the end of Consequences:

```markdown

Amendment (TASK-15581, 2026-11-19): the same wrappers also enforce the admin provider policy. Before the quota check, they refuse a provider or model the admin provider override disables, on every key, with a `ChatConfigurationError` that carries the reason as its `error_code`. A call in an exempt context (a key test or the setup check) is not refused. ADR-086's TASK-15581 amendment states the rule.
```

- [ ] **Step 3: Usage_Quotas.md.** In the paragraph at `:120`, replace everything from " A call whose model the admin provider override does not allow" through "...counts and checks the call as a server-key call (TASK-15571)." with:

```markdown
 A call to a provider the admin provider override turns off, or to a model outside its `allowed_models`, is refused on every key: the user's own, a team's, an org's and the server's, wherever its model came from (the provider's configured model, `DEFAULT_MODEL_<PROVIDER>`, or a model the request names). So is any call while the override store cannot be read: the server and every standalone worker or evaluation CLI that can make an LLM call load the store when they start, and a process that cannot read it refuses every LLM call, whatever its key. A call that names no model is checked for `provider_disabled` only. The refusal comes before the call's limit is checked, so it is not counted. The server logs one warning per refused call, naming the provider and the reason (`model_not_allowed`, `provider_disabled` or `credential_store_unavailable`), never a key. A provider the server has no key or credits for is not turned off: a user's own key still serves it (TASK-15581).
```

- [ ] **Step 4: The BYOK guide.** Replace " One of those calls whose model the operator's provider override does not allow runs on the server's key instead: it counts, and over a limit, or when the limit cannot be checked, it is refused. The study assistants are the exception: like chat, they refuse such a call." with:

```markdown
 A provider or model the operator has turned off in its provider override is refused whatever key the call uses, your own included. A provider the operator only has no key for is not turned off: your own key still serves it (the WebUI's model picker does not list such a provider yet, TASK-15583).
```

- [ ] **Step 5: llm-providers.md (M5).** Step 7 of the call flow becomes:

```markdown
7. Every adapter call is wrapped by the accounted base class: admin-override check → quota check → call → record. A provider or model the admin override disables is refused on every key (TASK-15581). Streaming feeds usage from provider `usage` blocks when present, else estimates.
```

and add this row after the "Quota exceeded" row of the failure table:

```markdown
| Provider or model the admin override disables, or an override store that cannot be read | `ChatConfigurationError` (`error_code` `provider_disabled`, `model_not_allowed` or `credential_store_unavailable`), before the quota check; RAG and the evaluations endpoints keep the code and answer 403 (503 for the store), and an endpoint that maps nothing gets the same from the app's `ChatConfigurationError` handler |
```

- [ ] **Step 6: The sidecar-workers guide (B1).** In `Docs/Deployment/Sidecar_Workers.template.md`, after the `TLDW_WORKERS_SIDECAR_MODE` line:

```markdown

A worker that can make an LLM call (audio, media ingest, data tables, Prompt Studio, long-form TTS, reading-list import with LLM claims extraction and the evaluations A/B test worker) loads the admin LLM provider overrides from the AuthNZ database when it starts, and keeps them fresh. It needs the same `DATABASE_URL` as the API process. It refuses to start if it cannot load them, and while it cannot read them it refuses every LLM call (TASK-15581).
```

then regenerate: `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/Deployment/generate_sidecar_files.py`.

Expected (measured): it rewrites `Docs/Deployment/Sidecar_Workers.md` and `Docs/Published/Deployment/Sidecar_Workers.md` only (no compose, systemd or launchd change).

- [ ] **Step 7: #3523's plan.** Under its title, add:

```markdown
> **Superseded (2026-11-19):** the owner reversed this rule. An admin-disabled provider or model is refused on every key and path, and nothing falls back to the server key: TASK-15581, `2026-11-19-refuse-admin-disabled-provider.md`.
```

- [ ] **Step 8: The changelog (fold 3, M7).** Under `## [Unreleased]`, `### Changed`:

```markdown
- **An admin-disabled provider or model is refused on every key and path.** A provider the admin provider override turns off, or a model off its `allowed_models` (each OpenRouter fallback model included), is refused for every LLM call: on the user's own key, a team or org key and the server's, in chat and in every server-side feature. This reverses the server-key fallback of #3523 (TASK-15581). **Before upgrading:** an existing disable or allowlist now also stops server-side features (RAG generation, claims, workflows, evaluation judges, summarization) configured with that provider or another model. Standalone workers and the `tldw-evals` LLM commands load the overrides from the AuthNZ database at start, so they need the API process's `DATABASE_URL`.
```

- [ ] **Step 9: The mirror.** `bash Helper_Scripts/refresh_docs_published.sh`.

Expected (measured): exit 0; `git status --short Docs/` shows the eight sources and five `Docs/Published` files: ADR-067, ADR-086, the BYOK guide, `Deployment/Sidecar_Workers.md` and `Deployment/Sidecar_Workers.template.md`. `Usage_Quotas.md`, `llm-providers.md` and the superpowers plans are not mirrored.

- [ ] **Step 10: tests/Docs.**

```bash
PYTEST tldw_Server_API/tests/Docs
```

Expected (measured): not re-run in fold round 3: the run stopped on `OSError: [Errno 28] No space left on device` (the machine's disk filled). Fold round 2, before this round's doc edits: 214 passed, 0 netprobe lines

- [ ] **Step 11: Item 6, recorded (no code).** The measurement behind D3 and TASK-15583, to quote in the PR: with no server key and a healthy store, `get_configured_providers()` lists `openai` and `anthropic` as `is_configured: False, provider_enabled: False, availability: "not-configured", readiness_reason_code: "provider_not_configured", models: []`, and the same with an override `is_enabled=True` (only `enabled: True` changes); discovery is not called. `TldwModels.isSelectableChatModel` drops such models (`TldwModels.ts:754-760`). The API side works: `test_the_users_own_key_serves_a_provider_the_server_has_no_key_for` (Task 2), the existing `test_byok_resolution_precedence` (user before team, org and server, `AuthNZ_SQLite/test_byok_runtime_sqlite.py:221`), and chat resolving the handle before its missing-key check (`chat.py:4680-4745`).

- [ ] **Step 12: Commit.**

```bash
git add Docs CHANGELOG.md
git commit -m "docs: an admin-disabled provider or model is refused on every key (TASK-15581)

Amends ADR-086 with the owner's reversal of TASK-15571 and ADR-067 with the
admin-policy check in its wrappers; replaces the server-key fallback text in
Usage_Quotas.md, the BYOK guide and the LLM provider architecture page; says
in the sidecar-workers guide that a worker loads the override store at start;
adds the changelog entry with its upgrade note.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Verification, close-out, review, PR

- [ ] **Step 1: The wider set.**
  - **Round 0's set** (157 files): the 144 files of `git grep -l -E 'set_llm_provider_overrides_cache_for_tests|LLMProviderOverride\(|_OVERRIDE_CACHE_HEALTHY|credential_store_unavailable|ambient_llm_provider_override_store|override_outcome|server_key_fallback|enforce_provider_override|user_to_check|check_user_sync|check_user_async|usage_accounting|instrument_provider_class|_summarize_via_adapter|Summarization_General_Lib|users_own_key_credentials|_admit_or_refuse_analysis|AuthoritativeProviderScope|ms_g_eval|eval_runner|prompt_studio.test_runner|ChatConfigurationError\(.*error_code'` over `tldw_Server_API/tests` and the MCP tests, plus 13 key-test, admin-provider, setup and catalog files, plus the whole MCP test directory (its conftest changes).
  - **This round's additions** (68 files): `git grep -l -E 'exception_handlers|standalone_html_provider|media_ingest_jobs_worker|audio_jobs_worker|Data_Tables\.jobs_worker|Data_Tables import jobs_worker|embeddings_abtest_jobs_worker|Explainer\.jobs_worker|Explainer import jobs_worker|cli\.evals_cli|evals_cli_enhanced|Evaluations\.cli\.evals_cli|ChatConfigurationError\('` over `tldw_Server_API/tests`.

  - **Fold round 2's additions:** `tests/LLM_Local`, `tests/LLamaCpp`, and the files `git grep -l -E 'endpoints\.llamacpp|endpoints import llamacpp|llamacpp/inference|reading_import'` finds outside `tests/lint` (`Admin/test_admin_event_loop_offload.py`, `AuthNZ_Unit/test_llamacpp_permissions_claims.py`, `Services/test_router_groups_contract.py`, `Services/test_startup_heavy_init.py` and four `Collections` reading tests).

  Run each on the branch, re-run the failing ids on `origin/dev`, and diff them.

Expected (measured): Rounds 0 and 1, measured on `93f5aaa684`: Round 0's set plus the MCP directory, on the branch: 12 failed, 9929 passed, 40 skipped, 8 errors. The 20 failing ids are exactly round 0's, which fail the same way on `origin/dev` (measured in round 0 on a full `origin/dev` run of the 157 files): `test_byok_runtime_sqlite.py` (2 failed, 2 teardown errors), `test_byok_helpers.py` (3 + 3), `test_messages_provider_override_store_boundary.py` (3 + 3) and `test_embeddings_orchestrator_endpoint_parity.py` (2), all "live network attempted" from an earlier test's egress worker thread (`egress.py:419`); and the 2 known `test_jobs_postgres_exports_recorded_coverage` cases (no `coverage`). Run alone on the branch, the four leaking files pass (217). The probe logged 2 lines, the same Embeddings leak as on `origin/dev`. `test_model_completion_contracts.py` runs alone with `-p no:xdist` (its random-UUID ids differ between xdist workers): 39 passed. This round's 68 files, on the branch: 87 failed, 2845 passed, 49 skipped: the 84 probe-blocked `test_standalone_html_provider.py` cases and 3 in `test_standalone_html_provider_model_inventory.py`, which fail with the same ids on `origin/dev` (they are in `tests/live_network_baseline.txt`; the probe logged 88 lines, all from those tests). Fold round 2's additions, on `f8f141b82e`: `tests/LLM_Local` 464 passed, 1 skipped; `tests/LLamaCpp` with the eight other files (and `Utils/test_sidecar_manifest_defaults.py`) 331 passed, 3 skipped; the four existing CLI test files 28 passed. 0 netprobe lines. Fold round 3: the 238 test files that name a symbol this round changed were not run: the machine's disk filled (the run stopped on `No space left on device`). This round's changed test files are measured above, and the 45 `reset_db_pool` files under M2.

- [ ] **Step 2: Postgres.** The PostgreSQL files in that set run serially (`-p no:xdist`) through the repo fixtures with a session-unique `TEST_DB_NAME`: `Evaluations/unit/test_eval_runner.py`, `AuthNZ_Postgres/test_byok_oauth_endpoints_pg.py`, `Evaluations/test_evaluations_core_hardening.py`, `Evaluations/test_run_progress_response.py`, `MediaIngestion_NEW/integration/test_email_authenticated_access.py`, `Admin/test_admin_byok_validation_jobs.py`, `DB_Management/test_media_ingest_worker_content_scope.py` and `MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`.

Expected (measured): With the repo's PostgreSQL fixtures (the shared test cluster on :5432, a session-unique `TEST_DB_NAME`), serially: 273 passed, 0 skipped, 0 netprobe lines (round 1, on `93f5aaa684`; fold round 2 changes none of these files or the code they drive). Not re-run in fold round 3, which changes `reset_db_pool` (the disk filled). The fold-3 check ran this set with no local cluster: 224 passed, 49 skipped, 0 failed. Run the 49 PostgreSQL cases if the cluster on :5432 is up; never start Docker for them. Otherwise CI's PostgreSQL jobs are the measure, and the controller checks them before merge.

- [ ] **Step 3: tests/lint and tests/CI.**

```bash
PYTEST tldw_Server_API/tests/lint --ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py
PYTEST tldw_Server_API/tests/CI
```

Expected (measured): Fold round 2, at `f8f141b82e`, before this round's changes: `tests/lint` 329 passed and 2 failed in one `-n 2` run, both an intermittent native abort in `test_rg_route_map_lint.py` that also hits `origin/dev` (7 of 7 pass serially); `tests/CI` 2 failed (the known `test_jobs_postgres_exports_recorded_coverage` cases, no `coverage`), 1078 passed, 6 skipped. Not re-run in fold round 3 (the disk filled). `ruff check` on every changed Python file reports nothing that `origin/dev`'s versions of those files do not already report (25 findings on both, in `main.py`, `evaluations_unified.py`, `health.py` and the files of round 2); the new `cli/utils/llm_store.py` is clean.

- [ ] **Step 4: The gates.**

```bash
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_backlog_task_ids.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/scope_predicate_ratchet.py
uvx bandit -q -r tldw_Server_API/app/core/LLM_Calls/usage_accounting.py \
  tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py \
  tldw_Server_API/app/core/Provider_Credentials/provider_credential_runtime.py \
  tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/app/core/exceptions.py \
  tldw_Server_API/app/api/v1/API_Deps/usage_quota_deps.py tldw_Server_API/app/api/v1/utils/exception_handlers.py \
  tldw_Server_API/app/core/Slides/standalone_html_provider.py \
  tldw_Server_API/app/core/Evaluations/eval_runner.py tldw_Server_API/app/core/Evaluations/ms_g_eval.py \
  tldw_Server_API/app/core/Usage/llm_quota_gate.py tldw_Server_API/app/services/media_ingest_jobs_worker.py \
  tldw_Server_API/app/services/audio_jobs_worker.py tldw_Server_API/app/core/Data_Tables/jobs_worker.py \
  tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py tldw_Server_API/app/core/Explainer/jobs_worker.py \
  tldw_Server_API/cli/evals_cli.py tldw_Server_API/app/core/Evaluations/cli/evals_cli.py \
  tldw_Server_API/app/core/Evaluations/cli/evals_cli_enhanced.py \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/core/Collections/reading_import_jobs_worker.py \
  tldw_Server_API/cli/utils/llm_store.py tldw_Server_API/app/core/LLM_Calls/provider_results.py \
  tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/app/main.py \
  tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py \
  tldw_Server_API/app/services/startup_auth.py tldw_Server_API/app/api/v1/endpoints/health.py \
  tldw_Server_API/app/core/AuthNZ/database.py tldw_Server_API/app/core/MCP_unified/adapters/model_completion/adapter.py
git diff --check origin/dev...HEAD
```

Expected (measured): `test_quality_triage.py --enforce`, `check_shard_coverage.py` ("no newly-unshared test files"), `check_backlog_task_ids.py` and `scope_predicate_ratchet.py` ("baseline exact") exit 0. (A first draft of the `python -m` test filled a list from a nested function and raised `tautology_suspect` for `test_llm_provider_overrides.py` from 2 to 3, as Task 1's first worker test once did; the version above raises instead and keeps the baseline.) Bandit on the 31 files: two Low B105 findings, both at unchanged lines of `eval_runner.py` (`:2350`, `:2376`, as on `origin/dev`), none in a changed line. `git diff --check`: clean.

- [ ] **Step 5: Function sizes.** No changed function is in `function_size_baseline.json`. Measured (origin/dev -> branch): `_summarize_via_adapter` 221 -> 218, `_summary_error_code` 29 -> 30, `ProviderCredentialRuntime._resolve_entry` 152 -> 134, `users_own_key_credentials` 39 -> 40, `eval_runner._call_adapter_text` 56 -> 48, `ms_g_eval._call_adapter_text` 69 -> 50, `user_to_check` 26 -> 23, `check_user_sync` 39 -> 37, `check_user_async` 38 -> 34, `_admit_or_refuse_analysis` 28 -> 24, `_wrap_chat`/`_wrap_achat` 22 -> 23, `_wrap_stream` 24 -> 25, `_wrap_astream` 41 -> 42, `_schedule_override_recovery` 66 -> 71, `generate_standalone_html` 54 -> 55, `run_llamacpp_inference_endpoint` 65 -> 66, `reading_import_jobs_worker.main` 18 -> 6 (its body is now `run_reading_import_jobs_worker`, 18), `classify_rag_provider_error` 31 -> 33, `_eval_provider_http_exception` 53 -> 58, `ModelCompletionAdapter._run` 70 -> 71, `api_health` 96 -> 110, `reset_db_pool` 74 -> 75; `global_unhandled_exception_handler` and the three CLI group functions are unchanged. New: `_refuse_admin_disabled` 26, `enforce_provider_override` 16, `standalone_llm_provider_overrides` 14, `standalone_llm_provider_overrides_sync` 31, `mark_llm_provider_overrides_stale` 10, `_verify_admin_override` 15, `_refuse_admin_disabled_llamacpp` 20, `LLMStoreGroup.resolve_command` 17, `admin_refusal_exception_handler` 16, `main._admin_refusal_exception_handler` 7, each worker `main` 6. `process_standalone_html_generation_job` is unchanged (425, its cap); the only baseline entry in a touched module, `evaluations_unified:batch_evaluate`, is untouched.

- [ ] **Step 6: Close-out (backlog-py).** The TASK-15577 edit assumes D1 (a): it replaces AC #1 with the corrected text, which backlog-py appends as the new #3 (old #2 and #3 become #1 and #2). Under D1 (b), drop `--remove-ac 1` and the `--ac`.

```bash
BACKLOG task edit TASK-15581 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured RED/GREEN per file, wider set, tests/Docs, tests/lint, tests/CI, gates, Bandit>" \
  --final-summary "An admin-disabled provider or model is refused on every key and path: the provider base class checks every adapter call before the quota check; the analyze path, the MCP completion tool, Slides standalone-HTML generation and the llama.cpp inference route keep their own check; each OpenRouter fallback model is checked too; RAG, evaluations, MCP and an endpoint that maps nothing keep the reason, 403 (503 for an unreadable store), and a store outage is retryable on analyze; PR #3523's fallback is removed. Every standalone worker and evaluation CLI that can make an LLM call loads the override store at start, and an AuthNZ pool reset keeps it loaded. A provider with no server key still serves a user's own key. ADR-086 and ADR-067 amended; docs updated; the WebUI listing gap is TASK-15583, an explicit disallowed state TASK-15582." \
  -s Done
BACKLOG task edit TASK-15576 \
  --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Owner, 2026-11-19: keep refusing; explicit disable is refused everywhere. No runtime adopts the fallback; TASK-15581 removed it and refuses at the provider base class too, closing the Prompt Studio DEFAULT_MODEL_* gap. The docs say every endpoint refuses." \
  -s Done
BACKLOG task edit TASK-15577 \
  --remove-ac 1 \
  --ac "A disabled provider and an unreadable override store refuse an own-key analyze request instead of running it on the server key, with tests (corrected 2026-11-19, TASK-15581 D1: refused at dispatch, as for a user with no stored key)" \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Done in TASK-15581: a disabled provider or an unreadable store no longer runs an own-key analyze request on the server key (refused at dispatch, D1 a); other failures keep the server key; M1 moot (no fallback), M2 fixed (MCP store fixture)." \
  -s Done
git add -A backlog/tasks
git commit -m "chore(backlog): close TASK-15581, TASK-15576 and TASK-15577

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Final review.** A clean-session opus review of the whole branch (`/superpowers:requesting-code-review`, in place of Qodo), with Review Focus above. Fold in every Critical and Important finding, re-run the affected files, and record the review in the PR body.

- [ ] **Step 8: The PR.** Push to `fix/admin-disabled-provider-refused-15581`, then `gh pr edit 3537 --body-file <plans dir>/body_15581.md` and `gh pr ready 3537`. The body has: Why (the owner's rule, verbatim, and the reversal of #3523), What changes (the choke point; the standalone bootstrap and its entry-point list; the deletions table; the docs), Measured (the counts above), Known limits (D2, D4, Q1's answer, the no-model gap, a custom endpoint's own routing, TASK-15583, TASK-15582), a "Before upgrading" line for the release inventory (an existing disable or allowlist now also stops server-side features configured with that provider or another model; standalone workers and `tldw-evals` LLM commands need the AuthNZ `DATABASE_URL`; a CLI LLM call made with a user in context is now recorded and quota-checked, so such a user over a limit gets the 402), Reviews, a full `## Change summary` section reading "ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-19.", and ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Merges go through the controller's queue (MERGE_QUEUE unset).
