# Move Chat Provider Defaults and API Keys out of the Schemas Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `api/v1/schemas/chat_request_schemas.py` stops loading config at import, the default provider is resolved at call time by `core/config_llm_providers.py` (which `.13` creates for API keys and the provider lists), every importer reads it there, and two guards keep it that way.

**Architecture:** Ratchet first, then cut. Slice 1 adds an AST guard that fails when a module under `api/v1/schemas` calls a `core.config*` function at import time, and a module-keyed ratchet on core importers of `chat_request_schemas`; both seed today's offenders. Slice 2 deletes the dead `API_KEYS` maps, appends `config_default_llm_provider` and `get_default_provider` (moved from `core/Chat/chat_target_resolution.py`) to `.13`'s `core/config_llm_providers.py` (layer `foundation` by the `core.config*` pattern), retargets every runtime importer of `DEFAULT_LLM_PROVIDER` and of the schema's `get_api_keys`/`ALL_SUPPORTED_PROVIDER_NAMES_LIST` re-exports, and deletes those re-exports and the schema's import-time reads. Nothing is shimmed: no importer or patch target is left, and a stale patch must fail loudly. Both seeds and the layering baseline shrink in the same PR.

**Tech Stack:** Python 3.12/3.13, FastAPI, Pydantic v2, pytest, AST lint tests under `tldw_Server_API/tests/lint/` sharing `_ast_cache.parse_tree`.

**Spec:**
- Backlog task: `backlog/tasks/task-13950.14 - Move-chat-provider-defaults-and-API-keys-out-of-the-schemas-package.md` (ACs #1-#4 bind).
- Roadmap row: `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`, `## Phase 1`, row `.14` (effort S, 1 PR, after `.13` is Done).
- Dependency plan: `Docs/superpowers/plans/2026-10-26-task-13950-13-llm-gateway.md` (`.13`), Task 3 (PR C1) and Task 4 (PR C2).

## Global Constraints

Copied from the roadmap's `## Global Constraints` where they bind this task:

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
- **Line-keyed legacy baseline.** `Helper_Scripts/ci/scope_predicate_baseline.txt` is still keyed by line number until `.23` re-keys it. Any PR that shifts lines in a file it lists regenerates it with `--write-baseline`. (Measured: it lists none of this task's files.)
- **Shims.** A move keeps a re-export shim at the old import path until `git grep` shows no importer and no string monkeypatch target still uses it.
- **Postgres proof.** Postgres-dependent acceptance criteria are proven in the Postgres shard with `TLDW_TEST_POSTGRES_REQUIRED=1`. A local skip is not evidence. (No AC here touches a database.)
- **Flipping a global default** (moderation per-surface flags, production detection, middleware stack, pool backend): `git grep` the symbol across all tests and run every hit. (Applied here: the default provider changes from import-time to call-time.)
- **Review gates.** Each AI-authored PR carries a `## Change summary` (ADR-004) and gets a clean-subagent `/superpowers:requesting-code-review` before merge. Each task's own plan gets an independent review plus a dry run before execution.

Task-specific:
- **Owner waiver.** The owner waived the ADR-004 Change summary for TASK-13950 PRs (2026-10-25) and authorized merging them after green CI plus a clean review.
- **Backlog.** TASK-13950 task files are not on `dev`; these PRs edit nothing under `backlog/`. Status, AC checks and the final summary are recorded on the docs branch by the orchestrator with `tools/backlog-py`.
- **Frozen functions.** Four edited functions are in `function_size_baseline.json`: `core.Data_Tables.jobs_worker:_handle_job` (318), `core.Streaming.speech_chat_service:run_speech_chat_turn` (563), `api.v1.endpoints.audio.audio_streaming:websocket_audio_chat_stream` (1761) and `api.v1.endpoints.chat:create_chat_completion` (2870 on `origin/chore/13950-8-size-ratchets`; `dev`'s copy says 2865 and already fails against the 2870-line function). Each edit is a same-line name swap; none may add a line, whatever the slice base's ceiling is.
- **Foundation pin.** `core/config_llm_providers.py` imports only foundation modules (`core.config`, `core.custom_openai_providers`, `core.testing`, stdlib). `get_default_provider` must not gain an import of AuthNZ, Chat or LLM_Calls, not even at function level: `.12`'s static foundation pin counts function-level imports.
- **#3321.** `.13` C1 moved the provider list after #3321 merged; nothing to do here.
- **ADR assessment:** not required. The task enforces the existing layer direction (`Docs/Architecture.md`, the `.9` contract); it records no new durable decision. State this in each PR body.

## What `.13` produces (cross-plan contract)

Both slices start after every `.13` PR (A, B, C1, C2, D) has merged. This plan relies on the following `.13` outputs. Task 1 Step 1 runs every check on the slice base; any output that differs stops the work until it is reconciled with the `.13` plan.

| Id | `.13` output (where its plan states it) | Check (expected output) |
|---|---|---|
| A1 | `core.LLM_Calls.gateway.resolve_provider_api_key(provider: str, *, prefer_module_keys_in_tests: bool = True, include_override: bool = True) -> tuple[str \| None, dict[str, Any]]`; `chat_service` re-exports it (`.13` Task 3 Interfaces and Step 3.5). This plan calls neither. | `git grep -n "^def resolve_provider_api_key" -- tldw_Server_API/app/core/LLM_Calls/gateway.py` (1 hit) |
| A2 | `core/config_llm_providers.py` exists with `get_api_keys() -> dict[str, str \| None]`, `ALL_SUPPORTED_PROVIDER_NAMES_LIST: list[str]` and `_BASE_SUPPORTED_PROVIDER_NAMES_LIST: list[str]`, moved verbatim, and `load_and_log_configs` bound as a module global (`.13` Task 3 Interfaces, Step 3.1). The gateway calls `config_llm_providers.get_api_keys()` as a module attribute at call time and imports nothing from `api.v1` (Step 3.2). The schema re-exports `ALL_SUPPORTED_PROVIDER_NAMES_LIST, get_api_keys` with `# noqa: F401 - re-exported until TASK-13950.14 retargets importers` (Step 3.4). No `core.LLM_Calls -> api.v1.schemas` baseline line is ever added (Step 7). | `git grep -n "^def get_api_keys" -- tldw_Server_API/app/core/config_llm_providers.py` (1 hit); `git grep -n "config_llm_providers.get_api_keys()" -- tldw_Server_API/app/core/LLM_Calls/gateway.py` (at least 1 hit); `git grep -n "api\.v1" -- tldw_Server_API/app/core/LLM_Calls/gateway.py` (0 hits); `grep -c "^core.LLM_Calls -> api.v1.schemas" tldw_Server_API/tests/lint/layering_baseline.txt` (0) |
| A3 | The `PYTEST_CURRENT_TEST`/`TEST_MODE` reach into both `API_KEYS` maps is deleted with `_resolve_test_module_provider_api_key` and `resolve_static_provider_fallback` (Step 3.5). Both definitions survive: the schema's `API_KEYS = get_api_keys()` "stays for `.14`" and `endpoints/chat.py` is untouched. Step 6 retargets every test writer of the maps to `config_llm_providers.get_api_keys` through `tests/helpers/provider_keys.py:api_keys_source`, except the read alias at `Chat/integration/test_chat_completions_integration.py:189` ("stays for `.14`"). It also retargets the autouse read seam at `Chat/integration/test_chat_endpoint_simplified.py:221` (coordinator ruling, `.13` fix round X1). | `git grep -nw API_KEYS -- 'tldw_Server_API/app/*.py'` (only `endpoints/chat.py`'s import alias, comment and `API_KEYS = SCHEMAS_API_KEYS`, and the schema's comment and `API_KEYS = get_api_keys()`; a `gateway.py` docstring hit, if `.13` kept the old sentence, is not a reader); `git grep -nw API_KEYS -- tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py` (no output) |
| A4 | `.13` C2 changes only `chat_target_resolution.py`'s import of `_split_inline_provider_model` (now from the gateway); `config_default_llm_provider`, `get_default_provider` and the schema import of `DEFAULT_LLM_PROVIDER` are untouched. | `git grep -nE "^def (config_default_llm_provider\|get_default_provider)" -- tldw_Server_API/app/core/Chat/chat_target_resolution.py` (2 hits) |
| A5 | C1 deletes `chat_service`'s two schema imports for keys (`:2156`, `:2251`). No `.13` slice touches `chat_helpers.py:14`, `prompt_improvement_dispatch.py:12` or the `_is_server_managed_chat_request_field` import in `chat_service.build_call_params_from_request`; those belong to `.15` S10. | `git grep -n "chat_request_schemas" -- 'tldw_Server_API/app/core/*.py'` (11 modules, 11 statements) |
| A6 | C1 rewrites `tests/Chat/unit/test_api_key_resolution.py` (Step 1). Three of its new lines name what this plan deletes: the two `raising=False` `API_KEYS` setattrs in `test_resolver_reads_what_production_reads_under_pytest_and_test_mode`, and `assert chat_request_schemas.get_api_keys is config_llm_providers.get_api_keys` in `test_the_old_paths_are_the_moved_objects`. | `git grep -nE "API_KEYS\|chat_request_schemas.get_api_keys" -- tldw_Server_API/tests/Chat/unit/test_api_key_resolution.py` (those three lines, plus a docstring) |

`.13` also produces the test helper `api_keys_source` and its regression test `test_api_keys_source_layers_test_keys_over_the_real_source`, which pins that the gateway follows `config_llm_providers.get_api_keys` at call time. This plan reuses that pin and adds no gateway test.

Order of writes to `core/config_llm_providers.py`: `.13` C1 creates it with the key function and both lists; `.14` slice 2 (Task 3) appends the default-provider functions; `.14` Task 5 deletes the schema's re-exports. No other Phase 1 plan writes to the module.

This plan produces, for `.15` S10, `.17`, `.21` and `.31`, these additions to `tldw_Server_API.app.core.config_llm_providers`:
- `config_default_llm_provider(config_data: dict[str, Any] | None = None) -> str | None`;
- `get_default_provider(*, config_resolver: Callable[[], str | None] = config_default_llm_provider, test_mode_resolver: Callable[[], bool] = is_test_mode, fallback_provider: str = "openai") -> str`.

## Review Focus

1. **A patch that silently stops steering.** A test that patches `chat_request_schemas.get_api_keys`, or a kept re-export, would pass while no runtime code reads that binding. The moved names are deleted rather than shimmed, so a leftover patch raises `AttributeError`. Pinned by `test_api_keys_maps_are_gone` (Task 2) and `test_chat_request_schemas_has_no_config_names` (Task 5); `.13`'s `test_api_keys_source_layers_test_keys_over_the_real_source` pins the gateway side. The proof greps in Tasks 2, 4 and 5 back these up.
2. **Precedence and call-time reads.** The order is `[llm_api_settings] default_api`, then `[API] default_api`, then env `DEFAULT_LLM_PROVIDER`, then test mode (`local-llm`), then `openai`. Blank values are skipped, and a change made after import is seen. Pinned by `test_default_provider_precedence` and `test_default_provider_is_read_at_call_time` (Task 3), `test_core_importer_reads_default_at_call_time` (Task 4) and `test_endpoint_reads_default_at_call_time` (Task 5).
3. **A read seam deleted instead of retargeted.** `test_chat_completions_integration.py:186-198` imports `API_KEYS` inside a `try/except ImportError`. Once the map is gone, the except branch empties both provider lists, and every provider-parametrized case vanishes without a failure. Pinned by `test_provider_parametrization_is_not_empty` (Task 2).
4. **Guard false negatives.** The guard must see an aliased or module import of `core.config`, a function-level import used by a module-level helper call (the `chat_validators.py:67` shape), and calls in class bodies and default arguments. Pinned by `test_every_import_time_config_call_form_is_seen` and `test_every_core_import_form_of_chat_request_schemas_is_seen` (Task 1).
5. **Schema-visible text.** The `api_provider` description embeds the import-time default today (`chat_request_schemas.py:910`), so the OpenAPI fingerprint depends on the machine's config. Pinned by `test_api_provider_description_names_no_provider` (Task 5) and `make openapi-drift-check`.

## PR slices

| Slice | Tasks | Base | Hand-written lines | Total changed (est.) |
|---|---|---|---|---|
| 1. Guards | Task 1 | `dev` after all `.13` PRs merge | ~265 (guard with seeds and self-tests ~260, as the dry run measured; workflow and contract ~5) | ~265 |
| 2. Cut | Tasks 2-6 | `dev` after slice 1 merges | ~200 (default-provider tests ~45, importer and endpoint edits ~65, test edits ~45, guard and docs ~15, the rest deletions) | ~400 (moved bodies ~40, deletions ~110) |

The roadmap says 1 PR. Two slices because (a) the roadmap's architecture is ratchet-first, so the guard lands before the cut, and (b) one PR would be about 665 changed lines. Each slice is green on its own: slice 1 allowlists today's offenders, and slice 2 deletes those entries.

CI-equivalent checks, per slice (run from the repo root; `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; pytest with `-p no:cacheprovider --basetemp "$TMPDIR/<name>"`):
- Both backend-required ratchet steps: the exact `pytest` file lists of "Enforce CI contracts and code ratchets" and "Enforce tenant isolation ratchets" in `.github/workflows/backend-required.yml`, with `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1`. Pass means no failure that does not also fail on the slice base. On `dev` + `.9` today, `test_function_size_ratchet` (2 cases) and `test_charactersragdb_surface_ratchet` fail before any change.
- mypy on changed modules (mypy is not in the venv): `uvx mypy --follow-imports=silent --ignore-missing-imports <changed tldw_Server_API/*.py>`. CI marks this step continue-on-error. Pass means new files are clean and no touched file gains an error line compared with the same command on the slice base.
- Slice 2 only: `make openapi-fingerprint`, then `make openapi-drift-check`. The description changes; `apps/packages/api-types/paths.d.ts` must not.
- Postgres shard: not applicable (no database behaviour changes).

## Inventory (measured)

Measured on `origin/dev` `35e711080a` merged with `origin/chore/13950-9-layering-contract` `9bff409144`, before `.13`. Line numbers shift after `.13` and `.15`, so every edit below is keyed by symbol, and Tasks 2, 4 and 5 re-measure on their base.

**Drift from the backlog task:**
- The backlog says six core modules read `DEFAULT_LLM_PROVIDER`; eight do. It misses `core/Chat/chat_target_resolution.py:11` and `core/Streaming/speech_chat_service.py:41`.
- "11 core files import the module" holds: 13 statements today, 11 after `.13` C1.
- Lines `:148`, `:166-174` and `:305` hold.
- Not in the backlog:
  - 2 services and 9 endpoint importers of `DEFAULT_LLM_PROVIDER`;
  - 6 endpoint importers of `get_api_keys`;
  - two more schema modules that load config at import;
  - 131 test references (R2), of which `.13` C1 owns 113.

**R1. Runtime importers of the names this plan moves or deletes** (paths under `tldw_Server_API/app/`):

| Importer (import line) | Names | Read sites | Task |
|---|---|---|---|
| `core/Chat/chat_target_resolution.py:11` | `DEFAULT_LLM_PROVIDER` | `:78` (default argument) | 3 |
| `core/Chunking/auto_boundary_assistant.py:630` (in function) | `DEFAULT_LLM_PROVIDER` | `:632` | 4 |
| `core/Data_Tables/jobs_worker.py:40` | `DEFAULT_LLM_PROVIDER` | `:1076` (frozen) | 4 |
| `core/Research_Workspace/output_jobs.py:19` | `DEFAULT_LLM_PROVIDER` | `:711` | 4 |
| `core/Skills/skill_executor.py:385` (in function) | `DEFAULT_LLM_PROVIDER` | `:386` | 4 |
| `core/Streaming/speech_chat_service.py:41` | `DEFAULT_LLM_PROVIDER` | `:669` (frozen) | 4 |
| `core/StudyPacks/generation_service.py:14` | `DEFAULT_LLM_PROVIDER` | `:185` | 4 |
| `core/Workflows/adapters/llm/llm.py:52` (in function) | `DEFAULT_LLM_PROVIDER` | `:79` | 4 |
| `services/osce_generator.py:14` | `DEFAULT_LLM_PROVIDER` | `:743` | 4 |
| `services/quiz_generator.py:12` | `DEFAULT_LLM_PROVIDER` | `:1507`, `:1923`, `:2011` | 4 |
| `api/v1/endpoints/audio/__init__.py:24` | `get_api_keys` | (re-export for the audio shim) | 5 |
| `api/v1/endpoints/audio/audio.py:14` | `get_api_keys` | (re-export) | 5 |
| `api/v1/endpoints/audio/audio_streaming.py:47` | `DEFAULT_LLM_PROVIDER`, `get_api_keys` | `:2267` (frozen), `:321`, `:732` | 5 |
| `api/v1/endpoints/character_chat_sessions.py:65` | `DEFAULT_LLM_PROVIDER` | `:888` | 5 |
| `api/v1/endpoints/chat.py:118` | `API_KEYS` | `:428` | 2 |
| `api/v1/endpoints/chat.py:121` | `DEFAULT_LLM_PROVIDER`, `get_api_keys` | `:2853`, `:3716` (frozen), `:2835` | 5 |
| `api/v1/endpoints/config_info.py:707` (in function) | `get_api_keys` | `:708` | 5 |
| `api/v1/endpoints/flashcards.py:22` | `DEFAULT_LLM_PROVIDER` | `:279` | 5 |
| `api/v1/endpoints/llm_providers.py:17` | `ALL_SUPPORTED_PROVIDER_NAMES_LIST`, `get_api_keys` | `:1845`, `:1582` | 5 |
| `api/v1/endpoints/media/document_insights.py:16` | `DEFAULT_LLM_PROVIDER` | `:278` | 5 |
| `api/v1/endpoints/messages.py:24` | `DEFAULT_LLM_PROVIDER` | `:947`, `:1025` | 5 |
| `api/v1/endpoints/reading.py:26` | `DEFAULT_LLM_PROVIDER` | `:996`, `:998` | 5 |
| `api/v1/endpoints/skills.py:906` (in function) | `DEFAULT_LLM_PROVIDER` | `:908` | 5 |
| `api/v1/endpoints/slides.py:33` | `DEFAULT_LLM_PROVIDER` | `:1183` | 5 |

Out of scope:
- Schema models, which `.15` S10 owns: `core/Chat/chat_helpers.py:14`, `core/Prompt_Management/prompt_improvement_dispatch.py:12`, `core/Chat/chat_service.py:2821`, `api/v1/schemas/llamacpp_schemas.py:7`, `mock_openai_server/tests/test_ci_prompt_chat.py:8`, `mock_openai_server/tests/test_live_tier_analysis_responses.py:147`.
- `.13` C1 deletes `chat_service.py:2156` and `:2251`.
- `Helper_Scripts/` imports none of these names.

`apps/` reads these values only through the API:
- `GET /api/v1/llm/providers` takes `default_provider` from config `default_api` and key presence from `get_api_keys()`.
- `GET /api/v1/config/providers` uses `config_info._resolve_provider_key`, which calls `get_api_keys()`.

Both payloads are unchanged, so no `apps/` file changes. `apps/packages/api-types` carries only paths (its rich `schema.d.ts` is gitignored), so the description change moves only `openapi.fingerprint.json`.

**R2. Test references** (paths under `tldw_Server_API/tests/`). Lines cite the line that opens each statement. `git grep -nw API_KEYS` finds 101 test lines in 20 files today: the sites below, plus comments and continuation lines.

| Class | Sites (measured tree) | Owner after `.13` C1 | `.14` step |
|---|---|---|---|
| T-a `chat_request_schemas.API_KEYS` | 23 in 13 files | `.13` C1, except the read alias `Chat/integration/test_chat_completions_integration.py:189` | Task 2 Step 4 (`:189`); Task 2 Step 1 verifies the rest |
| T-b `endpoints.chat.API_KEYS` | 72 in 19 files | `.13` C1, including the read seam `Chat/integration/test_chat_endpoint_simplified.py:221` | verification only (Task 2 Step 1) |
| T-c `chat_request_schemas.load_and_log_configs` | 2: `Chat/unit/test_api_key_resolution.py:184,194` | `.13` C1 (rewritten file) | none |
| T-d `chat_target_resolution.load_and_log_configs` | 2: `Flashcards/test_study_assistant_adapter_contract.py:67`; `Flashcards/test_study_assistant_service_prompts.py:50` | `.14` | Task 3 Step 4 |
| T-e `chat_request_schemas.get_api_keys` steering key resolution | 17 in 8 files | `.13` C1 | verification only (Task 4 Step 4) |
| T-f `chat_request_schemas.get_api_keys` steering `config_info` | 10: `Config/test_config_providers_endpoints.py:444,451,458,471,564,584,600,617,642,655` | `.14` (`.13` keeps them; fix-round X2) | Task 5 Step 4 |
| T-g `DEFAULT_LLM_PROVIDER` patched on an importer | 5: `DataTables/test_data_tables_worker.py:127,429,505`; `DataTables/test_data_tables_jobs_integration.py:40` (`jobs_worker`); `Media/test_document_insights_service_prompt.py:155` (`document_insights`) | `.14` | Task 4 Step 4 (`jobs_worker`), Task 5 Step 4 |
| T-h `.13`'s new lines (A6) | 3 in `Chat/unit/test_api_key_resolution.py` | `.14` | Task 2 Step 4 (2 setattrs), Task 5 Step 3 (identity assertion) |
| T-i comments naming the maps | `Chat/integration/test_chat_endpoint.py:616,624`; `Chat/integration/test_chat_simple.py:33`; `Chat/integration/test_chat_completions_integration.py:787`; `Chat/integration/conftest_isolated.py:246` (whichever survive `.13`) | `.14` | Task 2 Step 4 |

These keep working because the patched module keeps its own binding:
- `endpoints.chat.get_api_keys` (`Chat/integration/test_chat_endpoint.py:633`);
- `llm_providers.get_api_keys` (16 sites in 6 files);
- `audio.get_api_keys` (`Audio/test_ws_audio_chat_stream.py:813,1319`);
- `chat_target_resolution.get_default_provider` (`Chat/test_chat_target_resolution.py:199`, `Sharing/test_shared_workspace_chat_endpoint.py:636,686`);
- `chat._config_default_llm_provider` and `chat.load_and_log_configs` (`Chat/unit/test_chat_default_provider.py`).

Leave these alone:
- `Audio/test_speech_chat_service.py:899` patches a `get_api_keys` that `speech_chat_service` never had (`raising=False`).
- `Config/test_env_file_selection.py:85` reloads the schema to prove it loads no `.env` file. It keeps passing, and the guard now pins the stronger property.

**R3. Layering baseline cuts** (`tldw_Server_API/tests/lint/layering_baseline.txt`), as deltas from each slice's base. `.15` S3, S4 and S6 do not depend on `.13`, so they will likely land first and lower the same lines. A line that reaches 0 is deleted.

| Pair | Delta | Task | Measured value (before `.13`) |
|---|---|---|---|
| `core.Chat -> api.v1.schemas` | -1 (`chat_target_resolution`) | 3 | 6 (`.13` C1 makes it 4) |
| `core.Chunking -> api.v1.schemas` | -1 | 4 | 1 |
| `core.Data_Tables -> api.v1.schemas` | -1 | 4 | 1 |
| `core.Research_Workspace -> api.v1.schemas` | -1 | 4 | 5 |
| `core.Skills -> api.v1.schemas` | -1 | 4 | 1 |
| `core.Streaming -> api.v1.schemas` | -1 | 4 | 2 |
| `core.StudyPacks -> api.v1.schemas` | -1 | 4 | 3 |
| `core.Workflows -> api.v1.schemas` | -1 | 4 | 3 |
| `services -> api.v1.schemas` | -2 | 4 | 36 |

A simulated cut with `--tighten` on the measured tree took the baseline from 262 to 259 pairs and from 683 to 673 statements. Both SCC measures stayed at 102/130. Endpoint edits cut no line, because api -> api is not ruled. The appended functions add no ruled pair.

**R4. Guard seeds.**
- Import-time `core.config*` calls under `api/v1/schemas`:
  - `chat_request_schemas.py`: `_config = load_and_log_configs()` and `API_KEYS = get_api_keys()`. Before `.13` C1, `get_api_keys` is a local loader function; after C1 it is imported from `config_llm_providers`. Either way, the same two calls are flagged.
  - `chat_validators.py:67`: `_get_max_request_size`.
  - `rag_schemas_unified.py:29,36`: `_settings.get`. This entry is absent once `.15` Task 2 has moved the module to core.
- Core importers of `chat_request_schemas`: 11 modules, the same set before and after `.13`. There are 13 statements before C1 (`chat_service` x3) and 11 after it.
  - `core.Chat.chat_helpers`
  - `core.Chat.chat_service`
  - `core.Chat.chat_target_resolution`
  - `core.Chunking.auto_boundary_assistant`
  - `core.Data_Tables.jobs_worker`
  - `core.Prompt_Management.prompt_improvement_dispatch`
  - `core.Research_Workspace.output_jobs`
  - `core.Skills.skill_executor`
  - `core.Streaming.speech_chat_service`
  - `core.StudyPacks.generation_service`
  - `core.Workflows.adapters.llm.llm`

---

### Task 1: Schemas config-boundary guard and core-importer ratchet (slice 1)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_schemas_config_boundary.py`
- Modify: `.github/workflows/backend-required.yml` (step "Enforce CI contracts and code ratchets", after `test_core_to_api_import_boundary.py`)
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py` (`test_backend_required_enforces_layering_ratchets` parametrize list and docstring)

**Interfaces:**
- Consumes: `tldw_Server_API.tests.lint._ast_cache.parse_tree(root: Path) -> Mapping[Path, ast.Module]`, `APP_ROOT`.
- Produces (in the guard module):
  - `IMPORT_TIME_CONFIG_ALLOWLIST: dict[str, str]` maps an app-relative path (`"api/v1/schemas/chat_validators.py"`) to its reason.
  - `CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS: dict[str, str]` maps an app-relative module (`"core.Chat.chat_helpers"`) to its reason.
  - `import_time_config_calls(tree: ast.Module, module: str) -> list[tuple[int, str]]` returns `(line, callee source)` for each flagged call.
    - "Import time" means everything outside `def`, `async def` and `lambda` bodies: module and class bodies, decorators, argument defaults, and module-level `if`/`try`.
    - A call is flagged if its callee is reached through a name bound from a config module. It is also flagged if the callee is a module-level function of the same file that makes such a call, directly or transitively (computed to a fixpoint).
    - A config module is `tldw_Server_API.app.core.config`, or any name starting with `tldw_Server_API.app.core.config.` or `tldw_Server_API.app.core.config_`.
    - Bindings are collected file-wide, function-level imports included, so the matcher fails closed. The forms are `from <config> import x [as y]`, `from tldw_Server_API.app.core import config [as c]`, `import <config> as c`, and un-aliased `import <config>` (matched by dotted prefix). Relative imports are resolved against `module`.
    - Attribute calls count through their root name, as in `c.settings.get(...)`.
  - `schema_config_calls(root: Path = APP_ROOT) -> dict[str, list[tuple[int, str]]]` maps each app-relative path to its flagged calls, for every module under `root/api/v1/schemas`.
  - `core_chat_request_schemas_importers(root: Path = APP_ROOT) -> dict[str, list[int]]` maps each app-relative module to its import line numbers. It covers every module under `root/core` (directories named `tests` are skipped) with an import, at any level, of `tldw_Server_API.app.api.v1.schemas.chat_request_schemas`: `import`, `from ... import`, `from tldw_Server_API.app.api.v1.schemas import chat_request_schemas`, or relative.
  - `problems(found: Mapping[str, object], allowlist: Mapping[str, str]) -> list[str]` returns one message per key, each starting with `"<key>: "`, in three cases:
    - the key is found but not allowlisted;
    - the key is allowlisted but not found ("stale");
    - the key is allowlisted with a reason ("reason") that has no `TASK-<n>`/`#<n>` reference or contains an unfilled `<...>`.
  - `python -m tldw_Server_API.tests.lint.test_schemas_config_boundary` prints both measurements. This is the committed counter.

- [ ] **Step 1: Check the `.13` contract.** Run every check in the "What `.13` produces" table on the slice base. Expected: the listed outputs. On any mismatch, stop and reconcile with the `.13` plan before writing code. Record the results in the PR body.

- [ ] **Step 2: Write the self-tests** in `tldw_Server_API/tests/lint/test_schemas_config_boundary.py`. Use `pytestmark = pytest.mark.unit`. The helper `_seed(tmp_path, files: dict[str, str]) -> Path` writes the files under `tmp_path / "app"` and returns that root.

```python
@pytest.mark.parametrize("source", [
    "from tldw_Server_API.app.core.config import load_and_log_configs\nCFG = load_and_log_configs()\n",
    "from tldw_Server_API.app.core import config as c\nCFG = c.load_and_log_configs()\n",
    "import tldw_Server_API.app.core.config as c\nX = c.settings.get('k')\n",
    "import tldw_Server_API.app.core.config\nX = tldw_Server_API.app.core.config.load_comprehensive_config()\n",
    "from tldw_Server_API.app.core.config_llm_providers import get_api_keys\nKEYS = get_api_keys()\n",
    "def _k():\n    from tldw_Server_API.app.core.config import load_and_log_configs\n    return load_and_log_configs()\n\nK = _k()\n",
    "from tldw_Server_API.app.core.config import settings\nclass M:\n    n: int = int(settings.get('n', 1))\n",
    "from tldw_Server_API.app.core.config import load_and_log_configs\ndef f(cfg=load_and_log_configs()):\n    return cfg\n",
])
def test_every_import_time_config_call_form_is_seen(tmp_path, source):
    assert list(schema_config_calls(_seed(tmp_path, {"api/v1/schemas/x.py": source}))) == ["api/v1/schemas/x.py"]

def test_call_time_reads_and_pure_helpers_are_not_flagged(tmp_path):
    source = (
        "from tldw_Server_API.app.core.config import load_and_log_configs\n"
        "from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names\n"
        "NAMES = list(iter_custom_openai_provider_names())\n"
        "def f():\n    return load_and_log_configs()\n"
        "g = lambda: load_and_log_configs()\n"
        "class M:\n    def m(self):\n        return load_and_log_configs()\n"
    )
    assert schema_config_calls(_seed(tmp_path, {"api/v1/schemas/x.py": source})) == {}

@pytest.mark.parametrize("source", [
    "from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest\n",
    "def f():\n    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest\n",
    "from tldw_Server_API.app.api.v1.schemas import chat_request_schemas\n",
    "import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as s\n",
    "from ...api.v1.schemas.chat_request_schemas import ChatCompletionRequest\n",
])
def test_every_core_import_form_of_chat_request_schemas_is_seen(tmp_path, source):
    assert list(core_chat_request_schemas_importers(_seed(tmp_path, {"core/X/y.py": source}))) == ["core.X.y"]

def test_unlisted_stale_and_unreasoned_entries_fail():
    assert [p.split(":")[0] for p in problems({"a": [1]}, {})] == ["a"]
    assert "stale" in problems({}, {"b": "why (TASK-1)"})[0]
    assert "reason" in problems({"a": [1]}, {"a": "because"})[0]
    assert "reason" in problems({"a": [1]}, {"a": "see TASK-1 <fill me>"})[0]
    assert problems({"a": [1]}, {"a": "why (TASK-1)"}) == []
```

- [ ] **Step 3: Run them to verify they fail.**
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-guard" tldw_Server_API/tests/lint/test_schemas_config_boundary.py -q`
Expected: 15 failed. Each fails with a `NameError` naming `schema_config_calls`, `core_chat_request_schemas_importers` or `problems`.

- [ ] **Step 4: Implement** the five functions and `__main__` per the Interfaces block, over `parse_tree`. The module docstring covers:
  - the purpose (TASK-13950.14);
  - the matcher definition above;
  - the allowlist semantics;
  - the known limits. A config load reached through an imported non-config helper is not seen; importing `core.LLM_Calls.routing.models`, for example, loads config today. `importlib`, `getattr` and module rebinding are not seen either.

- [ ] **Step 5: Run the self-tests.** Same command as Step 3. Expected: the four tests pass, 15 cases.

- [ ] **Step 6: Seed both allowlists from the counter.**
Run: `$PY -m tldw_Server_API.tests.lint.test_schemas_config_boundary`
The dict contents are the same whether or not `.13` C1/C2 are on the base; only the counter's line lists differ.

| Base | Counter output |
|---|---|
| After `.13` C1 (the slice base) | Schema calls: `chat_request_schemas.py` (`load_and_log_configs`, imported `get_api_keys`), `chat_validators.py` (`_get_max_request_size`), `rag_schemas_unified.py` (`_settings.get` x2). Core importers: the 11 modules of R4, with `core.Chat.chat_service` at 1 line. |
| Before `.13` C1 (only if slice 1 is ever run early) | The same three schema files, with `get_api_keys` flagged as a local loader. The same 11 modules, with `core.Chat.chat_service` at 3 lines. |

There is no `core.LLM_Calls.gateway` entry on either base. If `.15` Task 2 is on the base, `rag_schemas_unified.py` is absent; omit its entry. Write the dicts with these reasons:
  - `api/v1/schemas/chat_request_schemas.py`: `"_config = load_and_log_configs() fixes DEFAULT_LLM_PROVIDER, and API_KEYS = get_api_keys() fixes the key map, at import; slice 2 of TASK-13950.14 deletes both"`.
  - `api/v1/schemas/chat_validators.py`: `"MAX_REQUEST_SIZE reads [Chat-Module] max_request_size_chars once at import, and tests patch and import the constant; config reads move to call time under TASK-13950.21"`.
  - `api/v1/schemas/rag_schemas_unified.py`: `"contextual-retrieval Field defaults come from core.config.settings at import and are OpenAPI-visible; config reads move to call time under TASK-13950.21"`.
  - The eight `DEFAULT_LLM_PROVIDER` importers: `"reads DEFAULT_LLM_PROVIDER from the schema; slice 2 of TASK-13950.14 retargets it"`.
  - `core.Chat.chat_helpers`, `core.Chat.chat_service` and `core.Prompt_Management.prompt_improvement_dispatch`: `"imports schema models; TASK-13950.15 moves them to core"`.

Then add the two gate tests:

```python
def test_schemas_modules_do_not_load_config_at_import():
    assert problems(schema_config_calls(), IMPORT_TIME_CONFIG_ALLOWLIST) == []

def test_core_imports_of_chat_request_schemas_only_shrink():
    assert problems(core_chat_request_schemas_importers(), CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS) == []
```

Run the Step 3 command. Expected: 17 passed.

- [ ] **Step 7: Wire the guard into the required gate.** Add `"tldw_Server_API/tests/lint/test_schemas_config_boundary.py"` to the parametrize list of `test_backend_required_enforces_layering_ratchets`, and make its docstring cite "TASK-13950.9 and .14".
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-ci" "tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py::test_backend_required_enforces_layering_ratchets" -q`
Expected: 1 failed, 2 passed (the new path is not in the step yet).
Then append `tldw_Server_API/tests/lint/test_schemas_config_boundary.py \` after `test_core_to_api_import_boundary.py \` in the step's `run:`, and re-run. Expected: 3 passed.

- [ ] **Step 8: Run the slice checks**: both ratchet steps, and `uvx mypy` on the two changed Python files. Expected: no failure that does not also fail on the base, and mypy reports `Success` for the new file.

- [ ] **Step 9: Commit and open the PR.**

```bash
git add tldw_Server_API/tests/lint/test_schemas_config_boundary.py .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): guard import-time config loads in api/v1/schemas and ratchet core imports of chat_request_schemas (TASK-13950.14)"
```

Open it with `gh pr create --base dev`, titled as the commit. The body covers:
- AC #1, AC #4, and the seed for AC #3;
- R4;
- the Step 1 check results;
- the ADR assessment (not required).

Then get a clean-subagent `/superpowers:requesting-code-review`, wait for green CI, and merge.

---

### Task 2: Delete the API_KEYS maps (slice 2)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py` (the `API_KEYS` comment block and definition), `tldw_Server_API/app/api/v1/endpoints/chat.py` (the `API_KEYS as SCHEMAS_API_KEYS` import and the comment block with `API_KEYS = SCHEMAS_API_KEYS`)
- Modify: `tldw_Server_API/tests/Chat/integration/test_chat_completions_integration.py` (`:186-198`, plus a new test), `tldw_Server_API/tests/Chat/unit/test_api_key_resolution.py` (T-h, 2 lines), and the T-i comment lines
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py`

**Interfaces:**
- Consumes: A3 and A6.
- Produces: neither `chat_request_schemas` nor `endpoints.chat` has an `API_KEYS` attribute.

- [ ] **Step 1: Verify `.13` retargeted every writer and the read seam.**
Run: `git grep -nw API_KEYS -- '*.py'`
Expected hits:
  - A3's app definitions;
  - `test_chat_completions_integration.py:189` and its comment line;
  - the T-h lines;
  - surviving T-i comments;
  - `.13`'s docstrings in `tests/helpers/provider_keys.py` and `Chat/unit/test_api_key_resolution.py`.

Stop on either of these:
  - a `patch`, `setattr`, `setitem` or assignment of a map anywhere else (T-a/T-b writers are `.13` C1's);
  - a read of `chat_endpoint.API_KEYS` in `test_chat_endpoint_simplified.py` (fix-round X1).

- [ ] **Step 2: Write the failing tests.** In `tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py`:

```python
def test_api_keys_maps_are_gone():
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
    assert not hasattr(chat_request_schemas, "API_KEYS")
    assert not hasattr(chat_endpoint, "API_KEYS")
```

In `tldw_Server_API/tests/Chat/integration/test_chat_completions_integration.py`, after `LOCAL_PROVIDERS_FOR_TEST = get_local_providers_integration()`:

```python
def test_provider_parametrization_is_not_empty():
    """The provider lists above come from a try/except ImportError; an empty list hides every case."""
    assert ALL_CONFIGURED_PROVIDERS_FROM_APP
    assert LOCAL_PROVIDERS_FOR_TEST
```

Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t2" tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py::test_api_keys_maps_are_gone tldw_Server_API/tests/Chat/integration/test_chat_completions_integration.py::test_provider_parametrization_is_not_empty -q`
Expected: 1 failed (`assert not True`), 1 passed (`API_KEYS` still imports).

- [ ] **Step 3: Delete the definitions.** In the schema, delete the comment block and `API_KEYS = get_api_keys()`. In `chat.py`, delete the `API_KEYS as SCHEMAS_API_KEYS` import, the comment block and `API_KEYS = SCHEMAS_API_KEYS`. Re-run the Step 2 command. Expected: 1 passed and 1 failed. The parametrization test now fails, because the `ImportError` branch emptied both lists. That failure is Review Focus 3.

- [ ] **Step 4: Fix what still names the maps.**
  - `test_chat_completions_integration.py:186-198`: inside the same `try`, replace the `API_KEYS` import with `from tldw_Server_API.app.core.config_llm_providers import get_api_keys`, followed by `APP_API_KEYS_FROM_SCHEMA = get_api_keys()`.
  - T-h: in `Chat/unit/test_api_key_resolution.py`, delete the two `raising=False` `API_KEYS` setattrs in `test_resolver_reads_what_production_reads_under_pytest_and_test_mode`. Drop the `chat_endpoint` import if nothing else uses it. `test_api_keys_maps_are_gone` now pins what those lines did.
  - T-i: reword each surviving comment that names a map to name `config_llm_providers.get_api_keys`, or drop it.
  - Any other hit from Step 1: if it reads a map that feeds a seam, retarget it to `config_llm_providers.get_api_keys()`; never delete it. If it writes a map, it is `.13`'s, so stop.

- [ ] **Step 5: Prove it and run the touched tests.**
Run: `git grep -nw API_KEYS -- '*.py'`. Expected: only `.13`'s two docstrings, plus the gateway docstring if `.13` kept it.
Run the following before committing; at that point `HEAD` is the slice base:

```bash
$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t2b" $(git diff --name-only --diff-filter=d HEAD -- 'tldw_Server_API/tests/**/test_*.py') tldw_Server_API/tests/Chat -q
```

Expected: pass, with the same skips as on the base.

- [ ] **Step 6: Commit.**

```bash
git add -A tldw_Server_API
git commit -m "refactor(chat): delete the API_KEYS maps from the chat schema and endpoint (TASK-13950.14)"
```

---

### Task 3: Default-provider resolver in `core/config_llm_providers.py` (slice 2)

**Files:**
- Modify: `tldw_Server_API/app/core/config_llm_providers.py` (created by `.13` C1; append only)
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py`, `tldw_Server_API/app/core/Chat/chat_target_resolution.py` (keyed by symbol; `.13` C2 changed an adjacent import)
- Modify: `tldw_Server_API/tests/lint/test_schemas_config_boundary.py` (delete the `core.Chat.chat_target_resolution` entry), `tldw_Server_API/tests/lint/layering_baseline.txt` (R3, Task 3 row)
- Modify (T-d): `tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py:67`, `tldw_Server_API/tests/Flashcards/test_study_assistant_service_prompts.py:50-51`
- Test: `tldw_Server_API/tests/Config/test_config_llm_providers.py` (new)

**Interfaces:**
- Consumes: A2 (`config_llm_providers.load_and_log_configs` is a module global) and `core.testing.is_test_mode() -> bool`, which reads `TEST_MODE` and `TLDW_TEST_MODE` only.
- Produces: `config_default_llm_provider` (moved verbatim from `chat_target_resolution`) and `get_default_provider`. The latter is moved too; its only change is the `fallback_provider` default, `"openai"`, which was the schema's import-time value. Both read config through the module global `load_and_log_configs`, so tests patch `config_llm_providers.load_and_log_configs`. `chat_target_resolution` re-exports both, with `__all__` unchanged.

- [ ] **Step 1: Write the failing tests** in `tldw_Server_API/tests/Config/test_config_llm_providers.py`:

```python
import pytest

from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.Chat import chat_target_resolution

pytestmark = pytest.mark.unit


def _set_env(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(("config", "env", "test_mode", "expected"), [
    ({"llm_api_settings": {"default_api": " cfg-a "}, "API": {"default_api": "cfg-b"}}, "env-x", "true", "cfg-a"),
    ({"API": {"default_api": "cfg-b"}}, "env-x", "true", "cfg-b"),
    ({"llm_api_settings": {"default_api": "  "}}, " env-x ", "true", "env-x"),
    ({}, None, "true", "local-llm"),
    (None, None, None, "openai"),
])
def test_default_provider_precedence(monkeypatch, config, env, test_mode, expected):
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: config)
    _set_env(monkeypatch, "DEFAULT_LLM_PROVIDER", env)
    _set_env(monkeypatch, "TEST_MODE", test_mode)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    assert config_llm_providers.get_default_provider() == expected


def test_default_provider_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "first")
    assert config_llm_providers.get_default_provider() == "first"
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "second")
    assert config_llm_providers.get_default_provider() == "second"


def test_chat_target_resolution_reexports_the_core_resolvers():
    assert chat_target_resolution.get_default_provider is config_llm_providers.get_default_provider
    assert chat_target_resolution.config_default_llm_provider is config_llm_providers.config_default_llm_provider
```

- [ ] **Step 2: Run them to verify they fail.**
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t3" tldw_Server_API/tests/Config/test_config_llm_providers.py -q`
Expected: 7 failed with `AttributeError: module 'tldw_Server_API.app.core.config_llm_providers' has no attribute 'get_default_provider'`.

- [ ] **Step 3: Append to `core/config_llm_providers.py`.** Add `config_default_llm_provider` and `get_default_provider` per the Interfaces block. Add the imports `from collections.abc import Callable`, `from typing import Any` and `from tldw_Server_API.app.core.testing import is_test_mode`. `.13`'s module docstring already says "TASK-13950.14 adds the default provider".

- [ ] **Step 4: Point the old homes at it.**
  - `chat_target_resolution.py`:
    - Delete the schema import of `DEFAULT_LLM_PROVIDER` and the two function bodies.
    - Import `config_default_llm_provider, get_default_provider` from `tldw_Server_API.app.core.config_llm_providers`.
    - Drop the now-unused `load_and_log_configs` and `is_test_mode` imports.
    - Keep `__all__`.
  - `chat_request_schemas.py`:
    - Delete these by name: `_config`, `_config_default_llm_provider`, `_cfg_default_provider`, `_env_default_provider`, `_test_mode_enabled`, `_get_setting` (dead), the `core.config` import, the `is_test_mode` import, `import os`, and the `DEFAULT_LLM_PROVIDER = "openai"` line.
    - Add `get_default_provider` to `.13`'s re-export line.
    - Where the deleted block was, before the `ChatCompletionRequest` description that uses it, add `DEFAULT_LLM_PROVIDER = get_default_provider()`.
    - This bridge and the re-exports are temporary; Task 5 deletes them. Until then, importers, the description and the fingerprint see the same values as before.
  - T-d: both Flashcards sites patch `config_llm_providers.load_and_log_configs` instead of `chat_target_resolution.load_and_log_configs`. This also changes what `config_llm_providers.get_api_keys()` returns in those tests: the adapter-contract fixture config carries `openai_api.api_key`. Step 5 checks that neither test now passes through a newly satisfied key path.
  - Ratchets: delete `core.Chat.chat_target_resolution` from `CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS`. Then run `$PY -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. Expected baseline diff: `core.Chat -> api.v1.schemas` -1, and nothing else.

- [ ] **Step 5: Run the tests.**
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t3b" tldw_Server_API/tests/Config/test_config_llm_providers.py tldw_Server_API/tests/Config/test_env_file_selection.py tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py tldw_Server_API/tests/Flashcards/test_study_assistant_service_prompts.py tldw_Server_API/tests/Chat/test_chat_target_resolution.py tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/lint/test_schemas_config_boundary.py tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py -q`
Expected: pass. The guard still lists `chat_request_schemas.py`, which now calls `get_default_provider()` at import.

Then run the two Flashcards files once more, with `monkeypatch.setattr(config_llm_providers, "get_api_keys", lambda: {name: None for name in config_llm_providers.get_api_keys()})` added temporarily to each patched test (every supported name maps to `None`, so an indexed lookup still finds the key and the test cannot fail with a KeyError for the wrong reason). Expected: the same test ids pass, so the key path is not what makes them pass. Revert the temporary lines.

- [ ] **Step 6: Commit.**

```bash
git add tldw_Server_API/app/core/config_llm_providers.py tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py tldw_Server_API/app/core/Chat/chat_target_resolution.py tldw_Server_API/tests/Config/test_config_llm_providers.py tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py tldw_Server_API/tests/Flashcards/test_study_assistant_service_prompts.py tldw_Server_API/tests/lint/test_schemas_config_boundary.py tldw_Server_API/tests/lint/layering_baseline.txt
git commit -m "refactor(config): resolve the default LLM provider in core/config_llm_providers at call time (TASK-13950.14)"
```

---

### Task 4: Core and services read the default at call time (slice 2)

**Files:**
- Modify: the 7 remaining core rows and the 2 services rows of R1 (`chat_target_resolution` was Task 3)
- Modify: `tldw_Server_API/tests/lint/test_schemas_config_boundary.py` (delete 7 importer entries), `tldw_Server_API/tests/lint/layering_baseline.txt` (R3, Task 4 rows)
- Modify (T-g): `tldw_Server_API/tests/DataTables/test_data_tables_worker.py:127,429,505`, `tldw_Server_API/tests/DataTables/test_data_tables_jobs_integration.py:40`
- Test: `tldw_Server_API/tests/Config/test_config_llm_providers.py`

**Interfaces:**
- Consumes: `get_default_provider() -> str` (Task 3).
- Produces: no core or services module reads `chat_request_schemas` for a default.

- [ ] **Step 1: Re-measure.** Run: `git grep -n "chat_request_schemas" -- 'tldw_Server_API/app/core/*.py' 'tldw_Server_API/app/services/*.py'`. Expected: R1's 7 core and 2 services rows (line numbers may have moved), the A5 model importers, and the out-of-scope `services` model imports.

- [ ] **Step 2: Write the failing test** (append to `test_config_llm_providers.py`):

```python
def test_core_importer_reads_default_at_call_time(monkeypatch):
    from tldw_Server_API.app.core.Chunking import auto_boundary_assistant
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "late-env")
    assert auto_boundary_assistant._configured_default_provider({}) == "late-env"
```

Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t4" "tldw_Server_API/tests/Config/test_config_llm_providers.py::test_core_importer_reads_default_at_call_time" -q`
Expected: 1 failed. The importer returns the schema's import-time default.

- [ ] **Step 3: Retarget the importers.** At each R1 import, import `get_default_provider` from `tldw_Server_API.app.core.config_llm_providers` instead. Function-level imports stay function-level, and the `try/except` around the `auto_boundary_assistant` and `skill_executor` imports stays. At each read site, `DEFAULT_LLM_PROVIDER` becomes `get_default_provider()` on the same line; frozen functions must not gain a line.

- [ ] **Step 4: Retarget T-g and verify T-e.**
  - T-g, `jobs_worker` (4 sites): `monkeypatch.setattr(jobs_worker, "DEFAULT_LLM_PROVIDER", "openai")` becomes `monkeypatch.setattr(jobs_worker, "get_default_provider", lambda: "openai")`.
  - Verify T-e, which `.13` C1 retargeted. Run: `git grep -nE 'chat_request_schemas[.", ]+get_api_keys' -- tldw_Server_API/tests`. Expected: only T-f's 10 lines and the T-h identity assertion.
  - Also run: `rg -nU --pcre2 "setattr\(\s*chat_request_schemas\s*,\s*[\"']get_api_keys" tldw_Server_API/tests`. Expected: no output.

- [ ] **Step 5: Shrink both ratchets.**
  - Delete these 7 modules from `CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS`: `core.Chunking.auto_boundary_assistant`, `core.Data_Tables.jobs_worker`, `core.Research_Workspace.output_jobs`, `core.Skills.skill_executor`, `core.Streaming.speech_chat_service`, `core.StudyPacks.generation_service` and `core.Workflows.adapters.llm.llm`.
  - Run `$PY -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. Expected baseline diff, from the slice base: R3's Task 4 rows. That is -1 on `core.Chunking`, `core.Data_Tables`, `core.Research_Workspace`, `core.Skills`, `core.Streaming`, `core.StudyPacks` and `core.Workflows` (each `-> api.v1.schemas`), and -2 on `services -> api.v1.schemas`. A line at 0 is deleted.
  - If any other line moves, stop and explain it.

- [ ] **Step 6: Run the tests.**
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t4b" tldw_Server_API/tests/Config/test_config_llm_providers.py tldw_Server_API/tests/DataTables tldw_Server_API/tests/lint/test_schemas_config_boundary.py tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py tldw_Server_API/tests/lint/test_function_size_ratchet.py $(git grep -lE "auto_boundary_assistant|skill_executor|speech_chat_service|StudyPacks.generation_service|adapters.llm.llm|osce_generator|quiz_generator|Research_Workspace.output_jobs" -- 'tldw_Server_API/tests/**/test_*.py') -q`
Expected: pass. The size ratchet passes, or fails only as it does on the base.

- [ ] **Step 7: Commit.**

```bash
git add -A tldw_Server_API
git commit -m "refactor(core): read the default LLM provider from core/config_llm_providers at call time (TASK-13950.14)"
```

---

### Task 5: Endpoints read the accessor; the schema stops loading config (slice 2)

**Files:**
- Modify: the 13 `api/v1/endpoints/*` rows of R1 other than `chat.py:118`; `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py` (the bridges and re-exports, and the `api_provider` description)
- Modify: `tldw_Server_API/tests/lint/test_schemas_config_boundary.py` (delete the `chat_request_schemas.py` entry)
- Modify (T-f, T-g, T-h): `tldw_Server_API/tests/Config/test_config_providers_endpoints.py`, `tldw_Server_API/tests/Media/test_document_insights_service_prompt.py:155`, `tldw_Server_API/tests/Chat/unit/test_api_key_resolution.py` (the identity assertion)
- Modify: `Docs/API-related/Chat_API_Documentation.md:415,448` and `Docs/Development/Running_Chat_Tests.md:34`. Regenerate `Docs/Published/API-related/Chat_API_Documentation.md` with the refresh script and `apps/packages/api-types/openapi.fingerprint.json` with `make openapi-fingerprint`.
- Test: `tldw_Server_API/tests/Config/test_config_llm_providers.py`, `tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py`

**Interfaces:**
- Consumes: Task 3's functions and A2's `get_api_keys` and `ALL_SUPPORTED_PROVIDER_NAMES_LIST`.
- Produces:
  - `chat_request_schemas` keeps `SUPPORTED_API_ENDPOINTS` and imports `ALL_SUPPORTED_PROVIDER_NAMES_LIST`, with no `noqa`, because its validator uses the name. It defines or imports no other config name.
  - The `api_provider` description is exactly `"[Extension] The target LLM provider (e.g., 'openai', 'anthropic'). If omitted, defaults to the server's configured default provider."`.

- [ ] **Step 1: Write the failing tests.** In `test_config_llm_providers.py`:

```python
def test_endpoint_reads_default_at_call_time(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import slides
    monkeypatch.setattr(config_llm_providers, "load_and_log_configs", lambda: {})
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "late-env")
    assert slides._resolve_provider(None) == "late-env"
```

In `test_chat_request_schemas.py`:

```python
_MOVED = ("DEFAULT_LLM_PROVIDER", "get_api_keys", "get_default_provider", "load_and_log_configs",
          "is_test_mode", "_config", "_get_setting", "_config_default_llm_provider")

def test_chat_request_schemas_has_no_config_names():
    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
    assert [name for name in _MOVED if hasattr(chat_request_schemas, name)] == []

def test_api_provider_description_names_no_provider():
    assert ChatCompletionRequest.model_fields["api_provider"].description == (
        "[Extension] The target LLM provider (e.g., 'openai', 'anthropic'). "
        "If omitted, defaults to the server's configured default provider."
    )
```

Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t5" tldw_Server_API/tests/Config/test_config_llm_providers.py::test_endpoint_reads_default_at_call_time tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py -q -k "call_time or config_names or description"`
Expected: 3 failed.

- [ ] **Step 2: Retarget the endpoints.**
  - Each endpoint's schema import of `DEFAULT_LLM_PROVIDER` becomes an import of `get_default_provider` from `tldw_Server_API.app.core.config_llm_providers`. `skills.py:906` stays function-level, inside its `try`.
  - Each `DEFAULT_LLM_PROVIDER` read becomes `get_default_provider()` on the same line. The exception is `chat.py:2853`, which drops the `fallback_provider=` keyword instead; `chat.py` already imports `get_default_provider` from `chat_target_resolution`.
  - The module-level `get_api_keys` bindings (`audio/__init__.py`, `audio/audio.py`, `audio/audio_streaming.py`, `chat.py`, `llm_providers.py`) keep their names and import from `tldw_Server_API.app.core.config_llm_providers`. So does `llm_providers.py`'s `ALL_SUPPORTED_PROVIDER_NAMES_LIST`.
  - `config_info.py:707` keeps its function-level import, now from the core module.

- [ ] **Step 3: Remove the bridges.**
  - The schema imports only `ALL_SUPPORTED_PROVIDER_NAMES_LIST` from the core module, without `noqa`.
  - Delete `DEFAULT_LLM_PROVIDER = get_default_provider()` and set the static description.
  - Delete the `api/v1/schemas/chat_request_schemas.py` entry from `IMPORT_TIME_CONFIG_ALLOWLIST`.
  - T-h: delete `assert chat_request_schemas.get_api_keys is config_llm_providers.get_api_keys` from `test_api_key_resolution.py`, and keep its `ALL_SUPPORTED_PROVIDER_NAMES_LIST` identity assertion.

- [ ] **Step 4: Retarget the tests.**
  - T-f: the 10 `_target` and patch strings in `test_config_providers_endpoints.py` become `"tldw_Server_API.app.core.config_llm_providers.get_api_keys"`.
  - T-g: `test_document_insights_service_prompt.py:155` becomes `monkeypatch.setattr(endpoint, "get_default_provider", lambda: "openai")`.

- [ ] **Step 5: Docs.**
  - Change the three snippets `from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys` to `from tldw_Server_API.app.core.config_llm_providers import get_api_keys`.
  - Run `bash Helper_Scripts/refresh_docs_published.sh`. Expected `git status --short Docs/Published`: only `Docs/Published/API-related/Chat_API_Documentation.md`. Any other file changing is pre-existing drift; stop.

- [ ] **Step 6: OpenAPI.** Run `make openapi-fingerprint`, then `make openapi-drift-check`. Expected:
  - only `apps/packages/api-types/openapi.fingerprint.json`'s `sha256` changes; `path_count` and `schema_count` (2140 and 3316 on the measured tree) stay;
  - `paths.d.ts` is unchanged;
  - the check prints `OK`.

- [ ] **Step 7: Run the tests and the proof greps.**
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-t5b" tldw_Server_API/tests/Config tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Media/test_document_insights_service_prompt.py tldw_Server_API/tests/lint/test_schemas_config_boundary.py tldw_Server_API/tests/Docs -q`. Expected: pass.
Run: `git grep -nw DEFAULT_LLM_PROVIDER -- 'tldw_Server_API/app/*.py' | grep -v '"DEFAULT_LLM_PROVIDER"'`. Expected: no output; only env-name strings remain.
Run: `git grep -nE "chat_request_schemas.*(get_api_keys|DEFAULT_LLM_PROVIDER|load_and_log_configs)" -- '*.py' 'Docs/API-related/*.md' 'Docs/Development/*.md'`. Expected: no output.
Run: `rg -nU --pcre2 "setattr\(\s*(chat_request_schemas|schema_chat|chat_schemas)\s*,\s*[\"'](get_api_keys|DEFAULT_LLM_PROVIDER|load_and_log_configs)[\"']" tldw_Server_API`. Expected: no output.

- [ ] **Step 8: Commit.**

```bash
git add -A tldw_Server_API Docs/API-related/Chat_API_Documentation.md Docs/Development/Running_Chat_Tests.md Docs/Published/API-related/Chat_API_Documentation.md apps/packages/api-types/openapi.fingerprint.json
git commit -m "refactor(chat): endpoints read provider defaults from core; chat_request_schemas loads no config at import (TASK-13950.14)"
```

---

### Task 6: Slice 2 verification and PR

**Files:** none new.

- [ ] **Step 1: Run every test that names a changed symbol** (the flipping-a-default rule).
Run: `$PY -m pytest -p no:cacheprovider --basetemp "$TMPDIR/t14-all" $(git grep -lE "chat_request_schemas|DEFAULT_LLM_PROVIDER|get_api_keys|config_llm_providers|chat_target_resolution|get_default_provider" -- 'tldw_Server_API/tests/**/test_*.py') -q`
Expected: pass (87 files on the measured tree). Compare any failure against the same command on the slice base before dismissing it.

- [ ] **Step 2: Run the slice checks:**
  - both backend-required ratchet steps, which include the new guard, the layering contract, the size ratchet and the test-quality ratchet;
  - `uvx mypy` on the changed Python files;
  - `make openapi-drift-check`.

Expected: no failure that does not also fail on the base. `test_schemas_config_boundary.py` must allowlist exactly `chat_validators.py`, plus `rag_schemas_unified.py` unless `.15` Task 2 has landed. The importer ratchet must list exactly `core.Chat.chat_helpers`, `core.Chat.chat_service` and `core.Prompt_Management.prompt_improvement_dispatch`: down from 11 on `dev`, which is also 11 after `.13`.

- [ ] **Step 3: Open the PR.** Use `gh pr create --base dev` with the title `refactor(config): move chat provider defaults and API keys out of the schemas package (TASK-13950.14)`. The body covers:
  - the AC map:
    - #1: the slice 1 guard, now allowlisting only the non-chat files;
    - #2: Tasks 3-5, with the keys from `.13` C1;
    - #3: 11 -> 3 core importers, ratcheted by the guard and by R3's baseline cuts;
    - #4: the slice 1 wiring;
  - the R3 diff;
  - the OpenAPI note;
  - the Step 1 results of Task 2;
  - the ADR assessment (not required).

Then get a clean-subagent `/superpowers:requesting-code-review`, wait for green CI, and merge. Report the AC status and the final summary to the orchestrator for the backlog.

## Self-review (fix round 1, 2026-10-26)

- Review C1 and the cross-plan contract: all 12 edits are applied. Task 3 now modifies `.13`'s module. The core importer count is 11. The `core.LLM_Calls -> api.v1.schemas` row and the gateway test are gone. T-a, T-b, T-c and T-e are verification only. The A-table cites `.13`'s Produces blocks and has no fallback column.
- I1 to I4 and the dry-run Importants are applied:
  - I1: Task 2 Step 4 retargets the `:189` read, and `test_provider_parametrization_is_not_empty` guards it.
  - I2: `:221` is treated as already retargeted, and Task 2 Step 1 checks it.
  - I3: R3 and both `--tighten` expectations are deltas.
  - I4: T-h lines are edited in Tasks 2 and 5.
  - The Task 1 seed is spelled out for both bases, with no gateway entry.
  - Task 1 Step 3 now expects 15 `NameError` failures.
  - The Task 3 tests are runnable, with their imports listed.
- Minors folded:
  - M1: greps are restricted to `*.py`, comments are handled, and the expected docstring hits are listed.
  - M2: the duplicate tests are dropped, and the re-export test is trimmed.
  - M3: precedence setup is written out.
  - M4: a Flashcards key-path check is added.
  - M5: the 2870 ceiling is stated.
  - Dry-run 3: `uvx mypy`.
  - Dry-run 4: R2 line convention and the 101-line count.
  - Dry-run 5: "no failure that does not also fail on the base", naming today's pre-existing failures.
  - Dry-run 6: the allowlist reason now names both import-time calls.
- Declined: review M6 (drop `CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS`). AC #3 asks that core imports of `chat_request_schemas` specifically be ratcheted. `.9` counts all `api.v1.schemas` statements per package, so only the module-keyed list names that module and catches a swap within a package. The dry run showed it costs about 40 lines.
- Also dropped: `test_supported_names_cover_base_providers_and_custom_slots`. `.13` owns the list and its tests, and #3321's allowlist tests already cover custom slots. Review Focus 3 is now the read-seam failure mode instead.
- Type consistency: `get_default_provider`, `config_default_llm_provider`, `get_api_keys`, `ALL_SUPPORTED_PROVIDER_NAMES_LIST`, `problems`, `schema_config_calls` and `core_chat_request_schemas_importers` are spelled the same in every task. The test names cited in Review Focus exist in their owning tasks.
