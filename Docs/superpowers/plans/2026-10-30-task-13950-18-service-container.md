# Introduce an App-Scoped Service Container and Retire Module Singletons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the module-singleton patterns with AST ratchets, then move the AuthNZ DB pool, session manager, JWT service, MCP server and ChaCha runtime into one `AppServices` container that the lifespan publishes as `app.state.services`, so tests swap services through a fresh-container fixture instead of `reset_*` helpers, and ChaCha's per-user DB cache and executor leave the API layer.

**Architecture:** `core/Utils/app_services.py` (foundation layer, imports nothing above it at run time) holds one `AppServices` dataclass per process with one slot per service. Each existing getter (`get_db_pool()`, `get_session_manager()`, `get_jwt_service()`, `get_mcp_server()`) keeps its name and get-or-create logic but reads and fills its slot instead of a module global, and the new `get_chacha_runtime()` (Task 9) does the same for the ChaCha runtime, so `get_x() is app_services().x` always holds and there is never a second instance. The lifespan publishes the container before startup and shuts down what the container holds, in ADR-021's existing order (pinned by a test before any slot moves). Tests get `fresh_services` (fixture) and `fresh_app_services()` (async context manager): empty slots, then close what the test built and restore the previous slots. ChaCha's executor, LRU cache, init coordination and health move into `ChaChaRuntimeManager`; its per-user DB factory and the worker-facing accessors move to `core/Character_Chat/chacha_db_provider.py`; `ChaCha_Notes_DB_Deps.py` keeps only the FastAPI dependency and the HTTP error mapping.

**Tech Stack:** Python 3.12/3.13, FastAPI lifespan, pytest + pytest-asyncio 1.4 (`asyncio_mode = "auto"`, function-scoped loops), AST lint ratchets under `tldw_Server_API/tests/lint/` sharing `_ast_cache.py` and `_ceilings.py`, SQLite and PostgreSQL (`TLDW_TEST_POSTGRES_REQUIRED=1`).

**Spec:**
- `backlog/tasks/task-13950.18 - Introduce-an-app-scoped-service-container-and-retire-module-singletons.md`, AC #1-#7 and the DoD.
- The `.18` row of `## Phase 2` in `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`: effort L; one PR slice per service (db_pool, session_manager, jwt_service, MCP server, ChaCha runtime); keep `get_x()` as a deprecated fallback that reads the container; check shutdown order (ADR-021) every slice; `reset_db_pool` callers under 50.
- `Docs/ADR/021-services-lifecycle-startup-and-shutdown.md` (the shutdown order this task must keep).

**Conventions:**
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Run every command from the repo root with `PYTHONDONTWRITEBYTECODE=1`.
- pytest runs as `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-<name>"`. Pass `-p pytest_asyncio.plugin` only together with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, as C1-C3 and CI do. With autoload on, it raises "Plugin already registered" in this venv.
- File:line references are on `origin/dev` b9eed09f91. Re-locate them by the quoted text.
- Every changed file ends with exactly one newline and has no trailing whitespace.

## Measured on 2026-10-30 (origin/dev b9eed09f91)

| Figure | Backlog (dca11342de) | Measured now | Note |
|---|---|---|---|
| `global` statements under `app/` | 456 in 231 files | 456 in 231 files | Task 1's counter; no in-app `tests` package has one |
| Bare zero-arg `get_x()` calls in endpoint function bodies | 574 | 551 in 103 files (`get_db_pool` 69, `get_acp_session_store` 37, `get_settings` 27) | The same counter gives exactly 574 on dca11342de, so the drift (-23) is code, not matcher |
| `API_Deps` modules with module-level locks, caches or executors | 11 of 38 | 11 of 38, 35 assignments | Audit 5, ChaCha 6, DB_Deps 3, Explainer 2, Prompts 1, Slides 2, Workflows 1, auth_deps 6, jobs 2, kanban 5, prompt_studio 2 |
| `services`/`core` modules importing `ChaCha_Notes_DB_Deps` | 18 (14 + 4) | 18 (14 + 4), 19 statements | `startup_chacha_warmup.py` imports twice; `core/Notes/README.md` mentions it too (doc, not counted) |
| `ChaCha_Notes_DB_Deps.py` | 1,030 lines | 1,030 lines | |
| Test files calling `reset_db_pool` | 155 | 168 | 6 conftests; 90 files call it only from `async def test*` bodies |
| `reset_session_manager` / `reset_jwt_service` in tests | not given | 32 files, 69 calls / 16 files, 27 calls | Neither has a caller in `app/` |
| `get_db_pool` | not given | 427 references in 133 app files; 257 test files | |
| Test patches of the MCP server singleton | not given | 34 `get_mcp_server` patches on `endpoints.mcp_unified_endpoint` (9 files), plus 6 `_server` patches on `MCP_unified.server` (`test_rag_module.py` ×5 and the `.4` race test) | Counted alias-aware by the seam scanner (Tool). 2 files call route functions directly |
| Names of `ChaCha_Notes_DB_Deps` that tests touch | not given | 53 names (the seam scanner); 198 files mention the module | Task 9's name table gives each one a disposition |

Where each service lives today:

| Service | Global | Getter | Reset | Created at startup by | Shut down by |
|---|---|---|---|---|---|
| AuthNZ DB pool | `core/AuthNZ/database.py:1763 _db_pool` | `get_db_pool()` :1844 | `reset_db_pool()` :1865 (also called by `session_manager.py:218`, `token_blacklist.py:207`, `initialize.py:771`) | `services/startup_auth.py:42` | `shutdown_auth_db_pool` via `shutdown_cleanup_timed_segments` |
| Session manager | `core/AuthNZ/session_manager.py:2076`, init lock :2079 | `get_session_manager()` :2082 | `reset_session_manager()` :2096 (tests only) | `services/startup_auth_runtime.py:36` | `shutdown_resource_cleanup._shutdown_session_manager` |
| JWT service | `core/AuthNZ/jwt_service.py:1160` | `get_jwt_service()` :1164 | `reset_jwt_service()` :1179 (tests only) | first use | none |
| MCP server | `core/MCP_unified/server.py:2315` | `get_mcp_server()` :2319 | `reset_mcp_server()` :2329 (called by `reset_db_pool`) | `services/startup_heavy_init.py:180` | `shutdown_resource_cleanup._shutdown_mcp_server` (only when heavy init succeeded) |
| ChaCha runtime | `api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:50-82, 359-369` | `_CHACHA_RUNTIME`, `_get_or_init_db_instance` | `reset_chacha_shutdown_state()` (startup) and `ChaChaRuntimeManager.reset_for_tests()` | `services/startup_chacha_warmup.py` | `shutdown_resource_cleanup._shutdown_chacha_resources` |

The final shutdown tail today (`shutdown_cleanup_timed_segments` then `shutdown_resource_cleanup`) closes: DB pool, session manager, MCP server (then waits for its deferred modules), MCP rate limiter, TTS service, TTS resource manager, HTTP client, ChaCha, Prompts, chat workflows, provider manager, request queue, local LLM manager, then the evaluations, unified audit, executors, CPU pools and telemetry segments.

**Drift since the base.** `origin/dev` is now 4d0701bece. #3371 merged and added `tests/Usage/test_llm_quota_byok_rows.py`, which calls `reset_db_pool` (169 files). The three ratchet counters are unchanged there: 456/231, 551/103 (the MCP endpoint at 22) and 35/11. Task 8a places the new file.

**Dry run of the design (scratch worktree on b9eed09f91).**
- The container in `core/Utils` with `TYPE_CHECKING`-only imports keeps `test_layering_contract.py` green (21 passed), including the foundation import and load pins.
- With `_db_pool` moved into the slot, the four test files that touch `database._db_pool` pass once retargeted (48 passed).
- Task 7's codemod changed 83 files. Run on dev and on the converted tree, all 83 gave the same result set.
  - 39 files (AuthNZ_SQLite and Usage): 141 passed, 1 skipped, and the same 3 failures, which already fail on dev (`test_llm_budget_402_sqlite.py::test_llm_budget_allows_under_budget_chat_sqlite` and two in `test_quota_enforcement_http_sqlite.py`).
  - The other 44: 179 passed, 1 xfailed.
- After the codemod, 88 of the 168 files still call `reset_db_pool`; Task 8's list converts 43 more.

## Rulings

1. **AC2 scope (controller ruling, 2026-10-30; binding).**
   - Every request handler can take each service through a dependency provider that reads the container: the existing `get_db`/`get_db_transaction` (`database.py:1940-1951`), `get_session_manager_dep` and `get_jwt_service_dep` (`auth_deps.py:725-730`), and the new `get_mcp_server_dep` (Task 6).
   - Task 6 converts the 11 HTTP routes of `mcp_unified_endpoint.py` to `Depends(get_mcp_server_dep)`.
   - The 69 `get_db_pool()` and 2 `get_jwt_service()` calls in endpoint bodies stay on the getter, frozen by Task 1's ratchet. A dependency resolves before the handler runs, which breaks behaviour that tests pin: `tests/Ingestion_Sources/test_ingestion_sources_api.py` asserts invalid input never fetches the pool (`_unexpected_get_db_pool`), and `admin_impersonation.py:100` builds the JWT service inside the handler's `try` after its 403 check. `.20` (endpoint shared code) owns those calls.
   - Workers reach the services through the getters, which read the same container. After Task 10, no worker imports the API layer. This task does not thread services through worker constructors. The ChaCha job workers (12 in `services`, 2 in `core`) are module-level functions, and their lifecycle specs pass them only a stop event (`lifecycle_worker_specs.stop_event_worker_spec`, `WorkerRegistry.register_custom`).
   - Task 11 rewrites AC2 to "...endpoints receive them through Depends providers that read the container (the MCP routes are converted; other bare getter calls are frozen by AC1's ratchet), and workers through the getters, which read the same container".
2. **Which reset helpers are deleted (AC4) (controller ruling, 2026-10-30; binding).**
   - AC4 deletes only the test-only helpers: `reset_session_manager` (Task 8b), `reset_jwt_service` (Task 5) and `ChaChaRuntimeManager.reset_for_tests` (Task 9).
   - `reset_db_pool` stays, because production code calls it: `SessionManager` and `TokenBlacklist` self-heal a closed or loop-switched pool, and `ensure_single_user_rbac_seed_if_needed` calls it in test mode or when the environment and settings disagree. `reset_mcp_server` stays because `reset_db_pool` calls it. Both read the container.
3. **One container object per process.** `fresh_app_services()` empties and restores the slots of the one object; it never swaps the object. So `app.state.services is app_services()` holds in every test, including a test that enters `TestClient(app)` after the fixture has run.
4. **Shutdown reads the container.**
   - The lifespan passes `db_pool=services.db_pool` and `session_manager=services.session_manager`, read at shutdown, so shutdown closes the live objects. Today it closes the startup handle, which is stale once test mode's `reset_db_pool` (`initialize.py:771`) has replaced the pool.
   - The MCP server keeps its heavy-init handle, because only an initialized server is shut down today. Task 6 pins that handle as the container's object.
   - The order of the shutdown helpers does not change.

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

The four 2026-10-25 owner decisions (moderation, keywords, scheduler, admin) do not bind this task.

Task-specific:

- **Controller rulings (2026-10-30), binding:**
  - *AC2 scope.* The MCP HTTP routes move to `Depends(get_mcp_server_dep)`. Every other bare getter call in an endpoint body stays on the getter, frozen by AC1's ratchet and handed to `.20`. Reason: a dependency resolves before the handler body runs, which breaks behaviour that tests pin. `test_ingestion_sources_api.py:151,184` asserts that invalid input never fetches the pool, and `admin_impersonation.py:100` builds the JWT service after its 403 check (Ruling 1).
  - *AC4 reading.* "Test-only reset helpers" means the helpers no production code calls: `reset_session_manager`, `reset_jwt_service` and `ChaChaRuntimeManager.reset_for_tests`. `reset_db_pool` stays because production self-heal code calls it (`session_manager.py:218`, `token_blacklist.py:207`, `initialize.py:771`). `reset_mcp_server` stays because `reset_db_pool` calls it (Ruling 2).
- **Order.** Task 2's shutdown-order pin merges (PR A) before any slot moves. Every later PR re-runs it unchanged (Task 10 changes only its ChaCha seam) and must keep its expected list byte-identical.
- **Identity, not a second instance.** A getter returns the slot's object or creates the object into the slot under its existing lock. No slice keeps a module-level copy. Each slice pins `get_x() is app_services().x`.
  - Known limits. All three exist on dev today; this task does not make any of them worse:
    - `get_session_manager` serializes only within one event loop, so two loops racing on first use can build two managers.
    - After a self-heal `reset_db_pool()` → `reset_mcp_server()`, the slot holds a new server, while shutdown closes the heavy-init handle (Ruling 4).
    - Four direct `JWTService(settings)` constructions bypass the slot: `endpoints/auth.py:1503`, `endpoints/workflows.py:1641`, `services/admin_e2e_support_service.py:668`, `services/workflows_scheduler.py:602`.
- **The shutdown pin runs in every slice.**
  - Every slice runs C3, which selects `tldw_Server_API/tests/unit/test_app_services_shutdown_order.py` through its `unit` marker.
  - Tasks 3-6, 9 and 10, which touch the services the pin covers, also name the file (or `tests/unit`) in their own run steps.
- **Layering.** `core/Utils/app_services.py` is foundation: no run-time import of any upper layer, at module or function level. Type names come from `if TYPE_CHECKING:` (bound by `from typing import TYPE_CHECKING`), which the `.9` scanner and `.12`'s foundation pins skip. `ChaChaRuntimeManager` stays in `core.DB_Management` and imports no domain package, so `.16`'s `db_management_logic_imports` ratchet does not move. The ChaCha factory and accessors go to `core.Character_Chat` (domain), because they seed the built-in character from `core.Visual_Identities`.
- **Ratchet upkeep per slice.** Each slice that removes globals, getter calls or API_Deps state runs `"$PY" -m tldw_Server_API.tests.lint.test_service_singleton_ratchet --tighten`. A count that falls to 0 leaves a stale entry, which fails until it is deleted. `--tighten` deletes it.
- **Function sizes.** No function over 300 lines is edited. The Task 6 routes are at most 115 lines. `ChaChaRuntimeManager.get_db` (moved from `_get_or_init_db_instance`, 130 lines) is not in `function_size_baseline.json`. `persona:persona_stream` (3,237 lines, frozen) is not touched: its one `get_mcp_server()` call stays.
- **No legacy baseline regeneration.** None of this task's files is in `scope_predicate_baseline.txt`. Check: `grep -nE "AuthNZ/(database|session_manager|jwt_service|initialize|token_blacklist)|MCP_unified/server|mcp_unified_endpoint|ChaCha_Notes_DB_Deps|chacha/runtime|chacha_db_provider|lifespan_|startup_(auth|chacha|core)|shutdown_resource|app/main" Helper_Scripts/ci/scope_predicate_baseline.txt` prints nothing.
- **Test-quality ratchet.** New tests add no `stub_injection`, `tautology_suspect` or `mock_density` offense (`Helper_Scripts/ci/test_quality_triage.py --enforce`).
  - Fakes go in through `monkeypatch.setattr(app_services(), "<slot>", fake)` or call arguments, never `obj.attr = _Fake()`.
  - A test whose only assert compares a collector list also asserts a real return value.
- **Codemod output is generated.** Tasks 7 and 8 report their hand-written and generated lines separately. Reviewers re-run Task 7's script on the PR base and diff the result.
- **Fixture order.** A test that only installs a fake with `monkeypatch.setattr(app_services(), "<slot>", fake)` needs no fixture: monkeypatch puts the slot back. A test that does that *and* requests `fresh_services` lists `fresh_services` before `monkeypatch` in its parameters. Pytest tears fixtures down in reverse order, so the monkeypatch undo runs first (the slot goes back to the emptied value) and `fresh_app_services` then restores the snapshot. The fake is never passed to `discard_app_services`, so it is not closed. The tests in Tasks 3-6 that patch a slot without `fresh_services` follow the first case.

## Review Focus

1. **A getter that builds a second instance.** If a slot and a module copy diverge, a fake installed in one path misses the code that runs, and shutdown closes the wrong object. Pinned by:
   - `test_get_db_pool_fills_and_reuses_the_container_slot` (Task 3);
   - `test_get_session_manager_fills_the_slot_once_under_concurrency` (Task 4);
   - `test_get_jwt_service_fills_the_slot_once_across_threads` (Task 5);
   - `test_get_mcp_server_single_instance_across_threads` and `test_heavy_init_returns_the_container_mcp_server` (Task 6);
   - `test_get_chacha_runtime_is_the_container_slot` (Task 9, `tests/unit/test_chacha_runtime_container.py`);
   - the lifespan tests, which check that shutdown receives `app.state.services`' objects (Tasks 3 and 4).
2. **Shutdown order drift.** If a slice reorders the final cleanup tail, the pool can close under a live session manager, or ChaCha can close before the workers that use it. Pinned by `test_final_cleanup_closes_the_container_services_in_adr_021_order` (Task 2), re-run in every slice.
3. **A ratchet that passes for the wrong reason.** A counter could miss nested functions, `global a, b`, `try:`-wrapped module state or attribute-called getters, or could count the wrong things. Pinned by `test_each_counter_counts_its_shape` (Task 1), which seeds every shape, including ones that must not count, and by the AC3 pin's seeded self-test (Task 10).
4. **A converted test that tests something else.** A codemod that drops a mid-test reconfiguration leaves the first pool live, and the test may still pass. Pinned by Task 7 Step 3: every converted file's pass/fail set equals the base's, file by file. Any difference reverts that file to a hand conversion. The Full Suite comparison covers the rest.
5. **ChaCha HTTP errors or tenant binding drift once the runtime raises core errors and the provider builds the factory.** Shutdown must stay 503, timeout 503, corruption 503, generic failure 500 with the same details, and worker callers must not see `HTTPException`. Pinned by the existing `tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py` and `tests/Chat/test_chacha_db_deps_error_mapping.py` (Task 9 retargets their seams, not their assertions), plus these new tests:
   - `test_runtime_errors_map_to_the_same_http_responses` (Task 9). It uses exact details, including a path planted in the abort sentinel.
   - `test_core_accessor_raises_core_errors_not_http` and `test_core_accessors_bind_the_tenant_client_id` (Task 10). The second checks that `client_id` is always the user id.

## CI-equivalent checks (each task names the ones it needs)

| Id | What | Command | Pass means |
|---|---|---|---|
| C1 | backend-required "Enforce CI contracts and code ratchets" | `BACKLOG_TASK_FORMAT_BASE=origin/dev PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-c1" -p pytest_asyncio.plugin` followed by every path in the step's first `pytest` command (`.github/workflows/backend-required.yml:290-317` today, plus Task 1's file) | all pass |
| C2 | backend-required "Enforce tenant isolation ratchets" | same env: `tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` | all pass |
| C3 | backend-required "Backend unit smoke" | `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-c3" -p pytest_asyncio.plugin -m "unit and not e2e and not jobs" tldw_Server_API/tests/unit` | all pass |
| C4 | mypy on changed modules (non-blocking in CI; not in the venv) | `uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports $(git diff --name-only --diff-filter=d origin/dev -- 'tldw_Server_API/app/*.py')` | no error on a line this PR changed |
| C5 | OpenAPI drift | `PYTHONPATH=. "$PY" Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | exit 0 |
| C6 | Postgres shard | `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$TMPDIR/t1318-pg" <files>`, with `POSTGRES_TEST_*` set as in ci.yml's Postgres shard, or Docker auto-start (`tests/_plugins/postgres.py`) | pass, and `-rs` lists no skipped `[postgres]` case |
| C7 | Full Suite comparison (fixture or conftest changes) | Compare the failing node ids of the PR's "CI" run with the union of the failing ids of the last 3 merged PRs' "CI" runs on dev (`gh run view <id> --log-failed`) | no failing id outside that union |

## Tool: the seam scanner

Tasks 6, 9 and 10 check test seams with `"$TMPDIR/t1318_seams.py"`. A grep cannot see aliases (`mcp_ep`, `mcp`, `deps`, `chacha_deps`, `ChaCha_Notes_DB_Deps`), so the scanner resolves them. It is not committed. Usage:

```text
python t1318_seams.py MODULE                 -> "<name>: attr N, import N, patch N; F files" for each name tests use
python t1318_seams.py MODULE --sites a,b,... -> "<file>:<line> <kind> <name>" for each use of the listed names
```

A *use* is one of:
- `from MODULE import name`;
- `alias.name`, where `alias` is bound by `import MODULE as alias`, `from PKG import LEAF [as alias]` or `alias = importlib.import_module("MODULE")`;
- a `setattr`, `patch.object` or `delattr` on such an alias (kind `patch`);
- a `"MODULE.name"` string passed to `setattr` or `patch`.

It scans `tldw_Server_API/tests` and every in-app `tests` package. On b9eed09f91, `... tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps` prints 53 lines, and `... tldw_Server_API.app.core.MCP_unified.server` prints `_server: patch 6; 2 files`.

```python
import ast
import sys
from collections import defaultdict
from pathlib import Path

MOD = sys.argv[1]
PKG, LEAF = MOD.rsplit(".", 1)
SITES = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[2] == "--sites" else None
roots = [Path("tldw_Server_API/tests"), *(p for p in Path("tldw_Server_API/app").rglob("tests") if p.is_dir())]
uses = defaultdict(lambda: defaultdict(list))


def note(name, kind, path, line):
    uses[name][kind].append(path)
    if SITES is not None and name in SITES:
        print(f"{path}:{line} {kind} {name}")


for root in roots:
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError:
            continue
        aliases = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                aliases |= {a.asname for a in n.names if a.name == MOD and a.asname}
            elif isinstance(n, ast.ImportFrom) and n.module == PKG:
                aliases |= {a.asname or a.name for a in n.names if a.name == LEAF}
            elif isinstance(n, ast.ImportFrom) and n.module == MOD:
                for a in n.names:
                    note(a.name, "import", path, n.lineno)
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and n.value.args and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == MOD:
                aliases |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if fname in ("setattr", "object", "delattr") and len(n.args) >= 2 and isinstance(n.args[0], ast.Name) and n.args[0].id in aliases and isinstance(n.args[1], ast.Constant):
                    note(n.args[1].value, "patch", path, n.lineno)
                elif fname in ("setattr", "patch") and n.args and isinstance(n.args[0], ast.Constant) and str(n.args[0].value).startswith(MOD + "."):
                    note(n.args[0].value[len(MOD) + 1 :].split(".")[0], "patch", path, n.lineno)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                note(n.attr, "attr", path, n.lineno)
if SITES is None:
    for name in sorted(uses):
        files = set().union(*map(set, uses[name].values()))
        print(f"{name}: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(uses[name].items())) + f"; {len(files)} files")
```

## Design

- **`AppServices`** (`core/Utils/app_services.py`). A `@dataclass` with one slot per service; each slice adds its own slots:
  - `db_pool: DatabasePool | None = None` (Task 3);
  - `session_manager: SessionManager | None = None` and `session_manager_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)` (Task 4);
  - `jwt_service: JWTService | None = None` (Task 5);
  - `mcp_server: MCPServer | None = None` (Task 6);
  - `chacha_runtime: ChaChaRuntimeManager | None = None` (Task 9).

  The module holds `_SERVICES = AppServices()`, and `app_services() -> AppServices` returns it. Nothing else lives there.
- **Getters.** Same names, same signatures, same locks (`_db_pool_lifecycle_lock`, `_jwt_service_lock`, `_server_lock` stay module-level threading locks). The only change is where the object is stored. Each docstring says: "Fallback for code with no request or constructor argument; request handlers take it via `<provider>`."
- **Lifespan.** `main.lifespan` runs `services = app_services(); app.state.services = services` right after `worker_runtime = LifespanWorkerRuntimeState()` (`main.py:1281`), and its `finally` passes `db_pool=services.db_pool, session_manager=services.session_manager`. The startup handle chain stops carrying `db_pool` and `session_manager`.
- **Tests.** `tests/_plugins/app_services_fixtures.py` (registered in `tests/conftest.py:8-14`) provides:
  - `async close_app_service(slot: str, *, restore: object = None) -> None`: the one helper that closes a slot's object.
    - If the slot already holds `restore`, it does nothing.
    - Otherwise it sets the slot to `restore` and then awaits the taken object's `aclose()` if it has one, else its `shutdown()`. The ChaCha runtime has both, and `aclose()` is the one that closes.
    - Errors are logged at debug and swallowed.
    - Used by `discard_app_services`, the root autouse fixture's session-slot reset (Task 4) and the root conftest's three ChaCha after-test blocks (Task 9).
  - `async discard_app_services() -> None`: closes what every slot holds and empties it, then resets the caches built on the pool.
    - The steps run in this order: `reset_db_pool()` (pool, MCP server, API-key manager, UsersDB, RBAC and MCP caches), `close_app_service("session_manager")`, `close_app_service("chacha_runtime")`, then every slot back to its default. Whichever of Tasks 4 and 9 lands second inserts its step at this position.
    - Teardown errors are logged at debug and swallowed, as `reset_db_pool` already does.
    - Fixtures that today reset singletons before and after each test call it (Task 8b).
  - `fresh_app_services()`, an async context manager:
    - on entry it snapshots the slots, sets each to its default (closing nothing), and runs `await reset_db_pool()` (on an empty slot that only drops the caches);
    - on exit it runs `discard_app_services()` and restores the snapshot.
  - `fresh_services`, an async fixture wrapping it. Sync tests can use it too; this was verified on pytest-asyncio 1.4.
- **ChaCha.**
  - `ChaChaRuntimeManager` (persistence) owns the executor, the per-user `LRUCache`, init events and errors, health counters, default-character task tracking and the shutting-down flag. It raises core errors.
  - `core/Character_Chat/chacha_db_provider.py` (domain) owns the per-user DB factory, default-character seeding, the tenant client id and the accessors workers call.
  - `ChaCha_Notes_DB_Deps.py` keeps `get_chacha_db_for_user` (Depends), HTTP wrappers for `get_chacha_db_for_user_id` and `get_chacha_db_for_owner`, the error-to-HTTP mapping, and re-exports of the provider's public names for API callers and tests.

## PR slices

| PR | Tasks | Base | Hand-written lines | Generated or moved |
|---|---|---|---|---|
| A | 1, 2 | dev | ~330: ratchet ~230, shutdown pin ~90, wiring 3 | 3 baselines (231 + 103 + 11 entries) |
| B | 3 | A | ~260 | none |
| C | 4 | B | ~140 | none |
| D | 5 | C | ~90 (27 call sites become slot clears) | none |
| E | 6 | D | ~170 | none |
| F1a | 7 (AuthNZ* dirs) | E | 0 | codemod: 45 files, ~530 lines |
| F1b | 7 (other dirs) | F1a | 0 | codemod: 38 files, ~550 lines |
| F2a | 8a | F1b | ~260 (43 files) | none |
| F2b | 8b | F2a | ~120 | none |
| G1 | 9 | B (independent of C-F; rebase on whichever is on dev) | ~330, including about 155 test uses retargeted (Task 9's table) | ~420 moved into `chacha/runtime.py` |
| G2 | 10 | G1 | ~240, including about 60 test uses retargeted | ~230 moved into `chacha_db_provider.py` |
| H | 11 | all | ~40 (ADR-021 amendment, backlog) | `Docs/Published` mirror |

C, D and E edit adjacent lines of `app_services.py` (C also edits the fixture plugin and the root conftest, D the root conftest), so they merge in order. F2b needs every `reset_session_manager` caller converted. G1 can run beside C-F. C7 applies to B, C, D, E, F1a, F1b, F2a, F2b, G1 and G2, i.e. every slice that changes a fixture, a conftest or the fixture plugin; E also deletes a global that 6 tests patch.

C6 applies to:
- B, C and F2b (AuthNZ on Postgres);
- F2a, on the AuthNZ files in Task 8a's table;
- G1 and G2 (the ChaCha Postgres files named in Task 9 Step 5);
- F1a (every converted AuthNZ* file, including `AuthNZ/integration/test_registration_role_membership_postgres.py`, `test_single_user_bootstrap_postgres.py` and `test_single_user_claims_permissions.py`);
- F1b on `tests/Usage/test_usage_tracker_sqlite.py` (its `[postgres]` case uses `pg_temp_db`) and `tests/Usage/test_usage_aggregator.py::test_aggregate_postgres_branch` (it runs only with a `postgresql` `DATABASE_URL`).

## Sibling coordination

| Sibling | Overlap | If it lands first | If this lands first |
|---|---|---|---|
| `.13` C1 #3369 (then C2 #3370, D #3373, D2 #3374, stacked) | `backend-required.yml` and `test_license_first_workflow_contracts.py` (one ratchet line each; PR A); `tests/conftest.py` (C1 adds an autouse override-store fixture after `healthy_no_override_tts_credential_snapshot`, around :927-950; PR B edits `pytest_plugins` at :8-14; C and D both edit `_restore_auth_env_and_singletons` at :373-421 (C adds the session-slot reset, D the JWT slot clears at :384-411); G1 edits :450-461, :800-806 and :1203-1212); `layering_baseline.txt` (G2) | Take both sides of each list or hunk; for the baseline, take dev's side and run `--tighten` | Same |
| `.17` A2/B (plan `2026-10-26-task-13950-17-provider-credentials.md`) | Its K1 codemod rewrites the `core.AuthNZ.llm_provider_overrides` import inside `reset_db_pool` (`database.py:1933-1935`; the function spans :1865-1938) and the string in `services/startup_auth.py:83`. PR B rewrites `database.py:1762-1897`, which ends 36 lines above it in the same function. None of the five singletons lives in a file `.17` moves (`database.py`, `session_manager.py` and `jwt_service.py` stay in `core/AuthNZ`). `.17`'s alias gate rejects new app imports of moved modules; this task adds none | Keep K1's import path in `reset_db_pool`; re-run `.17`'s alias gate (`test_app_code_imports_no_authnz_alias_path`) | K1's re-run (its porting recipe step 4) rewrites the one line |
| `.15` S4-S9 #3377-#3384 (stacked on S4), S10 | `layering_baseline.txt`. S4 also edits `core/Research_Workspace/capabilities.py` and `core/Sharing/shared_workspace_clone_jobs_worker.py`, two of G2's 18 importers | G2: take dev's side of the baseline, `--tighten`, and re-check the four API_Deps lines listed under Task 10 Files, Baseline. G2's edit to those two files is the one import line (:451, :1422); rebase it onto S4's version | S4 rebases the same import line |
| `.14` (plan only) | none | - | - |
| `.19` test-mode seams (plan `2026-10-30-task-13950-19-test-mode-seams.md`) | T1 adds a ratchet line next to Task 1's in `backend-required.yml` and the contract file. T9 edits `reset_singletons` (`tests/AuthNZ/conftest.py:565-575`), which F2b rewrites. T6 deletes the `get_session_manager_dep` pytest stub, after which every endpoint test uses the real session manager in the slot | Take both lines. F2b rebases its `discard_app_services` lines onto T9's hunk (different lines of the same fixture). After T6, per-test emptying of the session slot comes from `reset_singletons` (AuthNZ tests) and `fresh_services`; elsewhere, Task 4 adds the slot to the root autouse `_restore_auth_env_and_singletons` (controller ruling 2026-10-30: `.18` owns the container and its root-fixture reset; `.19` T6 runs after Task 4, consumes the slot and changes no root fixture) | T6 waits for Task 4 |
| `.20` endpoint shared code | Its endpoint-monkeypatch counter. PR E removes 34 patches on `endpoints.mcp_unified_endpoint`. Ruling 1 leaves the `get_db_pool` and `get_jwt_service` endpoint calls to `.20` | PR E runs `.20`'s `--tighten` | `.20` seeds after E |
| `.21` config consolidation | If its env-read ratchet is file-keyed, G1 moves the `CHACHA_EXECUTOR_MAX_WORKERS`, `CHACHA_INIT_WATCHDOG_SECS` and `CHACHA_INIT_TIMEOUT_SECS` reads from `API_Deps/ChaCha_Notes_DB_Deps.py` to `core/DB_Management/chacha/runtime.py` | G1 re-keys those entries with "moved from <old key> (<PR>)" | `.21` seeds after G1 |
| `.50` order-dependent MCP hub tests | MCP hub tests share the MCP server global that E moves into a slot | - | If E's `fresh_services` fixes them, `.50` cites E |
| `.17` B (planned) | Edits `tests/performance/test_authnz_multiuser_sqlite_load.py:72` (F2b's file), `tests/Privileges/test_privilege_snapshot_retention.py:10` (F1b's), `tests/AuthNZ/conftest.py:36` (F2b's) and `tests/conftest.py:75` (scheduler import lines) | Re-run Task 7's codemod on the rebased F1b; F2b rebases its call-site edits (other lines) | `.17` B rebases its import lines |
| #3187 (stale, conflicting since 2026-10-04) | `MCP_unified/server.py` (other hunks) | - | E's hunk at :2314-2340 is disjoint |

---

### Task 1: The singleton ratchet (AC1, AC7)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py`.
- Create (generated): `tldw_Server_API/tests/lint/global_statements_baseline.json`, `tldw_Server_API/tests/lint/endpoint_getter_calls_baseline.json` and `tldw_Server_API/tests/lint/api_deps_module_state_baseline.json`.
- Modify: `.github/workflows/backend-required.yml`. Add `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py \` after `tldw_Server_API/tests/lint/test_direct_llm_adapter_calls_ratchet.py \` (:305).
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add the path to the parametrize list of `test_backend_required_enforces_size_ratchets` (:1396-1403); that test checks `_ceilings.BASE_ENV`, which this ratchet uses.

**Interfaces:**
- Consumes: `_ceilings.Ratchet`, `violations`, `load`, `reasonless_bumps`, `warn_if_tightenable` and `main`, plus `_ast_cache.parse_tree` and `APP_ROOT`, all as `test_direct_llm_adapter_calls_ratchet.py` uses them.
- Produces:
  - `measure_globals(root: Path = APP_ROOT) -> dict[str, dict[str, int]]`: `{app-relative posix path: {"globals": n}}`, where n counts the `ast.Global` nodes in the module. Modules in an in-app `tests` package directory are skipped.
  - `measure_getter_calls(root: Path = APP_ROOT) -> dict[str, dict[str, int]]`: `{path: {"calls": n}}` for modules under `api/<version>/endpoints/` at any depth. n counts the distinct `ast.Call` nodes inside a function body (at any nesting) whose `func` is an `ast.Name` matching `get_\w+` and that have no args and no keywords. Module-level calls, decorators and defaults do not count, and neither do attribute calls (`mod.get_x()`).
  - `measure_module_state(root: Path = APP_ROOT) -> dict[str, dict[str, int]]`: `{path: {"state": n}}` for the direct children of `api/<version>/API_Deps/`. n counts the module-level `Assign`/`AnnAssign` statements whose value is a call whose callee name (an `ast.Name` id or an `ast.Attribute` attr) is in `STATE_CONSTRUCTORS`. "Module-level" means top-level statements and statements nested in module-level `if`/`try`/`with`/`for`/`while` blocks, never inside a `def` or `class`.
  - `STATE_CONSTRUCTORS = frozenset({"Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Event", "ThreadPoolExecutor", "ProcessPoolExecutor", "LRUCache", "TTLCache", "OrderedDict", "WeakValueDictionary", "WeakKeyDictionary"})`.
  - `GLOBALS`, `GETTERS` and `STATE`: `_ceilings.Ratchet`s with `command="python -m tldw_Server_API.tests.lint.test_service_singleton_ratchet"` and these texts:
    - `GLOBALS.fix`: "Keep the state on an object its owner holds (a slot in app_services() or a constructor argument), not in a module global."
    - `GLOBALS.stale`: "The module has no `global` statement any more."
    - `GETTERS.fix`: "Take the service as a parameter: Depends(<provider>) in a route, an argument in a helper. Bare zero-argument getters in endpoint modules only fall."
    - `GETTERS.stale`: "The endpoint module makes no bare getter call any more."
    - `STATE.fix`: "Move the lock, cache or executor into a core runtime object held by app_services() (as ChaChaRuntimeManager does); API_Deps modules stay adapters."
    - `STATE.stale`: "The module holds no module-level lock, cache or executor any more."
  - `RATCHETS = ((GLOBALS, measure_globals), (GETTERS, measure_getter_calls), (STATE, measure_module_state))`.
  - `__main__` runs `_ceilings.main(ratchet, measure)` for each entry of `RATCHETS` with the same argv and exits with the OR of the codes.

- [ ] **Step 1: Write the self-tests.**

```python
def test_each_counter_counts_its_shape(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/a.py": "_x = None\ndef f():\n    global _x\n    _x = 1\nclass C:\n    def g(self):\n        global _x, _y\n",
        "core/X/tests/test_a.py": "def f():\n    global _z\n",
        "api/v1/endpoints/e.py": (
            "X = get_y()\n"
            "async def route(dep=get_default()):\n"
            "    a = await get_db_pool()\n"
            "    b = get_settings(1)\n"
            "    c = mod.get_z()\n"
            "    def inner():\n        return get_w()\n"
            "    return get_q()\n"
        ),
        "api/v1/endpoints/sub/f.py": "def h():\n    return get_mcp_server()\n",
        "api/v1/API_Deps/d.py": (
            "import threading\nfrom cachetools import LRUCache\n"
            "_lock = threading.Lock()\n"
            "try:\n    _cache: LRUCache = LRUCache(maxsize=2)\nexcept ImportError:\n    _cache = {}\n"
            "_plain = {}\n"
            "def f():\n    lock = threading.Lock()\n"
            "class C:\n    lock = threading.RLock()\n"
        ),
    })
    assert measure_globals(root) == {"core/X/a.py": {"globals": 2}}
    assert measure_getter_calls(root) == {
        "api/v1/endpoints/e.py": {"calls": 3},  # get_db_pool, get_w (nested), get_q
        "api/v1/endpoints/sub/f.py": {"calls": 1},
    }
    assert measure_module_state(root) == {"api/v1/API_Deps/d.py": {"state": 2}}


@pytest.mark.parametrize(("ratchet", "measure", "field"), [(GLOBALS, measure_globals, "globals"), (GETTERS, measure_getter_calls, "calls"), (STATE, measure_module_state, "state")])
def test_growth_new_and_stale_entries_fail_and_shrinking_passes(tmp_path, ratchet, measure, field) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/a.py": "def f():\n    global a\ndef g():\n    global b\n",
        "api/v1/endpoints/e.py": "def r():\n    get_a()\n    get_b()\n",
        "api/v1/API_Deps/d.py": "import threading\nA = threading.Lock()\nB = threading.RLock()\n",
    })
    (key,) = measure(root)
    grown, gone, new = _ceilings.violations(ratchet, {**measure(root), "z.py": {field: 1}}, {key: {field: 1, "reason": None}, "gone.py": {field: 1, "reason": None}})
    assert grown.startswith(f"{key}: {field} 2 > ceiling 1")
    assert gone.startswith("gone.py: stale entry")
    assert new.startswith(f"z.py: {field} 1, and no ceiling")
    assert _ceilings.violations(ratchet, measure(root), {key: {field: 3, "reason": None}}) == []
```

`_seed` is the same helper as in `test_direct_llm_adapter_calls_ratchet.py:128-133`; copy it, since importing a `test_` module's names makes pytest collect them twice.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-t1" tldw_Server_API/tests/lint/test_service_singleton_ratchet.py`.
  Expected: errors with `NameError: name 'measure_globals' is not defined`.

- [ ] **Step 3: Implement the three counters and `__main__`** (Interfaces).
  - Walk `parse_tree(root)`.
  - The getter-call counter dedupes calls by node id, because `ast.walk` over nested functions visits a call once per enclosing function.
  - The module docstring states:
    - the three definitions above;
    - the known limits: getters called through an alias (`pool = get_db_pool; pool()`), state built by a module-level helper call, and `global` writes in comprehensions;
    - that each slice of TASK-13950.18 tightens it;
    - the commands.

- [ ] **Step 4: Seed the baselines.**
  Run: `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_service_singleton_ratchet --write-baseline`.
  Expected on b9eed09f91 (re-measure on the PR base; the script's output is the seed):
  - `tldw_Server_API/tests/lint/global_statements_baseline.json: 231 entries` (456 statements);
  - `tldw_Server_API/tests/lint/endpoint_getter_calls_baseline.json: 103 entries` (551 calls);
  - `tldw_Server_API/tests/lint/api_deps_module_state_baseline.json: 11 entries` (35 assignments);
  - spot checks: `"core/AuthNZ/database.py": {"globals": 2, ...}`, `"api/v1/endpoints/mcp_unified_endpoint.py": {"calls": 22, ...}` (12 `get_mcp_server`, 3 `get_jwt_manager`, 2 each of `get_config`, `get_settings` and `get_api_key_manager`, 1 `get_metrics_collector`) and `"api/v1/API_Deps/ChaCha_Notes_DB_Deps.py": {"state": 6, ...}`.

- [ ] **Step 5: Add the real-tree tests and wire the gate.**
  - `test_singleton_counts_do_not_grow(ratchet, measure)`, parametrized over `RATCHETS` with ids `globals`, `getters` and `state`, asserts `_ceilings.violations(ratchet, measure(), _ceilings.load(ratchet.baseline)) == []` and then calls `warn_if_tightenable`.
  - `test_raised_singleton_ceilings_carry_a_new_reason(ratchet, measure)`, with the same parametrization, asserts `_ceilings.reasonless_bumps(ratchet, measure()) == []`.
  - Make the workflow and contract edits from Files.

- [ ] **Step 6: Run the module and the contract.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-t1" tldw_Server_API/tests/lint/test_service_singleton_ratchet.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k "singleton or size_ratchets"`.
  Expected: 13 passed: 10 in the module (1 + 3 seeded, 3 + 3 real-tree) and 3 contract cases (the 2 existing ones and the new one). Then run C1.

- [ ] **Step 7: Commit.**

```bash
git add tldw_Server_API/tests/lint/test_service_singleton_ratchet.py tldw_Server_API/tests/lint/global_statements_baseline.json tldw_Server_API/tests/lint/endpoint_getter_calls_baseline.json tldw_Server_API/tests/lint/api_deps_module_state_baseline.json .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): freeze module globals, endpoint getter calls and API_Deps state (TASK-13950.18)"
```

### Task 2: Pin ADR-021's shutdown order before anything moves (AC5)

**Files:**
- Create: `tldw_Server_API/tests/unit/test_app_services_shutdown_order.py`, with `pytestmark = pytest.mark.unit`.
  - Nothing marks `tests/unit` automatically: there is no `tests/unit/conftest.py`, and 33 tests there are deselected today.
  - The marker is what puts the file in the required "Backend unit smoke" step (C3), which `test_backend_required_unit_smoke_runs_the_unit_tests` pins.

**Interfaces:**
- Consumes:
  - `services.shutdown_cleanup_timed_segments.shutdown_cleanup_timed_segments(**kwargs) -> CleanupTimedShutdownHandles`;
  - the private step helpers of `services.shutdown_resource_cleanup` and `services.shutdown_cleanup_timed_segments`, which the existing `tests/Services/test_shutdown_resource_cleanup.py` and `test_shutdown_cleanup_timed_segments.py` already patch by name.
- Produces: `ORDER: list[str]`, which later tasks must not edit:

```python
ORDER = [
    "db_pool", "session_manager", "mcp_server", "mcp_server.wait",
    "mcp_rate_limiter", "tts_service", "tts_resource_manager", "http_client",
    "chacha", "prompts", "chat_workflows", "provider_manager", "request_queue", "local_llm_manager",
    "evaluations", "unified_audit", "executors", "cpu_pools", "telemetry",
]
```

- [ ] **Step 1: Write the test.**
  - `pytestmark = pytest.mark.unit`.
  - `_Service(name, events)` is a module-level class:
    - `async close(self)`, `async shutdown(self)` and `async aclose(self, wait_timeout: float = 5.0)` each append `name`;
    - `async wait_for_shutdown_completion(self) -> bool` appends `f"{name}.wait"` and returns True.
  - `_step(name, events)` returns an `async def run(**_kwargs)` that appends `name`.
  - `@contextmanager def _untimed(_app, _name)` only yields.
  - Patch these with `monkeypatch.setattr(<module>, <helper>, _step(<label>, events))`:
    - in `shutdown_resource_cleanup`:
      - `_shutdown_mcp_rate_limiter` → `mcp_rate_limiter`;
      - `_shutdown_tts_service` → `tts_service`;
      - `_shutdown_tts_resource_manager` → `tts_resource_manager`;
      - `_shutdown_http_client` → `http_client`;
      - `_shutdown_prompts_resources` → `prompts`;
      - `_shutdown_chat_workflows_resources` → `chat_workflows`;
      - `_shutdown_provider_manager` → `provider_manager`;
      - `_shutdown_request_queue` → `request_queue`;
      - `_shutdown_local_llm_manager` → `local_llm_manager`;
    - in `shutdown_cleanup_timed_segments`:
      - `_shutdown_evaluations_resources` → `evaluations`;
      - `_shutdown_unified_audit_services` → `unified_audit`;
      - `_shutdown_executor_resources` → `executors`;
      - `_shutdown_cpu_pools` → `cpu_pools`;
      - `_shutdown_telemetry_services` → `telemetry`.
  - The ChaCha seam (until Task 10): `monkeypatch.setattr(ChaCha_Notes_DB_Deps, "shutdown_chacha_resources", _step("chacha", events))`.
  - Leave `_shutdown_auth_db_pool`, `_shutdown_session_manager`, `_shutdown_mcp_server` and `_shutdown_chacha_resources` real.

```python
async def test_final_cleanup_closes_the_container_services_in_adr_021_order(monkeypatch) -> None:
    events: list[str] = []
    ...  # the patches above
    handles = await timed.shutdown_cleanup_timed_segments(
        app=SimpleNamespace(state=SimpleNamespace()),
        db_pool=_Service("db_pool", events),
        session_manager=_Service("session_manager", events),
        heavy_startup_handles=SimpleNamespace(mcp_server=_Service("mcp_server", events), provider_manager=None, request_queue=None),
        in_pytest_for_db_pool_shutdown=False,
        in_pytest_for_tts_shutdown=False,
        import_exceptions=(ImportError,),
        startup_guard_exceptions=(RuntimeError,),
        timed_shutdown_segment=_untimed,
    )
    assert handles == timed.CleanupTimedShutdownHandles()
    assert events == ORDER
```

  The test asserts the real return value as well as the event list, so the test-quality triage does not flag it as `tautology_suspect`.

- [ ] **Step 2: Run it.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-t2" tldw_Server_API/tests/unit/test_app_services_shutdown_order.py`.
  Expected: PASS on dev (it characterizes current behaviour). To prove it fails for the right reason, swap the `_shutdown_session_manager` and `_shutdown_mcp_server` calls in `shutdown_resource_cleanup.py:25-32` locally. Expected: FAIL with a list diff at index 1. Revert the swap.

- [ ] **Step 3: Run the gates.**
  - Run C1 and C2 (the triage ratchet sees the new file).
  - Run C3 exactly as written. Expected on b9eed09f91: `428 passed, 33 deselected` (dev alone: `427 passed, 33 deselected`).
  - Prove the gate selects the file: run C3's command with `tldw_Server_API/tests/unit/test_app_services_shutdown_order.py` in place of `tldw_Server_API/tests/unit`. Expected: `1 passed`, with no `deselected`. Without the marker, the same command printed `1 deselected` in the dry run.
  - Commit.

```bash
git add tldw_Server_API/tests/unit/test_app_services_shutdown_order.py
git commit -m "test(lifecycle): pin ADR-021's final cleanup order for the services TASK-13950.18 containerises"
```

### Task 3: The container, the DB pool slot and the fresh-container fixture (AC2 db_pool, AC5)

**Files:**
- Create: `tldw_Server_API/app/core/Utils/app_services.py`.
- Modify: `tldw_Server_API/app/core/AuthNZ/database.py`.
  - Delete :1762-1763 (`# Global database pool instance` and `_db_pool: ...`).
  - `_get_db_pool_locked` (:1790-1841) reads `services = app_services()` and works on `services.db_pool` instead of `global _db_pool`. Name the local snapshot `pool`, not `_db_pool`, so no name suggests a module global.
  - `reset_db_pool` (:1865-1938): its pool part, :1865-1897, closes and empties `app_services().db_pool`. The cache resets after it are unchanged.
  - Import `from tldw_Server_API.app.core.Utils.app_services import app_services` after the settings import (:52).
- Modify: `tldw_Server_API/app/main.py`.
  - :1281: add `services = app_services()` and `app.state.services = services`. Import `app_services` function-locally, beside the other local imports in `lifespan` (:1272-1287).
  - :1306: delete `db_pool = startup_sequence_handles.db_pool`.
  - :1329: pass `db_pool=services.db_pool`.
- Modify: `tldw_Server_API/app/services/lifespan_startup_sequence.py` (delete the field at :21 and the argument at :110), `startup_core_initialization.py` (:17, :75) and `startup_auth_runtime.py` (delete the `db_pool` field at :15; :31 keeps its local `None` check and stops storing it).
- Modify tests:
  - `tests/Services/test_lifespan_startup_sequence.py:53,111,150,245,341` and `tests/Services/test_startup_core_initialization.py:25,88`: drop `db_pool` from the fakes and the asserts.
  - `tests/Services/test_startup_auth_runtime.py:80,124`: drop the `handles.db_pool` asserts; the `observed` call list stays.
  - `tests/Services/test_main_lifespan_cancellation.py:29,70`: drop `db_pool` from the handles namespace and delete the `kwargs["db_pool"]` assertion. That test is about cancellation; the new lifespan test owns "shutdown receives the container's pool", so the assertion lives in one place.
- Modify tests that touch `database._db_pool`. Rewrite `monkeypatch.setattr(database, "_db_pool", X)` as `monkeypatch.setattr(app_services(), "db_pool", X)`, and `database._db_pool` reads and writes as `app_services().db_pool`, in:
  - `tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py` (23 lines);
  - `tests/AuthNZ/unit/test_database_health_security.py:105`;
  - `tests/AuthNZ_Unit/test_auth_refresh_sqlite_connection_scope.py:73`;
  - `tests/Evaluations/conftest.py:74`.

  Each gets the `app_services` import after `from __future__ import annotations`.
- Create: `tldw_Server_API/tests/_plugins/app_services_fixtures.py`. Modify `tldw_Server_API/tests/conftest.py:8-14` to append `"tldw_Server_API.tests._plugins.app_services_fixtures",` to `pytest_plugins`.
- Create: `tldw_Server_API/tests/unit/test_app_services.py`.
- Baseline: `--tighten` deletes `"core/AuthNZ/database.py"` from `global_statements_baseline.json`.

**Interfaces:**
- Produces:
  - `AppServices` (dataclass), `app_services() -> AppServices`.
  - `async close_app_service(slot: str, *, restore: object = None) -> None`, `async discard_app_services() -> None`, `fresh_app_services() -> AbstractAsyncContextManager[AppServices]` and the fixture `fresh_services -> AppServices`, all in `tldw_Server_API.tests._plugins.app_services_fixtures`.
  - `close_app_service` is defined here, before anything calls it, so that G1 (Task 9) can build on B without C (Task 4).
  - `main.lifespan` sets `app.state.services`.
- Consumes: `reset_db_pool() -> Coroutine[None]` (unchanged signature).

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_app_services.py` (`pytestmark = pytest.mark.unit`).
  - An `authnz_sqlite` helper fixture sets `AUTH_MODE=multi_user` and `DATABASE_URL=sqlite:///<tmp>/users.db` with `monkeypatch.setenv`, then calls `reset_settings()`.
  - The module imports `main` (`tldw_Server_API.app`), `lifespan_startup_sequence` and `lifespan_shutdown_sequence` (`tldw_Server_API.app.services`), `AsyncMock`, `SimpleNamespace` and `FastAPI`.

```python
async def test_get_db_pool_fills_and_reuses_the_container_slot(fresh_services, authnz_sqlite) -> None:
    pool = await get_db_pool()
    assert fresh_services.db_pool is pool
    assert await get_db_pool() is pool


async def test_reset_db_pool_closes_and_empties_the_slot(fresh_services, authnz_sqlite) -> None:
    pool = await get_db_pool()
    await reset_db_pool()
    assert app_services().db_pool is None
    assert pool._initialized is False


async def test_fresh_app_services_empties_slots_then_closes_and_restores(monkeypatch, authnz_sqlite) -> None:
    previous = object()
    monkeypatch.setattr(app_services(), "db_pool", previous)
    async with fresh_app_services() as services:
        assert services is app_services()
        assert services.db_pool is None
        pool = await get_db_pool()
    assert app_services().db_pool is previous
    assert pool._initialized is False


async def test_lifespan_publishes_the_container_and_shuts_down_what_it_holds(monkeypatch) -> None:
    # main.py still reads .session_manager from the startup handles until Task 4.
    startup = AsyncMock(return_value=SimpleNamespace(heavy_startup_handles=None, session_manager=None))
    shutdown_spy = AsyncMock()
    monkeypatch.setattr(lifespan_startup_sequence, "run_lifespan_startup_sequence", startup)
    monkeypatch.setattr(lifespan_shutdown_sequence, "run_lifespan_shutdown_sequence", shutdown_spy)
    monkeypatch.setattr(main, "_run_startup_config_validation", lambda: None)
    pool = object()
    monkeypatch.setattr(app_services(), "db_pool", pool)
    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.services is app_services()
    assert startup.call_args.kwargs["app"] is app
    assert shutdown_spy.call_args.kwargs["db_pool"] is pool
```

- [ ] **Step 2: Run them.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-t3" tldw_Server_API/tests/unit/test_app_services.py`.
  Expected: errors with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Utils.app_services'`.

- [ ] **Step 3: Implement** `app_services.py` (Design), the `database.py` and lifespan edits, the handle-chain deletions and the plugin.
  - The plugin's `discard_app_services` and `fresh_app_services` follow Design. In this slice `discard_app_services` runs only `reset_db_pool()` and empties the slots. Tasks 4 and 9 each add one close step to it.
  - Use `dataclasses.fields`, with `f.default_factory()` when `f.default_factory is not MISSING`, so Task 4's lock field resets too.
  - `fresh_services` is `@pytest.fixture async def fresh_services(): async with fresh_app_services() as services: yield services`.

- [ ] **Step 4: Run the new tests, the retargeted tests and every test that names the touched symbols.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1318-t3" tldw_Server_API/tests/unit/test_app_services.py tldw_Server_API/tests/unit/test_app_services_shutdown_order.py tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py tldw_Server_API/tests/AuthNZ/unit/test_database_health_security.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_refresh_sqlite_connection_scope.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_startup_core_initialization.py tldw_Server_API/tests/Services/test_startup_auth_runtime.py tldw_Server_API/tests/Services/test_main_lifespan_cancellation.py tldw_Server_API/tests/Services/test_shutdown_auth_db_pool.py tldw_Server_API/tests/Services/test_startup_auth.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py tldw_Server_API/tests/Services/test_shutdown_resource_cleanup.py tldw_Server_API/tests/Services/test_shutdown_cleanup_timed_segments.py tldw_Server_API/tests/Evaluations/test_eval_test_mode_truthiness.py tldw_Server_API/tests/lint/test_layering_contract.py`.
  Expected: PASS.
  - The dry run retargeted all four `_db_pool` files: the three test files gave 48 passed together, and the layering contract gave 21 passed with the container in place.
  - `Evaluations/conftest.py` is an autouse fixture, so it runs through `test_eval_test_mode_truthiness.py` (3 tests, not gated by `RUN_EVALUATIONS`).
  Then both of these print nothing:
  - `grep -nE '^_db_pool:|^_db_pool =|global _db_pool' tldw_Server_API/app/core/AuthNZ/database.py`;
  - `git grep -nE '(database|authnz_db)\._db_pool|(database|authnz_db), "_db_pool"' -- tldw_Server_API/tests`.

- [ ] **Step 5: Tighten and run the gates.**
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_service_singleton_ratchet --tighten`. Expected diff: one line deleted, `core/AuthNZ/database.py`.
  - Run C1, C2, C3 and C4.
  - Run C6 on `tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py tldw_Server_API/tests/AuthNZ_Postgres` (the pool slot on Postgres).
  - Run C7 after pushing.

- [ ] **Step 6: Commit.**

```bash
git add tldw_Server_API/app/core/Utils/app_services.py tldw_Server_API/app/core/AuthNZ/database.py tldw_Server_API/app/main.py tldw_Server_API/app/services/lifespan_startup_sequence.py tldw_Server_API/app/services/startup_core_initialization.py tldw_Server_API/app/services/startup_auth_runtime.py tldw_Server_API/tests/_plugins/app_services_fixtures.py tldw_Server_API/tests/conftest.py tldw_Server_API/tests/unit/test_app_services.py tldw_Server_API/tests/Services tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py tldw_Server_API/tests/AuthNZ/unit/test_database_health_security.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_refresh_sqlite_connection_scope.py tldw_Server_API/tests/Evaluations/conftest.py tldw_Server_API/tests/lint/global_statements_baseline.json
git commit -m "feat(lifecycle): the AuthNZ DB pool lives in app.state.services; fresh_services replaces per-test pool resets (TASK-13950.18)"
```

### Task 4: The session manager slot (AC2 session_manager)

**Files:**
- Modify: `tldw_Server_API/app/core/Utils/app_services.py`. Add `session_manager` and `session_manager_lock` (Design).
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py:2075-2101`.
  - Delete the global and the lock.
  - `get_session_manager()` reads `services.session_manager`, serializes on `services.session_manager_lock` and publishes only after `initialize()` succeeds (keep `.4`'s order).
  - `reset_session_manager()` (kept until Task 8b) shuts down the slot's manager, empties the slot and replaces the lock with `asyncio.Lock()`.
- Modify: `tldw_Server_API/app/main.py:1307,1330`, `lifespan_startup_sequence.py:22,111`, `startup_core_initialization.py:18,76`, `startup_auth_runtime.py:16,36`. `startup_auth_runtime.py:36` still awaits `_get_session_manager()` to warm the slot; it does not store the result.
- Modify tests: drop `session_manager` from the fakes and asserts in the same four Services files:
  - `test_lifespan_startup_sequence.py:54,112,151,246,342`;
  - `test_startup_core_initialization.py:25,89`;
  - `test_startup_auth_runtime.py:81,125`;
  - `test_main_lifespan_cancellation.py:29,71`: delete the assertion at :71 too; the lifespan test owns it.
  - `tests/AuthNZ/unit/test_session_manager_init_race.py`:
    - `fresh_state` becomes `monkeypatch.setattr(app_services(), "session_manager", None)` plus `monkeypatch.setattr(app_services(), "session_manager_lock", asyncio.Lock())`;
    - `sm._session_manager is None` becomes `app_services().session_manager is None`;
    - `test_reset_recreates_lock` becomes `test_fresh_services_gives_a_fresh_session_init_lock`.
- Modify: the plugin. `discard_app_services` adds `await close_app_service("session_manager")` after `reset_db_pool()`, at its Design position.
- Modify: `tldw_Server_API/tests/conftest.py::_restore_auth_env_and_singletons` (root autouse, :373-421; Task 5 edits it too).
  - Before the `yield`: `session_before = app_services().session_manager`.
  - After it: `_run_coro_sync_best_effort(close_app_service("session_manager", restore=session_before))`, with `close_app_service` imported from `tldw_Server_API.tests._plugins.app_services_fixtures`. A manager the test published is shut down, and the slot goes back to what the test found, normally None. So no test inherits another's manager.
  - `.19` T6 relies on this and changes no root fixture (controller ruling 2026-10-30).
- Baseline: `--tighten` deletes `"core/AuthNZ/session_manager.py"` from the globals baseline.

**Interfaces:**
- Produces: `AppServices.session_manager: SessionManager | None` and `AppServices.session_manager_lock: asyncio.Lock`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_app_services.py`.
  - `test_get_session_manager_fills_the_slot_once_under_concurrency(fresh_services, monkeypatch)`:
    - patch `SessionManager.initialize` with an `async def` that sleeps 0.01 s, sets `self._initialized = True` and counts calls;
    - gather 20 `get_session_manager()` calls;
    - assert the set of ids has one element, `fresh_services.session_manager` is that object, and the count is 1.
  - In `tests/AuthNZ/unit/test_session_manager_init_race.py`, `test_reset_recreates_lock` is replaced by `test_fresh_services_gives_a_fresh_session_init_lock`: record `first = s.session_manager_lock` inside one `async with fresh_app_services() as s:` block, then assert the lock inside a second block `is not first`.
  - Extend the lifespan test: `monkeypatch.setattr(app_services(), "session_manager", manager)` and assert `kwargs["session_manager"] is manager`.
  - `tests/unit/test_app_services.py::test_root_autouse_empties_the_session_slot(fresh_services, monkeypatch)` exercises `close_app_service`, the helper the root fixture's teardown calls. `_StubManager` is a module-level class whose `shutdown()` counts its calls, or raises when built with `fail=True`. Three cases:
    - with `_StubManager()` in the slot, `close_app_service("session_manager")`: `shutdowns == 1` and the slot is None;
    - with `_StubManager(fail=True)`: the call returns without raising and the slot is None;
    - with `keep` in the slot, `close_app_service("session_manager", restore=keep)`: `keep.shutdowns == 0` and the slot is still `keep`.

- [ ] **Step 2: Run them.** Expected: FAIL with `AttributeError: 'AppServices' object has no attribute 'session_manager'`.

- [ ] **Step 3: Implement** (Files).

- [ ] **Step 4: Run.**
  Run `tests/unit/test_app_services.py tests/unit/test_app_services_shutdown_order.py tests/AuthNZ/unit/test_session_manager_init_race.py tests/AuthNZ/unit/test_singleton_edge_cases.py tests/AuthNZ/unit/test_session_manager_configured_key.py tests/AuthNZ/unit/test_session_manager_dep_production_guard.py tests/Services/` (the whole Services dir: lifecycle tests).
  Expected: PASS.
  Then run every test file from `git grep -lwE 'get_session_manager|reset_session_manager' -- tldw_Server_API/tests` whose import is `AuthNZ.session_manager` (filter out `Persona.session_manager`). Expected: same pass/fail set as dev.

- [ ] **Step 5: Tighten; run C1-C4, C6** (on `tests/AuthNZ/integration/test_token_blacklist_basic.py tests/AuthNZ_Postgres`) **and C7**. Commit:

```bash
git add tldw_Server_API/app/core/Utils/app_services.py tldw_Server_API/app/core/AuthNZ/session_manager.py tldw_Server_API/app/main.py tldw_Server_API/app/services/lifespan_startup_sequence.py tldw_Server_API/app/services/startup_core_initialization.py tldw_Server_API/app/services/startup_auth_runtime.py tldw_Server_API/tests/_plugins/app_services_fixtures.py tldw_Server_API/tests/conftest.py tldw_Server_API/tests/unit/test_app_services.py tldw_Server_API/tests/Services tldw_Server_API/tests/AuthNZ/unit/test_session_manager_init_race.py tldw_Server_API/tests/lint/global_statements_baseline.json
git commit -m "feat(lifecycle): the session manager and its init lock live in app.state.services (TASK-13950.18)"
```

### Task 5: The JWT service slot; `reset_jwt_service` is deleted (AC2 jwt, AC4)

**Files:**
- Modify: `app_services.py`. Add `jwt_service`.
- Modify: `tldw_Server_API/app/core/AuthNZ/jwt_service.py:1158-1183`.
  - Delete the global and `reset_jwt_service`.
  - Keep `_jwt_service_lock`.
  - `get_jwt_service()` takes its fast path when the slot is set, and otherwise double-checks under the lock and creates into the slot.
- Modify: `tldw_Server_API/tests/conftest.py:384-390,407-411`. The autouse `_restore_auth_env_and_singletons` replaces its two `reset_jwt_service()` blocks with `app_services().jwt_service = None`: every test starts and ends with an empty JWT slot.
- Modify: every other test call site. `git grep -n -w reset_jwt_service -- tldw_Server_API/tests` finds 27 calls in 16 files on b9eed09f91. Rewrite each call to `app_services().jwt_service = None`, and drop the import name (keep the import when other names remain):

```bash
git grep -l -w reset_jwt_service -- tldw_Server_API/tests | xargs perl -pi -e 's/^(\s*)(?:\w+\.)?_?reset_jwt_service\(\)\s*$/$1app_services().jwt_service = None\n/'
```

  Then fix the imports by hand: remove `reset_jwt_service` from each import list, and add `from tldw_Server_API.app.core.Utils.app_services import app_services` at the point where the removed import was (function-local imports stay function-local).
  One call site uses an alias the perl rule misses: `tests/AuthNZ/integration/test_auth_endpoints_integration.py:48-49` (`reset_jwt_service as _reset_jwt`, then `_reset_jwt()`). Rewrite it by hand.
  Check: `git grep -n -w reset_jwt_service -- tldw_Server_API` prints nothing.
- Modify: `tests/AuthNZ/test_admin_impersonation.py:81`. `monkeypatch.setattr(admin_impersonation, "get_jwt_service", ...)` becomes `monkeypatch.setattr(app_services(), "jwt_service", _StubJWTService())`; drop `raising=False`.
- Baseline: `--tighten` deletes `"core/AuthNZ/jwt_service.py"` from the globals baseline.

- [ ] **Step 1: Write the failing test.** `test_get_jwt_service_fills_the_slot_once_across_threads(fresh_services, monkeypatch)`:
  - patch `jwt_service.JWTService` with a class whose `__init__` sleeps 0.05 s and appends itself to `built`;
  - call `get_jwt_service()` from 8 threads behind a `threading.Barrier(8)`;
  - assert `len(built) == 1` and that every result `is fresh_services.jwt_service`.

- [ ] **Step 2: Run it.** Expected: FAIL, `AttributeError ... 'jwt_service'`.

- [ ] **Step 3: Implement** (Files).

- [ ] **Step 4: Run.**
  Run: `"$PY" -m pytest ... tldw_Server_API/tests/unit/test_app_services.py tldw_Server_API/tests/unit/test_app_services_shutdown_order.py $(git grep -l "app_services().jwt_service" -- tldw_Server_API/tests) tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit`.
  Expected: same pass/fail set as dev for the converted files; the new test passes.

- [ ] **Step 5: Tighten; run C1-C4; C7 (the root conftest changed).** Commit:

```bash
git add tldw_Server_API/app/core/Utils/app_services.py tldw_Server_API/app/core/AuthNZ/jwt_service.py tldw_Server_API/tests/lint/global_statements_baseline.json
git add -A tldw_Server_API/tests
git commit -m "feat(lifecycle): the JWT service lives in app.state.services; reset_jwt_service is gone (TASK-13950.18)"
```

### Task 6: The MCP server slot and its Depends provider (AC2 MCP, AC5)

**Files:**
- Modify: `app_services.py`. Add `mcp_server`.
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py:2314-2340`.
  - Delete `_server`; keep `_server_lock`.
  - `get_mcp_server()` creates into the slot under the lock (double-checked).
  - `reset_mcp_server()` shuts down the slot's server and empties the slot, then resets the module registry as today.
- Create: `tldw_Server_API/app/api/v1/API_Deps/mcp_deps.py` with `async def get_mcp_server_dep() -> MCPServer: return get_mcp_server()`. It is async so a first-time `MCPServer()` is built on the event loop, not in FastAPI's threadpool.
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`. Each of these routes gets a last parameter `server: MCPServer = Depends(get_mcp_server_dep)`, and its `server = get_mcp_server()` line is deleted:

| Route | Defined at | Deleted call at |
|---|---|---|
| `mcp_request` | :968 | :994 |
| `mcp_request_batch` | :1066 | :1091 |
| `get_server_status` | :1163 | :1176 |
| `get_server_metrics` | :1187 | :1200 |
| `list_tools` | :1259 | :1297 |
| `execute_tool` | :1434 | :1455 |
| `list_modules` | :1552 | :1564 |
| `get_modules_health` | :1604 | :1616 |
| `list_resources` | :1652 | :1664 |
| `list_prompts` | :1696 | :1710 |
| `health_check` | :1850 | :1858 |

  `websocket_endpoint` (:873) keeps `get_mcp_server()` after its cookie rejection (Ruling 1), so the `.7` WebSocket auth ratchets see no change.
  Imports: add `MCPServer` to the `from tldw_Server_API.app.core.MCP_unified import ...` line (:55), and add `from tldw_Server_API.app.api.v1.API_Deps.mcp_deps import get_mcp_server_dep`. `get_mcp_server` stays imported for the WebSocket route.
- Modify tests: 40 patch sites in 11 files. These are the 34 `get_mcp_server` patches on the endpoint module and the 6 `_server` patches on `MCP_unified.server`, found by the seam scanner (Tool), which resolves the `mcp_ep`, `mcp`, `mcp_unified_endpoint` and `mcp_server_module` aliases.
  - 7 files hit the routes over HTTP or WebSocket. Each rewrites `monkeypatch.setattr(<mcp module>, "get_mcp_server", lambda: X)` as `monkeypatch.setattr(app_services(), "mcp_server", X)`; the two `test_mcp_ws_*` files are WebSocket tests:
    - `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py` (8);
    - `tests/MCP_unified/test_mcp_prompts_http.py` (2);
    - `tests/MCP_unified/test_mcp_ws_connection_release_on_cancel.py` (2);
    - `tests/MCP_unified/test_mcp_ws_workspace_context.py` (5);
    - `tests/Resource_Governance/test_e2e_chat_audio_headers.py` (1);
    - `tests/AuthNZ/integration/test_auth_principal_mcp_monitoring_invariants.py` (1);
    - `tests/AuthNZ_Unit/test_mcp_admin_permissions_claims.py` (1).
  - The 2 files that call routes directly, `tests/unit/test_mcp_unified_error_mapping.py` (9) and `tests/MCP_unified/test_mcp_http_auth_paths.py` (5), pass `server=X` to the converted routes. The websocket call there uses the slot.
  - `tests/AuthNZ/unit/test_session_manager_init_race.py::test_get_mcp_server_single_instance_across_threads`: `monkeypatch.setattr(mcp_server, "_server", None)` becomes `monkeypatch.setattr(app_services(), "mcp_server", None)`, and the test also asserts `results[0] is app_services().mcp_server`.
  - `tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py:729,755,780,837,944`, the RAG source-permission tests: `monkeypatch.setattr(mcp_server_module, "_server", SimpleNamespace(protocol=protocol))` becomes `monkeypatch.setattr(app_services(), "mcp_server", SimpleNamespace(protocol=protocol))`. They name neither getter, so a grep-built run list would miss them.
- Baselines (`--tighten`):
  - delete the globals entry `core/MCP_unified/server.py`;
  - lower `api/v1/endpoints/mcp_unified_endpoint.py` from 22 to 11. The 11 converted routes' `get_mcp_server()` calls go; the WebSocket route's call and the 10 other getters stay;
  - total calls: 551 → 540.

- [ ] **Step 1: Write the failing tests.**
  - `test_heavy_init_returns_the_container_mcp_server(fresh_services, monkeypatch)`:
    - patch `server.MCPServer` with a class whose `async def initialize(self)` sets `self.initialized = True`;
    - `handle = await startup_heavy_init._init_mcp_server(SimpleNamespace(state=SimpleNamespace()), deferred=False)`;
    - assert `handle is fresh_services.mcp_server` and `handle.initialized is True`.
  - `test_mcp_routes_take_the_server_from_the_container(fresh_services, monkeypatch)`:
    - set the slot to an object whose `async def get_status()` returns `{"status": "healthy"}` and whose `initialized` is True;
    - build `app = FastAPI()`, `app.include_router(mcp_ep.router)` and `app.dependency_overrides[mcp_ep.enforce_http_security] = lambda: None` (one override: no `stub_injection` flag), then call `GET /mcp/health` through `TestClient(app)`;
    - assert 200 and `{"status": "healthy"}`;
    - assert `inspect.signature(mcp_ep.health_check).parameters["server"].default.dependency is get_mcp_server_dep`.

- [ ] **Step 2: Run them, each alone** (`-k <name>`).
  - `test_mcp_routes_take_the_server_from_the_container` fails at `monkeypatch.setattr` with `AttributeError: ... has no attribute 'mcp_server'`.
  - `test_heavy_init_returns_the_container_mcp_server` fails at the final `fresh_services.mcp_server` read with the same `AttributeError`. Before failing, `get_mcp_server()` has stored the fake `MCPServer` in the module global `_server`, which is why it runs alone.

- [ ] **Step 3: Implement** (Files).

- [ ] **Step 4: Run.**
  - Run:
    - the 11 retargeted files;
    - `tests/unit/test_app_services.py tests/unit/test_app_services_shutdown_order.py`;
    - `tests/Services/test_shutdown_resource_cleanup.py tests/Services/test_startup_core_initialization.py`;
    - every file from `git grep -lwE 'get_mcp_server|reset_mcp_server' -- tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests`, plus the two files the seam scanner lists for `_server` (`... tldw_Server_API.app.core.MCP_unified.server --sites _server` on the PR base). A grep for `_server` would also match every `self._server` attribute.

    Expected: the same pass/fail set as dev.
  - Alias-aware checks: `"$PY" "$TMPDIR/t1318_seams.py" tldw_Server_API.app.core.MCP_unified.server | grep -E "^_server(_lock)?:"` and `"$PY" "$TMPDIR/t1318_seams.py" tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint | grep "^get_mcp_server:"` print nothing. On b9eed09f91 they print `_server: patch 6; 2 files` and `get_mcp_server: patch 34; 9 files`.
  - Run `tests/AuthNZ/test_websocket_route_auth_matrix.py tests/lint/test_auth_token_parsing_ratchet.py` (the `.7` guards). Expected: PASS.

- [ ] **Step 5: Tighten; run C1-C5 and C7** (C5 must report no drift: dependencies with no parameters add nothing to the schema). Commit:

```bash
git add tldw_Server_API/app/core/Utils/app_services.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/api/v1/API_Deps/mcp_deps.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/tests/lint/global_statements_baseline.json tldw_Server_API/tests/lint/endpoint_getter_calls_baseline.json
git add -A tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests
git commit -m "feat(mcp): the MCP server lives in app.state.services and reaches the HTTP routes through Depends (TASK-13950.18)"
```

### Task 7: The reset codemod over async test bodies (AC4, AC6), PRs F1a and F1b

**Files:** the test files the script below changes. F1a takes every changed path under `tldw_Server_API/tests/AuthNZ*` (45 files on the dry run). F1b takes the rest (38).

**Interfaces:** consumes `fresh_services` (Tasks 3-6 cover the db, session, JWT and MCP slots).

- [ ] **Step 1: Save the codemod** as `"$TMPDIR/t1318_reset_codemod.py"`. It is not committed. Its rules:
  - A *reset run* is a maximal run of consecutive one-line expression statements in one block, each a bare call (awaited or not, no arguments) named `reset_db_pool`, `_reset_db_pool`, `reset_session_manager`, `_reset_session_manager`, `reset_settings` or `_reset_settings`.
  - Inside an `async def test*` function, a run that contains a `reset_db_pool` call loses its `reset_db_pool` and `reset_session_manager` calls.
  - If no `reset_settings` call is left in a run outside a `finally:` block, the first dropped call becomes `<name>()`, where `<name>` is the reset_settings spelling the file already uses. When the file has none, the run is left alone and reported as skipped.
  - A block left empty gets `pass`.
  - Each changed function gets a `fresh_services` parameter.
  - An import name from `AuthNZ.database` or `AuthNZ.session_manager` that is no longer used is removed.
  - A test that loses more than one non-`finally` run (a mid-test reconfiguration) is printed as a `MULTI-RUN` line, so the PR's reviewer reads those conversions.

```python
"""TASK-13950.18 codemod: async test bodies drop their AuthNZ reset runs and request fresh_services."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

DROP = {"reset_db_pool", "_reset_db_pool", "reset_session_manager", "_reset_session_manager"}
SETTINGS = {"reset_settings", "_reset_settings"}
DB = {"reset_db_pool", "_reset_db_pool"}
SOURCES = ("AuthNZ.database", "AuthNZ.session_manager")


def name_of(stmt):
    if not isinstance(stmt, ast.Expr):
        return None
    value = stmt.value.value if isinstance(stmt.value, ast.Await) else stmt.value
    if not isinstance(value, ast.Call) or value.args or value.keywords:
        return None
    func = value.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def runs(body):
    run = []
    for stmt in [*body, None]:
        if stmt is not None and name_of(stmt) in DROP | SETTINGS and stmt.lineno == stmt.end_lineno:
            run.append(stmt)
            continue
        if run:
            yield run
        run = []


def convert(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    tree, lines = ast.parse(text), text.splitlines(keepends=True)
    settings = next((name_of(s) for s in ast.walk(tree) if isinstance(s, ast.stmt) and name_of(s) in SETTINGS), None)
    edits, changed, skipped = {}, [], 0
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name.startswith("test")]:
        touched, converted_runs = False, 0
        for owner in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(owner, field, None)
                if not isinstance(body, list):
                    continue
                for run in runs(body):
                    if not DB & {name_of(s) for s in run}:
                        continue
                    drops = [s for s in run if name_of(s) in DROP]
                    if field != "finalbody" and not any(name_of(s) in SETTINGS for s in run):
                        if settings is None:
                            skipped += 1
                            continue
                        indent = re.match(r"\s*", lines[drops[0].lineno - 1]).group(0)
                        edits[drops[0].lineno - 1] = f"{indent}{settings}()\n"
                        drops = drops[1:]
                    for s in drops:
                        edits[s.lineno - 1] = ""
                    touched = True
                    converted_runs += field != "finalbody"
                if touched and body and all(edits.get(s.lineno - 1) == "" and s.lineno == s.end_lineno for s in body):
                    edits[body[0].lineno - 1] = re.match(r"\s*", lines[body[0].lineno - 1]).group(0) + "pass\n"
        if touched:
            changed.append(fn)
        if converted_runs > 1:
            print(f"MULTI-RUN {path}::{fn.name} ({converted_runs} setup runs dropped: read this conversion)")
    if not edits:
        return f"{path}: unchanged ({skipped} skipped)"
    for fn in changed:
        if "fresh_services" in [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]:
            continue
        start, stop = fn.lineno - 1, fn.body[0].lineno - 1
        while stop > start and not lines[stop - 1].rstrip().endswith(":"):
            stop -= 1
        sig = "".join(lines[start:stop])
        if fn.args.posonlyargs or fn.args.args or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
            sig = re.sub(r",?(\s*)\)(\s*(?:->[^:]*)?):(\s*(?:#.*)?)$", r", fresh_services\1)\2:\3", sig.rstrip("\n"), count=1) + "\n"
        else:
            sig = re.sub(r"\(\s*\)", "(fresh_services)", sig, count=1)
        for i in range(start, stop):
            edits[i] = ""
        edits[start] = sig
    out = "".join(edits.get(i, line) for i, line in enumerate(lines))
    tree2 = ast.parse(out)
    used = {n.id for n in ast.walk(tree2) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree2) if isinstance(n, ast.Attribute)}
    olines = out.splitlines(keepends=True)
    for node in sorted((n for n in ast.walk(tree2) if isinstance(n, ast.ImportFrom) and (n.module or "").endswith(SOURCES)), key=lambda n: -n.lineno):
        keep = [a for a in node.names if not (a.name in DROP and (a.asname or a.name) not in used)]
        if len(keep) != len(node.names):
            indent = re.match(r"\s*", olines[node.lineno - 1]).group(0)
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
            olines[node.lineno - 1 : node.end_lineno] = [f"{indent}from {node.module} import {names}\n"] if keep else []
    out = "".join(olines)
    ast.parse(out)
    path.write_text(out, encoding="utf-8")
    return f"{path}: {len(changed)} tests changed, {skipped} runs skipped"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(convert(Path(arg)))
```

- [ ] **Step 2: Run it on every test file that calls `reset_db_pool`.**
  Run: `"$PY" "$TMPDIR/t1318_reset_codemod.py" $(git grep -l -w reset_db_pool -- tldw_Server_API/tests) | tee "$TMPDIR/t1318_codemod.log" | grep -v unchanged`. Then `git diff --name-only -- tldw_Server_API/tests > "$TMPDIR/t1318_converted.txt"`.
  Expected on b9eed09f91: 83 files changed, plus 8 `MULTI-RUN` lines:
  - `test_token_blacklist_basic.py` (2 tests);
  - `test_org_stt_settings_repo_sqlite.py` (1);
  - `test_privilege_service_sqlite.py` (4);
  - `test_privilege_snapshot_retention.py` (1).

  Afterwards:
  - `git grep -l -w reset_db_pool -- tldw_Server_API/tests | wc -l` prints 88 (168 before);
  - `git grep -l -w reset_session_manager -- tldw_Server_API/tests | wc -l` prints 12 (32 before);
  - every changed file parses (the script asserts it).

  `git diff --check` must print nothing; it was already clean in the dry run.
- [ ] **Step 3: Compare each converted file's full outcomes with the base.**
  - Use a scratch worktree of the PR base, with `export BASE_WT=<its path> PY TMPDIR`.
  - Save the loop below as `"$TMPDIR/t1318_compare.sh"` and run `bash "$TMPDIR/t1318_compare.sh" "$TMPDIR/t1318_converted.txt"`.
  - It uses C1's env and compares every `PASSED`, `FAILED`, `ERROR`, `SKIPPED`, `XFAIL` and `XPASS` line, after stripping line numbers (the codemod shifts them). So a test that newly skips counts as a difference.

  ```bash
  set -u
  while read -r f; do
    for side in base head; do
      d=$([ "$side" = base ] && echo "$BASE_WT" || echo .)
      (cd "$d" && BACKLOG_TASK_FORMAT_BASE=origin/dev PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
        "$PY" -m pytest -q -rA -p no:cacheprovider -p pytest_asyncio.plugin --basetemp "$TMPDIR/t1318-cmp" "$f" 2>&1 \
        | grep -E "^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) " | sed -E "s/ - .*//; s/:[0-9]+:/:/" | sort) > "$TMPDIR/t1318-$side.txt"
    done
    cmp -s "$TMPDIR/t1318-base.txt" "$TMPDIR/t1318-head.txt" && echo "SAME: $f" || echo "DIFFERS: $f"
  done < "$1"
  ```

  - Pass: every line is `SAME`.
  - In the dry run, `Usage/test_usage_tracker_sqlite.py` reported `SAME` with 10 outcomes, including `test_log_llm_usage_derives_token_name_from_key[postgres]` passing on both sides, because the Postgres fixture auto-started.
  - A `[postgres]` case that is `SKIPPED` on both sides proves nothing; C6 covers it (Step 4).
  - A `DIFFERS` file is reverted with `git checkout origin/dev -- <file>` and removed from `t1318_converted.txt`. Task 8a then converts it by hand, or it keeps calling `reset_db_pool` if Task 8a's count still ends at or under 49.
- [ ] **Step 4: Split and push.** F1a takes the AuthNZ* paths; F1b takes the rest.
  - Each PR body carries:
    - the script, pasted in a collapsed block, and the base it ran on;
    - the files Step 3 reverted, so a reviewer who re-runs the script and `git diff`s the result can account for them;
    - the `MULTI-RUN` lines.
  - Run C1-C3 and C7 on each PR.
  - Run C6 on F1a: `$(git diff --name-only origin/dev -- 'tldw_Server_API/tests/AuthNZ*')`.
  - Run C6 on F1b: `tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py tldw_Server_API/tests/Usage/test_usage_aggregator.py`. In general, every converted file that requests a `pg_*` fixture, is parametrized with `postgres`, or skips on a non-Postgres `DATABASE_URL`: `grep -lE 'pg_temp_db|pg_server|"postgres"|startswith\("postgresql"\)' $(cat "$TMPDIR/t1318_converted.txt")`.
- [ ] **Step 5: Commit** (per PR). Stage exactly the converted files of that PR:

```bash
git add -- $(grep '^tldw_Server_API/tests/AuthNZ' "$TMPDIR/t1318_converted.txt")    # F1a
git add -- $(grep -v '^tldw_Server_API/tests/AuthNZ' "$TMPDIR/t1318_converted.txt") # F1b, on F1a's branch
git commit -m "test: async tests take fresh_services instead of resetting the AuthNZ pool and session manager (TASK-13950.18, generated)"
```

Dry-run result for the other 44 files (Admin, AuthNZ, AuthNZ_Unit, Audio, MCP_Hub, MCP_unified, MediaIngestion_NEW, Privileges, Resource_Governance, UserProfile): dev and the converted tree both gave 179 passed, 1 xfailed, with no failure on either side. So all 83 converted files matched dev in the dry run.

Two notes on that dry run:
- Its fixture emptied only the pool slot, and its codemod variant also dropped `reset_jwt_service`. Dropped `reset_session_manager` calls were therefore replaced by nothing, a stricter test than Task 4's fixture, which empties the session slot too.
- The script exactly as printed above was then re-run on a tree with Task 5's `reset_jwt_service` rewrite applied. It again changed 83 files and left 88 `reset_db_pool` files and 12 `reset_session_manager` files, and `git diff --check` was clean.

Files whose runs the codemod skipped stay in Task 8a Step 3's count.

### Task 8a: Fixtures and helpers take the fresh container; AC6 drops under 50 (PR F2a)

**Files.** The 43 files below. Each conversion:
- a fixture requests `fresh_services` and deletes its `reset_db_pool`/`reset_session_manager` calls;
- a helper drops the calls, and every test or fixture that calls the helper requests `fresh_services`;
- the env and `reset_settings()` lines stay where they are.

| Group | Files (under `tldw_Server_API/tests/`) |
|---|---|
| Single-file async fixtures (14) | `Admin/test_admin_backup_jobs.py`, `Admin/test_admin_backup_scheduler.py`, `Admin/test_admin_maintenance_rotation_jobs.py`, `AuthNZ_SQLite/test_admin_users_single_user_uuid_sqlite.py`, `AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py`, `AuthNZ_SQLite/test_authnz_rate_limits_repo_sqlite.py`, `AuthNZ/integration/test_api_key_rotation_audit.py`, `AuthNZ/integration/test_api_key_rotation_boundaries.py`, `AuthNZ/integration/test_single_user_cookie_session.py`, `AuthNZ/integration/test_single_user_login_api_key.py`, `Chatbooks/test_chatbooks_full_account_export_contract.py`, `Claims_Extraction/test_fva_verify_endpoint.py`, `MCP_unified/test_mcp_external_policy_revocation.py`, `MCP_unified/test_mcp_hub_targeted_discovery.py` |
| Single-file sync fixtures (7) | `AuthNZ_Federation/test_identity_provider_admin_api.py`, `AuthNZ_Federation/test_oidc_login_flow.py`, `Files/test_files_endpoint.py`, `Files/test_files_image_endpoint.py`, `Files/test_files_reference_images_endpoint.py`, `Privileges/test_privilege_endpoints.py`, `Watchlists/test_include_org_default.py` |
| Helpers (22) | `Resource_Governance/{test_coverage_endpoint_limit,test_e2e_chat_audio_headers,test_e2e_evals_authnz_character_headers,test_e2e_tokens_daily_cap,test_e2e_workflows_daily_cap,test_resource_governor_endpoint,test_rg_capabilities_endpoint,test_tokens_daily_cap,test_workflows_runs_daily_cap}.py` (the same `_init_authnz_sqlite` helper each), `MCP_unified/{test_mcp_hub_auth_template_status,test_mcp_hub_governance_pack_distribution,test_mcp_hub_governance_pack_import,test_mcp_hub_policy_overrides}.py`, `Integrations/{test_workspace_installation_registry_callbacks,test_workspace_provider_installations_repo}.py`, `Admin/{test_admin_conflicts,test_admin_conflicts_edgecases,test_admin_e2e_support_api,test_admin_llm_usage_reads,test_backup_schedules_api,test_bundle_ops,test_data_subject_requests_api}.py` |

A sync fixture that requests the async `fresh_services` keeps working on pytest-asyncio 1.4; this was verified in the dry run.

- Files added on dev after b9eed09f91 join the group their shape matches. #3371 added `Usage/test_llm_quota_byok_rows.py`: its helper calls `reset_db_pool` and `reset_session_manager`, so it joins Helpers, giving 44 files. Re-run `git grep -l -w reset_db_pool -- tldw_Server_API/tests` on the PR base and place any newcomer the same way.

- [ ] **Step 1: Convert the files** (rules above). None of the 22 helper files calls its helper twice in one test (checked).
- [ ] **Step 2: Compare each file with the base** (Task 7 Step 3's loop). Expected: no `DIFFERS`.
- [ ] **Step 3: Check AC6.** `git grep -l -w reset_db_pool -- tldw_Server_API/tests | wc -l` prints at most 45: 88 - 43 on b9eed09f91, and 89 - 44 once #3371's file is counted. Any file Task 7 Step 3 reverted adds 1, and the sum must stay at most 49.
  - The 45 files that keep calling or naming it, as measured on the dry-run tree:
    - 5 conftest fixtures and helpers: `AuthNZ/`, `Admin/`, `Evaluations/`, `UserProfile/`, `MediaIngestion_NEW/`;
    - `_plugins/authnz_fixtures.py`;
    - `MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py`;
    - 9 more async helpers;
    - 5 sync helpers, including the root conftest's `pytest_sessionfinish`;
    - 9 files that only patch or reference it;
    - 11 async tests whose runs the codemod skipped, including `AuthNZ/unit/test_database_openai_credential_lock_pool.py`, which tests `reset_db_pool` itself;
    - 4 sync tests.
  - Task 8b converts two of the async helpers, which brings the count to 43. `AuthNZ/conftest.py` keeps its bootstrap `reset_db_pool()` (Task 8b).
- [ ] **Step 4: Run C1-C3 and C7; C6** on the AuthNZ files in the table. Commit:

```bash
git add -A tldw_Server_API/tests
git commit -m "test: fixtures and helpers take fresh_services; reset_db_pool test callers fall below 50 (TASK-13950.18)"
```

### Task 8b: `reset_session_manager` is deleted (AC4) (PR F2b)

**Rule.** Every remaining `await reset_session_manager()` becomes `await discard_app_services()`, imported from `tldw_Server_API.tests._plugins.app_services_fixtures`. An `await reset_db_pool()` in the same reset run is deleted, because `discard_app_services` runs it.

`discard_app_services` closes and empties every slot, as `reset_db_pool` plus `reset_session_manager` do today:
- an inherited pool or session manager is closed, not snapshotted and handed back;
- the call sits exactly where the old calls sat, so no `async with` moves into a fixture body.

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ/conftest.py`.
  - `reset_singletons` (autouse): the pre-test pair at :498-499 and the post-test pair at :606-607 each become `await discard_app_services()`. Delete the `app_services().jwt_service = None` lines Task 5 left at :506 and :614 (`discard_app_services` empties that slot too), and the `reset_session_manager` name from the import block at :479-490. Task 5 already removed `reset_jwt_service` there. `close_all_chacha_db_instances` stays imported from the deps module; Task 10 keeps it there.
  - `isolated_test_environment`, step 5 (:1363-1364): the pair becomes `await discard_app_services()`.
    - The bootstrap `try: ... finally: await reset_db_pool()` (:1377-1382) stays as it is. It closes the fixture-loop pool before `TestClient` builds its own on the app's loop.
    - The `with TestClient(_app) as client: yield client, db_name` block (:1399-1400) is unchanged. Lifespan startup fills the slots there, and the test sees them.
  - `isolated_test_environment`, step 8 (:1403-1404): the pair becomes `await discard_app_services()`.
- Modify: `tests/AuthNZ/unit/test_session_manager_configured_key.py`, at :54, :77, :108, :114, :153, :162, :207, :216, :250 and :284 (10 calls). These tests construct `SessionManager()` directly, so each call is replaced by the rule. A plain deletion would be equivalent; the rule keeps the edit mechanical.
- Modify: `tests/AuthNZ/unit/test_singleton_edge_cases.py:59,76`, by the rule.
- Modify: `tests/Usage/test_llm_usage_recorded_through_executor.py` (`_fresh_pool`, :54-55) and `tests/performance/test_authnz_multiuser_sqlite_load.py` (`_build_app`, :84-85): each pair becomes `await discard_app_services()`.
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py`. Delete `reset_session_manager`.
- Check: `git grep -n -w reset_session_manager -- tldw_Server_API` prints nothing.

- [ ] **Step 1: Convert, delete and run** `tests/AuthNZ` (the whole directory, because its conftest changed) on SQLite and in C6 (`TLDW_TEST_POSTGRES_REQUIRED=1`). Expected: the same pass/fail set as dev on both backends. Use Task 7 Step 3's comparison loop over the directory's files.
- [ ] **Step 2: Run C1-C3 and C7.** Commit:

```bash
git add tldw_Server_API/app/core/AuthNZ/session_manager.py
git add -A tldw_Server_API/tests
git commit -m "refactor(authnz): delete reset_session_manager; AuthNZ fixtures discard the container (TASK-13950.18)"
```

### Task 9: ChaCha state moves into `ChaChaRuntimeManager` (AC3 runtime half, AC4) (PR G1)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/runtime.py`. It receives the G1 rows of the name table below as moved code, with module globals rewritten as instance attributes.
- Modify: `app_services.py`. Add `chacha_runtime`.
- Modify: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`.
  - Delete :50-55 and :65-82 (executor, health and shutdown globals, `_CHACHA_RUNTIME`) and :359-369 (cache, locks, init maps, task sets), together with the functions the table moves.
  - Keep the "Stays in deps" rows: names stay, and the bodies delegate to `get_chacha_runtime()`.
  - `_get_or_init_db_instance(user_id, client_id)` stays as the HTTP-mapping wrapper. It calls `await get_chacha_runtime().get_db(str(DatabasePaths.get_user_base_directory(user_id)), partial(_create_and_prepare_db, user_id, client_id), user_id=user_id)`, mapping errors in Interfaces order.
- Modify tests: every row of the name table with a G1 target, by rules R1-R7. That is about 155 test uses, plus the 6 `importlib.reload` sites (R3).
- Modify: `tldw_Server_API/tests/_plugins/singleton_guard.py`. The `_chacha_db_instances` watch reads `len(app_services().chacha_runtime.cached_instances())` from module `tldw_Server_API.app.core.Utils.app_services` (None when the slot is empty).
- Modify: `tldw_Server_API/tests/conftest.py:450-461, :800-806, :1203-1212`. Each of the three after-test ChaCha blocks becomes one call, `_run_coro_sync_best_effort(close_app_service("chacha_runtime"))` (the Task 3 helper). Every test then gets a fresh runtime.
- Modify: the plugin. `discard_app_services` adds `await close_app_service("chacha_runtime")` as its last close step (Design order). If Task 4 has not landed, that is directly after `reset_db_pool()`.
- Modify: `tests/Chat/test_chacha_runtime_contract.py:9-16`. `reset_for_tests` becomes `resume`.
- Create: `tldw_Server_API/tests/unit/test_chacha_runtime_container.py` (Step 1).
- Baselines (`--tighten`): delete the state entry `api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` (6 → 0) and its globals entry (3 → 0).

**Name table.** All 53 names Step 0's scan prints on b9eed09f91, with their test uses. "G1" is this task's target; "G2" is Task 10's.

| Deps name | Test uses | Target |
|---|---|---|
| **Stays in deps** (G1 delegates to the runtime; G2 re-exports or wraps the provider) | | |
| `get_chacha_db_for_user` | attr 31, import 165 (155 files) | unchanged FastAPI dependency |
| `get_chacha_db_for_user_id`, `get_chacha_db_for_owner` | 20, 8 | HTTP wrappers (G1: unchanged bodies over the runtime; G2: Task 10 Interfaces) |
| `_get_or_init_db_instance` | attr 20, patch 2 | HTTP-mapping wrapper (above) |
| `close_all_chacha_db_instances`, `shutdown_chacha_executor`, `shutdown_chacha_resources`, `reset_chacha_shutdown_state`, `get_chacha_health_snapshot` | 19, 2, 10, 10, 12 | one-line delegations to `runtime.close_all`, `.shutdown_executor`, `.aclose`, `.resume` and `.health_snapshot` (G1); re-exports of the provider (G2) |
| `warm_chacha_db_for_user` | 2 | G1: stays in deps with today's body, except that it uses `runtime.count("warm_startups")` and `runtime.track(task)`. G2: moves to the provider and is re-exported |
| `get_request_user`, `DEFAULT_CHARACTER_NAME`, `DatabasePaths` | 3, 38, 30 | unchanged imports; tests patch `deps.DatabasePaths.<method>` on the shared class (R4) |
| `ChaChaDatabaseCorruptionError`, `_CHACHA_CORRUPTION_ERROR_DETAIL`, `_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL` | 2, 2, 3 | re-exported from `chacha.runtime` |
| **G1: runtime state** (`runtime = get_chacha_runtime()`) | | |
| `_CHACHA_RUNTIME` | patch 7 | `monkeypatch.setattr(app_services(), "chacha_runtime", ChaChaRuntimeManager())` |
| `_chacha_db_instances`, `_chacha_db_lock` | 14, 8 | `runtime._instances`, `runtime._cache_lock` |
| `_chacha_db_init_events`, `_chacha_db_init_errors` | 15, 16 | `runtime._init_events`, `runtime._init_errors` |
| `_chacha_default_char_tasks`, `_chacha_default_char_futures` | 28, 4 | `runtime._tasks`, `runtime._futures` |
| `_CHACHA_HEALTH`, `_CHACHA_HEALTH_LOCK` | 8, 5 | `runtime._health`, `runtime._health_lock` |
| `_CHACHA_SHUTTING_DOWN` | patch 6 | `runtime._shutting_down` (the deps flag and the runtime flag merge) |
| `_CHACHA_EXECUTOR`, `_CHACHA_EXECUTOR_SHUTDOWN` | patch 1, 1 | `runtime._executor`, `runtime._executor_shutdown` |
| `_get_chacha_executor` | patch 7 | `runtime.executor`, patched on the instance |
| `_CHACHA_INIT_TIMEOUT_SECS`, `_CHACHA_WATCHDOG_SECS` | 1, patch 2 | `runtime.init_timeout_secs`, `runtime.watchdog_secs` (instance attributes, read on every `get_db`) |
| `_ChaChaInitializationAborted` | 1 | `ChaChaInitAbortedError` |
| `_record_init`, `_maybe_dump_traceback`, `_track_default_character_future` | 3, 1, 1 | `runtime._record_init(duration_ms, success, error=None)`, `runtime._maybe_dump_traceback(reason)`, `runtime.track(future)` |
| `_drain_default_character_tasks`, `_drain_default_character_futures` | 1, 1 | `runtime._drain_tasks(timeout)`, `runtime._drain_futures(timeout)` |
| `_get_chacha_cached_instance_count` | 3 | `len(runtime.cached_instances())` |
| `_health_check_instance` | 9 | `runtime._probe(db)` |
| `probe_chacha_connection`, `faulthandler`, `asyncio` | attr 2 + patch 2, 1, 1 | the same objects reached through `chacha.runtime` (R4) |
| `logger` | patch 11 (one file) | R2 |
| **G2: provider** (stay in deps during G1) | | |
| `_create_and_prepare_db` | 16 | `chacha_db_provider.create_and_prepare_chacha_db` |
| `_ensure_default_character`, `_ensure_default_character_async`, `_ensure_default_character_owned` | 9, 3, 2 | `chacha_db_provider.ensure_default_character`, `._ensure_default_character_async`, `._ensure_default_character_owned` |
| `_get_chacha_db_path_for_user`, `_verify_existing_chacha_db_integrity`, `_apply_sqlite_tuning` | patch 5, 1, 3 | the same names in `chacha_db_provider` |
| `_tenant_client_id` | import 1 | `chacha_db_provider.tenant_client_id` |
| `CharactersRAGDB`, `Path` | patch 4, 2 | patched on `chacha_db_provider` (R1, R4) |

Names no test uses move with their code and need no test edit:
- to `chacha.runtime` in G1: `_set_chacha_shutting_down` and `_is_chacha_shutting_down` (folded into `runtime._shutting_down`); `_record_default_character` (becomes `runtime.count`); `_get_or_init_db_instance_from_runtime` (folded into the deps wrapper); `_get_chacha_init_timeout_secs`, `_CHACHA_EXECUTOR_MAX_WORKERS` and `MAX_CACHED_CHACHA_DB_INSTANCES` (read by `get_chacha_runtime()`); the locks `_CHACHA_EXECUTOR_LOCK`, `_CHACHA_SHUTDOWN_LOCK` and `_chacha_default_char_futures_lock`; `_is_instance_healthy`, `_on_closed_loop`, the health helpers (`_is_sqlite_corruption_error`, `_safe_chacha_health_error`, `_sanitize_chacha_db_identifier`, `_build_chacha_failure_details`, `_copy_chacha_failure_details`), `_CHACHA_RECOVERY_*` and `_SQLITE_CORRUPTION_SIGNATURES`;
- to `chacha_db_provider` in G2: `SERVER_CLIENT_ID` and `resolve_chacha_user_base_dir`, both still re-exported by deps.

**Retarget rules.**
- **R1.** A patch follows the code it intercepts. It goes on the module or instance where that code lives after the slice: the `get_chacha_runtime()` instance for runtime state, `chacha.runtime` for module-level names the runtime uses, and `chacha_db_provider` for Task 10's code.
- **R2.** The 11 `logger` patches are all in `tests/Chat/test_chacha_db_deps_error_mapping.py`, and each follows its function under test:
  - to the `chacha.runtime` logger in G1: `:219` (health probe), `:236` (close all), `:256` (executor shutdown, which also patches `_CHACHA_EXECUTOR` and `_CHACHA_EXECUTOR_SHUTDOWN`) and `:297` (traceback dump);
  - unchanged in G1, and to the `chacha_db_provider` logger in G2: `:199` (tuning), `:277` (warm-up), `:316` (create and prepare), `:362` (default character, async) and `:385`, `:419`, `:437` (default character).

  The review's count of 26 also counted the Kanban, Prompts, prompt-studio and media deps tests, which this task does not touch: 34 `"logger"` patches in 6 deps test files, 11 of them on the ChaCha deps module.
- **R3.** Each `deps = importlib.reload(deps)` in `tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py` (:46, :74, :95, :144, :171, :213) reloaded the module to get fresh module state. That state now lives in the slot, so each test drops the reload and requests `fresh_services`.
- **R4.** A patch reached through the deps module on a shared object (`deps.DatabasePaths.x`, `deps.Path.mkdir`, `deps.faulthandler.dump_traceback`, `deps.asyncio.x`) patches the object, not a binding. It keeps working while the name stays reachable. `DatabasePaths` stays imported by deps. The others are rewritten to reach the same object through the module the code under test moved to.
- **R5.** The public-name patch sites are listed by `"$PY" "$TMPDIR/t1318_seams.py" tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps --sites get_chacha_db_for_user_id,get_chacha_db_for_owner,get_chacha_health_snapshot,shutdown_chacha_resources,warm_chacha_db_for_user,reset_chacha_shutdown_state,close_all_chacha_db_instances,shutdown_chacha_executor,_get_or_init_db_instance,CharactersRAGDB,probe_chacha_connection` (22 `patch` lines on b9eed09f91; filter with `grep " patch "`). G1 changes only `probe_chacha_connection` (`DB_Management/test_chacha_postgres_http_operation_lifecycle.py:349,587`). Task 10 handles the rest.
- **R6.** `close_all_chacha_db_instances` and `shutdown_chacha_executor` stay importable from deps through G2. `tests/AuthNZ/conftest.py:490` imports the first inside `reset_singletons`, where one missing name breaks every AuthNZ test. Other importers: `Admin/test_admin_smoke.py`, `Chatbooks/test_chatbooks_api_*`, `performance/test_authnz_multiuser_sqlite_load.py` and the Chat tests.
- **R7.** `runtime.executor()` recreates the executor after `shutdown_executor()`, as `_get_chacha_executor` does today for repeated lifespans in one process. No code holds a runtime past one call: always call `get_chacha_runtime()`.

**Interfaces** (`core/DB_Management/chacha/runtime.py`):
- Errors:
  - `class ChaChaRuntimeUnavailableError(RuntimeError)` is unchanged;
  - `class ChaChaInitAbortedError(ChaChaRuntimeUnavailableError)` is new;
  - `class ChaChaInitTimeoutError(RuntimeError)`, with message `"ChaChaNotes initialization timed out"`, is new;
  - `class ChaChaInitFailedError(RuntimeError)` is new. Its `__cause__` is the factory's exception, or None for "unknown error".
- The deps mapping (`_to_http(exc) -> HTTPException`) checks the errors in this order, and no detail ever includes `str(exc)`:

| Error | Status | Detail |
|---|---|---|
| `ChaChaInitAbortedError`, then `ChaChaRuntimeUnavailableError` | 503 | `_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL` |
| `ChaChaInitTimeoutError` | 503 | `"ChaChaNotes initialization timed out"` |
| `ChaChaInitFailedError` whose cause is a corruption error, `CharactersRAGDBError` or `InputError` | from `_map_chacha_init_db_error(cause)` | from `_map_chacha_init_db_error(cause)` |
| `ChaChaInitFailedError` with no cause | 500 | `"Could not initialize character & notes database for user: unknown error"` |
| any other `ChaChaInitFailedError` | 500 | `"Could not initialize character & notes database for user"` |

- `class ChaChaRuntimeManager`:
  - `__init__(self, *, max_cached: int = 20, max_workers: int = 4, init_timeout_secs: float = 30.0, watchdog_secs: float = 5.0) -> None`. The effective init timeout is `max(init_timeout_secs, watchdog_secs * 3, 5)`, computed on each `get_db`, as `_get_chacha_init_timeout_secs` does today.
  - `async get_or_create(self, factory, *args, **kwargs)` and `snapshot(self) -> dict[str, Any]` are unchanged.
  - `async get_db(self, key: str, factory: Callable[[], CharactersRAGDB], *, user_id: int) -> CharactersRAGDB`. This is today's `_get_or_init_db_instance` body:
    - raises `ChaChaRuntimeUnavailableError` once shut down;
    - raises `ChaChaInitTimeoutError` on a wait or init timeout;
    - raises `ChaChaInitAbortedError` from the stored sentinel;
    - raises `ChaChaInitFailedError` from the factory's error;
    - records health exactly where today's code calls `_record_init`.
  - `executor(self) -> ThreadPoolExecutor` (R7) and `shutdown_executor(self, wait: bool = False) -> None`.
  - `cached_instances(self) -> tuple[CharactersRAGDB, ...]` and `close_all(self) -> None`.
  - `track(self, pending: asyncio.Task[Any] | asyncio.Future[Any]) -> None`.
  - `count(self, counter: Literal["warm_startups", "default_char_ensures", "default_char_failures"]) -> None`.
  - `health_snapshot(self) -> dict[str, Any]` returns today's `get_chacha_health_snapshot()` payload, the same keys.
  - `shutdown(self) -> None` keeps its current meaning: stop accepting new work.
  - `async aclose(self, wait_timeout: float = 5.0) -> None` is today's `shutdown_chacha_resources`.
  - `resume(self) -> None` replaces `reset_for_tests` and the flag half of `reset_chacha_shutdown_state`.
  - `shutting_down` is a property returning `bool`.
- `get_chacha_runtime() -> ChaChaRuntimeManager`:
  - returns the slot, created on first use under a module `threading.Lock` (double-checked);
  - reads `CHACHA_EXECUTOR_MAX_WORKERS` (default 4, minimum 1), `CHACHA_INIT_WATCHDOG_SECS` (5), `CHACHA_INIT_TIMEOUT_SECS` (30) and `settings.get("MAX_CACHED_CHACHA_DB_INSTANCES", "20")` at creation.

- [ ] **Step 0: Measure the seams.**
  Run `"$PY" "$TMPDIR/t1318_seams.py" tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps > "$TMPDIR/t1318_deps_seams.txt"` on the PR base.
  - Expected on b9eed09f91: 53 lines of the form `<name>: attr N, import N, patch N; F files` (kinds with a zero count are omitted).
  - Pass: every printed name has a row in the name table: `cut -d: -f1 "$TMPDIR/t1318_deps_seams.txt"` lists no name the table lacks.
  - A name added since the base gets a row, by R1-R7, before Step 1.

- [ ] **Step 1: Write the failing tests** in a new file, `tests/unit/test_chacha_runtime_container.py` (`pytestmark = pytest.mark.unit`). A separate file keeps an import error here from erroring the earlier tasks' tests.
  - `test_get_chacha_runtime_is_the_container_slot(fresh_services)`: `get_chacha_runtime() is fresh_services.chacha_runtime is get_chacha_runtime()`.
  - `test_runtime_errors_map_to_the_same_http_responses(fresh_services, monkeypatch, tmp_path)`:
    - `monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))`, because the wrapper resolves the user directory before `get_db` runs;
    - parametrize over `(raised, status, detail)`;
    - set the slot (`monkeypatch.setattr`) to a runtime whose `get_db` raises `raised`;
    - call `deps._get_or_init_db_instance(1, "1")`;
    - assert the HTTPException's status and its exact detail:

| `raised` | status | detail (exact) |
|---|---|---|
| `ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")` | 503 | `"ChaChaNotes shutdown in progress"` |
| `ChaChaInitAbortedError("shutdown sentinel leaked /private/db/path")` | 503 | `"ChaChaNotes shutdown in progress"` |
| `ChaChaInitTimeoutError()` | 503 | `"ChaChaNotes initialization timed out"` |
| `ChaChaInitFailedError()` from `ChaChaDatabaseCorruptionError()` | 503 | `"ChaChaNotes DB corruption detected; repair or restore required"` |
| `ChaChaInitFailedError()` from `OSError("/private/db/path")` | 500 | `"Could not initialize character & notes database for user"` |
| `ChaChaInitFailedError()` with no cause | 500 | `"Could not initialize character & notes database for user: unknown error"` |

  - `test_runtime_owns_the_executor_and_cache(fresh_services)`:
    - `runtime = get_chacha_runtime()`; `db = await runtime.get_db("k", lambda: fake_db, user_id=1)`;
    - assert `db is fake_db` and `runtime.cached_instances() == (fake_db,)`;
    - after `await runtime.aclose()`, assert `runtime.cached_instances() == ()` and `fake_db.closed`;
    - then `runtime.resume()` and assert `runtime.executor()` is a live executor (R7).

    `fake_db` is an instance of a module-level `_FakeDB` class whose `close_all_connections()` sets `closed`. The first `get_db` runs the factory without a health probe.

- [ ] **Step 2: Run them.** Expected: collection fails with an `ImportError` for a new runtime name (the first new name the file imports).

- [ ] **Step 3: Move the code** per the table, the rules and Interfaces. Review it with `git diff -M --color-moved=dimmed-zebra origin/dev -- tldw_Server_API/app`: only state access, the raises and the deps wrappers should show as new.

- [ ] **Step 4: Run the ChaCha tests.**
  - Run: `"$PY" -m pytest ... tldw_Server_API/tests/unit tldw_Server_API/tests/API_Deps tldw_Server_API/tests/Chat/test_chacha_* tldw_Server_API/tests/DB_Management/test_chacha_* tldw_Server_API/tests/DB_Management/test_character_shared_owner_contract.py tldw_Server_API/tests/DB_Management/test_note_shared_owner_contract.py tldw_Server_API/tests/DB_Management/test_study_* tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py tldw_Server_API/tests/Services/test_startup_chacha_warmup.py tldw_Server_API/tests/Visual_Identities/test_builtin_pixel_migu.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py`.
  - Then run every file from `git grep -l ChaCha_Notes_DB_Deps -- tldw_Server_API/tests` (198 files) with Task 7 Step 3's comparison loop: every line `SAME`.
  - Then run `TLDW_SINGLETON_GUARD=error` on `tests/Chat` and `tests/Notes_NEW`: no leak reported.
  - Re-run Step 0's scan. Pass: it prints no name from the "G1: runtime state" rows except `logger`, whose 7 G2 patches stay on deps until Task 10 (R2). `probe_chacha_connection`, `faulthandler` and `asyncio` are gone, because R4 and R5 retarget them in this slice.
  - `git grep -nwE '_chacha_db_instances|_CHACHA_RUNTIME|_CHACHA_HEALTH|_CHACHA_SHUTTING_DOWN|_get_chacha_executor' -- tldw_Server_API` prints nothing.
  - `git grep -nw reset_for_tests -- tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api tldw_Server_API/tests/Chat tldw_Server_API/tests/unit` prints nothing. The grep is path-scoped because `ExtractionExecutorManager.reset_for_tests` (`core/Web_Scraping/orchestration/executor.py:383,678`, used by `tests/Web_Scraping/test_phase4_extraction_executor.py`) belongs to a service this task does not migrate.

- [ ] **Step 5: Tighten; run C1-C4, C6** (`tests/Chat/test_chacha_notes_db_deps_postgres_health.py tests/DB_Management/test_chacha_postgres_*`) **and C7.** Commit:

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/runtime.py tldw_Server_API/app/core/Utils/app_services.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/tests/lint/global_statements_baseline.json tldw_Server_API/tests/lint/api_deps_module_state_baseline.json
git add -A tldw_Server_API/tests
git commit -m "refactor(chacha): the runtime owns the executor, cache and init state; ChaCha deps hold none (TASK-13950.18)"
```

### Task 10: The ChaCha provider moves to core; services and core stop importing the deps module (AC3, AC5) (PR G2)

**Files:**
- Create: `tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py`. Move these from the deps module:
  - `_get_chacha_db_path_for_user`, `_verify_existing_chacha_db_integrity`, `_apply_sqlite_tuning`;
  - `_create_and_prepare_db`, renamed `create_and_prepare_chacha_db`;
  - `_ensure_default_character_async`, `_ensure_default_character_owned`, and `_ensure_default_character`, renamed `ensure_default_character`;
  - `_tenant_client_id`, renamed `tenant_client_id`;
  - `SERVER_CLIENT_ID`, `resolve_chacha_user_base_dir`, and the `Character_Chat.constants` re-export.
- Add the provider's public accessors (Interfaces).
- Modify: `ChaCha_Notes_DB_Deps.py`, a thin adapter (Interfaces). It keeps no private provider names.
- Modify the 18 importers to import the same names from `tldw_Server_API.app.core.Character_Chat.chacha_db_provider`:
  - `core/Chat/command_router.py:660`, `core/Chat_Macros/jobs.py:12`, `core/Research_Workspace/capabilities.py:451`, `core/Sharing/shared_workspace_clone_jobs_worker.py:1422`;
  - `services/notes_graph_projection_worker.py:34`, `notes_graph_suggestions_maintenance.py:11`, `notes_graph_suggestions_worker.py:12`, `notes_semantic_index_worker.py:17`, `notes_semantic_maintenance.py:14`, `research_workspace_output_jobs_worker.py:13`, `shutdown_resource_cleanup.py:171`, `startup_chacha_warmup.py:35,65`, `study_pack_jobs_worker.py:11`, `study_suggestions_jobs_worker.py:11`, `visual_identity_jobs_worker.py:13`, `vn_asset_jobs_worker.py:13`, `workspace_file_inventory_jobs_worker.py:12`, `writing_annotation_review_jobs_worker.py:13`.

  Also edit the `core/Notes/README.md:81` line, which names the deps module.
- Modify tests (about 60 uses):
  - every "G2: provider" row of Task 9's name table, by R1 and R4. The 7 G2 `logger` patches move to the `chacha_db_provider` logger (R2);
  - the R5 sites, listed by Task 9's `--sites` command (22 `patch` lines on b9eed09f91; filter with `grep " patch "`):
    - `get_chacha_health_snapshot`: `Health/test_readiness_health_sanitizers.py:57,85,113,216` and `Admin/test_admin_ops_dependencies.py:153` keep their deps target, because their callers (`endpoints/health.py:140`, `endpoints/admin/admin_ops.py:1409`) import from deps at call time;
    - `Sharing/test_sharing_endpoints.py:110,133,165,211,262,326` keep their deps target, because `endpoints/sharing.py:482,542,720,1787` imports from deps at call time;
    - `Sharing/test_shared_workspace_clone_jobs_worker.py:84` moves to `chacha_db_provider`, because its caller, `core/Sharing/shared_workspace_clone_jobs_worker.py:1422`, imports the provider after this slice;
    - `Sharing/test_shared_workspace_recipient_security_matrix.py:558-562` keeps both deps patches and adds the same two on `chacha_db_provider`, so the matrix still closes every route to a DB;
    - `Chat/test_chacha_db_deps_error_mapping.py:284` (warm-up log test): `warm_chacha_db_for_user` now lives in the provider and calls the runtime directly. So the test sets the slot to a runtime whose `get_db` raises `RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")`, patches the provider's `logger`, and keeps its assertions;
    - `Chat/test_chacha_notes_db_deps_sqlite_policy.py:130`: the provider validates ids before it touches the runtime. So the test sets the slot to a runtime whose `get_db` records calls, asserts it was never called, and still expects 400 `"Invalid owner_user_id."` from the wrapper;
    - `CharactersRAGDB` (`API_Deps/test_chacha_notes_db_deps_error_mapping.py:241`, `Chat/test_chacha_db_deps_error_mapping.py:345`, `DB_Management/test_chacha_postgres_http_operation_lifecycle.py:883`, `Visual_Identities/test_builtin_pixel_migu.py:270`) moves to `chacha_db_provider`.
  - `tests/unit/test_app_services_shutdown_order.py`: only the ChaCha seam changes, to `monkeypatch.setattr(app_services(), "chacha_runtime", _Service("chacha", events))`. `ORDER` is unchanged.
- Modify: `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py`. Add the AC3 pin:
  - `chacha_adapter_importers(root: Path = APP_ROOT) -> list[str]` lists modules under `core/` and `services/` (in-app `tests` packages skipped) with any import, at any level, of `tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps` (or `from ...API_Deps import ChaCha_Notes_DB_Deps`);
  - `test_services_and_core_do_not_import_the_chacha_deps_adapter` asserts `chacha_adapter_importers() == []`;
  - its seeded self-test plants one absolute, one from-package and one function-level import, and asserts all three are listed.
- Baseline: `layering_baseline.txt` via `"$PY" -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. Expected on b9eed09f91:
  - `services -> api.v1.API_Deps: 33` → `18`;
  - delete `core.Chat -> api.v1.API_Deps: 1` and `core.Chat_Macros -> api.v1.API_Deps: 1`;
  - `core.Research_Workspace -> api.v1.API_Deps: 2` → `1` and `core.Sharing -> api.v1.API_Deps: 2` → `1`.

  No new pair: `core.Character_Chat -> core.Visual_Identities` and `-> core.DB_Management` are not ratcheted.

**Interfaces:**
- `core/Character_Chat/chacha_db_provider.py`. Every function raises core errors, never `HTTPException`. The owner-id check lives only here.
  - `async get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB`. It raises `ValueError("Invalid owner_user_id.")` for a bool, non-int or non-positive id, before it touches the runtime, plus the runtime errors. It schedules the default-character task through `runtime.track`, as today.
  - `async get_chacha_db_for_owner(owner_user_id: int) -> CharactersRAGDB`. It raises `ValueError("Invalid owner_user_id.")` for a non-int id, today's check.
  - `async warm_chacha_db_for_user(user_id: int, client_id: str | None = None) -> None`.
  - `create_and_prepare_chacha_db(user_id: int, client_id: str) -> CharactersRAGDB`, `ensure_default_character(db: CharactersRAGDB) -> int | None` and `tenant_client_id(user_id: int, client_id: str | None) -> str`.
  - `get_chacha_health_snapshot() -> dict[str, Any]`, `snapshot_cached_chacha_db_instances() -> tuple[CharactersRAGDB, ...]`, `reset_chacha_shutdown_state() -> None`, `close_all_chacha_db_instances() -> None` and `shutdown_chacha_executor(wait: bool = False) -> None` delegate to `get_chacha_runtime()`.
  - `async shutdown_chacha_resources(wait_timeout: float = 5.0) -> None` reads `app_services().chacha_runtime` and does nothing when it is None, rather than build a runtime only to close it.
- `ChaCha_Notes_DB_Deps.py`:
  - `get_chacha_db_for_user` (Depends) and `get_chacha_db_for_user_id(user_id, client_id=None)` and `get_chacha_db_for_owner(owner_user_id)` call the provider's accessor of the same name and map errors with `_to_http`. A `ValueError` from the provider becomes 400 `"Invalid owner_user_id."`; the rest follow Task 9's Interfaces order.
    - There is no second validation in the wrapper.
    - Delegating keeps one tenant-binding implementation, the provider's `tenant_client_id`.
  - `_get_or_init_db_instance(user_id, client_id)` stays as the mapping wrapper that tests call. It now passes `partial(chacha_db_provider.create_and_prepare_chacha_db, user_id, client_id)`.
  - `_map_chacha_init_db_error` and `_to_http`.
  - Re-exports of the provider's public names, which keep R6's `close_all_chacha_db_instances` and `shutdown_chacha_executor` importable, plus `SERVER_CLIENT_ID`, `resolve_chacha_user_base_dir`, `DEFAULT_CHARACTER_NAME` and the runtime's error constants.

- [ ] **Step 0: Re-measure the seams.** Run Task 9's Step 0 scan and its R5 `--sites` command on the PR base. Every name and site must have a disposition here or in Task 9's table; add any newcomer before Step 1.

- [ ] **Step 1: Write the failing tests.**
  - The AC3 pin and its self-test (above).
  - `test_core_accessors_bind_the_tenant_client_id(fresh_services, monkeypatch)`, in `tests/unit/test_chacha_runtime_container.py`, importing `chacha_db_provider` inside the test. It pins the PostgreSQL tenant key that `test_chacha_client_id_is_the_tenant.py` documents: `"voice_assistant"` once pooled every user into one tenant.
    - Set the slot to a runtime whose `get_db(key, factory, *, user_id)` returns `factory()`.
    - Patch `chacha_db_provider.create_and_prepare_chacha_db` to record `(user_id, client_id)` and return a `_FakeDB`.
    - Patch `chacha_db_provider._ensure_default_character_async` with an async no-op.
    - Assert that each of these records exactly `(7, "7")`: `get_chacha_db_for_user_id(7, client_id="voice_assistant")`, `get_chacha_db_for_owner(7)` and `warm_chacha_db_for_user(7, "voice_assistant")`.
  - `test_core_accessor_raises_core_errors_not_http(fresh_services, monkeypatch)`, in `tests/unit/test_chacha_runtime_container.py`, also importing `chacha_db_provider` inside the test:
    - `await chacha_db_provider.get_chacha_db_for_user_id(0)` raises `ValueError`;
    - with the runtime slot patched so that `get_db` raises `ChaChaInitTimeoutError()`, `await chacha_db_provider.get_chacha_db_for_user_id(1)` raises `ChaChaInitTimeoutError`, not `HTTPException`.

- [ ] **Step 2: Run them.** Expected:
  - the pin fails, listing the 18 modules;
  - both provider tests fail with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Character_Chat.chacha_db_provider'`;
  - the file's Task 9 tests still pass, because the import is inside each new test.

- [ ] **Step 3: Move, rewrite the 18 imports, retarget the tests, `--tighten`.**

- [ ] **Step 4: Run.**
  - Task 9 Step 4's commands, plus every test file of the 18 importers' modules: `git grep -lE 'notes_graph|notes_semantic|study_pack|study_suggestions|visual_identity_jobs|vn_asset_jobs|workspace_file_inventory|writing_annotation_review|research_workspace_output|chat_macro|command_router|shared_workspace_clone|startup_chacha|shutdown_resource_cleanup|Research_Workspace/capabilities' -- tldw_Server_API/tests`. Expected: the same pass/fail set as dev.
  - `tests/unit/test_app_services_shutdown_order.py`: PASS with `ORDER` unchanged.
  - `git grep -n "ChaCha_Notes_DB_Deps" -- tldw_Server_API/app/core tldw_Server_API/app/services` prints nothing.
  - Re-run Task 9's Step 0 scan. Pass: it prints no "G2: provider" name. `close_all_chacha_db_instances` and `shutdown_chacha_executor` are still listed (R6).
  - `wc -l tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` is reported in the PR, with no target. The slices move about 650 of its 1,030 lines, so about 380 remain before the wrappers are thinned.

- [ ] **Step 5: Run C1-C4, C6 (as Task 9) and C7.** Commit:

```bash
git add tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/core/Notes/README.md tldw_Server_API/tests/lint/layering_baseline.txt tldw_Server_API/tests/lint/test_service_singleton_ratchet.py
git add -A tldw_Server_API/app/core tldw_Server_API/app/services tldw_Server_API/tests
git commit -m "refactor(chacha): workers open per-user ChaCha DBs through core; the deps module is a thin adapter (TASK-13950.18)"
```

### Task 11: Close-out (ADR-021 amendment, AC checks, DoD) (PR H)

**Files:**
- Modify: `Docs/ADR/021-services-lifecycle-startup-and-shutdown.md`. Append `## Amendment (TASK-13950.18)`:
  - the five services live in `app.state.services` (`core/Utils/app_services.py`);
  - shutdown closes the container's objects in the final cleanup tail's existing order, pinned by `tests/unit/test_app_services_shutdown_order.py`;
  - the getters are the fallback for code with no request or constructor argument;
  - tests swap services with `fresh_services`.
- Run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published/ADR/021-...` diff.
- Backlog: `backlog/tasks/task-13950.18 - ...md`, through `PYTHONPATH=tools/backlog-py/src "$PY" -m backlog_py --cwd . task edit TASK-13950.18` only.

- [ ] **Step 1: Verify each AC on dev.**

The commands are written outside a table so that they paste into zsh or bash unchanged.
- **AC1.** The three baselines exist and `test_singleton_counts_do_not_grow[*]` passes. Record the three totals (seed 456/551/35, now lower).
- **AC2.** `test_lifespan_publishes_the_container_and_shuts_down_what_it_holds`, the five slot tests and `test_mcp_routes_take_the_server_from_the_container` pass. The wording follows the controller's AC2 ruling (Global Constraints).
- **AC3.** `test_services_and_core_do_not_import_the_chacha_deps_adapter` passes, and the state baseline has no ChaCha entry.
- **AC4.** Both of these print nothing:
  - `git grep -nwE 'reset_session_manager|reset_jwt_service' -- tldw_Server_API`;
  - `git grep -nw reset_for_tests -- tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api tldw_Server_API/tests/Chat tldw_Server_API/tests/unit`. It is path-scoped because Web_Scraping's `ExtractionExecutorManager.reset_for_tests` belongs to a service this task does not migrate.
- **AC5.** `test_final_cleanup_closes_the_container_services_in_adr_021_order` passes, and Task 2 Step 3's marker check shows that C3 selects it. `ORDER` is as committed in PR A: `git diff <PR A merge> HEAD -- tldw_Server_API/tests/unit/test_app_services_shutdown_order.py` changes only the ChaCha seam.
- **AC6.** `git grep -l -w reset_db_pool -- tldw_Server_API/tests | wc -l` prints at most 49.
- **AC7.** `test_backend_required_enforces_size_ratchets[tldw_Server_API/tests/lint/test_service_singleton_ratchet.py]` and `test_backend_required_unit_smoke_runs_the_unit_tests` pass.

- [ ] **Step 2: Run Bandit on the touched app code.** Bandit is not in the venv, so use `uvx bandit -q -ll -r tldw_Server_API/app/core/Utils/app_services.py tldw_Server_API/app/core/DB_Management/chacha/runtime.py tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/api/v1/API_Deps/mcp_deps.py`. Expected: no new medium or high finding against dev.
- [ ] **Step 3: Record the results.**
  - Pass `--check-ac` for each AC proven in Step 1. First rewrite AC2's text to Ruling 1's wording with backlog-py's AC edit, citing the controller ruling of 2026-10-30.
  - Pass `--append-notes` with:
    - the before and after numbers (globals, getter calls, API_Deps state, `reset_db_pool` test files 168 on b9eed09f91 (169 after #3371) → N, services/core ChaCha deps importers 18 → 0, `ChaCha_Notes_DB_Deps.py` 1,030 → N lines, the `services -> api.v1.API_Deps` layering line 33 → 18);
    - the PR numbers;
    - the drift in the backlog's figures (574 → 551; 155 → 168);
    - Rulings 1 and 2 (controller rulings, 2026-10-30).
  - Pass `--check-dod` for each item met, and `--final-summary`.
  - Stage with `git add -A backlog/tasks`; `tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py` must pass.
- [ ] **Step 4: Commit.**

```bash
git add Docs/ADR/021-services-lifecycle-startup-and-shutdown.md Docs/Published/ADR/021-services-lifecycle-startup-and-shutdown.md
git add -A backlog/tasks
git commit -m "docs(adr): ADR-021 names app.state.services; close out TASK-13950.18"
```

## Self-review (2026-10-30)

- **Spec coverage.**
  - AC1: Task 1.
  - AC2: Tasks 3-6 and 9, scoped by Ruling 1.
  - AC3: Tasks 9-10.
  - AC4: Tasks 5, 8b and 9; Ruling 2 names the three deleted helpers.
  - AC5: Task 2, with each slice re-running it, and the lifespan tests in Tasks 3-4.
  - AC6: Tasks 7-8a.
  - AC7: Task 1 wiring. The AC3 pin lives in the same wired module. The AC5 and fixture tests sit in `tests/unit` with `pytestmark = pytest.mark.unit`, which the required unit-smoke step selects; Task 2 Step 3 proves the selection.
  - DoD: Task 11.
  - The roadmap's notes (deprecated fallback reads the container; shutdown order checked every slice; `reset_db_pool` callers under 50) are Global Constraints and Task 7-8 checks.
- **Names checked against each other.**
  - `fresh_services` and `fresh_app_services` (Task 3), `app_services()` and `AppServices`, `get_mcp_server_dep` (Task 6) and `get_chacha_runtime` (Task 9) are used under the same names in later tasks.
  - `ORDER` is defined once (Task 2).
  - Task 9 keeps the existing `ChaChaRuntimeManager.shutdown()` meaning and adds `aclose()`, so no caller of the old name changes behaviour.
- **Proportion.** Code blocks are limited to the ratchet's seeded self-test, the shutdown pin and the codemod, which the signatures do not determine. Task 9's move is given as a name table plus interfaces, not a transcript.
