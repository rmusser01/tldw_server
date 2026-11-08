# Move Shared Code Out of Endpoint Modules and Drain Endpoint Monkeypatches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze test patches of endpoint modules with a shrink-only counter, move the route-less endpoint modules and the core reach-ins out of `api/v1/endpoints`, and drain the endpoint-module monkeypatch count to at most 55% of its seed, so that `.26` and `.28` can move handler code without rewriting thousands of patches.

**Architecture:**
- A per-endpoint-module ceiling ratchet (`tests/lint/test_endpoint_monkeypatch_ratchet.py`, on `_ceilings`) counts `monkeypatch.setattr`/`delattr` calls that target an endpoint module, plus a second frozen field for mock patches and direct assignments. It resolves names the way Python scopes them, and it exports the matcher (`patch_sites`, `ScopeResolver`) that the codemod reuses, so the two agree by construction.
- Moves come next. `_pagination_utils` merges into `api/v1/utils/pagination.py`. Twenty route-less modules, plus the `_chatops` package `__init__`, move to `api/v1/utils/`, and `sandbox_service.py` moves to `core/Sandbox/default_service.py`. Every guard scoped to `api/v1/endpoints` widens to the moved files in the same PR. The four live reach-ins are cut.
- Then a codemod (Appendix A) works through one endpoint module at a time. Each collaborator that a test patches changes from `from M import f` + `f(...)` to `_m.f(...)`, and the test patch becomes `monkeypatch.setattr("M.f", ...)`.
  - Production behaviour does not change: the same object is called.
  - The patch now follows the code when `.26` or `.28` moves a handler body into core.
- A pytest tripwire (Appendix C) proves each wave:
  - no patch the codemod could not see goes dead;
  - a patch that already intercepted nothing on dev is deleted, not retargeted;
  - no home-module patch reaches further into the egress client or AuthNZ than the endpoint patch did.
- `.18`'s getter counter learns to count `_m.get_x()`, so the rewrite cannot lower it.

**Tech Stack:** Python 3.12/3.13, FastAPI, pytest (pytest-asyncio auto mode, pytest-timeout, pytest-xdist), AST lint ratchets under `tldw_Server_API/tests/lint/` sharing `_ast_cache.py` and `_ceilings.py`, SQLite and PostgreSQL (`TLDW_TEST_POSTGRES_REQUIRED=1`).

**Spec:**
- `backlog/tasks/task-13950.20 - Move-shared-code-out-of-endpoint-modules-and-drain-endpoint-monkeypatches.md`, AC #1-#5 and the DoD.
- `.18`'s ruling 1 in `backlog/tasks/task-13950.18 - Introduce-an-app-scoped-service-container-and-retire-module-singletons.md`. It hands the endpoint-body getters to this task.
- The `.20` row of `## Phase 2` in `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`:
  - effort L;
  - "pagination, route-less modules, then each core reach-in";
  - "Commit the monkeypatch counter first; lower the baseline every PR".
- The Phase 2 exit check in the same file (phase table, row 2): "Endpoint-module monkeypatch baseline at or below 55% of its committed seed".
- Fix round 1 (2026-11-08): `plan-20-review.md` and `plan-20-dryrun.md` under `.superpowers/sdd/2026-10-25-architecture-remediation-roadmap/` in the `arch-review` worktree, and the controller's rulings on them (Rulings 7-12).

**Conventions:**
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Run every command from the repo root with `PYTHONDONTWRITEBYTECODE=1`.
- pytest runs as `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-<name>"`, except Appendix B (no `--basetemp`, its docstring says why). Pass `-p pytest_asyncio.plugin` only together with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- File:line references are on `origin/dev` 7a020da14a. Re-locate them by the quoted text.
- Every changed file ends with exactly one newline and has no trailing whitespace: `git diff --check` prints nothing.
- `T1320_DIR=${T1320_DIR:-$HOME/.cache/t1320}`. `$TMPDIR` is purged, and the waves run for weeks.
  - Save Appendices A-E there verbatim as `t1320_seams.py`, `t1320_compare.py`, `plugin/t1320_tripwire.py`, `t1320_counts.py` and `t1320_totals.py`. The plugin sits in its own directory, because putting `$T1320_DIR` itself on `PYTHONPATH` could shadow an installed package.
  - Nothing there is committed. Each PR body pastes the scripts as run, plus its skip and inert lists, which makes the PR bodies the durable record.
- **Baseline reasons never contain `<` or `>`.** `_ceilings._UNFILLED` and the layering contract reject any `<...>` as an unfilled placeholder. Every reason in this plan is the literal string to commit.

## Measured on 2026-11-08 (origin/dev 7a020da14a)

| Figure | Backlog (dca11342de) | Measured now | Note |
|---|---|---|---|
| `monkeypatch.setattr`/`delattr` calls against endpoint modules | 5,266 (1,476 on private names) in 651 files | **5,729** (1,467 on private names) in 680 test files, against 200 endpoint modules. 168 are string targets and 5,561 are module-object targets | Task 1's matcher as now specified. Its first draft gave 5,816. The reviewers' readings gave 5,816 and 5,814, because whole-file name resolution attributed helper-returned and shadowed names. The lexical rule removes that ambiguity |
| Mock patches and direct assignments | not given | 937: 297 `patch("...")`, 593 `patch.object(...)`, 47 `<endpoint>.x = ...`/`del` | Task 1's `other` field. It is frozen, and it is not the exit metric |
| Phase 2 exit target | 55% of the seed | `monkeypatch` total ≤ **3,150** (⌊0.55 × 5,729⌋) | Re-measured at execution (Ruling 12) |
| Route-less endpoint modules | 24, 5,252 lines | 24, 5,258 lines, not counting package `__init__` files | 22 move (Ruling 5) |
| `_pagination_utils` importers | 62 | 62 app files with 64 statements, plus 1 test file | 61 top-level absolute statements, 2 function-level in `workflows.py:2086,2456`, 1 relative (`chatbooks.py:48`) |
| Cross-family endpoint→endpoint imports | 212 statements into 87 targets | 170 statements in 127 `layering_baseline.txt` lines, into 47 targets | Task 2 deletes 45 lines (64 statements); Task 3 deletes 38 (44) |
| Core reach-ins | sandbox `_service` :143,515,853; chat `API_KEYS` :2178,2259; `chat_service -> llm_providers`; `jobs_worker.py:42` (5 names, 1 private); `persistence.py:2592`; mcts `:57` | `sandbox_runner_client.py:18,143,515,853,858`; chat `API_KEYS` **gone** (7cffa1362f, `.13`); `chat_service.py:1522`; `jobs_worker.py:42` (5 names, **2** private); `persistence.py:2588`; `services/media_ingest_jobs_worker.py:406` (not in the backlog); `mcts_optimizer.py:55-61,247-256` | Ruling 6 |
| `TRUE_INVERSION_BASELINE` | named in AC3 | gone: `.9` replaced it with the pair-keyed baseline | Ruling 6 |
| Endpoint-body zero-arg getter calls (`.18`) | 551, then 540 | 549, with a baseline sum of 550 over 105 entries (`mcp_unified_endpoint.py` has 1 of slack). `get_db_pool`: 69 calls in 25 files. `get_jwt_service`: 2 (`admin_impersonation.py`, `evaluations_auth.py`) | With Task 1's matcher: 599 calls in 110 modules |
| `api_deps_module_state_baseline.json` | — | 29 in the baseline, 25 measured: `api/v1/API_Deps/auth_deps.py` is 6 in the baseline and 2 in code | Task 1's `--write-baseline` commits the lower ceiling (Task 1 Step 4) |
| Sandbox `_service` in tests | — | 35 patches in 7 files (30 resolvable, 5 through `src.sandbox_ep`) and 47 reads in 10 files; 16 files in all | Task 4 |
| Audio shim lookups | — | 48 `_audio_shim_attr(...)` calls, 4 definitions, 8 `getattr(<audio package>, ...)` sites, 8 function-level imports of the package or `audio.audio` | Task 8 |

Dry run of the design (scratch worktrees on 7a020da14a):
- **The full codemod run.** Appendix A, run over every endpoint module except `audio/`, changed 598 files (+5,249/−6,084): 141 endpoint modules and 457 test files.
  - It converted 655 pairs: 1,576 loads and 2,555 `monkeypatch` patches (2,446 object form, 109 string form), plus 550 mock patches. 63 converted pairs have a SENSITIVE home (Ruling 7).
  - It left the counter at 3,174.
  - `git diff --check` was clean, and all 280 endpoint modules imported.
  - Appendix E's totals were unchanged on the converted tree, the audio upper-bound run included: `.18`'s getters under Task 1's matcher (599), `.18`'s globals (445), `.19`'s test-mode lines and predicates (581 and 12), and `.13`'s adapter references (30). The reviewers' W1 run gave the same totals.
- **The tripwire (Appendix C)** reproduced the review's finding on `tests/Agent_Client_Protocol/test_async_api.py`.
  - Head (dead mode): 7 of 11 tests fail on `agent_client_protocol.get_acp_session_store`/`get_runner_client`, patched through a module returned by a helper.
  - Base (count mode): both of those sites count 0 calls, so they were inert on dev. That makes them deletions under Ruling 7, not skips.
- **Per-file outcome comparison.** The first draft's run covered 55 files and found 2 DIFFERS; the skip rules in Appendix A now cover both. The dry run's E1 sample (24 files, 591 items) matched node for node. Its 158 errors on both sides came from a `--basetemp` under the scratchpad, which Appendix B no longer passes.
- **Projection, from the dry run.** All figures are re-measured on each PR's base.
  - Starting from the 5,729 seed, the slices drain:
    - Task 3: −30;
    - Task 4: −30;
    - Task 5: −49;
    - W1: −482; W2: −544; W3: −359; W4: −565; W5: −594;
    - Task 8: about −196 (an upper bound: Appendix A with its by-string rule disabled on `audio/`).
  - Result: 2,880 (50.3%). Without Task 8 it is 3,076 (53.7%), against the 3,150 target.
  - Not yet counted:
    - patches Appendix C proves dead and live, whose pairs it skips;
    - patches it proves inert, which it deletes, and which also drain.
  - E1 re-projects from its real lists before E2 opens (Ruling 10).

## Rulings

Controller rulings (2026-11-08; binding):

1. **Phase exit.**
   - The seed is Task 1's committed count, measured on the dev tip at execution (Ruling 12). The task ends at or below ⌊0.55 × seed⌋.
   - E3 is the last drain PR, and Task 9 verifies the exit on dev after E3 merges.
   - If it falls short, Task 9 records the shortfall and files the rest against its owners: `.28` for `embeddings_v5_production_enhanced.py` (about 300 left after W4) and `.26` for the `persona.py` and `chat.py` own-name helpers.
   - Task 1 commits the counter with a seeded-violation self-test. Its required-gate assertion goes in `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`.
2. **Inherited getters (`.18` ruling 1): kept as getters.**
   - The endpoint-body `get_db_pool()` and `get_jwt_service()` calls stay lazy. `Depends` would resolve the pool before request validation (`tests/Ingestion_Sources/test_ingestion_sources_api.py:151-223` asserts invalid input never fetches it). It would also build the JWT service before `admin_impersonation.py`'s 403 (`:71` raises, `:100` builds).
   - Both getters already read the container (`app_services()`), so a test can swap the service with `monkeypatch.setattr(app_services(), "db_pool" | "jwt_service", fake)`, as `tests/AuthNZ/test_admin_impersonation.py:82` does.
   - Where tests patch the getter itself on an endpoint, the waves rewrite the call to `_database.get_db_pool()` (or `_jwt_service.get_jwt_service()`) and the patch to `"tldw_Server_API.app.core.AuthNZ.database.get_db_pool"`. The call stays lazy.
     - That is 11 pairs and 50 patches: 10 `get_db_pool` pairs (44 patches) and `evaluations_auth.py`'s `get_jwt_service` (6).
     - `telegram_support.py`'s pair (2) moves with Task 3.
     - These homes are SENSITIVE (Ruling 7), so each conversion needs Appendix C's no-broadening proof.
   - Task 1 extends `.18`'s counter to count module-attribute getter calls. It stays blind to a function-level `import ... as _db; _db.get_db_pool()`, a limit its docstring records.
   - Task 9 records in `.18`'s notes that the endpoint-body getters are re-handed. It files "Inject endpoint-body services through lazy request-scoped providers" as a dependency of `.26`, which rewrites those handler signatures anyway.
3. **`.21` coordination.** `.20` moves code with its config and env reads unchanged; `.21` retargets each read in its new home (Sibling coordination).

Scope rulings (this plan's, with reasons):

4. **Seam reading of AC4: accepted by the owner on 2026-11-08.**
   - The owner accepted this reading on 2026-11-08: migrated tests use overrides, container slots or a patch of the collaborator's home module, never an endpoint module.
   - Task 9 Step 2 rewords AC4's text to match.
   - **Proposed AC4 text, verbatim:** "Migrated tests patch no endpoint-module attribute: they use `app.dependency_overrides` for route dependencies, a container slot (`app_services()`) for container-held services, or a string patch of the collaborator's home module for names called in handler bodies."
   - **Why the third seam:**
     - Dependency overrides reach only `Depends` parameters. Of the waves' candidates, only the 17 whose names no function body loads are Depends-only, and those are already dead patches, because FastAPI captured the callable at decoration. The 2,555 that the waves migrate are on names called inside handler bodies.
     - Giving each of the 655 body collaborators a provider would change the signatures and resolution order of the handlers that call them (Ruling 2), and it would break the tests that call handlers directly.
     - The home-module patch is behaviour-neutral, and it survives `.26`'s and `.28`'s moves.
   - **The cost, stated for the owner:** about 29% of converted patches also reach their home module's own callers (Review Focus 1). Ruling 7's count check bounds that for the egress client, AuthNZ and the sandbox.
   - **Recommendation:** approve, and have new tests prefer a container slot wherever the service lives in `app_services()`.
   - **For the record, the literal reading** (overrides and container slots only) would drain at most 217 patches: 91 container-getter patches, 17 dead Depends-only patches and 109 move retargets. That leaves at least 5,512, 96.2% of the seed, so the 55% exit would not hold.
5. **AC2's set is 22 of the 24.**
   - `control_plane.py` holds route handlers that `main.py` registers with `add_api_route` (`main.py:1185,2866-2899`), so it is an endpoint module.
   - `evaluations/evaluations_auth.py` is TASK-13950.74's, which moves it onto the AuthNZ dependencies. It is pinned in the `.7` token-parsing ratchet citing that id, and in `.19`'s `SECURITY_FUNCTIONS`.
   - Task 3's route-less pin allows exactly these two, with those reasons.
   - No shim is kept, because every importer and patch target is retargeted in the moving PR. Task 9 records that as AC2's deviation from "with re-export shims while tests migrate".
6. **AC3 against current dev.**
   - The chat `API_KEYS` reach-in was removed by `.13` (7cffa1362f). Task 9 checks that box with that evidence.
   - `TRUE_INVERSION_BASELINE` no longer exists. "Drops their entries" means deleting these `layering_baseline.txt` lines:
     - `core.Agent_Client_Protocol -> api.v1.endpoints.sandbox`;
     - `core.Chat -> api.v1.endpoints.llm_providers`;
     - `core.Embeddings -> api.v1.endpoints.media_embeddings`;
     - `core.Ingestion_Media_Processing -> api.v1.endpoints.media_embeddings`;
     - `core.Prompt_Management -> api.v1.endpoints.prompt_studio`;
     - `services -> api.v1.endpoints.media_embeddings` (the same reach-in, from the media ingest worker).
   - Not in AC3, so not cut here:
     - `core.Embeddings`/`core.Evaluations -> api.v1.endpoints.embeddings_v5_production_enhanced` (`.28` AC2);
     - `core.Setup -> api.v1.endpoints.audio` (`install_manager.py:26`, new since the review; Task 9 files it).

Fix round 1 rulings (controller, 2026-11-08; binding):

7. **Dead, inert and broadened patches (review C1, I7).** Appendix C runs on every wave and on Tasks 4, 5 and 8.
   - **Dead and live.** A converted pair whose sites the tripwire reports dead on the head is not converted: it goes on the skip list. This covers a helper-returned module, a fixture parameter, a loop variable or a conditional expression, all invisible to the matcher. The pair is converted only if those sites are hand-retargeted; that is hand-written work, counted.
   - **Inert on dev (re-review N2).** A site is provably inert only if two things hold on the base:
     - every test that touched it passed;
     - the patched object recorded 0 calls, a Mock's attribute calls included.

     A site whose test skipped, errored, failed or xfailed on the base is unproven. One example is `tests/Audio/test_audio_transcriptions_adapter_path.py:106-110`, which patches five quota names and then skips at `:139` when the route is not mounted. An unproven site goes on the skip list and is never deleted. Appendix C records outcomes, and Appendix D's `inert` lists only proven sites.

     A provably inert site:
     - It is not retargeted.
     - Appendix A deletes it when it is a standalone `monkeypatch.setattr`/`delattr` statement. Otherwise it is left in place and allowlisted.
     - The PR body lists each deletion with Appendix D's `inert` line as the proof. The review's roughly 45 helper-returned candidate sites take this path when they are inert, and otherwise the skip path.
   - **Broadened.** For a SENSITIVE home (`core.http_client`, `core.Utils.egress`, `core.Security`, `core.AuthNZ`, `core.Sandbox`), a pair is converted only if no site file calls the home fake more on the head than the base called the endpoint fake. Appendix D's `broadened` output lists the pairs to skip.
     - Any wave that converts a SENSITIVE home runs C4 and C6, unconditionally.
8. **The Prompt Studio connection manager (dry run C1).**
   - `ConnectionManager.connect` needs the endpoint's `_accept_prompt_studio_websocket_if_needed` and `services.app_lifecycle.is_lifecycle_draining`. Neither may enter core.
   - So the class does not move. Core gains `core/Prompt_Management/prompt_studio/ws_connections.py`, a two-function slot: the endpoint publishes its manager, and core reads it at call time.
   - **Both core readers move to the slot in the same PR (re-review N1):**
     - `mcts_optimizer.py` itself;
     - `optimization_engine.py:1078`, which reads `mcts_optimizer_module.ws_connection_manager`.
   - The 7 test patches of that attribute become `monkeypatch.setattr(ws_connections, "current", lambda: <the old value>)`, which reaches both readers.
   - Outside the server process, `current()` returns None where the old code imported the endpoint on demand. That process has no WebSocket connections, so the old broadcast reached nobody either (review m8).
   - That adds no `core -> services` or `core -> api` edge, and Task 4 re-runs the layering contract.
9. **Moved security code (review I5).**
   - The PR that moves the code also widens each guard scoped to `api/v1/endpoints` to the moved files, so moved code is never unguarded:
     - `.7`'s bearer-parsing ratchet;
     - the auth-deps import boundary;
     - the two AuthNZ call-site guardrails;
     - the raw-SQL ratchet.
   - One shared list, `tests/lint/_moved_endpoint_code.py`, feeds all five. PR B1 creates it and widens the guards; each moving PR (B1, B2, C, D) appends its destinations.
   - Changed guards must run in a required gate (AC5), so PR B1 wires the four that were not in one.
   - Each guard's test is a mutation test: it plants a violation at a moved-destination path and points the guard's `moved_files` at it. It fails if the guard does not scan moved code (re-review N3).
10. **Full Suite gates (review I9).**
    - Three C7 gates: E1 (W1 alone), E2 (W2 + W4), then E3 (W3 + W5 + the audio shims).
    - Each PR stays at or under about 400 hand-written lines. Reviewers check the generated lines by re-running Appendix A, not by reading them.
    - E1 re-projects the exit from its real skip and inert lists before E2 opens.
11. **Dry-run Importants.**
    - Task 1 Step 4 states the state-baseline change.
    - The counter spec now pins resolution, arity and receivers, and the seed is re-measured under it.
    - `t1320_totals.py` is Appendix E.
    - The sandbox, shim and other counts are corrected above.
12. **Seeding.** Task 1 seeds on the dev tip at execution, after #3433 (`.19` PR 11) merges. A sibling PR that grows an entry afterwards takes a reason naming its PR, for example `"#3437 adds one endpoint patch after the TASK-13950.20 seed"`.

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

The four 2026-10-25 owner decisions (moderation, keywords, scheduler, admin) do not bind this task. The owner waived Change summaries for all TASK-13950 PRs and authorized merging them after green CI and a clean review. Record the summary section anyway.

Task-specific:

- **Lower the baseline in every drain PR.**
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet --tighten` and commit the diff. An entry whose module is fully drained or moved is stale and fails until `--tighten` deletes it.
  - Paste `"$PY" -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet --summary` into the PR body.
- **Shims: none needed** (Ruling 5).
  - Every move in Tasks 2-5 retargets every importer and every string patch target in the same PR. Each task's `git grep` proves it, so the old path is deleted, not aliased.
  - An open PR that still imports a deleted path fails on rebase with `ModuleNotFoundError` naming the module.
  - A path that cannot be fully retargeted in its PR keeps a `sys.modules` alias at the old path (the `core/Security/egress.py` pattern).
- **Deleted or moved symbols: grep and run (fix round 2, ruling 1).** Before it commits, a task that deletes, renames or moves a module-level symbol does two things:
  - It runs `git grep -nwE '<the symbols>'` over `tldw_Server_API`, `Helper_Scripts` and `.github` on the PR base. Every app hit is either changed by the task or proven unaffected.
  - Every test file the grep lists runs in the task's Appendix B comparison.

  Each task's step names its symbols. In the waves, Appendix A's referrer scan is the grep, and its `runset.txt` and `compare_files.txt` are the run.
- **Moved code keeps its config reads (Ruling 3)** and its function sizes. No moved function is in `function_size_baseline.json`. Codemod edits to frozen functions (`chat:create_chat_completion`, `persona:persona_stream`, `sandbox:start_run`, ...) rewrite names in place and add no lines.
- **Guards the slices can trip, and how each stays green.**
  - **Layering contract** (`layering_baseline.txt`).
    - Tasks 2-5 delete lines with `--tighten`, and Task 5 adds one `# bump:`.
    - The waves add only endpoint→core/`API_Deps`/`utils` imports, which are unruled. Appendix A skips sources inside `api/*/endpoints`, and it canonicalizes alias modules to their targets, so no alias path is imported.
  - **Size ratchets.** No moved function is frozen, and codemod edits are in place (above).
  - **HTTP usage guard.** `SCOPE_ALLOWLIST` lists `endpoints/llm_providers.py`, which stays put, and its HTTP code stays with it. No moved module uses raw httpx or requests.
  - **The five endpoint-scoped guards (Ruling 9).** Widened in PR B1 through `tests/lint/_moved_endpoint_code.py`, each with a mutation test:
    - `.7`'s `test_auth_token_parsing_ratchet.py` (bearer);
    - `test_endpoint_auth_deps_import_boundary.py`;
    - `tests/AuthNZ/unit/test_require_admin_callsite_guardrail.py`;
    - `tests/AuthNZ/unit/test_get_current_user_callsite_guardrail.py`;
    - `test_no_raw_sql_in_endpoints.py`.

    None flags a moved file on dev. The list holds exact files, not `api/v1/utils/` as a whole, because the import boundary would flag the pre-existing `utils/rag_cache.py:7` `User` import.
  - **`.6`'s seam and production-key ratchets.** Its `("api/v1/endpoints/sandbox.py", "core/config.py")` pair is unaffected: Task 4 changes only `_service` lookups.
  - **`.7`'s WebSocket ratchets.**
    - The bearer baseline's only entry, `evaluations_auth.py`, stays put.
    - Task 4 retargets one read in `test_websocket_route_auth_matrix.py`, and Task 8 changes the audio lookup but not the function. Both run C1's second command.
  - **`.5`, `.12`, `.16`.** No slice adds a module to `core/Utils`, `core/config*` or `core/DB_Management`. Task 4's `ws_connections.py` imports nothing.
  - **`.13` direct-adapter calls and the `LLM_Calls` import-isolation guard.** Task 4's `core/LLM_Calls/model_metadata.py` imports only `LLM_Calls` modules, and Task 4 Step 4 runs the guard. The waves keep the adapter-reference total equal.
  - **`.17`'s alias gate** (`test_authnz_alias_imports.py`). Appendix A imports canonical homes only, never an AuthNZ alias path.
  - **`.18`.**
    - Task 1 changes the getter matcher.
    - Task 3 re-keys five getter entries.
    - Task 4 drops `mcts_optimizer.py`'s `global`.
  - **`.19`.** Task 3 re-keys `telegram_support.py`'s `test_mode_branch_baseline.json` entry, and the `SECURITY_FUNCTIONS` pins stay where they are.
  - **Test-quality triage.** C2 and Task 6 Step 6 run it.
  - **`pyproject.toml` per-file ignores.** The `BLE001` keys at `:808` (`discord_support.py`), `:850` (`slack_support.py`) and `:855` (`telegram_support.py`) move with Task 3, and the key at `:833` (`media_embeddings.py`) gains the core path in Task 5.
- **Ratchet totals the codemod must not move.** Appendix E prints them, and each wave compares them before and after:
  - `.18` getters (Task 1's matcher counts `_m.get_x()`) and globals;
  - `.19` test-mode lines and predicates (its matcher counts attribute references);
  - `.13` adapter references;
  - `.21` env reads, once that ratchet lands (its matcher counts `<module alias>.<helper>(...)`).
- **No legacy baseline regeneration.**
  - No planned file is in `scope_predicate_baseline.txt`.
  - Check before each PR: `git diff --name-only origin/dev | sed 's#^#^#' | grep -f - Helper_Scripts/ci/scope_predicate_baseline.txt` prints nothing.
- **The codemod is generated output.**
  - Hand-written and generated lines are reported separately.
  - Reviewers re-run Appendix A on the PR base with the PR's `--skip` and `--inert` files, and diff the result.
  - On a merge conflict in a wave's files, take dev's side for that wave's endpoint and test files, re-run the wave, and re-run Appendices B and C on the files that changed.
- **Test-quality ratchet.**
  - The codemod keeps the number of patch calls in every test except inert deletions, which only lower it. `patch.object(...)` becomes `patch("...")`, and both are patch calls for `mock_density`.
  - New tests assert observable values and use no mocks.
- **Chat tests can hit the network.**
  - A converted pair's stub sits on the object the endpoint now calls.
  - Appendix C fails any test whose endpoint patch would have gone dead.
  - Appendix B runs with commercial provider keys unset.

## Review Focus

1. **A home-module patch reaches more code than the endpoint patch did.**
   - The home module's own callers of the name, and other converted endpoints, now see the fake too. In the first draft's full dry run, 959 of 3,298 converted patches (about 29%) named a function its home module also calls internally, `perform_chat_api_call` among them.
   - A test that relied on the unpatched internal call changes outcome, and for the egress client and AuthNZ the reach itself matters.
   - Pinned by each wave's Appendix B comparison over the converted files and every file that already patches a converted home name (`compare_files.txt`).
   - Also pinned by Appendix C's count check for SENSITIVE homes (Ruling 7) and by C7.
2. **A patch stops intercepting while its test still passes.**
   - Matcher-invisible patches of a converted endpoint attribute go dead, and Appendix B reports SAME (review C1).
   - Pinned by Appendix C in dead mode over `runset.txt`: every test file that imports a wave endpoint module (Task 6 Step 4).
3. **A ratchet passes for the wrong reason.**
   - The new counter could miss aliases, `importlib.import_module`, `sys.modules[...]`, string targets, attribute chains, `patch.object`, `mocker`, direct assignment or in-app `tests` packages. It could also misattribute shadowed and rebound names.
   - The codemod's spelling change could lower `.18`'s, `.19`'s, `.13`'s or `.21`'s counts.
   - Pinned by `test_the_counter_counts_each_patch_shape` (Task 1, with shadowed, rebound and function-local cases) and `.18`'s extended `test_each_counter_counts_its_shape`. Each wave re-checks with Appendix E.
4. **A second instance after a move, or moved security code left unguarded.**
   - Pinned by `test_no_module_copies_the_default_sandbox_service` (Task 4, in the gated ratchet module) and `test_mcts_reads_the_published_connection_manager` (Task 4).
   - Also pinned by `test_workers_and_the_endpoint_share_the_core_helpers` (Task 5), and by each widened guard's `test_<guard>_flags_a_violation_in_code_moved_out_of_endpoints` (Task 2).
5. **Route-less code grows back in `endpoints/`, or a reach-in returns.**
   - Pinned by `test_no_route_less_endpoint_modules` (Task 3).
   - Also pinned by the layering contract, whose deleted core→endpoint lines fail as new pairs if an import comes back (Tasks 4-5).

## CI-equivalent checks (each task names the ones it needs)

| Id | What | Command | Pass means |
|---|---|---|---|
| C1 | backend-required "Enforce CI contracts and code ratchets" | Run the step's two pytest commands from `.github/workflows/backend-required.yml` (`:295-338`) verbatim, with `-p no:cacheprovider --basetemp "$TMPDIR/t1320-c1"` added, under `BACKLOG_TASK_FORMAT_BASE=origin/dev PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1` | all pass |
| C2 | backend-required "Enforce tenant isolation ratchets", plus the triage | same env: `tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py`, then `"$PY" Helper_Scripts/ci/test_quality_triage.py --enforce` | all pass; exit 0 |
| C3 | mypy on changed modules (non-blocking in CI) | `uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports $(git diff --name-only --diff-filter=d origin/dev -- 'tldw_Server_API/app/*.py')` | no error on a line this PR changed |
| C4 | security-required "Security suites with TEST_MODE unset" (any PR that changes a file under `tests/AuthNZ*`, `tests/Security` or `tests/http_client`, or converts a SENSITIVE home) | `env -u TEST_MODE -u TESTING -u TLDW_TEST_MODE TLDW_UNSET_TEST_FLAGS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. DISABLE_HEAVY_STARTUP=1 TLDW_TEST_NO_DOCKER=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-c4" -p pytest_asyncio.plugin -p pytest_timeout --timeout=300 -o timeout_method=signal tldw_Server_API/tests/http_client tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Security tldw_Server_API/tests/CI/test_unset_test_flags_run.py` | all pass |
| C5 | Shard coverage and required-workflow contracts (any PR that adds a test file) | `"$PY" Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` and `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-c5" tldw_Server_API/tests/CI/test_required_workflow_contracts.py` | exit 0; pass |
| C6 | Postgres shard | `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$TMPDIR/t1320-pg" <files>`, with `POSTGRES_TEST_*` set as in ci.yml's Postgres shard, or Docker auto-start (`tests/_plugins/postgres.py`). `<files>`: the PR's changed test files matching `grep -lE 'pg_temp_db\|pg_server\|"postgres"\|startswith\("postgresql"\)\|_pg\.py'`, plus `tldw_Server_API/tests/AuthNZ_Postgres` whenever a SENSITIVE AuthNZ home converts | pass, and `-rs` lists no skipped Postgres case |
| C7 | Full Suite comparison, a merge gate for E1, E2 and E3 | After the PR's `CI` run: `gh run view <pr-run> --log-failed \| grep -oE '(FAILED\|ERROR) tldw_Server_API/[^ ]+' \| sort -u > pr.txt`. Do the same over the `CI` runs of the three most recent other backend PRs on dev, unioned into `base.txt`. Then run `comm -23 pr.txt base.txt`. Re-run any printed id on dev in a scratch worktree; one that fails there is not new. `test_model_completion_accounting.py` ("fixture 'mocker' not found") fails on every dev-based run. A printed id maps to its pairs through `pairs_by_test.json`, and Task 6 Step 5's remedy applies | prints nothing; the PR body pastes it |
| C8 | Import smoke | `SINGLE_USER_API_KEY=t1320-import-smoke-key-0123456789 PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1 "$PY" -c 'import importlib, pathlib; [importlib.import_module(".".join(p.with_suffix("").parts).removesuffix(".__init__")) for p in sorted(pathlib.Path("tldw_Server_API/app/api").rglob("endpoints/**/*.py"))]; print("ok")'` | prints `ok` (280 modules on 7a020da14a) |
| C9 | OpenAPI drift (Tasks 2-5, which move code imported by route modules) | `PYTHONPATH=. "$PY" Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | exit 0 |
| C10 | Docs mirror (Task 3, ADR-054) | `bash Helper_Scripts/refresh_docs_published.sh`, then `PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-c10" tldw_Server_API/tests/Docs` | pass, and the `Docs/Published` diff is committed |

## PR slices

| PR | Task | Base | Hand-written lines | Generated or moved | C7 |
|---|---|---|---|---|---|
| A | 1 | dev, after #3433 | ~340 (counter and matcher ~240, `.18` matcher ~45, self-tests, wiring 2, 11 reasons) | 200-entry baseline | no |
| B1 | 2 | A | ~240 (64 pagination import lines, the shared list ~25, five guards widened ~45, five mutation tests ~55, two guardrail extractions ~25, gate wiring and contract ~30) | three pagination functions moved | no |
| B2 | 3 | B1 | ~240 (~120 moved-module import lines, list entries ~15, re-keys ~25, the route-less pin ~60, `pyproject`/`path_classifier`/ADR ~20) | ~4,550 moved (22 `git mv`) | no |
| C | 4 | B2 | ~360 (sandbox ~90 app sites and ~82 test sites, the publish slot ~25, mcts and the engine ~20, 7 MCTS test patches, model-metadata imports ~6, new tests ~45) | ~820 moved (`MODEL_METADATA`) | no |
| D | 5 | B2 | ~300 (endpoint call-through ~40, four importers, ~60 test retargets, a layering bump, a new test ~30) | ~470 moved | no |
| E1 | 6 (W1, chat family) | B2 | ~40 (skip and inert lists, hand fixes, baseline) | codemod ~2,600 changed lines, −482 | gate 1 |
| E2 | 7 (W2 + W4) | B2, after E1 merges | ~60 | ~3,500, −1,109 | gate 2 |
| E3 | 8 (W3 + W5 + the audio shims) | C and D, after E2 merges | ~360 (shim removal ~300, lists ~60) | ~4,200, about −1,149 | gate 3 |
| G | 9 (close-out) | all | ~40 (backlog) | none | no |

- **Each PR is green on its own.**
  - B1 widens the guards and moves pagination, and B2 moves the route-less modules. Each PR lists what it moves in `_moved_endpoint_code.py`, so moved code is never unguarded (Ruling 9).
  - C and D both start from B2, and they are disjoint except for the baseline and layering files.
  - The E PRs start after B2, which rewrites import lines in many of their endpoint modules. E3 waits for C and D, whose code it converts.
- **If a PR runs over about 400 hand-written lines** (review m6):
  - C's model-metadata move becomes its own PR after C;
  - E3's audio shim removal and its audio-only codemod run become E3a (gate 3a), opened before W3 + W5 (gate 3b).
- **The E PRs are opened one at a time**, each once its base is on dev, so each needs one `CI` run.
  - Every wave edits `endpoint_monkeypatch_baseline.json` and shares test files with the others, so parallel E PRs would force a dev merge and a fresh run after every merge.
  - A wave is split only if its re-run on the rebased tree is not reproducible.
- **E1 re-projects the exit** from its real skip and inert lists, and records the re-projection in its PR body before E2 opens (Ruling 10).

## Sibling coordination

- **`.21` config consolidation (plan `2026-11-08-task-13950-21-config-consolidation.md`, read and reconciled).**
  - **Rule (both plans):** `.20` moves code with its env and config reads unchanged, and `.21` retargets each read in its new home.
  - **If `.21`'s PR 1 (`env_read_baseline.json`) is on dev** when a `.20` move lands, the `.20` PR tightens the old module's entry and adds the new module's entry with a literal reason. For example: `"moved from api/v1/endpoints/discord_support.py (TASK-13950.20, PR B2)"`.
  - **If `.20` lands first**, `.21` seeds after it.
  - **Reads this plan carries:**
    - Task 3: `_chatops/settings.py` (5), `chat_import_transport.py` (1), `discord_support.py` (3), `slack_support.py` (2).
    - Task 5: `media_embeddings.py`'s `ALLOW_ZERO_EMBEDDINGS_MEDIA_TYPES`, `EMBEDDINGS_SKIP_MEDIA_TYPES`, `EMBEDDINGS_JOBS_BACKEND` and `TLDW_JOBS_BACKEND`, plus the LazySettings reads in `_user_embedding_config`. All of them go to `core/Embeddings/media_embeddings.py`.
  - **Effect on `.21`'s count.**
    - The codemod never touches `os` (not first-party), so `.21`'s direct reads do not move.
    - `env_flag_enabled("X")` becomes `_testing.env_flag_enabled("X")`, which `.21`'s matcher counts. Appendix E checks `.21`'s total unchanged.
  - **Shared lines.**
    - `.21`'s Task 7 rows name endpoint lines that the waves also edit: `chat.py`, `setup.py`, `auth.py`, `embeddings_v5_production_enhanced.py`, `prompts.py`, `notes.py`. Whichever lands second rebases, and a wave re-runs Appendix A on the rebased file. `evaluations_auth.py:147` stays in place.
    - Both plans add lines to `backend-required.yml` and parametrize entries to `test_license_first_workflow_contracts.py`. The rebase is textual.
- **`.19` PR 11 (#3433, open).**
  - Task 1 seeds after it merges (Ruling 12).
  - It edits `test_mode_branch_baseline.json`, which Task 3 re-keys (`telegram_support.py`); whichever lands second takes both sides and re-runs `test_test_mode_branch_ratchet.py --tighten`.
  - It changes no endpoint patch.
- **`.18` (done).**
  - Task 1 changes its getter matcher and scope.
  - Task 3 re-keys five getter entries.
  - Task 4 deletes `mcts_optimizer.py`'s one `global`.
  - Task 9 records the re-handed getters in its notes.
- **`.74` (evaluations auth onto AuthNZ deps).**
  - `evaluations_auth.py` stays put.
  - E2's W4 rewrites its `get_jwt_service` lookup (6 patches), which is SENSITIVE (Ruling 7).
  - If `.74` lands first, W4 re-runs on `.74`'s version.
- **`.26` (handler use cases; depends on `.20`).**
  - It consumes the module-attribute seam: a handler body moved into a core use case keeps calling `_chat_service.perform_chat_api_call(...)`, and the string patches keep intercepting.
  - The lazy-provider follow-up (Ruling 2) is filed as a dependency of it.
  - It owns persona's and chat's own-name patches if Ruling 1's shortfall clause triggers.
- **`.28` (embeddings adapters; depends on `.20`).**
  - It consumes `core/Embeddings/media_embeddings.py`.
  - Task 5 raises `core.Embeddings -> api.v1.endpoints.embeddings_v5_production_enhanced` from 2 to 4 with a `# bump:` naming `.28` AC2, which cuts all four, plus the 3 in `core.Evaluations`.
  - About 300 patches remain on `embeddings_v5_production_enhanced.py` after W4.
- **`.31` (ProviderSpec).** It finds `MODEL_METADATA` in `core/LLM_Calls/model_metadata.py` (Task 4).
- **`.61`, `.73`, #2630 (audio WebSockets).**
  - Task 8 replaces the shim lookup of `_audio_ws_authenticate` with a direct `_streaming_service._audio_ws_authenticate` call. It is the same function, and `test_websocket_route_auth_matrix.py` runs in C1.
  - If #2630 lands first, Task 8 rebases its `audio_streaming.py` hunks.
- **Open PRs into dev that touch planned files** (`gh pr list --base dev --state open`, 2026-11-08):

  | PR | Planned files it touches | Sequence |
  |---|---|---|
  | #3405, #3394, #3294 | `backend-required.yml`, `test_license_first_workflow_contracts.py` (PR A's and PR B1's lines) | textual rebase either way |
  | #3437 (moderation) | `core/Chat/chat_service.py` (C), `global_statements_baseline.json` (C), `endpoints/chat.py` (E1), `audio/audio_transcriptions.py`, `audio/audio_streaming.py` (E3). It adds 1 endpoint patch | Whichever lands second rebases. After A, its new patch takes a reason naming #3437 (Ruling 12) or a home target |
  | #3180, #854 | `endpoints/llm_providers.py` (C: the `MODEL_METADATA` move; E2), `chat_service.py` (#854). #854 adds about 21 endpoint patches | Same. C's hunk is lines 143-960; #3180's hunks are the imports and `:706-780` |
  | #3404 | `tests/AuthNZ/test_websocket_route_auth_matrix.py` (C retargets its `sandbox_ep._service` read at `:973`) | textual rebase |
  | #2738 | `services/media_ingest_jobs_worker.py` (D, its `:406` lazy import) | textual rebase |
  | #3439, #3436, #3434, #3430, #3241, #3020, #3019, #2903, #2855, #2538 | endpoint modules in E1-E3 (`rag_unified`, `web_scraping`, `claims`, `jobs_admin`, `auth`, `rpg`, `character_chat_sessions`, `calendar`, `outputs`, `watchlists`, `admin/*`, `vector_stores_openai`). #3436 and #2903 add a few endpoint patches | The wave re-runs Appendix A on the merged tree |
  | #3398 (sandbox enforcement) | `core/Sandbox/service.py`, `store.py` only | none: no planned file |

---

### Task 1: The endpoint-patch ratchet; `.18`'s getter counter sees module-attribute calls and `api/*/utils` (AC1, AC5) (PR A)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py`.
- Create (generated): `tldw_Server_API/tests/lint/endpoint_monkeypatch_baseline.json`.
- Modify: `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py`:
  - `_ENDPOINTS` (`:55`) becomes `_GETTER_SCOPE = re.compile(r"api/[^/]+/(endpoints|utils)/")`;
  - `_is_bare_getter` (`:105-112`) becomes `_is_getter_call`;
  - `measure_getter_calls` (`:126-139`) passes the module names;
  - the docstring's getters bullet;
  - `test_each_counter_counts_its_shape` (`:231`).
- Modify (generated): `tldw_Server_API/tests/lint/endpoint_getter_calls_baseline.json` and `tldw_Server_API/tests/lint/api_deps_module_state_baseline.json` (Step 4).
- Modify: `.github/workflows/backend-required.yml`. Add `tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py \` after `tldw_Server_API/tests/lint/test_service_singleton_ratchet.py \` (`:316`).
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add the path to the parametrize list of `test_backend_required_enforces_size_ratchets` (`:1424-1432`). That test checks `_ceilings.BASE_ENV`, which this ratchet uses.

**Interfaces:**
- Consumes: `_ceilings.Ratchet`, `violations`, `load`, `reasonless_bumps`, `warn_if_tightenable` and `main`; `_ast_cache.REPO_ROOT` and `APP_ROOT`. The tests tree is parsed file by file and not cached: the app tree's cached parse is already about 2.5 GB.
- Produces (module `tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet`). The names below are public because Appendix A imports them.
  - `class PatchSite(NamedTuple)` with fields:
    - `node: ast.AST`;
    - `field: str` (`"monkeypatch"` or `"other"`);
    - `verb: str` (`"setattr"`, `"delattr"`, `"patch"`, `"patch.object"` or `"assign"`);
    - `form: str` (`"str"` or `"obj"`);
    - `receiver: str | None`: the dotted path of what is patched, without the attribute;
    - `attr: str | None`: the attribute, when it is a str literal.
  - `class ScopeResolver(tree: ast.Module)`, with `resolve(name: str, at: ast.AST) -> str | None` and `dotted(expr: ast.AST) -> str | None`.
    - A name resolves in the innermost function or lambda that contains the reference, then outward to the module, the way Python scopes it. Class bodies are transparent.
    - In the first scope that binds the name in any way, the binding used is the one with the greatest line number at or before the reference; if none precedes it, the scope's first binding. Ways to bind: an import, an assignment, a parameter, a loop/with/except target or a `del`.
    - That binding gives a dotted path only if it is a module binding:
      - `import a.b.c` gives `a`;
      - `import a.b.c as n` gives `a.b.c`;
      - absolute `from a.b import c [as n]` gives `a.b.c`;
      - `n = <x>.import_module("<literal>")` or `n = import_module("<literal>")` gives the literal;
      - `n = <x>.modules["<literal>"]` gives the literal.
    - Every other binding (a parameter, a relative import, a call's result, tuple unpacking, `__import__`) resolves to None, so a fixture-supplied or helper-returned module is unresolved, as is anything the binding shadows.
    - `dotted(expr)` resolves an `ast.Name` or an `ast.Attribute` chain on one.
  - `patch_sites(tree: ast.Module) -> Iterator[PatchSite]`. One site per counted shape, with no endpoint filter. The shapes:
    - `<x>.setattr(...)`/`<x>.delattr(...)`, where `<x>` is not the Name `object` or `builtins`: field `monkeypatch`;
    - `patch(...)`, `<x>.patch(...)`, and `patch.object(...)`/`<x>.patch.object(...)`: field `other`;
    - `Assign`, `AugAssign` and `Delete` targets that are `ast.Attribute`: field `other`, verb `assign`, `receiver = dotted(target.value)`.

    Their forms and arity:
    - **String form.** The first positional argument is a str literal containing `.`, and `receiver, attr = literal.rsplit(".", 1)`. It counts for `setattr` with exactly 2 positional arguments, `delattr` with exactly 1, and `patch` with any number. It never counts for `patch.object`.
    - **Object form.** Any other first positional argument, for `setattr`, `delattr` and `patch.object`: `receiver = dotted(arg0)`, and `attr` is the second positional argument if it is a str literal, else None.
  - `test_modules(repo_root: Path = REPO_ROOT) -> Iterator[Path]`: every `*.py` under `repo_root/tldw_Server_API` with a `tests` directory component, sorted.
  - `endpoint_module_key(dotted: str, repo_root: Path = REPO_ROOT) -> str | None`. The endpoints package itself counts too: `api/v1/endpoints/__init__.py` is a key, with 2 patches on dev (review m1).
    - It returns the app-relative POSIX path (`api/v1/endpoints/chat.py`, `api/v1/endpoints/audio/__init__.py`) of the longest prefix of `dotted` that is a module or package file under `repo_root/tldw_Server_API/app/api/<version>/endpoints/`, or None.
    - `...endpoints.chat.Thing` therefore keys to `chat.py`.
  - `measure_endpoint_patches(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, int]]`.
    - It returns `{key: {"monkeypatch": n, "other": m}}` over `test_modules`, for files whose bytes contain `b"endpoints"`.
    - Both fields are always present, and keys with `n + m == 0` are omitted.
  - `PATCHES = _ceilings.Ratchet(...)`:
    - `baseline=Path(__file__).resolve().with_name("endpoint_monkeypatch_baseline.json")`;
    - `command="python -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet"`;
    - `fix="Patch the collaborator where the endpoint looks it up: its home module as a string target, app.dependency_overrides for a dependency, or monkeypatch.setattr(app_services(), 'db_pool', fake) and the like for a container service (TASK-13950.20). Patches of endpoint modules only fall."`;
    - `stale="No test patches this endpoint module any more."`.
  - `SEED: dict[str, int]`, Step 4's totals. `EXIT_TARGET = SEED["monkeypatch"] * 55 // 100`.
  - `summary(measured: Mapping[str, Mapping[str, int]]) -> str`. One line: `monkeypatch N (P% of seed S; Phase 2 exit at or below T), other M`, where `P` has one decimal.
  - `__main__`: `--summary` prints `summary(measure_endpoint_patches())`. Any other argument goes to `_ceilings.main(PATCHES, measure_endpoint_patches)`.
- Produces (in `test_service_singleton_ratchet`):
  - `_first_party_modules(tree: ast.Module) -> frozenset[str]`. These are the names bound at module level (`_module_level(tree.body)`) by `import tldw_Server_API.<...> as n`, or by `from tldw_Server_API.<p> import n [as m]` where `<p>.n` is a module or package under `REPO_ROOT`.
  - `_is_getter_call(node: ast.AST, modules: frozenset[str]) -> bool`. True for a zero-argument call whose func is either:
    - an `ast.Name` matching `get_\w+`;
    - an `ast.Attribute` whose value is an `ast.Name` in `modules` and whose attr matches `get_\w+`.

- [ ] **Step 1: Write the self-tests.**

```python
SEED_TREE = {
    "tldw_Server_API/app/api/v1/endpoints/chat.py": "class Thing:\n    y = 1\n",
    "tldw_Server_API/app/api/v1/endpoints/audio/__init__.py": "",
    "tldw_Server_API/app/api/v1/endpoints/audio/tts.py": "",
    "tldw_Server_API/app/core/Chat/svc.py": "",
    "tldw_Server_API/tests/Chat/test_shapes.py": (
        "import importlib, sys\n"
        "from unittest import mock\n"
        "from unittest.mock import patch\n"
        "import tldw_Server_API.app.api.v1.endpoints.chat as chat_ep\n"
        "from tldw_Server_API.app.core.Chat import svc\n"
        "tts = importlib.import_module('tldw_Server_API.app.api.v1.endpoints.audio.tts')\n"
        "pkg = sys.modules['tldw_Server_API.app.api.v1.endpoints.audio']\n"
        "def test_shapes(monkeypatch, mocker, name):\n"
        "    monkeypatch.setattr(chat_ep, 'a', 1)\n"
        "    monkeypatch.setattr('tldw_Server_API.app.api.v1.endpoints.chat._b', 1)\n"
        "    monkeypatch.setattr(chat_ep.Thing, 'y', 2)\n"
        "    monkeypatch.setattr(chat_ep, name, 3)\n"
        "    monkeypatch.setattr(tts, 'c', 1)\n"
        "    monkeypatch.delattr(pkg, 'd')\n"
        "    from tldw_Server_API.app.api.v1.endpoints import audio\n"
        "    monkeypatch.setattr(audio.tts, 'e', 1)\n"
        "    patch('tldw_Server_API.app.api.v1.endpoints.chat.f')\n"
        "    patch.object(chat_ep, 'g')\n"
        "    mock.patch('tldw_Server_API.app.api.v1.endpoints.chat.h')\n"
        "    mocker.patch.object(chat_ep, 'i')\n"
        "    chat_ep.j = 1\n"
        "    del chat_ep.k\n"
        "    monkeypatch.setattr(svc, 'x', 1)\n"
        "    monkeypatch.setattr('tldw_Server_API.app.core.Chat.svc.x', 1)\n"
        "    monkeypatch.setenv('X', '1')\n"
        "    object.__setattr__(chat_ep, 'z', 1)\n"
        "def test_shadowed(chat_ep, monkeypatch):\n"
        "    monkeypatch.setattr(chat_ep, 'p', 1)\n"
        "def test_rebound(monkeypatch):\n"
        "    chat_ep = object()\n"
        "    monkeypatch.setattr(chat_ep, 'q', 1)\n"
        "def test_local(monkeypatch):\n"
        "    from tldw_Server_API.app.api.v1.endpoints.audio import tts as chat_ep\n"
        "    monkeypatch.setattr(chat_ep, 'r', 1)\n"
    ),
    "tldw_Server_API/app/core/MCP_unified/tests/test_in_app.py": (
        "from tldw_Server_API.app.api.v1.endpoints import chat\n"
        "def test_in_app(monkeypatch):\n    monkeypatch.setattr(chat, 'o', 1)\n"
    ),
    "tldw_Server_API/app/core/Chat/not_a_test.py": (
        "import tldw_Server_API.app.api.v1.endpoints.chat as c\n"
        "def f(monkeypatch):\n    monkeypatch.setattr(c, 'n', 1)\n"
    ),
}


def test_the_counter_counts_each_patch_shape(tmp_path: Path) -> None:
    assert measure_endpoint_patches(_seed(tmp_path, SEED_TREE)) == {
        # a, _b, Thing.y, a computed name, o (an in-app tests package); f, g, h, i, j, k.
        # Not p (a parameter shadows the import), not q (rebound locally), not n (not a test module).
        "api/v1/endpoints/chat.py": {"monkeypatch": 5, "other": 6},
        "api/v1/endpoints/audio/tts.py": {"monkeypatch": 3, "other": 0},  # c (import_module), e (package chain), r (a local alias)
        "api/v1/endpoints/audio/__init__.py": {"monkeypatch": 1, "other": 0},  # delattr through sys.modules
    }


def test_growth_a_new_module_and_a_stale_entry_fail_and_shrinking_passes(tmp_path: Path) -> None:
    measured = measure_endpoint_patches(_seed(tmp_path, SEED_TREE))
    key = "api/v1/endpoints/chat.py"
    grown, gone, new = _ceilings.violations(
        PATCHES,
        {**measured, "api/v1/endpoints/new.py": {"monkeypatch": 1, "other": 0}},
        {
            key: {"monkeypatch": 4, "other": 6, "reason": None},
            "api/v1/endpoints/audio/tts.py": {"monkeypatch": 3, "other": 0, "reason": None},
            "api/v1/endpoints/audio/__init__.py": {"monkeypatch": 1, "other": 0, "reason": None},
            "api/v1/endpoints/gone.py": {"monkeypatch": 1, "other": 0, "reason": None},
        },
    )
    assert grown.startswith(f"{key}: monkeypatch 5 > ceiling 4")
    assert gone.startswith("api/v1/endpoints/gone.py: stale entry")
    assert new.startswith("api/v1/endpoints/new.py: monkeypatch 1, other 0, and no ceiling")
    roomy = {k: {**v, "monkeypatch": v["monkeypatch"] + 1, "reason": None} for k, v in measured.items()}
    assert _ceilings.violations(PATCHES, measured, roomy) == []
```

  - `_seed` is the same helper as `test_direct_llm_adapter_calls_ratchet.py:128-133`. Copy it, since importing a `test_` module's names makes pytest collect them twice.
  - The seed's expected values were derived by running exactly this matcher over the seed.
  - In `test_service_singleton_ratchet.py::test_each_counter_counts_its_shape`:
    - prefix the `"api/v1/endpoints/e.py"` source with `"from tldw_Server_API.app.core.AuthNZ import database as _database\nimport tldw_Server_API.app.core.MCP_unified.server as _mcp\n"`;
    - add `"    d = _database.get_db_pool()\n    e = svc.get_w()\n    f = _mcp.get_mcp_server()\n"` after its `c = mod.get_z()` line;
    - add `"api/v1/utils/u.py": "def h():\n    return get_settings()\n"` and `"api/v1/schemas/s.py": "def h():\n    return get_settings()\n"` to the seed;
    - change the getter assertion to `{"api/v1/endpoints/e.py": {"calls": 5}, "api/v1/endpoints/sub/f.py": {"calls": 1}, "api/v1/endpoints/nest.py": {"calls": 1}, "api/v1/utils/u.py": {"calls": 1}}`.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-t1" tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py tldw_Server_API/tests/lint/test_service_singleton_ratchet.py -k "shape or stale"`.
  Expected:
  - `NameError: name 'measure_endpoint_patches' is not defined`;
  - the singleton shape test fails: `e.py` counts 3, not 5, and `api/v1/utils/u.py` is missing.

- [ ] **Step 3: Implement** the Interfaces: the counter, `summary` and `__main__`, then `_first_party_modules` and `_is_getter_call`.
  - The new module's docstring states:
    - the counted shapes, the two fields and the scope rule;
    - the exit metric (`monkeypatch` only) and the seed;
    - the known limits:
      - the builtin `setattr(ep, ...)`;
      - `patch.dict`, `patch.multiple` and `monkeypatch.setitem(<endpoint>.X, ...)`;
      - a module object supplied by a fixture, a helper's return value, tuple unpacking or a parameter;
      - attribute chains through another module's import (`runner_client.sandbox_ep._service`);
      - control-flow-dependent bindings;
      - `getattr`- or f-string-built targets;
    - that each drain PR runs `--tighten`;
    - the commands.
  - In `test_service_singleton_ratchet.py`'s docstring, replace "nor do attribute calls (``mod.get_x()``)" with this text: "attribute calls count when the receiver is a first-party module bound by an import at module level (``_database.get_db_pool()``, since TASK-13950.20), so moving a call onto its home module does not lower the count; method calls on objects (``svc.get_x()``) and receivers imported inside a function do not". The scope sentence then reads "modules under ``api/<version>/endpoints/`` or ``api/<version>/utils/``". That is docstring prose, not a baseline reason.
- [ ] **Step 4: Seed on the dev tip after #3433, and re-seed `.18`.**
  - Run `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet --write-baseline`.
    - Expected on 7a020da14a (the output on the execution tip is the seed): `tldw_Server_API/tests/lint/endpoint_monkeypatch_baseline.json: 200 entries`.
    - `--summary` reads `monkeypatch 5729 (100.0% of seed 5729; Phase 2 exit at or below 3150), other 937`.
    - Spot checks: `"api/v1/endpoints/chat.py": {"monkeypatch": 250, "other": 500, ...}`, `"api/v1/endpoints/embeddings_v5_production_enhanced.py": {"monkeypatch": 380, "other": 39, ...}` and `"api/v1/endpoints/audio/__init__.py": {"monkeypatch": 164, "other": 4, ...}`.
    - Set `SEED` to the printed totals.
  - Run `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_service_singleton_ratchet --write-baseline`. It rewrites all three of `.18`'s baselines.
    - **`endpoint_getter_calls_baseline.json`: 110 entries, 599 calls.**
      - 11 entries are new or raised:
        - `admin/admin_personalization.py` 1;
        - `admin/admin_registration.py` 1;
        - `admin/admin_settings.py` 3;
        - `admin/admin_system.py` 1→3;
        - `chat.py` 22→23;
        - `claims.py` 3→4;
        - `llamacpp.py` 2;
        - `llm_providers.py` 8→9;
        - `metrics.py` 7→10;
        - `setup.py` 3→35;
        - `api/v1/utils/cache.py` 3.
      - Give each of those 11 the reason `"TASK-13950.20: the counter now also counts first-party module-attribute getter calls and the api/v1/utils modules"`.
      - `mcp_unified_endpoint.py` falls 11→10 (slack on dev), which needs no reason.
    - **`api_deps_module_state_baseline.json`: `api/v1/API_Deps/auth_deps.py` falls 6→2.** That is slack on dev: the code holds 2. Commit it as a tighten.
    - **`global_statements_baseline.json`: unchanged (445).**
    - `git diff --stat` lists exactly those two baseline changes.
- [ ] **Step 5: Add the real-tree tests and wire the gate.**
  - `test_endpoint_patches_do_not_grow` and `test_raised_endpoint_patch_ceilings_carry_a_new_reason` are copies of `test_direct_llm_adapter_calls_ratchet.py:204-215` with `PATCHES` and `measure_endpoint_patches`.
  - Make the workflow and contract edits from Files.
- [ ] **Step 6: Run the modules and the contract.**
  Run: `BACKLOG_TASK_FORMAT_BASE=origin/dev "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-t1" tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py tldw_Server_API/tests/lint/test_service_singleton_ratchet.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k "patch or singleton or getter or size_ratchets"`.
  Expected:
  - all pass, which the dry run reproduced (20 passed);
  - the new module's 4 tests;
  - the singleton module's tests, including both `[getters]` cases;
  - 4 `test_backend_required_enforces_size_ratchets` cases.

  Then run C1 and C2.
- [ ] **Step 7: Commit.**

```bash
git add tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py tldw_Server_API/tests/lint/endpoint_monkeypatch_baseline.json tldw_Server_API/tests/lint/test_service_singleton_ratchet.py tldw_Server_API/tests/lint/endpoint_getter_calls_baseline.json tldw_Server_API/tests/lint/api_deps_module_state_baseline.json .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): freeze test patches of endpoint modules; getter counter sees module-attribute calls (TASK-13950.20)"
```

### Task 2: `_pagination_utils` merges into `api/v1/utils/pagination.py`, and the endpoint-scoped guards follow moved code (AC2, AC5) (PR B1)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/utils/pagination.py`.
  - Append `build_link_header`, `build_pagination_link_header` and `resolve_page_pagination_metadata`, moved verbatim from `endpoints/_pagination_utils.py:25-161`.
  - Add the imports they need: `urllib.parse as _u`, `Mapping`, `Any`, and `CursorPaginationMeta`/`OffsetPaginationMeta` from `api.v1.schemas.pagination`.
  - The file has no `__all__`; do not add one.
- Delete: `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`.
- Modify: the 62 app importers (64 statements). `from tldw_Server_API.app.api.v1.endpoints._pagination_utils import` and `from ._pagination_utils import` (`chatbooks.py:48`) become `from tldw_Server_API.app.api.v1.utils.pagination import`. Keep each statement's position, including the two function-level ones in `workflows.py:2086,2456`.
- Modify: `tldw_Server_API/tests/Utils/test_pagination_contract.py:4` (the import).
- Modify: `tldw_Server_API/tests/lint/layering_baseline.txt`, deleting its 45 `-> api.v1.endpoints._pagination_utils` lines (64 statements) with `--tighten`.
- Modify: `tldw_Server_API/tests/lint/test_private_coercion_ratchet.py:48`. The key becomes `"tldw_Server_API/app/api/v1/utils/pagination.py": 1` (the nested `_as_int`), in sorted order.
- Modify: `tldw_Server_API/tests/lint/test_layering_contract.py:24`. The family examples become ``.media``, ``.chat``, ``.audio``.
- Create: `tldw_Server_API/tests/lint/_moved_endpoint_code.py`, listing `tldw_Server_API/app/api/v1/utils/pagination.py`.
- Modify (Ruling 9): the five endpoint-scoped guards. Each scans `_scan_files()`, which is `[*sorted(ENDPOINTS_ROOT.rglob("*.py")), *moved_files()]` with `moved_files` imported by name, so a test can patch it. Each reports a path by `_display(path)`: the repo-relative POSIX path, or `str(path)` outside the repo. Each gains a mutation test (Step 1). The edits:
  - `tests/lint/test_auth_token_parsing_ratchet.py`: `current_bearer_parses` (`:144-145`) counts over `_scan_files()` through `_counts`. `_counts` takes the path list, and its key uses `_display`.
  - `tests/lint/test_endpoint_auth_deps_import_boundary.py`: `_direct_auth_dependency_imports` (`:72-78`).
  - `tests/AuthNZ/unit/test_require_admin_callsite_guardrail.py`: the inline loop (`:19-46`) becomes `_offenders(paths: Iterable[Path]) -> list[str]` with a module-level `ENDPOINTS_ROOT`, and the test asserts `_offenders(_scan_files()) == []`.
  - `tests/AuthNZ/unit/test_get_current_user_callsite_guardrail.py` (`:20-43`): the same extraction.
  - In both guardrails, offender lines change prefix from `app/api/...` (today relative to `tldw_Server_API`) to `tldw_Server_API/app/api/...` under `_display`. Nothing keys on that text.
  - `tests/lint/test_no_raw_sql_in_endpoints.py`: `_measure` (`:114-124`) and `test_no_endpoint_reaches_database_internals` (`:188`). An endpoint file keeps its `ENDPOINTS_ROOT`-relative key, and a moved file is keyed by `_display`.
- Modify: `.github/workflows/backend-required.yml`. Add `test_endpoint_auth_deps_import_boundary.py`, `test_no_raw_sql_in_endpoints.py`, `tests/AuthNZ/unit/test_require_admin_callsite_guardrail.py` and `tests/AuthNZ/unit/test_get_current_user_callsite_guardrail.py` to the step's first pytest command. AC5 applies, since this task changes them. `.7`'s ratchet already runs in the step's second command.
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add `test_backend_required_enforces_endpoint_scope_guards(guard)`, a copy of `test_backend_required_enforces_llm_gateway_guards` (`:1484-1507`) parametrized over those four paths.

**Interfaces:**
- Produces `tests/lint/_moved_endpoint_code.py`:
  - `MOVED_OUT_OF_ENDPOINTS: tuple[str, ...]`: repo-relative files, where a directory stands for its `*.py` files. PR B1 lists `tldw_Server_API/app/api/v1/utils/pagination.py`. Tasks 3, 4 and 5 append their destinations in the PR that moves them.
  - `moved_files(repo_root: Path = REPO_ROOT) -> list[Path]`, sorted. It raises `FileNotFoundError` naming a listed path that no longer exists, so every guard's real-tree run fails on a stale list.
- Produces `tldw_Server_API.app.api.v1.utils.pagination`'s `build_link_header`, `build_pagination_link_header` and `resolve_page_pagination_metadata`, with unchanged signatures.

- [ ] **Step 1: Write the failing tests.**
  - Change `test_pagination_contract.py:4` to import from `tldw_Server_API.app.api.v1.utils.pagination`, and add:

```python
def test_the_endpoint_pagination_module_is_gone() -> None:
    import importlib.util

    assert importlib.util.find_spec("tldw_Server_API.app.api.v1.endpoints._pagination_utils") is None
```

  - Add a mutation test to each guard's own file (N3). It seeds one violation in a file at a moved-destination path and points the guard's `moved_files` at it. The test fails if the guard does not scan moved code.

```python
def test_<guard>_flags_a_violation_in_code_moved_out_of_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seeded = tmp_path / "api" / "v1" / "utils" / "chatops" / "seeded.py"
    seeded.parent.mkdir(parents=True)
    seeded.write_text(VIOLATION, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "moved_files", lambda *_args: [seeded])
    assert REPORT
```

  | Guard file | `<guard>` | `VIOLATION` | `REPORT` |
  |---|---|---|---|
  | `tests/lint/test_auth_token_parsing_ratchet.py` | `bearer_ratchet` | `"def f(header):\n    return header.startswith('Bearer ')\n"` | `current_bearer_parses().get(str(seeded)) == 1` |
  | `tests/lint/test_endpoint_auth_deps_import_boundary.py` | `auth_deps_boundary` | `"from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User\n"` | `any(line.startswith(f"{seeded}:1:") for line in _direct_auth_dependency_imports())` |
  | `tests/AuthNZ/unit/test_require_admin_callsite_guardrail.py` | `require_admin_guardrail` | `"x = Depends(require_admin)\n"` | `_offenders(_scan_files()) == [f"{seeded}:1:x = Depends(require_admin)"]` |
  | `tests/AuthNZ/unit/test_get_current_user_callsite_guardrail.py` | `get_current_user_guardrail` | `"x = Depends(get_current_user)\n"` | `_offenders(_scan_files()) == [f"{seeded}:1:x = Depends(get_current_user)"]` |
  | `tests/lint/test_no_raw_sql_in_endpoints.py` | `raw_sql_ratchet` | `"def f(db):\n    db.execute('SELECT 1')\n"` | `_measure().get(str(seeded)) == 1` |

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-t2" -p pytest_asyncio.plugin tldw_Server_API/tests/Utils/test_pagination_contract.py tldw_Server_API/tests/lint/test_auth_token_parsing_ratchet.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py tldw_Server_API/tests/AuthNZ/unit/test_require_admin_callsite_guardrail.py tldw_Server_API/tests/AuthNZ/unit/test_get_current_user_callsite_guardrail.py tldw_Server_API/tests/lint/test_no_raw_sql_in_endpoints.py`, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
  Expected:
  - a collection error: `ImportError: cannot import name 'build_link_header' from 'tldw_Server_API.app.api.v1.utils.pagination'`;
  - each mutation test fails with `AttributeError: ... has no attribute 'moved_files'`, because the guard does not yet scan moved code;
  - a guard that imports `moved_files` but leaves it out of `_scan_files()` still fails, now on the assertion. For the `require_admin` guardrail the failure reads `assert [] == ['$TMP/api/v1/utils/chatops/seeded.py:1:x = Depends(require_admin)']`, where `$TMP` is the test's `tmp_path`.
- [ ] **Step 3: Move the functions, rewrite the importers and widen the guards** (Files).
  - Grep and run (the symbol rule): `git grep -nwE "_pagination_utils|build_link_header|build_pagination_link_header|resolve_page_pagination_metadata" -- tldw_Server_API Helper_Scripts .github` on the PR base lists every reader. After the change:
    - each app hit imports from `api.v1.utils.pagination`;
    - `_pagination_utils` appears only in the gone-pin;
    - every test file the base grep listed runs in Step 4's comparison.
- [ ] **Step 4: Re-run the ratchets, the comparison and the checks.**
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. Expected: the 45 lines are gone, and no line is added.
  - Run Step 2's command. Expected: pass. Also run `tests/lint/test_private_coercion_ratchet.py`. Expected: pass.
  - Run `BASE_WT=<scratch worktree of the PR base> "$PY" "$T1320_DIR/t1320_compare.py" <(git grep -lwE "build_link_header|build_pagination_link_header|resolve_page_pagination_metadata|build_offset_pagination_meta" -- 'tldw_Server_API/tests/*.py') -n 4`. Expected: no `DIFFERS` and no `VACUOUS` line.
  - Run C1 (it now runs the four newly wired guards), C2, C4 (the two AuthNZ guardrails), C8 and C9.
- [ ] **Step 5: Commit.** `git commit -m "refactor(api): endpoint pagination helpers merge into api/v1/utils/pagination; endpoint-scoped guards follow moved code (TASK-13950.20)"`.

### Task 3: The route-less endpoint modules leave `endpoints/` (AC2) (PR B2)

**Files:**
- Move with `git mv` (content unchanged except the import lines of moved modules):

  | From `api/v1/endpoints/` | To |
  |---|---|
  | `_chatops/{__init__,ingress,oauth_admin,policy,settings}.py` | `api/v1/utils/chatops/` (same names) |
  | `discord_oauth_admin.py`, `discord_support.py`, `slack_oauth_admin.py`, `slack_support.py`, `telegram_support.py` | `api/v1/utils/chatops/` (same names) |
  | `_in_memory_limits.py` | `api/v1/utils/in_memory_limits.py` |
  | `chat_import_transport.py`, `media_navigation_policy.py`, `notes_sync_errors.py`, `storage_helpers.py`, `workspace_chat_startup_transport.py`, `workspaces_rate_limit_policy.py` | `api/v1/utils/` (same names) |
  | `kanban/_kanban_utils.py` | `api/v1/utils/kanban.py` |
  | `media/deprecation_signals.py`, `media/input_contracts.py` | `api/v1/utils/media_deprecation_signals.py`, `api/v1/utils/media_input_contracts.py` |
  | `prompt_studio/resource_binding.py` | `api/v1/utils/prompt_studio_resource_binding.py` |
  | `sandbox_service.py` | `core/Sandbox/default_service.py` |

  - Destination rule: each module imports FastAPI, `API_Deps` or `api.v1.schemas`, or sits beside ones that do, so `api/v1/utils` is the layer that may import them. `sandbox_service.py` imports only `core.Sandbox.service`, and Task 4 makes core read it.
  - No moved module imports an unmoved endpoint module, so no `api.v1.utils -> api.v1.endpoints` pair appears.
- Modify: every importer. An AST scan finds 78 files: 41 app and 37 tests. Between them they hold 87 absolute and 9 relative from-imports, 13 package-from imports (`from ...endpoints import <module>`), 2 `import ... as` forms and 2 string constants. Every reference names the new absolute path.
  - The relative imports inside the moved modules change to absolute ones:
    - `discord_oauth_admin.py:17-18` and `slack_oauth_admin.py:15-16` (`from ._chatops import ...`);
    - `_chatops/__init__.py:27` (`from . ...`);
    - the two support modules' `from ._chatops...` imports.
  - `sandbox.py:39` and `sandbox_workspace_diagnostics.py:15` import `sandbox_service` from `tldw_Server_API.app.core.Sandbox.default_service`. Task 4 replaces that.
- Modify: the two tests that load moved modules by file path:
  - `tests/Media_Ingestion_Modification/test_input_contracts.py:8-14` (`/ "endpoints" / "media" / "input_contracts.py"`, `spec_from_file_location`);
  - `tests/Media_Ingestion_Modification/test_media_deprecation_signals.py:8-14`.

  Each becomes a plain import of `tldw_Server_API.app.api.v1.utils.media_input_contracts`/`media_deprecation_signals`.
- Modify: `tldw_Server_API/tests/lint/_moved_endpoint_code.py`. Append every Task 3 destination: `tldw_Server_API/app/api/v1/utils/chatops`, the 11 other moved `api/v1/utils` files and `tldw_Server_API/app/core/Sandbox/default_service.py`. Task 2's guards then scan them in this same PR (Ruling 9).
- Modify (re-key, each with its literal reason):
  - `endpoint_getter_calls_baseline.json`. Each new key gets its old key as a literal reason:
    - `api/v1/utils/chatops/ingress.py` 1, reason `"moved from api/v1/endpoints/_chatops/ingress.py (TASK-13950.20, PR B2)"`;
    - `api/v1/utils/chatops/oauth_admin.py` 9, reason `"moved from api/v1/endpoints/_chatops/oauth_admin.py (TASK-13950.20, PR B2)"`;
    - `api/v1/utils/chatops/discord_support.py` 2, reason `"moved from api/v1/endpoints/discord_support.py (TASK-13950.20, PR B2)"`;
    - `api/v1/utils/chatops/slack_support.py` 2, reason `"moved from api/v1/endpoints/slack_support.py (TASK-13950.20, PR B2)"`;
    - `api/v1/utils/chatops/telegram_support.py` 14, reason `"moved from api/v1/endpoints/telegram_support.py (TASK-13950.20, PR B2)"`.
  - `test_mode_branch_baseline.json:42`: key `api/v1/utils/chatops/telegram_support.py`, lines 3, reason `"moved from api/v1/endpoints/telegram_support.py (TASK-13950.20, PR B2)"`.
  - `test_private_coercion_ratchet.py:54,66,67`: the keys become `tldw_Server_API/app/api/v1/utils/chatops/{discord_support,slack_support,telegram_support}.py`, in sorted order. That ratchet takes no reason.
  - `env_read_baseline.json`, if `.21` PR 1 is on dev: the four modules in Sibling coordination, each re-keyed the same way. For example, `api/v1/utils/chatops/settings.py` gets the reason `"moved from api/v1/endpoints/_chatops/settings.py (TASK-13950.20, PR B2)"`.
  - `layering_baseline.txt`: `--tighten` deletes the 38 lines (44 statements) whose importer or target is a moved module, such as `api.v1.endpoints.slack -> api.v1.endpoints._chatops`. No line is added: a moved module imports only `API_Deps`, `schemas`, `utils`, core and the other moved modules.
  - `endpoint_monkeypatch_baseline.json`: `--tighten` deletes `discord_support.py`, `slack_support.py`, `storage_helpers.py` and `telegram_support.py` (30 patches, all module-object form, retargeted by the import rewrite).
  - `pyproject.toml:808,850,855`: the `BLE001` keys take the new paths.
- Modify: `Helper_Scripts/ci/path_classifier.py:57-64`. Add `"tldw_Server_API/app/api/v1/utils/**"` and `"tldw_Server_API/app/core/Sandbox/default_service.py"` to `E2E_BACKEND_GLOBS`, with a `tests/CI/test_path_classifier.py` case `test_moved_endpoint_code_still_enables_e2e`.
- Modify: `Docs/ADR/054-chatops-shared-shell.md:13,24,25,27,28` and `Docs/ADR/README.md:88`. `endpoints/_chatops/` becomes `api/v1/utils/chatops/`, and one sentence says TASK-13950.20 moved the shell out of `endpoints/`. Run C10.
- Modify: `tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py`. Add the route-less pin.

**Interfaces:**
- Consumes Task 2's `_moved_endpoint_code.py` and the five widened guards.
- Produces:
  - `ROUTE_LESS_ALLOWED: dict[str, str] = {"api/v1/endpoints/control_plane.py": "route handlers main.py registers with add_api_route", "api/v1/endpoints/evaluations/evaluations_auth.py": "TASK-13950.74 moves it onto the AuthNZ dependencies"}`.
  - `route_less_endpoint_modules(root: Path = APP_ROOT) -> list[str]`. These are the sorted app-relative keys of modules under `api/<v>/endpoints/` that are not `__init__.py` and contain neither of these:
    - a function decorated `@<x>.<verb>(...)` with verb in `{"get", "post", "put", "patch", "delete", "head", "options", "api_route", "websocket", "websocket_route", "route"}`;
    - a call to `<x>.add_api_route`, `<x>.add_api_websocket_route` or `<x>.include_router`.

    It reads `_ast_cache.parse_tree(root)`.

- [ ] **Step 1: Write the route-less pin.**

```python
def test_route_less_modules_are_found(tmp_path: Path) -> None:
    root = _seed(tmp_path, {
        "app/api/v1/endpoints/a.py": "from fastapi import APIRouter\nr = APIRouter()\n@r.get('/a')\ndef a():\n    return 1\n",
        "app/api/v1/endpoints/b.py": "def helper():\n    return 1\n",
        "app/api/v1/endpoints/c/__init__.py": "",
        "app/api/v1/endpoints/c/agg.py": "from fastapi import APIRouter\nr = APIRouter()\nr.include_router(object())\n",
        "app/api/v1/utils/u.py": "def helper():\n    return 1\n",
    })
    assert route_less_endpoint_modules(root / "app") == ["api/v1/endpoints/b.py"]


def test_no_route_less_endpoint_modules() -> None:
    assert sorted(set(route_less_endpoint_modules()) - set(ROUTE_LESS_ALLOWED)) == []
    assert sorted(set(ROUTE_LESS_ALLOWED) - set(route_less_endpoint_modules())) == []  # an allowed module that grew routes is stale
```

- [ ] **Step 2: Run it and confirm it fails.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-t3" tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py -k route_less`.
  Expected:
  - the self-test passes once implemented;
  - `test_no_route_less_endpoint_modules` lists 21 modules: the 20 this task moves to `api/v1/utils/` and `sandbox_service.py`. Package `__init__` files are not checked. `_pagination_utils.py` is already gone by Task 2, and the 24 on dev were exactly the backlog's 24.
- [ ] **Step 3: Make the moves and rewrite the importers** (Files). Then grep and run (the symbol rule). Each of these prints nothing:
  - `git grep -nE "endpoints(\.|/)(_chatops|_in_memory_limits|chat_import_transport|discord_oauth_admin|discord_support|kanban(\.|/)_kanban_utils|media(\.|/)(deprecation_signals|input_contracts)|media_navigation_policy|notes_sync_errors|prompt_studio(\.|/)resource_binding|sandbox_service|slack_oauth_admin|slack_support|storage_helpers|telegram_support|workspace_chat_startup_transport|workspaces_rate_limit_policy)" -- tldw_Server_API Helper_Scripts .github Docs/ADR pyproject.toml`;
  - `git grep -nE "from \.(_chatops|_in_memory_limits|discord_support|slack_support|discord_oauth_admin|slack_oauth_admin|telegram_support|storage_helpers|notes_sync_errors|workspaces_rate_limit_policy|chat_import_transport|workspace_chat_startup_transport|media_navigation_policy|sandbox_service|_kanban_utils|deprecation_signals|input_contracts|resource_binding)" -- tldw_Server_API`;
  - `git grep -nE '"(input_contracts|deprecation_signals|_kanban_utils|resource_binding|sandbox_service|_in_memory_limits)\.py"' -- tldw_Server_API/tests`, the path-built references.

  Then save the test files the first grep lists on the PR base as `$T1320_DIR/b2-readers.txt`. They run in Step 5.
- [ ] **Step 4: Re-key and tighten** (Files).
  - Run `--tighten` for the layering contract and the monkeypatch ratchet.
  - Run `"$PY" -m tldw_Server_API.tests.lint.test_service_singleton_ratchet --write-baseline`, then write the 5 literal reasons.
  - Re-key `.19`'s and `.21`'s entries by hand, since `.19`'s `--write-baseline` would drop other reasons.
- [ ] **Step 5: Run the tests, the comparison and the checks.**
  - Run Step 2's command, and Task 2 Step 2's guard files. Expected: pass. The guards now scan the moved files, and none flags one.
  - Run `BASE_WT=<scratch worktree of the PR base> "$PY" "$T1320_DIR/t1320_compare.py" <(cat "$T1320_DIR/b2-readers.txt"; git diff --name-only origin/dev -- 'tldw_Server_API/tests/*.py' | grep -v '/lint/'; ls tldw_Server_API/tests/Media_Ingestion_Modification/test_*.py) -n 4`. Expected: no `DIFFERS` and no `VACUOUS` line.
  - Run C1, C2, C4, C5, C8, C9 and C10. C4 covers the telegram authority tests.
  - Run C6 for `tests/AuthNZ_Postgres/*telegram*`, if the grep selects any.
- [ ] **Step 6: Commit.**
  - Stage the moves with `git add -A tldw_Server_API/app/api tldw_Server_API/app/core/Sandbox`.
  - Then stage the importers, `_moved_endpoint_code.py`, the baselines, `pyproject.toml`, `path_classifier.py` and its test, the ADR files and `Docs/Published`.
  - Then run `git commit -m "refactor(api): route-less endpoint modules move to api/v1/utils (TASK-13950.20)"`.

### Task 4: Core stops reading the sandbox endpoint, the Prompt Studio WebSocket module and `llm_providers` (AC3) (PR C)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`.
  - The import from Task 3 becomes `from tldw_Server_API.app.core.Sandbox import default_service as _sandbox_default`.
  - Delete `_service = sandbox_service` (`:179`).
  - Every Load of `_service` (38) becomes `_sandbox_default.sandbox_service`, rewritten in place so `start_run` (393) and `stream_run_logs` (355) keep their line counts.
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py`. `:15` gets the same import, the Store at `:37` is deleted, and its 3 `_service` Loads are rewritten the same way.
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py`. `:18` becomes `from tldw_Server_API.app.core.Sandbox import default_service as _sandbox_default`, and `sandbox_ep._service` at `:143,515,853,858` becomes `_sandbox_default.sandbox_service`.
- Modify (tests), in 16 files:
  - the 35 `setattr(<sandbox endpoint>, "_service", fake)` patches become `monkeypatch.setattr("tldw_Server_API.app.core.Sandbox.default_service.sandbox_service", fake)`. They are in 7 files, and 5 of them go through `src.sandbox_ep` in `test_acp_sandbox_runner_client.py`;
  - the 47 reads `<sandbox endpoint>._service...` in 10 files become `default_service.sandbox_service...`, with `from tldw_Server_API.app.core.Sandbox import default_service` imported where each read is.

  The patch files:
  - `Agent_Client_Protocol/test_acp_sandbox_runner_client.py`;
  - `AuthNZ_Unit/test_sandbox_admin_permissions_claims.py`;
  - `sandbox/test_admin_macos_diagnostics.py`, `test_admin_macos_image_store_cleanup_plan.py`, `test_admin_macos_image_store_cleanup.py`, `test_admin_macos_reconciliation_repair.py` and `test_session_upload.py`.

  The read files:
  - `AuthNZ/test_websocket_route_auth_matrix.py:973`;
  - `sandbox/test_admin_list_filters_pagination.py`, `test_artifact_content_type_and_path.py`, `test_artifact_range.py`, `test_artifact_range_additional.py`, `test_artifacts_perf_large_tree.py`, `test_sandbox_api.py`, `test_sandbox_cancel_error_mapping.py`, `test_session_upload.py` and `test_workspace_diagnostics.py`.
- Create: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/ws_connections.py` (Ruling 8). It imports nothing above stdlib.
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_websocket.py`. Right after `connection_manager = ConnectionManager()` (`:458`), add `ws_connections.publish(connection_manager)`. `ConnectionManager` (`:126-292`) and `register_shutdown_transport_family(...)` (`:459-463`) stay where they are.
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/mcts_optimizer.py`.
  - Delete the guarded endpoint import (`:55-61`).
  - Delete the comment and the `global` retry block in `MCTSOptimizer.optimize` (`:247-256`; the function is frozen at 442 lines and only shrinks).
  - At `:259-260`, read `manager = ws_connections.current()` into a local, with `from tldw_Server_API.app.core.Prompt_Management.prompt_studio import ws_connections` at module level.
  - Behaviour is unchanged in the server: there is no broadcaster unless the WebSocket module has loaded (Ruling 8). `_MCTS_IMPORT_EXCEPTIONS` stays, because `:80` uses it.
- Modify: `tldw_Server_API/app/core/Prompt_Management/prompt_studio/optimization_engine.py:1078`. `connection_manager = mcts_optimizer_module.ws_connection_manager` becomes `connection_manager = ws_connections.current()`, with the same module-level import (re-review N1).
- Modify (tests): the 7 patches of `mcts_optimizer.ws_connection_manager` become `monkeypatch.setattr(ws_connections, "current", lambda: <the value the test set>)`:
  - `tests/prompt_studio/unit/test_optimization_atomic_transitions.py:345,655`;
  - `tests/prompt_studio/unit/test_jobs_worker_completion_callbacks.py:477`;
  - `tests/prompt_studio/integration/test_optimization_job_engine_wiring.py:194,945`;
  - `tests/prompt_studio/integration/test_mcts_observability_and_persistence.py:60,277`.
- Create: `tldw_Server_API/app/core/LLM_Calls/model_metadata.py`. It holds `MODEL_METADATA`, `_default_model_metadata` and `get_model_metadata`, moved verbatim from `llm_providers.py:143-960`, with their imports (`canonical_anthropic_model_id`, `merge_routing_metadata`, `Any`).
- Modify: `llm_providers.py`. It imports `get_model_metadata` from the new module; its uses at `:2139,2885` are unchanged.
- Modify: `chat_service.py:1522-1524`. It imports from `tldw_Server_API.app.core.LLM_Calls.model_metadata`.
- Modify (tests): `tests/Chat/unit/test_model_router_metadata.py:3` and `tests/Chat/unit/test_anthropic_dotted_model_names.py:172`.
- Modify: `tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py`. Add `test_no_module_copies_the_default_sandbox_service`, which lives in a gated module (review I8).
- Modify: `tldw_Server_API/tests/lint/_moved_endpoint_code.py`. Append `tldw_Server_API/app/core/LLM_Calls/model_metadata.py` (Ruling 9).
- Create: `tldw_Server_API/tests/prompt_studio/test_ws_connections_slot.py`.
- Modify (ratchets):
  - `layering_baseline.txt`: `--tighten` deletes `core.Agent_Client_Protocol -> api.v1.endpoints.sandbox`, `core.Chat -> api.v1.endpoints.llm_providers` and `core.Prompt_Management -> api.v1.endpoints.prompt_studio`. No line is added;
  - `global_statements_baseline.json`: `--tighten` drops the `mcts_optimizer.py` entry;
  - `endpoint_monkeypatch_baseline.json`: `--tighten`, −30 on `sandbox.py`.

**Interfaces:**
- Produces:
  - `tldw_Server_API.app.core.Sandbox.default_service.sandbox_service: SandboxService`, the one process instance. Callers look it up as `default_service.sandbox_service` at call time, and nothing copies it.
  - `tldw_Server_API.app.core.Prompt_Management.prompt_studio.ws_connections.publish(manager: T) -> T` and `current() -> Any | None`.
    - `publish` stores the manager in a module-level `dict` (`_published["manager"]`, no `global` statement) and returns it. A re-import replaces the previous one.
    - `current` returns it, or None before the WebSocket module has loaded.
  - `tldw_Server_API.app.core.LLM_Calls.model_metadata.get_model_metadata(provider: str, model: str) -> dict[str, Any]`.
- Consumes: Task 3's `core/Sandbox/default_service.py`, and Task 2's `_moved_endpoint_code.py`.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/lint/test_endpoint_monkeypatch_ratchet.py
def test_no_module_copies_the_default_sandbox_service() -> None:
    """Callers read default_service.sandbox_service at call time, so one patch reaches routes and the ACP runner."""
    copies = []
    for path, tree in parse_tree(APP_ROOT).items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("Sandbox.default_service"):
                copies += [f"{path.relative_to(APP_ROOT)}:{node.lineno}" for a in node.names if a.name == "sandbox_service"]
        for node in tree.body:  # a module-level copy; a local read inside a function is a call-time lookup
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and node.value.attr == "sandbox_service":
                copies.append(f"{path.relative_to(APP_ROOT)}:{node.lineno}")
    assert copies == []
```

```python
# tests/prompt_studio/test_ws_connections_slot.py
import pytest

pytestmark = pytest.mark.unit


def test_mcts_reads_the_published_connection_manager() -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_websocket
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import mcts_optimizer, ws_connections

    assert ws_connections.current() is prompt_studio_websocket.connection_manager
    assert not hasattr(mcts_optimizer, "ws_connection_manager")  # no module-level copy that could go stale
```

  Also retarget `test_model_router_metadata.py:3` to `from tldw_Server_API.app.core.LLM_Calls.model_metadata import get_model_metadata`.
- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-t4" tldw_Server_API/tests/lint/test_endpoint_monkeypatch_ratchet.py tldw_Server_API/tests/prompt_studio/test_ws_connections_slot.py tldw_Server_API/tests/Chat/unit/test_model_router_metadata.py -k "copies or published or metadata"`.
  Expected:
  - `copies == ['api/v1/endpoints/sandbox.py:39', 'api/v1/endpoints/sandbox_workspace_diagnostics.py:15']` (Task 3's by-name imports);
  - `ImportError: cannot import name 'ws_connections'`;
  - `ModuleNotFoundError: ...model_metadata`.
- [ ] **Step 3: Implement** (Files). Then check:
  - `git grep -n "sandbox_ep\|endpoints import sandbox as\|endpoints.prompt_studio\|endpoints.llm_providers" -- tldw_Server_API/app/core` prints nothing;
  - `git grep -nE "(sandbox[a-z_]*|sb|src\.sandbox_ep)\._service|\"_service\"" -- tldw_Server_API/tests | grep -iv vn_assets` prints nothing.
  - Grep and run (the symbol rule). The symbols are `ws_connection_manager`, `sandbox_service`, `_service`, `sandbox_ep`, `MODEL_METADATA`, `_default_model_metadata` and `get_model_metadata`.
    - `git grep -nw "ws_connection_manager" -- tldw_Server_API` prints nothing.
    - On the PR base, save every test file that `git grep -lwE "ws_connection_manager|sandbox_service|sandbox_ep|MODEL_METADATA|_default_model_metadata|get_model_metadata|mcts_optimizer|optimization_engine" -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/*/tests/*.py'` lists, plus the 16 sandbox files, as `$T1320_DIR/c-readers.txt`.
- [ ] **Step 4: Prove the moves, then run the checks.**
  - Run Step 2's command (pass), `tests/lint/test_llm_calls_import_isolation.py` (pass: `model_metadata` loads only `LLM_Calls` modules), and `tests/lint/test_layering_contract.py` after `--tighten`. The contract must add no `core.Prompt_Management -> services` or `-> api` pair.
  - Write `"$T1320_DIR/t4-report.txt"` with two lines:
    - `CONVERT tldw_Server_API.app.api.v1.endpoints.sandbox._service -> tldw_Server_API.app.core.Sandbox.default_service.sandbox_service (35 patches, 38 loads)`;
    - `CONVERT tldw_Server_API.app.api.v1.endpoints.sandbox_workspace_diagnostics._service -> tldw_Server_API.app.core.Sandbox.default_service.sandbox_service (0 patches, 3 loads)`.
  - Run Appendix C in dead mode with that report over `tldw_Server_API/tests/sandbox tldw_Server_API/tests/Agent_Client_Protocol tldw_Server_API/tests/AuthNZ_Unit/test_sandbox_admin_permissions_claims.py tldw_Server_API/tests/AuthNZ/test_websocket_route_auth_matrix.py`. Expected: no `T1320 dead endpoint patch` failure.
  - Run `BASE_WT=... "$PY" "$T1320_DIR/t1320_compare.py" <(cat "$T1320_DIR/c-readers.txt"; git diff --name-only origin/dev -- 'tldw_Server_API/tests/*.py'; ls tldw_Server_API/tests/prompt_studio/unit/test_*.py tldw_Server_API/tests/prompt_studio/integration/test_*.py) -n 4`. This covers the MCTS and Prompt Studio optimization tests. Expected: no `DIFFERS` or `VACUOUS` line.
  - Tighten the three baselines.
  - Run C1 (it runs `test_websocket_route_auth_matrix.py`), C2, C4 (the `AuthNZ_Unit` and `AuthNZ` files) and C5. The SENSITIVE `core.Sandbox` home is one patch spelling, not a broadened fake: every caller already read `sandbox_ep._service` at call time.
  - Run C8 and C9.
- [ ] **Step 5: Commit.** `git commit -m "refactor(core): sandbox runner, MCTS and chat context windows stop importing endpoint modules (TASK-13950.20)"`.

### Task 5: The media-embeddings helpers move to core (AC3) (PR D)

**Files:**
- Create: `tldw_Server_API/app/core/Embeddings/media_embeddings.py`. It holds, verbatim from `api/v1/endpoints/media_embeddings.py`:
  - `:35-37`: `DEFAULT_EMBEDDING_MODEL`, `DEFAULT_EMBEDDING_PROVIDER`, `FALLBACK_EMBEDDING_MODEL`;
  - `:40-53`: the two exception tuples;
  - `:56-130`: `_user_embedding_config`, `_safe_int`, `_parse_media_type_list`, `_allow_zero_embeddings_for_media`, `_resolve_model_provider`, `_embeddings_jobs_backend`;
  - `:223-594`: `get_media_content`, `chunk_media_content`, `generate_embeddings_for_media`;
  - the imports they use. Its two lazy imports of `embeddings_v5_production_enhanced` (`:111`, `:331`) move unchanged.
- Modify: `api/v1/endpoints/media_embeddings.py`.
  - It keeps `router`, `MAX_MEDIA_EMBEDDING_JOBS_OFFSET`, the request/response models, the routes and the search route's lazy import (`:868`).
  - It imports `from tldw_Server_API.app.core.Embeddings import media_embeddings as _media_embeddings`, and every function-body Load of a moved name becomes `_media_embeddings.<name>`. Module-level uses import the constants and exception tuples by name.
- Modify:
  - `core/Embeddings/services/jobs_worker.py:42-48` and `core/Ingestion_Media_Processing/persistence.py:2588-2591` import from `tldw_Server_API.app.core.Embeddings.media_embeddings`. `jobs_worker.py` keeps importing the names by name, since tests patch `jobs_worker.<name>`.
  - `services/media_ingest_jobs_worker.py:406-409` does the same.
  - `api/v1/endpoints/media/reprocess.py:12,119-160` reads its helpers and constants from `_media_embeddings`. Its `.ChromaDBManager` read becomes `ChromaDB_Library.ChromaDBManager`, which removes the one call-time app referrer review I2 found.
- Modify (tests): every patch or read of a moved name on the endpoint module becomes the string target `"tldw_Server_API.app.core.Embeddings.media_embeddings.<name>"`. On dev that is 49 own-name patches: `_embeddings_jobs_backend` 7, `_resolve_model_provider` 9, `_user_embedding_config` 9, `chunk_media_content` 6, `generate_embeddings_for_media` 9, `get_media_content` 9.
  - A test that patches an imported name the moved code uses (`ChromaDBManager`, `get_media_by_id`, `EmbeddingsJobsAdapter`) and exercises a moved function patches that name on the core module. One that exercises a route body keeps the endpoint target, which W3 converts later. Appendices B and C decide each file.
  - Start from `git grep -ln "media_embeddings" -- tldw_Server_API/tests` (29 files on dev).
- Create: `tldw_Server_API/tests/Embeddings/test_media_embeddings_core_home.py`.
- Modify: `tldw_Server_API/tests/lint/_moved_endpoint_code.py`. Append `tldw_Server_API/app/core/Embeddings/media_embeddings.py` (Ruling 9).
- Modify: `pyproject.toml:833`. Add `"tldw_Server_API/app/core/Embeddings/media_embeddings.py" = ["BLE001"]` beside the endpoint's key, since both files keep broad handlers.
- Modify (ratchets):
  - `layering_baseline.txt`. Delete `core.Embeddings -> api.v1.endpoints.media_embeddings`, `core.Ingestion_Media_Processing -> api.v1.endpoints.media_embeddings`, `services -> api.v1.endpoints.media_embeddings` and `api.v1.endpoints.media -> api.v1.endpoints.media_embeddings`.
  - `layering_baseline.txt`. `api.v1.endpoints.media_embeddings -> api.v1.endpoints.embeddings_v5_production_enhanced` goes 3 → 1.
  - `layering_baseline.txt`. Replace `core.Embeddings -> api.v1.endpoints.embeddings_v5_production_enhanced: 2` with this line: `core.Embeddings -> api.v1.endpoints.embeddings_v5_production_enhanced: 4  # bump: the media-embeddings helpers moved from api.v1.endpoints.media_embeddings with their two lazy imports (TASK-13950.20); TASK-13950.28 AC2 removes all four`.
  - Net core→endpoint statements: −3 (`media_embeddings` from `core.Embeddings`, `core.Ingestion_Media_Processing` and `services`), +2 on the `embeddings_v5` line. Task 9's AC3 note says so.
  - `test_private_coercion_ratchet.py:60`: the key becomes `tldw_Server_API/app/core/Embeddings/media_embeddings.py` (`_safe_int`).
  - `test_core_to_api_import_boundary.py:14-19`: the docstring says core reaches `api.v1.endpoints.*` only through `.28`'s embeddings imports.
  - `endpoint_monkeypatch_baseline.json`: `--tighten`.
  - `env_read_baseline.json`, if `.21` PR 1 is on dev: the key `core/Embeddings/media_embeddings.py`, with the reason `"moved from api/v1/endpoints/media_embeddings.py (TASK-13950.20, PR D)"`.

**Interfaces:**
- Produces: `tldw_Server_API.app.core.Embeddings.media_embeddings` with the names above and unchanged signatures. These include `generate_embeddings_for_media(media_id: int, media_content: dict[str, Any], embedding_model: str, embedding_provider: str, chunk_size: int, chunk_overlap: int, user_id: str = "1") -> dict[str, Any]` and `get_media_content(media_id: int, db: Any) -> dict[str, Any]`.
- Consumed by `.28`.

- [ ] **Step 1: Write the failing test.**

```python
import pytest

pytestmark = pytest.mark.unit
MOVED = ("generate_embeddings_for_media", "get_media_content", "chunk_media_content", "_resolve_model_provider",
         "_allow_zero_embeddings_for_media", "_user_embedding_config", "_embeddings_jobs_backend")


def test_workers_and_the_endpoint_share_the_core_helpers() -> None:
    from tldw_Server_API.app.core.Embeddings import media_embeddings as core
    from tldw_Server_API.app.core.Embeddings.services import jobs_worker

    assert jobs_worker.generate_embeddings_for_media is core.generate_embeddings_for_media
    assert jobs_worker.FALLBACK_EMBEDDING_MODEL == core.FALLBACK_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_the_endpoint_no_longer_defines_the_helpers() -> None:
    from tldw_Server_API.app.api.v1.endpoints import media_embeddings as endpoint

    assert [name for name in MOVED if hasattr(endpoint, name)] == []  # a stale endpoint patch raises instead of missing
```

- [ ] **Step 2: Run it and confirm it fails.** Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1320-t5" tldw_Server_API/tests/Embeddings/test_media_embeddings_core_home.py`. Expected: `ModuleNotFoundError: ...core.Embeddings.media_embeddings`, and a list of all 7 names.
- [ ] **Step 3: Implement** (Files). Then:
  - `git grep -n "endpoints.media_embeddings import\|endpoints import media_embeddings" -- tldw_Server_API/app` prints nothing.
  - Grep and run (the symbol rule). On the PR base, save every file that `git grep -lwE "generate_embeddings_for_media|get_media_content|chunk_media_content|_resolve_model_provider|_allow_zero_embeddings_for_media|_user_embedding_config|_embeddings_jobs_backend|_safe_int|_parse_media_type_list|DEFAULT_EMBEDDING_MODEL|DEFAULT_EMBEDDING_PROVIDER|FALLBACK_EMBEDDING_MODEL" -- tldw_Server_API` lists, as `$T1320_DIR/d-readers.txt`.
    - Every app hit reads the core module now.
    - Every test hit runs in Step 4's comparison.
- [ ] **Step 4: Prove the move, then run the checks.**
  - Write `"$T1320_DIR/t5-report.txt"`, one line per name in `MOVED` and per imported name whose Loads moved to core. Each line reads `CONVERT tldw_Server_API.app.api.v1.endpoints.media_embeddings.<name> -> tldw_Server_API.app.core.Embeddings.media_embeddings.<name> (0 patches, 0 loads)`. Take the names from `analyse()`'s loads on the old endpoint file minus those on the new one.
  - Run Appendix C:
    - count mode on the base;
    - dead mode on the head with `T1320_ALLOW` set to Appendix D's `inert` output;
    - both over `git grep -ln "media_embeddings" -- 'tldw_Server_API/tests/*.py'`.

    Expected: no dead failure. Delete each inert site and list it in the PR body (Ruling 7).
  - Run Step 2 (pass).
  - Run `BASE_WT=... "$PY" "$T1320_DIR/t1320_compare.py" <(grep "/tests/" "$T1320_DIR/d-readers.txt"; git grep -ln "media_embeddings" -- 'tldw_Server_API/tests/*.py' | grep -v /lint/) -n 4`. Expected: no `DIFFERS` or `VACUOUS` line.
  - Run C1 (layering with the bump reason, `test_core_to_api_import_boundary.py`), C2, C5, C8 and C9.
  - Run C6 unconditionally on `tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py` and `tests/VectorStores/integration/test_vector_stores_real_db.py` (`ChromaDBManager`/pgvector paths).
- [ ] **Step 5: Commit.** `git commit -m "refactor(embeddings): media-embedding helpers move to core; workers stop importing the endpoint (TASK-13950.20)"`.

### Task 6: The seam codemod and its proof, wave W1: the chat family (AC4) (PR E1, gate 1)

**Files:**
- Modify (generated): the endpoint modules `api/v1/endpoints/{chat,character_chat_sessions,messages,character_messages,chat_documents,chat_dictionaries,characters_endpoint,character_memory,chat_grammars,chatbooks}.py`, and the test files Appendix A changes. Those include the three Chat fixture modules: `tests/Chat/credential_runtime_fixtures.py`, `tests/Chat/integration/conftest_isolated.py` and `tests/Chat_NEW/conftest.py`.
- Modify: `tldw_Server_API/tests/lint/endpoint_monkeypatch_baseline.json` (`--tighten`).
- Create (not committed): in `$T1320_DIR`:
  - `skip.txt`, shared by the waves, which starts empty;
  - `w1.txt`;
  - `w1-inert.txt`;
  - the count files.

**Interfaces:**
- Consumes Task 1's ratchet, matcher and getter counter, and Appendices A-E.
- Produces endpoint modules whose patched collaborators are looked up as `_<leaf>.<name>` on their home module, with no dead patch on any converted pair.

- [x] **Step 0: Owner gate. Satisfied: the owner accepted Ruling 4 on 2026-11-08.**
  - The E1 PR body cites that answer.
- [ ] **Step 1: Build the wave list and record the totals.**

```bash
bash -c 'fam() { for f in "$@"; do d=tldw_Server_API/app/api/v1/endpoints/$f; if [ -d "$d" ]; then find "$d" -name "*.py"; else echo "$d.py"; fi; done; }
fam chat character_chat_sessions messages character_messages chat_documents chat_dictionaries characters_endpoint character_memory chat_grammars chatbooks | sort > "$T1320_DIR/w1.txt"'
PYTHONPATH=. TEST_MODE=true "$PY" "$T1320_DIR/t1320_totals.py" > "$T1320_DIR/w1-totals-before.json"
```

  - Set up a scratch worktree of the PR base and `export BASE_WT=<its path>`.
- [ ] **Step 2: Run the codemod once for its pair list, then measure the base.**
  - Run `"$PY" "$T1320_DIR/t1320_seams.py" --skip "$T1320_DIR/skip.txt" --out "$T1320_DIR/w1" $(cat "$T1320_DIR/w1.txt")`, and keep `codemod_report.txt` and `runset.txt`. Then `git checkout -- tldw_Server_API`.
  - On `BASE_WT`, run Appendix C in count mode over `runset.txt`:
    - command: `T1320_REPORT="$T1320_DIR/w1/codemod_report.txt" T1320_MODE=count T1320_COUNT_OUT="$T1320_DIR/w1/base" PYTHONPATH=.:"$T1320_DIR/plugin" TLDW_TEST_NO_DOCKER=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 "$PY" -m pytest -q -p no:cacheprovider -p pytest_asyncio.plugin -p pytest_timeout --timeout=300 -p xdist.plugin -n 4 -p t1320_tripwire $(cat "$T1320_DIR/w1/runset.txt")`. The plugin unsets the provider keys itself;
    - then `"$PY" "$T1320_DIR/t1320_counts.py" inert "$T1320_DIR/w1/codemod_report.txt" "$T1320_DIR/w1/base" > "$T1320_DIR/w1-inert.txt"`.
- [ ] **Step 3: Run the codemod for real.**
  - Run `"$PY" "$T1320_DIR/t1320_seams.py" --skip "$T1320_DIR/skip.txt" --inert "$T1320_DIR/w1-inert.txt" --out "$T1320_DIR/w1" $(cat "$T1320_DIR/w1.txt")`.
  - Expected:
    - on the 7a020da14a dry run, W1 converted 482 `monkeypatch` patches, about 2,600 changed lines (the reviewers measured 2,620 with the first draft's script). `setattr:obj` + `setattr:str` in the JSON line is the `monkeypatch` drain;
    - `inert_deleted` counts deletions. A deletion can leave a fake's definition unused (F841; the `lint` job is non-blocking). The PR body lists them;
    - `git diff --check` prints nothing, and `python3 Helper_Scripts/checks/guard_http_client_patching.py` exits 0.
- [ ] **Step 4: Prove interception on the head (Ruling 7).**
  - Run Appendix C over `runset.txt` on the head twice, with Step 2's command except:
    - `T1320_MODE=dead T1320_ALLOW="$T1320_DIR/w1-inert.txt"`;
    - `T1320_MODE=count T1320_COUNT_OUT="$T1320_DIR/w1/head"`.
  - Then run `"$PY" "$T1320_DIR/t1320_counts.py" broadened "$T1320_DIR/w1/codemod_report.txt" "$T1320_DIR/w1/base" "$T1320_DIR/w1/head"`.
  - Known limit (review m5): `runset.txt` passes conftest and fixture modules to pytest, which collects no tests from them. A test that reaches a wave endpoint only through a conftest fixture therefore runs here only if its own file imports the endpoint. The dry run found no such patch site on dev; C7 covers the rest.
  - Each `T1320 dead endpoint patch` names `<site file>::<pair>`. Add that pair to `skip.txt`, unless its site was inert on the base, in which case allow or delete it. A pair carrying 20 or more patches may instead be hand-retargeted at the dead site, counted as hand-written and listed.
  - Add every pair a `broadened` line names to `skip.txt`.
  - Then `git checkout -- tldw_Server_API` and repeat Steps 3-4 until both print nothing.
- [ ] **Step 5: Compare outcomes (review I3).**
  - Run `BASE_WT=... "$PY" "$T1320_DIR/t1320_compare.py" "$T1320_DIR/w1/compare_files.txt" -n 4 | tee "$T1320_DIR/w1-compare.txt"`. The script already re-runs a differing file alone and calls it `FLAKY` when it then matches.
  - For each `DIFFERS: <file>`, take its pairs from `pairs_by_test.json`. Skip them one at a time, smallest first, re-running Steps 3-5 for that file until it is SAME.
  - If the file has no pairs, grep it for a patch of a CONVERT alias. That is a dead patch Appendix C missed (an untracked spelling): treat it as Step 4's case.
  - A `DIFFERS` on a pair with 20 or more patches may be fixed by hand in the test instead, counted and listed.
  - Stop after 3 rounds per file and record the pairs.
  - A `VACUOUS` file proves nothing. List it in the PR body, and rely on C7 for it.
  - Pass: every line is `SAME`, `FLAKY` (listed) or `VACUOUS` (listed).
- [ ] **Step 6: Check the totals and the gates.**
  - `"$PY" "$T1320_DIR/t1320_totals.py"` prints the same JSON as `w1-totals-before.json`.
  - C8 prints `ok`.
  - Run `--tighten`, then `--summary`.
  - C1, C2, C4 (any `tests/AuthNZ*` change or a SENSITIVE home), C6 (the grep plus `tests/AuthNZ_Postgres` for an AuthNZ home), and C7 after the PR's `CI` run: a merge gate.
- [ ] **Step 7: Re-project and commit.**
  - Re-project the exit (Ruling 10). From W1's real drain, skips and inert deletions, scale the remaining waves' dry-run drains by W1's ratio of real to dry-run drain, and add Tasks 3-5 and 8.
  - Write the result in the PR body. If it lands above `EXIT_TARGET`, say so, and name the extra pairs W2-W5 must hand-fix or the shortfall Ruling 1 files.
  - Stage exactly the files `git diff --name-only` lists, then run `git commit -m "refactor(chat): chat endpoints look patched collaborators up on their home modules; tests patch the home (TASK-13950.20, generated)"`.
  - The PR body carries:
    - Appendices A-E as run;
    - the base sha;
    - `skip.txt` and `w1-inert.txt`, with each deletion's `inert` line;
    - the codemod JSON line;
    - the `--summary` line;
    - the re-projection;
    - the hand-written and generated line counts.

### Task 7: Waves W2 and W4 (AC4) (PR E2, gate 2)

**Files:** the endpoint modules of W2 and W4, the test files Appendix A changes, and `endpoint_monkeypatch_baseline.json`. It is one PR, opened after E1 merges.

**Interfaces:** Consumes Task 6's procedure, its scripts and the shared `skip.txt`.

| Wave | `fam` arguments (Task 6 Step 1) | Dry-run drain |
|---|---|---|
| W2 | `persona llm_providers prompts mcp_unified_endpoint mcp_hub_management mcp_catalogs_manage agent_client_protocol agent_orchestration acp_multiplex acp_permissions voice_assistant writing writing_manuscripts skills tools mlx llamacpp` | −544 |
| W4 | `evaluations rag_unified embeddings_v5_production_enhanced chunking chunking_templates vector_stores_openai outputs notes notes_graph notes_graph_suggestions notes_semantic_index research research_runs research_workspace paper_search benchmark_api claims flashcards quizzes study_suggestions slides slides_standalone_html translate text2sql` | −565 |

- [ ] **Step 1: Run Task 6 Steps 1-6 once over the union of the W2 and W4 lists** (`e2.txt`).
  - W2 holds `agent_client_protocol.py`, the review's dead-patch reproduction. Its two sites in `tests/Agent_Client_Protocol/test_async_api.py:143,154` count 0 calls on the base, so they go to `e2-inert.txt` and are allowlisted, not skipped.
  - W4 converts `evaluations_auth.py`'s `get_jwt_service` lookup, which is SENSITIVE. It changes neither `.7`'s bearer count nor `.19`'s `SECURITY_FUNCTIONS` pin, and C1 proves both.
  - W4's computed-name tests in `test_embeddings_orchestrator_endpoint_parity.py` skip their pairs.
  - The `prompts.py` getter pair proves Ruling 2's lazy order, as long as Appendix E's getter total holds.
- [ ] **Step 2: Commit** (Task 6 Step 7's message, with `chat` replaced by `mcp, persona and evaluations`), with its own re-projection, and gate on C7.

### Task 8: Waves W3 and W5, and the audio package's shim lookups go (AC4, phase exit) (PR E3, gate 3)

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/{__init__,audio,audio_streaming,audio_tokenizer,audio_transcriptions,audio_tts}.py`.
  - Each of the 48 `_audio_shim_attr("<name>")` lookups and 8 `getattr(<audio package>, "<name>", <default>)` lookups becomes `<binding of the home module>.<name>`. The home module is the module `audio/__init__.py` imports `<name>` from, for example `_audio_quota.can_start_stream` or `_streaming_service._audio_ws_authenticate`.
  - For a name defined in an audio endpoint module (`get_tts_service`, `create_speech`), the lookup becomes `audio_tts.<name>`, which stays an endpoint patch target.
  - Delete the four `_audio_shim_attr` definitions (`audio_streaming.py:312`, `audio_tokenizer.py:43`, `audio_transcriptions.py:202`, `audio_tts.py:110`) and the 8 function-level imports of the package or `audio.audio` they used.
  - Delete each `_shim_*` wrapper that becomes a one-line pass-through, and call its target directly.
- Modify (generated): the W3 and W5 endpoint modules, every audio module through `--reexports`, and the test files Appendix A changes.
- Modify: `endpoint_monkeypatch_baseline.json` (`--tighten`) and `tests/lint/test_endpoint_monkeypatch_ratchet.py` (the shim pin).

**Interfaces:** Consumes Task 6's procedure. Produces an `audio` package whose `__init__.py` re-exports only what an importer outside it uses.

| Wave | `fam` arguments | Dry-run drain |
|---|---|---|
| W3 | `media prompt_studio media_embeddings ocr items files reading reading_highlights` | −359 |
| W5 | every other module. Write the list with: `bash -c 'fam() { ...as in Task 6 Step 1...; }; { fam <W1, W2, W3 and W4 arguments>; } \| sort -u > "$T1320_DIR/w1-4.txt"; find tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/api/v2/endpoints -name "*.py" -not -path "*/endpoints/audio/*" \| sort \| comm -23 - "$T1320_DIR/w1-4.txt" > "$T1320_DIR/w5.txt"'`. It is built from the `fam` arguments, so it does not depend on older list files | −594 |
| audio | `audio`, with `--reexports`, after Step 2 | about −196 (upper bound) |

- [ ] **Step 1: Write the failing pin.** Add to `tests/lint/test_endpoint_monkeypatch_ratchet.py`:

```python
def test_audio_endpoints_hold_no_shim_lookups() -> None:
    """A by-name lookup on the package lets a test patch hide behind a string; collaborators come from their home."""
    root = APP_ROOT / "api" / "v1" / "endpoints" / "audio"
    endpoints = "tldw_Server_API.app.api.v1.endpoints"
    hits = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.FunctionDef) and node.name == "_audio_shim_attr":
                hits.append(f"{path.name}:{node.lineno} def")
            elif isinstance(node, ast.ImportFrom) and node.col_offset > 0 and node.module in {endpoints, f"{endpoints}.audio"} \
                    and any(alias.name == "audio" for alias in node.names):
                hits.append(f"{path.name}:{node.lineno} import")
    assert hits == []
```

  Run it. Expected: FAIL listing 4 `def` hits and 8 `import` hits, which the dry run reproduced.
- [ ] **Step 2: Remove the shims** (Files).
  - `git grep -n "_audio_shim_attr\|getattr(_audio_pkg\|getattr(audio_pkg" -- tldw_Server_API/app` prints nothing.
  - Run Appendix C in dead mode over `git grep -ln "endpoints.audio\|endpoints import audio" -- 'tldw_Server_API/tests/*.py'`, with a report holding one CONVERT line per removed package name. Each line reads `CONVERT tldw_Server_API.app.api.v1.endpoints.audio.<name> -> <its home>.<name> (0 patches, 0 loads)`.
  - Its failures are the package patches Step 3's `--reexports` run must retarget, so run it before Step 3, and again after.
- [ ] **Step 3: Run Task 6 Steps 1-6 once over W3 + W5 + the audio modules** (`e3.txt`, with `--reexports`).
  - Grep and run (the symbol rule). On the PR base, `git grep -lwE "_audio_shim_attr|_shim_[a-z_]+|audio_pkg_shim|audio_module_shim|audio_shim" -- tldw_Server_API` lists the readers of what Step 2 deletes. Each test file in that list joins `e3.txt`'s comparison.
  - Every name Appendix A drops from `audio/__init__.py` has no remaining referrer, which is Appendix A's KEEP rule.
- [ ] **Step 4: Check the exit.**
  - `"$PY" -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet --summary` prints the total, which should be at or below `EXIT_TARGET` (projection about 2,880, 50.3%). E3 is the last drain PR.
  - If the total is above, E3 still merges, and Task 9 applies Ruling 1's shortfall clause.
- [ ] **Step 5: Commit.** `git commit -m "refactor(api): media, prompt-studio and remaining endpoints, and the audio package, look collaborators up on their home modules (TASK-13950.20)"`. Then gate on C7.

### Task 9: Close-out (AC1-AC5, DoD) (PR G)

**Files:** `backlog/tasks/task-13950.20 - ...md`, `backlog/tasks/task-13950.18 - ...md` (notes), and new follow-up task files, all through `tools/backlog-py` only; stage with `git add -A backlog/tasks`.

- [ ] **Step 1: Check each AC on dev, after E3 has merged.**
  - **AC1:**
    - `test_endpoint_monkeypatch_ratchet.py` passes on dev;
    - `--summary` shows `monkeypatch` ≤ `EXIT_TARGET`;
    - record the seed, the final count and the percentage.
  - **AC2:**
    - `test_no_route_less_endpoint_modules` and `test_the_endpoint_pagination_module_is_gone` pass;
    - record the 22 moves, Ruling 5's two exceptions, and the zero-shim deviation from "with re-export shims while tests migrate".
  - **AC3:**
    - `grep -E "^(core\.[^ ]*|services) -> api\.v1\.endpoints" tldw_Server_API/tests/lint/layering_baseline.txt` prints only the `embeddings_v5_production_enhanced` lines (`.28`) and `core.Setup -> api.v1.endpoints.audio`;
    - the chat `API_KEYS` item cites 7cffa1362f;
    - the note states Task 5's moved, not vanished, statements (−3, +2 on a line `.28` cuts).
  - **AC4:**
    - list the migrated pairs, with the E PRs' `--summary` lines, skip lists, inert deletions and hand fixes;
    - state the owner's confirmation of Ruling 4;
    - no converted test patches an endpoint module. Re-run Appendix A with `--out` to a scratch dir, then `git checkout -- .`, and show that no `CONVERT` line remains outside the skip list.
  - **AC5:**
    - `backend-required.yml` lists `test_endpoint_monkeypatch_ratchet.py` and the four guards Task 2 wired;
    - their contract cases pass;
    - `.18`'s and `.7`'s changed ratchets keep their existing assertions.
- [ ] **Step 2: Rewrite AC4's text** to Ruling 4's reading, as confirmed. Keep the AC numbering, as `.18` did for its AC2.
- [ ] **Step 3: File follow-ups** with `tools/backlog-py`:
  - "Inject endpoint-body services through lazy request-scoped providers", as a dependency of TASK-13950.26 (Ruling 2). Record the re-hand in `.18`'s notes;
  - "Cut core.Setup -> api.v1.endpoints.audio (install_manager imports audio_health)";
  - "Drain the endpoint logger patches" (538 on dev). They need a log-capture seam, which the codemod cannot generate;
  - "Retarget the skip-listed pairs", with the union of the E PRs' skip lists;
  - if Ruling 1's shortfall clause applied: the remaining count against `.26` and `.28`, recorded in their notes.
- [ ] **Step 4: Record the notes and the DoD.**
  - Write the final summary.
  - Run Bandit (`-ll`) on the touched app files.
  - Record known skips. The local Postgres cases are proven only by C6. Every dev-based CI run errors on `test_model_completion_accounting.py` with "fixture 'mocker' not found".
  - Record TASK-13950's Phase 2 metric line: endpoint-module monkeypatches, seed → final.
- [ ] **Step 5: Commit.** `git commit -m "docs(backlog): close TASK-13950.20 with AC evidence and follow-ups"`.

## Appendix A: `t1320_seams.py`

Save as `"$T1320_DIR/t1320_seams.py"`. It imports Task 1's matcher, so it runs only on a branch that has Task 1.

```python
"""TASK-13950.20 seam codemod: endpoints look patched collaborators up on their home module.

Usage (repo root, after Task 1 is on the branch):
    python t1320_seams.py [--skip FILE] [--inert FILE] [--reexports] [--out DIR] ENDPOINT.py ...

Patch sites come from Task 1's ``patch_sites`` (the counter's own matcher, so the codemod converts
exactly what the counter counts). For each endpoint module given, and each name ``alias`` it binds
with a top-level absolute ``from M import orig [as alias]`` (M first-party, outside api/*/endpoints;
an alias module, ``sys.modules[__name__] = X``, is replaced by X) that a test patches as
``<endpoint>.<alias>``:
  * app: each Load of ``alias`` in a function or lambda body becomes ``<binding of M>.<orig>``,
    reusing a top-level binding of M or adding ``import M as _<leaf>`` after the last import;
  * tests: each such patch becomes a string patch of ``"M.orig"`` (patch.object becomes patch);
  * the from-import drops ``alias`` unless a module-level Load, or an importer or reader of
    ``<endpoint>.<alias>`` elsewhere, needs it (reported KEEP).
SKIP (reported) when: the name is ``logger``; M is not a module or is an endpoint module; a
function body binds the name or declares it ``global``; no function body loads it; a test
assigns or deletes ``<endpoint>.<alias>``; a test patches the endpoint with a computed name that
appears as a string in that top-level test function or class; app code in the endpoint's
package (the endpoint itself for a top-level endpoint module) holds the name as an identifier
string outside ``__all__``; an app module reads ``<endpoint>.<alias>`` inside a function body;
or ``--skip`` lists ``<endpoint dotted>.<alias>``. ``--reexports`` also converts a package
``__init__``'s names no function body loads (tests only).
``--inert`` lists ``<test file>::<endpoint dotted>.<alias>`` sites Appendix C proved intercept
nothing on the base: a standalone ``monkeypatch.setattr``/``delattr`` statement there is deleted,
any other shape is left as it is (reported INERT). A converted home under SENSITIVE is reported.
Writes codemod_report.txt, pairs_by_test.json, compare_files.txt and runset.txt.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from functools import cache
from pathlib import Path

from tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet import ScopeResolver, patch_sites, test_modules

REPO = Path.cwd()
EP = ("tldw_Server_API.app.api.v1.endpoints", "tldw_Server_API.app.api.v2.endpoints")
SENSITIVE = ("tldw_Server_API.app.core.http_client", "tldw_Server_API.app.core.Utils.egress", "tldw_Server_API.app.core.Security",
             "tldw_Server_API.app.core.AuthNZ", "tldw_Server_API.app.core.Sandbox")
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def is_ep(d):
    return bool(d) and any(d == p or d.startswith(p + ".") for p in EP)


def module_file(d):
    p = REPO.joinpath(*d.split("."))
    for c in (p.with_suffix(".py"), p / "__init__.py"):
        if d.startswith("tldw_Server_API.") and c.is_file():
            return c
    return None


@cache
def canonical(d):
    """Follow ``sys.modules[__name__] = X`` alias modules to the module they stand for."""
    for _ in range(5):
        f = module_file(d)
        tree = ast.parse(f.read_bytes()) if f else None
        names = {a.asname or a.name: (f"{n.module}.{a.name}" if isinstance(n, ast.ImportFrom) else a.name)
                 for n in (tree.body if tree else []) if isinstance(n, (ast.Import, ast.ImportFrom)) and getattr(n, "level", 0) == 0
                 for a in n.names}
        target = next((names.get(n.value.id) for n in (tree.body if tree else []) if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Subscript) and isinstance(n.value, ast.Name)
                       and ast.unparse(n.targets[0]) == "sys.modules[__name__]"), None)
        if not target:
            return d
        d = target
    return d


def dotted_of(path):
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def inside_functions(tree):
    ids = set()
    for fn in ast.walk(tree):
        if isinstance(fn, FUNCS):
            for stmt in fn.body if isinstance(fn.body, list) else [fn.body]:
                ids |= {id(x) for x in ast.walk(stmt)}
    return ids


def identifier_strings(tree):
    exported = set()
    for n in tree.body:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign)) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            exported |= {id(c) for c in ast.walk(n.value)}
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier() and id(n) not in exported}


def analyse(tree):
    imports = {a.asname or a.name: (n.module, a.name, n) for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module for a in n.names}
    inside, bound, declared_global = inside_functions(tree), set(), set()
    for fn in ast.walk(tree):
        if isinstance(fn, FUNCS):
            a = fn.args
            bound |= {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y]}
    for x in ast.walk(tree):
        if id(x) in inside:
            if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
                bound.add(x.id)
            elif isinstance(x, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name).split(".")[0] for a in x.names}
            elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(x.name)
            elif isinstance(x, ast.Global):
                declared_global |= set(x.names)
    loads, top_loads = defaultdict(list), defaultdict(int)
    for x in ast.walk(tree):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in imports:
            (loads[x.id].append(x) if id(x) in inside else top_loads.__setitem__(x.id, top_loads[x.id] + 1))
    module_names = {}
    for n in tree.body:
        if isinstance(n, ast.Import):
            module_names.update({a.name: a.asname for a in n.names if a.asname and a.name not in module_names})
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            for a in n.names:
                if module_file(f"{n.module}.{a.name}"):
                    module_names.setdefault(f"{n.module}.{a.name}", a.asname or a.name)
    return imports, loads, top_loads, bound, declared_global, module_names


def edit(text, edits):
    """Replace (line, col, end_line, end_col, text) spans; 1-based lines, UTF-8 byte columns."""
    lines = text.splitlines(keepends=True)
    for ln, col, eln, ecol, rep in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        head, tail = lines[ln - 1].encode()[:col], lines[eln - 1].encode()[ecol:]
        lines[ln - 1] = (head + rep.encode() + tail).decode()
        for i in range(ln, eln):
            lines[i] = ""
    return "".join(lines)


def without(text, node, drop):
    """``text`` with the names in ``drop`` removed from the from-import ``node``, layout kept."""
    lines = text.splitlines(keepends=True)
    keep = [a for a in node.names if (a.asname or a.name) not in drop]
    span = range(node.lineno - 1, node.end_lineno)
    item = lambda i: lines[i].split("#")[0].strip().rstrip(",")  # noqa: E731
    dropped = [a for a in node.names if (a.asname or a.name) in drop]
    if not keep:
        gone = set(span)
    elif node.lineno != node.end_lineno and all(sum(item(i) in {a.name, f"{a.name} as {a.asname}"} for i in span) == 1 for a in dropped):
        gone = {i for i in span for a in dropped if item(i) in {a.name, f"{a.name} as {a.asname}"}}
    else:
        indent = lines[node.lineno - 1][: len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
        names = [a.name + (f" as {a.asname}" if a.asname else "") for a in keep]
        one = f"{indent}from {node.module} import {', '.join(names)}\n"
        lines[node.lineno - 1] = one if len(one) <= 120 else f"{indent}from {node.module} import (\n" + "".join(f"{indent}    {n},\n" for n in names) + f"{indent})\n"
        gone = set(span) - {node.lineno - 1}
    return "".join(line for i, line in enumerate(lines) if i not in gone)


def named_modules(tree):
    """Every dotted path a module imports (``from P import m`` gives ``P`` and ``P.m``) or holds as a string."""
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names |= {n.module} | {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("tldw_Server_API."):
            names.add(n.value)
    return names


def names_module(names, ep):
    """True when a module imports ``ep``, one of its submodules, or names either in a string."""
    return any(d == ep or d.startswith(ep + ".") for d in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip")
    ap.add_argument("--inert")
    ap.add_argument("--reexports", action="store_true")
    ap.add_argument("--out", default=".")
    ap.add_argument("endpoints", nargs="+")
    args = ap.parse_args()
    skip = set(Path(args.skip).read_text().split()) if args.skip else set()
    inert = set(Path(args.inert).read_text().split()) if args.inert else set()
    tests = {p: ast.parse(p.read_bytes(), filename=str(p)) for p in test_modules(REPO)}
    apps = {p: ast.parse(p.read_bytes(), filename=str(p)) for p in sorted((REPO / "tldw_Server_API" / "app").rglob("*.py"))
            if p not in tests and b"endpoints" in p.read_bytes()}
    sites, home_patchers, computed, stores = defaultdict(list), defaultdict(set), defaultdict(set), set()
    for p, tree in tests.items():
        top_of = {id(x): top for top in tree.body for x in ast.walk(top)}
        for s in patch_sites(tree):
            if not s.receiver:
                continue
            if is_ep(s.receiver) and module_file(s.receiver) and dotted_of(module_file(s.receiver)) == s.receiver:
                if s.verb == "assign":
                    stores.add((s.receiver, s.attr))
                elif s.attr is None:
                    computed[s.receiver] |= {n.value for n in ast.walk(top_of[id(s.node)]) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                else:
                    sites[(s.receiver, s.attr)].append((p, s))
            elif not is_ep(s.receiver) and s.attr:
                home_patchers[(canonical(s.receiver), s.attr)].add(p)
    referrers, call_time = defaultdict(set), set()
    for p, tree in {**tests, **apps}.items():
        resolver, inside = ScopeResolver(tree), inside_functions(tree)
        patched = {id(s.node.args[0]) for s in patch_sites(tree) if isinstance(s.node, ast.Call)}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    pkg = dotted_of(p).split(".")[: None if p.name == "__init__.py" else -1]
                    base = ".".join(pkg[: len(pkg) - n.level + 1] + ([n.module] if n.module else []))
                for a in n.names if is_ep(base) else []:
                    referrers[(base, a.name)].add(p)
                    if p in apps and id(n) in inside:
                        call_time.add((base, a.name))
            elif isinstance(n, ast.Attribute) and id(n.value) not in patched and isinstance(n.ctx, ast.Load):
                d = resolver.dotted(n.value)
                if d and is_ep(d):
                    referrers[(d, n.attr)].add(p)
                    if p in apps and id(n) in inside:
                        call_time.add((d, n.attr))
    named = {p: named_modules(tree) for p, tree in tests.items()}
    report, totals, test_edits, by_test, compare, runset = [], defaultdict(int), defaultdict(list), defaultdict(list), set(), set()
    for ep_path in [Path(e).resolve() for e in args.endpoints]:
        ep, text = dotted_of(ep_path), ep_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        runset |= {str(p.relative_to(REPO)) for p, names in named.items() if names_module(names, ep)}
        imports, loads, top_loads, bound, declared_global, module_names = analyse(tree)
        siblings = [ep_path] if ep_path.parent.name == "endpoints" else sorted(ep_path.parent.glob("*.py"))
        by_string = set().union(*(identifier_strings(ast.parse(s.read_bytes())) for s in siblings))
        taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(imports) | set(module_names.values())
        app_edits, new_imports, drops = [], {}, defaultdict(set)
        for (d, alias), plist in sorted(sites.items(), key=lambda kv: kv[0]):
            if d != ep or alias not in imports:
                continue
            m, orig, node = imports[alias]
            m = canonical(m) if module_file(m) else m
            reason = next((why for hit, why in (
                (f"{ep}.{alias}" in skip, "listed in --skip"),
                (alias == "logger", "logger"),
                (not module_file(m), "source is not a module"),
                (is_ep(m), "source is an endpoint module"),
                (alias in declared_global, "declared global"),
                (alias in bound, "bound in a function body"),
                (not loads.get(alias) and not (args.reexports and ep_path.name == "__init__.py"), "no function-body load"),
                ((ep, alias) in stores, "a module assigns or deletes the attribute directly"),
                (alias in computed.get(ep, set()), "a test patches this module with a computed name"),
                (alias in by_string, "app code looks the name up by string"),
                ((ep, alias) in call_time, "app referrer reads it at call time"),
            ) if hit), None)
            if reason:
                report.append(f"SKIP {ep}.{alias} ({len(plist)}): {reason}")
                totals["skipped"] += len(plist)
                continue
            binding = module_names.get(m) or new_imports.get(m)
            if binding is None:
                binding = "_" + m.rsplit(".", 1)[1]
                while binding in taken:
                    binding = "_" + binding
                taken.add(binding)
                new_imports[m] = binding
            app_edits += [(x.lineno, x.col_offset, x.end_lineno, x.end_col_offset, f"{binding}.{orig}") for x in loads[alias]]
            others = referrers.get((ep, alias), set()) - {ep_path}
            if top_loads.get(alias) or others:
                report.append(f"KEEP {ep}.{alias}: {top_loads.get(alias, 0)} module-level loads, referrers {sorted(str(o.relative_to(REPO)) for o in others)[:3]}")
            else:
                drops[node].add(alias)
            for p, s in plist:
                rel = str(p.relative_to(REPO))
                if f"{rel}::{ep}.{alias}" in inert:
                    block = next((body for owner in ast.walk(tests[p]) for f in ("body", "orelse", "finalbody")
                                  if isinstance(body := getattr(owner, f, None), list)
                                  and any(isinstance(st, ast.Expr) and st.value is s.node for st in body)), None)
                    stmt = next((st for st in block or [] if isinstance(st, ast.Expr) and st.value is s.node), None)
                    if stmt is not None and s.verb in {"setattr", "delattr"}:
                        # The only statement of its block becomes pass; otherwise its lines go.
                        rep = "pass" if len(block) == 1 else "#t1320-delete-line"
                        test_edits[p].append((stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset, rep))
                        totals["inert_deleted"] += 1
                    else:
                        totals["inert_kept"] += 1
                    report.append(f"INERT {rel}:{s.node.lineno} {ep}.{alias}")
                    continue
                a0 = s.node.args[0]
                end = a0 if s.form == "str" else s.node.args[1]
                test_edits[p].append((a0.lineno, a0.col_offset, end.end_lineno, end.end_col_offset, f'"{m}.{orig}"'))
                if s.verb == "patch.object":
                    f = s.node.func
                    test_edits[p].append((f.lineno, f.col_offset, f.end_lineno, f.end_col_offset, ast.unparse(f.value)))
                totals[f"{s.verb}:{s.form}"] += 1
                by_test[rel].append(f"{ep}.{alias}")
            compare |= {str(p.relative_to(REPO)) for p in home_patchers.get((m, orig), set())}
            totals["pairs"] += 1
            totals["app_loads"] += len(loads[alias])
            report.append(f"CONVERT {ep}.{alias} -> {m}.{orig} ({len(plist)} patches, {len(loads[alias])} loads)")
            if m.startswith(SENSITIVE):
                report.append(f"SENSITIVE {ep}.{alias} -> {m}.{orig}")
        if not app_edits and not drops:
            continue
        out = edit(text, app_edits)
        for node in sorted(drops, key=lambda n: -n.lineno):
            out = without(out, node, drops[node])
        if new_imports:
            lines = out.splitlines(keepends=True)
            last = max(n.end_lineno for n in ast.parse(out).body if isinstance(n, (ast.Import, ast.ImportFrom)))
            lines.insert(last, "".join(f"import {m} as {b}\n" for m, b in sorted(new_imports.items())))
            out = "".join(lines)
        ast.parse(out)
        ep_path.write_text(out, encoding="utf-8")
        totals["app_files"] += 1
    for p, edits in test_edits.items():
        out = edit(p.read_text(encoding="utf-8"), edits)
        out = "".join(line for line in out.splitlines(keepends=True) if not line.lstrip().startswith("#t1320-delete-line"))
        ast.parse(out)
        p.write_text(out, encoding="utf-8")
    totals["test_files"] = len(test_edits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "codemod_report.txt").write_text("\n".join(report) + "\n")
    (out_dir / "pairs_by_test.json").write_text(json.dumps(by_test, indent=1, sort_keys=True) + "\n")
    (out_dir / "compare_files.txt").write_text("\n".join(sorted(set(by_test) | compare)) + "\n")
    (out_dir / "runset.txt").write_text("\n".join(sorted(runset | set(by_test))) + "\n")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
```

## Appendix B: `t1320_compare.py`

Save as `"$T1320_DIR/t1320_compare.py"`. Validated on three files with `-n 3` in 1 min 17 s: two came back SAME, and one DIFFERS was caused deliberately by the audio upper-bound simulation, which converts without removing the shims.

```python
"""TASK-13950.20 outcome comparison, base against head, per test file (not committed).

Usage (head worktree root): BASE_WT=<base worktree> python t1320_compare.py LIST [-n WORKERS]
LIST holds test paths. ``conftest.py`` is replaced by its directory; another non-``test_`` module by
the test files that import it; a ``tests/_plugins`` module is dropped and reported (C7 covers it).
One pytest run per side over the whole list, then one solo re-run per side for each file that
differs. Prints, per file: SAME(<outcomes>), VACUOUS (the base passed nothing, so equality proves
nothing), FLAKY (differed together, same alone) or DIFFERS. Provider keys are unset; Docker
Postgres is not started (C6 covers Postgres); no --basetemp, since a basetemp outside pytest's
default root fails CharactersRAGDB's approved-root check.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PY = os.environ.get("PY", sys.executable)
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "COHERE_API_KEY", "HUGGINGFACE_API_KEY")
LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (\S+?\.py)(.*)$")


def expand(paths: list[str]) -> list[str]:
    tests = [str(p) for p in Path("tldw_Server_API").rglob("test_*.py")]
    out = []
    for p in paths:
        name = Path(p).name
        if "/_plugins/" in p:
            print(f"PLUGIN (C7 only): {p}")
        elif name == "conftest.py":
            out.append(str(Path(p).parent))
        elif not name.startswith("test_"):
            dotted = ".".join(Path(p).with_suffix("").parts)
            out += [t for t in tests if dotted in Path(t).read_text(encoding="utf-8", errors="ignore")]
        else:
            out.append(p)
    return sorted(set(out))


def run(cwd: str, files: list[str], workers: int) -> dict[str, set[str]]:
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(TLDW_TEST_NO_DOCKER="1", PYTHONPATH=".", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TEST_MODE="true",
               DISABLE_HEAVY_STARTUP="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-p", "pytest_asyncio.plugin",
           "-p", "pytest_timeout", "--timeout=300"]
    if workers > 1:
        cmd += ["-p", "xdist.plugin", "-n", str(workers)]
    proc = subprocess.run([*cmd, *files], cwd=cwd, env=env, capture_output=True, text=True)
    by_file: dict[str, set[str]] = defaultdict(set)
    for line in proc.stdout.splitlines():
        if m := LINE.match(line.split(" - ")[0]):
            by_file[m.group(2)].add(f"{m.group(1)} {m.group(2)}{m.group(3)}")
    return by_file


def main() -> None:
    files = expand([line.strip() for line in open(sys.argv[1], encoding="utf-8") if line.strip()])
    workers = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 1
    base, head = run(os.environ["BASE_WT"], files, workers), run(".", files, workers)
    for f in files:
        keys = [k for k in set(base) | set(head) if k == f or k.startswith(f.rstrip("/") + "/")]
        b = set().union(*(base.get(k, set()) for k in keys))
        h = set().union(*(head.get(k, set()) for k in keys))
        if b != h:
            b1, h1 = run(os.environ["BASE_WT"], [f], 1), run(".", [f], 1)
            same_alone = set().union(*b1.values()) == set().union(*h1.values())
            print(f"{'FLAKY' if same_alone else 'DIFFERS'}: {f}")
            for line in sorted(b ^ h)[:8]:
                print(f"    {'-' if line in b else '+'} {line}")
        elif not any(o.startswith("PASSED") for o in b):
            print(f"VACUOUS: {f}")
        else:
            print(f"SAME({len(b)}): {f}")


if __name__ == "__main__":
    main()
```

## Appendix C: `t1320_tripwire.py` (the review's tripwire, with a count mode)

Save as `"$T1320_DIR/plugin/t1320_tripwire.py"`, and load it with `PYTHONPATH=.:"$T1320_DIR/plugin"` and `-p t1320_tripwire`.

- **Validated** on 7a020da14a over `tests/Agent_Client_Protocol/test_async_api.py` and `tests/External_Sources/test_connectors_endpoints_api.py`.
- **Dead mode.** On the converted tree, 7 of 11 `test_async_api.py` tests fail on the two helper-returned sites.
- **Count mode.** It gives the same 20 site keys on both sides. The two dead sites count 0 calls on the base: they are inert.
- **Site attribution.** It skips frames under `tests/_plugins/`, because `http_client_patch_guard.py` wraps `monkeypatch.setattr`.
- **Outcome-aware (re-review N2).** A probe confirmed it:
  - a test that patches `audio.audio.finish_job` and then calls `pytest.skip` leaves its site `unproven`, and Appendix D `inert` does not list it;
  - a `MagicMock` reached only through an attribute call counts 1.

```python
"""TASK-13950.20 tripwire (pytest plugin, not committed): dead endpoint patches and per-site call counts.

Load with ``-p t1320_tripwire`` (``PYTHONPATH`` must reach this file). ``T1320_REPORT`` names a
``codemod_report.txt``; its CONVERT lines give the endpoint pairs ``(endpoint, alias)`` and their
homes ``(M, orig)``.

* ``T1320_MODE=dead`` (head): a test fails if it, or a fixture it uses, patches an endpoint pair
  that a CONVERT line names. After the conversion the endpoint reads the home, so such a patch,
  one the codemod could not see, intercepts nothing. ``T1320_ALLOW`` lists
  ``<test file>::<endpoint>.<alias>`` sites already proven inert on the base.
* ``T1320_MODE=count`` (base or head): a patch of a CONVERT pair (base) or of a CONVERT home (head)
  whose value is a function or a Mock is wrapped, and its calls are counted per
  ``<file that issued the patch>::<patched target>``. A Mock counts its ``mock_calls`` at teardown,
  so calls to its attributes count too. A key that any test which did not pass (failed, errored,
  skipped, xfailed) touched is also recorded as unproven: its zero proves nothing.
  ``T1320_COUNT_OUT`` gets ``{"calls": {...}, "unproven": [...]}`` per pytest process
  (``<path>.<xdist worker>.json``; Appendix D merges them). A value that is not a function or a
  Mock (a class, an object, a constant) is not counted, so its site is never shown inert.
The plugin unsets the commercial provider keys before the app loads, so a dead stub cannot reach a
real provider.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import re
import sys
import types
from collections import Counter
from unittest import mock

import pytest
from _pytest.monkeypatch import MonkeyPatch

for _name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY",
              "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "COHERE_API_KEY", "HUGGINGFACE_API_KEY"):
    os.environ.pop(_name, None)

_RX = re.compile(r"CONVERT (\S+)\.(\w+) -> (\S+)\.(\w+) ")
_PAIRS, _HOMES = {}, {}
for _m in map(_RX.match, open(os.environ["T1320_REPORT"], encoding="utf-8")):
    if _m:
        _PAIRS[(_m.group(1), _m.group(2))] = f"{_m.group(1)}.{_m.group(2)}"
        _HOMES[(_m.group(3), _m.group(4))] = f"{_m.group(3)}.{_m.group(4)}"
_MODE = os.environ.get("T1320_MODE", "dead")
_ALLOW = set(open(os.environ["T1320_ALLOW"], encoding="utf-8").read().split()) if os.environ.get("T1320_ALLOW") else set()
_HERE = os.path.abspath(__file__)
_counts: Counter[str] = Counter()
_unproven: set[str] = set()
_touched: set[str] = set()
_mocks: list[tuple[str, mock.NonCallableMock]] = []
_dead: list[str] = []
_item_file = [""]


def _site_file() -> str:
    """The test-tree file that issued the patch (``tests/_plugins`` wrappers skipped), else the running test's file."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if os.path.abspath(name) != _HERE and f"{os.sep}tests{os.sep}" in name and f"{os.sep}_plugins{os.sep}" not in name:
            return os.path.relpath(name)
        frame = frame.f_back
    return _item_file[0]


def _key(target, name):
    if isinstance(target, str):
        mod, _, name = target.rpartition(".")
    else:
        mod = target.__name__ if isinstance(target, types.ModuleType) else None
    return (mod, name) if mod else None


class _Counting:
    """Delegates to a Mock; its calls, its attributes' calls included, are counted at teardown."""

    def __init__(self, inner, key):
        object.__setattr__(self, "_t1320", (inner, key))

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "_t1320")[0](*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_t1320")[0], name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_t1320")[0], name, value)


def _counted(value, key):
    _touched.add(key)
    if inspect.iscoroutinefunction(value) and inspect.isfunction(value):
        @functools.wraps(value)
        async def wrapper(*args, **kwargs):
            _counts[key] += 1
            return await value(*args, **kwargs)
        return wrapper
    if inspect.isfunction(value) or inspect.ismethod(value) or isinstance(value, functools.partial):
        @functools.wraps(value)
        def wrapper(*args, **kwargs):
            _counts[key] += 1
            return value(*args, **kwargs)
        return wrapper
    if isinstance(value, mock.NonCallableMock) and callable(value):
        _mocks.append((key, value))
        return _Counting(value, key)
    _unproven.add(key)  # not a function or a Mock: its calls cannot be counted
    return value


def _check(target, name, value, has_value):
    key = _key(target, name)
    if key is None:
        return value
    site = _site_file()
    if _MODE == "dead" and key in _PAIRS and f"{site}::{_PAIRS[key]}" not in _ALLOW:
        _dead.append(f"{site}::{_PAIRS[key]}")
    if _MODE == "count" and has_value and (key in _PAIRS or key in _HOMES):
        return _counted(value, f"{site}::{_PAIRS.get(key) or _HOMES[key]}")
    return value


_mp_set, _mp_del, _p_enter = MonkeyPatch.setattr, MonkeyPatch.delattr, mock._patch.__enter__


def _set(self, target, name=None, value=None, *args, **kwargs):
    if isinstance(target, str):  # monkeypatch.setattr("a.b", value): the value arrives as ``name``
        name = _check(target, None, name, True)
        return _mp_set(self, target, name, *args, **kwargs) if value is None else _mp_set(self, target, name, value, *args, **kwargs)
    value = _check(target, name, value, True)
    return _mp_set(self, target, name, value, *args, **kwargs)


def _del(self, target, name=None, *args, **kwargs):
    _check(target, None if isinstance(target, str) else name, None, False)
    return _mp_del(self, target, name, *args, **kwargs) if name is not None else _mp_del(self, target, *args, **kwargs)


def _enter(self):
    result = _p_enter(self)
    holder = self.getter()
    current = getattr(holder, self.attribute, None)
    wrapped = _check(holder, self.attribute, current, True)
    if wrapped is not current:
        setattr(holder, self.attribute, wrapped)
    return result


MonkeyPatch.setattr, MonkeyPatch.delattr, mock._patch.__enter__ = _set, _del, _enter


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _item_file[0] = os.path.relpath(str(item.path))
    _dead.clear()  # fixture-time patches count too
    _touched.clear()
    _mocks.clear()
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    for key, inner in _mocks:  # before the fixtures' undo; the Mocks keep their records either way
        _counts[key] += len(inner.mock_calls)
    for key in _touched:
        _counts[key] += 0  # a wrapped site with no calls still shows up, as 0
    yield


def pytest_runtest_logreport(report):
    if report.outcome != "passed" or getattr(report, "wasxfail", None):
        _unproven.update(_touched)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield
    if _dead:
        pytest.fail(f"T1320 dead endpoint patch: {sorted(set(_dead))}")


def pytest_sessionfinish(session):
    if _MODE == "count" and os.environ.get("T1320_COUNT_OUT"):
        out = os.environ["T1320_COUNT_OUT"]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        with open(f"{out}.{worker}.json", "w", encoding="utf-8") as handle:
            json.dump({"calls": dict(sorted(_counts.items())), "unproven": sorted(_unproven)}, handle, indent=0)
```

## Appendix D: `t1320_counts.py`

Save as `"$T1320_DIR/t1320_counts.py"`.

```python
"""TASK-13950.20: read Appendix C's call counts (not committed).

python t1320_counts.py inert REPORT BASE_PREFIX
    Prints ``<site file>::<endpoint>.<alias>`` for every CONVERT pair site that Appendix C counted
    on the base with 0 calls and that no failing, erroring or skipped test touched: it provably
    intercepts nothing on dev. This is the codemod's --inert file and the tripwire's T1320_ALLOW file.
python t1320_counts.py broadened REPORT BASE_PREFIX HEAD_PREFIX
    Prints ``<site file> <home> base=<n> head=<m> skip: <pairs>`` for every SENSITIVE home whose head calls in a
    site file exceed the base calls there: the pairs' calls plus any direct patch of the home the
    file already made. Each listed pair is skipped.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict


def merged(prefix: str) -> tuple[Counter[str], set[str]]:
    calls: Counter[str] = Counter()
    unproven: set[str] = set()
    for path in glob.glob(f"{prefix}.*.json"):
        data = json.load(open(path, encoding="utf-8"))
        calls.update(data["calls"])
        unproven.update(data["unproven"])
    return calls, unproven


def main() -> None:
    mode, report = sys.argv[1], open(sys.argv[2], encoding="utf-8").read()
    pairs = dict(re.findall(r"^CONVERT (\S+) -> (\S+) \(", report, re.M))
    base, unproven = merged(sys.argv[3])
    if mode == "inert":
        for key, calls in sorted(base.items()):
            if calls == 0 and key not in unproven and key.split("::")[1] in pairs:
                print(key)
        return
    sensitive = set(re.findall(r"^SENSITIVE (\S+) -> ", report, re.M))
    head, by_home = merged(sys.argv[4])[0], defaultdict(lambda: [0, set()])
    for key, calls in base.items():
        site, pair = key.split("::")
        if pair in sensitive:
            by_home[(site, pairs[pair])][0] += calls
            by_home[(site, pairs[pair])][1].add(pair)
    for (site, home), (before, names) in sorted(by_home.items()):
        before += base.get(f"{site}::{home}", 0)  # the file's own direct patches of the home
        after = head.get(f"{site}::{home}", 0)
        if after > before:
            print(f"{site} {home} base={before} head={after} skip: {' '.join(sorted(names))}")


if __name__ == "__main__":
    main()
```

## Appendix E: `t1320_totals.py`

Save as `"$T1320_DIR/t1320_totals.py"`, and run it with `PYTHONPATH=. TEST_MODE=true`. On 7a020da14a, before Task 1, it prints `{"adapter_references": 30, "getters": 549, "globals": 445, "test_mode_lines": 581, "test_mode_predicates": 12}`. After Task 1, `getters` is 599.

```python
"""TASK-13950.20: totals a spelling change must not move (not committed). Run from the repo root."""
import importlib
import json

from tldw_Server_API.tests.lint import test_direct_llm_adapter_calls_ratchet as adapters
from tldw_Server_API.tests.lint import test_service_singleton_ratchet as singletons
from tldw_Server_API.tests.lint import test_test_mode_branch_ratchet as test_mode

modes = test_mode.measure().values()
totals = {
    "getters": sum(v["calls"] for v in singletons.measure_getter_calls().values()),
    "globals": sum(v["globals"] for v in singletons.measure_globals().values()),
    "test_mode_lines": sum(v["lines"] for v in modes),
    "test_mode_predicates": sum(v["predicates"] for v in modes),
    "adapter_references": sum(v["references"] for v in adapters.measure().values()),
}
try:  # TASK-13950.21's env-read ratchet, once it is on the branch
    env = importlib.import_module("tldw_Server_API.tests.lint.test_env_read_ratchet")
    totals["env_reads"] = sum(v["reads"] for v in env.measure().values())
except ModuleNotFoundError:
    pass
print(json.dumps(totals, sort_keys=True))
```

## Self-review (2026-11-08, after fix round 1)

- **Spec coverage:**
  - AC1 is Task 1 (counter, seed at execution, self-test, gate) with Task 9's exit check after E3.
  - AC2 is Tasks 2-3, with Ruling 5.
  - AC3 is Tasks 4-5, with Ruling 6 and Ruling 8's slot.
  - AC4 is Tasks 6-8, behind Ruling 4's owner gate and with the literal-AC4 fallback.
  - AC5 is Task 1 plus Task 2's wiring of the four guards it changes.
- **Every fix-round item has a place:**
  - C1: Ruling 7, Appendix C, Task 6 Steps 2-4, and Tasks 4, 5 and 8.
  - I1: the lexical `ScopeResolver`, shared by the counter and the codemod.
  - I2: Appendix A's call-time referrer skip.
  - I3: Task 6 Step 5.
  - I4: Ruling 10 and Task 6 Step 7.
  - I5: Ruling 9 and Task 2.
  - I6: Task 3 Files and Step 3.
  - I7: Ruling 7 and Appendix D `broadened`.
  - I8: Task 4's copy guard in the gated module.
  - I9: Ruling 10 and the PR slices.
  - I10: Ruling 4 and Task 6 Step 0.
  - Dry run C1: Ruling 8. Dry run C2: the reason convention.
  - Dry run I3-I8: Task 1 Step 4, the counter spec and the re-measured seed, Appendix E, and the corrected counts.
  - The Minors: VACUOUS and batch runs (Appendix B), `pyproject` and `path_classifier` (Task 3), literal reasons, AC2 and AC3 notes and the regex (Task 9), Task 5's statement accounting, `$T1320_DIR`, and the counter's known limits.
- **Types and names:**
  - Task 1 defines `PatchSite`, `ScopeResolver`, `patch_sites`, `test_modules`, `endpoint_module_key`, `measure_endpoint_patches`, `PATCHES`, `SEED`, `EXIT_TARGET` and `summary`. Appendix A uses them by those names.
  - `route_less_endpoint_modules` and `ROUTE_LESS_ALLOWED` are defined in Task 3. `moved_files` and `MOVED_OUT_OF_ENDPOINTS` are defined in Task 2 and extended in Tasks 3-5.
  - `ws_connections.publish`/`current`, `default_service.sandbox_service`, `model_metadata.get_model_metadata` and `core.Embeddings.media_embeddings` are produced once and consumed by name.
- **Fix round 2:**
  - N1: Task 4 now retargets `optimization_engine.py:1078` and the 7 test patches, and greps and runs `tests/prompt_studio`.
  - N2: Ruling 7 defines proof as passed with 0 calls; Appendix C records outcomes and Mock attribute calls, and Appendix D lists only proven sites.
  - N3 and review I5: each widened guard has a mutation test, and the two AuthNZ guardrails extract `_offenders(paths)`.
  - The grep-and-run rule is in Global Constraints and in Tasks 2-5 and 8.
  - B split into B1 (guards and pagination) and B2 (route-less moves), each about 240 hand-written lines.
  - Minors m1-m8: the endpoints-package key, `mkdir`, the `broadened` baseline (pair calls plus direct home patches), provider keys and timeout, the `runset` limit, split rules, the deletion sentinel and its note, and Ruling 8's out-of-process note.
- **Formatting:** every reason string in the plan was checked for `<` and `>`. The only angle brackets left are in prose and command placeholders.
- **Risk left open:**
  - The 55% exit rests on the dry run: 50.3% with Task 8's upper bound, 53.7% without it, against the 3,150 target. Dead-and-live skips lower it, and inert deletions raise it; E1's re-projection measures both.
  - Appendix C sees patches made through `MonkeyPatch.setattr`/`delattr` and `mock._patch`. A test that assigns an endpoint attribute directly is caught by Appendix A's store skip instead.
