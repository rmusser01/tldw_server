# Ratchet DB_Management to domain imports and move service logic out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the DB_Management imports of domain modules that load DB code at import (the cycle-closing edges), then cut the four named ones: Kanban vector indexing, roleplay-resume prompt formatting, the webhook lifecycle-reason rule and the Sync contracts.

**Architecture:** A new AST ratchet, `tests/lint/test_db_management_logic_imports.py`, reuses the `.9` layering scanner. It counts a DB_Management import as "DB-loading" when its target module loads `core.DB_Management` at import: module-level imports, followed transitively and through parent package `__init__`s. Any other target imports no DB code, which is AC4's own definition of a contract. Each slice then inverts one dependency and deletes its lines from both baselines:

- inject from the caller: the Kanban vector-search factory and the resume message builder;
- move down or out: the Sync contracts into a new `core.Sync_Contracts` package, and the reason rule into `Admin_Webhooks/domain.py`.

A re-export shim stays wherever importers remain.

**Tech Stack:** Python 3.12/3.13, pytest, AST lint tests under `tldw_Server_API/tests/lint/` sharing `_ast_cache.py`, SQLite and PostgreSQL (`TLDW_TEST_POSTGRES_REQUIRED=1`).

**Spec:**
- `backlog/tasks/task-13950.16 - Ratchet-DB-Management-to-domain-imports-and-move-service-logic-out.md`, AC #1-#6, with AC5 as amended on 2026-10-26.
- `backlog/tasks/task-13950.67 - Cut-the-remaining-DB-Management-imports-of-DB-loading-domain-modules.md`, which owns what this task leaves.
- The `.16` row of `## Phase 1` in `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`: effort M, with slices kanban, resume store, webhooks and Sync contracts.

**Conventions:**
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`.
- Run every command from the repo root.
- pytest runs as `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-<name>"`.
- File:line references are on `origin/dev` 35e711080a merged with #3337 (9bff409144). Re-locate them by the quoted text.

## Measured on 2026-10-26 (scratch merge of dev 35e711080a + #3337)

- **DB_Management size.** 86 top-level modules, 38 `*_DB.py` (backlog: same).
- **Layering baseline.** `layering_baseline.txt` has 33 `core.DB_Management -> *` pairs and 177 statements; 31 of the pairs target `core.*` packages. The backlog said 24 domain packages because it counted module-level imports only.
- **Logic vs contract split.** Of the 177 statements, 45 (37 importer→target module pairs) target a DB-loading module and 132 do not.
  - The independent dry run imported each of the 99 distinct targets in a fresh interpreter and checked `sys.modules`. It found 0 mismatches against the static definition.
- **Evidence drift.** `_roleplay_resume_messages` is now `chacha/conversation_resume_store.py:587-671` (backlog: 596-650). The other evidence lines still match:
  - `kanban_vector_search.py:27-28`
  - `Kanban_DB.py:40`
  - `admin_webhooks_repository.py:26-28`
  - `Sync_DB.py:21-105`
  - `core/Sync/v2/store.py:12`
  - `Evaluations_DB.py:23,29`
- **Size baselines.** #3337 carries a copy of #3334's baselines from before #3334 was re-seeded. That copy fails on this tree, while #3334's own head (855f25f984) already re-seeds and passes on dev. #3337 takes #3334's side when it rebases. This plan re-seeds nothing; it relies on #3334's keys, not its counts.
- **#2855.** OPEN, last push 2026-09-02, its merge base 4,757 commits behind dev. It keeps `registration_work_lifecycle_reason` in `delivery.py` and adds no import to `admin_webhooks_repository.py`.

## Rulings (2026-10-26, binding)

1. **AC5 is amended and TASK-13950.67 exists.** The new AC5 reads:

   > DB_Management imports of modules that load DB code at import are frozen by a module-keyed ratchet that fails on growth and on stale entries; the four named cuts (Kanban vector search, resume store formatting, the webhook lifecycle reason, Sync contracts) leave it, and the remaining entries are tracked by TASK-13950.67

   - `.67` depends on `.16` and owns the 30 pairs (37 statements) left after the four cuts.
   - Task 6 checks this wording. It does not rewrite AC5 or create a task.
2. **PR D does not wait for #2855.** `delivery.py` keeps re-exporting the moved function, so #2855's imports keep working. The porting note for #2855 is in Task 5.
   - The roadmap's #2855 conflict row and `.16` Phase 1 row were updated on the docs branch on 2026-10-26 to match ruling 2.
3. **PR C proves the Sync move changes nothing by comparing generator outputs.**
   - The contract generator runs before and after the move, and `cmp` must find the two outputs identical.
   - The committed `core/Sync/v2/contracts/personal-context-ongoing-v1.*.json` are already stale against dev. The committed schema is `sha256:fb64798e…` and a fresh export is `sha256:8292b30c…`; the difference is `notes.provenance` in the domain enum. No test compares them.
   - PR C does not regenerate them, because that would change a published v1 contract. A separate follow-up owns that.

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

- **Order.** Work starts after #3337 merges. #3337 is stacked on #3334, so #3334 merges first. `.16`'s only frontmatter dependency is `.9`.
- **Baseline conflicts.** On a conflict or stale failure in `layering_baseline.txt` after merging dev, take dev's side, run `--tighten`, and re-add your own `# bump:` lines (`.9`'s rule). The same applies to `db_management_logic_imports.txt`.
- **TASK-13950.15 owns `core.DB_Management -> api.v1.schemas`.**
  - `.15` AC2 removes every `core -> api.v1.schemas` line.
  - Its S1 slice completes AC3's Evaluations half ("Evaluations_DB schema imports move to core"), so `.16` does not move those two statements.
  - Task 6 waits for S1; see Task 6.
- **Kanban default flip.** `KanbanDB(...)` without a factory no longer indexes vectors (Task 2), so the flipping-a-default rule applies. Task 2 Step 4 runs every hit of its `git grep`.
- **Frozen function.** `messaging:run_kanban_adapter` is frozen at 483 lines in `function_size_baseline.json`.
  - Its edited `KanbanDB(...)` call stays on one line (107 characters; the limit is 120).
  - `create_kanban_vector_search` is imported at module level, never inside the function.
- **No legacy baseline regeneration.** None of this task's files is in `scope_predicate_baseline.txt`. Check: `grep -nE "Kanban_DB|kanban_vector_search|kanban_deps|database_retrievers|messaging|kanban_search|conversation_resume_store|Sync_DB|admin_webhooks_repository|Admin_Webhooks|core/Sync/" Helper_Scripts/ci/scope_predicate_baseline.txt` prints nothing.
- **CharactersRAGDB surface.** No slice moves a method out of CharactersRAGDB or renames a delegation, so `methods`, `delegated_methods` and `private_member_reaches` cannot rise.
  - The six `conversation_resume_store` delegations at `ChaChaNotes_DB.py:47472-47483` keep their names: `append_idempotent_user_message`, `prepare_roleplay_resume_generation`, `commit_roleplay_resume_generation`, `put_behavior_snapshot`, `get_conversation_behavior_snapshot` and `get_roleplay_resume_state`.
  - If a slice lowers one count, run `python -m tldw_Server_API.tests.lint.test_charactersragdb_surface_ratchet --tighten` in that PR to lock the gain in.
- **Function sizes.** No function listed in `function_size_baseline.json` moves, so no key is re-keyed.
  - The four listed functions in touched modules stay where they are and must not grow: `Kanban_DB:KanbanDB._get_schema_sql`, `Evaluations_DB:EvaluationsDatabase._initialize_database`, `delivery:AdminWebhookDeliveryService.redeliver_webhook` and `messaging:run_kanban_adapter`.
  - A function that does move must carry the reason "moved from <old key> (<PR>)".
- **Other path-keyed ratchets.** The only one that lists a moved file is `.5`'s keyword ratchet; see Task 4 and the hazards below.
  - Checked on #3328, #3331, #3333, #3335, #3337 and dev, against `kanban_vector_search` and the 13 Sync modules.
  - Dev's `test_core_to_api_import_boundary.py:130` path list disappears with #3337, which makes that test pair-based.
- **Test-quality ratchet.** New tests must not add test-quality offenses. The tenant-isolation step runs `tests/CI/test_test_quality_ratchet.py`, which runs `Helper_Scripts/ci/test_quality_triage.py --enforce` and flags `stub_injection`, `mock_density`, `tautology_suspect` and `status_only`.

## Conflict hazards (open PRs on 2026-10-26)

| PR | Overlap | What to do |
|---|---|---|
| #3328, #3339, #3342 (`.5` keyword stack) | `tests/lint/test_keyword_identity_ratchet.py` keys `"tldw_Server_API/app/core/Sync/v2/notes_task_contract.py": 2` by path. They also shift lines next to PR C's import edits in ChaChaNotes_DB, keyword_store and organization_sync_store; none edits a `Sync.v2` import line. | PR C re-keys the entry in either merge order (Task 4 Step 7) |
| #3331 (`.6`) | `database_retrievers.py` import block (`:10`, `:52`) | PR A adds an import there; trivial rebase |
| #3269 (open since 2026-10-21) | `kanban_deps.py` (`:17`, `:263`), `conversation_resume_store.py` (`:37`, `:1125`) | Hunks next to PR A's import and PR B's `TYPE_CHECKING` addition; trivial rebase |
| #3226 (draft), #2843 (stale) | `core/Sync/v2/models.py`, and #3226 also `errors.py` | The old path stays a file (the alias), so git sees a modification, not a rename. Their hunks must be ported by hand to `core/Sync_Contracts/`. PR C's description says so and suggests `git diff -B -M --color-moved` for review |
| #3328, #3339, #3342, #3345, #3334, #3337 | `backend-required.yml` and `test_license_first_workflow_contracts.py` | PR A's two-line wiring change conflicts textually; trivial |
| #2855 | `Admin_Webhooks/delivery.py` | Not blocking (ruling 2); Task 5 gives the porting note |

## CI-equivalent checks (each task names the ones it needs)

| Id | What | Command | Pass means |
|---|---|---|---|
| C1 | backend-required "Enforce CI contracts and code ratchets" | `BACKLOG_TASK_FORMAT_BASE=origin/dev PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-c1" -p pytest_asyncio.plugin tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py tldw_Server_API/tests/lint/test_function_size_ratchet.py tldw_Server_API/tests/lint/test_charactersragdb_surface_ratchet.py tldw_Server_API/tests/lint/test_layering_contract.py tldw_Server_API/tests/lint/test_db_management_logic_imports.py tldw_Server_API/tests/lint/test_core_to_api_import_boundary.py`, plus `tldw_Server_API/tests/lint/test_keyword_identity_ratchet.py` when the base has it | all pass, with no baseline edit beyond the ones the task names |
| C2 | backend-required "Enforce tenant isolation ratchets" | same env, `tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/lint/test_route_auth_ratchet.py tldw_Server_API/tests/lint/test_rg_route_map_lint.py tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` | all pass |
| C3 | mypy on changed modules (the CI step is non-blocking; mypy is not in the venv) | `uvx mypy --python-executable "$PY" --follow-imports=silent --ignore-missing-imports $(git diff --name-only --diff-filter=d origin/dev -- 'tldw_Server_API/*.py')` | no error reported on a line this PR changed |
| C4 | OpenAPI drift gate | `PYTHONPATH=. "$PY" Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | exit 0 |
| C5 | Postgres shard | `TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. TEST_MODE=true "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$TMPDIR/t1316-pg" <files>`, with `POSTGRES_TEST_*` set as in ci.yml's Postgres shard, or Docker auto-start (`tests/_plugins/postgres.py`) | pass, and `-rs` lists no skipped `[postgres]` case |

## Review Focus

1. **A Kanban write or search path silently stops indexing.** This happens if an app opener constructs `KanbanDB` without the factory, including the retriever's reopen after `close()`. Pinned in Task 2 Step 1 by `test_production_openers_wire_the_kanban_vector_index` (four openers, reopen included) and the Workflows adapter assertion.
2. **Resume append admits an input whose prompt cannot be built, or inserts it before the check.** Pinned in Task 3 Step 1 by `test_append_runs_the_injected_builder_before_inserting` and `test_append_without_a_builder_is_a_type_error`.
3. **The history byte gate and the prompt use different limits, so an oversize history passes the gate and the full payload is read.** Pinned in Task 3 Step 1 by the retargeted `test_resume_history_byte_gate_precedes_full_payload_read` and by `test_the_prompt_byte_limit_is_one_constant`.
4. **A moved Sync module loses its identity.** If the old path is a second module object, `except`, `isinstance` and monkeypatches on one path miss objects from the other. Pinned in Task 4 Step 1 by:
   - `test_the_old_sync_v2_path_is_the_moved_module` (all 13 modules);
   - `test_blob_store_raises_the_contract_error_class`;
   - C4 and the before/after `cmp` of the contract export (Task 4 Steps 3 and 5).
5. **The new ratchet passes for the wrong reason.** It might miss a parent `__init__` that loads DB code, or count a target whose DB import is only inside a function. Pinned in Task 1 Step 1 by `test_a_db_management_import_of_a_db_loading_module_is_counted`.

## Where each `core.DB_Management -> *` layering entry goes

"DB-loading" means the target module loads `core.DB_Management` at import (Task 1). "`.67`" means the entry stays in `db_management_logic_imports.txt` after this task, owned by TASK-13950.67.

| Baseline line (now) | Importing modules (statements) | Disposition |
|---|---|---|
| `-> api.v1.schemas: 3` | Evaluations_DB 2, ChaChaNotes_DB 1 (`quizzes`) | TASK-13950.15. Its AC2 covers all three; its S1 is `.16` AC3's Evaluations half. Not a `.16` slice |
| `-> core.Admin_Webhooks: 3` | admin_webhooks_repository 3 (`crypto`, `delivery`, `domain`) | PR D: 3 → 2; `crypto` and `domain` import no DB code |
| `-> core.Agent_Client_Protocol: 1` | ACP_Audit_DB (`config`, DB-loading) | `.67` |
| `-> core.Agent_Orchestration: 1` | Orchestration_DB (`models`) | stays (contract) |
| `-> core.AuthNZ: 24` | Users_DB 7, UserDatabase_v2 5, storage_quota_reservation_store 3, admin_webhooks_repository 2, seven modules 1 each | `.67` for `AuthNZ.database` ×6 and `AuthNZ.repos.generated_files_repo` ×2; the other 16 stay |
| `-> core.Character_Chat: 12` | conversation_resume_store 11, Guardian_DB 1 | PR B: 12 → 11, all contracts |
| `-> core.Chat: 25` | message_store 10, native_fork_store 6, native_chat_asset_store 3, conversation_resume_store 2, four modules 1 each | PR B: 25 → 24; `.67` for `native_fork_projection` ×4 and `history_context` ×1 |
| `-> core.CodeGraph: 1`, `-> core.Collections: 1`, `-> core.Explainer: 1`, `-> core.VN_Assets: 1`, `-> core.Writing: 1`, `-> core.claims_analytics_export_contract: 1` | one module each | stay (contracts) |
| `-> core.Embeddings: 2` | kanban_vector_search | PR A: line deleted |
| `-> core.Evaluations: 1` | Evaluations_DB (`embeddings_abtest_repository`, DB-loading) | `.67`. `.15` S1 raises this pair (see Coordination) |
| `-> core.Flashcards: 4`, `-> core.Workspaces: 6` | ChaChaNotes_DB | stay; pure modules, `.22` territory |
| `-> core.Ingestion_Media_Processing: 9` | eight media_db modules and `backends.sqlite_backend` (`logging_safety` ×8, email metrics) | stay |
| `-> core.Jobs: 4` | jobs_failed_requeue 3, kanban_vector_search 1 | PR A: 4 → 3; `.67` for `operations.{postgres,sqlite}.admission` ×2 |
| `-> core.Monitoring: 1`, `-> core.Visual_Identities: 1` | media_repository; VisualIdentity_DB | `.67` |
| `-> core.Notes: 3`, `-> core.Notes_Graph: 3`, `-> core.Persona: 2`, `-> core.RPG: 6`, `-> core.Watchlists: 2`, `-> core.Web_Scraping: 1` | chacha stores, RPG_DB, Collections/Watchlists_DB, dedupe_urls | stay |
| `-> core.Personalization: 6` | Personal_Context_Key_Store 2, Personal_Context_Repository 3, persona_state_store 1 | `.67` (all DB-loading) |
| `-> core.Prompt_Management: 7` | prompts_db_helpers 4, three modules 1 each | `.67` for `prompt_studio.monitoring` ×1; the other 6 stay |
| `-> core.RAG: 6` | six media_db modules | `.67` (`agentic_chunker` ×5, `synonyms_registry` ×1) |
| `-> core.Sharing: 5` | clone_snapshot_repository 4, ChaChaNotes_DB 1 | `.67` (`clone_models` is DB-loading) |
| `-> core.Sync: 30` | 12 modules (Task 4 table) | PR C: line deleted; `core.DB_Management -> core.Sync_Contracts: 29  # bump:` added |
| `-> services: 3` | ChaChaNotes_DB → `services.osce_practice` (not DB-loading) | stays for this ratchet; two layers up, Phase 3 `.22` |

The `.67` rows total 37 statements: the 45 DB-loading statements, less the 8 this task cuts.

## Coordination with TASK-13950.15

- **S1 (Evaluations).** When `.15` S1 moves `evaluation_recipe_schemas` and `evaluation_schemas_unified.RunStatus` into `core.Evaluations`:
  - `core.DB_Management -> core.Evaluations` rises from 1, and S1's PR carries the `# bump:`;
  - the new modules must import no DB code, or `test_db_management_logic_imports.py` fails. `.15`'s plan (line 90) says they import only `enum` and `pydantic`.
- **S9 (Sync contract exporter), either order.** S9 moves `export_personal_context_ongoing_contract`, together with its function-level `api.v1.schemas.sync_v2_models` import, up into `sync_v2_models`.
  - If S9 merges first, PR C finds no `core.Sync -> api.v1.schemas` line and adds no `core.Sync_Contracts -> api.v1.schemas` line (Task 4 Step 6).
  - If PR C merges first, S9 edits `core/Sync_Contracts/personal_context_ongoing_contract.py` and deletes `core.Sync_Contracts -> api.v1.schemas: 1`, as `.15`'s plan already says (its line 595).

## PR slices

| PR | Tasks | Base | Hand-written lines | Layering baseline | `db_management_logic_imports.txt` |
|---|---|---|---|---|---|
| A | 1, 2 | dev after #3334 and #3337 | ~370, measured by the dry run | delete `core.DB_Management -> core.Embeddings: 2`; `core.DB_Management -> core.Jobs: 4` → `3` | seeded at 37 lines / 45 statements; the 3 `kanban_vector_search` lines then deleted → 34 / 42 |
| B | 3 | dev after A | ~190 | `-> core.Character_Chat: 12` → `11`; `-> core.Chat: 25` → `24` | 2 lines deleted → 32 / 40 |
| C | 4 | dev after A and the `.5` stack (#3328, #3339, #3342) by default; Task 4 covers the other order and `.15` S9 in either order | ~310, plus 13 renames | delete `-> core.Sync: 30`; add `core.DB_Management -> core.Sync_Contracts: 29  # bump:`. Unless S9 merged first, also swap `core.Sync -> api.v1.schemas: 1` for `core.Sync_Contracts -> api.v1.schemas: 1  # bump:` | 1 line deleted (`native_chat_asset_store -> core.Sync.v2.blob_store: 2`) → 31 / 38 |
| D | 5 | dev after A | ~60 | `-> core.Admin_Webhooks: 3` → `2` | 1 line deleted → 30 / 37 |
| (close-out) | 6 | dev after A-D and `.15` S1 | backlog only | none | none |

B, C and D each need only A's ratchet, so they may merge in any order. Each later one resolves the two baseline files by the `--tighten` rule. The counts above assume A, B, C, D order; the final 30 / 37 holds in any order.

---

### Task 1: The DB-loading import ratchet

**Files:**
- Modify: `tldw_Server_API/tests/lint/test_layering_contract.py`. Split `_targets` (:266-285), hoist `_scan`'s `names`/`known` (:293-307) into a function, and give `unreasoned_bumps` (:452-470) a `path` parameter.
- Create: `tldw_Server_API/tests/lint/test_db_management_logic_imports.py` and `tldw_Server_API/tests/lint/db_management_logic_imports.txt`.
- Modify: `.github/workflows/backend-required.yml`. Add the test file after `test_layering_contract.py \` (:300), and add one comment line after :278.
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add the path to the parametrize list of `test_backend_required_enforces_layering_ratchets` (:1409).

**Interfaces:**
- Produces in `test_layering_contract`, with behaviour unchanged:
  - `module_names(root: Path) -> tuple[Mapping[Path, str], frozenset[str]]`, decorated `@cache`. It returns the per-file app-relative module names and every module and package prefix, i.e. the `names`/`known` pair `_scan` builds today.
  - `target_modules(node: ast.Import | ast.ImportFrom, module: str, is_package: bool, known: frozenset[str]) -> set[str]`, the app-relative names `_targets` resolves today. `_targets` becomes `{package_of(name) for name in target_modules(...)}`.
  - `unreasoned_bumps(baseline, base, *, path: Path = BASELINE) -> list[str]`, whose messages name `path`.
- Produces in `test_db_management_logic_imports`:
  - `BASELINE: Path` (`db_management_logic_imports.txt`).
  - `COMMAND = "python -m tldw_Server_API.tests.lint.test_db_management_logic_imports"`.
  - `db_loading(root: Path = APP_ROOT) -> frozenset[str]`.
  - `measure(root: Path = APP_ROOT) -> dict[Pair, Counter[str]]`, keyed `(importer module, target module)`, with value `Counter({importer module: statements})`.
  - `violations(measured: Mapping[Pair, Counter[str]], baseline: Mapping[Pair, tuple[int, str]]) -> list[str]`.
  - Its own `dump(path: Path, entries: Mapping[Pair, tuple[int, str]]) -> None`, writing the header below; `.9`'s `dump` hard-codes its header.
  - `main(argv: list[str] | None = None) -> int`, taking `--tighten | --write-baseline`.
- Consumes:
  - From `test_layering_contract`: `APP`, `Pair`, `_walk`, `package_of`, `rule`, `module_names`, `target_modules`, `parse`, `load`, `rewritten`, `base_version`, `unreasoned_bumps`, `_statements`, `_shown`, `REASON_PLACEHOLDER`, `_seed`, `_counts` and `BASE_ENV`.
  - From `tldw_Server_API.tests.lint._ast_cache`: `parse_tree` and `APP_ROOT`.
  - Import only non-`test_` names, so pytest collects nothing twice.

- [ ] **Step 1: Write the self-tests** in the new `test_db_management_logic_imports.py`. At this step its header imports only `APP`, `_seed` and `_counts` from `test_layering_contract`; the other names come with Step 4.

```python
def _tree(tmp_path: Path) -> Path:
    return _seed(tmp_path / "app", {
        "core/DB_Management/store.py": (
            f"from {APP}.core.Foo import lazy, logic, models\n"
            f"def f():\n    from {APP}.core.Bar.leaf import x\n"
        ),
        "core/DB_Management/y.py": "",
        "core/Foo/__init__.py": "",
        "core/Foo/logic.py": f"from {APP}.core.DB_Management import y\n",
        "core/Foo/models.py": "X = 1\n",
        "core/Foo/lazy.py": f"def f():\n    from {APP}.core.DB_Management import y\n",
        "core/Bar/__init__.py": f"from {APP}.core.DB_Management import y\n",
        "core/Bar/leaf.py": "x = 1\n",
    })


def test_a_db_management_import_of_a_db_loading_module_is_counted(tmp_path: Path) -> None:
    assert _counts(measure(_tree(tmp_path))) == {
        ("core.DB_Management.store", "core.Foo.logic"): 1,  # loads DB code at import
        ("core.DB_Management.store", "core.Bar.leaf"): 1,  # its package __init__ does; a function-level importer counts
    }  # core.Foo.models (no DB import) and core.Foo.lazy (DB import only inside a function) are contracts


def test_new_and_stale_entries_fail_with_the_fix(tmp_path: Path) -> None:
    new = violations(measure(_tree(tmp_path)), {})
    assert [p.split(" [")[0] for p in new] == [
        "core.DB_Management.store -> core.Bar.leaf: 1 statement, not in the baseline",
        "core.DB_Management.store -> core.Foo.logic: 1 statement, not in the baseline",
    ]
    assert "core.DB_Management.store -> core.Foo.logic: 1  # bump: <why" in new[1]
    (stale,) = violations({}, {("core.DB_Management.a", "core.X.y"): (1, "")})
    assert stale.startswith("core.DB_Management.a -> core.X.y: stale, the edge is gone. Delete the line")
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t1" tldw_Server_API/tests/lint/test_db_management_logic_imports.py`.
  Expected: `2 failed`, `NameError: name 'measure' is not defined`.

- [ ] **Step 3: Refactor `test_layering_contract.py`.** Add `module_names`, `target_modules` and the `path` parameter of `unreasoned_bumps` (Interfaces).
  Run `tldw_Server_API/tests/lint/test_layering_contract.py` and `test_core_to_api_import_boundary.py`.
  Expected: the same pass count as before the edit (19 passed on the dry-run tree).

- [ ] **Step 4: Implement the new module** and add the Consumes imports.
  - **`db_loading(root)`** is a fixpoint over module-level edges.
    - Use `_walk` records with `function_level` False. Class bodies and try/except-ImportError bodies count; `if TYPE_CHECKING:` bodies do not.
    - Skip in-app `tests` directories, as `_scan` does.
    - Start from every `core.DB_Management*` module.
    - Add a module when any of these is already in the set: one of its module-level targets, a parent package of such a target, or a parent package of the module itself.
  - **`measure(root)`** looks at every statement, at any level, in a module whose `package_of` is `core.DB_Management`. It counts each target module `t` once when `rule("core.DB_Management", package_of(t))` is not None and `t in db_loading(root)`.
  - **`violations`** mirrors `.9`'s wording and semantics: the count must equal the code, so a fall is stale.
    - Growth: `"{a} -> {b}: {n} statement(s){' > baseline M' | ', not in the baseline'} [loads DB code at import]. {b} loads core.DB_Management when it is imported, so this import closes a cycle. Inject the logic from the caller, or import a module that imports no DB code. If it is unavoidable, set this line in {_shown(BASELINE)}; the raise and its reason are reviewed in the PR:\n    {a} -> {b}: {n}  # bump: {REASON_PLACEHOLDER}"`.
    - Stale: `"{a} -> {b}: stale, the edge is gone. Delete the line …"` or `"… Lower it to {n} …"`, then `" in {_shown(BASELINE)} to lock the cut in, or run: {COMMAND} --tighten."`.
  - **Baseline header:** `# <importer module> -> <target module>: <statements>. DB_Management imports of modules that load DB code at import. Counts only fall; see test_db_management_logic_imports.py.`
  - **`main`** reuses `rewritten()` and prints raised and new pairs, like `.9`'s `main`.
  - **The docstring** states:
    - the definition, in AC5's words;
    - the known limit: a target's function-level DB import is not followed, though the layering pairs still count it;
    - that `.67` empties the baseline;
    - the commands.

- [ ] **Step 5: Seed the baseline.**
  Run: `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_db_management_logic_imports --write-baseline`.
  Expected: these 37 lines (45 statements) on this tree; the dry run's output was byte-identical. Re-measure on the PR base: the script's output is the seed.

```text
core.DB_Management.ACP_Audit_DB -> core.Agent_Client_Protocol.config: 1
core.DB_Management.ChaChaNotes_DB -> core.Sharing.clone_models: 1
core.DB_Management.Evaluations_DB -> core.Evaluations.embeddings_abtest_repository: 1
core.DB_Management.Personal_Context_Key_Store -> core.Personalization.personal_context_crypto: 1
core.DB_Management.Personal_Context_Key_Store -> core.Personalization.personal_context_repository_models: 1
core.DB_Management.Personal_Context_Repository -> core.Personalization.personal_context_crypto: 1
core.DB_Management.Personal_Context_Repository -> core.Personalization.personal_context_publication: 1
core.DB_Management.Personal_Context_Repository -> core.Personalization.personal_context_repository_models: 1
core.DB_Management.Resource_Daily_Ledger -> core.AuthNZ.database: 1
core.DB_Management.Resource_Governance_Policy_Admin -> core.AuthNZ.database: 1
core.DB_Management.Users_DB -> core.AuthNZ.database: 1
core.DB_Management.VNPolicy_DB -> core.AuthNZ.database: 1
core.DB_Management.VisualIdentity_DB -> core.Visual_Identities.source_context: 1
core.DB_Management.admin_retention_preview_counts -> core.AuthNZ.database: 1
core.DB_Management.admin_webhooks_repository -> core.Admin_Webhooks.delivery: 1
core.DB_Management.admin_webhooks_repository -> core.AuthNZ.database: 1
core.DB_Management.chacha.conversation_resume_store -> core.Character_Chat.roleplay_resume_prompt: 1
core.DB_Management.chacha.conversation_resume_store -> core.Chat.chat_service: 1
core.DB_Management.chacha.message_store -> core.Chat.history_context: 1
core.DB_Management.chacha.native_chat_asset_store -> core.Sync.v2.blob_store: 2
core.DB_Management.chacha.native_fork_store -> core.Chat.native_fork_projection: 4
core.DB_Management.chacha.persona_state_store -> core.Personalization.companion_activity: 1
core.DB_Management.jobs_failed_requeue -> core.Jobs.operations.postgres.admission: 1
core.DB_Management.jobs_failed_requeue -> core.Jobs.operations.sqlite.admission: 1
core.DB_Management.kanban_vector_search -> core.Embeddings.ChromaDB_Library: 1
core.DB_Management.kanban_vector_search -> core.Embeddings.redis_pipeline: 1
core.DB_Management.kanban_vector_search -> core.Jobs.manager: 1
core.DB_Management.media_db.legacy_maintenance -> core.RAG.rag_service.agentic_chunker: 1
core.DB_Management.media_db.repositories.clone_snapshot_repository -> core.Sharing.clone_models: 4
core.DB_Management.media_db.repositories.media_repository -> core.Monitoring.topic_monitoring_service: 1
core.DB_Management.media_db.repositories.media_repository -> core.RAG.rag_service.agentic_chunker: 1
core.DB_Management.media_db.runtime.document_version_rollback_ops -> core.RAG.rag_service.agentic_chunker: 1
core.DB_Management.media_db.runtime.fts_ops -> core.RAG.rag_service.synonyms_registry: 1
core.DB_Management.media_db.runtime.media_lifecycle_ops -> core.RAG.rag_service.agentic_chunker: 1
core.DB_Management.media_db.runtime.synced_document_update_ops -> core.RAG.rag_service.agentic_chunker: 1
core.DB_Management.prompt_studio_db.repositories.jobs -> core.Prompt_Management.prompt_studio.monitoring: 1
core.DB_Management.storage_quota_reservation_store -> core.AuthNZ.repos.generated_files_repo: 2
```

- [ ] **Step 6: Add the real-tree tests and wire the gate.**
  - Tests:
    - `test_db_management_imports_of_db_loading_modules_only_fall()` asserts `violations(measure(), load(BASELINE)) == []`.
    - `test_raised_counts_carry_a_new_reason()` copies `.9`'s version, calling `unreasoned_bumps(load(BASELINE), old, path=BASELINE)`.
  - Workflow: append `tldw_Server_API/tests/lint/test_db_management_logic_imports.py \` after `:300`, and add the comment line `# The DB_Management logic-import ratchet (TASK-13950.16) does the same for its import counts.`
  - Contract test: add `"tldw_Server_API/tests/lint/test_db_management_logic_imports.py"` to the parametrize list at `:1409`.

- [ ] **Step 7: Run the new tests and the contract.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t1" tldw_Server_API/tests/lint/test_db_management_logic_imports.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py -k "logic_imports or layering_ratchets or db_loading or stale"`.
  Expected: PASS (7 passed in the dry run). Then run C1 and C3.

- [ ] **Step 8: Commit.**

```bash
git add tldw_Server_API/tests/lint/test_db_management_logic_imports.py tldw_Server_API/tests/lint/db_management_logic_imports.txt tldw_Server_API/tests/lint/test_layering_contract.py .github/workflows/backend-required.yml tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
git commit -m "test(lint): freeze DB_Management imports of modules that load DB code at import (TASK-13950.16)"
```

### Task 2: Kanban vector indexing moves to `core.Kanban` (AC1)

**Files:**
- Create: `tldw_Server_API/app/core/Kanban/__init__.py`, containing only the docstring `"""Kanban services above the Kanban persistence layer."""`.
- Move: `git mv tldw_Server_API/app/core/DB_Management/kanban_vector_search.py tldw_Server_API/app/core/Kanban/vector_search.py`. The body is unchanged; only the header comment and the module docstring's name change.
- Modify `tldw_Server_API/app/core/DB_Management/Kanban_DB.py`:
  - drop the import at :40;
  - change `__init__` at :271;
  - store the factory beside `self._vector_search = None` at :291;
  - make `_get_vector_search` (:3832-3855) call the factory, or return None when there is none.
- Modify the openers to pass `vector_search_factory=create_kanban_vector_search`. In each file, the factory is imported at module level, beside the existing `Kanban_DB` import:
  - `api/v1/API_Deps/kanban_deps.py:146` (`_create_kanban_db`); import beside `:24`.
  - `core/MCP_unified/modules/implementations/kanban_module.py:1012` (`KanbanModule._open_db`); import beside the relative `Kanban_DB` import at `:13`.
  - `core/RAG/rag_service/database_retrievers.py:3295` and `:3307` (`KanbanDBRetriever.__init__` and `_get_db`); import beside `:43`.
  - `core/Workflows/adapters/integration/messaging.py:226`, keeping the call on one line; import beside `:24`.
- Unchanged openers, which never sync vectors:
  - `endpoints/admin/admin_rbac.py:132` (FTS maintenance);
  - `services/kanban_purge_service.py:64`;
  - `services/kanban_activity_cleanup_service.py:63`;
  - `Kanban_DB.py:138` `_kanban_card_indexable` (read-only);
  - `Helper_Scripts/DB-Related/kanban_fts_maintenance.py` (FTS only).
- Modify: `api/v1/endpoints/kanban/kanban_search.py:55-56` to import from the new path.
- Modify: `pyproject.toml:1006`. The per-file-ignore key becomes `"tldw_Server_API/app/core/Kanban/vector_search.py"`, placed in sort order (after `Jobs/worker_sdk.py`).
- Modify: `Docs/User_Guides/WebUI_Extension/Kanban_Board_Guide.md:596` to give the new path. Run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published/User_Guides/WebUI_Extension/Kanban_Board_Guide.md` diff.
- Test:
  - Create `tldw_Server_API/tests/kanban/test_kanban_vector_wiring.py`.
  - Modify `tests/kanban/test_kanban_db.py:70-101`: build the DB after defining the fake factory and pass it to the constructor, instead of `monkeypatch.setattr(kanban_db_module, "create_kanban_vector_search", ...)`.
  - Modify `tests/kanban/test_kanban_vector_search.py:12,242,285,364` to import `from tldw_Server_API.app.core.Kanban import vector_search as kvs`.
  - Modify `tests/Workflows/adapters/test_integration_adapters.py:1250-1272`.
- Shim: none.
  - The old path has 1 app importer, 1 test module and 0 string patch targets, and this task retargets all of them.
  - A shim in DB_Management would itself be a new upward `core.DB_Management -> core.Kanban` pair.
  - Check: `git grep -n "DB_Management.kanban_vector_search\|DB_Management import kanban_vector_search" -- ':!Docs/Reviews' ':!Docs/superpowers' ':!Docs/Plans' ':!backlog'` prints nothing.

**Interfaces:**
- Produces:
  - `Kanban_DB.KanbanVectorSearchFactory = Callable[[str, dict[str, Any]], Any]`.
  - `KanbanDB.__init__(self, db_path: str, user_id: str, *, vector_search_factory: KanbanVectorSearchFactory | None = None) -> None`, stored as `self._vector_search_factory`.
  - `tldw_Server_API.app.core.Kanban.vector_search`, with an unchanged API that includes `create_kanban_vector_search(user_id: str, embedding_config: dict[str, Any] | None = None) -> KanbanVectorSearch | None` and `KanbanVectorSearch`.

- [ ] **Step 1: Write the failing tests** in `test_kanban_vector_wiring.py`.
  - Module imports: `os`, `subprocess`, `sys`, `Path` and `SimpleNamespace`; `KanbanDB` and `DatabasePaths` from DB_Management; `_create_kanban_db` from `api/v1/API_Deps/kanban_deps.py`; `KanbanDBRetriever`; `KanbanModule`; `ModuleConfig` (`core/MCP_unified/modules/base.py`); and `create_kanban_vector_search`.
  - Define `REPO_ROOT = Path(__file__).resolve().parents[3]`.

```python
def test_importing_kanban_db_loads_neither_jobs_nor_embeddings() -> None:
    code = (
        "import sys\nimport tldw_Server_API.app.core.DB_Management.Kanban_DB\n"
        "print(sorted(m for m in sys.modules if m.startswith(("
        "'tldw_Server_API.app.core.Jobs', 'tldw_Server_API.app.core.Embeddings'))))\n"
    )
    result = subprocess.run(  # nosec B603 - fixed argv
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "TEST_MODE": "true"},
    )
    assert result.stdout.strip().splitlines()[-1] == "[]"  # 24 modules on dev today


def test_a_kanban_db_without_a_factory_has_no_vector_search(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    db = KanbanDB(db_path=str(DatabasePaths.get_kanban_db_path(1)), user_id="1")
    try:
        assert db.get_vector_search() is None
    finally:
        db.close()


def test_production_openers_wire_the_kanban_vector_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    path = str(DatabasePaths.get_kanban_db_path(1))
    retriever = KanbanDBRetriever(path, user_id="1")
    from_init = retriever.kanban_db
    assert from_init is not None
    retriever.close()  # drops kanban_db; the next retrieve() reopens through _get_db (database_retrievers.py:3307)
    dbs: list[KanbanDB] = [
        _create_kanban_db(1),
        from_init,
        retriever._get_db(),
        KanbanModule(ModuleConfig(name="kanban"))._open_db(SimpleNamespace(db_paths={"kanban": path}, user_id="1")),
    ]
    try:
        assert [db._vector_search_factory for db in dbs] == [create_kanban_vector_search] * 4
    finally:
        for db in dbs:
            db.close()
```

In `test_kanban_adapter_board_list`, bind the patch (`as kanban_db_cls`) and add `assert kanban_db_cls.call_args.kwargs["vector_search_factory"] is create_kanban_vector_search`.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t2" tldw_Server_API/tests/kanban/test_kanban_vector_wiring.py`.
  Expected: a collection error, `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Kanban'`. The session stops there, so neither the subprocess test nor the adapter assertion runs.
  To see the 24-module red, run the subprocess snippet on its own.

- [ ] **Step 3: Implement the move and the injection** (Files above).

- [ ] **Step 4: Run every test hit of the default flip.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t2" $(git grep -l "KanbanDB\|kanban_vector_search\|create_kanban_vector_search\|kanban_deps\|KanbanModule\|KanbanDBRetriever\|run_kanban_adapter" -- tldw_Server_API/tests tldw_Server_API/app/core/MCP_unified/tests ':!*.json')`.
  - 24 files on dev. The dry run ran 515 tests in 13 min.
  - Expected: PASS.
  - Kanban_DB is SQLite-only (`grep -ci postgres tldw_Server_API/app/core/DB_Management/Kanban_DB.py` prints 0), so C5 does not apply.

- [ ] **Step 5: Lower the baselines.**
  - Run `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_layering_contract --tighten`. Expected diff: `core.DB_Management -> core.Embeddings: 2` deleted, and `core.DB_Management -> core.Jobs: 4` → `3`.
  - Run `PYTHONPATH=. "$PY" -m tldw_Server_API.tests.lint.test_db_management_logic_imports --tighten`. Expected diff: the three `core.DB_Management.kanban_vector_search -> …` lines deleted.
  - Nothing else changes in either baseline.

- [ ] **Step 6: Checks.**
  - Run C1, C2 and C3. C3 must be clean on `test_kanban_vector_wiring.py`, which is why that test types `dbs` and asserts on `from_init`.
  - `git grep -n "[^A-Za-z_]KanbanDB(" -- tldw_Server_API/app ':!tldw_Server_API/app/core/MCP_unified/tests' | wc -l` prints 9.
  - `git grep -n -A2 "[^A-Za-z_]KanbanDB(" -- tldw_Server_API/app ':!tldw_Server_API/app/core/MCP_unified/tests' | grep -c vector_search_factory` prints 5. The `kanban_deps` and retriever calls wrap onto a second line; only `messaging` fits on one.

- [ ] **Step 7: Commit.**

```bash
git add -A tldw_Server_API/app/core/Kanban tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api/v1 tldw_Server_API/app/core/MCP_unified/modules tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py tldw_Server_API/app/core/Workflows/adapters/integration/messaging.py tldw_Server_API/tests pyproject.toml Docs/User_Guides/WebUI_Extension/Kanban_Board_Guide.md Docs/Published/User_Guides/WebUI_Extension/Kanban_Board_Guide.md
git commit -m "refactor(kanban): move vector indexing to core.Kanban and inject it into KanbanDB (TASK-13950.16)"
```

### Task 3: The resume store returns rows; Character_Chat formats (AC2)

**Files:**
- Modify `tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py`:
  - move `MAX_ROLEPLAY_PROMPT_BYTES = 32 * 1024 * 1024` here from `roleplay_resume_prompt.py:49`;
  - add `RoleplayResumeHistory` and `RoleplayResumeMessageBuilder`.
- Modify `tldw_Server_API/app/core/Character_Chat/roleplay_resume_prompt.py`:
  - re-export `MAX_ROLEPLAY_PROMPT_BYTES` from the contract;
  - add `format_roleplay_resume_messages` and `prepare_roleplay_resume`.
- Modify `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`:
  - `_roleplay_resume_messages` (:587-671) becomes `_roleplay_resume_history`, which only reads the DB;
  - `append_idempotent_user_message` (:456; its call at :536-541);
  - `prepare_roleplay_resume_generation` (:861; its call at :955);
  - add the contract types to the `if TYPE_CHECKING:` block.
- Modify the endpoints:
  - `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6767-6772` (`_complete_roleplay_resume`). Add `prepare_roleplay_resume` to that function's existing function-level imports.
  - `tldw_Server_API/app/api/v1/endpoints/character_messages.py:375-381` (`append_roleplay_resume_input`). The function has no imports today; add one function-level import of `format_roleplay_resume_messages` at its top. This keeps the endpoint module's import cost unchanged.
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py`.
  - Add the module-level imports below.
  - The `append` helper (:53-54) and :163 pass `build_messages=format_roleplay_resume_messages`.
  - The 13 `db.prepare_roleplay_resume_generation(` calls become `prepare_roleplay_resume(db, ...)`. They are at :405, 468, 579, 680, 727, 744, 754, 777, 897, 935, 955, 1273 and 1545.
  - In `test_resume_history_byte_gate_precedes_full_payload_read`, the local import at :1703 and the patch at :1720 move from `roleplay_resume_prompt` to `roleplay_resume_contract`.
  - Add the new tests below.
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py:434` becomes `prepare_roleplay_resume(db, ...)`.

**Interfaces:**
- Produces in the contract module, which imports no DB code:
  - `class RoleplayResumeHistory(NamedTuple)` with fields `snapshot: Any`, `content: Any` and `authority: dict[str, str]`. The authority keys are `source_digest` and `storage_context_digest`.
  - `RoleplayResumeMessageBuilder = Callable[[dict[str, Any], RoleplayResumeHistory, str | None], tuple[dict[str, Any], list[dict[str, Any]]]]`.
- Produces in the prompt module:
  - `format_roleplay_resume_messages(state: dict[str, Any], history: RoleplayResumeHistory, proposed_input: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]`. It holds the logic of today's :650-661, in this order:
    1. `build_roleplay_resume_messages(state, [])`.
    2. `chat_service._format_selected_history_messages(..., user_name=context["user_name"], replace_user_placeholders=False)`, imported inside the function as the store does today.
    3. Append `proposed_input` as a user turn.
    4. `build_roleplay_resume_messages(state, history_messages)`.

    Throughout, `HistorySelectionError` becomes `RoleplayResumeConflictError("resume_context_unsupported")`.
  - `prepare_roleplay_resume(db: CharactersRAGDB, conversation_id: str, request: dict[str, Any], *, owner_client_id: str) -> dict[str, Any]`. It calls `db.prepare_roleplay_resume_generation` and returns a `{"completed": ...}` replay as is. Otherwise it pops `state` and `history`, formats, and returns today's keys: `conversation_id, input_message_id, input_message_version, assistant_message_id, fence, effective_completion, messages, history_authority, context`.
- Produces in the store:
  - `_roleplay_resume_history(self, state: dict[str, Any], *, owner_client_id: str, conn: Any) -> RoleplayResumeHistory`. It holds today's :610-647 plus the `HistorySelectionError` → `resume_context_unsupported` mapping. The byte gate reads `roleplay_resume_contract.MAX_ROLEPLAY_PROMPT_BYTES` through the function-level contract import.
  - `append_idempotent_user_message(self, conversation_id: str, request: dict[str, Any], *, owner_client_id: str, build_messages: RoleplayResumeMessageBuilder, validate_message_count: Callable[[int], object] | None = None, conn: Any | None = None) -> dict[str, Any]`.
    - `build_messages` is required; it has no default.
    - It is called as `build_messages(state, history, body["content"])`, at the point where `_roleplay_resume_messages` is called today: inside the owner transaction and before `add_message`.
  - `prepare_roleplay_resume_generation` keeps its signature. It returns `{"completed": receipt}`, or the rows `{conversation_id, input_message_id, input_message_version, assistant_message_id, fence, effective_completion, history_authority, state, history}`. It no longer returns `context` or `messages`.
  - `commit_roleplay_resume_generation` is unchanged. Its internal prepare (:787) reads only `history_authority`. Equal fences and digests mean equal inputs, so skipping the re-format there cannot change the result.
- Runtime imports in the store go only through its existing function-level `roleplay_resume_contract` statements. The store gains no new statement, so the layering counts fall by exactly the two removed imports.

- [ ] **Step 1: Write the failing tests.**
  - Put them in `test_roleplay_resume_contract.py`. They use its `resume_db` fixture (parametrized sqlite/postgres), its `fence` helper (:42) and its `completion_request` helper (:365).
  - Add these module-level imports after the existing ones, in this order:
    - `from tldw_Server_API.app.core.Character_Chat import roleplay_resume_contract, roleplay_resume_prompt`
    - `from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError, RoleplayResumeHistory`
    - `from tldw_Server_API.app.core.Character_Chat.roleplay_resume_prompt import format_roleplay_resume_messages, prepare_roleplay_resume`

  `deepcopy` is already imported.

```python
def test_prepare_returns_rows_and_the_character_chat_wrapper_formats(resume_db):
    db, cid = resume_db
    request = completion_request(db, cid)
    rows = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    assert set(rows) == {"conversation_id", "input_message_id", "input_message_version", "assistant_message_id",
                         "fence", "effective_completion", "history_authority", "state", "history"}
    assert isinstance(rows["history"], RoleplayResumeHistory) and rows["history"].authority == rows["history_authority"]
    prepared = prepare_roleplay_resume(db, cid, request, owner_client_id="alice")
    assert set(prepared) == {"conversation_id", "input_message_id", "input_message_version", "assistant_message_id",
                             "fence", "effective_completion", "messages", "history_authority", "context"}


def test_append_runs_the_injected_builder_before_inserting(resume_db):
    db, cid = resume_db
    before = deepcopy(db.get_roleplay_resume_state(cid))
    def refuse(state, history, proposed_input):
        assert (proposed_input, isinstance(history, RoleplayResumeHistory)) == ("Continue", True)
        raise RoleplayResumeConflictError("resume_context_too_large")
    with pytest.raises(RoleplayResumeConflictError) as caught:
        db.append_idempotent_user_message(
            cid, {"message_id": "refused-input", "content": "Continue", "expected": fence(db, cid)},
            owner_client_id="alice", build_messages=refuse,
        )
    assert caught.value.code == "resume_context_too_large"
    assert db.get_message_by_id("refused-input") is None
    assert db.get_roleplay_resume_state(cid) == before


def test_append_without_a_builder_is_a_type_error(resume_db):
    db, cid = resume_db
    with pytest.raises(TypeError, match="build_messages"):
        db.append_idempotent_user_message(
            cid, {"message_id": "no-builder", "content": "Continue", "expected": fence(db, cid)}, owner_client_id="alice"
        )


def test_formatting_history_selection_errors_become_resume_context_unsupported(resume_db, monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError
    db, cid = resume_db
    def legacy(*_args, **_kwargs):
        raise HistorySelectionError("legacy_review_required")
    monkeypatch.setattr(chat_service, "_format_selected_history_messages", legacy)
    with pytest.raises(RoleplayResumeConflictError) as caught:
        format_roleplay_resume_messages(db.get_roleplay_resume_state(cid, owner_client_id="alice"),
                                        RoleplayResumeHistory(snapshot=None, content=[], authority={}))
    assert caught.value.code == "resume_context_unsupported"


def test_the_prompt_byte_limit_is_one_constant():
    assert roleplay_resume_prompt.MAX_ROLEPLAY_PROMPT_BYTES == roleplay_resume_contract.MAX_ROLEPLAY_PROMPT_BYTES == 32 * 1024 * 1024
```

In the byte-gate test, set `monkeypatch.setattr(roleplay_resume_contract, "MAX_ROLEPLAY_PROMPT_BYTES", byte_limit)`. Its assertions stay the same: byte_limit 3 gives `"resume_context_too_large"` with no full read, and byte_limit 4 reaches the full-read boundary.

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t3" tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py`.
  Expected: a collection error, `ImportError: cannot import name 'RoleplayResumeHistory' from 'tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract'`.

- [ ] **Step 3: Implement** the contract additions, the prompt functions, the store split and the two endpoint call sites (Interfaces). The endpoint calls become:
  - `await run_in_threadpool(prepare_roleplay_resume, db, chat_id, contract, owner_client_id=str(current_user.id))`;
  - `build_messages=format_roleplay_resume_messages` on the append call.

- [ ] **Step 4: Update the 16 test call sites** (Files), then run the resume suites on SQLite.
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t3" tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py tldw_Server_API/tests/Chat/unit/test_native_fork_projection.py tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py`.
  Expected: PASS.

- [ ] **Step 5: Postgres proof (C5).** Run the same five files with `TLDW_TEST_POSTGRES_REQUIRED=1`.
  Expected: PASS, including every `[postgres]` case of `resume_db`. These cases cover the `FOR UPDATE` lock and the `octet_length` budget query.

- [ ] **Step 6: Lower the baselines.**
  - Layering `--tighten`. Expected diff: `core.DB_Management -> core.Character_Chat: 12` → `11` and `core.DB_Management -> core.Chat: 25` → `24`, and nothing else.
  - Logic `--tighten`. Expected diff: the two `core.DB_Management.chacha.conversation_resume_store -> …` lines deleted.
  - Check: `git grep -n "chat_service\|roleplay_resume_prompt" -- tldw_Server_API/app/core/DB_Management` prints nothing.

- [ ] **Step 7: Checks.**
  - C1: the surface ratchet passes with no baseline edit.
  - C2 and C3.

- [ ] **Step 8: Commit.**

```bash
git add tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py tldw_Server_API/app/core/Character_Chat/roleplay_resume_prompt.py tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/endpoints/character_messages.py tldw_Server_API/tests tldw_Server_API/tests/lint/layering_baseline.txt tldw_Server_API/tests/lint/db_management_logic_imports.txt
git commit -m "refactor(resume): resume store returns history rows; Character_Chat formats the prompt (TASK-13950.16)"
```

### Task 4: Sync contracts move to `core.Sync_Contracts` (AC4)

**Files:**
- Create: `tldw_Server_API/app/core/Sync_Contracts/__init__.py`, docstring only: `"""Sync wire contracts shared by core.Sync and DB_Management. Imports no core.Sync or DB_Management code."""`.
- Move these 13 modules with `git mv tldw_Server_API/app/core/Sync/v2/<m>.py tldw_Server_API/app/core/Sync_Contracts/<m>.py`, bodies unchanged:
  - `errors`, `models`, `mutation_group_validation`;
  - `notes_moodboard_studio_readiness`, `notes_task_readiness`, `notes_provenance_contract`;
  - `notes_link`, `notes_link_contract`, `notes_moodboard_studio_contract`, `notes_task_contract`, `notes_organization`;
  - `personal_context_ongoing_contract`, `attachment_refs_v2`.

  The set is closed: their relative imports stay inside it. No moved module reads `__file__`. Outside the set they reach only:
  - `core.exceptions`;
  - `core.Notes.attachment_policy`;
  - if `.15` S9 has not merged, `api.v1.schemas.sync_v2_models`, function-level, at `personal_context_ongoing_contract:158`.

  The JSON files in `core/Sync/v2/contracts/` stay where they are and are not regenerated (ruling 3).
- Create: 13 alias shims at the old paths, `tldw_Server_API/app/core/Sync/v2/<m>.py`. Each is exactly:

```python
"""Moved to tldw_Server_API.app.core.Sync_Contracts.<m> (TASK-13950.16); this path is the same module."""

import sys

from tldw_Server_API.app.core.Sync_Contracts import <m> as _moved
from tldw_Server_API.app.core.Sync_Contracts.<m> import *  # noqa: F403 - names for type checkers; runtime is the alias below

sys.modules[__name__] = _moved
```

  - **Why an alias, not a re-export.** About 400 import statements use the old paths (406 or 410, depending on the matcher). Tests read private module attributes (`notes_moodboard_studio_contract._CREDENTIAL_QUALIFIERS`). `Sync/v2/service.py:58` imports the private `models._sync_v2_internal_domain_schemas`. A re-export module would be a second object.
  - **Prototype.** The review confirmed that identity, `except`/`isinstance`, string `mock.patch` and `monkeypatch` targets, pickling and `importlib.reload` all work with either path imported first.
  - **Patch targets.** No string patch target names one of the 13 old paths.
- Create: `tldw_Server_API/app/core/Sync_Contracts/blob_keys.py`, moved verbatim from `core/Sync/v2/blob_store.py`:
  - `SyncBlobStoreError` (:29-30), `_hash_digest` (:991-998) and `_storage_namespace_id` (:1001-1007);
  - the bodies of the two pure staticmethods, as `legacy_storage_key` (:227-232) and `namespace_storage_key` (:234-243).

  `blob_store.py` imports those five names from `blob_keys`. `LocalSyncBlobStore` keeps `legacy_storage_key = staticmethod(legacy_storage_key)` and `namespace_storage_key = staticmethod(namespace_storage_key)`. `_MissingBlobDirectoryError`, `_verify_open_blob` and the rest stay in `blob_store.py`; tests patch `_verify_open_blob` there.
- Modify: the 30 DB_Management statements, to import from `core.Sync_Contracts.<m>`. `git grep -n "app\.core\.Sync\.v2" -- tldw_Server_API/app/core/DB_Management` lists them:

| File | Statements (line → module) |
|---|---|
| `ChaChaNotes_DB.py` | 156 models, 157 notes_link, 164/176/179 notes_moodboard_studio_contract, 182 notes_task_contract, 11356/11446/40572 notes_organization (function-level) |
| `Sync_DB.py` | 21 errors, 31 models, 94 mutation_group_validation, 97 notes_moodboard_studio_readiness, 105 notes_task_readiness, 8147 notes_provenance_contract (function-level) |
| `Personal_Context_Repository.py` | 67 personal_context_ongoing_contract |
| `chacha/keyword_store.py` | 20 notes_organization |
| `chacha/moodboard_sync_store.py` | 16 notes_moodboard_studio_contract |
| `chacha/note_attachment_store.py` | 28 models |
| `chacha/note_link_store.py` | 21 models, 22 notes_link |
| `chacha/note_provenance_store.py` | 13 notes_provenance_contract |
| `chacha/note_store.py` | 30 models, 31 notes_moodboard_studio_contract |
| `chacha/organization_sync_store.py` | 18 models, 19 notes_organization |
| `chacha/task_store.py` | 19 models, 20 notes_task_contract |
| `chacha/native_chat_asset_store.py` | 518 and 670 (function-level `blob_store.LocalSyncBlobStore`) become one module-level `from tldw_Server_API.app.core.Sync_Contracts.blob_keys import namespace_storage_key`; the calls at 552 and 708 become `namespace_storage_key(...)` |

- Modify: `tldw_Server_API/tests/lint/test_keyword_identity_ratchet.py`, if the base has it (Step 7).
- Modify: `tldw_Server_API/app/core/Sync/README.md:23` and `Docs/Architecture/notes-sync.md:15`, which name `core/Sync/v2/models.py`, to give the real location. Neither file is mirrored in `Docs/Published`.
- Modify: `tldw_Server_API/tests/lint/test_db_management_logic_imports.py`, adding `sync_contract_problems` and two tests.
- Test: create `tldw_Server_API/tests/Sync/test_sync_contracts_relocation.py`.
- PR description: say that #3226 and #2843 must port their `models.py`/`errors.py` hunks to `core/Sync_Contracts/` by hand, and suggest `git diff -B -M --color-moved` for review.

**Interfaces:**
- Consumes: Task 1's `db_loading`, and `.9`'s `scan`, `scan_upward_edges` and `package_of`.
- Produces: `sync_contract_problems(root: Path = APP_ROOT) -> list[str]`, sorted. It returns:
  - `f"{module} imports {target}: Sync contracts must not import core.Sync or core.DB_Management"` for each scan record, at any level, from a `core.Sync_Contracts*` module into `core.Sync` or `core.DB_Management`;
  - `f"{module} imports DB code when it is imported"` for each `core.Sync_Contracts*` module in `db_loading(root)`.

- [ ] **Step 1: Write the failing tests.**

In `test_sync_contracts_relocation.py`:

```python
MOVED = ("errors", "models", "mutation_group_validation", "notes_moodboard_studio_readiness", "notes_task_readiness",
         "notes_provenance_contract", "notes_link", "notes_link_contract", "notes_moodboard_studio_contract",
         "notes_task_contract", "notes_organization", "personal_context_ongoing_contract", "attachment_refs_v2")


@pytest.mark.parametrize("name", MOVED)
def test_the_old_sync_v2_path_is_the_moved_module(name: str) -> None:
    old = importlib.import_module(f"tldw_Server_API.app.core.Sync.v2.{name}")
    assert old is importlib.import_module(f"tldw_Server_API.app.core.Sync_Contracts.{name}")


def test_the_sync_v2_package_reexports_the_moved_classes() -> None:
    from tldw_Server_API.app.core.Sync.v2 import SyncEnvelope, SyncV2Error
    from tldw_Server_API.app.core.Sync_Contracts.errors import SyncV2Error as MovedError
    from tldw_Server_API.app.core.Sync_Contracts.models import SyncEnvelope as MovedEnvelope
    assert (SyncEnvelope, SyncV2Error) == (MovedEnvelope, MovedError)


def test_blob_store_raises_the_contract_error_class() -> None:
    from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore, SyncBlobStoreError
    from tldw_Server_API.app.core.Sync_Contracts import blob_keys
    namespace, digest = "0" * 32, "a" * 64
    assert SyncBlobStoreError is blob_keys.SyncBlobStoreError
    assert LocalSyncBlobStore.namespace_storage_key(namespace, f"sha256:{digest}") == f"blobs/v2/{namespace}/{digest}.blob"
    with pytest.raises(blob_keys.SyncBlobStoreError, match="lowercase hex"):
        blob_keys.namespace_storage_key(namespace, "sha256:" + "A" * 64)
```

In `test_db_management_logic_imports.py`:

```python
def test_a_sync_contract_that_imports_sync_or_db_code_fails(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/Sync_Contracts/models.py": f"def f():\n    from {APP}.core.Sync.v2 import store\n",
        "core/Sync_Contracts/errors.py": f"from {APP}.core.DB_Management import Sync_DB\n",
        "core/Sync/v2/store.py": "",
        "core/DB_Management/Sync_DB.py": "",
    })
    assert sync_contract_problems(root) == [
        "core.Sync_Contracts.errors imports DB code when it is imported",
        "core.Sync_Contracts.errors imports core.DB_Management: Sync contracts must not import core.Sync or core.DB_Management",
        "core.Sync_Contracts.models imports core.Sync: Sync contracts must not import core.Sync or core.DB_Management",
    ]


def test_sync_contracts_import_no_sync_or_db_code_and_db_management_no_sync() -> None:
    assert sync_contract_problems() == []
    assert ("core.DB_Management", "core.Sync") not in scan_upward_edges()
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t4" tldw_Server_API/tests/Sync/test_sync_contracts_relocation.py tldw_Server_API/tests/lint/test_db_management_logic_imports.py`.
  Expected:
  - the 15 relocation tests fail with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Sync_Contracts'`;
  - the two new lint tests fail with `NameError: name 'sync_contract_problems' is not defined`;
  - the earlier lint tests still pass.

- [ ] **Step 3: Capture the contract export before the move,** on PR C's base.
  Run: `PYTHONPATH=. "$PY" Helper_Scripts/generate_personal_context_ongoing_contract.py --source-commit 0000000000000000000000000000000000000000 --output-dir "$TMPDIR/t1316-pc-before"`.
  Expected: two files, `personal-context-ongoing-v1.schema.json` and `personal-context-ongoing-v1.manifest.json`. If `.15` S9 has merged, the generator imports the exporter from `sync_v2_models`; the commands do not change.

- [ ] **Step 4: Implement** the renames, the shims, `blob_keys`, the 30 retargets and the two doc lines (Files).

- [ ] **Step 5: Prove the move changes nothing, then run the suites on SQLite.**
  - If PR C's branch was rebased after Step 3, retake Step 3's snapshot on the new base first.
  - Run Step 3's command with `--output-dir "$TMPDIR/t1316-pc-after"`, then `cmp "$TMPDIR/t1316-pc-before/personal-context-ongoing-v1.schema.json" "$TMPDIR/t1316-pc-after/personal-context-ongoing-v1.schema.json" && cmp "$TMPDIR/t1316-pc-before/personal-context-ongoing-v1.manifest.json" "$TMPDIR/t1316-pc-after/personal-context-ongoing-v1.manifest.json"`.
    - Expected: no output, exit 0. The generator imports through the old path, so this also exercises the alias.
    - Do not compare against, or regenerate, the committed JSON files (ruling 3).
  - Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t4" tldw_Server_API/tests/Sync tldw_Server_API/tests/ChaChaNotesDB tldw_Server_API/tests/Notes_Graph tldw_Server_API/tests/lint/test_db_management_logic_imports.py`.
    - Expected: PASS.

- [ ] **Step 6: Postgres proof (C5).** Run on `tldw_Server_API/tests/Sync/*postgres*.py tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py`.
  - The glob includes the notes, notes_link, notes_organization, notes_task, notes_task_activity, notes_task_readiness and unenrolled-delete Postgres contracts.
  - Expected: PASS, with no `[postgres]` skips.

- [ ] **Step 7: Rewrite the baselines and re-key the keyword ratchet.**
  - Run layering `--write-baseline`. Its output depends on whether `.15` S9 is on the base:

    | S9 on the base? | New pairs reported | Lines deleted |
    |---|---|---|
    | No | `core.DB_Management -> core.Sync_Contracts: 29` and `core.Sync_Contracts -> api.v1.schemas: 1` | `core.DB_Management -> core.Sync: 30` and `core.Sync -> api.v1.schemas: 1` |
    | Yes | `core.DB_Management -> core.Sync_Contracts: 29` only | `core.DB_Management -> core.Sync: 30` only |

    The 29 is the 30 retargeted statements, with the two function-level blob imports merged into one. Nothing else may change.
  - Give each new line a reason:
    - `core.DB_Management -> core.Sync_Contracts: 29  # bump: Sync contracts moved out of core.Sync so DB_Management no longer imports the Sync package; not DB-loading, pinned by test_db_management_logic_imports (TASK-13950.16)`
    - `core.Sync_Contracts -> api.v1.schemas: 1  # bump: moved from core.Sync with personal_context_ongoing_contract; TASK-13950.15 S9 removes it (TASK-13950.16)`. This line exists only when S9 is not on the base.
  - Run logic `--tighten`. Expected diff: `core.DB_Management.chacha.native_chat_asset_store -> core.Sync.v2.blob_store: 2` deleted.
  - Keyword ratchet (`.5`), which keys `"tldw_Server_API/app/core/Sync/v2/notes_task_contract.py": 2` by path:
    - **#3328 merged before PR C.** In `KEYWORD_NORMALIZATION_BASELINE`, delete that entry and add `"tldw_Server_API/app/core/Sync_Contracts/notes_task_contract.py": 2,  # bump: moved from tldw_Server_API/app/core/Sync/v2/notes_task_contract.py (TASK-13950.16 PR C)` after the `core/Sync/v2/` entries. Then run `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-kw" tldw_Server_API/tests/lint/test_keyword_identity_ratchet.py`. Expected: PASS.
    - **#3328 not yet merged when PR C merges.** PR C has no ratchet to edit. Post the same re-key line as a comment on #3328 and in PR C's description. The first `.5` PR that rebases onto PR C applies it: its ratchet reports the old entry stale (0) and the new path as growth (2) until then.
  - Check: `git grep -n "app\.core\.Sync\." -- tldw_Server_API/app/core/DB_Management` prints nothing.

- [ ] **Step 8: Checks.** Run C1 (with the keyword ratchet when present), C2, C3 and C4.
  - The fingerprint must not move: classes keep their names. The review exported the full document and found 59 module-qualified component names; none contains `Sync` or `core__`.
  - If the fingerprint moves anyway, stop and report.

- [ ] **Step 9: Commit.**

```bash
git add -A tldw_Server_API/app/core/Sync_Contracts tldw_Server_API/app/core/Sync tldw_Server_API/app/core/DB_Management tldw_Server_API/tests/Sync/test_sync_contracts_relocation.py tldw_Server_API/tests/lint Docs/Architecture/notes-sync.md
git commit -m "refactor(sync): move the Sync contracts DB_Management imports into core.Sync_Contracts (TASK-13950.16)"
```

### Task 5: The lifecycle-reason rule moves to `Admin_Webhooks/domain.py` (AC3, webhook half)

Per ruling 2, this task does not wait for #2855.

**Files:**
- Modify `tldw_Server_API/app/core/Admin_Webhooks/domain.py`: insert `registration_work_lifecycle_reason`, verbatim from `delivery.py:527-545`, after `parse_registration_etag` (:716-728). It lands between #2855's domain.py hunks, so that file merges cleanly.
- Modify `tldw_Server_API/app/core/Admin_Webhooks/delivery.py`:
  - delete the definition (:527-545);
  - add `registration_work_lifecycle_reason` to its `from .domain import (` block (:30), after `parse_registration_etag,`.

  This keeps the re-export and the `__all__` entry (:1904) for `worker.py:29`, for the repository's four call sites, and for #2855.
- Modify `tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py`: delete the `delivery` import (:26-28) and add the name to the `domain` import (:29-51).
- Test: `tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py`. Its existing assertions (:506-525) still go through `delivery`; add the test below.

**Interfaces:**
- Produces: `Admin_Webhooks.domain.registration_work_lifecycle_reason(delivery: WebhookDelivery, registration: WebhookRegistration) -> DeliveryReasonCode | None`, unchanged, re-exported from `delivery`.

- [ ] **Step 1: Write the failing test:**

```python
def test_the_lifecycle_reason_rule_lives_in_domain() -> None:
    from tldw_Server_API.app.core.Admin_Webhooks import delivery, domain
    assert domain.registration_work_lifecycle_reason.__module__ == "tldw_Server_API.app.core.Admin_Webhooks.domain"
    assert delivery.registration_work_lifecycle_reason is domain.registration_work_lifecycle_reason
```

- [ ] **Step 2: Run it and confirm it fails.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t5" tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py -k lives_in_domain`.
  Expected: FAIL with `AttributeError: module 'tldw_Server_API.app.core.Admin_Webhooks.domain' has no attribute 'registration_work_lifecycle_reason'`.

- [ ] **Step 3: Move the function** (Files).

- [ ] **Step 4: Run the suite.**
  Run: `"$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t1316-t5" tldw_Server_API/tests/Admin_Webhooks`.
  Expected: PASS.
  Then run C5 on `tldw_Server_API/tests/Admin_Webhooks/test_repository_postgres.py tldw_Server_API/tests/Admin_Webhooks/test_delivery_repository_postgres.py tldw_Server_API/tests/Admin_Webhooks/test_recovery_backend_matrix.py`.
  Expected: PASS, with no `[postgres]` skips.

- [ ] **Step 5: Lower the baselines.**
  - Layering `--tighten`: `core.DB_Management -> core.Admin_Webhooks: 3` → `2`.
  - Logic `--tighten`: `core.DB_Management.admin_webhooks_repository -> core.Admin_Webhooks.delivery: 1` deleted.
  - Then run C1, C2 and C3.

- [ ] **Step 6: Commit,** and after merge post the porting note on #2855.

```bash
git add tldw_Server_API/app/core/Admin_Webhooks/domain.py tldw_Server_API/app/core/Admin_Webhooks/delivery.py tldw_Server_API/app/core/DB_Management/admin_webhooks_repository.py tldw_Server_API/tests/Admin_Webhooks/test_delivery_domain.py tldw_Server_API/tests/lint
git commit -m "refactor(admin-webhooks): move the lifecycle reason rule into domain (TASK-13950.16)"
```

**Porting note for #2855.** I simulated it with `git merge-file`, using #2855's merge base b579bdea7d, its head, and dev plus this move.

- `domain.py` and `admin_webhooks_repository.py` merge cleanly.
- The merged `from .domain import (` block keeps `registration_work_lifecycle_reason`.
- `delivery.py` gets one extra conflict hunk:
  - Dev's side holds `_canonical_timestamp` through `validate_stored_event_body` (base `:425-526`), which #2855 deletes.
  - #2855's side holds `registration_work_lifecycle_reason`, which now lives in `domain.py`.
  - Resolve it by deleting the whole conflicted region, both sides.
- `worker.py:29`, the repository and `__all__` keep working through the `.domain` import.
- The other `delivery.py` conflict, in the `.crypto` import block, exists on dev without this move.

**Relation to TASK-13950.62 AC7.** That criterion makes legacy admin webhooks adopt `evaluate_platform_webhook_url_policy` through an `afetch` policy hook. It touches different files from this task:

| Task | Files |
|---|---|
| `.62` AC7 | `services/admin_webhooks_service.py` (the legacy path), `core/Security/egress.py`, and the `afetch` hook in `http_client` |
| This task | `Admin_Webhooks/domain.py`, `delivery.py`, `DB_Management/admin_webhooks_repository.py` |

Neither changes `evaluate_platform_webhook_url_policy` or `domain.validate_webhook_target`, so neither waits for the other; whichever lands second rebases. #2855, if it ever merges, deletes `services/admin_webhooks_service.py`, which moves `.62` AC7's target. It does not affect this move.

### Task 6: Close-out (AC3, AC5, AC6 and the DoD)

**When:** after PRs A-D and `.15` S1 have merged.
- `.15` S1 completes AC3's Evaluations half. It is `.15`'s first slice and has the same base as PR A, so waiting for it is unlikely to delay anything.
- If S1 merges after PR D, `.16` stays In Progress; once S1 merges, the docs branch checks AC3 with `tools/backlog-py` (each PR edits only its own backlog task).

**Files:** `backlog/tasks/task-13950.16 - Ratchet-DB-Management-to-domain-imports-and-move-service-logic-out.md`, edited through backlog-py only. This is a backlog-only PR, or it rides with the last slice if S1 is already on dev.

- [ ] **Step 1: Verify each AC on dev.** Each `git grep` below must print nothing.

| AC | Check |
|---|---|
| AC1 | `git grep -n "kanban_vector_search\|core\.Kanban\|core\.Jobs\|core\.Embeddings" -- tldw_Server_API/app/core/DB_Management/Kanban_DB.py`, and `test_importing_kanban_db_loads_neither_jobs_nor_embeddings` passes |
| AC2 | `git grep -n "chat_service" -- tldw_Server_API/app/core/DB_Management`, and Task 3's tests pass |
| AC3 | webhook half: `test_the_lifecycle_reason_rule_lives_in_domain` passes. Evaluations half: `git grep -n "api.v1.schemas" -- tldw_Server_API/app/core/DB_Management/Evaluations_DB.py`; cite `.15` S1's PR |
| AC4 | `git grep -n "app\.core\.Sync\." -- tldw_Server_API/app/core/DB_Management`, and `test_sync_contracts_import_no_sync_or_db_code_and_db_management_no_sync` passes |
| AC5 | See (a)-(d) below. |
| AC6 | `test_backend_required_enforces_layering_ratchets[tldw_Server_API/tests/lint/test_db_management_logic_imports.py]` passes |

AC5 is checked against the amended wording, clause by clause:
- (a) The ratchet is module-keyed and fails on growth and on stale entries. `test_a_db_management_import_of_a_db_loading_module_is_counted`, `test_new_and_stale_entries_fail_with_the_fix` and `test_db_management_imports_of_db_loading_modules_only_fall` pass.
- (b) The four named cuts left the baseline. `grep -E "kanban_vector_search|conversation_resume_store|admin_webhooks_repository -> core\.Admin_Webhooks\.delivery|-> core\.Sync\." tldw_Server_API/tests/lint/db_management_logic_imports.txt` prints nothing. These are the seven lines the cuts delete.
- (c) Every non-comment line has the form `<module> -> <module>: <n>`. `grep -vE '^#|^[A-Za-z0-9_.]+ -> [A-Za-z0-9_.]+: [0-9]+( +# bump:.*)?$' tldw_Server_API/tests/lint/db_management_logic_imports.txt` prints nothing.
- (d) TASK-13950.67 tracks the rest. `grep -A2 "^dependencies:" backlog/tasks/task-13950.67*.md` shows `TASK-13950.16`.
- The line count (37 → 30 if nothing else changed it) is information for the notes, not a check.

- [ ] **Step 2: Run Bandit on the touched app code.**
  Run: `"$PY" -m bandit -q -ll -r tldw_Server_API/app/core/Kanban tldw_Server_API/app/core/Sync_Contracts tldw_Server_API/app/core/DB_Management/Kanban_DB.py tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py tldw_Server_API/app/core/Character_Chat/roleplay_resume_prompt.py tldw_Server_API/app/core/Admin_Webhooks/domain.py`.
  Expected: no new medium or high finding against dev.

- [ ] **Step 3: Record the results.** Run `PYTHONPATH=tools/backlog-py/src "$PY" -m backlog_py --cwd . task edit TASK-13950.16` with:
  - `--check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6`, for each AC proven in Step 1;
  - `--append-notes` recording:
    - the logic baseline's lines and statements before and after (37/45 → 30/37 on the measured tree);
    - the layering `core.DB_Management` lines before and after;
    - both SCC sizes from `python -m tldw_Server_API.tests.lint.test_layering_contract`;
    - the PR numbers;
    - that the remaining entries belong to TASK-13950.67;
  - `--check-dod` for each DoD item met, and `--final-summary`.

  Do not edit AC5's text and do not create a task; both were done on 2026-10-26.
  Stage with `git add -A backlog/tasks`. `tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py` must pass.

- [ ] **Step 4: Commit.** `git commit -m "chore(backlog): close out TASK-13950.16 with its measurements"`.

## Self-review (after the 2026-10-26 review and dry run)

- **Spec coverage.** Each AC has an owning task, and AC5 has a check for each of its amended clauses:

  | AC | Owning task |
  |---|---|
  | AC1 | Task 2 |
  | AC2 | Task 3 |
  | AC3 | Task 5 for the webhook half; `.15` S1 for the Evaluations half (Task 6 waits for it) |
  | AC4 | Task 4 |
  | AC5 | Task 1, with the four cuts in Tasks 2-5 and its clause checks in Task 6 |
  | AC6 | Task 1 Step 6 |

- **Rulings.** All three are applied:
  1. Task 6 neither rewrites AC5 nor files a task.
  2. PR D does not wait, and Task 5 carries the porting note.
  3. Task 4 compares generator output before and after with `cmp`, and leaves the committed JSON alone.
- **Important findings.**

  | Finding | Fixed in |
  |---|---|
  | Review I1, dry-run I2 | the Rulings section, the slice table and Task 6 |
  | Review I2 | Task 6's AC5 checks (a)-(d) |
  | Review I3 | ruling 3 and Task 4 Steps 3 and 5 |
  | Review I4 | Coordination and Task 4 Step 7 |
  | Review I5 | the fourth case in `test_production_openers_wire_the_kanban_vector_index` |
  | Dry-run I1 | Task 4 Step 7, both merge orders |

- **Minor findings.** All are applied:
  - review M1 (grep-driven run list), M2 and M3 (Expected outputs), M4 (AC1 grep), M5 (C4 evidence), M6 (Postgres glob), M7 (hazards table), M8 (Task 6 timing), M9 (two docs) and M10 (endpoint import, :1703);
  - the review's term unification ("DB-loading");
  - dry-run M1 (NameError), M2 (Consumes and own `dump`), M3, M4 (`uvx mypy`; `KanbanDB | None` fixed by `assert from_init is not None` and a typed list), M5 (size-baseline wording), M6 (`-A2` count), M7 (test imports), M8 ("about 400"), M9 (`backlog` excluded) and M10 (PR A estimate ~370).

  None declined.
- **Type consistency.** Each name is spelled the same everywhere it appears: `db_loading`, `measure`, `violations`, `sync_contract_problems`, `vector_search_factory` / `_vector_search_factory`, `RoleplayResumeHistory`, `RoleplayResumeMessageBuilder`, `format_roleplay_resume_messages`, `prepare_roleplay_resume`, `_roleplay_resume_history` and `build_messages`.
