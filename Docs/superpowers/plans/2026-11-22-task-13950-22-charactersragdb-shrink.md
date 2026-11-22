# Freeze and shrink CharactersRAGDB (TASK-13950.22) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ChaChaNotes_DB.py` ends under 25,000 lines and `CharactersRAGDB` under 400 methods, as the surface ratchet counts them. On the way:
- a ratchet stops new app callers of delegated names;
- four stores reach the DB through a public context object, with no private reaches;
- migrations live in a version registry;
- two stores' delegates are retired, which proves the recipe for the other seven.

**Architecture:** Fourteen tasks, fifteen PRs, merged in order.
1. Freeze the app's delegated callers (Task 1).
2. Retire the two cheapest stores' delegates, as a pilot (Task 2).
3. Build `ChaChaContext` and move four stores onto it (Tasks 3-4).
4. Move schema SQL, then the SQLite and PostgreSQL schema helpers, then the migrations, then the bootstrap out of the class (Tasks 5-9). Helpers move before migrations because migrations call them, not the reverse.
5. Move the workspace and flashcard domains into stores: callers first, through a temporary façade, then the bodies (Tasks 10-13).
6. Close out (Task 14).

One stdlib script, `Helper_Scripts/chacha_split/split.py`, produces every bulk edit, with its own `verify` as PR evidence (Appendix A). Two checks share no code with it:
- `Helper_Scripts/chacha_split/diffpairs.py`, a structural diff of each moved body against its base;
- `ruff` F821/F841/F401/F811 on exactly the moved files.

**Tech Stack:** Python 3.12/3.13, stdlib `ast`/`tokenize`, SQLite and PostgreSQL 18, pytest, AST lint ratchets under `tldw_Server_API/tests/lint/` (`_ast_cache.py`, `_ceilings.py`), GitHub Actions (`backend-required.yml`).

**Spec:**
- `backlog/tasks/task-13950.22 - Freeze-and-shrink-the-CharactersRAGDB-god-class.md`, AC #1-#5.
- The roadmap's Global Constraints and Phase 3 `.22` row: `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`.
- Research behind every number, at dev ee0888b51b: `.superpowers/sdd/2026-11-22-task-13950-22-charactersragdb-shrink/research-A.md` and `research-B.md`, with their scripts. They are git-ignored, so the executor's ledger workspace holds them. The planner reran every number cited below.

## Measured (dev ee0888b51b)

- **File and class.** `ChaChaNotes_DB.py` is 47,924 lines, and `CharactersRAGDB` spans `:950-47397`.
- **The surface ratchet** reads `methods` 863 (860 class-body defs + 3 module-level bindings), `delegated_methods` 306 and `private_member_reaches` 553.
  - Delegated bindings are not methods.
  - A class-body `import` adds nothing: `_class_body_defs` counts only `FunctionDef`/`AsyncFunctionDef`.
- **Delegation.** Nine `for … setattr(CharactersRAGDB, _x, _delegate_store_method(...))` loops sit at `:47453-47840`, plus 89 TYPE_CHECKING stubs at `:975-1389`.
  - `resolve_moodboard_compatibility_dataset_id` and `resolve_studio_compatibility_dataset_id` are in the moodboard tuple, but module-level wrappers rebind both at `:47698-47699`.
- **AC1's callers, with Ruling 1's receiver rule.** There are 1,051 sites in 92 app files outside `core/DB_Management/`:
  - 1,025 attribute loads;
  - 26 `getattr`/`hasattr` probes with a literal delegated name;
  - 41 collision-name sites, excluded (none has a literal `chacha_db`/`notes_db`/`character_db` receiver).
  - Per store: persona_state 267, note 187, message 177, conversation 161, character 102, keyword 98, conversation_resume 30, task 29, moodboard_sync 0.
- **AC2's four stores** reach 246 times (note 102, keyword 77, conversation 51, message 16). They also use public DB members, cross-store attributes included, such as `note_graph_projection_store`, `workspace_chat_startups` and `conversation_resume_store`.
  - Tests build the real `ConversationStore`/`MessageStore` in 7 files and `NoteStore`/`KeywordStore` in 4. `tests/Slides/test_standalone_html_sources.py` defines its own fakes, and `tests/Notifications` has an unrelated `MessageStore`.
  - **Patches of the Task 3 helper names:** 37 sites, in every form.
    - 22 are call-form: `_get_current_utc_timestamp_iso` 14 (6 on `source_review_db`), `_map_table_for_backend` 3, `_selected_owner_filter` 2, `_generate_uuid`, `_get_current_db_version`, `_prepare_backend_statement`.
    - 8 assign the attribute: `db._get_current_utc_timestamp_iso = …` ×5, `_generate_uuid` ×2, `db._ensure_message_metadata_table = …` ×1.
    - 7 are keyword-built fakes (`_generate_uuid=` ×5, `_prepare_backend_statement=` ×2). All are `SimpleNamespace` fakes of Media DB or email-ingestion objects (`test_media_db_*`, `test_email_sensitive_logging_13376_2.py`).
  - Task 4's names add 3 more (`_note_task_v60_hash`, `_require_selected_owner_row`, `_deserialize_row_fields`).
  - One reach the counter misses today: `chacha/persona_transcript_store.py:177` `self.db._prepare_backend_statement`.
  - The bodies moving to the context read `self._transform_query_for_backend`, a private DB method, and the module privates `_owner_is_scoped` and `_SAFE_IDENTIFIER_RE` (`ChaChaNotes_DB.py:419`, `:340`).
- **55 sites in 33 test files build the DB with `CharactersRAGDB.__new__`/`object.__new__`**, skipping `__init__`. One of them is `test_independent_buddies.py:145`, which calls `_sqlite_linear_migration_steps()` on such an instance.
- **Schema/migration cluster.** These are research B's closure groups, reproduced by `split.py select` (Appendix A.5), with 13 shims staying on the class:

  | group | defs | lines |
  |---|---|---|
  | sqlite-schema | 76 | 4,429 |
  | postgres-schema | 69 | 6,345 |
  | migration-support | 9 | 227 |
  | migrate-sqlite | 80 | 2,222 |
  | migrate-postgres | 35 | 2,246 |
  | init | 2 | 957 |
  | shims (stay on the class) | 13 | 852 |

  - **Call edges:**
    - migrate-sqlite → sqlite-schema 48;
    - migrate-postgres → postgres-schema 76;
    - schema helpers → migrations 0;
    - sqlite-schema ↔ postgres-schema 0.
  - **Runtime members research B grouped as "shared":** `_search_generic_items_fts`, `_ensure_message_metadata_table` and `_ensure_bootstrap_for_backend`. Tasks 3 and 4 handle the first two; the third stays on the class.
- **Constants:** 107 schema constants, 7,077 lines. Tests read them in 38 files and patch them on the class 6 times.
  - Two of them, `_NOTE_TASK_V60_RELATIONS` and `_NOTES_MOODBOARD_STUDIO_V61_RELATIONS`, read a 108th class constant, `_NOTE_TASK_SCOPE_AUTHORITY_TABLE`, so it moves with them.
  - `_RETIRED_SYNC_PLACEHOLDER_REPAIR_SQL` is an f-string over two module constants, `RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND`/`_ID` (`:346-347`). `conversation_store.py:20-21` imports both from ChaChaNotes_DB.
- **Migrations:**
  - **SQLite.** `_sqlite_linear_migration_steps` (`:9827`) maps source versions 4..84; 84 is `functools.partial(persona_transcript_store.migrate_sqlite_transcripts, …)`, imported at call time.
  - **PostgreSQL.** `_initialize_schema_postgres` (`:27166`) runs the 72→91 steps in two if-chains:
    - the fast path at `:27234`, where v72 first ensures the study-pack triggers;
    - the full path at `:27612`, which also writes `_runtime_schema_version`.

    It calls 16 more PostgreSQL migrations (v54→v70) inline, and SQL scripts for v5..v53.
  - **Tests:** 63 direct `_migrate_*` calls in 26 files; 17 patches in 13 files; 3 class patches of `_sqlite_linear_migration_steps`.
  - **Key 84** today is a `functools.partial` with no `__name__`.
- **Class-receiver references** (`CharactersRAGDB.<name>`) to moved names: 62 test sites. 58 of them change; the 4 calls of `_prepare_sqlite_schema_drift_fixture` stay, because R24 keeps that classmethod on the class. Appendix D lists them by rule:
  - 22 static or classmethod calls;
  - 23 `inspect.getsource` reads, 6 of them on `_initialize_schema_postgres`, asserting ladder text and order;
  - 13 captures or identity checks;
  - 4 unbound calls with a fake `self`.
- **Moved classmethods:** 14 (2 schema, 12 workspace/flashcards).
- **Patches on the move set:** 47 class-level and 79 instance-level. The top targets are `_initialize_schema` 35, `_get_schema_version_postgres` 15 and `_verify_note_*_schema_postgres` 15.
- **Dead branch:** `:23304-23309` calls `_migrate_from_v65_to_v66_sqlite`/`_v66_to_v67_sqlite`, which do not exist. The same chain ran v65→v67 at `:23256-23260`.
- **Domains:**
  - **Workspace:** 195 defs, 6,223 lines, 158 app + 1,315 test call sites. 19 of its names are also defs elsewhere in app: mostly endpoint handlers, plus `update_workspace` on `Orchestration_DB`.
  - **Flashcards/decks/study packs:** 96 defs, 4,064 lines, 167 app + 910 test call sites, and 26 such collisions.
  - `read_workspace_clone_snapshot` holds the `core.Sharing.clone_models` logic import (`db_management_logic_imports.txt:3`).
- **Pilot:**
  - moodboard_sync retires 5 names (0 app callers, about 55 test sites in 3 files, 5 stubs).
  - conversation_resume retires 6 names:
    - 31 app attribute sites and 4 `getattr` probes;
    - about 290 test sites in 15 files;
    - 11 instance patches and 1 stub;
    - 13 fake-DB defs of `get_roleplay_resume_state` in 4 test files.
- **Baselines a move touches:**
  - `function_size_baseline.json` has 13 `CharactersRAGDB.*` entries, all in the schema cluster.
  - The `owner_scope_baseline.json` opt-outs (4) sit in skill-registry and note-moodboard runtime methods, and none of them moves.
  - `test_owner_scope_ratchet._is_scoped` scans only top-level `chacha/*.py`.
  - The layering ratchet counts import statements per package pair.
  - ruff's `exclude` skips every directory named `migrations`. A file passed explicitly is still checked unless `--force-exclude` is given, as measured with ruff 0.16.5.
  - `ruff check --select F821,F841,F401,F811` is clean on ChaChaNotes_DB.py and `chacha/` today.
  - ruff is non-blocking in CI.
  - `.gitignore:281` ignores `*.json`, so JSON manifests would not be committed.
  - `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md` is mirrored in `Docs/Published`, and its schema text is stale ("current version: 7").
- **Python:** the venv is 3.12.13, backend-required runs 3.12, and `pyproject.toml` says `requires-python = ">=3.12"`.
- **Open PRs touching these paths:**
  - #3512 adds `chacha/flashcard_chapter_queries.py` and edits `endpoints/flashcards.py`;
  - #3405 and #3294 edit `test_license_first_workflow_contracts.py`, and #3405 also edits `backend-required.yml`.
- **Disk:** 105 GiB free at planning time.
- **Runtime** (the dry run, `-n 2`, Docker `postgres:18`):
  - PROOF_MIG's 41 SQLite files: 630 tests in 102 s;
  - its 24 live-PostgreSQL files: 479 tests in 252 s.
  - **One known flake:** `test_chachanotes_db.py::TestNotes::test_list_notes` (a same-timestamp UUID tie-break); rerun it alone.
  - **One skip local runs cannot clear:** `test_sync_v2_notes_link_postgres_contract.py::test_postgres_notes_link_same_edge_scope_is_tenant_isolated_by_rls` needs `TEST_DATABASE_URL`.
  - PROOF_BOOT's 20 extra files and OWNER_ISO are timed in Task 5's first run, and recorded in the PR.

## Rulings

Rulings R1 to R10 are the controller's and binding; R11 to R18 are the planner's additions. Each gives its why and its cost if wrong.

1. **R1, the AC1 ratchet.** It counts app call sites (`tldw_Server_API/app/**` outside `core/DB_Management/`) of delegated names by research B's receiver rule.
   - The five collision names count only on a literal `chacha_db`/`notes_db`/`character_db` receiver.
   - Counts are keyed per file and shrink-only through `_ceilings`.
   - A seeded self-test, and names read from the class at test time.
   - Tests are not counted, because fake DBs make them noisy and each retirement codemods them anyway.
   - **Cost if wrong:** new tests keep calling `db.<delegated>` until their store retires.
2. **R2, the context.** `ChaChaContext` in `chacha/context.py`; `CharactersRAGDB.__init__` builds it once, as `db.ctx`, before the stores.
   - Its surface is curated, documented and all public.
   - The conversation, message, note and keyword stores receive it instead of the DB and make 0 private reaches.
   - Each DB-bound member is decided in Tasks 3-4's tables.
   - Other stores keep the DB for now.
   - It lands before any schema move.
3. **R3, migrations.**
   - They go in `chacha/migrations/`, with `SQLITE_STEPS` (source versions 4..84) and `POSTGRES_STEPS`.
   - Functions keep their names and take `(ctx, conn)`.
   - No `_migrate_*` shim stays on the class. The class keeps `_sqlite_linear_migration_steps`, `_run_sqlite_linear_migration_step` and both version attributes, read at call time.
   - Versions stay 85 (SQLite) and 91 (PostgreSQL).
   - A frozen-list test pins keys, names and order.
   - Tests that call or patch moved names change in the same PR.
4. **R4, schema SQL.** The 107 constants move to `chacha/schema_sql.py`, re-imported in the class body, with `_NOTE_TASK_SCOPE_AUTHORITY_TABLE` as a 108th because two of them read it. The ratchet does not count the import as a method (Measured).
5. **R5, schema code.** SQLite helpers, then PostgreSQL helpers, then bootstrap.
   - The 13 shims stay.
   - Store reaches into schema methods call module functions.
6. **R6, workspace and flashcards.**
   - Callers change by codemod, with no delegation transfer.
   - Every PR is green on its own.
   - Flashcards wait for #3512.
7. **R7, retirement.** This task closes AC4 by its numbers, plus a pilot: moodboard_sync, then conversation_resume.
   - Task 14 files the other seven stores as follow-ups and amends AC4.
   - **Cost if wrong:** the owner wanted all 306 retired inside `.22`; the follow-ups then run under this id.
8. **R8, verification.**
   - `split.py` produces each bulk edit, and its `verify` proves each moved body identical to the base, modulo declared rewrites.
   - A rebase re-runs the script on the new dev.
   - PRs merge within one or two days.
   - **Exception:** Tasks 3-4's context helpers are hand-written. They are about 150 lines, renamed and given a new receiver, and behaviour tests pin them.
9. **R9, bookkeeping.** In every move PR:
   - re-key `function_size_baseline.json` with the same ceilings;
   - `owner_scope_baseline.json` gets an entry only for a module with opt-outs. A `0/0` entry fails as stale, and no opt-out moves in this plan.
   - surface `--tighten`;
   - `db_management_logic_imports.txt`, `layering_baseline.txt` and `env_read_baseline.json` where touched.

   The only allowed raises are the named bumps in Global Constraints, and none of them raises `private_member_reaches` (R29). New modules stay under `chacha/`.
10. **R10, CI.**
    - No new job, and the owner-isolation job is not widened (that is `.141`).
    - The new ratchet and the new guards join "Enforce CI contracts and code ratchets", with workflow-contract assertions. That satisfies AC5.
    - **Cost:** the AC1 ratchet reuses the cached AST parse, so a few seconds.
    - Local proof uses Appendix B's sets.
11. **R11, order.** Schema helpers (Tasks 6-7) move before migrations (Task 8).
    - **Why:** migrations call 124 helper references, and helpers call no migration. Helpers first means moved migrations call module functions directly, so no temporary DB forwarders (private reaches) exist.
    - **Cost if wrong:** none; R5's order is kept.
12. **R12, registry values are names.** `SQLITE_STEPS: dict[int, str]` and `POSTGRES_STEPS: dict[int, str]` map a source version to a function name, which is resolved on its module at call time.
    - **Why:** the class's legacy init calls the same functions directly. One patch point, the module attribute, then covers both the runner and the direct calls; a dict of function objects would leave a module patch inert on one path.
    - `POSTGRES_STEPS` holds source versions 72..90, the 19 steps that take the schema from 72 to 91 (R3's "72..91").
    - The v5..v71 ladder stays one function: inline in `_initialize_schema_postgres`, which Task 9 moves verbatim.
    - **Cost if wrong:** tests call `getattr(module, SQLITE_STEPS[n])` instead of `SQLITE_STEPS[n]`.
13. **R13, AC1 counts probes.** `getattr`/`hasattr(<receiver>, "<delegated name>")` counts as a call site.
    - **Why:** a retired name turns such a probe silently False, as in `character_chat_sessions.py:4084`. The seed is 1,051, not 1,025.
14. **R14, no new receiver hides a reach.** From Task 3, the surface counter treats any private attribute on a receiver named `db`/`ctx`, or on an attribute `.db`/`._db`/`.ctx`/`._ctx`, as a reach.
    - It also pins the context's own reaches to `PINNED_DB_PRIVATES`, which are only class attributes tests patch or init mutates.
    - It pins the AC2 stores at 0.
    - **Why:** `self._ctx.db._x` would otherwise evade the count. The one existing hit (`persona_transcript_store.py:177`) is fixed in Task 3.
15. **R15, workspace and flashcard split.** The callers PR codemods app and tests to `db.<store>.<m>` behind a temporary façade store that forwards to the class. The move PR then moves the bodies, rewrites class-internal callers and retargets patches.
    - **Why:** each is green alone, and instance patches stay live in between.
16. **R16, the owner-scope ratchet scans `chacha/**`** (Task 6), so `chacha/migrations/` is covered. It is a no-op today, because chacha/ has no subdirectory.
17. **R17, `verify` is PR evidence, not a CI guard.** It compares a PR with its base, so it has nothing to check after merge. Its output goes in the PR body, and the reviewer re-runs it.
18. **R18, the dead v65/v66 branch** (`:23304-23309`) is deleted in Task 9, as a declared structural edit.
    - **Why:** it calls names that do not exist. It is unreachable unless a step fails to advance, and then a `SchemaError` replaces today's `AttributeError`.

The controller's fix-round-1 rulings (binding):

19. **R19, a lazy `ctx` (B1).** `ctx` is a `functools.cached_property` on `CharactersRAGDB`, returning `ChaChaContext(self)`. So the 55 `__new__`/`object.__new__` sites keep working.
    - `__init__` reads `self.ctx` before building the stores, so they share one instance.
    - The property is one more counted method: Task 3 takes the named +1 bump (Global Constraints), and every projection counts it.
    - Pinned by `test_ctx_exists_on_a_db_built_without_init`, which builds the DB with `CharactersRAGDB.__new__`, then reaches `ctx` and `_sqlite_linear_migration_steps()`.
20. **R20, the placeholder module constants (B2, I4).** `RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND` and `_ID` move into `chacha/schema_sql.py` with the constant that reads them.
    - ChaChaNotes_DB re-imports both at module level, so every importer, `conversation_store.py` included, keeps working.
    - `schema_sql.py` still imports nothing from ChaChaNotes_DB.
21. **R21, every new guard is in the required gate (I5, AC5).** Every guard, pin or self-test a task adds goes into backend-required's "Enforce CI contracts and code ratchets" step and into `TEST_SEAM_GATE_SUITES` (whose mutation tests are the contract assertion), in the PR that adds it.
    - Two files carry all of them:
      - `tldw_Server_API/tests/DB_Management/test_chacha_structure_pins.py`, which imports code and builds no DB;
      - `tldw_Server_API/tests/Helper_Scripts/test_chacha_split.py`, the tool's self-tests.
    - Task 2 creates and wires both, and later tasks add test functions to them.
    - AST-only pins go into `test_charactersragdb_surface_ratchet.py`, which is already in the step.
    - **Cost:** seconds, with no new job. An import of ChaChaNotes_DB measures 0.3 s, and the pins build no DB.
22. **R22, an independent check beside `verify` (I7).** `verify` stays as PR evidence. Two checks share no code with `apply`:
    - `ruff check --select F821,F841,F401,F811 --no-force-exclude` on exactly the moved and edited files. Passing the files explicitly overrides the `migrations` exclude, and the baseline is clean today.
    - `Helper_Scripts/chacha_split/diffpairs.py`, which imports nothing from `split.py`. It walks the base and head bodies in parallel after renaming only the receiver (`self` → `ctx`) and prints each distinct `old → new` subtree pair. It fails on any difference not rooted at a `self.<attr>` or a bare `self`. The PR body carries the pair list, and the reviewer checks each pair against the manifest.
    - **Python floor:** 3.12+, because AST positions inside f-strings need it. The venv is 3.12.13 and CI runs 3.12; `split.py` and `diffpairs.py` refuse to run on older interpreters.
    - **What neither check sees:** attributes resolved through a subclass (none today), `super()` (none) and nested defs (10, compared as part of their parent). A moved body that binds a name `ctx` fails closed (none today).
23. **R23, manifests are `.py` literals (I8).** `Helper_Scripts/chacha_split/manifests/t<NN>_<name>.py` hold one `MANIFEST = {...}` literal, read with `ast.literal_eval` and never imported.
    - **Why:** `.gitignore` ignores `*.json`, while `.py` under `Helper_Scripts/` is committed (`git check-ignore` exits 1), so no ignore-file change is needed.
24. **R24, class-receiver, static and classmethod references (I2).**
    - **Functions mode:** a static target, or a classmethod made static, is called as `<alias>.<m>(args)` from any receiver.
      - A classmethod moves as a module function only when `cls` appears solely as `cls.<x>`, with `x` moved, in `schema_sql`, or a static context member. Those become `<alias>.<x>`, `schema_sql.<x>` or `ChaChaContext.<member>`; anything else fails closed.
      - A classmethod that reads a version or schema-name attribute (`cls._CURRENT_SCHEMA_VERSION`, which tests patch) stays on the class. `select` excludes it.
      - Measured: `_verify_note_task_v59_authority_catalog` moves, while `_prepare_sqlite_schema_drift_fixture`, a test-fixture helper reading `cls._CURRENT_SCHEMA_VERSION`/`cls._SCHEMA_NAME`, stays.
    - **Store mode:** a classmethod stays a classmethod on the store class.
    - A class receiver calling a non-static target, and any non-call reference (`getsource`, captures, `__func__ is`), is never auto-rewritten. `split.py check` lists it, and the implementer fixes it by hand.
    - Appendix D lists the 62 sites by rule; 58 of them change.
    - `split.py check` (and `callers --check`) fails on any untouched reference to a moved or retired name.
25. **R25, patch detection (I1).** `split.py patches` sees:
    - `monkeypatch.setattr`/`delattr` and `patch.object`/`patch.multiple`/`mocker.patch.object`, in call, `with` and decorator form;
    - `"…CharactersRAGDB.<name>"` patch strings;
    - attribute assignment (`db._x = …`, `CharactersRAGDB._x = …`, `type(db)._x = …`);
    - keyword-built fakes (`SimpleNamespace(_x=…)`, `MagicMock(_x=…)`);
    - `getsource` of a watched name.

    The watched set is the manifest names, the class constants the moved bodies read, and the module-level free names they import from ChaChaNotes_DB (R30). It lands in Task 3 with a `--names` mode, and Task 3's census is 37 sites, not 12. A patch on a non-ChaCha object (the Media DB and email-ingestion fakes) is reported, marked as such in the PR, and left alone.
26. **R26, source-reading tests become ladder-data tests (I3).** The 23 `getsource` sites are never deleted.
    - Task 7, the first task that breaks the 6 `_initialize_schema_postgres` readers, re-expresses them through `tldw_Server_API/tests/helpers/chacha_postgres_ladder.py`:
      - `ladder_calls(fn) -> list[LadderCall]`: each call's callee name and keyword arguments, in source order, read by AST, so a call-site rewrite does not change them (R33);
      - `branch_calls(fn, version: int) -> list[LadderCall]`: the same, inside `if current_version < <version>:`.

      Order assertions compare callee names, and Task 9 repoints the helper's one `postgres_init_function()` line.
    - The 17 readers of other moved names retarget to the module function in the task that moves it.
    - A retargeted `not in` assertion keeps a positive anchor on the same source, so a shim or an empty body cannot pass it.
27. **R27, the context's privates (I6).**
    - `ctx.prepare_statement` passes `transform_sqlite_query_for_postgres` directly as the PostgreSQL transformer, so `_transform_query_for_backend` stays on the DB, used only by `_prepare_backend_many_statement`.
    - `context.py` imports `_owner_is_scoped` and `_SAFE_IDENTIFIER_RE` from ChaChaNotes_DB at its top. They are listed in `PINNED_MODULE_PRIVATES`, with this reason: ChaChaNotes_DB keeps using them, and moving them would make ChaChaNotes_DB import context at module load, which cycles.
    - The import is cycle-safe because ChaChaNotes_DB imports context only inside the `ctx` property.
    - `chacha.context` joins the standalone-import test in Task 3.
28. **R28, the AC4 file target is judged at close-out (T14),** after Tasks 9-13. G lands at about 24,850, which is not chased. The projected final size is about 12,820 lines.

The controller's fix-round-2 rulings (binding):

29. **R29, no PR raises `private_member_reaches` (N-1).** Each PR's reach count is at or below its base, and a new `PINNED_DB_PRIVATES` entry is offset in the same PR.
    - **Task 9's offset:** `_runtime_schema_version` is renamed to a public instance attribute `runtime_schema_version`, with a class-level `None` default.
      - That removes Task 4's pinned getter reach (−1).
      - The bootstrap writes `ctx.db.runtime_schema_version`, so the context needs no setter and makes no write reach.
      - `_POSTGRES_SCHEMA_VERSION` must stay a literal class attribute, because tests patch it 49 times. A property would cost a method, so it is the one new pin (+1).
      - Net 0: H stays at 296, and no fifth bump is needed.
    - Tasks 6-8 read only already-pinned attributes (`_SCHEMA_NAME`, `_CURRENT_SCHEMA_VERSION`; measured), so they add no pin.
30. **R30, free names are watched (N-2, closes I1).** `patches` and `check` also watch the module-level names each manifest's moved bodies import from ChaChaNotes_DB's namespace. A patch of `ChaChaNotes_DB.<name>` whose reader now reads `<newmodule>.<name>` is inert.
    - There are four known sites at base, retargeted in their tasks:
      - `test_notes_moodboard_studio_sync_postgres_tenancy.py:313` and `:332` (`time`; Task 7, or Task 8 if `select` places the reader there);
      - `test_source_review_plans.py:927` (`build_source_review_launch_metadata`; Task 11);
      - `test_flashcards_endpoint_integration.py:3044` (`_normalize_scheduler_settings_envelope`; Task 13).
    - The `psycopg_sql` patches in `test_chacha_postgres_session_scope.py:71,90` stay, because their reader does not move.
31. **R31, tool self-tests own their git identity (N-3).** A self-test that builds a tmp git base commits with `git -c user.name=chacha-split-test -c user.email=chacha-split-test@example.invalid -c commit.gpgsign=false`, and never reads global git config, which CI does not have. Task 2 also runs the two pin files once under the step's environment.
32. **R32, `diffpairs` normalizes only headers (N-4).**
    - It drops the `staticmethod`/`classmethod` decorator and the classmethod's `cls` parameter.
    - It accepts `cls.<attr>` as a rewrite root.
    - It classifies by the pre-rename text.
    - The body is never normalized, so a changed body still exits 1.
    - One self-test each, for a static and a classmethod move (Task 4).
33. **R33, the ladder helper keeps keyword arguments (N-5, closes I3).** `ladder_calls` and `branch_calls` return `LadderCall(name, kwargs)`. So `test_note_attachment_migration_v59.py:921`'s `lock=True` check becomes `kwargs == {"lock": True}` on the first `_get_schema_version_postgres` call, ahead of `_migrate_from_v58_to_v59_postgres`.
34. **R34, the legacy-marker order test is in the gate (N-6).** `test_postgres_init_repairs_legacy_markers_in_order` is AST-only, so it lives in `test_chacha_structure_pins.py`. The Task 7 re-expressed `getsource` tests stay in their existing files.

**Minors declined:** M3's "about 312" after Task 4. It adds the two Task 3 pins a second time and misses the −5 for `_supports_notes_moodboard_studio_v61` in task_store and moodboard_sync_store. The derivation is 488 − 179 − 5 + 1 = 305. Every other Minor (M1, M2, M4-M9) is fixed; see the changelog.

## Global Constraints

Copied into every implementer brief.
- **PRs:** target `dev`. Each branches from the latest `origin/dev`, one PR per slice in the table below.
  - Carry a `## Change summary`; the owner waived ADR-004 sign-off for TASK-13950.
  - Get a clean-subagent code review before merge.
  - Never arm auto-merge.
- **Git:** never `git stash`, `--no-verify`, force-push or `git clean -fdx`.
- **Python:** `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. There is no `python` on PATH. The shell is zsh, so `$VAR` lists don't word-split; use `$(cat list)` or `${=VAR}`.
- **Disk:**
  - Refuse to start a pytest run under 100 GiB free (`df -g /System/Volumes/Data`).
  - Each run uses `-p no:cacheprovider --basetemp "$TMPDIR/t22-<task>-<n>"`, deleted right after.
  - At most `-n 2`, one run at a time. OWNER_ISO runs serially.
  - No repo exports or extra worktrees.
- **Tests and network:**
  - `unset OPENAI_API_KEY ANTHROPIC_API_KEY`.
  - `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:9`, `NO_PROXY=localhost,127.0.0.1`.
  - Never pass `-p pytest_asyncio.plugin` or `-p xdist.plugin`.
  - PostgreSQL runs set `TLDW_TEST_POSTGRES_REQUIRED=1` (the plugin auto-starts Docker `postgres:18`; don't set `TLDW_TEST_NO_DOCKER`). A local skip is not evidence.
- **Known local failures (not regressions):**
  - `test_runtime_package_discovery`;
  - tests needing tiktoken;
  - 3 in `test_merge_queue_gate_workflows` and 3 in `test_required_workflow_contracts`;
  - `test_rg_route_map_lint` and `test_route_auth_ratchet` flake under `-n 2`;
  - `test_chachanotes_db.py::TestNotes::test_list_notes` flakes; rerun it alone;
  - `test_sync_v2_notes_link_postgres_contract.py::test_postgres_notes_link_same_edge_scope_is_tenant_isolated_by_rls` skips without `TEST_DATABASE_URL`.
- **Bandit:** `uvx --offline bandit -q` on changed app and `Helper_Scripts` files. No new findings against the base. Moved code keeps its `# nosec` comments.
- **Backlog:** `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py task …` only, with explicit ids. Stage with `git add -A backlog/tasks`.
- **Subagents:** implementers never dispatch subagents.
- **Ratchet rules (roadmap):** lint ratchets live in `tests/lint/`, are keyed by module or qualname, fail on growth and stale entries, ship a seeded-violation self-test and share `_ast_cache.py`.
- **The only allowed baseline raises** (R9), each with exactly this reason shape:
  - `function_size_baseline.json`: a re-keyed entry whose `lines`/`params` are at most the old entry's, reason `"moved from <old key> (TASK-13950.22 T<n>)"`. Delete the old key in the same PR.
  - `db_management_logic_imports.txt`: a re-keyed line, `# bump: moved from <old module> (TASK-13950.22 T<n>)`;
  - `layering_baseline.txt`: +1 on a pair when moved code needs a top-level import that ChaChaNotes_DB.py still uses, `# bump: <new module> needs <name>; ChaChaNotes_DB still imports it (TASK-13950.22 T<n>)`;
  - `charactersragdb_surface_baseline.json`, Task 3 only: `methods` 863 → 864, reason `"TASK-13950.22 T3: lazy ctx cached_property (R19) so __new__-built DBs reach the context"`.
- **No PR raises `private_member_reaches`** (R29). A new `PINNED_DB_PRIVATES` entry is offset in the same PR, so each PR's reach count is at or below its base.
- **Moves:**
  - Every bulk edit is `split.py` output, committed with its `.py` manifest (R23).
  - Run `split.py` and `diffpairs.py` with the 3.12+ venv.
  - A hand edit after `apply` must be a declared `structural` def, or `verify` fails.
  - A rebase re-runs the manifest's commands on the new dev, never a hand merge of moved bodies.
  - A move PR is opened, green and merged within one or two days.
  - Before opening one, `gh pr list --base dev --search "ChaChaNotes_DB.py"` and check the open PRs' files. If another open PR edits a body this PR moves, merge or rebase that one first.
- **Patching rule after a move:**
  - Patch the module attribute (`monkeypatch.setattr(sqlite_schema, "_x", fake)`), never the DB or the class.
  - A fake for a module function takes `(ctx, conn, …)`.
  - A class patch of a `schema_sql` constant moves to `schema_sql` once every reader of it has moved (`git grep -n 'self\.<NAME>\b'` in the class is empty). While readers remain on both sides, patch both.
  - An assignment-style patch (`db._x = f`) follows the same rule as `monkeypatch`.
  - A `MagicMock(spec=CharactersRAGDB)` has no store attributes (24 sites), so set `mock.<store_attr> = MagicMock()` where a rewritten call needs one.
  - `split.py patches` and `split.py check` must report 0, apart from the non-ChaCha hits marked in the PR body.

## Review Focus

1. **An inert patch after a move.** A test patches a name the code no longer runs through, in any of R25's forms, and passes vacuously.
   - `split.py patches` must be empty, apart from marked non-ChaCha hits, from Task 3 on.
   - R12 gives one patch point per migration.
   - Pinned by `test_patches_finds_each_patch_form` (Task 3) and `test_module_patch_reaches_registry_and_direct_calls` (Task 8).
2. **A retired or moved name still reached silently.** A `getattr` probe or a fake DB that defines the name turns the path off without failing.
   - `callers --check` fails on any probe, attribute load or fake class left behind.
   - Pinned by `test_check_flags_a_fake_without_the_store_attribute` and `test_probe_is_rewritten_through_the_store` (Task 2).
3. **Migration order or versions drift on either backend.**
   - Pinned by Task 8's frozen `test_registries_match_dev_before_the_move` and `test_versions_unchanged`.
   - Proved by PROOF_MIG on SQLite and live PostgreSQL in Tasks 5-9.
4. **`CharactersRAGDB._X` breaks, or the private-reach count rises.**
   - Task 5's `test_schema_constants_stay_class_attributes`.
   - Task 3's widened counter and `test_the_context_reaches_only_pinned_db_members` / `test_zero_reach_stores_stay_at_zero`.
5. **PostgreSQL advisory-lock and SQLite schema-init-lock semantics change in Task 9.**
   - **SQLite:** the lock-holding wrapper `_initialize_schema_sqlite` and `_sqlite_schema_init_lock_for_path` stay on the class. `_initialize_schema_sqlite_legacy` moves with its `TransactionContextManager(self)`, which becomes `ctx.db`.
   - **PostgreSQL:** the whole body of `_initialize_schema_postgres` moves verbatim into `schema_bootstrap.py`, its `with postgres_schema_migration(...)` blocks included, next to the unchanged lock implementation.
   - Pinned by `verify` and `diffpairs` (0 differences outside R18), `test_chacha_postgres_schema_lock.py`, the R26 ladder tests and PROOF_BOOT under `TLDW_TEST_POSTGRES_REQUIRED=1`.

## CI-equivalent checks

- **C1:** `tldw_Server_API/tests/lint` + `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Add the whole `tests/CI` when a workflow or contract file changes.
- **C2:** paste all five outputs into the PR body.
  - `split.py verify <manifests> --base origin/dev`: 0 differences.
  - `diffpairs.py <manifests> --base origin/dev`: exit 0, with its pair list.
  - `ruff check --select F821,F841,F401,F811 --no-force-exclude <moved and edited files>`: clean.
  - `split.py patches <manifests>` and `split.py check <manifests>`: 0, apart from marked non-ChaCha hits.
- **C3:** OWNER_ISO (Appendix B), serial, `TLDW_TEST_POSTGRES_REQUIRED=1`: 0 failed, 0 skipped. Record the wall time.
- **C4:** every file `split.py tests <manifests>` prints (or Task 2's `callers --tests`). PostgreSQL files run with `TLDW_TEST_POSTGRES_REQUIRED=1`.
- **C5:** the ChaCha SQLite suites: `tests/ChaChaNotesDB`, `tests/DB_Management/test_chacha_*.py`, `tests/Notes`.
- **C6:** PROOF_MIG (Tasks 5-8) or PROOF_BOOT (Task 9), with `TLDW_TEST_POSTGRES_REQUIRED=1`.

A run:
```bash
df -g /System/Volumes/Data   # refuse under 100 GiB free
unset OPENAI_API_KEY ANTHROPIC_API_KEY
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 NO_PROXY=localhost,127.0.0.1
BT="$TMPDIR/t22-<task>-<n>"
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$BT" -n 2 $(cat <list>)
rm -rf "$BT"
```

## PR slices and projections

| PR | Task | file lines after | methods | delegated | reaches | changed lines (est.) |
|---|---|---|---|---|---|---|
| A | 1 AC1 ratchet | 47,924 | 863 | 306 | 553 | ~400, tests only |
| B1 | 2a retire moodboard_sync | ~47,880 | 863 | 299 | 553 | ~550 (incl. ~250 tool, pins wiring) |
| B2 | 2b retire conversation_resume | ~47,860 | 863 | 293 | 553 | ~450 |
| C1 | 3 context + conversation/message | ~47,780 | 864 (R19 bump) | 293 | 488 | ~900 (incl. ~200 tool) |
| C2 | 4 context for note/keyword | ~47,450 | ~859 | 293 | ~305 | ~800 (incl. ~200 tool) |
| D | 5 schema SQL constants | ~40,840 | ~859 | 293 | ~305 | ~7,200, script output |
| E | 6 SQLite schema helpers | ~36,070 | ~774 | 293 | ~304 | ~4,900, script output |
| F | 7 PostgreSQL schema helpers | ~29,640 | ~705 | 293 | ~296 | ~6,700, script output + ladder tests |
| G | 8 migrations registry | ~24,850 | ~590 | 293 | ~296 | ~5,100, script output |
| H | 9 bootstrap | ~23,370 | ~588 | 293 | ~296 | ~1,600, script output |
| I1 | 10 workspace callers + façade | ~23,370 | ~588 | 293 | ~296 | ~1,600, codemod |
| I2 | 11 workspace move | ~16,980 | ~391 | 293 | ~296 | ~6,600, script output |
| J1 | 12 flashcard callers + façade | ~16,980 | ~391 | 293 | ~296 | ~1,200, codemod |
| J2 | 13 flashcard move | ~12,820 | ~295 | 293 | ~296 | ~4,300, script output |
| K | 14 close-out | | | | | backlog + doc |

Notes on the table:
- **The reach column:**
  - C1 is 553 − 67 + 2 pinned;
  - C2 is −179 in the stores, −5 for `_supports…` in task and moodboard, +1 pinned;
  - E is −1 for `note_graph_suggestion`;
  - F is −8 for the task and moodboard store reaches into schema methods;
  - H is 0: `_POSTGRES_SCHEMA_VERSION` is pinned (+1), and `_runtime_schema_version`'s pin goes with the public rename (−1, R29).
- **The file lines** follow the dry run's recount: the span plus one blank line per deleted def, with forwarders and imports added.
- **AC4 thresholds** are judged at close-out (R28), with about 12,820 lines and about 295 methods projected.
  - The file target is first met at G, with about 150 lines to spare, and held with margin from H.
  - The method target is met at I2 (~391; one workspace name has 3 defs), and J2 adds margin.
  - The surface ratchet's tightened ceiling holds each gain.
- **"~" values** are estimates. Each PR records the measured numbers.
- **The 400-line rule:** R8's script-plus-verify review replaces it for script-output PRs.
- **Sibling coordination:** every PR from B1 on edits `ChaChaNotes_DB.py`, so they merge strictly in order.
  - A and B1 edit `test_license_first_workflow_contracts.py` and `backend-required.yml`, as do open PRs #3405 and #3294.
  - Merge dev into an open PR before it merges, and take both sides.

---

### Task 1: The delegated-caller ratchet (AC1, AC5)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_charactersragdb_delegated_callers_ratchet.py`, `tldw_Server_API/tests/lint/charactersragdb_delegated_callers_baseline.json`
- Modify: `tldw_Server_API/tests/lint/test_charactersragdb_surface_ratchet.py` (add `delegated_names`)
- Modify: `.github/workflows/backend-required.yml` (the "Enforce CI contracts and code ratchets" step: one path and one comment line)
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`

**Interfaces:**
- Produces, in the surface ratchet module: `delegated_names(tree: ast.Module) -> dict[str, str]` (delegated name → store attribute).
  - It walks the same module-level loops and assignments as `_bindings` and fails closed the same way.
- Produces, in the new module:
  - `RECEIVER = re.compile(r"(^|_)db(\d+|_\w+)?$|database$|^cdb$|^chacha|chachanotes", re.I)`;
  - `COLLISIONS = frozenset({"get_task", "list_tasks", "create_task", "add_keyword", "soft_delete_keyword"})`;
  - `LITERAL_RECEIVERS = frozenset({"chacha_db", "notes_db", "character_db"})`;
  - `last_identifier(node: ast.expr) -> str`, which strips trailing calls and subscripts, then returns an `Attribute`'s attr or a `Name`'s id;
  - `counts(name: str, receiver: ast.expr) -> bool`;
  - `measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]`, keyed by app-relative posix path, with the field `calls`;
  - `RATCHET` and the `_ceilings.main` CLI.

**What counts:** in `root/**/*.py` outside `core/DB_Management/`:
- an `ast.Attribute` in Load context whose attr is a delegated name;
- a `getattr`/`hasattr` call whose second argument is a delegated-name literal.

The receiver (the attribute's value, or the call's first argument) must satisfy all of:
- `last_identifier` matches `RECEIVER`;
- it does not end in `store`;
- it is not `asyncio`;
- for a `COLLISIONS` name, it is in `LITERAL_RECEIVERS`.

Names come from `delegated_names(parse_tree(root)[root/"core/DB_Management/ChaChaNotes_DB.py"])` at test time.

- [ ] **Step 1: Write the failing self-tests.** Seed `tmp_path/"app"` with the surface module's `_DB` source at `core/DB_Management/ChaChaNotes_DB.py` and one app file, as `test_owner_scope_ratchet._seed` does, then call the real `measure(root)`:
  - `test_each_receiver_shape_counts`: `db.add_note()`, `self.db.add_note()`, `self._db.add_note()`, `chacha_db.add_note()`, `db_instance.add_note()` and `get_chacha_db().add_note()` give `{"api/x.py": {"calls": 6}}`.
  - `test_store_and_asyncio_receivers_do_not_count`: `db.note_store.add_note()`, `asyncio.add_note()` and `loop.add_note()` give `{}`.
  - `test_collision_names_count_only_on_literal_receivers`, with `create_task` added to the seeded tuple:
    - `db.create_task()` gives 0;
    - `chacha_db.create_task()` and `self.notes_db.create_task()` give 2.
  - `test_getattr_and_hasattr_probes_count`: `getattr(db, "add_note", None)` and `hasattr(chacha_db, "get_note")` give 2. `getattr(db, name)` gives 0.
  - `test_db_management_is_ignored`: the same calls under `core/DB_Management/chacha/s.py` give `{}`.
  - `test_names_are_read_from_the_class`: a `db.list_notes()` call counts only once `"list_notes"` is added to the seeded tuple.
  - `test_growth_and_stale_entries_fail`: through `_ceilings.violations`, messages contain `"calls 3 > ceiling 2"` and `"stale entry"`.
  - `test_raised_ceiling_without_reason_fails`: through `_ceilings.unreasoned_raises`.
  - In the surface module, `test_delegated_names_lists_each_bound_name`: `delegated_names(ast.parse(_DB)) == {"add_note": "note_store", "get_note": "note_store", "count_notes": "note_store"}`.
- [ ] **Step 2: Run them, and expect FAIL** (no module): `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t22-1-1" tldw_Server_API/tests/lint/test_charactersragdb_delegated_callers_ratchet.py tldw_Server_API/tests/lint/test_charactersragdb_surface_ratchet.py`
- [ ] **Step 3: Implement** `delegated_names` and the ratchet module. Model the module on `test_owner_scope_ratchet.py`, and include these real-tree tests:
  - `test_delegated_callers_do_not_grow` (with `warn_if_tightenable`);
  - `test_raised_delegated_caller_ceilings_carry_a_new_reason`;
  - `test_delegated_names_match_the_surface_count`: `len(delegated_names(...)) == measure()["delegated_methods"]["count"]` from the surface module.

  The `fix=` text: "Call the store, `db.<store>.<method>`, not the delegated `db.<method>` (TASK-13950.22)."
- [ ] **Step 4: Seed** with `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_charactersragdb_delegated_callers_ratchet --write-baseline`. Expect 92 entries totalling 1,051 calls, each with `reason: null`. If dev moved, record the measured numbers and say so.
- [ ] **Step 5: Wire it.**
  - Insert the path mid-list in "Enforce CI contracts and code ratchets", keeping line continuations intact. Add the comment line `# The delegated-caller ratchet (TASK-13950.22) does the same for its per-file call counts.`
  - Add the path to `test_backend_required_enforces_size_ratchets`'s list and to `TEST_SEAM_GATE_SUITES`.
- [ ] **Step 6: Verify.** Run C1. Then run the CLI with `--tighten`; it must be a no-op.
- [ ] **Step 7: Commit** `test(lint): ratchet app callers of delegated CharactersRAGDB names (TASK-13950.22 PR A)`.

### Task 2: Pilot retirement, moodboard_sync then conversation_resume (AC4 first clause)

Two PRs, B1 then B2, each green alone. B1 creates the tool.

**Files:**
- Create: `Helper_Scripts/chacha_split/__init__.py`, `Helper_Scripts/chacha_split/split.py` (the `callers` subcommand, Appendix A.4), `tldw_Server_API/tests/Helper_Scripts/test_chacha_split.py`, `tldw_Server_API/tests/DB_Management/test_chacha_structure_pins.py` (R21)
- Modify (B1): `.github/workflows/backend-required.yml` and `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Both new test files join the "Enforce CI contracts and code ratchets" step, mid-list, and `TEST_SEAM_GATE_SUITES`, with the comment line `# The ChaCha split tool and structure pins (TASK-13950.22) guard each move PR.`
- Modify (B1): `ChaChaNotes_DB.py`: delete the whole moodboard_sync loop (`:47683-47696`), which binds all 7 names, and the 5 stubs of the retired names. The two `resolve_*` names stay bound by the module-level assignments at `:47698-47699`, keeping their stubs. Removing only 5 names from the tuple would land on 301, not 299.
- Modify (B2): `ChaChaNotes_DB.py`: delete the conversation_resume loop (`:47495-47507`) and the `get_roleplay_resume_state` stub.
- Modify: every app and test file `callers` rewrites; the 4 test files with fake `get_roleplay_resume_state` (`tests/Chatbooks/test_openwebui_import_service.py`, `tests/Audio/test_ws_audio_chat_stream.py`, `tests/Chat/unit/test_chat_share_links_api.py`, `tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py`)
- Modify: `charactersragdb_surface_baseline.json` and `charactersragdb_delegated_callers_baseline.json` (`--tighten`)

**Interfaces:**
- Produces: `split.py callers --store ATTR --names N1,N2,… [--check]` (Appendix A.4); Tasks 10 and 12 reuse it.
- Retired names:
  - B1: `bind_local_moodboard_graph_to_dataset`, `bind_local_studio_graph_to_dataset`, `page_moodboards_for_sync_bootstrap`, `page_moodboard_placements_for_sync_bootstrap`, `page_studio_documents_for_sync_bootstrap`.
  - B2: `append_idempotent_user_message`, `prepare_roleplay_resume_generation`, `commit_roleplay_resume_generation`, `put_behavior_snapshot`, `get_conversation_behavior_snapshot`, `get_roleplay_resume_state`.

- [ ] **Step 1 (B1): Write the failing tool tests** in `test_chacha_split.py`. Seed a tmp tree with a mini `ChaChaNotes_DB.py`, an app file and a test file, and run `callers`:
  - `test_attribute_call_is_rewritten_through_the_store`: `db.get_x(1)` becomes `db.s_store.get_x(1)`.
  - `test_store_receiver_is_left_alone`: `db.s_store.get_x()` is unchanged.
  - `test_probe_is_rewritten_through_the_store`: `getattr(db, "get_x", None)` becomes `getattr(getattr(db, "s_store", None), "get_x", None)`, and `hasattr` the same way.
  - `test_class_self_callers_are_rewritten`: inside `class CharactersRAGDB`, `self.get_x()` becomes `self.s_store.get_x()`.
  - `test_collision_name_is_listed_not_rewritten`: a name another app class defines is reported and the file is unchanged.
  - `test_check_flags_a_fake_without_the_store_attribute`: a test class defining `get_x` with no `s_store` makes `--check` exit 1, naming the class.
  - `test_check_flags_assignment_and_class_receiver_references`: `db.get_x = f` and `CharactersRAGDB.get_x` left in a seeded test make `--check` exit 1.
  - `test_check_passes_when_nothing_is_left`.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement `callers`.** Edits use AST positions (Appendix A.3). Print every rewritten receiver that fails Task 1's receiver rule as `unusual receiver`, for review.
- [ ] **Step 4: Run the tool tests, and expect PASS.**
- [ ] **Step 5 (B1): Retire.**
  - Run `PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split callers --store moodboard_sync_store --names <B1 names>`.
  - Delete the loop and the 5 stubs.
  - Retarget any instance or class patch of a retired name to `db.moodboard_sync_store`. There are none at base; the 3 patches target the `resolve_*` names, which stay.
- [ ] **Step 6 (B1): Check** with `split.py callers --store moodboard_sync_store --names <B1 names> --check`. Expect exit 0.
  - Then `--tighten` both ratchets: `delegated_methods` becomes 299, and the AC1 baseline is unchanged.
  - Add `test_retired_names_are_gone` to `test_chacha_structure_pins.py`, import-only:
    - each retired name is not in `vars(CharactersRAGDB)`;
    - `resolve_moodboard_compatibility_dataset_id` still is;
    - each name is a function on `MoodboardSyncStore`.
- [ ] **Step 7 (B1): Verify.**
  - Run C1 and the whole `tests/CI` directory. Also C4 (`callers --tests` lists the files) and the tool tests.
  - Run the two new test files once more under the step's environment (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1`), since they are the first tests in that step to import ChaChaNotes_DB (R31).
  - Bandit on `Helper_Scripts/chacha_split/split.py`.
- [ ] **Step 8 (B1): Commit** `refactor(chacha): retire moodboard_sync delegates; callers use db.moodboard_sync_store (TASK-13950.22 PR B1)`.
- [ ] **Step 9 (B2): Retire conversation_resume the same way**, on a new branch from dev after B1 merges.
  - Run `callers` for the B2 names. Expect 26 app attribute loads, the 4 probes (`audio_streaming.py:824`, `character_chat_sessions.py:4084/4144/9570`) and about 290 test sites.
  - Give each of the 13 fake classes a store attribute, for example `conversation_resume_store = property(lambda self: self)`, so the rewritten calls still reach the fake.
  - Give each `MagicMock(spec=CharactersRAGDB)` that a rewritten call reaches a `conversation_resume_store = MagicMock()`.
  - Retarget the 11 instance patches to `db.conversation_resume_store`; fakes keep their signatures.
  - Delete the loop and the stub.
  - `--check` exits 0. `--tighten` both ratchets: `delegated_methods` 293, and the AC1 total falls by 30 to 1,021.
  - Extend `test_retired_names_are_gone` with the B2 names.
- [ ] **Step 10 (B2): Verify.** Run C1, C4 and C5, plus `tests/Character_Chat_NEW`, `tests/Audio/test_ws_audio_chat_stream.py`, `tests/Chat/unit` and `tests/Chatbooks`. Bandit on the changed app files.
- [ ] **Step 11 (B2): Commit** `refactor(chacha): retire conversation_resume delegates; callers use db.conversation_resume_store (TASK-13950.22 PR B2)`.

### Task 3: `ChaChaContext`; the conversation and message stores receive it (AC2 part 1, AC5)

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/context.py`, `tldw_Server_API/tests/DB_Management/test_chacha_context.py`
- Modify: `ChaChaNotes_DB.py`:
  - add `ctx`, a `functools.cached_property` returning `ChaChaContext(self)` with a function-local import like the stores (R19);
  - `__init__` reads `self.ctx` right after `_resolve_backend`, before building the stores;
  - the helpers the context now implements become one-line forwarders (`return self.ctx.utc_now_iso()`), and a static stays static;
  - rename `_ensure_message_metadata_table` to `ensure_message_metadata_table`;
  - delete the 7 conversation constants.
- Modify: `chacha/conversation_store.py`:
  - constructor `ConversationStore(ctx)`, stored as `self._ctx`;
  - the 7 constants become module constants with unchanged names.
- Modify: `chacha/message_store.py` (constructor `MessageStore(ctx)`), and `chacha/persona_transcript_store.py:177`, which becomes `self.db.ctx.prepare_statement`. That is R14's one existing hit, so `prepare_statement` lands in this task.
- Modify: `Helper_Scripts/chacha_split/split.py`:
  - the `rewire` subcommand;
  - `patches` with its `--names` mode (A.6, R25);
  - the tables `CTX_REWRITES` and `CTX_PUBLIC`.
- Modify: `tests/lint/test_charactersragdb_surface_ratchet.py`:
  - R14's widening, and the M4 known-limit self-test;
  - the AST pins `test_zero_reach_stores_stay_at_zero` and `test_the_context_reaches_only_pinned_db_members`.
- Modify: `charactersragdb_surface_baseline.json`: `--tighten`, plus R19's named `methods` bump to 864.
- Modify: `tests/DB_Management/test_chacha_structure_pins.py` (R21):
  - `test_ctx_exists_on_a_db_built_without_init`;
  - `test_context_surface_is_pinned`;
  - `test_every_split_rewrite_target_is_a_context_member`;
  - `test_context_module_privates_are_pinned`;
  - `test_new_chacha_modules_import_standalone`, parametrized with `chacha.context`.
- Modify tests:
  - the 7 files constructing the real `ConversationStore`/`MessageStore`: `tests/ChaChaNotesDB/test_chacha_conversation_store.py`, `test_chacha_message_store.py`, `tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py`, `tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py`, `tests/Chat/unit/test_chat_history_and_streaming.py`, `tests/DB_Management/test_character_behavior_snapshot_migration.py`, `tests/DB_Management/test_history_selection_transactions.py`;
  - every ChaCha hit of `split.py patches --names <Task 3 helpers>`: 37 sites at base (Measured), minus the 7 non-ChaCha keyword fakes, which stay;
  - the 3 files reading the moved conversation constants.

**Interfaces:**
- Produces `class ChaChaContext` in `chacha/context.py`, with a module docstring stating the rules:
  - every member reads the DB at call time and never captures a bound method;
  - its only private DB reads are `PINNED_DB_PRIVATES`;
  - its only private imports are `PINNED_MODULE_PRIVATES`.
- `PINNED_DB_PRIVATES: frozenset[str] = frozenset({"_CURRENT_SCHEMA_VERSION", "_SCHEMA_NAME"})`
- `PINNED_MODULE_PRIVATES: frozenset[str] = frozenset({"_owner_is_scoped", "_SAFE_IDENTIFIER_RE"})`, imported from ChaChaNotes_DB at the top of `context.py`, with R27's reason in a comment.
- `__init__(self, db: CharactersRAGDB) -> None` stores `self._db`.
- **`CharactersRAGDB.ctx`** is a `cached_property` that builds the context on first use. It works on a DB built with `__new__`.
- **Properties:**
  - `db -> CharactersRAGDB`, for public, cross-domain calls;
  - `backend`, `backend_type`, `client_id`, `owner_user_id`, `db_path_str`, `is_memory_db`;
  - `code_schema_version -> int`, reading `self._db._CURRENT_SCHEMA_VERSION`;
  - `schema_name -> str`, reading `self._db._SCHEMA_NAME`. R2 lists it, and Tasks 6-9 use it.
- **Call-through:** `transaction(*, preserve_existing: bool = False)`, `get_connection()`, `execute_query(query, params=None, **kwargs)`, `execute_many(query, params_list, *, commit=False)`.
- **Helpers:** bodies move from the DB, with `self.<x>` reading the context's own property. Signatures and error-message literals are as on the DB; `test_chacha_owner_scope_required.py` asserts the helper names in messages.
  - `utc_now_iso() -> str`;
  - `new_uuid() -> str`;
  - `@staticmethod normalize_nullable_text(value: Any) -> str | None`;
  - `map_table(table_name: str) -> str`;
  - `owner_filter(owner_client_id: str | OwnerScope | None, alias: str = "", *, table: str = "?") -> tuple[str, tuple[str, ...]]`;
  - `current_db_version(conn, table_name: str, pk_col_name: str, pk_value: Any, *, owner_client_id: str | OwnerScope | None = None) -> int`;
  - `prepare_statement(query: str, params: tuple | list | dict | None = None) -> tuple[str, tuple | list | dict | None]`. It calls `prepare_backend_statement`, with `transform_sqlite_query_for_postgres` as the transformer on PostgreSQL (R27).
- **Store constructors:** `ConversationStore(ctx: ChaChaContext)` and `MessageStore(ctx: ChaChaContext)`. Other stores keep `Store(db)`.
- **Tool `rewire --store <module path>`** rewrites each `self._db.<a>` in that module:
  - a `CTX_REWRITES` key → `self._ctx.<member>`;
  - a `CTX_PUBLIC` name → `self._ctx.<a>`;
  - any other public name → `self._ctx.db.<a>`;
  - bare `self._db` → `self._ctx.db`;
  - an unmapped private name fails closed and lists it;
  - the constructor's `self._db = db` becomes `self._ctx = ctx`.
- **Tool `patches --names N,…`** reports every R25 form for those names, outside any manifest.

**Task 3 decisions** (R2, per member; counts are reaches in these two stores):

| DB member | reaches | destination |
|---|---|---|
| `_normalize_nullable_text` | 26 | `ctx.normalize_nullable_text` |
| `_get_current_utc_timestamp_iso` | 13 | `ctx.utc_now_iso` |
| 7 `_ALLOWED_*`/`_DEFAULT_CONVERSATION_STATE` | 13 | module constants in `conversation_store.py`, the only user |
| `_ensure_message_metadata_table` | 6 | public DB `ensure_message_metadata_table` (init also calls it) |
| `_generate_uuid` | 3 | `ctx.new_uuid` |
| `_get_current_db_version` | 3 | `ctx.current_db_version` |
| `_map_table_for_backend` | 2 | `ctx.map_table` |
| `_CURRENT_SCHEMA_VERSION` | 1 | `ctx.code_schema_version` |

- [ ] **Step 1: Write the failing tests.**
  - **Tool** (`test_chacha_split.py`):
    - `test_patches_finds_each_patch_form`: on a seeded test file, `--names _x` reports each R25 form:
      - `monkeypatch.setattr(db, "_x", f)`;
      - `with patch.object(CharactersRAGDB, "_x")`;
      - `@patch.object(CharactersRAGDB, "_x")`;
      - `db._x = f`;
      - `type(db)._x = f`;
      - `SimpleNamespace(_x=f)`;
      - `patch("m.CharactersRAGDB._x")`;
      - `inspect.getsource(CharactersRAGDB._x)`.

      An unrelated `db._y = f` is not reported.
    - The `rewire` tests: map, public, cross-store public, bare receiver, and unmapped-fails-closed.
  - **`test_chacha_context.py`**, on in-memory SQLite plus a fake-PostgreSQL DB object (`backend_type = BackendType.POSTGRESQL`):
    - `test_db_builds_one_context_before_its_stores`: `db.ctx.db is db`, and `db.conversation_store._ctx is db.ctx is db.message_store._ctx`.
    - `test_context_reads_the_db_at_call_time`:
      - patch `db.execute_query` after construction, and `db.ctx.execute_query("SELECT 1")` hits the patch;
      - `monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 7)` gives `db.ctx.code_schema_version == 7`.
    - `test_map_table_and_owner_filter_by_backend`:
      - SQLite: `map_table("keywords") == "keywords"` and `owner_filter("1") == ("", ())`;
      - fake PostgreSQL: `map_table("keywords") == "chacha_keywords"`, `owner_filter(None, table="notes")` raises `OwnerScopeRequiredError` whose message names `_selected_owner_filter`, and `owner_filter(OwnerScope.NOT_APPLICABLE) == ("", ())`.
    - `test_prepare_statement_matches_the_db`: on fake PostgreSQL, `ctx.prepare_statement(q, p) == db._prepare_backend_statement(q, p)` for a `?`-placeholder query.
    - `test_db_forwarders_follow_a_context_patch`: patch `db.ctx.utc_now_iso` to return `"T"`, and `db._get_current_utc_timestamp_iso() == "T"`.
  - **Pins** (`test_chacha_structure_pins.py`, no DB):
    - `test_ctx_exists_on_a_db_built_without_init` (R19): `db = CharactersRAGDB.__new__(CharactersRAGDB)`. Then `db.ctx.db is db`, `db.ctx is db.ctx`, and `db._sqlite_linear_migration_steps()[65].__name__ == "_migrate_from_v65_to_v66"`.
    - `test_context_surface_is_pinned`: `{n for n in vars(ChaChaContext) if not n.startswith("_")}` equals the Interfaces list, and every member has a docstring.
    - `test_every_split_rewrite_target_is_a_context_member`: each value in `split.CTX_REWRITES` and `CTX_PUBLIC` is an attribute of `ChaChaContext`.
    - `test_context_module_privates_are_pinned`: by AST, the underscore names `context.py` imports from ChaChaNotes_DB equal `PINNED_MODULE_PRIVATES`.
    - `test_new_chacha_modules_import_standalone[chacha.context]`: `subprocess.run([sys.executable, "-c", "import <module>"])` exits 0 in a fresh interpreter, which proves the import cycle is safe.
  - **Surface ratchet** (R14):
    - `test_a_reach_through_a_context_or_db_attribute_counts`, parametrized over `self._ctx.db._x`, `ctx.db._x`, `self._ctx._db`, `self.db._x` and `ctx._x`: each adds 1. `self._ctx.db.public()` and `self._ctx.map_table()` add 0.
    - `test_an_alias_of_the_db_is_a_known_limit` (M4): `d = self._ctx.db; d._x` adds 0, matching the docstring's known limit.
    - `test_zero_reach_stores_stay_at_zero`, over `ZERO_REACH_STORES = ("core/DB_Management/chacha/conversation_store.py", "core/DB_Management/chacha/message_store.py")`: no `sources()` key starts with either.
    - `test_the_context_reaches_only_pinned_db_members`: the private attributes reached in `chacha/context.py` equal `context.PINNED_DB_PRIVATES`.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement** the context, the `ctx` property, the forwarders, the rename, the constants move and the widened `_is_db`, per R14. A private attribute counts when its receiver is a `Name` in `{"db", "ctx"}` or an `Attribute` whose attr is in `{"db", "_db", "ctx", "_ctx"}`.
  - Run `split.py rewire` on the two stores.
  - Fix the test constructions to `ConversationStore(ChaChaContext(fake))`. A fake's helper override that the context now implements is deleted when it equals the context's behaviour; otherwise it becomes a public attribute the context reads (`backend_type`, `client_id`).
  - Run `split.py patches --names <the 8 Task 3 names>`. Retarget every ChaCha hit to `db.ctx.<member>`: call form, assignment form (`db._get_current_utc_timestamp_iso = f` becomes `db.ctx.utc_now_iso = f`) and the rename (`db._ensure_message_metadata_table = …` becomes `db.ensure_message_metadata_table = …`).
  - In the PR body, mark the 7 keyword fakes (Media DB and email ingestion) as non-ChaCha.
- [ ] **Step 4: Run Step 1's tests, and expect PASS.** Then:
  - the surface ratchet with `--tighten`: reaches 553 → 488 (−67 in the stores, +2 pinned in the context);
  - R19's named bump: `methods` 863 → 864.
  - Re-run `patches --names` and expect only the marked non-ChaCha hits.
  - Review `git grep -nE "= .*\.(ctx\.)?db$" tldw_Server_API/app/core/DB_Management/chacha/` for aliases that evade the counter (M4); there must be none.
- [ ] **Step 5: Verify.**
  - Run C1, the whole `tests/CI` directory, C3 and C5.
  - C4 = `git grep -l -E '_get_current_utc_timestamp_iso|_generate_uuid|_normalize_nullable_text|_map_table_for_backend|_selected_owner_filter|_get_current_db_version|_prepare_backend_statement|_ensure_message_metadata_table|_ALLOWED_|_DEFAULT_CONVERSATION_STATE|ConversationStore\(|MessageStore\(' -- tldw_Server_API/tests`, with PostgreSQL required.
  - Bandit on the changed app files.
- [ ] **Step 6: Commit** `refactor(chacha): add ChaChaContext; conversation and message stores receive it (TASK-13950.22 PR C1)`.

### Task 4: The note and keyword stores receive the context (AC2 part 2)

**Files:**
- Modify: `chacha/context.py` (the members below), `chacha/note_store.py` and `chacha/keyword_store.py`:
  - constructors `NoteStore(ctx)` and `KeywordStore(ctx)`;
  - `_SQLITE_PARAM_LIMIT` becomes a module constant in `note_store.py`.
- Modify: `ChaChaNotes_DB.py`, with forwarders, the public renames and the store moves below. Also update callers of the renamed names in `chacha/task_store.py` and `chacha/moodboard_sync_store.py` (`supports_notes_moodboard_studio_v61`).
- Modify: `Helper_Scripts/chacha_split/split.py`: the `store` mode, `verify`, `check` and `tests` (Appendix A.3, A.6), and the `CTX_REWRITES` additions
- Create: `Helper_Scripts/chacha_split/diffpairs.py` (R22), `Helper_Scripts/chacha_split/manifests/t04_note_store.py`, `t04_keyword_store.py`
- Modify: the surface ratchet's `ZERO_REACH_STORES` gains the two stores; `context.PINNED_DB_PRIVATES` gains `_runtime_schema_version`
- Modify tests:
  - the real store constructions in `tests/ChaChaNotesDB/test_chacha_keyword_store.py`, `test_chacha_note_store.py`, `test_notes_moodboard_studio_sync_migration_v61.py` and `tests/DB_Management/test_chacha_postgres_fts.py`;
  - references to the renamed or moved names, which `split.py tests` lists;
  - the 3 ChaCha hits of `patches --names <Task 4 names>`;
  - `test_chacha_owner_scope_required.py`: its `_manage_link` key and message literal become `manage_link`. The other three helper literals stay unchanged in the context bodies.

**Interfaces:**
- **Context members added:**
  - `keyword_link_filter(link_table: str, alias: str, *, owner_client_id) -> tuple[str, tuple[str, ...]]`;
  - `require_owner_row(conn, table_name: str, item_id: Any, owner_client_id, *, include_deleted: bool = False) -> None`;
  - `case_insensitive_order_expression(column: str, direction: str | None = None) -> str` and `case_insensitive_order_clause(column: str, direction: str | None = None) -> str`;
  - `unique_text_match_sql(table_name: str, column: str) -> str`;
  - `is_unique_violation(error: Exception) -> bool`;
  - `@staticmethod chunk_list(lst: list, size: int) -> list[list]`;
  - `@staticmethod canonical_json_text_v61(value: object) -> str` and `@staticmethod note_task_v60_hash(value: object) -> str`;
  - `deserialize_row_fields(row, json_fields: Sequence[str]) -> dict[str, Any] | None`;
  - `runtime_schema_version -> int | None` (a property, `getattr(self._db, "_runtime_schema_version", None)`).
- **Public DB renames:** the private name is gone; callers in the class, stores and tests change:
  - `manage_link` (was `_manage_link`);
  - `add_generic_item`, `get_generic_item_by_unique_text`, `soft_delete_generic_item`, `update_generic_item`;
  - `supports_notes_moodboard_studio_v61`.
- **Moved into stores**, through `store` mode:
  - `_get_generic_item_by_id`, `_list_generic_items` and `_search_generic_items_fts` into `KeywordStore`, the only user;
  - `_invalidate_note_clipper_sidecars` and `_delete_note_clipper_sidecars` into `NoteStore`, the only user.

**Task 4 decisions** (reaches in note + keyword):

| DB member | reaches | destination |
|---|---|---|
| `_selected_owner_filter` | 35 | `ctx.owner_filter` (Task 3) |
| `_map_table_for_backend` | 25 | `ctx.map_table` |
| `_selected_keyword_link_filter` | 21 | `ctx.keyword_link_filter` |
| `_get_current_utc_timestamp_iso` | 13 | `ctx.utc_now_iso` |
| `_require_selected_owner_row` | 10 | `ctx.require_owner_row` |
| `_supports_notes_moodboard_studio_v61` | 10 | public DB `supports_notes_moodboard_studio_v61()` (reads DB init state) |
| `_case_insensitive_order_*` | 9 | `ctx.case_insensitive_order_*` |
| `_manage_link` | 8 | public DB `manage_link` (the class's moodboard opt-outs call it) |
| 4 shared generic-item helpers | 7 | public DB renames (writing_* uses them) |
| 3 keyword-only generic helpers | 6 | moved into `KeywordStore` |
| `_normalize_nullable_text` | 6 | `ctx.normalize_nullable_text` |
| `_get_current_db_version` | 5 | `ctx.current_db_version` |
| 2 clipper sidecar helpers | 5 | moved into `NoteStore` |
| `_chunk_list`, `_SQLITE_PARAM_LIMIT` | 3+3 | `ctx.chunk_list`, note-store module constant |
| `_prepare_backend_statement` | 3 | `ctx.prepare_statement` |
| `_canonical_json_text_v61`, `_note_task_v60_hash` | 3+1 | context statics |
| `_deserialize_row_fields`, `_runtime_schema_version`, `_CURRENT_SCHEMA_VERSION`, `_generate_uuid`, `_unique_text_match_sql`, `_is_unique_violation` | 1 each | context members |

A DB private helper whose last reference outside the class is gone keeps a forwarder only while the class body, another store or a test still uses it. Otherwise delete it.

- [ ] **Step 1: Write the failing tests.**
  - **Tool:** `store` mode moves a seeded method into a seeded store class, rewriting `self.backend_type` to `self._ctx.backend_type` and `self._map_table_for_backend` to `self._ctx.map_table`.
    - `test_verify_passes_on_tool_output`;
    - `test_verify_fails_on_a_hand_edit`, which names the function;
    - `test_diffpairs_lists_receiver_rewrites_and_fails_elsewhere`: a seeded move whose only change is `self._a` → `self._ctx.a` prints that one pair and exits 0. An edit to a literal elsewhere in the body exits 1;
    - `test_diffpairs_shares_no_code_with_split`: by AST, `diffpairs.py` imports nothing from `split`;
    - `test_diffpairs_accepts_a_static_move` and `test_diffpairs_accepts_a_classmethod_move` (R32): a correct move, with the decorator dropped and `cls` removed and `cls.x` → `mod.x`, exits 0. The same move with one changed literal exits 1;
    - `test_patches_watches_module_names_the_moved_bodies_import` (R30): a `monkeypatch.setattr("…ChaChaNotes_DB.time", f)`, a `patch("…ChaChaNotes_DB.time")` and a `setattr(ChaChaNotes_DB, "time", f)` are each reported when a moved body reads `time`, and are not reported otherwise;
    - every self-test that builds a git base (`verify`, `diffpairs`) commits with `git -c user.name=chacha-split-test -c user.email=chacha-split-test@example.invalid -c commit.gpgsign=false commit …` in its subprocess, never relying on global git config (R31);
    - `test_check_lists_an_untouched_reference_to_a_moved_name`;
    - `test_patches_watches_constants_the_moved_bodies_read`: with a manifest, a class patch of a constant a moved body reads is reported;
    - `test_manifest_is_read_without_import`: a manifest whose literal holds a call fails to load with a clear error;
    - `test_tests_lists_files_referencing_moved_names`.
  - **Context:** `test_keyword_link_filter_and_require_owner_row_by_backend` (SQLite gives no predicate; fake PostgreSQL gives the `client_id` predicate), and extend `test_context_surface_is_pinned`.
  - **Surface:** `ZERO_REACH_STORES` gains `note_store.py` and `keyword_store.py`, and the pinned set gains `_runtime_schema_version`.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement** the tool modes, then the context members, the renames and `rewire` on both stores.
  - Write the two manifests. Run `split.py store` for the five moved methods, then C2's checks.
  - Fix the test constructions and references.
- [ ] **Step 4: Run Step 1's tests, and expect PASS.** Surface `--tighten`: reaches about 305 (488 − 179 − 5 + 1) and methods about 859 (864 − 5). Record both.
- [ ] **Step 5: Verify.** C1, C2, C3, C4 (`split.py tests` for the manifests, plus `git grep -l -E '_manage_link|_generic_item|_note_clipper_sidecars|_supports_notes_moodboard_studio_v61|_selected_keyword_link_filter|_require_selected_owner_row|NoteStore\(|KeywordStore\(' -- tldw_Server_API/tests`) and C5. Bandit.
- [ ] **Step 6: Commit** `refactor(chacha): note and keyword stores receive ChaChaContext; no private DB reaches (TASK-13950.22 PR C2)`.

### Task 5: Schema SQL constants move to `chacha/schema_sql.py` (R4, R20)

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/schema_sql.py` (`split.py` output), `Helper_Scripts/chacha_split/manifests/t05_schema_sql.py`
- Modify: `ChaChaNotes_DB.py`:
  - the class constants are deleted, and one class-body `from tldw_Server_API.app.core.DB_Management.chacha.schema_sql import (...)` sits at the first constant's position;
  - the two module constants are deleted, and one module-level import of them from `schema_sql` replaces them at `:346-347` (R20);
  - `split.py` gains the `constants` mode.
- Modify: `tests/lint/test_charactersragdb_surface_ratchet.py` (one self-test), `tests/DB_Management/test_chacha_structure_pins.py` (R21)

**Interfaces:**
- Produces `chacha/schema_sql.py`: 110 constants with names unchanged, in source order, with no imports from `ChaChaNotes_DB`, so both re-imports are cycle-free.
  - The 110 are research B's 107 class constants, plus `_NOTE_TASK_SCOPE_AUTHORITY_TABLE` and the two module constants `RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND`/`_ID`, which they read.
  - A `ClassVar[T]` annotation becomes `T`.
  - Later tasks' moved code reads them as `schema_sql.<NAME>`, a module attribute read at call time.
- Its docstring: "Dependency-free constants of CharactersRAGDB: schema and migration SQL, and constants that moved chacha modules share with the class (TASK-13950.22)."
- The 107 are those the regex in `t22_plan.py` matches, copied into `split.py select schema-sql` (Appendix A.5):
  - the 90 `_MIGRATION_SQL_*`, `_FULL_SCHEMA_SQL_V4`, `_CHAT_MACROS_SCHEMA_SQL`, `_SHARED_WORKSPACE_CHAT_V61_POSTGRES_VERIFY_SQL`;
  - the 2 `_NOTE_GRAPH_SUGGESTION_RECEIPT_DELETE_*`, `_HISTORY_PROJECTIONS_SCHEMA_SQL`, `_RETIRED_SYNC_PLACEHOLDER_REPAIR_SQL`;
  - `_FTS_CONFIG`, `_POSTGRES_SEQUENCE_TABLES`, the 3 `_SQLITE_*_STATEMENTS`, `_SQLITE_CORE_SCHEMA_TABLES`;
  - `_NOTE_TASK_V60_TABLES`/`_RELATIONS`, `_NOTES_MOODBOARD_STUDIO_V61_RELATIONS`.
- `_CURRENT_SCHEMA_VERSION`, `_POSTGRES_SCHEMA_VERSION` and `_SCHEMA_NAME` stay literal on the class.

- [ ] **Step 1: Write the failing tests.**
  - **Tool:** `test_constants_move_keeps_values_and_class_access`, with a seeded module holding:
    - a module constant;
    - a class with a triple-quoted SQL constant with indented lines;
    - an f-string constant over the module constant;
    - a constant built from another (`_B = _A.replace("x", "y")`);
    - a `ClassVar` tuple.

    After `constants`: the target's values equal the originals by `ast.dump`, the class body and the module hold their imports, and `verify` and `diffpairs` pass. Also `test_constants_fail_closed_on_a_foreign_reference`, for a value reading a non-moved name.
  - **Pins** (`test_chacha_structure_pins.py`):
    - `test_schema_constants_stay_class_attributes`: for each class constant in the manifest, `getattr(CharactersRAGDB, n) is getattr(schema_sql, n)`.
    - `test_placeholder_constants_are_reexported`: `ChaChaNotes_DB.RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND is schema_sql.RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND`, and the same for `_ID`.
    - `test_schema_sql_imports_nothing_from_chachanotes_db`: by AST, `schema_sql.py` has no import naming `ChaChaNotes_DB`.
    - `test_versions_stay_literal`: `CharactersRAGDB._CURRENT_SCHEMA_VERSION == 85` and `_POSTGRES_SCHEMA_VERSION == 91`.
    - `test_new_chacha_modules_import_standalone` gains `chacha.schema_sql`.
  - **Surface:** `test_a_class_body_import_is_not_a_method` seeds `from m import X` in `_DB`'s class body, and `methods` stays 5.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement `constants`,** then run it with the manifest, then C2's checks.
  - `patches` must report 0: readers still read the class attribute, so the 6 class patches stay live.
  - No test changes in this PR.
- [ ] **Step 4: Run Step 1's tests, and expect PASS.** The surface ratchet is unchanged; record the file lines (expected about 40,840).
- [ ] **Step 5: Verify.** C1, C2, C3, C5 and C6 (PROOF_MIG), plus `tests/Flashcards/test_apkg_exporter.py` and `tests/ChaChaNotesDB/test_chacha_conversation_store.py`. Record the wall times.
- [ ] **Step 6: Commit** `refactor(chacha): move schema SQL constants to chacha/schema_sql.py (TASK-13950.22 PR D)`.

### Task 6: SQLite schema helpers move to `chacha/sqlite_schema.py` (R5, P4)

**Files:**
- Create (`split.py` output): `chacha/sqlite_schema.py`, `chacha/migrations/__init__.py` (docstring only), `chacha/migrations/support.py`; manifests `t06_sqlite_schema.py`, `t06_migration_support.py`
- Modify: `ChaChaNotes_DB.py` (moved defs deleted; call sites rewritten; shim bodies; end-of-file imports), `chacha/note_graph_suggestion_store.py` (its `_notes_fts_sqlite_contract` reach calls the module, R5), `split.py` (`functions` mode, `select`)
- Modify: `context.py`, adding any member the fail-closed list asks for. From the measured outside references, expect:
  - static context members: `normalize_postgres_catalog_expression`, `note_task_v60_json_safe`, `build_flashcard_search_shadow_fields`;
  - the public DB rename `repair_conversation_assistant_identity`, which is runtime code;
  - `_canonicalize_legacy_task_v60` and `_legacy_moodboard_rule_v61` either join the group or become context members, by A.3's rule.
- Modify: `tests/lint/test_owner_scope_ratchet.py` (R16: `_is_scoped` admits `chacha/**/*.py`, plus a self-test seeding `chacha/sub/x.py`)
- Modify: `function_size_baseline.json` (re-key `_copy_notes_moodboard_studio_graph_v61_sqlite`, `_ensure_study_pack_sync_triggers_sqlite` and `_ensure_workspace_subresource_schema_sqlite`, if `select` puts them here), the surface baseline (`--tighten`), and the layering or logic-import baselines only with the named bumps
- Modify tests: whatever `patches`, `check` and `split.py tests` list. That includes:
  - the `_FULL_SCHEMA_SQL_V4` class patch, since its reader `_apply_schema_v4` moves;
  - `_note_task_v60_migration_checkpoint` (5 class patches) and `_migration_v65_checkpoint` (2);
  - this group's Appendix D sites (for example the `_notes_graph_schema_sqlite` and `_ensure_note_studio_schema_sqlite` `getsource` readers, R26).

**Interfaces:**
- Consumes: Task 3-4's context and `CTX_REWRITES`; Task 5's `schema_sql`.
- Produces:
  - `split.py select {sqlite-schema,postgres-schema,migration-support,migrate-sqlite,migrate-postgres,init}` (Appendix A.5);
  - `split.py functions MANIFEST` (Appendix A.3);
  - module functions with unchanged names and signature `(ctx, conn, …)`, or `(conn, …)` for former static methods and classmethods (R24).
  - Leaf shims stay on the class with one-line forwarders: `_get_db_version`, `_sqlite_table_names`, `_sqlite_column_names`, `_sqlite_index_names`. `select` excludes shims, so fill the manifest's `shims` list by hand.
  - From now on, cross-module calls use `<module>.<name>(…)` attribute access, never a `from`-import of a moved function, so module patches stay live.

- [ ] **Step 1: Write the failing tests.**
  - **Tool `functions` self-tests**, on a seeded class:
    - `test_self_becomes_ctx_and_mapped_members_rewrite`;
    - `test_call_to_moved_function_gets_ctx_inserted` (`self._b(conn)` becomes `mod._b(ctx, conn)`, and a no-arg call becomes `mod._b(ctx)`);
    - `test_static_method_moves_without_ctx`;
    - `test_classmethod_with_moved_cls_refs_becomes_static` and `test_classmethod_with_other_cls_use_fails_closed` (R24);
    - `test_class_receiver_static_call_is_rewritten` (`CharactersRAGDB._s(x)` in a test becomes `mod._s(x)`);
    - `test_class_receiver_unbound_call_is_listed_not_rewritten`;
    - `test_bare_self_becomes_ctx_db`;
    - `test_self_ctx_becomes_ctx`;
    - `test_a_body_binding_ctx_fails_closed`;
    - `test_unmapped_private_fails_closed`, which lists the name and count;
    - `test_multiline_and_fstring_text_is_not_dedented`;
    - `test_free_names_import_from_their_source`;
    - `test_shim_forwards_and_class_call_sites_rewrite`;
    - `test_test_call_sites_take_receiver_ctx` (`db._b(conn)` in a test becomes `mod._b(db.ctx, conn)`);
    - `verify` and `diffpairs` pass on the output, and both fail on a one-token hand edit.
  - **`select`:** `test_select_matches_research_groups_on_a_seed` (a seeded class with one def per bucket).
  - **Pins:** `test_new_chacha_modules_import_standalone` gains `chacha.sqlite_schema` and `chacha.migrations.support`.
  - **Owner-scope:** `test_chacha_subpackage_is_scanned`.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement** `functions` and `select`. Generate both manifests: `select sqlite-schema` (expect about 76 defs) and `select migration-support` (expect 9 defs).
- [ ] **Step 4: Apply** both manifests. Each unmapped private name the tool lists is resolved by the Appendix A.3 rule; record each one in the manifest's `rewrites`.
- [ ] **Step 5: Run C2's checks.**
  - Retarget every `patches` hit to the module, adjusting fake signatures to `(ctx, …)` where an instance patch had none.
  - Fix every `check` hit by hand, by Appendix D's rule; `getsource` retargets follow R26.
  - Re-run both until they are 0.
- [ ] **Step 6: Bookkeeping.**
  - Run each lint ratchet's `--write-baseline` (function size, owner scope) or `--tighten` (surface, logic imports).
  - Check that `git diff` shows only re-keys and the named bumps.
  - The owner-scope baseline must be unchanged.
- [ ] **Step 7: Verify.** C1, C2, C3, C4, C5 and C6 (PROOF_MIG). Bandit.
- [ ] **Step 8: Commit** `refactor(chacha): move SQLite schema helpers to chacha/sqlite_schema.py (TASK-13950.22 PR E)`.

### Task 7: PostgreSQL schema helpers move to `chacha/postgres_schema.py` (R5, P5)

**Files:**
- Create (`split.py` output): `chacha/postgres_schema.py`, manifest `t07_postgres_schema.py`
- Create: `tldw_Server_API/tests/helpers/chacha_postgres_ladder.py` (R26):
  - `postgres_init_function() -> Callable`, which returns `CharactersRAGDB._initialize_schema_postgres` until Task 9 repoints it;
  - `class LadderCall(NamedTuple)`, with fields `name: str` and `kwargs: dict[str, object]`. A literal keyword value is kept as itself via `ast.literal_eval`; any other value is kept as its `ast.unparse` text (R33);
  - `ladder_calls(fn) -> list[LadderCall]`, one per `Call` in `fn`'s body, in source order, read by AST;
  - `branch_calls(fn, version: int) -> list[LadderCall]`, the same, inside the `If` whose test is `current_version < <version>`;
  - `names(calls) -> list[str]`, the names alone, for order checks.
- Modify: `ChaChaNotes_DB.py`; leaf shims stay with forwarders: `_apply_postgres_migration_script`, `_get_schema_version_postgres`, `_set_schema_version_postgres`.
- Modify the reaches R5 names, which `functions` rewrites as outside callers:
  - `chacha/moodboard_sync_store.py`: `_verify_notes_moodboard_studio_schema_postgres`, `_get_schema_version_postgres`;
  - `chacha/task_store.py`: `_verify_note_task_schema_postgres`, `_get_schema_version_postgres`;
  - `core/Character_Chat/chat_dictionary.py`: `_verify_dictionary_schema_postgres`.
- Modify: `function_size_baseline.json`: re-key `_ensure_manuscript_phase2_sync_triggers_postgres`, `_ensure_study_pack_sync_triggers_postgres`, `_verify_note_attachment_schema_postgres`, `_verify_note_task_schema_postgres`, `_verify_notes_moodboard_studio_schema_postgres`.
- Modify tests: whatever `patches` and `check` list. Expect:
  - the 15 `_get_schema_version_postgres` patches and the 15 `_verify_note_*_schema_postgres` patches;
  - `_configure_notes_moodboard_studio_v61_postgres_transaction` (6), `_ensure_postgres_fts` (4), `_ensure_chacha_rls_postgres` (2) and others;
  - this group's Appendix D sites;
  - R30's free-name patches: `tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py:313` and `:332` patch `…ChaChaNotes_DB.time` with `_FakeTime`. They become `…chacha.postgres_schema.time`, because their reader `_postgres_v61_fingerprint_phase_page` (`time.monotonic()`) moves here. If `select` places it in Task 8's group instead, the retarget moves with it.
- Modify, re-expressed and not deleted (R26): the 6 `getsource(CharactersRAGDB._initialize_schema_postgres)` tests:
  - `test_note_attachment_migration_v59.py:921`;
  - `test_note_task_sync_migration_v60.py:31`;
  - `test_notes_organization_migration_v57.py:307`;
  - `test_workspace_source_saved_views_postgres.py:192`;
  - `test_notes_moodboard_studio_sync_postgres_tenancy.py:226`;
  - `Sync/test_sync_v2_notes_link_postgres_contract.py:24`.

  This task's call-site rewrites (`postgres_schema._x(self.ctx, conn)`) break their text matches.

**Interfaces:**
- Consumes: Task 6's `functions` and `select`.
- Produces: `postgres_schema.<name>(ctx, conn, …)`. A shim keeps its name and forwards; a test patches the module function, never the shim, and that covers shim callers too.
- **The ladder helper's contract:**
  - callee names come from `Call.func` attribute or name ids, so `self._x(conn)`, `postgres_schema._x(self.ctx, conn)` and `postgres_steps._x(ctx, conn)` all yield `_x`;
  - keyword arguments are kept, and positional ones are not, so the moved call's extra `ctx` argument changes nothing.

- [ ] **Step 1: Write the tests.**
  - Add `chacha.postgres_schema` to `test_new_chacha_modules_import_standalone`.
  - Re-express each of the 6 source readers through the helper, keeping its intent. For example:
    - `test_note_attachment_migration_v59.py:921`'s "`_get_schema_version_postgres(conn, lock=True)` before `_migrate_from_v58_to_v59_postgres(conn)`" becomes: the first `LadderCall` named `_get_schema_version_postgres` has `kwargs == {"lock": True}`, and its index is below `names(calls).index("_migrate_from_v58_to_v59_postgres")`;
    - "`_ensure_workspace_source_saved_view_schema_postgres` before `_set_schema_version_postgres` inside the v54 step" becomes an index check on `names(branch_calls(fn, 54))`;
    - `"target_version = self._POSTGRES_SCHEMA_VERSION"` becomes "the function reads `_POSTGRES_SCHEMA_VERSION` or `postgres_schema_version`".

    Each re-expressed test must pass on the base before the move (run it on a dev checkout of the file), then pass after.
- [ ] **Step 2: Run the module-import test, and expect FAIL.** Run the 6 re-expressed tests on the unmoved tree, and expect PASS.
- [ ] **Step 3: Generate** the manifest with `select postgres-schema` (expect about 69 defs), then apply it, resolving the fail-closed list as in Task 6.
- [ ] **Step 4: Run C2's checks, and retarget.** Where a test holds two DB instances and patched only one, the module fake checks `ctx.db is <that instance>` and otherwise calls the original.
- [ ] **Step 5: Bookkeeping,** as in Task 6.
- [ ] **Step 6: Verify.** C1-C6, with PROOF_MIG. Bandit, including `chat_dictionary.py`.
- [ ] **Step 7: Commit** `refactor(chacha): move PostgreSQL schema helpers to chacha/postgres_schema.py (TASK-13950.22 PR F)`.

### Task 8: Migrations move to `chacha/migrations/` with a version registry (AC3, R3, R12)

**Files:**
- Create (`split.py` output): `chacha/migrations/sqlite_steps.py`, `chacha/migrations/postgres_steps.py`, manifests `t08_sqlite_steps.py`, `t08_postgres_steps.py`
- Modify: `ChaChaNotes_DB.py`, declared `structural`:
  - `_sqlite_linear_migration_steps` becomes `{v: bind(getattr(sqlite_steps, name), self.ctx) for v, name in SQLITE_STEPS.items()}`;
  - in `_initialize_schema_postgres`, each 73..91 if-chain (fast path `:27234`, full path `:27612`) becomes a loop over `POSTGRES_STEPS`, and the full path keeps its `_runtime_schema_version` writes.
    - The loop takes the keys in order and runs a step for each source `s` with `current_version <= s < target_version`: `getattr(postgres_steps, POSTGRES_STEPS[s])(self.ctx, conn)`, then `current_version = s + 1`.
    - That is the chain's `target_version >= N and current_version < N` with `N = s + 1`.
    - The v72 stamps before each chain stay as they are.

  `_run_sqlite_linear_migration_step` is unchanged. Direct calls in `_initialize_schema_sqlite_legacy` and the legacy PostgreSQL ladder become module calls.
- Modify: `split.py` (`registry-snapshot`), `tests/DB_Management/test_chacha_structure_pins.py` (the registry pins, R21) and `tests/helpers/chacha_postgres_ladder.py`. `ladder_calls` also expands a loop over `POSTGRES_STEPS` into the registry's names in key order, so the R26 tests keep seeing the 72→91 steps.
- Modify: `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md` §2 and §5 ("Schema Versioning and Initialization"):
  - where schema code and migrations live;
  - how to add a migration: a function in `sqlite_steps`/`postgres_steps`, a registry key, the version constant and the frozen pin;
  - what the context is.

  Then run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff.
- Modify tests:
  - the 63 direct calls (26 files) and the 17 patches (13 files);
  - the 3 `_sqlite_linear_migration_steps` class patches whose fakes return moved methods;
  - the migration-SQL class patches whose reader moves (`_V61_TO_V62` ×2, `_V85_TO_V86_POSTGRES`, `_V81_TO_V82`, `_V63_TO_V64`), by Global Constraints' constant rule;
  - this group's Appendix D sites, including the `_migrate_*` `getsource` readers (R26) and `test_sync_placeholder_assistant_migration.py:92`'s `step.__func__ is CharactersRAGDB._migrate_from_v75_to_v76`, which becomes `step.__wrapped__ is sqlite_steps._migrate_from_v75_to_v76`.

**Interfaces:**
- Produces, in `chacha/migrations/__init__.py`: `bind(step: Callable[[ChaChaContext, Any], None], ctx: ChaChaContext) -> Callable[[Any], None]`. The result calls `step(ctx, conn)` and copies `__name__`/`__qualname__`/`__doc__`/`__wrapped__` via `functools.update_wrapper`.
- Produces `sqlite_steps.SQLITE_STEPS: dict[int, str]` (4..84) and `postgres_steps.POSTGRES_STEPS: dict[int, str]` (72..90) per Appendix C.
- Produces `sqlite_steps.migrate_sqlite_transcripts(ctx, conn) -> None`, the adapter for key 84.
  - It calls `persona_transcript_store.migrate_sqlite_transcripts(conn, schema_name=ctx.schema_name, get_version=…)` through the module attribute at call time, so `test_persona_transcript_migration.py`'s module patch stays live.
  - For `get_version` it passes a callable over `sqlite_schema._get_db_version`.
  - On dev, key 84 is a `functools.partial` with no `__name__`. So the adapter's `__name__` is a deliberate new behaviour, which the frozen pin records (M9, N9).
- Migration functions keep their names, with signature `(ctx, conn)`.
- `split.py registry-snapshot` prints the two frozen lists from a tree, by AST:
  - keys 4..46 from the dict literal, 47..83 from the loop tuple, and 84 from the `steps[84] = bind_migration(migrate_sqlite_transcripts, …)` assignment, read as its first argument's name;
  - PostgreSQL from the full-path ladder. It asserts that the fast path yields the same list.

- [ ] **Step 1: Snapshot on the base,** before any edit: `PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split registry-snapshot`.
  - Implement the subcommand first, with its self-test on a seeded class.
  - The output must match Appendix C; if it does not, the snapshot wins and the PR says so.
  - Paste it as `FROZEN_SQLITE`/`FROZEN_POSTGRES` literals into the pins file.
- [ ] **Step 2: Write the failing pins** in `test_chacha_structure_pins.py`. They import code only and build no DB:
  - `test_registries_match_dev_before_the_move`: `list(SQLITE_STEPS.items()) == FROZEN_SQLITE` and `list(POSTGRES_STEPS.items()) == FROZEN_POSTGRES`.
  - `test_every_registry_name_resolves`: `getattr(sqlite_steps, n).__name__ == n` for each value, and the same for PostgreSQL.
  - `test_versions_unchanged`: `max(SQLITE_STEPS) + 1 == CharactersRAGDB._CURRENT_SCHEMA_VERSION == 85`, and `max(POSTGRES_STEPS) + 1 == CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 91`.
  - `test_class_steps_keep_names`: `CharactersRAGDB.__new__(CharactersRAGDB)._sqlite_linear_migration_steps()` maps each key to a callable whose `__name__` equals `FROZEN_SQLITE`'s (R19's `__new__` path).
  - `test_no_migrate_name_left_on_the_class`: no `vars(CharactersRAGDB)` key starts with `_migrate`.
  - `test_module_patch_reaches_registry_and_direct_calls`: patch `sqlite_steps._migrate_from_v49_to_v50` to record calls. Both `CharactersRAGDB.__new__(CharactersRAGDB)._sqlite_linear_migration_steps()[49](conn)` and a resolved `getattr(sqlite_steps, SQLITE_STEPS[49])` hit the fake.
  - `test_new_chacha_modules_import_standalone` gains `chacha.migrations.sqlite_steps` and `chacha.migrations.postgres_steps`.
- [ ] **Step 3: Run them, and expect FAIL.**
- [ ] **Step 4: Generate and apply** `select migrate-sqlite` (expect 80) and `select migrate-postgres` (expect 35). Then write the registries, `bind`, the adapter, the ladder-helper loop expansion and the structural class edits by hand.
- [ ] **Step 5: Run C2's checks.** The only `verify`/`diffpairs` differences are the two declared structural defs; review their unparsed diff in the PR body. `patches` and `check` are 0 after retargeting.
- [ ] **Step 6: Bookkeeping.**
  - Re-key `function_size_baseline.json` (`_migrate_from_v55_to_v56_postgres`, `_v56_to_v57_postgres`, `_v62_to_v63_postgres`).
  - Run surface `--tighten`; expect methods about 590.
  - Record the file lines, expected about 24,850. AC4 is judged at T14 (R28), so the margin is not chased here.
- [ ] **Step 7: Verify.**
  - C1-C6 (PROOF_MIG), plus `tests/DB_Management/test_persona_transcript_migration.py`, `tests/ChaChaNotesDB/test_persona_visuals_db.py` and `tests/Persona/test_independent_buddies.py`, with PostgreSQL required.
  - Upgrade paths: confirm that at least one PROOF_MIG file per backend starts below v72, and one at v86 (the dictionary bridge), from their fixtures. Name them in the PR.
- [ ] **Step 8: Commit** `refactor(chacha): migrations move to chacha/migrations with a version registry (TASK-13950.22 PR G)`.

### Task 9: Schema bootstrap moves to `chacha/schema_bootstrap.py` (R5, P6, the riskiest)

**Files:**
- Modify (`split.py` output): `chacha/schema_bootstrap.py` (existing; `postgres_schema_migration` stays). Manifest `t09_bootstrap.py`. It gains:
  - `_initialize_schema_sqlite_legacy(ctx, *, target_version)`;
  - `_postgres_schema_is_current(ctx, conn)`;
  - `_initialize_schema_postgres(ctx)`, the shim's body;
  - `_POSTGRES_SCHEMA_BOOTSTRAP_LOCK_TIMEOUT`, whose class attribute goes if no reader is left.
- Modify: `ChaChaNotes_DB.py`. `_initialize_schema_postgres` becomes a one-line shim.
  - These stay unchanged: `_initialize_schema`, `_initialize_schema_sqlite` (it holds the SQLite init lock), `_sqlite_schema_init_lock_for_path`, `_SQLITE_SCHEMA_INIT_LOCKS` and `_ensure_bootstrap_for_backend`.
  - `_initialize_schema_sqlite_legacy`'s `TransactionContextManager(self)` moves with it, as `TransactionContextManager(ctx.db)`.
  - Delete the dead branch `:23304-23309` (R18), declared `structural`.
  - R29's offset: rename the instance attribute `_runtime_schema_version` to a public `runtime_schema_version`, with a class-level default `runtime_schema_version: int | None = None` so `__new__`-built DBs read `None`. That covers the 24 class uses, which the manifest declares under `"renames"` so `verify` accepts them, plus `test_notes_provenance.py`'s 5 reads. A class-level annotation is not a method.
- Modify: `context.py`:
  - `postgres_schema_version -> int` (reads `_POSTGRES_SCHEMA_VERSION`);
  - `runtime_schema_version` now reads the public `self._db.runtime_schema_version`;
  - `PINNED_DB_PRIVATES` gains `_POSTGRES_SCHEMA_VERSION` and loses `_runtime_schema_version`: net 0, so reaches stay 296;
  - moved bootstrap code writes `ctx.db.runtime_schema_version = N`, a public attribute, so the context needs no setter;
  - plus any member the fail-closed list asks for. A new pin must be offset in this PR (R29).
- Modify: `function_size_baseline.json` (re-key `_initialize_schema_sqlite_legacy` and `_initialize_schema_postgres`, with its body)
- Modify: `tests/helpers/chacha_postgres_ladder.py`. `postgres_init_function()` returns `schema_bootstrap._initialize_schema_postgres`, a one-line change, and the R26 tests pass unchanged.
- Modify: `tests/DB_Management/test_chacha_structure_pins.py` (the pins below, and `chacha.schema_bootstrap` in the standalone-import test)

**Interfaces:**
- Consumes everything from Tasks 3-8.
- **Unchanged:** the class's `_initialize_schema*` shims are the only entry points.
  - Tests keep their 35 patches of `_initialize_schema` and 1 of `_initialize_schema_sqlite`.
  - Their references to `_initialize_schema_postgres` are calls and patches, which stay live through the shim. The source readers were re-expressed in Task 7.
- **Versions** are read at call time through `ctx.postgres_schema_version`/`ctx.code_schema_version`, because tests patch them on the class 139 times.

**The invariants this task must keep**, each checked by `verify`/`diffpairs` (a verbatim body) and by a named test:
- the PostgreSQL session advisory lock and its timeout. The `with postgres_schema_migration(...)` blocks move with the body, and the lock implementation is unchanged (`test_chacha_postgres_schema_lock.py`);
- the order of the legacy-marker repairs: v86 dictionary bridge → v68 history projections → v73/74 stamp to 72. This is a new AST-only, R26-style test, `test_postgres_init_repairs_legacy_markers_in_order`, in `test_chacha_structure_pins.py`, so it runs in the required gate (R34);
- the `runtime_schema_version` writes, in order;
- the SQLite init lock, which stays in the class, and the legacy init's transaction, which moves with its body.

- [ ] **Step 1: Write the failing tests:**
  - in the pins file, `test_bootstrap_reads_versions_at_call_time`: on a `__new__`-built DB, patch `CharactersRAGDB._POSTGRES_SCHEMA_VERSION`, and `db.ctx.postgres_schema_version` follows;
  - `test_runtime_schema_version_is_public_with_a_default`: `CharactersRAGDB.__new__(CharactersRAGDB).runtime_schema_version is None`, and `db.ctx.runtime_schema_version` reads the public attribute;
  - extend `test_the_context_reaches_only_pinned_db_members` (the set gains `_POSTGRES_SCHEMA_VERSION` and loses `_runtime_schema_version`);
  - in the pins file, write `test_postgres_init_repairs_legacy_markers_in_order` against the unmoved tree first, and expect PASS.
- [ ] **Step 2: Run them, and expect FAIL** for the context tests.
- [ ] **Step 3: Generate and apply** `select init` plus the `_initialize_schema_postgres` body, then the structural deletion.
- [ ] **Step 4: Run C2's checks.** The only differences are R18's deleted lines. `patches` and `check` are 0.
- [ ] **Step 5: Bookkeeping,** as in Task 6. Expect about 23,370 file lines, about 588 methods and reaches at 296, the same as base. The surface ratchet must pass without a reach bump.
- [ ] **Step 6: Verify.** C1-C5, plus C6 with PROOF_BOOT and `TLDW_TEST_POSTGRES_REQUIRED=1`, and OWNER_ISO. Record each wall time. Bandit.
- [ ] **Step 7: Commit** `refactor(chacha): schema bootstrap moves to chacha/schema_bootstrap.py (TASK-13950.22 PR H)`.

### Task 10: Workspace callers go through `db.workspace_store` (R6, R15, PR I1)

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/workspace_store.py`, a façade only:
  - `_MOVING: frozenset[str]`, the manifest's names;
  - `class WorkspaceStore`, with `__init__(self, ctx)` storing `self._ctx`;
  - `__getattr__(self, name)` returns `getattr(self._ctx.db, name)` for a name in `_MOVING`, else raises `AttributeError(name)`;
  - a `# ponytail: transitional façade, replaced by the moved bodies in TASK-13950.22 T11` comment.
- Create: manifest `t10_workspace.py`, from `split.py select workspace` (Appendix A.5; 195 names and 197 defs at base). Task 11 reuses it, extended by the fail-closed list.
- Modify: `ChaChaNotes_DB.py` `__init__`: `self.workspace_store = WorkspaceStore(self.ctx)`.
- Modify: every app file outside the class, DB_Management modules included, and every test file `callers --store workspace_store` rewrites. Class-internal `self.<m>` callers stay until Task 11.
- Modify: `tests/DB_Management/test_chacha_structure_pins.py` (R21): the façade pins and `chacha.workspace_store` in the standalone-import test

**Interfaces:**
- Produces `db.workspace_store.<m>` for each moving name; behaviour is unchanged. Instance and class patches on the DB stay live through the façade's call-time `getattr`, so this PR retargets no patches.

- [ ] **Step 0: Check the margin (M2).** Record `select workspace`'s def count `W`. If `588 − W` (Task 9's measured `methods` minus `W`) is 400 or more, stop and report: Task 11 would miss AC4's method target.
- [ ] **Step 1: Write the failing pins:**
  - `test_facade_forwards_moving_names_at_call_time`: on a `__new__`-built DB with `workspace_store` set, an instance patch on `db.<m>` is what `db.workspace_store.<m>` returns.
  - `test_facade_rejects_other_names`: `db.workspace_store.add_note` raises `AttributeError`.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement** the façade, then run `callers --store workspace_store --names $(…)` over app and tests.
  - The tool lists the collision names: methods of another app class, such as `update_workspace` on `Orchestration_DB`. Endpoint handlers that share a name are module functions, which no receiver reaches, so they are not collisions. Edit those sites by hand only when the receiver is a ChaCha DB, and record each decision in the PR body.
  - Review every `unusual receiver` line.
  - Give each `MagicMock(spec=CharactersRAGDB)` that a rewritten call reaches a `workspace_store` attribute (Global Constraints).
- [ ] **Step 4: Run `--check`, and expect 0,** apart from the hand-decided collision sites it prints as `collision`.
- [ ] **Step 5: Verify.** C1, the whole `tests/CI` directory, C3, C4 (`callers --tests`) and C5, plus `tests/Workspaces`, `tests/ChaChaNotesDB/test_workspace_*` and `tests/Notes`. Bandit.
- [ ] **Step 6: Commit** `refactor(chacha): workspace callers use db.workspace_store (TASK-13950.22 PR I1)`.

### Task 11: Workspace methods move into `WorkspaceStore` (R6, AC4 methods target, PR I2)

**Files:**
- Modify (`split.py store` output): `chacha/workspace_store.py`, with the façade replaced by the moved methods, and `ChaChaNotes_DB.py`, with the defs deleted and class-internal callers becoming `self.workspace_store.<m>`.
- Modify: constants only the workspace group reads (`_WORKSPACE_*`) become module constants in `workspace_store.py`. Shared ones move to `schema_sql.py` and are re-imported in the class body.
- Modify: `db_management_logic_imports.txt`: re-key `ChaChaNotes_DB -> core.Sharing.clone_models` to `chacha.workspace_store` (named bump). Re-key the `function_size` and keyword-identity entries if listed. Surface `--tighten`.
- Modify tests:
  - retarget every `patches` hit. An instance patch goes to `db.workspace_store`; a class patch goes to `WorkspaceStore`, and its fake's first parameter is now the store;
  - R30's free-name patch: `tests/ChaChaNotesDB/test_source_review_plans.py:927` patches `…ChaChaNotes_DB.build_source_review_launch_metadata`, which the workspace group reads, and becomes `…chacha.workspace_store.build_source_review_launch_metadata`;
  - fix the workspace Appendix D sites. The 7 static and 3 classmethod calls become `WorkspaceStore.<m>(…)`; `test_native_fork_workspace_lifecycle.py:450`'s unbound call is fixed by hand;
  - the pins file swaps the façade pins for `test_workspace_methods_live_on_the_store`.

**Interfaces:**
- Produces `WorkspaceStore(ctx)` with the moved methods. Classmethods stay classmethods (R24).
- **Receiver rewrites:** `self.<db public>` becomes `self._ctx.<…>` or `self._ctx.db.<…>`. DB privates go through `CTX_REWRITES`, resolving each unmapped one by Appendix A.3. Expect `_note_task_v60_json_safe`, `_coerce_mapping_row`, `_ensure_json_string_from_mixed` and the note-folder helpers.

- [ ] **Step 1: Write the failing pin** `test_workspace_methods_live_on_the_store`: no manifest name is in `vars(CharactersRAGDB)`; each is in `vars(WorkspaceStore)`; `WorkspaceStore` has no `__getattr__`.
- [ ] **Step 2: Run it, and expect FAIL.**
- [ ] **Step 3: Apply `store` mode,** then C2's checks, and retarget until `patches` and `check` are 0.
- [ ] **Step 4: Bookkeeping.** The surface ratchet must read `methods` under 400 (about 391). Record it.
- [ ] **Step 5: Verify.** C1-C5, as in Task 10. Bandit.
- [ ] **Step 6: Commit** `refactor(chacha): workspace methods move into WorkspaceStore (TASK-13950.22 PR I2)`.

### Task 12: Flashcard callers go through `db.flashcard_store` (R6, PR J1)

The same as Task 10, for `split.py select flashcards` (96 defs at base): `chacha/flashcard_store.py`, `FlashcardStore`, `db.flashcard_store`, manifest `t12_flashcards.py`. `chacha.flashcard_store` joins the standalone-import test (M8).
- **Gate:** `gh pr view 3512 --json state` must be `MERGED` before branching. If #3512 is still open when Task 11 merges, branch from its head; after it merges, rebase by re-running `callers` on dev.
- **Collisions:** the fake classes in `core/MCP_unified/tests/test_flashcards_module.py` define the same names; the tool lists them. Edit those sites by hand, as in Task 10. That file's `CharactersRAGDB._flashcard_visibility_filter` (Appendix D) waits for Task 13.
- **Verify:** add `tests/Flashcards`, `tests/StudyPacks` and `tests/MCP_unified` to Task 10's list.
- **Commit:** `refactor(chacha): flashcard callers use db.flashcard_store (TASK-13950.22 PR J1)`.

### Task 13: Flashcard methods move into `FlashcardStore` (R6, PR J2)

The same as Task 11, for the flashcard manifest:
- **Expected:** about 12,820 file lines and about 295 methods.
- **Constants:** `_STUDY_PACK_JSON_FIELDS` and `_SUGGESTION_SNAPSHOT_JSON_FIELDS` become module constants if only the group reads them.
- **R30's free-name patch:** `tests/Flashcards/test_flashcards_endpoint_integration.py:3044` patches `…ChaChaNotes_DB._normalize_scheduler_settings_envelope`, which the flashcard group reads. It becomes `…chacha.flashcard_store._normalize_scheduler_settings_envelope`.
- **Appendix D:** `app/core/MCP_unified/tests/test_flashcards_module.py:51`'s unbound `CharactersRAGDB._flashcard_visibility_filter(...)` is fixed by hand.
- **Commit:** `refactor(chacha): flashcard methods move into FlashcardStore (TASK-13950.22 PR J2)`.

### Task 14: Close-out (AC1-AC5, DoD)

**Files:**
- Modify (backlog-py only): `backlog/tasks/task-13950.22 - Freeze-and-shrink-the-CharactersRAGDB-god-class.md`; create the follow-ups.

- [ ] **Step 1: Check each AC on dev,** recording the command and the numbers.
  - **AC1:** the ratchet's seed (1,051) and its current total.
  - **AC2:** `test_zero_reach_stores_stay_at_zero`, and the surface ratchet's `private_member_reaches`.
  - **AC3:** the registry pins, versions 85/91, and PROOF_MIG passing on both backends with `TLDW_TEST_POSTGRES_REQUIRED=1`.
  - **AC4** (R28): file lines (`wc -l`) under 25,000, projected about 12,820; the surface `methods` under 400, projected about 295; and `delegated_methods` 293.
  - **AC5:** the contract tests for Task 1's ratchet and Task 2's two pin files, which hold every guard this task added (R21).
- [ ] **Step 2: Amend the AC text** with `task edit`:
  - AC4: "Delegated methods of moodboard_sync and conversation_resume are retired as a pilot; the other seven stores retire in follow-ups TASK-…; ChaChaNotes_DB.py ends under 25,000 lines and the class under 400 methods."
  - AC1: add "app code; `getattr`/`hasattr` probes count; tests are not counted".
  - AC3 (N10, M9): "The 115 `_migrate_*` methods move to `chacha/migrations/`, with version → function-name registries (`SQLITE_STEPS`, `POSTGRES_STEPS`) resolved to callables at call time (R12)…", the rest unchanged.
- [ ] **Step 3: File the follow-ups,** each depending on TASK-13950.22:
  - **seven retirements, one per store**, cheapest first: task (26 names), character (29), keyword (33), conversation (33), message (44), note (36), persona_state (92). Each follows Task 2's recipe;
  - "Move the remaining stores onto ChaChaContext" (character, task, persona_state, moodboard_sync and the non-delegated stores);
  - "Move the note-helper, quiz/OSCE, writing and skills domains out of CharactersRAGDB" (research B's P9);
  - "Delete DB private forwarders with no remaining caller";
  - "Count tests in the delegated-caller ratchet" (R1's cost-if-wrong);
  - "Make `test_list_notes` deterministic", which flakes on a same-timestamp UUID tie-break.
- [ ] **Step 4: Record** in the notes and the final summary: the rulings, the PR numbers, the measured counts per PR, the proof-set wall times, and the bandit results. Set the task to Done. Stage with `git add -A backlog/tasks`.

## Self-review

Rerun after fix round 2.

- **Spec coverage:**

  | AC | Task |
  |---|---|
  | AC1 | Task 1, plus R13 |
  | AC2 | Tasks 3-4, with the zero-reach and pinned-context tests |
  | AC3 | Tasks 6-8: the registry, versions 85/91, PROOF_MIG on both backends; the AC text is amended in Task 14 |
  | AC4 | Tasks 5-13 for the numbers, judged at Task 14 (R28); Task 2 for the pilot; Task 14 for the amendment (R7) |
  | AC5 | Task 1 (the ratchet's contract); Task 2 (both pin files in the step and `TEST_SEAM_GATE_SUITES`, R21); Tasks 3-13 add their guards to those files; Tasks 3 and 6 change ratchets already in the step |
  | DoD | Task 14; the docs update is in Task 8 |

- **Names across tasks:**
  - `ChaChaContext`, `db.ctx` (a `cached_property`), `PINNED_DB_PRIVATES`, `PINNED_MODULE_PRIVATES` and `ZERO_REACH_STORES` come from Task 3 and grow later:
    - context members in Tasks 4, 6, 7, 9, 11 and 13;
    - the DB pinned set in Tasks 4 and 9;
    - the zero-reach list in Task 4.

    `test_context_surface_is_pinned` changes in each of them.
  - `CTX_REWRITES`/`CTX_PUBLIC` start in Task 3, and `test_every_split_rewrite_target_is_a_context_member` keeps them in step.
  - Module aliases (`schema_sql`, `sqlite_schema`, `postgres_schema`, `sqlite_steps`, `postgres_steps`, `migration_support`, `schema_bootstrap`) are fixed in Appendix A.2.
  - `SQLITE_STEPS`/`POSTGRES_STEPS` are `dict[int, str]` everywhere (R12).
  - Pins go to `test_chacha_structure_pins.py` and tool tests to `test_chacha_split.py`, from Task 2 on (R21).
  - Manifests are `.py` literals (R23).
  - `LadderCall`, `ladder_calls`, `branch_calls`, `names` and `postgres_init_function` are in `tests/helpers/chacha_postgres_ladder.py`, from Task 7. Both call functions return `list[LadderCall]` everywhere (R33).
  - `runtime_schema_version` is a public DB attribute from Task 9 on. Before Task 9, `_runtime_schema_version` is read only through the context's pinned getter, added in Task 4 (R29).
  - The reach column never rises between PRs: 553, 553, 553, 488, 305, 305, 304, 296, 296, 296, … (R29).
- **Fails-before:**
  - Task 1's self-tests, before the module exists;
  - Task 2's `test_retired_names_are_gone`;
  - Task 3's `test_ctx_exists_on_a_db_built_without_init`, `test_db_builds_one_context_before_its_stores`, `test_patches_finds_each_patch_form` and the R14 self-tests;
  - Task 5's `test_schema_constants_stay_class_attributes` and `test_placeholder_constants_are_reexported`;
  - Task 8's registry pins;
  - Task 11's `test_workspace_methods_live_on_the_store`.
- **Passes before and after (R26):**
  - Task 7's 6 re-expressed ladder tests;
  - Task 9's `test_postgres_init_repairs_legacy_markers_in_order`, in the pins file (R34).
- **Proportion:** a few signatures and tables, with no function bodies. The algorithm is in Appendix A, because R8 and R22 rest on it.

---

## Appendix A: `Helper_Scripts/chacha_split/split.py`

**A.1 Shape.**
- A stdlib-only CLI (`ast`, `tokenize`, `argparse`, and `subprocess` for `git show`, with a fixed argv and a `# nosec` reason as in `_ceilings._git`).
- It refuses to run on Python below 3.12 (R22).
- Run it as `PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split <cmd>`.
- **`Helper_Scripts/chacha_split/diffpairs.py`** (R22) is a separate stdlib file that imports nothing from `split.py`. Run it as `… -m Helper_Scripts.chacha_split.diffpairs <manifests> --base REF`.
  - For each moved def, it parses the base def and the head def.
  - **It normalizes only the header** (R32):
    - it renames the receiver (`self` → `ctx`; nothing in store mode);
    - it drops a `staticmethod` decorator from the base;
    - for a classmethod moved as a module function, it drops the `classmethod` decorator and removes the base's first parameter `cls`.

    The body is not normalized.
  - **The walk** goes over both trees in parallel and descends while node types and field lengths match. At the first mismatch under a node, it records `(ast.unparse(old), ast.unparse(new))` and does not descend further.
  - **Each recorded pair is classified by the old side's text before the rename.** The allowed roots are `self.<attr>`, a call on one, a bare `self`, and `cls.<attr>` or a call on one. Any other root exits 1, so a changed literal, operator or control flow fails, in a static or classmethod move as anywhere else.
  - It prints each distinct pair with its count, and exits 0 when every pair has an allowed root.
  - The manifest's `renames` pairs are allowed roots too.
- Tests: `tldw_Server_API/tests/Helper_Scripts/test_chacha_split.py`, on seeded tmp trees. It is in the CI-contracts step (R21).
- **Each subcommand lands in the first task that uses it:**
  - `callers` and `check` for retirements (T2);
  - `rewire` and `patches` (T3);
  - `store`, `verify`, the manifest-wide `check`, `tests` and `diffpairs.py` (T4);
  - `constants` and `select schema-sql` (T5);
  - `functions` and the other `select` groups (T6; `workspace`/`flashcards` in T10/T12);
  - `registry-snapshot` (T8).

**A.2 Manifest** (`Helper_Scripts/chacha_split/manifests/t<NN>_<name>.py`, R23). It holds one literal, read with `ast.literal_eval`, never imported:
```python
MANIFEST = {"task": "T6", "mode": "functions",
            "source": "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
            "target": "tldw_Server_API/app/core/DB_Management/chacha/sqlite_schema.py",
            "alias": "sqlite_schema", "store_class": None, "store_attr": None,
            "names": ["..."], "shims": ["_get_db_version"], "rewrites": {"_x": "ctx.y"},
            "renames": {}, "structural": []}
```
- **Aliases:** `schema_sql`, `sqlite_schema`, `postgres_schema`, `migration_support` (`chacha.migrations.support`), `sqlite_steps`, `postgres_steps`, `schema_bootstrap`.
- **Every committed manifest** is read for `MOVED`, the earlier moves a later task's rewrite needs.
- **`renames`** maps a DB attribute renamed in the same PR, as in Task 9's `_runtime_schema_version`. `verify` and `diffpairs` treat it as a declared rewrite in moved and remaining defs alike.

**A.3 `functions` (and `store`) apply rules,** for each moved def, in order:
1. **Span:** from the first decorator to `end_lineno`, plus the comment lines directly above with no blank line between.
2. **Decorators:**
   - drop `@staticmethod`;
   - a `@classmethod` follows R24, with `cls.<x>` rewritten like `self.<x>` in rule 4 and the decorator dropped; in `store` mode it is kept;
   - any other decorator fails closed;
   - a body that binds a name `ctx` fails closed.
3. **First parameter:** `self` becomes `ctx` (in `store` mode it stays `self`).
4. **Each `self.<a>`:**
   - `a == "ctx"` → `ctx`;
   - `a` in `MOVED` as a module function → `<alias>.<a>`, and a call gets `ctx, ` inserted after its `(` (`ctx` alone when there are no args), unless the target was static;
   - `a` in `MOVED` as a store method → `ctx.db.<store_attr>.<a>`;
   - `a` in `CTX_REWRITES` or the manifest's `rewrites` → the mapped text;
   - `a` is a `schema_sql` name → `schema_sql.<a>`, read at call time;
   - `a` in `CTX_PUBLIC` → `ctx.<a>`;
   - another public name → `ctx.db.<a>`;
   - another private name → fail closed, listing each name and count.

   In `store` mode, read `self._ctx` for `ctx`, and a name in the group stays `self.<a>`.
5. **Bare `self`** (a Load that is not an attribute receiver) → `ctx.db` (`self._ctx.db` in `store` mode).
6. **Dedent** by 4 every line except continuation lines of a multi-line string or f-string token (found with `tokenize` on Python 3.12+, which tokenizes f-strings), so no string value changes. `store` mode does not dedent.
7. **Edits** use AST positions, with `col_offset` in UTF-8 bytes, applied right to left.
8. **Target module:** a docstring naming TASK-13950.22, `from __future__ import annotations`, and imports for each free name:
   - a name ChaChaNotes_DB.py imports is imported the same way;
   - a name it defines at module level is imported from it;
   - anything else fails closed.
9. **Source class:**
   - each moved def is deleted;
   - a shim's body becomes `return <alias>.<name>(self.ctx, <params>)`, or without `self.ctx` when the target is static;
   - a remaining `self.<moved>(…)` becomes `<alias>.<moved>(self.ctx, …)` (`self.<store_attr>.<moved>(…)` in `store` mode);
   - a non-call reference fails closed unless the def is `structural`;
   - ChaChaNotes_DB.py gets one end-of-file import per new module (`# noqa: E402`; cycle-safe, because every use is at call time) and drops top-level imports it no longer uses.
10. **Outside callers,** in `tldw_Server_API/app` and `tldw_Server_API/tests` (R24):
    - `<recv>.<moved>(…)` on an instance receiver becomes `<alias>.<moved>(<recv>.ctx, …)`, or `<recv>.<store_attr>.<moved>(…)` in `store` mode, and the file gains the import.
    - A static or classmethod target is called as `<alias>.<moved>(…)` (`<StoreClass>.<moved>(…)` in `store` mode) from any receiver, `CharactersRAGDB` included.
    - A `CharactersRAGDB.<moved>` call to an instance method, and every non-call reference, is not rewritten; `check` lists it (Appendix D).

**Resolving a fail-closed private,** in this order:
- the moved group alone uses it → add it to the group;
- it is pure, or needs only context state → a context member;
- it is a class constant tests patch, or init mutates → a context property reading the DB, added to `PINNED_DB_PRIVATES`;
- another constant → `schema_sql`;
- otherwise → a public DB rename.

Record each decision in the manifest's `rewrites`.

**A.4 `callers --store ATTR --names … [--check] [--tests]`.**
- **Attribute loads:** `<recv>.<name>` becomes `<recv>.<ATTR>.<name>`.
- **Probes:** `getattr(<recv>, "<name>", d)` becomes `getattr(getattr(<recv>, "<ATTR>", None), "<name>", d)`, and `hasattr` likewise.
- **Receivers:** any receiver not already ending in `<ATTR>`, with these exceptions:
  - the module that defines the store class is skipped;
  - a bare `self` is rewritten only inside `class CharactersRAGDB`, and only for retirements (Task 2); in Tasks 10 and 12 class callers wait for the move;
  - elsewhere, `self.<name>` is a class calling its own method and is left alone. `--check` lists any whose enclosing class does not define `<name>`.
- **Collisions:** a name that is also a method of another app class (a `def <name>` inside a class body in app, outside ChaChaNotes_DB.py and `chacha/`) is never rewritten; it is listed as `collision`.
- **Review output:**
  - a rewritten receiver failing Task 1's receiver rule prints `unusual receiver`;
  - a `MagicMock(spec=CharactersRAGDB)` in a rewritten file prints `spec mock`, needing the store attribute (Global Constraints).
- **`--check`** writes nothing, and exits 1 on any of:
  - a remaining load, probe or assignment (`db.<name> = …`) whose receiver is not `*.<ATTR>`;
  - any `CharactersRAGDB.<name>` reference;
  - a class outside `chacha/` that defines a name but has no `<ATTR>`;
  - a `"…CharactersRAGDB.<name>"` patch string.
- **`--tests`** prints the test files it touched.

**A.5 `select GROUP`,** a port of research B's `t22_defs.py`/`t22_closure.py`:
- **Buckets:**
  - `_migrate*` → migrate;
  - `rls|polic` → rls;
  - `fts` → fts;
  - `backfill` → backfill;
  - `^_?(ensure|verify|repair|rebuild|reconcile|bridge|configure|sync_postgres_seq)`, a name containing `schema`, or `_(create|apply)_.*(schema|table|index|trigger)` → schema;
  - everything else → other.
- **S₀:** class defs in a non-other bucket, minus 14 false positives: `_ensure_skill_registry_table`, `_ensure_json_string`, `_ensure_json_string_from_mixed`, `ensure_chat_grammars_table`, `_ensure_note_folder_path_locked`, `_ensure_note_exists_for_folder_sync`, `_validate_workspace_source_saved_view_schema_version`, `_verify_workspace_migration_delete_eligibility`, `_repair_flashcard_review_session_aggregates_if_unchanged`, `reconcile_flashcard_asset_refs`, `rebuild_full_text_indexes`, `backfill_tool_calls_from_inline`, `ensure_character_tables_ready`, `_repair_conversation_assistant_identity`.
- **Closure:** repeatedly add a private "other" def whose in-class callers are non-empty and ⊆ S, with no reference from `chacha/` and none from app code outside DB_Management.
- **Remove** the 13 shims, `_ensure_bootstrap_for_backend`, and classmethods that read `cls._CURRENT_SCHEMA_VERSION`/`_POSTGRES_SCHEMA_VERSION`/`_SCHEMA_NAME` (R24).
- **Groups:**
  - `migrate-sqlite` / `migrate-postgres`: a `_migrate*` name without or with `postgres`;
  - `init`: `_initialize_schema_sqlite_legacy`, `_postgres_schema_is_current`;
  - otherwise a name containing `postgres` goes to `postgres-schema` and one containing `sqlite` to `sqlite-schema`. Else by callers: all SQLite-side → `sqlite-schema`, all PostgreSQL → `postgres-schema`, mixed → `migration-support`.
- **`schema-sql`:** class-level assignments whose names match `t22_plan.py`'s `SQLRX`, plus every class or module constant their values read, here `_NOTE_TASK_SCOPE_AUTHORITY_TABLE` and `RETIRED_SYNC_PLACEHOLDER_ASSISTANT_KIND`/`_ID` (R20). The `constants` mode re-imports class constants in the class body and module constants at module level.
- **Domains** (Tasks 10 and 12): a class def in the "other" bucket whose name's first match in this ordered list is the domain. This is research B's `t22_domcallers.py` order:
  1. quiz `quiz|osce|attempt|question|study_assistant|_fill_blank|levenshtein|remediation`;
  2. `flashcards` `flashcard|deck|review_session|study_pack|suggestion|srs|apkg`;
  3. `workspace` `workspace|clone_target|source_review|saved_view|source_`.
- Measured base sizes are in Measured. Each PR records its own.

**A.6 `verify MANIFEST… --base REF`.**
- Load the base source with `git show REF:<source>`.
- A NodeTransformer applies rules 2-5 and 9-10 to the base AST, and the result must equal the head AST by `ast.dump(include_attributes=False)`, docstrings included, for:
  - each moved def against its target;
  - each remaining class def, with call sites rewritten and shims expanded;
  - each constant value.
- `structural` defs print an `ast.unparse` diff instead of failing.
- It checks that moved names are gone from the class and present once in the target.
- **Output:** `moved N identical; class defs M identical; structural [...]`. Exit 1 on any other difference.
- `verify` shares its rule code with `apply`, so it is evidence and not a trust anchor. R22's `diffpairs.py` and ruff are the independent checks.
- **`patches MANIFEST… | --names N,…`** (R25) scans `tldw_Server_API/tests` and `app/**/tests`. It lists every patch of a watched name whose target is not the alias module, `<recv>.<store_attr>`, the store class or `db.ctx`:
  - `monkeypatch.setattr`/`delattr`, `patch.object`, `patch.multiple` (keyword names), `mocker.patch.object` and `setattr`, in call, `with` and decorator form;
  - `patch("…CharactersRAGDB.<name>")` strings;
  - `ast.Assign`/`AugAssign` to `<x>.<name>`, `CharactersRAGDB.<name>` or `type(<x>).<name>`;
  - keyword arguments named `<name>` in any call (`SimpleNamespace`, `MagicMock`, `Mock`);
  - `getsource(<x>.<name>)`.

  The watched set has three parts. With `--names`, it is exactly the given names.
  - the manifest names;
  - every class constant the moved bodies read (`self._X` with `_X` in `schema_sql`);
  - every module-level free name the moved bodies import from ChaChaNotes_DB's namespace under A.3 rule 8, such as `time`, `build_source_review_launch_metadata` or `_normalize_scheduler_settings_envelope` (R30).

  For the third part, it reports these forms:
  - `monkeypatch.setattr("…ChaChaNotes_DB.<name>", …)`;
  - `patch("…ChaChaNotes_DB.<name>")`;
  - `monkeypatch.setattr`, `patch.object` or `setattr` on a ChaChaNotes_DB module object, under any alias, with `"<name>"`.

  A hit whose name ChaChaNotes_DB itself still reads is reported as `both sides`, and the test patches both modules.
- **`check MANIFEST…`** writes nothing, and exits 1 on any reference left to a moved name, or to a watched free name in a `ChaChaNotes_DB` string target or `getattr`/`hasattr` (R30):
  - a `CharactersRAGDB.<moved>` or `<recv>.<moved>` attribute access that the rewrite rules did not handle;
  - a string literal naming it in `getattr`/`hasattr`.

  It prints each with its Appendix D rule. `callers --check` is the same scan for retired names.
- **`tests MANIFEST…`** prints the test files containing any manifest name as a word.

## Appendix B: proof sets (dev ee0888b51b; `*` = live PostgreSQL)

**PROOF_MIG** (65 files, 24 live PostgreSQL). The dry run measured 102 s for the SQLite files and 252 s for the live-PostgreSQL ones, at `-n 2`:
```bash
git grep -lE "_migrate_from_v|_sqlite_linear_migration_steps|_run_sqlite_linear_migration_step|_initialize_schema_postgres|_MIGRATION_SQL_" -- tldw_Server_API/tests ':!tldw_Server_API/tests/lint' ':!tldw_Server_API/tests/prompt_studio' ':!tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py'
```
- **ChaChaNotesDB/:**
  - test_chachanotes_db, test_character_exemplars_db, test_conversation_assistant_identity_db, test_conversation_list_indexes;
  - test_note_attachment_migration_v59\*, test_note_task_sync_migration_v60, test_note_task_sync_postgres_tenancy\*;
  - test_notes_link_migration_v58, test_notes_moodboard_studio_sync_migration_v61, test_notes_moodboard_studio_sync_postgres_tenancy\*;
  - test_notes_organization_migration_v55, test_notes_organization_migration_v57;
  - test_persona_buddy_db, test_persona_exemplar_migration, test_persona_persistence_db, test_persona_visual_library_db, test_persona_visuals_db;
  - test_quizzes_basic\*, test_web_clipper_migration_v56;
  - test_workspace_assistant_defaults_db, test_workspace_source_saved_views_db, test_workspace_source_saved_views_postgres\*
- **Chat_Macros/unit/:** test_macro_repository
- **DB_Management/:**
  - test_chacha_conversations_fts_healing;
  - test_chacha_migration_v10, \_v39, \_v48_tasks, \_v62, \_v64, \_v73_persona_companion;
  - test_chacha_postgres_migration_v9, \_v10, \_v14, \_v15, \_v22, \_v26, \_v51, \_v52, \_v52_integration\*, \_v61\*, \_v62, \_v64\*, \_v77_persona_companion\*;
  - test_chacha_semantic_migration, \_postgres\*, \_v66, \_v66_postgres\*, \_v67_postgres\*;
  - test_character_behavior_snapshot_migration\*, test_conversation_assistant_startup\*, test_conversation_source_incarnation;
  - test_history_selection_migration\*, test_local_keyword_survivor_migration\*;
  - test_native_fork_migration\*, test_native_reference_role_migration\*, test_osce_postgres_migration_v67\*;
  - test_persona_transcript_migration\*, test_sync_placeholder_assistant_migration\*;
  - test_workspace_chat_startup_migration, test_workspace_persona_optout_v69\*
- **Flashcards/:** test_apkg_exporter
- **Persona/:** test_independent_buddies\*
- **Sync/:** test_sync_v2_notes_link_postgres_contract\*, test_sync_v2_notes_organization_postgres_contract
- **Writing/:** test_manuscript_annotations_db

**PROOF_BOOT** is PROOF_MIG plus these 20 (9 live PostgreSQL). They are the ChaCha files matching `_initialize_schema|_POSTGRES_SCHEMA_VERSION|_runtime_schema_version|_get_schema_version_postgres|_CURRENT_SCHEMA_VERSION|_sqlite_schema_init_lock_for_path|postgres_schema_migration`:
- **ChaChaNotesDB/:** test_chacha_message_store, test_chacha_task_store, test_conversation_create_fingerprint\*, test_notes_provenance\*, test_order_by_validation
- **Characters/:** test_character_functionality_db
- **Chat_NEW/integration/:** test_dictionary_postgres_ownership\*
- **DB_Management/:**
  - test_chacha_migration_v61;
  - test_chacha_postgres_notes_bootstrap_lifecycle\*, test_chacha_postgres_schema_lock\*, test_chacha_postgres_session_scope;
  - test_chacha_semantic_migration_v67, test_chacha_shared_sqlite_cleanup;
  - test_character_owner_name_migration_postgres\*, test_content_backend_cache, test_deck_owner_name_migration_postgres\*;
  - test_osce_migration_v67, test_study_pack_shared_sync_schema\*, test_workspace_chat_startup_receipts
- **Skills/integration/:** test_skill_registry_postgres_schema\*

**OWNER_ISO** (9 files, the `chacha-owner-isolation` job, run serially):
- `ChaChaNotesDB/`: test_chacha_message_store, test_web_clipper_postgres_tenancy
- `DB_Management/`: test_chacha_owner_scope_required, test_chacha_study_tables_rls, test_core_chat_owner_isolation, test_core_chat_rls_postgres, test_local_keyword_merge_survivor, test_note_shared_owner_contract, test_study_suggestion_owner_scope

## Appendix C: the frozen registries (from dev ee0888b51b; Task 8's snapshot confirms them)

**`SQLITE_STEPS`**, source → name, keys 4..84 in order:
- 4..58, 64..71 and 73..80: `_migrate_from_v{k}_to_v{k+1}`;
- 59..63 and 81..83: `_migrate_from_v{k}_to_v{k+1}_sqlite`;
- 72: `_migrate_from_v72_to_v73_persona_companion`;
- 84: `migrate_sqlite_transcripts`, the adapter. On dev, key 84 is a `functools.partial` with no `__name__`; `registry-snapshot` reads the name from the `bind_migration(...)` call's first argument. The adapter's `__name__` is new behaviour, recorded by the pin.

**`POSTGRES_STEPS`**, source → name, keys 72..90 in order:
- `_migrate_from_v{k}_to_v{k+1}_postgres`;
- except 76: `_migrate_from_v76_to_v77_persona_companion_postgres`.

The 16 PostgreSQL migrations v54→v70 are module functions, not registry entries; the legacy ladder calls them inline (R12).

## Appendix D: class-receiver references to moved names (dev ee0888b51b, 62 sites; 58 change)

By R24's rules. Paths are under `tldw_Server_API/tests/` unless they start with `app/`.
- **Auto-rewritten (22), static and classmethod calls:**
  - `ChaChaNotesDB/test_note_task_sync_migration_v60.py` 41, 65, 99, 155, 228;
  - `ChaChaNotesDB/test_persona_visuals_db.py` 115, 201, 249, 306. These call `_prepare_sqlite_schema_drift_fixture`, which stays on the class (R24), so the sites stay unchanged;
  - `ChaChaNotesDB/test_workspace_assistant_defaults_db.py` 378, 387;
  - `ChaChaNotesDB/test_workspace_sub_resources_db.py` 535;
  - `DB_Management/test_postgres_unique_conflict.py` 183, 226, 227;
  - `Sync/test_sync_v2_notes_task_activity_postgres_contract.py` 125, 126;
  - `Sync/test_sync_v2_notes_task_postgres_contract.py` 200;
  - `Workspaces/test_workspace_file_inventory_db.py` 397, 401, 408, 413.
- **`getsource` readers (23), retargeted by R26:**
  - `ChaChaNotesDB/test_note_attachment_migration_v59.py` 128, 131, 921\*;
  - `ChaChaNotesDB/test_note_task_sync_migration_v60.py` 31\*, 81, 109, 159, 232, 254;
  - `ChaChaNotesDB/test_notes_link_migration_v58.py` 26, 52;
  - `ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py` 295, 1857;
  - `ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py` 213, 215, 218, 221, 224, 226\*;
  - `ChaChaNotesDB/test_notes_organization_migration_v57.py` 307\*;
  - `ChaChaNotesDB/test_workspace_source_saved_views_postgres.py` 192\*;
  - `Sync/test_sync_v2_notes_link_postgres_contract.py` 24\*, 35.

  \* reads `_initialize_schema_postgres`: Task 7's ladder tests.
- **Hand: captures and identity checks (13):**
  - `ChaChaNotesDB/test_note_attachment_migration_v59.py` 305;
  - `ChaChaNotesDB/test_note_task_sync_migration_v60.py` 116, 117, 806;
  - `ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py` 1249, 1250;
  - `ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py` 791, 913;
  - `DB_Management/test_chacha_migration_v73_persona_companion.py` 147;
  - `DB_Management/test_persona_transcript_migration.py` 55;
  - `DB_Management/test_sync_placeholder_assistant_migration.py` 92, 93;
  - `DB_Management/test_workspace_persona_optout_v69.py` 322.

  Each captures the module function instead, and `__func__ is` becomes `__wrapped__ is`.
- **Hand: unbound calls with a fake `self` (4):**
  - `ChaChaNotesDB/test_notes_link_migration_v58.py` 38;
  - `ChaChaNotesDB/test_workspace_sub_resources_db.py` 278;
  - `DB_Management/test_native_fork_workspace_lifecycle.py` 450;
  - `app/core/MCP_unified/tests/test_flashcards_module.py` 51.

  Each calls the module function with `ChaChaContext(<fake>)`, or the store method on `<Store>(ChaChaContext(<fake>))`.

## Fix round 1 (changelog)

The findings come from `plan-review.md` (I, M) and `plan-dryrun.md` (B, N), with the controller's rulings R19-R28.

| Finding | Where it is addressed |
|---|---|
| B1 `__new__` instances | R19; Task 3 (`ctx` cached property, the named +1 bump, `test_ctx_exists_on_a_db_built_without_init`); Task 8's pins use `__new__`; projections count the property |
| B2 / I4 placeholder module constants | R20; Task 5 (110 constants, module-level re-import, `test_placeholder_constants_are_reexported`); A.5 `schema-sql` |
| I1 / N4 inert-patch forms, Task 3 census | R25; Task 3 (`patches --names`, census of 37); A.6 `patches`; Review Focus 1; Global Constraints (assignment and spec mocks) |
| I2 class-receiver, static and classmethod references | R24; A.3 rules 2 and 10; A.4 `--check`; A.6 `check`; Appendix D (62 sites, 58 change); Tasks 6, 8, 11 and 13 |
| I3 / N8 / N12 source-reading tests | R26; Task 7 (ladder helper, 6 tests re-expressed and run before the move); Task 8 (loop expansion); Task 9 (helper repointed, legacy-marker order test); `patches` flags `getsource` |
| I5 AC5 gate for new guards | R21; Task 2 (both pin files wired); Tasks 3-13 add to them; self-review table |
| I6 / N3 context privates | R27; Task 3 Interfaces (`prepare_statement` without `_transform_query_for_backend`, `PINNED_MODULE_PRIVATES`, `chacha.context` import test) |
| I7 verify circularity and the `migrations` lint gap | R22; A.1 `diffpairs.py`; A.6 note; C2; the Python 3.12 floor in Global Constraints and A.1 |
| I8 git-ignored manifests | R23; A.2; every task's manifest path |
| M1 / N11 the G file margin | R28; the table notes; Task 8 Step 6; Task 14 AC4 |
| M2 Task 11's method margin | Task 10 Step 0 |
| M3 reach projections | Declined (Rulings): derived as 488 − 179 − 5 + 1 = 305; every later reach value is derived in the table notes |
| M4 alias evasion | Task 3: `test_an_alias_of_the_db_is_a_known_limit`, plus a grep review in Step 4 |
| M5 Review Focus 5 wording | Review Focus 5 rewritten; Task 9 invariants |
| M6 `spec=CharactersRAGDB` mocks | Global Constraints; Task 2 Step 9; Task 10 Step 3; A.4 review output |
| M7 / N2 26 + 4 | Task 2 Step 9 |
| M8 standalone-import coverage | context (T3), schema_sql (T5), sqlite_schema and support (T6), postgres_schema (T7), steps (T8), schema_bootstrap (T9), workspace_store (T10), flashcard_store (T12) |
| M9 / N9 / N10 registry names, key 84, AC3 text | Task 8 Interfaces; Appendix C; Task 14 AC3 amendment |
| N1 the moodboard loop | Task 2 Files |
| N5 the Slides fakes | Measured; Tasks 3 and 4 file lists |
| N6 / N7 projections | PR table, from the dry run's recount |
| N13 | No change needed: none of the owner-scope opt-outs or function-size entries moves outside the plan |
| Dry-run timings, flake and skip | Measured; Known local failures; Appendix B; Task 14 follow-up for the flake |
| Open PRs #3405 and #3294 | Measured; sibling coordination |

## Fix round 2 (changelog)

The findings come from `plan-rereview.md`, with the controller's rulings R29-R34.

| Finding | Where it is addressed |
|---|---|
| N-1 Task 9 reach raise | R29; Task 9 Files (the `runtime_schema_version` rename and pin swap), Step 1 (`test_runtime_schema_version_is_public_with_a_default`) and Step 5 (296, no bump); PR table rows H-J2 at 296; table note; Global Constraints (no reach raise); R9 wording; A.2 `renames` |
| N-2 free-name patches (closes I1) | R30; R25 watched set; A.6 `patches`/`check`; Task 4 self-test `test_patches_watches_module_names_the_moved_bodies_import`; the four sites in Tasks 7, 11 and 13 |
| N-3 git identity in self-tests | R31; Task 4 Step 1; Task 2 Step 7 (step environment run) |
| N-4 diffpairs on static and classmethod moves | R32; A.1 `diffpairs.py`; Task 4 self-tests `test_diffpairs_accepts_a_static_move` and `test_diffpairs_accepts_a_classmethod_move` |
| N-5 ladder helper keyword arguments (closes I3) | R33; R26; Task 7 Files (`LadderCall`), Interfaces and Step 1 (the `lock=True` expression) |
| N-6 legacy-marker test outside the gate | R34; Task 9 invariants and Step 1; self-review |
