# Unconditional ChaCha owner scope on PostgreSQL (TASK-13950.23) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On PostgreSQL, a ChaCha owner-scope helper never silently drops its owner predicate.
- A missing owner raises.
- A ratchet stops new drop-on-None helpers.
- The owner-isolation tests run in a required gate and fail rather than skip.
- The scope-predicate baseline stops breaking on line shifts.

**Architecture:** Four ordered PRs plus a backlog close-out.
1. Re-key the existing scope-predicate baseline by qualified name.
2. Add a ChaCha owner-scope ratchet, seeded at today's 102 call sites.
3. Give owner-isolation tests a marker and a required Postgres step, where a skip counts as a failure.
4. Two PRs close the gap on PostgreSQL. SQLite behaviour is unchanged: per-user files, where `client_id` labels devices.
   - **4a:** the conversation-bound message-store helpers use the instance owner when given `None`. Their table is `conversations`, whose `client_id` is the tenant.
   - **4b:** the three table-agnostic helpers and `_manage_link` raise when the owner is `None`. Every caller that passes no owner today now passes the instance owner. A caller whose table has no tenant `client_id`, or whose parent lock already proved ownership, opts out explicitly with `OwnerScope.NOT_APPLICABLE`.

**Tech Stack:** Python 3.12/3.13, PostgreSQL 18, pytest (`pytester`), AST lint tests under `tldw_Server_API/tests/lint/` (`_ast_cache.py`, `_ceilings.py`), GitHub Actions (`backend-required.yml`).

**Spec:**
- `backlog/tasks/task-13950.23 - Make-ChaCha-owner-scoping-unconditional-on-PostgreSQL.md`, AC #1-#6. It depends on TASK-13950.3, which is Done.
- The roadmap's Global Constraints and Review Focus #1-#2: `Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md`.
- Research behind every number below, at dev 48204ff023: `.superpowers/sdd/2026-11-21-task-13950-23-unconditional-owner-scope/research-A.md` and `research-B.md`. These are git-ignored, so the executor's ledger workspace holds them.

## Measured (dev 48204ff023)

- **Helpers.** `_selected_owner_filter`, `_selected_keyword_link_filter` and `_require_selected_owner_row` are at `ChaChaNotes_DB.py:35004`, `:35013` and `:35037`.
  - Each returns no predicate when `owner_client_id is None` or the backend is not PostgreSQL. `_require_selected_owner_row` also returns early when `item_id is None`.
  - **Call sites (AST):** 102 in all. That is 59 + 21 + 22, by module `ChaChaNotes_DB.py` 36, `chacha/keyword_store.py` 29 and `chacha/note_store.py` 37. The task body's "77 + 22" has drifted; the committed counter defines the number.
- **Where `None` reaches the helpers.** Nearly every site passes `self.client_id` / `self._db.client_id`, and the constructor rejects an empty value (`ChaChaNotes_DB.py:8756`). `None` arrives only through the generic helpers' `owner_client_id=None` default:
  - **Generic helpers with that default:** `_get_current_db_version`, `_add/_get/_list/_update/_soft_delete_generic_item`, `_get_generic_item_by_unique_text`, `_search_generic_items_fts` and `_manage_link`.
  - **Callers that omit the owner:** research-A §1b lists every one.
    - The writing_* generic calls: `ChaChaNotes_DB.py:45860, 45884, 46169, 46185, 46297, 46323, 46377, 46393, 46472, 46498`.
    - `skill_registry` version lookups: `:28866`, `:28932`.
    - The moodboards version lookup: `:36185`.
    - `_manage_link("moodboard_notes", …)`: `:36451`, `:36577`. This is the pre-v61 branch.
    - Conversations: `conversation_store.py:1644`.
    - Messages: `message_store.py:2923`, `:3019`.
    - Keywords: `keyword_store.py:252, 566, 579, 746`.
- **Message-store owner-optional helpers.** Each queries `conversations`, applies `client_id = ?` on both backends when given an owner, and drops it on `None`:
  - `_lock_history_owner` (`message_store.py:72-84`). It raises `NotFoundError` from `db_errors`, not `ConflictError`, when the conversation is not found.
  - `get_messages_by_sync_stable_id` (`:2347`).
  - `tombstone_message_from_sync` (`:2162-2177`, public) has no owner logic of its own. It goes through the two helpers above, so it needs no edit.
  - `conversation_resume_store.get_roleplay_resume_state` (`:957`, `CAST(? AS TEXT) IS NULL`).
- **Their callers without an owner:**
  - `_lock_message_conversation` (`:93`), which serves edits and deletes;
  - `add_message` (`:1563`);
  - `tombstone_message_from_sync` (`:2210`);
  - `append_message_from_sync` (`:2058`, via `get_messages_by_sync_stable_id`);
  - Sync v2's `Sync/v2/materializers/chat.py:348`, `:399`.

  Raising there would break every `add_message` and Sync v2. Hence Ruling 1's split.
- **Writing getters.** `get_writing_session`, `list_writing_sessions` and `count_writing_sessions` (`ChaChaNotes_DB.py:45747-45820`) run raw SQL with no owner helper; RLS alone scopes them. The writing_* methods that go through generic helpers are `update_writing_session` (`:45860`), `soft_delete_writing_session` (`:45884`), `add_writing_template` (`:46169`), `get_writing_template_by_name` (`:46185`), `update_writing_template` (`:46297`), `soft_delete_writing_template` (`:46323`), and the themes equivalents (`:46377-46498`).
- **`_manage_link`'s pre-v61 moodboard branch** (`:36451`, `:36577`) is unreachable on PostgreSQL: the schema is at 91 and the gate is `>= 63`. A real owner would still raise "Unsupported scoped keyword link", so that call needs `NOT_APPLICABLE`.
- **Dry-run runtime.** The 7 Task 3 files take 7:01 locally against Docker Postgres with `TLDW_TEST_POSTGRES_REQUIRED=1`: 286 passed, 5 skipped, all in `test_core_chat_rls_postgres.py` ("CHAT_RLS_DB_URL not configured"). backend-required took 28 minutes of its 45-minute timeout on 2026-11-03.
- **Tables without a tenant `client_id`:** `skill_registry` has no `client_id` column (`ChaChaNotes_DB.py:3533`). For messages, `client_id` may name a sync device; the conversation owns the message (`_require_selected_owner_row`, `:35048`).
- **RLS gaps that the helpers alone cover.** `suggestion_snapshots`, `suggestion_generation_links` and `flashcard_keywords` have no policy (`Helper_Scripts/ci/rls_coverage_baseline.txt:250-251`). The startup BYPASSRLS refusal validates the content backend role only (`media_db/runtime/factory.py:93-129`).
- **Scope-predicate ratchet.** `Helper_Scripts/ci/scope_predicate_ratchet.py` keeps 28 entries keyed `path:LINE if <scope>`.
  - **Duplicates:** four `(path, qualname, scope)` pairs occur twice, 8 of the 28 entries: `JobManager.list_jobs`, `.count_jobs` and `.summarize_by_status`, plus the sqlite lifecycle `_candidate_sql`.
  - **Where it runs:** backend-required's "Enforce tenant isolation ratchets" step (`backend-required.yml:375`). Its contract is a substring check (`test_license_first_workflow_contracts.py:1626`).
- **Postgres in CI.** On dev PRs the Full Suite Postgres shards do not run; they are nightly or release tier (`ci.yml:590-594`). backend-required has a Postgres service (`tldw_user` / `TestPassword123!` / `tldw_test`) and runs exactly one Postgres test (`:290-316`).
  - `_plugins/postgres.py:210` skips on a missing psycopg even when `TLDW_TEST_POSTGRES_REQUIRED` is set.
  - `tests/DB_Management/test_core_chat_rls_postgres.py:34-38` skips unless `CHAT_RLS_DB_URL` is set, and no workflow sets it.
- **Stale superuser text:**
  - `ChaChaNotes_DB.py:19458-19461` (v79→v80 docstring);
  - `tests/_plugins/postgres.py:335-341` (docstring) and `:244` (hint);
  - `tests/DB_Management/test_sync_placeholder_assistant_migration.py:223-231` (docstring).
- **Detector prototypes.**
  - **App-wide** (both shapes): 45 helper names and 423 call sites, mostly false positives (quota, VN, Jobs).
  - **ChaCha-only, early-return shape** (Task 2's rule): exactly the three helpers and 102 calls.

## Rulings

1. **AC2: raise where the table is unknown, derive where it is fixed.** AC2 allows either.
   - **Raise:** on PostgreSQL, `_selected_owner_filter`, `_selected_keyword_link_filter`, `_require_selected_owner_row` and `_manage_link` raise `OwnerScopeRequiredError` when given `owner_client_id=None`.
     - **Why:** they serve every table. Deriving `self.client_id` would add `client_id = ?` to tables without a tenant `client_id`: `skill_registry` has no such column, and on messages `client_id` is a device label. Raising fails closed and matches the precedents at `moodboard_sync_store.py:91`, `task_store.py:186` and the constructor's empty-`client_id` check.
   - **Derive:** on PostgreSQL, `_lock_history_owner`, `get_messages_by_sync_stable_id` and `get_roleplay_resume_state` use `self._db.client_id` when given `None`.
     - **Why:** their table is always `conversations`, whose `client_id` is the tenant that `app.current_user_id` holds. So the derived predicate is exactly what RLS already enforces for the production role, and it keeps `add_message` and Sync v2 working.
   - **Behaviour change**, on PostgreSQL only: a conversation whose `client_id` differs from the instance owner is no longer reachable through these helpers.
     - Edits and deletes raise `NotFoundError`.
     - `add_message` raises `InputError` ("Cannot add message: ... not found or deleted"), because it re-raises at `message_store.py:1563-1567`.
     - A restricted role already behaves that way.
     - This includes direct `CharactersRAGDB(client_id="mcp_*" / "workflow_engine" / "persona_memory_*")` instances. On PostgreSQL they now fail closed on user-owned conversations even under a bypass role; Task 5 files the follow-up about them.
   - **Cost if wrong:** each opt-out (`OwnerScope.NOT_APPLICABLE`) is explicit and counted by the Task 2 ratchet, so it is reversible per site.
2. **The AC1 ratchet covers ChaCha modules only**, using the early-return shape: `ChaChaNotes_DB.py` and `chacha/*.py`.
   - **Why:** app-wide, the prototype is mostly false positives, and the finding and AC2 are about ChaCha.
   - **Cost if wrong:** other stores' optional-owner helpers stay unratcheted. Task 5 files a follow-up.
3. **AC6 step.** Both ratchets this task adds or changes run in "Enforce CI contracts and code ratchets":
   - the new owner-scope ratchet;
   - the re-keyed scope-predicate ratchet, which moves there from "Enforce tenant isolation ratchets".

   **Why:** the owner-scope ratchet is `_ceilings`-based and needs `BACKLOG_TASK_FORMAT_BASE`, which only that step sets. The scope-predicate ratchet moves only because AC6 names that step. **Cost if wrong:** moving two lines back.
4. **AC4: a path-gated required Postgres job plus skip-as-failure.** The owner decided this on 2026-11-21 ("Only when ChaCha changes").
   - **The job:** a new job, `chacha-owner-isolation` in `backend-required.yml`, runs every test file marked `owner_isolation` with `TLDW_TEST_POSTGRES_REQUIRED=1`. It has its own Postgres service and runs in parallel.
     - It runs only when a new path-classifier flag, `chacha_owner_scope_changed`, is true.
     - Otherwise it is skipped, and the required check accepts the skip.
   - **The required check:** `backend-required` gains `needs: chacha-owner-isolation` and a "Require ChaCha owner isolation to have passed" step, following the `non-utc-tests` pattern (`backend-required.yml:66-80`, `:245-258`).
   - **Skip-as-failure:** the Postgres plugin turns a skipped `owner_isolation` test into a failure whenever that variable is set, so this holds in the nightly shards too.
   - **Why:**
     - Postgres shards do not run on dev PRs.
     - The files take 7 to 27 minutes. 3a measured 27:03 serially under load; `-n 2` fails xdist collection on non-deterministic parametrize ids.
     - The owner does not want an extra runner on every backend PR, because CI runners are the bottleneck.
   - **Cost if wrong:** a ChaCha owner regression in a PR whose paths miss the flag is caught only nightly. The flag's path list is reviewed in Task 3b.
   - **All `backend-required.yml` edits for Task 3 are in PR C2 (Task 3b).** The classifier denied the 3a implementer's one-line workflow edit. The owner then chose this design, and C1 stays workflow-free.
5. **AC5 key shape:** `<path>::<qualname> if <scope>`.
   - A second finding with the same key takes a `#2` suffix. Number findings by `(lineno, col_offset)` within the key, since `ast.walk` order is not source order. Module-level code uses `<module>`.
   - Baseline semantics stay exact-match, a set of strings.

   **Why:** it is the smallest change that removes line numbers; the diff logic is unchanged. **Cost if wrong:** one regeneration.
6. **SQLite stays as it is.**
   - Every new raise sits after the existing non-PostgreSQL early return.
   - The derive applies only on PostgreSQL.
   - The message-store helpers keep their both-backend predicate when given an owner.
   - **Why:** per-user SQLite files label devices with `client_id`; that is the design.
   - **Cost if wrong:** none for SQLite users.
7. **Amend the AC text at close-out** (resolve, don't relay):
   - AC1's "77 filter + 22 require-row" becomes the counter's 102 (80 + 22).
   - AC4's "the Postgres shard" becomes "the required `chacha-owner-isolation` job, and the nightly shards".
   - The ratchet does not detect `_lock_history_owner`'s `if owner is not None:` shape or the `IfExp` shape. Task 5 files a follow-up for both.

## Global Constraints

Copied from the roadmap where marked (R).
- **(R) PRs.** Every PR targets `dev`, one backlog task per PR. Over about 400 changed lines, split into ordered PRs that each pass CI.
- **(R) New ratchets:**
  - live in `tldw_Server_API/tests/lint/`;
  - seed their baseline from a counter script committed in the same PR;
  - key entries by module or qualified name, never by line number;
  - fail on growth and on a stale entry;
  - run in backend-required's "Enforce CI contracts and code ratchets" step, with a workflow-contract assertion;
  - ship a self-test that seeds one violation and asserts the ratchet fails;
  - share `tests/lint/_ast_cache.py`.
- **(R) Raising a baseline** is allowed only through a bump-with-reason entry.
- **(R) Postgres proof.** Postgres-dependent ACs are proven with `TLDW_TEST_POSTGRES_REQUIRED=1`; a local skip is not evidence. Docker Postgres (`tldw_postgres_test`) is available locally.
- **(R) Flipping a default.** `git grep` each changed helper or method name across `tldw_Server_API/tests` and run every hit (Task 4).
- **(R) Backlog edits** go through `PYTHONPATH=tools/backlog-py/src python -m backlog_py task ...` only. Stage with `git add -A backlog/tasks`.
- **(R) Review gates.** Each PR carries a `## Change summary` section. The owner waived the ADR-004 sign-off for TASK-13950; record it anyway. Each PR gets a clean-subagent code review before merge.
- **Line-keyed baseline until Task 1 merges.** Any PR that shifts lines in a file that `Helper_Scripts/ci/scope_predicate_baseline.txt` lists regenerates it with `--write-baseline`.
- **Never log or print** a client id together with secrets. Error messages name the table, never the owner value.
- **Pytest hygiene:**
  - `-p no:cacheprovider --basetemp "$TMPDIR/t23-bt"`, then `rm -rf "$TMPDIR/t23-bt"` after every run;
  - at most `-n 2`, one run at a time;
  - check `df -h /System/Volumes/Data` first, and don't start under 100 GiB free;
  - with plugin autoload on, never pass `-p pytest_asyncio.plugin` or `-p xdist.plugin`;
  - unset `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, and set `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:9`.
- **Local Postgres runs:** `TLDW_TEST_POSTGRES_REQUIRED=1 NO_PROXY=localhost,127.0.0.1`. Don't set `TLDW_TEST_NO_DOCKER`; the plugin uses the running container.
- **No new worktrees, repo copies or `git archive` exports.**

## Review Focus

1. **Device-labelled message rows on PostgreSQL.** A message whose `client_id` differs from its conversation's owner must still be editable and deletable by that owner. Messages are scoped by their conversation lock, never by `client_id`. Pinned by Task 4a's `test_owner_edits_a_device_labelled_message_on_postgres`.
2. **SQLite per-user files.** `owner_client_id=None` still yields no predicate. A message edit in a conversation whose `client_id` is another device label still succeeds. Pinned by Task 4b's `test_sqlite_owner_none_stays_unscoped` and Task 4a's `test_sqlite_message_edit_ignores_device_labels`.
3. **A ratchet that passes for the wrong reason.** The detector must catch:
   - `if self.backend_type != X or owner is None: return "", ()`;
   - a bare `return`;
   - an `owner_user_id` parameter.

   It must not flag a `raise` body or a non-owner parameter. Pinned by Task 2's seeded self-tests.
4. **A gate that silently skips.** An `owner_isolation` test that skips under `TLDW_TEST_POSTGRES_REQUIRED=1` must fail. A marked file missing from the job, or a listed file without the marker, fails the contract. A file whose tests all skip at collection fails the job's junit check. Pinned by Task 3a's pytester self-test and Task 3b's contract mutation tests.
5. **Re-keying that still drifts.** Inserting lines above a finding must not change its key. Two findings in one function must get distinct, stable keys. Pinned by Task 1's `test_keys_survive_a_line_shift` and `test_duplicate_findings_get_ordinals`.

## CI-equivalent checks

- **C1, the lint and contract set:** `tldw_Server_API/tests/lint` plus `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`.
- **C2:** `python Helper_Scripts/ci/scope_predicate_ratchet.py` exits 0.
- **C3, the `chacha-owner-isolation` job's file list,** run locally with `TLDW_TEST_POSTGRES_REQUIRED=1`. It needs 0 skipped and 0 failed. Record the wall time.
- **C4, every test file that git grep finds for a changed symbol** (Task 4), plus `tldw_Server_API/tests/Sync`. Run Postgres-parametrized files with `TLDW_TEST_POSTGRES_REQUIRED=1`.
- **C5, the ChaCha SQLite suites:** `tldw_Server_API/tests/ChaChaNotesDB`, `tests/DB_Management/test_chacha_*.py` and `tests/Notes`. SQLite must be unchanged.

## PR slices

| PR | Task | Base |
|---|---|---|
| A | Task 1: scope-predicate re-key + step move | dev |
| B | Task 2: owner-scope ratchet | dev after A |
| C1 | Task 3a: owner-isolation marker, skip-as-failure, the 8 marked files, test docstrings | dev after B |
| C2 | Task 3b: the required `chacha-owner-isolation` job and its contracts | dev after C1 |
| D1 | Task 4a: conversation-bound message-store helpers use the instance owner on PostgreSQL | dev after C2 |
| D2 | Task 4b: table-agnostic helpers raise; every caller passes an owner or opts out; app docstring | dev after D1 |
| E | Task 5: close-out (backlog only) | dev after D2 |

A–C2 edit `test_license_first_workflow_contracts.py`, and A, B and C2 edit `backend-required.yml`, so they merge in order. The dry run estimates about 220 changed lines for D1 and about 300 for D2.

**Sibling coordination:**
- **#3508** (TASK-15572, open) also edits `test_license_first_workflow_contracts.py`. Merge dev into each PR before it merges; take both sides.
- **#3071** (open, behind dev) overlaps only on `ci.yml`, which this plan does not edit.

---

### Task 1: Re-key the scope-predicate baseline by qualified name (AC5, AC6)

**Files:**
- Modify: `Helper_Scripts/ci/scope_predicate_ratchet.py`, `Helper_Scripts/ci/scope_predicate_baseline.txt`
- Modify: `tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py`
- Modify: `.github/workflows/backend-required.yml` (move one path between steps)
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`

**Interfaces:**
- Produces:
  - Finding keys `"<posix path>::<qualname> if <scope>"`, with an optional `" #<n>"` suffix (n ≥ 2), from `iter_findings(...)`.
  - `qualified_name(tree: ast.Module, lineno: int) -> str`, which returns `"Class.method"`, `"outer.inner"` or `"<module>"`.

- [ ] **Step 1: Write the failing tests** in `test_scope_predicate_ratchet.py`:
  - `test_keys_survive_a_line_shift`: seed `pkg/q.py` with a `def f(user_id):` holding an `if user_id:` that adds `"user_id = ?"`. Prepend 5 blank lines to a copy. Assert `iter_findings` yields the same keys for both, equal to `["pkg/q.py::f if user_id"]`.
  - `test_duplicate_findings_get_ordinals`: one method `C.m(self, user_id)` with two such `if user_id:` blocks. Keys are `["pkg/q.py::C.m if user_id", "pkg/q.py::C.m if user_id #2"]`.
  - `test_module_level_finding_uses_module_sentinel`: the key is `"pkg/q.py::<module> if user_id"`.
  - Update `test_findings_use_posix_baseline_paths_on_windows` to expect `"pkg/query.py::<module> if user_id"`, or the qualname its fixture implies.
- [ ] **Step 2: Run them, and expect FAIL** (old keys are `path:LINE if scope`):
  `PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/t23-bt" tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py`
- [ ] **Step 3: Implement.** In `iter_findings`, replace `:LINE` with `::<qualname>` from `qualified_name`. That is the innermost enclosing `FunctionDef`/`AsyncFunctionDef`/`ClassDef` chain, joined by `.`. Then sort each key's findings by `(lineno, col_offset)` and append `#n` from the second one on.
  - Keep `read_baseline`, `diff_against_baseline`, the `--write-baseline` CLI and the exact-match semantics.
  - Update every "line numbers move" text: the baseline header, the message in `test_no_new_conditional_tenant_predicates`, and the hint in `main()`. They should say that entries are keyed by qualified name, so line shifts no longer change them.
  - Regenerate: `PYTHONPATH=. $PY Helper_Scripts/ci/scope_predicate_ratchet.py --write-baseline`. Expect "Recorded 28 ...", with 4 keys carrying `#2`.
- [ ] **Step 4: Move the ratchet in `backend-required.yml`.**
  - Delete `tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py` from "Enforce tenant isolation ratchets" (`:375-388`). If it was the last path, move the trailing `\` correctly.
  - Insert it mid-list in "Enforce CI contracts and code ratchets" (`:407`), keeping every line continuation intact.
  - In the contract file, remove it from `test_backend_required_enforces_isolation_ratchets`'s tuple. Add it to `TEST_SEAM_GATE_SUITES`, so the existing dropped and commented-out mutation tests cover it.
  - Add a comment line to the step's comment block: `# The scope-predicate ratchet (TASK-13950.23) is keyed by qualified name.`
- [ ] **Step 5: Verify.** Run C1 and C2. Both must pass, with "28 conditional scope predicates; baseline exact".
- [ ] **Step 6: Commit** `refactor(ci): key the scope-predicate baseline by qualified name (TASK-13950.23 PR A)`.

### Task 2: The ChaCha owner-scope ratchet (AC1, AC6)

**Files:**
- Create: `tldw_Server_API/tests/lint/test_owner_scope_ratchet.py`, `tldw_Server_API/tests/lint/owner_scope_baseline.json`
- Modify: `.github/workflows/backend-required.yml` (append to the CI-contracts step), `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`

**Interfaces:**
- Consumes: `_ast_cache.parse_tree(root)`, `_ceilings.Ratchet`, `violations`, `warn_if_tightenable`, `reasonless_bumps` and `main`. Follow `test_env_read_ratchet.py` as the model.
- Produces:
  - `SCOPED_MODULES`: `core/DB_Management/ChaChaNotes_DB.py` and every `core/DB_Management/chacha/*.py`, relative to the app root.
  - `OWNER_PARAMS = {"owner_client_id", "client_id", "owner_user_id", "user_id", "owner_id", "tenant_id"}`.
  - `drop_on_none_helpers(trees) -> set[str]`, the names of the helpers.
  - `measure(root=APP_ROOT) -> dict[str, dict[str, int]]`, keyed by module path relative to the app root, with fields `calls` and `opt_outs`.
  - The `__main__` CLI, via `_ceilings.main`.

**The detector, which decides what counts as a helper:** a function in `SCOPED_MODULES` is a drop-on-None helper when all of these hold:
- a parameter is in `OWNER_PARAMS`. Scan `posonlyargs`, `args` and `kwonlyargs`; `_selected_keyword_link_filter`'s owner is keyword-only;
- it contains an `ast.If` whose test has `Compare(<that param>, [Is], [None])`, directly or as an operand of a `BoolOp` (any depth);
- that `If`'s body is exactly one `Return`, either bare or returning a tuple whose first element is the constant `""`.

A `raise` body, `return None` and non-owner parameters do not count.

**The metrics:**
- `calls` counts every `ast.Call` in `SCOPED_MODULES` whose func name (`Name.id` or `Attribute.attr`) is a helper name.
- `opt_outs` counts call arguments, positional or keyword values, that are exactly `OwnerScope.NOT_APPLICABLE`. That is an `Attribute(value=Name("OwnerScope"), attr="NOT_APPLICABLE")` among a `Call`'s `args` or `keywords` values. The helpers' own `is OwnerScope.NOT_APPLICABLE` comparisons are not counted. Task 4 introduces the name.
- A module with both at 0 has no entry.

- [ ] **Step 1: Write the failing self-tests.** Seed `tmp_path/"app"` trees with `_seed`, as `test_env_read_ratchet.py` does, and call the real `measure(root)`:
  - `test_early_return_tuple_helper_and_its_calls_count`: `def _f(self, owner_client_id): if owner_client_id is None or self.x != 1: return "", ()` plus 2 calls. Expect `{"core/DB_Management/chacha/s.py": {"calls": 2, "opt_outs": 0}}`.
  - `test_bare_return_and_owner_user_id_count`.
  - `test_raise_body_is_not_a_helper` gives `{}`.
  - `test_return_none_is_not_a_helper` gives `{}`.
  - `test_non_owner_param_is_not_a_helper` gives `{}`.
  - `test_module_outside_chacha_is_ignored` gives `{}`.
  - `test_opt_outs_are_counted`: one call with `owner_client_id=OwnerScope.NOT_APPLICABLE` and one passing it positionally give `opt_outs: 2`. A bare `x is OwnerScope.NOT_APPLICABLE` comparison in the same file adds nothing.
  - `test_growth_and_stale_entries_fail`: through `_ceilings.violations`, assert the messages contain "calls 3 > ceiling 2" and "stale entry".
  - `test_raised_ceiling_without_reason_fails`: mirror the env-read test of the same shape.
- [ ] **Step 2: Run them, and expect FAIL** (the module does not exist yet).
- [ ] **Step 3: Implement** `test_owner_scope_ratchet.py`, with the real-tree tests `test_owner_scope_does_not_grow` and `test_raised_owner_scope_ceilings_carry_a_new_reason`, and the `RATCHET` definition (`fix=`/`stale=` text naming TASK-13950.23).
- [ ] **Step 4: Seed** the baseline with `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_owner_scope_ratchet --write-baseline`. The expected file content, with each `reason: null`:
  ```json
  {"core/DB_Management/ChaChaNotes_DB.py": {"calls": 36, "opt_outs": 0, "reason": null},
   "core/DB_Management/chacha/keyword_store.py": {"calls": 29, "opt_outs": 0, "reason": null},
   "core/DB_Management/chacha/note_store.py": {"calls": 37, "opt_outs": 0, "reason": null}}
  ```
  The file uses `_ceilings`' one-entry-per-line layout. If the counts differ, dev has moved: record the measured numbers, and say so in the report.
- [ ] **Step 5: Wire.**
  - Insert `tldw_Server_API/tests/lint/test_owner_scope_ratchet.py` mid-list in "Enforce CI contracts and code ratchets", keeping line continuations intact, and add a comment line to the step's comment block: `# The ChaCha owner-scope ratchet (TASK-13950.23) does the same for its call and opt-out counts.`
  - Add the path to `test_backend_required_enforces_config_ratchets`'s parametrize list, which checks the `base_sha` env. Add it to `TEST_SEAM_GATE_SUITES` too.
- [ ] **Step 6: Verify.** Run C1. Then run the CLI with `--tighten`; it must be a no-op.
- [ ] **Step 7: Commit** `test(lint): ratchet ChaCha owner-scope helpers that drop the predicate on None (TASK-13950.23 PR B)`.

### Task 3a: Owner-isolation tests fail rather than skip (AC4 part 1, AC3 test text, AC6)

**Files:**
- Modify: `pyproject.toml`: register the marker `"owner_isolation: ChaCha owner-isolation test; a skip fails when TLDW_TEST_POSTGRES_REQUIRED is set"`.
- Modify: `tldw_Server_API/tests/_plugins/postgres.py`:
  - `:210`: a missing psycopg fails when required;
  - add a `pytest_runtest_makereport` hookwrapper;
  - docstring `:335-341` and hint `:244`.
- Modify, adding `pytest.mark.owner_isolation` to the module-level `pytestmark`, these test files (the job's file list, "OWNER_ISOLATION_FILES"):
  - `tests/DB_Management/test_note_shared_owner_contract.py`
  - `tests/DB_Management/test_study_suggestion_owner_scope.py`
  - `tests/DB_Management/test_local_keyword_merge_survivor.py`
  - `tests/DB_Management/test_chacha_study_tables_rls.py`
  - `tests/DB_Management/test_core_chat_owner_isolation.py`
  - `tests/DB_Management/test_core_chat_rls_postgres.py`
  - `tests/ChaChaNotesDB/test_web_clipper_postgres_tenancy.py`
  - `tests/ChaChaNotesDB/test_chacha_message_store.py`, which covers Task 4a's helpers.

  Selection rule: ChaCha test files that prove owner isolation as a restricted role or `SET LOCAL ROLE`, plus the message-store suite that Task 4 changes. research-B §4 lists the other roughly 35 owner-related files. They stay in the nightly shards, and Task 5 files a follow-up to widen the job if the runtime allows.
- Modify: `tests/DB_Management/test_core_chat_rls_postgres.py`:
  - replace the module-level `pytest.importorskip("psycopg")` (`:27-30`) with a plain `import psycopg`, because a collection-time skip bypasses the hook;
  - when `CHAT_RLS_DB_URL` is unset (`:34-38`), use the plugin fixture `pg_temp_db["dsn"]` instead of skipping. `pg_database_config` has no DSN attribute.
- Modify: `tests/DB_Management/test_sync_placeholder_assistant_migration.py:223-231`, docstring only (AC3).
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`: add the self-test to `TEST_SEAM_GATE_SUITES`.
- Test: `tldw_Server_API/tests/DB_Management/test_owner_isolation_skip_fails.py` (pytester; no Postgres needed).

**Interfaces:**
- Produces: the marker `owner_isolation`, and the hook's failure text `"owner_isolation test skipped while TLDW_TEST_POSTGRES_REQUIRED is set: <reason>"`.

- [ ] **Step 1: Write the failing tests.**
  - **`test_owner_isolation_skip_fails_when_postgres_required`**, in a module with `pytest_plugins = ["pytester"]`. Use `runpytest_subprocess` with `PYTHONPATH` set, as `test_pg_restricted_backend_teardown.py` does:
    1. Make an inner test module marked `owner_isolation` whose test calls `pytest.skip("no pg")`.
    2. Run it with `-p tldw_Server_API.tests._plugins.postgres`. With `TLDW_TEST_POSTGRES_REQUIRED=1`, expect `failed=1`.
    3. With the variable unset, expect `skipped=1`.
    4. An unmarked skipping test still skips with the variable set.
    5. A marked `xfail` test stays `xfailed`.
    6. A fixture that skips in setup fails when the variable is set.
  - **Wiring:** add `tldw_Server_API/tests/DB_Management/test_owner_isolation_skip_fails.py` to `TEST_SEAM_GATE_SUITES`. That puts the self-test in a required step and satisfies AC6.
- [ ] **Step 2: Run them, and expect FAIL.**
- [ ] **Step 3: Implement the plugin changes.**
  - `:210` calls `pytest.fail` when `require_pg` is set.
  - **The hookwrapper:** for an item carrying `owner_isolation`, a report whose outcome is `skipped`, in setup or call, and which is not `wasxfail`, becomes `failed`. Its longrepr is `"owner_isolation test skipped while TLDW_TEST_POSTGRES_REQUIRED is set: <reason>"`. It applies only when the variable is truthy, using the plugin's existing truthy set. Changing the outcome is what junit reads.
  - **Docstrings:** reword `:335-341` and `:244`.
    - The test admin DSN is a superuser for fixture setup only.
    - Deployments must use a NOSUPERUSER NOBYPASSRLS role, which startup enforces for the content backend at `media_db/runtime/factory.py:93-129`.
    - That is why owner-isolation tests run as `pg_restricted_backend`.

    Reword the sync-placeholder docstring the same way: a bypass role is a test or migration case, not a documented deployment.
  - **SQLite parameters:** whole-module marking also covers a module's SQLite parameters. They do not skip today; any that ever skip while the variable is set will fail, which is intended.
- [ ] **Step 4: Mark the 8 files and fix `test_core_chat_rls_postgres.py`.**
- [ ] **Step 5: Verify.**
  - Run C1 and the whole `tldw_Server_API/tests/CI` directory.
  - Run C3 locally, recording the wall time and the counts, with 0 skipped.
  - Then point `POSTGRES_TEST_PORT` at a closed port and run one marked file with `TLDW_TEST_POSTGRES_REQUIRED=1`. Confirm it **fails**, and record the output.
- [ ] **Step 6: Commit** `test(db): ChaCha owner-isolation tests fail rather than skip when Postgres is required (TASK-13950.23 PR C1)`.

### Task 3b: The required `chacha-owner-isolation` job (AC4 part 2, AC6)

**Files:**
- Modify: `.github/workflows/backend-required.yml`:
  - a new job, plus one step and a `needs` entry in `backend-required`;
  - the `changes` job's `outputs` gains `chacha_owner_scope_changed`;
  - the CI-contracts step gains the 3a self-test path.
- Modify: `Helper_Scripts/ci/path_classifier.py` (`classify_paths`) and its tests, adding the flag `chacha_owner_scope_changed`. `.github/actions/detect-required-gate-changes/action.yml` gains the matching output. The flag is true when any changed path is one of:
  - `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`;
  - `tldw_Server_API/app/core/DB_Management/chacha/**`;
  - `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`;
  - `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`;
  - `tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py`;
  - `tldw_Server_API/tests/_plugins/postgres.py`;
  - any file carrying the `owner_isolation` marker (list them explicitly; the contract test checks the list equals the marked set);
  - `.github/workflows/backend-required.yml`;
  - `Helper_Scripts/ci/path_classifier.py`;
  - `pyproject.toml`, `uv.lock`.
- Modify: `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`:
  - the bidirectional contract and its mutation tests;
  - `TEST_SEAM_GATE_SUITES`, plus the CI-contracts step line, for Task 3a's self-test `tldw_Server_API/tests/DB_Management/test_owner_isolation_skip_fails.py`. 3a deliberately left both out;
  - `CHANGE_CLASSIFIER_OUTPUTS` (`:273`), for the new flag;
  - the license-first job inventories the new job forces: `ORIGINAL_JOB_NAMES` (`:111`), `ORIGINAL_DEPENDENCIES` (`:166-169`; the new job needs `changes` and `await_license` like `non-utc-tests`, and `backend-required` gains it), `DIRECT_ADMISSION_JOBS` (`:225-227`), `FETCH_DEPTH_CHECKOUTS` (`:258`), and the `extra_condition` branch near `:726`. Mirror every place `non-utc-tests` appears. `DIRECT_TRIGGER_DIGESTS` hashes only the `on:` block, so it is unchanged.

**Interfaces:**
- Consumes: Task 3a's marker and marked files.
- Produces: the job id `chacha-owner-isolation` (name `ChaCha owner isolation (PostgreSQL)`), the step `Require ChaCha owner isolation to have passed`, and `_owner_isolation_marked_files() -> set[str]` and `_owner_isolation_job_problems(workflow, marked_files) -> list[str]` in the contract file.

- [ ] **Step 1: Write the failing contract tests.**
  - **`test_backend_required_runs_every_owner_isolation_file`:**
    1. `_owner_isolation_marked_files()` returns the repo-relative path of every `tldw_Server_API/tests/**/test_*.py` whose module-level `pytestmark` assignment contains `pytest.mark.owner_isolation`, either alone or in a list/tuple. Find it by AST, not text search, so this contract file and pytester strings don't match.
    2. Assert `_owner_isolation_job_problems(workflow, marked_files) == []`. The function checks:
       - exactly one job `chacha-owner-isolation`;
       - it is not `continue-on-error`;
       - a step env holds `TLDW_TEST_POSTGRES_REQUIRED: '1'`;
       - the `tldw_Server_API/tests/...py` tokens in its pytest step (comments stripped, via `_step_run_tokens`) equal `marked_files`;
       - `backend-required`'s `needs` includes it;
       - a `backend-required` step named "Require ChaCha owner isolation to have passed" reads `needs.chacha-owner-isolation.result`.
  - **The mutation tests:**

    | Mutation | Expected problems |
    |---|---|
    | drop a listed path | `["job does not run <path>"]` |
    | comment a listed path out | `["job does not run <path>"]` |
    | list an unmarked file | `["job runs unmarked <path>"]` |
    | `continue-on-error: true` | `["job is continue-on-error"]` |
    | remove the env | `["job does not require PostgreSQL"]` |
    | remove it from `backend-required.needs` | `["backend-required does not wait for chacha-owner-isolation"]` |
- [ ] **Step 2: Run them, and expect FAIL** (no job yet).
- [ ] **Step 3: Add the job and the required check.**
  - Copy the `non-utc-tests` job's `needs` and `if:` (`:66-80`), and add `&& needs.changes.outputs.chacha_owner_scope_changed == 'true'` to the `if:`.
  - Copy the `backend-required` job's `services.postgres` block, including its conditional `image:` guard, conditioned on the same flag.
  - Run serially, with no `-n`, since `-n 2` fails collection. Set `timeout-minutes: 40`, and record `--durations=15` in the report.
  - The required check accepts:
    - `true:true:success`;
    - `true:false:skipped`;
    - `false:*:skipped`.

    Here the tuple is `backend_changed:chacha_owner_scope_changed:result`. Anything else fails.
  - Give the job the same checkout and setup steps as `backend-required` (`:263-288`).
  - Its test step:
    ```yaml
          - name: ChaCha owner isolation on PostgreSQL
            env:
              PYTHONPATH: .
              PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1'
              TEST_MODE: 'true'
              DISABLE_HEAVY_STARTUP: '1'
              MINIMAL_TEST_APP: '1'
              PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: python
              TLDW_TEST_POSTGRES_REQUIRED: '1'
              TLDW_TEST_NO_DOCKER: '1'
              POSTGRES_TEST_HOST: 127.0.0.1
              POSTGRES_TEST_PORT: ${{ job.services.postgres.ports['5432'] }}
              POSTGRES_TEST_USER: tldw_user
              POSTGRES_TEST_PASSWORD: TestPassword123!
              POSTGRES_TEST_DB: tldw_test
            run: |
              set -euo pipefail
              report="$RUNNER_TEMP/owner-isolation.xml"
              pytest -q --disable-warnings -p pytest_asyncio.plugin -p pytest_timeout --timeout=180 \
                --junitxml="$report" \
                <the 8 paths, one per line>
              python - "$report" <the 8 paths> <<'PY'
              # Fail unless every listed file contributed at least one passing testcase and nothing was skipped.
              PY
    ```
    The dry run confirmed the env names. The plugin loads through the root `conftest.py` `pytest_plugins`, so autoload being off is fine. Keep `-p pytest_asyncio.plugin`, as in the teardown step.
  - **The junit check** maps each testcase's `classname` to its file. It exits non-zero if any listed file has zero passing cases, or if any case is skipped. That catches collection-time skips the hook cannot see.
  - **The required check** goes in `backend-required`, next to "Require non-UTC timestamp tests to have passed", with the same `case` shape on `needs.chacha-owner-isolation.result`. Add `chacha-owner-isolation` to `backend-required.needs`.
- [ ] **Step 4: Verify.** Run C1 and the whole `tldw_Server_API/tests/CI` directory. Then run `actionlint .github/workflows/backend-required.yml` if it is installed; otherwise rely on the `actionlint.yml` CI job.
- [ ] **Step 5: Commit** `ci: run ChaCha owner isolation on PostgreSQL as a required job (TASK-13950.23 PR C2)`.

### Task 4a: The conversation-bound message-store helpers use the instance owner on PostgreSQL (AC2, part 1)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`:
  - `_lock_history_owner` (`:72-84`);
  - `get_messages_by_sync_stable_id` (`:2347`);
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`, `get_roleplay_resume_state` (`:957`).
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_owner_scope_required.py` (new). Its Postgres tests are `pytest.mark.postgres`, and its module `pytestmark` includes `pytest.mark.owner_isolation`. Add the file to the Task 3b job's list.

**Interfaces:**
- Produces: on PostgreSQL, each helper above treats `owner_client_id=None` as `str(self._db.client_id)`. On SQLite `None` still means no predicate. Callers are unchanged.

- [ ] **Step 1: Write the failing tests.** The Postgres tests use `pg_database_config` (the superuser admin DSN, where RLS does not apply) and two instances, `"930201"` (A) and `"930202"` (B):
  - **`test_owner_edits_a_device_labelled_message_on_postgres`** (Review Focus 1):
    - A's conversation holds a message inserted with `client_id="device-x"`;
    - A's instance edits and soft-deletes it, and both succeed;
    - B's instance editing it raises `NotFoundError` (`db_errors`, from `_lock_history_owner`).

    It fails before the change, because B's edit succeeds.
  - **`test_other_owner_cannot_add_a_message_to_a_conversation_on_postgres`:** `B.add_message({"conversation_id": <A's>, ...})` raises `InputError` (it re-raises the lock's `NotFoundError` at `message_store.py:1563-1567`). It fails before the change.
  - **`test_sync_append_and_tombstone_still_work_for_the_owner_on_postgres`:** for A's own conversation, `append_message_from_sync`, then `tombstone_message_from_sync`, both succeed.
  - **`test_sqlite_message_edit_ignores_device_labels`** (Review Focus 2): on SQLite, a message with `client_id="device-x"` is editable through an instance with `client_id="1"`.
- [ ] **Step 2: Run with `TLDW_TEST_POSTGRES_REQUIRED=1`, and expect** the first two to FAIL and the rest to PASS.
- [ ] **Step 3: Implement.** One line per helper: `if owner_client_id is None and self._db.backend_type == BackendType.POSTGRESQL: owner_client_id = str(self._db.client_id)`. Place it before the existing predicate code.
- [ ] **Step 4: Verify.** Run C3, plus `tests/ChaChaNotesDB/test_chacha_message_store.py`, `tests/Sync` and the C4 grep for `_lock_history_owner|get_messages_by_sync_stable_id|tombstone_message_from_sync|append_message_from_sync|get_roleplay_resume_state|add_message`. Postgres files run with `TLDW_TEST_POSTGRES_REQUIRED=1`.
- [ ] **Step 5: Commit** `fix(chacha): message-store owner locks use the instance owner on PostgreSQL (TASK-13950.23 PR D1)`.

### Task 4b: The table-agnostic helpers raise on PostgreSQL; every caller passes an owner or opts out (AC2, part 2; AC3 app text)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`:
  - errors block near `:365`;
  - the three helpers `:35004-35070`;
  - `_manage_link` `:36880`;
  - the callers that pass no owner, listed in Measured: writing_*, `skill_registry`, the moodboards version lookup, and `_manage_link("moodboard_notes", …)`;
  - the docstring at `:19458-19461`.
- Modify: `chacha/keyword_store.py` (`:252, 566, 579, 746`), `chacha/conversation_store.py:1644`, `chacha/message_store.py:2923, :3019`.
- Modify: `tldw_Server_API/tests/lint/owner_scope_baseline.json`.
- Test: extend `tldw_Server_API/tests/DB_Management/test_chacha_owner_scope_required.py`.

**Interfaces:**
- Consumes: the Task 2 ratchet, the Task 3b job, and Task 4a's message-store derive.
- Produces, in `ChaChaNotes_DB.py` next to `ConflictError`:
  - `class OwnerScopeRequiredError(CharactersRAGDBError)`, with the message `"<helper> needs an owner on PostgreSQL (table <name>)"` and never the owner value;
  - `class OwnerScope(enum.Enum): NOT_APPLICABLE = "not_applicable"`, with a docstring: the table has no tenant `client_id`, or a parent lock taken with the owner in this transaction already proved ownership.

  The stores import them the way they import `ConflictError` today.

**The helper rule, in order:**
1. Not PostgreSQL → return `("", ())`, or return for the row helper, as today.
2. `owner_client_id is OwnerScope.NOT_APPLICABLE` → the same empty result.
3. `owner_client_id is None` → raise `OwnerScopeRequiredError`.
4. Otherwise, today's predicate.

Each owner-taking parameter's annotation becomes `str | OwnerScope | None`. `_require_selected_owner_row` keeps its `item_id is None` early return.

**Applying the rule elsewhere:**
- `_manage_link` applies the same rule. On PostgreSQL with `NOT_APPLICABLE`, it skips the parent checks.
- The generic helpers (`_get_current_db_version`, `_add/_get/_list/_update/_soft_delete_generic_item`, `_get_generic_item_by_unique_text`, `_search_generic_items_fts`) pass their `owner_client_id` through unchanged and widen their annotations.

**Callers, by table:**
- **Tenant tables** pass `self.client_id` / `self._db.client_id`: writing_*, moodboards, conversations (`conversation_store.py:1644`) and keywords (`keyword_store.py:252, 566, 579, 746`).
- **`skill_registry`** (`:28866`, `:28932`) passes `OwnerScope.NOT_APPLICABLE`, because it has no `client_id` column.
- **The message version lookups** (`message_store.py:2923`, `:3019`) pass `OwnerScope.NOT_APPLICABLE`. `_lock_message_conversation` has just locked the conversation with the owner, which Task 4a derives.
- **`_manage_link("moodboard_notes", …)`** (`:36451`, `:36577`) passes `OwnerScope.NOT_APPLICABLE`. That branch runs only on pre-v63 SQLite schemas, and a real owner would raise "Unsupported scoped keyword link".

Each opt-out carries a one-line `# OwnerScope.NOT_APPLICABLE: <reason>` comment. Before editing, re-derive every caller's current line with `git grep -n`; the dry run's list is in `plan-dryrun.md` §4.

- [ ] **Step 1: Write the failing tests:**
  - **`test_owner_none_raises_on_postgres`**, parametrized over the three helpers and `_manage_link`, via `link_note_to_keyword`'s path with `owner_client_id=None` forced:
    - on a `pg_restricted_backend` instance, calling with `owner_client_id=None` raises `OwnerScopeRequiredError`;
    - `str(exc)` does not contain the instance's client id.

    It fails before the change, because nothing is raised.
  - **`test_bypass_role_no_longer_reaches_another_owners_writing_rows`**, on `pg_database_config` (A `"930201"`, B `"930202"`):
    - `A.get_writing_template_by_name(<B's template name>)` is `None`;
    - `A.update_writing_session(<B's session id>, …, expected_version=1)` raises `ConflictError`.

    It fails before the change, because both reach B's row. `get_writing_session` is raw SQL that RLS scopes, so it is out of scope here.
  - **`test_restricted_role_paths_return_only_the_owners_rows`** (AC2's restricted-role proof), on `pg_restricted_backend` with A and B seeded:
    - through A's instance, B's template name, B's session update, B's moodboard version update and B's keyword (`search_keywords`) never return or touch B's rows;
    - the owner-`None` raise holds on this role too.
  - **`test_skill_registry_version_lookup_still_works_on_postgres`.**
  - **`test_sqlite_owner_none_stays_unscoped`**: on SQLite, `_selected_owner_filter(None) == ("", ())`, and `_require_selected_owner_row(conn, "notes", 1, None)` returns `None`.
- [ ] **Step 2: Run with `TLDW_TEST_POSTGRES_REQUIRED=1`, and expect** the first two to FAIL and the rest to PASS.
- [ ] **Step 3: Implement** the errors, the helper rule, `_manage_link` and every caller.
  - To find stragglers, run Step 1's tests plus C4. Any remaining `None` path on PostgreSQL now raises, and the traceback names it.
  - Reword `ChaChaNotes_DB.py:19458-19461`. Startup refuses a SUPERUSER/BYPASSRLS role for the content backend. This migration still handles a bypass role, because the migration may run under an admin DSN. Keep the `bypasses_rls` branch at `:19475`.
- [ ] **Step 4: Update the ratchet baseline.**
  1. Run `PYTHONPATH=. $PY -m tldw_Server_API.tests.lint.test_owner_scope_ratchet --write-baseline`.
  2. Expect `keyword_store.py` and `note_store.py` to leave the file.
  3. Expect `ChaChaNotes_DB.py` and `message_store.py` to remain as `{"calls": 0, "opt_outs": N}`.
  4. Give each opt-out entry the reason `"TASK-13950.23: tables without a tenant client_id (skill_registry), or scoped by a parent lock (messages via their conversation; moodboard_notes pre-v63 SQLite only)"`.
- [ ] **Step 5: Verify.** Run C1, C2, C3, C4 and C5, and record the counts per check.
  - For C4: `git grep -l -E '_selected_owner_filter|_selected_keyword_link_filter|_require_selected_owner_row|_manage_link|_get_current_db_version|_generic_item|writing_session|writing_template|writing_theme|moodboard|skill_registry|search_keywords|merge_keywords|rename_keyword' -- tldw_Server_API/tests`. Run every hit, plus `tests/Sync`. Postgres-parametrized files run with `TLDW_TEST_POSTGRES_REQUIRED=1`.
  - Also run `bandit -q -r` on the changed app files. There must be no new findings compared with the base.
- [ ] **Step 6: Commit** `fix(chacha): a missing owner raises on PostgreSQL instead of dropping the predicate (TASK-13950.23 PR D2)`.

### Task 5: Close-out (AC1-AC6, DoD)

**Files:**
- Modify (backlog-py only): `backlog/tasks/task-13950.23 - Make-ChaCha-owner-scoping-unconditional-on-PostgreSQL.md`; create the follow-ups.

- [ ] **Step 1: Check each AC on dev**, recording the command and the numbers:
  - **AC1:** the ratchet's seed (102) and its final state.
  - **AC2:** Tasks 4a and 4b's tests in the required job on dev's CI run.
  - **AC3:** the reworded locations.
  - **AC4:** the job's run, with 0 skipped, plus Task 3a's failure proof.
  - **AC5:** the re-keyed baseline.
  - **AC6:** the contract tests.

  Amend AC1's and AC4's text per Ruling 7 with `task edit`.
- [ ] **Step 2: File the follow-ups**, each depending on TASK-13950.23:
  - "Scope `skill_registry` per owner on PostgreSQL": its `name`/`uuid` are globally UNIQUE and it has no tenant column (`ChaChaNotes_DB.py:3533`).
  - "Add RLS to `suggestion_snapshots`, `suggestion_generation_links` and `flashcard_keywords`" (`rls_coverage_baseline.txt:250-251`).
  - "Refuse a SUPERUSER/BYPASSRLS role for the ChaCha backend at startup". Today only the content backend is checked (`media_db/runtime/factory.py:93-129`).
  - "Decide whether direct `CharactersRAGDB(...)` constructions with non-user client ids may run on PostgreSQL": workflow_engine, mcp_*, persona_memory_* and Notes_Library (research-A §2).
  - "`endpoints/chat.py:7032` passes an owner taken from the row, possibly empty".
  - "Extend the owner-scope ratchet to the conditional-predicate shapes": `if owner is not None:` plus predicate, and `IfExp` (Ruling 7).
  - "Ratchet optional-owner helpers outside ChaCha": the app-wide prototype found 45 names.
  - "`message_store.py:2292` passes `sync_client_id` as the conversation owner". This predates the task; check that it is the tenant.
  - "Widen the `chacha-owner-isolation` job to the other ChaCha owner-isolation files" (research-B §4), if the runtime allows.
- [ ] **Step 3: Record** in the backlog notes and the final summary:
  - the rulings;
  - the PR numbers;
  - the measured counts;
  - the job's runtime.

  Then set the task to Done. Stage with `git add -A backlog/tasks`.

## Self-review

- **Spec coverage:**

  | AC | Task |
  |---|---|
  | AC1 | Task 2 |
  | AC2 | Tasks 4a (derive) and 4b (raise), per Ruling 1, with the restricted-role proof `test_restricted_role_paths_return_only_the_owners_rows` |
  | AC3 | Task 3a (tests' text), Task 4b (app docstring) |
  | AC4 | Tasks 3a and 3b |
  | AC5 | Task 1 |
  | AC6 | Tasks 1, 2 and 3b contract tests; Task 3a's pytester self-test is in `TEST_SEAM_GATE_SUITES` |
  | DoD | Task 5 |

- **Names across tasks:**
  - `OwnerScope.NOT_APPLICABLE` is counted in Task 2 (as call arguments) and introduced in Task 4b. A count of 0 before Task 4b needs no entry.
  - The `owner_isolation` marker and the `chacha-owner-isolation` job come from Tasks 3a and 3b. Task 4a adds `test_chacha_owner_scope_required.py` to that job, and Task 4b extends it.
  - `TEST_SEAM_GATE_SUITES` gains the scope ratchet (Task 1), the owner-scope ratchet (Task 2) and the skip-as-failure self-test (Task 3a).
  - Task 4b's message-version opt-outs rely on Task 4a's derived conversation lock, so 4a merges first.
- **Fails-before (Review Focus 2 of the roadmap):** these tests fail on the base:
  - Task 4a: `test_owner_edits_a_device_labelled_message_on_postgres` (B's edit) and `test_other_owner_cannot_add_a_message_to_a_conversation_on_postgres`;
  - Task 4b: `test_owner_none_raises_on_postgres` and `test_bypass_role_no_longer_reaches_another_owners_writing_rows`;
  - Task 3a: the pytester test, before the hook exists.
- **Resolved from the review and dry run** (`plan-review.md`, `plan-dryrun.md`):
  - **Message-store callers without an owner** (review C1, dry run C1) → Task 4a derives the owner instead of raising.
  - **add_message behaviour change** (review C2) → Ruling 1, plus a test.
  - **`NotFoundError` vs `ConflictError`** (review I1, dry run C3) → fixed in the tests.
  - **Self-test outside the required gate and the text-search marker detection** (review I2) → added to `TEST_SEAM_GATE_SUITES`; detection is now by AST.
  - **Collection-time skips** (review I3, dry run C5) → removed `importorskip`, plus the junit check.
  - **File selection** (review I4) → added `test_chacha_message_store.py` and wrote down the selection rule.
  - **Runtime** (review I5, dry run C4) → a separate parallel job, path-gated per the owner's 2026-11-21 decision, run serially with `timeout-minutes: 40`.
  - **moodboard_notes** (review I6) → that branch is unreachable on PostgreSQL; it opts out because a real owner would raise.
  - **Writing getters** (dry run C2) → the tests target the generic-helper methods.
  - **Dry run C6-C12 and review M1-M6** → each applied in its step.
- **Resolved from the scoped re-review** (`plan-rereview.md`):
  - **`add_message`'s error** (I1) → `InputError` in Ruling 1 and the 4a test.
  - **Contract inventories for the new job** (I2) → Task 3b's Files list.
  - **PR C over 400 lines** (I3) → split into Tasks 3a and 3b.
  - **Minors:**
    - the direct instances named in Ruling 1;
    - `(lineno, col_offset)` ordering;
    - `tombstone_message_from_sync` is no longer an edit site;
    - the `:2292` follow-up;
    - the whole `tests/CI` directory in verify.
