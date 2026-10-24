# PostgreSQL Dev Test Failures: ChaCha Sequence Sync, Admission-Capture Preconditions, RAG Teardown Hang (TASK-13900, TASK-13901)

> **Review folded in (r2, 2026-10-22).** The review is `<scratchpad>/review_plan_fu2_pg_tests.md`: Ready with fixes, 0 Blocker, 1 Important, 5 Minor. r1 is kept as `<scratchpad>/plan_fu2_pg_tests.r1.md`. All r2 counts were re-measured on `e06eeec68c`.
> - **I1, adopted per the coordinator's ruling.**
>   - Task 2's parametrized metadata case now changes its setup, not its expectation. At `:1014` the admission no longer carries `extra_metadata`, so the drift is a version-neutral first metadata write.
>   - `:1022`'s `== 1` and its comment stay as they are. The case now pins `match="stale_parent"` (`:1040`).
>   - Only the A→B→A case's expectation changes (`:1050`, `== 3`), justified by #3229's quoted hunk.
>   - So one setup line and one expectation line change, plus the `match` pin.
>   - Review Focus 1 no longer cites `..._drift_between_validation_and_proof`, which pins a different fence (`stale_source`).
> - **Minor 1.** Task 5's `normalize --check` takes the task files, not the directory, which crashed.
> - **Minor 2.** Task 4 gains two contract tests. One is a 3 s RED that cannot hang (no PostgreSQL session left idle in a transaction). The other pins read-your-writes inside a transaction (Review Focus 3).
> - **Minor 3.** The #3269 conflict picture is restated: 164 conflicting paths both before and after this plan; one removed (the helper's add/add) and one added (the history test, three one-line hunks); `test_chacha_postgres_sequence_monotonicity.py` add/add either way.
> - **Minor 4.** Task 4's caller run widens to 24 files: the media-detail route's tests are added.
> - **Minor 5.** Task 1 Step 5 runs Bandit itself.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- ChaCha's PostgreSQL sequence sync logs a table it cannot advance and moves on, without touching the caller's transaction, as it did before #3084 (TASK-13901 case 1).
- The four admission-capture tests pass and keep the coverage #3084 designed:
  - the parametrized metadata case writes metadata the version-neutral way;
  - the A→B→A case expects the version bump that TASK-13569 (#3229) deliberately added before #3084 wrote it (TASK-13901 cases 2-5).
- A Media DB `MediaFiles` lookup no longer holds the request's PostgreSQL connection idle in a transaction. So `GET /media/{id}` stops stalling and leaking that session (TASK-13900 AC 1).
- The RAG restricted Media store's teardown fails fast and names a leaked session, instead of hanging the run (TASK-13900 AC 2).

**Architecture:**
- **TASK-13901 case 1 (product).**
  - `backends/postgres_sequence_maintenance.py` becomes byte-identical to open PR #3269's copy (head `81b0bb5836`, blob `f169543591`): the loop calls the existing `advance_postgres_serial` per table and takes `continue_on_error: bool = False`.
  - ChaCha's `_sync_postgres_sequences` passes `continue_on_error=True`.
  - Media keeps the default and still raises. A new unit guard pins that.
- **TASK-13901 cases 2-5 (tests).** In `test_history_selection_transactions.py`:
  - **Setup (`:1014`).** The parametrized test admits without `extra_metadata`, so its metadata drift is a first write. #3229 keeps a first write version-neutral, so `:1022`'s `== 1` holds as written.
  - **Pin (`:1040`).** The metadata case pins `match="stale_parent"`: the input state digest refuses it.
  - **Expectation (`:1050`).** The A→B→A test expects version 3, because #3229's bump is what makes a metadata undo observable.
  - The product is unchanged.
- **TASK-13900 AC 2 (test fixture).** `restricted_media_store`'s teardown calls the shared `_drop_restricted_role(admin, role)` from `tests/_plugins/postgres.py` (TASK-13861 PR D) instead of a bare `DROP OWNED BY`.
- **TASK-13900 AC 1 (product).**
  - `MediaFilesRepository.get_for_media` reads through `db.execute_query(...)` instead of `db.get_connection()`. TASK-13233 already made `list_for_media` do the same.
  - Two contract tests pin the fix: no session is left idle in a transaction, and a lookup inside a transaction still reads its own uncommitted row.
  - One test that monkeypatched the old internal call (`_fetchall_with_connection`) moves its patch to `execute_query`; its expectation is unchanged.

**Tech Stack:** Python 3.12, psycopg 3.3.6 / psycopg_pool 3.3.2, FastAPI 0.142.1, pytest 9.1.1 with pytest-timeout (thread method), and the running `tldw_postgres_test` container (PostgreSQL 18) through the repo's fixtures.

**Spec:**
- `backlog/tasks/task-13900 - RAG-restricted-Media-DB-test-leaves-a-session-idle-in-transaction-and-hangs-its-own-teardown.md`:
  - AC 1: the test is fixed or its cause recorded, and no session of the test role is left in a transaction.
  - AC 2: the teardown fails fast.
  - Its note (PR #3300 review, M3) names the failing node: `test_permission_first_auth_trash_restore_without_preseeded_scope[postgresql-X-API-KEY]`.
- `backlog/tasks/task-13901 - Tests-five-PostgreSQL-tests-have-failed-on-dev-since-3084.md`:
  - AC 1: each test passes, or is fixed or removed with a reason.
  - AC 2: the cause is identified and stated.
- Both were filed by TASK-13861 PR D (#3300). Measurements: `<scratchpad>/impl_chacha_prD.md` (Ruling 4, Concerns 1-2, "M3: what the re-run showed") and `<scratchpad>/clean3300_review.md` (M3, Recommendations).

## What was measured

All on `origin/dev` **`e06eeec68c`** (merge of #3300), in this planner's isolated worktree, with Python 3.12.13 and PostgreSQL through `pg_database_config` / `pg_restricted_backend`. Line numbers are `e06eeec68c`'s.

Logs and probe scripts are in `<scratchpad>` with the prefix `fu2_`:
- r1: `fu2_red_13901.log`, `fu2_dev_baseline.log`, `fu2_rag_alone.log`, `fu2_rag_teardown_only.log`, `fu2_rag_fix_file.log`, `fu2_probe_nobump.log`, `fu2_probe_codes.log`;
- r2: `fu2_r2_task2_subset.log`, `fu2_r2_task2_files.log`, `fu2_r2_mutant_state.log`, `fu2_r2_contract_dev.log`, `fu2_r2_contract_preseam.log`, `fu2_r2_contract_fix.log`, `fu2_r2_callers.log`, `fu2_r2_combined.log`;
- the full r2 change set: `fu2_final_r2.diff`.

Every change was written into the worktree, run, and reverted. The worktree is clean.

### TASK-13901 case 1: the sequence test (a #3084 regression in product code)

- **RED.** `test_chacha_postgres_sequence_ownership.py::test_chacha_sync_continues_after_table_error_without_settling_caller` fails in its call phase with `DatabaseError: PostgreSQL query execution failed`, raised from `db._sync_postgres_sequences(conn)` at `:251`. The test's patched `MAX("id")` query on `character_cards` fails, and the error leaves the loop.
- **Cause.** Two PRs extracted Media's forward-only loop independently.
  - **#3253** (`1f7f780a4d`, merged 2026-10-17) added `backends/postgres_sequences.py::advance_postgres_serial`, together with this test.
    - ChaCha kept its per-table `try/except BackendDatabaseError`, which logs `"Failed to synchronize PostgreSQL sequence for {}.{}: {}"` and continues.
    - Its commit message: "Preserve each wrapper's existing ownership and error policy, with native owner and rollback coverage."
    - The helper's docstring: "The caller owns this public-schema serial and chooses its error policy."
  - **#3084** (`eb0c8c9e52`, merged 2026-10-21 as `d47c9ccf58`) added `backends/postgres_sequence_maintenance.py::sync_owned_postgres_sequences`, an inlined copy of the same body with no error handling. It replaced ChaCha's whole loop with one call to it, which deleted the `except` above.
- **#3084 did not change the error policy deliberately.** Its plan, `Docs/Plans/IMPLEMENTATION_PLAN_uat_chacha_postgres_sequence_20261008.md`, says: "Extract the existing Media fixed-inventory forward-only sequence loop into a shared backend helper. Preserve Media's public function, inventory and callers. ChaCha delegates using its existing inventory." It names no error-policy change. It ran "three ChaCha ownership cases", the three this file had before #3253 added the restricted ones. So the product is wrong, not the test.
- **`advance_postgres_serial` is dead on dev.** `git grep advance_postgres_serial` finds only its definition.
- **Open PR #3269 already has the same fix.** It is the owner's UAT C04 branch (1168 files, head `81b0bb5836`), and the fix is in commit `12c929582d`, "fix(uat): integrate validated workflow repairs through C04", 2026-10-18:
  - `continue_on_error`;
  - the call to `advance_postgres_serial`;
  - `continue_on_error=True` in ChaCha.

  Task 1 copies it byte-for-byte (see Conflicts).
- **GREEN with Task 1's code.** The five sequence files (`test_chacha_postgres_sequence_ownership.py`, `test_chacha_postgres_sequence_monotonicity.py`, `test_media_postgres_sequence_monotonicity.py`, `test_media_postgres_sequence_ownership.py`, `test_media_db_schema_bootstrap.py`) give `130 passed` (79.96 s): dev's 129 plus the new guard.
- **Mutant.** With `continue_on_error: bool = True` as the default, the new Media guard fails with `Failed: DID NOT RAISE DatabaseError` (`1 failed`).
- **Bandit.** `uvx bandit -q -ll` on `postgres_sequence_maintenance.py`, `postgres_sequences.py` and `ChaChaNotes_DB.py` gives 0 results, on dev and with Task 1.

### TASK-13901 cases 2-5: the admission-capture tests (written against a reversed contract)

- **RED.** Two pairs of cases fail; the two functions give `4 failed, 14 passed`.
  - `test_native_admission_capture_refuses_changed_input_or_capture[sqlite-metadata]` and `[postgres-metadata]`: `assert 2 == 1` at `:1022`.
  - `test_native_admission_capture_refuses_metadata_change_and_restore[sqlite]` and `[postgres]`: `assert 3 == 1` at `:1050`.
- **Cause.** `message_store.py:3210-3212` was added by **#3229** (`69e594c495`, "Protect Edit resend with original target admission (TASK-13569)", merged 2026-10-15, before #3084):

  ```python
          if advance_history:
              if metadata_message_id is not None:
                  # Undo must not restore an opaque revision within one timestamp tick.
                  conn.execute("UPDATE messages SET version = version + 1 WHERE id = ?", (message_id,))
  ```

  A metadata rewrite, on a message that already has a metadata row, bumps `messages.version`. A first write and a true no-op do not. #3229 pinned both halves in `test_history_branch_admission.py`:
  - `test_metadata_undo_cannot_restore_the_original_opaque_revision` (`:296-327`) asserts `current["revision"] != original["revision"]` after an A→B→A rewrite inside one fixed SQL clock tick;
  - `test_initial_metadata_and_true_noop_preserve_the_accepted_input_version` (`:330-340`) pins the version-neutral first write.
- **What #3084 meant.** It added the four cases in `eb0c8c9e52`. Its design record wants metadata drift that leaves the physical version alone: "Real storage regressions … cover … metadata and asset drift without physical-version changes" (`Docs/Design/uat_a01_native_retry_validation_20261011.md:20`). It also says it redefines neither revision domain: "it does not redefine either revision domain" (`:4`).
  - Its test comments say the same: "Metadata must be fenced without a physical version bump".
  - Its mistake was the write path. Both tests admit with `extra_metadata={"original": True}`, so the drift that follows is a rewrite, which #3229 bumps.
  - `69e594c495` is an ancestor of `eb0c8c9e52`'s parent, so these four cases have never passed in this history.
  - #3084's own API test, `Chat_NEW/integration/test_history_selection_api.py::test_native_admission_capture_is_opt_in_read_only_and_binds_the_real_hash`, admits without metadata and writes it once (reviewer, SQLite-only).
- **The rewrite contract cannot be changed: it is load-bearing.** With the bump line disabled, the four #3084 cases pass and all four `test_metadata_undo_cannot_restore_the_original_opaque_revision` cases fail (`assert 'sha256:…' != 'sha256:…'`): `4 failed, 22 passed`. So the product stays.
- **What refuses each drift** (probe `fu2_probe_codes.log`, both backends):
  - A first metadata write keeps version 1. `validate_history_admission` refuses it with `stale_parent`, from `_validate_history_parent`'s input-state-digest check (`message_store.py:1125-1126`).
  - A rewrite gives version 2, and A→B→A gives version 3. Both are refused with `stale_parent`, because the receipt's `input_message_revision` "1" no longer matches.
- **r2 change: setup at `:1014`, the `match` pin at `:1040`, `== 3` at `:1050`.**
  - The two functions give `18 passed`: 9 SQLite and 9 PostgreSQL, including `[sqlite-metadata]` and `[postgres-metadata]` (`fu2_r2_task2_subset.log`, 28 s).
  - The two whole files (`test_history_selection_transactions.py` and `test_history_branch_admission.py`) give `200 passed, 1 skipped` (306.90 s, `fu2_r2_task2_files.log`). Of 201 collected items, 99 carry the `postgres` parameter: 98 passed, plus 1 skip. The other 102 (the `sqlite` parameter, and SQLite-only tests) passed.
  - The skip is `test_history_sqlite_function_follows_real_connection_lifetime[postgres]` (`:645`, `pytest.skip("SQLite connection lifetime")`), as on dev.
- **The `stale_parent` pin catches a lost digest check.** With `message_store.py:1125`'s input-state-digest check disabled, the metadata case is refused by the capture fence instead. `[sqlite-metadata]` and `[postgres-metadata]` then fail with `Regex pattern did not match. Expected regex: 'stale_parent' Actual message: 'stale_source'` (`2 failed, 14 passed`).

### TASK-13900: the RAG teardown hang (a product connection leak, plus a fixture that cannot fail fast)

- **The hang, once, on unchanged dev.** I ran `-v -o timeout=150` on the single node `test_permission_first_auth_trash_restore_without_preseeded_scope[postgresql-X-API-KEY]`.
  - The test reported `FAILED` 46 s after start. It fails alone, not only in file order: it is the first PostgreSQL case of `authenticated_media_client`.
  - Its teardown then blocked. A read-only `pg_stat_activity` probe showed `DROP OWNED BY "rag_retrieval_38b7…"` (`active`, wait `Lock`) behind the role's session `idle in transaction` for 37 s on `SELECT * FROM MediaFiles WHERE media_id = $1 AND file_type = $2 AND deleted = FALSE ORDER BY id DESC LIMIT 1`.
  - pytest-timeout (thread method, 150 s) never ended it. I killed it at 4:40 (`rc=143`).
  - It left role `rag_retrieval_38b77f4b453041a2861624a919fc4044` and database `tldw_test_be9ab08c`. I dropped those two, by name, with `<scratchpad>/fu2_pg_drop_mine.py`. Nothing else was touched.
- **Bounded RED** (same node, with Task 3's teardown only): `1 failed, 1 error in 50.29s`.
  - Call phase: `assert client.get(f"/media/{media_id}", headers=headers).status_code == 200` → `assert 500 == 200`, logged `Unexpected error fetching details for media 1`.
  - Teardown: `AssertionError: The test left 1 session(s) of rag_retrieval_5557… or its database holding locks (a leaked checkout?): [{… 'state': 'idle in transaction', 'query': 'SELECT * FROM MediaFiles WHERE media_id = $1 AND file_type = $2 AND deleted = FALSE ORDER BY id DESC LIMIT 1'}]`.
  - No role was left behind.
- **Cause (product).**
  - The request path is: `GET /media/{id}` → `get_full_media_details_rich` (`media_details_service.py:261-317`) → `_has_original_file` (`:280`) → `MediaFilesRepository.get_for_media` (`media_files_repository.py:76-96`).
  - `get_for_media` calls `db.get_connection()` (`:84`). On PostgreSQL that checks a connection out of the pool and pins it in the request's context var (`connection_lifecycle.py:67-85`). The `SELECT` then opens a transaction that nothing ends. The session sits `idle in transaction`, holding `AccessShareLock` on `MediaFiles`.
  - The rest of the request (`_load_keywords`, `_load_versions`) goes through `execute_query`, which checks out a second connection.
  - The fixture's pool has one slot (`pool_size=1`; psycopg_pool's `max_size` is `pool_size`). So the second checkout waits `pool_timeout` (30 s), fails, and the route answers 500.
  - In production the default pool has 10 slots, so each detail `GET` holds two of them; the reviewer did not measure the production cost. `get_media_db_for_user`'s teardown (`DB_Deps.py:231-250`) then calls `release_context_connection()`, and the pool rolls the session back.
  - A path that never calls `release_context_connection()` keeps the session idle in its transaction for good. This test's `media_db` override (`:298-299`) is one such path; any non-request caller would be another.
  - `list_for_media`, a sibling read, was fixed the same way by TASK-13233 (`76d60dbbd6`, `:112`): "# execute_query owns its connection, including when called from a worker."
- **Cause (fixture).** `restricted_media_store`'s teardown (`:73-76`) runs a bare `DROP OWNED BY` / `DROP ROLE` with no `lock_timeout`, so any leaked session hangs the run. `_drop_restricted_role` (`tests/_plugins/postgres.py:430-463`) is TASK-13861's fail-fast version.
- **Contract-level RED (r2, 3 s, cannot hang).** On dev's repository, `test_lookup_leaves_no_postgres_session_in_a_transaction` fails with `assert 1 == 0`. `test_lookup_inside_a_transaction_reads_its_uncommitted_row` passes on SQLite and PostgreSQL. Together: `1 failed, 2 passed` (`fu2_r2_contract_dev.log`).
- **GREEN with Tasks 3-4:**
  - the 4 `trash_restore` cases: `4 passed` (15.04 s);
  - the whole RAG file: `68 passed` (40.25 s). Nothing after the 19th item had run on dev before, and nothing else in the file fails.
- **The contract file.**
  - After the fix, still patching the old call: `2 failed, 11 passed` (22.07 s). The two are `test_database_query_failure_is_not_reported_as_missing_file[sqlite|postgresql]`: `Failed: DID NOT RAISE DatabaseError`. The test monkeypatches `_fetchall_with_connection`, the call the fix removes.
  - With the patch moved to `execute_query`: `13 passed` (5.36 s).
- **Callers.** 23 test files name `get_media_file`, `MediaFilesRepository`, `has_original_file`, `get_for_media`, `get_full_media_details_rich` or `_media_db_legacy_stub`. With `Media_Ingestion_Modification/test_media_versions.py` (the versions route, which the grep misses), that makes 24 files, giving `563 passed` (144.31 s, `fu2_r2_callers.log`). That is r1's 15 files (454) plus the reviewer's 9 (106) plus the 3 new contract items.

### Together, gates, PR #3269, and the cluster

- **Dev baseline** for the focused set without the RAG file (the five sequence files, the contract file and the two admission-capture functions): `5 failed, 152 passed` (224.57 s). The 5 are exactly TASK-13901's.
- **Branch, all r2 changes:** that set plus the RAG file gives `229 passed` (333.98 s at load 11-13, `fu2_r2_combined.log`). That is r1's 226 plus the 3 new contract items.
- **Lint and security:**
  - `ruff check` on all 7 touched files: `All checks passed!`.
  - `uvx bandit -q -ll` on `postgres_sequence_maintenance.py`, `postgres_sequences.py`, `media_files_repository.py` and `ChaChaNotes_DB.py`: 0 results, on dev and on the branch.
- **Ratchets:**
  - `test_quality_triage.py --enforce`: `[test-triage] OK — no offenses beyond the baseline.`
  - `test_rls_coverage_ratchet.py`, `test_test_quality_ratchet.py`, `test_scope_predicate_ratchet.py` and `test_private_coercion_ratchet.py`: `37 passed`.
  - None of the touched files is in `Helper_Scripts/ci/scope_predicate_baseline.txt`.
  - `task normalize --check` on the two task files: rc 0.
- **PR #3269** (`git merge-tree --write-tree --name-only` with head `81b0bb5836`): 164 conflicting paths onto dev, and 164 onto this plan's r2 tree.
  - Removed: the helper's `CONFLICT (add/add)`.
  - Added: `test_history_selection_transactions.py`, three one-line hunks at `:1014`, `:1040` and `:1050`.
  - `ChaChaNotes_DB.py`: in `_sync_postgres_sequences` only one blank line conflicts. It conflicts with dev in 8 other regions either way.
  - `test_chacha_postgres_sequence_monotonicity.py`: add/add either way, unaffected by this plan.
- **Cluster.** After every run I listed the `tldw_rls_*` / `rag_retrieval_*` roles and the `tldw_test_*` databases (`<scratchpad>/fu2_pg_inventory.py`, `fu2_pg_db_activity.py`). Mine are all gone.
  - During r1, another session's sweep (pid 74265, worktree `agent-aab5fe3febc96709e`) was hung on this same defect. The reviewer saw it removed later.
  - At r2's end, `tldw_test_aa9139dc` (created 2026-10-20) remains, plus databases that other sessions' running pytests had just created. I left them alone.
- **Runs of the hanging test.** Exactly one run was unbounded, on dev; it hung and was killed. Every later run of the RAG file, in r1 and r2, had Task 3's fail-fast teardown applied.

## Global Constraints

- **Environment:**
  - `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`.
  - Tests: `TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 $PY -m pytest -p no:cacheprovider -q -o timeout=200 <paths>`. Use the running `tldw_postgres_test` container through the fixtures; never roll your own database setup.
  - Never run the full suite. Redis is not involved.
- **The hang rule.**
  - Never run `tests/RAG/test_restricted_postgres_media_retrieval.py` on a tree without Task 3's teardown. It hangs for minutes and leaves a role behind (the RED is recorded above).
  - After any run that is killed, list the `tldw_rls_*` / `rag_retrieval_*` roles and the `tldw_test_*` databases. Drop only the ones your run created, by name.
- **Branch:** `fix/pg-dev-test-failures-13900-13901`, from `origin/dev`. The PR targets `dev`.
  - If `origin/dev` has moved past `e06eeec68c`, re-run each RED on the new tip first and record the commit in the notes.
  - If PR #3269 has merged by then, Task 1 is already on dev and its RED passes. Skip Task 1's product steps, keep its guard test, and say so in TASK-13901's notes.
- **Commits:**
  - Every commit ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - Never `--no-verify`, `git stash` or force-push.
- **Test first:** each task's RED is the measured failure above, on the tree as the previous task left it.
- **Test changes (only these):**
  - **Task 2:**
    - one setup line: `:1014` admits without `extra_metadata`;
    - one expectation line: `:1050` becomes `== 3`, justified by #3229's quoted hunk;
    - one added pin: `:1040` gains `match="stale_parent"` for the metadata case.

    `:1022`'s `== 1` and its comment are kept.
  - **Task 4:** a monkeypatch target moves, and its expectation is kept. Two tests are added.
  - No test is skipped or deleted.
- **Unchanged:**
  - Media's sequence error policy: it raises, pinned by the new guard;
  - `message_store.py`;
  - `connection_lifecycle.get_connection` and `release_context_connection`;
  - the RAG test's `media_db` override;
  - `tests/_plugins/postgres.py`.
- **`postgres_sequence_maintenance.py` stays byte-identical to PR #3269's copy.** `git hash-object` on it prints `f1695435919059df6b340ab055c8784e6965aadf`. Do not reformat or edit it: any difference brings back #3269's add/add conflict.
- **Test-quality ratchet (required gate):** `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` exits 0, and `tldw_Server_API/tests/CI/test_test_quality_ratchet.py` passes.
- **Global defaults:** none flips.
  - `continue_on_error` defaults to `False`. Only ChaCha opts in.
  - `get_for_media` changes connection for all its callers. Task 4 Step 6 runs every test file that reaches it: a `git grep -l` over six names (23 files on `e06eeec68c`), plus `test_media_versions.py`. Re-run the grep at execution.
- **Docs:** none changes. Nothing user-facing changes, and no mirrored doc is touched. The plan copy goes to `Docs/superpowers/plans/` (not mirrored), with no trailing whitespace.
- **Backlog:**
  - `tools/backlog-py` only, through the shell function `B` defined in Setup (`PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task "$@"`). `T` runs pytest with the flags above. Both are functions, so they work in zsh and bash.
  - `--append-notes` only, never `--notes`. Stage with `git add -A backlog/tasks`.
  - New ids are computed under the id lock right before each `task create` (Task 4 Step 7).

## Decisions for the owner (recommendation first)

None. The code settles what looked like choices:
- **Cases 2-5: change the tests, not the product.**
  - TASK-13569 (#3229) made a metadata rewrite bump the version on purpose, under test, as its defence against A→B→A.
  - #3084's design wanted version-neutral metadata drift, which a first write still gives. Its own record says it redefined no revision domain.
  - The parametrized case therefore changes its write path, and keeps its expectation. Only the A→B→A case, which cannot be version-neutral under #3229, changes its expectation.
- **Case 1: restore ChaCha's continue-on-error policy.** #3253 set it on purpose, and #3084 dropped it in a merge without saying so. PR #3269 carries the same restoration.

The PR needs the owner's own `## Change summary` (ADR-004). No recorded waiver covers this round. It is an input, not a decision.

## Review Focus

1. **Task 2 changes one expectation (`:1050`), and the change it follows is #3229's, not #3084's.**
   - The brief allows changing an expectation only "where the test pins behaviour that #3084 deliberately changed". Here #3084 added an A→B→A test that contradicted TASK-13569's earlier, deliberate bump (quoted above). #3229 has no metadata delete path, so a version-neutral A→B→A is impossible.
   - The parametrized metadata case keeps `== 1` by changing its setup to a first write. It now pins `stale_parent`, which the input-state-digest check produces. The mutant without that check fails it with `stale_source`.
   - `test_native_admission_capture_refuses_drift_between_validation_and_proof` is not evidence here. It pins a different fence: `stale_source`, for a change injected after `_validate_history_parent` has passed.
   - The design record's "metadata and asset drift without physical-version changes" still holds for the parametrized metadata and asset cases. It no longer covers A→B→A, which bumps the version and is refused through it. TASK-13901's notes record this.
2. **A ChaCha migration with a table it cannot advance now succeeds and logs `Could not advance owned PostgreSQL sequence for {table}.{column}`, as before #3084.**
   - That table keeps its old high-water mark, so a later insert can collide (`IntegrityError`) instead of the migration failing.
   - The backend logs the SQL error itself at ERROR (`postgresql_backend.py:1152`). #3269's warning drops the exception text, which #3253's included.
   - The failing table's savepoint is rolled back, and the caller's rows stay, as the test pins (`count(*) == 1`, keyword `nextval == 902`).
   - Media still raises: `test_sync_postgres_sequences_raises_a_failing_table`.
   - Narrowing: if the single up-front `pg_depend` scan fails, ChaCha now raises, where #3253's per-table lookups were caught. Nothing tests this. The reviewer judged it acceptable.
3. **`get_for_media` now reads on its own pooled connection outside a transaction.**
   - It no longer sees uncommitted writes made on the request's pinned connection outside `db.transaction()`.
   - Inside a transaction it still uses the transaction's connection (`execute_query` takes `_get_txn_conn()`). That is now pinned by `test_lookup_inside_a_transaction_reads_its_uncommitted_row` on SQLite and PostgreSQL.
   - This matches `list_for_media` since TASK-13233. The 24 caller files pass.
   - On SQLite file DBs it reads through an ephemeral connection, so it sees committed data only. The `existing_only` SQLite wrapper (`bootstrap_lifecycle_ops.py:90-97`) still routes it through `get_connection()`.
4. **Six other Media DB calls still pin the request connection on PostgreSQL outside a transaction:**
   - reads: `keyword_access_ops.py:86` (`fetch_media_for_keywords`), `sync_log_ops.py:35` (`get_sync_log_entries`), `chunk_batch_ops.py:181` (`process_chunks`' parent check) and `visual_document_ops.py:114` (`list_visual_documents_for_media`);
   - writes: `visual_document_ops.py:59` and `:142`.

   Each one costs two pool slots per request, and leaks on a path that never releases. None is on this test's path. TASK-<N1> tracks them.
5. **The fixture now terminates sessions on teardown.**
   - `_drop_restricted_role` terminates every non-idle client session on the test's own database before its retry (TASK-13861 M1). The fixture's `admin` connects to the per-test database (`pg_database_config` → `pg_temp_db`), so this stays per test. `_drop_database` does the same a moment later.
   - A future leak fails that test with the session named, and leaves no role behind.
   - It does not catch a session that is `idle` (not in a transaction), which holds no lock.

## ADR assessment

- **No new ADR.**
  - **ADR-020** (shared PostgreSQL content backend): Task 1 keeps its forward-only sequence rule, which #3084's plan cites, and restores only ChaCha's error policy.
  - **ADR-049** (history selection ownership): unchanged. Task 2 corrects two tests to the contract TASK-13569 implemented.
  - **ADR-080** governs ChaCha checkouts, not the Media DB's request connection. Task 4 follows the existing TASK-13233 `execute_query` pattern, which is not a durable new rule.
- **Process ADRs:** ADR-004 (the owner's Change summary), ADR-006 (Bandit on the touched scope), ADR-059 (backlog-py).
- I searched `Docs/ADR/README.md` for connection, checkout, sequence, serial, revision and history selection.

## PR split

**One PR, one commit per task:**
- **Small, and verified together.** About 20 product lines change, and the focused set gives `229 passed` with all four tasks.
- **No hold-up either way.** The two backlog tasks share no file, and neither needs the other. Two PRs would pay the seven required statuses twice, and would need two owner Change summaries and two reviews.
- **Reverts stay targeted.** Each task is its own commit.

## Conflicts with other plans and open PRs

| Place | This plan | Other | How to merge |
|---|---|---|---|
| `backends/postgres_sequence_maintenance.py` | Task 1 makes it byte-identical to #3269's copy (blob `f169543591`) | PR #3269 (open, 1168 files) adds the same file | `git merge-tree` of #3269 onto dev today: `CONFLICT (add/add)`. Onto this plan's tree: no conflict in this file. |
| `ChaChaNotes_DB.py:28199` | Task 1's 3-line call, identical to #3269's | #3269 | Onto this plan's tree, this method's only conflict is one blank line. #3269 already conflicts with dev in 8 other regions of this file. Other open PRs touch the file too: #3299, #3298, #3226, #3071, #3020, #2959 and #2957. Re-check with `git merge-tree` at execution. |
| `test_history_selection_transactions.py:1014`, `:1040`, `:1050` | Task 2's three one-line edits | #3269 (its copy matches dev); #3071; #3159 (base not `dev`) | Onto this plan's tree, #3269 conflicts on the three lines. Keep dev's side, which is this plan's. |
| `test_chacha_postgres_sequence_monotonicity.py` | untouched | #3269 | add/add either way, unaffected by this plan. |
| `test_media_db_schema_bootstrap.py` after `:872` | Task 1 adds one test | #2903 (62 files) | Re-check at execution. An insertion between two tests merges unless #2903 edits `:870-875`. |
| `media_files_repository.py`, `test_media_files_backend_contract.py`, `test_restricted_postgres_media_retrieval.py` | Tasks 3-4 | No open PR among 44, and no other `plan_fu2_*` | None. `plan_fu2_claims.md` hand-rolls a new restricted fixture that copies this file's pattern, with a `lock_timeout`; no file overlaps. TASK-<N2> covers the pattern. |

Across all paths, #3269 conflicts in 164 paths both before and after this plan: this plan removes one conflict and adds one.

---

## Setup (once)

```bash
cd <your worktree>
git fetch origin dev --quiet
git switch -c fix/pg-dev-test-failures-13900-13901 origin/dev
git log -1 --format='%h %s'   # record it; the plan measured e06eeec68c
export PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
B() { PYTHONPATH=tools/backlog-py/src "$PY" -m backlog_py --cwd . task "$@"; }
T() { TLDW_TEST_NO_DOCKER=1 TLDW_TEST_POSTGRES_REQUIRED=1 "$PY" -m pytest -p no:cacheprovider -q -o timeout=200 "$@"; }
mkdir -p Docs/superpowers/plans
cp <scratchpad>/plan_fu2_pg_tests.md Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md
grep -c ' $' Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md   # expect 0
B edit TASK-13901 -s "In Progress"
B edit TASK-13900 -s "In Progress"
docker ps --format '{{.Names}} {{.Status}}' | grep tldw_postgres_test   # expect "Up"
```

The plan copy goes in Task 1's commit.

---

### Task 1: ChaCha's sequence sync continues past a failing table (TASK-13901 case 1)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py`: the whole file, replaced by PR #3269's version.
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:28199` (in `_sync_postgres_sequences`, `:28193-28199`).
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py:221-272` (existing; the RED).
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`: add one test after `test_sync_postgres_sequences_positive_scalar_uses_max_id_branch`, which ends at `:872`.

**Interfaces:**
- Consumes, unchanged: `advance_postgres_serial(backend, conn, *, table_name, column_name, sequence_schema, sequence_name) -> None` and `PostgresSequenceBackend` from `backends/postgres_sequences.py`; `backends.base.DatabaseError`.
- Produces: `sync_owned_postgres_sequences(db, conn, owned_columns, *, continue_on_error: bool = False) -> None`. `PostgresSequenceMaintenanceDB` is still exported, and Media's wrapper imports it.

- [ ] **Step 1: Run the existing test to see it fail**

```bash
T "tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py::test_chacha_sync_continues_after_table_error_without_settling_caller"
```

Expected: `1 failed`. The error is `E   tldw_Server_API.app.core.DB_Management.backends.base.DatabaseError: PostgreSQL query execution failed`, raised from `db._sync_postgres_sequences(conn)` (`:251`).

- [ ] **Step 2: Add the Media guard**

The new flag must keep Media's default: it raises. This guard passes on dev, and fails if the default flips (measured mutant).

In `test_media_db_schema_bootstrap.py`, after `test_sync_postgres_sequences_positive_scalar_uses_max_id_branch`, add the following, with two blank lines before and after:

```python
@pytest.mark.unit
def test_sync_postgres_sequences_raises_a_failing_table() -> None:
    """Media keeps raising a table's error; only ChaCha skips it (TASK-13901)."""
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sequence_maintenance as postgres_sequence_maintenance_module,
    )

    conn = SimpleNamespace(transaction=lambda **_kwargs: nullcontext())
    queries: list[str] = []

    class FakeBackend:
        @staticmethod
        def escape_identifier(value: str) -> str:
            return f'"{value}"'

        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> SimpleNamespace:
            queries.append(query.strip())
            if len(queries) == 1:
                return SimpleNamespace(
                    rows=[
                        {"sequence_schema": "public", "sequence_name": f"{table}_id_seq", "table_name": table, "column_name": "id"}
                        for table in ("keywords", "media")
                    ]
                )
            if query.startswith("SELECT last_value"):
                return SimpleNamespace(rows=[{"last_value": 1, "is_called": False}])
            if query.startswith("SELECT COALESCE(MAX"):
                raise BackendDatabaseError("PostgreSQL query execution failed")
            return SimpleNamespace()

    with pytest.raises(BackendDatabaseError):
        postgres_sequence_maintenance_module.sync_postgres_sequences(SimpleNamespace(backend=FakeBackend()), conn)
    assert not any("media_id_seq" in query for query in queries)
```

`BackendDatabaseError`, `SimpleNamespace`, `nullcontext` and `pytest` are already imported at the top of the file (`:4-13`).

```bash
T "tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py::test_sync_postgres_sequences_raises_a_failing_table"
```

Expected: `1 passed`. Dev already raises.

- [ ] **Step 3: Replace the shared helper with PR #3269's copy**

Write `tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py` with exactly this content (82 lines):

```python
"""Forward-only PostgreSQL serial maintenance for fixed module inventories."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.postgres_sequences import (
    PostgresSequenceBackend,
    advance_postgres_serial,
)


class PostgresSequenceMaintenanceDB(Protocol):
    """Protocol for DB objects exposing a PostgreSQL backend."""

    backend: PostgresSequenceBackend


def sync_owned_postgres_sequences(
    db: PostgresSequenceMaintenanceDB,
    conn: Any,
    owned_columns: Collection[tuple[str, str]],
    *,
    continue_on_error: bool = False,
) -> None:
    """Advance owned serials without reusing allocated IDs hidden by tenant RLS.

    The schema owner calls this inside its existing migration transaction.
    Only a temporary lock savepoint is rolled back; caller work is untouched.
    PostgreSQL's nontransactional setval deliberately retains forward advances.
    ChaCha opts into per-table continuation; Media retains fail-fast maintenance.
    """

    backend = db.backend
    sequence_rows = backend.execute(
        """
        SELECT
            sequence_ns.nspname AS sequence_schema,
            seq.relname AS sequence_name,
            tab.relname AS table_name,
            col.attname AS column_name
        FROM pg_class seq
        JOIN pg_namespace sequence_ns ON sequence_ns.oid = seq.relnamespace
        JOIN pg_depend dep ON dep.objid = seq.oid AND dep.deptype = 'a'
        JOIN pg_class tab ON tab.oid = dep.refobjid
        JOIN pg_namespace tab_ns ON tab_ns.oid = tab.relnamespace
        JOIN pg_attribute col ON col.attrelid = tab.oid AND col.attnum = dep.refobjsubid
        WHERE seq.relkind = 'S' AND tab_ns.nspname = 'public'
        ORDER BY sequence_ns.nspname, seq.relname;
        """,
        connection=conn,
    )

    for row in sequence_rows.rows:
        table_name = row.get("table_name")
        column_name = row.get("column_name")
        sequence_schema = row.get("sequence_schema", "public")
        sequence_name = row.get("sequence_name")

        if not table_name or not column_name or not sequence_name:
            continue
        if (table_name, column_name) not in owned_columns:
            continue

        try:
            advance_postgres_serial(
                backend,
                conn,
                table_name=table_name,
                column_name=column_name,
                sequence_schema=sequence_schema,
                sequence_name=sequence_name,
            )
        except DatabaseError:
            if not continue_on_error:
                raise
            # The primitive rolls back its own savepoint, preserving caller work.
            logger.warning("Could not advance owned PostgreSQL sequence for {}.{}", table_name, column_name)
```

If PR #3269's branch is still fetchable, this writes the same bytes: `git fetch origin codex/uat-wins-through-c04-20261018 && git show 81b0bb58365f22a28ff662497395807cba8ca563:tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py > tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py`. Either way, check:

```bash
git hash-object tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py
```

Expected: `f1695435919059df6b340ab055c8784e6965aadf`.

- [ ] **Step 4: ChaCha opts in**

In `ChaChaNotes_DB.py`, `_sync_postgres_sequences`, replace

```python
        sync_owned_postgres_sequences(self, conn, self._POSTGRES_SEQUENCE_TABLES)
```

with

```python
        sync_owned_postgres_sequences(
            self, conn, self._POSTGRES_SEQUENCE_TABLES, continue_on_error=True
        )
```

- [ ] **Step 5: GREEN, lint and Bandit**

```bash
T tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_monotonicity.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_monotonicity.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_ownership.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
$PY -m ruff check tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
uvx bandit -q -ll tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py \
  tldw_Server_API/app/core/DB_Management/backends/postgres_sequences.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
```

Expected:
- `130 passed`;
- `All checks passed!` (`$PY -m ruff` and the venv's `bin/ruff` are ruff 0.15);
- no Bandit `Issue:` lines (0 on dev as well).

- [ ] **Step 6: Commit**

```bash
git add Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md \
  tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
git add -A backlog/tasks
git commit -m "fix(chacha): sequence sync continues past a failing table again (TASK-13901)

#3253 kept ChaCha's per-table error policy when it shared Media's
serial advance; #3084 replaced ChaCha's loop with one call to a new
helper that had no error handling, so one table's error stopped the
sync. The helper now calls advance_postgres_serial and takes
continue_on_error, as in open PR #3269 (byte-identical file). ChaCha
opts in; Media still raises, now pinned by a unit guard.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: the admission-capture tests keep #3084's coverage under TASK-13569's contract (TASK-13901 cases 2-5)

**Files:**
- Modify: `tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py:1014` (setup), `:1040` (`match` pin), `:1050` (expectation).
- Unchanged, cited: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py:1125-1126` and `:3210-3212`; `tldw_Server_API/tests/DB_Management/test_history_branch_admission.py:296-340`.

**Interfaces:**
- Consumes, unchanged:
  - `CharactersRAGDB.add_message_metadata`. A rewrite bumps `messages.version` by 1; a first write and a no-op do not (TASK-13569).
  - `validate_history_admission`. It refuses a changed input state with `HistorySelectionError("stale_parent")`.
- Produces: nothing new.

- [ ] **Step 1: RED**

```bash
T "tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_changed_input_or_capture" \
  "tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_metadata_change_and_restore"
```

Expected: `4 failed, 14 passed`. The failures:
- `[sqlite-metadata]` and `[postgres-metadata]`: `E   assert 2 == 1`;
- `[sqlite]` and `[postgres]` of the restore test: `E   assert 3 == 1`.

- [ ] **Step 2: Why these edits, and not a product change**

The product line both tests run into is #3229's (`69e594c495`, TASK-13569, merged 2026-10-15). #3084 merged on top of it and did not redefine it:

```python
        if advance_history:
            if metadata_message_id is not None:
                # Undo must not restore an opaque revision within one timestamp tick.
                conn.execute("UPDATE messages SET version = version + 1 WHERE id = ?", (message_id,))
```

- Without this line, TASK-13569's own `test_metadata_undo_cannot_restore_the_original_opaque_revision` fails in all 4 cases (measured: `4 failed, 22 passed` with the line disabled). So the product stays.
- **The parametrized metadata case.** #3084's design wanted "metadata … drift without physical-version changes". That is still reachable with a first metadata write, which #3229 keeps version-neutral. So that case changes its setup, not its expectation.
- **The A→B→A case.** It needs a rewrite, which #3229 bumps on purpose, and there is no metadata delete path. So it is the one expectation that changes, and the hunk above is its justification.

- [ ] **Step 3: Edit three lines**

At `:1014`, in `test_native_admission_capture_refuses_changed_input_or_capture`, replace

```python
    reference = admission_reference(admit(db, cid, selected(db, cid), extra_metadata={"original": True}))
```

with

```python
    reference = admission_reference(admit(db, cid, selected(db, cid)))
```

No other drift case uses the admitted metadata. Leave `:1022`, `assert ... ["version"] == 1  # nosec B101: Metadata must be fenced without a physical version bump.`, as it is.

At `:1040`, in the same test, replace

```python
    with pytest.raises(HistorySelectionError):
```

with

```python
    with pytest.raises(HistorySelectionError, match="stale_parent" if drift == "metadata" else None):
```

At `:1050`, in `test_native_admission_capture_refuses_metadata_change_and_restore`, leave its own `admit(... extra_metadata={"original": True})` at `:1046` as it is, and replace

```python
    assert db.get_message_by_id(reference["input_message_id"])["version"] == 1  # nosec B101: Metadata ABA leaves the physical version unchanged.
```

with

```python
    assert db.get_message_by_id(reference["input_message_id"])["version"] == 3  # nosec B101: Each rewrite of the ABA bumps the physical version (TASK-13569).
```

- [ ] **Step 4: GREEN on both backends**

```bash
T "tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_changed_input_or_capture" \
  "tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_metadata_change_and_restore"
T tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py tldw_Server_API/tests/DB_Management/test_history_branch_admission.py
$PY -m ruff check tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py
```

Expected:
- `18 passed`: 9 SQLite and 9 PostgreSQL cases, `[sqlite-metadata]` and `[postgres-metadata]` among them.
- `200 passed, 1 skipped`. That is about 5 min under load. 98 PostgreSQL-parametrized items pass, and the other 102 pass. The skip is `test_history_sqlite_function_follows_real_connection_lifetime[postgres]`, as on dev.
- `All checks passed!`.

Optional check, measured while planning: with `message_store.py:1125`'s `if state != authority["input_state_digest"]:` made `if False:`, the metadata cases fail with `Expected regex: 'stale_parent' Actual message: 'stale_source'`. Restore the file afterwards.

- [ ] **Step 5: Close TASK-13901 and commit**

```bash
B edit TASK-13901 \
  --append-notes "Case 1 cause (measured on <commit>): #3253 (1f7f780a4d) added advance_postgres_serial and this test and kept ChaCha's per-table try/except that logs and continues ('Preserve each wrapper's existing ownership and error policy'); #3084 (eb0c8c9e52) replaced ChaCha's loop with one call to its new sync_owned_postgres_sequences, an inlined copy with no error handling, and its plan names no error-policy change (it ran the 3 ownership tests that predate #3253). Fix: postgres_sequence_maintenance.py is now byte-identical to open PR #3269's copy (81b0bb5836, blob f169543591): the loop calls advance_postgres_serial and takes continue_on_error=False; ChaCha passes True. RED: 1 failed, DatabaseError at :251. GREEN: the five sequence files 130 passed (129 + test_sync_postgres_sequences_raises_a_failing_table, which pins Media's raise; flipping the default fails it: DID NOT RAISE). uvx bandit -q -ll on postgres_sequence_maintenance.py, postgres_sequences.py and ChaChaNotes_DB.py: 0 results (0 on dev)." \
  --append-notes "Cases 2-5 cause: the product is right; the tests' write path and one expectation were not. #3229 (69e594c495, TASK-13569, merged 2026-10-15) made a metadata rewrite bump messages.version ('Undo must not restore an opaque revision within one timestamp tick', message_store.py:3210-3212); a first write stays version-neutral. Both pinned by its tests. #3084 later added these cases, admitting with extra_metadata so that its drift was a rewrite, and asserting the version stays 1; its design record wanted 'metadata and asset drift without physical-version changes' and 'does not redefine either revision domain'. With the bump disabled these 4 pass and #3229's 4 undo cases fail (4 failed, 22 passed). Fix: the parametrized test admits without extra_metadata (:1014), so its metadata drift is a version-neutral first write and :1022's == 1 holds unchanged; that case now pins match='stale_parent' (:1040), the input-state-digest refusal (disabling that check turns it into stale_source and fails the pin). The A->B->A test (:1050) now expects 3: under #3229 a version-neutral ABA is impossible (no metadata delete path). The design record's claim still holds for the parametrized metadata and asset cases; it no longer covers ABA, which is refused through the version bump. Cases 2-5 failed on SQLite too. RED: 4 failed, 14 passed. GREEN: the two functions 18 passed (9 sqlite, 9 postgres); test_history_selection_transactions.py + test_history_branch_admission.py 200 passed, 1 skipped (the SQLite-only lifetime test on postgres)." \
  --append-notes "No docs change: the dated #3084 design record (Docs/Design/uat_a01_native_retry_validation_20261011.md) is left as written; the note above records where its claim no longer applies. PR #3269: the add/add conflict on postgres_sequence_maintenance.py disappears; test_history_selection_transactions.py now conflicts with it on three one-line hunks (:1014, :1040, :1050; keep dev's side). Plan: Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "The five tests pass. Case 1 was a #3084 regression: its new shared sequence helper had no error handling and replaced ChaCha's loop, which #3253 had kept tolerant of a failing table. The helper now takes continue_on_error (the same file as open PR #3269), ChaCha opts in, and Media still raises, pinned by a new unit guard. Cases 2-5 were tests #3084 wrote against TASK-13569's contract (#3229), which bumps the message version on a metadata rewrite. The metadata-drift case now writes metadata the version-neutral way #3084 intended and pins the stale_parent refusal; only the A->B->A case changes its expectation, to the version 3 that #3229's bump produces." \
  -s Done
git add tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py
git add -A backlog/tasks
git commit -m "test(chat): admission-capture drift tests follow TASK-13569's contract (TASK-13901)

#3084's metadata-drift case admitted with metadata, so its drift was a
rewrite, which TASK-13569 (#3229) bumps on purpose. It now admits
without metadata: the drift is a version-neutral first write, as
#3084 intended, its version-1 check stands, and it pins the
stale_parent refusal. The A->B->A case cannot be version-neutral, so
it now expects the bumped version 3.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

`<commit>` is the dev tip recorded in Setup.

---

### Task 3: the restricted Media store's teardown fails fast (TASK-13900 AC 2)

**Files:**
- Modify: `tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py:27` (import) and `:73-78` (teardown).

**Interfaces:**
- Consumes: `_drop_restricted_role(admin, role: str) -> None` from `tldw_Server_API.tests._plugins.postgres` (`:430-463`; an import with precedent in `Admin/test_bundle_restore_postgres.py:18`).
  - It drops the role under `lock_timeout = 10s`.
  - On a lock wait, it terminates the non-idle sessions on the test's database and retries.
  - It then raises `AssertionError("The test left N session(s) of <role> or its database holding locks (a leaked checkout?): [...]")`.
- Produces: nothing new.

- [ ] **Step 1: RED (recorded; do not re-run)**

On dev this teardown hangs. Measured while planning: the `DROP OWNED BY` was still waiting at 4:40, past `-o timeout=150`, and it left its role behind (What was measured). Do not run the file on a tree without Step 2.

- [ ] **Step 2: Use the shared fail-fast drop**

After `from tldw_Server_API.app.main import app as fastapi_app` (`:27`), add:

```python
from tldw_Server_API.tests._plugins.postgres import _drop_restricted_role
```

In `restricted_media_store`'s `finally`, replace

```python
        if role is not None:
            with admin.transaction() as conn:
                conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        if admin is not None:
            admin.get_pool().close_all()
```

with

```python
        try:
            if role is not None:
                _drop_restricted_role(admin, role)
        finally:
            if admin is not None:
                admin.get_pool().close_all()
```

`sql` is still used by the setup's `CREATE ROLE` and `GRANT`s.

- [ ] **Step 3: Run the failing node; it now fails fast**

```bash
T "tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py::test_permission_first_auth_trash_restore_without_preseeded_scope[postgresql-X-API-KEY]"
```

Expected, in about 50 s: `1 failed, 1 error`.
- `FAILED ... - AssertionError: assert 500 == 200`, on `client.get(f"/media/{media_id}", ...)`.
- `ERROR ... - AssertionError: The test left 1 session(s) of rag_retrieval_… or its database holding locks (a leaked checkout?): [{… 'state': 'idle in transaction', 'query': 'SELECT * FROM MediaFiles WHERE media_id = $1 AND file_type = $2 AND deleted = FALSE ORDER BY id DESC LIMIT 1'}]`.

Then confirm that no `rag_retrieval_*` role is left: run `SELECT rolname FROM pg_roles WHERE rolname LIKE 'rag_retrieval_%'` through the fixture's admin DSN. Nothing from this run should be listed.

- [ ] **Step 4: Commit**

```bash
$PY -m ruff check tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py
git add tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py
git commit -m "test(rag): restricted Media store teardown fails fast on a leaked session (TASK-13900)

The fixture's bare DROP OWNED BY waited forever behind a Media DB
session left idle in a transaction, past pytest-timeout, and left its
role in the shared cluster. It now uses TASK-13861's
_drop_restricted_role: a lock timeout, then a failure that names the
session.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: the MediaFiles lookup no longer holds the request connection (TASK-13900 AC 1)

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/repositories/media_files_repository.py:76-96` (`get_for_media`).
- Modify: `tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py:132` (monkeypatch target). Two tests are added at the end of the file.
- Test: `tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py` (Task 3 Step 3's failure).

**Interfaces:**
- Consumes: `MediaDatabase.execute_query(query, params) -> BackendCursorAdapter`. Inside `db.transaction()` it uses the transaction's connection; outside one, its own pooled checkout.
- Consumes, in a test: `pg_database_config` (`tests/_plugins/postgres.py:298`) and `DatabaseBackendFactory.create_backend(config).execute(sql).scalar`.
- Produces: `MediaFilesRepository.get_for_media(media_id, file_type="original", *, include_deleted=False) -> dict[str, Any] | None`. Its signature, result and errors are unchanged.

- [ ] **Step 1: Write the contract tests (RED in 3 s, no hang)**

At the end of `test_media_files_backend_contract.py`, add:

```python
def test_lookup_leaves_no_postgres_session_in_a_transaction(pg_database_config):
    """A lookup outside a transaction must not pin a connection idle in one (TASK-13900)."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(":memory:", client_id="1", backend=backend)
    try:
        media_id = add_media(db)
        repo = MediaFilesRepository(db)
        repo.insert(media_id, "original", "original.txt")
        db.close_connection()
        assert repo.get_for_media(media_id)["storage_path"] == "original.txt"
        idle = backend.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND state = 'idle in transaction'"
        ).scalar
        assert idle == 0
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_lookup_inside_a_transaction_reads_its_uncommitted_row(media_db):
    media_id = add_media(media_db)
    repo = MediaFilesRepository(media_db)
    with pytest.raises(RuntimeError, match="rollback requested"):
        with media_db.transaction():
            pending_uuid = repo.insert(media_id, "original", "pending.txt")
            assert repo.get_for_media(media_id)["uuid"] == pending_uuid
            raise RuntimeError("rollback requested")
    assert repo.get_for_media(media_id) is None
```

```bash
T "tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py::test_lookup_leaves_no_postgres_session_in_a_transaction" \
  "tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py::test_lookup_inside_a_transaction_reads_its_uncommitted_row"
```

Expected: `1 failed, 2 passed`.
- The first test fails with `E   assert 1 == 0`: dev's lookup leaves one session idle in a transaction.
- The second is a guard for Review Focus 3. It passes on SQLite and PostgreSQL, before and after the fix.

Task 3 Step 3's `1 failed, 1 error` is the same defect seen through the route.

- [ ] **Step 2: Read through `execute_query`**

In `get_for_media`, replace

```python
        db = self.session
        conn = db.get_connection()
        clauses: list[str] = ["media_id = ?", "file_type = ?"]
```

with

```python
        db = self.session
        clauses: list[str] = ["media_id = ?", "file_type = ?"]
```

and replace

```python
            rows = db._fetchall_with_connection(conn, sql, params)
            return rows[0] if rows else None
```

with

```python
            # execute_query owns its connection; a held request connection would idle in a transaction.
            rows = db.execute_query(sql, params).fetchall()
            return dict(rows[0]) if rows else None
```

- [ ] **Step 3: GREEN on the RAG file**

```bash
T "tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py::test_permission_first_auth_trash_restore_without_preseeded_scope"
T tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py
```

Expected: `4 passed` (about 15 s), then `68 passed` (about 40 s).

- [ ] **Step 4: Move the contract test's monkeypatch to the call the fix uses**

```bash
T tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py
```

Expected: `2 failed, 11 passed`. The two failures are `test_database_query_failure_is_not_reported_as_missing_file[sqlite]` and `[postgresql]`, with `Failed: DID NOT RAISE DatabaseError`.

The test patches `_fetchall_with_connection`, which `get_for_media` no longer calls. Its expectation stays: a query failure raises `DatabaseError` matching "controlled query failure", rather than returning `None`. Only the injection point follows the fix, and `list_for_media` (TASK-13233) already reads through `execute_query`.

At `:132`, replace

```python
    monkeypatch.setattr(media_db, "_fetchall_with_connection", fail_query)
```

with

```python
    monkeypatch.setattr(media_db, "execute_query", fail_query)
```

```bash
T tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py
```

Expected: `13 passed`.

- [ ] **Step 5: Lint and Bandit**

```bash
$PY -m ruff check tldw_Server_API/app/core/DB_Management/media_db/repositories/media_files_repository.py tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py
uvx bandit -q -ll tldw_Server_API/app/core/DB_Management/media_db/repositories/media_files_repository.py
```

Expected: `All checks passed!`, and no Bandit `Issue:` lines.

- [ ] **Step 6: Every caller's tests**

```bash
git grep -l "get_media_file\|MediaFilesRepository\|has_original_file\|get_for_media\|get_full_media_details_rich\|_media_db_legacy_stub" -- tldw_Server_API/tests
T tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py \
  tldw_Server_API/tests/DB_Management/test_db_manager_wrappers.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_maintenance_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_reads_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_db_legacy_transcript_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_db_media_update_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py \
  tldw_Server_API/tests/Files/test_files_image_endpoint.py \
  tldw_Server_API/tests/Image_Generation/test_reference_images.py \
  tldw_Server_API/tests/Media/test_document_figures_endpoint_sanitization.py \
  tldw_Server_API/tests/Media/test_document_outline.py \
  tldw_Server_API/tests/Media/test_document_outline_endpoint_sanitization.py \
  tldw_Server_API/tests/Media/test_media_file_endpoint_sanitization.py \
  tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py \
  tldw_Server_API/tests/Media/test_media_item_error_contracts.py \
  tldw_Server_API/tests/Media/test_media_navigation.py \
  tldw_Server_API/tests/Media/test_media_navigation_db_error_contracts.py \
  tldw_Server_API/tests/MediaDB2/test_media_files.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_original_file_replacement.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py
```

- The grep lists 24 paths: these 23 test files plus `_media_db_legacy_stub.py`, a helper that is not a test.
- `test_media_versions.py` is added by hand: it reaches `get_for_media` through `media/versions.py`'s `get_full_media_details_rich` call.
- Add any new file the grep lists at execution.

Expected: `563 passed`.

For any other failure, re-run that test with `origin/dev`'s `media_files_repository.py` before calling it unrelated:
- `cp <path> <scratch>/media_files_repository.branch.py`;
- `git show origin/dev:<path> > <path>`, then run the test;
- `cp <scratch>/media_files_repository.branch.py <path>`, and confirm the restore with `git diff --stat`.

- [ ] **Step 7: File the follow-ups**

Take the id lock: `mkdir <scratchpad>/id_lock`; if it already exists, wait. Then compute the highest id across `origin/dev` and every open PR. This is AGENTS.md's command, extended to page through PRs with more than 100 files:

```bash
git fetch origin dev --quiet
{ git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1
```

Use the next id for `<N1>`, then `<N2>`, then `<N3>`. Release the lock (`rmdir <scratchpad>/id_lock`) only once the PR carrying the files is open and `gh pr view` lists them.

```bash
B create "Media DB calls that use the request connection outside a transaction pin a PostgreSQL pool slot idle in a transaction" --id "TASK-<N1>" -l media,postgres,backend,bug \
  -d "Found by TASK-13900. On PostgreSQL, MediaDatabase.get_connection() checks a connection out and pins it in the request's context var (media_db/runtime/connection_lifecycle.py:67-85); a statement on it opens a transaction nothing ends, so the session sits idle in transaction until release_context_connection() runs, and every later execute_query in the same request takes a second pool slot. TASK-13900 measured it on MediaFilesRepository.get_for_media: with a 1-slot pool the next statement waited pool_timeout (30 s) and GET /media/{id} answered 500; with a dependency that never releases, the session held AccessShareLock on MediaFiles for good and hung DROP OWNED BY. get_for_media and list_for_media now read through execute_query, pinned by test_lookup_leaves_no_postgres_session_in_a_transaction. The same pattern remains in keyword_access_ops.py:86 (fetch_media_for_keywords), sync_log_ops.py:35 (get_sync_log_entries), chunk_batch_ops.py:181 (process_chunks' parent check), visual_document_ops.py:114 (list_visual_documents_for_media), and the writes visual_document_ops.py:59 (insert_visual_document) and :142 (soft_delete_visual_documents_for_media) (measured on <commit>). backup_ops.py:37 and api.py describe_media_schema are SQLite-only." \
  --ac "Each listed read goes through execute_query or an explicit transaction, so it leaves no PostgreSQL session idle in a transaction, with a PostgreSQL test like test_lookup_leaves_no_postgres_session_in_a_transaction" \
  --ac "The two visual-document writes run in db.transaction() (or are shown to commit), with a test"
B create "PostgreSQL test fixtures hand-roll a restricted-role teardown that waits forever on a leaked session" --id "TASK-<N2>" -l tests,postgres \
  -d "34 test files (git grep -l 'DROP OWNED BY' -- tldw_Server_API/tests, minus tests/_plugins/postgres.py and the RAG file TASK-13900 fixed; measured on <commit>) create a restricted role and drop it with a bare DROP OWNED BY. A session the test leaks in a transaction makes that DROP wait for its lock forever: TASK-13900's RAG fixture waited past pytest-timeout (150 s) for 4:40 and more, and left its role in the shared cluster. pg_restricted_backend and _drop_restricted_role (tests/_plugins/postgres.py, TASK-13861) fail fast and name the session. plan_fu2_claims.md adds another hand-rolled fixture with only a lock_timeout. Move these fixtures to pg_restricted_backend or _drop_restricted_role (consider making it public)." \
  --ac "No test file drops a restricted role with a bare DROP OWNED BY: each uses pg_restricted_backend or the shared fail-fast drop, with a guard test that greps for the pattern"
B create "pytest-timeout does not end a run blocked in a PostgreSQL lock wait" --id "TASK-<N3>" -l tests,postgres,ci \
  -d "With timeout_method = thread (pyproject.toml:669) and -o timeout=150, a test whose teardown waits on a PostgreSQL lock (DROP OWNED BY behind a session idle in transaction) was not ended: measured 4:40 while planning TASK-13900 (origin/dev <commit>), and 5:13, 5:37 and 13 min in TASK-13861's runs; another session's sweep was stuck 12+ min the same day. Not reproduced by the plan's reviewer (the unbounded run is forbidden). A hung CI shard waits for its job timeout instead. Find why the timer does not fire or cannot exit (psycopg's wait holding the GIL, the stack dump blocking, or another plugin), and make a hang end within the timeout." \
  --ac "A test blocked in a PostgreSQL lock wait is ended by pytest-timeout within its timeout, shown by a test or a recorded reproduction"
```

- [ ] **Step 8: Close TASK-13900 and commit**

```bash
B edit TASK-13900 \
  --append-notes "Cause (measured on <commit>): the failing node, [postgresql-X-API-KEY], fails alone too; it is the first PostgreSQL case of authenticated_media_client. Its first request, GET /media/{id}, reaches MediaFilesRepository.get_for_media through has_original_file. get_for_media used db.get_connection(), which on PostgreSQL pins a pooled connection in the request's context var; its SELECT left the session idle in transaction holding AccessShareLock on MediaFiles. The request's next statements (_load_keywords, _load_versions) needed a second slot on the fixture's 1-slot pool, waited pool_timeout (30 s) and the GET answered 500 (assert 500 == 200). The test's media_db override never calls release_context_connection, so the session stayed in its transaction, and the fixture's bare DROP OWNED BY waited on its lock past pytest-timeout (killed at 4:40, role left behind and dropped by name). Production's get_media_db_for_user releases at request end, but each detail GET held two of the pool's 10 slots, and any non-releasing path keeps the session." \
  --append-notes "Fix: get_for_media reads through db.execute_query, as list_for_media has since TASK-13233 (76d60dbbd6); restricted_media_store's teardown uses _drop_restricted_role (TASK-13861). RED: test_lookup_leaves_no_postgres_session_in_a_transaction assert 1 == 0 on dev (3 s; the in-transaction read-your-writes guard passes before and after); with the teardown only, the RAG node gave 1 failed, 1 error in 50 s (GET 500; teardown named the idle-in-transaction MediaFiles session; no role left). GREEN: the 4 trash_restore cases 4 passed; the whole RAG file 68 passed (nothing after the 19th item had run on dev). test_database_query_failure_is_not_reported_as_missing_file patched _fetchall_with_connection, which get_for_media no longer calls (2 failed, DID NOT RAISE); its patch moved to execute_query, expectation unchanged; the contract file 13 passed. The 24 test files that reach get_for_media (the six-name grep plus test_media_versions.py): 563 passed." \
  --append-notes "Verification: ruff All checks passed; uvx bandit -q -ll on media_files_repository.py: 0 results (0 on dev). No docs change. Follow-ups: TASK-<N1> (six more Media DB calls pin the request connection on PostgreSQL), TASK-<N2> (34 test files hand-roll a bare DROP OWNED BY teardown), TASK-<N3> (pytest-timeout does not end a PostgreSQL lock wait). Plan: Docs/superpowers/plans/2026-10-22-pg-dev-test-failures.md." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Fixed. The hang came from a product leak: MediaFilesRepository.get_for_media read on the request's pinned PostgreSQL connection and left it idle in a transaction, so GET /media/{id} stalled for a second pool slot (500 on the fixture's one-slot pool) and the session's lock blocked the fixture's DROP OWNED BY for good. get_for_media now reads through execute_query, as list_for_media does, pinned by a contract test; and the fixture's teardown uses TASK-13861's fail-fast drop, so a future leak fails the test with the session named and leaves no role behind. The whole file passes on PostgreSQL. TASK-<N1>, TASK-<N2> and TASK-<N3> track the same connection pattern elsewhere, the other hand-rolled teardowns, and pytest-timeout not firing." \
  -s Done
git add tldw_Server_API/app/core/DB_Management/media_db/repositories/media_files_repository.py \
  tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py
git add -A backlog/tasks
git commit -m "fix(media): MediaFiles lookup no longer holds the request connection (TASK-13900)

get_for_media read on the request's pinned PostgreSQL connection and
left it idle in a transaction. The rest of the request then needed a
second pool slot, and a caller that never releases kept the session,
and its MediaFiles lock, for good: GET /media/{id} answered 500 on a
one-slot pool and the RAG fixture's teardown hung. It now reads
through execute_query, as list_for_media has since TASK-13233, pinned
by a contract test. Files TASK-<N1>, TASK-<N2> and TASK-<N3>.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: verification, review and PR

- [ ] **Step 1: The focused set, together**

```bash
T tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_ownership.py \
  "tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_changed_input_or_capture" \
  "tldw_Server_API/tests/DB_Management/test_history_selection_transactions.py::test_native_admission_capture_refuses_metadata_change_and_restore" \
  tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py \
  tldw_Server_API/tests/DB_Management/test_media_files_backend_contract.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_chacha_postgres_sequence_monotonicity.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_monotonicity.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_sequence_ownership.py
```

Expected: `229 passed`. Dev gives `5 failed, 152 passed` without the RAG file, which hangs there.

- [ ] **Step 2: The gates**

```bash
$PY Helper_Scripts/ci/test_quality_triage.py --enforce
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
  $PY -m pytest -q --disable-warnings -p pytest_asyncio.plugin -p no:cacheprovider \
  tldw_Server_API/tests/CI/test_rls_coverage_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py \
  tldw_Server_API/tests/lint/test_scope_predicate_ratchet.py tldw_Server_API/tests/lint/test_private_coercion_ratchet.py
BACKLOG_TASK_FORMAT_BASE=$(git merge-base origin/dev HEAD) $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py
B normalize --check backlog/tasks/task-13900\ -*.md backlog/tasks/task-13901\ -*.md backlog/tasks/task-<N1>\ -*.md backlog/tasks/task-<N2>\ -*.md backlog/tasks/task-<N3>\ -*.md
uvx bandit -q -ll tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py \
  tldw_Server_API/app/core/DB_Management/backends/postgres_sequences.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/media_db/repositories/media_files_repository.py
git diff origin/dev --check
git hash-object tldw_Server_API/app/core/DB_Management/backends/postgres_sequence_maintenance.py
```

`normalize --check` takes task files: given the directory, it crashes with `IsADirectoryError`. With no argument it scans every task, and 14 unrelated tasks already fail on dev.

Expected:
- the triage prints `[test-triage] OK — no offenses beyond the baseline.`;
- the ratchets give `37 passed`;
- the task-format ratchet passes;
- `normalize --check` exits 0 (measured rc 0 on the two existing files);
- Bandit prints no `Issue:` lines;
- `git diff --check` prints nothing;
- the hash is `f1695435919059df6b340ab055c8784e6965aadf`.

Then list the `tldw_rls_*` / `rag_retrieval_*` roles and the `tldw_test_*` databases. None should come from your runs.

- [ ] **Step 3: Clean review before the PR merges**

Run `/superpowers:requesting-code-review` in a clean subagent. Point it at:
- this plan's Review Focus and "What was measured";
- `<scratchpad>/review_plan_fu2_pg_tests.md`;
- `<scratchpad>/clean3300_review.md` M3;
- `<scratchpad>/impl_chacha_prD.md` Ruling 4.

Ask it specifically to judge Task 2's setup change and its one changed expectation against #3229's quoted hunk. Fix every Critical and Important finding test-first. Record the outcome in the PR body, and hold the merge until the findings are addressed.

- [ ] **Step 4: Push and open the PR against `dev`**

```bash
git push -u origin fix/pg-dev-test-failures-13900-13901
gh pr create --base dev --title "fix: ChaCha sequence sync, admission-capture drift tests and the RAG Media DB teardown hang (TASK-13900, TASK-13901)" --body-file <scratchpad>/pr_body_fu2_pg_tests.md
```

The PR body has these sections:
- **Summary:** one bullet per fix, saying what changes for a user or a test run.
- **How two #3084 tests change:**
  - the parametrized metadata case changes one setup line (no `extra_metadata`), keeps `== 1`, and pins `stale_parent`;
  - the A→B→A case changes one expectation, to `== 3`.
  - Quote #3229's hunk and the mutual-exclusion probe (`4 failed, 22 passed`).
  - Say that the design record's "without physical-version changes" still covers the parametrized metadata and asset cases, but not A→B→A.
- **`## Change summary`:** the owner's own words (ADR-004). Leave the placeholder `Owner to write` and ask the owner. Do not write it for them.
- **Test plan:** the RED and GREEN counts per task, the mutants, Task 5's run, and the gates.
- **Review focus:** this plan's five items.
- **PR #3269:**
  - Task 1's helper file is byte-identical to #3269's, which removes #3269's add/add conflict.
  - Its remaining conflicts here are one blank line in `ChaChaNotes_DB.py` and three one-line hunks in the history test.
  - It conflicts in 164 paths either way.
- **Follow-ups filed:** TASK-<N1>, TASK-<N2> and TASK-<N3>, one line each.
- **Clean review:** the outcome.
- The last line is `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Check the merge-queue mode with `gh variable get MERGE_QUEUE`, and follow CLAUDE.md's "Merging into dev". Confirm `MERGED` before deleting the branch.

## Follow-ups (filed at execution)

| Id | Title | Filed in | Why |
|---|---|---|---|
| TASK-<N1> | Media DB calls that use the request connection outside a transaction pin a PostgreSQL pool slot idle in a transaction | Task 4 Step 7 | Four reads and two writes still do what `get_for_media` did (Review Focus 4). |
| TASK-<N2> | PostgreSQL test fixtures hand-roll a restricted-role teardown that waits forever on a leaked session | Task 4 Step 7 | 34 test files use a bare `DROP OWNED BY`; one leak hangs a run and leaves a role behind. |
| TASK-<N3> | pytest-timeout does not end a run blocked in a PostgreSQL lock wait | Task 4 Step 7 | Measured 4:40 here, and 5-13 min in TASK-13861; a CI shard would wait for its job timeout. |

## Out of scope

- **Undoing TASK-13569's metadata version bump.** It is deliberate and tested, and #3084's design did not ask for it.
- **The RAG test's `media_db` override not calling `release_context_connection()`.** With Task 4, nothing on this file's paths pins the connection, and the fail-fast teardown reports any future leak. Mirroring production's release there would hide one.
- **Rewriting the dated #3084 design record** (`Docs/Design/uat_a01_native_retry_validation_20261011.md:20`). TASK-13901's notes record that its "metadata … drift without physical-version changes" no longer covers A→B→A.
- **Merging or rebasing PR #3269.** This plan removes one of its conflicts, the helper's add/add, and adds one: the three history-test lines.
