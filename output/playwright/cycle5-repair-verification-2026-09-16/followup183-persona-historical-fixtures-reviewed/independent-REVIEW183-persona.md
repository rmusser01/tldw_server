# Independent UAT183 review — historical Persona fixtures

Task: TASK13260.120. Verdict: **clear; no source changes requested**.

## Scope and source identity

Reviewed only the historical fixture changes in `tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py`, SHA256 `4e0b8f236bdfefc34608410d8af4ccc04eecd2f1692349324de701ff3191ce73`. The frozen author manifest matches. Exact baseline and current AST comparisons were independently repeated. All13 original v25 assertions and all4 original v36 assertions remain. All unrelated functions are AST-identical and the ordinary CRUD-test remainder is byte-identical.

The independent run also captured unchanged dependencies: ChaCha `97f1a6ef791ca8f5fa4ebb5bbeba2c5c82a229d0d05ed70345fdc6064699dc5a` (PostgreSQL head69, SQLite head67), PersonaStateStore `52652fa347b2d945abcdf01674a16f2f76d997875bc6a56995457656dee9cfea`, and WorldBook test `0b37e00942c60e47e9020ce6ff256606d9ef1560b6cc0cff29fe7f144f981efa`. These shared production bytes are dependencies, not changes attributed to UAT183. Before/after hashes and exact snapshots are retained.

## Correctness review

The seed helper constructs the real V4 base and invokes registered migration methods in order through25 or36, checking every version increment. Its temporary initializer and version cap are restored on leaving the monkeypatch context, before current-head reopen. It does not initialize the latest schema and relabel it as historical, suppress collision guards, or alter production code.

The v25 control proves persona tables and the later Notes attachment registry are absent, seeds a valid historical character, invokes the exact25→26 migration, and checks the new Persona tables/version26. Historical Persona and memory records then survive ordinary current-head initialization, with every original stored field compared afterward.

The v36 control proves the later Notes registry and voice linkage columns are absent, seeds historical Persona/voice records, invokes the exact36→37 migration and checks version37/data preservation. It then reopens normally and checks all prior head/voice/index assertions, complete historical field preservation, and null defaults on new linkage columns. Source inspection confirms36→37 is workspace/Flashcard migration; voice linkage is supplied by the existing recent-voice compatibility routine. The author's report makes that distinction accurately.

Both latest-head assertions use SQLite `_CURRENT_SCHEMA_VERSION`, correctly independent of concurrent PostgreSQL-only head69. No PG migration guard or historical exact transition was weakened. The retained original RED is attributed to constructing an inconsistent fake historical schema that correctly hit the later Notes registry collision guard.

## Independent verification

Command (official required-PG runner; no exclusions):

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat183-persona-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py -q --tb=short
```

Result: **12 passed, 0 skipped, 4 warnings, 10.38s**. The historical migration cases are SQLite; the adjacent WorldBook tests exercise actual official PostgreSQL and SQLite fixtures. No native databases, runtime services or browser were used.

Fresh Ruff:0 findings. Fresh Bandit:0 findings/0 errors, excluding only B101 for test assertions. Python AST parses and scoped diff whitespace check passes. An extra `ruff format --check` returns1 because of three unchanged ordinary-test formatting hunks; baseline replay contains these same hunks plus one removed old fixture hunk. There are no new formatter differences in the changed fixture/helper scope. Both formatting diffs are retained; no unrelated formatting rewrite was requested.

## Limits

This is a test-only correction and focused verification, not certification of every historical database, all PG69 tenant migrations, or native UAT completion. Historical schemas are synthetic products of repository migration steps, not recovered user databases. Author's broader105-case run is recorded in their packet; it was not repeated here. No production/test edits, task/tracker, git, runtime, browser, credentials, or live database actions were performed by this reviewer.
