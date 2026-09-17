# UAT200 / TASK13260.138 — optional persona-memory read on PostgreSQL

## Change and cause

The real completion helper `_inject_character_memory_from_db` calls `list_persona_memory_entries` before WorldBookService initialization. The shared memory filter compared the PostgreSQL BOOLEAN `archived` column to integer `0`. The helper intentionally catches optional-memory failures, but the unowned failing SELECT left the connection in INERROR, so the next real dependency failed too.

Exactly two statements change in `persona_state_store.py`: the archive predicate uses FALSE for PostgreSQL (SQLite retains 0); the list SELECT opts into existing `execute_query(..., read_only=True)` transaction ownership. That mechanism settles only a read that starts with an idle connection and no caller transaction. It does not commit or roll back caller-owned writes. All owner/persona/scope/session/type/archive/deleted filtering, ordering, pagination, row conversion and optional-error handling remain intact. No SQL translator, schema, backend or endpoint changes.

The separate UAT203 count-row fix shares the final file; `owned.patch` here isolates only UAT200 and its new test. UAT203's packet has the one additional statement. The snapshot/manifest records the final combined bytes.

## Permanent causal evidence

`test_persona_memory_filter_backends.py` has 26 real database cases, including actual optional injection → WorldBookService initialization, empty/populated memory, all four archived/deleted combinations, owner exclusion with and without a persona filter, exact scope/session/type and pagination, shared-filter backfill, and successful caller-owned explicit/implicit commit and rollback. A real PostgreSQL SELECT 1/0 on the same backend connection proves standalone optional-read cleanup. A separate error-in-existing-caller-transaction control explicitly retains INERROR until the caller rolls back.

- First test attempt: 9 failures / 10 passes. One SQLite fixture used an autocommitted UPDATE without an explicit transaction, so its rollback expectation was invalid. The receipt and first test snapshot are retained; the fixture was corrected to use the actual caller transaction boundary.
- Corrected causal run before production: primary 22 cases produced 10 PostgreSQL failures / 12 passes. The same run contained a separately labeled private count probe (1 PostgreSQL failure / 1 SQLite pass), total 11 failures / 13 passes / 0 skips.
- Additional owner/backfill controls replayed against exact pre-UAT200 store bytes: 2 PostgreSQL failures / 2 SQLite passes. The private loader imports the original bytes without replacing product files.
- Final combined run: **103 passed, 0 skipped, 2 explicitly deselected, 63.84 seconds**. This includes all 26 UAT200 and 12 UAT203 cases, existing WorldBook backend controls, persona store/persistence controls, and character-memory unit controls.

The two deselections are existing migration tests `test_migration_v25_to_latest_creates_persona_tables` and `test_migration_v36_to_latest_adds_voice_command_persona_columns`. The first broader run retained 87 passes and those 2 failures (Notes attachment v59 registry collision). Exact pre-UAT200 loader replay also fails both. Parent reopened UAT183 / TASK13260.120 for that fixture repair; neither is represented as passing here.

## Verification

The complete command is in `final-green-command.json`; activate `.venv` before invoking its official required-PG runner. It uses official per-test PostgreSQL databases and SQLite files, requires PostgreSQL availability, and does not create ad hoc databases or touch native app data. Independent focused command:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat200-203-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_persona_memory_filter_backends.py tldw_Server_API/tests/DB_Management/test_persona_memory_count_backends.py tldw_Server_API/tests/DB_Management/test_world_book_initialization_backends.py -q --tb=short
```

Ruff: exactly 2 existing production findings (I001, C420), identical at the same logical source path; 0 new-test findings. Bandit: production 0 findings / 0 parse errors; both new tests 0 findings / 0 parse errors with test assertion rule B101 excluded. Three changed Python files parse; scoped diff whitespace check passes. `static-summary.json` and underlying JSON receipts retain the details.

Production and both tests are frozen. `owned-manifest.json` SHA256: `8386008719828bf077cd76524b61161a338d63d4f02bb6768b912e3ac59b500f`.
Combined store SHA256: `52652fa347b2d945abcdf01674a16f2f76d997875bc6a56995457656dee9cfea`.

## Limits

No browser, model/provider call, application runtime restart, native database, task/tracker, staging or commit mutation was performed. This is the actual memory-read/next-dependency boundary, not a claim that all completion stages or native Bob Retry now succeed. Optional driver errors inside an already-owned transaction still require that caller to recover its transaction; broad rollback would violate ownership. Other persona read methods are unchanged. Native acceptance and independent review remain parent-owned.
