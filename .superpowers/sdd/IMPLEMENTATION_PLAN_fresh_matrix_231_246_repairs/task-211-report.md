# TASK13260.211 / UAT270 legacy World Book test-double repair

## Cause

Root’s read-only baseline source-selection run proved the same ten legacy failures on committed UAT260/265/266 source (`0d7f2a23c9`, module SHA-256 `1ffb3839115e34daa162abae2f9b6648741597814744798d435c70960010c870`). The retained baseline log is `.tmp/uat-repairs-231-246/worldbook-lifecycle210/root-legacy-baseline.log` with SHA-256 `fb8fc846caa0e3ad0456966c35cc02561c49565431b39251d7d3394173d9e587`.

The legacy fixture represented only the old context-managed connection path. Current World Book reads use `execute_query`, and cache eligibility inspects the connection’s `in_transaction` state. It also expected attachment upsert before the current required foreign-record validation. These were stale test doubles and assertions, not product faults.

## Test-only change

`test_world_book_manager_legacy.py` now supplies:

- an SQLite backend identity and non-transactional connection state for the test double;
- the same cursor through `execute_query` for portable read-path assertions;
- both successful reference rows before attachment upsert; and
- assertions that preserve the validation-plus-upsert contract instead of asserting the obsolete one-call behavior.

No production, runtime, browser, database, Git, tracker, or configuration file changed. UAT210 source and its new lifecycle test remain frozen.

## Evidence

```sh
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py -q --tb=short
# 39 passed, 4 warnings
```

The original causal red was 29 passed / 10 failed on both current and read-only baseline source. An intermediate stale-double update reduced that to 3 failures, then the SQLite cache-state and attachment cursor bindings brought the suite to 39 passing.

`compileall` passes. Bandit reports 123 low-severity test-only assertions (`B101`); it has no production scope. Ruff reports 13 existing legacy diagnostics (unused imports, import ordering, and legacy boolean assertion style); this test-only repair neither broadens nor suppresses them.

Final test-file SHA-256:

```text
765300eec44a17e80b97ddd46d9e4e9040d987717b6243c76a76af15a3d82caf  tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py
```

Independent root review remains required before commit.
