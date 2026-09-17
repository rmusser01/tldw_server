# UAT252 — SQLite MCP missing-schema test fixture

**TASK13260.194. Test-only repair, frozen for independent review.**

## Change

The original `test_repo_ensure_tables_requires_governance_pack_distribution_tables` attempted to remove its fixture tables through the managed AuthNZ pool. The production profile write guard correctly rejected DROP TABLE before the intended missing-table check.

Only that function in `tldw_Server_API/tests/AuthNZ_Unit/test_mcp_hub_repo.py` changed. The same two DROP statements now run through a dedicated `sqlite3.connect(db_path)` schema connection, explicitly committed and closed with `contextlib.closing`. This matches existing AuthNZ schema-corruption fixtures. The managed pool/repository remains the system under test; no guard is disabled or changed.

The exact original `pytest.raises(RuntimeError, match="mcp_governance_pack_source_candidates")` context and its `await repo.ensure_tables()` body are unchanged. `assertion-and-scope.json` verifies the complete module AST outside this one function and every existing assertion are unchanged.

## Evidence

| Receipt | Result |
| --- | --- |
| `uat252-fixture-red` | 1 failed, 0 skipped, 4 warnings, 8.83s: `ProfileUserWriteRejected` at the original setup DROP. |
| `uat252-fixture-green` | 1 passed, 0 skipped, 4 warnings, 8.09s: unchanged missing-table assertion is reached. |
| `uat252-combined-green` | 139 passed, 0 skipped, 280 warnings, 66.83s. |

The combined run includes 61 existing MCP repository/setup service/catalog cases, UAT251's 44 real backend cases (22 PostgreSQL, 22 SQLite), and 34 existing protected-write negative controls. All run through the approved explicit-Jobs required-PG runner. Official fixtures own disposable PostgreSQL provisioning; the repaired missing-schema fixture itself is **SQLite**, not PostgreSQL.

Scoped static verification: Python compile and diff whitespace check pass. Ruff has the same four preexisting findings before/after (I001, B017, two F811 duplicate-definition findings); the six-line offset in F811 messages is normalized in the comparison. Bandit has the same five preexisting B106 findings on the unrelated `secret_kind="bearer_token"` enum argument, zero added/removed findings and zero parse errors. B101 is excluded for test assertions only. No finding is in the changed function. No unrelated formatting or fixture cleanup was performed.

The two UAT251 source/test hashes remain unchanged; this packet records them as unowned dependencies. UAT251's own earlier 60-pass/1-failure adjacent receipt remains valid historical evidence and was not overwritten. Its baseline replay had already reproduced the same fixture failure using pre251 repository bytes.

## Scope and reproduction

`commands.json` and `receipts/*-command.json` contain exact commands. Activate `.venv`, set a unique `TLDW_UAT_EVIDENCE_LABEL`, and invoke `.tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs` with the recorded arguments. No skipped or disabled test was used.

No production, runtime, native database/profile/archive, browser, task/tracker or Git mutations. This repair restores the intended malformed-schema regression; it is not native MCP workflow acceptance. Parent owns review/integration and UAT251 native acceptance.
