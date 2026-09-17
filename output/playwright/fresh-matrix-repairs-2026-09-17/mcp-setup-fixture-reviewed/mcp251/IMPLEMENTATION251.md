# UAT251 — PostgreSQL MCP setup Save packs

**Task:** TASK13260.193. **Source/test frozen for independent review.** Native acceptance remains pending with the parent.

## Problem and change

The native Save packs request failed because PostgreSQL cannot infer the type of an independently bound parameter used only in `IS NULL`. Giving its separately bound equality partner a column type does not type that parameter. Real setup-chain controls found the same failure in profile/assignment create readback queries. Assignment get/list had a second proven error: `COALESCE(boolean, 0)` mixes incompatible PostgreSQL types.

The repair changes fourteen SQL lines in five `McpHubRepo` methods. Nullable operands now use portable `CAST(? AS TEXT)` or `CAST(? AS INTEGER)` matching the existing column types. Assignment projections use Boolean `FALSE`. Lists still treat omitted/None filters as unfiltered, while create readback retains exact NULL matching. Parameterization, ordering, IDs, scope predicates and transaction ownership remain unchanged.

Owned production: `tldw_Server_API/app/core/AuthNZ/repos/mcp_hub_repo.py`.

Owned test: `tldw_Server_API/tests/DB_Management/test_mcp_setup_scope_filters_backends.py`.

The pre-change source snapshot exactly matches frozen native commit `86458ab88ce3fa62e6518c9d813c3860254ddb2c`. `ast-scope.json` verifies only the five declared methods changed, with no added or removed methods. `owned.patch` includes the new dedicated test.

## Causal evidence and final verification

| Receipt | Result | Meaning |
| --- | --- | --- |
| `uat251-causal-red` | 40 fixture errors | Initial test pool maximum 2 was below supported minimum 5; no product verdict. |
| `uat251-causal-red-corrected-fixture` | 25 failed / 15 passed | Product failures reproduced; also retained test-only invalid target `tool` and expected exception mismatch. The actual target is `group`; transaction wrapper correctly raises `TransactionError`. |
| `uat251-permanent-red` | 22 failed / 22 passed, 0 skips | Every PostgreSQL case failed on the actual query boundary; all SQLite cases passed. Includes actual setup service apply. |
| `uat251-first-green` | 44 passed, 0 skips | Minimal production correction passes both backends. |
| `uat251-final-green` | 44 passed, 0 skips, 4 warnings, 11.17s | Final source/test bytes after test-only formatting, literal cleanup and generated test signing key. |
| `uat251-adjacent-green` | 60 passed / 1 failed, 0 skips | Existing repository/setup service/catalog regression suite. One unrelated preexisting fixture issue below; label is retained as originally run. |
| `uat251-adjacent-baseline` | Same 1 failed | Exact original repository loaded nonmutating; proves adjacent failure predates251. |

The dedicated tests use official `pg_temp_db` provisioning and the real asyncpg-backed `DatabasePool`; temporary SQLite uses normal AuthNZ migrations. The real `SetupMcpToolsService` → `McpHubService` → repository/pool path is exercised for initial apply and repeat. Only independent tool inventory and external audit emission are replaced; no model calls. Controls cover owner/target filtering and stable ordering, None semantics, same-name foreign owner preservation, nullable create readback both with and without caller transaction, caller rollback preserving previously committed rows, and absent/active/inactive policy overrides.

Static checks on the final two owned files: Ruff zero findings; production Bandit zero findings/errors; test Bandit zero findings/errors with assertion rule B101 excluded for test assertions only; Python compile passes; production diff whitespace check passes; new test formatter check passes. Original test-only Ruff C408 and Bandit B106 receipts are retained; both were corrected. Production baseline Ruff/Bandit were also zero.

## Adjacent baseline failure — separate fixture repair

`tests/AuthNZ_Unit/test_mcp_hub_repo.py::test_repo_ensure_tables_requires_governance_pack_distribution_tables` invokes `pool.execute("DROP TABLE mcp_governance_pack_source_candidates", ())` at line136. The established profile-write guard rejects this DDL with `ProfileUserWriteRejected` before `repo.ensure_tables` is reached. The same failure occurs with the exact pre251 source. The fixture was not edited or skipped. Parent will track its correction separately.

`replay_mcp251_baseline.py` loads only the baseline repository module; `baseline-replay-source.json` binds its exact source hash. The command receipt and redacted log retain both current and baseline failures. This is not an all-adjacent-green claim.

## Reproduction and limits

Exact commands are in `commands.json` and copied official command receipts under `receipts/`. Activate `.venv`; use the approved explicit-Jobs runner with a unique evidence label. Required PostgreSQL mode fails instead of skipping when unavailable; the official fixture owns database creation/drop. No held matrix database, runtime/profile/archive, browser, credentials, tasks, or Git state were changed.

This is actual service/repository acceptance on temporary databases, not native HTTP/browser acceptance or a new auth bypass test. The existing endpoint auth remains unchanged. The original safe native failure belongs to `.tmp/uat-repairs-231-246/native-targeted/pg-single/{setup-mcp-settled.txt,mcp-save-cause.redacted.log}`; no raw native logs are copied here. Other MCP repository methods with optional filters are outside this proven setup workflow and were not converted.
