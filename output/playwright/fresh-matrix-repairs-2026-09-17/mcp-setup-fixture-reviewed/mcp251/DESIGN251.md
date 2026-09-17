# UAT251 — MCP setup nullable scope queries

Associated task: TASK13260.193. Parent owns task/tracker/runtime/native/Git.

Native PostgreSQL single-user Save packs fails in `McpHubRepo.list_permission_profiles`: asyncpg cannot infer the type of the parameter used only by `IS NULL`. The separately bound equality parameter does not supply its type. Setup then reaches assignment listing and profile/assignment create readbacks with the same pattern.

## Bounded design

- Production scope: `tldw_Server_API/app/core/AuthNZ/repos/mcp_hub_repo.py`, only actual setup-chain queries demonstrated by causal tests.
- Give nullable parameters their existing column types via portable explicit `CAST`; preserve list `None` as no filter and create readback `None` as SQL NULL. Keep equality predicates, parameterization, sort order and transactions.
- Do not convert unrelated repository methods, add query translation, alter schema/auth, or suppress errors.
- Actual PostgreSQL controls also prove `COALESCE(boolean, 0)` fails in assignment get/list. Parent approved including this second setup-chain cause; the portable correction is `COALESCE(o.is_active, FALSE)` at these two projections only.
- Real service test uses `SetupMcpToolsService` → `McpHubService` → `McpHubRepo` → actual `DatabasePool`; only tool inventory and external audit emission are isolated. No model/network/native data.

## Validation

New test: `tests/DB_Management/test_mcp_setup_scope_filters_backends.py`. Official `pg_temp_db` owns PostgreSQL provisioning; SQLite uses migrated temporary AuthNZ storage. Test empty/list/create/readback, owner and target filters, repeated setup apply preserving IDs and unrelated owner rows. Add transaction/error controls only for demonstrated changed branches. Run required PostgreSQL with zero skips, existing setup service and repository controls, Ruff/Bandit/compile, retain original RED plus any intermediate failures, exact source hashes and attributable patch.

Native acceptance remains parent-owned.

## Final scope and result

Exactly five methods changed: `create_permission_profile`, `list_permission_profiles`, `create_policy_assignment`, `get_policy_assignment`, `list_policy_assignments`. Fourteen SQL lines changed. Explicit INTEGER casts match the existing owner ID columns; TEXT casts match scope/target columns. Both ordinary-pool and caller-transaction create readbacks are covered. Permanent RED: 22 PostgreSQL failures / 22 SQLite passes. Final GREEN: 44 passes, zero skips. The unrelated existing guarded-DROP fixture failure is baseline-confirmed and remains outside this repair.
