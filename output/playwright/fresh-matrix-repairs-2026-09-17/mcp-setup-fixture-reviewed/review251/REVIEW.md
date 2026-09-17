# Independent UAT251 review

**CLEAR for the bounded two-file repair; native acceptance remains pending.**

Task TASK13260.193. Frozen owned manifest `c626d10dced8c2cd7033673a26c15d28f7774b3a835032a12dab7393e71451ee`; production `mcp_hub_repo.py` SHA `a63c6cf5c51ffa1a702894740240a4767e27f677ddf8512ed1131ad1edd02fda`. Both live/snapshot hashes stayed stable. All author evidence-manifest entries validate; baseline production exactly matches recorded native commit86458ab88ce3fa62e6518c9d813c3860254ddb2c.

## Source review

Only seven SQL string constants across the declared five methods change (fourteen SQL lines): profile create/list, assignment create/get/list. Independent AST comparison confirms everything else is identical.

The otherwise untyped nullable operands now use `CAST(? AS TEXT/INTEGER)` matching actual PostgreSQL and SQLite schema columns. This fixes PostgreSQL's independently bound `IS NULL` parameters without changing values, parameter order, filters or SQL injection exposure. List `None` remains unfiltered; create readback preserves exact NULL matching and owner/type/name/target constraints. `COALESCE(o.is_active,FALSE)` matches PostgreSQL BOOLEAN and SQLite's existing integer/boolean normalization. Missing, active and inactive override semantics are retained.

No authorization predicate, transaction boundary, ordering, returned ID logic or write statement is weakened. Existing caller connections are still used rather than committed by the repository. The real service repeat control preserves a same-name foreign profile/default assignment while reusing the intended global pair. This does not turn the repository into a new endpoint authorization boundary.

## Independent verification

- **44 passed,0 skipped**, required PostgreSQL/SQLite focused suite,4 warnings,13.11s. Official `pg_temp_db` and real asyncpg-backed `DatabasePool` exercise repository plus actual SetupMcpToolsService→McpHubService flow. Only external inventory/audit collaborators are controlled.
- **60 passed,1 known failure,0 skipped**, adjacent repository/setup/catalog suites,124 warnings,29.99s. The failure occurs in fixture DDL before the intended assertion: `DROP TABLE mcp_governance_pack_source_candidates` is rejected by `ProfileUserWriteRejected`. The author exact-pre251 replay has the same failure and its bound baseline module was independently checked. That replay was inspected, not rerun here; the fixture remains separately tracked and untouched.
- Causal retained RED is **22 PostgreSQL failures /22 SQLite passes**,0 skips. Failures include untyped nullable parameters and incompatible Boolean COALESCE. Earlier pool-size/target-name/exception expectation setup mistakes are explicitly preserved as fixture corrections, not product evidence.
- Scoped Ruff:0 findings. Production Bandit:0 findings/0 errors. Test Bandit:0 findings/0 errors with B101 excluded only for pytest assertions. Both files compile. AST proves exactly five changed methods and only seven changed SQL constants.

Permanent controls cover empty/populated/omitted/partial filters, same-ID different owner type, same-name foreign rows, stable ordering, nullable owner/target readback with and without caller transactions, rollback preserving prior committed data, override activity and repeated real Save packs service application. Exact independent commands and sanitized runner receipts are retained alongside this review.

## Limits

No native browser/HTTP acceptance, model inference, held profile/database alteration, source edit, task/tracker or Git action was performed. This review does not claim all MCP optional-filter methods are portable, all adjacent tests are green, or endpoint authorization was newly tested. Root owns the original Save packs native retry and acceptance.
