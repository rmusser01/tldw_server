# UAT252 — missing MCP schema fixture setup

Task TASK13260.194 was associated by the parent before edits. Only `tests/AuthNZ_Unit/test_mcp_hub_repo.py::test_repo_ensure_tables_requires_governance_pack_distribution_tables` is owned.

The original test initializes a real temporary SQLite AuthNZ database, verifies MCP readiness, then removes two tables to exercise readiness rejection. Its setup incorrectly issues DROP TABLE through `DatabasePool.execute`; the production profile write guard rejects this before the intended assertion. UAT251 baseline replay already confirmed the failure predates that repository repair.

Use a dedicated `sqlite3.connect(db_path)` schema connection for those two fixture-only DROP statements, explicitly closing it with `contextlib.closing`. This matches direct SQLite schema-corruption fixtures in `tests/AuthNZ/unit/test_profile_version_migration.py`; normal migrations similarly own schema setup. Keep the production pool/repository and original `pytest.raises(RuntimeError, match="mcp_governance_pack_source_candidates")` unchanged. No guard bypass is added to product code or managed connections.

The target remains a SQLite readiness test. Required-PG verification additionally runs UAT251's existing actual PostgreSQL/SQLite service and repository tests with official fixture provisioning. No manual database provisioning, native data, runtime, browser, task/tracker or Git mutations.
