## Provenance

- Reviewed branch: `codex/wave1-data-bootstrap-hardening`
- Reviewed head: `02e6ec1e9`
- Focused suite: 47 passed, 2 skipped, 2 failed
- Remaining red tests: `test_content_backend_cache.py::test_get_content_backend_closes_superseded_cached_backend` and `test_content_backend_cache.py::test_reset_media_runtime_defaults_closes_cached_backend`
- Cache slice rerun: 2 failed, 2 passed; the same two cache-close assertions remained red

## Rebaseline Summary

- Closed in current tree:
  - RLS fail-closed contract (`test_pg_rls_policies_contract.py`)
  - migration loader and planning contract (`test_db_migration_loader.py`, `test_db_migration_planning.py`)
  - CLI no-backup passthrough (`test_migration_cli_integration.py`)
  - UserDatabase_v2 fail-closed bootstrap checks (`test_userdatabase_v2_bootstrap_failclosed.py`)
  - trusted-path contract (`test_db_path_utils.py`)
  - media_db API DB-error propagation (`test_media_db_api_error_contracts.py`)
  - backend FTS normalization (`test_database_backend_fts_normalization.py`)

- Still live:
  - shared content-backend cache close semantics

- Follow-up, non-defect:
  - add migration verification gap coverage

- Next action:
  - fix deterministic cache-close behavior
