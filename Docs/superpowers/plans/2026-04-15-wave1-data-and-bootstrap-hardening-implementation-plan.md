# Wave 1 Data And Bootstrap Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-baseline the DB/bootstrap hardening wave against the current tree, fix the remaining live shared-backend cache contract failure, and close the remaining verification/documentation gaps for Wave 1.

**Architecture:** Treat the April DB_Management review as a historical baseline, not a source of assumed still-live defects. Start by running the focused contract suite and recording which findings are already closed in the current tree. Then fix the only currently reproduced contract failure in the shared PostgreSQL content-backend cache path, add the one still-missing migration verification coverage gap, and finish by updating the DB-management docs and wave rebaseline artifact.

**Tech Stack:** Python, FastAPI, SQLite, PostgreSQL, pytest, loguru, Bandit

---

## File Map

### Rebaseline Artifact

- Create: `Docs/superpowers/reviews/db-management/2026-04-15-rebaseline.md`
  - Record which April 2026 DB-management findings are already closed in the current tree and which ones remain live after the focused Wave 1 suite.
- Modify: `Docs/superpowers/reviews/db-management/README.md`
  - Link the rebaseline artifact so later waves do not re-open already-closed findings.

### Shared Content Backend Cache Contract

- Modify: `tldw_Server_API/app/core/DB_Management/content_backend.py`
  - Remove or tighten the deferred-close behavior so superseded cached backends are deterministically closed in the cache lifecycle contract.
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
  - Keep reset helpers aligned with the tightened cache-close semantics.
- Modify: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
  - Only if caller-side cache clearing still bypasses the shared lifecycle helper.
- Test:
  - `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
  - `tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`

### Migration Verification Coverage

- Modify: `tldw_Server_API/app/core/DB_Management/db_migration.py`
  - Report version-gap issues in `verify_migrations()` for available migration sets, not only missing files and checksum mismatches in applied rows.
- Create: `tldw_Server_API/tests/DB_Management/test_db_migration_verification.py`
  - Cover checksum mismatch, missing migration file, and non-contiguous available-version reporting.
- Modify: `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`
  - Add `verify` command output coverage if the new verification issue needs CLI-facing assertions.

### Documentation And Verification

- Modify: `tldw_Server_API/app/core/DB_Management/README.md`
  - Document the Wave 1 fail-closed invariants that are now either verified closed or explicitly still live.
- Verify:
  - Focused pytest slices for the rebaseline suite
  - Shared-backend cache tests after the fix
  - Migration verification tests
  - Bandit on the touched backend scope

## Notes

- The focused rebaseline suite at plan-authoring time produced:
  - `47 passed`
  - `2 skipped`
  - `2 failed`
- The only reproduced failures were:
  - `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py::test_get_content_backend_closes_superseded_cached_backend`
  - `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py::test_reset_media_runtime_defaults_closes_cached_backend`
- The following April review areas already passed in the current tree during rebaseline and should not be re-implemented blindly:
  - PostgreSQL RLS installer fail-closed contract
  - migration loader duplicate/malformed handling
  - contiguous migration planning checks
  - CLI `--no-backup` passthrough
  - `UserDatabase_v2` fail-closed bootstrap checks
  - trusted-path helper tests
  - `media_db.api` DB-error propagation checks
  - backend-level FTS normalization checks

### Task 1: Re-Baseline Wave 1 Against The Current Tree

**Files:**
- Create: `Docs/superpowers/reviews/db-management/2026-04-15-rebaseline.md`
- Modify: `Docs/superpowers/reviews/db-management/README.md`
- Test:
  - `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`
  - `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`
  - `tldw_Server_API/tests/DB_Management/test_db_migration_planning.py`
  - `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`
  - `tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py`
  - `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
  - `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`
  - `tldw_Server_API/tests/DB_Management/test_media_db_api_error_contracts.py`
  - `tldw_Server_API/tests/DB_Management/test_database_backend_fts_normalization.py`

- [ ] **Step 1: Run the focused Wave 1 rebaseline suite**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/DB_Management/test_db_migration_loader.py tldw_Server_API/tests/DB_Management/test_db_migration_planning.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_db_path_utils.py tldw_Server_API/tests/DB_Management/test_media_db_api_error_contracts.py tldw_Server_API/tests/DB_Management/test_database_backend_fts_normalization.py`
Expected: Only the shared content-backend cache slice should remain red if the tree still matches the current baseline.

- [ ] **Step 2: Write the rebaseline artifact**

```markdown
## Rebaseline Summary

- Closed in current tree:
  - RLS fail-closed contract
  - migration loader and planning contract
  - CLI no-backup passthrough
  - UserDatabase_v2 fail-closed bootstrap checks
  - trusted-path contract
  - media_db API DB-error propagation
  - backend FTS normalization

- Still live:
  - shared content-backend cache close semantics

- Next action:
  - fix deterministic cache-close behavior
  - add migration verification gap coverage
```

- [ ] **Step 3: Update the review README to link the rebaseline**

```markdown
- `2026-04-15-rebaseline.md`
  - Current-tree rebaseline for Wave 1; identifies already-closed April findings and the remaining live cache-contract failure.
```

- [ ] **Step 4: Re-run the rebaseline suite after the artifact and README edits**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
Expected: Still FAIL before the cache-contract fix lands.

- [ ] **Step 5: Commit**

```bash
git add Docs/superpowers/reviews/db-management/2026-04-15-rebaseline.md Docs/superpowers/reviews/db-management/README.md
git commit -m "docs: record db management wave1 rebaseline"
```

### Task 2: Fix Shared Content Backend Cache Close Semantics

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/content_backend.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
- Modify: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- Test: `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`

- [ ] **Step 1: Keep the existing red cache tests as the regression guard**

```python
def test_get_content_backend_closes_superseded_cached_backend(monkeypatch) -> None:
    old_backend = _FakeBackend()
    new_backend = _FakeBackend()
    monkeypatch.setattr(content_backend, "_cached_backend", old_backend)
    monkeypatch.setattr(content_backend, "_cached_backend_signature", ("old",))
    monkeypatch.setattr(content_backend.DatabaseBackendFactory, "create_backend", staticmethod(lambda _cfg: new_backend))
    backend = content_backend.get_content_backend(_make_config("pw-new", "prefer"))
    assert backend is new_backend
    assert old_backend.pool.closed == 1
```

- [ ] **Step 2: Run the shared-backend cache tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`
Expected: FAIL because the cache currently defers pool closing instead of satisfying the deterministic close contract encoded by the tests.

- [ ] **Step 3: Implement deterministic close-before-forget behavior**

```python
def clear_cached_backend() -> None:
    global _cached_backend, _cached_backend_signature
    with _cache_lock:
        old_backend = _cached_backend
        _cached_backend = None
        _cached_backend_signature = None
    _close_backend_pool(old_backend)


def get_content_backend(config: ConfigParser):
    ...
    with _cache_lock:
        old_backend = _cached_backend
        backend = DatabaseBackendFactory.create_backend(settings.database_config)
        _cached_backend = backend
        _cached_backend_signature = signature
    if old_backend is not None and old_backend is not backend:
        _close_backend_pool(old_backend)
    return backend
```

- [ ] **Step 4: Re-run the cache lifecycle verification**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/content_backend.py tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py tldw_Server_API/app/core/DB_Management/DB_Manager.py tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py
git commit -m "fix: close cached content backends deterministically"
```

### Task 3: Add Migration Verification Gap Coverage

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/db_migration.py`
- Create: `tldw_Server_API/tests/DB_Management/test_db_migration_verification.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`

- [ ] **Step 1: Write the failing verification regressions**

```python
def test_verify_migrations_reports_noncontiguous_available_versions(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(migrator, "get_applied_migrations", lambda: [{"version": 1, "checksum": "a"}])
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [
            SimpleNamespace(version=1, checksum="a"),
            SimpleNamespace(version=3, checksum="c"),
        ],
    )

    issues = migrator.verify_migrations()
    assert any(issue["issue"] == "migration_version_gap" for issue in issues)


def test_verify_migrations_returns_empty_list_when_no_migration_files_exist(tmp_path):
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "missing"))
    assert migrator.verify_migrations() == []
```

- [ ] **Step 2: Run the migration verification tests to verify failure**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_db_migration_verification.py`
Expected: FAIL because `verify_migrations()` currently reports missing applied files and checksum mismatches but not available-version gaps.

- [ ] **Step 3: Implement version-gap reporting in `verify_migrations()`**

```python
if not available:
    return issues

available_versions = sorted(available)
expected_versions = list(range(available_versions[0], available_versions[-1] + 1))
missing_versions = [version for version in expected_versions if version not in available]
for version in missing_versions:
    issues.append(
        {
            "version": version,
            "issue": "migration_version_gap",
            "message": f"Migration file for version {version} is missing from the available set",
        }
    )
```

- [ ] **Step 4: Re-run the migration verification slice and CLI verify coverage**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_db_migration_verification.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py -k "verify or create_backup"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/db_migration.py tldw_Server_API/tests/DB_Management/test_db_migration_verification.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py
git commit -m "test: cover migration verification gaps"
```

### Task 4: Update DB_Management Contracts And Verify Wave 1

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/README.md`
- Verify: focused Wave 1 pytest slices and Bandit output

- [ ] **Step 1: Document the rebaselined Wave 1 contracts**

```markdown
## Wave 1 Contract Notes

- The current tree already closes the April 2026 RLS, migration-loader, UserDatabase_v2, trusted-path, media_db API, and backend-FTS findings covered by the Wave 1 rebaseline.
- Shared content-backend cache replacement and reset must close superseded backends deterministically.
- `verify_migrations()` reports missing migration files, checksum mismatches, and non-contiguous available-version gaps.
```

- [ ] **Step 2: Run the final focused Wave 1 verification suite**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/DB_Management/test_db_migration_loader.py tldw_Server_API/tests/DB_Management/test_db_migration_planning.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_db_path_utils.py tldw_Server_API/tests/DB_Management/test_media_db_api_error_contracts.py tldw_Server_API/tests/DB_Management/test_database_backend_fts_normalization.py tldw_Server_API/tests/DB_Management/test_db_migration_verification.py`
Expected: PASS

- [ ] **Step 3: Run Bandit on the touched backend scope**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py tldw_Server_API/app/core/StudyPacks/source_resolver.py tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py tldw_Server_API/app/api/v1/endpoints/media/navigation.py -f json -o /tmp/bandit_wave1_data_bootstrap.json`
Expected: JSON report written to `/tmp/bandit_wave1_data_bootstrap.json` with no new high-severity findings in the touched code.

- [ ] **Step 4: If PostgreSQL fixtures are available, run the PG follow-up smoke slice**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py -k "postgres or rls or create_backup"`
Expected: PASS, or an explicit skip if PostgreSQL is unavailable.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/README.md
git commit -m "docs: record rebaselined db hardening contracts"
```
