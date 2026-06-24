# Ingestion Sources Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the validated Ingestion_Sources review findings without changing unrelated module behavior.

**Architecture:** Keep the existing adapters and service boundaries. Add bounded-read helpers to source adapters, replace archive member reverse lookup with an explicit normalized-path map, and keep degradation behavior item-scoped where possible.

**Tech Stack:** Python, pytest, SQLite via the existing async DB fixture patterns, standard library `zipfile`, `tarfile`, `subprocess`, and Loguru-adjacent project conventions.

---

## Stage 1: Archive Correctness And Expansion Limits
**Goal**: Archive snapshots map each original member to the correct normalized relative path and reject excessive expanded payloads.
**Success Criteria**: Suffix-colliding paths both retain text; oversized ZIP/TAR members and excessive member counts raise `ValueError`.
**Tests**: `tldw_Server_API/tests/Ingestion_Sources/test_archive_snapshot_adapter.py`
**Status**: Complete

### Task 1: Add Failing Archive Collision And Limit Tests

**Files:**
- Modify: `tldw_Server_API/tests/Ingestion_Sources/test_archive_snapshot_adapter.py`

- [x] Add `test_archive_snapshot_preserves_suffix_colliding_member_content` with a ZIP containing `export/a.md` and `export/dir/a.md`; assert both `items["a.md"]["text"]` and `items["dir/a.md"]["text"]` match their source content.
- [x] Add `test_validate_archive_members_rejects_zip_member_over_size_limit` by monkeypatching `INGESTION_SOURCES_ARCHIVE_MEMBER_MAX_BYTES=8` and validating a ZIP with one 9-byte file.
- [x] Add `test_validate_archive_members_rejects_tar_total_uncompressed_limit` by monkeypatching `INGESTION_SOURCES_ARCHIVE_TOTAL_MAX_BYTES=8` and validating a TAR with two supported files totaling more than 8 bytes.
- [x] Add `test_validate_archive_members_rejects_excessive_member_count` by monkeypatching `INGESTION_SOURCES_ARCHIVE_MAX_MEMBERS=1` and validating a ZIP with two files.
- [x] Run:
  `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && PYTHONPATH=$PWD python -m pytest tldw_Server_API/tests/Ingestion_Sources/test_archive_snapshot_adapter.py -v`
  Expected: new tests fail before implementation.

### Task 2: Implement Archive Mapping And Bounded Reads

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Sources/archive_snapshot.py`

- [x] Add `_archive_limit_int(env_name: str, default: int) -> int` with positive integer parsing.
- [x] Add defaults: max members 10,000, per-member bytes 50 MiB, total uncompressed bytes 250 MiB.
- [x] Add bounded member, count, and total checks used before archive member reads.
- [x] In ZIP validation, use `ZipInfo.file_size` and a bounded streaming read instead of unbounded `handle.read()`.
- [x] In TAR validation, use `TarInfo.size` and a bounded streaming read instead of unbounded `extracted.read()`.
- [x] In `build_archive_snapshot_with_failures`, build an explicit `member_to_relative_path` map from normalized archive members and remove `_normalized_relative_path` suffix matching.
- [x] Run the archive test file again and confirm PASS.

---

## Stage 2: Local Source Bounds
**Goal**: Local directory and local git sources avoid unbounded scan/read behavior and degrade oversized files item-by-item.
**Success Criteria**: Oversized local files appear in `failed_items`; local git enumeration has a timeout.
**Tests**: `tldw_Server_API/tests/Ingestion_Sources/test_local_directory_adapter.py`, `tldw_Server_API/tests/Ingestion_Sources/test_git_repository_adapter.py`
**Status**: Complete

### Task 3: Add Failing Local Source Bound Tests

**Files:**
- Modify: `tldw_Server_API/tests/Ingestion_Sources/test_local_directory_adapter.py`
- Modify: `tldw_Server_API/tests/Ingestion_Sources/test_git_repository_adapter.py`

- [x] Add a local directory test that monkeypatches `INGESTION_SOURCES_LOCAL_FILE_MAX_BYTES=4`, writes `too-large.md` with 5 bytes, and asserts the file is absent from `items` and present in `failures` with an oversize error.
- [x] Add a local git test that monkeypatches `INGESTION_SOURCES_LOCAL_FILE_MAX_BYTES=4`, creates a repo with `too-large.md`, and asserts item-level failure.
- [x] Add a git subprocess test that monkeypatches `subprocess.run` to raise `subprocess.TimeoutExpired`; assert `_git_ls_files` raises `ValueError` with timeout context.
- [x] Run the two test files and confirm the new tests fail before implementation.

### Task 4: Implement Local Source Limits And Git Timeout

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Sources/local_directory.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Sources/git_repository.py`

- [x] Add shared-style positive integer env parsing in each adapter for local file max bytes, defaulting to 50 MiB.
- [x] Check `stat.st_size` before reading or converting local files; record a failure entry and continue when over limit.
- [x] Add `INGESTION_SOURCES_GIT_LS_FILES_TIMEOUT_SECONDS`, default 30 seconds, and pass it to `subprocess.run`.
- [x] Catch `subprocess.TimeoutExpired` and raise `ValueError("Timed out enumerating git repository files ...")`.
- [x] Run local directory and git repository adapter tests and confirm PASS.

---

## Stage 3: Service State, Schema Indexes, And Bool Parsing
**Goal**: Service-level data handling becomes stricter and state transitions become observable.
**Success Criteria**: String booleans are rejected, schema indexes exist, and job finish detects fence mismatches.
**Tests**: `tldw_Server_API/tests/Ingestion_Sources/test_service_sqlite_state.py`, `tldw_Server_API/tests/Ingestion_Sources/test_models_and_service_contract.py`
**Status**: Complete

### Task 5: Add Failing Service Tests

**Files:**
- Modify: `tldw_Server_API/tests/Ingestion_Sources/test_service_sqlite_state.py`
- Modify: `tldw_Server_API/tests/Ingestion_Sources/test_models_and_service_contract.py`

- [x] Add tests asserting `normalize_source_payload({"enabled": "false", ...})` and `normalize_source_payload({"schedule_enabled": "true", ...})` raise `IngestionSourceValidationError`.
- [x] Add a SQLite schema test querying `sqlite_master` for expected index names after `ensure_ingestion_sources_schema`.
- [x] Add a job-state test that starts job `job-1`, attempts to finish with `job-2`, and asserts `finish_source_sync_job` raises `RuntimeError`.
- [x] Run the service/model tests and confirm the new tests fail before implementation.

### Task 6: Implement Service Hardening

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Sources/service.py`

- [x] Add `_normalize_bool(value, field_name, default)` that accepts actual bool and `None`, rejects strings and other types with `IngestionSourceValidationError`.
- [x] Use `_normalize_bool` in create normalization and update-source patch handling.
- [x] Add `CREATE INDEX IF NOT EXISTS` statements for user listing, scheduler filtering, snapshot lookup, artifact lookup, item event lookup, and active job checks.
- [x] Capture the cursor from the fenced `UPDATE` in `finish_source_sync_job`; if `rowcount == 0`, raise `RuntimeError` before returning state.
- [x] Run service/model tests and confirm PASS.

---

## Stage 4: Focused Verification And Task Closeout
**Goal**: Prove the touched behavior passes and record security verification.
**Success Criteria**: Focused pytest suite passes, Bandit reports no new findings, task notes and final summary are updated.
**Tests**: Focused Ingestion_Sources pytest files and Bandit on touched Python files.
**Status**: Complete

### Task 7: Run Verification

**Files:**
- Modify: `backlog/tasks/task-9936 - Harden-Ingestion-Sources-review-findings.md`

- [x] Run:
  `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && PYTHONPATH=$PWD python -m pytest tldw_Server_API/tests/Ingestion_Sources/test_archive_snapshot_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_local_directory_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_git_repository_adapter.py tldw_Server_API/tests/Ingestion_Sources/test_service_sqlite_state.py tldw_Server_API/tests/Ingestion_Sources/test_models_and_service_contract.py -v`
- [x] Run:
  `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && PYTHONPATH=$PWD python -m bandit -r tldw_Server_API/app/core/Ingestion_Sources -f json -o /tmp/bandit_ingestion_sources_review.json`
- [x] Run `git diff --check`.
- [x] Update TASK-9936 acceptance criteria, implementation notes, and final summary with actual verification results.

### Verification Results

- RED: focused pytest suite collected 31 tests and failed the 11 newly added regression cases before implementation.
- GREEN: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD python -m pytest -p pytest_asyncio.plugin ... -v` passed 31 tests with 76 warnings in 33.19s.
- Compile: `python -m py_compile` on touched Ingestion Sources implementation files exited 0.
- Security: Bandit over `tldw_Server_API/app/core/Ingestion_Sources` exited 0 with 0 JSON findings.
- Whitespace: `git diff --check` exited 0.
- Caveat: the default pytest command with all auto-loaded plugins was interrupted after stalling in broader import/cleanup work; the focused suite passed with plugin autoload disabled and `pytest_asyncio` explicitly enabled.

### PR Review Response Results

- Rebased the PR branch onto latest `origin/dev`.
- Dropped an unrelated Claims_Extraction design commit from the PR branch.
- Addressed review feedback for actual archive byte accounting, defensive archive member path lookup, and the unqualified SLF001 suppression in the git timeout test.
- Added ZIP/TAR regression tests for actual bytes exceeding total archive limits.
- GREEN: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD python -m pytest -p pytest_asyncio.plugin ... -v` passed 33 tests with 80 warnings in 32.29s.
- Compile: `python -m py_compile` on touched Ingestion Sources implementation files exited 0.
- Security: Bandit over `tldw_Server_API/app/core/Ingestion_Sources` exited 0 with 0 JSON findings.
- Whitespace: `git diff --check` exited 0.
