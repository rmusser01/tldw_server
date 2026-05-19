# Character Chat DB Health Release Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make corrupt per-user ChaChaNotes databases diagnosable and recoverable enough to unblock first-class Character Chat GA.

**Architecture:** Extend the existing ChaChaNotes dependency health snapshot instead of adding a parallel health service. Keep raw filesystem paths out of generic health responses, but expose a sanitized affected database identifier, reason code, and user-action recovery guidance. Document manual recovery as an explicit backup/recover/validate/restore operation; do not mutate user data automatically.

**Tech Stack:** FastAPI, pytest, SQLite, existing ChaChaNotes dependency module, existing `/api/v1/health` aggregate health endpoint, Backlog.md.

---

### Task 1: Sanitized ChaChaNotes Failure Details

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- Test: `tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py`
- Test: `tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py`

- [x] **Step 1: Write failing dependency diagnostics tests**

Add tests proving a corrupt existing user DB records a sanitized `last_failure` with:
- `reason_code: sqlite_corruption`
- affected database identifier such as `user:987/ChaChaNotes.db`
- no raw `/private` or temp path leakage
- recovery documentation/action metadata

- [x] **Step 2: Verify the new tests fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py::test_create_and_prepare_db_records_corrupt_db_recovery_details \
  tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py::test_api_health_exposes_chacha_recovery_details_without_path_leak \
  -q
```

Expected: fail because `last_failure` and recovery metadata do not exist.

- [x] **Step 3: Implement the minimal health metadata**

Update `ChaChaDatabaseCorruptionError` and `_record_init` so corruption preflight failures carry sanitized affected-DB and recovery fields into `get_chacha_health_snapshot()`.

- [x] **Step 4: Verify focused tests pass**

Run the same focused pytest command. Expected: pass.

### Task 2: Startup Fail-Open Contract

**Files:**
- Test: `tldw_Server_API/tests/Services/test_startup_chacha_warmup.py`
- Modify only if needed: `tldw_Server_API/app/services/startup_chacha_warmup.py`
- Modify only if needed: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`

- [x] **Step 1: Write startup warm-up regression test**

Add a test that schedules/runs warm-up against a corrupt user DB and verifies:
- no exception escapes the startup warm-up path
- health snapshot records the corrupt `ChaChaNotes.db`
- no raw path is exposed in the snapshot

- [x] **Step 2: Verify failure, then implement only if needed**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Services/test_startup_chacha_warmup.py::test_warm_chacha_db_for_user_records_corrupt_db_and_fails_open \
  -q
```

Expected: fail until Task 1 metadata exists. If Task 1 already makes this pass, do not change startup code.

### Task 3: Recovery Documentation And Task Closeout

**Files:**
- Create: `Docs/Operations/ChaChaNotes_DB_Recovery.md`
- Modify: `Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md`
- Modify: `backlog/tasks/task-429 - Add-Character-Chat-DB-health-and-recovery-release-gate.md`

- [x] **Step 1: Document recovery flow**

Create a recovery guide covering backup, `PRAGMA quick_check`, `PRAGMA integrity_check`, `sqlite3 .recover`, validation of recovered DB, intentional restore, and rollback.

- [x] **Step 2: Link the release gate**

Update the Character Chat PRD R11 section and `TASK-429` with the recovery guide, verification commands, and remaining risk.

- [x] **Step 3: Final verification**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py \
  tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py \
  tldw_Server_API/tests/Services/test_startup_chacha_warmup.py \
  -q
```

Run touched-scope Bandit:
```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py \
  tldw_Server_API/app/services/startup_chacha_warmup.py \
  -f json -o /tmp/bandit_character_chat_db_health.json
```

Run:
```bash
git diff --check
```

Expected: tests pass, Bandit has no new findings in touched production code, and diff check is clean.
