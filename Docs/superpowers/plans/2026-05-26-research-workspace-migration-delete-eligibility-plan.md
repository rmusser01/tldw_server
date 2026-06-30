# Research Workspace Migration Delete Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the Research Workspace migration protocol to authorize client legacy-storage deletion only after finalized server read-back verification proves all declared chunks were accepted with matching integrity metadata.

**Architecture:** Keep eligibility inside the existing migration protocol, not in source ingestion/indexing Jobs. Finalize performs a fresh read-back comparison between the persisted session declarations and accepted chunk receipts, persists a recovery manifest with explicit verification details, and sets `client_delete_eligible=true` only for non-empty, fully verified migrations. Client delete acknowledgement remains a separate explicit write gated by finalized status, matching manifest hash, and eligibility.

**Tech Stack:** FastAPI endpoint layer, `CharactersRAGDB` SQLite/PostgreSQL abstraction, existing workspace migration schemas, pytest integration tests.

---

## Files

- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `Docs/Design/Research_Workspace_Migration_Protocol_API.md`
- Modify: `backlog/tasks/task-515 - Enable-server-deletion-eligibility-for-Research-Workspace-migrations.md`

## Task 1: Update API Tests for Eligible Finalize and Ack

- [x] **Step 1: Write failing test for verified finalize**

Change the existing finalize test so, after all declared chunks are accepted, `POST /api/v1/workspaces/migrations/{id}/finalize` returns:

- `status == "finalized"`
- `client_delete_eligible is True`
- `recovery_manifest["can_delete_legacy_storage"] is True`
- `recovery_manifest["server_readback_verified"] is True`
- `recovery_manifest["verification_status"] == "verified"`

- [x] **Step 2: Write failing test for client delete ack success and idempotent retry**

After verified finalize, call `POST /client-delete-ack` twice with the matching manifest hash. Both calls should return `200 {"ok": true}`.

- [x] **Step 3: Write failing test for ineligible zero-chunk finalize**

Create a migration with no declared chunks, finalize it, and assert:

- `client_delete_eligible is False`
- `recovery_manifest["verification_status"] == "no_declared_chunks"`
- client delete ack returns `409`

- [x] **Step 4: Write failing test for manifest mismatch ack**

Finalize an eligible migration, then call delete ack with a different 64-character manifest hash. Assert `409`.

- [x] **Step 5: Run tests to verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py -q
```

Expected: FAIL because finalize still persists `client_delete_eligible=false` and delete ack still rejects eligible sessions.

Observed: failed as expected before implementation. Verified finalization still returned `client_delete_eligible=false`.

## Task 2: Add Finalize-Time Read-Back Verification

- [x] **Step 1: Add a small verifier helper**

In `CharactersRAGDB`, add a private helper that compares persisted declarations to accepted receipts by chunk id:

- every declared chunk has an accepted receipt;
- accepted receipt `sha256`, `byte_count`, and `chunk_kind` match the declaration;
- declared chunk count is greater than zero;
- no accepted receipt exists for an undeclared chunk when declarations exist.

Return a small dict such as:

```python
{
    "eligible": True,
    "status": "verified",
    "mismatch_chunk_ids": [],
    "undeclared_chunk_ids": [],
}
```

For zero declared chunks, return `eligible=False`, `status="no_declared_chunks"`.

- [x] **Step 2: Persist eligibility during finalize**

Update `finalize_workspace_migration()` to:

- call the verifier after the missing-chunk check;
- set `client_delete_eligible` from the verifier result;
- include `server_readback_verified`, `verification_status`, `mismatch_chunk_ids`, and `undeclared_chunk_ids` in the recovery manifest;
- keep `can_delete_legacy_storage` equal to `client_delete_eligible`.

- [x] **Step 3: Strengthen delete ack gate**

Update `record_workspace_migration_client_delete_ack()` to require:

- matching `acknowledged_manifest_hash`;
- `status == "finalized"`;
- `client_delete_eligible` truthy.

Repeated eligible ack calls remain accepted.

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py -q
```

Expected: PASS.

Observed: passed with 10 tests after adding the read-back verifier and persisted eligibility.

## Task 3: Documentation and Tracker

- [x] **Step 1: Update migration protocol design**

Change the API design text so finalize no longer says it always keeps `client_delete_eligible=false`. Document that eligibility is only true after finalized server read-back verification, and that zero-chunk or mismatch cases remain ineligible.

- [x] **Step 2: Update TASK-515**

Mark acceptance criteria and DoD according to verification results. Record any known skips, especially if PostgreSQL-specific tests are not run locally.

- [x] **Step 3: Run final verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py -q
git diff --check
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py -f json -o /tmp/bandit_task515.json
```

Expected: pytest and diff-check pass. Bandit should report no new high-confidence findings in touched code; if baseline findings appear, document them precisely.

Observed: pytest passed with 10 tests, `git diff --check` passed, and Bandit wrote `/tmp/bandit_task515.json` with an empty `results` array.

- [x] **Step 4: Commit**

Stage only TASK-515 files and commit:

```bash
git add tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  Docs/Design/Research_Workspace_Migration_Protocol_API.md \
  Docs/superpowers/plans/2026-05-26-research-workspace-migration-delete-eligibility-plan.md \
  "backlog/tasks/task-515 - Enable-server-deletion-eligibility-for-Research-Workspace-migrations.md"
git commit -m "feat: enable workspace migration delete eligibility"
```
