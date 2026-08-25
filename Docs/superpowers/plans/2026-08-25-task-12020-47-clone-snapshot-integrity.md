# Shared Workspace Clone Snapshot Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make workspace cloning produce one deterministic, isolated point-in-time copy without exposing partial targets or mutating unrelated recipient media.

**Architecture:** Add explicit source snapshot readers at the ChaChaNotes and Media repository boundaries, deterministic staged-target lifecycle methods in the Workspace DB, and an operation-owned Media insert path that never enters ordinary URL/content deduplication. Refactor `CloneService` to consume those primitives synchronously, check cooperative cancellation between bounded units, publish only after validation, and return truthful copy/readiness facts for the later Jobs worker.

**Tech Stack:** Python 3.10+, SQLite/PostgreSQL database abstractions, context managers, dataclasses, pytest, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-25-shared-workspace-clone-jobs-design.md`

## Global Constraints

- A clone is a point-in-time source snapshot; mixed fresh reads are not accepted as success.
- Target Workspace identity is supplied by the caller and is deterministic for the durable operation.
- Staged and publication-pending targets are hidden from all ordinary Workspace lists.
- Clone media is always operation-owned; ordinary URL/content deduplication is never used.
- Cleanup may delete only rows whose operation marker and content identity both match.
- Embeddings are not copied or generated; vector readiness remains explicit.
- Permission checks are supplied by the caller and run before reservation, between top-level items, and before publication.
- This task does not implement Jobs admission, API routes, WorkerSDK lifecycle, audit emission, or frontend behavior.

---

## File Structure

- `tldw_Server_API/app/core/Sharing/clone_models.py`: immutable snapshot, copy-result, readiness, warning, and controlled-failure contracts.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`: Workspace snapshot reader plus reserve/publish/confirm/discard/reconcile target methods and hidden-list behavior.
- `tldw_Server_API/app/core/DB_Management/media_db/repositories/clone_snapshot_repository.py`: repeatable Media reads, operation-owned inserts, ownership verification, and cleanup.
- `tldw_Server_API/app/core/DB_Management/media_db/api.py`: narrow facade methods for the new repository.
- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`: bind facade methods and forward-migrate clone ownership fields/indexes.
- `tldw_Server_API/app/core/DB_Management/media_db/schema/features/core_media.py`: PostgreSQL/SQLite schema parity for operation ownership metadata.
- `tldw_Server_API/app/core/Sharing/clone_service.py`: orchestrate deterministic snapshot copy using only the new repository methods.
- `tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py`: staged Workspace lifecycle and collision tests.
- `tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py`: media isolation and cleanup tests, placed with the repository's existing Media DB test family.
- `tldw_Server_API/tests/Sharing/test_clone_service.py`: service behavior, cancellation, partial results, and readiness tests.
- `.github/workflows/ci.yml`: assign newly added tests if the shard coverage guard requires it.

### Task 1: Define Immutable Clone Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Sharing/clone_models.py`
- Modify: `tldw_Server_API/tests/Sharing/test_clone_service.py`

**Interfaces:**
- Consumes: source Workspace, source membership, media, chunk, transcript, note, and artifact records.
- Produces: `WorkspaceCloneRequest`, `WorkspaceCloneSnapshot`, `MediaCloneSnapshot`, `CloneCopyCounts`, `CloneRetrievalReadiness`, `WorkspaceCloneResult`, `CloneCancelled`, and `CloneSnapshotUnavailable`.

- [ ] **Step 1: Write failing immutability and bounds tests**

```python
def test_clone_result_rejects_unbounded_warnings():
    with pytest.raises(ValueError, match="at most 8"):
        WorkspaceCloneResult(
            workspace_id="target",
            name="Copy",
            outcome="partial",
            publication_confirmed=False,
            counts=CloneCopyCounts.empty(),
            readiness=CloneRetrievalReadiness("ready", "ready", "needs_indexing"),
            warnings=tuple(CloneWarning(code=f"w{i}", count=1) for i in range(9)),
        )

def test_snapshot_defensively_copies_mutable_rows():
    row = {"id": "source-1", "title": "Original"}
    snapshot = WorkspaceCloneSnapshot.from_rows(workspace={"id": "ws"}, sources=[row], notes=[], artifacts=[])
    row["title"] = "Changed"
    assert snapshot.sources[0]["title"] == "Original"
```

- [ ] **Step 2: Run tests and verify missing contracts**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_clone_service.py -k 'contract or snapshot' -v`

- [ ] **Step 3: Add frozen contracts and controlled errors**

`WorkspaceCloneRequest` contains `source_workspace_id`, `target_workspace_id`, `operation_id`, `request_fingerprint`, and normalized `name`. `WorkspaceCloneResult` contains attempted/copied/failed counts by item class, operation-owned media count, readiness, at most eight stable warnings, and `publication_confirmed`. Validate all identifiers and warning codes as bounded ASCII; never store source titles, URLs, paths, content, or exception strings in result diagnostics.

- [ ] **Step 4: Run focused tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_clone_service.py -k 'contract or snapshot' -v`

```bash
git add tldw_Server_API/app/core/Sharing/clone_models.py tldw_Server_API/tests/Sharing/test_clone_service.py
git commit -m "feat(sharing): define immutable clone snapshot contracts"
```

### Task 2: Add Deterministic Staged Workspace Lifecycle

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py`

**Interfaces:**
- Consumes: deterministic target ID, operation ID, request fingerprint, normalized Workspace fields.
- Produces: `reserve_clone_target`, `publish_clone_target`, `confirm_clone_target_publication`, `discard_clone_target`, and `list_clone_targets_for_reconciliation`.

- [ ] **Step 1: Write failing lifecycle tests**

Cover first reservation, idempotent same-operation reservation, collision with an ordinary Workspace, collision with another operation, staged target exclusion from `list_workspaces`, publish-to-pending, confirmation marker clearing, operation-fenced discard, and failed/cancelled reconciliation lookup.

- [ ] **Step 2: Run tests and verify missing methods**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py -v`

- [ ] **Step 3: Forward-migrate explicit operation marker fields**

Add nullable `system_operation_id`, `system_operation_kind`, `system_operation_state`, and `system_request_fingerprint` fields to `workspaces`, with allowed states `staged` and `publication_pending` and an index on `(system_operation_kind, system_operation_state, system_operation_id)`. These are internal ownership metadata, not a second operation status store.

- [ ] **Step 4: Implement fenced lifecycle methods**

Add these exact methods:

- `reserve_clone_target(self, *, workspace_id: str, operation_id: str, request_fingerprint: str, name: str, description: str | None, workspace_profile: str) -> dict[str, Any]`
- `publish_clone_target(self, *, workspace_id: str, operation_id: str) -> dict[str, Any]`
- `confirm_clone_target_publication(self, *, workspace_id: str, operation_id: str) -> dict[str, Any]`
- `discard_clone_target(self, *, workspace_id: str, operation_id: str) -> bool`

Reservation inserts `archived=true` and `system_operation_state=staged`. Publication verifies exact ownership, sets `archived=false`, and transitions to `publication_pending`. Confirmation clears all system marker fields. Discard soft-deletes only an exact operation-owned row. Ordinary list methods add `system_operation_state IS NULL`; direct internal reads may opt into staged rows explicitly.

- [ ] **Step 5: Run Workspace regression tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -v`

- [ ] **Step 6: Commit staged lifecycle**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py
git commit -m "feat(workspaces): add fenced clone target lifecycle"
```

### Task 3: Add Repeatable Workspace And Media Source Snapshots

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/app/core/DB_Management/media_db/repositories/clone_snapshot_repository.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py`
- Create: `tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py`

**Interfaces:**
- Consumes: source Workspace ID and the media IDs referenced by its membership snapshot.
- Produces: `CharactersRAGDB.read_workspace_clone_snapshot(workspace_id: str) -> WorkspaceCloneSnapshot` and `MediaDatabase.read_media_clone_snapshots(media_ids: Sequence[int]) -> dict[int, MediaCloneSnapshot]`.

- [ ] **Step 1: Write consistency tests with concurrent source edits**

Start the snapshot read, mutate source Workspace/media state from a second connection, and assert every returned collection belongs to one transaction snapshot. Simulate a backend that cannot establish repeatable read and assert `CloneSnapshotUnavailable` before any target reservation.

- [ ] **Step 2: Run tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py -k snapshot -v`

- [ ] **Step 3: Implement repository-owned read transactions**

For SQLite, open a read-only connection where supported, issue `BEGIN`, execute all Workspace or media/chunk/transcript reads through that same connection, materialize immutable snapshot objects, then commit/close. For PostgreSQL, use a read-only `REPEATABLE READ` transaction. Any transaction setup failure, missing referenced media row, or snapshot loss raises `CloneSnapshotUnavailable`; never fall back to ordinary fresh reads.

- [ ] **Step 4: Verify no source handle leaks**

Add tests for success and exception paths that prove snapshot connections close and no transaction remains open.

- [ ] **Step 5: Run repository suites and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py -v`

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/media_db tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py
git commit -m "feat(sharing): read clone sources from repeatable snapshots"
```

### Task 4: Add Operation-Owned Media Snapshot Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/repositories/clone_snapshot_repository.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/features/core_media.py`
- Modify: `tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py`

**Interfaces:**
- Consumes: `MediaCloneSnapshot`, operation ID, source media identity, and expected SHA-256 content hash.
- Produces: `insert_operation_owned_clone_media(self, *, snapshot: MediaCloneSnapshot, operation_id: str, source_identity: str, expected_content_hash: str) -> OperationOwnedMediaResult` and `delete_operation_owned_clone_media(self, *, operation_id: str, source_identity: str, expected_content_hash: str) -> int`.

- [ ] **Step 1: Write collision and cleanup tests**

Seed recipient media with the same original URL and separately with identical content. Assert clone insertion creates a new deterministic operation-owned row and does not change either existing row's version, timestamp, chunks, or transcripts. Assert same-operation replay returns only the matching owned row after hash validation; ownership/hash mismatch fails closed. Assert cleanup removes exact operation-owned rows and cannot delete unrelated media.

- [ ] **Step 2: Run tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py -k 'owned or collision or cleanup' -v`

- [ ] **Step 3: Add first-class media ownership metadata**

Forward-migrate nullable `system_operation_id`, `system_operation_kind`, `system_source_identity`, and `system_content_hash` fields on `Media`, plus a unique partial index for `(system_operation_kind, system_operation_id, system_source_identity)` when the operation ID is present. Mirror schema behavior for SQLite and PostgreSQL.

- [ ] **Step 4: Implement the isolated insert**

Build storage URL `tldw-clone://workspace/{operation_digest}/{source_digest}` from bounded digests; preserve the original URL only in safe provenance and the Workspace source row. Insert Media, document version, keywords, chunks, and transcripts in one target Media DB transaction. Do not call `add_media_with_keywords`; do not invoke topic monitoring or ingestion side effects.

- [ ] **Step 5: Implement ownership-fenced cleanup**

Delete only rows matching operation kind, operation ID, source identity, and expected content hash. Return the exact count and raise a controlled conflict if any candidate's marker/hash differs.

- [ ] **Step 6: Run Media DB parity tests and commit**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py -v`

```bash
git add tldw_Server_API/app/core/DB_Management/media_db tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py
git commit -m "feat(media): isolate operation-owned clone snapshots"
```

### Task 5: Refactor CloneService Around Deterministic Snapshots

**Files:**
- Modify: `tldw_Server_API/app/core/Sharing/clone_service.py`
- Modify: `tldw_Server_API/tests/Sharing/test_clone_service.py`

**Interfaces:**
- Consumes: `WorkspaceCloneRequest`, immutable source snapshots, `should_cancel: Callable[[], bool]`, and `on_progress: Callable[[str, float], None] | None`.
- Produces: `CloneService.clone_workspace(request: WorkspaceCloneRequest, *, should_cancel: Callable[[], bool], on_progress: Callable[[str, float], None] | None = None) -> WorkspaceCloneResult`.

- [ ] **Step 1: Replace stale tests with deterministic-target behavior tests**

Assert the caller-provided target ID is used, the nonexistent `create_workspace` path is never called, operation-owned media persistence is used, source IDs map to exact created media, and warning/count bounds remain truthful.

- [ ] **Step 2: Add cancellation and publication tests**

Check cancellation before reservation, between each source/note/artifact, and before publication. Assert controlled cancellation discards the staged Workspace and operation-owned media. Assert item-level failures produce `outcome=partial`; fatal snapshot/publication failures never return success.

- [ ] **Step 3: Add readiness tests**

Verify text/citation readiness from copied target chunks/source state. When vectors are configured but absent, return `vector_search=needs_indexing`, partial outcome, and one `vector_index_not_generated` warning. When vector retrieval is disabled, return `not_configured` without warning that indexing happened.

- [ ] **Step 4: Run tests and verify old implementation fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_clone_service.py -v`

- [ ] **Step 5: Implement the new orchestration**

Load both source snapshots before target writes; reserve deterministically; copy each top-level item through operation-owned repository methods; invoke `should_cancel()` and the later worker-supplied authorization callback at every required boundary; validate target state; transition to `publication_pending`; and return `publication_confirmed=False`. `TASK-12020.48` completes the fenced Job and then calls `confirm_clone_target_publication`, after which its public result sets `publication_confirmed=true`.

- [ ] **Step 6: Remove obsolete random-ID and dedupe paths**

Remove internal `uuid.uuid4()` target/source identity generation and `_copy_media_item` usage of `add_media_with_keywords(overwrite=False)`. Preserve only bounded exception-class logging.

- [ ] **Step 7: Run Sharing and repository regressions**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_clone_service.py tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py -v`

- [ ] **Step 8: Commit service refactor**

```bash
git add tldw_Server_API/app/core/Sharing/clone_service.py tldw_Server_API/tests/Sharing/test_clone_service.py
git commit -m "fix(sharing): make workspace clones deterministic and isolated"
```

### Task 6: Verify Security, Parity, And CI Ownership

**Files:**
- Modify if required: `.github/workflows/ci.yml`
- Modify: `backlog/tasks/task-12020.47 - Make-shared-workspace-cloning-snapshot-isolated-and-cleanup-safe.md`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: the snapshot/copy foundation consumed by `TASK-12020.48`.

- [ ] **Step 1: Run all focused tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_clone_service.py tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py -v`

- [ ] **Step 2: Run static and security checks**

Run: `source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/Sharing/clone_models.py tldw_Server_API/app/core/Sharing/clone_service.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/media_db/repositories/clone_snapshot_repository.py`

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sharing tldw_Server_API/app/core/DB_Management/media_db/repositories/clone_snapshot_repository.py -f json -o /tmp/bandit_task_12020_47.json`

- [ ] **Step 3: Verify shard coverage**

Run: `source .venv/bin/activate && python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`

Add only uncovered new modules to their existing Sharing, Workspaces, and Media DB shards.

- [ ] **Step 4: Self-review ownership fences**

Search: `rg -n "add_media_with_keywords|uuid\.uuid4|create_workspace" tldw_Server_API/app/core/Sharing/clone_service.py`

Expected: no ordinary media dedupe, random target identity, or nonexistent Workspace creation call remains. Review every delete/update to ensure both operation marker and expected identity are in the predicate.

- [ ] **Step 5: Check the patch and close the task**

Run: `git diff --check && git status --short`

Record exact verification, Bandit output, PostgreSQL fixture availability, and any residual hard-exit cleanup limitation in `TASK-12020.47`; leave worker reconciliation implementation to `TASK-12020.48`.

```bash
git add .github/workflows/ci.yml backlog/tasks/task-12020.47\ -\ Make-shared-workspace-cloning-snapshot-isolated-and-cleanup-safe.md
git commit -m "chore(sharing): close clone snapshot foundation"
```

## Self-Review

- Spec coverage: source snapshot consistency, deterministic staging, hidden targets, publication fencing, operation-owned media, ownership-safe cleanup, cancellation boundaries, truthful counts, and readiness each map to a task.
- Placeholder scan: every implementation step names concrete methods, predicates, tests, and commands.
- Type consistency: `WorkspaceCloneRequest` and `WorkspaceCloneResult` are created in Task 1 and consumed unchanged by Task 5; target lifecycle and Media methods expose exact operation-fenced signatures.
- Scope control: the plan deliberately returns publication-pending state; Jobs completion, final publication confirmation, reconciliation scheduling, API projection, and audit semantics remain in `TASK-12020.48`.
