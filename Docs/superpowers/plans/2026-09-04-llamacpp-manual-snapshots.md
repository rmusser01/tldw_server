# Manual llama.cpp Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Administrators can manually save and restore compatible processed context on managed llama.cpp runtimes.

**Architecture:** Private immutable artifacts and durable receipts belong to the managed profile. The supervisor owns launch-fenced single-dispatch operations; the existing admin API and shared Admin page expose explicit actions and recovery states.

**Tech Stack:** Python, FastAPI, Pydantic, asyncio, local files, React/TypeScript, pytest, Vitest and Playwright; existing dependencies only.

**Spec:** `Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md`

**Tracking:** TASK-13160; execution TASK-13161 → TASK-13162 → TASK-13163.

ADR required: yes
ADR path: Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
Reason: Sensitive storage ownership and non-replayable supervisor operations. Record acceptance of this proposed ADR with implementation review before code begins. ADR-003 remains the general Jobs default; ADR-021 and ADR-030 govern lifecycle and egress.

## Global constraints

- Release one is admin-only, opt-in per managed profile, and restricted to a single-model runtime on the supervisor's host.
- No external endpoints, router mode, cross-profile restore, import/export, user-provided paths, scheduled saves, automatic restore, or changes to existing Pause/Resume semantics.
- Retention defaults to the newest 10 committed snapshots per managed profile, configurable from 1 to 1000.
- Use short capability probes (5 seconds) and a bounded 10-minute mutation deadline including staging; connect timeout 5 seconds, write 30 seconds, read bounded by remaining operation time.
- Record a retention window of at least 30 days for receipts and refuse keys older than that window using a server-issued request token.
- Unknown outcomes require confirmed child exit before further snapshot mutation. Explicit Stop remains available for recovery.
- Do not claim conversation recovery, encryption at rest, inference isolation, or multimodal support without evidence.
- Read repository/nested AGENTS.md before execution. Mark each implementation task In Progress and attach its plan before code edits. Implementation tasks remain To Do in this documentation change.
- Use targeted verification and the existing server virtualenv. Check available runtime/model assets; do not install dependencies or download models implicitly.

## Module boundaries and shared interfaces

Create these files under `tldw_Server_API/app/core/Local_LLM/`:

| File | Responsibility |
| --- | --- |
| `llamacpp_snapshot_models.py` | Strict fingerprint, metadata, request and receipt types. |
| `llamacpp_snapshot_store.py` | Private paths, catalog, atomic publication, receipts, process ownership and retention. |
| `llamacpp_snapshot_compatibility.py` | Stable content fingerprints and mismatch reasons. |
| `llamacpp_snapshot_operations.py` | Admission, single-dispatch state machine, deadlines and checked upstream calls. |

New shared types, not existing symbols:

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Fingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_sha256: str
    executable_sha256: str
    projector_sha256: str | None = None
    effective_options_sha256: str
    adapters_sha256: str
    format_version: Literal[1] = 1

class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_id: int = Field(ge=0)
    expected_launch_generation: str
    request_id: str
    replace_confirmed: bool = False
```

Add bounded validation: digest strings are exactly 64 lowercase hex characters;
IDs are opaque generated IDs, not paths; tokens and all strings have explicit
maximum lengths. Define `SnapshotMetadata` with profile/snapshot IDs, source slot,
UTC creation time, monotonic commit sequence, positive byte/token counts, SHA-256,
fingerprint, actor ID and format version 1. Define `OperationReceipt` with IDs,
launch generation, request digest, save/restore kind, dispatched flag, optional
snapshot ID/token count, safe error code and state:
`validating|saving|verifying|restoring|complete|failed|outcome_unknown`.
Never serialize internal filesystem paths or request tokens into responses.

## Stage 1 / TASK-13161: Private storage and compatibility

**Files:** Create the models/store/compatibility modules above and
`tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_store.py`,
`test_llamacpp_snapshot_compatibility.py`. Modify
`llamacpp_runtime_models.py`, `app/api/v1/schemas/llamacpp_admin_schemas.py`
and `test_llamacpp_profile_store.py` for persisted opt-in and retention defaults.

**Produces:**

```python
def compare_fingerprints(saved: Fingerprint, current: Fingerprint | None) -> list[str]:
    if current is None:
        return ["compatibility_unknown"]
    return [name for name in type(saved).model_fields
            if getattr(saved, name) != getattr(current, name)]

# SnapshotStore(root: Path) public methods:
# list(profile_id: str) -> list[SnapshotMetadata]
# commit(profile_id: str, staged: Path, metadata: SnapshotMetadata) -> SnapshotMetadata
# stage_restore(profile_id: str, snapshot_id: str, working: Path) -> Path
# prune(profile_id: str, keep: int) -> list[str]  # failed deletion IDs
# delete(profile_id: str, snapshot_id: str) -> None
# write_receipt(receipt: OperationReceipt) -> None
# read_receipt(profile_id: str, operation_id: str) -> OperationReceipt
```

- [ ] Write this failing comparison test before the new implementation:

```python
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_models import Fingerprint
from tldw_Server_API.app.core.Local_LLM.llamacpp_snapshot_compatibility import compare_fingerprints

def test_unknown_and_changed_model_never_match():
    saved = Fingerprint(model_sha256="a" * 64, executable_sha256="b" * 64,
                        effective_options_sha256="c" * 64, adapters_sha256="d" * 64)
    assert compare_fingerprints(saved, None) == ["compatibility_unknown"]
    changed = saved.model_copy(update={"model_sha256": "e" * 64})
    assert compare_fingerprints(saved, changed) == ["model_sha256"]
```

- [ ] Run `python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_compatibility.py -q`; verify missing-module failure. Implement the strict types and comparison above; rerun green.
- [ ] Write store tests using temporary directories and real files: known hash publication, corrupted restore, traversal IDs, symlink targets, oversized manifests, disk-full writes, and interruption at each commit boundary. Verify outside files and previous snapshots remain unchanged.
- [ ] Implement private no-follow file access, exclusive creation and process-held OS ownership lock. Unsupported confinement fails closed. Generate file names internally; no process-global umask changes. Run blocking file/hash work off-loop.

```python
# Commit ordering under store ownership:
# validate staged file and metadata -> chunked copy/hash to exclusive temp file
# -> compare digest/size -> fsync -> rename binary -> publish/fsync manifest last.
# Catalog entries require valid manifests. Orphan binaries are never listed.
```

- [ ] Add failure-injection tests before implementing recovery. Catalog recovery ignores incomplete publication; cleanup requires proven dead child for launch staging. Test timestamp rollback against monotonic pruning. Failed pruning returns warnings; no pruning before verified commit.
- [ ] Add `snapshots_enabled: bool = False` and `snapshot_retention: int = Field(default=10, ge=1, le=1000)` to profile models/requests. Test old-profile defaults and round-trip persistence. Updating retention alone never deletes; disabling preserves browse/delete.
- [ ] Fingerprint content, executable, projector and canonical effective options/adapters. Check file identity before/after hashing. Unknown mutable state is unsupported, not compatible. Test mismatches individually; model aliases and paths are not identity.
- [ ] Run new tests plus `test_llamacpp_profile_store.py`, touched-file lint/format and Bandit. Record evidence and commit the reviewed stage.

## Stage 2 / TASK-13162: Fenced operations and admin API

**Files:** Create `llamacpp_snapshot_operations.py`,
`tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_operations.py` and
`test_llamacpp_snapshot_api.py`. Modify `llamacpp_supervisor_service.py`,
`llamacpp_process_runner.py`, `llamacpp_runtime_models.py`,
`llamacpp_server_args.py`, `app/api/v1/endpoints/llamacpp.py`,
`app/api/v1/schemas/llamacpp_admin_schemas.py`; wire lifecycle in
`app/services/startup_heavy_init.py` only if supervisor shutdown wiring needs it.
Extend existing supervisor/process-runner tests.

**Consumes:** Stage 1 store/types. **Produces:** supervisor methods:

```python
# async snapshot_slots(profile_id: str) -> dict[str, object]
# async snapshot_catalog(profile_id: str, offset: int, limit: int) -> dict[str, object]
# async save_snapshot(profile_id: str, request: SnapshotRequest, actor_id: str) -> OperationReceipt
# async restore_snapshot(profile_id: str, snapshot_id: str,
#                        request: SnapshotRequest, actor_id: str) -> OperationReceipt
# async delete_snapshot(profile_id: str, snapshot_id: str) -> None
# async snapshot_operation(profile_id: str, operation_id: str) -> OperationReceipt
```

- [ ] Reuse existing admin API auth fixtures for a six-route parametrized test: slots, catalog, save, restore, delete and receipt. Assert non-admin denial before resource lookup. Add cross-profile ID and extra-field rejection cases.
- [ ] Run `python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshot_api.py -q` and inspect missing-route failures. Add the six spec routes with role/rate-limit dependencies and whitelist response models. Catalog offset is nonnegative; limit is 1..100.
- [ ] Add launch-generation UUID and generated per-launch save path. Reject conflicting launch arguments. Never restart on enablement. Return stopped/disabled/unsupported/busy/restart-required diagnostics without probing user URLs. Initially support only explicitly tested text configurations.
- [ ] Write duplicate and stale-generation tests with a fake transport counting calls. Same signed token/input gives the same operation ID and one upstream call; different input gives 409. A changed generation or wrong process owner gives zero upstream calls.
- [ ] Implement admission and execution with these exact transition boundaries:

```python
# Admission under profile reservation:
# validate token signature/profile/age -> existing receipt lookup/digest comparison
# -> validate owner/generation -> persist receipt -> register owned operation.
# Execution outside general profile lock:
# prepare -> recheck generation/slot -> persist dispatched=True -> send ONCE
# -> validate acknowledgement -> verify/commit or finish restore -> persist result.
# Any failure after dispatch: outcome_unknown, no retry, quarantine launch.
```

- [ ] Sign bounded tokens containing profile, issuance time and random nonce using a private persistent HMAC key; never use provider credentials. Validate in constant time, reject future/expired tokens before new admission, retain receipts at least 30 days. Never log tokens. GET slots issues fresh tokens and returns latest operation ID for reload recovery, even while stopped.
- [ ] Add timeout-after-send, malformed-acknowledgement, disconnect, disk-full and crash tests. Assert no commit/prune/retry after uncertainty. Add a blocking fake transport to prove Stop/Pause/Restart conflict during work, then explicit Stop works after Outcome unknown.
- [ ] Integrate per-profile reservation and configurable server-wide concurrency, default 1. Register/drain operations through supervisor lifecycle. Shutdown stops admission, boundedly drains, persists uncertainty before cancellation, then stops child; ownership is released only after confirmed death. Startup never replays dispatched receipts. Block profile deletion while snapshots exist.
- [ ] Use central checked egress with captured server-owned origin, no mutation retry or cross-origin redirect. Validate response slot, basename and counts against the operation; recheck generation before success. Keep working files quarantined until child death. Do not infer death from timeout or a reused PID.
- [ ] Run the new API/operation tests, existing supervisor/process-runner/admin tests, touched-file static checks and Bandit. Record the fault-injection results and commit the reviewed stage.

## Stage 3 / TASK-13163: Admin workflow and live reuse proof

**Files:** Create
`apps/packages/ui/src/components/Option/Admin/LlamacppSnapshotsPanel.tsx` and
`__tests__/LlamacppSnapshotsPanel.test.tsx` in that directory. Modify
`LlamacppAdminPage.tsx`, `LlamacppRuntimePanel.tsx`,
`apps/packages/ui/src/services/tldw/TldwApiClient.ts`,
`apps/packages/ui/src/types/llamacpp-admin.ts` and
`apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts`.
Create `Docs/Guides/llamacpp-manual-snapshots.md` and
`tldw_Server_API/tests/LLM_Local/test_llamacpp_snapshots_live.py`.

**Consumes:** Six Stage 2 routes. **Produces:** approved manual workflow, operator guidance and measured real-runtime evidence.

- [ ] Add typed client methods matching Stage 2 names/fields. Keep network work in the Admin container and controlled presentation in the panel. Use `onRestore(snapshotId: string, slotId: number): void` as the panel callback; the container supplies the token and generation.
- [ ] Build a test fixture with one idle slot and compatible snapshot, then test that selecting Restore cannot mutate without confirmation:

```tsx
// onRestore is a vi.fn() passed to the rendered fixture panel.
await user.click(screen.getByRole("button", { name: "Restore" }))
expect(onRestore).not.toHaveBeenCalled()
expect(screen.getByText(/Failure may also clear it/)).toBeVisible()
await user.click(screen.getByRole("button", { name: "Restore into slot 0" }))
expect(onRestore).toHaveBeenCalledTimes(1)
```

- [ ] From `apps/tldw-frontend`, run `npm run test:run -- ../packages/ui/src/components/Option/Admin/__tests__/LlamacppSnapshotsPanel.test.tsx`; verify red, implement the panel, rerun green.
- [ ] Implement enablement with restart-required guidance, slot refresh/save, sorted catalog, retention, compatibility reasons and inline restore/delete confirmations. Reuse shared primitives, themes and i18n. Preserve spec warnings. No path inputs, prompt previews, automatic chat sends or provider changes.
- [ ] Poll active operations only while visible; abort reads on unmount/profile switch and reject late responses using profile/generation fences. Recover latest operation from GET slots after reload, never resubmit. New manual mutations get fresh tokens. Show unknown outcomes with explicit Stop recovery, not Retry Restore.
- [ ] Test stopped/busy/unsupported/incompatible/unknown/error/empty states, keyboard focus return, live status announcements, narrow viewport and both themes. Keep disabled reasons visible without hover. Confirm deletion names the permanent target and does not erase a slot.
- [ ] Add opt-in live pytest gated by `TLDW_SNAPSHOT_LIVE=1`, operator-supplied test runtime/model and an explicitly disposable profile. Never download assets or use a production profile. Record executable/model hashes and effective options. Seed a known long text prefix, save, stop, start, restore, then submit identical prefix plus suffix. Compare cached/processed token evidence against an identical request on a separate cold process. File existence, HTTP 200 and similar answers are insufficient.
- [ ] Run the Admin flow against that runtime with Playwright. Verify Chatbook's original messages/tools remain unchanged and Pause/Resume do not save/restore. Record routing limitations; one successful reuse does not promise reuse for every chat request. Unsupported multimodal remains disabled unless separately proven.
- [ ] Write the operator guide covering sensitivity, quiescing callers, exact supported build, retention, disk encryption/backups, manual-only semantics and unknown-outcome Stop recovery. Record actual commands and sanitized runtime metrics, not private prompts.
- [ ] Run targeted Vitest/Playwright, live pytest, touched-file lint/format and Bandit for the harness. A skipped live test keeps this task open. Commit reviewed work; no merge or full-suite run is implied.

## Coverage and completion

| Approved requirement | Stage |
| --- | --- |
| Private ownership, integrity, compatibility, atomic publication, retention | 1 |
| Admin API, generation/owner fences, egress, deduplication, unknown recovery | 2 |
| Guided setup, manual controls, accessibility, diagnostics, reload recovery | 3 |
| Crash/disk/stale-process fault evidence | 1 + 2 |
| Pinned-build cache reuse, Chatbook semantics, operator documentation | 3 |

For each stage, check Backlog acceptance criteria against actual evidence, add
implementation notes and ADR links, run review, then mark Done. Missing live proof
keeps TASK-13163 open. Automatic checkpoints remain outside this plan.
