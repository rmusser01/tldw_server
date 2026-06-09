---
id: TASK-2300
title: Add MCP filesystem lock lease tools
status: Done
labels:
- mcp
- filesystem
- concurrency
- security
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-lock-leases-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-lock-leases-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add optional filesystem lock leases alongside hashes/read receipts. Include `fs.lock_acquire`, `fs.lock_release`, lease expiry/cleanup, safe lock-conflict responses, and optional policy-required lock validation for `fs.patch` and `fs.write`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `fs.lock_acquire` and `fs.lock_release` are exposed with strict schemas, `lock` path-action metadata, safe result payloads, and path-boundable descriptors.
- [x] #2 Process-local in-memory leases support acquire, renew, conflict, expiry cleanup, release, and wrong-token conflict without leaking absolute paths.
- [x] #3 `fs.edit`, `fs.patch`, and `fs.write` accept optional lock lease validation and can require active matching locks through module settings without weakening hash/read-receipt preimage checks.
- [x] #4 Regression tests cover lock tools, safe conflicts, TTL expiry, path escapes, and mutation validation for `fs.patch` and `fs.write`.
- [x] #5 Documentation and task notes clearly state this slice is process-local advisory locking, with shared/persistent stores deferred.
- [x] #6 Focused filesystem tests, compile checks, Bandit on touched Python scope, and `git diff --check` pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started implementation slice on 2026-06-09 in worktree `.worktrees/mcp-fs-lock-leases`.

Approved scope:
- Process-local in-memory advisory locks for the first slice.
- Future filesystem/DB/shared lock stores stay behind a follow-up seam.

Baseline:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`
  - Result: 99 passed, 4 warnings.

Implementation:
- Added a process-local `InMemoryFilesystemLockManager` with acquire, renew, release, validate, TTL expiry cleanup, and safe conflict payload helpers.
- Added `fs.lock_acquire` and `fs.lock_release` descriptors/execution paths with `lock` file-policy metadata.
- Added optional `lock_lease_id` validation to `fs.edit` and `fs.write`, and `lock_lease_id_by_path` validation to `fs.patch`.
- Added `require_lock_for_mutation` enforcement and a second pre-commit lease validation so a lease that expires after preimage authorization cannot still commit a write.
- Marked the `lock` file-policy action as implemented and updated package user-guide documentation with the process-local advisory limitation.

Red checks:
- Lock-focused tests initially failed for missing `fs.lock_*` tools/module and missing `require_lock_for_mutation` enforcement.
- `test_filesystem_write_rejects_lock_that_expires_before_commit` initially failed because writes still committed after a lease expired between preimage authorization and commit.

Verification:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`
  - Result: 109 passed, 4 warnings.
- `source .venv/bin/activate && python -m py_compile tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py mcp_unified/interfaces/file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
  - Result: passed.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py mcp_unified/interfaces/file_policy_actions.py -f json -o /tmp/bandit_mcp_fs_lock_leases.json`
  - Result: passed, 0 findings.
- `git diff --check`
  - Result: passed.

PR review follow-up:
- Rebased PR #2331 on latest `origin/dev`.
- Offloaded lock acquire/release path checks through `asyncio.to_thread(...)`.
- Added explicit `lock_missing` response for failed renewals of expired or missing leases.
- Added bounded rotating expired-lease sweeps in the process-local lock manager.
- Normalized release lease tokens consistently with renewal and mutation validation.
- Rejected blank mutation lease IDs instead of treating them as omitted.
- Revalidated leases before creating parent directories in patch/write commit paths.
- Removed `time` from `filesystem_locks.__all__`.
- Cleaned task status and replaced machine-specific verification paths with repo-relative commands.

Review follow-up verification:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`
  - Result: 115 passed, 4 warnings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented in PR #2331: https://github.com/rmusser01/tldw_server/pull/2331

This slice adds process-local advisory filesystem lock leases, exposes `fs.lock_acquire` and `fs.lock_release`, wires optional lease validation into `fs.edit`, `fs.patch`, and `fs.write`, and documents that shared/persistent lock stores are deferred. Verification completed with focused MCP filesystem tests, compile checks, Bandit on touched implementation scope, and `git diff --check`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
