---
id: TASK-2300
title: Add MCP filesystem lock lease tools
status: In Progress
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
- [ ] #1 `fs.lock_acquire` and `fs.lock_release` are exposed with strict schemas, `lock` path-action metadata, safe result payloads, and path-boundable descriptors.
- [ ] #2 Process-local in-memory leases support acquire, renew, conflict, expiry cleanup, release, and wrong-token conflict without leaking absolute paths.
- [ ] #3 `fs.edit`, `fs.patch`, and `fs.write` accept optional lock lease validation and can require active matching locks through module settings without weakening hash/read-receipt preimage checks.
- [ ] #4 Regression tests cover lock tools, safe conflicts, TTL expiry, path escapes, and mutation validation for `fs.patch` and `fs.write`.
- [ ] #5 Documentation and task notes clearly state this slice is process-local advisory locking, with shared/persistent stores deferred.
- [ ] #6 Focused filesystem tests, compile checks, Bandit on touched Python scope, and `git diff --check` pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started implementation slice on 2026-06-09 in worktree `.worktrees/mcp-fs-lock-leases`.

Approved scope:
- Process-local in-memory advisory locks for the first slice.
- Future filesystem/DB/shared lock stores stay behind a follow-up seam.

Baseline:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py -q`
  - Result: 99 passed, 4 warnings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
