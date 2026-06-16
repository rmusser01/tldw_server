---
id: TASK-2365
title: Use image-store-backed disposable clones for VZ smoke
status: In Progress
labels:
- sandbox
- vz_linux
- image_store
- tools
references:
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tldw_Server_API/app/core/Sandbox/image_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Specify and implement the smallest slice of image-store-backed disposable clone behavior for the VZ Linux host smoke path so real VM execution no longer mutates the canonical source bundle. Scope includes a design/spec update first, then a wrapper-level abstraction and focused tests/docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec/update documents the image-store-backed disposable smoke-bundle design and its boundaries.
- [ ] #2 Host smoke wrapper uses a disposable image-store run bundle for helper bundle smoke, real host smoke, and optional failure drills.
- [ ] #3 The canonical source bundle path is validated and registered/planned, but not passed to VM-executing stages by default.
- [ ] #4 Focused tests cover dry-run command output, real-run materialization with fake helper/Python, clone metadata, and source-bundle immutability.
- [ ] #5 Operator docs/evidence guidance explain the disposable clone behavior and how to record source-vs-run bundle hashes.
- [ ] #6 Verification and Bandit results are recorded in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
