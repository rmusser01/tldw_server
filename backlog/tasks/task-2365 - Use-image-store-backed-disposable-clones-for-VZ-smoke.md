---
id: TASK-2365
title: Use image-store-backed disposable clones for VZ smoke
status: Done
labels:
- sandbox
- vz_linux
- image_store
- tools
references:
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tldw_Server_API/app/core/Sandbox/image_store.py
modified_files:
- Docs/Sandbox/macos-runtime-operator-notes.md
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
- Docs/superpowers/plans/2026-06-16-vz-smoke-image-store-clone.md
- Docs/superpowers/specs/2026-06-16-vz-smoke-image-store-clone-design.md
- tldw_Server_API/app/core/Sandbox/image_store.py
- tldw_Server_API/tests/unit/test_sandbox_image_store.py
- tools/vz-linux-image/README.md
- tools/vz-linux-image/scripts/prepare-smoke-bundle.py
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
- tools/vz-linux-image/tests/test_prepare_smoke_bundle.py
- backlog/tasks/task-2365 - Use-image-store-backed-disposable-clones-for-VZ-smoke.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Specify and implement the smallest slice of image-store-backed disposable clone behavior for the VZ Linux host smoke path so real VM execution no longer mutates the canonical source bundle. Scope includes a design/spec update first, then a wrapper-level abstraction and focused tests/docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec/update documents the image-store-backed disposable smoke-bundle design and its boundaries.
- [x] #2 Host smoke wrapper uses a disposable image-store run bundle for helper bundle smoke, real host smoke, and optional failure drills.
- [x] #3 The canonical source bundle path is validated and registered/planned, but not passed to VM-executing stages by default.
- [x] #4 Focused tests cover dry-run command output, real-run materialization with fake helper/Python, clone metadata, and source-bundle immutability.
- [x] #5 Operator docs/evidence guidance explain the disposable clone behavior and how to record source-vs-run bundle hashes.
- [x] #6 Verification and Bandit results are recorded in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/specs/2026-06-16-vz-smoke-image-store-clone-design.md`
  and `Docs/superpowers/plans/2026-06-16-vz-smoke-image-store-clone.md` before
  implementation.
- Extended `SandboxImageStore.prepare_run_clone()` with optional
  `target_subdir` support so smoke run files can live under
  `runs/<run-id>/bundle/` while the image-store run manifest remains at
  `runs/<run-id>/manifest.json`.
- Added `tools/vz-linux-image/scripts/prepare-smoke-bundle.py` to register the
  source bundle, prepare the run clone manifest, materialize clone items with
  macOS `clonefile(2)` fallback to `shutil.copy2()`, and copy bundle metadata
  into the disposable run bundle.
- Updated `run-host-e2e-smoke.sh` so `--bundle` is the canonical source bundle
  and helper smoke, real host smoke, and optional failure drills receive the
  disposable run bundle path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented image-store-backed disposable smoke bundles for the lower-level
  VZ Linux host smoke wrapper. The canonical source bundle is validated and
  registered/planned, but VM-executing stages now use
  `<image-store-root>/runs/<run-id>/bundle`.
- Added focused materializer, shell-wrapper, and image-store tests covering
  run-bundle materialization, dry-run output, fake real-run integration, clone
  metadata, and source rootfs immutability.
- Updated operator and evidence docs to record source bundle hashes separately
  from disposable run bundle hashes. Fresh prepared-host evidence is still
  needed before closing the residual direct-bundle mutability gap.
- Verification: focused pytest suite passed with `59 passed, 6 warnings`;
  `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` passed;
  `git diff --check` passed; Bandit reported `0` findings for
  `tools/vz-linux-image/scripts/prepare-smoke-bundle.py` and
  `tldw_Server_API/app/core/Sandbox/image_store.py`.
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
