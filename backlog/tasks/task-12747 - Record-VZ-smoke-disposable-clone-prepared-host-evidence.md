---
id: TASK-12747
title: Record VZ smoke disposable-clone prepared-host evidence
status: Done
labels:
- sandbox
- vz_linux
- evidence
- image_store
references:
- https://github.com/rmusser01/tldw_server/pull/2370
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
modified_files:
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the real VZ Linux host smoke on the prepared Apple silicon host after PR #2370 and record evidence that the canonical source bundle remains unchanged while VM stages execute from an image-store-backed disposable run bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Prepared Apple silicon host smoke runs through the image-store disposable clone path.
- [x] Evidence records the source bundle before/after hashes and proves the source bundle was not mutated.
- [x] Evidence records the disposable run bundle path, run manifest, and changed run rootfs hash.
- [x] Evidence records helper build/signing, runtime paths, cleanup state, artifacts, expected skips, residual gaps, and follow-up ownership.
- [x] Verification commands are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vz-smoke-clone-evidence`
- Branch: `codex/vz-smoke-clone-evidence`
- Base commit: `ab1c55c67c852040a5162308ef987ea124937baa`
- Artifact root: `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222`
- Source bundle: `/Users/macbook-dev/Library/Application Support/tldw/sandbox-images/source-bundles/debian-bookworm-arm64/bundle`
- Run id: `clone-evidence-20260616-130222`
- Run bundle: `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-clone-evidence-20260616-130222/image-store/runs/clone-evidence-20260616-130222/bundle`
- Baseline portable tests before the live smoke: `python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q` returned `27 passed, 2 warnings in 3.49s`.
- Dry-run smoke expansion used the disposable run bundle for both `TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH` and `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE`.
- Real host smoke built and signed the helper, then returned helper daemon smoke `2 passed` and real `vz_linux` smoke `3 passed, 11 deselected`.
- Source bundle hash lines were identical before and after the smoke run. Source `rootfs.img` stayed `e52c82e96667f6daa8f7e1d40be8a655aad110cd2c5acedb0a9fb5fa01118cbf`.
- Disposable run bundle `rootfs.img` ended at `ba04818c7f99b8742481b184bcb98eabbcfcdd476760bf13926be82f3cf7bb7c`, confirming writes landed in the clone.
- The accepted helper socket was removed after cleanup and the recorded helper pid `44279` was no longer running. A separate helper from an earlier worktree remained running outside this packet.
- Failure drills, launchd drill, stale-socket drill, stuck-boot drill, and host reboot drill were intentionally skipped for this packet and documented as residual/manual-gated follow-ups.
- Bandit was not run because this task only changes documentation and the Backlog task record; no Python or executable code was edited.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added a new `2026-06-16` disposable image-store clone evidence packet to `Docs/Sandbox/vz-linux-prepared-host-evidence.md`.
- Recorded the prepared-host dry-run and real smoke commands, helper signing state, runtime paths, artifact pointers, source bundle before/after hashes, run bundle hashes, helper cleanup state, expected skips, and residual follow-ups.
- Verified the canonical source bundle stayed unchanged while the disposable run bundle rootfs changed, proving the smoke wrapper used the image-store/disposable-clone abstraction instead of mutating the source bundle directly.
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
