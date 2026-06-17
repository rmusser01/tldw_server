---
id: TASK-2366
title: Address PR 2370 review feedback for VZ smoke image-store clone
status: Done
labels:
- sandbox
- vz_linux
- image_store
- pr_review
references:
- https://github.com/rmusser01/tldw_server/pull/2370
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tools/vz-linux-image/scripts/prepare-smoke-bundle.py
- tldw_Server_API/app/core/Sandbox/image_store.py
modified_files:
- backlog/tasks/task-2366 - Address-PR-2370-review-feedback-for-VZ-smoke-image-store-clone.md
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tools/vz-linux-image/scripts/prepare-smoke-bundle.py
- tldw_Server_API/app/core/Sandbox/image_store.py
- tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
- tools/vz-linux-image/tests/test_prepare_smoke_bundle.py
- tldw_Server_API/tests/unit/test_sandbox_image_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-valid review comments on PR #2370 after rebasing onto latest dev. Scope covers dry-run path consistency, materializer input hardening, manifest target_subdir validation, and focused regression tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2370 is rebased onto latest `origin/dev`.
- [x] #2 Dry-run disposable bundle path resolution uses the materializer/image-store normalization path and rejects invalid run ids.
- [x] #3 Materializer JSON and bundle-artifact validation fails cleanly for invalid UTF-8 and manifest path traversal.
- [x] #4 `clonefile(2)` lookup and run manifest `target_subdir` parsing handle malformed/unsupported inputs defensively.
- [x] #5 Focused regression tests cover every still-valid review finding.
- [x] #6 Backlog task-id collision from the rebase is resolved without changing the unrelated dev task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased `codex/vz-smoke-image-store-clone` onto latest `origin/dev`; no code conflicts occurred.
- Replaced dry-run raw path concatenation with a `prepare-smoke-bundle.py --print-path-only` mode that validates the source bundle, resolves the image-store root, normalizes the run id, and avoids filesystem writes.
- Hardened `prepare-smoke-bundle.py` to report invalid UTF-8 JSON as a clean one-line error, reject absolute/traversing bundle artifact names, and fall back when `ctypes.CDLL(None)` raises `OSError`.
- Hardened image-store run manifest loading by rejecting non-string `target_subdir` payloads before normalization.
- Added regression tests for dry-run normalization, invalid run ids, invalid UTF-8 manifests, manifest path traversal, rootfs materialization assertion, and invalid `target_subdir` payloads.
- Removed the branch-local duplicate `TASK-2365` file after latest `dev` introduced an unrelated `TASK-2365`; this follow-up is tracked as `TASK-2366`.
- Verification: `python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/unit/test_sandbox_image_store.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` passed with 65 tests.
- Verification: `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`, `git diff --check`, and Bandit on touched Python code all passed; Bandit reported 0 findings and 0 errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2370 onto latest dev and addressed all open review comments from Qodo, Gemini, and CodeRabbit. Also resolved a Backlog task-id collision introduced by the rebase by replacing the branch-local duplicate TASK-2365 record with TASK-2366. Verification passed: focused pytest suite including host-gated workflow test reported 65 passed, shell syntax check passed, git diff --check passed, and Bandit reported 0 findings / 0 errors for touched Python code.
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
