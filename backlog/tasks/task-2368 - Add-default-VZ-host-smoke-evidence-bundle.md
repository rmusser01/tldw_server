---
id: TASK-2368
title: Add default VZ host smoke evidence bundle
status: Done
labels:
- sandbox
- vz_linux
- evidence
- host_gated
references:
- Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- .github/workflows/vz-linux-host-gated.yml
modified_files:
- tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
- tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
- tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py
- Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
- Docs/superpowers/specs/2026-06-17-vz-host-smoke-evidence-bundle-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add default-on structured evidence capture to the VZ Linux host smoke wrapper so local operator runs and the host-gated workflow retain concise, redacted proof of source/run bundle hashes, runtime paths, phase outcomes, and cleanup state without manual ad hoc commands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `run-host-e2e-smoke.sh` defaults evidence output to the private runtime directory under `evidence/`.
- [x] Operators can override evidence output with `--evidence-dir PATH`.
- [x] Evidence directory preflight refuses symlinks, non-directories, wrong-owner directories, and group/world-accessible directories; missing directories are created `0700`.
- [x] Dry-run prints the resolved evidence directory and planned evidence files without creating them.
- [x] Real/fake-helper runs write structured evidence files including source hashes, run hashes, runtime paths, phase outcomes, and cleanup state.
- [x] Trap/cleanup/finalization preserves the original smoke exit code.
- [x] Evidence JSON stores log pointers, sizes, and hashes only, not raw serial/helper log content.
- [x] Host-gated workflow/docs/tests expect the evidence bundle to be retained by the existing runtime artifact upload.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Baseline verification before edits: `python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` returned `45 passed, 6 warnings`.
- Design spec: `Docs/superpowers/specs/2026-06-17-vz-host-smoke-evidence-bundle-design.md`.
- Added implementation plan: `Docs/superpowers/plans/2026-06-17-vz-host-smoke-evidence-bundle.md`.
- RED wrapper verification: new evidence tests failed before implementation because `--evidence-dir` and evidence outputs were not implemented.
- RED host-gated policy verification: the policy contract test failed before docs mentioned `host-smoke-evidence.json`.
- Final focused verification: `python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` returned `51 passed, 6 warnings`.
- Shell syntax verification: `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` exited `0`.
- Whitespace verification: `git diff --check` exited `0`.
- Security verification: `python -m bandit -r tools/vz-linux-image/scripts -f json -o /tmp/bandit_vz_host_smoke_evidence.json` exited `0` with `0` results and `0` errors.
- Known skip: no real Apple Virtualization VM smoke was run for this local code slice; validation used the existing portable fake-helper wrapper tests plus the host-gated workflow contract tests.
- Review fixes: rebased on latest `origin/dev`; addressed reviewer findings for empty run-bundle hash roots, best-effort unreadable log metadata, direct `mkdir -p -m 700` evidence directory creation, validated `--python` fallback for evidence serialization, owner-only evidence files, evidence finalize phase accuracy, exact dry-run evidence paths, and new evidence-test docstrings.
- Review-fix RED verification: `python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q` failed before wrapper fixes on unreadable log metadata, empty run-bundle hashing, invalid PATH Python fallback, and exact evidence path assertions.
- Review-fix final focused verification: `python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` returned `53 passed, 6 warnings`.
- Review-fix shell syntax verification: `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` exited `0`.
- Review-fix whitespace verification: `git diff --check` exited `0`.
- Review-fix security verification: `python -m bandit -r tools/vz-linux-image/scripts -f json -o /tmp/bandit_vz_host_smoke_evidence_review.json` exited `0` with `0` results and `0` errors.
- CI note: PR check inspection found only canceled GitHub Actions runs for the old head SHA, with logs unavailable; no code failure was available to patch locally before pushing the review-fix commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented default-on VZ host smoke evidence capture under the runtime `evidence/` directory, added `--evidence-dir PATH` override support, hardened evidence directory permissions, and preserved smoke exit codes through cleanup/finalization. Evidence now records streamed SHA-256 bundle hashes, runtime paths, phase outcomes, cleanup state, and log path/size/hash metadata without embedding raw log contents. Host-gated acceptance docs and workflow contract tests now require the structured evidence bundle to be retained by the existing runtime artifact upload.
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
