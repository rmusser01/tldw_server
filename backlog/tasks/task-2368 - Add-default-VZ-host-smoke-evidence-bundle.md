---
id: TASK-2368
title: Add default VZ host smoke evidence bundle
status: In Progress
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
- [ ] `run-host-e2e-smoke.sh` defaults evidence output to the private runtime directory under `evidence/`.
- [ ] Operators can override evidence output with `--evidence-dir PATH`.
- [ ] Evidence directory preflight refuses symlinks, non-directories, wrong-owner directories, and group/world-accessible directories; missing directories are created `0700`.
- [ ] Dry-run prints the resolved evidence directory and planned evidence files without creating them.
- [ ] Real/fake-helper runs write structured evidence files including source hashes, run hashes, runtime paths, phase outcomes, and cleanup state.
- [ ] Trap/cleanup/finalization preserves the original smoke exit code.
- [ ] Evidence JSON stores log pointers, sizes, and hashes only, not raw serial/helper log content.
- [ ] Host-gated workflow/docs/tests expect the evidence bundle to be retained by the existing runtime artifact upload.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Baseline verification before edits: `python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` returned `45 passed, 6 warnings`.
- Design spec: `Docs/superpowers/specs/2026-06-17-vz-host-smoke-evidence-bundle-design.md`.
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
