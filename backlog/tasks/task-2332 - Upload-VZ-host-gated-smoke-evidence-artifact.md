---
id: TASK-2332
title: Upload VZ host-gated smoke evidence artifact
status: In Review
labels:
- sandbox
- vz_linux
- host-gated
- ci
priority: Medium
documentation:
- Docs/superpowers/specs/2026-06-17-vz-host-gated-evidence-artifact-design.md
- Docs/superpowers/plans/2026-06-17-vz-host-gated-evidence-artifact-plan.md
references:
- https://github.com/rmusser01/tldw_server/pull/2382
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the structured evidence bundle emitted by the VZ Linux host smoke wrapper first-class in host-gated CI by passing an explicit evidence directory, uploading it as a separate artifact, and documenting the operator expectation. Keep VM/helper behavior unchanged and preserve raw helper logs for fallback debugging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Host-gated workflow passes an explicit --evidence-dir under the private runtime directory to run-host-e2e-smoke.sh.
- [x] #2 Host-gated workflow uploads the evidence directory as a separate always-run artifact without replacing helper log upload.
- [x] #3 Operator docs/policy identify the structured evidence artifact as the primary artifact to inspect and keep raw logs as fallback debugging.
- [x] #4 Focused tests cover workflow wiring and documentation expectations without requiring a real VZ host.
- [x] #5 Verification includes focused pytest, shell syntax check, git diff check, and Bandit where Python files are touched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create an isolated worktree from dev. 2. Write the approved short design spec. 3. Add failing workflow/doc tests for explicit evidence artifact upload. 4. Update the workflow and docs minimally. 5. Run focused verification and commit. 6. Push and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created approved design spec for the host-gated evidence artifact slice. Subagent spec review was not spawned because the available multi-agent tool policy requires explicit user authorization for subagents in this turn; performing local spec risk review instead.
Spec review found that preserving the broad runtime-tree helper-log upload would now risk uploading disposable image-store/rootfs clones after the image-store smoke bundle work. Updated the spec so the helper-log artifact remains but is narrowed to raw serial/helper logs while structured evidence becomes the primary artifact.
Implemented workflow/docs/tests. RED verification: `python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` failed with 3 expected failures for missing explicit evidence dir, missing separate artifact upload, and missing policy artifact guidance. GREEN verification: focused workflow contract passed with 21 tests. Broader focused verification passed with 49 tests across host-gated workflow and smoke wrapper contracts, plus `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` and `git diff --check`. Bandit on the touched workflow test file with B101 excluded reported 0 findings and 0 errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Host-gated VZ Linux CI now passes an explicit smoke evidence directory, uploads `vz-linux-host-gated-evidence` as a separate always-run artifact, and narrows `vz-linux-host-gated-helper-logs` to serial/helper logs so disposable image-store/rootfs clones are not uploaded through the raw-log artifact. Operator docs and policy now direct maintainers to inspect structured evidence first and raw logs only as fallback debugging.

PR: https://github.com/rmusser01/tldw_server/pull/2382
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
