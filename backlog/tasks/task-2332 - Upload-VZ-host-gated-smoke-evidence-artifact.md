---
id: TASK-2332
title: Upload VZ host-gated smoke evidence artifact
status: In Progress
labels:
- sandbox
- vz_linux
- host-gated
- ci
priority: Medium
documentation:
- Docs/superpowers/specs/2026-06-17-vz-host-gated-evidence-artifact-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the structured evidence bundle emitted by the VZ Linux host smoke wrapper first-class in host-gated CI by passing an explicit evidence directory, uploading it as a separate artifact, and documenting the operator expectation. Keep VM/helper behavior unchanged and preserve raw helper logs for fallback debugging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Host-gated workflow passes an explicit --evidence-dir under the private runtime directory to run-host-e2e-smoke.sh.
- [ ] #2 Host-gated workflow uploads the evidence directory as a separate always-run artifact without replacing helper log upload.
- [ ] #3 Operator docs/policy identify the structured evidence artifact as the primary artifact to inspect and keep raw logs as fallback debugging.
- [ ] #4 Focused tests cover workflow wiring and documentation expectations without requiring a real VZ host.
- [ ] #5 Verification includes focused pytest, shell syntax check, git diff check, and Bandit where Python files are touched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create an isolated worktree from dev. 2. Write the approved short design spec. 3. Add failing workflow/doc tests for explicit evidence artifact upload. 4. Update the workflow and docs minimally. 5. Run focused verification and commit. 6. Push and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created approved design spec for the host-gated evidence artifact slice. Subagent spec review was not spawned because the available multi-agent tool policy requires explicit user authorization for subagents in this turn; performing local spec risk review instead.
Spec review found that preserving the broad runtime-tree helper-log upload would now risk uploading disposable image-store/rootfs clones after the image-store smoke bundle work. Updated the spec so the helper-log artifact remains but is narrowed to raw serial/helper logs while structured evidence becomes the primary artifact.
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
