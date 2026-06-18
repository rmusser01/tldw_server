---
id: TASK-2381
title: Add advisory VZ host smoke evidence summary
status: In Progress
labels:
- sandbox
- vz_linux
- host-gated
- ci
- diagnostics
priority: Medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an advisory-first host-gated evidence summary path for VZ Linux smoke runs. The summary should read structured smoke evidence when available, write operator-friendly GitHub step-summary output, and never mask the primary smoke result in this first slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design spec captures advisory-only summary behavior, malformed/missing evidence handling, and non-goals before implementation.
- [ ] #2 The host-gated workflow can run an always-run advisory evidence summary step after smoke/evidence generation.
- [ ] #3 The summary reports evidence present/missing, required file presence, phase outcomes, final exit code, cleanup status, and artifact/log pointers when available.
- [ ] #4 Malformed or missing evidence produces warnings and exit 0 in this first advisory slice.
- [ ] #5 Focused tests cover complete evidence, missing evidence, malformed JSON, and workflow wiring without requiring a real VZ host.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and commit the approved design spec. 2. Review the spec for issues before implementation planning. 3. After approval, create an implementation plan and implement TDD-first in a separate commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Design spec drafted at `Docs/superpowers/specs/2026-06-17-vz-host-gated-evidence-summary-advisory-design.md`.
- Local design review tightened the read-only contract: append-only GitHub step summary output, no evidence mutation, direct-child-only probes, symlink/non-regular-file skips, and bounded JSON reads.
- Subagent spec review was not spawned because the available multi-agent tool requires explicit user authorization for delegation.
- Planning verification: `git diff --check`; Bandit not run for the design-only docs/backlog commit.
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
