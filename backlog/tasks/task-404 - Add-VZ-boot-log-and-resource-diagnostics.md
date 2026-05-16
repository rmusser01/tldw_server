---
id: TASK-404
title: Add VZ boot log and resource diagnostics
status: In Progress
labels:
- sandbox
- macos
- vz-linux
- diagnostics
priority: medium
documentation:
- Docs/superpowers/specs/2026-05-16-vz-boot-resource-diagnostics-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the next sandbox roadmap slice for VZ Linux admin diagnostics: stable serial/boot log pointers, bounded helper log metadata, and resource snapshots without reading log contents or mutating diagnostics state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review the current diagnostics/helper contracts and document a focused design with risks and mitigations.
- [ ] #2 Expose stable VZ Linux boot/serial/helper log metadata in admin diagnostics without returning log contents.
- [ ] #3 Expose allowlisted resource snapshot fields when helper metadata provides them, with deterministic unavailable/unknown states when absent.
- [ ] #4 Add focused portable tests for diagnostics behavior and schema stability.
- [ ] #5 Update operator docs and record verification including Bandit for touched Python code when applicable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec created for the narrowed diagnostics gap: keep existing read-only boot/helper/serial log pointers, add accurate helper-owned resource snapshot fields (cpu_count, memory_size_mb, wall_time_sec), and explicitly reject fake CPU/RSS/I/O telemetry until real per-VM counters exist.
<!-- SECTION:PLAN:END -->

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
