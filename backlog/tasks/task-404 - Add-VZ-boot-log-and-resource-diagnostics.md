---
id: TASK-404
title: Add VZ boot log and resource diagnostics
status: Done
labels:
- sandbox
- macos
- vz-linux
- diagnostics
priority: medium
documentation:
- Docs/superpowers/specs/2026-05-16-vz-boot-resource-diagnostics-design.md
implementation_plan:
- Docs/superpowers/plans/2026-05-16-vz-boot-resource-diagnostics-implementation-plan.md
modified_files:
- Docs/Sandbox/macos-runtime-operator-notes.md
- tldw_Server_API/app/core/Sandbox/README.md
- tldw_Server_API/app/core/Sandbox/macos_diagnostics.py
- tldw_Server_API/tests/sandbox/test_macos_diagnostics.py
- tools/macos-vz-helper/PROTOCOL.md
- tools/macos-vz-helper/Sources/Protocol/Response.swift
- tools/macos-vz-helper/Sources/Server/HelperService.swift
- tools/macos-vz-helper/Sources/VM/VMRegistry.swift
- tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift
- tools/macos-vz-helper/Sources/VM/VirtualizationLinuxBootDriver.swift
- tools/macos-vz-helper/Tests/HelperServiceVMTests.swift
- tools/macos-vz-helper/Tests/TestDoubles.swift
- tools/macos-vz-helper/Tests/VMBootTests.swift
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the next sandbox roadmap slice for VZ Linux admin diagnostics: stable serial/boot log pointers, bounded helper log metadata, and resource snapshots without reading log contents or mutating diagnostics state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review the current diagnostics/helper contracts and document a focused design with risks and mitigations.
- [x] #2 Expose stable VZ Linux boot/serial/helper log metadata in admin diagnostics without returning log contents.
- [x] #3 Expose allowlisted resource snapshot fields when helper metadata provides them, with deterministic unavailable/unknown states when absent.
- [x] #4 Add focused portable tests for diagnostics behavior and schema stability.
- [x] #5 Update operator docs and record verification including Bandit for touched Python code when applicable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec created for the narrowed diagnostics gap: keep existing read-only boot/helper/serial log pointers, add accurate helper-owned resource snapshot fields (cpu_count, memory_size_mb, wall_time_sec), and explicitly reject fake CPU/RSS/I/O telemetry until real per-VM counters exist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented helper-owned VM resource snapshots for `vz_linux` status/list/create responses. The helper now stores configured CPU count and memory size from the validated `Virtualization.framework` configuration, preserves that snapshot across registry state updates, and emits `cpu_count`, `memory_size_mb`, and diagnostic `wall_time_sec` in existing string-encoded helper details. Python diagnostics allowlists `cpu_count` and `memory_size_mb` into the existing read-only `resource_snapshot` block. Docs clarify these are configured VM facts plus uptime, not live CPU/RSS/I/O utilization telemetry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: focused Python red test failed before allowlist implementation, then `test_macos_diagnostics.py` passed with 26 passed and 2 warnings. Focused Swift red test failed before helper implementation, then full `tools/macos-vz-helper` Swift tests passed with 88 tests. `git diff --check` passed. Bandit on `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py` wrote `/tmp/bandit_vz_boot_resource_diagnostics.json` with errors=[] and results=[].
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
