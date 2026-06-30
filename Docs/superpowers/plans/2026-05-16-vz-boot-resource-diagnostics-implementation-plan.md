# VZ Boot Resource Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real `vz_linux` helper VMs populate accurate resource snapshot fields in existing admin diagnostics.

**Architecture:** Keep diagnostics read-only and additive. The Swift helper owns VM configuration facts, stores them on the VM registry record, and emits them through existing status/list `details`; Python diagnostics continues to project only allowlisted integer details into `resource_snapshot`.

**Tech Stack:** Swift 5.9 helper package, Python diagnostics module, pytest, Swift Testing, Backlog.md task `TASK-404`.

---

## Scope Review

The existing Python diagnostics already expose helper log pointers, serial log
pointers, guest readiness metadata, and an allowlisted `resource_snapshot`.
This plan does not rewrite log discovery. It only makes the helper populate
accurate configured resource facts and lets Python surface the new allowlisted
keys.

## Files

- Modify: `tools/macos-vz-helper/Sources/Protocol/Response.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VMRegistry.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VZLinuxVMManager.swift`
- Modify: `tools/macos-vz-helper/Sources/VM/VirtualizationLinuxBootDriver.swift`
- Modify: `tools/macos-vz-helper/Sources/Server/HelperService.swift`
- Modify: `tools/macos-vz-helper/Tests/TestDoubles.swift`
- Modify: `tools/macos-vz-helper/Tests/HelperServiceVMTests.swift`
- Modify: `tools/macos-vz-helper/Tests/VMBootTests.swift`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Modify: `tools/macos-vz-helper/PROTOCOL.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `backlog/tasks/task-404 - Add-VZ-boot-log-and-resource-diagnostics.md`

## Task 1: Swift Helper Resource Snapshot Tests

- [x] Add a failing test in `HelperServiceVMTests.swift` proving create/status/list details include `cpu_count`, `memory_size_mb`, and non-negative `wall_time_sec`.
- [x] Add a failing test in `VMBootTests.swift` proving the VM registry preserves a boot driver resource snapshot through the running-state update.
- [x] Run focused Swift tests and confirm they fail because resource snapshot support does not exist.

Expected command:

```bash
cd tools/macos-vz-helper && swift test --filter 'HelperService|VMBoot'
```

## Task 2: Swift Helper Resource Snapshot Implementation

- [x] Add `VMResourceSnapshot` with `cpuCount` and `memorySizeBytes`.
- [x] Change `VZBootDriving.boot(...)` to return `VMResourceSnapshot`.
- [x] Make `RecordingBootDriver` return configurable snapshots for tests.
- [x] Store snapshots on `VMRecord` and preserve them through registry updates unless a new snapshot is supplied.
- [x] Make `VirtualizationLinuxBootDriver.boot(...)` return the built configuration's CPU count and memory size.
- [x] Emit `cpu_count`, `memory_size_mb`, and `wall_time_sec` from `HelperService.vmDetails(for:)`.
- [x] Re-run focused Swift tests and confirm they pass.

## Task 3: Python Diagnostics Allowlist Tests And Implementation

- [x] Extend the existing `test_probe_vz_linux_observability_reports_log_pointers_and_vm_resources` fixture with `cpu_count` and `memory_size_mb`.
- [x] Run the focused pytest and confirm it fails because the new keys are filtered.
- [x] Add `cpu_count` and `memory_size_mb` to `_VZ_LINUX_RESOURCE_DETAIL_KEYS`.
- [x] Re-run the focused pytest and confirm it passes.

Expected command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py::test_probe_vz_linux_observability_reports_log_pointers_and_vm_resources -q
```

## Task 4: Docs And Tracker

- [x] Update `tools/macos-vz-helper/PROTOCOL.md` list/status examples and stability notes for the new resource detail fields.
- [x] Update `Docs/Sandbox/macos-runtime-operator-notes.md` to clarify that current resource snapshots are configured VM facts plus uptime, not live utilization telemetry.
- [x] Update `TASK-404` implementation notes with touched files and test results.

## Task 5: Verification And Commit

- [x] Run focused Swift helper tests.
- [x] Run focused Python diagnostics tests.
- [x] Run `git diff --check`.
- [x] Run Bandit against touched Python diagnostics code.
- [x] Commit the implementation.

Expected commands:

```bash
cd tools/macos-vz-helper && swift test --filter 'HelperService|VMBoot'
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py::test_probe_vz_linux_observability_reports_log_pointers_and_vm_resources -q
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/macos_diagnostics.py -f json -o /tmp/bandit_vz_boot_resource_diagnostics.json
```

## Plan Review

- Avoid fake telemetry: only emit configured resource facts and uptime.
- Keep diagnostics read-only: do not read log contents or mutate diagnostics state.
- Keep helper details string-encoded to preserve the existing protocol shape.
- Keep Python diagnostics permissive for mixed helper versions: absent new fields simply produce an empty or partial `resource_snapshot`.
