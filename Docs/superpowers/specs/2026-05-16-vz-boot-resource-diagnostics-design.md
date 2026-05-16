# VZ Boot Log And Resource Diagnostics Design

## Goal

Close the remaining `vz_linux` diagnostics gap by making real helper-backed VMs
surface useful resource snapshot fields in admin diagnostics while preserving the
existing read-only boot/serial/helper log pointer behavior.

The existing Python diagnostics already report helper stdout/stderr pointers,
per-VM serial log pointers, guest readiness metadata, and allowlisted resource
snapshot fields. The gap is that the Swift helper currently does not populate
real VM resource fields in `create_vm`, `get_vm_status`, or `list_vms`, so real
operator diagnostics usually show an empty `resource_snapshot`.

## Current State

- `probe_vz_linux_observability()` reads only metadata: helper log file pointers,
  serial log file pointers, VM status, guest metadata, and allowlisted integer
  resource fields from helper `details`.
- The diagnostics endpoint does not read log contents and should stay
  side-effect-free.
- `tools/macos-vz-helper` creates serial log files through
  `TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR`.
- `HelperService.vmDetails(for:)` emits transport, network policy, guest-agent
  details, and helper generation metadata, but no resource snapshot fields.
- The helper has accurate configured VM resource facts at boot time:
  `VZVirtualMachineConfiguration.cpuCount` and `memorySize`.

## Design

### Resource Fields

Add helper-owned resource snapshot fields to VM status `details`:

- `cpu_count`: configured VCPU count for the VM.
- `memory_size_mb`: configured VM memory in MiB.
- `wall_time_sec`: elapsed seconds since the VM ownership metadata `created_at`
  timestamp, when that timestamp parses successfully.

These are intentionally not CPU utilization, process RSS, disk I/O, or network
I/O counters. Apple `Virtualization.framework` does not currently expose cheap,
stable per-VM utilization counters through the helper code path. Publishing
configured facts plus uptime is accurate and immediately useful for operators;
fake CPU/RSS values would create misleading diagnostics.

### Swift Helper Contract

- Introduce a small `VMResourceSnapshot` value in the helper protocol/model
  layer.
- Change `VZBootDriving.boot(...)` to return `VMResourceSnapshot`.
- Have `VirtualizationLinuxBootDriver.boot(...)` derive the snapshot from the
  validated `VZVirtualMachineConfiguration` it already builds.
- Store the snapshot in `VMRecord` through `VMRegistry.upsert(...)`.
- Preserve an existing snapshot on state-only updates unless a new snapshot is
  supplied, mirroring the existing guest-info preservation behavior.
- Emit resource fields from `HelperService.vmDetails(for:)` for create/status/list.
- Compute `wall_time_sec` at detail-render time from `record.metadata.createdAt`;
  omit it if the timestamp is missing or invalid.

### Python Diagnostics Contract

- Extend `_VZ_LINUX_RESOURCE_DETAIL_KEYS` to include `cpu_count` and
  `memory_size_mb`.
- Keep `_resource_snapshot()` integer-only and allowlist-only.
- Do not add filesystem reads beyond existing file existence/size checks.
- Do not change the public diagnostics shape; this remains an additive field
  population in the existing `resource_snapshot` dictionary.

### Documentation

- Update `tools/macos-vz-helper/PROTOCOL.md` so `list_vms` examples show the new
  resource fields.
- Update sandbox operator notes to clarify that current resource snapshots are
  configured VM resources plus uptime, not live utilization telemetry.

## Design Review

### Issue: misleading utilization counters

The obvious but wrong improvement would be to populate `cpu_time_sec`,
`peak_rss_mb`, or I/O counters from helper-process data. That would be
misleading because the helper process can manage multiple VMs and does not own a
clear per-VM RSS/CPU counter. This design avoids those fields until a real
per-VM source exists.

### Issue: boot driver protocol churn

Changing `VZBootDriving.boot(...)` affects tests and fake drivers. The change is
still the cleanest boundary because the boot driver is the component that builds
the authoritative VM configuration. Returning a tiny snapshot avoids teaching
`HelperService` or `VMRegistry` about `Virtualization.framework`.

### Issue: uptime depends on wall clock

`wall_time_sec` is diagnostic metadata, not scheduling logic. It should be
non-negative, omitted on parse failure, and tested with a deterministic old
timestamp rather than exact current-time assertions.

### Issue: existing log pointer behavior is already implemented

This PR should not rewrite serial/helper log discovery. It should preserve the
existing read-only pointer contract and only tighten documentation/tests where
needed.

## Tests

- Swift helper unit tests:
  - `create_vm`, `get_vm_status`, and `list_vms` include `cpu_count`,
    `memory_size_mb`, and `wall_time_sec` when the boot driver returns a
    snapshot and metadata has a valid `created_at`.
  - state updates preserve the resource snapshot.
  - invalid or missing `created_at` omits `wall_time_sec`.
- Python diagnostics tests:
  - `_resource_snapshot()`/`probe_vz_linux_observability()` surfaces
    `cpu_count` and `memory_size_mb`.
  - unknown helper details remain filtered.
- Verification:
  - focused Python diagnostics tests
  - focused Swift helper tests if Swift tooling is available
  - `git diff --check`
  - Bandit on touched Python files

## Non-Goals

- Reading or returning log contents.
- Adding live CPU/RSS/I/O utilization telemetry.
- Changing helper lifecycle, launchd behavior, image-store behavior, or repair
  semantics.
- Requiring real Apple VZ execution for portable unit coverage.
