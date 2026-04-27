# vz_linux Lifecycle And Recovery Hardening Design

**Date:** 2026-04-27
**Status:** Approved for implementation planning
**Scope:** `tldw_Server_API/app/core/Sandbox/`, sandbox admin API schemas/endpoints, macOS sandbox docs, and helper protocol compatibility handling

## Summary

This design hardens the `vz_linux` runtime lifecycle after the real execution, vsock transport, operator smoke, and image-store foundations have landed.

The next slice should make session reuse and helper failure behavior deterministic before expanding into APFS clone provisioning, launchd installation, hosted Apple Silicon CI, or `vz_macos` execution.

The PR should add:

- a read-only reconciliation report for persisted `sandbox_vz_sessions` versus live helper VM facts
- an explicit admin repair path for stale persisted session-control rows
- fail-closed helper protocol compatibility handling
- clearer session-reuse behavior for helper unavailable, protocol mismatch, missing VM, and unhealthy VM cases
- updated docs that reflect the current real VZ boot and real vsock implementation

## Source Documents

This design follows the durable sandbox doctrine:

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/Plans/2026-03-10-vz-linux-helper-stability-design.md`
- `Docs/Plans/2026-03-10-vz-linux-real-execution-design.md`
- `Docs/Plans/2026-03-11-vz-linux-vsock-transport-design.md`
- `Docs/Design/2026-04-27-vz-linux-operator-image-store-design.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `tldw_Server_API/app/core/Sandbox/README.md`

The active doctrine requires:

- Python owns policy, sessions, runs, artifacts, queueing, and API behavior.
- The native helper owns live runtime availability, VM state, guest transport health, and reconciliation facts.
- Diagnostics must reuse runtime truth.
- Session reuse must check live health before trusting persisted metadata.
- Cleanup must tolerate already-gone runtime state.

## Current State

`vz_linux` now has a real helper-backed execution path on prepared Apple silicon macOS hosts:

- Python talks to the helper over a Unix socket through `MacOSVirtualizationHelperClient`.
- The Swift helper owns `Virtualization.framework` boot, VM registry state, vsock session management, and guest exec bridging.
- The guest-side `tldw-agent` connects over vsock and serves structured exec requests.
- `VZLinuxRunner` supports ephemeral execution and same-session VM reuse.
- `SandboxImageStore` has a filesystem-backed manifest path.
- `run-host-e2e-smoke.sh` exercises helper build/sign/start, template validation, ephemeral execution, same-session reuse, and shutdown.

The remaining lifecycle gaps are:

- Reconciliation is currently a diagnostics helper shape, not a reusable lifecycle service concept.
- Existing diagnostics report stale persisted rows and orphaned helper VMs, but do not expose enough structured outcome categories for operator action.
- There is no explicit admin repair operation for stale persisted session-control rows.
- Helper protocol mismatch is raised by the client, but diagnostics and runtime lifecycle handling do not consistently classify it as a compatibility failure.
- Orphaned helper VM termination is unsafe with the current helper metadata, because `list_vms()` does not expose `run_id`, `session_id`, `runtime`, creation time, or ownership.
- Two docs still incorrectly say the VZ boot driver and real vsock binding are incomplete.

## Goals

1. Make persisted `vz_linux` session-control metadata recoverable after helper restarts, API restarts, helper crashes, and stale rows.
2. Keep diagnostics read-only and safe to call repeatedly.
3. Add an explicit admin repair path for stale persisted session rows.
4. Treat helper protocol mismatch as a fail-closed compatibility state with clear diagnostics.
5. Avoid terminating helper VMs that Python cannot prove it owns.
6. Preserve the Python/helper ownership boundary.
7. Update docs so future runtime plans start from current reality.

## Non-Goals

1. Full launchd install/uninstall support.
2. Managed helper auto-upgrade.
3. APFS clone-backed provisioning.
4. Apple Silicon host-gated GitHub Actions runner wiring.
5. `vz_macos` real execution.
6. Allowlist networking.
7. Orphan VM termination without extending helper VM metadata.
8. Making the helper a persistence layer for sandbox sessions.

## Recommended Approach

Implement Python-side lifecycle and recovery hardening first.

This is the pragmatic next step because it uses the helper truth that already exists, keeps the PR reviewable, and reduces the risk of future provisioning work compounding lifecycle ambiguity.

Alternatives considered:

- Full launchd/service management now. This has higher operator value, but it is too broad before the recovery semantics are stable.
- Minimal stale-row deletion only. This is safer, but leaves helper compatibility, diagnostics clarity, and documentation drift unresolved.
- APFS cloning next. This improves speed, but it does not address stale session state or helper crash behavior.

## Architecture

### Read-Only Reconciliation

Add a reusable reconciliation report function under the sandbox core. It should compare:

- persisted rows from `SandboxOrchestrator.list_vz_session_controls()`
- live helper VM records from `MacOSVirtualizationHelperClient.list_vms()`
- active queued, starting, or running session work from the sandbox service/orchestrator

The report should be machine-readable and include outcome categories:

- `healthy`: persisted session has a live, healthy helper VM
- `stale_session`: persisted session references a missing live VM
- `unhealthy_vm`: persisted session references a live VM that is not healthy
- `orphaned_vm`: helper reports a VM not referenced by persisted session controls
- `skipped_active_session`: row is stale or unhealthy but repair must skip because the session has active work
- `helper_unavailable`: helper cannot be reached
- `protocol_mismatch`: helper responded with an incompatible protocol version
- `reconciliation_unavailable`: local store/orchestrator support is unavailable

Diagnostics should call this report path and return the report. Diagnostics must not delete rows or terminate VMs.

### Explicit Repair

Add an admin-only repair operation that applies a conservative subset of the reconciliation report.

Repair should:

- require helper reachability
- require protocol compatibility
- delete stale persisted session-control rows only when the session has no active queued, starting, or running runs
- delete unhealthy persisted session-control rows only when the helper confirms the VM is not healthy and the session has no active runs
- skip active sessions and report them as skipped
- never terminate orphaned VMs in this PR
- tolerate rows already deleted by concurrent cleanup
- return a structured repair summary

Repair should not try to make the helper state authoritative for product sessions. It only repairs Python-owned session-control metadata that is provably stale against helper-owned runtime truth.

### Orphan VM Policy

Orphan VMs should remain report-only in this slice.

The current helper `list_vms()` response does not expose enough ownership metadata to safely distinguish:

- an active ephemeral run VM
- a VM created by another local process
- a VM from a previous helper lifecycle
- a legitimate session VM that Python has not persisted yet because of a race

Future orphan termination should first extend helper VM metadata to include at least `run_id`, `session_id`, `runtime`, `created_at`, `session_mode`, and possibly workspace/template identifiers.

### Helper Compatibility

Centralize helper compatibility classification.

The Python client already rejects mismatched protocol versions. This slice should make that failure consistently visible as `protocol_mismatch` in diagnostics and reconciliation instead of collapsing it into generic helper unavailability.

Runtime behavior should fail closed:

- no fallback to fake execution
- no deletion of persisted metadata when the helper is unavailable
- no deletion of persisted metadata when the helper protocol is incompatible
- explicit remediation in diagnostics to update the helper or Python client together

### Session Reuse

`VZLinuxRunner` already checks `get_vm_status()` before reusing a persisted VM. This slice should tighten the outcomes:

- healthy VM: reuse
- missing VM: delete stale row and create a fresh VM for the session run
- unhealthy VM: delete stale row and create a fresh VM for the session run
- helper unavailable: fail the run clearly without deleting metadata
- protocol mismatch: fail the run clearly without deleting metadata

This keeps reuse optimistic only after the helper confirms live health.

### Startup Behavior

Startup should remain non-destructive.

The service may compute and log a reconciliation report during startup if a helper is configured, but it should not mutate session metadata or terminate VMs automatically. Operators should use the explicit repair path after reviewing diagnostics.

This avoids surprising cleanup on hosts where the helper starts after the API service or where a protocol mismatch is temporarily caused by deployment ordering.

## API Shape

The existing endpoint remains read-only:

```text
GET /api/v1/sandbox/admin/macos-diagnostics
```

It should keep returning host, helper, template, runtime, and reconciliation data.

Add a new admin endpoint for explicit repair:

```text
POST /api/v1/sandbox/admin/macos-reconciliation/repair
```

Recommended request shape:

```json
{
  "delete_stale_session_controls": true,
  "delete_unhealthy_session_controls": true,
  "terminate_orphaned_vms": false,
  "dry_run": true
}
```

Rules:

- `dry_run` should default to `true`.
- `terminate_orphaned_vms=true` should be rejected in this PR with `orphan_termination_not_supported`.
- actual mutation requires `dry_run=false`.
- admin role is required.

Recommended response shape:

```json
{
  "dry_run": true,
  "helper": {
    "ready": true,
    "protocol_version": "1",
    "helper_version": "0.1.0"
  },
  "summary": {
    "stale_session_controls": 1,
    "unhealthy_session_controls": 0,
    "deleted_session_controls": 0,
    "skipped_active_sessions": 0,
    "orphaned_vms": 1,
    "terminated_orphaned_vms": 0
  },
  "actions": [
    {
      "type": "delete_session_control",
      "session_id": "session-1",
      "vm_id": "vm-missing",
      "status": "planned",
      "reason": "stale_session"
    }
  ],
  "reasons": []
}
```

## Error Handling

Use explicit reason strings:

- `macos_virtualization_helper_unavailable`
- `macos_virtualization_helper_protocol_mismatch`
- `vz_reconciliation_unavailable`
- `vz_session_active_runs_present`
- `orphan_termination_not_supported`
- `vz_session_control_delete_failed`

Read-only diagnostics should report failures as data and still return HTTP 200 when possible.

The mutating repair endpoint should use 4xx only for invalid requests or unsupported options, and 503 for helper unavailable or protocol mismatch when mutation was requested.

## Observability And Audit

Emit structured logs for:

- reconciliation report computed
- stale session row detected
- unhealthy session row detected
- active session skipped
- repair dry-run planned
- session-control row deleted
- repair failure

The repair endpoint should also emit an admin/audit event if the existing audit pattern for sandbox admin endpoints supports it. If there is no straightforward existing audit hook in this endpoint cluster, structured Loguru events are the minimum for this PR, and audit integration can follow separately.

## Testing Strategy

### Unit Tests

- reconciliation reports healthy persisted sessions
- reconciliation reports stale persisted sessions
- reconciliation reports unhealthy persisted sessions
- reconciliation reports orphaned helper VMs without planning termination
- helper unavailable leaves `computed=false` or an explicit helper-unavailable reason
- protocol mismatch is classified distinctly
- active sessions are marked `skipped_active_session`

### Service/API Tests

- diagnostics endpoint remains read-only
- repair endpoint defaults to dry-run
- repair endpoint deletes stale rows only with `dry_run=false`
- repair endpoint skips sessions with active runs
- repair endpoint rejects orphan termination
- repair endpoint is admin-only

### Runner Tests

- session reuse fails closed on helper unavailable without deleting control metadata
- session reuse fails closed on protocol mismatch without deleting control metadata
- missing or unhealthy session VM causes the control row to be deleted and a new VM to be created

### Host-Gated Tests

No new host-gated VZ test is required for the first lifecycle hardening PR. Existing smoke coverage remains:

- helper daemon host-gated smoke
- real `vz_linux` host E2E smoke with ephemeral execution and same-session reuse

## Documentation Updates

Update:

- `tldw_Server_API/app/core/Sandbox/README.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- optionally `tools/macos-vz-helper/README.md`

Required doc corrections:

- remove stale claims that real `Virtualization.framework` boot and real vsock binding are incomplete
- describe diagnostics as read-only
- describe explicit reconciliation repair
- clarify that orphan VM termination is deferred pending richer helper metadata
- keep launchd, auto-upgrade, APFS clone execution, and host-gated CI listed as future work

## Design Review Notes

The initial design was tightened in four ways before implementation planning:

1. Diagnostics must remain non-mutating.
2. Orphan VM termination must be deferred because helper VM records lack ownership metadata.
3. Repair must skip sessions with active queued, starting, or running work.
4. Helper unavailability and protocol mismatch must not delete persisted metadata.

These constraints prevent the recovery work from creating a more dangerous cleanup path than the stale state it is meant to repair.

## Open Follow-Up Work

- Add helper VM metadata for safe orphan ownership decisions.
- Add launchd install/uninstall and managed helper lifecycle commands.
- Add helper auto-upgrade/version compatibility policy.
- Add APFS clone-backed provisioning behind the stable image-store/helper contract.
- Add Apple Silicon host-gated CI/nightly workflow.
- Add stricter network policy enforcement once a provable allowlist mechanism exists.
