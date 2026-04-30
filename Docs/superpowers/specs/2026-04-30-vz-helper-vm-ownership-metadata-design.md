# VZ Helper VM Ownership Metadata Design

**Date:** 2026-04-30
**Status:** Draft for implementation planning
**Scope:** `tools/macos-vz-helper/`, `tldw_Server_API/app/core/Sandbox/`, sandbox tests, and macOS operator docs

## Summary

The next `vz_linux` hardening slice should make helper VM identity explicit before adding more lifecycle automation. The merged orphan-VM repair path can terminate helper VMs when an operator passes `terminate_orphaned_vms=true`, but the helper's live VM records currently expose only `vm_id`, `state`, and `healthy`. That is not enough to distinguish a VM created by this sandbox service from a VM created by another helper user, a stale helper process, or a race during session persistence.

This PR should extend helper VM records with first-party ownership metadata, surface that metadata through the helper protocol and Python client, and tighten reconciliation repair so only owned orphan VMs are eligible for termination. Unknown or foreign orphan VMs should remain report-only.

## Source Documents

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/superpowers/specs/2026-04-27-vz-linux-lifecycle-recovery-hardening-design.md`
- `Docs/superpowers/specs/2026-04-29-vz-helper-lifecycle-hardening-design.md`
- `Docs/superpowers/plans/2026-04-30-vz-orphan-vm-repair-implementation-plan.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `tools/macos-vz-helper/PROTOCOL.md`

The sandbox doctrine keeps Python as the trusted control plane and the helper as the owner of live VM state. This design preserves that boundary: Python supplies identity metadata when creating VMs, and the helper preserves and reports it as live runtime truth.

## Current State

The helper flow is:

- Python calls `MacOSVirtualizationHelperClient.create_vm()` with `vm_name`, `runtime`, `run_id`, `session_mode`, `workspace_path`, and the template source under the existing `template` key.
- `HelperService.createVM()` passes only `vmID`, `templatePath`, `workspacePath`, and readiness timeout to `VZLinuxVMManager`.
- `VMRegistry.upsert()` stores `VMRecord(vmID, state, healthy)`.
- `get_vm_status` and `list_vms` return `vm_id`, `state`, `healthy`, and generic transport details.
- Reconciliation treats any live helper VM whose id is not referenced by persisted session controls as `orphaned_vm`.
- Repair can terminate every reported orphan when explicitly requested.

That last step is too broad for long-term safety because orphan classification lacks ownership proof.

## Goals

1. Preserve helper VM ownership metadata at VM creation time.
2. Expose ownership metadata through `get_vm_status` and `list_vms`.
3. Parse and retain the metadata in Python helper models.
4. Classify orphan VMs by termination eligibility.
5. Allow repair to terminate only VMs proven to be owned by this sandbox control plane.
6. Keep unknown or foreign helper VMs visible but non-mutating.
7. Keep the protocol backward-compatible within protocol version `1` unless a true breaking field is required.

## Non-Goals

1. Persisting helper VM records across helper restarts.
2. Making the helper authoritative for sandbox sessions or run history.
3. Adding launchd install/uninstall behavior.
4. Adding APFS clone-backed provisioning.
5. Adding `vz_macos` real execution.
6. Terminating unknown or foreign orphan VMs.
7. Introducing cryptographic VM ownership attestation.

## Recommended Approach

Use additive helper metadata fields and conservative repair gating.

The helper should store a `VMOwnershipMetadata` value with each `VMRecord`. Python should provide the metadata during `create_vm`; if a caller omits it, the helper should mark the ownership as `unknown` rather than inventing ownership. `VMRegistry.upsert()` must preserve existing metadata across state transitions unless explicit replacement metadata is supplied, because `VZLinuxVMManager.createVM()` currently writes `booting` and then `running` records.

Reconciliation should classify live VMs as:

- `owned_orphaned_vm`: helper VM satisfies the exact ownership eligibility contract below
- `unknown_orphaned_vm`: helper VM lacks ownership metadata or has partial legacy metadata
- `foreign_orphaned_vm`: helper VM has ownership metadata that does not match the expected sandbox owner/runtime

Only `owned_orphaned_vm` is eligible for `terminate_orphaned_vms=true`. The existing `orphaned_vm` summary field can remain as a total for compatibility, but action items should carry the more precise status/reason.

### Ownership Eligibility Contract

A live helper VM is termination-eligible only when all of these conditions are true:

- `metadata.owner == "tldw"`
- `metadata.runtime == "vz_linux"`
- `metadata.run_id` is non-empty
- `metadata.created_at` is non-empty and parseable enough to display as helper-created metadata
- when `metadata.session_mode == true`, `metadata.session_id` is non-empty

Everything else must be non-eligible:

- missing metadata -> `unknown_orphaned_vm`
- missing `run_id` or `created_at` -> `unknown_orphaned_vm`
- session-mode VM with no `session_id` -> `unknown_orphaned_vm`
- owner/runtime mismatch -> `foreign_orphaned_vm`

The first implementation should not require `vm_id == run_id` because the helper protocol already allows callers to choose `vm_name`, but docs should note that the current `vz_linux` runner uses `run_id` as `vm_name`.

## Metadata Contract

### Helper Request Metadata

`create_vm` should accept these optional fields in the request object:

```json
{
  "owner": "tldw",
  "runtime": "vz_linux",
  "run_id": "run-123",
  "session_id": "session-456",
  "session_mode": true,
  "template": "/path/to/bundle",
  "workspace_path": "/path/to/workspace"
}
```

Rules:

- `owner` defaults to `unknown` when omitted.
- `runtime` defaults to `vz_linux` only for the existing `create_vm` operation, but reconciliation should still require the explicit returned value before terminating orphans.
- `run_id` should be the sandbox run id.
- `session_id` may be empty for ephemeral runs.
- `session_mode` should be true only when a sandbox session VM is intended for reuse.
- the helper should continue accepting the existing `template` key and may also accept `template_path` as an alias; response metadata should normalize this value as `template_path`.
- `template_path` and `workspace_path` should be preserved as strings but should not be used as authority for filesystem access decisions.
- `created_at` should be assigned by the helper at creation time in UTC ISO-8601 format.
- Python's `vz_linux` runner must add `owner="tldw"` and `session_id` to the existing `create_vm` request. It should keep sending `template` for compatibility with the current helper request parser.

### Helper Response Metadata

`get_vm_status` and `list_vms` should include explicit metadata fields or a nested `metadata` object. Prefer a nested object to keep the top-level status fields stable:

```json
{
  "protocol_version": "1",
  "helper_version": "0.1.0",
  "vm_id": "run-123",
  "state": "running",
  "healthy": true,
  "metadata": {
    "owner": "tldw",
    "runtime": "vz_linux",
    "run_id": "run-123",
    "session_id": "session-456",
    "session_mode": true,
    "template_path": "/path/to/bundle",
    "workspace_path": "/path/to/workspace",
    "created_at": "2026-04-30T18:00:00Z"
  },
  "details": {
    "transport": "vsock"
  }
}
```

Python should tolerate both the new nested `metadata` object and older helper responses with no metadata. Missing metadata must not be treated as owned.

## Reconciliation Behavior

Reconciliation should keep existing persisted-session checks intact:

- healthy persisted sessions remain `healthy`
- missing persisted VMs remain `stale_session`
- unhealthy persisted VMs remain `unhealthy_vm`
- active sessions remain `skipped_active_session`

For live VMs not referenced by persisted session-control rows:

- return `owned_orphaned_vm` when metadata proves `owner=tldw` and `runtime=vz_linux`
- return `foreign_orphaned_vm` when metadata exists but owner/runtime does not match
- return `unknown_orphaned_vm` when metadata is absent or incomplete

The report should include:

- `orphaned_vm_ids`: all orphan VM ids for compatibility
- `owned_orphaned_vm_ids`
- `unknown_orphaned_vm_ids`
- `foreign_orphaned_vm_ids`

Each item should carry a `termination_eligible` boolean and a reason string, such as `owned_orphan`, `unknown_ownership`, or `foreign_owner`.

Compatibility note: any existing code that counts `status == "orphaned_vm"` must be updated to count all three precise orphan statuses. The aggregate `orphaned_vm_ids` list should remain the complete set of owned, unknown, and foreign orphan VM ids.

## Repair Behavior

`terminate_orphaned_vms=true` should:

- plan termination actions only for `owned_orphaned_vm`
- execute termination only for `owned_orphaned_vm` when `dry_run=false`
- report unknown and foreign orphan VMs as skipped actions
- preserve helper-specific failure reasons from `terminate_vm`
- keep dry-run as the default

Recommended action examples:

```json
{
  "type": "terminate_orphaned_vm",
  "vm_id": "run-owned",
  "status": "planned",
  "reason": "owned_orphan",
  "termination_eligible": true
}
```

```json
{
  "type": "skip_orphaned_vm",
  "vm_id": "vm-unknown",
  "status": "skipped",
  "reason": "unknown_ownership",
  "termination_eligible": false
}
```

## Protocol Compatibility

This can remain an additive protocol-version-`1` change if:

- existing clients ignore unknown helper response fields
- Python parser treats missing metadata as unknown
- helper tests cover old response parsing behavior

If the Swift helper needs to reject malformed metadata, it should reject only invalid types that would make VM creation ambiguous. Missing metadata should be allowed for compatibility but should produce `owner=unknown`.

## Error Handling

Use explicit reason strings:

- `owned_orphan`
- `unknown_ownership`
- `foreign_owner`
- `vm_metadata_invalid`
- `macos_virtualization_helper_unavailable`
- `macos_virtualization_helper_protocol_mismatch`
- helper-specific `MacOSVirtualizationHelperFailure.error_code`

Repair should not turn metadata ambiguity into a hard failure. It should skip ambiguous orphan termination and include the reason in the response.

## Testing Strategy

### Swift Helper Tests

- `VMRegistry` stores and returns metadata.
- `VMRegistry.upsert()` preserves metadata when state/health is updated without replacement metadata.
- `HelperService.createVM()` records metadata from request fields.
- `UnixSocketServer` forwards `owner`, `runtime`, `run_id`, `session_id`, `session_mode`, `template`, and `workspace_path` into `HelperService.createVM()`.
- `get_vm_status` includes metadata for created VMs.
- `list_vms` includes metadata for every record.
- missing metadata defaults to unknown ownership.

### Python Helper Model Tests

- parser reads nested VM metadata.
- parser tolerates missing metadata.
- parser coerces malformed metadata to empty or unknown values without marking ownership as trusted.

### Reconciliation Tests

- owned orphan is classified as `owned_orphaned_vm` and termination eligible.
- unknown orphan is classified as `unknown_orphaned_vm` and not termination eligible.
- foreign orphan is classified as `foreign_orphaned_vm` and not termination eligible.
- partial tldw metadata missing `run_id`, `created_at`, or session `session_id` is not termination eligible.
- existing healthy/stale/unhealthy persisted-session behavior is unchanged.
- orphan summary counts include owned, unknown, and foreign orphan statuses.

### Repair Tests

- dry-run plans termination for owned orphan VMs only.
- mutating repair calls `terminate_vm` for owned orphan VMs only.
- unknown and foreign orphan VMs produce skipped actions.
- helper termination errors preserve specific reason codes.

### Docs Tests

No dedicated docs test is required, but docs should be updated with the new ownership gate and reviewed via `git diff --check`.

## Design Review

### Issue: Metadata Can Be Spoofed By A Direct Helper Caller

The helper currently trusts local socket clients. Metadata proves that a VM was created through the helper with tldw-shaped metadata, not that the caller was authenticated. This is acceptable for this slice only because the lifecycle hardening work already makes the helper socket owner-private. The docs should avoid calling this cryptographic proof.

### Issue: Existing Legacy VMs Will Lack Metadata

Legacy helper VMs created before this PR should be reported as `unknown_orphaned_vm` and skipped by repair. Operators can still inspect and terminate them manually through lower-level tools if needed.

### Issue: Session Persist Race Could Temporarily Look Like An Orphan

A newly created session VM could appear in `list_vms()` before Python persists session-control metadata. Classification as owned orphan should not imply automatic termination. Since repair is explicit and dry-run-first, this is acceptable. Docs should tell operators to re-run diagnostics before mutating repair if a run is in progress.

### Issue: Template And Workspace Paths Are Sensitive

Returning full paths improves diagnostics but may expose local filesystem layout to admin API users. The existing diagnostics endpoint is admin-only. Still, docs and tests should avoid treating these paths as secrets, and future API redaction can be added if admin surface scope changes.

### Issue: Protocol Version Drift

Because this is additive, keeping protocol version `1` is reasonable. If later clients require metadata for normal execution, that should become protocol version `2`. This slice only requires metadata for orphan termination eligibility.

## Success Criteria

- Helper VM records expose ownership metadata through status and list calls.
- Python reconciliation distinguishes owned, unknown, and foreign orphan VMs.
- Explicit orphan repair terminates only owned orphan VMs.
- Unknown and foreign orphan VMs remain visible and skipped.
- Existing session reuse and stale-session repair behavior remains unchanged.
- Portable Swift and Python tests cover the metadata contract.
- Operator docs describe the ownership gate and avoid overstating proof strength.
