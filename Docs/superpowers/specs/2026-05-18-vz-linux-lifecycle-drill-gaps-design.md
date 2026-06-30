# VZ Linux Lifecycle Drill Gaps Design

**Date:** 2026-05-18
**Status:** Approved design; implementation planning pending
**Backlog:** TASK-432
**Scope:** Remaining `vz_linux` lifecycle drill gaps after prepared-host evidence tracking, stale VM recovery, helper restart recovery, launchd-drill, and host-gated CI policy work.

## Summary

The `vz_linux` Apple silicon path now has real helper-backed execution,
same-session VM reuse, recovery diagnostics, dry-run repair planning,
manual failure drills for stale VM state, a managed helper `restart-drill`, and
a prepared-host evidence tracker. The remaining lifecycle gaps are narrower:

- stale socket handling evidence
- stuck boot and stuck guest-readiness behavior
- guest-agent mismatch behavior
- host reboot recovery boundaries

This slice should define the contract before implementation. The goal is not to
add more automation immediately. The goal is to make each future drill small,
safe, operator-owned, and reviewable.

## Current Baseline

- `vz-helperctl.py check`, `status`, `start`, `stop`, `restart-drill`,
  `launchd`, `launchd-drill`, and `smoke` already exist.
- Helper-side socket safety and wrapper-side path checks are documented and
tested: stale socket removal must be limited to real Unix sockets, never
symlinks or non-socket files.
- Host-gated CI is limited to trusted refs, manual dispatch, and opted-in
nightly runs.
- Failure drills are manual opt-in and currently cover drill-owned stale VM
replacement plus helper restart recovery.
- Host reboot validation remains a manual operator procedure and is explicitly
outside scheduled CI.
- Prepared-host evidence is tracked in
  `Docs/Sandbox/vz-linux-prepared-host-evidence.md`.

## Goals

- Define manual drill contracts for stale socket, stuck boot/readiness,
  guest-agent mismatch, and host reboot recovery.
- Keep each drill independently implementable as a later small PR.
- Preserve the existing manual/host-gated security posture.
- Distinguish safe host-independent simulation from real prepared-host
  destructive or slow behavior.
- Define evidence packet fields for pass/fail/skip results before adding more
  runtime code.

## Non-Goals

- Do not add PR or push triggers for real `vz_linux` execution.
- Do not enable scheduled destructive drills.
- Do not automate host reboot.
- Do not terminate broad helper or VM state.
- Do not make repair mutation the default.
- Do not introduce networking, `vz_macos` execution, APFS clones, or Apple
  `containerization` adapters.
- Do not require operators to run launchd unless they explicitly selected
  launchd validation.

## Approach Options

### Option A: Documentation Contract First

Define the drill contracts, safety rules, expected skips, evidence fields, and
implementation slices now. Implement each drill later as a narrow PR.

This is the recommended path. The remaining gaps touch risky lifecycle edges,
and a shared contract prevents future PRs from accidentally adding broad
automation or destructive cleanup.

### Option B: Implement All Manual Drills Now

Add stale socket, stuck boot/readiness, and guest-agent mismatch drills in one
PR.

This moves faster, but it would mix helper behavior, real-host pytest changes,
operator docs, and workflow-policy edits in one review. It also risks inventing
fault-injection mechanisms before the drill boundary is agreed.

### Option C: Only Record Evidence From Existing Smoke

Run the current prepared-host smoke and add evidence entries.

This is useful, but it does not close the known lifecycle gaps because the
default smoke does not intentionally exercise stale socket, stuck readiness, or
guest-agent mismatch behavior.

## Chosen Design

Use Option A. This PR should create the drill contract and lock the manual-only
boundaries with doc-contract tests. Later implementation PRs should each choose
one drill, add the minimum host-independent coverage, and only then expose an
operator command or manual host-gated path.

## Drill Contract: Stale Socket

Purpose: prove that stale helper socket recovery is safe and diagnosable without
ever unlinking arbitrary paths.

Accepted implementation shape:

1. Create a private runtime directory with owner-only permissions.
2. Create or preserve an accepted socket path under that directory.
3. Simulate a stale socket only as a Unix socket path that is not backed by a
   responding helper.
4. Run `vz-helperctl.py check` or `status` and record the path-safety result.
5. Start the helper through `vz-helperctl.py start` or a future dedicated manual
   drill command.
6. Verify that only a safe stale Unix socket under the accepted private runtime
   directory may be removed.
7. Verify symlinks, non-socket files, directories, and user-controlled parent
   paths fail closed.
8. Preserve helper stdout/stderr logs and serial-log directory evidence.

Expected outcome:

- safe stale socket under the private runtime directory is recoverable
- unsafe socket path shapes are refused before unlink
- no arbitrary path is removed
- evidence records socket path, runtime dir mode, command, result, and logs

## Drill Contract: Stuck Boot And Stuck Readiness

Purpose: prove that boot-driver and guest-readiness failures do not leave
registered VMs, stale session-control rows, or ambiguous diagnostics.

The first implementation should prefer host-independent helper tests and
service tests using fake boot drivers or fake guest bridges. Real-host drills
should come later only if a prepared host can run them reliably without corrupting
operator bundles.

Accepted implementation shape:

1. Use a test-controlled boot-driver failure or readiness-timeout failure.
2. Create the VM through the same helper create path used by real execution.
3. Fail before command execution.
4. Verify helper registry cleanup.
5. Verify Python runner cleanup of active VM/run bookkeeping when VM creation
   partially succeeds and readiness then fails.
6. Verify diagnostics expose a stable boot/readiness reason without reading raw
   serial logs into API output.
7. Record helper stdout/stderr and serial-log artifact pointers when available.

Expected outcome:

- stuck boot does not leave a reusable VM
- stuck guest readiness does not mark a session VM healthy
- session reuse falls through to fresh provisioning only after helper truth
  proves the old candidate is absent or unhealthy
- diagnostics classify the failure as readiness/boot failure rather than a
  generic runtime error

## Drill Contract: Guest-Agent Mismatch

Purpose: prove that an incompatible or missing guest agent fails closed and is
visible to operators.

Accepted implementation shape:

1. Simulate agent mismatch through a fake guest bridge, helper unit test, or
   purpose-built test bundle. Do not corrupt the canonical operator bundle.
2. Exercise helper readiness and execution paths, not only docs.
3. Verify helper metadata reports guest readiness details when available.
4. Verify command execution fails before returning misleading guest output.
5. Verify session reuse does not reuse a VM whose guest-agent state is missing,
   incompatible, or unhealthy.
6. Verify diagnostics and evidence packet fields identify the mismatch.

Expected outcome:

- agent mismatch is a blocking runtime failure on real execution
- no fake helper/template env flag can make the mismatch look like a pass
- evidence records guest-agent version/protocol if known, mismatch reason,
  command, helper version, and artifact pointers

## Host Reboot Boundary

Host reboot remains a manual operator procedure, not a CI drill.

Accepted manual procedure:

1. Preserve pre-reboot helper status, diagnostics, session-control state, and
   artifact pointers.
2. Reboot only a prepared host that can tolerate disruption.
3. After reboot, start or verify the helper through the operator-managed path.
4. Run `vz-helperctl.py status` and confirm protocol-compatible helper ping.
5. Run `/api/v1/sandbox/admin/macos-diagnostics`.
6. Inspect stale, unhealthy, skipped-active, and orphan classifications.
7. Run reconciliation repair in dry-run mode before any mutation.
8. Apply mutating repair only after reviewing the dry-run plan and only for
   ownership-checked candidates.
9. Run the real host smoke again to verify fresh ephemeral execution and
   same-session behavior.

Non-goals for host reboot:

- no scheduled reboot CI
- no hidden startup repair
- no broad orphan VM termination
- no launchd takeover unless the operator explicitly chose launchd validation

## Evidence Tracker Updates

Each future drill PR should update the prepared-host evidence tracker with:

- drill name
- command or workflow run
- pass/fail/skip result
- explicit skip reason when manual prerequisites were not selected
- first failing command/log pointer for failures
- artifact names, byte sizes, or checksums
- residual follow-up owner

Evidence entries should not paste secrets, raw user data, or full runner logs.

## Implementation Slices

Recommended order:

1. Stale socket operator drill or documented check.
2. Host-independent stuck boot/readiness tests.
3. Guest-agent mismatch tests and diagnostics contract.
4. Host reboot manual playbook and evidence template expansion.

Each slice should be independently reviewable and should include:

- Backlog task
- focused tests
- operator docs update
- prepared-host evidence tracker update
- no workflow trigger expansion

## Test Strategy

This design slice should add doc-contract tests that verify:

- the spec exists and names the four drill areas
- stale socket safety refuses symlinks/non-socket paths and permits only safe
  stale Unix sockets under a private runtime directory
- stuck boot/readiness and guest-agent mismatch are not default scheduled
  destructive drills
- host reboot stays manual-only and out of scheduled CI
- future implementations must remain dry-run-first for repair mutation and
  ownership-checked for VM termination

Future implementation slices should add targeted tests in the subsystem they
touch:

- helper unit tests for socket safety, boot failure, and readiness timeout
- Python runner tests for cleanup and reuse fallback
- host-gated workflow tests only when workflow inputs or policy change
- real-host pytest markers only for explicit manual drills

## Design Risks And Mitigations

- Risk: one broad "lifecycle drill" command grows into unsafe automation.
  Mitigation: split stale socket, stuck readiness, guest-agent mismatch, and
  host reboot into separate implementation slices.
- Risk: real-host stuck boot drills become flaky or damage prepared bundles.
  Mitigation: start with fake boot/guest-bridge tests; require a separate review
  before adding real-host fault injection.
- Risk: stale socket cleanup becomes a path-deletion primitive.
  Mitigation: require private runtime directories, helper-side `lstat`, and
  refusal of symlinks and non-socket files.
- Risk: host reboot testing disrupts shared runners.
  Mitigation: keep reboot manual-only until a dedicated runner and log retention
  story exists.
- Risk: diagnostics overclaim certainty after helper restart or reboot.
  Mitigation: treat persisted session-control rows as provenance, not live VM
  proof; helper truth is required before reuse or mutation.

## Open Questions For Implementation Planning

- Should the stale socket drill be a new `vz-helperctl.py socket-drill` command,
  or should `check`/`start` evidence be sufficient?
- Should boot/readiness failures be simulated entirely in Swift helper tests, or
  should Python service tests also inject helper-client failures?
- Should guest-agent mismatch report a dedicated stable reason code before the
  first implementation drill lands?
- What minimum prepared-host evidence is required before any of these drills can
  be promoted from manual local run to manual host-gated workflow input?
