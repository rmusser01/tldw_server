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

## 2026-09-13 Reproducible Guest-Fault Workflow

TASK-13243.5 packages the two already accepted real guest drills (missing `exec`
capability and acknowledged-handshake readiness withholding) into one explicit
operator workflow, `tools/macos-vz-helper/scripts/vz-failure-drill.py`. This is
preparation and repeatability work, not a new runtime or a general fault engine.
The existing individual pytest entrypoints remain available.

- Require explicit fault-injection consent, a known-good Debian arm64 ext4
  bundle, a signed helper path, and a new private evidence directory.
- Keep fault behavior in checked-in test fixtures applied through Go build
  overlays. Refuse ambiguous/missing source anchors rather than silently
  producing a healthy guest. No production flags or source edits.
- Use the existing image-store materializer for all clones. Install each test
  binary into an offline clone through a separate healthy disposable preparer
  VM; compare installed bytes and run filesystem checks. Never boot source
  bundles, and independently verify their hashes even after a failed drill.
- Use the existing managed direct-helper lifecycle at a fresh private socket,
  with signing/entitlement preflight and PID ownership checks. Do not attach to
  another helper, install launchd services, or change default smoke/CI triggers.
- Run both positive tests and negative controls on fresh clones. Negative
  acceptance requires the intended assertion failure **and** recorded completed
  execution, exit zero, exact stdout, and dispatch to the fault VM. Cancelled
  runs, unrelated failures, missing reports, and skips are not acceptance.
- Retain receipts, input/binary/source hashes, overlays, logs, and image-store
  clones. Independently attempt VM cleanup, disk-handle checks, helper stop, and
  socket/PID absence verification. Any cleanup uncertainty makes the command
  fail. Only the empty short runtime directory is automatically removed.

Host reboot, kernel boot hangs, missing-agent, protocol-version mismatch,
workspace mismatch, arbitrary crash classes, and scheduled fault injection
remain outside this slice. See the helper README for the operator command and
`Docs/Sandbox/vz-linux-prepared-host-evidence.md` for the actual host results.

## 2026-09-14 Guest Protocol Mismatch Extension

TASK-13243.6 adds the next bounded guest-compatibility drill to the existing
workflow. A test-only Go overlay sends initial guest handshake version `999`;
the normal helper rejects it with `guest_protocol_mismatch` rather than the
generic `helper_internal_error`. Unrelated malformed messages retain their
existing classification. This is the guest VSock protocol, not the separate
host-helper protocol or the already tested missing-exec capability gate.

The dedicated diagnostic also covers version mismatches in later exec replies.
Swift server/bridge regression tests cover that path and preserve request-ID and
malformed-response classifications; the live drill injects only the handshake.

The existing disposable image-store/offline-preparer/isolated-helper lifecycle
remains authoritative. A fresh workspace nonce correlates a guest-written proof
with the attempted VM and wire version. Proof alone cannot pass the test:
acceptance requires the specific rejection, no exec dispatch, no surviving
reusable VM state, then real healthy execution and same-session VM reuse.
The negative control changes only the fixture's version to the supported value;
helper validation stays enabled, and real successful execution plus the intended
assertion failure are both required. The command now requires all six cases.

Keep canonical/source hashes and cleanup verification, retain the evidence,
and do not introduce production fault flags, automatic injection or reboot.
Host reboot, missing-agent, early kernel boot hangs and other deferred fault
classes remain separate. Record actual host results in the prepared-host evidence
document; this design extension is not itself proof of real VM acceptance.

## 2026-09-15 Advertised Workspace Mismatch Extension

TASK-13243.7 isolates the existing runner workspace-metadata admission guard.
Reuse the same disposable image-store, offline installer, isolated helper and
evidence workflow, now with four profiles and eight required positive/negative
cases. No production admission or VM lifecycle change is intended.

The test-only Go overlay alters only the initial handshake's advertised root to
`/workspace-mismatch/<fresh nonce>`. The actual `/workspace` mount, guest protocol
and `exec`/`output_cap_v1` capabilities remain unchanged. The host must observe
that exact root from the real helper create reply for the attempted VM, then
require only `vz_linux_guest_agent_workspace_mismatch`, no exec dispatch, no
reusable VM state, and healthy replacement execution plus same-session reuse.
Public reconciliation, session APIs and helper inventory establish cleanup;
do not reach into private orchestrator state or bypass admission.

The negative control changes only the challenge to advertise the supported root.
Acceptance requires supported metadata and actual completed, exit-zero,
exact-output execution before the intended rejection assertion fails. A skip,
readiness failure, capability fault, foreign VM, or uncertain cleanup is not
acceptance. Existing source-hash and helper/disk cleanup gates remain mandatory.

This proves advertised metadata admission, not mount isolation or path-escape
resistance. Missing-agent, early boot-hang and host-reboot live evidence remain
separate; no reboot, source-bundle mutation or scheduled fault injection is added.

## 2026-09-26 Initramfs Boot-Stall Extension

TASK-13243.10 extends the existing manual failure workflow to six profiles and
twelve mandatory positive/negative cases. Unlike the missing-agent launcher,
this test-only wrapper runs as initramfs PID1 before Debian's original `/init`,
rootfs mounting, systemd, or guest-agent startup. It prints a fresh nonce,
requested VM ID, mode, and `initramfs` stage through the helper's per-VM serial
log, then stalls. A transport timeout without matching serial proof is a failed
drill, not acceptance.

The existing disposable healthy preparer extracts the original `/init` from an
offline fault-source clone. Append an aligned native `newc` archive containing
the wrapper, preserved original init, and fixture settings to that clone's
manifest-selected initrd. Do not modify the source bundle, kernel, rootfs, or
production agent. Each case clones that prepared image through the existing
image store and appends only its fresh nonce/mode settings. The continue-mode
negative control executes the preserved original init through the same wrapper;
acceptance requires completed real fault-VM execution plus the intended failed
assertion. Normal host validation remains enabled.

Positive acceptance requires a bounded guest-transport timeout, zero exec,
no reusable session VM state, healthy replacement execution and same-session
reuse. Existing VM enumeration, disk-handle, helper shutdown, source-hash and
evidence-retention gates remain mandatory. Record preparation initrd hashes and
original-init hash as well as input hashes and live serial proof. This extends
test fixtures only: no production fault flag, Docker dependency, default-helper
takeover, launchd change, reboot, or default CI trigger.

This is explicitly early-userspace-stall evidence, not closure of arbitrary
kernel hangs, a stalled Virtualization.framework start callback, mount isolation,
or host reboot recovery. Portable success does not establish live acceptance;
the prepared-host evidence tracker must record an actual twelve-case receipt.
